# Qwen3.8-27B (Fable) + DFlash2 bring-up — verified facts and plan

Everything below was read from the local checkpoints' GGUF headers, the
z-lab/dflash source (cloned to /tmp/dflash during research), and Rusty
Llama's `src/models/qwen35.cpp` / `qwen35moe_mtp.cpp` references. No
guesswork. User constraints for testing: **KV cache q8, flash attention on,
context < 64k** — see "gaps" for what FreeToken supports today.

## Checkpoints (verified on disk)

- Target: `/mnt/HDD-2/Models/TeichAI/Qwen3.8-27B-Fable-Distill-GGUF/Qwen3.8-27B-Fable-Distill-Q6_K.gguf`
  — 22.9 GiB, arch `qwen35` (DENSE hybrid, no experts), 65 blocks, H=5120,
  FFN 17408, vocab 248320, untied output (Q6_K).
- Draft: `/mnt/HDD-2/Models/incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf`
  — 2.06 GiB, arch `dflash`, 5 blocks, H=5120 (matches target), 32q/8kv × 128,
  causal=False, sliding_window 2048, enc.* tensor prefix.

## Target structure (from GGUF tensors + qwen35.cpp)

- Layers idx%4==3 are FULL ATTENTION (17 of 65): attn_q [5120→12288]
  (per-head q|gate interleaved, 24 heads × 256), k/v [5120→1024] (4 kv),
  attn_output [6144→5120], q/k norms [256]. Partial NeoX rope: 64 of 256
  dims, base 1e7 (text path ignores mrope sections — same as Ornith).
- Other 48 layers are GDN: attn_qkv [5120→10240] with layout
  [q 2048 | k 2048 | v 6144] = key_dim(16×128)×2 + value_dim(48×128);
  attn_gate [5120→6144] = the output gate (FreeToken's in_proj_z);
  ssm_beta|alpha [5120→48] = in_proj_ba [b|a]; ssm_a [48] = A_log;
  ssm_dt.bias [48]; ssm_norm [128]; ssm_conv1d [4, 10240]; ssm_out
  [6144→5120]. FreeToken's GDN module split
  ([key,key,value] + z + b|a) matches this EXACTLY — parameterize
  num_key_heads=16, num_value_heads=48, key/value_head_dim=128, conv 4.
- Dense FFN every layer: gate/up [5120→17408], down [17408→5120] →
  Qwen3_5DenseMLP + the existing ggml kquant dense path (dense_quant on).
- Norms pre-baked (1+w) by the converter, same as qwen35moe.
- `nextn_predict_layers=1` exists in the file — IGNORE (user direction:
  DFlash2, not the built-in MTP).

Adapter estimate: config parser for the `qwen35` arch keys + a weight-name
mapping onto the existing Qwen3_5 decoder-layer modules (dense-MoE-off path
already exists via `moe_enabled=False`). No new kernels needed for inference.

## DFlash2 semantics (from /tmp/dflash/dflash/model.py + GGUF metadata)

- Draft config: block_size=8, target_layers=[6,20,34,48,62] (trunk hidden
  at these depths), conv_kernel=2, conv_group=16, selector_rank=256,
  selector_top_k=16, mask_token_id=248070, vocab 248320.
- Draft inputs per cycle: (a) context feature = concat of the trunk's
  hidden states at the 5 target layers for the newly produced rows →
  fc [25600→5120] + enc.output_norm; (b) noise embedding = the TARGET's
  token-embedding rows of [anchor token, MASK×(k−1)].
- 5 non-causal layers (Q from block, K/V = cat(context, block), sliding
  window 2048 on context); DFlash2 adds grouped dynamic causal convs on
  hidden (attn_conv + ffn_conv per layer: base [2,2,5120] + proj
  [2,2,320,5120]) and a CandidateSelector (unary logits + bigram codebooks:
  selector_predecessor/successor [vocab,256], hidden_projection
  [256,5120], top-16 chained per position).
- Verify loop (reference greedy): feed the target the block
  [anchor y_n, c1..ck]; trunk logits row j accepts c_{j+1} iff argmax_j
  matches; commit longest prefix + one BONUS token (the trunk argmax at the
  first mismatch). Crop the target KV to the committed length (dense KV —
  no GDN-state rollback issue on the verify path itself, but the TARGET has
  GDN layers: their state advanced over the speculative tokens — snapshot
  per-request GDN state before verify, restore on partial accept, exactly
  the pattern documented for M2d).
- Economics (z-lab published + video math): draft read 1.14 GiB @ 4-bit
  (ours is Q8_0, 2.06 GiB), target read 13.9 GiB; ceiling 5.05×, realized
  3.43× arithmetic / 2.67× conversation at bs=1; degrades to ~1.0 under
  concurrent load. block_size 5 recommended with quantized drafts
  (--draft-bits 4); our target is Q6_K → start k=5, sweep k∈{4,5,8}.

## Placement (dual-GPU, verified fits)

22.9 GiB dense weights do not fit one card with KV. Split layers across
GPU0 (XTX, ~layers 0-32) and GPU1 (7800 XT, ~33-64 + the 2 GiB DFlash
engine); hidden crosses at the boundary (10 KB/step). KV: only full-attn
layers carry paged KV (17 layers × 4 kv × 256): ~70 KB/token bf16 → 64k
context ≈ 4.4 GiB split by layer ownership. GDN state ~30 MB/request.
The draft's context features come from trunk layers 6,20 (GPU0) and
34,48,62 (GPU1) — gather at the boundary (~30 KB/row).
Reuse the piecewise-graph philosophy for cross-device seams; the MTP
engine's private-stream + pinned-staging exchange pattern applies verbatim.

## Gaps vs user constraints (honest)

- **FreeToken has no quantized KV cache today** (bf16 paged KV only). With
  <64k context the budget fits in bf16 (4.4 GiB), so bring-up proceeds
  without it; q8 KV becomes a capacity feature later if wanted.
- Flash attention: FreeToken's ROCm attention backend is the Triton
  flash-style path already (auto-resolves on HIP) — nothing to toggle.
- FreeToken has no layer-split-across-GPUs infra yet (TP=1 only for GGUF);
  the boundary seam is the new machinery (small, reuses piecewise seams).

## Build order

1. `qwen35` dense GGUF adapter (config + weight mapping; reference-check
   logits against Rusty Llama running the same GGUF).
2. Layer-split placement + serve the Fable target on both GPUs (<64k ctx).
3. DFlash2 draft engine from the incoai GGUF on GPU1 (non-causal forward +
   candidate selector; standalone-testable against the z-lab reference).
4. Block-verify loop in the scheduler (k=5): trunk forward over
   [anchor + drafts] with the 5-layer hidden tap for the draft context,
   GDN-state snapshot/restore, greedy longest-prefix accept + bonus token.
5. Sweep k, measure tok/s + acceptance; compare against the video's
   3.43×/2.67× reference points.

## S1/S2a results (2026-08-24, measured)

- **S1 adapter + serving both landed.** Strict-load clean; then two bring-up
  fixes: (1) `_TOKENIZER_ARCH["qwen35"] = "qwen2"`; (2) the loader's
  `include_moe_experts` flag is the resident-path default True for dense
  models too (my assert was wrong, not the engine).
- **v-head regroup is r-generalized**: the converter's reorder is
  grouped-by-K -> tiled (`_LinearAttentionVReorderBase`). For Ornith
  (nv=32, nk=16, r=2) tiled == [evens|odds], which the qwen35moe loader
  hardcodes; **Fable has r=3 (48v/16k)** where that pair split garbles the
  GDN v-heads -> degenerate "covering Paris, covering Paris..." output.
  Correct inverse: `src = (j % r)*nk + j//r` (now `_v_src_index` in
  qwen35_dense/gguf.py). Symptom-to-cause signature for future ports.
- **Single-GPU baseline (user-confirmed LM Studio parity)**: Q6_K weights
  22.9 GiB on the XTX with 0.89 GiB spare at 4k ctx (flags:
  `--max-seq-len-override 4096 --memory-ratio 0.96 --max-extend-length 1024
  --max-prefill-length 1024 --kv-reserve-tokens 2048 --max-running-requests 2`;
  bf16 KV — q8 KV remains a gap).
- **Empiricals**: decode 28.2 tok/s (early ctx) -> 26.3 @ 2.5k; warm TTFT
  0.50 s; coherent reasoning AND content (enable_thinking=false path
  verified); graphs captured bs [1,2].
- **Known anomaly for next pass**: FIRST request after startup pays ~160 s
  (prefill of ~26 tokens; suspect per-shape kernel JIT/compile in the
  extend path — warm prefill is 0.5 s).

## Harness-usability fixes (2026-08-25, for maestro-terminal et al.)

Symptom: agent harness saw "recognized and processes but no output". Three
stacked causes, all fixed:

1. **Reasoning budget**: Fable thinks at length before any `content`; a
   harness with a modest max_tokens gets an all-reasoning response.
   `FREETOKEN_DEFAULT_REASONING_EFFORT` (e.g. `none`) now sets a server-wide
   default; per-request `reasoning_effort` and explicit
   `chat_template_kwargs` still win.
2. **Sampler Triton cliff**: torch 2.13's multinomial compiles ~190 kernels
   per shape on first use -- measured 74-207 s stalls on the first sampled
   request (attribution: fresh TRITON_CACHE_DIR, kernel names
   `_draw_*`/`_count_hist`/`_refine_mass`). `_warmup_sampler()` now
   pre-compiles all top-k/top-p variants at startup (FREETOKEN_WARM_SAMPLER=0
   to skip).
3. **Prefill warmup ladder** [80..1024] + `torch.cuda.empty_cache()` after
   the sampler warmup: on a nearly-full GPU the retained warmup cache
   starved the tokenizer worker's lazy CUDA context and requests hung
   indefinitely (0.68 vs 0.84 GiB free was the difference).

Measured after fixes (warm triton disk cache): FIRST short request 0.8 s,
sampled mid-length 7.3 s, tools+system 3.9 s. On a fresh triton cache the
compiles move to startup (~1-2 + ~10 min one-time).

## Harness reality check (2026-08-25, maestro-terminal)

- **The no-output cause**: the harness's initial prompt is **23,346 tokens**
  vs the single-XTX test cap of 4,096. The scheduler drops it with an
  ErrorReplyMsg ("prompt is too long: N tokens > M") -- the harness client
  swallowed the error. Real agent contexts are ~24k+; the <64k testing
  target needs ~2.2 GiB of KV, which with 22.9 GiB of dense weights does
  not fit one 24 GB card. **S2b (dual-GPU layer split) is the critical
  path.**
- **Prefill throughput measured**: ~100 tok/s warm (308 tokens in ~3 s),
  spiky GPU utilization 16-20% floor with 85-100% bursts -- the eager
  host-bound chunked-prefill path (kernel bursts, Python gaps between
  layers/chunks). A 23k prompt would take ~4 min even with capacity. Fix
  direction: chunked prefill runs at a FIXED chunk size (constant shapes)
  -> prefill chunks are graph-capturable; schedule with S2b.

## S2b layer split -- machinery landed, one environmental blocker (2026-08-25)

Landed (engine/layer_split.py + hooks, uncommitted-to-committed WIP):
- Routing pools: SplitMHAKVCache / SplitLinearStatePool keep global layer
  ids; page/slot ids identical on both sides; store_kv moves out_loc to the
  owning device. Far-side KV budget cap with draft reserve
  (FREETOKEN_LAYER_SPLIT_FAR_RESERVE_GB).
- Seam crossing: (x, residual) + batch metadata (out_loc/positions/
  attn+fla metadata, recursively) cross to cuda:1 at layer FREETOKEN_LAYER_SPLIT;
  hidden crosses BACK before the final norm + lm_head (KBs, not MBs of logits;
  the k-quant head keeps its working ggml GEMV on cuda:0).
- FarSideLinear: the vendored ggml kernels HANG on the 7800 XT even with a
  dual-arch fatbin (reproduced standalone, single GPU). Far-side projections
  convert at load to Q4_K (Triton kq_gemv decode) + bf16-transposed (prefill
  GEMM); 48-72 projections depending on split point. empty_cache is
  device-scoped (the plain call frees cuda:0's cache only -- cost a full
  debug cycle).
- The inter-GPU flags the user recalled are real and inherited from the
  stack env: HSA_ENABLE_SDMA=0 (SDMA off; crossing copies stage via CPU --
  why pinned crossings worked), HSA_OVERRIDE_GFX_VERSION=11.0.0 (7800 XT
  presents as gfx1100; needed for RCCL, UNUSED by this TP=1 server),
  RCCL/NCCL_P2P_DISABLE=1.

**THE BLOCKER**: Triton JIT for a kernel first invoked on cuda:1 with BOTH
GPUs visible is pathologically slow -- a trivial kernel churns 200s+ at
~200% CPU (compile subprocess), which is what "hung" the warmup (the server
spun at 99.8% CPU for an hour in Prefill-warmup). With the 7800 XT as the
SOLE visible device the same kernel compiles in 0.3 s and runs CORRECTLY.
Under HSA_OVERRIDE it also returned a WRONG result (sum=0.0) in a
contended run. Next-session opening moves, in order:
1. If the dual-visible compile eventually completes correctly (900 s test
   queued in /tmp/tri_dual.log), the fix is a Triton cache pre-warm: run
   the far-side kernel set once with HIP_VISIBLE_DEVICES=1 (cache keys are
   arch-based, not device-index-based) so the dual-visible server hits
   disk cache and never compiles on device 1.
2. Else root-cause Triton's second-device compile (TRITON_CACHE_DIR per
   process, llvm target probing both devices).
NOTE: with HSA_OVERRIDE active the far GPU also returned a wrong sum once
-- never run the split server with the override set.
