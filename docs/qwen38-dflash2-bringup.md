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

## S2b verdict after the power cycle (2026-08-25): pivot to torch-only far side

Post-reboot re-verification, minimal repros, clean box:
- ggml C++ ext on the 7800 XT: still HANGS (solo). Real, not transient.
- Triton kernels on cuda:1 with the XTX active: FAULT or HANG broadly --
  rope faults, conv1d hangs (cache-hit!), trivial kernels fault; the fla
  chunk was the lone exception. Solo-visible (HIP_VISIBLE_DEVICES=1) every
  kernel is 0.3s and correct.
- Pure torch (hipBLAS/SDPA/matmul) on cuda:1 with XTX active: works
  perfectly, at full speed. The MTP bf16 draft engine ran this way all of
  2026-08-24.
- The 30-60 min "warmup livelocks" pre-reboot were STALE TORCH-EXTENSION
  BATONS (faulthandler stack: _warmup_prefill -> ggml_dequantize ->
  file_baton.wait) -- rm -rf ~/.cache/torch_extensions/<py-tag> clears.

CONCLUSION: on this ROCm stack (7800 XT = gfx1101 as second device),
heterogeneous Triton is not viable. The layer split lives ONLY if the far
side is torch-only:
- FarSideLinear (bf16 x @ Wt) already proven on GPU1.
- Far GDN layers: torch causal-conv (F.conv1d groups=dim, trivial) + the
  reference chunked delta-rule (gdn_reference.py) -- decode is a handful of
  einsums (fine for 6 layers), prefill ~50-200ms/layer/2048-chunk.
- Far full-attn layers: torch SDPA + explicit KV gather on GPU1's sub-pool.
Tooling landed: tools/mtp/far_seed.py (solo-GPU1 Triton cache seeder; conv1d
kernel specializes num_cache_lines = state-pool slots 13 -- seeder must
match), FREETOKEN_FAULTHANDLER=1 (+_SECS) periodic stack dumps in the
scheduler (py-spy is ptrace-blocked in the sandbox).

## S2b: THE REAL ROOT CAUSE (2026-08-25, user-directed deep research — supersedes 4a20e67's 'torch-only far side' verdict)

**Triton runs fine on the 7800 XT with the XTX active.** The user was right
that a flag/architecture-level fix existed. Deep research trail: the
transformers finegrained_fp8.py docstring describes the exact mechanism
class ("CUfunction handle binds to the CUDA context live at load time;
driving that cached handle from another device launches it against the
wrong context"); inspecting the installed triton/backends/amd/driver.c
`loadBinary` shows the `device` argument is parsed and IGNORED (no
hipSetDevice before hipModuleLoadDataEx), and jit.py's `run()` launches on
the THREAD-CURRENT device's stream, never inferring device from tensor
args. So every far-side Triton launch executed a device-0-bound function
on device-1 pointers -> faults / hangs / silent sum=0.0. THAT was the
"Triton can't run on the second GPU" mystery, not the hardware.

**Fix**: `torch.cuda.set_device(1)` / `with torch.cuda.device(1)` around
far-side Triton launches. PROVEN (dual-visible, dev0 warmed first):
trivial kernel 0.6s correct; full far-side battery (fla chunk T=80..2048,
fla decode bs=1-4, causal conv1d, rope, extend paged attention, kq_gemv)
ALL pass, 0.02-0.6s each. Also proven: dev0-then-dev1 kernel sequences and
engine-stream(near)+default-stream(far) sequences pass.

In-server integration state (fable39): seam machinery is correct (near
layers OUTSIDE the device ctx, far block inside, cross-back + far-queue
drain inside the ctx — an earlier iteration wrongly wrapped the near
layers too and faulted the ggml ext with 'invalid resource handle').
Remaining: the server still HANGS in `causal_conv1d_varlen`
(causal_conv1d_triton.py:457 <- gdn.py:_conv_prefill) on the far side
during _warmup_prefill — GPU-side hang (CPU 1.2%), all miniature repros
pass. Next iteration: print the far conv1d launch args (shapes/strides of
the _RoutingLayerTensor conv_states view + x layout) at the hang site, or
bisect FREETOKEN_LAYER_SPLIT=63 (2 far layers) to shrink it. Everything
else (weights converted, pools split, seam executed) already works.

## S2b state at handoff (2026-08-25 late): every component PROVEN, one server-side delta left

Clean-env recipe (PROVEN, dual-visible, dev0 warmed, engine stream current,
ggml ext loaded): `env -u HSA_OVERRIDE_GFX_VERSION -u HSA_TOOLS_LIB -u
LD_PRELOAD HSA_ENABLE_SDMA=0` + device-ctx far block. Under this recipe the
FULL server pattern passes in a probe: near ggml work on the engine stream,
far fla chunk, far gemma_rmsnorm fresh-compiled (never seeded) — correct
results, sub-second. HSA_OVERRIDE unset is load-bearing: with it, both GPUs
report gfx1100 and triton.compile's GLOBAL cache returns the dev0-bound
CompiledKernel object for dev1 (same target key) -> wrong-context launch.
Without it, gfx1100 vs gfx1101 = distinct targets = per-device kernels.
HSA_TOOLS_LIB unset: with it, the first far launch segfaulted inside the
rocprofiler-wrapped hipModuleLaunchKernel dispatch.

Server (fable41) with this exact env STILL segfaulted at the first far
Triton launch after the seam. Remaining deltas vs the passing probe (in
likelihood order): (1) the SEAM CROSSING itself — x/residual cross-device
.to(non_blocking=True) enqueued after switching current device to 1 with
SDMA disabled (peer copy staged through host); the probe never crosses, it
allocates directly on dev1. Fix: event-order the crossing (record event on
the near engine stream, far stream waits) or make the crossing blocking.
(2) near-engine-stream vs far-default-stream race. Next session: add the
event-ordering at the seam, relaunch, then the full verification battery
(coherence, 23k prompt, tok/s, TTFT) and the phase commit.

## S2b close-but-not-passed state (2026-08-25, end of session)

The far side now runs its ENTIRE tail on cuda:1 (all far layers + trunk
norm + a converted Q4_K-only Triton FarHead with the last-indices prefill
trick; only [T, vocab] logits cross back, pinned two-hop with D2H inside
the far ctx). With the proven config (split=56, ratio 0.92, fp32 SSM,
mr=2, no num-pages override, `env -u HSA_OVERRIDE_GFX_VERSION -u
HSA_TOOLS_LIB -u LD_PRELOAD HSA_ENABLE_SDMA=0`) the server COMPLETES THE
FULL PREFILL WARMUP on both GPUs and reaches ready-to-serve. The FIRST
REAL REQUEST then dies: an async far-side fault that masquerades as
`CUDA error: out of memory` at the next host alloc (impossible OOM: [mem]
log shows cuda:0 free=2.4-4.2 GiB at the failure point).

Machinery landed this session (uncommitted): far-tail restructure (norm +
head inside the far device ctx), FarHead (Q4_K requant, kq_gemv, chunked
conversion, last-indices), head-then-cross logits ordering, fresh-vs-cached
pinned staging discipline (cached long-lived buffers for x/r/logits;
metadata crosses via small blocking direct copies — fresh pinned per tensor
per forward EXHAUSTED pinable memory and surfaced as the same fake-OOM),
num_slots on the split linear pool, dev0 empty_cache after far-weight
moves, [mem] logging at warmup.

NEXT SESSION, IN ORDER:
1. `CUDA_LAUNCH_BLOCKING=1` on the split server (warmup is CPU-bound but
   tolerable; first request will fault SYNCHRONOUSLY at the exact kernel
   launch — that names the culprit op directly). One run answers it.
2. Suspects ranked: the far full-attn layers' store_kv/extend attention
   with crossed TritonMetadata (only exercised by REAL batches, not the
   dummy warmup batches — warmup uses the dummy KV slot and dummy pages;
   a real request hits fresh page indices + the radix insert path);
   get_last_indices on crossed metadata inside FarHead.
3. If the far attention is the culprit, the fallback is the far side =
   last 4-8 GDN/FFN layers only with full-attn layers pinned near (the
   split point can skip full-attn layers to dev0 by mapping them back).

## S2b PASSED (2026-08-26, commits 368586e)

Phase gate: server/model running on BOTH GPUs — MET.

Final blocker chain (this session):
1. fable77 "capture crash" = CUDA_LAUNCH_BLOCKING=1 self-sabotage inside graph
   capture. The gfx1101 wmma "[0,0,K]" compile stderr is non-fatal noise
   (FMA fallback: solo-7800XT probe compiles + launches fine, exit 0).
   Rule: blocking mode ONLY with --cuda-graph-max-bs 0.
2. True seam bug: cross_to_dev1 moved batch.input_ids to cuda:1 -> CausalLM's
   logits-crossing branch (output.device != input_ids.device) silently
   disabled -> raw far-side [T,248320] logits handed to the near sampler ->
   HSA page fault at the next idle sync. Fix: input_ids never crosses;
   branch keys on output.device == dev1(). Trace-proven coherent Fable run.
3. Graph capture with the split still OOMs at 2.42 GiB free (capture pools
   double-account far-side weights). Decode-runs-eager for now; split-aware
   capture/piecewise decode is S2c.

Ridge pivot (user directive): empero-ai Qwen3.8-27B-Ridge-3.7bpw.
- Types IQ2_S/IQ3_S/Q8_0/Q5_K added to Python dequant (source = vendored
  ggml-common.h/dequantize.cuh; _iq_tables.py generated). Verified vs GPU ext.
- '!!!' degenerate root cause: ext MMQ prefill kernels BROKEN on IQ2_S/IQ3_S
  (rel~1.0/NaN); MMVQ clean -> every prefill poisoned KV. Fix _ext_packed():
  requant unsafe types to Q4_K at load; native only {Q8_0,Q4_K,Q5_K,Q6_K}.
- Beware the pip cwd trap AGAIN: installs from repo root install Rusty-Stack,
  not freetoken (ridge04 learned this twice).

Measured (Ridge split=56 eager): decode 24.4 tok/s e2e coherent; ~4.2k-token
prefill 100 s; Ridge has NO no-think mode — effort "none" yields empty chat
outputs by design (reasoning stream itself is coherent). Fable empty-think
answering is the exception.

DFlash2 note: incoai Qwen3.8-27B-DFlash2-Q8_0 exists for Fable trunk. NO
Ridge-matched draft head exists — S3 needs a decision if the target stays
Ridge (train head? use Fable pair first? switch trunk back to Fable?).

## S2c decode-graphs design (parked for next session, 2026-08-27)

Goal: base decode 4.7@68k → mid-high 20s; DFlash2 compounds AFTER that.

Approach: reuse PiecewiseCapture but SPLIT-AWARE — two graphs, two devices:
  near-graph  (dev0): layers 0..55 + all near-side attention/meta kernels
  seam        (eager, outside graphs): cross_to_dev1 into PERSISTENT dev1
              buffers (currently cross allocates fresh dev1 tensors each step
              — must switch to fixed staging buffers the far graph reads)
  far-graph   (dev1): layers 56..63 + trunk norm + FarHead, captured under
              torch.cuda.device(dev1); logits cross back eagerly (already
              pinned two-hop)

Blockers closed before starting:
- capture_seam is invoked from inside model.forward (OffloadMoELayer today);
  Qwen3_5Model.forward needs its own capture_seam call at the layer-56
  boundary when piecewise capture is active (gate: split_enabled()).
- Graph pools are PER-DEVICE: PiecewiseCapture shares one pool handle —
  extend _CaptureState to per-device pools {dev: pool}.
- GraphCaptureBuffer must gain dev1 twins for what the far graph reads:
  positions (rotary), fresh_state_indices, and the seam hidden buffer.
- replay(): runner order becomes buf.copy_from → near_graph.replay →
  seam copies (pinned D2H+H2D, eager) → far_graph.replay → logits out.
- Memory: earlier monolithic capture OOM'd at 2.42GiB free; piecewise with
  per-device pools + mrr=1 + memory-ratio 0.88-0.90 leaves headroom for two
  small decode graphs (activations only; weights/KV are outside pools).

Also parked in-tree: probe.py live hook (env-disabled) + kquant_gemm_wip.py
(Triton dequant-GEMM, debug notes inline).

## S2c status at context close (2026-08-28, commit 1858a9a+)

BREAKTHROUGH: both split graphs (bs 2+1) CAPTURE and server reaches ready
(6.1 GiB free). Fix ladder: dead twin-swap flag unified, fail-fast guard on
in-capture crossings, per-pass near-attr restore, per-(attr,shape) twin
keying, per-segment pools (ROCm returns None pool handles), stream-paired
open/close, split branch skips monolithic pool tail.

REMAINING BLOCKER: first real request → scheduler (or tokenizer) worker
spins 100% CPU indefinitely; GPUs ~idle; no Prefill log. Exception-swallow
retry loop is the leading theory; py-spy needs sudo (ptrace denied).
NEXT: one-line unguarded traceback print in the engine.forward_batch
exception path OR scheduler receive loop, one boot, read the answer. Then
fix + verify graph replay + depth ladder (target: base 25-29 tok/s).
Do NOT spin a serve process at 100% CPU again — kill within one probe cycle
(user hard rule).

## S2c discriminator result (2026-08-28, ridge50-55)

DECISIVE: far-eager tail fed from the NEAR GRAPH's replay output ALSO
produces degenerate text ("#" repetition) — while seam hashes prove the near
output CHANGES every step. Conclusion: the NEAR graph replays produce
plausible-but-wrong hiddens. The near segment's capture is missing /
mishandling some kernel family's work: prime suspect is a launch-stream
mismatch (a kernel inside near layers — ggml ext GEMV/GEMM, fla chunk
kernels, or causal_conv1d — issuing on a non-captured stream, so manual
capture_begin on stream0 misses it; eager mode unaffected). torch.cuda.graph
ctx wrapper handles cross-stream capture bookkeeping that raw
capture_begin/capture_end does not — next session: capture the near segment
via the torch.cuda.graph ctx manager (stream=self.stream0) and move the
seam hooks to call capture_seam (piecewise.py) which is proven under that
wrapper, rather than raw begin/end in SplitGraphCapture.

Also verified this round: the earlier "spin" no longer reproduces with the
spin-diag commit (replay ran, 3.1s/80tok = ~26 tok/s class decode at tiny
context — the graphs DO execute fast); correctness is the only gap.

## S2c wrapper-capture ladder (2026-08-28, ridge56-61)

Wrapper (torch.cuda.graph ctx) ladder results:
- ridge57: far on dev1 default stream → "must be non-default" → fixed with
  explicit stream=stream1 + stream ctx entered for the segment
- ridge59: near open/close both on stream0 ✓ (with-block) — wrapper's
  __exit__ calls torch.cuda.synchronize() which is ILLEGAL while the OTHER
  device's capture is open → replaced wrapper __exit__ with direct
  current.capture_end() (stream ctx exit after)
- ridge61: far capture_end still fails with async StreamCaptureUnsupported
  surfacing at end → an illegal op INSIDE the far segment is being
  async-flagged. Far layer inventory: FarSideLinear (triton kq_gemv), FarHead,
  GDN fla kernels, far pool store_kv/gather, conv1d, far attention reading
  metadata TWINS (dev1 ✓). Remaining suspects: (1) triton backend's ctx
  (TritonAttentionBackend built with device=dev0) touching dev0 scratch
  inside far attention; (2) ctx.page_table (dev0) read by far attention path;
  (3) fla build_fla_metadata creating a dev0 tensor from a batch attr not in
  the twin set (e.g. reading batch.input_ids? NOT crossed deliberately!).

NEXT SESSION: enumerate every tensor the far segment reads (grep far-layer
forward paths for ctx./batch. accessors), add each to the twin set, retry.
If all twins covered and still failing, hook HIP stream ops via
AMD_SERIALIZE_KERNEL=3 + CUDA_LOG_FILE to name the op.

## 2026-08-29: first-request stall class SOLVED (warmup + prefetch, 4bfafb9)

**Forensics trail.** The "416.6s for a greeting" session, the 4m23s prefill->decode
gap, and the 8.5k no-decode repro all decomposed into TWO stacked causes — plus one
phantom:

- **Phantom**: the "final-chunk -> decode transition failure" was never real. The
  tail-chunk Req passes `can_decode` (log's `#running-req: 1` on the *previous*
  chunk's report proves it) and is then legitimately finished by `hit_eos` on its
  first sampled token: the model argmaxes `<|im_end|>` on degenerate repeated-pangram
  raw prompts at >=~140 tokens. Natural prose continues perfectly (`Margaret's novel
  was a paperback copy of The Little Prince...`). Synthetic-prompt artifact.
- **Cause 1 — cold Triton autotune**: fla `chunk_fwd`/`chunk_delta_h` autotune keys
  include BC (chunk count), so every novel extend length sweeps whole config grids
  *inside the request's forward*. Measured first hits: 46s (T=575), 70s (266), 129s
  (255), 270s (505). Identical replays 3-6s (`cache_results=True` persists winners).
  The pre-existing engine warmup covered only [80,128,256,512,1024] — no ±1
  shoulders, no decode step, no depth buckets.
- **Cause 2 — page-cache eviction of the mmap'd expert banks** (checkpoint on the
  shared HDD): the raw-socket stall sentinel caught a 135.5s probe while the worker
  faulted **81MB at 0.6MB/s** behind concurrent desktop IO. Stalls correlate with
  the user's disk activity, not idle time / shapes / client stack. Depth walks ran
  disk-bound (~250 tok/s) vs ~2000+ page-cache-warm.

**Fixes (all in 4bfafb9, verified on ridge67).**
- `Scheduler._prefill_warmup` (launch.py calls it BEFORE the ready ack): 33-class
  extend ladder (1..16, 2^k ±1, max_extend) each +1 decode step, then a chunked
  depth walk to serving depth — through the real request path. Env:
  `FREETOKEN_PREFILL_WARMUP=1`, `FREETOKEN_WARMUP_MAX_DEPTH`.
- `utils/prefetch.py`: `posix_fadvise(WILLNEED)` the whole checkpoint at boot +
  `KeepResident` re-advises every 120s (`FREETOKEN_PREFETCH_MODEL`,
  `FREETOKEN_KEEP_RESIDENT_S`). No steady-state disk cost.
- `scripts/stall_sentinel.py`: 45s raw-socket probes with api/worker read_bytes
  deltas; >10s responses snapshot per-thread state+wchan+io (py-spy is
  ptrace_scope-blocked). This is what caught cause 2.
- Results: x14 503s->12.5s(cache filling)->~3s; x13 46->3.0; para 70->2.5; 8.5k
  prompt end-to-end 11.4s (~750 tok/s wall incl. queue+detok); engine warmup
  230->13s on second boot; depth walk 280->149s.

**Parked: >=~50k-context HSA hardware exception.** ridge65's full-depth (73728)
warmup walk crashed the worker at KV usage 0.68 (~52k) with
`HSA_STATUS_ERROR_EXCEPTION` — no Python traceback (async fault). 40960 walks
clean (ridge66/67). No traffic has ever exceeded ~40k, so this is a *new* latent
bug on the path to the 72k target, not a regression. Next: diagnostic boot with
`AMD_SERIALIZE_KERNEL=3` + full-depth walk to name the faulting kernel; suspects
are the fla deep-BC grids or an autotune config that OOBs at deep context.

**New host trap**: an exported `PYTHONPATH` with a trailing empty entry breaks
uv-venv prefix detection (`sys.prefix` resolves to the base interpreter, venv
site-packages vanishes). `env -u PYTHONPATH` for every venv-python invocation.

## 2026-08-29 (later): fadvise was not enough — activity-gated residency (29e5988)

The fadvise prefetch lost to real desktop load: page cache drained 24GB -> 5.6GB,
refill faults crawled at 0.2-0.6MB/s behind video/desktop IO (sentinel: 116.7s
probe stall, 21MB faulted; a live maestro request wedged 9+ minutes mid-prefill).
Final design per user contract — aggressive while serving, silent when idle:
- **SSD staging**: one boot-time copy of the checkpoint to the SN850X (95s), served
  from there forever; refills hit NVMe even when RAM is drained. Resolution MUST
  run before Scheduler(args) maps the file; ServerArgs is frozen (object.__setattr__).
- **mlock2(MLOCK_ONFAULT)** while requests run + 300s idle grace; munlock past it.
  Live-verified: VmLck 11.7GiB active -> 0 kB idle; "no background IO while idle".
- Sentinel is passive-by-default now (active probes = permanent activity = pins
  forever — it violated the idle contract itself).
- Numbers, pinned+SSD: 8.5k cold 13.8s (~615 tok/s true compute; 280 was disk
  contention interleaved), cached re-serve 3.3s. Remaining TTFT lever for very
  deep prompts is pure compute (FIX 2) + the parked >=50k HSA fault.

## 2026-08-29 (diag): >=50k HSA fault REPRODUCED deterministically; attribution next

`scripts/hsa_deep_diag.sh 73728` (AMD_SERIALIZE_KERNEL=3 + HSA_ENABLE_ASSERTS=1):
crash at KV usage **0.68-0.69 (~52k)** — matches ridge65's 0.68. Facts:
- HSA_STATUS_ERROR_EXCEPTION 0x1016 (HSAIL op -> hw exception), async queue abort;
  serialized launches did NOT name the kernel, dmesg clean (user-mode queue
  exception, not a driver page fault).
- Walk ran 170-184 tok/s under serialization; crash ~10 min into boot.
- Crash is in the depth-walk prefill path only; <=40960 walks clean on every boot.
Next probe (fresh session): rerun the walk under rocprofv2/ROCTx to name the last
kernel before the abort; prime suspects are fla chunk intermediates scaling with
BC (~812 chunks at 52k: A/M/state buffers) overflowing GPU1's 16GB (far side =
8 layers + head + its KV share), or a Triton kq_gemv int32 stride overflow past
~52k rows. Both are code-readable once the kernel is named.

## 2026-08-29 (final): ">=50k HSA fault" CLOSED — warmup seq-len boundary violation (f4894e3)

The bisect ladder told the story: walks to 49152/55296/58368 under max_seq_len
73728 completed; 73728-walks crashed at walk-end (0.68x108k pool = 73.4k);
61440-walk with max_seq_len_override=61440 crashed at 0.57x108k = 61.5k; the
SAME 61440-walk under a 73728 cap COMPLETED (boot H). Trigger = decoding at
position == max_seq_len: warmup submits via add_one_req, bypassing the
admission guard that clamps input+max_tokens for real traffic — so the walk's
2 decode steps sampled one past the rope/positions buffers (async HSA 0x1016,
no kernel attribution, exactly matching the rocprofv2 evidence that every
dispatched kernel COMPLETED and the abort followed a silent gap).

Fix: warmup walk length = min(env, engine.max_seq_len - 4); honest INCOMPLETE
log instead of a false success line. Regression verified: boot-C config now
completes (61436-token walk, 0 aborts). Real traffic was never exposed.

Consequences: the parked "72k blocker" never existed for serving; lift
FREETOKEN_WARMUP_MAX_DEPTH to 73728-8 in production boots to compile
deep-context kernels at boot. Also from this hunt: /tmp is 32GB tmpfs (the
4.5GB rocprof trace + 11.7GB mlock pins during diagnostics likely OOM-killed
the user's editor session) — artifacts to disk only, FREETOKEN_RESIDENCY=0 on
diag boots.

## 2026-08-29 (wave 2 close): OOM forensics exonerate the server; cache gardener ships (5ffb642)

Kill 4 happened with the server fully UNPINNED — kernel journal shows both
global-OOM kills targeted `rustc` builds at 11-13.5GB RSS (cargo codegen
units). Coordination ask sent: bound cargo (CARGO_BUILD_JOBS, codegen-units).
Posture now: mlock pins default-OFF (opt-in via FREETOKEN_RESIDENCY=1, with
the da05987 pressure guard), warmup walk capped at 40960 unless explicitly
raised, and the **cache gardener**: rotating 512MB fadvise(WILLNEED) windows
over the staged checkpoint at <=34MB/s sustained — keeps the page cache warm
with ZERO unpageable bytes, pauses under MemAvailable < 16GB.

Measured (ridge72, gardener on, pins off): 8.5k novel prompt 12.9s wall
(= the pinned config's 13.8s), late chunks at 512 tok/s (full-residency
compute rate), one residual dip on the first request while the gardener
catches up. Decode at ~9k depth ~16-32 tok/s steady.

Scoreboard vs targets: prefill compute ~512-615 tok/s (target 800: remaining
levers are fla GDN tuning + chunk-size A/B, both measured work); decode at
depth unchanged (S2c graphs + S4 DFlash2 are the levers for 60+ @ 72k);
system contract now holds on a loaded box (no unpageable memory, bounded IO,
pressure-yielding everything).

## 2026-08-30 (wave 3): FIX 2 step 1 — chunk 2048 lands (+40% prefill); twin cache OOMs

A/B on the 27B Ridge split config (gardener on, pins off, 40k warmup):
- **chunk 2048 (keeper)**: sustained prefill 347-382 tok/s (vs 264 @1024, +40%);
  31k prompt 126.3s wall; depth walk to 40960 in 110s (was 208-478s); final
  chunks 742 tok/s. Warm rerun of a cached prompt: 2.5s. This is now the
  serving recommendation: --max-extend-length 2048 --max-prefill-length 2048.
  User maestro corroboration: first prompt 158s (was 416s), follow-up 14.1s.
- **bf16-twin cache FREETOKEN_BF16_CACHE_GB=1024: CUDA OOM at boot** — at
  memory_ratio 0.90 the KV+weights+banks leave only ~600MB slack on GPU0;
  the per-use dequant (20-40ms/layer) can only be attacked after a VRAM
  budget redesign (lower KV budget, MoE bank sizing, or dequant-free GEMV
  path). Parked with that scope note.

Remaining prefill gap to 800+: fla GDN chunk kernel tuning (BT/BK grids) and
the dequant path. Decode to 60+: unchanged levers (S2c graphs, S4 DFlash2).

## 2026-08-30 (wave 4 close): bf16 prefill gate PASSED on correctness; streaming wall confirmed sole prefill limiter

A/B on real weights (identical greedy prompt, fresh boots, split56/2048):
- Outputs **byte-identical** (239/239 chars) — the double-index fix holds; the
  path is numerically sound end-to-end.
- Sustained wall: 375 tok/s (6151-token novel prompt, 16.4s) — same as the MMQ
  baseline. The GEMM win is masked exactly as the profile predicted: the chunk
  wall is expert streaming. Smoking gun in the same run: one chunk whose layer
  banks were already GPU-resident (identical para → same experts routed) ran
  at **9556 tok/s** — when streaming isn't the bottleneck the bf16 path is
  ~10x the sustained rate.
- Config: bf16 path stays ON (FREETOKEN_MOE_BF16_PREFILL_MIN_T=64 default).

Wave-5 target = the streaming wall itself: per-chunk whole-layer bank
materialization is ~5GB/chunk (48 layers x ~110MB) through PAGEABLE staged
copies (~1-3GB/s) = the ~3.5s/chunk floor. Levers: pinned staging buffers,
keep-previous-chunk banks (natural text re-routes the same experts → D2D
hits, the moe_prefill_hit_d2d flag exists), or per-chunk delta streaming.
