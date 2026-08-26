# DFlash2 bring-up (S3/S4/S5) — spec decode for the qwen35 trunks

Goal: block-verify speculative decode on Ridge (primary; Fable deferred until
Ridge + DFlash2 hit target tok/s per user). Draft head: incoai
Qwen3.8-27B-DFlash2-Q8_0 (2.06 GiB, arch `dflash`, 81 tensors). User hypothesis:
same base model ⇒ existing head transfers across trunk finetunes. If acceptance
is below break-even, fallback = train a Ridge-matched head (unsloth, hardware
available). Success gate: measured tok/s ≥ published-reference multiplier vs
the S2b eager baseline (24.4 tok/s decode).

All facts below are READ from sources: GGUF metadata via the fork reader,
reference semantics from `Fork/dflash-reference/dflash/model.py` (z-lab/dflash).
No guessing. Sections marked ⚠ are ambiguities resolved empirically at S3-parity.

## Draft checkpoint inventory (GGUF, ggml order `[in?, out?]` as stored)

Config (`dflash.*`):
- block_count 5 · H 5120 · FFN 17408 · attn 32q/8kv × head_dim 128
- causal **False** · sliding_window 2048 · sliding_window_pattern [True×5]
- rope freq_base 1e7 (matches trunks) · eps 1e-6
- block_size **8** · conv_kernel_size 2 · conv_group_size 16
- selector_rank 256 · selector_top_k **16**
- **target_layers [6, 20, 34, 48, 62]**
- context_length 262144

Tensors per block N∈0..4 (all projections Q8_0, norms F32):
- attn_q (5120→4096), attn_k (5120→1024), attn_v (5120→1024),
  attn_output (4096→5120); attn_q_norm/attn_k_norm F32(128);
  attn_norm/ffn_norm F32(5120)
- ffn_gate (5120→17408), ffn_up (5120→17408), ffn_down (17408→5120)
- attn_conv_base F32 (5120,2,2) · attn_conv_proj (5120→1280)
- ffn_conv_base F32 (5120,2,2) · ffn_conv_proj (5120→1280)

Global: fc.weight Q8_0 (25600→5120 = 5×H → H); output_norm F32(5120);
enc.output_norm F32(5120); selector_hidden (5120→256);
selector_predecessor (256↔248320 codebook); selector_successor (same).
NO embedding table and NO lm_head in the draft — both borrowed from TARGET.

## Reference math (z-lab model.py — source of truth)

Draft forward (`DFlash2DraftModel.forward` / generate loop lines 179-323):
1. `noise_embedding = embed(block_tokens) * input_embedding_scale` where
   block = [anchor, MASK×k−1], embeddings from the TARGET's token_embd.
2. `ctx_feat = hidden_norm(fc(cat(taps)))`, taps = trunk hidden_states at
   `layer_id+1` (HF offsets by embedding row 0) for each of the 5 ids.
3. Block stack over noise hidden: every layer is NON-CAUSAL bidirectional;
   K/V = cat(k/v_proj(ctx_feat broadcast-attendees), k/v_proj(noise)) so each
   draft position attends BOTH to the projected trunk context AND its peer
   proposals (+draft KV cache of past steps). All 5 layers windowed ±2048.
   RoPE base 1e7, per-head RMSNorm q/k (eps 1e-6).
4. Dynamic convs sandwich attention AND ffn inside each layer
   (`prepare` before, `finish` after): grouped dynamic causal conv,
   kernel generated per-position by `kernel_projection(x)` + static
   `base_kernel[0|1]` half per stage; `_grouped_dynamic_convolve`
   iterates offset<kernel with left-padding, addcmul dynamic term,
   group_size=16 over H.
5. `propose`: logits = TARGET lm_head(final_norm(hidden)); top_k=16 candidates
   per position; iterative chained pick (temp 0 → argmax):
   score(pos,k) = unary_topk(pos,k) + Σ_r pred_cb(anchor_chain)[r] · hproj(h[pos])[r] · succ_cb(cand[pos,k])[r]
   predecessor chain starts at the anchor token id.
6. Verify (greedy): trunk forward over [anchor + c1..ck]; longest-prefix
   argmax accept, bonus = posterior at accept point; crop trunk KV to
   accepted prefix; refresh taps from that forward's hidden_states.

Generate-loop bookkeeping we must mirror (line refs above): draft cache lives
across steps; verify crops it to `start`; position_ids passed to draft cover
`start − len(target_hidden) : start + verify` (positions ALIGN with the tap
window, not from zero); after accept, taps = hidden_states[:, :produced].

Ambiguity 1 RESOLVED by parity (2026-08-26): `output_norm` = fc-context
hidden_norm; `enc.output_norm` = FINAL pre-selector norm (llama.cpp naming
convention is inverted here vs their field names). Locked in the loader.

⚠ Ambiguity 2: `selector_predecessor/successor` codebooks — nn.Embedding
weight is [vocab, rank]; raw t.shape prints (256, 248320) (in-fastest-first).
Dequant then reshape so rows=tokens, assert rstride==256 exactly.

⚠ Ambiguity 3: conv `base` ggml layout (5120,2,2) vs Parameter(2, k=2, H):
expect reshape(H,2,2)-permute → (2,2,H); settled numerically at parity.

## FreeToken-side plan

S3a loader: `freetoken/models/dflash/` — parse config from dflash.* keys;
iter_weights dequantizing Q8_0 → bf16 modules standing alone (draft engine is
GPU-resident bf16; NO ext kernels needed on this path ⇒ immune to the IQ/MMQ
trap); registry key `dflash`; tokenizer map reuse qwen2.
S3b module: standalone torch implementation of the reference forward above
(bf16 GPU1 resident ≈ 3.9 GiB weights at bf16; fits far side budget).
S3c parity harness (offline, tools/mtp/dflash_parity.py): feed identical
synthetic taps/noise into z-lab PyTorch impl (their classes imported straight
from Fork/dflash-reference path w/ HF available) and ours; require max-cos ≥
0.9999 per component incl. proposal paths; resolves ⚠1/2/3 by construction.
S3d acceptance probe vs Ridge trunk offline replay (mirror
tools/mtp/offline_draft_replay.py pattern used for Ornith MTP): greedy
trunk continuation corpus → mean accepted-prefix length per verify k∈{4,5,8}.
Break-even for the free-GPU1 draft ≈ >35-40% first-token-equivalent
acceptance (measured ledger math, memory file).

S4 engine integration: tap buffers stashed in Qwen3_5Model.forward per depth
(taps 6/20/34/48 near side, 62 FAR SIDE under split=56 — cross taps to dev1
once per step through the pinned two-hop; draft engine already planned to live
on dev1 which also owns the head/logits channel → propose consumes Ridge's
lm_head output directly, no extra crossing for logits). Verify loop =
existing batch prefill machinery with fixed shape k ⇒ graph-friendly later;
KV crop; GDN state snapshot/rollback reusing gdn.py snapshot pattern.

S5 sweep: k ∈ {4,5,8} greedy + sampled temp≥0.7 (rejection-sampler variant for
non-greedy ports from reference lines 94-124); targets published arithmetic /
conversational multipliers at bs=1, ~1.0× under load.

## Status log

- 2026-08-26: config + tensor inventory decoded; reference math captured;
  doc written. Loader next.

## Status log (cont.)

- 2026-08-26 (2): **S3 PARITY GATE PASSED.** Full-forward parity vs the z-lab
  reference classes: per-attention standalone cos 0.99999; end-to-end hidden
  cos ~0.997 (bf16 accumulation drift, max 1e-4 of activation scale);
  chained greedy SELECTOR PROPOSALS EXACTLY EQUAL under the locked norm
  mapping. Attention subtlety that cost an hour: reference concatenates
  K as [ctx, noise] (their line 388) and rotates q against the table's last
  q_len rows — probes must mirror this exactly or cos collapses to ~0.84.
  RoPE contract for our engine forward: position_ids length MUST equal
  Tc + Tq (ctx rows first, then noise rows), matching their
  `position_ids[start - ctx_len : start + verify]`.

## Status log (cont.)

- 2026-08-26 (3): **S3d ACCEPTANCE MEASURED — head transfer VERDICT: GO.**
  Capture: split trunk taps [6,20,34,48,62] + greedy continuation over 1383
  decode steps / 7 prose prompts (tools/mtp via FREETOKEN_DFLASH_TAPS +
  TAP_DUMP; pinned-staging per-depth buffers — raw `.to("cpu")` in the far
  ctx hard-faults, and unpickled views serialized whole pinned storages ->
  always clone). Replay (CPU, fp32): teacher-forced chained-selector proposals
  vs recorded continuations:
    k=4 mean acc 1.858, P(acc>0) .762, E[tok/verify] 2.86
    k=5           2.131            .807                 3.13
    k=8           2.704            .837                 3.70
  Projection from the 24.4 tok/s eager baseline (draft overlapped on GPU1):
  ~70-90 tok/s — squarely in the published-DFlash2 multiple band WITHOUT any
  Ridge-specific training. Remaining gap to the paper's 3.43x: prefill chunk
  graphs (S2c) raise the base and shrink verify cost share.
