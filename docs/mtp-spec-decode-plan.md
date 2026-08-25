# MTP scheduler draft-verify loop (M2d) — design + touch points

Measured inputs (Ornith-1.5-35B, file tier, dual-GPU draft, 2026-08-24):

- k=1 greedy acceptance: p1 ≈ 0.26–0.35 (content-dependent; 0.40 on the
  early-generation window), chain decay p2 ≈ 0.10, p3 ≈ 0.02 (chain uses the
  draft's own hidden back — DeepSeek-MTP recursion).
- E[committed tokens / verify step] ≈ 1.40 (k=1), 1.52 (k=4), k=2 already 1.50.
- Draft cost: 0.9 ms enqueue + 2.1 ms GPU1 compute, fully hidden behind an
  19 ms trunk step → draft is FREE; break-even acceptance ≈ 0.
- Therefore every accepted draft token is pure throughput: k=1 ⇒ ~+40%,
  k=2 ⇒ ~+50% (if verify ≈ 1 decode step).

## Hard rules learned (do not re-learn)

- ggml expert rows are [ne1=out rows][ne0=in elements]; never transpose.
- LM head in the draft engine: requantize Q6_K→Q4_K and use our Triton
  kq_gemv. Never run freetoken's ggml C++ ext on the 7800 XT (code object is
  gfx1100-only) and never trust `pip install -q .` — md5-verify the venv.
- ROCm legacy null streams synchronize cross-device: the GPU1 exchange must
  live on private streams; cross-device `.to()` must go via pinned host.
- Graph capture rebinding: any per-step ctx tensor handed across capture
  levels must be a single stable slice-written buffer (see model.py
  `_trunk_prenorm_buf`).

## Architecture

Dual-GPU is the steady state: draft engine on GPU1 produces candidates
asynchronously; trunk on GPU0 verifies in one forward. Same code, single GPU,
falls back to the in-process quantized probe (FREETOKEN_MTP_DRAFT_GPU=0).

Per commit step:

1. Drift: after the trunk commits token `y_n` at position `n`, the engine
   already holds (h_n, y_n) → c1. For k>1 it chains: c_j from
   (last_h2, c_{j-1}) — engine.last_h2 (residual stream, pre head_norm).
2. Scheduler builds a VERIFY batch: this request extends by [c1..ck]
   (positions n+1..n+k) — i.e. input [y_n, c1..ck], k+1 rows.
3. Snapshot the request's GDN per-layer state (recurrent_states[slot] +
   conv_states[slot], ~1 MB across layers — the index_copy pattern at
   `gdn.py:_write_track_snapshot`, but into a per-request pending buffer, not
   the hybrid-radix track slots).
4. Trunk forward (verify): one pass → logits rows a0 (after y_n) .. ak.
5. Accept: commit a0; then while c_i == a_{i-1}, commit a_i. Committed
   tokens total = 1 + longest accepted prefix.
6. If fewer than k accepted: restore the GDN snapshot (index_copy_ back);
   the full-attention KV rows for rejected positions are free-tail and get
   overwritten by the next step (scheduler's cached_len decides).
   If all k accepted: drop the snapshot.

## Verify must be graph-captured (else it regresses)

The chunk path is eager; per estimate an eager 2-row verify over 40 layers
costs ~2× a graphed decode step which erases k=1's +40%. Before flipping the
default: capture piecewise verify graphs for shape [1 req × (k+1)] —
piecewise machinery is shape-agnostic (it captures whatever model.forward
runs at a given Batch); what needs adding is the attn metadata for a >1-row
decode (positions array, plus the GDN snap/restore kernels inside the graph
— index_copy_ ops are capture-safe).

## Touch points (fork)

- scheduler/scheduler.py: speculative mode — build verify batches; accept
  logic in _process_last_data next to the probe hook; snapshot/restore call
  sites around engine._forward.
- engine/graph.py: extra capture pass over verify batch shapes [1 req × k+1].
- models/qwen3_5_moe/gdn.py: expose per-request snapshot/restore helpers
  (pool.recurrent_states[li, slot] index_copy both directions — reuse the
  _write_track_snapshot dtype/copy discipline).
- models/qwen3_5_moe/mtp_draft.py: engine already exposes last_h2 and is
  async; add `chain(k)` producing [c1..ck] in one GPU1 burst after each
  commit instead of per-depth python loops.
- Probe stays ON by default in speculative mode: report realized acceptance
  vs measured tokens/step (verified acceptance telemetry).

## Risks

- GDN rollback correctness if a verify step races prefill of another request
  (snapshot scope is per-request slot — fine, but confirm index ops share the
  forward stream).
- Attention KV for rejected c_i rows: mark the rows free-tail; radix cache
  must not index them (schedule is decompress_before_verify ordering).
- Chained drafts beyond k=2 add ~1 ms GPU1 each (free) but the trunk verify
  cost grows with rows: k=2 is the sweet spot (E=1.50 vs 1.52 at k=4).
- Sampling mode: this is greedy-argmax verify. For temperature>0, swap the
  accept rule for stochastic verify (draft-vs-target distribution) — later.
