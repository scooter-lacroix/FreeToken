# Prefill phase profiler

`profile_prefill_chunk.py` loads the 27B offline (split56, chunk 2048, graphs
off), runs a warmup pass, then a torch-profiler pass over a 2048x2-token
prompt. Run it with the standard serving env (FREETOKEN_LAYER_SPLIT=56 etc.);
the server must be stopped (it owns the GPUs).

## Wave-3 findings (2026-08-30, ridge73 config)

- `mul_mat_q4_K` (ggml MMQ, the MoE expert path via `ggml_moe_a8_vec`):
  **94.4% of GPU time** (238 calls x 9.4ms per 2-chunk window). The entire
  fla GDN stack is <1%. Tuning fla is worthless for prefill.
- But: total GPU compute ~= 1.2s of a ~10.8s two-chunk wall (380 tok/s) —
  **~78% of prefill wall is NOT GPU compute**: expert-bank streaming
  (page-cache/PCIe DMA), host sync, launch gaps.
- Wave-4 plan, in order:
  1. Phase-wall instrumentation inside OffloadMoELayer (materialize/gemm/
     rest) — name the 4.3s before optimizing it.
  2. bf16 grouped-GEMM prefill for `fused_experts_ggml_mixed/_split`
     (dequant per streamed expert + rocBLAS over sorted route groups):
     worth ~10x on the GPU-time component (MMQ runs ~1% of peak at T=2048).
  3. Only then fla tuning if it ever shows up.
