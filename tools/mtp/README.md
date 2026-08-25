# MTP acceptance / debugging harness (qwen3_5_moe GGUF)

Tools from the 2026-08-24 acceptance bring-up (hand-rolled, driven by env, no
server needed except where noted). All assume `FREETOKEN_MTP=1` and the venv
install of the fork.

- `offline_draft_replay.py` — reference draft forward in pure dequantized
  bf16 from raw GGUF bytes (CPU). The ground truth for every live-stage diff.
  Reads `/tmp/mtp_trace/step_*.pt` tuples (set FREETOKEN_MTP_TRACE=<dir> and
  FREETOKEN_MTP_PROBE=1 on the server to capture).
- `stage_diff.py` — per-stage live-vs-offline comparison from the probe's
  `mid` dict (x0 / attn / moe-in / router / routed / shared / h2); the first
  divergent stage localizes a live bug exactly.
- `kernel_test.py`, `fetch_test.py` — standalone ggml-file offload cache +
  slot-fetch + `fused_experts_ggml_split` reproduction (GPU).
- `chain_measure.py` — chain-acceptance depth decay: after each real traced
  step the bf16 draft engine chains up to k=4 (its own hidden recurs as the
  h input); scores each depth against the actual future tokens. 2026-08-24
  Ornith result: p1=0.40, p2=0.10, p3≈p4≈0.02; E[tokens/verify]≈1.52 (k=4).

GGUF orientation fact (learned the hard way): expert rows are
[ne1 = out-rows][ne0 = in-elements] — `ffn_gate_exps` dims are
[2048 (H, in), 512 (I, out), 256 (E)]; transpose nothing; `_to_bf16()` already
puts tensors in torch [out, in] shape via `dims[::-1]`.
