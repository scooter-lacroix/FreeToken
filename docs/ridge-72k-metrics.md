# Ridge serving metrics — 72k-context verification session (2026-08-27)

Build: fork feature/rocm @ 5811e0a+ (tokenizer specials fix, tool-format
auto-sniff, prefill bf16-twin fast path, per-use dequant default, split=56
dual-GPU). Server: `--max-seq-len-override 73728 --max-running-requests 1
--cuda-graph-max-bs 0` (eager; split-aware decode graphs are the next lever).

## Decode tok/s vs KV depth (single stream, greedy, measured n=3-9 windows/point)

| depth | tok/s |   | depth | tok/s |
|---|---|---|---|---|
| 4k  | 12.3 |  | 40k | 6.5 |
| 8k  | 13.1 |  | 44k | 6.0 |
| 12k | 11.8 |  | 52k | 5.8 |
| 16k | 10.1 |  | 56k | 5.5 |
| 20k | 9.3  |  | 60k | 5.2 |
| 28k | 7.3  |  | 64k | 4.9 |
| 32k | 5.7  |  | 68k | 4.7 |
| 36k | 7.1  |  | **72k** | **~4.6 (extrapolated)** |

17 full-attn layers pay growing KV reads (≈68 KB/token/layer-pass; 72k ⇒
~4.9 GB/pass ≈ 6 ms at bandwidth — small); the 213 ms/step at 68k is
dominated by per-layer host/launch overhead (64 layers eager, split seam
syncs) + constant quant-GEMV weight streaming. Depth decay is therefore
mostly FIXED-overhead × per-step syncs, not attention math — graphs attack it.

## Prompt processing

- Cold prefill ~530 tok/s effective (33k in 62 s measured; bf16-twin path).
- Incremental prefill on fully-cached prefix: **~2,000 tok/s** (4,800 new
  tokens in 2.3 s at the 66k→71k rung) — above LM Studio's 800 headline for
  this case.
- Cold 72k-class prompt projection: ~135–190 s one-time (fits maestro's 300 s
  SSE window; warm retries are seconds).

## DFlash2 (S3+S4 inputs, all measured)

- Acceptance vs Ridge trunk (real taps, 1383 steps): mean accepted prefix
  1.86 / 2.13 / 2.70 and E[tokens/verify] 2.86 / 3.13 / 3.70 at k = 4/5/8.
- Proposal chain cost: **48.4 ms per k=8 step** on cuda:1 (serial selector
  dependency; graph-capture is the shrink lever, target ≤15 ms).
- Live-probe harness in-tree (FREETOKEN_DFLASH_ENGINE) but DISABLED for
  serving after it destabilized a session; measurement now runs offline.

## Path to 60+ tok/s @ 72k (stacked levers, projections)

| lever | effect at 72k | projected |
|---|---|---|
| today | — | 4.6 |
| + split-aware decode CUDA graphs (host/launch removal) | base → 9–12 | |
| + S4 verify loop, k=8, graphed proposal ≤15 ms, KV crop + GDN rollback | E 3.7 on top | **33–44** |
| + draft-overlap pipelining (propose during trunk tail) | hide 12–15 ms | 38–50 |
| + far-tail fusion / per-step sync reduction | +10–20% | **45–60** |

Honest verdict: 60 at 72k needs ALL stacked levers landed and tuned — the
graphs lever is the long pole (earlier capture attempt OOM'd; must capture
near-piece + far-piece separately around the seam).

## Re-baseline 2026-09-01 (ridge178, single-GPU 7900XTX, post radix-cache + ledger fixes)

Method: incremental prefix via radix cache (delta prefill per step), 96 forced decode
steps (ignore_eos), sustained rate = server's last 40-step gen-throughput line.

| depth | tok/s | | depth | tok/s |
|---|---|---|---|---|
| 0.08k | 29.0 | | 33.8k | 9.7 |
| 2.9k | 25.3 | | 36.6k | 9.2 |
| 5.7k | 19.2 | | 39.3k | 8.3 |
| 8.5k | 19.9 | | 41.8k | 8.3 |
| 11.3k | 17.4 | | 44.2k | 7.8 |
| 14.1k | 15.8 | | 46.7k | 7.5 |
| 16.9k | 14.5 | | 49.2k | 7.1 |
| 19.7k | 13.8 | | 51.6k | 7.0 |
| 22.5k | 12.1 | | 54.1k | 7.0 |
| 25.3k | 11.5 | | 56.6k | 6.6 |
| 28.1k | 10.8 | | 59.0k | 6.6 |
| 30.9k | 10.2 | | 61.4k | 6.3 |
| | | | 63.9k | 6.1 |
| | | | 66.3k | 5.9 |

vs 08-27 eager baseline (12.3 @ 4k -> 4.7 @ 68k): mid-range +50-100%, deep end +26%.
Extrapolated 72k ~ 5.5. Practical prompt ceiling this config: ~66.3k (admission 400
above ~68.8k; num_pages 67527, kv_reserve 2048).

Config note: serving is SINGLE-GPU now (tp=1, CUDA_VISIBLE_DEVICES=1 = 7900 XTX in HIP
enumeration). The 2-GPU split56 seam (near/far pieces, dev twins) is historical -- S2c
graph work = plain single-device capture of the decode step (cuda_graph_max_bs>=1),
no seam machinery. Old split-only failure modes (stream mismatch across devices,
far-segment dev0 scratch) do not apply; what remains is ROCm capture legality of the
vendored ggml/fla launches (capture under torch.cuda.graph, graphs OFF today:
cuda_graph_max_bs=0).
