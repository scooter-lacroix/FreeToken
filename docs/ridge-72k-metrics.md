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
