"""Chain-acceptance measurement: after each real traced step, let the draft
chain up to K deeper (its own hidden recurs as the h input) and score each
depth against the actual future tokens. Tells us the block-acceptance
economics before building the verify machinery.
"""
import glob
import os
import time
import torch

os.environ.setdefault("FREETOKEN_MTP", "1")

from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config
from freetoken.utils import cached_load_hf_config
from freetoken.models.qwen3_5_moe.mtp_draft import Bf16DraftEngine

MP = "/mnt/HDD-2/Models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf"
K = 4

cfg = parse_gguf_config(cached_load_hf_config(MP))
eng = Bf16DraftEngine(MP, cfg, "cuda:0")

fs = sorted(glob.glob("/tmp/mtp_trace/step_*.pt"))
steps = [torch.load(f, map_location="cpu", weights_only=False) for f in fs]
T = len(steps)
print(f"traces: {T}")

hits = [0] * (K + 1)   # hits[d] = chain-depth-d predictions correct
tot = [0] * (K + 1)
first_accept = [0] * (K + 2)  # distribution of first miss position (1..K, K+1 = all accepted)
nchain = 0

t0 = time.time()
for i, d in enumerate(steps):
    h = d["h"][0].to("cuda:0").bfloat16()
    tok = int(d["token"])
    pos = int(d["pos"])
    # real step: appends KV row i with the true token
    logits = eng.step_async(h, tok, pos, i)
    torch.cuda.synchronize()

    preds = [int(logits.argmax().item())]
    h_chain = eng.last_h2
    # chain deeper from the draft's own stream state
    for dep in range(2, K + 1):
        if i + dep - 1 >= eng.max_pos - 1:
            break
        lg = eng.step_async(h_chain[0], preds[-1], pos + dep - 1, i + dep - 1)
        torch.cuda.synchronize()
        preds.append(int(lg.argmax().item()))
        h_chain = eng.last_h2

    # score against the actual future tokens (steps i+1 .. i+K hold them)
    if i + len(preds) < T:
        for dep, p in enumerate(preds, start=1):
            actual = int(steps[i + dep]["token"])
            tot[dep] += 1
            if p == actual:
                hits[dep] += 1
        # first-miss distribution
        acc = 0
        for dep, p in enumerate(preds, start=1):
            if i + dep < T and p == int(steps[i + dep]["token"]):
                acc += 1
            else:
                break
        first_accept[acc] += 1
        nchain += 1

dt = time.time() - t0
for dep in range(1, K + 1):
    if tot[dep]:
        print(f"p{dep} = {hits[dep]}/{tot[dep]} = {hits[dep]/tot[dep]:.3f}")
print("P(first miss after d accepted):", {d: first_accept[d] for d in range(K + 1)})
print("P(>=1 draft accepted, k<=4):", f"{sum(first_accept[1:]) / max(nchain,1):.3f}")
print(f"chain cost: {(dt - 0) / max(len(steps),1):.2f} s per real step incl {K} chained drafts")
