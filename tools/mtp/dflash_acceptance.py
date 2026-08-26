"""S3d DFlash2 acceptance probe — teacher-forced replay of a capture dump.

Reads the FREETOKEN_DFLASH_TAP_DUMP pickle-lines written by a capture server
run ({position, taps{depth:[H] bf16}, argmax, topk(64)}) and replays the
reference generate-loop's PROPOSAL side only: anchor chain proposals for
k∈4/5/8 are teacher-verified against the recorded greedy continuation.
No trunk re-run needed — that is what makes this runnable on CPU alongside
the live server (the encoder is 5 small layers; the head's unary logits come
from the recorded topk).

Usage:
    python tools/mtp/dflash_acceptance.py <dump.pkl> [--steps N]
"""
import argparse
import pickle
import sys

import torch

sys.path.insert(0, "/home/scooter/Documents/Product/Stan-s-ML-Stack/Fork/FreeToken/python")

GGUF_DRAFT = "/mnt/HDD-2/Models/incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf"
MASK_ID = 248070


def load_dump(path):
    """Pickle-lines -> list of CONSECUTIVE runs (one per captured prompt; the
    position counter restarts at each prefill, so runs break there and block
    windows must not straddle a boundary)."""
    raw = []
    with open(path, "rb") as f:
        while True:
            try:
                raw.append(pickle.load(f))
            except EOFError:
                break
    runs, cur = [], [raw[0]]
    for prev, e in zip(raw, raw[1:]):
        if e["position"] == prev["position"] + 1:
            cur.append(e)
        else:
            runs.append(cur)
            cur = [e]
    runs.append(cur)
    return runs


def _split_runs(entries):
    runs, cur = [], [entries[0]]
    for prev, e in zip(entries, entries[1:]):
        if e["position"] == prev["position"] + 1:
            cur.append(e)
        else:
            runs.append(cur); cur = [e]
    runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--steps", type=int, default=0, help="limit decode steps")
    args = ap.parse_args()

    from freetoken.models.dflash.gguf import parse_dflash_config, iter_dflash_weights
    from freetoken.models.dflash.model import DFlash2Draft
    from freetoken.models.gguf.reader import load_gguf_metadata

    md = load_gguf_metadata(GGUF_DRAFT)
    cfg = parse_dflash_config(md)
    tap_layers = list(cfg["target_layers"])

    print("loading draft (bf16, CPU) ...", flush=True)
    draft = DFlash2Draft(cfg, vocab_size=248320)
    draft.load_state_dict(dict(iter_dflash_weights(GGUF_DRAFT)), strict=True)
    draft.eval()

    runs = load_dump(args.dump)
    if args.steps:
        kept = []
        for r in runs:
            kept.extend(r[: args.steps])
        runs = _split_runs(kept)
    total = sum(len(r) for r in runs)
    print(f"capture: {len(runs)} consecutive runs, {total} steps "
          f"(first run {len(runs[0])} steps)", flush=True)

    # token embeddings for the [anchor | MASK...] noise block come from the
    # TARGET trunk's embedding table (Ridge): gather once from GGUF.
    from gguf import GGUFReader

    emb_t = None
    for t in GGUFReader(TARGET_GGUF).tensors:
        if t.name == "token_embd.weight":
            ne = [int(d) for d in t.shape]
            from freetoken.models.gguf.reader import iter_gguf_tensors

            gt = next(
                g
                for g in iter_gguf_tensors(TARGET_GGUF)
                if g.name == t.name
            )
            pk = gt.packed()
            from freetoken.models.gguf.dequant import dequantize

            emb_t = dequantize(pk.reshape(-1), gt.ggml_type, torch.bfloat16).reshape(
                gt.shape)
            break
            break
    assert emb_t is not None
    scale = 1.0  # input_embedding_scale default

    def embed(ids: torch.Tensor) -> torch.Tensor:
        return emb_t[ids] * scale

    results = {}
    for k in (4, 5, 8):
        accepted_hist = []
        for entries in runs:
          s = 0
          while s + k + 1 < len(entries):
            ent = entries[s]
            future = [entries[s + i]["argmax"] for i in range(k)]
            # anchor == the token just emitted at this step
            anchor_id = int(ent["argmax"])
            ids = torch.tensor([anchor_id] + [MASK_ID] * (k - 1), dtype=torch.long)
            noise_emb = embed(ids).to(torch.bfloat16)[None]
            th = torch.stack(
                [ent["taps"][d].reshape(-1) for d in tap_layers], dim=0
            )[None].permute(1, 0, 2).reshape(1, 1, -1).to(torch.bfloat16)  # [B,Tc=1,5H]
            # rope contract covers ctx(Tc=1) THEN noise(k): end at the
            # current position so queries take the trailing k rows
            p0 = int(ent["position"])
            pos = torch.arange(p0 - 1, p0 + k)
            with torch.inference_mode():
                h_final, _kv = draft(
                    noise_emb.float(), th.float(), pos)
                import json as _json
                top_ids = torch.tensor(ent["topk_ids"], device="cpu").view(1, -1)
                top_vals = torch.tensor(ent["topk_vals"], dtype=torch.float32).view(1, -1)

                proj = draft.candidate_selector.hidden_projection(h_final[:, :]).float()
                pred = anchor_id
                picked = []
                for pi in range(k):
                    # rank-recorded unary: score recorded top64 candidates with
                    # the selector bilinear term, take argmax — equivalent to
                    # reference top-k over full logits when true rank fits 64.
                    pred_vec = draft.candidate_selector.predecessor_codebook.weight[pred].float()
                    cand = draft.candidate_selector.successor_codebook.weight[top_ids[0]].float()
                    scores = top_vals[0] + (pred_vec * proj[0, pi].float()) @ cand.T
                    best = int(scores.argmax())
                    pick_id = int(top_ids[0][best])
                    picked.append(pick_id)
                    pred = pick_id
            acc = 0
            for pi in range(k):
                if picked[pi] == future[pi]:
                    acc += 1
                else:
                    break
            accepted_hist.append(acc)
            s += 1
        hist = torch.tensor(accepted_hist, dtype=torch.float32)
        results[k] = {
            "mean_acc": hist.mean().item(),
            "acc_rate": (hist > 0).float().mean().item() if len(hist) else 0.0,
            "tokens_per_verify": 1 + hist.mean().item(),
        }
        print(f"k={k}: mean accepted prefix {hist.mean():.3f} | "
              f"P(acc>0) {(hist > 0).float().mean().item():.3f} | "
              f"E[tokens/verify] {1 + hist.mean().item():.3f} "
              f"({len(hist)} windows)")

    base_tok_s = 24.4  # measured Ridge split56 eager decode baseline
    print("\nprojected (GPU1-resident draft assumed overlapped):")
    for k, r in results.items():
        ideal = min(base_tok_s * r["tokens_per_verify"], 10_000)
        print(f"  k={k}: {base_tok_s:.1f} -> up to ~{ideal:.0f} tok/s")


TARGET_GGUF = "/mnt/HDD-2/Models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf"

if __name__ == "__main__":
    main()
