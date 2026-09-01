"""S4 draft service vs the S3d replay formulation: identical picks on identical inputs.

The replay (tools/mtp/dflash_acceptance.py) teacher-verified E[tok/verify]=3.70 @ k=8
against real Ridge continuations — its exact proposal math is the contract. The service
must reproduce it bit-for-bit (same device, same dtype) before it feeds a live verify
loop. Synthetic taps/ids/vals: we test FORMULATION equality, not acceptance (that was
S3d's job and needs a real capture).
"""
import sys

import torch

sys.path.insert(0, "/home/scooter/Documents/Product/Stan-s-ML-Stack/Fork/FreeToken/python")

GGUF_DRAFT = "/mnt/HDD-2/Models/incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf"
MASK_ID = 248070
VOCAB = 248320
K = 8


def _load_draft():
    from freetoken.models.dflash.gguf import iter_dflash_weights, parse_dflash_config
    from freetoken.models.dflash.model import DFlash2Draft
    from freetoken.models.gguf.reader import load_gguf_metadata

    cfg = parse_dflash_config(load_gguf_metadata(GGUF_DRAFT))
    d = DFlash2Draft(cfg, vocab_size=VOCAB)
    d.load_state_dict(dict(iter_dflash_weights(GGUF_DRAFT)), strict=True)
    return d.eval(), cfg


def test_service_matches_replay_formulation():
    from freetoken.models.dflash.service import DFlashService

    draft, cfg = _load_draft()
    tap_layers = list(cfg["target_layers"])
    g = torch.Generator().manual_seed(1234)
    H = 5120
    taps = {d: torch.randn(H, generator=g).to(torch.bfloat16) for d in tap_layers}
    anchor_id = int(torch.randint(0, VOCAB, (1,), generator=g))
    position = 4321
    order = torch.randperm(VOCAB, generator=g)[:64]
    top_ids = order.to(torch.int64)
    top_vals = torch.randn(64, generator=g) * 4  # logit-scale
    emb = torch.randn(VOCAB, H, generator=g).to(torch.bfloat16)

    # --- replay formulation (verbatim structure from dflash_acceptance.py) ---
    ids = torch.tensor([anchor_id] + [MASK_ID] * (K - 1), dtype=torch.long)
    noise_emb = (emb[ids]).to(torch.bfloat16)[None]
    th = (
        torch.stack([taps[d].reshape(-1) for d in tap_layers], dim=0)[None]
        .permute(1, 0, 2)
        .reshape(1, 1, -1)
        .to(torch.bfloat16)
    )
    pos = torch.arange(position - 1, position + K)
    with torch.inference_mode():
        h_final, _kv = draft(noise_emb.float(), th.float(), pos)
        proj = draft.candidate_selector.hidden_projection(h_final[:, :]).float()
        pred = anchor_id
        replay_picks = []
        for pi in range(K):
            pred_vec = draft.candidate_selector.predecessor_codebook.weight[pred].float()
            cand = draft.candidate_selector.successor_codebook.weight[top_ids].float()
            scores = top_vals + (pred_vec * proj[0, pi].float()) @ cand.T
            best = int(scores.argmax())
            replay_picks.append(int(top_ids[best]))
            pred = replay_picks[-1]

    # --- service formulation (CPU device for exactness) ---
    svc = DFlashService(k=K, device="cpu")
    svc_picks = svc.propose(
        anchor_id, position, taps, top_ids, top_vals,
        embed_row=emb[anchor_id], mask_row=emb[MASK_ID],
    )
    assert svc_picks == replay_picks, f"divergence: {svc_picks} vs {replay_picks}"
