"""S4 live DFlash2 draft service — the acceptance-verified proposal algorithm
(replay contract from tools/mtp/dflash_acceptance.py, S3d E[tok/verify]=3.70 @ k=8)
as an in-process engine on the SECOND GPU.

Device contract (S4 layout): trunk on cuda:0 (7900 XTX via CUDA_VISIBLE_DEVICES=1,0),
draft on cuda:1 (7800 XT). Every cross-device transfer is explicit and small:
taps [5,H] bf16 ~51 KB, top-64 ids/vals, one embedding row. No trunk weights are
materialized on the draft device — the unary side is the trunk's OWN top-64 logits
recorded at the anchor row, exactly what the replay consumed.

propose(anchor_id, position, taps, top_ids, top_vals) -> list[int] of k picks.
Threaded through the scheduler by the S4 verify loop (spec.py).
"""
from __future__ import annotations

import os
import time

import torch

GGUF_DRAFT = "/mnt/HDD-2/Models/incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf"
MASK_ID = 248070
VOCAB = 248320
DEFAULT_K = 8


class DFlashService:
    def __init__(self, k: int = DEFAULT_K, device: str = "cuda:1"):
        from freetoken.models.dflash.gguf import iter_dflash_weights, parse_dflash_config
        from freetoken.models.dflash.model import DFlash2Draft
        from freetoken.models.gguf.reader import load_gguf_metadata

        self.k = k
        self.device = torch.device(device)
        cfg = parse_dflash_config(load_gguf_metadata(GGUF_DRAFT))
        self.tap_layers = list(cfg["target_layers"])
        self.draft = DFlash2Draft(cfg, vocab_size=VOCAB)
        self.draft.load_state_dict(dict(iter_dflash_weights(GGUF_DRAFT)), strict=True)
        self.draft.eval().to(self.device)
        # float32, matching the S3d replay formulation exactly (bf16 storage is a
        # post-parity optimization; the encoder is 5 small layers).
        self.sel = self.draft.candidate_selector
        # MASK embedding gathered per-call from the trunk table (quant type varies);
        # only the anchor+MASK rows are ever needed, dequantized trunk-side.
        self.n_propose = 0
        self.ms = 0.0

    @torch.inference_mode()
    def propose(
        self,
        anchor_id: int,
        position: int,
        taps: dict[int, torch.Tensor],      # {tap_layer: [H] hidden on trunk device}
        top_ids: torch.Tensor,              # [64] trunk logits top-k ids (trunk device)
        top_vals: torch.Tensor,             # [64] trunk logits top-k vals (trunk device)
        embed_row: torch.Tensor,            # [H] anchor embedding row, bf16, trunk device
        mask_row: torch.Tensor,             # [H] MASK embedding row, bf16, trunk device
    ) -> list[int]:
        """One chained-greedy proposal. Bit-identical formulation to the S3d replay:
        noise rows [anchor, MASK*(k-1)] embedded TRUNK-side, context taps at the anchor
        position, rope window ending at the anchor position, selector bilinear scoring
        over the trunk's top-64 unary."""
        t0 = time.perf_counter()
        dev = self.device
        noise = torch.stack([embed_row, mask_row]).to(dev, non_blocking=False)  # reuse mask k-1x below
        noise = torch.cat([noise[:1], noise[1:2].expand(self.k - 1, *noise.shape[1:])])
        noise = noise.unsqueeze(0)                                            # [1,k,H]
        th = torch.stack([taps[d].reshape(-1) for d in self.tap_layers]).reshape(1, 1, -1)
        th = th.to(dev, dtype=torch.float32, non_blocking=False)              # [1,1,5H]
        pos = torch.arange(position - 1, position + self.k, device=dev)
        h, _kv = self.draft(noise.float(), th, pos)
        proj = self.sel.hidden_projection(h[:, :]).float()                    # [1,k,rank]
        cbp = self.sel.predecessor_codebook.weight
        cbs = self.sel.successor_codebook.weight
        ids = top_ids.to(dev, non_blocking=False).view(-1)
        vals = top_vals.to(dev, dtype=torch.float32, non_blocking=False).view(-1)
        cand = cbs[ids].float()                                               # [64,rank]
        pred = int(anchor_id)
        picked = []
        for pi in range(self.k):
            scores = vals + (cbp[pred].float() * proj[0, pi].float()) @ cand.T
            pick = int(ids[int(scores.argmax())])
            picked.append(pick)
            pred = pick
        self.n_propose += 1
        self.ms += (time.perf_counter() - t0) * 1000.0
        return picked


_svc: DFlashService | None = None


def get_service(k: int = DEFAULT_K) -> DFlashService | None:
    """Singleton; None unless FREETOKEN_SPEC_K>0 and a second device exists."""
    global _svc
    if _svc is not None:
        return _svc
    if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
        return None
    try:
        _svc = DFlashService(k=k)
        print(
            f"[dflash-svc] armed on {_svc.device} (k={k}, "
            f"taps={_svc.tap_layers})", flush=True,
        )
    except Exception as e:                                  # noqa: BLE001
        print(f"[dflash-svc] FAILED to arm: {e!r}", flush=True)
        _svc = None
    return _svc
