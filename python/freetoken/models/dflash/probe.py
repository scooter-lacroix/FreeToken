"""S4 milestone-1 LIVE DFlash2 probe (measure-only; MTPProbe pattern).

FREETOKEN_DFLASH_ENGINE=1 arms a singleton on cuda:1. After every convergent
decode step CausalLM.forward feeds it {taps, argmax}; it proposes a chained
block of up to 8 candidates and scores matured chains against the REAL emitted
stream. Serving outputs are untouched (greedy identical); the log carries
per-report stats: mean accepted prefix / P(acc>0) / E[tokens per verify] /
draft+head ms per step.
"""
from __future__ import annotations

import os
import time
from collections import deque

import torch

GGUF_DRAFT = "/mnt/HDD-2/Models/incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf"
MASK_ID = 248070


class LiveDFlashProbe:
    def __init__(self, cfg, vocab_size: int):
        from freetoken.models.dflash.model import DFlash2Draft

        self.tap_layers = list(cfg["target_layers"])
        self.k = min(8, cfg["block_size"])
        self.vocab = vocab_size
        self.draft = DFlash2Draft(cfg, vocab_size)
        from freetoken.models.dflash.gguf import iter_dflash_weights

        from freetoken.models.dflash.gguf import iter_dflash_weights as _idw

        sd = dict(_idw(GGUF_DRAFT, "cpu"))
        self.draft.load_state_dict(sd, strict=True)
        self.draft.eval().to("cuda:1", dtype=torch.bfloat16)
        # target-side pieces attached by ensure_target():
        self.embed_cpu = None          # bf16 [V,H] pinned CPU
        self.head_fn = None            # callable h[T,H]@cuda:1 -> logits[T,V]
        self.pending: deque = deque()  # [picked, matched_count]
        self.done_acc: list[int] = []
        self.ms_draft = 0.0
        self.n = 0

    def ensure_target(self, causal_lm):
        if self.head_fn is not None:
            return
        # NO bulk dequant here: dequantizing the full [V,H] table spiked
        # ~4.7 GB fp32 on cuda:0 during load and OOM'd the engine. Per-step
        # gather+dequant of just the k=8 needed rows is negligible.
        self.emb_mod = causal_lm.model.embed_tokens
        self.head_fn = causal_lm.lm_head.forward   # far-side callable

    @torch.inference_mode()
    def propose_and_score(self, tap_store: dict[int, torch.Tensor],
                          anchor_id: int, position: int):
        t0 = time.perf_counter()
        from freetoken.kernel.gguf import ggml_dequantize

        idx = torch.tensor([anchor_id] + [MASK_ID] * (self.k - 1),
                           dtype=torch.long, device="cuda:0")
        rows = self.emb_mod.packed[idx].contiguous()
        nv = ggml_dequantize(rows, int(self.emb_mod.quant_type),
                             len(idx), int(self.emb_mod.embedding_dim))
        noise = nv.to(torch.bfloat16).to("cuda:1", non_blocking=False
                                        ).unsqueeze(0)
        th = torch.stack(
            [tap_store[d].reshape(-1) for d in self.tap_layers]
        ).reshape(1, -1).to("cuda:1", dtype=torch.bfloat16, non_blocking=False)
        th = th.unsqueeze(1)                                    # [1,1,5H]
        pos = torch.arange(position - 1, position + self.k,
                           device="cuda:1")
        h, _kv = self.draft(noise, th, pos)
        proj = self.draft.candidate_selector.hidden_projection(h[:, :]).float()
        cbp = self.draft.candidate_selector.predecessor_codebook.weight
        cbs = self.draft.candidate_selector.successor_codebook.weight
        picked, pred = [], anchor_id
        logits_rows = self.head_fn(h)                            # [1,k,V]
        for pi in range(self.k):
            tv, ti = torch.topk(logits_rows[0, pi].float(), 16)
            sc = tv + (cbp[pred].float() * proj[0, pi].float()) @ cbs[
                ti.to(cbp.device)].T.float()
            tid = int(ti[int(sc.argmax())])
            picked.append(tid)
            pred = tid
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.ms_draft += dt_ms

        # score PROPOSALS created earlier against THIS emitted token;
        # the fresh chain starts checking from the NEXT emission
        still = deque()
        for ent in self.pending:
            hit = ent[0][ent[1]] == anchor_id
            if hit:
                ent[1] += 1
            if not hit or ent[1] >= len(ent[0]):
                self.done_acc.append(ent[1])
            else:
                still.append(ent)
        still.append([picked, 0])
        self.pending = still
        self.n += 1
        report = None
        ev = int(os.environ.get("DFLASH_PROBE_EVERY", "200"))
        if self.n % ev == 0 and self.done_acc:
            import statistics as st

            m = st.mean(self.done_acc[-ev * 2:])
            p = sum(v > 0 for v in self.done_acc[-ev * 2:]) / min(
                len(self.done_acc), ev * 2)
            report = f"[dflash-probe] n={self.n} meanAcc={m:.3f} P(acc>0)={p:.3f} E[tok/vfy]={1+m:.2f} draftMs={self.ms_draft/self.n:.1f}"
        return picked, report


_p: LiveDFlashProbe | None = None


def maybe_probe() -> LiveDFlashProbe | None:
    global _p
    if _p is not None:
        return _p
    if os.environ.get("FREETOKEN_DFLASH_ENGINE", "0") not in {"1", "true", "yes"}:
        return None
    try:
        from freetoken.models.dflash.gguf import parse_dflash_config
        from freetoken.models.gguf.reader import load_gguf_metadata

        md = load_gguf_metadata(GGUF_DRAFT)
        _p = LiveDFlashProbe(parse_dflash_config(md), vocab_size=248320)
        print(f"[dflash-probe] armed on cuda:1 (k={_p.k})", flush=True)
    except Exception as e:                                  # noqa: BLE001
        print(f"[dflash-probe] build FAILED: {e!r}", flush=True)
        _p = None
    return _p
