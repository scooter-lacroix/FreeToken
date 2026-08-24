"""MTP draft acceptance probe (milestone 2b).

Runs the loaded draft head EAGERLY beside normal serving -- no behavior
change, no graph interaction -- and scores it: after each decode step the
draft predicts the next token from (trunk hidden, current token); the
following step's real token settles the bet. The reported acceptance rate
is exactly the k=1 speculative-decode win rate and the tuning target for
the scheduler loop (milestone 2d).

The draft's single full-attention layer keeps a private eager KV cache per
request slot (no attention-backend surgery for the probe); its MoE rides
the offload cache at trunk layer index ``num_layers``. The layer's own
rope/q/k norms are reused verbatim from ``Qwen3_5Attention``.
"""

from __future__ import annotations

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


class MTPProbe:
    """Eager k=1 draft probe. Enable: FREETOKEN_MTP=1 FREETOKEN_MTP_PROBE=1."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.draft = model.draft
        self.device = torch.device("cuda")
        self.kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.max_pos = 16384
        self.k_cache = torch.zeros(
            self.max_pos, self.kv_heads, self.head_dim, dtype=torch.bfloat16,
            device=self.device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.fill: dict = {}
        self.pending = None
        self.hits = 0
        self.total = 0
        self.last_h_std = -1.0
        self.last_hp_std = -1.0

    def _score_step(self, req_id, actual_token: int):
        if self.pending is not None and self.pending[0] == req_id:
            self.total += 1
            if self.pending[1] == int(actual_token):
                self.hits += 1
            if self.total % 200 == 0:
                logger.info(
                    f"[mtp-probe] acceptance {self.hits}/{self.total} = "
                    f"{self.hits / self.total:.3f} | h_std={getattr(self, 'last_h_std', -1):.4f}"
                    f" post_std={getattr(self, 'last_hp_std', -1):.4f}"
                )
        self.pending = None

    @torch.no_grad()
    def step(self, req_id, pos: int, token: int, trunk_hidden, trunk_hidden_post=None):
        """Called after a decode step committed ``token`` at ``pos``."""
        d = self.draft
        at = d.layer.self_attn
        if trunk_hidden is None:
            self._score_step(req_id, token)
            return
        # Snapshot immediately: the shared trunk buffers are rewritten by the
        # NEXT decode replay (forward stream) while this drain thread works.
        h = trunk_hidden.detach().reshape(1, -1).clone().to(torch.bfloat16)
        hp = (
            trunk_hidden_post.detach().reshape(1, -1).clone().to(torch.bfloat16)
            if trunk_hidden_post is not None
            else None
        )
        self.last_h_std = float(h.float().std().item())
        self.last_hp_std = float(hp.float().std().item()) if hp is not None else -1.0
        self._score_step(req_id, token)

        n = self.fill.get(req_id, 0)
        if n >= self.max_pos - 1 or pos >= self.max_pos:
            return

        # --- eh_proj combine (llama.cpp qwen35moe_mtp: concat(e_norm, h_norm)) ---
        emb = self.model.model.embed_tokens.forward(
            torch.tensor([token], dtype=torch.int64, device=self.device)
        ).reshape(1, -1)
        x0 = d.eh_proj.forward(
            torch.cat([d.enorm.forward(emb), d.hnorm.forward(h)], dim=-1)
        )

        # --- the draft layer, eager, mirroring Qwen3_5DecoderLayer/_project ---
        x = d.layer.input_layernorm.forward(x0)
        qkv = torch.cat(
            [at.q_proj.forward(x), at.k_proj.forward(x), at.v_proj.forward(x)], dim=-1
        )
        qg, k, v = torch.split(qkv, at._qkv_split, dim=-1)
        qg = qg.view(1, at.num_q, at.head_dim * 2)
        q = qg[..., : at.head_dim].contiguous()
        gate = qg[..., at.head_dim :].reshape(1, at.qo_attn_dim)
        k = k.view(1, at.num_kv, at.head_dim).contiguous()
        v = v.contiguous()
        q = at.q_norm.forward(q).reshape(1, -1)
        k = at.k_norm.forward(k).reshape(1, -1)
        positions = torch.tensor([pos], dtype=torch.int32, device=self.device)
        q, k = at.rotary.forward(positions, q, k)

        self.k_cache[n] = k.view(at.num_kv, at.head_dim)
        self.v_cache[n] = v.view(at.num_kv, at.head_dim)
        self.fill[req_id] = n + 1

        K = self.k_cache[: n + 1].float()   # [T, kvh, d]
        V = self.v_cache[: n + 1].float()
        rep = at.num_q // at.num_kv  # GQA: expand KV to the query heads
        if rep > 1:
            K = K.repeat_interleave(rep, dim=1)
            V = V.repeat_interleave(rep, dim=1)
        qf = q.view(1, at.num_q, at.head_dim).float()
        scores = torch.einsum("ihd,thd->iht", qf, K) * (at.head_dim ** -0.5)
        probs = torch.softmax(scores, dim=-1)
        o = torch.einsum("iht,thd->ihd", probs.to(V.dtype), V).reshape(1, -1)
        o = at.o_proj.forward(o.bfloat16() * torch.sigmoid(gate).bfloat16())

        x, residual = d.layer.post_attention_layernorm.forward_add_residual(o, x0)
        mlp = d.layer.mlp
        # ctx-free MoE (the scheduler drain has no active forward batch):
        # mirror Qwen3_5MoE.forward onto the eager decode path
        router_logits = mlp.gate.forward(x)
        shared = mlp.shared_expert.forward(x)
        shared = shared * torch.sigmoid(mlp.shared_expert_gate.forward(x))
        routed = mlp.experts.decode_forward(x, router_logits)
        h2 = routed + shared

        logits = self.model.lm_head.forward(d.head_norm.forward(h2))
        pred = int(logits[0].argmax())
        self.pending = (req_id, pred)
        self._trace(h, hp, token, pos, logits)

    _TRACE_N = 0

    def _trace(self, h, hp, token, pos, logits):
        """FREETOKEN_MTP_TRACE=<dir>: dump inputs/outputs for offline debugging."""
        import os

        d = os.environ.get("FREETOKEN_MTP_TRACE")
        if not d or MTPProbe._TRACE_N >= 64:
            return
        MTPProbe._TRACE_N += 1
        import torch as _t

        _t.save(
            {
                "h": h.detach().float().cpu(),
                "h_post": hp.detach().float().cpu() if hp is not None else None,
                "token": int(token),
                "pos": int(pos),
                "logits": logits.detach().float().cpu(),
                "pred": int(logits[0].argmax()),
                "fill": self.fill.get(self.pending[0], 0) if self.pending else 0,
            },
            os.path.join(d, f"step_{MTPProbe._TRACE_N:03d}.pt"),
        )

    def reset_req(self, req_id):
        self.fill.pop(req_id, None)
        if self.pending and self.pending[0] == req_id:
            self.pending = None

    def report(self):
        if self.total:
            rate = self.hits / self.total
            logger.info(
                f"[mtp-probe] FINAL acceptance {self.hits}/{self.total} = {rate:.3f}"
            )
            return rate
        return None


__all__ = ["MTPProbe"]
