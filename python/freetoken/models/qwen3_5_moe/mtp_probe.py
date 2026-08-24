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

    def __init__(self, model, config, model_path=None):
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
        self._latch = None
        self.engine = self._maybe_attach_bf16_engine(model_path)
        if self.engine is not None:
            # Two-hop xfer buffer: a direct cuda:0 -> cuda:1 .to() does a
            # cross-device staged copy that forces device-wide syncs every
            # step (measured 45 -> 17 tok/s). Pinned host staging keeps each
            # hop a plain stream-ordered copy. Everything also runs on PRIVATE
            # streams: the legacy null stream synchronizes across devices on
            # ROCm, which serialized the drain with in-flight replays (~30ms
            # per step measured on the null streams).
            self._pin_h = torch.empty(
                1, self.config.hidden_size, dtype=torch.bfloat16, pin_memory=True
            )
            self._pin_ev = torch.cuda.Event()
            self._s0 = torch.cuda.Stream()
            self._s1 = torch.cuda.Stream(device=self.engine.device)
            self._eng_ms = 0.0
            self._xfer_ms = 0.0
            # one-step latch: (event, logits, req_id) resolved at the NEXT
            # step's entry (GPU 1 is long idle by then -> free sync)
            self._latch = None
        else:
            self._attach_private_expert_cache(model_path)

    def _maybe_attach_bf16_engine(self, model_path):
        """Dual-GPU mode: the draft head on the second GPU, dequantized bf16.

        The trunk keeps GPU 0 (and its graphs) to itself; the draft gets the
        idle device with full-precision weights (no Q4_K compute noise) and
        its own KV. Per-step cross-GPU traffic: one [H] hidden vector over,
        one token id back. FREETOKEN_MTP_DRAFT_GPU=0 forces the in-process
        quantized path (single-GPU portability); "1"/"auto" (default) use
        cuda:1 when a second device exists.
        """
        import os

        sel = os.environ.get("FREETOKEN_MTP_DRAFT_GPU", "auto").strip().lower()
        if sel in {"0", "off", "false", "no"} or model_path is None:
            return None
        idx = 1 if sel in {"auto", "1", "on", "true", "yes"} else int(sel)
        if torch.cuda.device_count() <= idx:
            if sel == "auto":
                return None
            logger.warning(f"[mtp-probe] FREETOKEN_MTP_DRAFT_GPU={idx} but only "
                           f"{torch.cuda.device_count()} devices; falling back")
            return None
        try:
            from .mtp_draft import Bf16DraftEngine

            eng = Bf16DraftEngine(model_path, self.config, f"cuda:{idx}")
            logger.info(f"[mtp-probe] draft engine on cuda:{idx} (trunk keeps cuda:0)")
            return eng
        except Exception as e:
            logger.warning(f"[mtp-probe] bf16 draft engine failed: {e!r}; "
                           "falling back to the in-process quantized path")
            return None

    def _attach_private_expert_cache(self, model_path):
        """Give the draft layer its OWN OffloadMoeCache for the file tier.

        The probe runs eagerly on the scheduler-drain thread while the trunk's
        next graph replay mutates the ENGINE cache's LRU tables and slot cache
        on the forward stream -- sharing them evicts the probe's slots before
        its GEMM reads them (uncorrelated routed outputs, measured cos ~= 0).
        A private cache (512 slots ~= 370 MB for the whole draft layer) keeps
        both sides race-free; the mmap file sources cost nothing to attach.
        """
        experts = getattr(self.draft.layer.mlp, "experts", None)
        cache = getattr(experts, "offload_cache", None) if experts is not None else None
        if (
            model_path is None
            or experts is None
            or cache is None
            or cache.quant_format != "ggml_file"
        ):
            return
        try:
            from freetoken.moe.offload_cache import OffloadMoeCache
            from freetoken.models.gguf.reader import iter_gguf_tensors  # noqa: F401
            from .gguf import load_ggml_expert_sources_file, parse_gguf_config
            from freetoken.utils import cached_load_hf_config

            cfg = parse_gguf_config(cached_load_hf_config(model_path))
            srcs = load_ggml_expert_sources_file(model_path, cfg)
            PL = len(srcs["gate"])
            priv = OffloadMoeCache(
                num_layers=PL,
                num_experts=cfg.num_experts,
                cache_size=2 * cfg.num_experts,
                device=self.device,
                quant_format="ggml_file",
            )
            priv.set_bank_sources({k: srcs[k] for k in ("gate", "up", "down")})
            priv.ggml_quant_types = srcs["quant_types"]
            experts.offload_cache = priv
            logger.info(
                "[mtp-probe] private expert cache attached "
                f"({PL} layers x {cfg.num_experts} experts, file tier)"
            )
        except Exception as e:  # shared cache fallback: measurement may race
            logger.warning(f"[mtp-probe] private expert cache failed: {e!r}")

    def _score_step(self, req_id, actual_token: int):
        if self.pending is not None and self.pending[0] == req_id:
            self.total += 1
            if self.pending[1] == int(actual_token):
                self.hits += 1
            if self.total % 200 == 0:
                eng = (
                    f" eng={self._eng_ms / 200:.2f}ms xfer={self._xfer_ms / 200:.2f}ms"
                    if self.engine is not None else ""
                )
                self._eng_ms = 0.0
                self._xfer_ms = 0.0
                logger.info(
                    f"[mtp-probe] acceptance {self.hits}/{self.total} = "
                    f"{self.hits / self.total:.3f} | h_std={getattr(self, 'last_h_std', -1):.4f}"
                    f" post_std={getattr(self, 'last_hp_std', -1):.4f}{eng}"
                )
        self.pending = None

    @torch.no_grad()
    def step(self, req_id, pos: int, token: int, trunk_hidden, trunk_hidden_post=None):
        """Called after a decode step committed ``token`` at ``pos``."""
        d = self.draft
        at = d.layer.self_attn
        if self.engine is not None and self._latch is not None:
            # resolve the previous step's async draft (predicts THIS step's
            # token) before scoring -- pending is then (req, pred) vs token.
            ev, lg, rid = self._latch
            ev.synchronize()
            self.pending = (rid, int(lg.argmax().item()))
            self._latch = None
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

        if self.engine is not None:
            # Dual-GPU: hand the snapshot to the bf16 engine via pinned host
            # staging on private streams (see __init__); the whole forward is
            # ENQUEUED async and the argmax resolves at the next entry (GPU 1
            # finishes long before the next decode step commits).
            import time as _time

            t0 = _time.perf_counter()
            s0, s1 = self._s0, self._s1
            with torch.cuda.stream(s0):
                self._pin_h.copy_(h, non_blocking=True)
                self._pin_ev.record(s0)
            self._pin_ev.synchronize()
            t1 = _time.perf_counter()
            with torch.cuda.stream(s1):
                h1 = self._pin_h.to(self.engine.device, non_blocking=True)
                logits = self.engine.step_async(h1[0], token, pos, n)
                ev = torch.cuda.Event()
                ev.record(s1)
            self._latch = (ev, logits, req_id)
            self.fill[req_id] = n + 1
            t2 = _time.perf_counter()
            self._xfer_ms += (t1 - t0) * 1e3
            self._eng_ms += (t2 - t1) * 1e3
            self._trace(h, hp, token, pos, None, mid=None)
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
        # llama.cpp qwen35moe_mtp: h2 = ffn_residual + moe_out + shared --
        # dropping the residual stream (routed + shared alone) decimates the
        # head input (measured: acceptance 6.6% vs 40%+ with it).
        h2 = residual + routed + shared

        logits = self.model.lm_head.forward(d.head_norm.forward(h2))
        pred = int(logits[0].argmax())
        self.pending = (req_id, pred)
        self._trace(
            h, hp, token, pos, logits,
            mid=dict(
                x0=x0.detach().clone(),
                attn=o.detach().clone(),
                x=x.detach().clone(),
                router=router_logits.detach().clone(),
                routed=routed.detach().clone(),
                shared=shared.detach().clone(),
                h2=h2.detach().clone(),
            ),
        )

    _TRACE_N = 0

    def _trace(self, h, hp, token, pos, logits, mid=None, engine_top5=None):
        """FREETOKEN_MTP_TRACE=<dir>: dump inputs/outputs for offline debugging."""
        import os

        d = os.environ.get("FREETOKEN_MTP_TRACE")
        if not d or MTPProbe._TRACE_N >= 64:
            return
        MTPProbe._TRACE_N += 1
        import torch as _t

        payload = {
            "h": h.detach().float().cpu(),
            "h_post": hp.detach().float().cpu() if hp is not None else None,
            "token": int(token),
            "pos": int(pos),
            "logits": logits.detach().float().cpu() if logits is not None else None,
            "pred": int(logits[0].argmax()) if logits is not None else (self.pending[1] if self.pending else -1),
            "fill": self.fill.get(self.pending[0], 0) if self.pending else 0,
        }
        if mid is not None:
            payload["mid"] = {
                k: (v.detach().float().cpu() if torch.is_tensor(v) else v)
                for k, v in mid.items()
            }
        if engine_top5 is not None:
            t5i, t5v = engine_top5
            payload["top5_idx"] = t5i.detach().cpu().tolist()
            payload["top5_val"] = t5v.detach().float().cpu().tolist()
        _t.save(payload, os.path.join(d, f"step_{MTPProbe._TRACE_N:03d}.pt"))

    def reset_req(self, req_id):
        self.fill.pop(req_id, None)
        if self.pending and self.pending[0] == req_id:
            self.pending = None
        if self._latch is not None and self._latch[2] == req_id:
            self._latch = None

    def report(self):
        if self.total:
            rate = self.hits / self.total
            logger.info(
                f"[mtp-probe] FINAL acceptance {self.hits}/{self.total} = {rate:.3f}"
            )
            return rate
        return None


__all__ = ["MTPProbe"]
