"""Full-bf16 MTP draft engine on a second GPU (dual-GPU mode).

The measurement probe and (next milestone) the scheduler draft-verify loop
run the nextn head EAGERLY beside the trunk. On a single GPU that costs
~1 ms/token of eager kernels contending with the trunk's graph replays. When
a second GPU is present, this class serves the draft from it: the trunk keeps
GPU 0 to itself, the draft gets the idle device, and the per-step cross-GPU
traffic is one [H] hidden vector over plus one token id back.

The math mirrors the validated offline replay (llama.cpp qwen35moe_mtp
semantics): eh_proj over concat(enorm(emb), hnorm(h)) -- embedding first --
full-attention draft block with per-head q|gate, qk-norms, partial NeoX rope,
sigmoid output gate, top-8 renormalized routing plus gated shared expert, the
FFN residual, and the shared head over head_norm.

Perf notes (measured, gfx1100-class, idle GPU): torch's rocBLAS bf16 GEMV
(``W @ x``) runs at ~50 GB/s (a 20 ms LM head!), while ``x @ W.t()`` and our
Triton Q4_K GEMV run 400-650 GB/s. So every projection stores TRANSPOSED
weights and computes ``x @ Wt``; the LM head stays GGUF-packed and goes
through :func:`freetoken.kernel.triton.kquant_linear.kq_gemv` (654 GB/s on
this exact shape). Host launch cost is ~75 us/op on this stack, so the
forward also minimizes op count (fused rms_norm, batched einsums, one-hop
casts). No host syncs anywhere inside -- routing stays device-side and the
caller resolves argmax one step later.

Single-GPU deployments fall back to the in-process quantized probe path
(FREETOKEN_MTP_DRAFT_GPU=0).
"""

from __future__ import annotations

import math

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


class Bf16DraftEngine:
    """Dequantized draft head resident on one device. Enable via
    FREETOKEN_MTP_DRAFT_GPU=1 (or "auto", the default, when a second GPU
    exists)."""

    def __init__(self, model_path, config, device: str):
        import time

        from freetoken.models.gguf.reader import iter_gguf_tensors
        from freetoken.models.gguf.dequant import dequantize

        t0 = time.time()
        self.device = torch.device(device)
        self.H = H = config.hidden_size
        self.nq = config.num_qo_heads
        self.nkv = config.num_kv_heads
        self.hd = config.head_dim
        self.eps = config.rms_norm_eps
        self.E = E = config.num_experts
        self.topk = config.num_experts_per_tok
        self.FF = I = config.moe_intermediate_size
        rc = config.rotary_config
        self.rd = rc.rotary_dim
        self.max_pos = 16384
        L = config.num_layers  # draft block id

        want = {"token_embd.weight", "output.weight"}
        for nm in (
            "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
            "nextn.shared_head_norm.weight",
            "attn_norm.weight", "post_attention_norm.weight",
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
            "attn_q_norm.weight", "attn_k_norm.weight",
            "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        ):
            want.add(f"blk.{L}.{nm}")
        raw = {}
        for t in iter_gguf_tensors(model_path):
            if t.name in want:
                raw[t.name] = t
        missing = want - set(raw)
        assert not missing, f"draft tensors missing from checkpoint: {sorted(missing)}"

        def bf16(name):
            t = raw[name]
            return dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16) \
                .reshape(t.shape).to(self.device)

        def W(name):
            return bf16(f"blk.{L}.{name}")

        def Wt(name):
            # transposed storage: every projection runs as fast x @ Wt
            return W(name).t().contiguous()

        self.emb = bf16("token_embd.weight")       # [vocab, H] (gather only)
        # LM head: requantized to Q4_K for OUR Triton GEMV. The ggml C++
        # extension's code object is gfx1100-only and HANGS on gfx1101 (the
        # 7800 XT), so the engine must stay on torch/Triton kernels; Triton
        # JITs per-device at runtime. Requant noise is negligible (the trunk
        # already requantizes Q6_K->Q4_K for out_proj and the bank down-proj).
        from freetoken.models.gguf.dequant import requantize_q4_k

        lm_t = raw["output.weight"]
        lm_f32 = dequantize(lm_t.packed().reshape(-1), lm_t.ggml_type, torch.float32)
        self.lm_qt = 12
        self.lm_packed = requantize_q4_k(lm_f32).to(self.device).reshape(
            lm_t.shape[0], -1
        ).contiguous()
        del lm_f32

        self.eh_t = bf16(f"blk.{L}.nextn.eh_proj.weight").t().contiguous()   # [2H, H]
        self.enorm = bf16(f"blk.{L}.nextn.enorm.weight").float()
        self.hnorm = bf16(f"blk.{L}.nextn.hnorm.weight").float()
        self.head_norm = bf16(f"blk.{L}.nextn.shared_head_norm.weight").float()
        self.attn_norm = W("attn_norm.weight").float()
        self.post_norm = W("post_attention_norm.weight").float()
        self.wq_t = Wt("attn_q.weight")            # [H, nq*hd*2]
        self.wk_t = Wt("attn_k.weight")            # [H, nkv*hd]
        self.wv_t = Wt("attn_v.weight")
        self.wo_t = Wt("attn_output.weight")       # [nq*hd, H]
        self.q_norm = W("attn_q_norm.weight").float()
        self.k_norm = W("attn_k_norm.weight").float()
        self.gate_t = Wt("ffn_gate_inp.weight")    # [H, E]
        self.ginp_sh_t = Wt("ffn_gate_inp_shexp.weight")
        self.gate_sh_t = Wt("ffn_gate_shexp.weight")   # [H, FF]
        self.up_sh_t = Wt("ffn_up_shexp.weight")
        self.down_sh_t = Wt("ffn_down_shexp.weight")  # [FF, H]
        # routed experts: [E, rows, cols] in ggml orientation (gate/up rows
        # are FF out-rows over H in; down rows are H out-rows over FF in)
        self.exp_gate = bf16(f"blk.{L}.ffn_gate_exps.weight").reshape(E, I, H)
        self.exp_up = bf16(f"blk.{L}.ffn_up_exps.weight").reshape(E, I, H)
        self.exp_down = bf16(f"blk.{L}.ffn_down_exps.weight").reshape(E, H, I)

        # rope tables (partial NeoX, first rd dims of each head)
        inv = 1.0 / (rc.base ** (
            torch.arange(0, self.rd, 2, dtype=torch.float32, device=self.device) / self.rd
        ))
        t = torch.arange(self.max_pos, dtype=torch.float32, device=self.device)
        fr = torch.einsum("i,j->ij", t, inv)       # [pos, rd/2]
        self._cos = fr.cos().bfloat16()
        self._sin = fr.sin().bfloat16()

        self.k_cache = torch.zeros(self.max_pos, self.nkv, self.hd, dtype=torch.bfloat16, device=self.device)
        self.v_cache = torch.zeros_like(self.k_cache)
        gb = (self.emb.numel() * 2 + self.lm_packed.numel()) / 2**30
        xgb = (self.exp_gate.numel() + self.exp_up.numel() + self.exp_down.numel()) * 2 / 2**30
        logger.info(
            f"[mtp-draft] bf16 draft engine on {self.device}: "
            f"{gb:.2f} GiB head/emb + experts {xgb:.2f} GiB ({time.time() - t0:.0f}s load)"
        )

    def _rms(self, x, w):
        # one fused op (fp32 in/out keeps the replay-validated numerics)
        return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), weight=w, eps=self.eps)

    def _rope(self, x, pos):
        # x [heads, hd]; NeoX half-rotation on the first rd dims
        c, s = self._cos[pos], self._sin[pos]      # [rd/2] bf16
        xr = x[:, : self.rd].float()
        x1, x2 = xr[:, : self.rd // 2], xr[:, self.rd // 2:]
        rot = torch.cat([x1 * c.float() - x2 * s.float(), x1 * s.float() + x2 * c.float()], dim=-1).bfloat16()
        return torch.cat([rot, x[:, self.rd:]], dim=-1)

    @torch.no_grad()
    def step_async(self, h, token: int, pos: int, n: int):
        """Enqueue the whole draft forward on the CURRENT stream of this
        device; no host syncs inside. Returns the logits tensor ASYNC (the
        caller resolves argmax one step later -- the prediction is only
        compared against the NEXT committed token).

        h: [H] bf16 already on this device.
        """
        import torch.nn.functional as F

        H, nq, nkv, hd = self.H, self.nq, self.nkv, self.hd
        e = self.emb[token]
        cat = torch.cat(
            [self._rms(e, self.enorm), self._rms(h, self.hnorm)], dim=-1
        ).bfloat16().unsqueeze(0)                       # [1, 2H] (2-D: x @ Wt, never mv)
        x0 = cat @ self.eh_t                            # [1, H]

        # -- attention block --
        x = self._rms(x0, self.attn_norm).bfloat16()    # [1, H]
        qf = (x @ self.wq_t).view(nq, 2 * hd)
        q, gate = qf[:, :hd], qf[:, hd:]
        k = (x @ self.wk_t).view(nkv, hd)
        v = (x @ self.wv_t).view(nkv, hd)
        q = self._rms(q, self.q_norm).bfloat16()
        k = self._rms(k, self.k_norm).bfloat16()
        q, k = self._rope(q, pos), self._rope(k, pos)

        self.k_cache[n] = k
        self.v_cache[n] = v
        T = n + 1
        # GQA without materializing the KV repeat: q heads are consecutive
        # groups of rep per kv head (repeat_interleave semantics)
        qg = q.view(nkv, -1, hd).float()                    # [nkv, rep, hd]
        s_ = torch.einsum("grd,tkd->grt", qg, self.k_cache[:T].float()) / math.sqrt(hd)
        p = torch.softmax(s_, dim=-1)
        o = torch.einsum("grt,tkd->grd", p, self.v_cache[:T].float())
        o = (o * torch.sigmoid(gate.float()).view(nkv, -1, hd)).reshape(1, nq * hd).bfloat16()
        attn = o @ self.wo_t                               # [1, H]

        xr = attn + x0                                   # [1, H] FFN residual stream
        x2 = self._rms(xr, self.post_norm).bfloat16()    # [1, H]

        # -- MoE: top-8 renormalized routing, device-side ids --
        rl = (x2 @ self.gate_t).float()                     # [1, E]
        top = torch.topk(torch.softmax(rl, dim=-1), self.topk)
        wts = (top.values / top.values.sum())               # [1, K] fp32
        ids = top.indices[0]                                # [K]
        g = torch.einsum("eih,th->ei", self.exp_gate[ids], x2)
        u = torch.einsum("eih,th->ei", self.exp_up[ids], x2)
        inter = (F.silu(g.float()) * u.float()).bfloat16()  # [K, I]
        d = torch.einsum("ehi,ei->eh", self.exp_down[ids], inter)
        routed = (d.float() * wts[0].unsqueeze(-1)).sum(0).bfloat16()
        gsh, ush = x2 @ self.gate_sh_t, x2 @ self.up_sh_t
        shared = (F.silu(gsh.float()) * ush.float()).bfloat16() @ self.down_sh_t
        shared = shared * torch.sigmoid(x2 @ self.ginp_sh_t).bfloat16()

        h2 = xr + routed.unsqueeze(0) + shared
        # the draft's own hidden feeds back as the chain input when drafting
        # deeper than k=1 (DeepSeek-MTP recursion: the head recurs on its own
        # stream since no trunk hidden exists for speculative positions)
        self.last_h2 = h2
        y = self._rms(h2, self.head_norm).bfloat16()        # [1, H]
        from freetoken.kernel.triton.kquant_linear import kq_gemv

        return kq_gemv(self.lm_packed, y, self.lm_qt)       # [1, vocab]


__all__ = ["Bf16DraftEngine"]
