"""Standalone DFlash2 draft encoder (torch, engine-free).

A direct FreeToken port of z-lab/dflash ``DFlash2DraftModel`` semantics from
the preserved reference source (Fork/dflash-reference/dflash/model.py) — every
formula traces to that file. Differences vs the reference: no transformers
dependency (plain SDPA under the reference's symmetric-window mask), and the
draft KV is plain tensors passed in/out so the ENGINE owns cropping at verify
time (mirrors DynamicCache.update semantics: caller passes past K/V per layer,
call returns the concatenated K/V).

The lm_head and token embedding are NOT part of this module — they belong to
the TARGET trunk and are injected by callers (parity harness wires a probe
head; the engine passes Ridge's far-side head / embedding gather).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _bidir_window_mask(q_len: int, k_len: int, window: int | None, device):
    """Reference _attention_mask with is_causal=False: everything visible
    within a symmetric window around the query position."""
    q_pos = (k_len - q_len + torch.arange(q_len, device=device))[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    visible = torch.ones((q_len, k_len), dtype=torch.bool, device=device)
    if window is not None:
        visible &= (q_pos - k_pos) < window
        visible &= (k_pos - q_pos) < window
    return visible[None, None]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        v = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        return (x * self.weight.float()).to(dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * (hidden_size // group_size), bias=False)

    @staticmethod
    def _convolve(hidden, dynamic, base, group_size):
        batch, length, hidden_size = hidden.shape
        groups = hidden_size // group_size
        blocks = hidden.view(batch, length, groups, group_size)
        dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
        output = torch.zeros_like(blocks)
        for offset in range(base.shape[0]):
            values = (
                blocks if offset == 0
                else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
            )
            kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
            output = output + kernel * values
            output = torch.addcmul(output, dynamic[:, :, offset].to(hidden.dtype), values)
        return output.view_as(hidden)

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1], 2, self.kernel_size, groups)
        return (
            self._convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return self._convolve(hidden, dynamic, self.base_kernel[1], self.group_size)


def apply_rotary(q, k, cos, sin):
    """cos/sin: [1, P, head_dim] (pairs duplicated); q: [B,H,Tq,D]; k:
    [B,H,tk,D]. Follows the reference's qwen3 rotate_half convention."""
    q = q * cos[:, -q.shape[-2]:, :].unsqueeze(1) + rotate_half(q) * sin[:, -q.shape[-2]:, :].unsqueeze(1)
    k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
    return q, k


class DFlashAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h, hd = cfg["hidden_size"], cfg["head_dim"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = hd
        self.scale = hd ** -0.5
        self.q_proj = nn.Linear(h, self.num_heads * hd, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.num_heads * hd, h, bias=False)
        self.q_norm = RMSNorm(hd, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(hd, cfg["rms_norm_eps"])
        self.window = cfg.get("sliding_window")

    def _project_kv(self, x: torch.Tensor):
        bsz, t, _ = x.shape
        k = self.k_norm(self.k_proj(x).view(bsz, t, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(self, hidden_states, target_hidden, cos_sin, past_kv=None):
        """past_kv: this layer's (K, V) over ALL previous steps (context+
        noise rows exactly as the reference's DynamicCache would hold them);
        returns (attn_out, updated full (K, V))."""
        bsz, q_len, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)).transpose(1, 2)

        k_n, v_n = self._project_kv(hidden_states)
        k_c, v_c = self._project_kv(target_hidden)
        if k_c.shape[0] != bsz:  # ctx projection may be shared across batch
            k_c = k_c.expand(bsz, -1, -1, -1)
            v_c = v_c.expand(bsz, -1, -1, -1)

        # RoPE tables span [past? ctx(Tc) + noise(Tq)] positions exactly like
        # the reference's position_ids[start-Tc : start+Tq]: ALL current keys
        # use the full table; queries take the trailing q_len rows. Past K/V
        # arrive already-rotated.
        cos, sin = cos_sin
        assert cos.shape[1] >= target_hidden.shape[1] + q_len, (
            f"rope table {cos.shape[1]} < ctx+noise "
            f"{target_hidden.shape[1] + q_len}")
        k_full = torch.cat([k_c, k_n], dim=2)
        out_dtype = hidden_states.dtype
        k_rot = (k_full * cos.unsqueeze(1) + rotate_half(k_full) * sin.unsqueeze(1)).to(out_dtype)
        q_rot = (q * cos[:, -q_len:, :].unsqueeze(1) + rotate_half(q) * sin[:, -q_len:, :].unsqueeze(1)).to(out_dtype)

        if past_kv is not None:
            pk, pv = past_kv
            k_all = torch.cat([pk.to(k_rot.dtype), k_rot], dim=2)
            v_all = torch.cat([pv.to(v_n.dtype), torch.cat([v_c, v_n], dim=2)], dim=2)
        else:
            k_all = k_rot
            v_all = torch.cat([v_c, v_n], dim=2)

        total = k_all.shape[2]
        mask = _bidir_window_mask(q_len, total, self.window, q.device)
        out = F.scaled_dot_product_attention(
            q_rot, k_all, v_all,
            attn_mask=mask, scale=self.scale, enable_gqa=True)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(out), (k_all, v_all)


class _MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h = cfg["hidden_size"]
        self.gate_proj = nn.Linear(h, cfg["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(h, cfg["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(cfg["intermediate_size"], h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h = cfg["hidden_size"]
        self.self_attn = DFlashAttention(cfg)
        self.mlp = _MLP(cfg)
        self.input_layernorm = RMSNorm(h, cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(h, cfg["rms_norm_eps"])
        ks, gs = cfg["conv_kernel_size"], cfg["conv_group_size"]
        self.attention_conv = GroupedDynamicCausalConv(h, ks, gs)
        self.mlp_conv = GroupedDynamicCausalConv(h, ks, gs)

    def forward(self, hidden_states, target_hidden, cos_sin, past_kv=None):
        residual = hidden_states
        hs, attn_dyn = self.attention_conv.prepare(self.input_layernorm(hidden_states))
        attn_out, kv_out = self.self_attn(hs, target_hidden, cos_sin, past_kv=past_kv)
        hs = self.attention_conv.finish(attn_out, attn_dyn)
        hidden_states = residual + hs

        residual = hidden_states
        ms, mlp_dyn = self.mlp_conv.prepare(self.post_attention_layernorm(hidden_states))
        mlp_out = self.mlp(ms)
        hs = self.mlp_conv.finish(mlp_out, mlp_dyn)
        return residual + hs, kv_out


class CandidateSelector(nn.Module):
    """Reference CandidateSelector (greedy branch; sampled proposals arrive
    with the S4 rejection-sampler port)."""

    def __init__(self, cfg: dict, vocab_size: int):
        super().__init__()
        rank = cfg["selector_rank"]
        self.top_k = cfg["selector_top_k"]
        self.predecessor_codebook = nn.Embedding(vocab_size, rank)
        self.successor_codebook = nn.Embedding(vocab_size, rank)
        self.hidden_projection = nn.Linear(cfg["hidden_size"], rank, bias=False)

    @torch.inference_mode()
    def select_greedy(self, hidden, logits, anchor_ids):
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        proj = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path = []
        for position in range(hidden.shape[1]):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * proj[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            index = torch.argmax(scores, dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        return torch.stack(path, dim=1), candidates


class DFlash2Draft(nn.Module):
    """Portable DFlash2 draft. Embeddings/logits come from the TARGET trunk:
    noise_embedding is precomputed by the caller from the trunk's embedding
    table, and proposals score against a trunk-head callable."""

    def __init__(self, cfg: dict, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        h = cfg["hidden_size"]
        self.layers = nn.ModuleList(DFlashLayer(cfg) for _ in range(cfg["num_layers"]))
        self.norm = RMSNorm(h, cfg["rms_norm_eps"])
        self.fc = nn.Linear(len(cfg["target_layers"]) * h, h, bias=False)
        self.hidden_norm = RMSNorm(h, cfg["rms_norm_eps"])
        self.block_size = cfg["block_size"]
        self.candidate_selector = CandidateSelector(cfg, vocab_size)
        inv = 1e7 ** (-torch.arange(0, cfg["head_dim"], 2).float() / cfg["head_dim"])
        self.register_buffer("inv_freq", inv, persistent=False)

    def rope_tables(self, positions: torch.Tensor):
        """[1, P, head_dim] duplicated-pair cos/sin (rotate_half convention)."""
        pos = positions.float().view(-1)
        ang = pos[:, None] * self.inv_freq.to(pos.device)[None, :]
        emb = torch.cat([ang, ang], dim=-1)[None]
        return torch.cos(emb), torch.sin(emb)

    def forward(self, noise_embedding, target_hidden, position_ids, past_kvs=None):
        """One pass over the noise block [B, Tq, H]. past_kvs: per-layer
        (K, V) tuples or None. Returns (normed hidden, new per-layer KVs).

        position_ids covers the FULL attended window: the Tc context rows'
        positions followed by the Tq noise positions (mirrors the reference's
        ``position_ids[:, start - Tc : start + Tq]``)."""
        ctx = self.hidden_norm(self.fc(target_hidden))
        cos, sin = self.rope_tables(position_ids)
        hs = noise_embedding
        new_kvs = []
        for i, layer in enumerate(self.layers):
            hs, kv = layer(
                hs, ctx, (cos, sin),
                past_kv=past_kvs[i] if past_kvs is not None else None,
            )
            new_kvs.append(kv)
        return self.norm(hs), new_kvs

    def propose_greedy(self, hidden, anchor_ids, output_head):
        logits = output_head(hidden)
        return self.candidate_selector.select_greedy(hidden, logits, anchor_ids)
