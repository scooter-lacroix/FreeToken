"""Grouped expert GEMM over native GGUF Q4_0 banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed Q4_0 block bytes and
dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We
use the MMVQ (vector) kernel for both prefill and decode: it consumes ``topk_ids``
directly (no ``moe_align_block_size`` needed) and on small batches it is the right
choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_0

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_and_mul}


def _triton_moe_ok(quant_types: tuple[int, int]) -> bool:
    """Parked (default OFF): the byte-space kernels are ~36 us each, but the
    ggml MMVQ path already serves a layer in ~52 us in-graph; the fragmented
    per-expert row counts (512-2048) starve this variant where the dense form
    wins. The routed-expert lever belongs to a Rusty-Llama-style kernel port
    (async dequant / rocWMMA over the vendored ggml sources). Enable with
    FREETOKEN_TRITON_MOE=1 for A/B."""
    import os

    return (
        quant_types == (12, 12)
        and os.environ.get("FREETOKEN_TRITON_MOE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def fused_experts_ggml_triton(
    hidden_states: torch.Tensor,
    gate_q: torch.Tensor,   # [slots, I, row_bytes(H)]
    up_q: torch.Tensor,     # [slots, I, row_bytes(H)]
    down_q: torch.Tensor,   # [slots, H, row_bytes(I)]
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Byte-space Triton variant of :func:`fused_experts_ggml_split`.

    The uniform-Q4_K SSD-tier pack makes every bank Q4_K, so the dense kquant
    GEMV's per-expert form serves all three projections with no q8 pre-pass:
    measured ~534 GB/s effective vs the ggml MMVQ's ~180 in-profile (which
    also pays a quantize_row_q8_1 launch per call).
    """
    import torch.nn.functional as F

    from freetoken.kernel.triton.kquant_linear import kq_moe_gemv

    assert activation == "silu", "Triton MoE path currently covers silu"
    g = kq_moe_gemv(gate_q, topk_ids, hidden_states)   # [T, K_top, I]
    u = kq_moe_gemv(up_q, topk_ids, hidden_states)
    inter = (F.silu(g.float()) * u.float()).to(torch.bfloat16).contiguous()
    d = kq_moe_gemv(down_q, topk_ids, inter)            # [T, K_top, H]
    return (d.float() * topk_weights.unsqueeze(-1)).sum(dim=1).to(
        hidden_states.dtype
    )


def fused_experts_ggml(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, row_bytes(H)] uint8
    down_q: torch.Tensor,  # [num_slots, H, row_bytes(I)] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_type: int = GGML_Q4_0,
) -> torch.Tensor:
    """Grouped expert GEMM over native GGML block-quant banks (Q4_0 or the
    k-quants Q4_K/Q6_K; ``quant_type`` is the ggml enum of gate_up -- down shares
    it in the Q4_0 path and carries its own in the mixed k-quant path via
    :func:`fused_experts_ggml_mixed`)."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]
    qt = int(quant_type)

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt, n2, num_tokens)
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, qt, h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def fused_experts_ggml_mixed(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_types: tuple[int, int],
) -> torch.Tensor:
    """k-quant variant where gate_up and down carry different ggml types
    (e.g. Q4_K_M mixes Q4_K gate/up with Q6_K down on some layers)."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT[activation]
    qt_gu, qt_dn = int(quant_types[0]), int(quant_types[1])
    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]
    h = down_q.shape[1]
    top_k = topk_ids.shape[1]

    gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt_gu, n2, num_tokens)
    inter = act_fn(gate_up)
    out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, qt_dn, h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def fused_experts_ggml_split(
    hidden_states: torch.Tensor,
    gate_q: torch.Tensor,   # [slots, I, row_bytes(H)]
    up_q: torch.Tensor,     # [slots, I, row_bytes(H)]
    down_q: torch.Tensor,   # [slots, H, row_bytes(I)]
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_types: tuple[int, int],
) -> torch.Tensor:
    """SSD-tier variant over the unfused gate/up/down banks (file-backed views
    stream into the slot caches; same math as the fused path, three MMVQ
    calls)."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT[activation]
    qt_gu, qt_dn = int(quant_types[0]), int(quant_types[1])
    num_tokens = hidden_states.shape[0]
    i_size = gate_q.shape[1]
    h = down_q.shape[1]
    top_k = topk_ids.shape[1]

    g = ggml_moe_a8_vec(hidden_states, gate_q, topk_ids, top_k, qt_gu, i_size, num_tokens)
    u = ggml_moe_a8_vec(hidden_states, up_q, topk_ids, top_k, qt_gu, i_size, num_tokens)
    inter = act_fn(torch.cat([g, u], dim=-1))
    out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, qt_dn, h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


# Back-compat alias for the Q4_0 path.
fused_experts_gguf_q4_0 = fused_experts_ggml

__all__ = [
    "fused_experts_ggml",
    "fused_experts_ggml_mixed",
    "fused_experts_ggml_split",
    "fused_experts_ggml_split_bf16_prefill",
    "fused_experts_gguf_q4_0",
]


def fused_experts_ggml_split_bf16_prefill(
    hidden_states: torch.Tensor,
    gate_q: torch.Tensor,   # [slots, I, row_bytes(H)]
    up_q: torch.Tensor,     # [slots, I, row_bytes(H)]
    down_q: torch.Tensor,   # [slots, H, row_bytes(I)]
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_types: tuple[int, int],
) -> torch.Tensor:
    """Prefill fast path for the split ggml banks: per-expert dequant + bf16
    GEMM over sorted route groups.

    The MMVQ/MMQ path runs the decode-class ``mul_mat_q4_K`` kernel at prefill
    batch sizes -- profiled at 94% of prefill GPU time and ~1% of gfx1100 peak
    at T=2048. Here the banks are already on-GPU (prefill materializes the
    layer), so dequantizing each routed expert (a ~1-2MB device-local read)
    and running rocBLAS over its sorted token group amortizes the same packed
    bytes into a compute-bound GEMM (~30x measured per call shape).
    """
    from freetoken.models.gguf.dequant import dequantize

    act_fn = _ACT[activation]
    qt_gu, qt_dn = int(quant_types[0]), int(quant_types[1])
    num_tokens, h = hidden_states.shape
    i_size = gate_q.shape[1]
    top_k = topk_ids.shape[1]
    slots = gate_q.shape[0]

    flat_e = topk_ids.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    # token per SORTED row: sorted row i serves flat position order[i], whose
    # source token is order[i] // top_k. Group i's rows are the contiguous
    # slice [start, start+cnt) of this mapping — index it by the slice, never
    # by the flat positions again (double-indexing silently permutes tokens).
    tokens = order.div(top_k, rounding_mode="floor")
    counts = torch.bincount(flat_e, minlength=slots).tolist()

    inter = torch.empty(
        (num_tokens * top_k, i_size), dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    out = torch.empty(
        (num_tokens * top_k, h), dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    start = 0
    for e, cnt in enumerate(counts):
        if cnt == 0:
            continue
        grp = order[start:start + cnt]
        xg = hidden_states.index_select(0, tokens[start:start + cnt])
        w_g = dequantize(gate_q[e], qt_gu, torch.bfloat16).view(i_size, -1)
        w_u = dequantize(up_q[e], qt_gu, torch.bfloat16).view(i_size, -1)
        g = xg @ w_g.t()
        u = xg @ w_u.t()
        inter[grp] = act_fn(torch.cat([g, u], dim=-1))
        start += cnt
    start = 0
    for e, cnt in enumerate(counts):
        if cnt == 0:
            continue
        grp = order[start:start + cnt]
        w_dn = dequantize(down_q[e], qt_dn, torch.bfloat16).view(h, -1)
        out[grp] = inter[start:start + cnt] @ w_dn.t()
        start += cnt
    out = out.to(torch.float32) * topk_weights.reshape(-1, 1).to(torch.float32)
    return (
        out.reshape(num_tokens, top_k, h).to(hidden_states.dtype).sum(dim=1)
    )
