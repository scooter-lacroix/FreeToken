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

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


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


# Back-compat alias for the Q4_0 path.
fused_experts_gguf_q4_0 = fused_experts_ggml

__all__ = ["fused_experts_ggml", "fused_experts_ggml_mixed", "fused_experts_gguf_q4_0"]
