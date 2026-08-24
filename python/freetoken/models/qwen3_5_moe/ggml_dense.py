"""Native-GGML k-quant dense modules for the GGUF path (lm_head / embeddings).

Profiling on gfx1100 showed the bf16-dequant LM head cost ~2 ms/token
(248k x 2048 at ~390 GB/s effective) plus ~970 MiB of resident bf16 weight.
Serving the checkpoint's own Q4_K/Q6_K block bytes through the borrowed ggml
kernels cuts the bytes ~4x with dequant-in-kernel; the tied embedding table
gathers rows and dequantizes just those instead of holding a full bf16 copy.

The packed buffers are assigned by the GGUF weight loader under
``<prefix>.packed`` (uint8 ``[rows, row_bytes]``) + ``<prefix>.quant_type``
(scalar int tensor). Untied checkpoints give the head its own tensor; tied
checkpoints yield the same underlying storage under both names (one copy).
"""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP
from freetoken.layers.base import _concat_prefix


class QuantGGMLEmbedding(BaseOP):
    """Token embedding served from native GGML block bytes (gather + dequant)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.packed = torch.empty(0, 0, dtype=torch.uint8)
        self.quant_type = 12

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        w = state_dict.pop(_concat_prefix(prefix, "packed"))
        qt = state_dict.pop(_concat_prefix(prefix, "quant_type")).item()
        from freetoken.models.gguf.dequant import row_bytes

        assert w.dtype == torch.uint8, w.dtype
        assert w.shape == (self.num_embeddings, row_bytes(self.embedding_dim, int(qt))), w.shape
        self.packed = w
        self.quant_type = int(qt)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize

        rows = self.packed[ids.reshape(-1).long()]
        out = ggml_dequantize(rows, self.quant_type, ids.numel(), self.embedding_dim)
        return out.to(torch.bfloat16).reshape(*ids.shape, self.embedding_dim)


class GgufKQuantLMHead(BaseOP):
    """k-quant LM head: Q8-1-quantized activation x ggml block weights.

    Mirrors ``ParallelLMHead``/``Nvfp4LMHead`` at TP=1 (last-token slice at
    prefill, then the quantized GEMV/GEMM over the ``[V, row_bytes]`` table).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.packed = torch.empty(0, 0, dtype=torch.uint8)
        self.quant_type = 12

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        w = state_dict.pop(_concat_prefix(prefix, "packed"))
        qt = state_dict.pop(_concat_prefix(prefix, "quant_type")).item()
        assert w.dtype == torch.uint8, w.dtype
        self.packed = w
        self.quant_type = int(qt)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_mul_mat_a8, ggml_mul_mat_vec_a8

        # tolerate contexts without an active batch (the eager MTP probe)
        batch = getattr(get_global_ctx(), "_batch", None)
        if batch is not None and batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        fn = ggml_mul_mat_vec_a8 if x.shape[0] <= 8 else ggml_mul_mat_a8
        return fn(self.packed, x, self.quant_type, self.num_embeddings)


def _kq_gemv(w, x, quant_type: int, out_features: int):
    """Triton k-quant GEMV when profitable (large-N Q4_K); ggml otherwise.

    Measured on gfx1100: the Triton byte-space kernel beats the ggml vec
    kernel ~8-16% from N~2k upward (534 vs 462 GB/s at [8192,2048]) and loses
    below ~1k rows (launch-bound), so small matrices and other quant types
    stay on the ggml path.
    """
    import os

    if (
        quant_type == 12
        and x.shape[0] <= 8
        and out_features >= 2048
        and os.environ.get("FREETOKEN_TRITON_KQ", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    ):
        from freetoken.kernel.triton.kquant_linear import kq_gemv

        return kq_gemv(w, x, quant_type)
    from freetoken.kernel.gguf import ggml_mul_mat_a8, ggml_mul_mat_vec_a8

    if x.shape[0] <= 8:
        return ggml_mul_mat_vec_a8(w, x, quant_type, out_features)
    return ggml_mul_mat_a8(w, x, quant_type, out_features)


class QuantGgmlLinear(BaseOP):
    """Dense projection over native GGML block bytes (W8A8 k-quant GEMV/GEMM).

    Same fused-matrix layout as ``LinearColParallelMerged`` (consumers split the
    merged output rows exactly as before); the weight stays in the checkpoint's
    k-quant blocks and the borrowed ggml kernels dequantize in-loop. The bf16
    dequant of these projections dominated gfx1100 decode (~16 ms/token at
    ~200 GB/s effective over hipBLASLt GEMV).
    """

    def __init__(self, out_features: int, in_features: int):
        self.out_features = out_features
        self.in_features = in_features
        self.packed = torch.empty(0, 0, dtype=torch.uint8)
        self.quant_type = 12

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        w = state_dict.pop(_concat_prefix(prefix, "packed"))
        qt = state_dict.pop(_concat_prefix(prefix, "quant_type")).item()
        from freetoken.models.gguf.dequant import row_bytes

        assert w.dtype == torch.uint8, w.dtype
        assert w.shape == (self.out_features, row_bytes(self.in_features, int(qt))), (
            w.shape, self.out_features, self.in_features, qt
        )
        self.packed = w
        self.quant_type = int(qt)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _kq_gemv(self.packed, x, self.quant_type, self.out_features)


__all__ = ["GgufKQuantLMHead", "QuantGGMLEmbedding", "QuantGgmlLinear"]
