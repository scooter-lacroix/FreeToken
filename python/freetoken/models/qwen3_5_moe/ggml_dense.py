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
        if batch is not None and batch.is_prefill and not getattr(batch, "is_verify", False):
            # spec-verify needs EVERY row's logits (per-row argmax accept), not just
            # each request's last row
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

    # NOTE: the historical exclusion of verify batches (T=k>1) from the ggml
    # GEMM path was RIGHT to leave (the a8 GEMM at M=8 measured ~2.7s/step —
    # the prefill profile's 196ms-vs-4.9ms class), but the Triton GEMV is
    # only validated at T=1 (T=8 produced garbage). Verify batches fall
    # through to the ggml VEC kernel below: validated for M<=8 (the LM head
    # has run it since the S4 driver) and its 8x weight re-read is ~15x
    # cheaper than the a8 GEMM.
    _verify_batch = False
    try:
        from freetoken.core import get_global_ctx as _gctx

        _verify_batch = getattr(getattr(_gctx(), "batch", None), "is_verify", False)
    except Exception:                                   # noqa: BLE001
        pass
    if (
        quant_type == 12
        and x.shape[0] <= 8
        and out_features >= 2048
        and os.environ.get("FREETOKEN_TRITON_KQ", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    ):
        import os as _os_fused

        if (_verify_batch and x.shape[0] > 1
                and _os_fused.environ.get("FREETOKEN_FUSED_M8", "1")
                not in {"0", "false", "no"}):
            if not globals().get("_m8_dbg"):
                globals()["_m8_dbg"] = True
                print(f"[m8-dbg] x shape={tuple(x.shape)} stride={x.stride()} "
                      f"dtype={x.dtype} contig={x.is_contiguous()} "
                      f"ptr%16={x.data_ptr() % 16} w ptr%16={w.data_ptr() % 16} "
                      f"N={out_features}", flush=True)
            if _os_fused.environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
                from freetoken.kernel.triton.kquant_linear import kq_gemm_q4k_m8 as _m8k

                st = globals().setdefault("_m8_stash", [])
                if len(st) < 2:
                    y_m8 = _m8k(w, x, quant_type)
                    st.append((w, x, y_m8, len(st)))
                    return y_m8
            # fused skinny-M GEMM (DEFAULT ON, gate-green 09-04): bit-exact vs
            # the T=1 GEMV oracle on real weights, faithful under isolated and
            # in-serving graph capture, and the full spec gate (same-prompt,
            # two-thread, repeats) passed coherent end-to-end with it live:
            # 'The user wants me to say hello in one sentence...' identical to
            # normal decode. ~2x on the verify step (177ms vs 344ms).
            from freetoken.kernel.triton.kquant_linear import kq_gemm_q4k_m8

            return kq_gemm_q4k_m8(w, x, quant_type)
        from freetoken.kernel.triton.kquant_linear import kq_gemv

        return kq_gemv(w, x, quant_type)
    from freetoken.kernel.gguf import ggml_mul_mat_a8, ggml_mul_mat_vec_a8

    # vec kernel for skinny batches including verify (see note above: the
    # a8 GEMM at M=8 measured ~2.7s/step; 8x weight re-reads win by ~15x)
    if x.shape[0] <= 8:
        if quant_type == 14 and out_features * 256 <= w.shape[1] * 210 * 4096:
            # Q6_K Triton GEMV (real-weight parity ~bf16 rounding; ~578GB/s vs the
            # vendored vec kernel's ~500 at [5120,6144]) — same guard shape as Q4_K.
            from freetoken.kernel.triton.kquant_linear import kq_gemv_q6k

            return kq_gemv_q6k(w, x, quant_type)
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

    _stats: dict = {}

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
        import os as _os

        _trace = _os.environ.get("FREETOKEN_QGL_TRACE", "0") == "1"
        if _trace:
            import time as _t

            _t0 = _t.perf_counter()
        used_bf16 = False
        if x.shape[0] > _PREFILL_BF16_MIN_T:
            fast = _prefill_bf16_matmul(self.packed, self.quant_type)
            if fast is not None:
                used_bf16 = True
                # fast is [N,K] row-major == col-major [K,N]: rocBLAS consumes
                # the transpose without a copy
                out = x @ fast.t()
                if _trace:
                    QuantGgmlLinear._stats[("bf16", x.shape[0])] = (
                        QuantGgmlLinear._stats.get(("bf16", x.shape[0]), 0) + 1
                    )
                    QuantGgmlLinear._stats["_bf16_s"] = (
                        QuantGgmlLinear._stats.get("_bf16_s", 0.0) + _t.perf_counter() - _t0
                    )
                return out
        out = _kq_gemv(self.packed, x, self.quant_type, self.out_features)
        if _trace:
            QuantGgmlLinear._stats[("mmq", x.shape[0])] = (
                QuantGgmlLinear._stats.get(("mmq", x.shape[0]), 0) + 1
            )
            QuantGgmlLinear._stats["_mmq_s"] = (
                QuantGgmlLinear._stats.get("_mmq_s", 0.0) + _t.perf_counter() - _t0
            )
            n = sum(v for k, v in QuantGgmlLinear._stats.items() if k != "_bf16_s" and k != "_mmq_s")
            if n % 500 == 0:
                bf = QuantGgmlLinear._stats.get("_bf16_s", 0.0)
                mq = QuantGgmlLinear._stats.get("_mmq_s", 0.0)
                top = sorted(
                    ((k, v) for k, v in QuantGgmlLinear._stats.items()
                     if isinstance(k, tuple)),
                    key=lambda kv: -kv[1],
                )[:6]
                print(
                    f"[qgl-stats] n={n} bf16_s={bf:.1f} mmq_s={mq:.1f} top={top}",
                    flush=True,
                )
        return out


# ---------------------------------------------------------------------------
# Prefill bf16 materialization cache (prompt-processing fast path).
#
# Profiled 2026-08-26: the vendored mul_mat_q4_K MMQ runs 196 ms/call at
# [1024,5120]x[5120,17408] on gfx1100 vs 4.9 ms for rocBLAS over
# pre-dequantized bf16 (40x), and owns >99% of prefill GPU time (the q5/q6/IQ
# families are all fine at 1-2 ms/call). For big-M calls we lazily dequantize
# the packed weight ONCE into a device-cached bf16 twin and route through
# rocBLAS; the twin lives in a FIFO cache bounded by FREETOKEN_BF16_CACHE_GB
# (weights stay resident normally — requests touch layers sequentially, so one
# long prompt's chunks plus its decode hits re-materialize nothing).

_PREFILL_BF16_MIN_T = int(
    __import__("os").environ.get("FREETOKEN_PREFILL_BF16_MIN_T", "64")
)
_bf16_cache: dict[int, tuple[torch.Tensor, int]] = {}
_bf16_bytes = 0


def _prefill_bf16_matmul(packed: torch.Tensor, qt: int):
    """bf16 [N,K] twin for this packed weight, or None when the type/env/
    VRAM budget says use the quant kernels instead."""
    import os

    if os.environ.get("FREETOKEN_PREFILL_BF16", "1").strip().lower() in {
        "0", "false", "no", "off"
    }:
        return None
    if qt not in (8, 12, 13, 14, 21, 22):   # verified dequant families only
        return None
    key = packed.data_ptr()
    hit = _bf16_cache.get(key)
    if hit is not None:
        return hit[0]

    from freetoken.models.gguf.dequant import BLOCK_SHAPE, dequantize

    n, rb = packed.shape
    bnum, bsize = BLOCK_SHAPE[qt]
    assert rb % bsize == 0
    k = (rb // bsize) * bnum
    need = n * k * 2
    budget = int(
        float(os.environ.get("FREETOKEN_BF16_CACHE_GB", "0")) * (1 << 30)
    )
    if budget <= 0:
        # no persistence: dequant per use (~20-40ms/layer) — removes the
        # 48-layer-vs-cap eviction storms that surfaced as multi-second
        # serving freezes
        return _dequant_chunked(packed, qt, n, k)
    global _bf16_bytes
    while _bf16_bytes + need > budget and _bf16_cache:
        evict_key = next(iter(_bf16_cache))
        _, sz = _bf16_cache.pop(evict_key)
        _bf16_bytes -= sz
    w = _dequant_chunked(packed, qt, n, k)
    _bf16_cache[key] = (w, need)
    _bf16_bytes += need
    return w


def _dequant_chunked(packed: torch.Tensor, qt: int, n: int, k: int) -> torch.Tensor:
    """Row-chunked dequant to bound fp32 transients (~70MB per 4096 rows).

    Q4_K goes through the fused single-kernel Triton dequant (bit-exact,
    ~139 GB/s vs the elementwise port's ~16 on gfx1100 -- the per-use twin
    path is the dominant prefill phase at ~30ms/layer x 65 layers).
    """
    if packed.is_cuda and k % 256 == 0:
        from freetoken.kernel.triton import kquant_dequant as _kdf

        flat = packed.reshape(-1)
        if qt == 12:
            return _kdf.dequant_q4_k_fused(flat, torch.bfloat16).view(n, k)
        if qt == 22:
            return _kdf.dequant_iq2_s_fused(flat, torch.bfloat16).view(n, k)
        if qt == 21:
            return _kdf.dequant_iq3_s_fused(flat, torch.bfloat16).view(n, k)

    from freetoken.models.gguf.dequant import dequantize

    out = torch.empty((n, k), dtype=torch.bfloat16, device=packed.device)
    CH = 4096
    for lo in range(0, n, CH):
        hi = min(lo + CH, n)
        out[lo:hi] = dequantize(packed[lo:hi], qt, torch.bfloat16).view(hi - lo, k)
    return out


__all__ = ["GgufKQuantLMHead", "QuantGGMLEmbedding", "QuantGgmlLinear"]
