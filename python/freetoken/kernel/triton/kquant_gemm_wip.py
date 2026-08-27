"""Triton tiled dequant-GEMM for GGML K-quant weights (the prefill fast path).

The vendored ggml ``mul_mat_q4_K`` MMQ kernel is catastrophically slow on this
stack (measured 196 ms/call at [1024,5120]x[5120,17408] vs 4.9 ms rocBLAS bf16;
it owns >99% of prefill GPU time on the qwen35-dense trunk). This kernel
streams packed rows like the memory-bound bf16 GEMM it replaces, so no bf16
weight twin is needed; scales are decoded once per tensor via dequant.py's
verified ``_scale_min_k4`` so the hot loop only reads quant bytes.

Layout mirrors models/gguf/dequant.py ``dequant_q4_k``: raw [N, rb] rows are
output features with blocks along K; per sub-block,
w = d*sc*q_nibble - dmin*mn_q.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.models.gguf.dequant import _scale_min_k4


@triton.jit
def _mm_dequant_q4k_kernel(
    x_ptr, qs_ptr, sc_ptr, mn_ptr, d_ptr, dmin_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_om,
    QSTRIDE: tl.constexpr,           # byte stride of packed rows
    NUM_SB: tl.constexpr,            # K // 256
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mn2 = mask_m[:, None] & mask_n[None, :]

    d_all = tl.load(d_ptr + offs_n, mask=mask_n, other=0.0)[:, None]
    d_min = tl.load(dmin_ptr + offs_n, mask=mask_n, other=0.0)[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    x_row = x_ptr + offs_m[:, None] * stride_xm

    for il_any in range(NUM_SB * 4):
        sb = il_any // 4
        il = il_any % 4
        qs = tl.load(
            qs_ptr + offs_n[:, None] * QSTRIDE + sb * 128 +
            (il * 32 + tl.arange(0, 32))[None, :],
            mask=mask_n[:, None], other=0,
        ).to(tl.int32)                                    # [BN,32]

        c_lo = sb * 256 + il * 64
        sl_a = tl.load(sc_ptr + offs_n[:, None] * 8 + (2 * il), mask=mask_n[:, None], other=0.0).to(tl.float32)
        ml_a = tl.load(mn_ptr + offs_n[:, None] * 8 + (2 * il), mask=mask_n[:, None], other=0.0).to(tl.float32)
        sl_b = tl.load(sc_ptr + offs_n[:, None] * 8 + (2 * il + 1), mask=mask_n[:, None], other=0.0).to(tl.float32)
        ml_b = tl.load(mn_ptr + offs_n[:, None] * 8 + (2 * il + 1), mask=mask_n[:, None], other=0.0).to(tl.float32)

        w_lo = sl_a * (qs & 0x0F).to(tl.float32) - ml_a                # [BN,32]
        w_hi = sl_b * ((qs >> 4) & 0x0F).to(tl.float32) - ml_b

        xa_lo = tl.load(x_row + (c_lo + tl.arange(0, 32))[None, :],
                        mask=mask_m[:, None] & ((c_lo + tl.arange(0, 32)) < K)[None, :],
                        other=0.0).to(tl.float32)
        xa_hi = tl.load(x_row + (c_lo + 32 + tl.arange(0, 32))[None, :],
                        mask=mask_m[:, None] & ((c_lo + 32 + tl.arange(0, 32)) < K)[None, :],
                        other=0.0).to(tl.float32)

        acc += tl.dot(xa_lo, tl.trans(w_lo.to(xa_lo.dtype)), out_dtype=tl.float32)
        acc += tl.dot(xa_hi, tl.trans(w_hi.to(xa_hi.dtype)), out_dtype=tl.float32)

    om = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(om, acc.to(out_ptr.dtype.element_ty), mask=mn2)


_CACHE: dict[int, tuple] = {}


def _q4k_scales(raw: torch.Tensor):
    """One-time per-tensor decode of (d, dmin, sc[N,8] f32, mn[N,8] f32)."""
    key = raw.data_ptr()
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    d_all, d_min = _f16_pair(raw[:, 0:2])
    sc, mn = _scale_min_k4(raw[:, 4:16])
    vals = (
        d_all.contiguous(), d_min.contiguous(),
        sc.to(torch.float32).contiguous(), mn.to(torch.float32).contiguous(),
    )
    if len(_CACHE) > 256:
        _CACHE.clear()
    _CACHE[key] = vals
    return vals


def _f16_pair(raw2: torch.Tensor):
    """fp16 bit-pairs decoded via numpy view (torch lacks usable uint16 views);
    4 bytes per row pulled to host ONCE per weight tensor."""
    import numpy as np

    host = raw2.contiguous().cpu().numpy()
    d = np.ascontiguousarray(host[:, :2]).view("<f2").astype("float32")
    mn = np.ascontiguousarray(host[:, 2:4]).view("<f2").astype("float32")
    dev = raw2.device
    return (
        torch.from_numpy(d).to(dev),
        torch.from_numpy(mn).to(dev),
    )


def mm_dequant_q4k(x: torch.Tensor, raw: torch.Tensor, out=None) -> torch.Tensor:
    """[T,K] @ (packed Q4_K [N, rb]) -> [T,N]. K multiple of 256."""
    T, K = x.shape
    N, rb = raw.shape[0], raw.shape[1]
    assert K % 256 == 0 and rb == K // 256 * 144
    assert x.is_contiguous() and raw.is_contiguous()
    d_all, d_min, sc, mn = _q4k_scales(raw)
    if out is None:
        out = torch.empty((T, N), dtype=x.dtype, device=x.device)
    bm, bn, nw = (16, 128, 8) if T <= 64 else (64, 64, 8)
    grid = (triton.cdiv(T, bm), triton.cdiv(N, bn))
    _mm_dequant_q4k_kernel[grid](
        x, raw[:, 16:], sc, mn, d_all, d_min, out, T, N, K,
        x.stride(0), out.stride(0),
        QSTRIDE=raw.stride(0), NUM_SB=K // 256,
        BM=bm, BN=bn, num_warps=nw, num_stages=3,
    )
    return out
