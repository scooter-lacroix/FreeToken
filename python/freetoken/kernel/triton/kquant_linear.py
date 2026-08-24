"""Triton k-quant GEMV for dense projections (Q4_K / Q6_K, W4A16).

The borrowed ggml ``mul_mat_vec`` kernels run the dense projections at
~130-150 GB/s effective on gfx1100 (Q8-quantized activations, un-tuned
port). This kernel streams the packed block bytes with wide coalesced
loads, dequantizes in registers against the validated CPU reference
(models/gguf/dequant.py), and dots directly with the bf16 activation --
no q8 pre-pass, no extra launches.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kq_gemv_q4k(
    w_ptr, x_ptr, y_ptr, K,
    N, rb,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    nblk = K // 256
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for kb in range(nblk):
        b = rows.to(tl.int64) * rb + kb * 144
        lo16 = tl.load(w_ptr + b).to(tl.uint16)
        hi16 = tl.load(w_ptr + b + 1).to(tl.uint16)
        dall = (lo16 | (hi16 << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        lo16 = tl.load(w_ptr + b + 2).to(tl.uint16)
        hi16 = tl.load(w_ptr + b + 3).to(tl.uint16)
        dmin = (lo16 | (hi16 << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        q0 = tl.load(w_ptr + b + 4, mask=rmask, other=0)
        q1 = tl.load(w_ptr + b + 5, mask=rmask, other=0)
        q2 = tl.load(w_ptr + b + 6, mask=rmask, other=0)
        q3 = tl.load(w_ptr + b + 7, mask=rmask, other=0)
        q4 = tl.load(w_ptr + b + 8, mask=rmask, other=0)
        q5 = tl.load(w_ptr + b + 9, mask=rmask, other=0)
        q6 = tl.load(w_ptr + b + 10, mask=rmask, other=0)
        q7 = tl.load(w_ptr + b + 11, mask=rmask, other=0)
        q8 = tl.load(w_ptr + b + 12, mask=rmask, other=0)
        q9 = tl.load(w_ptr + b + 13, mask=rmask, other=0)
        q10 = tl.load(w_ptr + b + 14, mask=rmask, other=0)
        q11 = tl.load(w_ptr + b + 15, mask=rmask, other=0)
        sc0 = (q0 & 63).to(tl.float32)
        sc1 = (q1 & 63).to(tl.float32)
        sc2 = (q2 & 63).to(tl.float32)
        sc3 = (q3 & 63).to(tl.float32)
        mn0 = (q4 & 63).to(tl.float32)
        mn1 = (q5 & 63).to(tl.float32)
        mn2 = (q6 & 63).to(tl.float32)
        mn3 = (q7 & 63).to(tl.float32)
        sc4 = ((q8 & 0x0F) | ((q0 >> 6) << 4)).to(tl.float32)
        sc5 = ((q9 & 0x0F) | ((q1 >> 6) << 4)).to(tl.float32)
        sc6 = ((q10 & 0x0F) | ((q2 >> 6) << 4)).to(tl.float32)
        sc7 = ((q11 & 0x0F) | ((q3 >> 6) << 4)).to(tl.float32)
        mn4 = ((q8 >> 4) | ((q4 >> 6) << 4)).to(tl.float32)
        mn5 = ((q9 >> 4) | ((q5 >> 6) << 4)).to(tl.float32)
        mn6 = ((q10 >> 4) | ((q6 >> 6) << 4)).to(tl.float32)
        mn7 = ((q11 >> 4) | ((q7 >> 6) << 4)).to(tl.float32)
        c = tl.arange(0, 256)
        il = c // 64
        within = c % 64
        j = within % 32
        hi = (within >= 32).to(tl.int32)
        # column c's nibble lives at byte (il*32 + j), high nibble for the
        # second 32-lane half of each chunk
        byte_idx = il * 32 + j
        nibs = tl.load(w_ptr + b[:, None] + 16 + byte_idx[None, :],
                       mask=rmask[:, None], other=0).to(tl.int32)  # [BN,256]
        nib = (nibs >> (4 * hi[None, :])) & 15
        sc_lo = tl.where(il == 0, sc0[:, None], tl.where(il == 1, sc1[:, None],
                    tl.where(il == 2, sc2[:, None], sc3[:, None])))
        sc_hi = tl.where(il == 0, sc4[:, None], tl.where(il == 1, sc5[:, None],
                    tl.where(il == 2, sc6[:, None], sc7[:, None])))
        mn_lo = tl.where(il == 0, mn0[:, None], tl.where(il == 1, mn1[:, None],
                    tl.where(il == 2, mn2[:, None], mn3[:, None])))
        mn_hi = tl.where(il == 0, mn4[:, None], tl.where(il == 1, mn5[:, None],
                    tl.where(il == 2, mn6[:, None], mn7[:, None])))
        sc = tl.where(hi == 1, sc_hi, sc_lo) * dall[:, None]
        mn = tl.where(hi == 1, mn_hi, mn_lo) * dmin[:, None]
        w = sc * nib.to(tl.float32) - mn
        xk = tl.load(x_ptr + t.to(tl.int64) * K + kb * 256 + c).to(tl.float32)
        acc += tl.sum(w * xk[None, :], axis=1)
    tl.store(y_ptr + t.to(tl.int64) * N + rows, acc.to(tl.bfloat16), mask=rmask)


def kq_gemv(w: torch.Tensor, x: torch.Tensor, quant_type: int) -> torch.Tensor:
    """Dense GEMV over Q4_K-packed rows (``w`` [N, 144*K/256] uint8, ``x`` [T,K])."""
    assert quant_type == 12, "Triton path currently covers Q4_K"
    N = w.shape[0]
    K = (w.shape[1] // 144) * 256
    T = x.shape[0]
    y = torch.empty(T, N, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(N, 16), T)
    _kq_gemv_q4k[grid](w, x, y, K, N, w.shape[1], BLOCK_N=16, num_warps=8)
    return y


__all__ = ["kq_gemv"]
