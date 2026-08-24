"""Triton k-quant GEMV for dense projections (Q4_K, W4A16).

Byte-space formulation: every packed nibble byte is loaded ONCE and feeds
both of its output columns (low-nibble element ``64*(q//32) + q%32`` and
its +32 high-nibble partner), so DRAM traffic equals the packed size --
unlike a per-column gather, which reads each byte twice. Scales/mins come
from per-lane gathers over the 12 header bytes (L1-resident). bf16
activations, no q8 pre-pass.

Numerics note (hard-won): every per-lane value is computed from loads
whose offsets come from POINTER arithmetic on lane index vectors, and all
selects run between same-shaped tensors -- mixed-shape ``tl.where``
broadcasts silently mis-selected scales in an earlier draft. The torch
replication of this math matches the CPU reference bit-exactly, and the
kernel matches the ggml vec kernel to q8-activation-quant noise (~5e-3).
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

    # byte-space lanes: byte q of the 128 nibble bytes
    q = tl.arange(0, 128)
    qb = q // 32          # 32-byte chunk (64 output columns each)
    r = q % 32
    c_lo = 64 * qb + r    # low-nibble column of byte q
    # sub-block scales for the two columns of byte q
    sb_lo = 2 * qb
    sb_hi = 2 * qb + 1
    u_lo = sb_lo % 4
    u_hi = sb_hi % 4
    lo_sel_lo = sb_lo < 4
    lo_sel_hi = sb_hi < 4

    for kb in range(nblk):
        b = rows.to(tl.int64) * rb + kb * 144
        d_lo = tl.load(w_ptr + b).to(tl.int32)
        d_hi = tl.load(w_ptr + b + 1).to(tl.int32)
        dall = (d_lo | (d_hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        m_lo = tl.load(w_ptr + b + 2).to(tl.int32)
        m_hi = tl.load(w_ptr + b + 3).to(tl.int32)
        dmin = (m_lo | (m_hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)

        A_s = tl.load(w_ptr + b[:, None] + 4 + u_lo[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        B_s = tl.load(w_ptr + b[:, None] + 12 + u_lo[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        A_m = tl.load(w_ptr + b[:, None] + 8 + u_lo[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        sc_lo = tl.where(lo_sel_lo[None, :], (A_s & 63).to(tl.float32),
                         ((B_s & 15) | ((A_s >> 6) << 4)).to(tl.float32)) * dall[:, None]
        mn_lo = tl.where(lo_sel_lo[None, :], (A_m & 63).to(tl.float32),
                         ((B_s >> 4) | ((A_m >> 6) << 4)).to(tl.float32)) * dmin[:, None]
        A_s = tl.load(w_ptr + b[:, None] + 4 + u_hi[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        B_s = tl.load(w_ptr + b[:, None] + 12 + u_hi[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        A_m = tl.load(w_ptr + b[:, None] + 8 + u_hi[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        sc_hi = tl.where(lo_sel_hi[None, :], (A_s & 63).to(tl.float32),
                         ((B_s & 15) | ((A_s >> 6) << 4)).to(tl.float32)) * dall[:, None]
        mn_hi = tl.where(lo_sel_hi[None, :], (A_m & 63).to(tl.float32),
                         ((B_s >> 4) | ((A_m >> 6) << 4)).to(tl.float32)) * dmin[:, None]

        nb = tl.load(w_ptr + b[:, None] + 16 + q[None, :],
                     mask=rmask[:, None], other=0).to(tl.int32)  # [BN,128] coalesced
        w_lo = sc_lo * (nb & 15).to(tl.float32) - mn_lo
        w_hi = sc_hi * (nb >> 4).to(tl.float32) - mn_hi
        xl = tl.load(x_ptr + t.to(tl.int64) * K + kb * 256 + c_lo).to(tl.float32)
        xh = tl.load(x_ptr + t.to(tl.int64) * K + kb * 256 + c_lo + 32).to(tl.float32)
        acc += tl.sum(w_lo * xl[None, :] + w_hi * xh[None, :], axis=1)

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


@triton.jit
def _kq_moe_gemv_q4k(
    w_ptr, ids_ptr, x_ptr, y_ptr,
    K, R, rb, TOPK,
    PER_ROW_X: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One expert's row-block: ``y[t, k, rows] = W[slot(t,k)][rows] @ x``.

    ``w`` is the slot-cache bank ``[S, R, rb]``; ``ids`` holds slot ids
    (post-ensure rewrite). ``PER_ROW_X`` selects the down-projection input
    (per-(t,k) activations) vs the shared gate/up input.
    """
    pid = tl.program_id(0)
    t = tl.program_id(1)
    k = tl.program_id(2)
    slot = tl.load(ids_ptr + t * TOPK + k).to(tl.int64)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < R
    nblk = K // 256
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # slot ids index a [S, R, rb] bank: the slot base offset can exceed int32
    expert_base = slot * (R.to(tl.int64) * rb)

    q = tl.arange(0, 128)
    qb = q // 32
    r = q % 32
    c_lo = 64 * qb + r
    sb_lo = 2 * qb
    sb_hi = 2 * qb + 1
    u_lo = sb_lo % 4
    u_hi = sb_hi % 4
    lo_sel_lo = sb_lo < 4
    lo_sel_hi = sb_hi < 4

    if PER_ROW_X:
        xb = (t.to(tl.int64) * TOPK + k) * K
    else:
        xb = t.to(tl.int64) * K

    for kb in range(nblk):
        b = expert_base + rows.to(tl.int64) * rb + kb * 144
        d_lo = tl.load(w_ptr + b).to(tl.int32)
        d_hi = tl.load(w_ptr + b + 1).to(tl.int32)
        dall = (d_lo | (d_hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        m_lo = tl.load(w_ptr + b + 2).to(tl.int32)
        m_hi = tl.load(w_ptr + b + 3).to(tl.int32)
        dmin = (m_lo | (m_hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)

        A_s = tl.load(w_ptr + b[:, None] + 4 + u_lo[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        B_s = tl.load(w_ptr + b[:, None] + 12 + u_lo[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        A_m = tl.load(w_ptr + b[:, None] + 8 + u_lo[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        sc_lo = tl.where(lo_sel_lo[None, :], (A_s & 63).to(tl.float32),
                         ((B_s & 15) | ((A_s >> 6) << 4)).to(tl.float32)) * dall[:, None]
        mn_lo = tl.where(lo_sel_lo[None, :], (A_m & 63).to(tl.float32),
                         ((B_s >> 4) | ((A_m >> 6) << 4)).to(tl.float32)) * dmin[:, None]
        A_s = tl.load(w_ptr + b[:, None] + 4 + u_hi[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        B_s = tl.load(w_ptr + b[:, None] + 12 + u_hi[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        A_m = tl.load(w_ptr + b[:, None] + 8 + u_hi[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        sc_hi = tl.where(lo_sel_hi[None, :], (A_s & 63).to(tl.float32),
                         ((B_s & 15) | ((A_s >> 6) << 4)).to(tl.float32)) * dall[:, None]
        mn_hi = tl.where(lo_sel_hi[None, :], (A_m & 63).to(tl.float32),
                         ((B_s >> 4) | ((A_m >> 6) << 4)).to(tl.float32)) * dmin[:, None]

        nb = tl.load(w_ptr + b[:, None] + 16 + q[None, :],
                     mask=rmask[:, None], other=0).to(tl.int32)
        w_lo = sc_lo * (nb & 15).to(tl.float32) - mn_lo
        w_hi = sc_hi * (nb >> 4).to(tl.float32) - mn_hi
        xl = tl.load(x_ptr + xb + kb * 256 + c_lo).to(tl.float32)
        xh = tl.load(x_ptr + xb + kb * 256 + c_lo + 32).to(tl.float32)
        acc += tl.sum(w_lo * xl[None, :] + w_hi * xh[None, :], axis=1)

    yb = (t.to(tl.int64) * TOPK + k) * R
    tl.store(y_ptr + yb + rows, acc.to(tl.bfloat16), mask=rmask)


def kq_moe_gemv(bank: torch.Tensor, ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Grouped expert GEMV over the Q4_K slot-cache bank.

    ``bank`` [S, R, rb]; ``ids`` [T, topk] int32 slot ids; ``x`` [T, K] (shared
    gate/up input) or a contiguous [T, topk, K] (per-expert down input).
    Returns [T, topk, R] bf16.

    Split-K form: each (row-block, k-block, expert) is its own program and
    partial dots land in an fp32 scratch via atomics -- the per-expert row
    counts (512-2048) are too small to fill the GPU without splitting K,
    which starved the non-split variant at ~73 GB/s effective.
    """
    S, R, rb = bank.shape
    K = (rb // 144) * 256
    nblk = K // 256
    T, topk = ids.shape
    per_row = x.dim() == 3
    if per_row:
        assert x.shape[0] == T and x.shape[1] == topk
    y32 = torch.zeros(T, topk, R, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(R, 16), nblk, T * topk)
    _kq_moe_gemv_q4k_sk[grid](
        bank, ids, x, y32, K, R, rb, topk,
        PER_ROW_X=per_row, BLOCK_N=16, num_warps=4,
    )
    return y32.to(torch.bfloat16)


@triton.jit
def _kq_moe_gemv_q4k_sk(
    w_ptr, ids_ptr, x_ptr, y_ptr,
    K, R, rb, TOPK,
    PER_ROW_X: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    kb = tl.program_id(1)
    tk = tl.program_id(2)
    t = tk // TOPK
    k = tk % TOPK
    slot = tl.load(ids_ptr + tk).to(tl.int64)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < R
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    expert_base = slot * (R.to(tl.int64) * rb)

    q = tl.arange(0, 128)
    qb = q // 32
    r = q % 32
    c_lo = 64 * qb + r
    sb_lo = 2 * qb
    sb_hi = 2 * qb + 1
    u_lo = sb_lo % 4
    u_hi = sb_hi % 4
    lo_sel_lo = sb_lo < 4
    lo_sel_hi = sb_hi < 4

    if PER_ROW_X:
        xb = tk.to(tl.int64) * K
    else:
        xb = t.to(tl.int64) * K

    b = expert_base + rows.to(tl.int64) * rb + kb * 144
    d_lo = tl.load(w_ptr + b).to(tl.int32)
    d_hi = tl.load(w_ptr + b + 1).to(tl.int32)
    dall = (d_lo | (d_hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    m_lo = tl.load(w_ptr + b + 2).to(tl.int32)
    m_hi = tl.load(w_ptr + b + 3).to(tl.int32)
    dmin = (m_lo | (m_hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)

    A_s = tl.load(w_ptr + b[:, None] + 4 + u_lo[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    B_s = tl.load(w_ptr + b[:, None] + 12 + u_lo[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    A_m = tl.load(w_ptr + b[:, None] + 8 + u_lo[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    sc_lo = tl.where(lo_sel_lo[None, :], (A_s & 63).to(tl.float32),
                     ((B_s & 15) | ((A_s >> 6) << 4)).to(tl.float32)) * dall[:, None]
    mn_lo = tl.where(lo_sel_lo[None, :], (A_m & 63).to(tl.float32),
                     ((B_s >> 4) | ((A_m >> 6) << 4)).to(tl.float32)) * dmin[:, None]
    A_s = tl.load(w_ptr + b[:, None] + 4 + u_hi[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    B_s = tl.load(w_ptr + b[:, None] + 12 + u_hi[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    A_m = tl.load(w_ptr + b[:, None] + 8 + u_hi[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    sc_hi = tl.where(lo_sel_hi[None, :], (A_s & 63).to(tl.float32),
                     ((B_s & 15) | ((A_s >> 6) << 4)).to(tl.float32)) * dall[:, None]
    mn_hi = tl.where(lo_sel_hi[None, :], (A_m & 63).to(tl.float32),
                     ((B_s >> 4) | ((A_m >> 6) << 4)).to(tl.float32)) * dmin[:, None]

    nb = tl.load(w_ptr + b[:, None] + 16 + q[None, :],
                 mask=rmask[:, None], other=0).to(tl.int32)
    w_lo = sc_lo * (nb & 15).to(tl.float32) - mn_lo
    w_hi = sc_hi * (nb >> 4).to(tl.float32) - mn_hi
    xl = tl.load(x_ptr + xb + kb * 256 + c_lo).to(tl.float32)
    xh = tl.load(x_ptr + xb + kb * 256 + c_lo + 32).to(tl.float32)
    acc += tl.sum(w_lo * xl[None, :] + w_hi * xh[None, :], axis=1)

    tl.atomic_add(y_ptr + tk.to(tl.int64) * R + rows, acc, mask=rmask)


__all__ = ["kq_gemv", "kq_moe_gemv"]
