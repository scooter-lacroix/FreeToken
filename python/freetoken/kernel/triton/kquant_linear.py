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


@triton.jit
def _kq_gemv_iq3s(
    w_ptr, x_ptr, y_ptr, tbl_ptr, K,
    N, rb,
    BLOCK_N: tl.constexpr,
):
    """IQ3_S GEMV: 110-byte 256-elem superblocks (fp16 d | 64B qs | 8B qh |
    32B signs | 4B nibble scales). Lane q = 4*ib + il covers 8 consecutive
    columns (base 8q); two 9-bit grid indices per lane gather 4 signed values
    each from the [512,4] iq3xs table (2 KB -> L1)."""
    pid = tl.program_id(0)
    t = tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    nblk = K // 256
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    q = tl.arange(0, 32)                    # lane = 4*ib + il
    ib = q // 4
    il = q % 4
    sh1 = 8 - 2 * il                        # idx1 high-bit shifts
    sh2 = 7 - 2 * il
    km = 1 << tl.arange(0, 4)               # sign bits 1,2,4,8 (g1) / 16..128 (g2)

    for kb in range(nblk):
        b = rows.to(tl.int64) * rb + kb * 110
        d_lo = tl.load(w_ptr + b).to(tl.int32)
        d_hi = tl.load(w_ptr + b + 1).to(tl.int32)
        d = (d_lo | (d_hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)

        g1b = tl.load(w_ptr + b[:, None] + 2 + 2 * q[None, :],
                      mask=rmask[:, None], other=0).to(tl.int32)
        g2b = tl.load(w_ptr + b[:, None] + 3 + 2 * q[None, :],
                      mask=rmask[:, None], other=0).to(tl.int32)
        qhb = tl.load(w_ptr + b[:, None] + 66 + ib[None, :],
                      mask=rmask[:, None], other=0).to(tl.int32)
        idx1 = g1b | ((qhb << sh1[None, :]) & 0x100)
        idx2 = g2b | ((qhb << sh2[None, :]) & 0x100)

        sc = tl.load(w_ptr + b[:, None] + 106 + (ib[None, :] // 2),
                     mask=rmask[:, None], other=0).to(tl.int32)
        nib = (sc >> (4 * (ib[None, :] % 2))) & 0xF
        d_eff = d[:, None] * (0.5 + nib.to(tl.float32)) * 0.5   # [BN,32]

        sgn = tl.load(w_ptr + b[:, None] + 74 + q[None, :],
                      mask=rmask[:, None], other=0).to(tl.int32)

        xb = kb * 256 + 8 * q               # column base per lane
        for k in tl.static_range(4):
            g1 = tl.load(tbl_ptr + idx1 * 4 + k, mask=rmask[:, None], other=0
                         ).to(tl.int8).to(tl.float32)
            g2 = tl.load(tbl_ptr + idx2 * 4 + k, mask=rmask[:, None], other=0
                         ).to(tl.int8).to(tl.float32)
            s1 = tl.where((sgn & (1 << k)) != 0, -1.0, 1.0)
            s2 = tl.where((sgn & (16 << k)) != 0, -1.0, 1.0)
            x1 = tl.load(x_ptr + t.to(tl.int64) * K + xb[None, :] + k).to(tl.float32)
            x2 = tl.load(x_ptr + t.to(tl.int64) * K + xb[None, :] + 4 + k).to(tl.float32)
            acc += tl.sum(d_eff * (g1 * s1 * x1 + g2 * s2 * x2), axis=1)

    tl.store(y_ptr + t.to(tl.int64) * N + rows, acc.to(tl.bfloat16), mask=rmask)


def kq_gemv_iq3s(w: torch.Tensor, x: torch.Tensor, quant_type: int) -> torch.Tensor:
    assert w.shape[1] * 256 % 110 == 0, "Triton path currently covers IQ3_S (110B blocks)"
    from freetoken.models.gguf._iq_tables import iq3xs_grid_u8

    N = w.shape[0]
    K = (w.shape[1] // 110) * 256
    T = x.shape[0]
    y = torch.empty(T, N, dtype=torch.bfloat16, device=x.device)
    tbl = iq3xs_grid_u8().to(x.device, non_blocking=True)
    if N >= 8192:
        BN, W = 32, 8
    else:
        BN, W = 64, 16
    grid = (triton.cdiv(N, BN), T)
    _kq_gemv_iq3s[grid](w, x, y, tbl, K, N, w.shape[1], BLOCK_N=BN, num_warps=W)
    return y


@triton.jit
def _kq_gemv_iq2s(
    w_ptr, x_ptr, y_ptr, tbl_ptr, K,
    N, rb,
    BLOCK_N: tl.constexpr,
):
    """IQ2_S GEMV: 82-byte 256-elem superblocks (fp16 d | 64B qs | 8B qh |
    8B nibble scales). Lane q = 4*ib + il covers 8 consecutive columns (base
    8q); ONE 10-bit grid index per lane gathers 8 signed values from the
    [1024,8] iq2s table (8 KB -> L1). Sign bits: qs[32+q] & (1<<k)."""
    pid = tl.program_id(0)
    t = tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    nblk = K // 256
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    q = tl.arange(0, 32)                    # lane = 4*ib + il
    il = q % 4
    shifts = 8 - 2 * il                     # per-lane qh shift

    for kb in range(nblk):
        b = rows.to(tl.int64) * rb + kb * 82
        d_lo = tl.load(w_ptr + b).to(tl.int32)
        d_hi = tl.load(w_ptr + b + 1).to(tl.int32)
        d = (d_lo | (d_hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)

        qs_lo = tl.load(w_ptr + b[:, None] + 2 + q[None, :],
                        mask=rmask[:, None], other=0).to(tl.int32)
        sgn = tl.load(w_ptr + b[:, None] + 34 + q[None, :],
                      mask=rmask[:, None], other=0).to(tl.int32)
        qhb = tl.load(w_ptr + b[:, None] + 66 + (q[None, :] // 4),
                      mask=rmask[:, None], other=0).to(tl.int32)
        idx = qs_lo | ((qhb << shifts[None, :]) & 0x300)

        sc = tl.load(w_ptr + b[:, None] + 74 + (q[None, :] // 4),
                     mask=rmask[:, None], other=0).to(tl.int32)
        nib = (sc >> (4 * (il[None, :] // 2))) & 0xF
        d_eff = d[:, None] * (0.5 + nib.to(tl.float32)) * 0.25   # [BN,32]

        xb = kb * 256 + 8 * q
        for k in tl.static_range(8):
            gv = tl.load(tbl_ptr + idx * 8 + k, mask=rmask[:, None], other=0
                         ).to(tl.int8).to(tl.float32)
            s1 = tl.where((sgn & (1 << k)) != 0, -1.0, 1.0)
            xv = tl.load(x_ptr + t.to(tl.int64) * K + xb[None, :] + k).to(tl.float32)
            acc += tl.sum(d_eff * gv * s1 * xv, axis=1)

    tl.store(y_ptr + t.to(tl.int64) * N + rows, acc.to(tl.bfloat16), mask=rmask)


def kq_gemv_iq2s(w: torch.Tensor, x: torch.Tensor, quant_type: int) -> torch.Tensor:
    from freetoken.models.gguf._iq_tables import iq2s_grid_u8

    N = w.shape[0]
    K = (w.shape[1] // 82) * 256
    T = x.shape[0]
    y = torch.empty(T, N, dtype=torch.bfloat16, device=x.device)
    tbl = iq2s_grid_u8().to(x.device, non_blocking=True)
    if N >= 8192:
        BN, W = 32, 8
    else:
        BN, W = 64, 16
    grid = (triton.cdiv(N, BN), T)
    _kq_gemv_iq2s[grid](w, x, y, tbl, K, N, w.shape[1], BLOCK_N=BN, num_warps=W)
    return y


@triton.jit
def _kq_gemv_q6k(
    w_ptr, x_ptr, y_ptr, K,
    N, rb,
    BLOCK_N: tl.constexpr,
):
    """Q6_K GEMV: 210-byte 256-elem superblocks (128B ql nibbles | 64B qh
    2-bits | 16 int8 sub-scales | fp16 d). Two 128-elem halves; per half, 32
    lanes produce four quads (lo/hi nibbles of ql[l], ql[l+32] + 2 qh bits)."""
    pid = tl.program_id(0)
    t = tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    nblk = K // 256
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    l = tl.arange(0, 32)

    for kb in range(nblk):
        b = rows.to(tl.int64) * rb + kb * 210
        d_lo = tl.load(w_ptr + b + 208).to(tl.int32)
        d_hi = tl.load(w_ptr + b + 209).to(tl.int32)
        d = (d_lo | (d_hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)

        for h in tl.static_range(2):
            a = tl.load(w_ptr + b[:, None] + h * 64 + l[None, :],
                        mask=rmask[:, None], other=0).to(tl.int32)
            b2 = tl.load(w_ptr + b[:, None] + h * 64 + 32 + l[None, :],
                         mask=rmask[:, None], other=0).to(tl.int32)
            hb = tl.load(w_ptr + b[:, None] + 128 + h * 32 + l[None, :],
                         mask=rmask[:, None], other=0).to(tl.int32)
            q1 = ((a & 0x0F) | ((hb & 3) << 4)) - 32
            q2 = ((b2 & 0x0F) | (((hb >> 2) & 3) << 4)) - 32
            q3 = ((a >> 4) | (((hb >> 4) & 3) << 4)) - 32
            q4 = ((b2 >> 4) | (((hb >> 6) & 3) << 4)) - 32
            scb = b[:, None] + 192 + h * 8 + (l // 16)[None, :]
            s1 = tl.load(w_ptr + scb, mask=rmask[:, None], other=0).to(tl.int8).to(tl.float32)
            s2 = tl.load(w_ptr + scb + 2, mask=rmask[:, None], other=0).to(tl.int8).to(tl.float32)
            s3 = tl.load(w_ptr + scb + 4, mask=rmask[:, None], other=0).to(tl.int8).to(tl.float32)
            s4 = tl.load(w_ptr + scb + 6, mask=rmask[:, None], other=0).to(tl.int8).to(tl.float32)
            xb = kb * 256 + h * 128
            x1 = tl.load(x_ptr + t.to(tl.int64) * K + xb + l[None, :]).to(tl.float32)
            x2 = tl.load(x_ptr + t.to(tl.int64) * K + xb + 32 + l[None, :]).to(tl.float32)
            x3 = tl.load(x_ptr + t.to(tl.int64) * K + xb + 64 + l[None, :]).to(tl.float32)
            x4 = tl.load(x_ptr + t.to(tl.int64) * K + xb + 96 + l[None, :]).to(tl.float32)
            acc += tl.sum(
                d[:, None] * (s1 * q1.to(tl.float32) * x1
                              + s2 * q2.to(tl.float32) * x2
                              + s3 * q3.to(tl.float32) * x3
                              + s4 * q4.to(tl.float32) * x4), axis=1)

    tl.store(y_ptr + t.to(tl.int64) * N + rows, acc.to(tl.bfloat16), mask=rmask)


def kq_gemv_q6k(w: torch.Tensor, x: torch.Tensor, quant_type: int) -> torch.Tensor:
    assert quant_type == 14, "Triton path currently covers Q6_K"
    N = w.shape[0]
    K = (w.shape[1] // 210) * 256
    T = x.shape[0]
    y = torch.empty(T, N, dtype=torch.bfloat16, device=x.device)
    if N >= 16384:
        BN, W = 64, 16
    else:
        BN, W = 32, 8
    _kq_gemv_q6k[(triton.cdiv(N, BN), T)](w, x, y, K, N, w.shape[1],
                                          BLOCK_N=BN, num_warps=W)
    return y


@triton.jit
def _kq_gemm_q4k_m8(
    w_ptr, x_ptr, y_ptr, K,
    N, rb,
    BLOCK_N: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
):
    """Fused skinny-M Q4_K GEMM: one program per row-block reads its weight
    block ONCE and produces all T<=8 token rows via tensor-core dots --
    the verify-batch shape (M=k). The per-token GEMV grid re-read weights T
    times; the ggml a8 GEMM is ~40x slower at this M. Weights are dequantized
    to bf16 and accumulated in fp32 (llama.cpp-class rounding)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    nblk = K // 256
    acc = tl.zeros((TP, BLOCK_N), dtype=tl.float32)

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
    t_ar = tl.arange(0, TP)
    t_live = t_ar < T

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
                     mask=rmask[:, None], other=0).to(tl.int32)
        w_lo = sc_lo * (nb & 15).to(tl.float32) - mn_lo
        w_hi = sc_hi * (nb >> 4).to(tl.float32) - mn_hi

        # No tl.dot: the MMA lowering does not survive CUDA-graph capture on
        # this ROCm stack (garbage from the first replay). A static per-token
        # elementwise accumulate reads the weight block ONCE and keeps fp32
        # math like the proven T=1 GEMV.
        for t in tl.static_range(T):
            xl_t = tl.load(x_ptr + t * K + kb * 256 + c_lo).to(tl.float32)
            xh_t = tl.load(x_ptr + t * K + kb * 256 + c_lo + 32).to(tl.float32)
            contrib = tl.sum(w_lo * xl_t[None, :] + w_hi * xh_t[None, :], axis=1)
            acc = acc + tl.where(t_ar[:, None] == t, contrib[None, :], 0.0)

    for t in tl.static_range(T):
        # acc is (TP, BLOCK_N); pick token row t
        col = tl.sum(tl.where(t_ar[:, None] == t, acc, 0.0), axis=0)
        tl.store(y_ptr + t * N + rows, col.to(tl.bfloat16), mask=rmask)


def kq_gemm_q4k_m8(w: torch.Tensor, x: torch.Tensor, quant_type: int) -> torch.Tensor:
    """Verify-batch dense projection: Q4_K GEMM at 1 < T <= 8 (weights read
    once per block; grid (row-blocks,) -- no per-token weight re-read)."""
    assert quant_type == 12, "fused M8 path covers Q4_K"
    N = w.shape[0]
    K = (w.shape[1] // 144) * 256
    T = x.shape[0]
    assert 1 < T <= 8, T
    y = torch.empty(T, N, dtype=torch.bfloat16, device=x.device)
    if N >= 16384:
        BN, W = 64, 8
    else:
        BN, W = 32, 4
    _kq_gemm_q4k_m8[(triton.cdiv(N, BN),)](
        w, x, y, K, N, w.shape[1], BLOCK_N=BN, T=T, TP=16, num_warps=W)
    return y


def kq_gemv(w: torch.Tensor, x: torch.Tensor, quant_type: int) -> torch.Tensor:
    """Dense GEMV over Q4_K-packed rows (``w`` [N, 144*K/256] uint8, ``x`` [T,K])."""
    assert quant_type == 12, "Triton path currently covers Q4_K"
    N = w.shape[0]
    K = (w.shape[1] // 144) * 256
    T = x.shape[0]
    y = torch.empty(T, N, dtype=torch.bfloat16, device=x.device)
    # gfx1100-swept (idle-card clocks, wide grid): [17408,5120] (64,16)=899GB/s,
    # [10240,5120] (32,8)=791, [6144,5120] (32,8)=761.
    if N >= 16384:
        BN, W = 64, 16
    else:
        BN, W = 32, 8
    grid = (triton.cdiv(N, BN), T)
    _kq_gemv_q4k[grid](w, x, y, K, N, w.shape[1], BLOCK_N=BN, num_warps=W)
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
