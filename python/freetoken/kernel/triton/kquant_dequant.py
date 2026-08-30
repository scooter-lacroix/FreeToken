"""Fused Triton dequant for GGML k-quants (per-use bf16 twin path).

The torch-elementwise ``dequantize`` port runs ~5-9 GB/s on gfx1100 (a dozen
intermediate tensors per super-block); at 65 layers x ~178MB bf16 twin per
2048-token chunk that is 2-3s/chunk of pure dequant overhead -- measured as
the dominant prefill phase (mlp gate_up 2.06s + down 1.15s per chunk,
FREETOKEN_PHASE_TIMING=1). One fused kernel per block layout reads the packed
bytes once and writes the twin once: DRAM-bound, ~10-20x the elementwise path.

Numerics: bit-exact to ``freetoken.models.gguf.dequant`` (same
``w = d*sc*q - dmin*mn`` math in fp32, cast to bf16 at the store).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_q4_k_kernel(
    src_ptr,  # uint8 flat, 144 bytes per super-block
    dst_ptr,  # bf16 [n_blocks, 256]
    n_blocks,
    BLOCKS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_q = tl.arange(0, 32)
    for b in range(BLOCKS_PER_PROG):
        blk = pid * BLOCKS_PER_PROG + b
        base = blk * 144

        # fp16 d / dmin (LE bytes 0..3)
        d_bits = tl.load(src_ptr + base + 0).to(tl.uint16) | (
            tl.load(src_ptr + base + 1).to(tl.uint16) << 8
        )
        dm_bits = tl.load(src_ptr + base + 2).to(tl.uint16) | (
            tl.load(src_ptr + base + 3).to(tl.uint16) << 8
        )
        d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)
        dmin = dm_bits.to(tl.float16, bitcast=True).to(tl.float32)

        # 8 sub-scales + 8 sub-mins (6-bit packed, bytes 4..15). Scalar
        # unroll: shuffles are awkward in Triton and 16 byte-loads per
        # 256-output block are noise next to the nibble payload traffic.
        sc0 = tl.load(src_ptr + base + 4).to(tl.float32)
        sc1 = tl.load(src_ptr + base + 5).to(tl.float32)
        sc2 = tl.load(src_ptr + base + 6).to(tl.float32)
        sc3 = tl.load(src_ptr + base + 7).to(tl.float32)
        mn0 = tl.load(src_ptr + base + 8).to(tl.float32)
        mn1 = tl.load(src_ptr + base + 9).to(tl.float32)
        mn2 = tl.load(src_ptr + base + 10).to(tl.float32)
        mn3 = tl.load(src_ptr + base + 11).to(tl.float32)
        sc4 = (
            (tl.load(src_ptr + base + 12).to(tl.float32) * 0
             + ((tl.load(src_ptr + base + 12).to(tl.uint32) & 0x0F)
                | ((tl.load(src_ptr + base + 4).to(tl.uint32) >> 6) << 4)).to(tl.float32))
        )
        sc5 = (
            ((tl.load(src_ptr + base + 13).to(tl.uint32) & 0x0F)
             | ((tl.load(src_ptr + base + 5).to(tl.uint32) >> 6) << 4)).to(tl.float32)
        )
        sc6 = (
            ((tl.load(src_ptr + base + 14).to(tl.uint32) & 0x0F)
             | ((tl.load(src_ptr + base + 6).to(tl.uint32) >> 6) << 4)).to(tl.float32)
        )
        sc7 = (
            ((tl.load(src_ptr + base + 15).to(tl.uint32) & 0x0F)
             | ((tl.load(src_ptr + base + 7).to(tl.uint32) >> 6) << 4)).to(tl.float32)
        )
        mn4 = (
            ((tl.load(src_ptr + base + 12).to(tl.uint32) >> 4)
             | ((tl.load(src_ptr + base + 8).to(tl.uint32) >> 6) << 4)).to(tl.float32)
        )
        mn5 = (
            ((tl.load(src_ptr + base + 13).to(tl.uint32) >> 4)
             | ((tl.load(src_ptr + base + 9).to(tl.uint32) >> 6) << 4)).to(tl.float32)
        )
        mn6 = (
            ((tl.load(src_ptr + base + 14).to(tl.uint32) >> 4)
             | ((tl.load(src_ptr + base + 10).to(tl.uint32) >> 6) << 4)).to(tl.float32)
        )
        mn7 = (
            ((tl.load(src_ptr + base + 15).to(tl.uint32) >> 4)
             | ((tl.load(src_ptr + base + 11).to(tl.uint32) >> 6) << 4)).to(tl.float32)
        )
        # sub-block masks are & 63 in the reference; the loads above kept the
        # raw byte for lanes 0..3 -- apply here.
        sc0m = (tl.load(src_ptr + base + 4).to(tl.uint32) & 63).to(tl.float32)
        sc1m = (tl.load(src_ptr + base + 5).to(tl.uint32) & 63).to(tl.float32)
        sc2m = (tl.load(src_ptr + base + 6).to(tl.uint32) & 63).to(tl.float32)
        sc3m = (tl.load(src_ptr + base + 7).to(tl.uint32) & 63).to(tl.float32)
        mn0m = (tl.load(src_ptr + base + 8).to(tl.uint32) & 63).to(tl.float32)
        mn1m = (tl.load(src_ptr + base + 9).to(tl.uint32) & 63).to(tl.float32)
        mn2m = (tl.load(src_ptr + base + 10).to(tl.uint32) & 63).to(tl.float32)
        mn3m = (tl.load(src_ptr + base + 11).to(tl.uint32) & 63).to(tl.float32)

        # nibble payload bytes 16..143: chunk il drives sub-blocks 2il (low)
        # and 2il+1 (high) at outputs [64il, 64il+64).
        scales = tl.zeros((8,), dtype=tl.float32) + (
            sc0m * 0 + sc1m * 0 + sc2m * 0 + sc3m * 0
        )  # keep dtype stable; real assembly below
        s_lo_0 = d * sc0m
        s_hi_0 = d * sc1m
        s_lo_1 = d * sc2m
        s_hi_1 = d * sc3m
        s_lo_2 = d * sc4
        s_hi_2 = d * sc5
        s_lo_3 = d * sc6
        s_hi_3 = d * sc7
        m_lo_0 = dmin * mn0m
        m_hi_0 = dmin * mn1m
        m_lo_1 = dmin * mn2m
        m_hi_1 = dmin * mn3m
        m_lo_2 = dmin * mn4
        m_hi_2 = dmin * mn5
        m_lo_3 = dmin * mn6
        m_hi_3 = dmin * mn7

        for il in tl.static_range(4):
            q_bytes = tl.load(src_ptr + base + 16 + il * 32 + offs_q).to(tl.uint32)
            lo = (q_bytes & 0x0F).to(tl.float32)
            hi = (q_bytes >> 4).to(tl.float32)
            if il == 0:
                s_lo, m_lo, s_hi, m_hi = s_lo_0, m_lo_0, s_hi_0, m_hi_0
            elif il == 1:
                s_lo, m_lo, s_hi, m_hi = s_lo_1, m_lo_1, s_hi_1, m_hi_1
            elif il == 2:
                s_lo, m_lo, s_hi, m_hi = s_lo_2, m_lo_2, s_hi_2, m_hi_2
            else:
                s_lo, m_lo, s_hi, m_hi = s_lo_3, m_lo_3, s_hi_3, m_hi_3
            o_lo = il * 64 + offs_q
            o_hi = il * 64 + 32 + offs_q
            tl.store(dst_ptr + blk * 256 + o_lo, (s_lo * lo - m_lo).to(tl.bfloat16))
            tl.store(dst_ptr + blk * 256 + o_hi, (s_hi * hi - m_hi).to(tl.bfloat16))


def dequant_q4_k_fused(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Fused Q4_K dequant; falls back to the torch port off-CUDA or for
    non-bf16 outputs."""
    if not raw.is_cuda or out_dtype != torch.bfloat16:
        from freetoken.models.gguf.dequant import dequant_q4_k

        return dequant_q4_k(raw, out_dtype)
    n = raw.numel() // 144
    if n == 0:
        return torch.empty((0, 256), dtype=torch.bfloat16, device=raw.device)
    BLOCKS_PER_PROG = 4
    pad = (-n) % BLOCKS_PER_PROG
    raw_flat = raw.reshape(-1)
    if pad:
        raw_flat = torch.cat(
            [raw_flat, torch.zeros(pad * 144, dtype=torch.uint8, device=raw.device)]
        )
    raw_flat = raw_flat.contiguous()
    out = torch.empty(
        (n + pad, 256), dtype=torch.bfloat16, device=raw.device
    )
    grid = (triton.cdiv(n + pad, BLOCKS_PER_PROG),)
    _dequant_q4_k_kernel[grid](raw_flat, out, n + pad, BLOCKS_PER_PROG=BLOCKS_PER_PROG)
    return out[:n]


@triton.jit
def _dequant_iq2_s_kernel(src_ptr, dst_ptr, grid_ptr, n_blocks, BLOCKS_PER_PROG: tl.constexpr):
    pid = tl.program_id(0)
    off32 = tl.arange(0, 32)   # (ib,il) flattened: ib=off//4, il=off%4
    ib = off32 // 4
    il = off32 % 4
    offs8 = tl.arange(0, 8)
    for b in range(BLOCKS_PER_PROG):
        blk = pid * BLOCKS_PER_PROG + b
        base = blk * 82
        d_bits = tl.load(src_ptr + base + 0).to(tl.uint16) | (tl.load(src_ptr + base + 1).to(tl.uint16) << 8)
        d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)
        qs_lo = tl.load(src_ptr + base + 2 + off32).to(tl.uint32)
        sign_b = tl.load(src_ptr + base + 34 + off32).to(tl.uint32)
        qh = tl.load(src_ptr + base + 66 + ib).to(tl.uint32)
        sc_b = tl.load(src_ptr + base + 74 + ib).to(tl.uint32)
        shift = 8 - 2 * il
        idx = qs_lo | ((qh << shift) & 0x300)
        nib = (sc_b >> (4 * (il // 2))) & 0xF
        d_eff = d * (0.5 + nib.to(tl.float32)) * 0.25
        # grid gather: [32, 8]
        g = tl.load(grid_ptr + idx[:, None] * 8 + offs8[None, :]).to(tl.float32)
        sgn = tl.where((sign_b[:, None] >> offs8[None, :]) & 1 != 0, -1.0, 1.0)
        y = d_eff[:, None] * g * sgn
        tl.store(dst_ptr + blk * 256 + off32[:, None] * 8 + offs8[None, :], y.to(tl.bfloat16))


@triton.jit
def _dequant_iq3_s_kernel(src_ptr, dst_ptr, grid_ptr, n_blocks, BLOCKS_PER_PROG: tl.constexpr):
    pid = tl.program_id(0)
    off32 = tl.arange(0, 32)
    ib = off32 // 4
    il = off32 % 4
    offs4 = tl.arange(0, 4)
    for b in range(BLOCKS_PER_PROG):
        blk = pid * BLOCKS_PER_PROG + b
        base = blk * 110
        d_bits = tl.load(src_ptr + base + 0).to(tl.uint16) | (tl.load(src_ptr + base + 1).to(tl.uint16) << 8)
        d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)
        q1 = tl.load(src_ptr + base + 2 + 8 * ib + 2 * il).to(tl.uint32)
        q2 = tl.load(src_ptr + base + 2 + 8 * ib + 2 * il + 1).to(tl.uint32)
        qh = tl.load(src_ptr + base + 66 + ib).to(tl.uint32)
        sign_b = tl.load(src_ptr + base + 74 + off32).to(tl.uint32)
        sc = tl.load(src_ptr + base + 106 + ib // 2).to(tl.uint32)
        idx1 = q1 | ((qh << (8 - 2 * il)) & 0x100)
        idx2 = q2 | ((qh << (7 - 2 * il)) & 0x100)
        nib = (sc >> (4 * (ib % 2))) & 0xF
        d_eff = d * (0.5 + nib.to(tl.float32)) * 0.5
        g1 = tl.load(grid_ptr + idx1[:, None] * 4 + offs4[None, :]).to(tl.float32)
        g2 = tl.load(grid_ptr + idx2[:, None] * 4 + offs4[None, :]).to(tl.float32)
        s1 = tl.where((sign_b[:, None] >> offs4[None, :]) & 1 != 0, -1.0, 1.0)
        s2 = tl.where((sign_b[:, None] >> (offs4[None, :] + 4)) & 1 != 0, -1.0, 1.0)
        o = blk * 256 + off32[:, None] * 8
        tl.store(dst_ptr + o + offs4[None, :], (d_eff[:, None] * g1 * s1).to(tl.bfloat16))
        tl.store(dst_ptr + o + 4 + offs4[None, :], (d_eff[:, None] * g2 * s2).to(tl.bfloat16))


_GRID_CACHE: dict = {}


def _grid_u8(name: str):
    t = _GRID_CACHE.get(name)
    if t is None:
        from freetoken.models.gguf._iq_tables import iq2s_grid_u8, iq3xs_grid_u8

        t = (iq2s_grid_u8() if name == "iq2s" else iq3xs_grid_u8()).cuda().contiguous()
        _GRID_CACHE[name] = t
    return t


def _fused(raw: torch.Tensor, block_bytes: int, kernel, grid: torch.Tensor) -> torch.Tensor:
    n = raw.numel() // block_bytes
    if n == 0:
        return torch.empty((0, 256), dtype=torch.bfloat16, device=raw.device)
    BLOCKS_PER_PROG = 4
    pad = (-n) % BLOCKS_PER_PROG
    raw_flat = raw.reshape(-1)
    if pad:
        raw_flat = torch.cat([raw_flat, torch.zeros(pad * block_bytes, dtype=torch.uint8, device=raw.device)])
    raw_flat = raw_flat.contiguous()
    out = torch.empty((n + pad, 256), dtype=torch.bfloat16, device=raw.device)
    gridsize = (triton.cdiv(n + pad, BLOCKS_PER_PROG),)
    kernel[gridsize](raw_flat, out, grid, n + pad, BLOCKS_PER_PROG=BLOCKS_PER_PROG)
    return out[:n]


def dequant_iq2_s_fused(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    if not raw.is_cuda or out_dtype != torch.bfloat16:
        from freetoken.models.gguf.dequant import dequant_iq2_s

        return dequant_iq2_s(raw, out_dtype).view(-1, 256)
    return _fused(raw, 82, _dequant_iq2_s_kernel, _grid_u8("iq2s"))


def dequant_iq3_s_fused(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    if not raw.is_cuda or out_dtype != torch.bfloat16:
        from freetoken.models.gguf.dequant import dequant_iq3_s

        return dequant_iq3_s(raw, out_dtype).view(-1, 256)
    return _fused(raw, 110, _dequant_iq3_s_kernel, _grid_u8("iq3xs"))
