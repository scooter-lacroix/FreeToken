"""GGML block-quant dequantization in pure torch (the formats this repo's GGUF
checkpoints use: Q4_0, Q6_K, plus trivial F32/F16/BF16).

This is the *reference / CPU* path, NOT the engine's hot path: GGUF weights stay
packed and are dequantized inside the borrowed ggml CUDA kernels (see
``freetoken.kernel.gguf``). These routines are used only to (a) materialize the few
dense F32/F16 tensors at load (norms, scales, router) via :func:`dequantize`, and
(b) cross-check the CUDA kernels in tests. The ``BLOCK_SHAPE`` table and
:func:`row_bytes` are the type metadata the packed (kernel) path also relies on.

Each ``dequant_*`` takes the raw little-endian bytes as a ``uint8`` tensor whose
final axis spans whole blocks, and returns the values in *storage order* (ggml's
fastest axis first); the caller reshapes to the torch shape (``dims[::-1]``). The
math mirrors ``ggml-quants.c``.
"""

from __future__ import annotations

import torch

# ggml_type enum values (subset present in these checkpoints).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q6_K = 14
GGML_BF16 = 30

# (block numel, bytes per block) per ggml type.
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q8_0: (32, 34),
    GGML_Q4_K: (256, 144),
    GGML_Q6_K: (256, 210),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_BF16: "BF16",
    GGML_Q4_0: "Q4_0",
    GGML_Q8_0: "Q8_0",
    GGML_Q4_K: "Q4_K",
    GGML_Q6_K: "Q6_K",
}


def row_bytes(numel: int, ggml_type: int) -> int:
    """Packed byte length of one row of ``numel`` elements in ``ggml_type`` blocks.

    Single source of truth for the ``numel // block * type_size`` math shared by the
    packed-weight ops (``GGUFLinear``/``GGUFEmbedding``) and the expert bank loaders.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    assert numel % block == 0, (
        f"{numel} not a multiple of block {block} for {GGML_NAME.get(ggml_type, ggml_type)}"
    )
    return numel // block * type_size


def _f16_scales(raw: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Reinterpret bytes ``[lo:hi]`` (2 per block) of each block row as fp16 -> fp32 [N,1]."""
    return raw[:, lo:hi].contiguous().view(torch.float16).to(torch.float32)


def dequant_q4_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0: per 32-elem block = fp16 scale ``d`` + 16 packed nibbles; ``w = d*(q-8)``.

    Byte ``j`` of the 16 holds element ``j`` in its low nibble and ``j+16`` in its high
    nibble, so storage order within the block is ``[lo0..lo15, hi0..hi15]``.
    """
    raw = raw.reshape(-1, 18)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    qs = raw[:, 2:18]  # [N,16] uint8
    lo = (qs & 0x0F).to(torch.float32)
    hi = (qs >> 4).to(torch.float32)
    q = torch.cat([lo, hi], dim=1)  # [N,32]
    return ((q - 8.0) * d).reshape(-1).to(out_dtype)


def dequant_q6_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q6_K: 256-elem super-block = 128B low nibbles + 64B high 2-bits + 16 int8
    sub-scales + fp16 ``d``. Direct vectorization of ggml's two-half loop."""
    raw = raw.reshape(-1, 210)
    n = raw.shape[0]
    ql = raw[:, 0:128]  # [n,128]
    qh = raw[:, 128:192]  # [n,64]
    sc = raw[:, 192:208].view(torch.int8).to(torch.float32)  # [n,16]
    d = _f16_scales(raw, 208, 210)  # [n,1]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # l in 0..15 -> is=0; l in 16..31 -> is=1 (per ggml: is = l/16).
    is_idx = (torch.arange(32, device=raw.device) // 16)  # [32] in {0,1}
    for h in range(2):  # two 128-elem halves of the super-block
        qlh = ql[:, h * 64:(h + 1) * 64]  # [n,64]
        qhh = qh[:, h * 32:(h + 1) * 32]  # [n,32]
        sch = sc[:, h * 8:(h + 1) * 8]  # [n,8]
        a = qlh[:, 0:32].to(torch.int32)  # ql[l]
        b = qlh[:, 32:64].to(torch.int32)  # ql[l+32]
        hb = qhh.to(torch.int32)  # qh[l]
        q1 = ((a & 0x0F) | (((hb >> 0) & 3) << 4)) - 32
        q2 = ((b & 0x0F) | (((hb >> 2) & 3) << 4)) - 32
        q3 = ((a >> 4) | (((hb >> 4) & 3) << 4)) - 32
        q4 = ((b >> 4) | (((hb >> 6) & 3) << 4)) - 32
        s1 = sch.index_select(1, is_idx + 0).to(torch.float32)
        s2 = sch.index_select(1, is_idx + 2).to(torch.float32)
        s3 = sch.index_select(1, is_idx + 4).to(torch.float32)
        s4 = sch.index_select(1, is_idx + 6).to(torch.float32)
        base = h * 128
        y[:, base + 0:base + 32] = d * s1 * q1.to(torch.float32)
        y[:, base + 32:base + 64] = d * s2 * q2.to(torch.float32)
        y[:, base + 64:base + 96] = d * s3 * q3.to(torch.float32)
        y[:, base + 96:base + 128] = d * s4 * q4.to(torch.float32)
    return y.reshape(-1).to(out_dtype)


def _scale_min_k4(q12: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack Q4_K's 12 packed 6-bit scale/min bytes into 8 (d, m) pairs.

    Direct port of get_scale_min_k4 (dequantize.cuh): the low 6 bits of bytes
    0..3 / 4..7 hold scales 0..3 / mins 0..3; the high 2 bits of those bytes
    splice into bytes 8..11, which carry the remaining nibble of each.'''
    """
    n = q12.shape[0]
    sc = torch.empty((n, 8), dtype=torch.float32, device=q12.device)
    mn = torch.empty((n, 8), dtype=torch.float32, device=q12.device)
    sc[:, 0:4] = q12[:, 0:4] & 63
    mn[:, 0:4] = q12[:, 4:8] & 63
    sc[:, 4:8] = (q12[:, 8:12] & 0x0F) | ((q12[:, 0:4] >> 6) << 4)
    mn[:, 4:8] = (q12[:, 8:12] >> 4) | ((q12[:, 4:8] >> 6) << 4)
    return sc, mn


def dequant_q4_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_K: 256-elem super-block = fp16 ``d`` + fp16 ``dmin`` + 12 packed 6-bit
    sub-scales + 128 packed nibbles; 8 sub-blocks of 32 with ``w = d*sc*q - dmin*mn``.

    Vectorization of dequantize_block_q4_K: byte chunk ``il`` (32 bytes) drives
    outputs ``[64*il, 64*il+32)`` via low nibbles (sub-block ``2*il``) and
    ``[64*il+32, 64*il+64)`` via high nibbles (sub-block ``2*il+1``)."""
    raw = raw.reshape(-1, 144)
    n = raw.shape[0]
    dall = _f16_scales(raw, 0, 2)  # [n,1]
    dmin = _f16_scales(raw, 2, 4)  # [n,1]
    sc, mn = _scale_min_k4(raw[:, 4:16])  # [n,8] each
    qs = raw[:, 16:144].reshape(n, 4, 32)

    d_sc = dall * sc  # [n,8]
    d_mn = dmin * mn  # [n,8]
    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    for il in range(4):
        lo = (qs[:, il, :] & 0x0F).to(torch.float32)  # [n,32]
        hi = (qs[:, il, :] >> 4).to(torch.float32)
        s_lo = d_sc[:, 2 * il : 2 * il + 1]  # [n,1]
        m_lo = d_mn[:, 2 * il : 2 * il + 1]
        s_hi = d_sc[:, 2 * il + 1 : 2 * il + 2]
        m_hi = d_mn[:, 2 * il + 1 : 2 * il + 2]
        base = 64 * il
        y[:, base : base + 32] = s_lo * lo - m_lo
        y[:, base + 32 : base + 64] = s_hi * hi - m_hi
    return y.reshape(-1).to(out_dtype)


def requantize_q4_k(w: torch.Tensor) -> torch.Tensor:
    """Quantize flat fp32 values to raw Q4_K block bytes (uint8).

    Not ggml's exact encoder -- a scale-quantized round-trip encoder matched to
    :func:`dequant_q4_k`: per 32-value sub-block ``w' = D*q - M`` with
    ``D = (hi-lo)/15, M = -lo``; the super-block fp16 ``dall/dmin`` scale the
    integer 6-bit sub-scales. Used to uniformize mixed-quant GGUF checkpoints
    (e.g. Q4_K_M's Q6_K tensors) into the offload bank format.
    """
    w = w.reshape(-1, 256).to(torch.float32)
    n = w.shape[0]
    sub = w.reshape(n, 8, 32)
    lo = sub.min(dim=2).values  # [n,8]
    hi = sub.max(dim=2).values
    # representable range per sub-block is [-M, 15*D - M] with M >= 0 (dequant
    # subtracts dmin*mn): M = max(0, -lo); D = (hi + M) / 15 covers [lo, hi].
    M = (-lo).clamp_min(0.0)
    D = ((hi + M) / 15.0).clamp_min(1e-12)
    dall = (D.max(dim=1, keepdim=True).values / 63.0).clamp_min(1e-12)
    dmin = (M.max(dim=1, keepdim=True).values / 63.0).clamp_min(1e-12)
    sc = (D / dall).clamp(0, 63).round().to(torch.int64)  # [n,8]
    mn = (M / dmin).clamp(0, 63).round().to(torch.int64)
    # encode against the quantized effective scales to minimize error
    Dq = (dall * sc.to(torch.float32)).clamp_min(1e-12)
    Mq = dmin * mn.to(torch.float32)
    q = ((sub + Mq.unsqueeze(2)) / Dq.unsqueeze(2)).round().clamp(0, 15).to(torch.uint8)

    out = torch.empty((n, 144), dtype=torch.uint8, device=w.device)
    out[:, 0:2] = dall.to(torch.float16).view(torch.uint8).reshape(n, 2)
    out[:, 2:4] = dmin.to(torch.float16).view(torch.uint8).reshape(n, 2)
    # 12 kquant scale bytes: inverse of _scale_min_k4. sc/mn 0..3 = low 6 bits
    # of bytes 0..3 / 4..7; the high 2 bits of those bytes carry the top bits of
    # sc/mn 4..7, whose low nibbles live in bytes 8..11.
    out[:, 4:8] = ((sc[:, 0:4] & 63) | (((sc[:, 4:8] >> 4) & 3) << 6)).to(torch.uint8)
    out[:, 8:12] = ((mn[:, 0:4] & 63) | (((mn[:, 4:8] >> 4) & 3) << 6)).to(torch.uint8)
    out[:, 12:16] = ((sc[:, 4:8] & 0xF) | ((mn[:, 4:8] & 0xF) << 4)).to(torch.uint8)
    # nibble packing mirrors dequant: chunk il's byte t packs output[64il+t] of
    # sub-block 2il (low) with output[64il+32+t] of sub-block 2il+1 (high).
    q8 = q.reshape(n, 8, 32)
    qs = torch.empty((n, 128), dtype=torch.uint8, device=w.device)
    for il in range(4):
        lo_vals = q8[:, 2 * il, :]
        hi_vals = q8[:, 2 * il + 1, :]
        qs[:, 32 * il : 32 * il + 32] = (lo_vals & 0xF) | ((hi_vals & 0xF) << 4)
    out[:, 16:144] = qs
    return out.reshape(-1)


_DEQUANT = {
    GGML_Q4_0: dequant_q4_0,
    GGML_Q4_K: dequant_q4_k,
    GGML_Q6_K: dequant_q6_k,
}


def dequantize(raw: torch.Tensor, ggml_type: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``raw`` (uint8) of any supported ggml type to flat ``out_dtype``."""
    if ggml_type == GGML_F32:
        return raw.view(torch.float32).to(out_dtype)
    if ggml_type == GGML_F16:
        return raw.view(torch.float16).to(out_dtype)
    if ggml_type == GGML_BF16:
        return raw.view(torch.bfloat16).to(out_dtype)
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"dequant for ggml type {GGML_NAME.get(ggml_type, ggml_type)} not implemented"
        )
    return fn(raw, out_dtype)


__all__ = [
    "GGML_F32",
    "GGML_F16",
    "GGML_BF16",
    "GGML_Q4_0",
    "GGML_Q8_0",
    "GGML_Q4_K",
    "GGML_Q6_K",
    "GGML_NAME",
    "BLOCK_SHAPE",
    "row_bytes",
    "dequant_q4_0",
    "dequant_q4_k",
    "requantize_q4_k",
    "dequant_q6_k",
    "dequantize",
]
