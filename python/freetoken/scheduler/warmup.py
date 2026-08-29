"""Boot-time prefill-shape warmup: compile the autotuned Triton kernels' full
specialization space before the first real request.

The GDN (fla) and attention prefill kernels are ``@triton.autotune``-decorated with
keys over chunk counts (BC) and integer-arg divisibility classes, so every novel
extend length -- and every novel KV-depth bucket a chunked prefill walks through --
triggers a fresh config sweep (compile + bench per config) *inside the request's
forward*. Measured on the 27B Ridge split config: a first-hit extend of 266 tokens
stalled 70s, 255 stalled 129s, ~620 stalled 270s; the identical replay ran in 3-6s
(warm, with ``cache_results=True`` persisting winners to the on-disk triton cache).
A single stall over the frontend's 300s SSE idle limit is what killed long-prompt
sessions at the first deep request.

The warmup submits short synthetic requests through the REAL scheduler path
(admission -> prepare -> forward -> drain -> free), so every kernel the serving
path can launch -- including the decode step at full context depth -- is compiled
while the server is still booting, and its winning autotune config lands in the
persistent cache. Steady-state cost (fully cached) is pure GPU time: the shape
ladder is tens of sub-second prefills and the depth walk is one chunked prefill
at warm throughput.
"""

from __future__ import annotations

import os


def prefill_warmup_lengths(max_extend: int) -> list[int]:
    """Extend lengths whose first forward compiles a new specialization class.

    Covers each integer-divisibility class exactly once (1..16), every GDN chunk
    boundary and power of two up to ``max_extend`` with a +/-1 shoulder (the
    off-by-one variants hit different divisibility/remainder specializations),
    and ``max_extend`` itself (the chunked-prefill steady shape).
    """
    if max_extend <= 0:
        return []
    lengths: set[int] = set(range(1, min(16, max_extend) + 1))
    for base in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        if base > max_extend:
            break
        for t in (base - 1, base, base + 1):
            if 1 <= t <= max_extend:
                lengths.add(t)
    lengths.add(max_extend)
    return sorted(lengths)


def warmup_enabled() -> bool:
    return os.getenv("FREETOKEN_PREFILL_WARMUP", "1") in {"1", "true", "yes", "on"}


def warmup_depth(max_seq_len: int) -> int:
    """Context depth the warmup walks to (tokens). Full ``max_seq_len`` by default --
    the deep buckets are exactly where a first-hit stall is fatal (300s SSE idle
    limit); cap with FREETOKEN_WARMUP_MAX_DEPTH on slower or shared boots."""
    cap = int(os.getenv("FREETOKEN_WARMUP_MAX_DEPTH", "0") or 0)
    return min(max_seq_len, cap) if cap > 0 else max_seq_len
