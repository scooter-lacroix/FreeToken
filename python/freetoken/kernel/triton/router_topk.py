"""Single-launch router top-k for MoE decode (the no-triton_kernels path).

The pure-torch fallback pays a softmax + a sort-based ``torch.topk`` +
renormalization -- five-plus kernels including hipC++ merge-sort topk -- per
MoE layer per step. On RDNA3 profiling showed that cluster costs ~1.8 ms/token
at Ornith shapes (bs<=8, E=256, k=8). One small Triton program per token does
the whole thing:

- ``renormalize=True``: top-k of the LOGITS is top-k of the softmax, and the
  renormalized weights are exactly a softmax over the selected logits
  (``e^{x_i}/sum_topk e^{x_j}``), so no full-row softmax is needed.
- ``renormalize=False``: one extra reduction computes the full-row log-sum-exp
  so selected probabilities are exact.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _router_topk_kernel(
    logits_ptr,
    w_ptr,
    id_ptr,
    E,
    K: tl.constexpr,
    RENORM: tl.constexpr,
    BLOCK: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < E
    base = logits_ptr + pid.to(tl.int64) * E
    lg = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)

    lse_sum = 0.0
    mx = 0.0
    if not RENORM:
        # full-row log-sum-exp for exact selected probabilities
        mx = tl.max(lg, 0)
        lse_sum = tl.sum(tl.where(mask, tl.exp(lg - mx), 0.0), 0)

    # iterative top-k by masked argmax extraction (k is tiny; row fits registers)
    offs_k = tl.arange(0, BK)
    sel_v = tl.full((BK,), float("-inf"), dtype=tl.float32)
    work = lg
    for j in tl.static_range(K):
        v = tl.max(work, 0)
        i = tl.argmax(work, 0).to(tl.int32)
        tl.store(id_ptr + pid.to(tl.int64) * K + j, i)
        sel_j = tl.sum(tl.where(offs == i, v, 0.0), 0)
        # knock out the winner for the next pass
        work = tl.where(offs == i, float("-inf"), work)
        sel_v = tl.where(offs_k == j, sel_j, sel_v)

    if RENORM:
        # softmax over the selected logits (== renormalized softmax probs);
        # padding lanes are -inf so they contribute zero
        mxs = tl.max(sel_v, 0)
        e = tl.exp(sel_v - mxs)
        s = tl.sum(e, 0) + 1e-20
        out = e / s
    else:
        out = tl.exp(sel_v - mx) / (lse_sum + 1e-20)
    tl.store(w_ptr + pid.to(tl.int64) * K + offs_k, out, mask=offs_k < K)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def router_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton fused router: returns (float32 weights [T,K], int32 ids [T,K])."""
    assert gating_output.is_cuda and gating_output.dtype in (torch.bfloat16, torch.float16, torch.float32)
    T, E = gating_output.shape
    w = torch.empty(T, topk, dtype=torch.float32, device=gating_output.device)
    ids = torch.empty(T, topk, dtype=torch.int32, device=gating_output.device)
    _router_topk_kernel[(T,)](
        gating_output,
        w,
        ids,
        E,
        K=topk,
        RENORM=renormalize,
        BLOCK=_next_pow2(E),
        BK=_next_pow2(topk),
        num_warps=4,
    )
    return w, ids
