"""Far-side Triton cache seeder (slot count 13 = hybrid_radix mr=2 live/ping-pong/snapshot/cache/padding — the conv1d kernel specializes num_cache_lines as a constexpr, so this must match the server) (dual-GPU layer split, XTX + 7800 XT).

On this stack, Triton kernels COMPILE/AUTOTUNE fine on whichever GPU
initializes first, but compiling (or autotuning -- each config is a launch)
on the SECOND visible GPU livelocks the process at ~90-100% CPU with zero
cache writes. Cache-HIT LOADS on the second GPU work fine (proven by the
MTP engine running Triton on cuda:1 all session from disk-cached kernels).

So: run this seeder with the 7800 XT as the SOLE visible device --

    HIP_VISIBLE_DEVICES=1 python tools/mtp/far_seed.py

It invokes every kernel family the far side of the split uses, with the
same signatures/constexprs the server will request, writing compile +
autotune results into the shared Triton disk cache (keys are arch-based;
the HSA_OVERRIDE in the user env makes both GPUs report gfx1100, so the
keys match the dual-GPU server's cuda:1 launches). The subsequent
FREETOKEN_LAYER_SPLIT server then only ever cache-hits on the far side.
"""

import torch


def seed():
    from freetoken.models.qwen3_5_moe.gdn_kernels import (
        gdn_decode_fla, gdn_prefill_chunk_fla,
    )
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
    from freetoken.kernel.triton.attention import (
        decode_paged_attention, extend_paged_attention, paged_attention,
    )
    from freetoken.kernel.triton.kquant_linear import kq_gemv

    dev = torch.device("cuda:0")  # sole visible device = the 7800 XT
    NK, NV, D = 16, 48, 128       # Qwen3.8-27B GDN heads
    H, KVH, HD = 5120, 4, 256     # full-attention shapes
    V, K = 5120, 2048

    # --- fla GDN chunk (prefill ladder shapes) + decode ---
    for T in (80, 128, 256, 512, 1024, 2048):
        q = torch.randn(1, T, NK, D, device=dev).bfloat16()
        k = torch.randn_like(q)
        v = torch.randn(1, T, NV, D, device=dev).bfloat16()
        g = torch.zeros(1, T, NV, device=dev)
        beta = torch.ones(1, T, NV, device=dev)
        st = torch.zeros(13, NV, D, D, device=dev)
        cu = torch.tensor([0, T], dtype=torch.int64, device=dev)
        idx = torch.tensor([1], dtype=torch.int32, device=dev)
        gdn_prefill_chunk_fla(q, k, v, g, beta, state_source=st,
                              indices=idx, cu_seqlens=cu, scale=D ** -0.5)
        torch.cuda.synchronize()
        print(f"  fla chunk T={T} ok", flush=True)
    for bs in (1, 2, 4):
        q = torch.randn(1, bs, NK, D, device=dev).bfloat16()
        k, v = torch.randn_like(q), torch.randn(1, bs, NV, D, device=dev).bfloat16()
        a = torch.randn(bs, NV, device=dev)
        b = torch.randn(bs, NV, device=dev)
        st = torch.zeros(13, NV, D, D, device=dev)
        idx = torch.arange(bs, dtype=torch.int32, device=dev)
        cu = torch.arange(bs + 1, dtype=torch.int32, device=dev)
        gdn_decode_fla(q, k, v, a, b, A_log=torch.zeros(NV, device=dev),
                       dt_bias=torch.zeros(NV, device=dev), state_source=st,
                       indices=idx, cu_seqlens=cu, scale=D ** -0.5)
        torch.cuda.synchronize()
        print(f"  fla decode bs={bs} ok", flush=True)

    # --- causal conv1d (varlen prefill + decode) ---
    conv_dim = 2 * NK * D + NV * D
    w = torch.randn(conv_dim, 4, device=dev).bfloat16()  # [conv_dim, kernel]
    st_conv = torch.zeros(13, conv_dim, 3, device=dev).bfloat16()
    for T in (80, 128, 256, 512, 1024, 2048):
        x = torch.randn(conv_dim, T, device=dev).bfloat16()  # channels-first
        cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
        idx = torch.tensor([1], dtype=torch.int32, device=dev)
        causal_conv1d_varlen(x, w, st_conv, cu, idx,
                             torch.tensor([True], device=dev))
        torch.cuda.synchronize()
    for bs in (1, 2, 4):
        x = torch.randn(bs, conv_dim, device=dev).bfloat16()
        causal_conv1d_decode(x, st_conv, w, torch.arange(bs, device=dev))
        torch.cuda.synchronize()
    print("  causal conv1d ok", flush=True)

    # --- paged attention: extend (ladder) + decode + generic ---
    kc = torch.randn(4096, KVH, HD, device=dev).bfloat16()
    vc = torch.randn_like(kc)
    for ql in (80, 128, 256, 512, 1024):
        q = torch.randn(ql, H, HD, device=dev).bfloat16()
        qi = torch.tensor([0, ql], device=dev)
        ki = torch.tensor([0, 4096], device=dev)
        ix = torch.arange(4096, device=dev)
        pl = torch.tensor([4096 - ql], device=dev)
        extend_paged_attention(q, kc, vc, qi, ki, ix, pl, ql, HD ** -0.5,
                               None, None, None,
                               kc[:ql].view(-1, KVH, HD), vc[:ql].view(-1, KVH, HD))
        torch.cuda.synchronize()
        print(f"  extend attn ql={ql} ok", flush=True)
    for bs in (1, 2, 4):
        q = torch.randn(bs, H, HD, device=dev).bfloat16()
        logits = torch.empty(bs, H, 8, HD, device=dev, dtype=torch.float32)
        lse = torch.empty(bs, H, 8, device=dev, dtype=torch.float32)
        nks = torch.full((bs,), 8, dtype=torch.int32, device=dev)
        decode_paged_attention(q, kc, vc, torch.arange(bs + 1, device=dev),
                               torch.arange(4096, device=dev),
                               torch.zeros(bs, dtype=torch.int32, device=dev),
                               logits, lse, nks, 8, HD ** -0.5)
        torch.cuda.synchronize()
        print(f"  decode attn bs={bs} ok", flush=True)

    # --- kq_gemv (far-side Q4_K projections, decode) ---
    for N in (34816, 17408, 12288, 10240, 6144, 5120):
        wq = torch.randint(0, 255, (N, (K // 256) * 144), dtype=torch.uint8, device=dev)
        x = torch.randn(1, K, device=dev).bfloat16()
        kq_gemv(wq, x, 12)
        torch.cuda.synchronize()
    print("  kq_gemv ok", flush=True)

    # --- rope (far-side full-attention layers; head 256, partial rd 64) ---
    from freetoken.kernel.triton.rope import apply_rope_with_cos_sin_cache_inplace

    RD = 64
    inv = 1.0 / (1e7 ** (torch.arange(0, RD, 2, dtype=torch.float32, device=dev) / RD))
    t = torch.arange(8192, dtype=torch.float32, device=dev)
    cs = torch.cat([(torch.outer(t, inv)).cos(), (torch.outer(t, inv)).sin()], dim=-1).contiguous()
    for T in (1, 2, 80, 128, 256, 512, 1024, 2048):
        q = torch.randn(T, 24 * 256, device=dev).bfloat16()
        k = torch.randn(T, 4 * 256, device=dev).bfloat16()
        pos = torch.arange(T, dtype=torch.int64, device=dev)
        apply_rope_with_cos_sin_cache_inplace(pos, q, k, 256, cs, True)
        torch.cuda.synchronize()
    print("  rope ok", flush=True)

    print("FAR-SIDE TRITON CACHE SEEDED", flush=True)


if __name__ == "__main__":
    seed()
