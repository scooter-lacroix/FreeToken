"""S4-B keystone: GDN spec-verify mode == k sequential decode steps.

The verify branch processes k rows of one request sequentially through the decode
kernel (in-place state chaining) and logs per-position all-layer snapshots. It must
reproduce, bit-tolerantly: (a) each position's layer output, (b) the final state slot,
(c) per-position snapshots == state after j decode steps. Random weights, small dims,
real kernels on cuda:0.
"""
import sys

import torch

sys.path.insert(0, "/home/scooter/Documents/Product/Stan-s-ML-Stack/Fork/FreeToken/python")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _build(hidden=256, nk=2, nv=4, dk=32, conv=4):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet

    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=nk, num_value_heads=nv,
        key_head_dim=dk, value_head_dim=dk, conv_kernel_dim=conv, output_gate=True,
    )
    pool = LinearStatePool(group=g, num_slots=4, dtype=torch.bfloat16, device=torch.device(DEVICE), tp_size=1)
    layer = Qwen3_5GatedDeltaNet(
        hidden_size=hidden, num_k_heads=nk, num_v_heads=nv, head_k_dim=dk,
        head_v_dim=dk, conv_kernel_size=conv, rms_norm_eps=1e-6, layer_id=0,
    )

    def _to_dev(obj):
        import torch.nn as nn
        from freetoken.layers.base import BaseOP as OP

        keep_f32 = {"A_log", "dt_bias"}
        for name, v in list(obj.__dict__.items()):
            if isinstance(v, nn.Module):
                setattr(obj, name, v.to(DEVICE, dtype=torch.bfloat16))
                for pn, p in v.named_parameters():
                    with torch.no_grad():
                        p.copy_(torch.randn_like(p.float()) * 0.02)
            elif isinstance(v, torch.Tensor):
                dt = torch.float32 if name in keep_f32 else torch.bfloat16
                if v.dtype in (torch.bfloat16, torch.float32):
                    if name not in keep_f32:
                        v = torch.randn_like(v.float()) * 0.02
                setattr(obj, name, v.to(DEVICE, dtype=dt))
            elif isinstance(v, OP):
                _to_dev(v)
        return obj

    return _to_dev(layer), pool


def _ctx_with(pool, batch):
    import freetoken.core as core
    from freetoken.core import Context

    core._GLOBAL_CTX = None  # single test process owns the global
    ctx = Context(page_size=1)
    ctx.linear_state_pool = pool
    core.set_global_ctx(ctx)
    return ctx


def _fla(slot, dev):
    from freetoken.attention.linear import FLAMetadata

    return FLAMetadata(
        cu_seqlens=torch.arange(0, 2, dtype=torch.int32, device=dev),
        cache_indices=torch.tensor([slot], dtype=torch.int32, device=dev),
        has_initial_state=None, fresh_state_indices=None,
    )


def test_spec_verify_equals_sequential_decode():
    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.distributed import set_tp_info

    set_tp_info(rank=0, size=1)

    layer, pool = _build()
    dev = torch.device(DEVICE)
    K = 5
    slot = 1
    pool.recurrent_states[:, slot].zero_()
    pool.conv_states[:, slot].zero_()
    init = pool.snapshot_slot(slot)

    g = torch.Generator(device="cpu").manual_seed(7)
    h = (torch.randn(K, 256, generator=g) * 0.5).to(torch.bfloat16).to(dev)

    def mkbatch(phase):
        r = Req(input_ids=torch.zeros(8, dtype=torch.int32), table_idx=0, cached_len=0,
                output_len=0, uid=1, sampling_params=SamplingParams(), cache_handle=None)
        r.linear_slot_idx = slot
        b = Batch(reqs=[r], phase=phase)
        b.fla_metadata = _fla(slot, dev)
        return b

    # Path A: K sequential decode forwards
    ctx = _ctx_with(pool, None)
    outs_a, snaps_a = [], []
    with ctx.forward_batch(mkbatch("decode")):
        for j in range(K):
            if j:
                pool.restore_slot(slot, snaps_a[-1])
            outs_a.append(layer.forward(h[j : j + 1].clone()))
            snaps_a.append(pool.snapshot_slot(slot))

    # Path B: one verify forward from the same initial state
    pool.restore_slot(slot, init)
    state_log = [(torch.zeros_like(pool.conv_states[:, slot]),
                  torch.zeros_like(pool.recurrent_states[:, slot])) for _ in range(K)]
    vb = mkbatch("prefill")
    vb.is_verify = True
    ctx2 = _ctx_with(pool, None)
    with ctx2.forward_batch(vb):
        ctx2.spec_state_log = state_log
        out_b = layer.forward(h.clone())
    final_b = pool.snapshot_slot(slot)

    do, da = out_b.float(), torch.cat(outs_a).float()
    assert do.shape == da.shape
    md = (do - da).abs().max().item()
    assert md < 0.05, f"output mismatch {md}"
    fd = (final_b[1] - snaps_a[-1][1]).abs().max().item()
    assert fd < 1e-4, f"final recurrent state mismatch {fd}"
    for j in range(K):
        sj = (state_log[j][1] - snaps_a[j][1]).abs().max().item()
        assert sj < 1e-4, f"position {j} snapshot mismatch {sj}"
