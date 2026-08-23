"""Qwen3.5/3.6 MoE GGUF adapter: build the FreeToken ``ModelConfig`` and stream
weights from a llama.cpp ``qwen35moe`` GGUF checkpoint.

The GGUF checkpoint carries the same hybrid architecture as the HF safetensors
release — Gated-DeltaNet (linear) layers interleaved with gated full-attention
layers, 256 routed experts + a gated shared expert, one MTP (nextn) layer — so
this produces the *same* ``ModelConfig`` as ``qwen3_5_moe.config.parse_config``,
only sourced from GGUF KV metadata. Dense weights (attention, GDN, shared
expert, router, embedding, lm_head) dequantize to bf16 at load; the routed
experts stay in their native Q4_K/Q6_K block layout and stream to the ggml MoE
kernels through the ``ggml`` offload-cache format (per-layer ggml type).

Tested against Ornith-1.5-35B-A3B (Q4_K_M): head_count 16 x 256 qk-dim with a
2x fused q|gate half, 2 kv heads, GDN 16x128 key / 32x128 value heads, conv 4,
partial rope 64/256 @ 1e7, 256 experts of 512, shared expert 512.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import (
    GGML_F32,
    GGML_Q4_K,
    GGML_Q6_K,
    dequantize,
    row_bytes,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str, default=None):
        val = m.get(f"qwen35moe.{key}", default)
        if val is None and default is None:
            raise KeyError(f"missing GGUF metadata key qwen35moe.{key}")
        return val

    block_count = int(g("block_count"))
    nextn = int(m.get("qwen35moe.nextn_predict_layers", 0) or 0)
    num_layers = block_count - (1 if nextn else 0)  # the MTP layer is dropped
    interval = int(g("full_attention_interval", 4))
    layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    head_dim = int(g("attention.key_length"))
    assert int(g("attention.value_length")) == head_dim, "qk/v head dims diverge"
    num_qo = int(g("attention.head_count"))
    num_kv = int(g("attention.head_count_kv"))
    rotary_dim = int(m.get("qwen35moe.rope.dimension_count", head_dim))
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(g("context_length")),
        base=float(g("rope.freq_base")),
        scaling=None,
    )

    state = int(g("ssm.state_size"))  # == key/value head dim
    inner = int(g("ssm.inner_size"))  # == value_dim total
    num_k_heads = int(g("ssm.group_count"))
    num_v_heads = inner // state

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=int(g("embedding_length")),
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        norm_topk_prob=True,
        model_type="qwen3_5_moe",
        architectures=list(shim.architectures),
        moe_enabled=True,
        expert_quant="ggml",
        moe_weight_format="ggml",
        use_qk_norm=True,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_ids,
                num_kv_heads=num_kv,
                head_dim=head_dim,
                rotary_config=rotary,
            ),
            LinearGatedDeltaGroupConfig(
                name="linear",
                layer_ids=linear_ids,
                num_key_heads=num_k_heads,
                num_value_heads=num_v_heads,
                key_head_dim=state,
                value_head_dim=state,
                conv_kernel_dim=int(g("ssm.conv_kernel")),
                output_gate=True,
            ),
        ),
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken qwen3_5_moe module params.
# --------------------------------------------------------------------------------------

def _vhead_src_index(nv: int = 32) -> list[int]:
    """GGUF (llama.cpp qwen35moe layout) stores the GDN value heads de-interleaved:
    blocks [even heads | odd heads] (verified against the HF checkpoint: per-head
    cosine 0.98-0.999 with GGUF block i == HF head 2i for i < nv/2). FreeToken (and
    HF) use plain head order; gather source rows HF[j] <- GGUF[j//2] (even j) /
    GGUF[nv//2 + j//2] (odd j). Q/K projections are NOT permuted."""
    return [j // 2 if j % 2 == 0 else nv // 2 + j // 2 for j in range(nv)]


def _reinterleave_vheads_rows(t: torch.Tensor, nv: int, vd: int, v_offset: int = 0) -> torch.Tensor:
    """Regroup a row-major [.., nv*vd, ..] tensor's v-head blocks from the GGUF's
    [even | odd] layout into plain head order. ``v_offset`` skips leading non-v rows
    (the q|k parts of the fused qkv projection)."""
    idx = _vhead_src_index(nv)
    head = t.shape[0]
    total_v = nv * vd
    assert (head - v_offset) == total_v, (head, v_offset, total_v)
    v = t[v_offset:].reshape(nv, vd, *t.shape[1:])
    gathered = v[torch.tensor(idx)]
    if v_offset:
        return torch.cat([t[:v_offset], gathered.reshape(total_v, *t.shape[1:])], dim=0)
    return gathered.reshape(total_v, *t.shape[1:])


def _reinterleave_vheads_cols(t: torch.Tensor, nv: int, vd: int) -> torch.Tensor:
    """Column (input-side) variant for out_proj: [out, nv*vd] with v-head columns
    in the GGUF's de-interleaved order."""
    idx = torch.tensor(_vhead_src_index(nv))
    cols = t.shape[1]
    assert cols == nv * vd, (cols, nv * vd)
    c = t.reshape(t.shape[0], nv, vd)
    return c[:, idx].reshape(t.shape)


def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q4_K/Q6_K) to a dense bf16 tensor of its
    torch shape (``dims[::-1]`` of the ggml layout)."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _vhead_perm_tensor(nv: int) -> torch.Tensor:
    return torch.tensor(_vhead_src_index(nv))


def _norm_plus_one(t) -> torch.Tensor:
    """Gemma-style (1+weight) norms; the llama.cpp converter pre-bakes the +1
    for qwen35moe, so the GGUF value is (1+w) -- pass through verbatim."""
    return _to_bf16(t)


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert qwen3_5_moe param.

    Dense projections dequantize to bf16 (attention 2x-q|gate half, GDN in_proj
    [qkv|z|b|a] fusion, shared expert gate_up fusion); norms bake the Gemma +1
    (except ``linear_attn.norm``, a standard weight*x norm). The MTP (nextn)
    layer and its tensors are dropped (served text-only, like the HF loader).
    Routed experts stream from the offload cache (``load_ggml_expert_sources``).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    assert not include_moe_experts, (
        "qwen35moe GGUF stores experts natively quantized (Q4_K/Q6_K) and only "
        "supports the offload backend; experts load via load_ggml_expert_sources()."
    )
    assert include_non_moe

    config = parse_gguf_config(cached_load_hf_config(model_path))
    linear_ids = set(config.linear_attention_group().layer_ids)
    num_layers = config.num_layers
    lg = config.linear_attention_group()
    cfg_linear_dims = {
        lid: {"nv": lg.num_value_heads, "vd": lg.value_head_dim, "k_dim": lg.num_key_heads * lg.key_head_dim}
        for lid in linear_ids
    }

    # Per-layer fusion buffers: GGUF parts arrive in tensor-table order; fuse as
    # soon as every part of a merged projection is present.
    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    in_proj_buf: dict[int, dict[str, torch.Tensor]] = {}
    shexp_buf: dict[int, dict[str, torch.Tensor]] = {}

    def flush_qkv(layer: int):
        parts = qkv_buf.pop(layer)
        fused = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
        yield f"model.layers.{layer}.self_attn.qkv_proj.weight", fused

    def flush_in_proj(layer: int):
        parts = in_proj_buf.pop(layer)
        # [conv_dim(q|k|v) | z | b | a] -- the GDN module's merged split order.
        fused = torch.cat(
            [parts["qkv"], parts["z"], parts["b"], parts["a"]], dim=0
        )
        yield f"model.layers.{layer}.linear_attn.in_proj.weight", fused

    def flush_shexp(layer: int):
        parts = shexp_buf.pop(layer)
        fused = torch.cat([parts["gate"], parts["up"]], dim=0)
        yield f"model.layers.{layer}.mlp.shared_expert.gate_up_proj.weight", fused

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if ".nextn." in name:
            continue  # MTP layer extras: served text-only, dropped
        if not name.startswith("blk."):
            if name == "token_embd.weight":
                yield "model.embed_tokens.weight", _to_bf16(t)
            elif name == "output_norm.weight":
                yield "model.norm.weight", _norm_plus_one(t)
            elif name == "output.weight":
                yield "lm_head.weight", _to_bf16(t)
            continue

        layer = int(name.split(".")[1])
        if layer >= num_layers:
            continue  # the trailing MTP block (block_count includes it)
        suffix = name.split(".", 2)[2]
        p = f"model.layers.{layer}"

        if suffix == "attn_norm.weight":
            yield f"{p}.input_layernorm.weight", _norm_plus_one(t)
        elif suffix == "post_attention_norm.weight":
            yield f"{p}.post_attention_layernorm.weight", _norm_plus_one(t)
        elif suffix == "attn_q_norm.weight":
            yield f"{p}.self_attn.q_norm.weight", _norm_plus_one(t)
        elif suffix == "attn_k_norm.weight":
            yield f"{p}.self_attn.k_norm.weight", _norm_plus_one(t)
        elif suffix == "attn_output.weight":
            yield f"{p}.self_attn.o_proj.weight", _to_bf16(t)
        elif suffix in ("ffn_gate_inp.weight",):
            yield f"{p}.mlp.gate.weight", _to_bf16(t)
        elif suffix in ("ffn_gate_inp_shexp.weight",):
            yield f"{p}.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
        elif suffix == "ffn_down_shexp.weight":
            yield f"{p}.mlp.shared_expert.down_proj.weight", _to_bf16(t)
        elif layer in linear_ids:
            # GDN layer tensors. The GGUF's value-head blocks are de-interleaved
            # ([even | odd]); every v-head-dim tensor is regrouped to plain order
            # (see _vhead_src_index). Q/K rows are not permuted.
            cfg_l = cfg_linear_dims[layer]
            if suffix == "attn_qkv.weight":
                w = _to_bf16(t)
                w = _reinterleave_vheads_rows(w, cfg_l["nv"], cfg_l["vd"], v_offset=2 * cfg_l["k_dim"])
                in_proj_buf.setdefault(layer, {})["qkv"] = w
            elif suffix == "attn_gate.weight":
                in_proj_buf.setdefault(layer, {})["z"] = _reinterleave_vheads_rows(_to_bf16(t), cfg_l["nv"], cfg_l["vd"])
            elif suffix == "ssm_beta.weight":
                in_proj_buf.setdefault(layer, {})["b"] = _to_bf16(t)[_vhead_perm_tensor(cfg_l["nv"])]
            elif suffix == "ssm_alpha.weight":
                in_proj_buf.setdefault(layer, {})["a"] = _to_bf16(t)[_vhead_perm_tensor(cfg_l["nv"])]
            elif suffix == "ssm_a":
                # llama.cpp stores A pre-negated (-exp(A_log)) in its head order;
                # FreeToken keeps A_log and negates at use.
                a_neg = t.packed().view(torch.float32).reshape(-1)
                a_neg = a_neg[_vhead_perm_tensor(cfg_l["nv"])]
                yield f"{p}.linear_attn.A_log", torch.log(a_neg.clamp(max=-1e-8).neg())
            elif suffix == "ssm_dt.bias":
                db = t.packed().view(torch.float32).reshape(-1)[_vhead_perm_tensor(cfg_l["nv"])]
                yield f"{p}.linear_attn.dt_bias", db
            elif suffix == "ssm_conv1d.weight":
                w = _to_bf16(t)
                w = _reinterleave_vheads_rows(w, cfg_l["nv"], cfg_l["vd"], v_offset=2 * cfg_l["k_dim"])
                yield f"{p}.linear_attn.conv1d.weight", w.reshape(w.shape[0], 1, -1)
            elif suffix == "ssm_norm.weight":
                yield f"{p}.linear_attn.norm.weight", _to_bf16(t)
            elif suffix == "ssm_out.weight":
                w = _to_bf16(t)
                yield f"{p}.linear_attn.out_proj.weight", _reinterleave_vheads_cols(w, cfg_l["nv"], cfg_l["vd"])
            # ffn_*_exps / ffn_*_shexp handled below (shared with attention layers)
        if suffix == "attn_q.weight":
            qkv_buf.setdefault(layer, {})["q"] = _to_bf16(t)
        elif suffix == "attn_k.weight":
            qkv_buf.setdefault(layer, {})["k"] = _to_bf16(t)
        elif suffix == "attn_v.weight":
            qkv_buf.setdefault(layer, {})["v"] = _to_bf16(t)
        elif suffix == "ffn_gate_shexp.weight":
            shexp_buf.setdefault(layer, {})["gate"] = _to_bf16(t)
        elif suffix == "ffn_up_shexp.weight":
            shexp_buf.setdefault(layer, {})["up"] = _to_bf16(t)
        # ffn_*_exps.weight -> offload banks (never here)

        if layer in qkv_buf and len(qkv_buf[layer]) == 3:
            yield from flush_qkv(layer)
        if layer in in_proj_buf and len(in_proj_buf[layer]) == 4:
            yield from flush_in_proj(layer)
        if layer in shexp_buf and len(shexp_buf[layer]) == 2:
            yield from flush_shexp(layer)

    assert not qkv_buf, f"unfused qkv parts: { {k: sorted(v) for k, v in qkv_buf.items()} }"
    assert not in_proj_buf, f"unfused in_proj parts: { {k: sorted(v) for k, v in in_proj_buf.items()} }"
    assert not shexp_buf, f"unfused shared-expert parts: { {k: sorted(v) for k, v in shexp_buf.items()} }"


# --------------------------------------------------------------------------------------
# Routed expert banks: native Q4_K/Q6_K blocks, streamed to the ggml MoE kernels.
# --------------------------------------------------------------------------------------

def load_ggml_expert_sources(model_path: str, config: ModelConfig, *, layer_sink=None) -> dict:
    """Per-layer host banks of the routed experts' native GGML block bytes.

    ``gate_up`` is one ``[E, 2I, row_bytes(H)]`` tensor per layer (gate rows over
    up rows, fused by packed-row concatenation -- each row's blocks are
    independent, so the concat is layout-exact) and ``down`` one
    ``[E, H, row_bytes(I)]`` per layer, all in ONE uniform ggml type: mixed-quant
    checkpoints (Q4_K_M mixes Q6_K into some tensors) are requantized to the
    gate/up type here, because the offload slot-cache pool requires uniform
    expert row bytes. ``quant_types`` carries the uniform (gate_up, down) type
    pair for the compute path.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size

    types_gate, types_up, types_dn = {}, {}, {}
    for t in iter_gguf_tensors(model_path):
        if not t.name.startswith("blk.") or ".nextn." in t.name:
            continue
        layer = int(t.name.split(".")[1])
        if layer >= L:
            continue
        if t.name.endswith("ffn_gate_exps.weight"):
            types_gate[layer] = t.ggml_type
        elif t.name.endswith("ffn_up_exps.weight"):
            types_up[layer] = t.ggml_type
        elif t.name.endswith("ffn_down_exps.weight"):
            types_dn[layer] = t.ggml_type
    want = set(range(L))
    assert set(types_gate) == want and set(types_up) == want and set(types_dn) == want, (
        "checkpoint is missing routed-expert tensors"
    )
    assert types_gate == types_up, "gate/up expert tensors have different ggml types"
    qt = types_gate[0]
    assert all(v == qt for v in list(types_gate.values()) + list(types_up.values())), (
        "gate/up expert ggml types vary across layers"
    )
    assert qt == GGML_Q4_K, f"uniform requant target implemented for Q4_K, got {qt}"
    mixed_down = {l for l, v in types_dn.items() if v != qt}

    specs = {
        "gate_up": ((E, 2 * I, row_bytes(H, qt)), torch.uint8),
        "down": ((E, H, row_bytes(I, qt)), torch.uint8),
    }
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen_gu, seen_dn = set(), set()

    def _packed(t, rows, cols, t_qt):
        return t.packed().reshape(rows, row_bytes(cols, t_qt))

    def _to_uniform(t, out_rows, in_cols, from_qt):
        """Packed rows in the uniform qt; requantizes when the tensor's native
        type differs (mixed-quant checkpoints)."""
        packed = _packed(t, out_rows, in_cols, from_qt)
        if from_qt == qt:
            return packed
        from freetoken.models.gguf.dequant import dequantize, requantize_q4_k
        dense = dequantize(packed.reshape(-1), from_qt, torch.float32)
        return requantize_q4_k(dense).reshape(out_rows, row_bytes(in_cols, qt))

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None
        gate_parts: dict[int, torch.Tensor] = {}
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk.") or ".nextn." in t.name:
                continue
            layer = int(t.name.split(".")[1])
            if layer >= L:
                continue
            if t.name.endswith("ffn_gate_exps.weight"):
                gate_parts[layer] = _to_uniform(t, E * I, H, types_gate[layer]).reshape(E, I, -1)
            elif t.name.endswith("ffn_up_exps.weight"):
                up = _to_uniform(t, E * I, H, types_up[layer]).reshape(E, I, -1)
                gu = torch.cat([gate_parts.pop(layer), up], dim=1)  # [E, 2I, bytes]
                banks["gate_up"][layer].copy_(gu)
                seen_gu.add(layer)
            elif t.name.endswith("ffn_down_exps.weight"):
                dn = _to_uniform(t, E * H, I, types_dn[layer]).reshape(E, H, -1)
                banks["down"][layer].copy_(dn)
                seen_dn.add(layer)
            else:
                continue
            if tracker is not None:
                tracker.note(layer)
        assert not gate_parts, f"unpaired gate/up expert tensors: {sorted(gate_parts)}"

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    assert seen_gu == want and seen_dn == want, (
        f"missing expert tensors: gate_up={sorted(want - seen_gu)} down={sorted(want - seen_dn)}"
    )
    return {
        "gate_up": banks["gate_up"],
        "down": banks["down"],
        "quant_types": [(qt, qt)] * L,
    }


def dummy_ggml_expert_sources(config: ModelConfig) -> dict:
    """Fabricate zeroed banks (install smoke tests / dry runs)."""
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    gu_rows, dn_rows = row_bytes(H, GGML_Q4_K), row_bytes(I, GGML_Q4_K)
    return {
        "gate_up": [torch.zeros(E, 2 * I, gu_rows, dtype=torch.uint8) for _ in range(L)],
        "down": [torch.zeros(E, H, dn_rows, dtype=torch.uint8) for _ in range(L)],
        "quant_types": [(GGML_Q4_K, GGML_Q4_K)] * L,
    }


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "load_ggml_expert_sources",
    "dummy_ggml_expert_sources",
]
