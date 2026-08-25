"""GGUF (native k-quant) adapter for the DENSE qwen35 family (Qwen3.8-27B
class: hybrid GDN + every-4th full attention, dense SwiGLU FFN, no experts).

Reuses the qwen3_5_moe model classes end to end -- the decoder layer already
branches to ``Qwen3_5DenseMLP`` when ``moe_enabled=False`` and the GDN /
full-attention modules are dimension-parameterized. This module contributes
the config parse (``qwen35.*`` metadata keys) and the weight-name mapping.

Verified layout facts (GGUF header + Rusty Llama qwen35.cpp +
convert_hf_to_gguf.py's shared ``_LinearAttentionVReorderBase``):

- full attention at layers ``(i+1) % interval == 0``: q carries the per-head
  q|gate interleave (rows = nq*hd*2), k/v are GQA, partial rope 64-of-256
  base 1e7, q/k norms.
- GDN: ``attn_qkv`` rows = [q 2*key_dim | k ... | v value_dim] with layout
  [q|k|v] = key_dim*2 + value_dim (16 k-heads, 48 v-heads, 128 dims);
  ``attn_gate`` = the output gate z (value_dim rows); ``ssm_beta``/``ssm_alpha``
  = b|a per v-head; ``ssm_a`` = -exp(A_log) pre-negated; ``ssm_dt.bias``;
  ``ssm_norm`` = plain (no +1) head norm; ``ssm_conv1d`` [kernel, conv_dim];
  ``ssm_out`` [value_dim, n_embd].
- The converter de-interleaves v heads ([even | odd] blocks) exactly like
  qwen35moe, so every v-head-dim tensor is regrouped to plain order here
  (row permutations apply identically to packed bytes).
- The trailing ``nextn`` block (block_count includes it) is dropped --
  the dense family serves text-only here; speculation is DFlash2's job.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

# Layout/regroup helpers shared with the (battle-tested) qwen35moe loader.
from freetoken.models.qwen3_5_moe.gguf import (
    _gguf_dense_quant_flag,
    _norm_plus_one,
    _to_bf16,
)


def _qt_scalar(qt: int):
    return torch.tensor(qt, dtype=torch.int32)


def _v_src_index(nv: int, nk: int) -> list[int]:
    """Inverse of the converter's grouped->tiled v-head reorder
    (_LinearAttentionVReorderBase): HF head j (group g=j//r, sub s=j%r,
    r = nv//nk) lives at GGUF position s*nk + g. For r==2 this reduces to
    the [even | odd] split the qwen35moe loader hardcodes; Qwen3.8-27B has
    r==3 (48 v-heads over 16 k-heads), where that pair split is WRONG."""
    r = nv // nk
    return [(j % r) * nk + j // r for j in range(nv)]


def _v_perm(nv: int, nk: int) -> torch.Tensor:
    return torch.tensor(_v_src_index(nv, nk))


def _regroup_v_rows(t: torch.Tensor, nv: int, nk: int, vd: int, v_offset: int = 0):
    """Regroup v-head ROW blocks from the GGUF's tiled order to plain HF
    order. Row permutations, so they apply identically to packed bytes."""
    idx = _v_perm(nv, nk)
    total_v = nv * vd
    assert (t.shape[0] - v_offset) == total_v, (t.shape, v_offset, total_v)
    v = t[v_offset:].reshape(nv, vd, *t.shape[1:])
    gathered = v[idx]
    if v_offset:
        return torch.cat([t[:v_offset], gathered.reshape(total_v, *t.shape[1:])], dim=0)
    return gathered.reshape(total_v, *t.shape[1:])


def _regroup_v_cols(t: torch.Tensor, nv: int, nk: int, vd: int):
    """Column (input-side) variant for out_proj."""
    idx = _v_perm(nv, nk)
    c = t.reshape(t.shape[0], nv, vd)
    return c[:, idx].reshape(t.shape)


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str, default=None):
        val = m.get(f"qwen35.{key}", default)
        if val is None and default is None:
            raise KeyError(f"missing GGUF metadata key qwen35.{key}")
        return val

    block_count = int(g("block_count"))
    nextn = int(m.get("qwen35.nextn_predict_layers", 0) or 0)
    num_layers = block_count - (1 if nextn else 0)  # drop the trailing MTP block
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
    rotary_dim = int(m.get("qwen35.rope.dimension_count", head_dim))
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
        intermediate_size=int(g("feed_forward_length")),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        norm_topk_prob=True,
        model_type="qwen35_dense",
        architectures=list(shim.architectures),
        moe_enabled=False,
        expert_quant="none",
        use_qk_norm=True,
        dense_quant=_gguf_dense_quant_flag(),
        lm_head_quant=_gguf_dense_quant_flag(),
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


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for the dense qwen35 params: packed k-quant
    bytes for every projection (per-type modules for the fused pieces), the
    (1+w) norm convention, and the v-head regrouping to plain order."""
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    # include_moe_experts is True on the resident path; a dense model simply
    # has no expert tensors to yield either way.
    assert include_non_moe

    config = parse_gguf_config(cached_load_hf_config(model_path))
    dense_q = config.dense_quant == "ggml_kquant"
    num_layers = config.num_layers
    linear_ids = set(config.linear_attention_group().layer_ids)
    lg = config.linear_attention_group()
    nv, vd = lg.num_value_heads, lg.value_head_dim
    nk = lg.num_key_heads
    k_dim = nk * lg.key_head_dim

    # Fusion buffers: parts arrive in tensor-table order; flush when complete.
    qkv_buf: dict[int, dict[str, tuple]] = {}
    in_proj_buf: dict[int, dict[str, tuple]] = {}
    ffn_buf: dict[int, dict[str, tuple]] = {}

    def flush_qkv(layer: int):
        parts = qkv_buf.pop(layer)
        p = f"model.layers.{layer}.self_attn"
        if dense_q:
            for part, mod in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
                pk, qt = parts[part]
                yield f"{p}.{mod}.packed", pk
                yield f"{p}.{mod}.quant_type", _qt_scalar(qt)
        else:
            fused = torch.cat([parts["q"][0], parts["k"][0], parts["v"][0]], dim=0)
            yield f"{p}.qkv_proj.weight", fused

    def flush_in_proj(layer: int):
        parts = in_proj_buf.pop(layer)
        p = f"model.layers.{layer}.linear_attn"
        if dense_q:
            qkv, z, b, a = parts["qkv"], parts["z"], parts["b"], parts["a"]
            yield f"{p}.in_proj_qkv.packed", qkv[0]
            yield f"{p}.in_proj_qkv.quant_type", _qt_scalar(qkv[1])
            yield f"{p}.in_proj_z.packed", z[0]
            yield f"{p}.in_proj_z.quant_type", _qt_scalar(z[1])
            assert b[1] == a[1], "mixed k-quant types in in_proj_ba"
            yield f"{p}.in_proj_ba.packed", torch.cat([b[0], a[0]], dim=0)
            yield f"{p}.in_proj_ba.quant_type", _qt_scalar(b[1])
        else:
            fused = torch.cat(
                [parts["qkv"][0], parts["z"][0], parts["b"][0], parts["a"][0]], dim=0
            )
            yield f"{p}.in_proj.weight", fused

    def flush_ffn(layer: int):
        parts = ffn_buf.pop(layer)
        p = f"model.layers.{layer}.mlp"
        g, u, d = parts["gate"], parts["up"], parts["down"]
        if dense_q:
            assert g[1] == u[1], "mixed k-quant types in gate_up"
            yield f"{p}.gate_up_proj.packed", torch.cat([g[0], u[0]], dim=0)
            yield f"{p}.gate_up_proj.quant_type", _qt_scalar(g[1])
            yield f"{p}.down_proj.packed", d[0]
            yield f"{p}.down_proj.quant_type", _qt_scalar(d[1])
        else:
            yield f"{p}.gate_up_proj.weight", torch.cat([g[0], u[0]], dim=0)
            yield f"{p}.down_proj.weight", d[0]

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if not name.startswith("blk."):
            if name == "token_embd.weight":
                if dense_q:
                    pk = t.packed()
                    qt = _qt_scalar(t.ggml_type)
                    yield "model.embed_tokens.packed", pk
                    yield "model.embed_tokens.quant_type", qt
                    if config.tie_word_embeddings:
                        yield "lm_head.packed", pk
                        yield "lm_head.quant_type", qt
                else:
                    yield "model.embed_tokens.weight", _to_bf16(t)
            elif name == "output_norm.weight":
                yield "model.norm.weight", _norm_plus_one(t)
            elif name == "output.weight":
                if dense_q:
                    yield "lm_head.packed", t.packed()
                    yield "lm_head.quant_type", _qt_scalar(t.ggml_type)
                else:
                    yield "lm_head.weight", _to_bf16(t)
            continue

        layer = int(name.split(".")[1])
        if layer >= num_layers:
            continue  # trailing nextn block: text-only serving
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
            if dense_q:
                yield f"{p}.self_attn.o_proj.packed", t.packed()
                yield f"{p}.self_attn.o_proj.quant_type", _qt_scalar(t.ggml_type)
            else:
                yield f"{p}.self_attn.o_proj.weight", _to_bf16(t)
        elif layer in linear_ids:
            # GDN layer: v-head blocks are de-interleaved in the GGUF
            # ([even | odd]); regroup every v-head-dim tensor to plain order
            # (row permutations apply identically to packed bytes).
            if suffix == "attn_qkv.weight":
                if dense_q:
                    w = _regroup_v_rows(t.packed(), nv, nk, vd, v_offset=2 * k_dim)
                else:
                    w = _regroup_v_rows(_to_bf16(t), nv, nk, vd, v_offset=2 * k_dim)
                in_proj_buf.setdefault(layer, {})["qkv"] = (w, t.ggml_type)
            elif suffix == "attn_gate.weight":
                if dense_q:
                    w = _regroup_v_rows(t.packed(), nv, nk, vd)
                else:
                    w = _regroup_v_rows(_to_bf16(t), nv, nk, vd)
                in_proj_buf.setdefault(layer, {})["z"] = (w, t.ggml_type)
            elif suffix == "ssm_beta.weight":
                perm = _v_perm(nv, nk)
                in_proj_buf.setdefault(layer, {})["b"] = (
                    (t.packed()[perm], t.ggml_type) if dense_q else _to_bf16(t)[perm]
                )
            elif suffix == "ssm_alpha.weight":
                perm = _v_perm(nv, nk)
                in_proj_buf.setdefault(layer, {})["a"] = (
                    (t.packed()[perm], t.ggml_type) if dense_q else _to_bf16(t)[perm]
                )
            elif suffix == "ssm_a":
                # llama.cpp stores A pre-negated (-exp(A_log)); FreeToken
                # keeps A_log and negates at use.
                a_neg = t.packed().view(torch.float32).reshape(-1)
                a_neg = a_neg[_v_perm(nv, nk)]
                yield f"{p}.linear_attn.A_log", torch.log(a_neg.clamp(max=-1e-8).neg())
            elif suffix == "ssm_dt.bias":
                db = t.packed().view(torch.float32).reshape(-1)[_v_perm(nv, nk)]
                yield f"{p}.linear_attn.dt_bias", db
            elif suffix == "ssm_conv1d.weight":
                w = _to_bf16(t)
                w = _regroup_v_rows(w, nv, nk, vd, v_offset=2 * k_dim)
                yield f"{p}.linear_attn.conv1d.weight", w.reshape(w.shape[0], 1, -1)
            elif suffix == "ssm_norm.weight":
                yield f"{p}.linear_attn.norm.weight", _to_bf16(t)
            elif suffix == "ssm_out.weight":
                if dense_q:
                    # Requantize the column-regrouped plain-order matrix to
                    # Q4_K once at load (packed columns cannot be permuted):
                    # same recipe as the qwen35moe GDN out_proj.
                    from freetoken.models.gguf.dequant import requantize_q4_k

                    w = _regroup_v_cols(_to_bf16(t), nv, nk, vd)
                    rows_n = w.shape[0]
                    pk = (
                        requantize_q4_k(w.reshape(-1).float())
                        .reshape(rows_n, -1)
                        .to(torch.uint8)
                    )
                    yield f"{p}.linear_attn.out_proj.packed", pk
                    yield f"{p}.linear_attn.out_proj.quant_type", _qt_scalar(12)
                else:
                    w = _to_bf16(t)
                    yield f"{p}.linear_attn.out_proj.weight", _regroup_v_cols(w, nv, nk, vd)
            elif suffix == "ffn_gate.weight":
                ffn_buf.setdefault(layer, {})["gate"] = (
                    (t.packed(), t.ggml_type) if dense_q else _to_bf16(t)
                )
            elif suffix == "ffn_up.weight":
                ffn_buf.setdefault(layer, {})["up"] = (
                    (t.packed(), t.ggml_type) if dense_q else _to_bf16(t)
                )
            elif suffix == "ffn_down.weight":
                ffn_buf.setdefault(layer, {})["down"] = (
                    (t.packed(), t.ggml_type) if dense_q else _to_bf16(t)
                )
            if layer in in_proj_buf and len(in_proj_buf[layer]) == 4:
                yield from flush_in_proj(layer)
            if layer in ffn_buf and len(ffn_buf[layer]) == 3:
                yield from flush_ffn(layer)
        else:
            # Full-attention layer
            if suffix == "attn_q.weight":
                qkv_buf.setdefault(layer, {})["q"] = (
                    (t.packed(), t.ggml_type) if dense_q else _to_bf16(t)
                )
            elif suffix == "attn_k.weight":
                qkv_buf.setdefault(layer, {})["k"] = (
                    (t.packed(), t.ggml_type) if dense_q else _to_bf16(t)
                )
            elif suffix == "attn_v.weight":
                qkv_buf.setdefault(layer, {})["v"] = (
                    (t.packed(), t.ggml_type) if dense_q else _to_bf16(t)
                )
            elif suffix == "ffn_gate.weight":
                ffn_buf.setdefault(layer, {})["gate"] = (
                    (t.packed(), t.ggml_type) if dense_q else _to_bf16(t)
                )
            elif suffix == "ffn_up.weight":
                ffn_buf.setdefault(layer, {})["up"] = (
                    (t.packed(), t.ggml_type) if dense_q else _to_bf16(t)
                )
            elif suffix == "ffn_down.weight":
                ffn_buf.setdefault(layer, {})["down"] = (
                    (t.packed(), t.ggml_type) if dense_q else _to_bf16(t)
                )
            if layer in qkv_buf and len(qkv_buf[layer]) == 3:
                yield from flush_qkv(layer)
            if layer in ffn_buf and len(ffn_buf[layer]) == 3:
                yield from flush_ffn(layer)

    assert not qkv_buf, f"unpaired q/k/v tensors: {sorted(qkv_buf)}"
    assert not in_proj_buf, f"unpaired in_proj tensors: {sorted(in_proj_buf)}"
    assert not ffn_buf, f"unpaired ffn tensors: {sorted(ffn_buf)}"


__all__ = ["parse_gguf_config", "iter_gguf_weights"]
