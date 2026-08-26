"""DFlash2 draft GGUF loading: config decode + bf16 materialization.

The draft is a standalone 5-layer non-causal encoder (no causal-LM surface),
so it does NOT register in the engine's GGUF_ARCH_TO_REGISTRY — it is loaded
directly by :func:`load_dflash2_gguf` into a reference-compatible state_dict
(z-lab/dflash ``DFlash2DraftModel`` naming) for parity testing and later
engine wiring. All projections dequantize to bf16 at load; the draft engine
is bf16-resident, so no ggml-ext quant kernels are on this path at all.

Orientation facts (from the reader contract, verified against Ridge):
``GgufTensor.shape`` is torch order ``[rows=out, in]`` with rows spanning
whole blocks, so plain dequant + reshape yields Linear weights directly;
the codebooks arrive as raw ``(rank, vocab)`` ne-tuples whose reversed
torch shape is exactly Embedding weight ``[vocab, rank]``.
"""
from __future__ import annotations

import torch

# ggml types present in the incoai DFlash2-Q8_0 checkpoint.
_F32 = 0


def parse_dflash_config(metadata: dict) -> dict:
    """Decode the ``dflash.*`` KV block into a plain config dict."""
    from freetoken.models.gguf.reader import load_gguf_metadata

    md = metadata or load_gguf_metadata  # never autoloader; caller passes

    def g(key, default=None):
        val = metadata.get(f"dflash.{key}", default)
        if val is None and default is None:
            raise KeyError(f"missing GGUF metadata key dflash.{key}")
        return val

    return {
        "num_layers": int(g("block_count")),
        "hidden_size": int(g("embedding_length")),
        "intermediate_size": int(g("feed_forward_length")),
        "num_attention_heads": int(g("attention.head_count")),
        "num_key_value_heads": int(g("attention.head_count_kv")),
        "head_dim": int(g("attention.key_length")),
        "causal": bool(g("attention.causal", False)),
        "sliding_window": int(g("attention.sliding_window", 0)) or None,
        "sliding_window_pattern": list(g("attention.sliding_window_pattern", [])),
        "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon")),
        "rope_base": float(g("rope.freq_base")),
        "block_size": int(g("block_size")),
        "conv_kernel_size": int(g("conv_kernel_size")),
        "conv_group_size": int(g("conv_group_size")),
        "selector_rank": int(g("selector_rank")),
        "selector_top_k": int(g("selector_top_k")),
        # Hidden-state depths consumed from the TARGET trunk. The HF draft
        # indexes hidden_states[layer+1]; the GGUF stores the llama.cpp-side
        # (0-based decoder-layer) ids verbatim.
        "target_layers": [int(v) for v in g("target_layers")],
    }


def iter_dflash_weights(model_path: str, device="cpu"):
    """Yield ``(reference_param_name, bf16_tensor)`` for the whole draft.

    Names mirror z-lab ``DFlash2DraftModel`` so parity harnesses can diff
    against an HF dump without remapping.
    """
    from freetoken.models.gguf.dequant import dequantize
    from freetoken.models.gguf.reader import iter_gguf_tensors

    norms = {}  # two global norm roles resolved by caller (ambiguity note)
    for t in iter_gguf_tensors(model_path):
        name = t.name
        qt = t.ggml_type
        if qt == _F32:
            w = t.packed().view(torch.float32).reshape(t.shape).to(device)
        else:
            flat = dequantize(t.packed().reshape(-1), qt, torch.bfloat16)
            w = flat.reshape(t.shape).to(device)

        if name == "fc.weight":
            yield "fc.weight", w
        elif name == "output_norm.weight" or name == "enc.output_norm.weight":
            norms[name] = w
        elif name == "selector_hidden.weight":
            yield "candidate_selector.hidden_projection.weight", w
        elif name == "selector_predecessor.weight":
            yield "candidate_selector.predecessor_codebook.weight", w
        elif name == "selector_successor.weight":
            yield "candidate_selector.successor_codebook.weight", w
        else:
            parts = name.split(".", 2)  # blk.<N>.<rest...>
            layer = int(parts[1])
            rest = parts[2]
            out = _map_block(layer, rest, w)
            if out is not None:
                yield out

    # Ambiguity 1 RESOLVED empirically (tools/mtp/dflash_parity.py: chained
    # greedy proposals match the z-lab reference ONLY under this assignment):
    # `output_norm` = fc/context-side hidden_norm, `enc.output_norm` = final
    # pre-selector norm. (llama.cpp's usual convention is inverted here.)
    if "output_norm.weight" in norms:
        yield "hidden_norm.weight", norms["output_norm.weight"]
    if "enc.output_norm.weight" in norms:
        yield "norm.weight", norms["enc.output_norm.weight"]


def _map_block(layer: int, rest: str, w: torch.Tensor):
    p = f"layers.{layer}"
    if rest == "attn_norm.weight":
        return f"{p}.input_layernorm.weight", w
    if rest == "ffn_norm.weight":
        return f"{p}.post_attention_layernorm.weight", w
    if rest.startswith("attn_q_norm"):
        return f"{p}.self_attn.q_norm.weight", w
    if rest.startswith("attn_k_norm"):
        return f"{p}.self_attn.k_norm.weight", w
    if rest == "attn_q.weight":
        return f"{p}.self_attn.q_proj.weight", w
    if rest == "attn_k.weight":
        return f"{p}.self_attn.k_proj.weight", w
    if rest == "attn_v.weight":
        return f"{p}.self_attn.v_proj.weight", w
    if rest == "attn_output.weight":
        return f"{p}.self_attn.o_proj.weight", w
    if rest == "ffn_gate.weight":
        return f"{p}.mlp.gate_proj.weight", w
    if rest == "ffn_up.weight":
        return f"{p}.mlp.up_proj.weight", w
    if rest == "ffn_down.weight":
        return f"{p}.mlp.down_proj.weight", w
    if rest == "attn_conv_base":
        # reader already reverses ne=(H,k,halves) → (halves,k,H) row-major
        return f"{p}.attention_conv.base_kernel", w.contiguous()
    if rest == "attn_conv_proj.weight":
        return f"{p}.attention_conv.kernel_projection.weight", w
    if rest == "ffn_conv_base":
        return f"{p}.mlp_conv.base_kernel", w.contiguous()
    if rest == "ffn_conv_proj.weight":
        return f"{p}.mlp_conv.kernel_projection.weight", w
    return None
