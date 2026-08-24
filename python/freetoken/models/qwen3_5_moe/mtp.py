"""MTP (nextn) draft head for the qwen35moe GGUF path — milestone 1.

DeepSeek-V3-style multi-token-prediction head, served from the checkpoint's
own ``blk.<L>.nextn.*`` tensors (dropped by the text-only loader unless
``FREETOKEN_MTP=1``):

    x      = eh_proj( concat( hnorm(h_trunk), enorm(embed(token)) ) )
    h'     = decoder_layer_L(x)          # a GDN layer, its own state
    logits = lm_head( head_norm(h') )    # the shared (quantized) head

This milestone loads and verifies the weights; the scheduler-side
draft-verify loop (speculative decode with GDN-state snapshot/rollback) is
the next milestone. The draft layer's routed experts ride the same offload
bank machinery once the pack includes layer ``num_layers``.
"""

from __future__ import annotations

from freetoken.layers import BaseOP, GemmaRMSNorm, LinearReplicated


class Qwen35MTPDraft(BaseOP):
    """The nextn head; weights land under ``draft.*`` from the GGUF loader."""

    def __init__(self, config):
        from .model import Qwen3_5DecoderLayer

        H = config.hidden_size
        self.enorm = GemmaRMSNorm(H, eps=config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(H, eps=config.rms_norm_eps)
        if getattr(config, "dense_quant", "none") == "ggml_kquant":
            from .ggml_dense import QuantGgmlLinear

            self.eh_proj = QuantGgmlLinear(H, 2 * H)
        else:
            self.eh_proj = LinearReplicated(2 * H, H, has_bias=False)
        # the nextn layer is a FULL-ATTENTION layer (blk.<L> carries
        # attn_q/k/v/o, no GDN tensors): register its id with the full
        # group so the decoder layer constructs the right mixer
        import dataclasses

        fg = next(
            g for g in config.attention_groups
            if getattr(g, "name", "") == "full"
        )
        fg2 = dataclasses.replace(fg, layer_ids=fg.layer_ids + (config.num_layers,))
        cfg2 = dataclasses.replace(config, attention_groups=(fg2, config.linear_attention_group()))
        self.layer = Qwen3_5DecoderLayer(cfg2, config.num_layers)
        self.head_norm = GemmaRMSNorm(H, eps=config.rms_norm_eps)

    def forward(self, h_trunk: torch.Tensor, token_emb: torch.Tensor, lm_head):
        """One draft step: trunk hidden + current token embedding -> logits."""
        x = self.eh_proj.forward(
            torch.cat([self.hnorm.forward(h_trunk), self.enorm.forward(token_emb)], dim=-1)
        )
        h, _ = self.layer.forward(x, None)
        return lm_head.forward(self.head_norm.forward(h))


import torch  # noqa: E402  (kept last: only the annotation needs it)

__all__ = ["Qwen35MTPDraft"]
