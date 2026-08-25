from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
                dense_quant=getattr(config, "dense_quant", "none"),
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        if getattr(config, "lm_head_quant", "none") == "ggml_kquant":
            # GGUF: the (tied) embedding table stays in its native k-quant bytes;
            # rows dequantize on gather instead of a resident bf16 copy.
            from .ggml_dense import QuantGGMLEmbedding

            self.embed_tokens = QuantGGMLEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        from freetoken.engine.layer_split import split_at, split_enabled, cross_to_dev1

        sa = split_at() if split_enabled() else 0
        for i, layer in enumerate(self.layers.op_list):
            if sa and i == sa:
                # layer-split seam: hidden stream + far-side batch metadata
                # cross to cuda:1 once per forward (eager path)
                x, residual = cross_to_dev1(get_global_ctx().batch, x, residual)
            x, residual = layer.forward(x, residual)
        if sa:
            # cross back for the final norm + lm_head on the near side (the
            # crossing carries [T, H] hidden, not [T, vocab] logits)
            d0 = x.device  # near side = the embedding's device
            near = get_global_ctx().batch.input_ids.device
            x = x.to(near, non_blocking=True)
            residual = residual.to(near, non_blocking=True)
            from freetoken.engine.layer_split import _to_device_deep

            b = get_global_ctx().batch
            for attr in ("out_loc", "positions", "attn_metadata", "fla_metadata"):
                val = getattr(b, attr, None)
                if val is not None:
                    try:
                        setattr(b, attr, _to_device_deep(val, near))
                    except Exception:
                        pass
        xn, rn = self.norm.forward_add_residual(x, residual)
        # Pre-final-norm trunk residual for the MTP head (llama.cpp feeds the
        # trunk's final residual to nextn hnorm). ONE stable buffer written by
        # slice: graph capture walks the batch-size list largest -> smallest
        # with an eager warmup forward before each level, so a per-shape
        # buffer would rebind at every level and ctx would end up pointing at
        # whichever level was captured LAST -- frozen for every other replay
        # width. Decode-only: prefill must NOT touch the channel (decode
        # replays never re-run Python, so a prefill clobber persists for the
        # whole request) and prefill widths could exceed the buffer, forcing a
        # rebind that orphans the captured copy_ targets.
        ctx = get_global_ctx()
        if not getattr(ctx.batch, "is_prefill", False):
            buf = getattr(self, "_trunk_prenorm_buf", None)
            if buf is None:
                self._trunk_prenorm_buf = rn.new_empty(rn.shape)
                buf = self._trunk_prenorm_buf
            if rn.shape[0] <= buf.shape[0]:
                buf[: rn.shape[0]].copy_(rn)
                ctx.trunk_hidden_prenorm = buf
            else:
                ctx.trunk_hidden_prenorm = rn  # oversized eager decode
        return xn


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        lm_q = getattr(config, "lm_head_quant", "none")
        if lm_q == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        elif lm_q == "ggml_kquant":
            from .ggml_dense import GgufKQuantLMHead

            self.lm_head = GgufKQuantLMHead(config.vocab_size, config.hidden_size)
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()
        if __import__("os").environ.get("FREETOKEN_MTP", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            # nextn (MTP) draft head; weights arrive under draft.* when the
            # GGUF loader runs with the same env (see gguf.iter_gguf_weights)
            from .mtp import Qwen35MTPDraft

            self.draft = Qwen35MTPDraft(config)

    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        output = self.model.forward(ctx.batch.input_ids)
        if output.device != ctx.batch.input_ids.device:
            # layer split: logits are produced on cuda:1; the sampler and the
            # output pipeline live on cuda:0
            output = output.to(ctx.batch.input_ids.device, non_blocking=True)
        # Post-norm trunk hidden (MTP diagnostics; lm_head consumes `output`,
        # so it is live every replay). `output` itself is a per-capture-level
        # pool tensor -- publish it through the same stable slice-written
        # buffer pattern as the pre-norm channel. Decode-only, like the
        # pre-norm channel (a prefill clobber would persist across replays).
        if not getattr(ctx.batch, "is_prefill", False):
            buf = getattr(self, "_trunk_post_buf", None)
            if buf is None:
                self._trunk_post_buf = output.new_empty(output.shape)
                buf = self._trunk_post_buf
            if output.shape[0] <= buf.shape[0]:
                buf[: output.shape[0]].copy_(output)
                ctx.trunk_hidden = buf
            else:
                ctx.trunk_hidden = output  # oversized eager decode
        return self.lm_head.forward(output)


__all__ = ["Qwen3_5MoEForCausalLM"]
