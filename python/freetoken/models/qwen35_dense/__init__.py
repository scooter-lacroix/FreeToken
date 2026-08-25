"""Dense qwen35 (Qwen3.8-27B class) GGUF support: the qwen3_5_moe model
classes with a dense config + this package's GGUF config/weight loaders.

No ``setup_offload_expert_banks`` export on purpose: the family is dense
(num_experts == 0), so expert-bank machinery never engages and the generic
providers are never consulted."""

from .gguf import iter_gguf_weights, parse_gguf_config

# Same model classes as the MoE family: the decoder layer branches to
# Qwen3_5DenseMLP when moe_enabled=False and the GDN / full-attention
# modules are dimension-parameterized.
from freetoken.models.qwen3_5_moe import Qwen3_5MoEForCausalLM

__all__ = ["Qwen3_5MoEForCausalLM", "parse_gguf_config", "iter_gguf_weights"]
