"""Layer-split placement across two GPUs (dense models, eager path first).

FREETOKEN_LAYER_SPLIT=N puts trunk layers >= N (plus the final norm and
lm_head) on cuda:1; layers < N and the embedding stay on cuda:0. The hidden
stream crosses at the seam once per forward -- a small pinned-friendly copy
of (x, residual) plus the per-batch index/metadata tensors the far-side
layers and kernels need (out_loc, positions, attention + fla metadata).

Storage splits with the layers: the paged-KV pool and the GDN state pool
each become two sub-pools (one per device) behind routing wrappers that keep
the global-layer-id interface, so page/slot ids stay global and identical on
both sides. Cross-device index tensors are copied to the owning device
inside the wrappers (they are KBs).

Decode graphs do not span devices; run with --cuda-graph-max-bs 0 for the
eager split (graph-captured split = follow-up: two segment graphs with a
host-mediated seam, the piecewise pattern).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Sequence

import torch

_cache: tuple[int, bool] | None = None


def split_at() -> int:
    """First layer id placed on cuda:1 (0 = split disabled)."""
    global _cache
    if _cache is None:
        n = int(os.environ.get("FREETOKEN_LAYER_SPLIT", "0") or 0)
        ok = n > 0 and torch.cuda.is_available() and torch.cuda.device_count() > 1
        _cache = (n if ok else 0, ok)
    return _cache[0]


def split_enabled() -> bool:
    split_at()
    return _cache[1] and _cache[0] > 0


def dev1() -> torch.device:
    return torch.device("cuda:1")


def layer_device(layer_id: int) -> torch.device:
    return dev1() if split_enabled() and layer_id >= split_at() else torch.device("cuda:0")


def weight_device_for(key: str) -> torch.device:
    """Device for a dense state-dict key under the split."""
    if not split_enabled():
        return torch.device("cuda:0")
    if key.startswith("model.layers."):
        layer_id = int(key.split(".")[2])
        return layer_device(layer_id)
    # final norm + head stay on the NEAR side: the crossing carries the
    # [T, H] hidden (KBs) instead of [T, vocab] logits (MBs), and the
    # k-quant lm_head keeps its working ggml GEMV on cuda:0
    return torch.device("cuda:0")


class FarSideLinear:
    """Dense projection for a far-side (cuda:1) layer.

    The vendored ggml kernels hang on gfx1101 (the 7800 XT) even with a
    dual-arch fatbin -- far-side compute must be Triton/torch only. Decode
    (T<=8, N>=2048) runs our Q4_K Triton GEMV over requantized packed rows;
    everything else (prefill, small projections) runs a bf16 transposed
    GEMM. Dual storage: Q4_K + bf16 ~= 0.78 GB per big layer.
    """

    def __init__(self, packed_q4k: torch.Tensor, wt_bf16: torch.Tensor):
        self.packed = packed_q4k        # [N, rb(K)] uint8, cuda:1
        self.quant_type = 12
        self.wt = wt_bf16               # [K, N] bf16, cuda:1 (x @ wt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.kquant_linear import kq_gemv

        T, K = x.shape[0], x.shape[-1]
        if T <= 8 and self.packed.shape[0] >= 2048:
            return kq_gemv(self.packed, x, 12)
        return x @ self.wt

    __call__ = forward


def convert_far_linears(model) -> int:
    """Replace every QuantGgmlLinear under the far-side layers with a
    FarSideLinear (Q4_K requant + bf16 transposed). Returns the count."""
    from freetoken.models.gguf.dequant import (
        dequantize, requantize_q4_k, row_bytes,
    )
    from freetoken.models.qwen3_5_moe.ggml_dense import QuantGgmlLinear

    def walk(obj, depth=0):
        n = 0
        if depth > 4:
            return n
        for name, val in list(obj.__dict__.items()):
            if isinstance(val, QuantGgmlLinear):
                packed, qt = val.packed, int(val.quant_type)
                out_f = packed.shape[0]
                k = (packed.shape[1] * 256) // row_bytes(256, qt)
                with torch.no_grad():
                    f32 = dequantize(packed.reshape(-1), qt, torch.float32)
                    wt = f32.reshape(out_f, k).t().contiguous().to(torch.bfloat16)
                    q4 = (
                        requantize_q4_k(f32).reshape(out_f, -1)
                        if out_f >= 2048 else packed
                    )
                setattr(obj, name, FarSideLinear(q4, wt))
                n += 1
            elif hasattr(val, "__dict__"):
                n += walk(val, depth + 1)
        return n

    layers = model.model.layers.op_list
    total = 0
    for i in range(split_at(), len(layers)):
        total += walk(layers[i])
    # release the conversion's fp32/bf16 transients back to the driver:
    # the far-side KV budget is measured with mem_get_info right after this
    # (empty_cache frees the CURRENT device's cache -- switch to cuda:1 or
    # the far side's reservations stay held)
    torch.cuda.synchronize(dev1())
    with torch.cuda.device(dev1()):
        torch.cuda.empty_cache()
    return total


# ---------------------------------------------------------------------------

def _to_device_deep(obj: Any, device: torch.device) -> Any:
    """Recursively move every tensor in a metadata structure to ``device``.

    Handles the namedtuple/dataclass/dict/list/tuple shapes the batch
    metadata objects use; anything else passes through untouched.
    """
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True) if obj.device != device else obj
    if isinstance(obj, dict):
        return {k: _to_device_deep(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        conv = type(obj)
        vals = [_to_device_deep(v, device) for v in obj]
        try:
            return conv(vals)
        except TypeError:  # plain tuple constructor is fine; others fall back
            return tuple(vals)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj, **{
                f.name: _to_device_deep(getattr(obj, f.name), device)
                for f in dataclasses.fields(obj)
            }
        )
    if hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*[_to_device_deep(getattr(obj, f), device) for f in obj._fields])
    return obj


def cross_to_dev1(batch, x: torch.Tensor, residual: torch.Tensor | None):
    """The seam: move the hidden stream and the far side's view of the batch
    metadata to cuda:1. Mutates ``batch`` in place (single forward in flight;
    the next batch rebuilds its metadata)."""
    d1 = dev1()
    x = x.to(d1, non_blocking=True)
    if residual is not None:
        residual = residual.to(d1, non_blocking=True)
    for attr in ("out_loc", "positions", "attn_metadata", "fla_metadata", "input_ids"):
        val = getattr(batch, attr, None)
        if val is not None:
            try:
                setattr(batch, attr, _to_device_deep(val, d1))
            except Exception as e:  # noqa: BLE001
                if not getattr(cross_to_dev1, "_warned", False):
                    cross_to_dev1._warned = True
                    import logging

                    logging.getLogger(__name__).warning(
                        "layer-split seam: could not move %r (%s); far-side kernels "
                        "may see cross-device tensors", attr, e,
                    )
    if not getattr(cross_to_dev1, "_logged", False):
        cross_to_dev1._logged = True
        fla = getattr(batch, "fla_metadata", None)
        fi = getattr(fla, "fresh_state_indices", "absent")
        print(
            f"[layer-split seam] x->{x.device} residual->"
            f"{residual.device if residual is not None else None} "
            f"fla.fresh_state_indices->"
            f"{fi.device if torch.is_tensor(fi) else fi}",
            flush=True,
        )
    return x, residual


def redevice_rotaries(model) -> None:
    """Deep-copy the rope tables of the far side's attention layers onto
    cuda:1 (get_rope's cache is device-bound at construction)."""
    import copy

    layers = model.model.layers.op_list
    for i in range(split_at(), len(layers)):
        attn = getattr(layers[i], "self_attn", None)
        rotary = getattr(attn, "rotary", None)
        if rotary is not None and getattr(rotary, "_cos_sin_cache", None) is not None:
            r2 = copy.deepcopy(rotary)
            r2._cos_sin_cache = rotary._cos_sin_cache.to(dev1())
            attn.rotary = r2


# ---------------------------------------------------------------------------
# Routing pool wrappers

class _RoutingLayerTensor:
    """Reads like the per-layer storage ``conv_states[local_layer]`` while
    routing a GLOBAL dense layer id to its owning sub-pool."""

    def __init__(self, subs: list[torch.Tensor], starts: list[int]):
        self._subs = subs
        self._starts = starts
        self.shape = (sum(s.shape[0] for s in subs), *subs[0].shape[1:])

    def __getitem__(self, global_li: int):
        for sub, start in zip(self._subs, self._starts):
            if start <= global_li < start + sub.shape[0]:
                return sub[global_li - start]
        raise IndexError(global_li)


class SplitLinearStatePool:
    """Two LinearStatePools (one per device) behind the global interface.

    Slots are per-request ids, identical on both sides; every slot operation
    applies to both sub-pools. ``local_index`` returns a GLOBAL dense layer
    id that ``conv_states``/``recurrent_states`` route to the right side.
    """

    def __init__(self, sub0, sub1, global_local_index: dict[int, int]):
        self._subs = [sub0, sub1]
        self._global_index = global_local_index
        n0 = sub0.conv_states.shape[0]
        self._starts = [0, n0]
        self.conv_states = _RoutingLayerTensor(
            [sub0.conv_states, sub1.conv_states], self._starts)
        self.recurrent_states = _RoutingLayerTensor(
            [sub0.recurrent_states, sub1.recurrent_states], self._starts)
        self.padding_slot = sub0.padding_slot
        # bookkeeping delegates to sub0 (identical free-list on both)
        self._lead = sub0

    @property
    def num_free_slots(self) -> int:
        return self._lead.num_free_slots

    def alloc(self, n: int = 1) -> list[int]:
        ids = self._lead.alloc(n)
        for sub in self._subs[1:]:
            sub.alloc(n)
        return ids

    def free(self, slots) -> None:
        for sub in self._subs:
            sub.free(slots)

    def reclaim_all_slots(self) -> None:
        for sub in self._subs:
            sub.reclaim_all_slots()

    def clear_slots(self, slots) -> None:
        for sub in self._subs:
            sub.clear_slots(slots)

    def copy_from(self, src: int, dst: int) -> None:
        for sub in self._subs:
            sub.copy_from(src, dst)

    def is_linear_layer(self, layer_id: int) -> bool:
        if layer_id in self._global_index and not getattr(self, "_dbg", False):
            self._dbg = True
            li = self._global_index[layer_id]
            sub_idx = next(i for i, (sub, start) in enumerate(
                zip(self._subs, self._starts)) if start <= li < start + sub.conv_states.shape[0])
            print(
                f"[layer-split dbg] first local_index call: layer={layer_id} li={li} "
                f"starts={self._starts} sub_sizes={[s.conv_states.shape[0] for s in self._subs]} "
                f"routed_sub={sub_idx} sub_devs={[s._device for s in self._subs]}",
                flush=True,
            )
        return layer_id in self._global_index

    def local_index(self, layer_id: int) -> int:
        li = self._global_index[layer_id]
        if not getattr(self, "_dbg2", False):
            self._dbg2 = True
            sub_idx = next(i for i, (sub, start) in enumerate(
                zip(self._subs, self._starts)) if start <= li < start + sub.conv_states.shape[0])
            print(
                f"[layer-split dbg] first local_index: layer={layer_id} li={li} "
                f"starts={list(self._starts)} sizes={[s.conv_states.shape[0] for s in self._subs]} "
                f"sub={sub_idx} devs={[str(s._device) for s in self._subs]} "
                f"nslots={[s.conv_states.shape[1] for s in self._subs]}",
                flush=True,
            )
        return li

    def rebuild(self, num_slots: int) -> None:
        for sub in self._subs:
            sub.rebuild(num_slots)
        self._lead = self._subs[0]


class SplitMHAKVCache:
    """Two MHAKVCache sub-pools behind the global-layer-id KV interface.

    NOT a BaseKVCachePool subclass on purpose (the ABC's __init__ allocates);
    the engine-facing classmethods (kv_cost / solve_num_pages /
    validate_rebuild) are invoked before the split exists, on the plain
    class, so the wrapper only needs the instance surface.
    """

    def __init__(self, sub0, sub1, split_at: int):
        self._subs = [(split_at, sub0), (1 << 62, sub1)]
        self._split_at = split_at
        self.needs_rebind_on_rebuild = sub0.needs_rebind_on_rebuild

    # class-level cost surface delegates statically (used pre-construction)
    @staticmethod
    def kv_cost(config, **kwargs):
        from freetoken.kvcache.mha_pool import MHAKVCache

        return MHAKVCache.kv_cost(config, **kwargs)

    def _for_layer(self, layer_id: int):
        for bound, sub in self._subs:
            if layer_id < bound:
                return sub
        raise IndexError(layer_id)

    @property
    def device(self) -> torch.device:
        return self._subs[0][1].device

    @property
    def dtype(self) -> torch.dtype:
        return self._subs[0][1].dtype

    @property
    def num_layers(self) -> int:
        return self._subs[0][1].num_layers

    def unit_bytes(self) -> tuple[int, int]:
        return self._subs[0][1].unit_bytes()

    def k_cache(self, layer_id: int) -> torch.Tensor:
        return self._for_layer(layer_id).k_cache(layer_id)

    def v_cache(self, layer_id: int) -> torch.Tensor:
        return self._for_layer(layer_id).v_cache(layer_id)

    def store_kv(self, k, v, out_loc, layer_id: int) -> None:
        sub = self._for_layer(layer_id)
        if torch.is_tensor(out_loc) and out_loc.device != sub.device:
            out_loc = out_loc.to(sub.device, non_blocking=True)
        sub.store_kv(k, v, out_loc, layer_id)

    def rebuild(self, num_pages: int) -> None:
        for _, sub in self._subs:
            sub.rebuild(num_pages)

    def rebuild_from_config(self, config, num_pages: int, **kw) -> None:
        for _, sub in self._subs:
            sub.rebuild_from_config(config, num_pages, **kw)

    def attach_page_table(self, page_table: torch.Tensor) -> None:
        for _, sub in self._subs:
            pt = page_table.to(sub.device) if page_table.device != sub.device else page_table
            sub.attach_page_table(pt)


def maybe_split_kv_pool(base_factory, model_config, num_pages, page_size, dtype, device):
    """Wrap the plain pool factory: when the split is on, build one sub-pool
    per device over its side's full-attention layer ids."""
    if not split_enabled():
        return base_factory()
    from .layer_split import split_at as _sa  # self-import guard for picklers

    sa = _sa()
    full_ids = ()
    if model_config.has_linear_attention:
        specs = [s for s in model_config.kv_cache_group_specs() if s.num_layers > 0]
        full_ids = specs[0].layer_ids
        num_kv_heads, head_dim = specs[0].num_kv_heads, specs[0].head_dim
    else:
        num_kv_heads, head_dim = model_config.num_kv_heads, model_config.head_dim
    from freetoken.kvcache.mha_pool import MHAKVCache

    def make(ids):
        return MHAKVCache(
            num_kv_heads=num_kv_heads, num_pages=num_pages, page_size=page_size,
            num_layers=model_config.num_layers, head_dim=head_dim,
            device=torch.device("cuda:0") if not ids or min(ids) < sa else dev1(),
            dtype=dtype, layer_ids=ids or None,
        )

    sub0 = make(tuple(i for i in full_ids if i < sa))
    sub1 = make(tuple(i for i in full_ids if i >= sa))
    return SplitMHAKVCache(sub0, sub1, sa)


def maybe_split_linear_pool(factory, group, num_slots, dtype, device, tp_size):
    """Same wrap for the GDN state pool: one sub-pool per device over its
    side's linear layer ids, behind SplitLinearStatePool."""
    if not split_enabled():
        return factory()
    import dataclasses as _dc

    sa = split_at()
    ids = list(group.layer_ids)
    g0 = _dc.replace(group, layer_ids=tuple(i for i in ids if i < sa))
    g1 = _dc.replace(group, layer_ids=tuple(i for i in ids if i >= sa))
    sub0 = factory(g0)
    sub1 = factory(g1)
    # the engine's factory closure pins device=cuda:0; move the far side's
    # storage onto cuda:1 explicitly (two tensors, one-time)
    d1 = dev1()
    sub1._device = d1
    sub1.conv_states = sub1.conv_states.to(d1)
    sub1.recurrent_states = sub1.recurrent_states.to(d1)
    global_idx: dict[int, int] = {}
    n = 0
    for i in ids:
        global_idx[i] = n
        n += 1
    return SplitLinearStatePool(sub0, sub1, global_idx)


__all__ = [
    "SplitLinearStatePool", "SplitMHAKVCache", "cross_to_dev1", "dev1",
    "layer_device", "maybe_split_kv_pool", "maybe_split_linear_pool",
    "redevice_rotaries", "split_at", "split_enabled", "weight_device_for",
]
