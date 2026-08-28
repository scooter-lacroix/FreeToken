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
    # final norm + head go to the FAR side: every near-tail launch after
    # the far block (norm Triton, ggml MMQ head) faulted in-server even
    # though each passed standalone probes -- keeping the whole post-far
    # tail on dev1 (Triton norm + Triton/bf16 head, no ggml) eliminates
    # the class. Only the [T, vocab] logits cross back (1 MB decode,
    # 40 MB prefill).
    if key.startswith(("model.norm.", "lm_head.")):
        return dev1()
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


class FarHead:
    """LM head for the far side: Triton Q4_K GEMV, no ggml kernels (they hang
    on the 7800 XT). Q4_K-only storage (350 MB vs 1.35 GB dual): sampling
    only ever needs the LAST indices per request (T=1-2, see
    GgufKQuantLMHead.forward), which kq_gemv covers; the T>8 path
    (unused in serving) falls back to a chunked dequant GEMM. Output stays
    on the far device; the caller stages logits back."""

    def __init__(self, packed_q4k, k: int):
        self.packed = packed_q4k      # [vocab, rb] uint8 on cuda:1
        self.quant_type = 12
        self.k = k

    def forward(self, x):
        from freetoken.kernel.triton.kquant_linear import kq_gemv
        from freetoken.core import get_global_ctx

        _trace(f"FarHead.forward: x {tuple(x.shape)} on {x.device}")
        # mirror GgufKQuantLMHead: prefill only needs the LAST index per
        # request (computing all T rows through the 8-row GEMV loop would
        # allocate ~1 GB of logit transients on the far side)
        batch = getattr(get_global_ctx(), "_batch", None)
        if batch is not None and getattr(batch, "is_prefill", False):
            _trace("FarHead: get_last_indices")
            indices = batch.attn_metadata.get_last_indices(batch.size)
            _trace(f"FarHead: indices {tuple(indices.shape)} on {indices.device}; indexing x")
            x = x[indices].contiguous()
            _trace("FarHead: x indexed")
        if x.shape[0] <= 8:
            _trace(f"FarHead: kq_gemv T={x.shape[0]}")
            out = kq_gemv(self.packed, x, 12)
            _trace(f"FarHead: kq_gemv done -> {tuple(out.shape)}")
            return out
        outs = []
        for lo in range(0, x.shape[0], 8):
            hi = min(lo + 8, x.shape[0])
            outs.append(kq_gemv(self.packed, x[lo:hi], 12))
        _trace(f"FarHead: chunked gemv done ({len(outs)} chunks)")
        return torch.cat(outs, 0)

    __call__ = forward


def convert_far_head(model) -> bool:
    """Replace the GgufKQuantLMHead with a FarHead (Q4_K requant + bf16)."""
    from freetoken.models.gguf.dequant import dequantize, requantize_q4_k, row_bytes
    from freetoken.models.qwen3_5_moe.ggml_dense import GgufKQuantLMHead

    head = getattr(model, "lm_head", None)
    if not isinstance(head, GgufKQuantLMHead):
        return False
    packed, qt = head.packed, int(head.quant_type)
    out_f = packed.shape[0]
    k = (packed.shape[1] * 256) // row_bytes(256, qt)
    rb = packed.shape[1]
    # chunked on-device requant: a whole-vocab fp32 dequant is 2 GB of
    # transients the far side does not have; 16k-row chunks cap it ~200 MB.
    # Q4_K-only artifact (no bf16 copy) -- see FarHead.
    q4_rows = []
    CH = 16384
    with torch.no_grad(), torch.cuda.device(packed.device):
        flat = packed.reshape(-1)
        for lo in range(0, out_f, CH):
            hi = min(lo + CH, out_f)
            f32 = dequantize(flat[lo * rb: hi * rb], qt, torch.float32)
            q4_rows.append(requantize_q4_k(f32).reshape(hi - lo, -1))
            del f32
        q4 = torch.cat(q4_rows, 0)
        del q4_rows
        torch.cuda.empty_cache()
    model.lm_head = FarHead(q4, k)
    return True


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
    # the far packed weights live on cuda:1 (the device map moved them) and
    # dequantize launches the ggml ext -- WITHOUT this context the ext runs
    # under device-0 against device-1 tensors, the silent wrong-context
    # corruption that later surfaces as a memory fault at the first
    # post-far ggml call in the warmup
    with torch.cuda.device(dev1()):
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
        # blocking direct copy: the metadata tensors are small; the pinned
        # two-hop is reserved for the big hidden/logit crossings (fresh
        # pinned buffers per tensor per forward exhausted pinable memory and
        # surfaced as CUDA OOM at the next warmup alloc)
        return obj.to(device, non_blocking=False) if obj.device != device else obj
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


_staging: dict = {}


def _pin_for(tag, shape, dtype) -> torch.Tensor:
    """The seam's cached pinned staging buffer for ``tag`` (tag=None ->
    fresh buffer; cached tags must be long-lived single-purpose buffers --
    an id()-derived tag can collide across objects and silently deliver the
    WRONG dtype, e.g. int32 fresh_state_indices into index_fill_)."""
    if tag is None:
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    key = (tag, tuple(shape))
    pin = _staging.get(key)
    if pin is None:
        pin = torch.empty(shape, dtype=dtype, pin_memory=True)
        _staging[key] = pin
    return pin


def _stage_cross(src: torch.Tensor, dst_dev: torch.device, tag: str) -> torch.Tensor:
    """Cross a tensor between GPUs through a pinned host buffer.

    DIRECT .to() between the cards -- async OR blocking, with SDMA disabled
    -- corrupts the far context (probe: kernel sequence passes but the
    process segfaults at teardown; the server faulted at the first far
    launch). The pinned two-hop is the pattern the MTP engine ran correctly
    all day. Buffers are cached per tag+shape (KBs each).
    """
    import os as _os

    if _SPLIT_CAPTURE["active"] and _os.environ.get(
        "FREETOKEN_SPLIT_GRAPHS_ALLOW_CROSS", "0"
    ) not in {"1", "true", "yes"}:
        raise RuntimeError(
            "split-graph capture: blocking device crossing inside an open "
            f"segment (tag={tag}, src={src.device} -> {dst_dev}). Pre-cross "
            "this tensor in the seam instead — a blocking copy inside a "
            "capturing stream deadlocks HIP."
        )
    pin = _pin_for(tag, src.shape, src.dtype)
    pin.copy_(src, non_blocking=False)
    return pin.to(dst_dev, non_blocking=False)


def _trace(msg: str) -> None:
    import os

    if os.environ.get("FREETOKEN_LAYER_SPLIT_TRACE", "").strip().lower() not in (
        "", "0", "false", "no", "off",
    ):
        print(f"[split-trace] {msg}", flush=True)


_SEAM_BUF_STATE: dict = {}
_SPLIT_CAPTURE: dict = {"active": False, "ctx": None}


def split_capture_active() -> bool:
    """True while SplitGraphCapture walks the forward (see engine/graph.py)."""
    return _SPLIT_CAPTURE["active"]


def cross_seam_meta_replay(batch) -> None:
    """Refresh the fixed dev1 metadata twins from the live (near) batch so the
    far graph replays against current positions/KV indices."""
    if not _SPLIT_META.get("twins"):
        return
    for attr in ("out_loc", "positions", "attn_metadata", "fla_metadata"):
        val = getattr(batch, attr, None)
        if val is None:
            continue
        twin = _seam_meta_twins(batch).get(attr)
        if twin is not None:
            _copy_dev1(val, twin)


def cross_seam_replay() -> None:
    """Replay-time seam: copy the near graph's OUTPUT hidden/residual into the
    persistent dev1 seam buffers. The near graph writes its final hidden into
    fixed near-side seam SOURCE buffers registered at capture (the same
    tensors cross_to_dev1 read); here we just re-run the pinned two-hop."""
    st = _SPLIT_SRC.get("x")
    if st is None:
        return
    d1 = dev1()
    pin_x = _pin_for("x", st.shape, st.dtype)
    pin_x.copy_(st, non_blocking=False)
    bufs = _seam_dev1_buffers(st.shape[0], st.shape[-1], st.dtype)
    bufs["x"][: st.shape[0]].copy_(pin_x, non_blocking=False)
    sr = _SPLIT_SRC.get("r")
    if sr is not None:
        pin_r = _pin_for("r", sr.shape, sr.dtype)
        pin_r.copy_(sr, non_blocking=False)
        bufs["r"][: sr.shape[0]].copy_(pin_r, non_blocking=False)


_SPLIT_META: dict = {}


def _copy_dev1(src, dst) -> None:
    """Deep copy tensors/containers of tensors from any device into matching
    fixed dev1 structures (recursive for dataclasses/dicts/lists/tuples)."""
    import dataclasses as _dc

    if torch.is_tensor(src):
        if torch.is_tensor(dst):
            dst.copy_(src.to(dst.device, non_blocking=False), non_blocking=False)
        return
    if _dc.is_dataclass(src) and not isinstance(src, type):
        for f in _dc.fields(src):
            _copy_dev1(getattr(src, f.name), getattr(dst, f.name))
    elif isinstance(src, dict):
        for k, v in src.items():
            _copy_dev1(v, dst[k])
    elif isinstance(src, (list, tuple)):
        for a, b in zip(src, dst):
            _copy_dev1(a, b)


def _shape_sig(obj) -> tuple:
    """Recursive shape signature of every tensor inside obj (dataclasses,
    dicts, lists, tuples). Two objects with the same signature are
    _copy_dev1-compatible."""
    import torch as _t
    import dataclasses as _dc

    if _t.is_tensor(obj):
        return ("t", tuple(obj.shape), str(obj.dtype))
    if _dc.is_dataclass(obj) and not isinstance(obj, type):
        return ("dc", type(obj).__name__,
                tuple((f.name, _shape_sig(getattr(obj, f.name)))
                      for f in _dc.fields(obj)))
    if isinstance(obj, dict):
        return ("dict", tuple((k, _shape_sig(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return ("seq", tuple(_shape_sig(v) for v in obj))
    return ("leaf", repr(type(obj)))


def _seam_meta_twins(batch) -> dict:
    """Fixed dev1 mirrors of the far side's batch metadata (created once per
    shape at capture; replays refresh via _copy_dev1)."""
    import copy as _copy
    import dataclasses as _dc

    d1 = dev1()
    if "twins" not in _SPLIT_META:
        _SPLIT_META["twins"] = {}
    tw = _SPLIT_META["twins"]
    out = {}
    for attr in ("out_loc", "positions", "attn_metadata", "fla_metadata"):
        val = getattr(batch, attr, None)
        if val is None:
            continue
        # key per (attr, nested-shape-signature): bs=2 and bs=1 graphs need
        # SEPARATE twins — _copy_dev1 copies into fixed storage, so every
        # nested tensor shape must match exactly (cu_seqlens_q_gpu is
        # [bs+1], indptr [bs+1], logits scratch [bs, ...] — all bs-bound).
        key = (attr, _shape_sig(val))
        if key not in tw:
            tw[key] = _to_device_deep(_copy.deepcopy(val), d1)
        out[attr] = tw[key]
    return out


_SPLIT_SRC: dict = {}


def split_capture_far_output(t: torch.Tensor) -> None:
    """Stash the far trunk's normed output during capture (called from
    Qwen3_5Model.forward just before split_capture_close)."""
    ctx = _SPLIT_CAPTURE["ctx"]
    if ctx is not None:
        ctx.set_far_output(t)


def split_tail_forward(causal_lm, output):
    """Eager per-replay tail: head under far ctx, pinned logits crossing to
    near. Mirrors CausalLM.forward's split branch exactly."""
    from freetoken.core import get_global_ctx

    with torch.cuda.device(output.device):
        logits = causal_lm.lm_head.forward(output)
    near = get_global_ctx().batch.input_ids.device
    pin = _pin_for(None, logits.shape, logits.dtype)
    with torch.cuda.device(logits.device):
        pin.copy_(logits, non_blocking=False)
    return pin.to(near, non_blocking=False)


def restore_batch_near(batch) -> None:
    """Move the batch's far-crossed metadata attrs back to dev0. Captures are
    multi-pass (warmup eager + per-bs), and cross_to_dev1 mutates the batch in
    place every pass — without this, pass N's dev1 attrs leak into pass N+1's
    NEAR segment and near layers hit the fail-fast crossing guard."""
    d0 = torch.device("cuda", 0)
    for attr in ("out_loc", "positions", "attn_metadata", "fla_metadata"):
        val = getattr(batch, attr, None)
        if val is not None:
            try:
                setattr(batch, attr, _to_device_deep(val, d0))
            except Exception:
                pass


def split_capture_close() -> None:
    """Close the far segment at the end of Qwen3_5Model.forward (before
    CausalLM's eager head + logits crossing), and restore the batch's
    original near-side metadata attrs — the in-place twin swap must not
    leak into the next walk's near layers."""
    if not _SPLIT_CAPTURE["active"]:
        return
    ctx = _SPLIT_CAPTURE["ctx"]
    if ctx is not None:
        ctx._close_if_open()
    batch = ctx._batch if ctx is not None else None
    orig = _SPLIT_META.get("orig", {})
    if batch is not None and orig:
        for attr, val in orig.items():
            setattr(batch, attr, val)
        _SPLIT_META["orig"] = {}


def split_capture_seam(layer_id: int) -> None:
    """Close the current device segment and open the next one on the OTHER
    device. Both ends call in: near end (after layer sa-1, still on dev0) and
    far entry (post-crossing, on dev1). Between the two calls the eager seam
    runs OUTSIDE any graph -- the pinned two-hop cannot be captured."""
    if not _SPLIT_CAPTURE["active"]:
        return
    ctx = _SPLIT_CAPTURE["ctx"]
    if ctx is not None:
        ctx._at_seam(layer_id)


def capture_seam_active() -> bool:
    """True while a split-graph capture walk is in flight (twin swap reads
    this; unified on _SPLIT_CAPTURE — the old _SEAM_BUF_STATE flag was never
    set, which silently disabled the twin swap during capture)."""
    return _SPLIT_CAPTURE["active"]


def _seam_dev1_buffers(t: int, h: int, dtype) -> dict:
    """Persistent cuda:1 landing buffers for the seam hidden/residual.

    Grown once to the capture width; every decode step writes through a slice
    so a far-side CUDA graph can be captured against the fixed storage."""
    d1 = dev1()
    cap_t = _SEAM_BUF_STATE.get("capacity", 0)
    if cap_t < t:
        _SEAM_BUF_STATE["x"] = torch.empty((t, h), dtype=dtype, device=d1)
        _SEAM_BUF_STATE["r"] = torch.empty((t, h), dtype=dtype, device=d1)
        _SEAM_BUF_STATE["capacity"] = t
    return _SEAM_BUF_STATE


def cross_to_dev1(batch, x: torch.Tensor, residual: torch.Tensor | None):
    """The seam: move the hidden stream and the far side's view of the batch
    metadata to cuda:1. Mutates ``batch`` in place (single forward in flight;
    the next batch rebuilds its metadata).

    MUST set the thread's current device: Triton's launcher binds kernel
    modules and launches on the CURRENT device's stream (it does not infer
    the device from tensor args), so far-side launches under a device-0
    context execute device-0 streams against device-1 pointers -> memory
    faults / silent garbage. torch ops are unaffected (tensor-device driven).
    """
    d1 = dev1()
    _trace(f"seam: staging x {tuple(x.shape)} pin D2H")
    # D2H to pinned BEFORE switching devices: the producer (near layers) runs
    # on the near stream; a blocking D2H issued under the far device would
    # order against the wrong stream
    pin_x = _pin_for("x", x.shape, x.dtype)
    pin_x.copy_(x, non_blocking=False)
    pin_r = None
    if residual is not None:
        pin_r = _pin_for("r", residual.shape, residual.dtype)
        pin_r.copy_(residual, non_blocking=False)
    _trace("seam: pin D2H done, switching device + H2D")
    torch.cuda.set_device(d1)
    if capture_seam_active():
        _SPLIT_SRC["x"] = x
        _SPLIT_SRC["r"] = residual
        # metadata twins: the far graph's captured reads must land on FIXED
        # dev1 storage; replay re-fills them from the near capture buffer.
        tw = _seam_meta_twins(batch)
        for k, twin in tw.items():
            val = getattr(batch, k, None)
            if val is not None:
                _copy_dev1(val, twin)
    if capture_seam_active():
        # Graph-capture mode: the far graph was captured reading FIXED dev1
        # seam buffers -- landing the H2D into fresh tensors would rebind x
        # and freeze the captured channel. Write through the persistent
        # buffers instead (grow-only, exactly like the near-side graph
        # buffers; bs<=captured width reuses the same storage).
        bufs = _seam_dev1_buffers(x.shape[0], x.shape[-1], x.dtype)
        bufs["x"][: x.shape[0]].copy_(pin_x, non_blocking=False)
        x = bufs["x"][: x.shape[0]]
        if pin_r is not None:
            bufs["r"][: residual.shape[0]].copy_(pin_r, non_blocking=False)
            residual = bufs["r"][: residual.shape[0]]
    else:
        x = pin_x.to(d1, non_blocking=False)
        if pin_r is not None:
            residual = pin_r.to(d1, non_blocking=False)
    _trace("seam: x/resident on far side; crossing metadata attrs")
    # input_ids is deliberately NOT crossed: the embedding ran near-side, no
    # far consumer exists, and CausalLM.forward keys its logits-crossing
    # branch on output.device != batch.input_ids.device -- a far-side
    # input_ids silently disables that branch and hands the near-side sampler
    # a raw cuda:1 logits tensor (cross-device access fault).
    for attr in ("out_loc", "positions", "attn_metadata", "fla_metadata"):
        val = getattr(batch, attr, None)
        if val is not None:
            if capture_seam_active():
                twin = _seam_meta_twins(batch).get(attr)
                if twin is not None:
                    _SPLIT_META.setdefault("orig", {})[attr] = val
                    _SPLIT_CAPTURE["ctx"].bind_batch(batch)
                    setattr(batch, attr, twin)
                    _trace(f"seam: {attr} -> dev1 twin (capture)")
                    continue
            _trace(f"seam: crossing {attr}")
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
            _trace(f"seam: crossed {attr}")
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

    @property
    def num_slots(self) -> int:
        return self._lead.num_slots

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
            if os.environ.get("FREETOKEN_SPLIT_TRACE", ""):
                print(
                    f"[split-trace] store_kv cross: out_loc={out_loc.device} "
                    f"sub={sub.device} layer={layer_id}",
                    flush=True,
                )
            # cross-device index move must be capture-safe: async .to() on
            # the legacy stream inside a capture violates HIP stream rules
            # (hipErrorStreamCaptureImplicit). Route through the pinned
            # staging path — small ints, blocking copies are ~10us.
            out_loc = _stage_cross(out_loc, sub.device, tag="outloc")
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
    "FarHead", "SplitLinearStatePool", "SplitMHAKVCache", "convert_far_head",
    "cross_to_dev1", "dev1",
    "layer_device", "maybe_split_kv_pool", "maybe_split_linear_pool",
    "redevice_rotaries", "split_at", "split_enabled", "weight_device_for",
]
