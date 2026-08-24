"""Piecewise (segment) CUDA-graph capture for host-mediated expert tiers.

A monolithic decode graph cannot serve the SSD expert tier (``ggml_file``
banks): the miss copies are driven by the HOST (the GPU cannot fault page-cache
pages), so a single capture would have to freeze a variable-length batch of
H2D transfers into every replay. Instead the trunk is captured as a chain of
SEGMENTS that end right after each layer's ``ensure_experts`` -- the LRU
bookkeeping is device-side and capturable -- and the runner plays

    seg[0] ; copy_missing(layer 0) ; seg[1] ; copy_missing(layer 1) ; ...

with the eager staged copies running OUTSIDE any graph between two replays
(see :mod:`freetoken.moe.file_staging`). Only the expert fetch pays eager
dispatch; attention/dense/norm kernels replay as graphs, which is where most
of the per-step launch cost lives.

Cross-seam tensors (hidden states, topk ids/weights) are registered into a
keepalive list by the layer at seam time, so the shared graph pool cannot
recycle their storage while an earlier segment still writes it on replay.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch


class _CaptureState:
    def __init__(self) -> None:
        self.active = False
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.pool = None  # shared graph memory-pool handle (None until first close)
        self.graphs: List[torch.cuda.CUDAGraph] = []
        self.seams: List[int] = []  # MoE layer ids in encounter order
        self.keepalive: List[torch.Tensor] = []

    def reset(self, pool) -> None:
        self.graph = None
        self.pool = pool
        self.graphs = []
        self.seams = []
        self.keepalive = []


_state = _CaptureState()


def piecewise_capture_active() -> bool:
    return _state.active


def capture_seam(layer_id: int, tensors=()) -> None:
    """Close the current segment and open the next one (capture time only).

    Called by OffloadMoELayer at the ensure/copy seam while the piecewise
    runner captures; a no-op outside a capture walk. ``tensors`` are any
    cross-seam activations whose storage must survive until the consuming
    segment's replay.
    """
    if not _state.active:
        return
    if tensors:
        _state.keepalive.extend(tensors)
    _state.seams.append(layer_id)
    _state.graph.capture_end()
    self_graphs = _state.graphs
    self_graphs.append(_state.graph)
    if _state.pool is None:
        _state.pool = _state.graph.pool()
    g = torch.cuda.CUDAGraph()
    _state.graph = g
    g.capture_begin(pool=_state.pool)


class PiecewiseCapture:
    """Drives one segmented capture pass over ``model.forward``."""

    def __init__(self, stream: torch.cuda.Stream, pool=None):
        self._stream = stream
        self._incoming_pool = pool
        # Set after capture(): the pool handle later batch sizes must reuse.
        self.pool = pool
        self.graphs: List[torch.cuda.CUDAGraph] = []
        self.seams: List[int] = []

    def capture(self, fn: Callable[[], object]) -> "PiecewiseCapture":
        assert not _state.active, "nested piecewise capture"
        _state.reset(self._incoming_pool)
        _state.active = True
        try:
            torch.cuda.synchronize(self._stream.device)
            with torch.cuda.stream(self._stream):
                try:
                    g = torch.cuda.CUDAGraph()
                    _state.graph = g
                    g.capture_begin(pool=self._incoming_pool)
                    try:
                        fn()
                    finally:
                        _state.graph.capture_end()
                        _state.graphs.append(_state.graph)
                finally:
                    if _state.pool is None and _state.graphs:
                        _state.pool = _state.graphs[0].pool()
            self.graphs = list(_state.graphs)
            self.seams = list(_state.seams)
            self.pool = _state.pool
        finally:
            _state.active = False
        return self

    @property
    def keepalive(self) -> List[torch.Tensor]:
        return _state.keepalive


__all__ = ["PiecewiseCapture", "capture_seam", "piecewise_capture_active"]
