"""Split-aware decode CUDA graphs (S2c): one graph per device around the seam.

Why piecewise-per-device: a torch.cuda.CUDAGraph captures exactly ONE
device's stream work. The split forward is dev0(near) → eager pinned crossing
→ dev1(far+head), so the runner captures the two compute spans separately and
keeps the crossing OUTSIDE any graph::

    near.replay() ; seam copies (pinned D2H/H2D) ; far.replay() ; logits out

Capture walk mirrors PiecewiseCapture (engine/piecewise.py) but keys pools
PER DEVICE (a shared pool handle across devices is invalid) and registers
itself into layer_split._SPLIT_CAPTURE so the model's seam hooks call back at
the layer boundary. The far graph reads the PERSISTENT dev1 seam buffers
(engine/layer_split._seam_dev1_buffers) written eagerly by each replay's
crossing — never fresh allocations, which would rebind and freeze the wrong
storage.

CPU discipline: replay is two graph launches + four small pinned copies per
step (no Python in the layer loop, no polling, no host spin).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


class SplitGraphCapture:
    """Drives one capture pass over ``model.forward`` for a given bs."""

    def __init__(self, stream_dev0: torch.cuda.Stream):
        self.stream0 = stream_dev0
        self.pools: Dict[torch.device, list] = {}
        self.graphs: Dict[str, Optional[torch.cuda.CUDAGraph]] = {
            "near": None, "far": None,
        }
        self._phase = "near"          # near -> (seam) -> far
        self._current: Optional[torch.cuda.CUDAGraph] = None
        self.bs: Optional[int] = None

    # -- called by layer_split.split_capture_seam --------------------------
    # Hook order inside fn():  NEAR end (dev0) -> [eager crossing executes] ->
    # FAR entry (dev1). Closing at near-end leaves the crossing OUTSIDE the
    # near graph; far-entry opens the dev1 segment around the far layers.
    def bind_batch(self, batch) -> None:
        self._batch = batch

    def _at_seam(self, layer_id: int) -> None:
        if self._phase == "near":
            self._close_segment("near")
            self._phase = "crossing"
        elif self._phase == "crossing":
            from freetoken.engine.layer_split import dev1

            # switch the WALK to dev1's side stream before opening the far
            # segment; blocking seam copies on dev0 finished on the walk
            # stream BEFORE the near close, so no cross-stream hazard.
            torch.cuda.set_stream(self.stream1)
            self._open_segment(dev1())
            self._phase = "far"

    # -- segment lifecycle --------------------------------------------------
    def _open_segment(self, dev: torch.device) -> None:
        g = torch.cuda.CUDAGraph()
        # per-segment pools: torch 2.13-ROCm returns None pool() handles for
        # some graphs; sharing None across segments crashes. Two tiny pools
        # (activations only) cost nothing.
        self._open_stream = torch.cuda.current_stream(dev)
        g.capture_begin(pool=None)
        self._current = g
        self._cur_dev = dev

    def _close_if_open(self) -> None:
        if self._current is not None and self._phase == "far":
            self._close_segment("far")

    def _close_segment(self, name: str) -> None:
        assert self._current is not None
        open_stream = getattr(self, "_open_stream", None)
        if open_stream is not None:
            torch.cuda.set_stream(open_stream)
        self._current.capture_end()
        self.graphs[name] = self._current
        dev = self._cur_dev
        try:
            pool_handle = self._current.pool()
        except Exception:
            pool_handle = None
        if dev not in self.pools:
            self.pools[dev] = [pool_handle] if pool_handle is not None else []
        elif self.pools[dev] and self.pools[dev][0] is None and pool_handle is not None:
            self.pools[dev][0] = pool_handle
        self._current = None

    # -- driver --------------------------------------------------------------
    def capture(self, bs: int, fn, batch=None) -> "SplitGraphCapture":
        """``fn`` runs the FULL forward once (both devices + eager seam)."""
        from freetoken.engine import layer_split as ls

        self.bs = bs
        if batch is not None:
            self.bind_batch(batch)
            ls.restore_batch_near(batch)
        assert not ls._SPLIT_CAPTURE["active"]
        ls._SPLIT_CAPTURE.update(active=True, ctx=self)
        dev0 = torch.device("cuda", 0)
        dev1 = ls.dev1()
        try:
            torch.cuda.synchronize(dev0)
            torch.cuda.synchronize(dev1)
            self.stream1 = torch.cuda.Stream(device=dev1)
            with torch.cuda.stream(self.stream0):
                self._open_segment(dev0)
                self._phase = "near"
                fn()          # seam hooks close near / open far / close far
                if self._current is not None:      # safety net
                    self._close_segment("far")
                if self.graphs.get("far") is None:
                    raise RuntimeError("split capture: far segment missing")
                torch.cuda.set_stream(self.stream0)
        finally:
            ls._SPLIT_CAPTURE.update(active=False, ctx=None)
        return self


    def far_output(self):
        """The dev1 tensor the far graph's trunk norm wrote (captured last
        output). Kept alive by the capture dict."""
        return getattr(self, "_far_out", None)

    def set_far_output(self, t) -> None:
        self._far_out = t
    # -- replay --------------------------------------------------------------
    def replay_seam(self) -> None:
        """Eager crossing between graph replays: dev0 tail -> pinned -> dev1
        seam buffers. Mirrors cross_to_dev1's staging minus metadata (the far
        graph reads the buffers captured at capture time; metadata tensors it
        touches must ALSO be persistent — handled by GraphCaptureBuffer
        dev1 twins in the runner)."""
        raise NotImplementedError("wired in GraphRunner.split_replay")

    def replay(self) -> None:
        g_near = self.graphs["near"]
        g_far = self.graphs["far"]
        if g_near is not None:
            g_near.replay()
        if g_far is not None:
            g_far.replay()


__all__ = ["SplitGraphCapture"]
