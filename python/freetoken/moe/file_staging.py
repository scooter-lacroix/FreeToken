"""Pinned-staging H2D for page-cache-backed (``ggml_file``) expert banks.

The GPU cannot dereference unpinned host memory, so every file-bank byte must
cross PCIe through a bounce buffer. ``tensor.copy_(pageable, non_blocking=_)``
makes the driver stage each call internally -- correct, but one driver round
trip per row with the host blocked until it completes. This module pins small
rotating scratch buffers instead: rows pack into pinned memory with plain
memcpys (page-cache friendly, prefetch-friendly), cross as one large async H2D
per chunk, and scatter device-side into the slot cache. While the GPU drains
chunk k the host already packs chunk k+1 into the other buffer.
"""

from __future__ import annotations

import os

import numpy as np
import torch


def _staging_bytes() -> int:
    raw = os.getenv("FREETOKEN_FILE_STAGING_MB", "32")
    try:
        mb = int(raw)
    except ValueError:
        mb = 32
    return max(1, mb) * (1 << 20)


class FileBankStager:
    """Two-deep pinned staging ring, bucketed by row size.

    One instance per OffloadMoeCache (lazy: only the file tier builds one).
    Buffers are uint8; callers pass flattened byte views of their banks.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._budget = _staging_bytes()
        # feat_bytes -> (host [2, cap, feat] uint8 pinned, dev [2, cap, feat], events [2])
        self._rings: dict[int, tuple[torch.Tensor, torch.Tensor, list]] = {}
        # Linear (whole-tensor) ring, sized in bytes rather than rows.
        self._lin: tuple[torch.Tensor, torch.Tensor, list] | None = None
        self._rotate = 0

    def _pick(self) -> int:
        b = self._rotate % 2
        self._rotate += 1
        return b

    def _ring_for(self, feat: int) -> tuple[torch.Tensor, torch.Tensor, list]:
        entry = self._rings.get(feat)
        if entry is None:
            cap = max(1, self._budget // max(1, feat))
            host = torch.empty(2, cap, feat, dtype=torch.uint8, pin_memory=True)
            dev = torch.empty(2, cap, feat, dtype=torch.uint8, device=self.device)
            entry = (host, dev, [torch.cuda.Event(), torch.cuda.Event()])
            self._rings[feat] = entry
        return entry

    def _linear_ring(self, chunk: int) -> tuple[torch.Tensor, torch.Tensor, list]:
        if self._lin is None or self._lin[0].shape[1] < chunk:
            host = torch.empty(2, chunk, dtype=torch.uint8, pin_memory=True)
            dev = torch.empty(2, chunk, dtype=torch.uint8, device=self.device)
            self._lin = (host, dev, [torch.cuda.Event(), torch.cuda.Event()])
        return self._lin

    def gather_rows(
        self,
        src_flat: torch.Tensor,
        rows,
        dst_flat: torch.Tensor,
        slots: torch.Tensor,
        n: int,
    ) -> None:
        """``dst_flat[slots[i]] = src_flat[rows[i]]`` for ``i < n``.

        ``src_flat``: CPU uint8 ``[num_experts, feat]`` (an mmap view);
        ``dst_flat``: GPU uint8 ``[cache_size, feat]`` (the slot cache);
        ``rows``: host expert ids (numpy int array); ``slots``: GPU int32
        cache-slot ids written by ``ensure_experts``.
        """
        if n <= 0 or self.device.type != "cuda":
            return
        feat = src_flat.shape[1]
        host, dev, events = self._ring_for(feat)
        cap = host.shape[1]
        pos = 0
        cur = torch.cuda.current_stream(self.device)
        while pos < n:
            b = self._pick()
            m = min(cap, n - pos)
            events[b].synchronize()  # buffer's previous H2D has drained
            r = torch.from_numpy(rows[pos : pos + m].astype(np.int64))
            torch.index_select(src_flat, 0, r, out=host[b, :m])
            dev[b, :m].copy_(host[b, :m], non_blocking=True)
            events[b].record(cur)
            dst_flat.index_copy_(0, slots[pos : pos + m].long(), dev[b, :m])
            pos += m

    def linear(self, dst_gpu: torch.Tensor, src_cpu: torch.Tensor) -> None:
        """Chunked staged copy of one flat CPU byte range to a flat GPU range."""
        if self.device.type != "cuda":
            dst_gpu.copy_(src_cpu, non_blocking=True)
            return
        total = src_cpu.numel()
        if total == 0:
            return
        host, dev, events = self._linear_ring(min(total, self._budget))
        cap = host.shape[1]
        off = 0
        cur = torch.cuda.current_stream(self.device)
        while off < total:
            b = self._pick()
            m = min(cap, total - off)
            events[b].synchronize()
            host[b, :m].copy_(src_cpu[off : off + m])
            dev[b, :m].copy_(host[b, :m], non_blocking=True)
            events[b].record(cur)
            dst_gpu[off : off + m].copy_(dev[b, :m])
            off += m


__all__ = ["FileBankStager"]
