"""Keep the model file's expert banks resident in the page cache.

The MoE-offload tiers read expert weights out of the (mmap'd) checkpoint on
demand. When those pages are resident, prefill runs at page-cache speed; when
they have been evicted -- this box's checkpoints live on a shared spinning
disk -- every fault queues behind whatever else is using the disk. Measured
(2026-08-29): a pipeline probe stalled 135s while faulting 81MB (0.6MB/s)
behind concurrent desktop IO, and a cold boot's chunked prefill ran at
~250 tok/s (disk-bound) versus ~2000 tok/s page-cache-warm.

``prefetch_model_file`` asks the kernel to populate the page cache for the
whole file (POSIX_FADV_WILLNEED: asynchronous, reclaimable, cannot OOM), and
``KeepResident`` periodically re-issues it -- on resident ranges the advice is
effectively free, on evicted ranges the kernel restores them at background
priority before serving traffic needs them.
"""

from __future__ import annotations

import os
import threading

from . import init_logger

logger = init_logger(__name__)


def prefetch_model_file(path: str) -> int:
    """Populate the page cache for ``path`` (best-effort). Returns the file size."""
    import ctypes
    import ctypes.util

    POSIX_FADV_WILLNEED = 3
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            libc.posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED)
            return size
        finally:
            os.close(fd)
    except OSError as e:
        logger.warning_rank0(f"model prefetch skipped for {path}: {e!r}")
        return 0


def prefetch_enabled() -> bool:
    return os.getenv("FREETOKEN_PREFETCH_MODEL", "1") in {"1", "true", "yes", "on"}


class KeepResident(threading.Thread):
    """Daemon that re-advises WILLNEED on the model file on an interval.

    FADV_WILLNEED only reads pages that are not already resident, so the steady
    state costs no disk IO; it exists to undo eviction pressure from other
    tenants of the machine between requests. Interval 0 disables the loop (a
    single boot-time prefetch still ran).
    """

    def __init__(self, path: str, interval_s: float):
        super().__init__(daemon=True, name="model-keep-resident")
        self._path = path
        self._interval = interval_s
        self._halt = threading.Event()

    def run(self) -> None:
        while self._interval > 0 and not self._halt.wait(self._interval):
            prefetch_model_file(self._path)

    def stop(self) -> None:
        self._halt.set()


def start_prefetch(path: str) -> KeepResident | None:
    """Boot-time entry point: prefetch once, then keep resident. None if disabled."""
    if not prefetch_enabled():
        return None
    size = prefetch_model_file(path)
    if size:
        logger.info_rank0(
            f"model prefetch: advised WILLNEED for {size / (1 << 30):.1f} GiB of "
            f"{os.path.basename(path)} (page cache fills in the background)"
        )
    interval = float(os.getenv("FREETOKEN_KEEP_RESIDENT_S", "120") or 0)
    keeper = KeepResident(path, interval)
    keeper.start()
    return keeper
