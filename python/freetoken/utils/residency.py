"""Activity-gated model residency: SSD staging while serving, silence while idle.

Contract (user-directed, 2026-08-29): while inference runs, FreeToken may use
the machine aggressively -- serve the checkpoint from the fastest SSD and pin
its pages so requests never stall behind desktop disk IO. When idle it must be
a non-hog: no background IO, no CPU, and pinned memory returned to the page
cache for the desktop.

Two layers:

- **SSD staging** (``resolve_staged_model``, boot-time): the checkpoint is
  served from a copy on the fastest SSD with room, so page refills never
  touch the spinning origin even after the page cache has been drained by
  desktop work (observed: cache 24GB -> 5.6GB under video/desktop IO, refills
  faulting 81MB at 0.6MB/s behind it = the 135s / 116s stalls). One
  sequential HDD->SSD copy at boot; refreshed only when size/mtime drifts.
  All failures fall back to the origin path -- staging must never block
  serving.

- **Page pinning** (``ActivityResidency``): while a request is in flight (or
  within a grace window after the last one), the worker applies
  ``mlock2(MLOCK_ONFAULT)`` to its mapping of the checkpoint: resident pages
  become unevictable, and pages fault in on demand and pin as they arrive --
  no synchronous mass-fault. Past the grace window the thread ``munlock``s
  and goes fully quiet: pages return to normal reclaimable cache, zero IO,
  zero CPU beyond a slow idle poll.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import threading
import time

from . import init_logger

logger = init_logger(__name__)

_MLOCK_ONFAULT = 0x01

_STAGING_DIR_DEFAULTS = (
    "/mnt/WD-SSD/.mlstack/model-staging",
    os.path.expanduser("~/.cache/mlstack/model-staging"),
)

libc = None


def _libc():
    global libc
    if libc is None:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    return libc


def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default) in {"1", "true", "yes", "on"}


def mapping_ranges(path: str) -> list[tuple[int, int]]:
    """(start, end) virtual ranges this process has mmapped from ``path``."""
    real = os.path.realpath(path)
    ranges = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split(maxsplit=5)
                if len(parts) == 6 and parts[5].strip() == real:
                    lo, hi = parts[0].split("-")
                    ranges.append((int(lo, 16), int(hi, 16)))
    except OSError:
        pass
    return ranges


def _mlock2(start: int, length: int, onfault: bool) -> bool:
    lc = _libc()
    fn = getattr(lc, "mlock2", None)
    if fn is None:
        # kernel/glibc without mlock2: plain mlock would synchronously fault
        # the whole range (a self-inflicted stall) -- refuse rather than stall.
        return False
    return fn(ctypes.c_void_p(start), ctypes.c_size_t(length), _MLOCK_ONFAULT if onfault else 0) == 0


def _munlock(start: int, length: int) -> bool:
    return _libc().munlock(ctypes.c_void_p(start), ctypes.c_size_t(length)) == 0


def pin_model(path: str) -> int:
    """mlock2(MLOCK_ONFAULT) every mapping of ``path``; returns bytes pinned."""
    pinned = 0
    for start, end in mapping_ranges(path):
        if _mlock2(start, end - start, onfault=True):
            pinned += end - start
    return pinned


def unpin_model(path: str) -> int:
    """munlock every mapping of ``path``; returns bytes released."""
    released = 0
    for start, end in mapping_ranges(path):
        if _munlock(start, end - start):
            released += end - start
    return released


def prefetch_model_file(path: str) -> int:
    """Asynchronously advise the kernel to populate the page cache for ``path``."""
    POSIX_FADV_WILLNEED = 3
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            _libc().posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED)
            return size
        finally:
            os.close(fd)
    except OSError as e:
        logger.warning_rank0(f"page-cache advice skipped for {path}: {e!r}")
        return 0


# --------------------------------------------------------------------------
# SSD staging
# --------------------------------------------------------------------------

def _staging_candidates() -> list[str]:
    env_dir = os.getenv("FREETOKEN_STAGING_DIR", "").strip()
    if env_dir:
        return [env_dir]
    return list(_STAGING_DIR_DEFAULTS)


def _copy_progress(desc: str, done: int, total: int) -> None:
    from .progress import emit_progress

    emit_progress(desc, done, total)


def resolve_staged_model(origin: str) -> str:
    """Serve ``origin`` from an SSD copy when one can be provisioned now.

    Boot-time only (called before the scheduler exists): the initial copy is a
    sequential read the boot window already pays for via warmup; refreshes
    happen only when the origin's size or mtime drifts. Returns the path to
    serve from -- never raises; any failure returns ``origin``.
    """
    if not _env_on("FREETOKEN_SSD_STAGING", "1"):
        return origin
    try:
        st = os.stat(origin)
        size = st.st_size
    except OSError as e:
        logger.warning_rank0(f"staging skipped, origin unreadable: {e!r}")
        return origin

    for cand in _staging_candidates():
        try:
            if os.path.realpath(cand) == os.path.realpath(os.path.dirname(origin)):
                return origin  # already on this filesystem
            os.makedirs(cand, exist_ok=True)
            du = shutil.disk_usage(cand)
            if du.free < size * 1.1 + (4 << 30):
                continue
            staged = os.path.join(cand, os.path.basename(origin))
            if os.path.realpath(staged) == os.path.realpath(origin):
                return origin
            if (
                os.path.isfile(staged)
                and os.stat(staged).st_size == size
                and os.stat(staged).st_mtime_ns == st.st_mtime_ns
            ):
                logger.info_rank0(
                    f"model staging: serving {size / (1 << 30):.1f} GiB from SSD copy "
                    f"{staged}"
                )
                return staged
            t0 = time.perf_counter()
            tmp = staged + ".part"
            logger.info_rank0(
                f"model staging: copying {size / (1 << 30):.1f} GiB to SSD ({staged}); "
                "origin stays authoritative, one-time boot cost"
            )
            _copy_progress("staging model to SSD", 0, 1)
            with open(origin, "rb") as src, open(tmp, "wb") as dst:
                copied = 0
                last = 0
                while chunk := src.read(32 << 20):
                    dst.write(chunk)
                    copied += len(chunk)
                    now = time.perf_counter()
                    if now - last > 5.0:
                        last = now
                        _copy_progress("staging model to SSD", copied, size)
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(tmp, staged)
            shutil.copystat(origin, staged)
            _copy_progress("staging model to SSD", 1, 1)
            logger.info_rank0(
                f"model staging: copy done in {time.perf_counter() - t0:.0f}s; "
                f"serving from {staged}"
            )
            return staged
        except OSError as e:
            logger.warning_rank0(f"staging candidate {cand} unusable: {e!r}")
            continue
    logger.info_rank0("model staging: no SSD candidate with space; serving from origin")
    return origin


# --------------------------------------------------------------------------
# Activity-gated residency
# --------------------------------------------------------------------------

class ActivityResidency(threading.Thread):
    """Pin the checkpoint's pages while serving; release and go quiet when idle.

    ACTIVE means: ``activity_fn()`` true now, or it was true within ``grace_s``
    (bridges the natural pauses between a user's requests so mid-session pages
    stay put). Entering ACTIVE: mlock2(MLOCK_ONFAULT) the mapping -- resident
    pages become unevictable, future faults pin as they arrive -- and advise
    WILLNEED once so the refill starts immediately. LEAVING ACTIVE: munlock
    (memory returns to the desktop) and stop touching the disk entirely.
    """

    def __init__(self, path: str, activity_fn, grace_s: float,
                 poll_active_s: float = 5.0, poll_idle_s: float = 30.0):
        super().__init__(daemon=True, name="model-residency")
        self._path = path
        self._activity_fn = activity_fn
        self._grace = grace_s
        self._poll_active = poll_active_s
        self._poll_idle = poll_idle_s
        self._halt = threading.Event()
        self._pinned = False

    def run(self) -> None:
        last_active = time.monotonic()  # boot counts as active (warmup pins)
        while not self._halt.wait(self._poll_active if self._pinned else self._poll_idle):
            try:
                if self._activity_fn():
                    last_active = time.monotonic()
                active = (time.monotonic() - last_active) < self._grace
                if active and not self._pinned:
                    n = pin_model(self._path)
                    self._pinned = n > 0
                    prefetch_model_file(self._path)
                    if self._pinned:
                        logger.info_rank0(
                            f"residency: serving-active, pinned {n / (1 << 30):.1f} GiB "
                            "of checkpoint pages (MLOCK_ONFAULT; refills pin as they "
                            "fault in)"
                        )
                elif not active and self._pinned:
                    n = unpin_model(self._path)
                    self._pinned = n == 0 and bool(mapping_ranges(self._path))
                    if n:
                        logger.info_rank0(
                            f"residency: idle past {self._grace:.0f}s grace, released "
                            f"{n / (1 << 30):.1f} GiB back to the page cache; no "
                            "background IO while idle"
                        )
            except Exception as e:  # residency must never kill the worker
                logger.warning_rank0(f"residency error (continuing): {e!r}")

    def stop(self) -> None:
        self._halt.set()


def start_residency(path: str, activity_fn) -> ActivityResidency | None:
    """Entry point. ``activity_fn`` returns True while requests are in flight."""
    if not _env_on("FREETOKEN_RESIDENCY", "1"):
        return None
    grace = float(os.getenv("FREETOKEN_IDLE_RELEASE_S", "300") or 0)
    res = ActivityResidency(path, activity_fn, grace)
    res.start()
    return res
