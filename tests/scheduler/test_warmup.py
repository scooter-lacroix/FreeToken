"""Unit coverage for the boot-time prefill-shape warmup.

The ladder/env logic is pure and tested directly; the Scheduler-side driver is
exercised against a shell (no Engine, no GPU) to pin the scrub-on-failure and
env-gate contracts.
"""

from __future__ import annotations

from types import SimpleNamespace

import time

import pytest

from freetoken.scheduler.warmup import (
    prefill_warmup_lengths,
    warmup_depth,
    warmup_enabled,
)


def test_ladder_covers_small_lengths_and_divisibility_classes():
    lengths = prefill_warmup_lengths(1024)
    # every integer-divisibility class 1..16 is present exactly once
    assert lengths[:16] == list(range(1, 17))
    # boundaries +/-1 around the GDN chunk size and powers of two are present
    # (the +1 shoulder of max_extend itself is capped away by design)
    for base in (64, 128, 256, 512):
        assert base - 1 in lengths and base in lengths and base + 1 in lengths
    assert 1023 in lengths and 1024 in lengths and 1025 not in lengths
    # max_extend itself (the chunked-prefill steady shape) is included
    assert 1024 in lengths
    # sorted, deduped, in range
    assert lengths == sorted(set(lengths))
    assert all(1 <= n <= 1024 for n in lengths)


def test_ladder_stops_at_max_extend_and_handles_degenerate():
    lengths = prefill_warmup_lengths(8)
    assert lengths == list(range(1, 9))
    assert prefill_warmup_lengths(0) == []
    # a max_extend between boundary +/-1 shoulders keeps everything <= max_extend
    lengths = prefill_warmup_lengths(63)
    assert lengths[-1] == 63
    assert 64 not in lengths


def test_env_gates(monkeypatch):
    assert warmup_enabled() is True  # default on
    monkeypatch.setenv("FREETOKEN_PREFILL_WARMUP", "0")
    assert warmup_enabled() is False
    monkeypatch.setenv("FREETOKEN_PREFILL_WARMUP", "true")
    assert warmup_enabled() is True

    monkeypatch.delenv("FREETOKEN_WARMUP_MAX_DEPTH", raising=False)
    # unset env caps at the shared-box-safe default, never the full window
    assert warmup_depth(73728) == 40960
    assert warmup_depth(8192) == 8192  # never exceeds max_seq_len
    monkeypatch.setenv("FREETOKEN_WARMUP_MAX_DEPTH", "16384")
    assert warmup_depth(73728) == 16384
    monkeypatch.setenv("FREETOKEN_WARMUP_MAX_DEPTH", "73720")
    assert warmup_depth(73728) == 73720  # explicit opt-in reaches full depth


def _warmup_scheduler_shell(monkeypatch, enabled: str, fail: bool = False):
    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setenv("FREETOKEN_PREFILL_WARMUP", enabled)
    calls = {"submit": 0, "loop": 0}

    calls["lengths"] = []

    def fake_submit(uid, n_tokens):
        calls["submit"] += 1
        calls["lengths"].append(n_tokens)
        if fail:
            raise RuntimeError("synthetic warmup failure")

    sched = Scheduler.__new__(Scheduler)
    sched.config = SimpleNamespace(max_extend_tokens=1024)
    sched.prefill_manager = SimpleNamespace(
        runnable=False, pending_list=[], add_one_req=lambda msg: None
    )
    sched.decode_manager = SimpleNamespace(runnable=False, running_reqs=set())
    sched._submit_warmup_req = fake_submit
    sched._run_warmup_loop = lambda max_iters, desc=None: calls.__setitem__(
        "loop", calls["loop"] + 1
    )
    sched.engine_stream_ctx = _null_ctx()
    sched.engine = SimpleNamespace(
        max_seq_len=73728, stream=SimpleNamespace(wait_stream=lambda s: None)
    )
    sched.stream = None
    return sched, calls


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_scheduler_warmup_disabled_by_env(monkeypatch):
    sched, calls = _warmup_scheduler_shell(monkeypatch, enabled="0")
    sched._prefill_warmup()
    assert calls["submit"] == 0 and calls["loop"] == 0


def test_scheduler_warmup_runs_ladder_and_depth(monkeypatch):
    sched, calls = _warmup_scheduler_shell(monkeypatch, enabled="1")
    sched._prefill_warmup()
    # one submit+drain per ladder entry, plus the single depth-walk request
    assert calls["submit"] == len(prefill_warmup_lengths(1024)) + 1
    assert calls["loop"] == calls["submit"]


def test_scheduler_warmup_failure_is_contained(monkeypatch):
    sched, calls = _warmup_scheduler_shell(monkeypatch, enabled="1", fail=True)
    # must not raise: serving continues after a warmup failure
    sched._prefill_warmup()
    # the depth walk never launched after the ladder's first submit raised
    assert calls["submit"] == 1


def test_staging_serves_from_ssd_copy(monkeypatch, tmp_path):
    from freetoken.utils import residency

    origin = tmp_path / "origin" / "tiny.gguf"
    origin.parent.mkdir()
    origin.write_bytes(b"x" * 8192)
    stage_dir = tmp_path / "stage"
    monkeypatch.setenv("FREETOKEN_STAGING_DIR", str(stage_dir))

    # first call provisions the copy and serves from it
    served = residency.resolve_staged_model(str(origin))
    assert served == str(stage_dir / "tiny.gguf")
    assert (stage_dir / "tiny.gguf").stat().st_size == 8192
    assert not (stage_dir / "tiny.gguf.part").exists()  # atomic rename, no temp litter
    # second call sees a fresh copy (size+mtime match) and skips the recopy
    assert residency.resolve_staged_model(str(origin)) == served

    # drift (size change) triggers a refresh
    origin.write_bytes(b"y" * 4096)
    assert residency.resolve_staged_model(str(origin)) == served
    assert (stage_dir / "tiny.gguf").stat().st_size == 4096


def test_staging_falls_back_to_origin(monkeypatch, tmp_path):
    from freetoken.utils import residency

    monkeypatch.setenv("FREETOKEN_SSD_STAGING", "0")
    assert residency.resolve_staged_model("/nonexistent/m.gguf") == "/nonexistent/m.gguf"
    monkeypatch.setenv("FREETOKEN_SSD_STAGING", "1")
    monkeypatch.setenv("FREETOKEN_STAGING_DIR", str(tmp_path / "nope"))
    # unreadable origin: staging must never raise
    assert residency.resolve_staged_model("/nonexistent/m.gguf") == "/nonexistent/m.gguf"


def test_residency_pin_unpin_and_idle_release(monkeypatch, tmp_path):
    import mmap as _mmap

    from freetoken.utils import residency

    f = tmp_path / "tiny.gguf"
    f.write_bytes(b"z" * (1 << 20))
    # the mapping must stay open for the thread phase: pinning works on the
    # ranges /proc/self/maps reports for THIS process (the server keeps its
    # GGUF mapped for the process lifetime; the test mirrors that here).
    fh = open(f, "r+b")
    mm = _mmap.mmap(fh.fileno(), 0)
    try:
        ranges = residency.mapping_ranges(str(f))
        assert ranges, "test mapping must appear in /proc/self/maps"
        assert residency.pin_model(str(f)) >= mm.size()
        assert residency.unpin_model(str(f)) >= mm.size()

        # the activity loop: pins while activity_fn is true, releases after grace
        flag = {"active": False}
        res = residency.ActivityResidency(
            str(f), lambda: flag["active"], grace_s=0.2,
            poll_active_s=0.1, poll_idle_s=0.1,
        )
        res.start()
        flag["active"] = True
        deadline = time.monotonic() + 5
        while not res._pinned and time.monotonic() < deadline:
            time.sleep(0.05)
        assert res._pinned, "residency must pin while active"
        flag["active"] = False
        deadline = time.monotonic() + 5
        while res._pinned and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not res._pinned, "residency must release after the idle grace"
        res.stop()
        res.join(timeout=2)
    finally:
        mm.close()
        fh.close()


def test_warmup_walk_respects_seq_len_boundary(monkeypatch):
    """The walk ends with 2 decode steps; a request decoded AT position
    max_seq_len indexes one past max_seq_len-sized buffers (rope cache) and
    faults the GPU with an async HSA exception. Warmup requests bypass the
    admission guard, so the walk length itself must carry the headroom."""
    sched, calls = _warmup_scheduler_shell(monkeypatch, enabled="1")
    monkeypatch.setenv("FREETOKEN_WARMUP_MAX_DEPTH", "73728")
    sched._prefill_warmup()
    walk_len = max(calls["lengths"])
    assert walk_len <= sched.engine.max_seq_len - 4
    # env asks for more than the cap allows -> clamped
    assert walk_len == sched.engine.max_seq_len - 4


def test_warmup_walk_clamped_when_env_below_cap(monkeypatch):
    sched, calls = _warmup_scheduler_shell(monkeypatch, enabled="1")
    monkeypatch.setenv("FREETOKEN_WARMUP_MAX_DEPTH", "40960")
    sched._prefill_warmup()
    walk_len = max(calls["lengths"])
    assert walk_len == 40960


def test_residency_yields_under_memory_pressure(monkeypatch, tmp_path):
    import mmap as _mmap

    from freetoken.utils import residency

    f = tmp_path / "tiny.gguf"
    f.write_bytes(b"z" * (1 << 20))
    fh = open(f, "r+b")
    mm = _mmap.mmap(fh.fileno(), 0)
    try:
        avail = {"gb": 40.0}
        monkeypatch.setattr(residency, "mem_available_gb", lambda: avail["gb"])
        flag = {"active": True}
        res = residency.ActivityResidency(
            str(f), lambda: flag["active"], grace_s=300.0,
            poll_active_s=0.05, poll_idle_s=0.05,
        )
        res.start()
        deadline = time.monotonic() + 5
        while not res._pinned and time.monotonic() < deadline:
            time.sleep(0.02)
        assert res._pinned, "pins while healthy"

        # pressure: availability crashes below low-water -> pins released
        avail["gb"] = 4.0
        deadline = time.monotonic() + 5
        while res._pinned and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not res._pinned, "pressure must release pins even while serving"

        # hysteresis: still serving, availability recovers past high-water -> re-pin
        avail["gb"] = 40.0
        deadline = time.monotonic() + 5
        while not res._pinned and time.monotonic() < deadline:
            time.sleep(0.02)
        assert res._pinned, "re-pins after headroom recovers"
        res.stop()
        res.join(timeout=2)
    finally:
        mm.close()
        fh.close()
