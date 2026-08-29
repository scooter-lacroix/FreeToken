"""Unit coverage for the boot-time prefill-shape warmup.

The ladder/env logic is pure and tested directly; the Scheduler-side driver is
exercised against a shell (no Engine, no GPU) to pin the scrub-on-failure and
env-gate contracts.
"""

from __future__ import annotations

from types import SimpleNamespace

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
    assert warmup_depth(73728) == 73728
    monkeypatch.setenv("FREETOKEN_WARMUP_MAX_DEPTH", "16384")
    assert warmup_depth(73728) == 16384
    assert warmup_depth(8192) == 8192  # never exceeds max_seq_len


def _warmup_scheduler_shell(monkeypatch, enabled: str, fail: bool = False):
    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setenv("FREETOKEN_PREFILL_WARMUP", enabled)
    calls = {"submit": 0, "loop": 0}

    def fake_submit(uid, n_tokens):
        calls["submit"] += 1
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


def test_prefetch_env_gate_and_best_effort(monkeypatch, tmp_path):
    from freetoken.utils import prefetch

    monkeypatch.setenv("FREETOKEN_PREFETCH_MODEL", "0")
    assert prefetch.prefetch_enabled() is False
    assert prefetch.start_prefetch("/nonexistent/model.gguf") is None

    monkeypatch.setenv("FREETOKEN_PREFETCH_MODEL", "1")
    assert prefetch.prefetch_enabled() is True
    # missing file: best-effort, logs and returns without raising
    f = tmp_path / "tiny.gguf"
    f.write_bytes(b"x" * 4096)
    size = prefetch.prefetch_model_file(str(f))
    assert size == 4096
    monkeypatch.setenv("FREETOKEN_KEEP_RESIDENT_S", "0")
    keeper = prefetch.start_prefetch(str(f))
    assert keeper is not None
    keeper.stop()
    keeper.join(timeout=2)
    assert not keeper.is_alive()
