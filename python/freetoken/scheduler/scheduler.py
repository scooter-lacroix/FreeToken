from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from freetoken.utils.progress import emit_progress

from .warmup import prefill_warmup_lengths, warmup_depth, warmup_enabled
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


_sched_pin_keepalive: list = []


def _pinned_to_device(t: torch.Tensor, device) -> torch.Tensor:
    """Async H2D of a pinned HOST tensor, with the pinned source KEPT ALIVE.

    Handing a function-local pinned tensor to ``.to(device, non_blocking=True)``
    frees the pinned pages at scope exit while the async copy engine still
    reads them -- GPU "Page not present" faults on host addresses,
    timing-dependent (masked under launch serialization). Retain recent
    sources; the scheduler loop syncs every iteration so the ring drains."""
    out = t.to(device, non_blocking=True)
    _sched_pin_keepalive.append(t)
    if len(_sched_pin_keepalive) > 128:
        del _sched_pin_keepalive[:64]
    return out


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        import os as _os_spec_stream

        if int(_os_spec_stream.environ.get("FREETOKEN_SPEC_K", "0") or 0) > 0:
            # Spec mode: ONE stream for everything. The verify graph is only
            # stable when capture, replay, and ALL other GPU work (prefill and
            # decode forwards included) share a single stream -- cross-stream
            # work while it is live surfaces as all-NaN replays or hard GPU
            # page faults (measured both). The scheduling-overlap pipelining
            # this second stream provided is not worth a wedged graph.
            self.stream = self.engine.stream
        else:
            self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # The verify graph captured during Engine init (and its post-warmup
        # re-capture) lives on the stream that was current THERE (the default
        # stream). Replaying it on the scheduler's stream is a cross-stream
        # replay -- measured as an all-NaN-or-hard-fault class on this ROCm
        # stack. Re-capture on OUR stream so capture and replay coincide.
        _vr_boot = getattr(self.engine.graph_runner, "verify_runner", None)
        if _vr_boot is not None:
            _pool_boot = self.engine.linear_state_pool
            if _pool_boot is not None:
                _ps = _pool_boot.padding_slot
                _pool_boot.conv_states[:, _ps].zero_()
                _pool_boot.recurrent_states[:, _ps].zero_()
            self.engine.graph_runner._capture_verify(
                _vr_boot.k, self.engine.model, stream=self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        import os as _pos

        _prof_n = int(_pos.environ.get("FREETOKEN_PROFILE_STEPS", "0") or 0)
        _real_decode = (
            not self.prefill_manager.runnable
            and len(self.decode_manager.running_reqs) == 1
            and next(iter(self.decode_manager.running_reqs)).uid < self.WARMUP_UID_BASE
        )
        if _prof_n and _real_decode and not getattr(self, "_prof_started", False):
            try:
                from torch.profiler import ProfilerActivity, profile

                self._prof = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                self._prof_started = True
                self._prof_iter = 0
                self._prof_max = _prof_n
                self._prof.__enter__()
                print(f"[prof] profiling next {_prof_n} iterations", flush=True)
            except Exception as e:                     # noqa: BLE001
                print(f"[prof] failed to start: {e!r}", flush=True)
                self._prof_started = True
                self._prof_max = 0
        if getattr(self, "_prof_started", False) and getattr(self, "_prof_max", 0):
            self._prof_iter = getattr(self, "_prof_iter", 0) + 1
            if self._prof_iter > self._prof_max:
                self._prof_max = 0
                try:
                    self._prof.__exit__(None, None, None)
                    out = "/home/scooter/Documents/Product/Stan-s-ML-Stack/Fork/FreeToken/logs/decode_step_trace.json"
                    self._prof.export_chrome_trace(out)
                    print(f"[prof] trace saved: {out}", flush=True)
                except Exception as e:                 # noqa: BLE001
                    print(f"[prof] export failed: {e!r}", flush=True)
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        import os as _tos
        import time as _tt

        _lt = _tos.environ.get("FREETOKEN_LOOP_TIMING", "0") == "1"
        if _lt:
            _t0 = _tt.perf_counter()
        if _lt:
            self._lt_msgs = getattr(self, "_lt_msgs", 0.0) + _tt.perf_counter() - _t0
            _ta = _tt.perf_counter()
        if self._spec_step():
            self._flush_abort_acks()
            return

        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if _lt:
            self._lt_pick = getattr(self, "_lt_pick", 0.0) + _tt.perf_counter() - _ta
            _tb = _tt.perf_counter()
        if batch is not None and _tos.environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
            print(f"[phase] prepare phase={batch.phase} "
                  f"n={len(batch.reqs)} uid={batch.reqs[0].uid} "
                  f"cached={batch.reqs[0].cached_len}", flush=True)
        forward_input = self._prepare_batch(batch) if batch is not None else None
        if _lt:
            self._lt_prep = getattr(self, "_lt_prep", 0.0) + _tt.perf_counter() - _tb
            self._lt_sched = getattr(self, "_lt_sched", 0.0) + _tt.perf_counter() - _t0
            _t1 = _tt.perf_counter()
        # Spin diagnostic: runnable managers + a None schedule + non-blocking
        # receive = a hot retry loop (the 100%-CPU signature). Name the gate
        # once after a few iterations instead of spinning silently.
        if forward_input is None and (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            import os as _os

            n = getattr(self, "_spin_count", 0) + 1
            self._spin_count = n
            if n == 50:
                print(
                    "[sched-diag] SPIN: runnable but unschedulable — "
                    f"prefill.runnable={self.prefill_manager.runnable} "
                    f"decode.runnable={self.decode_manager.runnable} "
                    f"runnable_reqs={[getattr(r, 'input_len', '?') for r in self.prefill_manager.runnable_reqs] if hasattr(self.prefill_manager, 'runnable_reqs') else 'n/a'} "
                    f"mamba_free={getattr(self.cache_manager.linear_state_pool, 'num_free_slots', '?')}",
                    flush=True,
                )
            if n == 500:
                import faulthandler as _fh

                _fh.dump_traceback()
                self._spin_count = 50  # keep reporting every 450 iters
        else:
            if getattr(self, "_spin_count", 0):
                print(f"[sched-diag] spin ended at n={self._spin_count}", flush=True)
            self._spin_count = 0
        ongoing_data = None
        if _lt:
            _t2 = _tt.perf_counter()
        if forward_input is not None:
            # a normal (non-spec) forward ran: the verify graph's external
            # state may be invalidated by the decode/prefill machinery (fault
            # at the next replay, measured at request boundaries) -- mark it
            # so the next spec step re-captures before replaying
            self._spec_graph_dirty = True
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._fwd_timed(forward_input))
            if _lt:
                self._lt_fwd = getattr(self, "_lt_fwd", 0.0) + _tt.perf_counter() - _t1
                _t2 = _tt.perf_counter()

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        if _lt:
            self._lt_drain = getattr(self, "_lt_drain", 0.0) + _tt.perf_counter() - _t2
            self._lt_n = getattr(self, "_lt_n", 0) + 1
            if self._lt_n % 200 == 0:
                n = self._lt_n
                print(f"[loop-timing] n={n} msgs={self._lt_msgs/n*1000:.1f}ms "
                      f"pick={self._lt_pick/n*1000:.1f}ms prep={self._lt_prep/n*1000:.1f}ms "
                      f"fwd={self._lt_fwd/n*1000:.1f}ms drain={self._lt_drain/n*1000:.1f}ms",
                      flush=True)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        if self._spec_step():
            self._flush_abort_acks()
            return

        forward_input = self._schedule_next_batch()
        # Spin diagnostic: runnable managers + a None schedule + non-blocking
        # receive = a hot retry loop (the 100%-CPU signature). Name the gate
        # once after a few iterations instead of spinning silently.
        if forward_input is None and (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            import os as _os

            n = getattr(self, "_spin_count", 0) + 1
            self._spin_count = n
            if n == 50:
                print(
                    "[sched-diag] SPIN: runnable but unschedulable — "
                    f"prefill.runnable={self.prefill_manager.runnable} "
                    f"decode.runnable={self.decode_manager.runnable} "
                    f"runnable_reqs={[getattr(r, 'input_len', '?') for r in self.prefill_manager.runnable_reqs] if hasattr(self.prefill_manager, 'runnable_reqs') else 'n/a'} "
                    f"mamba_free={getattr(self.cache_manager.linear_state_pool, 'num_free_slots', '?')}",
                    flush=True,
                )
            if n == 500:
                import faulthandler as _fh

                _fh.dump_traceback()
                self._spin_count = 50  # keep reporting every 450 iters
        else:
            if getattr(self, "_spin_count", 0):
                print(f"[sched-diag] spin ended at n={self._spin_count}", flush=True)
            self._spin_count = 0
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    # uids at or above this base belong to warmup requests; they can never collide with the
    # frontend's per-process counter and are scrubbed from scheduler state on warmup failure.
    # ---------------------------------------------------------------- S4 spec decode
    def _spec_step(self) -> bool:
        """One DFlash2 speculative verify step for the single greedy running request
        (FREETOKEN_SPEC_K): extend forward over z = [anchor, picks[1..k-1]], per-row
        argmax longest-prefix accept + bonus, KV crop, GDN state restore from the
        per-position log on partial accept, all committed tokens detokenized."""
        import os as _os

        k = int(_os.environ.get("FREETOKEN_SPEC_K", "0") or 0)
        if k <= 0:
            return False
        running = self.decode_manager.running_reqs
        n_real = sum(1 for r in running if r.uid < self.WARMUP_UID_BASE)
        has_warmup = len(running) - n_real > 0
        if not (
            k > 0
            and self.cache_manager.is_hybrid
            and not self.prefill_manager.runnable
            and n_real == 1
            and not has_warmup  # a co-resident warmup req would starve behind spec steps
        ):
            if k > 0 and self.cache_manager.is_hybrid and not self.prefill_manager.runnable:
                print(f"[spec-gate] decline: n_real={n_real} warmup={has_warmup} "
                      f"failed={getattr(self, '_spec_failed', False)}", flush=True)
            return False
        req = next(r for r in running if r.uid < self.WARMUP_UID_BASE)
        if req.table_idx == -1 or req.aborted:
            return False
        sp = req.sampling_params
        if (sp.temperature or 0.0) != 0.0 or req.linear_slot_idx is None:
            return False
        if getattr(self, "_spec_failed", False):
            return False
        try:
            return self._spec_step_inner(req, k)
        except Exception as e:                              # noqa: BLE001
            import traceback as _tb

            print("[spec] DISABLED after error: " + repr(e) + "\n"
                  + "".join(_tb.format_exception(e)), flush=True)
            self._spec_failed = True
            return False

    def _spec_replay_step(self, req, vr, st, gctx, z, z_dev, L, kn, slot):
        """Graphed verify: allocate the block's pages, stage the capture buffers,
        ONE replay, consume in-stream (argmax/topk), then the shared accept/
        crop/restore/emit/propose tail."""
        import time as _time

        import torch as _t

        pool = self.cache_manager.linear_state_pool
        vreq_device_len = L + kn
        # page allocation via a minimal transient (host bookkeeping only)
        from freetoken.core import Req

        # The transient must carry L+kn host ids so allocate_paged reserves the
        # WHOLE block (req.input_ids holds only L..L+1; slicing it short silently
        # allocated 1 page while the replay writes k rows).
        _ids = req.input_ids
        if _ids.numel() < vreq_device_len:
            _ids = _t.cat([_ids, _t.zeros(vreq_device_len - _ids.numel(), dtype=_t.int32)])
        _alloc_req = Req(input_ids=_ids, table_idx=req.table_idx,
                         cached_len=L, output_len=0, uid=req.uid + self.WARMUP_UID_BASE,
                         sampling_params=req.sampling_params, cache_handle=req.cache_handle)
        _t_ap0 = _time.perf_counter()
        self.cache_manager.allocate_paged([_alloc_req])
        self._t_pre_alloc = _time.perf_counter()
        self._spec_t_allocpg = getattr(self, "_spec_t_allocpg", 0.0) + (
            _time.perf_counter() - _t_ap0)

        # All spec-path GPU work runs on the scheduler's OWN stream
        # (current here): the verify graph is captured AND replayed on it, and
        # only same-stream ops between replays keep it valid -- cross-stream
        # work (engine-stream or otherwise) invalidates the next replay with
        # all-NaN logits on this ROCm stack.
        import time as _tp

        _t_alloc0 = _tp.perf_counter()
        self._spec_entry_snap = pool.snapshot_slot(slot)
        self._spec_t_snap = getattr(self, "_spec_t_snap", 0.0) + (
            _tp.perf_counter() - _t_alloc0)

        # ENTRY-state fingerprints (pre-replay): across a step cycle the
        # PREFIX pages [0:L), the GDN slot bytes, and tokpool[0:L] must be
        # IDENTICAL at entry (same logical context); any view whose ENTRY
        # bytes change across a cycle names the between-steps writer.
        import os as _os_efp

        if _os_efp.environ.get("FREETOKEN_SPEC_AB", "0") in {"1", "true", "yes"}:
            import hashlib

            def _h(_t):
                return hashlib.md5(
                    _t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()[:10]

            _efp = {
                "prefix_pages": _h(self.cache_manager.page_table[req.table_idx][:L]),
                "slot_entry": _h(pool.recurrent_states[:, slot]) + "/" + _h(
                    pool.conv_states[:, slot]),
                "tokpool_prefix": _h(self.token_pool[req.table_idx][:L]),
            }
            _pefp = getattr(self, "_efp_prev", None)
            if _pefp is not None and _pefp[1] == L:
                _chg = [k for k in _efp if _efp[k] != _pefp[0][k]]
                print(f"[EFP] L={L} entry-view changes vs prev cycle: {_chg} "
                      f"now={_efp}", flush=True)
            elif _pefp is not None:
                print(f"[EFP] L={L} (prev L={_pefp[1]} — prefix grew, expected)",
                      flush=True)
            self._efp_prev = (_efp, L)

        # HOST-PULL keep-alive (last coherence candidate): every coherent
        # probe boot D2H-synced the GDN slot and page-table row each step via
        # fingerprint host pulls; production never did. Dummy launches, device
        # syncs, eager passes, and double replays are all falsified -- this is
        # the remaining delta. Gated FREETOKEN_SPEC_HOSTPULL (default ON for
        # the experiment; ~1-2ms/step).
        import os as _os_hp

        if _os_hp.environ.get("FREETOKEN_SPEC_HOSTPULL", "0") in {"1", "true", "yes"}:
            _ = pool.recurrent_states[:, slot].detach().cpu().contiguous().view(
                torch.uint8)
            _ = self.cache_manager.page_table[req.table_idx, : L + kn
                                              ].detach().cpu().contiguous().view(
                torch.uint8)

        _t1 = _time.perf_counter()
        _gr = self.engine.graph_runner
        import os as _os_ph

        if _os_ph.environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
            print(f"[phase] spec-replay uid={req.uid} L={L}", flush=True)
        import os as _os_ab

        if (_os_ab.environ.get("FREETOKEN_SPEC_AB", "0") in {"1", "true", "yes"}
                and getattr(self, "_spec_n_ab", 0) < 1):
            # Decode-path invalidator bisect at the natural fault point (the
            # first spec step after another request's normal decode run).
            # Order matters: first NaN poisons the pool for later probes.
            self._spec_n_ab = getattr(self, "_spec_n_ab", 0) + 1
            entry_ab = self._spec_entry_snap
            _pr_ab = self.cache_manager.page_table[req.table_idx]
            _dev = self.device

            def _pr():
                with torch.cuda.stream(torch.cuda.current_stream()):
                    vr.replay_step(z_ids=z_dev.to(_dev), slot=slot, L=L,
                                   page_row=_pr_ab,
                                   stream=torch.cuda.current_stream())
                    torch.cuda.current_stream().synchronize()
                return int(vr.nan_out[0])

            def _op_rocblas():
                _a = torch.randn(64, 5120, device=_dev, dtype=torch.bfloat16)
                _b = torch.randn(5120, 256, device=_dev, dtype=torch.bfloat16)
                _c = torch.mm(_a, _b)
                del _a, _b, _c

            def _op_scratch():
                _s1 = torch.empty((1, 64, 128, 128), dtype=torch.float32, device=_dev)
                _s2 = torch.empty((1, 64, 128), dtype=torch.float32, device=_dev)
                del _s1, _s2

            def _op_triton():
                torch.arange(L, L + kn, dtype=torch.int32, device=_dev)

            print(f"[ABP] L={L} slot={slot} (post-decode-traffic)", flush=True)
            for _nm, _op in [("baseline", None), ("consecutive", None),
                             ("triton_arange", _op_triton),
                             ("scratch_alloc", _op_scratch),
                             ("rocblas_mm", _op_rocblas)]:
                pool.restore_slot(slot, entry_ab)
                if _nm == "baseline":
                    pass  # restore + replay directly
                elif _nm == "consecutive":
                    pass  # no op between restore and replay
                else:
                    _op()
                _nan = _pr()
                print(f"[ABP] after {_nm}: nan={_nan}", flush=True)
            pool.restore_slot(slot, entry_ab)
            _pr()

        if getattr(self, "_spec_graph_dirty", False):
            # normal forwards ran since the last replay: re-capture BEFORE
            # replaying (the stale graph hard-faults, which the NaN heal
            # cannot catch)
            print("[spec] normal traffic since last replay -> proactive re-capture",
                  flush=True)
            vr = self._spec_recapture(pool, vr)
            self._spec_graph_dirty = False
        _sched_stream = torch.cuda.current_stream()
        _sub = {}
        for _attempt in range(2):
            with torch.cuda.stream(_sched_stream):
                self._spec_t_stage = getattr(self, "_spec_t_stage", 0.0) + (_time.perf_counter() - _t1)
                _tr0 = _time.perf_counter()
                # REPLAY-TWICE experiment: probe-active boots (extra graph
                # replay between production steps) were coherent while
                # no-probe boots garble -- test whether replay #1 after a
                # gap is wrong and #2 right. First result discarded.
                if _os_ab.environ.get("FREETOKEN_SPEC_2X", "0") in {"1", "true", "yes"}:
                    vr.replay_step(
                        z_ids=z_dev.to(self.device), slot=slot, L=L,
                        page_row=self.cache_manager.page_table[req.table_idx],
                        stream=_sched_stream)
                _z_gpu = z_dev.to(self.device)
                _sub["h2d"] = _sub.get("h2d", 0.0) + _time.perf_counter() - _tr0
                _t_l = _time.perf_counter()
                vr.replay_step(
                    z_ids=_z_gpu, slot=slot, L=L,
                    page_row=self.cache_manager.page_table[req.table_idx],
                    stream=_sched_stream)
                _sub["launch"] = _sub.get("launch", 0.0) + _time.perf_counter() - _t_l
                _t_s = _time.perf_counter()
                _sched_stream.synchronize()
                _sub["sync"] = _sub.get("sync", 0.0) + _time.perf_counter() - _t_s
                _t_n = _time.perf_counter()
                _stale = int(vr.nan_out[0]) > 0
                _sub["nanread"] = _sub.get("nanread", 0.0) + _time.perf_counter() - _t_n
            if not _stale:
                break
            # The captured verify graph was invalidated by eager traffic since
            # its capture (any first-of-shape prefill -- warmup ladders, a new
            # request -- leaves later replays all-NaN; captured-kernel reads of
            # allocator-recycled memory). Self-heal: re-capture with a clean
            # GDN padding slot and replay once more.
            print(f"[spec] verify graph stale (attempt {_attempt}) -> re-capturing",
                  flush=True)
            # the stale replay already wrote through: undo its GDN slot advance
            # so the retry starts from the same entry state
            pool.restore_slot(slot, self._spec_entry_snap)
            _ps = pool.padding_slot
            pool.conv_states[:, _ps].zero_()
            pool.recurrent_states[:, _ps].zero_()
            # capture-time store_kv writes through out_loc: point it at the
            # engine's guard page so a live request's KV rows 0-7 survive
            _dp = int(self.cache_manager.page_table[
                self.engine.dummy_req.table_idx][0].item())
            vr.out_loc.fill_(_dp)
            # free the old graph's private pool before capturing the new one
            # (mid-serving free memory is ~0). NOTE: no torch.cuda.empty_cache()
            # here -- on this ROCm stack it destabilizes live verify-graph
            # replays (the next step goes stale again -> re-capture loop).
            vr = self._spec_recapture(pool, vr)
        with torch.cuda.stream(_sched_stream):
            if int(vr.nan_out[0]) > 0:
                raise RuntimeError(
                    "verify graph NaN after re-capture; falling back to decode")
            _t_a = _time.perf_counter()
            logits = vr.logits_out
            rows = [int(v) for v in logits.argmax(-1).reshape(-1).tolist()]
            _sub["argmax"] = _sub.get("argmax", 0.0) + _time.perf_counter() - _t_a
            _t_k = _time.perf_counter()
            tk = _t.topk(logits.float(), 64, dim=-1)
            _sub["topk"] = _sub.get("topk", 0.0) + _time.perf_counter() - _t_k
            _t_l1 = _time.perf_counter()
            tk_ids = tk.indices.tolist()
            _sub["tolist1"] = _sub.get("tolist1", 0.0) + _time.perf_counter() - _t_l1
            _t_l2 = _time.perf_counter()
            tk_vals = tk.values.tolist()
            _sub["tolist2"] = _sub.get("tolist2", 0.0) + _time.perf_counter() - _t_l2
            self._spec_t_replay = getattr(self, "_spec_t_replay", 0.0) + (_time.perf_counter() - _tr0)
        import os as _os_dump

        if (_os_dump.environ.get("FREETOKEN_SPEC_AB", "0") in {"1", "true", "yes"}
                and getattr(self, "_spec_n_dump", 0) < 12
                and getattr(self, "_spec_n", 0) % 3 == 0):
            # m8 serving dump: the fused kernel's captured launch vs an eager
            # re-run on the SAME (w, x) at the first live replay.
            self._spec_n_dump = getattr(self, "_spec_n_dump", 0) + 1
            import freetoken.models.qwen3_5_moe.ggml_dense as _gd
            from freetoken.kernel.triton.kquant_linear import kq_gemm_q4k_m8

            for _wq, _xq, _yq, _idx in getattr(_gd, "_m8_stash", []) or []:
                _ye = kq_gemm_q4k_m8(_wq, _xq, 12)
                _d = (_yq.float() - _ye.float()).abs().max().item()
                print(f"[m8-dump] step={getattr(self, '_spec_n', 0)} call{_idx}: "
                      f"replay-vs-eager maxdiff={_d:.4f}", flush=True)

        import os as _os_eab

        if (_os_eab.environ.get("FREETOKEN_SPEC_AB", "0") in {"1", "true", "yes"}
                and getattr(self, "_spec_n_r12", 0) < 1):
            # replay #1 vs #2 with IDENTICAL inputs and restored slot: any tap
            # depth that changes between them is pool-carry (the buffer whose
            # value differs while inputs don't is the accumulate-without-init).
            self._spec_n_r12 = 1
            _snap12 = self._spec_entry_snap
            _pr12 = self.cache_manager.page_table[req.table_idx]

            def _tap_snapshot():
                pool.restore_slot(slot, _snap12)
                with torch.cuda.stream(torch.cuda.current_stream()):
                    vr.replay_step(z_ids=z_dev.to(self.device), slot=slot, L=L,
                                   page_row=_pr12,
                                   stream=torch.cuda.current_stream())
                    torch.cuda.current_stream().synchronize()
                return {d: t.clone() for d, t in vr.taps.items()}

            _t1_ = _tap_snapshot()
            _t2_ = _tap_snapshot()
            _t3_ = _tap_snapshot()
            _d12 = {d: round((_t1_[d].float() - _t2_[d].float()).abs().max().item(), 3)
                    for d in _t1_}
            _d23 = {d: round((_t2_[d].float() - _t3_[d].float()).abs().max().item(), 3)
                    for d in _t2_}
            print(f"[R12] r1-vs-r2={_d12}", flush=True)
            print(f"[R12] r2-vs-r3={_d23}", flush=True)
            pool.restore_slot(slot, _snap12)
            with torch.cuda.stream(torch.cuda.current_stream()):
                vr.replay_step(z_ids=z_dev.to(self.device), slot=slot, L=L,
                               page_row=_pr12,
                               stream=torch.cuda.current_stream())
                torch.cuda.current_stream().synchronize()
        import os as _os_eab

        if (_os_eab.environ.get("FREETOKEN_SPEC_AB", "0") in {"1", "true", "yes"}
                and getattr(self, "_spec_n_eab", 0) < 5):
            # Eager-vs-graph rows probe: run the eager verify on the SAME
            # entry state and compare argmax rows. rows==erows while the
            # client stream garbles => GPU path faithful, emission/staging
            # protocol guilty.
            from freetoken.attention.linear import FLAMetadata as _FLA
            from freetoken.core import Batch as _B, Req as _R

            self._spec_n_eab = getattr(self, "_spec_n_eab", 0) + 1
            _no_eager = _os_eab.environ.get(
                "FREETOKEN_SPEC_PUREGRAPH", "0") in {"1", "true", "yes"}
            if _no_eager:
                print(f"[EAB] step={getattr(self, '_spec_n', 0)} L={L} "
                      f"graph_rows={rows[:4]} (PUREGRAPH: eager skipped)",
                      flush=True)
            if not _no_eager:
                _esnap = self._spec_entry_snap
                pool.restore_slot(slot, _esnap)
                _vrq = _R(
                    input_ids=_t.cat([req.input_ids[:L], z_dev]),
                    table_idx=req.table_idx, cached_len=L, output_len=0,
                    uid=req.uid + self.WARMUP_UID_BASE,
                    sampling_params=req.sampling_params, cache_handle=req.cache_handle)
                _vrq.device_len = L + kn
                _vrq.linear_slot_idx = slot
                _vb = _B(reqs=[_vrq], phase="prefill")
                _vb.padded_reqs = _vb.reqs
                _fi = self._prepare_batch(_vb)
                _vb.is_verify = True
                _vb.fla_metadata = _FLA(
                    cu_seqlens=_t.tensor([0, kn], dtype=_t.int32, device=self.device),
                    cache_indices=_t.tensor([slot], dtype=_t.int32, device=self.device),
                    has_initial_state=None, fresh_state_indices=None)
                with self.engine_stream_ctx:
                    self._restore_linear_states(_vb)
                    self._forward(_fi)
                    self.engine.stream.synchronize()
                    _elog = gctx.spec_logits
                    _erows = [int(v) for v in _elog.argmax(-1).reshape(-1).tolist()]
                self.decode_manager.remove_req(_vrq)
                _etaps = gctx.spec_taps or {}
                _tapd = {}
                for _d, _tg in _etaps.items():
                    _vg = vr.taps.get(_d)
                    if _vg is None:
                        continue
                    _tapd[_d] = round((_vg.float() - _tg.float().to(_vg.device))
                                      .abs().max().item(), 3)
                print(f"[EAB] step={getattr(self, '_spec_n', 0)} L={L} "
                      f"graph_rows={rows[:4]} eager_rows={_erows[:4]} "
                      f"match={rows[:4] == _erows[:4]} tapdiff={_tapd}", flush=True)
            # z-ablation (replaces the faulting EABX restage): replay a HYBRID
            # z = [current z[0]] + [prev z[1:]] at the current entry state and
            # compare row-0 against both the production rows and the previous
            # step's rows. row-0 following prev z[1:] => row-0 couples to
            # z[1:] inside the captured block math (chunk-lane leakage);
            # row-0 following production => z[1:] is innocent.
            _pzab = getattr(self, "_zab_prev", None)
            self._zab_prev = (z_dev.clone(), rows[:4])
            if _pzab is not None:
                _pzz, _pzrows = _pzab
                _hy = _t.cat([z_dev[:1], _pzz[1:]])
                pool.restore_slot(slot, self._spec_entry_snap)
                with torch.cuda.stream(torch.cuda.current_stream()):
                    vr.replay_step(z_ids=_hy.to(self.device), slot=slot, L=L,
                                   page_row=self.cache_manager.page_table[
                                       req.table_idx],
                                   stream=torch.cuda.current_stream())
                    torch.cuda.current_stream().synchronize()
                _hrows = [int(v) for v in vr.logits_out.argmax(-1)
                          .reshape(-1).tolist()[:4]]
                _r0 = _hrows[0]
                _verdict = ("COUPLES-TO-PREV" if _r0 == _pzrows[0]
                            else "FOLLOWS-PRODUCTION" if _r0 == rows[0]
                            else "NEITHER")
                print(f"[ZAB] hybrid row0={_r0} prod_row0={rows[0]} "
                      f"prev_row0={_pzrows[0]} -> {_verdict} "
                      f"hybrid_rows={_hrows}", flush=True)
                pool.restore_slot(slot, self._spec_entry_snap)

        _sub["total"] = _sub.get("total", 0.0) + (_time.perf_counter() - _t1)
        _acc = getattr(self, "_spec_sub_acc", None)
        if _acc is None:
            _acc = {}
            self._spec_sub_acc = _acc
        for _kk, _vv in _sub.items():
            _acc[_kk] = _acc.get(_kk, 0.0) + _vv
        self._spec_t_fwd = getattr(self, "_spec_t_fwd", 0.0) + (_time.perf_counter() - _t1)
        self._spec_n_vr = getattr(self, "_spec_n_vr", 0) + 1
        self._spec_graph_dirty = False
        taps = vr.taps

        # Retry-block protocol: on partial accept (a < kn-1) revert the GDN
        # slot to the ENTRY snapshot and keep cached_len at L — the next block
        # re-extends positions L.. with the accepted rows as its prefix (their
        # KV rows recompute identically; determinism makes the prefix re-accept,
        # so progress >= 1 token/step is guaranteed). Full accept keeps the
        # graph's final state and advances cached_len past the block. No
        # per-position state log needed at all.
        import os as _os_ab

        if (_os_ab.environ.get("FREETOKEN_SPEC_AB", "0") in {"1", "true", "yes"}
                and getattr(self, "_spec_n_ab", 0) < 1):
            # Time-ladder probe: is the invalidation a function of the HOST
            # GAP between replays rather than any operation? (Everything
            # scheduler-visible now runs on the engine stream; the only
            # uncontrolled variable left is elapsed time / GPU idle.)
            import time as _time_ab

            self._spec_n_ab = getattr(self, "_spec_n_ab", 0) + 1
            entry_ab = self._spec_entry_snap
            _pr_ab = self.cache_manager.page_table[req.table_idx]

            def _pr_replay():
                with self.engine_stream_ctx:
                    vr.replay_step(z_ids=z_dev.to(self.device), slot=slot, L=L,
                                   page_row=_pr_ab, stream=self.engine.stream)
                    self.engine.stream.synchronize()
                return int(vr.nan_out[0])

            for _gap in (0.0, 0.05, 0.25, 1.0, 3.0):
                pool.restore_slot(slot, entry_ab)
                if _gap:
                    _time_ab.sleep(_gap)
                _nan = _pr_replay()
                print(f"[ABT] L={L} gap={_gap}s: nan={_nan}", flush=True)
            pool.restore_slot(slot, entry_ab)
            _pr_replay()

        import os as _os_b

        if _os_b.environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
            _tnan = {d: int(t.isnan().sum()) for d, t in taps.items()} if taps else {}
            self._sd2_ctx = (req.uid, L, st['pending'].get('pre', 1), z, rows, slot)
        a = 0
        while a < kn - 1 and rows[a] == z[a + 1]:
            a += 1
        bonus = rows[a]
        full = a == kn - 1
        entry_snap = self._spec_entry_snap
        if _os_b.environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
            _uid_d, _L_d, _pre_d, _z_d, _rows_d, _slot_d = getattr(
                self, "_sd2_ctx", (req.uid, L, 1, z, rows, slot))
            _emit_d = (z[_pre_d:a + 1] + [bonus]) if not full else (
                (z[_pre_d:] + [bonus]) if _pre_d < kn else [bonus])
            print(f"[sd2] uid={_uid_d} L={_L_d} pre={_pre_d} a={a} full={full} "
                  f"z[:3]={_z_d[:3]} rows[:3]={_rows_d[:3]} slot={_slot_d} "
                  f"emit={_emit_d} lsum={float(logits.abs().sum()):.1f} "
                  f"nan={vr.nan_out.tolist()}", flush=True)

        reply = []
        finished = False
        # rows [0, pre) of this block are already in input_ids (prior bonus +
        # retry-prefix appends); only emit genuinely new tokens so input_ids
        # grows exactly with the output budget (append_host caps at max_device_len)
        _t_acc0 = _time.perf_counter()
        # STREAM-GROUND-TRUTH watermark: the number of z-prefix tokens
        # ACTUALLY in input_ids beyond KV depth L. The old `pre` z-algebra
        # desynced here (it claimed z[:pre] was appended while the stream
        # held the previous bonus instead) -- the root cause of the spec
        # garbage class (wrong text from right logits).
        pre = int(req.input_ids.numel()) - L
        pre = max(0, min(pre, kn))
        if full:
            req.cached_len = req.device_len = L + kn
            emit = z[pre:] + [bonus] if pre < kn else [bonus]
        else:
            pool.restore_slot(slot, entry_snap)
            self.cache_manager._free(
                self.cache_manager.page_table[req.table_idx, L : L + kn])
            emit = z[pre : a + 1] + [bonus]
        for tok in emit:
            if req.input_ids.numel() >= req.max_device_len:   # budget BEFORE append
                finished = True                                # (append_host asserts at cap)
                reply.append(self._spec_msg(req, tok, ("length", None)))
                break
            req.append_host(_t.tensor([tok], dtype=_t.int32))
            fin = self._spec_emit_fin(req, tok)
            if fin is not None:
                finished = True
                if not full:
                    # EOS mid-emission on a partial block: the committed KV is
                    # only through this token; crop this step's rewrite rows.
                    self.cache_manager._free(self.cache_manager.page_table[
                        req.table_idx, L : L + kn]) if False else None
                reply.append(self._spec_msg(req, tok, fin))
                break
            self._spec_toolcall_anchor(req, tok)
            reply.append(self._spec_msg(req, tok, None))
        if not finished:
            self.token_pool[req.table_idx, L + a + 1] = _t.tensor(
                [bonus], dtype=_t.int32, device=self.device)
            _t3 = _time.perf_counter()
            taps_row = {d: t[a] for d, t in taps.items()}
            emb_row, mask_row = gctx.spec_embed(bonus)
            if _os_b.environ.get("FREETOKEN_SPEC_NODRAFT", "0") in {"1", "true", "yes"}:
                # invalidator bisect: skip the DFlash propose entirely
                picks = [bonus] * kn
            else:
                picks = self._spec_svc.propose(
                    bonus, L + a, taps_row,
                    torch.tensor(tk_ids[a]),
                    torch.tensor(tk_vals[a], dtype=torch.float32),
                    embed_row=emb_row, mask_row=mask_row)
            self._spec_t_prop = getattr(self, "_spec_t_prop", 0.0) + _time.perf_counter() - _t3
            if full:
                nxt = [bonus] + picks[1:]
            else:
                keep = z[: a + 1]                       # accepted prefix re-extends
                nxt = keep + [bonus] + picks[1 : kn - a - 1]
            st["pending"] = {"anchor": bonus, "position": L + a, "z": nxt[:kn]}
        else:
            if not full:
                # finished mid-partial-block: crop to emitted depth is implicit
                # (this step's pages were freed above); cached_len unchanged.
                pass
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
            self.finished_reqs.add(req)
            st["pending"] = None
        self._spec_t_emit = getattr(self, "_spec_t_emit", 0.0) + (
            _time.perf_counter() - _t_acc0)
        _t_sr0 = _time.perf_counter()
        self.send_result(reply)
        self._flush_abort_acks()
        self._spec_t_send = getattr(self, "_spec_t_send", 0.0) + (
            _time.perf_counter() - _t_sr0)
        self._spec_n = getattr(self, "_spec_n", 0) + 1
        if self._spec_n % 25 == 0:
            n = self._spec_n
            _acc2 = getattr(self, "_spec_sub_acc", {}) or {}
            _subs = " ".join(f"{k}={_acc2.get(k, 0.0)/n*1000:.0f}" for k in
                             ("total", "h2d", "launch", "sync", "nanread",
                              "argmax", "topk", "tolist1", "tolist2"))
            _subs += (f" | allocpg={getattr(self, '_spec_t_allocpg', 0.0)/n*1000:.0f}"
                      f" snap={getattr(self, '_spec_t_snap', 0.0)/n*1000:.0f}"
                      f" emit={getattr(self, '_spec_t_emit', 0.0)/n*1000:.0f}"
                      f" send={getattr(self, '_spec_t_send', 0.0)/n*1000:.0f}")
            print(f"[spec-timing] n={n} fwd={self._spec_t_fwd/n*1000:.0f}ms "
                  f"replay={getattr(self, '_spec_t_replay', 0.0)/n*1000:.0f}ms "
                  f"prop={self._spec_t_prop/n*1000:.0f}ms vr_n={getattr(self, '_spec_n_vr', 0)} "
                  f"| sub(ms): {_subs}", flush=True)
        return True

    def _spec_recapture(self, pool, vr):
        """Re-capture the verify graph on the current (scheduler/engine) stream
        with a clean GDN padding slot and out_loc aimed at the guard page."""
        _ps = pool.padding_slot
        pool.conv_states[:, _ps].zero_()
        pool.recurrent_states[:, _ps].zero_()
        _dp = int(self.cache_manager.page_table[
            self.engine.dummy_req.table_idx][0].item())
        vr.out_loc.fill_(_dp)
        _k = vr.k
        _gr = self.engine.graph_runner
        _gr.verify_runner = None
        del vr
        import gc as _gc

        _gc.collect()
        _gr._capture_verify(_k, self.engine.model,
                            stream=torch.cuda.current_stream())
        return _gr.verify_runner

    def _spec_arm(self, k: int):
        from freetoken.models.dflash.service import get_service

        svc = get_service(k=k)
        assert svc is not None, "draft service unavailable (needs 2 visible GPUs)"
        self._spec_svc = svc
        self._spec_state = {"pending": None, "log": None, "slot": None}
        print(f"[spec] armed k={k}", flush=True)

    def _spec_propose(self, anchor: int, position: int, taps_row, row_logits):
        import torch as _t

        from freetoken.core import get_global_ctx

        gctx = get_global_ctx()
        row = row_logits.reshape(-1).float()
        tv, ti = _t.topk(row, 64)
        emb_row, mask_row = gctx.spec_embed(int(anchor))
        return self._spec_svc.propose(
            int(anchor), int(position), taps_row, ti, tv,
            embed_row=emb_row, mask_row=mask_row)

    def _spec_pending_from_last_row(self, req):
        """First step after prefill: anchor/taps from the FINAL prefill row."""
        from freetoken.core import get_global_ctx

        gctx = get_global_ctx()
        logits = getattr(gctx, "spec_logits", None)
        taps = getattr(gctx, "spec_taps", None) or {}
        if logits is None or logits.shape[0] == 0 or not taps:
            return None
        L = req.cached_len
        row_logits = logits[logits.shape[0] - 1]
        # STREAM-GROUND-TRUTH anchor: if the prefill already appended its
        # sampled first token, THAT token is position L's content -- the
        # verify block must extend over it (deriving the anchor from logits
        # again duplicated it: 'TheThe' boundary artifact).
        if req.input_ids.numel() > L:
            anchor = int(req.input_ids[-1].item())
        else:
            anchor = int(row_logits.reshape(-1).argmax().item())
            # The prefill's reply SENDS this token to the client (its batch
            # reply path); make the stream match by appending it here so the
            # first spec block's pre-watermark counts it (z[0] skipped, not
            # re-emitted -- the prefill->spec 'TheThe' boundary duplicate).
            if req.input_ids.numel() < req.max_device_len:
                req.append_host(_t.tensor([anchor], dtype=_t.int32))
        taps_row = {d: t[t.shape[0] - 1] for d, t in taps.items()}
        return {"anchor": anchor, "position": L - 1,
                "picks": self._spec_propose(anchor, L - 1, taps_row, row_logits)}

    def _spec_step_inner(self, req, k: int) -> bool:
        import torch as _t

        from freetoken.attention.linear import FLAMetadata
        from freetoken.core import Batch, Req, get_global_ctx

        if getattr(self, "_spec_svc", None) is None:
            self._spec_arm(k)
        st = self._spec_state
        if st["pending"] is None:
            st["pending"] = self._spec_pending_from_last_row(req)
            if st["pending"] is None:
                return False  # taps not armed -> normal decode
        from freetoken.message.tokenizer import DetokenizeMsg

        gctx = get_global_ctx()
        pool = self.cache_manager.linear_state_pool
        slot = req.linear_slot_idx
        L = req.cached_len
        pend = st["pending"]
        if "z" in pend:
            z = [int(x) for x in pend["z"]]
        else:
            z = [int(pend["anchor"])] + [int(x) for x in pend["picks"][1:k]]
        # stream-ground-truth: replace the assumed z-prefix with the tokens
        # ACTUALLY appended beyond KV depth L (input_ids is ground truth)
        _tail_ids = req.input_ids[L:].tolist()
        if _tail_ids and _tail_ids != z[: len(_tail_ids)]:
            z = [int(x) for x in _tail_ids] + z[len(_tail_ids):]
            z = z[: max(k, len(z))]
        kn = len(z)
        _vr_k = getattr(getattr(self.engine.graph_runner, "verify_runner", None), "k", 0)
        if _vr_k and kn < _vr_k:
            z = z + [z[-1]] * (_vr_k - kn)      # pad short retry blocks to the fixed shape
        z_dev = _t.tensor(z, dtype=_t.int32)

        # z[0] (the pending bonus) was ALREADY appended to req.input_ids at the
        # previous step's emission -- including it again duplicates the token and
        # shifts the whole extension (attention metadata assert on step 2). The
        # first step after prefill has no appended bonus yet (input_ids == KV
        # depth), so append the full block only then.
        _base = req.input_ids
        _z_ext = z_dev if _base.numel() == L else z_dev[1:]

        # uid in the warmup range: if this transient ever leaks into a manager,
        # every real-traffic gate (spec included) skips it by uid alone.
        vreq = Req(input_ids=_t.cat([_base, _z_ext]),
                   table_idx=req.table_idx, cached_len=L, output_len=req.output_len,
                   uid=req.uid + self.WARMUP_UID_BASE, sampling_params=req.sampling_params,
                   cache_handle=req.cache_handle)
        vreq.device_len = vreq.input_ids.numel()
        # The transient verify req must NEVER enter decode_manager: _forward calls
        # filter_reqs(batch.reqs), which adds can_decode reqs -- a stale vreq in the
        # running set breaks the spec single-request gate AND serves garbage pages
        # on the next normal batch (the step-2 decode assert). Zero output budget
        # makes can_decode False for the transient only.
        vreq.output_len = 0
        vreq.linear_slot_idx = slot
        self.token_pool[vreq.table_idx, L : L + kn].copy_(z_dev)
        import time as _time

        vr = getattr(self.engine.graph_runner, "verify_runner", None)
        import os as _os_e

        if (vr is not None and kn == vr.k
                and _os_e.environ.get("FREETOKEN_SPEC_EAGER", "0") not in {"1", "true", "yes"}):
            return self._spec_replay_step(req, vr, st, gctx, z, z_dev, L, kn, slot)

        _t0 = _time.perf_counter()
        vbatch = Batch(reqs=[vreq], phase="prefill")
        forward_input = self._prepare_batch(vbatch)
        self._spec_t_prepare = getattr(self, "_spec_t_prepare", 0.0) + _time.perf_counter() - _t0
        vbatch.is_verify = True
        # cu_seqlens declares the varlen sequence boundaries: the k-row verify
        # block is ONE sequence of length kn. [0,1] (arange(0,2)) silently left
        # rows 1..kn-1 unprocessed -- their outputs were uninitialized pool
        # memory (finite garbage at capture, NaN once the pool dirtied).
        vbatch.fla_metadata = FLAMetadata(
            cu_seqlens=_t.tensor([0, kn], dtype=_t.int32, device=self.device),
            cache_indices=_t.tensor([slot], dtype=_t.int32, device=self.device),
            has_initial_state=None, fresh_state_indices=None)
        if st["log"] is None or st["slot"] != slot or len(st["log"]) < kn:
            conv0, rec0 = pool.conv_states[:, slot], pool.recurrent_states[:, slot]
            st["log"] = [(_t.zeros_like(conv0), _t.zeros_like(rec0)) for _ in range(kn)]
            st["slot"] = slot
        gctx.spec_state_log = st["log"]
        _t1 = _time.perf_counter()
        with self.engine_stream_ctx:  # engine asserts its stream is current
            self._restore_linear_states(vbatch)
            self._forward(forward_input)
                # ALL host consumption of the verify forward's outputs happens INSIDE
            # the engine-stream context after an explicit sync: per-row argmax +
            # top-64 launch here and the small results are read via .tolist() while
            # the stream is ours. Reading engine-stream tensors from the ambient
            # context deadlocks on ROCm (the original post-forward stall).
            self.engine.stream.synchronize()
            logits = gctx.spec_logits
            rows = [int(v) for v in logits.argmax(-1).reshape(-1).tolist()]
            tk = torch.topk(logits.float(), 64, dim=-1)
            tk_ids = tk.indices.tolist()
            tk_vals = tk.values.tolist()
        gctx.spec_state_log = None
        # The transient must not linger in decode_manager (filter_reqs ran inside
        # _forward with can_decode semantics we don't fully control) — drop it
        # explicitly; remove_req is an idempotent discard.
        self.decode_manager.remove_req(vreq)
        _t2 = _time.perf_counter()
        self._spec_t_fwd = getattr(self, "_spec_t_fwd", 0.0) + (_t2 - _t1)

        taps = gctx.spec_taps or {}
        import os as _os_dbg2

        if _os_dbg2.environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
            _tnan = {d: int(t.isnan().sum()) for d, t in taps.items()} if taps else {}
            print(f"[sd2e] uid={req.uid} L={L} pre={st['pending'].get('pre', 1)} "
                  f"z[:3]={z[:3]} rows[:3]={rows[:3]} slot={slot} "
                  f"lsum={float(logits.abs().sum()):.1f} tap_nan={_tnan}", flush=True)
        a = 0
        while a < kn - 1 and rows[a] == z[a + 1]:
            a += 1
        bonus = rows[a]
        committed = a + 1                          # z[0..a] get KV
        self.cache_manager._free(
            self.cache_manager.page_table[req.table_idx, L + committed : L + kn])
        if committed < kn:
            pool.restore_slot(slot, st["log"][committed - 1])
        req.cached_len = req.device_len = L + committed

        reply = []
        finished = False
        # z[0] is the PREVIOUS step's bonus -- already emitted and appended then.
        # Emit only the newly-verified tokens z[1..committed-1]; re-sending z[0]
        # duplicated every bonus into the stream (tripled text with echo accepts).
        for i, tok in enumerate(z[1:committed], start=1):
            if i > 0:
                req.append_host(_t.tensor([tok], dtype=_t.int32))
            fin = self._spec_emit_fin(req, tok)
            if fin is not None:
                finished = True
                if i + 1 < committed:
                    self.cache_manager._free(self.cache_manager.page_table[
                        req.table_idx, L + i + 1 : L + committed])
                    pool.restore_slot(slot, st["log"][i])
                    req.cached_len = req.device_len = L + i + 1
                reply.append(self._spec_msg(req, tok, fin))
                break
            self._spec_toolcall_anchor(req, tok)
            reply.append(self._spec_msg(req, tok, None))
        if not finished:
            if req.input_ids.numel() >= req.max_device_len:
                finished = True
                reply.append(self._spec_msg(req, bonus, ("length", None)))
            else:
                req.append_host(_t.tensor([bonus], dtype=_t.int32))
            # normal-decode compatibility: a fallback non-spec step reads its input
            # token from token_pool[cached_len]; stage the pending bonus there.
            self.token_pool[req.table_idx, req.cached_len] = _t.tensor(
                [bonus], dtype=_t.int32, device=self.device)
            fin = self._spec_emit_fin(req, bonus)
            if fin is not None:
                finished = True
                reply.append(self._spec_msg(req, bonus, fin))
            else:
                self._spec_toolcall_anchor(req, bonus)
                reply.append(self._spec_msg(req, bonus, None))
                _t3 = _time.perf_counter()
                taps_row = {d: t[a] for d, t in taps.items()}
                emb_row, mask_row = gctx.spec_embed(bonus)
                st["pending"] = {
                    "anchor": bonus, "position": L + a,
                    "picks": self._spec_svc.propose(
                        bonus, L + a, taps_row,
                        torch.tensor(tk_ids[a]),
                        torch.tensor(tk_vals[a], dtype=torch.float32),
                        embed_row=emb_row, mask_row=mask_row),
                }
                self._spec_t_prop = getattr(self, "_spec_t_prop", 0.0) + _time.perf_counter() - _t3
                _svc = self._spec_svc
                if _svc.n_propose > 0 and _svc.n_propose % 25 == 0:
                    print(f"[svc-direct] n={_svc.n_propose} total={_svc.ms/_svc.n_propose:.1f} "
                          f"pre={_svc.t_pre/_svc.n_propose:.1f} fwd={_svc.t_fwd/_svc.n_propose:.1f} "
                          f"chn={_svc.t_chn/_svc.n_propose:.1f} graph={hasattr(_svc,'_graph')}",
                          flush=True)
        if finished:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
            self.finished_reqs.add(req)
            st["pending"] = None
        self.send_result(reply)
        self._flush_abort_acks()
        self._spec_n = getattr(self, "_spec_n", 0) + 1
        if self._spec_n % 25 == 0:
            n = self._spec_n
            print(f"[spec-timing] n={n} prep={self._spec_t_prepare/n*1000:.0f}ms "
                  f"fwd={self._spec_t_fwd/n*1000:.0f}ms prop={self._spec_t_prop/n*1000:.0f}ms "
                      f"emb={getattr(self, '_spec_t_emb', 0.0)/n*1000:.0f}ms",
                  flush=True)
        return True

    def _spec_emit_fin(self, req, tok):
        hit_length = not req.can_decode
        hit_eos = (not req.sampling_params.ignore_eos) and tok in self.eos_token_ids
        matched = (self._match_stop_str(req)
                   if not hit_eos and req.sampling_params.stop_strs else None)
        if hit_length or hit_eos or matched is not None:
            return ("stop" if (hit_eos or matched is not None) else "length", matched)
        return None

    def _spec_msg(self, req, tok, fin):
        from freetoken.message.tokenizer import DetokenizeMsg

        return DetokenizeMsg(
            uid=req.uid, next_token=int(tok), finished=fin is not None,
            finish_reason=(fin[0] if fin is not None else None),
            matched_stop=(fin[1] if fin is not None else None),
            stop_strs=req.sampling_params.stop_strs or None)

    def _spec_toolcall_anchor(self, req, tok):
        if tok == self.toolcall_anchor_id and req.toolcall_anchor_len is None:
            req.toolcall_anchor_len = req.input_ids.numel()

    WARMUP_UID_BASE = 1 << 60

    def _submit_warmup_req(self, uid: int, n_tokens: int) -> None:
        from freetoken.core import SamplingParams

        # Plain-text vocabulary range: far from every special/control/eos id, and
        # ignore_eos so the model cannot terminate the request before its one
        # decode step -- that step is what compiles the decode kernels at depth.
        ids = (torch.arange(n_tokens, dtype=torch.int32) % 24000) + 2000
        self.prefill_manager.add_one_req(
            UserMsg(
                uid=uid,
                input_ids=ids,
                sampling_params=SamplingParams(
                    temperature=0.0, max_tokens=2, ignore_eos=True
                ),
            )
        )

    def _run_warmup_loop(self, max_iters: int, desc: str | None = None) -> None:
        for i in range(max_iters):
            if not (self.prefill_manager.runnable or self.decode_manager.runnable):
                return
            if desc is not None and i % 8 == 0:
                emit_progress(desc, i, max_iters)
            self.normal_loop()
        # Bound tripped: an admission gate refuses our synthetic request. Drop warmup
        # state so run_forever does not inherit a hot spin over an unschedulable req.
        logger.critical_rank0(
            "prefill warmup did not drain within its iteration bound; discarding "
            "warmup requests (first real request may still pay cold-compile cost)"
        )
        self.prefill_manager.pending_list.clear()
        self.decode_manager.running_reqs = {
            r for r in self.decode_manager.running_reqs if r.uid < self.WARMUP_UID_BASE
        }

    def _prefill_warmup(self) -> None:
        """Compile the autotuned prefill/decode Triton kernels' specialization space
        before the first real request (see scheduler/warmup.py for why: a first-hit
        extend length or KV-depth bucket sweeps whole autotune grids inside the
        request's forward -- measured 70-270s stalls, fatal against the frontend's
        300s SSE idle limit). Runs the real request path end-to-end so admission,
        chunking, GDN state, and the drain/free paths are exercised too; winning
        configs persist in the on-disk triton cache, so steady-state boots pay only
        the GPU time. Failure is never fatal: any exception logs and leaves serving
        running (worst case the old cold-first-hit behavior)."""
        import time as _time

        if not warmup_enabled():
            return
        max_extend = self.config.max_extend_tokens
        # Warmup requests bypass the admission guard (they enter via add_one_req,
        # not the guarded _process_one_msg path), so THEY must enforce the
        # sequence-length boundary themselves: the walk ends with 2 decode steps,
        # and decoding at position == max_seq_len indexes one past every
        # max_seq_len-sized buffer (rope cache, positions) -> async HSA 0x1016
        # hardware exception with no kernel attribution. Real traffic is safe —
        # the guard clamps max_tokens — this is warmup-only.
        seq_cap = int(getattr(self.engine, "max_seq_len", 0) or 0)
        depth = warmup_depth(seq_cap)
        if seq_cap:
            depth = min(depth, seq_cap - 4)
        t_all = _time.perf_counter()
        drained_all = True
        try:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                # Phase 1: one request per extend-length class. max_tokens=2 so each
                # also runs one decode step at its (shallow) depth.
                ladder = prefill_warmup_lengths(max_extend)
                t0 = _time.perf_counter()
                for i, n in enumerate(ladder):
                    emit_progress("compiling prefill kernels", i + 1, len(ladder))
                    self._submit_warmup_req(self.WARMUP_UID_BASE + i, n)
                    # ladder lengths never exceed max_extend, so each is one prefill
                    # iteration + a couple of decode iterations
                    self._run_warmup_loop(max_iters=8)
                logger.info_rank0(
                    f"prefill warmup: {len(ladder)} extend-length classes up to "
                    f"{max_extend} done in {_time.perf_counter() - t0:.1f}s"
                )
                # Phase 2: one chunked prefill walking the context depth buckets to
                # the serving depth (the maestro-class first-deep-request stall),
                # ending with a decode step at full depth.
                if depth > max_extend:
                    t0 = _time.perf_counter()
                    uid = self.WARMUP_UID_BASE + len(ladder)
                    self._submit_warmup_req(uid, depth)
                    expected = depth // max(1, max_extend) + 8
                    self._run_warmup_loop(
                        max_iters=expected + 64, desc="compiling deep-context kernels"
                    )
                    if self.prefill_manager.runnable or self.decode_manager.runnable:
                        drained_all = False
                        logger.critical_rank0(
                            "prefill warmup: depth walk did NOT drain (admission or "
                            "scheduling stuck); warmup requests discarded"
                        )
                        self.prefill_manager.pending_list.clear()
                        self.decode_manager.running_reqs = {
                            r for r in self.decode_manager.running_reqs
                            if r.uid < self.WARMUP_UID_BASE
                        }
                    else:
                        logger.info_rank0(
                            f"prefill warmup: depth walk to {depth} tokens done in "
                            f"{_time.perf_counter() - t0:.1f}s"
                        )
        except Exception:
            import traceback

            logger.critical_rank0(
                "prefill warmup failed (serving continues; first hits pay cold-compile): "
                + traceback.format_exc()
            )
            self.prefill_manager.pending_list.clear()
            self.decode_manager.running_reqs = {
                r for r in self.decode_manager.running_reqs
                if r.uid < self.WARMUP_UID_BASE
            }
        else:
            state = "complete" if drained_all else "INCOMPLETE (see critical above)"
            logger.info_rank0(
                f"prefill warmup {state} in {_time.perf_counter() - t_all:.1f}s "
                "(triton autotune cache warm; first requests will not stall)"
            )

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt an in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        import os as _os

        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                        continue
                    # Hybrid radix: commit the chunk's ×64 snapshot + KV prefix NOW. The old
                    # unconditional skip left chunked prefills with NO reusable snapshot below
                    # the finish-donate depth (prompt+generated) -- identical-prompt reruns
                    # matched 0 and re-prefilled. The commit donates the frozen slot at the
                    # tracked boundary, re-points this request's row to the canonical pages,
                    # Default ON (2026-09-01): chunk commits are what make identical-prompt
                    # reruns hit the radix cache (probe pair 15s -> 3.1s). The lock-per-commit
                    # and stale-floor double-free that wedged the 9-slot GDN pool are fixed in
                    # CacheManager._cache_req_hybrid (reserve-first commit + mamba_commit_upto
                    # dedup floor); counters are derived via _recount at read time. Set =0 to
                    # fall back to finish-donate-only caching.
                    if (
                        _os.environ.get("FREETOKEN_CHUNK_COMMIT", "1") in {"1", "true", "yes"}
                        and self.cache_manager.is_hybrid
                        and req.table_idx != -1
                    ):
                        self.cache_manager.cache_req(req, finished=False)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                mtp_probe = getattr(getattr(self, "engine", None), "mtp_probe", None)
                if mtp_probe is not None and not batch.is_prefill:
                    # k=1 acceptance probe: draft predicts from (h, token);
                    # the NEXT committed token settles it. Both trunk channels
                    # ride ctx as stable graph-rewritten buffers; row i is
                    # this request's.
                    from freetoken.core import get_global_ctx

                    gctx = get_global_ctx()
                    h = getattr(gctx, "trunk_hidden_prenorm", None)
                    hp = getattr(gctx, "trunk_hidden", None)
                    mtp_probe.step(
                        id(req),
                        int(batch.positions[i].item()) if batch.positions is not None else 0,
                        next_token,
                        h[i] if h is not None else None,
                        hp[i] if hp is not None else None,
                    )
                # EOS / stop-string -> "stop", output budget exhausted -> "length";
                # EOS and stop strings win over length.
                hit_length = not req.can_decode
                hit_eos = (
                    not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                )
                matched_stop = (
                    self._match_stop_str(req)
                    if not hit_eos and req.sampling_params.stop_strs
                    else None
                )
                finished = hit_length or hit_eos or matched_stop is not None
                finish_reason = (
                    ("stop" if (hit_eos or matched_stop is not None) else "length")
                    if finished
                    else None
                )
                if (
                    next_token == self.toolcall_anchor_id
                    and req.toolcall_anchor_len is None
                    and not finished
                ):
                    req.toolcall_anchor_len = req.input_ids.numel()
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=matched_stop,
                        stop_strs=req.sampling_params.stop_strs or None,
                    )
                )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        self.send_result(reply)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                import os as _os_m

                if _os_m.environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
                    print(f"[phase] cow-restore uid={req.uid} "
                          f"src={req.mamba_restore_src} dst={req.linear_slot_idx}",
                          flush=True)
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1
        # Drop the MTP probe's eager KV rows for this request before its id()
        # can be recycled by a later request.
        mtp_probe = getattr(getattr(self, "engine", None), "mtp_probe", None)
        if mtp_probe is not None:
            mtp_probe.reset_req(id(req))

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        import os as _os, time as _t

        _pt = _os.environ.get("FREETOKEN_PREP_TIMING", "0") == "1"
        if _pt:
            _m = {}
            _t0 = _t.perf_counter()
        self.engine.graph_runner.pad_batch(batch)
        if _pt:
            _m["pad"] = _t.perf_counter() - _t0; _t0 = _t.perf_counter()
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        if _pt:
            _m["swa"] = _t.perf_counter() - _t0; _t0 = _t.perf_counter()
        self.cache_manager.allocate_paged(batch.reqs)
        if _pt:
            _m["alloc"] = _t.perf_counter() - _t0; _t0 = _t.perf_counter()
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if _pt:
            _m["mappings"] = _t.perf_counter() - _t0; _t0 = _t.perf_counter()
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = _pinned_to_device(
                        torch.tensor(slots, dtype=torch.int32, device="cpu",
                                     pin_memory=True), self.device)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        if _pt:
            _m["fla"] = _t.perf_counter() - _t0; _t0 = _t.perf_counter()
        self.engine.attn_backend.prepare_metadata(batch)
        if _pt:
            _m["attn"] = _t.perf_counter() - _t0; _t0 = _t.perf_counter()
            self._pt_n = getattr(self, "_pt_n", 0) + 1
            acc = getattr(self, "_pt_acc", None)
            if acc is None:
                acc = {}; self._pt_acc = acc
            for k2, v2 in _m.items():
                acc[k2] = acc.get(k2, 0.0) + v2
            if self._pt_n % 200 == 0:
                print("[prep-timing] n=" + str(self._pt_n) + " " +
                      " ".join(f"{k2}={v2/self._pt_n*1000:.1f}ms" for k2, v2 in acc.items()),
                      flush=True)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate per-request vision soft tokens (in request order) for a prefill
        batch so the model can scatter them at image-token positions. ``req.mm_embeds``
        is kept (not cleared) so the cache manager can recognize multimodal requests and
        keep them out of the shared prefix cache (image placeholders share a token id but
        carry per-image content)."""
        parts = [req.mm_embeds for req in batch.reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        if getattr(self, "_sched_diag", None) is not None:
            # a previously-stuck gate unblocked itself — report the recovery
            print(f"[sched-diag] unblocked after: {self._sched_diag}", flush=True)
            self._sched_diag = None
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _fwd_timed(self, forward_input: ForwardInput) -> ForwardOutput:
        """Step-probe wrapper: per-decode-step wall time, logged every N steps
        (FREETOKEN_STEP_PROBE=1, FREETOKEN_STEP_PROBE_EVERY=10)."""
        import os as _os
        import time as _time

        import time as _t2

        _w0 = _t2.perf_counter()
        out = self._forward(forward_input)
        if _os.environ.get("FREETOKEN_STEP_PROBE", "0") in {"1", "true", "yes"}:
            self._step_n = getattr(self, "_step_n", 0) + 1
            self._step_t = getattr(self, "_step_t", 0.0) + (_t2.perf_counter() - _w0)
            every = int(_os.environ.get("FREETOKEN_STEP_PROBE_EVERY", "10"))
            if self._step_n % every == 0:
                print(
                    f"[step-probe] n={self._step_n} avg="
                    f"{self._step_t / self._step_n * 1000:.1f} ms/step",
                    flush=True,
                )
        return out

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        import os as _os
        import time as _time

        _probe = _os.environ.get("FREETOKEN_STEP_PROBE", "0") in {"1", "true", "yes"}
        _t0 = _time.perf_counter() if _probe else None
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return _pinned_to_device(indices_host, device)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return _pinned_to_device(mapping_host, device), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return _pinned_to_device(mapping_host, device), _pinned_to_device(write_host, device)
