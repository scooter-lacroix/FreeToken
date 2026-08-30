from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from freetoken.distributed import DistributedInfo
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs
    from .supervisor import BackendHandle


def _report_startup_error(ack_queue: mp.Queue, exc: BaseException) -> None:
    """Tell the parent WHY this worker is dying — push an ("error", reason) ack before it exits,
    so the supervisor reports the real cause (e.g. a config ValueError) instead of the generic
    "backend worker … exited during load". Best-effort and flushed (close + join_thread), since
    the process is about to terminate; a failure to report must never mask the original error."""
    try:
        ack_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        ack_queue.close()
        ack_queue.join_thread()
    except Exception:  # noqa: BLE001 -- reporting is a nicety; never shadow the real exception
        pass


def _detach_process_group() -> None:
    """Shell mode only: move this worker out of the terminal's foreground process group.

    The shell binds ^C to "cancel this turn", but a terminal delivers SIGINT to the whole
    foreground group — which, with the engine running in this same process, includes the
    workers. They would take the same ^C and exit (``_run_scheduler`` below stops gracefully on
    KeyboardInterrupt), leaving the shell chatting with a dead engine. Nothing depends on the
    signal reaching them: the parent tears the workers down explicitly on every stop path
    (uvicorn's lifespan, plus the SIGTERM/SIGHUP handler and the reap backstop in api_server).

    ``ft serve`` keeps the default — uvicorn owns ^C there, and the group-wide delivery is part
    of how it stops."""
    try:
        os.setpgrp()
    except OSError:  # no job control (already a group leader / unusual environment)
        pass


def _run_tokenize_worker(detach: bool, **kwargs) -> None:
    """Module-level so it survives the spawn pickle; exists only to detach the group first."""
    if detach:
        _detach_process_group()
    from freetoken.tokenizer import tokenize_worker

    tokenize_worker(**kwargs)


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    if args.shell_mode:
        _detach_process_group()

    import os as _os

    if _os.environ.get("FREETOKEN_FAULTHANDLER", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        # periodic python-stack dumps for livelock diagnosis (py-spy needs
        # ptrace the sandbox denies); FREETOKEN_FAULTHANDLER_SECS sets the
        # interval (default 120)
        import faulthandler

        faulthandler.dump_traceback_later(
            int(_os.environ.get("FREETOKEN_FAULTHANDLER_SECS", "120") or 120),
            repeat=True,
        )

    import torch
    from freetoken.scheduler import Scheduler

    import os as _osd

    if _osd.environ.get("FREETOKEN_MOE_PHASE_LOG", "") == "PING":
        import freetoken.layers.moe as _m

        print(
            f"[moe-boot] moe.py={_m.__file__} PHASE_LOG={_osd.environ.get('FREETOKEN_MOE_PHASE_LOG')!r}",
            flush=True,
        )
    if args.tp_info.is_primary():
        from freetoken.utils.progress import set_progress_sink

        set_progress_sink(
            lambda desc, done, total: ack_queue.put(("progress", desc, done, total))
        )

    # Serve from the SSD-staged copy when one can be provisioned now: one
    # boot-time sequential copy BEFORE the model maps it, so the worker's
    # mapping (and therefore page pinning) lives on the SSD and refills never
    # touch the spinning origin. Residency (pin while serving, release when
    # idle) starts right after, still before the ready ack.
    from freetoken.utils.residency import (
        resolve_staged_model,
        start_cache_gardener,
        start_residency,
    )

    # ServerArgs is frozen; the args.py tool-parser sniff sets the precedent
    # for object.__setattr__ here.
    object.__setattr__(args, "model_path", resolve_staged_model(args.model_path))
    # Bounded-rate page-cache warm-keeper: keeps the checkpoint resident without
    # unpageable memory (the mlock path is opt-in via FREETOKEN_RESIDENCY=1).
    start_cache_gardener(args.model_path)

    with torch.inference_mode():
        try:
            scheduler = Scheduler(args)
            scheduler.sync_all_ranks()
        except Exception as exc:  # noqa: BLE001 -- surface the reason, then let it propagate
            # A startup failure (bad config, OOM, corrupt weights) would otherwise reach the
            # parent only as a dead process -> a generic "exited during load". Push the real
            # reason first so the supervisor (and the desktop failure modal) can surface it;
            # the traceback still prints and the process still exits non-zero.
            _report_startup_error(ack_queue, exc)
            raise

        if args.tp_info.is_primary():
            # Report the real per-unit cache VRAM costs (KV/expert/mamba), the device-wide free
            # VRAM, and the per-pool rebuild floors before the ready ack, so the supervisor has
            # them (and the desktop's slider bounds) by the time the gate flips. Optional +
            # best-effort: a failure here must never keep the model from serving, and older
            # consumers ignore ("meta", …).
            # Warm the prefill/decode Triton specialization space BEFORE the ready ack:
            # a first-hit extend length or KV-depth bucket would otherwise sweep whole
            # autotune grids inside a live request's forward (measured 70-270s stalls).
            # Self-contained: logs, reports progress, and never raises on failure.
            start_residency(
                args.model_path,
                activity_fn=lambda: scheduler.prefill_manager.runnable
                or scheduler.decode_manager.runnable,
            )
            scheduler._prefill_warmup()
            try:
                from freetoken.kvcache.cache_status import compute_cache_status_meta

                ack_queue.put(("meta", compute_cache_status_meta(scheduler.engine)))
            except Exception:  # noqa: BLE001 -- metadata is a nicety; readiness is not
                pass
            ack_queue.put("Scheduler is ready")
            # The supervisor stops draining ack_queue once ready, so uninstall the sink:
            # runtime cache rebuilds re-run the graph capture (which emits progress) and
            # would otherwise push onto a queue nobody reads for the server's lifetime.
            set_progress_sink(None)

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(
        sys.argv[1:] if argv is None else argv,
        run_shell,
        prog=prog,
    )
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> "BackendHandle":
        import multiprocessing as mp

        from .supervisor import BackendHandle

        mp.set_start_method("spawn", force=True)
        detach = server_args.shell_mode  # see _detach_process_group

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue = mp.Queue()
        processes: list[mp.Process] = []

        for i in range(world_size):
            new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
            p = mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"freetoken-TP{i}-scheduler",
            )
            p.start()
            processes.append(p)

        num_tokenizers = server_args.num_tokenizer
        p = mp.Process(
            target=_run_tokenize_worker,
            kwargs={
                "detach": detach,
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="freetoken-detokenizer-0",
        )
        p.start()
        processes.append(p)
        for i in range(num_tokenizers):
            p = mp.Process(
                target=_run_tokenize_worker,
                kwargs={
                    "detach": detach,
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"freetoken-tokenizer-{i}",
            )
            p.start()
            processes.append(p)

        # Expected ready acks: 1 primary scheduler + num_tokenizers + 1 detokenizer.
        return BackendHandle(
            ack_queue=ack_queue,
            processes=processes,
            expected_acks=num_tokenizers + 2,
        )

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
