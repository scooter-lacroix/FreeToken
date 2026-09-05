from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


class VerifyGraphRunner:
    """S4 spec-verify forward as ONE CUDA graph: the k-row extend (attention
    over the full context + sequential GDN per-position state logging + taps +
    full-row lm_head) is a FIXED shape every step, so the ~530 eager launches
    collapse to one replay. Inputs land in persistent buffers before replay;
    logits / taps / per-position GDN state come out of captured buffers after
    an explicit sync."""

    def __init__(self, k: int, max_seq_len: int, vocab_size: int, hidden: int,
                 device: torch.device, tap_layers):
        self.k = k
        self.device = device
        self.input_ids = torch.zeros(k, dtype=torch.int32, device=device)
        self.positions = torch.zeros(k, dtype=torch.int32, device=device)
        self.out_loc = torch.zeros(k, dtype=torch.int32, device=device)
        self.cu_q = torch.tensor([0, k], dtype=torch.int32, device=device)
        self.kv_indptr = torch.zeros(2, dtype=torch.int32, device=device)
        self.prefix_lens = torch.zeros(1, dtype=torch.int32, device=device)
        self.indices = torch.zeros(max_seq_len + k + 64, dtype=torch.int32, device=device)
        self.linear_idx = torch.zeros(1, dtype=torch.int32, device=device)
        self.q_to_req = torch.zeros(k, dtype=torch.int32, device=device)
        # fp32 (not bf16): the scheduler argmaxes/topk's this buffer directly --
        # bf16 mantissa could flip argmax between near-tied top-1/top-2 logits
        # and corrupt greedy acceptance.
        self.logits_out = torch.empty(k, vocab_size, dtype=torch.float32, device=device)
        self.taps = {d: torch.zeros(k, hidden, dtype=torch.bfloat16, device=device)
                     for d in tap_layers}
        # per-position GDN snapshots: k pairs of full-slot columns, written by the
        # captured verify branch (restore log[a] on partial accept)
        # NaN/Inf counters written by a captured kernel (readable post-replay):
        # localizes whether the replay forward itself NaNs vs. the copy.
        self.nan_out = torch.zeros(2, dtype=torch.int32, device=device)
        self.graph = None
        # Piecewise variant (FREETOKEN_SPEC_PIECEWISE=1): segments split at the
        # MoE ensure/copy seams, host-driven expert fetches between replays --
        # same contract as the decode graphs. A monolithic verify capture bakes
        # whatever host-side miss copies fire DURING capture (suppress_inline_copy
        # is not set there), so whether the graph is clean is a per-boot
        # residency lottery (exact or stable-wrong + captured-copy host-page
        # faults). Piecewise removes the lottery by construction.
        self.pw_graphs = None
        self.pw_seams = []
        self.moe_cache = None

    def _forward_ctx(self, gctx, batch, model):
        with gctx.forward_batch(batch):
            return model.forward()

    def _replay_pw(self) -> None:
        cache = self.moe_cache
        cache.suppress_inline_copy = True
        try:
            self.pw_graphs[0].replay()
            for i, layer_id in enumerate(self.pw_seams):
                # Host-driven miss fetch for this layer, then the segment
                # holding its expert GEMM (mirrors GraphRunner.replay).
                cache.copy_missing_staged(layer_id)
                self.pw_graphs[i + 1].replay()
        finally:
            cache.suppress_inline_copy = False

    def capture(self, attn_backend, model, dummy_req, stream=None, moe_cache=None):
        import torch

        from freetoken.core import Batch, Context, get_global_ctx
        from freetoken.attention.linear import FLAMetadata
        from freetoken.attention.triton import TritonMetadata
        from freetoken.kvcache.linear_state_pool import LinearStatePool

        pool = get_global_ctx().linear_state_pool
        # NOTE: no per-position state_log here -- nothing reads it since the
        # gdn is_verify branch was deleted (partial-accept rollback uses the
        # scheduler's entry snapshot instead), and k x full-width GDN columns
        # cost ~144 MiB, which OOMs a mid-serving re-capture.

        batch = Batch(reqs=[dummy_req], phase="prefill")
        batch.padded_reqs = batch.reqs
        batch.is_verify = True
        batch.input_ids = self.input_ids
        batch.positions = self.positions
        batch.out_loc = self.out_loc
        batch.linear_table_idx = self.linear_idx
        batch.fla_metadata = FLAMetadata(
            # one varlen sequence spanning the whole k-row block: [0,1] would
            # leave rows 1..k-1 unprocessed by the fla chunk kernel (uninit
            # outputs -> NaN/garbage logits on replay)
            cu_seqlens=torch.tensor([0, self.k], dtype=torch.int32, device=self.device),
            cache_indices=self.linear_idx,
        )
        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q_gpu=self.cu_q,
            indptr=self.kv_indptr,
            indices=self.indices,
            q_to_req=self.q_to_req,
            q_positions=self.positions,
            is_decode=False,
            prefix_lens=self.prefix_lens,
            max_q_len=self.k,
            swa_indices=None,
        )
        gctx = get_global_ctx()
        gctx.spec_tap_dev = self.taps
        gctx.spec_logits = self.logits_out  # the flush stash points here at capture
        # THE sink: without this the captured flush rebinds ctx.spec_logits to a
        # transient pool tensor and logits_out is never written on replay (all
        # rows argmaxed token 0 -> '!' garbage loop at full acceptance).
        gctx.spec_logits_sink = self.logits_out
        gctx.spec_nan_sink = self.nan_out

        # ``stream``: the stream the graph will be REPLAYED on. torch's
        # default capture uses an internal side stream; for replay stability
        # on this ROCm stack capture, replay, and every inter-replay op must
        # share ONE stream (the boot selftest proved that config clean;
        # cross-stream inter-replay work yields all-NaN or hard faults), so
        # serving captures pass the scheduler's stream explicitly.
        _cs = stream if stream is not None else torch.cuda.Stream()
        self.replay_stream = _cs
        _pace = __import__("os").environ.get("FREETOKEN_SPEC_PACE", "")
        if _pace == "capture":
            __import__("time").sleep(0.02)
        elif _pace == "selftest":
            # pace between each selftest cycle (the print-dense interval of
            # the 5/5 passing boots)
            class _Pacer:
                def __init__(self):
                    self.n = 0
                def tick(self):
                    self.n += 1
                    __import__("time").sleep(0.005)
            self._pace_ticks = _Pacer()
        _moe_cache = moe_cache
        if (_moe_cache is not None
                and __import__("os").environ.get("FREETOKEN_SPEC_PIECEWISE", "0")
                in {"1", "true", "yes"}):
            # PIECEWISE verify capture: warmups run eagerly (misses fetch for
            # real, populating the cache for the capture batch's routing),
            # then the captured walk runs with inline copies SUPPRESSED -- the
            # MoE layer closes a segment at each ensure/copy seam and the
            # runner replays segments with copy_missing_staged between them.
            self.moe_cache = _moe_cache
            with torch.cuda.device(self.device):
                with torch.cuda.stream(_cs):
                    for _ in range(2):
                        with gctx.forward_batch(batch):
                            model.forward()
                    torch.cuda.synchronize()
                    from freetoken.engine.piecewise import PiecewiseCapture

                    _moe_cache.suppress_inline_copy = True
                    try:
                        cap = PiecewiseCapture(_cs)
                        cap.capture(lambda: self._forward_ctx(gctx, batch, model))
                    finally:
                        _moe_cache.suppress_inline_copy = False
                    torch.cuda.synchronize()
            self.pw_graphs = cap.graphs
            self.pw_seams = cap.seams
            self.graph = None
            if __import__("os").environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
                print(f"[vr-cap] piecewise verify captured: "
                      f"{len(self.pw_graphs)} segments, seams={self.pw_seams}",
                      flush=True)
            gctx.spec_tap_dev = None
            gctx.spec_logits_sink = None
            gctx.spec_nan_sink = None
            return
        with torch.cuda.device(self.device):
            with torch.cuda.stream(_cs):
                # CAPTURE STAGING (FREETOKEN_SPEC_CAPSTAGE=1): the monolithic
                # walk runs with the buffers' ZERO initialization unless we
                # pre-stage them -- and any host value derived from the staged
                # DATA at launch (grid dims, split counts) then freezes the
                # dummy-shape variant into the graph. Stage realistic maximum
                # metadata so the captured kernels match replay-time shapes.
                if __import__("os").environ.get(
                        "FREETOKEN_SPEC_CAPSTAGE", "0") in {"1", "true", "yes"}:
                    _L = self.indices.shape[0] - self.k - 64
                    self.positions.copy_(
                        torch.arange(_L, _L + self.k, dtype=torch.int32,
                                     device=self.device))
                    _dp0 = int(self.out_loc.shape[0])  # placeholder, overwritten
                    self.out_loc.zero_()
                    self.indices.zero_()
                    self.kv_indptr[0].zero_()
                    self.kv_indptr[1].fill_(_L + self.k)
                    self.prefix_lens[0].fill_(_L)
                    torch.cuda.synchronize()
                    print(f"[vr-cap] CAPSTAGE staged L={_L} for warmups+capture",
                          flush=True)
                for _ in range(2):
                    with gctx.forward_batch(batch):
                        model.forward()
                torch.cuda.synchronize()
                if __import__("os").environ.get(
                        "FREETOKEN_SPEC_POOLISO", "0") in {"1", "true", "yes"}:
                    # POOL-ISOLATION: unmap every freed warmup allocation before
                    # the captured walk. If a captured kernel later reads a
                    # FREED warmup address (the allocation-escape class), the
                    # replay now FAULTS (deterministic) instead of silently
                    # reading stale bytes — discriminates freed-read vs
                    # live-but-wrong-read escapes.
                    import gc as _gc_iso

                    _gc_iso.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("[vr-cap] POOL-ISO: cache emptied pre-capture",
                          flush=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=_cs):
                    with gctx.forward_batch(batch):
                        model.forward()
                torch.cuda.synchronize()
        import os as _os_dbg

        _selftest_on = _os_dbg.environ.get(
            "FREETOKEN_SPEC_SELFTEST", "0") in {"1", "true", "yes"}
        if _selftest_on:
            _stv = set(x.strip() for x in _os_dbg.environ.get(
                "FREETOKEN_SPEC_STV",
                "dummy,l66both,kvsplit,pos,kvmeta").split(",") if x.strip())
            # Boot self-test: one more eager warmup vs a replay of the fresh
            # graph, same inputs -> logits must match. Divergence here means
            # the capture itself broke the forward (pool/config freeze),
            # independent of any scheduler staging.
            with torch.cuda.device(self.device):
                with torch.cuda.stream(_cs):
                    _pool = get_global_ctx().linear_state_pool
                    _snap = _pool.snapshot_slot(0)

                    def _eager_ref():
                        with gctx.forward_batch(batch):
                            model.forward()
                        torch.cuda.synchronize()
                        return self.logits_out.clone(), {
                            d: t.clone() for d, t in self.taps.items()}

                    def _replay_cmp(ref, ref_taps, tag):
                        _pool.restore_slot(0, _snap)
                        graph.replay()
                        torch.cuda.synchronize()
                        _d = (self.logits_out.float() - ref.float()).abs().max().item()
                        _n = int(self.logits_out.isnan().sum())
                        _td = {d: round((self.taps[d].float() - ref_taps[d].float())
                                        .abs().max().item(), 3) for d in self.taps}
                        print(f"[vr-selftest] {tag}: maxdiff={_d:.3f} nan={_n} "
                              f"tapdiff={_td}", flush=True)

                    # variant A: boot-dummy buffers
                    self._pace_ticks.tick() if _pace == "selftest" else None
                    if "dummy" in _stv:
                        _ref, _rt = _eager_ref()
                        _replay_cmp(_ref, _rt, "dummy")

                    def _stage_l66(do_pos=True, do_kv=True):
                        if do_kv:
                            self.kv_indptr[0].zero_()
                            self.kv_indptr[1].fill_(66)
                            self.prefix_lens[0].fill_(58)
                        if do_pos:
                            self.positions.copy_(torch.arange(58, 58 + self.k,
                                                              dtype=torch.int32,
                                                              device=self.device))

                    # variant B: both groups staged
                    self._pace_ticks.tick() if _pace == "selftest" else None
                    if "l66both" in _stv:
                        _stage_l66()
                        _ref, _rt = _eager_ref()
                        _e2, _t2 = _eager_ref()  # determinism control
                        _d_ee = (_ref.float() - _e2.float()).abs().max().item()
                        print(f"[vr-selftest] B eager_vs_eager maxdiff={_d_ee:.3f} "
                              f"(determinism control)", flush=True)
                        _replay_cmp(_ref, _rt, "L66both")
                    # KVSPLIT (FREETOKEN_SPEC_KVSPLIT=1): at L=66, compare the
                    # KV rows each side WRITES through out_loc (staged to the
                    # dummy req's scratch pages, one per block row). Stored rows
                    # differ => the STORE side under capture; identical => the
                    # GATHER-side read is the corruption.
                    if (_os_dbg.environ.get(
                            "FREETOKEN_SPEC_KVSPLIT", "0") == "1"
                            and "kvsplit" in _stv):
                        # ZEROINIT (FREETOKEN_SPEC_ZEROINIT=1): swap torch.empty
                        # /empty_like for zeros across the KVSPLIT runs. If the
                        # degraded eager control (4.3 on later captures) and the
                        # graph divergence collapse to ~0, an uninit-read in an
                        # empty-allocated workspace is CONFIRMED; site-bisect
                        # the zeroing next. Restored after.
                        _zi = _os_dbg.environ.get(
                            "FREETOKEN_SPEC_ZEROINIT", "0") in {"1", "true", "yes"}
                        if _zi:
                            _e_empty, _e_el = torch.empty, torch.empty_like
                            torch.empty = torch.zeros
                            torch.empty_like = torch.zeros_like
                            print("[KVSPLIT] ZEROINIT: empties zeroed", flush=True)
                        _kvc = attn_backend.kvcache
                        _layers_ok = []
                        for _li in (3, 7, 27):
                            try:
                                _kvc.k_cache(_li)
                                _layers_ok.append(_li)
                            except KeyError:
                                continue
                        for _li in _layers_ok:
                            _kc = _kvc.k_cache(_li)
                            _vc = _kvc.v_cache(_li)
                            _kc2 = _kc.view(-1, _kc.shape[-2], _kc.shape[-1])
                            _vc2 = _vc.view(-1, _vc.shape[-2], _vc.shape[-1])
                            # REAL ROWS (artifact #3 fix): the dummy row is
                            # SENTINEL-filled (fill_(num_tokens)) — staging
                            # from it gathered one-past-end rows (UB, the
                            # entire in-server control-wobble family). Stage
                            # onto REAL warmup-allocated pages instead:
                            # indices -> pages [0, 66+k), out_loc -> pages
                            # [66+k, 66+2k). All touched rows snapshotted and
                            # rewound between runs; gather and store disjoint.
                            _nk = 66 + self.k

                            def _stage_clean():
                                _stage_l66()
                                self.indices[:_nk].copy_(
                                    torch.arange(0, _nk, dtype=torch.int32,
                                                 device=self.device))
                                self.out_loc.copy_(
                                    torch.arange(_nk, _nk + self.k,
                                                 dtype=torch.int32,
                                                 device=self.device))

                            _rows = torch.arange(0, _nk + self.k,
                                                 device=self.device)

                            def _snap_rows():
                                return _kc2[_rows].clone(), _vc2[_rows].clone()

                            def _rewind(rk, rv):
                                _kc2[_rows].copy_(rk)
                                _vc2[_rows].copy_(rv)

                            _pool.restore_slot(0, _snap)
                            _stage_clean()
                            _rk0, _rv0 = _snap_rows()
                            _eager_ref()
                            _rkE, _rvE = _snap_rows()
                            _rewind(_rk0, _rv0)
                            _pool.restore_slot(0, _snap)
                            _stage_clean()
                            _eager_ref()
                            _rkE2, _rvE2 = _snap_rows()
                            _dEE = max(
                                (_rkE.float() - _rkE2.float()).abs().max().item(),
                                (_rvE.float() - _rvE2.float()).abs().max().item())
                            _rewind(_rk0, _rv0)
                            _pool.restore_slot(0, _snap)
                            _stage_clean()
                            graph.replay()
                            torch.cuda.synchronize()
                            _rkG, _rvG = _snap_rows()
                            _dk = (_rkE.float() - _rkG.float()).abs().max().item()
                            _dv = (_rvE.float() - _rvG.float()).abs().max().item()
                            print(f"[KVSPLIT] layer={_li} "
                                  f"control_eager_vs_eager={_dEE:.4f} | "
                                  f"stored-KV K={_dk:.4f} V={_dv:.4f} "
                                  f"-> {'STORE-SIDE' if max(_dk, _dv) > 0.05 else 'GATHER-SIDE'}",
                                  flush=True)
                        if _zi:
                            torch.empty, torch.empty_like = _e_empty, _e_el
                    # variant C: positions only (RoPE path)
                    self._pace_ticks.tick() if _pace == "selftest" else None
                    if "pos" in _stv:
                        _pool.restore_slot(0, _snap)
                        _stage_l66(do_pos=False)
                        _ref, _rt = _eager_ref()
                        _replay_cmp(_ref, _rt, "pos_only")
                    # variant D: KV metadata only
                    self._pace_ticks.tick() if _pace == "selftest" else None
                    if "kvmeta" in _stv:
                        _pool.restore_slot(0, _snap)
                        _stage_l66(do_kv=False)
                        self.positions.copy_(torch.arange(0, self.k, dtype=torch.int32,
                                                          device=self.device))
                        _ref, _rt = _eager_ref()
                        _replay_cmp(_ref, _rt, "kvmeta_only")
        if _os_dbg.environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
            print(f"[vr-cap] sink_sum={float(self.logits_out.abs().sum()):.1f} "
                  f"row0_top={int(self.logits_out[0].argmax())}", flush=True)
        self.graph = graph
        gctx.spec_tap_dev = None
        gctx.spec_logits_sink = None  # eager forwards rebind as before
        gctx.spec_nan_sink = None
        # WARMUP REPLAY (FREETOKEN_SPEC_WARMREPLAY=1): one replay with REAL
        # page staging right after capture. Correlates 4/4 with a correct
        # FIRST production replay (boots running the real-rows KVSPLIT probe
        # — which does exactly this — always drew exact step-0s; without it,
        # wrong from replay #1). The staging writes KV into scratch pages
        # [66+k, 66+2k) and gathers [0, 66+k) — warmup-owned pages at boot;
        # later captures leave harmless leftovers on scratch-range pages.
        if __import__("os").environ.get(
                "FREETOKEN_SPEC_WARMREPLAY", "0") in {"1", "true", "yes"}:
            _k = self.k
            _nk = 66 + _k
            # one EAGER pass first, then the replay — mirroring the probe
            # sequence that correlates 4/4 with exact step-0s. Hypothesis:
            # the eager forward's prepare_metadata populates backend-shared
            # metadata buffers that the captured graph reads (captured as
            # views into them at capture time); cold buffers = garbage
            # replay, warmed = exact.
            self.input_ids.zero_()
            with gctx.forward_batch(batch):
                model.forward()
            torch.cuda.synchronize()
            self.positions.copy_(
                torch.arange(66, 66 + _k, dtype=torch.int32,
                             device=self.device))
            self.out_loc.copy_(
                torch.arange(_nk, _nk + _k, dtype=torch.int32,
                             device=self.device))
            self.indices[:_nk].copy_(
                torch.arange(0, _nk, dtype=torch.int32, device=self.device))
            self.kv_indptr[0].zero_()
            self.kv_indptr[1].fill_(_nk)
            self.prefix_lens[0].fill_(66)
            self.linear_idx[0].zero_()
            with torch.cuda.stream(self.replay_stream):
                graph.replay()
                self.replay_stream.synchronize()
            if __import__("os").environ.get("FREETOKEN_SPEC_DBG", "0") == "1":
                print(f"[vr-cap] warmup eager+replay done "
                      f"(row0_top={int(self.logits_out[0].argmax())})",
                      flush=True)

    @torch.inference_mode()
    def replay_step(self, *, z_ids, slot: int, L: int, page_row, stream):
        """Stage one verify block and replay. ``page_row`` is the request's
        page-table row (device, int32); positions are L..L+k-1."""
        import torch

        k = self.k
        self.input_ids.copy_(z_ids, non_blocking=False)
        self.positions.copy_(torch.arange(L, L + k, dtype=torch.int32,
                                          device=self.device))
        self.out_loc.copy_(page_row[L : L + k])
        self.indices[: L + k].copy_(page_row[: L + k])
        self.kv_indptr[0].zero_()
        self.kv_indptr[1].fill_(L + k)
        self.prefix_lens[0].fill_(L)
        self.linear_idx[0].fill_(slot)
        if __import__("os").environ.get("FREETOKEN_SPEC_PACE", "") == "prereplay":
            __import__("time").sleep(0.02)
        if self.pw_graphs is not None:
            self._replay_pw()
        else:
            self.graph.replay()


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.stream = stream
        self.device = device
        # Piecewise mode (SSD expert tier): capture segments that end at each
        # MoE layer's ensure/copy seam instead of one monolithic graph, so the
        # host-driven miss copies can run between replays (see engine/piecewise).
        self.split_graph = (
            __import__("os").environ.get("FREETOKEN_SPLIT_GRAPHS", "0")
            in {"1", "true", "yes"}
        )
        self.split_map = {}
        self.piecewise = bool(
            self.max_graph_bs > 0
            and moe_offload_cache is not None
            and getattr(moe_offload_cache, "quant_format", "") == "ggml_file"
        )
        self.pw_map: Dict[int, List[torch.cuda.CUDAGraph]] = {}
        self.pw_seams: List[int] = []
        # split-graph mode needs the model at replay (eager head tail)
        self.model = model if self.split_graph else None
        self.max_seq_len = max_seq_len
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_runner = None
        if self.max_graph_bs == 0:
            # decode graphs disabled, but the S4 verify graph is independent
            import os as _os

            _k = int(_os.environ.get("FREETOKEN_SPEC_K", "0") or 0)
            if _k > 0:
                self._capture_verify(_k, model)
            return logger.info_rank0("CUDA graph is disabled.")

        # Capture the VERIFY graph BEFORE the decode bs graphs: running it
        # after left the attention backend in decode-capture state
        # (init_capture_graph + TritonCaptureData) and the verify capture
        # baked wrong logits (acceptance=0, garbage rows -- gate-green in the
        # max_bs=0 path where no decode capture preceded it).
        import os as _os_vfirst

        _kvf = int(_os_vfirst.environ.get("FREETOKEN_SPEC_K", "0") or 0)
        if _kvf > 0:
            self._capture_verify(_kvf, model)

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = None
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                if self.piecewise:
                    from freetoken.engine.piecewise import PiecewiseCapture

                    cache = self.moe_offload_cache
                    assert cache is not None and cache.quant_format == "ggml_file"
                    cache.suppress_inline_copy = True
                    try:
                        cap = PiecewiseCapture(self.stream, pool=pool)
                        cap.capture(
                            lambda: self.buffer.logits.__setitem__(
                                slice(0, bs), model.forward()
                            )
                        )
                        if pool is None:
                            pool = cap.pool
                        if not self.pw_seams:
                            self.pw_seams = list(cap.seams)
                        else:
                            assert self.pw_seams == cap.seams, (
                                "piecewise segment layout changed between batch sizes"
                            )
                        assert len(cap.graphs) == len(self.pw_seams) + 1, (
                            len(cap.graphs), len(self.pw_seams)
                        )
                        self.pw_map[bs] = cap.graphs
                    finally:
                        cache.suppress_inline_copy = False
                elif self.split_graph:
                    from freetoken.engine.split_graph import SplitGraphCapture

                    cap = SplitGraphCapture(self.stream)
                    cap.capture(
                        bs,
                        lambda: self.buffer.logits.__setitem__(
                            slice(0, bs), model.forward()
                        ),
                        batch=batch,
                    )
                    self.split_map[bs] = cap
                else:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if self.split_graph and bs in self.split_map:
                pass  # split graphs own their per-segment pools
            elif pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            if graph is not None:
                self.graph_map[bs] = graph

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_verify(self, k: int, model, stream=None) -> None:
        import json as _json
        import os as _os

        from freetoken.engine.graph import VerifyGraphRunner

        tap_layers = _json.loads(_os.environ.get("FREETOKEN_DFLASH_TAP_LAYERS", "[6, 20, 34, 48, 62]"))
        _cfg = getattr(model, "config", None)
        hidden = int(getattr(_cfg, "hidden_size", 0) or 5120)
        vocab = int(getattr(_cfg, "vocab_size", 0) or 248320)
        vr = VerifyGraphRunner(k, self.max_seq_len, vocab, hidden, self.device, tap_layers)
        vr.capture(self.attn_backend, model, self.dummy_req, stream=stream,
                   moe_cache=self.moe_offload_cache)
        self.verify_runner = vr
        logger.info_rank0(f"verify graph captured (k={k})")

        # Alternation probe (FREETOKEN_SPEC_VR2=1): a SECOND verify graph with
        # its own private memory pool. Deep (>=16k) evidence so far: on the SAME
        # graph, a second replay in one step faults (restore variant) or wedges
        # (raw variant) while single-replay-per-step production survives. If
        # alternating vr/vr2 replays completes and agrees at depth, per-graph
        # pool reuse is the bug -> fix is double-buffering or per-request
        # recapture. Only captured once: _spec_recapture calls back into here,
        # and re-adding a second live pool mid-serving would double the cost.
        import os as _os2

        if (int(_os2.environ.get("FREETOKEN_SPEC_VR2", "0") or 0) > 0
                and getattr(self, "verify_runner2", None) is None):
            vr2 = VerifyGraphRunner(k, self.max_seq_len, vocab, hidden,
                                    self.device, tap_layers)
            vr2.capture(self.attn_backend, model, self.dummy_req, stream=stream,
                        moe_cache=self.moe_offload_cache)
            self.verify_runner2 = vr2
            logger.info_rank0("verify graph #2 captured (VR2 alternation probe)")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        self.attn_backend.prepare_for_replay(batch)
        if self.split_graph and batch.padded_size in self.split_map:
            cap = self.split_map[batch.padded_size]
            cap.graphs["near"].replay()
            # eager seam: near tail -> pinned -> persistent dev1 buffers, and
            # refresh the far graph's fixed metadata twins from this batch.
            from freetoken.engine.layer_split import (
                cross_seam_meta_replay,
                cross_seam_replay,
            )

            cross_seam_meta_replay(batch)
            cross_seam_replay()
            import os as _ose

            if _ose.environ.get("FREETOKEN_SPLIT_FAR_EAGER", "0") in {"1", "true", "yes"}:
                out_dev1 = self.model.model.forward_far_eager(batch)
                from freetoken.engine.layer_split import split_tail_forward

                self.buffer.logits[: batch.size] = split_tail_forward(
                    self.model, out_dev1
                )
                return self.buffer.logits[: batch.size]
            cap.graphs["far"].replay()
            import os as _os

            if _os.environ.get("FREETOKEN_SPLIT_TRACE"):
                fo = cap.far_output()
                if fo is not None:
                    fh = hash(fo[:1, :16].float().cpu().numpy().tobytes()) & 0xFFFFFF
                    print(f"[seam-replay] far_out_hash={fh:06x}", flush=True)
            # eager tail (outside graphs by design): far hidden -> head ->
            # pinned crossing -> near logits buffer. The far trunk output is
            # whatever tensor the far graph's norm wrote (kept alive in
            # _SPLIT_DST); run the head exactly as CausalLM.forward does.
            from freetoken.engine.layer_split import split_tail_forward

            out = split_tail_forward(self.model, cap.far_output())
            self.buffer.logits[: batch.size] = out
            return self.buffer.logits[: batch.size]
        if self.piecewise:
            cache = self.moe_offload_cache
            segments = self.pw_map[batch.padded_size]
            cache.suppress_inline_copy = True
            try:
                segments[0].replay()
                for i, layer_id in enumerate(self.pw_seams):
                    # Host-driven miss fetch for this layer (staged through the
                    # pinned ring), then the segment holding its expert GEMM.
                    cache.copy_missing_staged(layer_id)
                    segments[i + 1].replay()
            finally:
                cache.suppress_inline_copy = False
        else:
            g = self.graph_map[batch.padded_size]
            g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.pw_map = {}
        self.split_map = {}
        self.buffer = None
        gc.collect()
