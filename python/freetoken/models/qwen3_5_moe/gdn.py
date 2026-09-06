from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearColParallelMerged

from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged
from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

from .gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla
_STAGED: dict = {}
_LAST_EAGER: dict = {}
from .quant_linear import make_replicated_quant


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class _GatedRMSNorm(BaseOP):
    """RMSNorm of x followed by a silu(z) gate (HF Qwen3_5MoeRMSNormGated).

    Uses the fused fla ``rms_norm_gated`` triton kernel (norm(x) * silu(z) in one
    kernel) instead of the unfused pow/mean/rsqrt/mul/silu chain, matching sglang's
    ``RMSNormGated`` -- collapses ~8 elementwise kernels per GDN layer into one."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation="silu",
        )


class Qwen3_5GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, expert_quant: str = "none",
        attn_quant: str = "none", dense_quant: str = "none",
    ):
        self.layer_id = layer_id
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # qkv|z carry a weight scale (block-fp8 weight_scale_inv, or per-tensor FP8
        # weight_scale); b|a stay bf16. Both quant modes therefore split the four-way
        # fusion into an fp8 qkvz GEMM + a bf16 ba GEMM (matches sglang/vLLM).
        self._block_fp8 = expert_quant == "fp8_block"
        self._pertensor_fp8 = attn_quant == "fp8_pertensor"
        self._fp8 = self._block_fp8 or self._pertensor_fp8

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if self._fp8:
            ColMerged = Fp8BlockColMerged if self._block_fp8 else Fp8PerTensorColMerged
            self.in_proj_qkvz = ColMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            if dense_quant == "ggml_kquant":
                # GGUF: native k-quant blocks in per-type modules (qkv / z / b|a --
                # llama.cpp mixes Q4_K and Q6_K within the four-way fusion). The
                # v-head regroupings are row permutations, so they apply to the
                # packed bytes unchanged.
                from .ggml_dense import QuantGgmlLinear

                self._in_proj_kquant = True
                self.in_proj_qkv = QuantGgmlLinear(self.conv_dim, hidden_size)
                self.in_proj_z = QuantGgmlLinear(self.value_dim, hidden_size)
                self.in_proj_ba = QuantGgmlLinear(2 * num_v_heads, hidden_size)
            else:
                self._in_proj_kquant = False
                self.in_proj = LinearColParallelMerged(hidden_size, self._in_proj_split, has_bias=False)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = _GatedRMSNorm(head_v_dim, eps=rms_norm_eps)
        # out_proj follows the checkpoint quant: block-fp8 / per-tensor-fp8 / compressed-tensors
        # NVFP4 (W4A16) / bf16. in_proj_* stay bf16 in every mode (above), so a compressed-tensors
        # NVFP4 checkpoint (attn_quant=="nvfp4") only makes out_proj native FP4.
        self.out_proj = None  # set below (quant dispatch)
        if dense_quant == "ggml_kquant":
            # ssm_out arrives REQUANTIZED to Q4_K in plain column order (the
            # loader regroups columns on the bf16 matrix before packing), so the
            # packed projection needs no activation-side permutation.
            from .ggml_dense import QuantGgmlLinear

            self.out_proj = QuantGgmlLinear(hidden_size, self.value_dim)
            self._out_col_perm = None
            self._out_nv_vd = None
        else:
            self.out_proj = make_replicated_quant(
                expert_quant, attn_quant, self.value_dim, hidden_size, has_bias=False
            )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        # max_seq_len from host-side metadata keeps this launch graph-capturable
        # (the kernel fallback derives it via a .item() device sync otherwise —
        # illegal inside the verify-graph capture). Fixed-shape verify batches
        # bake the constant; prefill callers pass their known max extend.
        _mq = getattr(getattr(get_global_ctx().batch, "attn_metadata", None), "max_q_len", None)

        def _conv_call():
            return causal_conv1d_varlen(
                x, self._conv_weight(), pool.conv_states[li],
                cu_seqlens, cache_indices, has_initial_state,
                batch=None, max_seq_len=_mq)

        # PIECEWISE-GDN seam (same class as the fla seam): the conv1d triton
        # launch runs UNRECORDED between graph segments during a piecewise
        # capture (Triton argument staging would freeze its conv-state reads
        # — the residual after the fla seam); the replay job re-runs it.
        from freetoken.engine.piecewise import (
            capture_seam as _seam,
            piecewise_capture_active as _pw_active,
        )

        if (_pw_active() and __import__("os").environ.get(
                "FREETOKEN_SPEC_PIECEWISE_GDN", "0") in {"1", "true", "yes"}):
            _seam(self.layer_id, (x,), job=_conv_call)
        out = _conv_call()
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._fp8:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        elif getattr(self, "_in_proj_kquant", False):
            conv_in = self.in_proj_qkv.forward(hidden_states)
            z = self.in_proj_z.forward(hidden_states)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)

        if batch.is_decode:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
            mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            def _decode_call():
                return gdn_decode_fla(
                    q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                    state_source=pool.recurrent_states[li],
                    indices=fla.cache_indices,
                    cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                )

            # PIECEWISE-GDN seam (same class as the prefill fla + conv1d
            # seams): unrecorded launch between segments; replay job re-runs.
            from freetoken.engine.piecewise import (
                capture_seam as _seam_d,
                piecewise_capture_active as _pw_active_d,
            )

            if (_pw_active_d() and __import__("os").environ.get(
                    "FREETOKEN_SPEC_PIECEWISE_GDN", "0") in {"1", "true", "yes"}):
                _seam_d(self.layer_id, (q, k, v, a, b), job=_decode_call)
            core_out = _decode_call()
        else:
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state)
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None

            def _fla_call():
                return gdn_prefill_chunk_fla(
                    q, k, v, g, beta,
                    state_source=pool.recurrent_states[li],
                    indices=fla.cache_indices,
                    cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                    return_h=track,
                )

            # PIECEWISE-GDN seam (active only during a piecewise capture
            # walk): close the segment BEFORE the fla triton call and open
            # the next AFTER — the launch runs UNRECORED (host/eager) between
            # segments, so Triton never argument-stages it into frozen graph
            # bytes (the captured-verify step-N defect). The replay job
            # re-runs the same call between segment replays.
            from freetoken.engine.piecewise import (
                capture_seam as _seam,
                piecewise_capture_active as _pw_active,
            )

            _pw_gdn = (
                _pw_active()
                and __import__("os").environ.get(
                    "FREETOKEN_SPEC_PIECEWISE_GDN", "0") in {"1", "true", "yes"}
            )
            if _pw_gdn:
                import os as _os_ji

                # STAGED-JOB (default when the GDN seam is on): copy the job
                # inputs into RUNNER-OWNED PERSISTENT buffers with a captured
                # IN-GRAPH copy_ just before the seam closes — the captured
                # segments write the graph-pool copies at replay, so eager
                # closure references never refresh; the bufs DO (each replay
                # re-executes the in-graph copy). The job reads the bufs.
                _key = ("gdn", self.layer_id)
                _st = _STAGED.get(_key)
                if _st is None:
                    _st = {
                        "q": torch.empty_like(q),
                        "k": torch.empty_like(k),
                        "v": torch.empty_like(v),
                        "g": torch.empty_like(g),
                    }
                    _STAGED[_key] = _st
                _st["q"].copy_(q)
                _st["k"].copy_(k)
                _st["v"].copy_(v)
                _st["g"].copy_(g)

                # OUTPUT staging too: the captured continuation (norm/
                # out_proj in the NEXT segment) must read a persistent buffer
                # that the replay job rewrites — otherwise it reads the frozen
                # capture-time output address (the constant-output root cause).
                _ob = _STAGED.get(("out", self.layer_id))
                _osh = (q.shape[1], v.shape[2], v.shape[3])  # [total, Hv, V]
                if _ob is None or tuple(_ob.shape) != _osh:
                    _ob = torch.empty(_osh, dtype=q.dtype, device=q.device)
                    _STAGED[("out", self.layer_id)] = _ob

                # FULL-TAIL JOB (JI2 verdict fix): the captured continuation
                # (norm+out_proj segment) after the seam was convicted of the
                # numeric divergence (L0 staged inputs bit-match eager, L1
                # diverge). Move the WHOLE tail — fla + norm + out_proj —
                # into the eager seam job; stage z too; the final layer
                # output lands in a persistent buffer the next layer's
                # captured segment reads. Captured region per GDN layer
                # shrinks to projections-only.
                _st["z"] = z.detach().clone()
                _st["beta"] = beta.detach().clone()
                _fkey = ("fout", self.layer_id)
                _fout = _STAGED.get(_fkey)
                _fshape = (z.shape[0], self.out_proj_weight_out_features
                           if hasattr(self, "out_proj_weight_out_features")
                           else None)

                def _tail(_r):
                    _co = _r[0] if track else _r
                    if track:
                        self._write_track_snapshot(
                            pool, li, conv_in, _r[1], fla)
                    _co = _co.reshape(-1, self.head_v_dim)
                    _zz = _st["z"].reshape(-1, self.head_v_dim)
                    return self.out_proj.forward(
                        self.norm.forward(_co, _zz).reshape(
                            _st["z"].shape[0], -1))

                def _fla_job():
                    import hashlib as _hl_fo

                    if not __import__("torch").cuda.is_current_stream_capturing():
                        _fo_h = _hl_fo.md5(
                            _fout.detach().float().cpu().numpy().tobytes()
                        ).hexdigest()[:8] if _fout is not None else "none"
                        print(f"[FO] L{self.layer_id} REPLAY-IN "
                              f"fout={_fo_h}", flush=True)
                    _r = gdn_prefill_chunk_fla(
                        _st["q"], _st["k"], _st["v"], _st["g"], _st["beta"],
                        state_source=pool.recurrent_states[li],
                        indices=fla.cache_indices,
                        cu_seqlens=fla.cu_seqlens,
                        scale=self.head_k_dim ** -0.5,
                        return_h=track,
                    )
                    _o = _tail(_r)
                    if _fout is not None and tuple(_fout.shape) == tuple(
                            _o.shape):
                        _fout.copy_(_o)
                    return _o

                _LAST_EAGER[self.layer_id] = (
                    q.detach().clone(), k.detach().clone(),
                    v.detach().clone(), g.detach().clone())
                _seam(self.layer_id, (q, k, v, g, beta), job=_fla_job)
                _r_cap = _fla_call()
                _o_cap = _tail(_r_cap)
                if _fout is None or tuple(_fout.shape) != tuple(
                        _o_cap.shape):
                    _fout = _o_cap.detach().clone()
                    _STAGED[_fkey] = _fout
                else:
                    _fout.copy_(_o_cap)
                return _fout  # next layer's captured segment reads the buffer
            else:
                _LAST_EAGER[self.layer_id] = (
                    q.detach().clone(), k.detach().clone(),
                    v.detach().clone(), g.detach().clone())
                result = _fla_call()
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["Qwen3_5GatedDeltaNet"]
