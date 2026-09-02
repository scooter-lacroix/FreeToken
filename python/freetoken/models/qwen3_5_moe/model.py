from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    _ph: dict = {}
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
                dense_quant=getattr(config, "dense_quant", "none"),
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        import os as _os

        _pt = _os.environ.get("FREETOKEN_PHASE_TIMING", "0") in {"1", "true", "yes"}
        if _pt:
            import time as _tt

            def _lap(key, _st=[None]):
                torch.cuda.synchronize()
                now = _tt.perf_counter()
                if _st[0] is not None:
                    Qwen3_5DecoderLayer._ph[key] = (
                        Qwen3_5DecoderLayer._ph.get(key, 0.0) + now - _st[0]
                    )
                _st[0] = now

            _lap(None)  # start the clock
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        if _pt:
            _lap("norm_in")
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        if _pt:
            _lap("linear_attn" if self._is_linear else "self_attn")
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        if _pt:
            _lap("norm_mid")
        hidden = self.mlp.forward(hidden)
        if _pt:
            _lap("mlp")
            Qwen3_5DecoderLayer._ph["_n"] = Qwen3_5DecoderLayer._ph.get("_n", 0) + 1
            if Qwen3_5DecoderLayer._ph["_n"] % 128 == 0:
                ph = {k: v for k, v in Qwen3_5DecoderLayer._ph.items() if k != "_n"}
                tot = sum(ph.values())
                parts = " ".join(f"{k}={v:.2f}s" for k, v in sorted(ph.items(), key=lambda kv: -kv[1]))
                print(
                    f"[layer-phase] n={Qwen3_5DecoderLayer._ph['_n']} total={tot:.2f}s {parts}",
                    flush=True,
                )
                for k in Qwen3_5DecoderLayer._ph:
                    if k != "_n":
                        Qwen3_5DecoderLayer._ph[k] = 0.0
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        if getattr(config, "lm_head_quant", "none") == "ggml_kquant":
            # GGUF: the (tied) embedding table stays in its native k-quant bytes;
            # rows dequantize on gather instead of a resident bf16 copy.
            from .ggml_dense import QuantGGMLEmbedding

            self.embed_tokens = QuantGGMLEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_far_eager(self, batch):
        """S2c discriminator: run ONLY the far portion eagerly (used when the
        near segment is graph-replayed but the far graph is under test)."""
        from freetoken.engine.layer_split import (
            dev1 as _dev1,
            _SPLIT_SRC,
            _seam_meta_twins,
        )

        # split_capture_close restored batch attrs to dev0 originals; the
        # eager far tail needs the dev1 twins again (same as capture-time).
        for attr, twin in _seam_meta_twins(batch).items():
            setattr(batch, attr, twin)
        # _SPLIT_SRC holds the NEAR-side (cuda:0) tail output — move to dev1.
        x = _SPLIT_SRC["x"].to(_dev1(), non_blocking=False)
        residual = _SPLIT_SRC.get("r")
        if residual is not None:
            residual = residual.to(_dev1(), non_blocking=False)
        layers = self.layers.op_list
        sa = 56
        with torch.cuda.device(_dev1()):
            for i in range(sa, len(layers)):
                x, residual = layers[i].forward(x, residual)
            xn, rn = self.norm.forward_add_residual(x, residual)
        return xn
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        from freetoken.engine.layer_split import split_at, split_enabled, cross_to_dev1

        import os as _os
        _dbg = _os.environ.get("FREETOKEN_LAYER_SPLIT_DEBUG", "0") in {"1", "true", "yes"}
        # DFlash2 tap capture (S3d acceptance probing): stash the hidden AFTER
        # each target depth into a module-held CPU dump dict, keyed by global
        # decoder-layer id. Decode-only (prefill taps are the same-for-all
        # batch noise the proposals never see). Default off.
        _taps = None
        if _os.environ.get("FREETOKEN_DFLASH_TAPS", "0") in {"1", "true", "yes"}:
            import json as _json

            _taps = {
                int(v) for v in _json.loads(
                    _os.environ.get(
                        "FREETOKEN_DFLASH_TAP_LAYERS",
                        "[6, 20, 34, 48, 62]",
                    )
                )
            }
            if not hasattr(self, "_dflash_tap_dump"):
                self._dflash_tap_dump = {}
            store = self._dflash_tap_dump

        def _tap(hidden, depth: int):
            """D2H into a per-depth LONG-LIVED pinned buffer (blocking). A raw
            ``.to("cpu")`` from inside the far device context hard-faults this
            HIP stack; pinned-staging copy is what the lm_head logits crossing
            already proves safe."""
            pins = getattr(self, "_dflash_tap_pins", None)
            if pins is None:
                pins = {}
                self._dflash_tap_pins = pins
            buf = pins.get(depth)
            if buf is None or buf.shape[0] < hidden.shape[0]:
                buf = torch.empty(
                    (max(hidden.shape[0], 1), *hidden.shape[1:]),
                    dtype=torch.bfloat16,
                    pin_memory=True,
                )
                pins[depth] = buf
            view = buf[: hidden.shape[0]]
            view.copy_(hidden.detach(), non_blocking=False)
            store[depth] = view.clone()
        sa = split_at() if split_enabled() else 0
        if sa:
            # Triton binds kernel modules + launches to the THREAD's current
            # device (never inferred from tensor args). NEAR layers run under
            # the ambient (cuda:0) context; the FAR block runs inside an
            # explicit cuda:1 context so no interior call can reset it, then
            # restores cuda:0 for the final norm + lm_head.
            from freetoken.engine.layer_split import (
                dev1 as _dev1,
                _to_device_deep,
                split_capture_active,
                split_capture_seam,
            )

            layers = self.layers.op_list
            for i in range(sa):
                x, residual = layers[i].forward(x, residual)
                if split_capture_active() and i == sa - 1:
                    split_capture_seam(sa)  # close NEAR segment (dev0 side)
                if _taps and i in _taps:
                    _tap(x, i)
            with torch.cuda.device(_dev1()):
                x, residual = cross_to_dev1(get_global_ctx().batch, x, residual)
                if split_capture_active():
                    split_capture_seam(sa)
                for i in range(sa, len(layers)):
                    x, residual = layers[i].forward(x, residual)
                    if _taps and i in _taps:
                        _tap(x, i)
                # trunk norm stays INSIDE the far device context: its Triton
                # launch under the near device would be a wrong-context launch
                xn, rn = self.norm.forward_add_residual(x, residual)
                if _dbg:
                    torch.cuda.synchronize()
                    print("[split-dbg] far trunk norm complete", flush=True)
        else:
            for i, layer in enumerate(self.layers.op_list):
                x, residual = layer.forward(x, residual)
                if _taps and i in _taps:
                    _tap(x, i)
            xn, rn = self.norm.forward_add_residual(x, residual)
        # Pre-final-norm trunk residual for the MTP head (llama.cpp feeds the
        # trunk's final residual to nextn hnorm). ONE stable buffer written by
        # slice: graph capture walks the batch-size list largest -> smallest
        # with an eager warmup forward before each level, so a per-shape
        # buffer would rebind at every level and ctx would end up pointing at
        # whichever level was captured LAST -- frozen for every other replay
        # width. Decode-only: prefill must NOT touch the channel (decode
        # replays never re-run Python, so a prefill clobber persists for the
        # whole request) and prefill widths could exceed the buffer, forcing a
        # rebind that orphans the captured copy_ targets.
        if split_enabled():
            from freetoken.engine.layer_split import (
                split_capture_active,
                split_capture_close,
                split_capture_far_output,
            )

            if split_capture_active():
                split_capture_far_output(xn)
                split_capture_close()
        ctx = get_global_ctx()
        if not getattr(ctx.batch, "is_prefill", False):
            buf = getattr(self, "_trunk_prenorm_buf", None)
            if buf is None:
                self._trunk_prenorm_buf = rn.new_empty(rn.shape)
                buf = self._trunk_prenorm_buf
            if rn.shape[0] <= buf.shape[0]:
                buf[: rn.shape[0]].copy_(rn)
                ctx.trunk_hidden_prenorm = buf
            else:
                ctx.trunk_hidden_prenorm = rn  # oversized eager decode
        return xn


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        lm_q = getattr(config, "lm_head_quant", "none")
        if lm_q == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        elif lm_q == "ggml_kquant":
            from .ggml_dense import GgufKQuantLMHead

            self.lm_head = GgufKQuantLMHead(config.vocab_size, config.hidden_size)
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()
        if __import__("os").environ.get("FREETOKEN_MTP", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            # nextn (MTP) draft head; weights arrive under draft.* when the
            # GGUF loader runs with the same env (see gguf.iter_gguf_weights)
            from .mtp import Qwen35MTPDraft

            self.draft = Qwen35MTPDraft(config)

    def forward(self) -> torch.Tensor:
        import os as _os_arm
        if (_os_arm.environ.get("FREETOKEN_DFLASH_ENGINE", "0") in {"1","true","yes"}
                and not hasattr(self, "_live_dflash_armed")):
            self._live_dflash_armed = True
            self._ensure_live_dflash()
        ctx = get_global_ctx()
        output = self.model.forward(ctx.batch.input_ids)
        from freetoken.engine.layer_split import split_enabled

        from freetoken.engine.layer_split import dev1 as _dev1

        if split_enabled() and output.device == _dev1():
            # layer split: the whole trunk tail + head run on cuda:1 (the
            # hidden NEVER crosses back). Head under the far device context
            # (Triton), then only the [T, vocab] logits cross (pinned
            # two-hop; D2H inside the far ctx).
            from freetoken.engine.layer_split import _pin_for, _trace

            with torch.cuda.device(output.device):
                _trace(f"head: entering lm_head with {tuple(output.shape)} on {output.device}")
                logits = self.lm_head.forward(output)
                _trace(f"head: logits {tuple(logits.shape)} on {logits.device}")
            near = ctx.batch.input_ids.device
            pin = _pin_for(None, logits.shape, logits.dtype)
            with torch.cuda.device(logits.device):
                pin.copy_(logits, non_blocking=False)
            _trace(f"head: logits staged to pinned ({logits.shape[0]} rows); crossing to {near}")
            logits = pin.to(near, non_blocking=False)
            _trace("head: logits on near side")
            return self._s3d_maybe_flush(ctx, logits)
        # Post-norm trunk hidden (MTP diagnostics; lm_head consumes `output`,
        # so it is live every replay). `output` itself is a per-capture-level
        # pool tensor -- publish it through the same stable slice-written
        # buffer pattern as the pre-norm channel. Decode-only, like the
        # pre-norm channel (a prefill clobber would persist across replays).
        if not getattr(ctx.batch, "is_prefill", False):
            buf = getattr(self, "_trunk_post_buf", None)
            if buf is None:
                self._trunk_post_buf = output.new_empty(output.shape)
                buf = self._trunk_post_buf
            if output.shape[0] <= buf.shape[0]:
                buf[: output.shape[0]].copy_(output)
                ctx.trunk_hidden = buf
            else:
                ctx.trunk_hidden = output  # oversized eager decode
        return self._s3d_maybe_flush(ctx, self.lm_head.forward(output))


    def _ensure_live_dflash(self):
        pr = getattr(self, "_live_dflash_obj", None)
        if pr is None:
            from freetoken.models.dflash.probe import maybe_probe

            pr = maybe_probe()
            if pr is not None:
                pr.ensure_target(self)
            self._live_dflash_obj = pr
        return pr

    def _live_dflash_step(self, ctx, logits):
        """S4 milestone-1: live draft proposal on cuda:1 (measure-only)."""
        tap_store = getattr(self.model, "_dflash_tap_dump", None)
        pr = self._live_dflash_obj
        if pr is None or not tap_store:
            return
        try:
            anchor = int(logits.argmax(-1).reshape(-1)[-1].item())
            pos = int(ctx.batch.positions.reshape(-1)[-1].item()) + 1
            _picked, report = pr.propose_and_score(tap_store, anchor, pos)
            if report:
                print(report, flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"[dflash-probe] step error {e!r}", flush=True)

    def _s3d_maybe_flush(self, ctx, logits):
        # S4 spec-driver bridge: the scheduler-side verify loop (FREETOKEN_SPEC_K)
        # reads the last forward's full logits + tap store + an embedding-row
        # callable from the ctx. Refs only -- no copies, no device traffic.
        try:
            ctx.spec_logits = logits
            ctx.spec_taps = getattr(self.model, "_dflash_tap_dump", None)
            if getattr(ctx, "spec_embed", None) is None:
                emb = self.model.embed_tokens

                def _rows(ids):
                    import torch as _t

                    from freetoken.kernel.gguf import ggml_dequantize

                    idx = _t.tensor(ids, device=emb.packed.device)
                    out = ggml_dequantize(
                        emb.packed[idx].contiguous(), int(emb.quant_type),
                        len(ids), int(emb.embedding_dim))
                    return out.to(_t.bfloat16)

                _mask_row = _rows([248070])[0]
                ctx.spec_embed = lambda aid: (_rows([aid])[0], _mask_row)
        except Exception:                                  # noqa: BLE001
            pass
        """S3d capture flush shared by both serving paths: on a T==1 decode
        step with taps armed, append one pickle line {position, taps[depth][H]
        bf16, argmax, topk_ids/vals(64)}. The replay teacher-verifies proposals
        against these recorded greedy continuations WITHOUT re-running the
        trunk."""
        tap_store = getattr(self.model, "_dflash_tap_dump", None)
        import os as _os2

        if (
            tap_store is not None
            and not getattr(ctx.batch, "is_prefill", False)
            and logits.shape[0] == 1
            and (dump_path := _os2.environ.get("FREETOKEN_DFLASH_TAP_DUMP"))
        ):
            entry = {
                "position": int(
                    ctx.batch.positions.reshape(-1)[-1].item()
                ),
                "taps": {
                    d: t.to(torch.bfloat16) for d, t in tap_store.items()
                },
                "argmax": int(logits.argmax(-1).reshape(-1)[-1].item()),
            }
            # selector-unary substitute: top-64 logits let the offline probe
            # run proposals WITHOUT the 248320-wide trunk head
            tk = torch.topk(logits.reshape(-1).float(), 64)
            entry["topk_ids"] = [int(v) for v in tk.indices.tolist()]
            entry["topk_vals"] = [float(v) for v in tk.values.tolist()]
            import pickle as _p

            with open(dump_path, "ab") as f:
                f.write(_p.dumps(entry))

        import os as _os3
        if (_os3.environ.get("FREETOKEN_DFLASH_ENGINE", "0") in {"1","true","yes"}
                and hasattr(self, "_live_dflash_armed")
                and not getattr(ctx.batch, "is_prefill", False)
                and logits.shape[0] == 1):
            self._live_dflash_step(ctx, logits)
        if (_os3.environ.get("FREETOKEN_SPEC_SMOKE", "0") in {"1","true","yes"}
                and not getattr(ctx.batch, "is_prefill", False)
                and logits.shape[0] == 1):
            self._s4_smoke_step(ctx, logits)
        return logits

    def _s4_smoke_step(self, ctx, logits):
        """S4 live smoke (measure-only): the DFlashService on the SECOND GPU proposes a
        k=8 chain after every convergent decode step, using the SAME inputs the S3d
        replay consumed (taps, position, anchor argmax, trunk top-64). Pending chains are
       scored against the real emitted stream -> live acceptance stats. Serving output is
        untouched; the FIRST exception disables the smoke (the old probe's per-step crash
        class must never reach serving)."""
        if getattr(self, "_s4_svc", None) is None:
            import os as _os0

            if _os0.environ.get("FREETOKEN_SPEC_K", "0").strip() not in {"", "0"}:
                return  # the scheduler verify driver owns proposals
            if getattr(self, "_s4_smoke_failed", False):
                return
            try:
                from freetoken.models.dflash.service import get_service

                svc = get_service(k=8)
                if svc is None:
                    self._s4_smoke_failed = True
                    return
                self._s4_svc = svc
                self._s4_pending = []
                self._s4_done = []
                self._s4_n = 0
                emb = self.model.embed_tokens
                from freetoken.kernel.gguf import ggml_dequantize
                import torch as _t

                mid = _t.tensor([248070], device=emb.packed.device)
                self._s4_mask_row = ggml_dequantize(
                    emb.packed[mid].contiguous(), int(emb.quant_type), 1,
                    int(emb.embedding_dim)).to(_t.bfloat16).reshape(-1)
                print("[s4-smoke] service armed (k=8)", flush=True)
            except Exception as e:                              # noqa: BLE001
                print(f"[s4-smoke] arm FAILED (disabled): {e!r}", flush=True)
                self._s4_smoke_failed = True
                return
        try:
            import torch as _t

            svc = self._s4_svc
            row = logits.reshape(-1).float()
            y = int(row.argmax().item())
            tk = _t.topk(row, 64)
            pos = int(ctx.batch.positions.reshape(-1)[-1].item())
            tap_store = getattr(self.model, "_dflash_tap_dump", None) or {}
            emb = self.model.embed_tokens
            from freetoken.kernel.gguf import ggml_dequantize

            aid = _t.tensor([y], device=emb.packed.device)
            emb_row = ggml_dequantize(
                emb.packed[aid].contiguous(), int(emb.quant_type), 1,
                int(emb.embedding_dim)).to(_t.bfloat16).reshape(-1)
            picked = svc.propose(
                y, pos, tap_store, tk.indices, tk.values,
                embed_row=emb_row, mask_row=self._s4_mask_row)
            if self._s4_n < 3 and __import__("os").environ.get(
                    "FREETOKEN_SPEC_SMOKE_DEBUG", "0") in {"1", "true", "yes"}:
                tn = {d: float(t.float().norm()) for d, t in tap_store.items()}
                print(
                    f"[s4-dbg] n={self._s4_n} pos={pos} y={y} "
                    f"top5={tk.indices[:5].tolist()} vals5={[round(v,2) for v in tk.values[:5].tolist()]} "
                    f"pick={picked[:4]} tapnorms={ {d: round(v,1) for d,v in tn.items()} } "
                    f"tapsizes={ {d: tuple(t.shape) for d,t in list(tap_store.items())[:2]} }",
                    flush=True,
                )
            # Tool contract (dflash_acceptance.py): chain from step s checks
            # picks[j] against argmax_{s+j}, starting with j=0 = THIS step's own
            # emission (the anchor). Evaluate pick[0] at creation, then extend.
            still = []
            for ent in self._s4_pending:
                hit = ent[0][ent[1]] == y
                if hit:
                    ent[1] += 1
                if not hit or ent[1] >= len(ent[0]):
                    self._s4_done.append(ent[1])
                else:
                    still.append(ent)
            m0 = 1 if picked[0] == y else 0
            if m0 == 0 or m0 >= len(picked):
                self._s4_done.append(m0)
            else:
                still.append([picked, m0])
            self._s4_pending = still
            self._s4_n += 1
            ev = int(__import__("os").environ.get("FREETOKEN_SPEC_SMOKE_EVERY", "200"))
            if self._s4_n % ev == 0 and self._s4_done:
                import statistics as st

                m = st.mean(self._s4_done[-ev * 2:])
                p0 = sum(v > 0 for v in self._s4_done[-ev * 2:]) / min(
                    len(self._s4_done), ev * 2)
                print(
                    f"[s4-smoke] n={self._s4_n} meanAcc={m:.3f} P(acc>0)={p0:.3f} "
                    f"E[tok/vfy]={1 + m:.2f} draftMs={svc.ms / max(svc.n_propose, 1):.1f}",
                    flush=True,
                )
        except Exception as e:                                  # noqa: BLE001
            print(f"[s4-smoke] disabled after error: {e!r}", flush=True)
            self._s4_svc = None
            self._s4_smoke_failed = True


__all__ = ["Qwen3_5MoEForCausalLM"]
