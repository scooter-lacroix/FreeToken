"""S4 live DFlash2 draft service — the acceptance-verified proposal algorithm
(replay contract from tools/mtp/dflash_acceptance.py, S3d E[tok/verify]=3.70 @ k=8)
as an in-process engine on the SECOND GPU.

Device contract (S4 layout): trunk on cuda:0 (7900 XTX via CUDA_VISIBLE_DEVICES=1,0),
draft on cuda:1 (7800 XT). Every cross-device transfer is explicit and small:
taps [5,H] bf16 ~51 KB, top-64 ids/vals, one embedding row. No trunk weights are
materialized on the draft device — the unary side is the trunk's OWN top-64 logits
recorded at the anchor row, exactly what the replay consumed.

propose(anchor_id, position, taps, top_ids, top_vals) -> list[int] of k picks.
Threaded through the scheduler by the S4 verify loop (spec.py).
"""
from __future__ import annotations

import os
import time

import torch

GGUF_DRAFT = "/mnt/HDD-2/Models/incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf"
MASK_ID = 248070
VOCAB = 248320
DEFAULT_K = 8


class DFlashService:
    def __init__(self, k: int = DEFAULT_K, device: str | None = None):
        from freetoken.models.dflash.gguf import iter_dflash_weights, parse_dflash_config
        from freetoken.models.dflash.model import DFlash2Draft
        from freetoken.models.gguf.reader import load_gguf_metadata

        self.k = k
        import os as _os
        # Default now cuda:0 (the trunk device): every cuda:1 interaction from
        # this process -- launches, replays, even pinned readbacks -- costs
        # 7-56ms of ROCm cross-device submission latency (measured; the far-GPU
        # draft is dead until that's fixed at the driver level). Same-device
        # submissions are microseconds. FREETOKEN_SPEC_DEVICE overrides.
        self.device = torch.device(device or _os.environ.get("FREETOKEN_SPEC_DEVICE", "cuda:0"))
        cfg = parse_dflash_config(load_gguf_metadata(GGUF_DRAFT))
        self.tap_layers = list(cfg["target_layers"])
        self.draft = DFlash2Draft(cfg, vocab_size=VOCAB)
        self.draft.load_state_dict(dict(iter_dflash_weights(GGUF_DRAFT)), strict=True)
        self.draft.eval()
        # The selector codebooks (~2GB f32, the bulk of the GGUF) stay on CPU (the
        # chain runs there); ONLY the encoder parameters move to the device -- the
        # trunk GPU has ~70MB free after its pools, so an all-module .to() OOMs.
        with torch.no_grad():
            # bf16 encoder: halves the claim (the trunk's load-time requant needs
            # the headroom); bf16 only shifts WHICH drafts are proposed, never
            # acceptance correctness (the trunk verifies every draft).
            for name, param in self.draft.named_parameters():
                if "codebook" not in name:
                    param.data = param.data.to(self.device, dtype=torch.bfloat16)
            for name, buf in self.draft.named_buffers():
                buf.data = buf.data.to(self.device)
        # float32, matching the S3d replay formulation exactly (bf16 storage is a
        # post-parity optimization; the encoder is 5 small layers).
        self.sel = self.draft.candidate_selector
        # CPU mirrors for the chain: every cross-device submission from this
        # process costs ~7-12ms on this ROCm stack -- the 8-step chain runs on
        # the host against these (vocab x rank f32, ~254MB at rank 256).
        self.cbp_cpu = self.sel.predecessor_codebook.weight.detach().float().cpu()
        self.cbs_cpu = self.sel.successor_codebook.weight.detach().float().cpu()
        self._h_pin = None  # lazy: [k, rank] pinned, sized after first forward
        # MASK embedding gathered per-call from the trunk table (quant type varies);
        # only the anchor+MASK rows are ever needed, dequantized trunk-side.
        self.n_propose = 0
        self.ms = 0.0
        self.t_pre = 0.0
        self.t_fwd = 0.0
        self.t_chn = 0.0

    @torch.inference_mode()
    def propose(
        self,
        anchor_id: int,
        position: int,
        taps: dict[int, torch.Tensor],      # {tap_layer: [H] hidden on trunk device}
        top_ids: torch.Tensor,              # [64] trunk logits top-k ids (trunk device)
        top_vals: torch.Tensor,             # [64] trunk logits top-k vals (trunk device)
        embed_row: torch.Tensor,            # [H] anchor embedding row, bf16, trunk device
        mask_row: torch.Tensor,             # [H] MASK embedding row, bf16, trunk device
    ) -> list[int]:
        """One chained-greedy proposal. Bit-identical formulation to the S3d replay:
        noise rows [anchor, MASK*(k-1)] embedded TRUNK-side, context taps at the anchor
        position, rope window ending at the anchor position, selector bilinear scoring
        over the trunk's top-64 unary."""
        t0 = time.perf_counter()
        dev = self.device
        # Pinned staging for every cross-device input: pageable H2D and P2P-less
        # D2D each cost ~10ms of driver staging on this ROCm stack (58ms/propose
        # before). Route everything through persistent pinned host buffers with
        # non_blocking H2D; the final .tolist() is the single sync.
        if not hasattr(self, "_pin"):
            nl = len(self.tap_layers)
            H = taps[next(iter(taps))].numel()
            self._pin = {
                "th": torch.empty(nl, H, dtype=torch.bfloat16, pin_memory=True),
                "ids": torch.empty(64, dtype=torch.int64, pin_memory=True),
                "vals": torch.empty(64, dtype=torch.float32, pin_memory=True),
                "emb": torch.empty(H, dtype=torch.bfloat16, pin_memory=True),
                "th_dev": torch.empty(1, 1, nl * H, dtype=torch.float32, device=dev),
                "ids_dev": torch.empty(64, dtype=torch.int64, device=dev),
                "vals_dev": torch.empty(64, dtype=torch.float32, device=dev),
                "emb_dev": torch.empty(1, H, dtype=torch.float32, device=dev),
                "mask_dev": mask_row.to(dev, dtype=torch.float32).unsqueeze(0),
            }
        pin = self._pin
        pin["th"].copy_(torch.stack([taps[d].reshape(-1) for d in self.tap_layers]).to(torch.bfloat16))
        pin["ids"].copy_(top_ids.to("cpu", torch.int64) if top_ids.is_cuda else top_ids.to(torch.int64))
        pin["vals"].copy_(top_vals.to("cpu", torch.float32) if top_vals.is_cuda else top_vals.to(torch.float32))
        pin["emb"].copy_(embed_row.to("cpu", torch.bfloat16) if embed_row.is_cuda else embed_row.to(torch.bfloat16))
        pin["th_dev"].copy_(pin["th"].view(1, 1, -1).to(torch.float32), non_blocking=True)
        pin["ids_dev"].copy_(pin["ids"], non_blocking=True)
        pin["vals_dev"].copy_(pin["vals"], non_blocking=True)
        pin["emb_dev"].copy_(pin["emb"].to(torch.float32), non_blocking=True)
        th = pin["th_dev"]
        pos = torch.arange(position - 1, position + self.k, device=dev)
        _tA = time.perf_counter()
        # CUDA-graphed draft encoder on cuda:1: the 5-layer encoder is ~55 tiny
        # kernels, and launches into the far device cost ~1ms each from this
        # process -- graph replay is ONE launch. Inputs land in persistent
        # buffers (written from the pinned staging above); pos is a buffer too.
        g = getattr(self, "_graph", None)
        if g is None:
            with torch.cuda.device(dev):
                noise_buf = torch.empty(1, self.k, pin["emb_dev"].shape[-1],
                                        dtype=torch.bfloat16, device=dev)
                th_buf = torch.empty(pin["th_dev"].shape, dtype=torch.bfloat16, device=dev)
                pos_buf = torch.empty(self.k + 1, dtype=torch.int64, device=dev)
                side = torch.cuda.Stream()
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self.draft(noise_buf, th_buf, pos_buf)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, stream=side):
                    h_buf, _ = self.draft(noise_buf, th_buf, pos_buf)
                self._graph = gr
                self._gbuf = (noise_buf, th_buf, pos_buf, h_buf)
            g = gr
        noise_buf, th_buf, pos_buf, h = self._gbuf
        noise_buf.copy_(torch.cat([pin["emb_dev"], pin["mask_dev"].expand(self.k - 1, -1)]).unsqueeze(0))
        th_buf.copy_(th)
        pos_buf.copy_(pos)
        g.replay()
        _tB = time.perf_counter()
        proj = self.sel.hidden_projection(h[:, :]).float()                    # [1,k,rank]
        cbp = self.sel.predecessor_codebook.weight
        cbs = self.sel.successor_codebook.weight
        # CPU chain (see arm-time comment): h read back once via pinned buffer;
        # ids/vals/candidate scoring all host-side -- no cuda:1 submissions.
        if self._h_pin is None or self._h_pin.shape != proj.shape:
            self._h_pin = torch.empty(proj.shape, dtype=torch.float32, pin_memory=True)
        self._h_pin.copy_(proj, non_blocking=False)
        proj_c = self._h_pin
        ids_c = pin["ids"]
        vals_c = pin["vals"]
        cbp, cbs = self.cbp_cpu, self.cbs_cpu
        cand = cbs[ids_c]                                                     # [64,rank]
        picked = []
        pred = int(anchor_id)
        for pi in range(self.k):
            scores = vals_c + (cbp[pred] * proj_c[0, pi]) @ cand.T
            pick = int(ids_c[int(scores.argmax())])
            picked.append(pick)
            pred = pick
        self.n_propose += 1
        self.ms += (time.perf_counter() - t0) * 1000.0
        self.t_pre += (_tA - t0) * 1000.0
        self.t_fwd += (_tB - _tA) * 1000.0
        self.t_chn += (time.perf_counter() - _tB) * 1000.0
        if self.n_propose % 25 == 0:
            print(f"[dflash-svc] n={self.n_propose} pre={self.t_pre/self.n_propose:.1f}ms "
                  f"fwd={self.t_fwd/self.n_propose:.1f}ms chain={self.t_chn/self.n_propose:.1f}ms",
                  flush=True)
        return picked


_svc: DFlashService | None = None


def get_service(k: int = DEFAULT_K) -> DFlashService | None:
    """Singleton; None unless FREETOKEN_SPEC_K>0 and a second device exists."""
    global _svc
    if _svc is not None:
        return _svc
    if not torch.cuda.is_available():
        return None
    try:
        _svc = DFlashService(k=k)
        print(
            f"[dflash-svc] armed on {_svc.device} (k={k}, "
            f"taps={_svc.tap_layers})", flush=True,
        )
    except Exception as e:                                  # noqa: BLE001
        print(f"[dflash-svc] FAILED to arm: {e!r}", flush=True)
        _svc = None
    return _svc
