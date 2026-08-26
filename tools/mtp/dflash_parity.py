"""DFlash2 parity harness: our standalone draft vs the z-lab reference.

Loads the SAME iter_dflash_weights state into BOTH implementations, feeds
identical synthetic inputs (noise block + tap-ctx feature + positions +
probe head), and reports per-component cosine. Also resolves the
`output_norm` vs `enc.output_norm` role ambiguity by trying both assignments.
"""
import sys
from types import SimpleNamespace

import torch

FT = "/home/scooter/Documents/Product/Stan-s-ML-Stack/Fork/FreeToken/python"
REF = "/home/scooter/Documents/Product/Stan-s-ML-Stack/Fork/dflash-reference"
sys.path.insert(0, FT)
sys.path.insert(0, REF)

GGUF = "/mnt/HDD-2/Models/incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf"

from freetoken.models.dflash.gguf import parse_dflash_config, iter_dflash_weights
from freetoken.models.gguf.reader import load_gguf_metadata


def build_state(norm_swap: bool):
    sd = {}
    for k, v in iter_dflash_weights(GGUF):
        if norm_swap:
            # swap only the two global roles
            if k == "norm.weight":
                k = "hidden_norm.weight"
            elif k == "hidden_norm.weight":
                k = "norm.weight"
        sd[k] = v.clone()
    return sd


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


def main():
    md = load_gguf_metadata(GGUF)
    cfg = parse_dflash_config(md)
    vocab_size = 248320
    dev = "cpu"  # reference imports transformers Qwen3; CPU keeps this light

    # ---------- ours ----------
    from freetoken.models.dflash.model import DFlash2Draft as Ours
    ours = Ours(cfg, vocab_size).to(torch.bfloat16).eval()

    # ---------- theirs ----------
    from dflash.model import DFlash2DraftModel  # z-lab reference classes

    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    rcfg = Qwen3Config(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=1e7,
        rope_scaling=None,
        max_position_embeddings=262144,
        sliding_window=cfg["sliding_window"],
        layer_types=["sliding_attention"] * cfg["num_layers"],
        is_causal=False,
        vocab_size=vocab_size,
        hidden_act="silu",
    )
    rcfg.dflash_config = {
            "block_size": cfg["block_size"],
            "conv_kernel_size": cfg["conv_kernel_size"],
            "conv_group_size": cfg["conv_group_size"],
            "selector_rank": cfg["selector_rank"],
            "selector_top_k": cfg["selector_top_k"],
            "target_layer_ids": list(cfg["target_layers"]),
            "mask_token_id": 248070,
            "input_embedding_scale": 1.0,
        }
    # transformers' config wrapper can shadow arbitrary dict attrs; mirror
    # every draft knob at top level too (reference getters fall back here)
    for key, val in dict(rcfg.dflash_config).items():
        setattr(rcfg, key, val)
    # the reference's eager default reads this even when target_layer_ids wins
    rcfg.num_target_layers = 64
    ref = DFlash2DraftModel(rcfg).to(torch.bfloat16).eval()
    # reference attention drops straight to sdpa; ensure non-causal windows used
    for lay in ref.layers:
        lay.self_attn.is_causal = False

    # ---------------- inputs ----------------
    g = torch.Generator().manual_seed(0)
    B, Tc, Tq = 1, 4, 3
    taps = torch.randn(B, len(cfg["target_layers"]), Tc, cfg["hidden_size"], generator=g).to(torch.bfloat16)
    target_hidden = taps.reshape(B, Tc, -1)  # concat along last dim
    noise = torch.randn(B, Tq, cfg["hidden_size"], generator=g).to(torch.bfloat16)
    base_pos = 1000
    positions_window = torch.arange(base_pos - Tc, base_pos + Tq)

    probe_head_w = (torch.randn(vocab_size, cfg["hidden_size"], generator=g) * 0.02).to(torch.bfloat16)

    def head(hidden):
        return hidden @ probe_head_w.t()

    results = {}
    for swap in (False, True):
        tag = "SWAPPED" if swap else "default"
        sd = build_state(swap)
        missing, unexpected = ours.load_state_dict(sd, strict=True), None
        r_missing, r_unexpected = ref.load_state_dict(
            {k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=True)

        with torch.inference_mode():
            cos_t, sin_t = ours.rope_tables(positions_window)
            oh, okv = ours(noise, target_hidden, positions_window)
            rpast = None
            rh = noise
            pos_full = positions_window[None]
            position_embeddings = ref.rotary_emb(rh, pos_full)
            tctx = ref.hidden_norm(ref.fc(target_hidden))
            for i, layer in enumerate(ref.layers):
                rh = layer(
                    hidden_states=rh,
                    target_hidden=tctx,
                    position_ids=pos_full[None],
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
            rh = ref.norm(rh)

            cos_h = cosine(oh[:, -1], rh[:, -1])
            oh_logits = head(oh)
            rh_logits = head(rh)
            cos_l = cosine(oh_logits, rh_logits)

            anchor = torch.randint(0, vocab_size, (B,), generator=g)
            opath, ocand = ours.propose_greedy(oh, anchor, head)
            sel = ref.candidate_selector.select(rh, rh_logits, anchor, 0.0)
            rpath = sel[0]
            path_eq = bool((opath == rpath).all())
            print(f"[{tag}] hidden_cos={cos_h:.6f} logits_cos={cos_l:.6f} "
                  f"path_equal={path_eq} ours={opath[0].tolist()} ref={rpath[0].tolist()}")
            results[tag] = (cos_h, cos_l, path_eq)

    best = max(results.items(), key=lambda kv: abs(kv[1][0]))
    print(f"\nBEST NORM ASSIGNMENT: {best[0]} (cos={best[1][0]:.6f})")


if __name__ == "__main__":
    main()
