"""Offline MTP draft replay from raw GGUF weights (bf16) vs traced (h, token).

Discriminates: probe/quant/pack machinery bug vs weight-semantics bug vs
untrained draft head. Replays the draft layer exactly as llama.cpp
qwen35moe_mtp does: eh_proj(cat(enorm(emb), hnorm(h))) -> full-attn block
(per-head q|gate, qk-norm, partial NeoX rope rd=64 base=1e7, sigmoid gate)
-> MoE (top-8 renorm + gated shared) -> shared_head_norm -> lm_head.

KV attention is suffix-only (only the traced steps are available), so scores
are approximate -- enough to tell "predictions track actual next tokens"
from "uniform garbage".
"""
import glob
import math
import os
import torch

os.environ.setdefault("FREETOKEN_MTP", "1")

from freetoken.models.gguf.reader import iter_gguf_tensors
from freetoken.models.gguf.dequant import dequantize

MP = "/mnt/HDD-2/Models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf"
H, NQ, NKV, HD, RD, BASE = 2048, 16, 2, 256, 64, 1e7
E, TOPK, FF = 256, 8, 512
EPS = 1e-6
DEV = "cpu"
F = torch.float32

print("scanning gguf tensors ...", flush=True)
want_exact = {"token_embd.weight", "output.weight"}
want_blk = ("blk.40.",)
T = {}
for t in iter_gguf_tensors(MP):
    n = t.name
    if n in want_exact or n.startswith(want_blk):
        T[n] = t
print("kept:", len(T), flush=True)


def dq(t):
    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16).reshape(t.shape).to(DEV)


def rmsnorm(x, w):  # w is the pre-baked (1+w) vector
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)) * w.float()


W = {}
for n in [
    "blk.40.nextn.eh_proj.weight", "blk.40.nextn.enorm.weight",
    "blk.40.nextn.hnorm.weight", "blk.40.nextn.shared_head_norm.weight",
    "blk.40.attn_norm.weight", "blk.40.post_attention_norm.weight",
    "blk.40.attn_q.weight", "blk.40.attn_k.weight", "blk.40.attn_v.weight",
    "blk.40.attn_output.weight", "blk.40.attn_q_norm.weight", "blk.40.attn_k_norm.weight",
    "blk.40.ffn_gate_inp.weight", "blk.40.ffn_gate_inp_shexp.weight",
    "blk.40.ffn_gate_shexp.weight", "blk.40.ffn_up_shexp.weight", "blk.40.ffn_down_shexp.weight",
]:
    W[n] = dq(T[n])
    print("  ", n, tuple(W[n].shape), flush=True)

EMB = dq(T["token_embd.weight"])
LM = dq(T["output.weight"])
print("emb", EMB.shape, "lm", LM.shape, flush=True)

exp_names = {
    "gate": "blk.40.ffn_gate_exps.weight",
    "up": "blk.40.ffn_up_exps.weight",
    "down": "blk.40.ffn_down_exps.weight",
}
exp_packed = {k: T[v].packed().reshape(-1) for k, v in exp_names.items()}
exp_type = {k: T[v].ggml_type for k, v in exp_names.items()}
# per-expert packed size (torch layout [E, in, out]; ggml flat = [E][in][out])
per_exp = {k: v.numel() // E for k, v in exp_packed.items()}
exp_cache: dict = {}


def expert(i):
    if i in exp_cache:
        return exp_cache[i]
    mats = {}
    for k in exp_names:
        b = exp_packed[k][i * per_exp[k]: (i + 1) * per_exp[k]]
        d = dequantize(b, exp_type[k], torch.bfloat16).float()
        # ggml rows: gate/up [FF out rows, H in]; down [H out rows, FF in]
        mats[k] = d.reshape(FF, H) if k != "down" else d.reshape(H, FF)
    exp_cache[i] = mats
    return mats


inv = 1.0 / (BASE ** (torch.arange(0, RD, 2, dtype=F) / RD))


def rope(x, pos):  # x [heads, HD] float32, NeoX rotate-half on first RD dims
    fr = (pos * inv).to(F)
    cos, sin = fr.cos(), fr.sin()
    xr = x[:, :RD].clone()
    x1, x2 = xr[:, : RD // 2], xr[:, RD // 2:]
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([rot, x[:, RD:]], dim=-1)


def moe(x):
    logits = W["blk.40.ffn_gate_inp.weight"].float() @ x          # [E]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, TOPK)
    wts = top.values / top.values.sum()                            # norm_topk_prob
    routed = torch.zeros(H, dtype=F)
    for wi, ei in zip(wts.tolist(), top.indices.tolist()):
        m = expert(ei)
        g = m["gate"] @ x
        u = m["up"] @ x
        routed += wi * (m["down"] @ (torch.nn.functional.silu(g) * u))
    gsh = W["blk.40.ffn_gate_shexp.weight"].float() @ x
    ush = W["blk.40.ffn_up_shexp.weight"].float() @ x
    shared = W["blk.40.ffn_down_shexp.weight"].float() @ (torch.nn.functional.silu(gsh) * ush)
    shared = shared * torch.sigmoid(W["blk.40.ffn_gate_inp_shexp.weight"].float() @ x)
    return routed + shared


def draft_step(h, tok, pos, Kc, Vc, order="eh"):
    e = EMB[tok].float()
    hn = rmsnorm(h, W["blk.40.nextn.hnorm.weight"])
    en = rmsnorm(e, W["blk.40.nextn.enorm.weight"])
    cat = torch.cat([en, hn]) if order == "eh" else torch.cat([hn, en])
    x0 = W["blk.40.nextn.eh_proj.weight"].float() @ cat
    x = rmsnorm(x0, W["blk.40.attn_norm.weight"])
    qf = W["blk.40.attn_q.weight"].float() @ x                    # [NQ*HD*2]
    qg = qf.view(NQ, 2 * HD)
    q, gate = qg[:, :HD].clone(), qg[:, HD:].clone()
    k = (W["blk.40.attn_k.weight"].float() @ x).view(NKV, HD).clone()
    v = (W["blk.40.attn_v.weight"].float() @ x).view(NKV, HD)
    q = rmsnorm(q, W["blk.40.attn_q_norm.weight"])
    k = rmsnorm(k, W["blk.40.attn_k_norm.weight"])
    q, k = rope(q, pos), rope(k, pos)
    Kc = torch.cat([Kc, k[None]]) if Kc is not None else k[None]   # [T,NKV,HD]
    Vc = torch.cat([Vc, v[None]]) if Vc is not None else v[None]
    rep = NQ // NKV
    Kx = Kc.repeat_interleave(rep, dim=1)                          # [T,NQ,HD]
    s = torch.einsum("qd,tqd->qt", q, Kx) / math.sqrt(HD)
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("qt,tqd->qd", p, Vc.repeat_interleave(rep, dim=1))
    o = o * torch.sigmoid(gate)
    attn = W["blk.40.attn_output.weight"].float() @ o.reshape(-1)
    xr = attn + x0
    x2 = rmsnorm(xr, W["blk.40.post_attention_norm.weight"])
    h2 = xr + moe(x2)
    lg = LM.float() @ rmsnorm(h2, W["blk.40.nextn.shared_head_norm.weight"])
    return lg, Kc, Vc


REQUANT_DOWN = os.environ.get("REQUANT_DOWN", "0") == "1"
if REQUANT_DOWN:
    from freetoken.models.gguf.dequant import requantize_q4_k
    for k in ("down",):
        raw = exp_packed[k]
        d32 = dequantize(raw, exp_type[k], torch.float32)
        rows = d32.reshape(E * FF, H)          # torch [in, out] rows
        q = requantize_q4_k(rows.reshape(-1).contiguous())
        exp_packed[k] = q
        exp_type[k] = 12
        per_exp[k] = q.numel() // E
    print("down requantized Q6_K -> Q4_K (bank emulation)", flush=True)

fs = sorted(glob.glob("/tmp/mtp_trace/step_*.pt"))
steps = [torch.load(f, map_location="cpu", weights_only=False) for f in fs]
print(f"traced steps: {len(steps)}", flush=True)

import torch as _t
for order in ("eh",):
    Kc = Vc = None
    hit = 0
    n = 0
    print(f"--- order {order} ---")
    for i, d in enumerate(steps):
        h = d["h"][0].float()
        lg, Kc, Vc = draft_step(h, d["token"], d["pos"], Kc, Vc, order=order)
        pred = int(lg.argmax())
        if i + 1 < len(steps):
            actual = int(steps[i + 1]["token"])
            top5 = lg.topk(5)
            in5 = actual in top5.indices.tolist()
            hit += int(pred == actual)
            n += 1
            live = d["logits"][0].float()
            cos = _t.nn.functional.cosine_similarity(live.unsqueeze(0), lg.unsqueeze(0)).item()
            if i < 6:
                print(f"  live-vs-offline logit cos {cos:.3f} live lstd {live.std().item():.2f} off lstd {lg.std().item():.2f}")
            if i < 6:
                print(f"  pos {d['pos']:4d} tok {d['token']:6d} pred {pred:6d} actual {actual:6d}"
                      f" {'HIT' if pred==actual else ''} {'top5' if in5 else ''}"
                      f" lmax {lg.max().item():.2f} lstd {lg.std().item():.2f}")
    print(f"order {order}: offline acceptance {hit}/{n} = {hit/max(n,1):.3f}", flush=True)
