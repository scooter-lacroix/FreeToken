"""Stage-by-stage diff: live probe intermediates (trace 'mid') vs offline replay.

Recomputes the draft forward from raw GGUF weights for each traced (h, token)
and reports per-stage cosine/relative error against the live dump. The first
stage that diverges names the live bug. Attention outputs are compared
loosely (live KV has full prefill context; offline is suffix-only) -- x0
must match tightly (no attention involved).
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
F = torch.float32

T = {}
for t in iter_gguf_tensors(MP):
    n = t.name
    if n in {"token_embd.weight"} or n.startswith("blk.40."):
        T[n] = t


def dq(n):
    t = T[n]
    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16).reshape(t.shape).float()


EMB = dq("token_embd.weight")
W = {n: dq(n) for n in [
    "blk.40.nextn.eh_proj.weight", "blk.40.nextn.enorm.weight",
    "blk.40.nextn.hnorm.weight", "blk.40.nextn.shared_head_norm.weight",
    "blk.40.attn_norm.weight", "blk.40.post_attention_norm.weight",
    "blk.40.attn_q.weight", "blk.40.attn_k.weight", "blk.40.attn_v.weight",
    "blk.40.attn_output.weight", "blk.40.attn_q_norm.weight", "blk.40.attn_k_norm.weight",
    "blk.40.ffn_gate_inp.weight", "blk.40.ffn_gate_inp_shexp.weight",
    "blk.40.ffn_gate_shexp.weight", "blk.40.ffn_up_shexp.weight", "blk.40.ffn_down_shexp.weight",
]}

exp_packed = {k: T[f"blk.40.ffn_{k}_exps.weight"].packed().reshape(-1) for k in ("gate", "up", "down")}
exp_type = {k: T[f"blk.40.ffn_{k}_exps.weight"].ggml_type for k in ("gate", "up", "down")}
per_exp = {k: v.numel() // E for k, v in exp_packed.items()}
exp_cache = {}


def expert(i):
    if i not in exp_cache:
        m = {}
        for k in ("gate", "up", "down"):
            b = exp_packed[k][i * per_exp[k]: (i + 1) * per_exp[k]]
            m[k] = dequantize(b, exp_type[k], torch.bfloat16).float().reshape(
                (FF, H) if k != "down" else (H, FF))
        exp_cache[i] = m
    return exp_cache[i]


def rmsn(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


inv = 1.0 / (BASE ** (torch.arange(0, RD, 2, dtype=F) / RD))


def rope(x, pos):
    fr = (pos * inv).to(F)
    c, s = fr.cos(), fr.sin()
    xr = x[:, :RD].clone()
    x1, x2 = xr[:, : RD // 2], xr[:, RD // 2:]
    return torch.cat([torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1), x[:, RD:]], -1)


def silu(x):
    return x * torch.sigmoid(x)


fs = sorted(glob.glob("/tmp/mtp_trace/step_*.pt"))
print("traces:", len(fs), flush=True)

cos = torch.nn.functional.cosine_similarity
stats: dict = {}
Kc = Vc = None
for i, f in enumerate(fs):
    d = torch.load(f, map_location="cpu", weights_only=False)
    if "mid" not in d:
        continue
    h = d["h"][0].float()
    mid = d["mid"]
    e = EMB[d["token"]]
    x0 = W["blk.40.nextn.eh_proj.weight"] @ torch.cat([
        rmsn(e, W["blk.40.nextn.enorm.weight"]), rmsn(h, W["blk.40.nextn.hnorm.weight"])])
    x = rmsn(x0, W["blk.40.attn_norm.weight"])
    qf = (W["blk.40.attn_q.weight"] @ x).view(NQ, 2 * HD)
    q, gate = qf[:, :HD].clone(), qf[:, HD:].clone()
    k = (W["blk.40.attn_k.weight"] @ x).view(NKV, HD).clone()
    v = (W["blk.40.attn_v.weight"] @ x).view(NKV, HD)
    q = rmsn(q, W["blk.40.attn_q_norm.weight"])
    k = rmsn(k, W["blk.40.attn_k_norm.weight"])
    q, k = rope(q, d["pos"]), rope(k, d["pos"])
    Kc = k[None] if Kc is None else torch.cat([Kc, k[None]])
    Vc = v[None] if Vc is None else torch.cat([Vc, v[None]])
    rep = NQ // NKV
    s_ = torch.einsum("qd,tqd->qt", q, Kc.repeat_interleave(rep, 1)) / math.sqrt(HD)
    p = torch.softmax(s_, -1)
    o = torch.einsum("qt,tqd->qd", p, Vc.repeat_interleave(rep, 1)) * torch.sigmoid(gate)
    attn = W["blk.40.attn_output.weight"] @ o.reshape(-1)
    xr = attn + x0
    x2 = rmsn(xr, W["blk.40.post_attention_norm.weight"])
    rl = W["blk.40.ffn_gate_inp.weight"] @ x2
    top = torch.topk(torch.softmax(rl, -1), TOPK)
    wts = top.values / top.values.sum()
    routed = torch.zeros(H)
    for wi, ei in zip(wts.tolist(), top.indices.tolist()):
        m = expert(ei)
        routed += wi * (m["down"] @ (silu(m["gate"] @ x2) * (m["up"] @ x2)))
    gsh, ush = W["blk.40.ffn_gate_shexp.weight"] @ x2, W["blk.40.ffn_up_shexp.weight"] @ x2
    shared = (W["blk.40.ffn_down_shexp.weight"] @ (silu(gsh) * ush)) * torch.sigmoid(
        W["blk.40.ffn_gate_inp_shexp.weight"] @ x2)
    h2 = xr + routed + shared

    for name, off, live in (
        ("x0", x0, mid["x0"][0]), ("attn", attn, mid["attn"][0]),
        ("x(moe-in)", x2, mid["x"][0]),
        ("router", rl, mid["router"][0]),
        ("routed", routed, mid["routed"][0]), ("shared", shared, mid["shared"][0]),
        ("h2", h2, mid["h2"][0]),
    ):
        c = cos(off.reshape(1, -1), live.reshape(1, -1).float()).item()
        rel = ((off - live.float()).norm() / (live.float().norm() + 1e-9)).item()
        st = stats.setdefault(name, [])
        st.append((c, rel))
    if i == 2:
        lt = torch.topk(mid["router"][0].float(), TOPK)
        print("live top8:", lt.indices.tolist())
        print(" off top8:", top.indices.tolist())

for name, st in stats.items():
    cs = [x[0] for x in st]
    rs = [x[1] for x in st]
    print(f"{name:10s} cos min/mean {min(cs):+.3f}/{sum(cs)/len(cs):+.3f}  rel mean {sum(rs)/len(rs):.3f}")
# routing overlap
ov = []
for f in fs[:20]:
    d = torch.load(f, map_location="cpu", weights_only=False)
    if "mid" not in d:
        continue
    ov.append(d["mid"]["router"][0].float().argmax().item())
print("live router argmax sequence (first 20):", ov)
