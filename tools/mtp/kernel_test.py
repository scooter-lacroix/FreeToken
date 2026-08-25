"""Kernel-level check: fused_experts_ggml_split (the ggml_file eager decode
kernel) vs offline dequant reference, on real traced routing/input, with a
private cache whose slot bytes were already verified byte-exact.
"""
import os
import torch

os.environ["FREETOKEN_MTP"] = "1"
os.environ["FREETOKEN_EXPERT_BANK_STORAGE"] = "file"

MP = "/mnt/HDD-2/Models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf"
DEV = torch.device("cuda")

from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config, load_ggml_expert_sources_file
from freetoken.utils import cached_load_hf_config

cfg = parse_gguf_config(cached_load_hf_config(MP))
E = cfg.num_experts
srcs = load_ggml_expert_sources_file(MP, cfg)

from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.fused_q4_0 import fused_experts_ggml_split
from freetoken.models.gguf.dequant import dequantize

cache = OffloadMoeCache(
    num_layers=len(srcs["gate"]), num_experts=E, cache_size=2 * E,
    device=DEV, quant_format="ggml_file",
)
cache.set_bank_sources({k: srcs[k] for k in ("gate", "up", "down")})
cache.ggml_quant_types = srcs["quant_types"]

d = torch.load("/tmp/mtp_trace/step_002.pt", map_location="cpu", weights_only=False)
mid = d["mid"]
x = mid["x"][0].to(DEV).to(torch.bfloat16).reshape(1, -1)
rl = mid["router"][0].to(DEV).float().reshape(1, -1)
probs = torch.softmax(rl, -1)
tw, tid = torch.topk(probs, 8, dim=-1)
tw = tw / tw.sum(-1, keepdim=True)
ids = tid.to(torch.int32)
print("ids:", ids.reshape(-1).tolist())

ids_slot = ids.clone()
cache.ensure_experts(40, ids_slot)
cache.copy_missing()
torch.cuda.synchronize()
print("slots:", ids_slot.reshape(-1).tolist())

gate, up, down = cache.bank_views()
out = fused_experts_ggml_split(
    x, gate, up, down, tw.float(), ids_slot, "silu", cache.ggml_quant_types[40]
)
torch.cuda.synchronize()
out = out[0].float().cpu()

# offline reference from raw GGUF expert bytes (dequant), same ids/weights/x
from freetoken.models.gguf.reader import iter_gguf_tensors
T = {}
for t in iter_gguf_tensors(MP):
    if t.name.startswith("blk.40.ffn_") and t.name.endswith("_exps.weight"):
        T[t.name] = t
H, FF = 2048, 512
per = {k: T[f"blk.40.ffn_{k}_exps.weight"].packed().reshape(-1).numel() // E for k in ("gate", "up", "down")}
typ = {k: T[f"blk.40.ffn_{k}_exps.weight"].ggml_type for k in ("gate", "up", "down")}


def silu(v):
    return v * torch.sigmoid(v)


xc = x[0].float().cpu()
ref = torch.zeros(H)
for w_, e in zip(tw.reshape(-1).tolist(), ids.reshape(-1).tolist()):
    m = {}
    for k in ("gate", "up", "down"):
        pk = T[f"blk.40.ffn_{k}_exps.weight"].packed().reshape(-1)
        b = pk[e * per[k]: (e + 1) * per[k]]
        m[k] = dequantize(b, typ[k], torch.float32).reshape((H, FF) if k != "down" else (FF, H))
    ref += w_ * ((silu(xc @ m["gate"]) * (xc @ m["up"])) @ m["down"])

cos = torch.nn.functional.cosine_similarity(out.reshape(1, -1), ref.reshape(1, -1)).item()
rel = ((out - ref).norm() / ref.norm()).item()
print(f"kernel vs offline ref: cos {cos:+.4f} rel {rel:.4f}")
print("kernel head:", [round(v, 4) for v in out[:6].tolist()])
print("ref    head:", [round(v, 4) for v in ref[:6].tolist()])
print("live routed head:", [round(v, 4) for v in mid["routed"][0][:6].tolist()])
