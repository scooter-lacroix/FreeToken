"""Standalone eager-fetch test: ensure_experts(40) + copy_missing on a real
OffloadMoeCache with the file-tier banks, then byte-compare slot rows vs the
host pack views. No server involved.
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
L = cfg.num_layers  # 40 trunk
E = cfg.num_experts
PL = L + 1

srcs = load_ggml_expert_sources_file(MP, cfg)
print("file sources:", {k: (len(v) if isinstance(v, list) else v) for k, v in srcs.items()}, flush=True)

from freetoken.moe.offload_cache import OffloadMoeCache

cache = OffloadMoeCache(
    num_layers=PL, num_experts=E, cache_size=4 * E, device=DEV,
    quant_format="ggml_file",
)
cache.set_bank_sources({k: srcs[k] for k in ("gate", "up", "down")})

# the exact top-8 the live probe routed at step 002 (trace)
raw = [148, 59, 218, 206, 107, 186, 132, 121]
ids = torch.tensor([raw], dtype=torch.int32, device=DEV)
raw_t = ids.clone()

cache.ensure_experts(40, ids)
print("post-ensure ids (slots):", ids.reshape(-1).tolist(), flush=True)
cache.copy_missing()
torch.cuda.synchronize()

ok = 0
for i, e in enumerate(raw):
    slot = int(ids[0, i])
    if slot < 0:
        print(f"expert {e}: slot {slot} NEGATIVE")
        continue
    for name in ("gate", "up", "down"):
        got = cache.bank_caches[name][slot].cpu()
        exp = srcs[name][40][e].cpu()
        same = torch.equal(got, exp)
        if name == "gate":
            print(f"expert {e:3d} -> slot {slot}: gate equal={same}")
        ok += int(same)
print(f"row groups equal: {ok}/24")
print("slot0 gate head:", cache.bank_caches['gate'][int(ids[0,0])].view(-1)[:8].tolist())
print("host  e=148 gate head:", srcs['gate'][40][148].view(-1)[:8].tolist())

# ---- kernel-level check: fused_experts_ggml_split vs offline reference ----
from freetoken.moe.fused_q4_0 import fused_experts_ggml_split
from freetoken.models.gguf.dequant import dequantize

x_live = torch.load("/tmp/mtp_trace/step_002.pt", map_location="cpu", weights_only=False)["mid"]["x"][0].to(DEV).to(torch.bfloat16).reshape(1, -1)
# routing: recompute from saved router logits exactly as the probe's fused_topk would
rl = torch.load("/tmp/mtp_trace/step_002.pt", map_location="cpu", weights_only=False)["mid"]["router"][0].to(DEV).float()
probs = torch.softmax(rl.reshape(1, -1), -1)
tw, tid = torch.topk(probs, 8, dim=-1)
tw = (tw / tw.sum(-1, keepdim=True)).to(torch.float32)
print("ids again:", tid.reshape(-1).tolist(), "w:", [round(v,3) for v in tw.reshape(-1).tolist()])

ids2 = tid.to(torch.int32).reshape(-1)
raw2 = ids2.clone()
cache.ensure_experts(40, ids2.reshape(1, -1))
cache.copy_missing()
torch.cuda.synchronize()

views = cache.bank_views()
gate, up, down = views
qt_pair = cache.ggml_quant_types[40]
out = fused_experts_ggml_split(x_live, gate, up, down, tw, ids2.reshape(1, -1), "silu", qt_pair)
torch.cuda.synchronize()

# offline reference with GGUF dequant experts (same ids/weights/x)
xc = x_live[0].float().cpu()
ref = torch.zeros(2048)
def silu(v): return v * torch.sigmoid(v)
for w_, e in zip(tw.reshape(-1).tolist(), tid.reshape(-1).tolist()):
    m = {}
    for k, shp in (("gate", (512, 2048)), ("up", (512, 2048)), ("down", (2048, 512))):
        tname = f"blk.40.ffn_{k}_exps.weight"
        raise SystemExit("placeholder")  # replaced below
