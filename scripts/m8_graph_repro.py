import sys
import gguf, torch

sys.path.insert(0, "/home/scooter/.mlstack/venvs/freetoken/lib/python3.12/site-packages")
from freetoken.kernel.triton.kquant_linear import kq_gemv, kq_gemm_q4k_m8

r = gguf.GFUFReader if False else gguf.GGUFReader("/mnt/HDD-2/Models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf")
t = next(t for t in r.tensors if t.name == "blk.0.attn_qkv.weight")
w = torch.from_numpy(t.data.copy()).cuda()
N, rb = w.shape
K = (rb // 144) * 256

torch.manual_seed(0)
x = (torch.randn(8, K, device="cuda") * 0.1).to(torch.bfloat16)

# eager reference (fused)
y_ref = kq_gemm_q4k_m8(w, x, 12).clone()

# tiny graph capture of the SAME kernel on the same inputs
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(2):
        kq_gemm_q4k_m8(w, x, 12)
    torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    y_cap = kq_gemm_q4k_m8(w, x, 12)
torch.cuda.synchronize()
print("capture ok; y_cap", tuple(y_cap.shape), y_cap.dtype)

g.replay()
torch.cuda.synchronize()
d = (y_cap.float() - y_ref.float()).abs()
print(f"replay#1 vs eager ref: maxdiff={d.max().item():.4f}")
g.replay()
torch.cuda.synchronize()
d2 = (y_cap.float() - y_ref.float()).abs()
print(f"replay#2 vs eager ref: maxdiff={d2.max().item():.4f}")
