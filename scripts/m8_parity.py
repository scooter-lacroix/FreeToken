import sys
import gguf, torch

sys.path.insert(0, "/home/scooter/.mlstack/venvs/freetoken/lib/python3.12/site-packages")
from freetoken.kernel.triton.kquant_linear import kq_gemv, kq_gemm_q4k_m8
from freetoken.kernel.gguf import ggml_mul_mat_vec_a8

r = gguf.GGUFReader("/mnt/HDD-2/Models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf")
t = next(t for t in r.tensors if t.name == "blk.0.attn_qkv.weight")
w = torch.from_numpy(t.data.copy()).cuda()  # uint8 [N, rb]
N, rb = w.shape
K = (rb // 144) * 256
print(f"w {tuple(w.shape)} N={N} K={K}")

torch.manual_seed(0)
x = (torch.randn(8, K, device="cuda") * 0.1).to(torch.bfloat16)

# oracle 1: per-row GEMV (proven T=1)
y_gemv = torch.cat([kq_gemv(w, x[i:i+1], 12) for i in range(8)], 0)
# oracle 2: ggml vec (LM-head-proven M<=8)
y_vec = ggml_mul_mat_vec_a8(w, x, 12, N)
# fused
y_fused = kq_gemm_q4k_m8(w, x, 12)

for name, y in (("vec", y_vec), ("fused", y_fused)):
    d = (y.float() - y_gemv.float()).abs()
    print(f"{name}: shape {tuple(y.shape)} dtype {y.dtype} maxdiff={d.max().item():.4f} "
          f"meandiff={d.mean().item():.5f}")
print("gemv row0[:6]:", y_gemv[0,:6].float().tolist())
print("vec  row0[:6]:", y_vec[0,:6].float().tolist())
print("fused row0[:6]:", y_fused[0,:6].float().tolist())
print("fused row1[:6]:", y_fused[1,:6].float().tolist())
print("gemv  row1[:6]:", y_gemv[1,:6].float().tolist())
# per-row error pattern: which rows are wrong?
for i in range(8):
    d = (y_fused[i].float() - y_gemv[i].float()).abs().max().item()
    print(f"row {i}: maxdiff {d:.4f}")
