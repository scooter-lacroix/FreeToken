"""Profile one 2048-token prefill forward of the 27B Ridge (kernel breakdown)."""
import os, sys, time
import torch

MODEL = "/mnt/HDD-2/Models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf"

t0 = time.time()
from freetoken.llm import LLM
from freetoken.core import SamplingParams

llm = LLM(
    MODEL,
    max_extend_tokens=2048,
    max_running_req=1,
    max_seq_len_override=16384,
    cuda_graph_max_bs=0,
)
print(f"load: {time.time()-t0:.0f}s", flush=True)

prompt_ids = [(i * 7919) % 24000 + 2000 for i in range(4096)]  # 2 chunks of 2048

# warmup pass (compiles/cache-warm), then profiled pass
llm.generate([prompt_ids], SamplingParams(max_tokens=1, ignore_eos=True))
print("warmup pass done", flush=True)

torch.cuda.synchronize()
llm.pending_requests = [(prompt_ids, SamplingParams(max_tokens=1, ignore_eos=True))]
llm.status_map = {}
llm.counter = 0
t0 = time.time()
try:
    llm.run_forever()
except Exception as e:
    print("loop end:", repr(e), flush=True)
torch.cuda.synchronize()
print(f"pass wall: {time.time()-t0:.2f}s for 2x2048-token chunks", flush=True)
