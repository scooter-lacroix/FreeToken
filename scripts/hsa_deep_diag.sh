#!/usr/bin/env bash
# HSA >=50k-context fault diagnostic boot (the 72k blocker).
#
# Ridge65's full-depth (73728) warmup walk crashed the worker at KV usage ~0.68
# (~52k) with HSA_STATUS_ERROR_EXCEPTION and no Python traceback; 40960 walks
# clean. This boot: (1) serializes kernel launches so the FIRST faulting kernel
# is attributed in the log, (2) removes the warmup depth cap so the walk goes
# full-depth on purpose, (3) captures dmesg + rocm-smi state at crash time.
#
# GPU contract: heavy on BOTH cards for ~10-15 min (boot + full-depth walk) and
# WILL crash a worker by design. Run only inside a coordinated window.
#
# Usage: scripts/hsa_deep_diag.sh [depth]   (default 73728)
set -euo pipefail
DEPTH="${1:-73728}"
cd "$(dirname "$0")/.."
LOG="logs/ridge-diag-hsa-$(date +%H%M).log"

echo "=== HSA deep diagnostic: depth=$DEPTH log=$LOG ==="
env -u HSA_OVERRIDE_GFX_VERSION -u HSA_TOOLS_LIB -u LD_PRELOAD -u PYTHONPATH \
  HIP_VISIBLE_DEVICES=0,1 HSA_ENABLE_SDMA=0 \
  PYTORCH_ROCM_ARCH=gfx1100 PYTORCH_ROCM_DEVICE=0,1 \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512 \
  TRITON_CACHE_DIR=/home/scooter/.cache/mlstack/triton/cache \
  AMD_SERIALIZE_KERNEL=3 \
  HSA_ENABLE_ASSERTS=1 \
  FREETOKEN_LAYER_SPLIT=56 \
  FREETOKEN_WARMUP_MAX_DEPTH="$DEPTH" \
  ~/.mlstack/venvs/freetoken/bin/ft serve \
  --model /mnt/HDD-2/Models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf \
  --port 1919 --cuda-graph-max-bs 0 --max-seq-len-override 73728 --memory-ratio 0.90 \
  --max-extend-length 1024 --max-prefill-length 1024 --kv-reserve-tokens 2048 \
  --max-running-requests 1 \
  2>&1 | tee "$LOG"

STATUS=$?
echo "=== exited status=$STATUS; last kernel context ==="
grep -anE "aborting with error|Fault|fault|Page|kernel|Signal" "$LOG" | tail -25
echo "=== dmesg tail (GPU/AMD lines) ==="
dmesg 2>/dev/null | grep -iE "amdgpu|hsa|kfd" | tail -15 || sudo -n dmesg 2>/dev/null | tail -15 || echo "(dmesg unavailable without sudo)"
