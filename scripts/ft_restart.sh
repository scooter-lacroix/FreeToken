#!/bin/bash
# Robust FreeToken server restart: kill main+workers, wait for ports 1919/1920 to
# actually clear (orphaned spawn_main workers hold 1920 after the main dies), then boot.
LOG="$1"; shift
for attempt in 1 2 3 4 5 6; do
  pkill -9 -f "\.mlstack/venvs/freetoken/bin/ft serve" 2>/dev/null
  pkill -9 -f "multiprocessing.spawn import spawn_main" 2>/dev/null
  sleep 4
  if ! ss -tln 2>/dev/null | grep -qE ':19(19|20)\s'; then
    echo "PORTS-CLEAR (attempt $attempt)"; break
  fi
  if [ "$attempt" = 6 ]; then echo "PORTS-STILL-HELD"; ss -tlnp | grep -E ':19(19|20)\s'; exit 1; fi
done
rm -f ~/.cache/torch_extensions/*/freetoken_gguf_kernels/lock 2>/dev/null
cd /home/scooter/Documents/Product/Stan-s-ML-Stack/Fork/FreeToken || exit 1
setsid env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" FREETOKEN_FREE_AUDIT="${FREETOKEN_FREE_AUDIT:-0}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}" \
  "$HOME/.mlstack/venvs/freetoken/bin/ft" serve \
  --model /mnt/HDD-2/Models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf \
  --port 1919 --max-seq-len-override 73728 --kv-reserve-tokens 2048 --max-extend-length 2048 \
  --max-running-requests 1 --page-size 1 --cache-type radix --memory-ratio 0.9 "$@" \
  > "$LOG" 2>&1 < /dev/null &
echo "BOOTED $!"
for i in $(seq 1 45); do
  sleep 10
  tr -d '\0' < "$LOG" 2>/dev/null | grep -aq "ready to serve" && { echo READY; exit 0; }
  tr -d '\0' < "$LOG" 2>/dev/null | grep -aqE "EADDRINUSE|Traceback" && { echo BOOT-FAILED; exit 1; }
done
echo TIMEOUT; exit 1
