#!/usr/bin/env python3
"""Stall sentinel: catch the intermittent first-request-after-idle stall in the act.

Context (2026-08-29 forensics): occasional 46-503s delays between an HTTP
completion POST and its scheduler Prefill batch. Never reproduced on demand;
during stalls the scheduler shows no spin (no [sched-diag]), no CPU pin, and
only ~12MB of worker disk reads. Raw-socket and curl probes were always fast,
python-urllib sometimes slow -- but the stall also preceded real traffic
(maestro 416.6s session), so it is server-relevant either way.

This watcher fires a tiny radix-cacheable completion every interval via a RAW
socket (pure pipeline latency, no python-http client stack) and, on any
response slower than the threshold, snapshots the API + scheduler-worker
processes' per-thread syscall/wchan/CPU state to a timestamped file -- the
kernel-level evidence py-spy cannot get here (ptrace_scope=1).

Run: nohup python3 scripts/stall_sentinel.py >/dev/null 2>&1 &
Stop: kill $(cat /tmp/freetoken-stall-sentinel.pid)
"""
import json
import os
import socket
import subprocess
import sys
import time

HOST, PORT = "127.0.0.1", 1919
INTERVAL_S = 45.0
STALL_THRESHOLD_S = 10.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "stall-sentinel")
PID_FILE = "/tmp/freetoken-stall-sentinel.pid"
# fixed prompt -> radix-cached after the first probe -> measures pipeline latency
PROMPT = "Sentinel pipeline heartbeat probe. "
MODEL = "Qwen3.8-27B-Ridge-3.7bpw.gguf"


def _pids():
    api = worker = None
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "ft serve" in line and "grep" not in line:
            api = int(line.split()[0])
        if "spawn_main" in line and "grep" not in line and worker is None:
            worker = int(line.split()[0])
    return api, worker


def _snapshot(reason, elapsed):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"stall-{time.strftime('%H%M%S')}-{reason}.txt")
    api, worker = _pids()
    with open(path, "w") as f:
        f.write(f"stall: {elapsed:.1f}s ({reason}) at {time.ctime()}\n")
        for name, pid in (("api", api), ("worker", worker)):
            if not pid:
                continue
            f.write(f"\n===== {name} pid={pid} cpu={_cpu(pid)} =====\n")
            for t in sorted(os.listdir(f"/proc/{pid}/task")):
                try:
                    comm = open(f"/proc/{pid}/task/{t}/comm").read().strip()
                    state = open(f"/proc/{pid}/task/{t}/stat").read().split(") ", 1)[1].split()[0]
                    wchan = open(f"/proc/{pid}/task/{t}/wchan").read().strip()
                    f.write(f"  {t} {comm}: {state} {wchan}\n")
                except OSError:
                    pass
            try:
                io = open(f"/proc/{pid}/io").read()
                f.write("io:\n" + io)
            except OSError:
                pass
    return path


def _io(pid):
    try:
        d = {}
        for line in open(f"/proc/{pid}/io"):
            k, v = line.split(":", 1)
            if k in ("read_bytes", "write_bytes", "rchar"):
                d[k] = int(v.strip())
        return d
    except OSError:
        return {}


def _cpu(pid):
    try:
        return subprocess.run(
            ["ps", "-o", "%cpu=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "?"


def probe():
    data = json.dumps(
        {"model": MODEL, "prompt": PROMPT, "max_tokens": 1,
         "temperature": 0.0, "ignore_eos": True}
    ).encode()
    req = (
        b"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\nConnection: close\r\n"
        + f"Content-Length: {len(data)}\r\n\r\n".encode()
        + data
    )
    t0 = time.time()
    s = socket.create_connection((HOST, PORT), timeout=600)
    s.sendall(req)
    buf = b""
    while b"\r\n\r\n" not in buf or not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return time.time() - t0, b"200" in buf.split(b"\r\n")[0]


def main():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    print(f"sentinel up (every {INTERVAL_S}s, threshold {STALL_THRESHOLD_S}s)", flush=True)
    prev = {}
    while True:
        try:
            api, worker = _pids()
            io0 = {n: _io(p) for n, p in (("api", api), ("worker", worker)) if p}
            elapsed, ok = probe()
            io1 = {n: _io(p) for n, p in (("api", api), ("worker", worker)) if p}
            now = time.strftime("%H:%M:%S")
            delta = {
                n: (io1[n].get("read_bytes", 0) - io0[n].get("read_bytes", 0)) // (1 << 20)
                for n in io0
            }
            tag = " ".join(f"{n}_rd={v}MB" for n, v in delta.items())
            if elapsed > STALL_THRESHOLD_S or not ok:
                p = _snapshot(f"{elapsed:.0f}s-ok{ok}", elapsed)
                print(f"[{now}] STALL {elapsed:.1f}s ok={ok} {tag} -> {p}", flush=True)
            else:
                print(f"[{now}] {elapsed:.2f}s {tag}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] probe error: {e!r}", flush=True)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
