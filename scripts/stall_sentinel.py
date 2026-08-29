#!/usr/bin/env python3
"""Stall sentinel — passive by default: NEVER generates requests.

Contract (user-directed, 2026-08-29): FreeToken must be silent when idle.
Generating probe traffic keeps the server permanently "active" (it would hold
residency pins forever and mask real idle behavior), so passive log-watching
is the default; the request-probe mode from the original forensics is opt-in
via --active.

Passive mode tails the server log and flags stalls: a Prefill/Decode batch
in flight (queue or running > 0) whose NEXT batch line is >THRESHOLD seconds
away. Active mode additionally fires a raw-socket completion every --interval
and snapshots per-thread state + io deltas on any response over threshold
(the evidence py-spy cannot get at ptrace_scope=1).

Run:   nohup python3 scripts/stall_sentinel.py --log logs/<server>.log >/dev/null 2>&1 &
Probe: python3 scripts/stall_sentinel.py --active --interval 45
Stop:  kill $(cat /tmp/freetoken-stall-sentinel.pid)
"""
import json
import os
import re
import socket
import subprocess
import sys
import time

HOST, PORT = "127.0.0.1", 1919
STALL_THRESHOLD_S = 10.0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "stall-sentinel")
PID_FILE = "/tmp/freetoken-stall-sentinel.pid"
PROMPT = "Sentinel pipeline heartbeat probe. "  # radix-cacheable, tiny
MODEL = "Qwen3.8-27B-Ridge-3.7bpw.gguf"
BATCH_RE = re.compile(r"Prefill batch|Decode batch|#running-req: (\d+), #queue-req: (\d+)")


def _pids():
    api = worker = None
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "ft serve" in line and "grep" not in line:
            api = int(line.split()[0])
        if "spawn_main" in line and "grep" not in line and worker is None:
            worker = int(line.split()[0])
    return api, worker


def _io(pid):
    try:
        return {
            k: int(v)
            for k, v in (line.split(":", 1) for line in open(f"/proc/{pid}/io"))
            if k in ("read_bytes", "write_bytes", "rchar")
        }
    except OSError:
        return {}


def _cpu(pid):
    try:
        return subprocess.run(
            ["ps", "-o", "%cpu=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "?"


def snapshot(reason, elapsed, extra=""):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"stall-{time.strftime('%H%M%S')}-{reason}.txt")
    api, worker = _pids()
    with open(path, "w") as f:
        f.write(f"stall: {elapsed:.1f}s ({reason}) at {time.ctime()} {extra}\n")
        for name, pid in (("api", api), ("worker", worker)):
            if not pid:
                continue
            f.write(f"\n===== {name} pid={pid} cpu={_cpu(pid)} =====\n")
            try:
                for line in open(f"/proc/{pid}/status"):
                    if line.startswith(("VmLck", "VmRSS")):
                        f.write(line)
            except OSError:
                pass
            for t in sorted(os.listdir(f"/proc/{pid}/task")):
                try:
                    comm = open(f"/proc/{pid}/task/{t}/comm").read().strip()
                    state = open(f"/proc/{pid}/task/{t}/stat").read().split(") ", 1)[1].split()[0]
                    wchan = open(f"/proc/{pid}/task/{t}/wchan").read().strip()
                    f.write(f"  {t} {comm}: {state} {wchan}\n")
                except OSError:
                    pass
            try:
                f.write("io:\n" + open(f"/proc/{pid}/io").read())
            except OSError:
                pass
    return path


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


def run_active(interval):
    while True:
        try:
            api, worker = _pids()
            io0 = {n: _io(p) for n, p in (("api", api), ("worker", worker)) if p}
            elapsed, ok = probe()
            io1 = {n: _io(p) for n, p in (("api", api), ("worker", worker)) if p}
            now = time.strftime("%H:%M:%S")
            delta = " ".join(
                f"{n}_rd={(io1[n].get('read_bytes', 0) - io0[n].get('read_bytes', 0)) >> 20}MB"
                for n in io0
            )
            if elapsed > STALL_THRESHOLD_S or not ok:
                p = snapshot(f"{elapsed:.0f}s-ok{ok}", elapsed)
                print(f"[{now}] STALL {elapsed:.1f}s ok={ok} {delta} -> {p}", flush=True)
            else:
                print(f"[{now}] {elapsed:.2f}s {delta}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] probe error: {e!r}", flush=True)
        time.sleep(interval)


def run_passive(log_path, threshold):
    """Tail the server log; flag in-flight requests whose batch lines gap out."""
    print(f"passive sentinel: watching {log_path} (gap >{threshold}s while in flight)",
          flush=True)
    fh = open(log_path, "r", errors="replace")
    fh.seek(0, os.SEEK_END)
    inflight_since = None  # time of the last batch line that showed work in flight
    while True:
        line = fh.readline()
        if not line:
            time.sleep(2.0)
            if inflight_since is not None and time.time() - inflight_since > threshold:
                p = snapshot(f"passive-{threshold:.0f}s-gap", time.time() - inflight_since,
                             extra=f"log={log_path}")
                print(f"[{time.strftime('%H:%M:%S')}] PASSIVE STALL: in-flight batch "
                      f"silent >{threshold}s -> {p}", flush=True)
                inflight_since = time.time()  # re-arm: one alert per gap
            continue
        m = BATCH_RE.search(line)
        if not m:
            continue
        running = int(m.group(1) or 0)
        queued = int(m.group(2) or 0)
        if running > 0 or queued > 0:
            if inflight_since is not None and time.time() - inflight_since > threshold:
                p = snapshot(f"passive-{threshold:.0f}s-gap", time.time() - inflight_since,
                             extra=f"log={log_path}")
                print(f"[{time.strftime('%H:%M:%S')}] PASSIVE STALL: gap between "
                      f"in-flight batch lines >{threshold}s -> {p}", flush=True)
            inflight_since = time.time()
        else:
            inflight_since = None


def main():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    if "--active" in sys.argv:
        interval = (float(sys.argv[sys.argv.index("--interval") + 1])
                    if "--interval" in sys.argv else 45.0)
        print(f"ACTIVE sentinel: probe every {interval}s (generates traffic!)", flush=True)
        run_active(interval)
    else:
        log_path = (sys.argv[sys.argv.index("--log") + 1]
                    if "--log" in sys.argv else None)
        threshold = (float(sys.argv[sys.argv.index("--threshold") + 1])
                     if "--threshold" in sys.argv else 60.0)
        if not log_path:
            print("passive mode needs --log <server.log>", flush=True)
            sys.exit(2)
        run_passive(log_path, threshold)


if __name__ == "__main__":
    sys.exit(main())
