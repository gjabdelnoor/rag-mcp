#!/usr/bin/env python3
"""_poll_vram.py — VRAM-guard poller for CONTRACT 3 AMENDMENT.

Runs in the background and reports every `--interval` seconds whether
`cuda:0` has ≥ 5 GB free AND no `embedder_worker` is running. Writes
each check to a JSONL file at `kimi_kernel/output/vram_poll.jsonl`
and prints a single line per check on stdout.

Exits with code 0 the moment the guard passes (caller can `wait` on
this process). Exits with code 1 on `--max-wait` exceeded.

NEVER kills anything.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VRAM_FLOOR_BYTES = 5 * 1024 ** 3


def free_vram_bytes(device_idx: int) -> int:
    try:
        import torch
        if torch.cuda.is_available() and device_idx < torch.cuda.device_count():
            free, _ = torch.cuda.mem_get_info(device_idx)
            return int(free)
    except Exception:
        pass
    return 0


def embedder_worker_running() -> bool:
    if shutil.which("pgrep") is None:
        return False
    r = subprocess.run(["pgrep", "-af", "embedder_worker"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def check() -> dict:
    free = free_vram_bytes(0)
    has_worker = embedder_worker_running()
    return {
        "ts": time.time(),
        "free_vram_gb": free / 1024 ** 3,
        "embedder_worker_running": has_worker,
        "guard_pass": (free >= VRAM_FLOOR_BYTES) and not has_worker,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-wait", type=int, default=20 * 60,
                    help="Max seconds to poll before giving up.")
    args = ap.parse_args()

    log_path = OUT_DIR / "vram_poll.jsonl"
    deadline = time.time() + args.max_wait
    print(f"[poll] interval={args.interval}s  max_wait={args.max_wait}s  "
          f"log={log_path}", flush=True)

    with open(log_path, "a") as f:
        while time.time() < deadline:
            c = check()
            f.write(json.dumps(c) + "\n")
            f.flush()
            tag = "READY" if c["guard_pass"] else "WAIT"
            print(f"[poll] [{tag}] free={c['free_vram_gb']:.2f}GB  "
                  f"worker={'yes' if c['embedder_worker_running'] else 'no'}",
                  flush=True)
            if c["guard_pass"]:
                print("[poll] GUARD PASSED — GPU is available", flush=True)
                sys.exit(0)
            time.sleep(args.interval)

    print(f"[poll] GUARD NEVER PASSED in {args.max_wait}s — DEFER GPU work",
          flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()