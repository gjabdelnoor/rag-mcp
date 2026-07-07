#!/usr/bin/env python3
"""miopen_harness.py — CONTRACT 3 AMENDMENT §1.

The cheapest fix for the 7700S vision-encoder hang is to disable
MIOpen's exhaustive algorithm search. Per
https://rocm.docs.amd.com/projects/MIOpen/en/latest/MIOpen-find-mode.html
the env var `MIOPEN_FIND_MODE` accepts:

  1 = NORMAL (default — exhaustive; the one that hangs)
  2 = FAST    (immediate, use cached or first-found; skips benchmark)
  3 = HYBRID  (use cache when available, otherwise fast)
  4 = FAST_HYBRID
  5 = DYNAMIC (use cache if recent, otherwise fall back to NORMAL)

CONTRACT 2 tried `MIOPEN_FORCE_ALGO_FWD=1` (which forces one specific
algorithm) but did NOT touch find-mode — that's what this harness
explores.

This script:
  - Sweeps `MIOPEN_FIND_MODE` ∈ {2, 3, 5} (the recommended set; 1 is
    the broken default, 4 is rarely useful).
  - Optionally warms the per-user FindDb at ~/.cache/miopen (FIND_DB_PATH
    / MIOPEN_CUSTOM_CACHE_DIR).
  - For each mode, attempts a single 720×720 still embedder.process()
    on cuda:0 and reports whether it returns within the timeout.
  - Records per-mode elapsed time + per-mode PASS/FAIL/HANG.
  - Writes a JSON summary to `kimi_kernel/output/miopen_find_modes.json`.

GPU-free preprocessing: the harness itself is just orchestration; it
will SKIP GPU runs (writing "DEFERRED" in the JSON) if the VRAM guard
fails (free < 5 GB on cuda:0 OR `pgrep embedder_worker` non-empty).

Per CONTRACT 3: **NEVER** kill any process to free VRAM — if VRAM is
stuck, the harness defers and the JSON shows `gpu_deferred: true`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VRAM_FLOOR_BYTES = 5 * 1024 ** 3
# Find-modes to test (1 is the broken default — included as a baseline
# reference only when the user passes --include-default).
RECOMMENDED_MODES = [2, 3, 5]


def free_vram_bytes(device_idx: int) -> int:
    try:
        import torch
        if torch.cuda.is_available() and device_idx < torch.cuda.device_count():
            free, _ = torch.cuda.mem_get_info(device_idx)
            return int(free)
    except Exception:
        pass
    return 0


def vram_guard_passes(device_idx: int = 0) -> tuple[bool, str]:
    """Returns (ok, reason)."""
    if shutil.which("pgrep") is not None:
        r = subprocess.run(["pgrep", "-af", "embedder_worker"],
                           capture_output=True, text=True)
        if r.stdout.strip():
            return False, "embedder_worker is running (per pgrep)"
    free = free_vram_bytes(device_idx)
    if free < VRAM_FLOOR_BYTES:
        return False, f"cuda:{device_idx} only {free/1024**3:.2f}GB free (<5GB)"
    return True, f"cuda:{device_idx} {free/1024**3:.1f}GB free"


def run_one_mode(mode: int, framedir: str, W: int, timeout_s: int,
                 device: str = "cuda:0") -> dict:
    """Spawn a subprocess that runs a single embed call under this
    find-mode, with a hard timeout. NEVER kills the subprocess — uses
    SIGTERM (graceful) if it overruns, and reports HANG.
    """
    # Build a tiny one-shot Python that:
    #  1. Loads the embedder under the env.
    #  2. Embeds 1 still image at 720x720.
    #  3. Embeds 1 video window of size W.
    #  4. Prints elapsed time + status.
    script = HERE / "_miopen_one.py"
    if not script.exists():
        return {"mode": mode, "status": "INFRA-ERROR",
                "error": "_miopen_one.py missing"}

    env = os.environ.copy()
    env["MIOPEN_FIND_MODE"] = str(mode)
    env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1] if ":" in device else "0"
    env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
    env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"   # from CONTRACT 3 §3
    # Reduce noise.
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    # Ensure the subprocess can find the parent-dir modules (qwen3_vl_embedding,
    # embedder_worker, etc.). Without this, _miopen_one.py fails with
    # ModuleNotFoundError when run from /tmp/ or via a launcher.
    parent_dir = str(HERE.parent.resolve())
    env["PYTHONPATH"] = parent_dir + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-u", str(script),
           "--frames", framedir, "--window", str(W), "--device", device,
           "--limit", str(max(64, W * 2))]
    print(f"\n[mode={mode}] launching: MIOPEN_FIND_MODE={mode} "
          f"{' '.join(cmd[2:])}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        stdout, _ = proc.communicate(timeout=timeout_s)
        elapsed = time.time() - t0
        rc = proc.returncode
        status = "PASS" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[mode={mode}] rc={rc} elapsed={elapsed:.1f}s status={status}",
              flush=True)
        return {"mode": mode, "status": status, "elapsed_s": elapsed,
                "returncode": rc, "stdout_tail": stdout[-2000:]}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        # Graceful terminate — NEVER SIGKILL per CONTRACT 3.
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Even SIGTERM timed out — leave the process; do NOT kill.
            print(f"[mode={mode}] SIGTERM did not complete in 10s — leaving "
                  f"the subprocess alive (NO KILL per CONTRACT 3)", flush=True)
        print(f"[mode={mode}] TIMEOUT after {elapsed:.0f}s "
              f"(>= {timeout_s}s)  status=HANG", flush=True)
        return {"mode": mode, "status": "HANG", "elapsed_s": elapsed,
                "timeout_s": timeout_s}


def warm_find_db():
    """Optional: pre-warm ~/.cache/miopen by touching the directory."""
    db = Path.home() / ".cache" / "miopen"
    db.mkdir(parents=True, exist_ok=True)
    print(f"[harness] FindDb path: {db}  (entries: "
          f"{len(list(db.glob('*.db')))})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=str,
                    default="/home/gabriel/Documents/Personal/"
                            "How-and-why-to-take-a-logarithm-of-an-image-"
                            "2026-07-04T095531Z.frames")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-mode wall timeout (seconds).")
    ap.add_argument("--include-default", action="store_true",
                    help="Also test MIOPEN_FIND_MODE=1 (the broken baseline) "
                         "for reference. Default skips it.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"[harness] frames={args.frames}  W={args.window}  "
          f"timeout={args.timeout}s  device={args.device}", flush=True)

    warm_find_db()

    ok, reason = vram_guard_passes()
    summary = {
        "framedir": args.frames,
        "W": args.window,
        "device": args.device,
        "vram_guard_ok": ok,
        "vram_guard_reason": reason,
        "results": [],
    }
    if not ok:
        print(f"[harness] VRAM guard FAILED: {reason}", flush=True)
        print("[harness] NOT running any GPU work — the live rag-mcp daemon "
              "owns cuda:0 (CONTRACT 3 AMENDMENT forbids killing it).",
              flush=True)
        summary["status"] = "DEFERRED"
        out = OUT_DIR / "miopen_find_modes.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"[harness] summary: {out}", flush=True)
        sys.exit(0)

    modes = list(RECOMMENDED_MODES)
    if args.include_default:
        modes.insert(0, 1)

    for m in modes:
        res = run_one_mode(m, args.frames, args.window, args.timeout,
                           device=args.device)
        summary["results"].append(res)
        # Brief pause between modes (let MIOpen state settle; no kill).
        time.sleep(2)

    # Recommend the winner (if any).
    passes = [r for r in summary["results"] if r["status"] == "PASS"]
    if passes:
        passes.sort(key=lambda r: r.get("elapsed_s", float("inf")))
        summary["recommended_mode"] = passes[0]["mode"]
        summary["recommended_elapsed_s"] = passes[0]["elapsed_s"]
    summary["status"] = "COMPLETE"

    out = OUT_DIR / "miopen_find_modes.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[harness] summary: {out}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()