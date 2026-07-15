"""Torch-free hardware probe + preset picker for the rag-mcp first-time setup.

Public surface:
    probe() -> dict   # raw hardware facts
    pick(probe_result) -> str   # one of "tiny" | "small" | "medium" | "large"
    lookup_tflops(name) -> float | None   # GPU name -> peak FP32 TFLOPs
    PRESETS            # the tier table (model, dim, mins, backend)

The probe shells out to `nvidia-smi`, `rocm-smi`, `lspci`, and `sysctl` rather
than importing torch — so the no-torch-at-idle invariant of the rag-mcp server
is preserved. CPU performance is estimated by a short numpy matmul (2-second
budget) so even a headless box without `nvidia-smi` gets a usable score.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Preset table — the model ladder exposed to the user.
# ---------------------------------------------------------------------------

PRESETS = {
    "tiny": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_local": "all-MiniLM-L6-v2",
        "dim": 384,
        "min_tflops_fp16": 0.0,
        "min_vram_gb": 0,
        "backend": "sentence_transformers",
        "supports_images": False,
        "summary": "all-MiniLM-L6-v2 (22M, 384-d) — CPU-friendly, text only",
    },
    "small": {
        "model": "nomic-embed-text-v1.5.Q8_0.gguf",
        "model_local": "nomic-embed-text-v1.5.Q8_0.gguf",
        "dim": 768,
        "min_tflops_fp16": 2.0,           # any modern iGPU (780M = 8.4 FP16)
        "min_vram_gb": 4,
        "backend": "llamacpp",
        "supports_images": False,
        "summary": "nomic-embed-text-v1.5 (137M, 768-d) — text only",
    },
    "medium": {
        "model": "Qwen/Qwen3-VL-Embedding-2B",
        "model_local": "Qwen/Qwen3-VL-Embedding-2B",
        "dim": 2048,
        "min_tflops_fp16": 20.0,          # RX 7700S = 23.6 FP16; RTX 3060 = 25.6
        "min_vram_gb": 6,
        "backend": "hf_transformers",
        "supports_images": True,
        "summary": "Qwen3-VL-Embedding-2B (2B, 2048-d) — text + images",
    },
    "large": {
        "model": "Qwen/Qwen3-VL-Embedding-8B",
        "model_local": "Qwen/Qwen3-VL-Embedding-8B",
        "dim": 4096,
        "min_tflops_fp16": 25.0,          # >= RTX 3060-class tensor throughput
        "min_vram_gb": 12,                # VL-8B 8-bit fits on a 3060 (12 GB)
        "backend": "hf_transformers_quantized",
        "supports_images": False,         # image side stays on the VL-2B path
        "summary": "Qwen3-VL-Embedding-8B (8B, 4096-d, 8-bit) — text only",
        "warning": (
            "Large preset (VL-8B, 4096-d) requires >= RTX 3060-class FP16 "
            "tensor throughput (~25 TFLOPs) and >= 12 GB VRAM. Quality gains "
            "over `medium` are MARGINAL for most knowledge bases (textbooks, "
            "papers, notes) and only show up on very complex / domain-specific "
            "corpora with heavy jargon, multilingual content, or fine-grained "
            "conceptual distinctions. Costs: ~5-15 s cold start (vs ~15 s on a "
            "7700S), ~3x slower ingest, ~2x VRAM. Image ingest still uses the "
            "`medium`-tier embedder (VL-2B). Downgrade: `python setup.py "
            "--preset medium --force` then `python ingest.py --reset`."
        ),
    },
}

VALID_PRESETS = tuple(PRESETS.keys())


# ---------------------------------------------------------------------------
# Static device -> peak FP16 TFLOPs table.
#
# FP16 (tensor-core / matrix) throughput is what actually runs an embedding
# model: NV tensor cores deliver ~2x FP32 CUDA cores; AMD RDNA 2/3 FP16
# matrix ops are ~2x FP32 shader; Apple GPU FP16 throughput is ~2x FP32.
# FP32 (shader/CUDA) is roughly half of these numbers and NOT what we want
# to size an embedding model against.
#
# Sources: NVIDIA spec sheets (Tensor TFLOPS, dense, no sparsity), AMD
# product pages (peak FP16 matrix), Apple developer docs. Numbers are
# non-boosted; we err on the conservative side.
#
# Add new devices as they hit the field. Unknown devices -> None -> the
# picker conservatively assumes "small or lower".
# ---------------------------------------------------------------------------

NVIDIA_TFLOPS_FP16 = {
    # RTX 40 series (Ada — FP16 tensor, dense, FP32 accumulate)
    "RTX 4090": 165.2, "RTX 4080": 97.5, "RTX 4070 Ti": 79.0,
    "RTX 4070": 57.0, "RTX 4060 Ti": 44.1, "RTX 4060": 30.2,
    # RTX 30 series (Ampere — FP16 tensor, dense)
    "RTX 3090 Ti": 80.0, "RTX 3090": 71.2, "RTX 3080 Ti": 68.2,
    "RTX 3080": 59.5, "RTX 3070 Ti": 43.6, "RTX 3070": 40.6,
    "RTX 3060 Ti": 32.4, "RTX 3060": 25.6, "RTX 3050": 18.2,
    # Data center
    "A100": 312.0, "H100": 989.0, "L4": 121.0, "L40": 181.0,
    # Laptop (common ones)
    "RTX 4080 Laptop": 67.4, "RTX 4070 Laptop": 42.8,
    "RTX 4060 Laptop": 29.2, "RTX 4050 Laptop": 19.2,
    "RTX 3080 Ti Laptop": 47.0, "RTX 3070 Ti Laptop": 33.2,
    "RTX 3070 Laptop": 29.0, "RTX 3060 Laptop": 21.4,
}

AMD_TFLOPS_FP16 = {
    # RX 7000 series (RDNA 3 — FP16 matrix)
    "Radeon RX 7900 XTX": 122.8, "Radeon RX 7900 XT": 103.0,
    "Radeon RX 7800 XT": 74.6, "Radeon RX 7700 XT": 68.4,
    "Radeon RX 7600": 43.0,
    # RX 6000 series (RDNA 2 — FP16 matrix)
    "Radeon RX 6950 XT": 94.6, "Radeon RX 6900 XT": 83.0,
    "Radeon RX 6800 XT": 74.8, "Radeon RX 6800": 61.2,
    "Radeon RX 6700 XT": 42.0, "Radeon RX 6650 XT": 25.4,
    "Radeon RX 6600": 21.0,
    # Mobile / integrated (the user's own machines)
    "Radeon RX 7700S": 23.6, "Radeon RX 6700S": 17.2,
    "Radeon 780M Graphics": 8.4, "Radeon 760M Graphics": 5.4,
    "Radeon 680M Graphics": 6.0, "Radeon Graphics": 2.0,
}

APPLE_TFLOPS_FP16 = {
    "M4 Max": 54.4, "M4 Pro": 27.2, "M4": 8.0,
    "M3 Max": 51.2, "M3 Pro": 22.0, "M3": 6.8,
    "M2 Max": 27.0, "M2 Pro": 13.6, "M2": 7.2,
    "M1 Max": 21.2, "M1 Pro": 10.6, "M1": 5.2,
}


def lookup_tflops_fp16(name: str) -> Optional[float]:
    """Find peak FP16 TFLOPs (tensor / matrix) for a GPU by name.
    Best-effort substring match. This is the metric the picker uses."""
    if not name:
        return None
    n = name.strip()
    for table in (NVIDIA_TFLOPS_FP16, AMD_TFLOPS_FP16, APPLE_TFLOPS_FP16):
        for key, val in table.items():
            if key in n or n in key:
                return val
    return None


def lookup_tflops_fp32(name: str) -> Optional[float]:
    """Reference only: peak FP32 (shader / CUDA-core) TFLOPs. Roughly half
    of FP16 for tensor-core GPUs; used for the CPU benchmark's comparison
    column. The picker does NOT use this."""
    if not name:
        return None
    # Half of FP16 is a fine rule of thumb for tensor-core hardware and for
    # Apple GPUs; for older AMD (pre-RDNA) it's closer to identical. Without
    # a more authoritative source we just divide.
    fp16 = lookup_tflops_fp16(name)
    return round(fp16 / 2.0, 1) if fp16 is not None else None


# ---------------------------------------------------------------------------
# Probe backends (each returns a list of GPU dicts or []).
# ---------------------------------------------------------------------------

def _run(cmd, timeout=5):
    """Run a shell command; return (returncode, stdout, stderr) or None on FileNotFoundError."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        return out.returncode, out.stdout, out.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _probe_nvidia() -> list:
    if not shutil.which("nvidia-smi"):
        return []
    rc = _run(["nvidia-smi",
               "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader,nounits"])
    if not rc or rc[0] != 0:
        return []
    gpus = []
    for line in rc[1].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name, mem_mib = parts[0], parts[1]
        try:
            vram_gb = float(mem_mib) / 1024.0
        except ValueError:
            vram_gb = 0.0
        fp16 = lookup_tflops_fp16(name)
        gpus.append({
            "vendor": "nvidia",
            "name": name,
            "vram_gb": round(vram_gb, 2),
            "tflops_fp16": fp16,
            "tflops_fp32": lookup_tflops_fp32(name),
            "raw_query": line.strip(),
        })
    return gpus


def _probe_rocm() -> list:
    if not shutil.which("rocm-smi"):
        return []
    rc = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not rc or rc[0] != 0:
        return []
    try:
        data = json.loads(rc[1])
    except json.JSONDecodeError:
        return []
    gpus = []
    for key, card in data.items():
        if not isinstance(card, dict):
            continue
        series = card.get("Card Series") or card.get("Card Name") or card.get("Product Name")
        model = card.get("Card Model") or ""
        vram_total = card.get("VRAM Total Memory (B)") or card.get("vram_total_memory")
        name = series or model or key
        vram_gb = 0.0
        if vram_total:
            try:
                vram_gb = float(vram_total) / (1024 ** 3)
            except (TypeError, ValueError):
                vram_gb = 0.0
        else:
            # try parsing memory.use_info text
            mem = card.get("memory") or {}
            tot = mem.get("vram Total Memory (B)")
            if tot:
                try:
                    vram_gb = float(tot) / (1024 ** 3)
                except (TypeError, ValueError):
                    pass
        fp16 = lookup_tflops_fp16(name)
        gpus.append({
            "vendor": "amd",
            "name": name,
            "vram_gb": round(vram_gb, 2),
            "tflops_fp16": fp16,
            "tflops_fp32": lookup_tflops_fp32(name),
            "raw_query": key,
        })
    return gpus


def _probe_apple_silicon() -> list:
    if platform.system() != "Darwin":
        return []
    rc = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if not rc or rc[0] != 0:
        return []
    cpu_name = rc[1].strip()
    # Unified memory — use the system's total RAM as a rough VRAM proxy.
    rc2 = _run(["sysctl", "-n", "hw.memsize"])
    mem_gb = 0.0
    if rc2 and rc2[0] == 0 and rc2[1].strip().isdigit():
        mem_gb = int(rc2[1].strip()) / (1024 ** 3)
    fp16 = lookup_tflops_fp16(cpu_name)
    return [{
        "vendor": "apple",
        "name": cpu_name,
        "vram_gb": round(mem_gb, 2),     # unified memory
        "tflops_fp16": fp16,
        "tflops_fp32": lookup_tflops_fp32(cpu_name),
        "raw_query": cpu_name,
    }]


def _probe_lspci() -> list:
    """Last-resort probe: enumerate PCI devices, return raw names (no TFLOPs)."""
    if not shutil.which("lspci"):
        return []
    rc = _run(["lspci", "-mm"])
    if not rc or rc[0] != 0:
        return []
    gpus = []
    for line in rc[1].splitlines():
        # "VGA compatible controller" or "3D controller" or "Display controller"
        if not any(k in line for k in ("VGA", "3D controller", "Display controller")):
            continue
        # lspci -mm format: "Slot | Class | Vendor | Device | SVendor | SDevice"
        parts = [p.strip().strip('"') for p in line.split("|")]
        if len(parts) < 4:
            continue
        name = parts[3]
        gpus.append({
            "vendor": "unknown",
            "name": name,
            "vram_gb": 0.0,
            "tflops_fp16": None,
            "tflops_fp32": None,
            "raw_query": line.strip(),
        })
    return gpus


# ---------------------------------------------------------------------------
# CPU benchmark (numpy matmul) — short, deterministic-ish budget.
# ---------------------------------------------------------------------------

def _cpu_tflops_estimate(budget_s: float = 2.0) -> float:
    """Run repeated (1024 x 1024) @ (1024 x 1024) matmuls for `budget_s` seconds
    and report FP32 TFLOPS. Uses BLAS through numpy — single-threaded is fine
    for the rough signal we need."""
    n = 1024
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    flops_per = 2.0 * n ** 3
    # warm-up
    np.dot(a, b)
    iters = 0
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        np.dot(a, b)
        iters += 1
    elapsed = max(1e-6, time.monotonic() - (deadline - budget_s))
    return round(flops_per * iters / elapsed / 1e12, 3)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def probe(include_cpu_bench: bool = True, cpu_bench_budget_s: float = 2.0) -> dict:
    """Run all probes. Returns a dict ready for `pick()` and for the CLI to display.

    `include_cpu_bench=False` skips the numpy matmul (faster, useful for tests)."""
    gpus = []
    for backend in (_probe_nvidia, _probe_rocm, _probe_apple_silicon, _probe_lspci):
        try:
            found = backend()
        except Exception:                          # noqa: BLE001
            found = []
        # If a stronger backend already identified GPUs, skip the weaker lspci
        # fallback (it can't give TFLOPs and would just duplicate). Keep the
        # first non-empty list.
        if found and not gpus:
            gpus = found
        elif found and backend is _probe_lspci and all(g.get("tflops_fp16") for g in gpus):
            # richer probes already covered it
            continue
    # dGPU > iGPU: prefer the one with the highest tflops_fp16.
    best_gpu = None
    if gpus:
        scored = [g for g in gpus if isinstance(g.get("tflops_fp16"), (int, float))]
        if scored:
            best_gpu = max(scored, key=lambda g: g["tflops_fp16"])["name"]

    if include_cpu_bench:
        cpu_tflops_fp32 = _cpu_tflops_estimate(cpu_bench_budget_s)
    else:
        cpu_tflops_fp32 = None

    return {
        "schema": 2,                              # bumped: probe schema now reports FP16
        "cpu_tflops_fp32": cpu_tflops_fp32,        # CPU bench is FP32 (x86 has no FP16 SIMD)
        "gpus": gpus,
        "best_gpu": best_gpu,
    }


def pick(detected: dict) -> str:
    """Pick the highest preset whose min FP16 TFLOPs and min VRAM are both met.
    Falls back to `tiny` if nothing fits. CPU-only installs get `tiny`.

    GPU scoring uses FP16 tensor throughput (what actually runs an embedding
    model); the CPU fallback uses the FP32 numpy benchmark as a rough proxy,
    halved to estimate equivalent FP16 throughput on a CPU."""
    gpus = detected.get("gpus") or []
    scored = [g for g in gpus if isinstance(g.get("tflops_fp16"), (int, float))]
    best = max(scored, key=lambda g: g["tflops_fp16"]) if scored else None
    tflops = (best["tflops_fp16"] if best
              else ((detected.get("cpu_tflops_fp32") or 0.0) / 2.0))
    vram = (best["vram_gb"] if best else 0.0)
    for name in ("large", "medium", "small", "tiny"):
        p = PRESETS[name]
        if tflops >= p["min_tflops_fp16"] and vram >= p["min_vram_gb"]:
            return name
    return "tiny"


def alternatives(detected: dict) -> list:
    """All presets the hardware can run, descending."""
    gpus = detected.get("gpus") or []
    scored = [g for g in gpus if isinstance(g.get("tflops_fp16"), (int, float))]
    best = max(scored, key=lambda g: g["tflops_fp16"]) if scored else None
    tflops = (best["tflops_fp16"] if best
              else ((detected.get("cpu_tflops_fp32") or 0.0) / 2.0))
    vram = (best["vram_gb"] if best else 0.0)
    out = []
    for name in ("large", "medium", "small", "tiny"):
        p = PRESETS[name]
        if tflops >= p["min_tflops_fp16"] and vram >= p["min_vram_gb"]:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# CLI (so you can `python device_probe.py` to see what we see).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="rag-mcp hardware probe")
    ap.add_argument("--no-cpu-bench", action="store_true",
                    help="skip the numpy matmul benchmark")
    ap.add_argument("--json", action="store_true", help="print JSON instead of pretty text")
    ap.add_argument("--bench-budget", type=float, default=2.0,
                    help="seconds to spend on the CPU matmul benchmark")
    args = ap.parse_args()

    result = probe(include_cpu_bench=not args.no_cpu_bench,
                   cpu_bench_budget_s=args.bench_budget)
    result["recommendation"] = pick(result)
    result["alternatives"] = alternatives(result)

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        print(f"CPU: ~{result['cpu_tflops_fp32']} TFLOPS FP32 (numpy matmul, "
              f"~halved for FP16)")
        if result["gpus"]:
            for g in result["gpus"]:
                fp16 = g.get("tflops_fp16")
                fp32 = g.get("tflops_fp32")
                fp16_s = f"{fp16} FP16 / {fp32} FP32" if fp16 else "unknown"
                print(f"GPU: {g['name']} ({g['vendor']}) — "
                      f"{g['vram_gb']} GB VRAM, {fp16_s} TFLOPS")
        else:
            print("GPU: none detected")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Alternatives:   {', '.join(result['alternatives']) or '(none)'}")
