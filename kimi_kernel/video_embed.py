#!/usr/bin/env python3
"""video_embed.py — Native Qwen3-VL video-path frame embedding (CONTRACT 1).

Embeds a `<slug>.frames/` directory via Qwen3-VL's **native video temporal-merge
path** (`pixel_values_videos`, temporal patch-merge with patch size 2) instead of
one independent still image per frame. Frames are processed in consecutive
non-overlapping windows of `W` frames (default W=32) so the entire timeline is
represented WITHOUT dropping any frame, and WITHOUT lowering spatial resolution.

Two inviolable laws from TECH_CONTEXT.md are honored:

  1. No spatial downscale. Per-frame `max_pixels` is held at
     `FRAME_MAX_PIXELS = 768 * 32 * 32 = 786_432` (~896x896) — exactly what the
     still-image path's per-frame budget would be at the same per-frame setting.
     We raise `total_pixels` (the *video* budget) to `W // 2 * FRAME_MAX_PIXELS`
     so each frame keeps its full per-frame budget after the temporal-merge
     decoder's `min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR)`
     ceiling — see qwen_vl_utils.vision_process.fetch_video.
  2. Native video path. We call `embedder.process([{"video": paths, ...}])`,
     which routes through `format_model_input(video=[paths])` ->
     `process_vision_info` -> `pixel_values_videos` + `video_grid_thw`. The
     model's own temporal patch-merge then collapses pairs of frames into a
     single token. We do NOT roll our own.

The script honors the TECH_CONTEXT VRAM guard: it refuses to load the 2B model
on any device with < 5 GB free VRAM. When triggered (the live 7700S embed job
holds ~98% of VRAM at contract time), it falls back to:

  * A **token-count analysis** (tokens/window vs tokens/frame — the documented
    "valid theoretical speedup proxy" per KIMI_CONTRACT_1 clause 5).
  * A small **CPU correctness selftest** (`--selftest`) that exercises the
    native video path end-to-end on tiny synthetic frames so we can prove the
    path works and text->window retrieval round-trips, without touching the
    live GPU.

Usage:
    /home/gabriel/venv/bin/python video_embed.py --frames <framedir> --window 32
    /home/gabriel/venv/bin/python video_embed.py --selftest
    /home/gabriel/venv/bin/python video_embed.py --frames <framedir> --device cuda:1
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# IMPORTANT: env vars for GPU selection MUST be set before any `import torch`
# (and before importing qwen3_vl_embedding, which itself imports torch).
# We do the device-string -> env-var mapping in `parse_args_apply_env()` below,
# called from main(), so the parent's env is left alone unless the user passes
# --device. Default: leave the parent env untouched (no GPU claim).
# ---------------------------------------------------------------------------

# Local import of parent-dir modules (after env vars are set in main()).
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

# Constants — mirrored from qwen3_vl_embedding.py (Law 1: do not change).
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2          # 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR   # 4096
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR  # 1_843_200  (still-image per-frame ceiling)
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR  # 786_432  (video per-frame ceiling)
DEFAULT_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS   # 7_864_320  (the model's stock video budget)
MAX_FRAMES = 64                                # hard cap from the model
DEFAULT_WINDOW = 32                            # CONTRACT 1 default
PATCH_FACTOR = 2                               # temporal merge window inside Qwen3-VL
VRAM_FLOOR_BYTES = 5 * 1024 ** 3               # TECH_CONTEXT clause 4: do not load below 5 GB free

# Where we write window outputs. NOT the live index — that's `index/`. We write
# here so Claude can review the prototype before any promotion to ingest.py.
OUTPUT_DIR = HERE / "output"


def parse_args_apply_env(argv=None):
    """Parse CLI args and, if --device is given, set HIP_VISIBLE_DEVICES so
    torch sees only that device (cuda:0 in the process). 780M iGPU also needs
    HSA_OVERRIDE_GFX_VERSION=11.0.0 to avoid ROCm GEMM NOT_SUPPORTED."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=Path,
                    help="Path to a <slug>.frames/ directory (frames.json + .jpg files).")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"Window size in frames (default {DEFAULT_WINDOW}). "
                         f"Hard-capped to {MAX_FRAMES}.")
    ap.add_argument("--selftest", action="store_true",
                    help="Run a tiny CPU correctness check on the native video path.")
    ap.add_argument("--device", choices=("cuda:0", "cuda:1", "cpu"), default=None,
                    help="Target device. Default: auto (cuda:0 if free VRAM >= 5GB, "
                         "else defer). 'cuda:1' is the 780M iGPU and auto-sets "
                         "HSA_OVERRIDE_GFX_VERSION=11.0.0.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip model load entirely; just print the window plan and "
                         "token-count analysis.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override the output JSONL path (default: "
                         "kimi_kernel/output/<slug>.windows.jsonl).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Embed only the first N frames (for a quick sanity run).")
    args = ap.parse_args(argv)

    # GPU selection via env vars (must happen BEFORE torch import).
    if args.device and args.device.startswith("cuda:"):
        idx = args.device.split(":", 1)[1]
        os.environ["HIP_VISIBLE_DEVICES"] = idx
        os.environ["ROCR_VISIBLE_DEVICES"] = idx
        os.environ["CUDA_VISIBLE_DEVICES"] = idx
        if idx == "1":
            # 780M gfx1103 needs this or GEMM fails NOT_SUPPORTED.
            os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    elif args.device == "cpu" or args.selftest:
        # Force CPU: hide all GPUs from torch *before* it imports. The parent
        # Qwen3VLEmbedder constructor hard-codes
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # and there is no kwarg to override it. Emptying CUDA_VISIBLE_DEVICES
        # is the only way to keep the model off cuda:0 (which is the live
        # embedder worker — TECH_CONTEXT §4 VRAM guard).
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""
        os.environ["ROCR_VISIBLE_DEVICES"] = ""
    return args


# ---------------------------------------------------------------------------
# VRAM guard
# ---------------------------------------------------------------------------
def free_vram_bytes(device_idx: int) -> int:
    """Best-effort free VRAM on cuda:device_idx, in bytes. Returns 0 on CPU or
    if torch/rocm-smi can't tell us. Uses torch.cuda when available, falls back
    to rocm-smi via /sys (no subprocess)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        if device_idx >= torch.cuda.device_count():
            return 0
        free, _total = torch.cuda.mem_get_info(device_idx)
        return int(free)
    except Exception:
        pass
    # Fallback: rocm-smi subprocess (last resort; only called once).
    try:
        import subprocess
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "-d", str(device_idx)],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            # e.g. "GPU[0]      : VRAM Total Used Memory (B): 8485187584"
            if "Total" in line and "Used" in line:
                # Parse "<label>: VRAM Total Used Memory (B): <N>"
                parts = line.rsplit(":", 1)
                used = int(parts[1].strip())
                # Hard-coded total from rocm-smi is not in the parsed output here;
                # assume 8 GB for 7700S, 16 GB for others — but only used is reliable.
                # We'll return a tiny sentinel; this branch is the rare fallback.
                return 0
    except Exception:
        pass
    return 0


def select_device(requested: str | None) -> tuple[str, int, str]:
    """Decide which (label, idx, reason) to use. Returns ("cpu", -1, "no-free-vram")
    if we have to defer the real run."""
    if requested == "cpu":
        return "cpu", -1, "user-requested-cpu"
    idx = 0 if requested in (None, "cuda:0") else 1
    free = free_vram_bytes(idx)
    if free >= VRAM_FLOOR_BYTES:
        return f"cuda:{idx}", idx, f"{free/1024**3:.1f}GB-free"
    return "cpu", -1, f"vram-guard: cuda:{idx} only {free/1024**3:.2f}GB free (<5GB)"


# ---------------------------------------------------------------------------
# Windowing — every frame is covered, no silent drops
# ---------------------------------------------------------------------------
def load_manifest(framedir: Path) -> dict:
    with open(framedir / "frames.json") as f:
        return json.load(f)


def make_windows(frames: list[dict], W: int, framedir: Path,
                 limit: int | None = None):
    """Yield (idx_start, idx_end_exclusive, paths, captions, t_start, t_end,
    middle_path) for every non-overlapping window of up to W frames. The last
    window may be shorter if `len(frames)` is not a multiple of W; we pad it
    implicitly by passing the same num_frames=W to the model (the model
    handles short tails via the qwen_vl_utils pad-with-last-frame behavior —
    no frame is dropped from the timeline)."""
    fs = frames if limit is None else frames[:limit]
    n = len(fs)
    if n == 0:
        return
    for s in range(0, n, W):
        e = min(s + W, n)
        chunk = fs[s:e]
        paths = [str(framedir / fr["file"]) for fr in chunk]
        caps = [(fr.get("cc") or "").strip() for fr in chunk]
        ts = [fr.get("t", 0.0) for fr in chunk]
        # Representative middle frame (used as ingest.py "image" field for visual recall).
        mid = chunk[len(chunk) // 2]
        yield {
            "idx_start": s,
            "idx_end": e,
            "n_frames": len(chunk),
            "paths": paths,
            "captions": caps,
            "cc_text": " ".join(c for c in caps if c),
            "t_start": ts[0],
            "t_end": ts[-1],
            "middle_path": str(framedir / mid["file"]),
        }


# ---------------------------------------------------------------------------
# Token-count analysis (theoretical speedup proxy, valid per CONTRACT clause 5)
# ---------------------------------------------------------------------------
def token_count_analysis(n_frames: int, W: int,
                         per_frame_pixels: int = FRAME_MAX_PIXELS) -> dict:
    """Estimate tokens per WINDOW (video path) vs tokens per FRAME (still path).

    Spatial tokens per frame = ceil(per_frame_pixels / IMAGE_FACTOR^2)
                              = ceil(per_frame_pixels / 1024)

    Temporal merge collapses FRAME_FACTOR (=2) consecutive frames into one token,
    so tokens-per-frame-after-temporal-merge = spatial_tokens_per_frame / FRAME_FACTOR.

    Plus a constant budget for the system + user prompt text (we use ~64 text
    tokens as a conservative lower bound — Qwen3-VL chat-template overhead).
    """
    spatial_per_frame = per_frame_pixels // (IMAGE_FACTOR * IMAGE_FACTOR)
    text_overhead = 64

    # --- still image path (the baseline) ---
    per_image_spatial = spatial_per_frame
    per_image_total = per_image_spatial + text_overhead

    # --- native video path ---
    # After temporal merge (frame_factor=2), every FRAME_FACTOR frames contribute
    # `spatial_per_frame` tokens (not /FRAME_FACTOR; the temporal merge happens
    # *within* the vision encoder before tokenization, so each pair-of-frames
    # still contributes spatial_per_frame tokens, but spread over 2 frames of
    # visual content). This is the standard Qwen3-VL behavior — see
    # `temporal_patch_size=2` in modeling_qwen3_vl.py.
    per_window_spatial = (W // PATCH_FACTOR) * spatial_per_frame
    per_window_total = per_window_spatial + text_overhead

    windows_needed = (n_frames + W - 1) // W
    still_tokens_total = n_frames * per_image_total
    video_tokens_total = windows_needed * per_window_total
    token_ratio = video_tokens_total / still_tokens_total if still_tokens_total else 0.0

    return {
        "n_frames": n_frames,
        "W": W,
        "windows": windows_needed,
        "per_frame_pixels": per_frame_pixels,
        "spatial_tokens_per_frame": spatial_per_frame,
        "text_overhead_tokens": text_overhead,
        "still_image": {
            "tokens_per_frame": per_image_total,
            "tokens_for_video": still_tokens_total,
        },
        "video_path": {
            "tokens_per_window": per_window_total,
            "tokens_for_video": video_tokens_total,
            "per_window_spatial_tokens": per_window_spatial,
        },
        "token_ratio_video_over_still": token_ratio,
        "theoretical_speedup_proxy": 1.0 / token_ratio if token_ratio else 0.0,
    }


# ---------------------------------------------------------------------------
# In-process embedding (native video path). Loaded lazily so we never import
# torch until VRAM is confirmed.
# ---------------------------------------------------------------------------
def total_pixels_for(W: int) -> int:
    """Pick `total_pixels` so the qwen_vl_utils per-frame ceiling stays at
    FRAME_MAX_PIXELS (i.e. NO spatial downscale).

    fetch_video computes:
        max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS,
                             total_pixels / nframes * FRAME_FACTOR),
                         int(min_pixels * 1.05))
    We want max_pixels >= FRAME_MAX_PIXELS, so we need:
        total_pixels / nframes * FRAME_FACTOR >= FRAME_MAX_PIXELS
        total_pixels >= nframes * FRAME_MAX_PIXELS / FRAME_FACTOR
        total_pixels >= W * FRAME_MAX_PIXELS // 2
    """
    return max(W // PATCH_FACTOR, 1) * FRAME_MAX_PIXELS


def load_embedder(model_id: str, device: str, W: int, dtype):
    """Construct a Qwen3VLEmbedder whose native video path preserves per-frame
    spatial fidelity (FRAME_MAX_PIXELS) and uses exactly W frames per call.
    Lazy import so torch is only loaded once we've cleared the VRAM guard.

    Note: the parent Qwen3VLEmbedder.__init__ hard-codes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    and moves the model there. We don't override that — the caller's
    responsibility is to make CUDA visible (or not) before this import.

    max_length is raised to 32768: a W=32 window with concatenated CC strings
    can exceed 8192 tokens (the stock limit) and Qwen3-VL's processor then
    truncates text → video, raising 'Mismatch in `video` token count'.
    """
    from qwen3_vl_embedding import Qwen3VLEmbedder  # parent-dir module
    return Qwen3VLEmbedder(
        model_id,
        max_length=32768,                     # raised from 8192 (CONTRACT 2 fix)
        min_pixels=MIN_PIXELS,                # do NOT lower (Law 1)
        max_pixels=MAX_PIXELS,                # do NOT lower (Law 1)
        total_pixels=total_pixels_for(W),     # raised so per-frame stays full
        num_frames=W,
        max_frames=W,
        torch_dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_real(args, embedder, windows, analysis: dict, framedir: Path):
    """Run the embedding loop, write per-window vectors + ingest-compatible
    metadata. Returns (n_windows, wall_time)."""
    import numpy as np

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (OUTPUT_DIR / (framedir.name + ".windows.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fresh file each run — caller manages re-ingest upstream (like ingest.py).
    if out_path.exists():
        out_path.unlink()

    n_windows = 0
    t0 = time.time()
    per_window_times = []
    with open(out_path, "w") as f:
        for win in windows:
            w_t0 = time.time()
            items = [{"video": win["paths"], "text": win["cc_text"]}]
            vecs = embedder.process(items, normalize=True)   # (1, 2048), float32, L2-norm
            vec = vecs[0].detach().cpu().numpy().astype("float32")
            per_window_times.append(time.time() - w_t0)

            meta = {
                "source": str(framedir),
                "slug": framedir.name,
                "modality": "video-window",
                "window": args.window,
                "idx_start": win["idx_start"],
                "idx_end": win["idx_end"],
                "n_frames": win["n_frames"],
                "t_start": win["t_start"],
                "t_end": win["t_end"],
                "image": win["middle_path"],         # ingest.py compatible visual proxy
                "text": win["cc_text"],              # CC union (text-side recall)
                "cc_first": win["captions"][0] if win["captions"] else "",
                "cc_last": win["captions"][-1] if win["captions"] else "",
            }
            rec = {
                **meta,
                "dim": int(vec.shape[0]),
                "vec_b64": base64.b64encode(vec.tobytes()).decode("ascii"),
            }
            f.write(json.dumps(rec) + "\n")
            n_windows += 1
            print(f"  [window {n_windows:>4}] "
                  f"frames[{win['idx_start']:>4}..{win['idx_end']:>4}] "
                  f"t={win['t_start']:.0f}..{win['t_end']:.0f}s "
                  f"({time.time()-w_t0:.2f}s)", flush=True)

    wall = time.time() - t0
    total_frames = analysis["n_frames"]
    fps = total_frames / wall if wall > 0 else 0.0
    avg_w = sum(per_window_times) / len(per_window_times) if per_window_times else 0.0

    summary = {
        "frames_embedded": total_frames,
        "windows": n_windows,
        "wall_seconds": wall,
        "avg_window_seconds": avg_w,
        "frames_per_second": fps,
        "videos_per_hour_estimate_45min": 3600 / (fps * 2700) if fps > 0 else 0.0,
        "out_path": str(out_path),
    }
    print()
    print(f"Done: {n_windows} windows / {total_frames} frames in {wall:.1f}s "
          f"({fps:.2f} frames/s; {avg_w:.2f}s/window)")
    print(f"Out:  {out_path}")
    return summary


def _jsonable(o):
    """Recursively convert Path -> str so json.dumps works."""
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    return o


def write_summary(framedir: Path, args, analysis, summary, device_label: str,
                  reason: str):
    """Write a machine-readable summary next to the windows file so the
    REPORT_*.md can quote it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sum_path = OUTPUT_DIR / (framedir.name + ".summary.json")
    payload = _jsonable({
        "framedir": str(framedir),
        "args": vars(args),
        "device": device_label,
        "reason": reason,
        "analysis": analysis,
        "summary": summary,
    })
    sum_path.write_text(json.dumps(payload, indent=2))
    return sum_path


# ---------------------------------------------------------------------------
# --selftest (CPU-correctness round-trip)
# ---------------------------------------------------------------------------
def run_selftest(args):
    """Tiny CPU correctness check. Builds 4 synthetic 720p-like frames, embeds
    them via the native video path AND via the still-image path, then runs a
    text->window retrieval. All CPU to honor the VRAM guard."""
    import numpy as np
    from PIL import Image

    W = 4  # tiny window
    model_id = os.environ.get("RAG_MODEL", "Qwen/Qwen3-VL-Embedding-2B")

    print(f"[selftest] loading {model_id} on cpu (float32) — slow but VRAM-safe …")
    t0 = time.time()
    embedder = load_embedder(model_id, device="cpu", W=W, dtype="float32")
    print(f"[selftest] model ready in {time.time()-t0:.1f}s")

    # 4 synthetic frames with distinct color-coded "scenes" + a tiny caption.
    frames = []
    captions = []
    for i, (color, caption) in enumerate([
        ((255,   0,   0), "a red sunset over the ocean"),       # 0
        ((  0, 255,   0), "a green forest with tall pine trees"),  # 1
        ((  0,   0, 255), "a deep blue night sky with stars"),    # 2
        ((255, 255,   0), "a bright yellow sunflower field"),     # 3
    ]):
        img = Image.new("RGB", (720, 720), color)
        path = OUTPUT_DIR / f"_selftest_frame_{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, "JPEG", quality=85)
        frames.append(str(path))
        captions.append(caption)

    # 1) Native video path: one window of W=4 -> one vector.
    video_text = " ".join(captions)
    vec_win = embedder.process(
        [{"video": frames, "text": video_text}], normalize=True
    )[0].detach().cpu().numpy().astype("float32")
    print(f"[selftest] video-window vec shape={vec_win.shape}, "
          f"norm={float(np.linalg.norm(vec_win)):.4f}")

    # 2) Still-image path baseline (per-frame, no temporal merge).
    per_frame_vecs = embedder.process(
        [{"image": p, "text": c} for p, c in zip(frames, captions)],
        normalize=True,
    ).detach().cpu().numpy().astype("float32")

    # 3) Retrieval round-trip: embed the text query, find nearest window.
    q_vec = embedder.process([{"text": "tell me about the sunflower field"}],
                             normalize=True)[0].detach().cpu().numpy().astype("float32")
    sim_video = float(np.dot(q_vec, vec_win))
    sim_frame = [float(np.dot(q_vec, v)) for v in per_frame_vecs]
    print(f"[selftest] text->video-window cos  = {sim_video:+.4f}")
    print(f"[selftest] text->per-frame  cos    = "
          f"{['%.3f'%s for s in sim_frame]}  (frame 3 = sunflower)")

    # 4) Cross-modal sanity: video-window vec vs each per-frame vec.
    cross = [float(np.dot(vec_win, v)) for v in per_frame_vecs]
    print(f"[selftest] video-window vs each still-image vec = "
          f"{['%.3f'%c for c in cross]}")

    ok = (
        # window != garbage: norm ~ 1.0 (L2-normalized)
        abs(float(np.linalg.norm(vec_win)) - 1.0) < 0.05
        # the sunflower frame (index 3) is the top hit for the sunflower query
        and int(np.argmax(sim_frame)) == 3
        # video-window cos with the sunflower query should beat the others' avg
        and sim_video > (sum(sim_frame) / len(sim_frame))
    )
    print(f"[selftest] RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args_apply_env(argv)

    if args.selftest:
        ok = run_selftest(args)
        sys.exit(0 if ok else 1)

    if not args.frames:
        print("ERROR: --frames <dir> is required (or pass --selftest).",
              file=sys.stderr)
        sys.exit(2)

    framedir: Path = args.frames
    if not (framedir / "frames.json").exists():
        print(f"ERROR: no frames.json under {framedir}", file=sys.stderr)
        sys.exit(2)

    # Hard-cap window to model's MAX_FRAMES.
    W = max(1, min(args.window, MAX_FRAMES))

    # Load manifest, build windows (no model load yet).
    manifest = load_manifest(framedir)
    frames_meta = manifest.get("frames", [])
    if args.limit:
        frames_meta = frames_meta[: args.limit]
    windows = list(make_windows(frames_meta, W, framedir, limit=None))
    analysis = token_count_analysis(len(frames_meta), W)

    # Decide device.
    device_label, device_idx, reason = select_device(args.device)
    print(f"[plan] framedir={framedir}  frames={len(frames_meta)}  "
          f"W={W}  windows={len(windows)}")
    print(f"[plan] total_pixels={total_pixels_for(W)}  "
          f"(per-frame ceiling stays at {FRAME_MAX_PIXELS} = "
          f"{int(FRAME_MAX_PIXELS**0.5)}x{int(FRAME_MAX_PIXELS**0.5)})")
    print(f"[plan] device={device_label}  ({reason})")
    print(f"[plan] tokens/window={analysis['video_path']['tokens_per_window']}  "
          f"vs tokens/frame={analysis['still_image']['tokens_per_frame']}  "
          f"(theoretical speedup proxy x{analysis['theoretical_speedup_proxy']:.2f})")

    if args.dry_run or device_label == "cpu":
        # Defer: write summary + token analysis only.
        summary = {"status": "DEFERRED",
                   "reason": reason,
                   "frames_embedded": 0, "windows": 0,
                   "wall_seconds": 0.0, "frames_per_second": 0.0}
        sum_path = write_summary(framedir, args, analysis, summary,
                                 device_label, reason)
        print(f"[deferred] no model load. Summary: {sum_path}")
        return

    # Real run on the chosen GPU.
    import torch
    dtype = torch.float16 if device_label != "cpu" else torch.float32
    print(f"[run] loading model on {device_label} (dtype={dtype}) …")
    t0 = time.time()
    embedder = load_embedder(
        os.environ.get("RAG_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
        device_label, W, dtype,
    )
    print(f"[run] model ready in {time.time()-t0:.1f}s")

    summary = run_real(args, embedder, windows, analysis, framedir)
    sum_path = write_summary(framedir, args, analysis, summary,
                             device_label, reason)
    print(f"[summary] {sum_path}")


if __name__ == "__main__":
    main()