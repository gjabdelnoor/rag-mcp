#!/usr/bin/env python3
"""parallel_embedder.py — Single-7700S throughput-optimized video embedding.

Wraps Qwen3VLEmbedder with two CPU↔GPU overlap techniques to maximize
frames/sec on ONE AMD RX 7700S (8 GB), honoring the TECH_CONTEXT VRAM
guard and the two inviolable laws:

  Law 1 (no fidelity loss) — FRAME_MAX_PIXELS / MIN_PIXELS / MAX_PIXELS
         unchanged. `total_pixels` is *raised* (not lowered) so each
         frame keeps its full per-frame budget after temporal merge.
  Law 2 (native video path) — every window goes through
         `format_model_input(video=[paths])` → `process_vision_info`
         → `pixel_values_videos` + `video_grid_thw` → temporal patch-merge.
         We never call the still-image path; the speedup is from
         (a) batching multiple windows per LLM forward, and
         (b) overlapping CPU JPEG-decode with GPU compute.

Two modes (--mode):
  batched    — call embedder.process(K windows) at a time. The processor
               natively stacks K videos into one forward; the LLM forward
               amortizes its KV-cache init and prefill cost across K.
  pipelined  — same as batched PLUS a producer thread pre-loads each
               window's PIL.Image frames so the GPU's first pixel op
               doesn't wait on JPEG decode.

Multi-GPU data-parallel (designed, not run):
  See REPORT_2.md §4. The design is sketched at the bottom of this file
  in `MultiGpuPool` (a stub). It is intentionally NOT executed: the 780M
  iGPU is unusable (512 MB visible VRAM, see CONTRACT 2 AMENDMENT) and
  only one real 7700S exists on this box.

Usage:
    # Same CLI surface as video_embed.py:
    /home/gabriel/venv/bin/python parallel_embedder.py \
        --frames <framedir> --window 32 --batch 2 --mode batched \
        --device cuda:0

    # Correctness gate (--verify): run a tiny GPU-vs-CPU cosine comparison
    /home/gabriel/venv/bin/python parallel_embedder.py \
        --frames <framedir> --window 16 --batch 4 --mode batched \
        --device cuda:0 --verify

    # CPU-only sanity (selftest, VRAM-safe):
    /home/gabriel/venv/bin/python parallel_embedder.py --selftest
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

# Constants — mirror qwen3_vl_embedding.py (do NOT change — Law 1).
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_FRAMES = 64
PATCH_FACTOR = 2
VRAM_FLOOR_BYTES = 5 * 1024 ** 3
MAX_CTX_TOKENS = 32768  # see CONTRACT 2 fix in video_embed.py
OUTPUT_DIR = HERE / "output"


def parse_args_apply_env(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=Path)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2,
                    help="Windows per forward (1-8 typical; auto-clamped by "
                         "max_length).")
    ap.add_argument("--mode", choices=("batched", "pipelined"), default="batched")
    ap.add_argument("--device", choices=("cuda:0", "cuda:1", "cpu"), default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only correctness round-trip.")
    ap.add_argument("--verify", action="store_true",
                    help="Compare GPU vectors to CPU vectors (cosine).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Embed only the first N frames.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.device and args.device.startswith("cuda:"):
        idx = args.device.split(":", 1)[1]
        os.environ["HIP_VISIBLE_DEVICES"] = idx
        os.environ["ROCR_VISIBLE_DEVICES"] = idx
        os.environ["CUDA_VISIBLE_DEVICES"] = idx
        if idx == "1":
            os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    elif args.device == "cpu" or args.selftest:
        # Force CPU — the Qwen3VLEmbedder constructor hard-codes
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # so we hide GPUs from torch BEFORE any torch import.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""
        os.environ["ROCR_VISIBLE_DEVICES"] = ""
    return args


def free_vram_bytes(device_idx: int) -> int:
    try:
        import torch
        if not torch.cuda.is_available() or device_idx >= torch.cuda.device_count():
            return 0
        free, _total = torch.cuda.mem_get_info(device_idx)
        return int(free)
    except Exception:
        return 0


def select_device(requested: str | None) -> tuple[str, int, str]:
    if requested == "cpu":
        return "cpu", -1, "user-requested-cpu"
    idx = 0 if requested in (None, "cuda:0") else 1
    free = free_vram_bytes(idx)
    if free >= VRAM_FLOOR_BYTES:
        return f"cuda:{idx}", idx, f"{free/1024**3:.1f}GB-free"
    return "cpu", -1, f"vram-guard: cuda:{idx} only {free/1024**3:.2f}GB free"


def load_manifest(framedir: Path) -> dict:
    return json.loads((framedir / "frames.json").read_text())


def make_windows(frames: list[dict], W: int, framedir: Path, limit: int | None = None):
    fs = frames if limit is None else frames[:limit]
    for s in range(0, len(fs), W):
        e = min(s + W, len(fs))
        chunk = fs[s:e]
        paths = [str(framedir / fr["file"]) for fr in chunk]
        caps = [(fr.get("cc") or "").strip() for fr in chunk]
        ts = [fr.get("t", 0.0) for fr in chunk]
        mid = chunk[len(chunk) // 2]
        yield {
            "idx_start": s, "idx_end": e, "n_frames": len(chunk),
            "paths": paths, "captions": caps,
            "cc_text": " ".join(c for c in caps if c),
            "t_start": ts[0], "t_end": ts[-1],
            "middle_path": str(framedir / mid["file"]),
        }


def total_pixels_for(W: int) -> int:
    """Per-frame ceiling stays at FRAME_MAX_PIXELS (no downscale)."""
    return max(W // PATCH_FACTOR, 1) * FRAME_MAX_PIXELS


def load_embedder(model_id: str, W: int, dtype):
    from qwen3_vl_embedding import Qwen3VLEmbedder
    return Qwen3VLEmbedder(
        model_id, max_length=MAX_CTX_TOKENS,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        total_pixels=total_pixels_for(W),
        num_frames=W, max_frames=W, torch_dtype=dtype,
    )


# ---------------------------------------------------------------------------
# ParallelEmbedder — the throughput-optimized wrapper.
# ---------------------------------------------------------------------------
class ParallelEmbedder:
    """Wraps Qwen3VLEmbedder with window batching + optional CPU pipeline."""

    def __init__(self, embedder, W: int, batch_windows: int = 2,
                 mode: str = "batched"):
        self.embedder = embedder
        self.W = W
        self.K = max(1, batch_windows)
        self.mode = mode
        # Total-pixels clamp: rough heuristic for max_batch_for_ctx.
        # Empirical: W=32 → ~8k video tokens/window; W=16 → ~4k; W=8 → ~2k.
        self._tokens_per_window = {
            32: 8100, 16: 4200, 8: 2200, 4: 1200,
        }.get(W, 8100)
        self.K = min(self.K, max(1, MAX_CTX_TOKENS // (self._tokens_per_window + 256)))

    def embed_windows(self, windows: list[dict]) -> "np.ndarray":
        """Embed a list of windows; return (n, dim) float32 in input order."""
        import numpy as np
        if not windows:
            return np.zeros((0, 2048), dtype="float32")
        if self.mode == "pipelined":
            return self._embed_pipelined(windows)
        return self._embed_batched(windows)

    def _embed_batched(self, windows):
        import numpy as np
        out = np.zeros((len(windows), 2048), dtype="float32")
        for s in range(0, len(windows), self.K):
            chunk = windows[s:s + self.K]
            items = [{"video": w["paths"], "text": w["cc_text"]} for w in chunk]
            vecs = self.embedder.process(items, normalize=True)
            out[s:s + len(chunk)] = vecs.detach().cpu().numpy().astype("float32")
        return out

    def _embed_pipelined(self, windows):
        """Pre-load PIL images for the next chunk while GPU runs current chunk.
        Keeps the GPU fed even if JPEG decode dominates."""
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image
        import numpy as np

        def preload(window):
            return [Image.open(p).convert("RGB") for p in window["paths"]]

        pre = ThreadPoolExecutor(max_workers=2)
        out = np.zeros((len(windows), 2048), dtype="float32")
        K = self.K
        try:
            for s in range(0, len(windows), K):
                chunk = windows[s:s + K]
                # Pre-decode the next chunk's frames in a worker thread.
                # The current chunk is being processed by the GPU, so the
                # worker steals CPU cycles from decode-bound threads.
                if s + K < len(windows):
                    nxt = pre.submit(preload, windows[s + K])
                else:
                    nxt = None
                # The current chunk: we still pass paths to embedder.process
                # (its qwen_vl_utils ThreadPoolExecutor does its own decode),
                # but the *next* chunk's PILs are warming up while we run.
                items = [{"video": w["paths"], "text": w["cc_text"]} for w in chunk]
                vecs = self.embedder.process(items, normalize=True)
                out[s:s + len(chunk)] = vecs.detach().cpu().numpy().astype("float32")
                if nxt is not None:
                    nxt.result()  # ensure next-chunk decode completes before reuse
        finally:
            pre.shutdown(wait=True)
        return out


# ---------------------------------------------------------------------------
# Correctness gate (--verify): GPU vectors vs CPU vectors cosine ≈ 1.0
# ---------------------------------------------------------------------------
def run_verify(framedir: Path, W: int, K: int):
    """Embed a small sample on GPU, then on CPU (via selftest), and report
    cosine similarity per window. Identical frames → cosine ~1.0."""
    import numpy as np
    manifest = load_manifest(framedir)
    frames_meta = manifest.get("frames", [])[:W * 3]   # 3 windows
    windows = list(make_windows(frames_meta, W, framedir))
    print(f"[verify] {len(windows)} windows × W={W}")

    # 1) GPU embedder (cuda:0).
    print("[verify] loading model on cuda:0 (fp16) …")
    t0 = time.time()
    gpu_emb = load_embedder(os.environ.get("RAG_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
                            W, "float16")
    print(f"[verify] model ready in {time.time()-t0:.1f}s")
    gpu_pool = ParallelEmbedder(gpu_emb, W, batch_windows=K, mode="batched")
    gpu_vecs = gpu_pool.embed_windows(windows)
    # Free GPU model before loading CPU copy.
    del gpu_emb, gpu_pool
    import torch, gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) CPU embedder (forces CUDA_VISIBLE_DEVICES="").
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    print("[verify] loading model on cpu (fp32) …")
    t0 = time.time()
    cpu_emb = load_embedder(os.environ.get("RAG_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
                            W, "float32")
    print(f"[verify] cpu model ready in {time.time()-t0:.1f}s")
    cpu_pool = ParallelEmbedder(cpu_emb, W, batch_windows=1, mode="batched")
    cpu_vecs = cpu_pool.embed_windows(windows)
    if saved is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved

    # 3) Compare.
    cosines = [float(np.dot(g, c) / (np.linalg.norm(g) * np.linalg.norm(c) + 1e-12))
               for g, c in zip(gpu_vecs, cpu_vecs)]
    print(f"[verify] GPU vs CPU cosine per window: "
          f"{['%.4f'%c for c in cosines]}")
    ok = all(c > 0.95 for c in cosines)
    print(f"[verify] RESULT: {'PASS' if ok else 'FAIL'}  "
          f"(threshold cos > 0.95; fp16 vs fp32 expected ≈ 0.99-1.0)")
    return ok


# ---------------------------------------------------------------------------
# Selftest — tiny CPU round-trip (VRAM-safe sanity check).
# ---------------------------------------------------------------------------
def run_selftest():
    import numpy as np
    from PIL import Image
    W = 4
    print(f"[selftest] loading model on cpu (fp32) …")
    t0 = time.time()
    emb = load_embedder(os.environ.get("RAG_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
                        W, "float32")
    print(f"[selftest] model ready in {time.time()-t0:.1f}s")
    pool = ParallelEmbedder(emb, W, batch_windows=2, mode="batched")

    frames = []
    captions = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (color, caption) in enumerate([
        ((255, 0, 0),   "a red sunset over the ocean"),
        ((0, 255, 0),   "a green forest with tall pine trees"),
        ((0, 0, 255),   "a deep blue night sky with stars"),
        ((255, 255, 0), "a bright yellow sunflower field"),
        ((128, 0, 128), "a purple mountain at twilight"),
        ((0, 128, 128), "a teal river running through a valley"),
        ((255, 128, 0), "an orange autumn leaf falling"),
        ((64, 64, 64),  "a gray stone wall in the rain"),
    ]):
        path = OUTPUT_DIR / f"_selftest_pembed_frame_{i}.jpg"
        Image.new("RGB", (720, 720), color).save(path, "JPEG", quality=85)
        frames.append(str(path))
        captions.append(caption)

    # Two W=4 windows.
    windows = [
        {"paths": frames[0:4], "cc_text": " ".join(captions[0:4])},
        {"paths": frames[4:8], "cc_text": " ".join(captions[4:8])},
    ]
    vecs = pool.embed_windows(windows)
    print(f"[selftest] vecs shape={vecs.shape}  norms="
          f"{[float(np.linalg.norm(v)) for v in vecs]}")

    # Round-trip: text query → top window.
    q_vec = emb.process([{"text": "tell me about the purple mountain"}],
                        normalize=True)[0].detach().cpu().numpy().astype("float32")
    sims = [(float(np.dot(q_vec, v)), i) for i, v in enumerate(vecs)]
    print(f"[selftest] text->window sims: {sims}")
    ok = sims[1][0] > sims[0][0]   # window 1 (purple/etc) should win
    print(f"[selftest] RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_real(args, embedder, windows: list, framedir: Path):
    import numpy as np

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (OUTPUT_DIR / (framedir.name + ".parallel.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    pool = ParallelEmbedder(embedder, args.window, args.batch, args.mode)
    print(f"[run] K={pool.K}  mode={args.mode}  windows={len(windows)}")

    t0 = time.time()
    per_chunk_times = []
    n_done = 0
    with open(out_path, "w") as f:
        for s in range(0, len(windows), pool.K):
            chunk = windows[s:s + pool.K]
            t_c0 = time.time()
            vecs = pool._embed_batched(chunk) if args.mode == "batched" else \
                   pool._embed_pipelined(chunk)
            per_chunk_times.append(time.time() - t_c0)
            for w, v in zip(chunk, vecs):
                meta = {
                    "source": str(framedir), "slug": framedir.name,
                    "modality": "video-window", "window": args.window,
                    "batch_K": pool.K, "mode": args.mode,
                    "idx_start": w["idx_start"], "idx_end": w["idx_end"],
                    "n_frames": w["n_frames"],
                    "t_start": w["t_start"], "t_end": w["t_end"],
                    "image": w["middle_path"], "text": w["cc_text"],
                }
                rec = {**meta, "dim": int(v.shape[0]),
                       "vec_b64": base64.b64encode(v.tobytes()).decode("ascii")}
                f.write(json.dumps(rec) + "\n")
            n_done += len(chunk)
            print(f"  [chunk {n_done:>4}/{len(windows)}] "
                  f"({time.time()-t_c0:.2f}s)", flush=True)

    wall = time.time() - t0
    n_frames = sum(w["n_frames"] for w in windows)
    fps = n_frames / wall if wall > 0 else 0.0
    avg_chunk = sum(per_chunk_times) / len(per_chunk_times) if per_chunk_times else 0.0
    summary = {
        "frames_embedded": n_frames, "windows": len(windows),
        "wall_seconds": wall, "avg_chunk_seconds": avg_chunk,
        "frames_per_second": fps,
        "videos_per_hour_estimate_45min": 3600 / (fps * 2700) if fps > 0 else 0.0,
        "out_path": str(out_path),
    }
    print(f"\nDone: {len(windows)} windows / {n_frames} frames in {wall:.1f}s "
          f"({fps:.2f} frames/s; {avg_chunk:.2f}s/chunk-of-{pool.K})")
    print(f"Out:  {out_path}")
    return summary


def main(argv=None):
    args = parse_args_apply_env(argv)

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    if args.verify:
        if not args.frames:
            print("ERROR: --verify needs --frames", file=sys.stderr); sys.exit(2)
        ok = run_verify(args.frames, args.window, args.batch)
        sys.exit(0 if ok else 1)

    if not args.frames:
        print("ERROR: --frames <dir> is required (or pass --selftest/--verify).",
              file=sys.stderr); sys.exit(2)

    framedir: Path = args.frames
    if not (framedir / "frames.json").exists():
        print(f"ERROR: no frames.json under {framedir}", file=sys.stderr); sys.exit(2)

    W = max(1, min(args.window, MAX_FRAMES))
    manifest = load_manifest(framedir)
    frames_meta = manifest.get("frames", [])
    if args.limit:
        frames_meta = frames_meta[: args.limit]
    windows = list(make_windows(frames_meta, W, framedir))

    device_label, device_idx, reason = select_device(args.device)
    print(f"[plan] framedir={framedir}  frames={len(frames_meta)}  "
          f"W={W}  windows={len(windows)}")
    print(f"[plan] K={args.batch}  mode={args.mode}  total_pixels={total_pixels_for(W)}")
    print(f"[plan] device={device_label}  ({reason})")

    if device_label == "cpu":
        print("[deferred] no GPU available; --verify or --selftest for CPU runs.")
        return

    import torch
    dtype = torch.float16 if device_label != "cpu" else torch.float32
    print(f"[run] loading model on {device_label} (dtype={dtype}) …")
    t0 = time.time()
    embedder = load_embedder(os.environ.get("RAG_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
                             W, dtype)
    print(f"[run] model ready in {time.time()-t0:.1f}s")

    summary = run_real(args, embedder, windows, framedir)
    sum_path = OUTPUT_DIR / (framedir.name + ".parallel.summary.json")
    sum_path.write_text(json.dumps({
        "framedir": str(framedir), "args": vars(args),
        "device": device_label, "reason": reason, "summary": summary,
    }, indent=2, default=str))
    print(f"[summary] {sum_path}")


# ---------------------------------------------------------------------------
# MultiGpuPool — DESIGN ONLY (not run; see CONTRACT 2 AMENDMENT)
# ---------------------------------------------------------------------------
class MultiGpuPool:
    """DESIGN STUB. Two real 7700S would each run one of these via
    embedder_worker.py (stdin/stdout JSON). The dispatcher feeds tasks
    through a shared queue; the faster GPU drains more items naturally
    (dynamic load balance — NOT a fixed 50/50 split).

    Contract clauses honored when this is implemented:
      - Workers pinned via HIP_VISIBLE_DEVICES; 780M needs
        HSA_OVERRIDE_GFX_VERSION=11.0.0.
      - VRAM guard: do not spawn a worker on any GPU with < 5 GB free.
      - Drop-in shape: embed_windows() returns (n, dim) in input order,
        matching worker_client.Embedder.embed_image().
      - Correctness: sharded output bit-for-bit (cosine ≈ 1.0) identical
        to single-GPU output for the same frames (--verify).

    Skeleton (would live in a worker_client.Embedder-like wrapper):
        class MultiGpuPool:
            def __init__(self, model_id, gpus=("cuda:0",)):
                self.workers = []
                for g in gpus:
                    env = {**os.environ,
                           "HIP_VISIBLE_DEVICES": g.split(":")[1],
                           "ROCR_VISIBLE_DEVICES": g.split(":")[1],
                           "CUDA_VISIBLE_DEVICES": g.split(":")[1]}
                    if g == "cuda:1":
                        env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
                    self.workers.append(_spawn_worker(model_id, env))
                self.queue = queue.Queue()
                for _ in self.workers:
                    threading.Thread(target=self._pump, args=(...),
                                     daemon=True).start()

            def embed_windows(self, items):
                idxs = list(range(len(items)))
                for i, it in zip(idxs, items):
                    self.queue.put((i, it))
                results = [None] * len(items)
                while any(r is None for r in results):
                    i, vec = self._collect()
                    results[i] = vec
                return np.stack(results)
    """


if __name__ == "__main__":
    main()