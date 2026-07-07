#!/usr/bin/env python3
"""_miopen_one.py — single embed call under a given MIOPEN_FIND_MODE.

Invoked by `miopen_harness.py` once per find-mode. Loads the embedder,
embeds a still 720×720 frame and a small video window, prints elapsed
times, and exits. The harness reads stdout and uses the exit code to
classify PASS / FAIL / HANG.

Never kills anything; relies on the harness's outer timeout.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# Parse args first so we can set the device BEFORE torch import.
ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True)
ap.add_argument("--window", type=int, default=16)
ap.add_argument("--limit", type=int, default=32)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

idx = args.device.split(":", 1)[1] if ":" in args.device else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = idx
os.environ["HIP_VISIBLE_DEVICES"] = idx
os.environ["ROCR_VISIBLE_DEVICES"] = idx

print(f"[one] MIOPEN_FIND_MODE={os.environ.get('MIOPEN_FIND_MODE', 'unset')}  "
      f"device={args.device}  W={args.window}  limit={args.limit}", flush=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

print("[one] loading model …", flush=True)
t0 = time.time()
from qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402
emb = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-2B",
                      max_length=32768, torch_dtype=torch.float16)
print(f"[one] model ready in {time.time()-t0:.1f}s", flush=True)

# Still-path test: 720×720 solid color (the size that hung in CONTRACT 2).
img = Image.new("RGB", (720, 720), (180, 120, 60))
img.save("/tmp/_miopen_one_720.jpg", "JPEG", quality=85)
print("[one] embedding 720×720 still …", flush=True)
t0 = time.time()
v_still = emb.process([{"image": "/tmp/_miopen_one_720.jpg", "text": "brown"}],
                      normalize=True)
still_dt = time.time() - t0
print(f"[one] still 720×720 done in {still_dt:.2f}s  shape={tuple(v_still.shape)}",
      flush=True)

# Video-path test: read frames.json for --limit frames.
framedir = Path(args.frames)
manifest = json.loads((framedir / "frames.json").read_text())
fs = manifest["frames"][: args.limit]
paths = [str(framedir / fr["file"]) for fr in fs]
cc = " ".join(fr.get("cc", "") for fr in fs)
W = min(args.window, len(paths))
print(f"[one] embedding {W}-frame video window ({len(paths)} frames avail) …",
      flush=True)
t0 = time.time()
v_vid = emb.process([{"video": paths[:W], "text": cc}], normalize=True)
vid_dt = time.time() - t0
print(f"[one] video W={W} done in {vid_dt:.2f}s  shape={tuple(v_vid.shape)}",
      flush=True)

# Print a one-line JSON the harness can grep (best-effort).
print(f"[one] RESULT still_dt={still_dt:.3f} vid_dt={vid_dt:.3f}", flush=True)
print("MIOPEN_ONE_OK", flush=True)
sys.exit(0)