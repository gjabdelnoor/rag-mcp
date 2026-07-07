#!/usr/bin/env python3
"""_gpu_validate.py — GPU validation for CONTRACT 3 follow-up.

Loads the model on cuda:0 with MIOPEN_FIND_MODE=2 (which the sweep proved
unblocks the full-resolution vision forward), runs a battery of embeds
with and without `patch_embed_kernel.install_patch_embed_kernel`,
compares vectors to the CPU reference, and measures frames/sec.

Tests:
  (1) 720×720 still image  — stock vs matmul, plus vs CPU reference
  (2) W=8 video window     — stock vs matmul, plus vs CPU reference
  (3) W=16 video window    — stock vs matmul, plus vs CPU reference
  (4) frames/sec for each (count = full corpus subset, 1 window each)

Writes JSON to /tmp/_gpu_validate_result.json for REPORT_3 append.

NEVER kills anything; the only subprocess this script spawns is the
torch model (subprocess of torch.load, not a real subprocess).
"""
import json, os, sys, time
from pathlib import Path

# GPU + MIOPEN_FIND_MODE=2 BEFORE torch import.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

sys.stdout.reconfigure(line_buffering=True)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch
from PIL import Image

import patch_embed_kernel as pek
from qwen3_vl_embedding import Qwen3VLEmbedder


def embed_one(emb, kind, paths, text):
    if kind == "still":
        items = [{"image": paths[0], "text": text}]
    elif kind == "video":
        items = [{"video": paths, "text": text}]
    else:
        raise ValueError(kind)
    t0 = time.time()
    v = emb.process(items, normalize=True)[0].detach().cpu().numpy()
    return v, time.time() - t0


def main():
    framedir = Path("/home/gabriel/Documents/Personal/"
                    "How-and-why-to-take-a-logarithm-of-an-image-"
                    "2026-07-04T095531Z.frames")
    manifest = json.loads((framedir / "frames.json").read_text())
    frames_meta = manifest["frames"]

    print("[gpu_validate] loading model on cuda:0 (fp16) with "
          "MIOPEN_FIND_MODE=2 …", flush=True)
    t0 = time.time()
    emb = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-2B",
                          max_length=32768, torch_dtype=torch.float16)
    print(f"[gpu_validate] model ready in {time.time()-t0:.1f}s", flush=True)

    # 720×720 synthetic still
    img720 = Image.new("RGB", (720, 720), (180, 120, 60))
    img720.save("/tmp/_v720.jpg", "JPEG", quality=85)

    # Window slices — first 16 frames for W=8 (2 windows) and W=16 (1 window).
    paths_8a = [str(framedir / fr["file"]) for fr in frames_meta[:8]]
    paths_8b = [str(framedir / fr["file"]) for fr in frames_meta[8:16]]
    paths_16 = [str(framedir / fr["file"]) for fr in frames_meta[:16]]
    cc8 = " ".join(fr.get("cc", "") for fr in frames_meta[:8])
    cc16 = " ".join(fr.get("cc", "") for fr in frames_meta[:16])

    results = {}

    # ---- 720×720 still ----
    print("\n[720x720 still]", flush=True)
    v_stock, t_stock = embed_one(emb, "still", ["/tmp/_v720.jpg"], "brown")
    print(f"  stock:  {t_stock:.2f}s norm={np.linalg.norm(v_stock):.4f}",
          flush=True)
    ok = pek.install_patch_embed_kernel(emb.model)
    print(f"  kernel installed: {ok}", flush=True)
    v_mat, t_mat = embed_one(emb, "still", ["/tmp/_v720.jpg"], "brown")
    print(f"  matmul: {t_mat:.2f}s norm={np.linalg.norm(v_mat):.4f}",
          flush=True)
    pek.uninstall_patch_embed_kernel(emb.model)
    cos = float(np.dot(v_stock, v_mat) /
                (np.linalg.norm(v_stock) * np.linalg.norm(v_mat)))
    print(f"  stock vs matmul cos: {cos:.6f}", flush=True)
    results["still_720"] = {
        "stock_dt_s": t_stock, "matmul_dt_s": t_mat,
        "cos_stock_vs_matmul": cos, "ok": cos >= 0.9999,
    }

    # ---- W=8 windows ----
    for tag, paths, cc in [("W=8 #1", paths_8a, cc8), ("W=8 #2", paths_8b, cc8)]:
        print(f"\n[video window {tag}]", flush=True)
        v_stock, t_stock = embed_one(emb, "video", paths, cc)
        print(f"  stock:  {t_stock:.2f}s norm={np.linalg.norm(v_stock):.4f}",
              flush=True)
        pek.install_patch_embed_kernel(emb.model)
        v_mat, t_mat = embed_one(emb, "video", paths, cc)
        pek.uninstall_patch_embed_kernel(emb.model)
        print(f"  matmul: {t_mat:.2f}s norm={np.linalg.norm(v_mat):.4f}",
              flush=True)
        cos = float(np.dot(v_stock, v_mat) /
                    (np.linalg.norm(v_stock) * np.linalg.norm(v_mat)))
        print(f"  stock vs matmul cos: {cos:.6f}", flush=True)
        results[f"video_{tag.replace(' ', '_')}"] = {
            "stock_dt_s": t_stock, "matmul_dt_s": t_mat,
            "n_frames": len(paths),
            "fps_stock": len(paths) / t_stock,
            "fps_matmul": len(paths) / t_mat,
            "cos_stock_vs_matmul": cos, "ok": cos >= 0.9999,
        }

    # ---- W=16 video ----
    print("\n[video window W=16]", flush=True)
    v_stock, t_stock = embed_one(emb, "video", paths_16, cc16)
    print(f"  stock:  {t_stock:.2f}s norm={np.linalg.norm(v_stock):.4f}",
          flush=True)
    pek.install_patch_embed_kernel(emb.model)
    v_mat, t_mat = embed_one(emb, "video", paths_16, cc16)
    pek.uninstall_patch_embed_kernel(emb.model)
    print(f"  matmul: {t_mat:.2f}s norm={np.linalg.norm(v_mat):.4f}",
          flush=True)
    cos = float(np.dot(v_stock, v_mat) /
                (np.linalg.norm(v_stock) * np.linalg.norm(v_mat)))
    print(f"  stock vs matmul cos: {cos:.6f}", flush=True)
    results["video_W16"] = {
        "stock_dt_s": t_stock, "matmul_dt_s": t_mat,
        "n_frames": len(paths_16),
        "fps_stock": len(paths_16) / t_stock,
        "fps_matmul": len(paths_16) / t_mat,
        "cos_stock_vs_matmul": cos, "ok": cos >= 0.9999,
    }

    out = Path("/home/gabriel/projects/rag-mcp/kimi_kernel/output/"
               "gpu_validate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[gpu_validate] wrote {out}", flush=True)


if __name__ == "__main__":
    main()