#!/usr/bin/env python3
"""test_kernel.py — CONTRACT 3 §2 correctness gate.

For any new kernel that claims to replace the stock Conv3d patch-embed,
the rule is: vectors must match the stock path to cos ≥ 0.9999.

This script:
  1. Loads the Qwen3-VL-Embedding-2B model on **CPU** (VRAM-safe;
     honors the TECH_CONTEXT VRAM guard — GPU[0] is live-busy with the
     rag-mcp daemon's worker right now, per the CONTRACT 3 AMENDMENT).
  2. Picks the FIRST `Qwen3VLVisionPatchEmbed` it can find under
     `model.visual` and runs a random-input forward both with the
     stock Conv3d and with our `patch_embed_matmul` replacement.
  3. Computes the per-row cosine similarity and reports min/mean/max.
  4. Reports PASS/FAIL with the CONTRACT 3 threshold (0.9999).

Additionally, runs a small full-model forward (still path, synthetic
640×640 image) with the matmul kernel installed via
`patch_embed_kernel.install_patch_embed_kernel` and asserts the
embedding vector still matches the un-patched model to cos ≥ 0.9999.

CLI:
    /home/gabriel/venv/bin/python test_kernel.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# CPU-only — the GPU is busy with the rag-mcp daemon's worker
# (CONTRACT 3 AMENDMENT — must NOT kill anything). Force-hide GPUs
# from torch BEFORE importing it. setdefault is not enough because the
# shell may already export HIP_VISIBLE_DEVICES=0.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch
import torch.nn as nn

import patch_embed_kernel as pek

CONTRACT_THRESHOLD = 0.9999


def _find_patch_embed(model):
    """Walk the model tree to find Qwen3VLVisionPatchEmbed.

    Accepts either a `Qwen3VLEmbedder` (whose `.model` is
    `Qwen3VLForEmbedding` whose `.model` is `Qwen3VLModel`) or a
    bare `Qwen3VLForEmbedding`.
    """
    # Walk down: Qwen3VLEmbedder → Qwen3VLForEmbedding → Qwen3VLModel
    root = model
    if hasattr(root, "model") and hasattr(root.model, "model") and \
       root.model.__class__.__name__ == "Qwen3VLForEmbedding":
        inner = root.model.model
    elif hasattr(root, "model"):
        inner = root.model
    else:
        inner = root
    visual = getattr(inner, "visual", None)
    if visual is None:
        return None
    pe = getattr(visual, "patch_embed", None)
    if pe is not None:
        return pe
    for _, mod in visual.named_modules():
        if mod.__class__.__name__ == "Qwen3VLVisionPatchEmbed":
            return mod
    return None


def unit_test_patch_embed(model) -> tuple[float, float, float, bool]:
    """Direct patch-embed unit test. Returns (min_cos, mean_cos, max_cos, ok)."""
    pe = _find_patch_embed(model)
    assert pe is not None, "No Qwen3VLVisionPatchEmbed found under model.visual"

    # Use the actual config from the model.
    T = pe.temporal_patch_size
    H = pe.patch_size
    W = pe.patch_size
    in_channels = pe.in_channels
    embed_dim = pe.embed_dim
    print(f"  patch_embed: in_channels={in_channels} T={T} H={H} W={W} "
          f"embed_dim={embed_dim} dtype={pe.proj.weight.dtype}", flush=True)

    # Random pre-patched input — the shape the vision processor hands us:
    #   (N, in_channels * T * H * W)
    N = 4096
    x = torch.randn(N, in_channels * T * H * W, dtype=pe.proj.weight.dtype)
    # Conv3d path (stock). Wrap in no_grad for fairness.
    with torch.no_grad():
        # Stock forward: x → view → proj (Conv3d) → view
        stock_view = x.view(-1, in_channels, T, H, W)
        stock_out = pe.proj(stock_view).view(-1, embed_dim)
        # Matmul path.
        matmul_out = pek.patch_embed_matmul(
            x, pe.proj.weight, pe.proj.bias,
            in_channels=in_channels, temporal_patch=T, spatial_patch=H,
        )
    # Per-row cosine similarity.
    a = stock_out.float()
    b = matmul_out.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    cos_min = float(cos.min())
    cos_mean = float(cos.mean())
    cos_max = float(cos.max())
    ok = cos_min >= CONTRACT_THRESHOLD
    print(f"  per-row cos: min={cos_min:.6f}  mean={cos_mean:.6f}  max={cos_max:.6f}",
          flush=True)
    print(f"  threshold: cos >= {CONTRACT_THRESHOLD}  →  "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return cos_min, cos_mean, cos_max, ok


def end_to_end_test(model) -> tuple[float, bool]:
    """Full model forward with the kernel installed vs uninstalled.

    Uses a synthetic 640×640 image (solid-color, so JPEG encode is fast).
    Verifies the final embedding vector matches.
    """
    from PIL import Image
    from qwen3_vl_embedding import Qwen3VLEmbedder

    img = Image.new("RGB", (640, 640), (180, 120, 60))
    tmp = Path("/tmp/_test_kernel_img.jpg")
    img.save(tmp, "JPEG", quality=85)
    prompt = "a brown square"

    # 1) Stock baseline.
    was_installed = pek.is_installed()
    if was_installed:
        pek.uninstall_patch_embed_kernel(model.model)
    with torch.no_grad():
        v_stock = model.process([{"image": str(tmp), "text": prompt}],
                               normalize=True)[0].detach().cpu().numpy()

    # 2) Matmul kernel installed.
    ok = pek.install_patch_embed_kernel(model.model)
    if not ok:
        print("  could not install patch_embed kernel — skipping end-to-end",
              flush=True)
        return 0.0, False
    with torch.no_grad():
        v_kernel = model.process([{"image": str(tmp), "text": prompt}],
                                normalize=True)[0].detach().cpu().numpy()

    cos = float(np.dot(v_stock, v_kernel) /
                (np.linalg.norm(v_stock) * np.linalg.norm(v_kernel) + 1e-12))
    ok_e2e = cos >= CONTRACT_THRESHOLD
    print(f"  end-to-end embedding cos (stock vs matmul): {cos:.6f}", flush=True)
    print(f"  threshold: cos >= {CONTRACT_THRESHOLD}  →  "
          f"{'PASS' if ok_e2e else 'FAIL'}", flush=True)

    # 3) Restore.
    pek.uninstall_patch_embed_kernel(model.model)
    return cos, ok_e2e


def main():
    from qwen3_vl_embedding import Qwen3VLEmbedder

    print("[test_kernel] loading Qwen/Qwen3-VL-Embedding-2B on CPU (float32) — "
          "VRAM-safe per CONTRACT 3 AMENDMENT …", flush=True)
    t0 = time.time()
    model = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-2B",
                            max_length=32768, torch_dtype=torch.float32)
    print(f"[test_kernel] model ready in {time.time()-t0:.1f}s", flush=True)

    print("\n[unit] patch_embed (matmul vs Conv3d):", flush=True)
    u_min, u_mean, u_max, u_ok = unit_test_patch_embed(model)

    print("\n[end-to-end] full model forward (kernel on/off):", flush=True)
    e2e_cos, e2e_ok = end_to_end_test(model)

    print("\n" + "=" * 56, flush=True)
    print(f"  unit-test  cos_min = {u_min:.6f}  →  {'PASS' if u_ok else 'FAIL'}")
    print(f"  end-to-end cos     = {e2e_cos:.6f}  →  "
          f"{'PASS' if e2e_ok else 'FAIL'}")
    overall = u_ok and e2e_ok
    print(f"  OVERALL            →  {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()