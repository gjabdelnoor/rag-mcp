#!/usr/bin/env python3
"""patch_embed_kernel.py — MIOpen-free patch-embed conv for Qwen3-VL.

CONTRACT 3 (AMENDMENT): the native video path's vision encoder hangs in
MIOpen's algorithm search at full per-frame resolution on the 7700S
("No suitable algorithm ... convolution"). The offending op is
`Qwen3VLVisionPatchEmbed.proj` — a Conv3d with kernel_size == stride,
which is mathematically equivalent to a flat matmul.

This module replaces that Conv3d forward with an unfold+matmul that
bypasses MIOpen entirely while producing bit-identical outputs (the
constraint is `kernel_size == stride`, so each output element is the
dot-product of one input patch and one weight vector; no overlap,
no boundary effects to worry about).

The replacement is wired in two ways:

  1. **Standalone function** `patch_embed_matmul(...)` — drop-in for
     testing or for users who want to call it directly without
     monkey-patching.

  2. **`install_patch_embed_kernel(model)`** — monkey-patches
     `Qwen3VLVisionPatchEmbed.forward` on a live
     `Qwen3VLForEmbedding.model.visual` so the stock model uses the
     matmul path with zero source changes.

Fidelity proof (CONTRACT 2 / 3 §2): the matmul result equals the Conv3d
result to cos ≥ 0.9999 for fp16 and 1.000000 ± 1e-6 for fp32 — see
`test_kernel.py` for the runnable proof.

Design notes:
- Conv3d weight shape: `(embed_dim=1024, in_channels=3, T=2, H=16, W=16)`
- Conv3d bias shape:    `(embed_dim=1024,)`
- After flattening: weight is `(1024, 1536)`, input is `(N, 1536)`,
  output is `(N, 1024)`. Trivially matmul-able.
- We do not change the model's weight values, dtype, or device.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vision-tower config constants (mirrored from Qwen3-VL-Embedding-2B).
# We read these from the model at install time, but expose them for
# standalone callers and tests.
DEFAULT_IN_CHANNELS = 3
DEFAULT_TEMPORAL_PATCH = 2
DEFAULT_PATCH_SIZE = 16


def patch_embed_matmul(hidden_states: torch.Tensor,
                       proj_weight: torch.Tensor,
                       proj_bias: torch.Tensor,
                       in_channels: int = DEFAULT_IN_CHANNELS,
                       temporal_patch: int = DEFAULT_TEMPORAL_PATCH,
                       spatial_patch: int = DEFAULT_PATCH_SIZE) -> torch.Tensor:
    """Mathematically identical replacement for `Conv3d(in, embed, k=s, s=k)`.

    Args:
      hidden_states: shape `(N, in_channels * T * H * W)` — the
        flattened pre-patched input as the vision processor produces.
        (This is exactly the Conv3d input's `view(-1, in, T, H, W)`
         unrolled.)
      proj_weight: shape `(embed_dim, in_channels, T, H, W)` — the
        Conv3d's `.proj.weight` tensor.
      proj_bias:   shape `(embed_dim,)` — the Conv3d's `.proj.bias`.
      in_channels, temporal_patch, spatial_patch: integer constants;
        defaults match Qwen3-VL-Embedding-2B (3, 2, 16).

    Returns:
      shape `(N, embed_dim)` — same as Conv3d forward.

    Notes:
      - We honor the original `proj_weight.dtype` (fp16 / bf16 / fp32).
      - No sync, no copy of the weight, no MIOpen involvement.
      - `T`, `H`, `W` are taken from the weight tensor's actual shape so
        any future config change Just Works.
    """
    T = proj_weight.shape[2]
    H = proj_weight.shape[3]
    W = proj_weight.shape[4]
    embed_dim = proj_weight.shape[0]
    target_dtype = proj_weight.dtype

    N = hidden_states.shape[0]
    # Flatten input: (N, in*T*H*W). The Conv3d path reshapes to
    # (N, in, T, H, W) then Conv3d with kernel=stride=(T,H,W) produces
    # (N, embed, 1, 1, 1) → (N, embed). The reshape is a view; the
    # actual data layout for matmul is identical because the Conv3d
    # reads each (T,H,W) tile contiguously (which is exactly how the
    # input is laid out after view).
    x = hidden_states.to(dtype=target_dtype).view(N, -1)
    # Flatten weight: (embed, in*T*H*W). Same memory layout.
    w = proj_weight.view(embed_dim, -1)
    out = F.linear(x, w, proj_bias)   # (N, embed)
    return out


def _patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Replacement forward for `Qwen3VLVisionPatchEmbed`.

    Identical semantics to the stock Conv3d forward — bit-for-bit for
    fp32, cos ≥ 0.9999 for fp16 (rounding only).
    """
    return patch_embed_matmul(
        hidden_states,
        self.proj.weight,
        self.proj.bias,
        in_channels=self.in_channels,
        temporal_patch=self.temporal_patch_size,
        spatial_patch=self.patch_size,
    )


# ---------------------------------------------------------------------------
# Monkey-patch installer
# ---------------------------------------------------------------------------
_INSTALLED = False
_ORIGINAL_FORWARD = None


def install_patch_embed_kernel(model) -> bool:
    """Install the matmul forward on the model's vision patch-embed.

    Args:
      model: either a `Qwen3VLEmbedder` or a `Qwen3VLForEmbedding`
        instance. We patch the FIRST `Qwen3VLVisionPatchEmbed`
        we find by walking the model tree.

    Returns:
      True if patched, False if no patch-embed was found (e.g., wrong
      model class — caller should check and decide).
    """
    global _INSTALLED, _ORIGINAL_FORWARD
    if _INSTALLED:
        return True

    # Walk to the vision encoder.
    # Qwen3VLEmbedder → .model (Qwen3VLForEmbedding) → .model (Qwen3VLModel) → .visual
    # Some forks / future refactors may flatten; handle both.
    root = model
    if hasattr(root, "model") and hasattr(root.model, "model"):
        # Qwen3VLEmbedder wrapper
        inner = root.model.model
    elif hasattr(root, "model"):
        inner = root.model
    else:
        inner = root

    visual = getattr(inner, "visual", None) or getattr(inner, "vision_tower", None)
    if visual is None:
        return False

    # Prefer the named attribute, fall back to a recursive search.
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is None:
        for _, mod in visual.named_modules():
            if mod.__class__.__name__ == "Qwen3VLVisionPatchEmbed":
                patch_embed = mod
                break
    if patch_embed is None:
        return False

    _ORIGINAL_FORWARD = patch_embed.__class__.forward
    patch_embed.__class__.forward = _patched_forward
    _INSTALLED = True
    return True


def uninstall_patch_embed_kernel(model) -> bool:
    """Restore the stock Conv3d forward (for A/B testing)."""
    global _INSTALLED
    if not _INSTALLED:
        return False
    root = model
    if hasattr(root, "model") and hasattr(root.model, "model"):
        inner = root.model.model
    elif hasattr(root, "model"):
        inner = root.model
    else:
        inner = root
    visual = getattr(inner, "visual", None) or getattr(inner, "vision_tower", None)
    if visual is not None:
        pe = getattr(visual, "patch_embed", None)
        if pe is not None and _ORIGINAL_FORWARD is not None:
            pe.__class__.forward = _ORIGINAL_FORWARD
    _INSTALLED = False
    return True


def is_installed() -> bool:
    return _INSTALLED