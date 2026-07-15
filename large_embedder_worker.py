#!/usr/bin/env python3
"""Large embedder worker — Qwen3-VL-Embedding-8B in a subprocess.

Only file in the `large` preset that imports torch + transformers. Tries
8-bit quantization via bitsandbytes first (so a 12 GB VRAM card can host
the 8B model); falls back to FP16 with a clear error if bitsandbytes is
missing.

Set RAG_LARGE_QUANT to one of: "8bit" (default), "4bit", "fp16".
Set RAG_LARGE_MODEL to override the HF id (default: Qwen/Qwen3-VL-Embedding-8B).

Protocol (mirrors the other embedder workers):
  request:  {"id": int, "op": "embed_text"|"ping"|"shutdown", "texts": [...],
             "is_query": bool}
  response: {"id": int, "ok": bool, "dim": int, "vecs_b64": str}
"""
from __future__ import annotations

import base64
import json
import os
import sys


def log(*a):
    print("[large]", *a, file=sys.stderr, flush=True)


MODEL_NAME = os.environ.get(
    "RAG_LARGE_MODEL",
    "Qwen/Qwen3-VL-Embedding-8B",
)
QUANT = os.environ.get("RAG_LARGE_QUANT", "8bit").lower()
MAX_LENGTH = int(os.environ.get("RAG_LARGE_MAXLEN", "8192"))


def _load_model():
    import torch
    from transformers import BitsAndBytesConfig
    from qwen3_vl_embedding import Qwen3VLEmbedder

    log(f"loading {MODEL_NAME} (quant={QUANT}, max_length={MAX_LENGTH}) ...")

    kwargs = dict(torch_dtype=torch.float16)

    if QUANT == "8bit":
        try:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        except Exception as e:                       # noqa: BLE001
            raise RuntimeError(
                f"8-bit quantization requested but bitsandbytes is unavailable "
                f"({e!r}). Install with `pip install bitsandbytes`, or set "
                f"RAG_LARGE_QUANT=fp16 to skip quantization."
            )
    elif QUANT == "4bit":
        try:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = "auto"
        except Exception as e:                       # noqa: BLE001
            raise RuntimeError(
                f"4-bit quantization requested but bitsandbytes is unavailable "
                f"({e!r}). Install with `pip install bitsandbytes`."
            )
    elif QUANT == "fp16":
        # plain fp16 on cuda:0 — needs >= 16 GB VRAM
        kwargs["device_map"] = "auto"
    else:
        raise RuntimeError(f"unknown RAG_LARGE_QUANT={QUANT!r}; "
                           "expected 8bit, 4bit, or fp16")

    embedder = Qwen3VLEmbedder(
        MODEL_NAME,
        max_length=MAX_LENGTH,
        **kwargs,
    )
    log("model ready")
    return embedder


def main() -> None:
    import numpy as np
    embedder = _load_model()
    dim = int(embedder.model.config.hidden_size)

    def reply(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    reply({"id": 0, "ok": True, "event": "ready",
           "dim": dim, "model": MODEL_NAME, "quant": QUANT})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = {}
        try:
            req = json.loads(line)
            op = req.get("op")
            rid = req.get("id", 0)
            if op == "ping":
                reply({"id": rid, "ok": True})
                continue
            if op == "shutdown":
                reply({"id": rid, "ok": True})
                break
            if op == "embed_text":
                texts = req["texts"]
                vecs = embedder.process(
                    [{"text": t} for t in texts], normalize=True
                ).float().cpu().numpy().astype(np.float32)
                b64 = base64.b64encode(vecs.tobytes()).decode("ascii")
                reply({"id": rid, "ok": True,
                       "n": int(vecs.shape[0]),
                       "dim": int(vecs.shape[1]),
                       "vecs_b64": b64})
                continue
            reply({"id": rid, "ok": False, "error": f"bad op {op}"})
        except Exception as e:                       # noqa: BLE001
            import traceback
            log("ERROR", traceback.format_exc())
            reply({"id": req.get("id", 0), "ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
