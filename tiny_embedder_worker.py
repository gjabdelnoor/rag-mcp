#!/usr/bin/env python3
"""Tiny embedder worker — sentence-transformers/all-MiniLM-L6-v2 in a subprocess.

Only file in the `tiny` preset that imports `sentence_transformers` (and only
when started). The MCP server / ingest / watcher stay torch-free at idle.

Protocol (mirrors embedder_worker.py and worker_client.py):
  request:  {"id": int, "op": "embed_text"|"ping"|"shutdown", "texts": [...],
             "is_query": bool}
  response: {"id": int, "ok": bool, "dim": int, "vecs_b64": str}

Embeddings are returned as base64 of a contiguous float32 array (n*dim),
already L2-normalized.
"""
from __future__ import annotations

import base64
import json
import os
import sys


def log(*a):
    print("[tiny]", *a, file=sys.stderr, flush=True)


MODEL_NAME = os.environ.get(
    "RAG_TINY_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


def main() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    log(f"loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    dim = int(model.get_sentence_embedding_dimension())
    log(f"model ready (dim={dim})")

    def reply(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    reply({"id": 0, "ok": True, "event": "ready",
           "dim": dim, "model": MODEL_NAME})

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
                is_query = bool(req.get("is_query", False))
                vecs = model.encode(
                    texts,
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)
                # `normalize_embeddings=True` already returns unit-norm rows,
                # but double-check (some models behave differently).
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0.0, 1.0, norms)
                vecs = (vecs / norms).astype(np.float32)
                _ = is_query    # all-MiniLM-L6-v2 has no asymmetric prefix
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
