#!/usr/bin/env python3
"""Embedder worker — the ONLY process that imports torch / loads the model.

Runs as a subprocess of the MCP server. The model (Qwen3-VL-Embedding-2B) is
loaded onto the 7700S (cuda:0) at startup and stays resident until the parent
kills this process (after 15 min idle). Killing the process is what frees VRAM
and drops RAM back to ~nothing — torch's HIP context cannot be fully released
in-process, so process death is the clean unload.

Protocol: newline-delimited JSON on stdin -> newline-delimited JSON on stdout.
  request:  {"id": int, "op": "embed_text"|"embed_image"|"ping", ...}
  response: {"id": int, "ok": bool, "dim": int, "vecs_b64": str, ...}
Embeddings are returned as base64 of a contiguous float32 array (n*dim).
"""
import sys, os, json, base64

# Keep transformers from importing TensorFlow/Flax (faster start, less RAM, no noise)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = os.environ.get("RAG_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
EMBED_DIM = int(os.environ.get("RAG_DIM", "2048"))


def log(*a):
    print("[worker]", *a, file=sys.stderr, flush=True)


def main():
    import numpy as np
    import torch
    from qwen3_vl_embedding import Qwen3VLEmbedder

    maxlen = int(os.environ.get("RAG_MAXLEN", "8192"))
    log(f"loading {MODEL_ID} onto cuda:0 ({torch.cuda.get_device_name(0)}) ...")
    embedder = Qwen3VLEmbedder(
        MODEL_ID,
        max_length=maxlen,
        torch_dtype=torch.float16,
    )
    log("model ready")

    def encode(items):
        # items: list of dicts, e.g. {"text": "..."} or {"image": "/path.png"}.
        vecs = embedder.process(items, normalize=True)   # last-token pooled, L2-norm
        return vecs.float().cpu().numpy().astype(np.float32)

    def reply(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    # signal readiness once the model is loaded
    reply({"id": 0, "ok": True, "event": "ready", "dim": EMBED_DIM,
           "device": torch.cuda.get_device_name(0)})

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
            if op == "embed_text":
                vecs = encode([{"text": t} for t in req["texts"]])
            elif op == "embed_image":
                # Optional per-image captions (e.g. the closed captions of a
                # video frame) are fused INTO the same vector so the frame
                # surfaces on both visual and spoken-content queries.
                paths = req["paths"]
                caps = req.get("captions") or [None] * len(paths)
                items = []
                for p, c in zip(paths, caps):
                    it = {"image": p}
                    if c:
                        it["text"] = c
                    items.append(it)
                vecs = encode(items)
            else:
                reply({"id": rid, "ok": False, "error": f"bad op {op}"})
                continue
            b64 = base64.b64encode(vecs.tobytes()).decode("ascii")
            reply({"id": rid, "ok": True, "n": int(vecs.shape[0]),
                   "dim": int(vecs.shape[1]), "vecs_b64": b64})
        except Exception as e:
            import traceback
            log("ERROR", traceback.format_exc())
            reply({"id": req.get("id", 0), "ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
