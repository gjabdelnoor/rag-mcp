#!/usr/bin/env python3
"""OCR worker — runs Surya-2 VLM OCR over PDF pages on the 7700S.

Surya 0.20's recognition runs through an inference *backend*; we use the
``llamacpp`` backend, which spawns a `llama-server` (Vulkan build) that does the
heavy compute on the dGPU, and talks to it over a local OpenAI-style API. This
worker is a subprocess of the MCP server so the server itself stays torch-free
at idle; the worker is spawned only for an ingest OCR-prepass and killed right
after (which makes Surya tear down its llama-server and free the ~3 GB of VRAM).

GPU vs CPU: with ``LLAMA_CPP_NGL`` > 0 the model is offloaded to the GPU
(Vulkan). The parent may set ``LLAMA_CPP_NGL=0`` to fall back to pure-CPU
inference if the GPU spawn fails (e.g. OOM).

We store Surya's **raw HTML blocks** (math/headings/tables preserved) verbatim
into the cache. The search-time embedder (Qwen3-VL-Embedding-2B) reads HTML fine
and Gemma-4 (the target consumer) renders the math + structure directly. The
HTML is what `get_book_image` searches against for page-locate and what
`search` snippets show.

Protocol: newline-delimited JSON on stdin -> newline-delimited JSON on stdout.
  request:  {"id": int, "op": "ocr"|"ping"|"shutdown", "path": str, "dpi": int}
  response: {"id": int, "ok": bool, "pages": [{"page": int, "html": str}], ...}
On "shutdown" (or stdin EOF) the worker returns from main() so the interpreter
exits cleanly and Surya's atexit hook kills the llama-server.
"""
import sys, os, json

# Default to the llamacpp backend on the Vulkan llama-server unless overridden.
os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")


def log(*a):
    print("[ocr]", *a, file=sys.stderr, flush=True)


def main():
    import fitz  # pymupdf
    from surya.recognition import RecognitionPredictor

    backend = os.environ.get("SURYA_INFERENCE_BACKEND")
    ngl = os.environ.get("LLAMA_CPP_NGL", "99")
    log(f"starting Surya recognition (backend={backend}, ngl={ngl}) ...")
    rec = RecognitionPredictor()   # lazy: spawns llama-server on first call
    log("predictor constructed (llama-server spawns on first page)")

    def reply(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    def ocr_pdf(path, dpi):
        doc = fitz.open(path)
        imgs = []
        from PIL import Image
        for pno in range(doc.page_count):
            pix = doc.load_page(pno).get_pixmap(dpi=dpi)
            imgs.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()
        if not imgs:
            return []
        results = rec(imgs, full_page=True)     # one VLM call per page
        pages = []
        for i, r in enumerate(results):
            # store the raw HTML per block; concatenate so the embedder sees one
            # chunk and Gemma renders the math/structure directly
            blocks = getattr(r, "blocks", []) or []
            html = "\n".join((getattr(b, "html", "") or "").strip()
                             for b in blocks if (getattr(b, "html", "") or "").strip())
            pages.append({"page": i + 1, "html": html})
        return pages

    # signal readiness (model/llama-server is lazy, so this is immediate)
    reply({"id": 0, "ok": True, "event": "ready", "backend": backend, "ngl": ngl})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
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
            if op == "ocr":
                pages = ocr_pdf(req["path"], int(req.get("dpi", 150)))
                reply({"id": rid, "ok": True, "pages": pages,
                       "n_pages": len(pages)})
                continue
            reply({"id": rid, "ok": False, "error": f"bad op {op}"})
        except Exception as e:
            import traceback
            log("ERROR", traceback.format_exc())
            reply({"id": req.get("id", 0) if 'req' in dir() else 0,
                   "ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
