#!/usr/bin/env python3
"""Ingest a folder into a RAG collection's index. Run on-demand (NOT at idle).

  python ingest.py <folder|file> [--index DIR] [--reset] [--dpi 150] [--chunk 1200]

Modalities (per user):
  * **PDF** -> the page is OCR'd with **Surya-2** on the 7700S and the OCR text
    is embedded (one chunk per page, split if long). The OCR text is the richest
    representation of slides/scans; the page image itself is rendered on demand
    by the server's `get_book_image` tool so a multimodal model can still see
    figures. OCR results are cached by file hash so re-ingest is free.
  * **.txt/.md** -> embedded as text (sliding-window chunks).
  * **.epub**    -> embedded as text, one chunk per chapter-section heading.
"""
import os, sys, time, argparse, glob, zipfile, posixpath, re, warnings, hashlib
import numpy as np
import fitz  # pymupdf
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from worker_client import Embedder
from index_store import IndexStore
import ocr_cache

TEXT_EXTS = {".txt", ".md", ".markdown", ".text"}
PDF_EXTS = {".pdf"}
EPUB_EXTS = {".epub"}
# Image extensions — kept in sync with server.IMAGE_EXTS; the server's set is
# the source of truth and this duplicates it only to avoid a server<->ingest
# import cycle. If server adds a new image ext, mirror it here.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Videos pushed by yt-dlpcc (`<slug>.mp4` + `<slug>.srt` + `<slug>.video.json`
# next to the transcript) are NOT embedded: the transcript text carries the
# timestamps, and the server's `screenshot_video` tool grabs captioned stills
# on demand. Ingest simply ignores those files.


def chunk_text(s, size, overlap=150):
    s = s.replace("\r\n", "\n")
    out, i = [], 0
    while i < len(s):
        out.append(s[i:i + size])
        i += size - overlap
    return [c for c in out if c.strip()]


def epub_spine_docs(zf):
    """Return list of (href) XHTML documents in reading (spine) order."""
    container = zf.read("META-INF/container.xml")
    opf_path = BeautifulSoup(container, "xml").find("rootfile")["full-path"]
    opf_dir = posixpath.dirname(opf_path)
    opf = BeautifulSoup(zf.read(opf_path), "xml")
    manifest = {it["id"]: it["href"] for it in opf.find_all("item")}
    hrefs = []
    for ref in opf.find("spine").find_all("itemref"):
        href = manifest.get(ref["idref"])
        if href:
            hrefs.append(posixpath.normpath(posixpath.join(opf_dir, href)))
    return hrefs


def epub_sections(epub_path):
    """Yield (label, text) per section. A section is a heading (h1-h4) and the
    text up to the next heading; documents with no headings yield as one block."""
    zf = zipfile.ZipFile(epub_path)
    HEAD = ("h1", "h2", "h3", "h4")
    for href in epub_spine_docs(zf):
        try:
            raw = zf.read(href)
        except KeyError:
            continue
        soup = BeautifulSoup(raw, "lxml")
        body = soup.body or soup
        for tag in body(["script", "style"]):
            tag.decompose()
        heads = body.find_all(HEAD)
        if not heads:
            text = re.sub(r"\n{3,}", "\n\n", body.get_text("\n").strip())
            if text:
                yield (posixpath.basename(href), text)
            continue
        for h in heads:
            label = h.get_text(" ", strip=True)
            parts = []
            for sib in h.next_siblings:
                if getattr(sib, "name", None) in HEAD:
                    break
                if hasattr(sib, "get_text"):
                    parts.append(sib.get_text("\n"))
                elif isinstance(sib, str):
                    parts.append(sib)
            text = re.sub(r"\n{3,}", "\n\n", "\n".join(parts).strip())
            body_text = (label + "\n" + text).strip()
            if body_text:
                yield (label or posixpath.basename(href), body_text)
    zf.close()


def ext_kind(path):
    e = os.path.splitext(path)[1].lower()
    if e in PDF_EXTS:
        return "pdf"
    if e in TEXT_EXTS:
        return "text"
    if e in EPUB_EXTS:
        return "epub"
    if e in IMAGE_EXTS:
        return "image"
    return None


def supported_files(root):
    """All ingestable files under root (or [root] if root is a single file)."""
    if os.path.isfile(root):
        return [root] if ext_kind(root) else []
    out = []
    for f in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if os.path.isfile(f) and ext_kind(f):
            out.append(f)
    return out


def file_signature(path):
    """Cheap identity (size+mtime) plus a content hash for change detection."""
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return {"size": st.st_size, "mtime": st.st_mtime, "hash": h.hexdigest()}


def file_hash(path):
    return file_signature(path)["hash"]


# PDFs longer than this skip Surya OCR entirely and use their embedded text
# layer instead (books/papers have one; OCR is for short scans/slides).
MAX_OCR_PAGES = int(os.environ.get("RAG_MAX_OCR_PAGES", "20"))


def pdf_page_count(path):
    with fitz.open(path) as doc:
        return doc.page_count


def pdf_text_pages(path):
    """Extract the embedded text layer per page, same shape as OCR output
    (list of {"page", "html"}). Pages with no text layer come out empty and
    are simply not embedded."""
    with fitz.open(path) as doc:
        return [{"page": i + 1, "html": doc[i].get_text().strip()}
                for i in range(doc.page_count)]


def pdf_needs_ocr_worker(path, index_dir):
    """True if ingesting this PDF would actually run the OCR model: not in
    the cache AND short enough to OCR (> MAX_OCR_PAGES uses the text layer)."""
    if ocr_cache.get(index_dir, file_hash(path)) is not None:
        return False
    return pdf_page_count(path) <= MAX_OCR_PAGES


def ocr_pages_for(path, index_dir, ocr_worker, dpi=150):
    """Return the per-page text for a PDF (list of {"page", "html"}), using the
    cache when possible. On a miss, PDFs with more than MAX_OCR_PAGES pages get
    their embedded text layer extracted (no GPU); shorter ones are OCR'd by the
    worker. Both results are cached by file hash. Requires a live `ocr_worker`
    only on a cache miss for a short PDF."""
    h = file_hash(path)
    pages = ocr_cache.get(index_dir, h)
    if pages is not None:
        return pages, True                       # cache hit
    if pdf_page_count(path) > MAX_OCR_PAGES:
        pages = pdf_text_pages(path)
        ocr_cache.put(index_dir, h, pages,
                      meta={"source": path, "method": "textlayer"})
        return pages, False                      # text layer, no OCR
    if ocr_worker is None:
        raise RuntimeError(f"OCR needed for {os.path.basename(path)} but no OCR "
                           f"worker available")
    pages = ocr_worker.ocr_pdf(path, dpi=dpi)
    ocr_cache.put(index_dir, h, pages,
                  meta={"source": path, "dpi": dpi, "method": "ocr"})
    return pages, False                          # freshly OCR'd


def ingest_file(path, text_emb=None, image_emb=None,
                 text_store=None, image_store=None,
                 ocr_pages=None, dpi=150, chunk=1200):
    """Embed one file and append to the appropriate store. Returns
    (n_chunks, modality). Dispatched by file kind:
      * pdf / text / epub -> `text_emb.embed_text(...)` -> `text_store`
        (PDF requires `ocr_pages` to already be supplied — list of
        {"page","html"} — exactly like before)
      * image               -> `image_emb.embed_image([path])` -> `image_store`,
        appending 1 vector with meta {"source": path, "modality": "image"}
    Caller is responsible for `remove_source` on BOTH stores before calling
    when re-ingesting. Pass only the embedder+store pair relevant to the kind:
    ingest_file raises ValueError if the other pair is missing."""
    kind = ext_kind(path)
    if kind == "pdf":
        if ocr_pages is None:
            raise RuntimeError("ingest_file: PDF requires ocr_pages")
        if text_emb is None or text_store is None:
            raise ValueError("ingest_file: pdf kind needs text_emb + text_store")
        metas = []
        for pg in ocr_pages:
            html = (pg.get("html") or "").strip()
            if not html:
                continue
            for piece in chunk_text(html, chunk):
                metas.append({"source": path, "page": pg["page"],
                              "modality": "text", "text": piece})
        n, B = 0, 16
        for i in range(0, len(metas), B):
            batch = metas[i:i+B]
            text_store.append(text_emb.embed_text([m["text"] for m in batch]), batch)
            n += len(batch)
        return n, "ocr"
    if kind in ("text", "epub"):
        if text_emb is None or text_store is None:
            raise ValueError("ingest_file: text/epub kind needs text_emb + text_store")
        if kind == "text":
            with open(path, errors="ignore") as f:
                chunks = chunk_text(f.read(), chunk)
            metas = [{"source": path, "chunk": j, "modality": "text", "text": c}
                     for j, c in enumerate(chunks)]
        else:
            metas = [{"source": path, "section": label, "modality": "text",
                      "text": piece}
                     for label, text in epub_sections(path)
                     for piece in chunk_text(text, chunk)]
        n, B = 0, 16
        for i in range(0, len(metas), B):
            batch = metas[i:i+B]
            text_store.append(text_emb.embed_text([m["text"] for m in batch]), batch)
            n += len(batch)
        return n, "text"
    if kind == "image":
        if image_emb is None or image_store is None:
            raise ValueError("ingest_file: image kind needs image_emb + image_store")
        # image_emb.embed_image returns shape (n, dim) float32 L2-normalized.
        vecs = image_emb.embed_image([path])
        meta = {"source": path, "modality": "image"}
        image_store.append(vecs, [meta])
        return 1, "image"
    return 0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--index", default=os.environ.get(
        "RAG_INDEX", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "index")))
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=1200)
    args = ap.parse_args()

    text_store = IndexStore(os.path.join(args.index, "text"))
    image_store = IndexStore(os.path.join(args.index, "image"))
    if args.reset:
        text_store.reset()
        image_store.reset()

    files = supported_files(args.folder)
    print(f"found {len(files)} ingestable file(s) under {args.folder}")
    if not files:
        print("nothing to ingest."); return
    pdfs = [f for f in files if ext_kind(f) == "pdf"]
    images = [f for f in files if ext_kind(f) == "image"]

    # ---- Phase 1: OCR prepass (GPU owned by llama-server; VL-2B embedder
    #      stopped because it shares the dGPU)
    ocr = None
    if pdfs:
        # Only launch the OCR llama-server if some PDF actually needs OCR
        # (cache miss AND <= MAX_OCR_PAGES); long PDFs use their text layer.
        if any(pdf_needs_ocr_worker(f, args.index) for f in pdfs):
            from ocr_client import OCRWorker
            ocr = OCRWorker(dpi=args.dpi)
            print(f"OCR prepass for {len(pdfs)} PDF(s) on {ocr.llama} ...")
        else:
            print(f"text prepass for {len(pdfs)} PDF(s) (no OCR needed) ...")
        t0 = time.time()
        for f in pdfs:
            _, hit = ocr_pages_for(f, args.index, ocr, dpi=args.dpi)
            print(f"  [pdf{'/cache' if hit else ''}] {os.path.basename(f)}")
        if ocr:
            ocr.stop()                           # frees the llama-server VRAM
        print(f"  PDF prepass done in {time.time()-t0:.1f}s")

    # ---- Phase 2: embed text-ish files with the small text embedder on the
    # 7700S (xlarge preset). The 780M iGPU is reserved for search-query
    # inference only — see server.py's module docstring for the policy.
    text_files = [f for f in files if ext_kind(f) in ("pdf", "text", "epub")]
    t_start = time.time()
    if text_files:
        import dgpu_embed_ctl as dctl
        dctl.wait_for_dgpu_idle()
        text_emb = dctl._make_embedder("xlarge", dctl.DEFAULT_PORT)
        print(f"cold-starting text embedder on 7700S/xlarge ({len(text_files)} file(s))...")
        t0 = time.time(); text_emb.start()
        print(f"  text embedder ready in {time.time()-t0:.1f}s on {text_emb.device}")

        n_text = 0
        for path in text_files:
            text_store.remove_source(path)        # idempotent re-ingest
            op = None
            if ext_kind(path) == "pdf":
                op, _ = ocr_pages_for(path, args.index, None, dpi=args.dpi)
            t0 = time.time()
            n, modality = ingest_file(path, text_emb=text_emb, image_emb=None,
                                      text_store=text_store, image_store=None,
                                      ocr_pages=op, dpi=args.dpi, chunk=args.chunk)
            n_text += n
            print(f"  [{modality}] {os.path.basename(path)}: {n} chunks "
                  f"({time.time()-t0:.1f}s)")
        text_emb.stop()
    else:
        n_text = 0

    # ---- Phase 3: embed raster images with the big VL-2B on 7700S
    if images:
        emb = Embedder()
        print(f"cold-starting image embedder (VL-2B) on 7700S "
              f"({len(images)} file(s))...")
        t0 = time.time(); emb.start()
        print(f"  image embedder ready in {time.time()-t0:.1f}s on {emb.device}")

        n_images = 0
        for path in images:
            image_store.remove_source(path)
            t0 = time.time()
            n, modality = ingest_file(path, text_emb=None, image_emb=emb,
                                      text_store=None, image_store=image_store)
            n_images += n
            print(f"  [{modality}] {os.path.basename(path)}: {n} vectors "
                  f"({time.time()-t0:.1f}s)")
        emb.stop()
    else:
        n_images = 0

    print("-" * 56)
    print(f"ingested {n_text} text chunks + {n_images} images across "
          f"{len(files)} file(s)")
    print(f"  text store: {text_store.count()} vectors  "
          f"({len(text_store.sources())} files)")
    print(f"  image store: {image_store.count()} vectors  "
          f"({len(image_store.sources())} files)")
    print(f"wall time (incl load): {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
