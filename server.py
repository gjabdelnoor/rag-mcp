#!/usr/bin/env python3
"""RAG MCP server — multimodal knowledge base over named collections.

A "collection" (a.k.a. notebook) is a watched folder + its own vector index + a
description. Lecture transcripts (.txt) embed as text, textbook chapters (.epub)
as per-section text, and PDFs are OCR'd with Surya-2 on the 7700S and the OCR
text is embedded — while the page image stays available via `get_book_image` so
a multimodal model can see figures the text only references.

Idle profile: this process is a thin server (mcp + numpy only, NO torch). The
embedding model lives in a child subprocess spawned cold on first use and KILLED
after 15 min idle. OCR runs in a *separate* short-lived subprocess only during
ingest; the embedder and the OCR llama-server never hold the GPU at the same
time (OCR is a prepass, with the embedder stopped, then embedding runs).

Videos (yt-dlpcc `--kb`) are NOT frame-embedded: the transcript text (with its
[H:MM:SS] markers) is what's indexed, the CC'd video sits on disk next to it,
and `screenshot_video` grabs captioned stills at requested timestamps on
demand (up to 10 per call, pure ffmpeg — no GPU).

Tools:
  list_collections()                       -> the notebooks you can search
  search(query, collection, k)             -> top-k passages from a collection
  get_book_image(collection, source, ...)  -> render the page of a passage
  screenshot_video(collection, video, timestamps) -> stills from a saved video
  ingest_path(collection, path)            -> incremental (re)ingest
  status()                                 -> warm/cold, per-collection counts
"""
import os, json, time, threading
from mcp.server.fastmcp import FastMCP, Image
from worker_client import Embedder
from index_store import IndexStore
import rag_collections

IDLE_SECONDS = int(os.environ.get("RAG_IDLE", "900"))   # 15 min
INGEST_VERSION = 3                                       # 3 = OCR'd PDFs (raw HTML)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}          # indexed still images

COLLECTIONS = rag_collections.load()
stores = {name: IndexStore(c["index_dir"]) for name, c in COLLECTIONS.items()}

emb = Embedder()
_last_used = [0.0]
_lock = threading.Lock()          # guards embedder start/stop + OCR prepass
_ingest_lock = threading.Lock()   # serializes ingest_path calls
_render_lock = threading.Lock()   # fitz docs are not thread-safe
_doc_cache = {}                   # source path -> laid-out fitz document

mcp = FastMCP("rag")


def _touch():
    _last_used[0] = time.time()


def _ensure_warm():
    with _lock:
        _touch()
        cold = not emb.alive
        if cold:
            emb.start()
    return cold


def _collection(name):
    """Resolve a collection name to (name, store, config); default when there is
    exactly one collection or none is given."""
    if not name:
        if len(stores) == 1:
            name = next(iter(stores))
        else:
            raise ValueError("specify `collection` — one of: "
                             + ", ".join(stores))
    if name not in stores:
        raise ValueError(f"unknown collection {name!r}; one of: "
                         + ", ".join(stores))
    return name, stores[name], COLLECTIONS[name]


def _fmt_ts(t):
    """Seconds -> m:ss (or h:mm:ss) for citing a video-frame timestamp."""
    try:
        t = int(float(t))
    except (TypeError, ValueError):
        return "?"
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _within_collections(path):
    """True if `path` lives inside a collection folder or its index dir. Guards
    get_book_image from reading arbitrary files off disk."""
    ap = os.path.abspath(path)
    for c in COLLECTIONS.values():
        for base in (c["folder"], c["index_dir"]):
            b = os.path.abspath(base)
            if ap == b or ap.startswith(b + os.sep):
                return True
    return False


def _manifest_path(index_dir):
    return os.path.join(index_dir, "manifest.json")


def _load_manifest(index_dir):
    try:
        return json.load(open(_manifest_path(index_dir)))
    except Exception:
        return {}


def _save_manifest(index_dir, man):
    os.makedirs(index_dir, exist_ok=True)
    tmp = _manifest_path(index_dir) + ".tmp"
    json.dump(man, open(tmp, "w"), indent=0)
    os.replace(tmp, _manifest_path(index_dir))


def _idle_watchdog():
    """Kill the embedder subprocess after IDLE_SECONDS with no requests."""
    while True:
        time.sleep(30)
        with _lock:
            if emb.alive and (time.time() - _last_used[0]) > IDLE_SECONDS:
                emb.stop()   # frees VRAM + RAM -> back to cold


threading.Thread(target=_idle_watchdog, daemon=True).start()


@mcp.tool()
def list_collections() -> str:
    """List the available knowledge-base collections (notebooks) you can search,
    with a description of what each one contains and how much is indexed. Use the
    returned collection name as the `collection` argument to `search`."""
    lines = ["Available collections (use the name as `collection`):"]
    for name, c in COLLECTIONS.items():
        st = stores[name]
        srcs = st.sources()
        lines.append(f"\n• {name} — {st.count()} chunks across {len(srcs)} file(s)"
                     f"\n    {c['description']}")
    return "\n".join(lines)


@mcp.tool()
def search(query: str, collection: str = "", k: int = 5) -> str:
    """Search a knowledge-base collection for the query and return the top-k most
    relevant passages. Pass `collection` (see `list_collections`); it may be
    omitted only if there is a single collection. Cold-starts the embedding model
    on the 7700S if it is not already warm (~15s the first time)."""
    name, store, _ = _collection(collection)
    if store.count() == 0:
        return f"Collection {name!r} is empty. Add files to its folder or call ingest_path."
    with _lock:
        _touch()
        cold = not emb.alive
        qvec = emb.embed_text([query], is_query=True)[0]
    hits = store.search(qvec, k=k)
    _touch()
    lines = [f"Top {len(hits)} results in {name!r} for: {query!r}"
             + ("  (cold start)" if cold else "")]
    videos = {}          # mp4 basename -> title, for the footer hint
    for i, h in enumerate(hits, 1):
        snippet = " ".join((h.get("text") or "").split())[:400]
        src_path = h.get("source", "?")
        src = os.path.basename(src_path)
        loc = (f"p.{h['page']}" if h.get("page")
               else f"§{h.get('chunk', h.get('section', '?'))}")
        vm = _video_manifest_for(src_path)
        note = ""
        if vm:
            videos[vm["video"]] = vm.get("title") or src
            note = f"  [VIDEO on disk: {vm['video']}]"
        lines.append(f"\n[{i}] {h['score']:.3f}  {src} {loc} "
                     f"({h.get('modality')}){note}\n{snippet}")
    if videos:
        vlist = ", ".join(repr(v) for v in videos)
        lines.append(
            "\nSome passages are video transcripts and the full video (with "
            "closed captions) is saved on disk: " + vlist + ". To SEE any "
            f"moment, call screenshot_video(collection={name!r}, "
            "video=<mp4 shown above>, timestamps=[…]) with up to 10 "
            "timestamps (seconds or 'H:MM:SS') picked from the [H:MM:SS] "
            "markers inside the passage text.")
    lines.append("\nTo SEE a figure/chart/table/equation a book/PDF passage "
                 f"refers to, call get_book_image(collection={name!r}, "
                 "source=<file>, page=<n> or section=<§…>, text=<snippet>).")
    return "\n".join(lines)


@mcp.tool()
def ingest_path(collection: str = "", path: str = "", recursive: bool = True) -> str:
    """Incrementally ingest a file or folder into a collection. Only NEW or
    content-CHANGED files (by hash) are (re-)embedded; deleted files are dropped.
    PDFs are OCR'd on the 7700S first (cached by hash), then everything is
    embedded. Safe to call repeatedly. With no `path`, reconciles the
    collection's watched folder. This is what the file-watcher daemon calls."""
    import ingest as ing            # lazy: pulls in fitz/bs4 only on ingest
    name, store, cfg = _collection(collection)
    index_dir = cfg["index_dir"]
    target = path or cfg["folder"]
    if not os.path.exists(target):
        return f"path does not exist: {target}"

    with _ingest_lock:
        files = ing.supported_files(target) if recursive or os.path.isfile(target) \
            else [f for f in ing.supported_files(target)
                  if os.path.dirname(f) == os.path.abspath(target)]
        man = _load_manifest(index_dir)
        on_disk = set(files)

        # decide what needs (re)ingest: new, changed hash, or stale ingest version
        todo = []
        skipped = 0
        for f in files:
            sig = ing.file_signature(f)
            prev = man.get(f)
            indexed = f in store.sources()
            fresh = (prev and prev.get("hash") == sig["hash"]
                     and prev.get("iv") == INGEST_VERSION and indexed)
            if fresh:
                skipped += 1
            else:
                todo.append((f, sig, bool(prev)))

        # ---- Phase 1: OCR prepass for any PDFs that need it (embedder stopped,
        #      OCR llama-server owns the GPU). Hold _lock so no search can
        #      cold-start the embedder into the same VRAM.
        pdfs = [(f, sig) for f, sig, _ in todo if ing.ext_kind(f) == "pdf"]
        need_ocr = [(f, sig) for f, sig in pdfs
                    if __import__("ocr_cache").get(index_dir, sig["hash"]) is None]
        if need_ocr:
            from ocr_client import OCRWorker
            with _lock:
                if emb.alive:
                    emb.stop()
                ocr = OCRWorker()
                ocr.start()
                try:
                    for f, sig in need_ocr:
                        ing.ocr_pages_for(f, index_dir, ocr)
                finally:
                    ocr.stop()           # frees the llama-server VRAM

        # ---- Phase 2: embed (embedder owns the GPU)
        added, updated, removed = [], [], []
        warmed = False
        for f, sig, had_prev in todo:
            if not warmed:
                _ensure_warm(); warmed = True
            _touch()
            if f in store.sources():
                store.remove_source(f)
            op = None
            if ing.ext_kind(f) == "pdf":
                op, _ = ing.ocr_pages_for(f, index_dir, None)   # cache hit now
            n, modality = ing.ingest_file(f, emb, store, ocr_pages=op)
            _touch()
            man[f] = {**sig, "n": n, "modality": modality, "iv": INGEST_VERSION}
            (updated if had_prev else added).append((os.path.basename(f), n))

        # legacy frame units (pre-video-tool design): frames are no longer
        # embedded — screenshots are grabbed from the saved video on demand.
        # Drop any leftover image vectors + manifest entries.
        for key in list(man.keys()):
            if man[key].get("kind") == "frames":
                store.remove_source(key)
                man.pop(key, None)
                removed.append(os.path.basename(key) + " (legacy frames)")

        # files that vanished from a watched folder (prune within target only)
        tgt_abs = os.path.abspath(target)
        for f in list(man.keys()):
            gone = (f not in on_disk) and (
                os.path.isfile(target) and os.path.abspath(f) == tgt_abs
                or (not os.path.isfile(target)
                    and os.path.abspath(f).startswith(tgt_abs + os.sep)
                    and not os.path.exists(f)))
            if gone:
                store.remove_source(f)
                man.pop(f, None)
                removed.append(os.path.basename(f))

        _save_manifest(index_dir, man)

    def fmt(items):
        return ", ".join(f"{b}({n})" for b, n in items) if items else "none"
    return (f"ingest reconcile of {target} -> collection {name!r}:\n"
            f"  added:   {fmt(added)}\n"
            f"  updated: {fmt(updated)}\n"
            f"  removed: {', '.join(removed) if removed else 'none'}\n"
            f"  unchanged: {skipped}\n"
            f"  total vectors now: {store.count()}")


def _resolve_source(store, source):
    """Map a basename/partial path (as shown in search results) to the exact
    indexed source path within a collection."""
    srcs = list(store.sources().keys())
    if source:
        for s in srcs:
            if s == source or os.path.basename(s) == source or s.endswith(source):
                return s
        return None
    return srcs[0] if len(srcs) == 1 else None


def _open_doc(path):
    """Open (and for reflowable EPUBs, paginate) a doc, caching the result."""
    if path in _doc_cache:
        return _doc_cache[path]
    import fitz
    doc = fitz.open(path)
    if doc.is_reflowable:
        doc.layout(width=600, height=800, fontsize=11)   # must match locate-time
    _doc_cache[path] = doc
    return doc


def _locate_page(doc, section, text, page):
    """Return a 0-based page index for the passage. PDFs: use `page`. EPUBs:
    search for a distinctive text phrase, then the section heading."""
    if page and not doc.is_reflowable:
        return max(0, min(page - 1, doc.page_count - 1))
    needles = []
    if text:
        words = " ".join(text.split()).split(" ")
        for start in (0, 8, 16):                 # try a few phrases from the chunk
            if len(words) > start:
                needles.append(" ".join(words[start:start + 6]))
    if section:
        needles.append(section)
    for i in range(doc.page_count):
        pg = doc[i]
        for nd in needles:
            if nd and pg.search_for(nd):
                return i
    if page:                                     # last resort if nothing matched
        return max(0, min(page - 1, doc.page_count - 1))
    return None


@mcp.tool()
def get_book_image(collection: str = "", source: str = "", section: str = "",
                   text: str = "", page: int = 0, dpi: int = 130) -> Image:
    """Render the page of a source document where a retrieved passage appears and
    return it as an IMAGE, so a multimodal model can inspect charts, figures,
    tables, or equations a text chunk only references. Pass the `collection` and
    `source` (basename is fine); for PDFs pass `page` (from the search result);
    for reflowable EPUBs pass `section` and/or a `text` snippet to locate it.
    For a video FRAME, pass `source` = the frame path shown in the search result
    (it is returned directly, no rendering)."""
    _, store, _ = _collection(collection)
    # A video frame (or any indexed image) is returned as-is — no page render.
    if source and os.path.splitext(source)[1].lower() in IMAGE_EXTS \
            and os.path.isfile(source) and _within_collections(source):
        _touch()
        return Image(path=source)
    path = _resolve_source(store, source)
    if not path:
        srcs = [os.path.basename(s) for s in store.sources()]
        raise ValueError(f"specify `source` (one of {len(srcs)} indexed files), "
                         f"e.g. {srcs[0] if srcs else 'n/a'!r}")
    if not os.path.exists(path):
        raise ValueError(f"source file no longer on disk: {path}")
    with _render_lock:
        doc = _open_doc(path)
        idx = _locate_page(doc, section, text, page)
        if idx is None:
            raise ValueError("could not locate the passage to render; pass a "
                             "`page`, `text` snippet, or `section`")
        png = doc[idx].get_pixmap(dpi=dpi).tobytes("png")
    _touch()
    return Image(data=png, format="png")


# ---- Video screenshots (yt-dlpcc video units) -----------------------------

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov"}
MAX_SCREENSHOTS = 10


def _video_manifest_for(source_path):
    """If `source_path` (a transcript) has a `<base>.video.json` sibling
    written by yt-dlpcc, return the parsed manifest, else None."""
    base = os.path.splitext(source_path)[0]
    mpath = base + ".video.json"
    try:
        if os.path.isfile(mpath):
            return json.load(open(mpath))
    except Exception:
        pass
    return None


def _parse_timestamp(v):
    """Accept 90, 90.5, '90.5', '1:30' or '0:01:30.500' -> seconds (float)."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if ":" not in s:
        return float(s)
    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError(f"bad timestamp {v!r}")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return sec


def _video_duration(path):
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30, check=True)
        return float(out.stdout.strip())
    except Exception:
        return None


def _resolve_video(cfg, store, video):
    """Map a video name (mp4 basename, transcript name, or partial path) to
    the on-disk video file inside the collection folder."""
    folder = cfg["folder"]
    base, ext = os.path.splitext(video)
    names = {video}
    if ext.lower() not in VIDEO_EXTS:      # given the transcript/manifest name
        names |= {base + e for e in VIDEO_EXTS}
        names |= {os.path.basename(base) + e for e in VIDEO_EXTS}
    cands = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() not in VIDEO_EXTS:
                continue
            p = os.path.join(root, f)
            if f in {os.path.basename(n) for n in names} or \
                    any(p.endswith(n) for n in names):
                return p
            cands.append(p)
    if len(cands) == 1 and not video:
        return cands[0]
    return None


def _sub_filter_path(p):
    """Escape a path for use inside an ffmpeg filter argument."""
    return p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


@mcp.tool()
def screenshot_video(collection: str = "", video: str = "",
                     timestamps: list = [], captions: bool = True,
                     height: int = 720) -> list:
    """Grab up to 10 still frames from a video saved in a collection, at the
    timestamps you choose, and return them as IMAGES (each preceded by its
    timestamp label). Use this to visually investigate a video whose transcript
    you found via `search`: pick moments from the [H:MM:SS] markers in the
    transcript text. `video` is the mp4 name shown in the search result (the
    transcript name also works). `timestamps` entries may be seconds (90.5) or
    clock strings ('1:30', '0:01:30.500'). The video's closed captions are
    burned onto each frame (pass captions=false for clean frames). No GPU and
    no embedding model involved — this is a cheap ffmpeg seek."""
    import subprocess, tempfile
    _, store, cfg = _collection(collection)
    if not timestamps:
        raise ValueError("pass 1-10 `timestamps` (seconds or 'H:MM:SS')")
    path = _resolve_video(cfg, store, video)
    if not path or not _within_collections(path):
        raise ValueError(f"video {video!r} not found in collection folder "
                         f"{cfg['folder']} — use the mp4 name from a search "
                         "result")
    ts = [_parse_timestamp(t) for t in timestamps]
    truncated = len(ts) > MAX_SCREENSHOTS
    ts = ts[:MAX_SCREENSHOTS]
    dur = _video_duration(path)
    srt = os.path.splitext(path)[0] + ".srt"
    burn = captions and os.path.isfile(srt)
    out = []
    if truncated:
        out.append(f"(note: {len(timestamps)} timestamps given; "
                   f"only the first {MAX_SCREENSHOTS} are returned)")
    for t in ts:
        if dur is not None:
            t = max(0.0, min(t, max(0.0, dur - 0.05)))
        vf = f"scale=-2:{height}"
        if burn:
            vf = (f"subtitles='{_sub_filter_path(srt)}'"
                  f":force_style='FontSize=20,Outline=1'," + vf)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-y", "-ss", f"{t:.3f}"]
            if burn:
                cmd += ["-copyts"]      # keep pts so the right cue is burned
            cmd += ["-i", path, "-vf", vf, "-frames:v", "1", "-q:v", "3",
                    "-f", "image2", tmp_path]
            subprocess.run(cmd, check=True, timeout=120, capture_output=True)
            data = open(tmp_path, "rb").read()
            if not data:
                raise RuntimeError("empty frame")
            out.append(f"frame @{_fmt_ts(t)} ({os.path.basename(path)}):")
            out.append(Image(data=data, format="jpeg"))
        except Exception as e:
            out.append(f"frame @{_fmt_ts(t)}: FAILED ({e})")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    _touch()
    return out


@mcp.tool()
def status() -> str:
    """Report whether the embedding model is warm or cold, per-collection counts,
    the GPU device, and seconds since the last request."""
    warm = emb.alive
    idle = time.time() - _last_used[0] if _last_used[0] else None
    lines = [f"model: {'WARM' if warm else 'COLD'}"
             f"{' on ' + (emb.device or '?') if warm else ''}",
             f"idle shutdown after: {IDLE_SECONDS}s",
             f"seconds since last request: "
             f"{f'{idle:.0f}' if idle is not None else 'n/a'}",
             "collections:"]
    for name, c in COLLECTIONS.items():
        st = stores[name]
        lines.append(f"  • {name}: {st.count()} chunks, "
                     f"{len(st.sources())} files  [{c['folder']}]")
    return "\n".join(lines)


if __name__ == "__main__":
    transport = os.environ.get("RAG_TRANSPORT", "stdio")
    if transport in ("http", "streamable-http"):
        mcp.settings.host = os.environ.get("RAG_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("RAG_PORT", "8077"))
        mcp.run(transport="streamable-http")
    else:
        mcp.run()   # stdio
