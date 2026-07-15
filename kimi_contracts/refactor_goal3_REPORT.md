# Refactor Goal 3 report

## Files changed
`git diff --stat` output against commit 0c24882:
```
 .gitignore |   3 +
 ingest.py  | 122 ++++++++++++++++++++++++++++----------
 server.py  | 193 ++++++++++++++++++++++++++++++++++++++++++++-----------------
 3 files changed, 235 insertions(+), 83 deletions(-)
```
(`.gitignore` is the goal-1 "models/" addition carried forward; the
substantive refactor is the two `.py` files.)

## Archive locations
- **STATF401**: `index/STATF401/_archive_pre_dualindex/` — files present: `embeddings.npy` (10 862 720 bytes), `meta.jsonl` (3 008 997 bytes)
- **workbook**:   `index/workbook/_archive_pre_dualindex/` — files present: `embeddings.npy` (15 024 256 bytes), `meta.jsonl` (4 838 750 bytes)
- **EVO2**:       `index/EVO2/_archive_pre_dualindex/` — files present: `embeddings.npy` (41 088 bytes), `meta.jsonl` (10 208 bytes)
- **Personal**:   `index/Personal/_archive_pre_dualindex/` — files present: `embeddings.npy` (254 080 bytes), `meta.jsonl` (84 677 bytes)

`manifest.json` and `ocr_cache/` were left in place at `index/<name>/` for each of the four — only `embeddings.npy` and `meta.jsonl` were moved, exactly as the contract asked.

## Dual-store wiring
- `text_stores` and `image_stores` dicts in `server.py`, replacing the single
  `stores` dict.
- Each collection gets two `IndexStore` instances rooted at
  `index/<name>/text/` and `index/<name>/image/` respectively. `IndexStore`
  itself was not touched — it already had no notion of "collection" or
  "modality", and that generality is what made the split free.
- `INGEST_VERSION` bumped: **4** (was 3). The existing manifest-version guard
  in `ingest_path`'s "fresh = (...)" check now rejects every prior `iv:3` row,
  so on next reconcile every file is force-re-embedded via the new dual-store
  path. No manual manifest editing required.
- `_collection(name)` keeps returning the 3-tuple `(name, text_store, cfg)` so
  `get_book_image` and `screenshot_video` (which the contract said NOT to
  touch) keep unpacking with `_, store, _ = _collection(...)` unchanged.
  PDF/EPUB text chunks land in `text_store` so those tools'
  `store.sources()` lookups still resolve.
- New helper `_image_store(name)` returns `image_stores.get(name)`; only
  `ingest_path` (which is in scope to edit) calls it.
- Two embedders kept as separate globals with the same names the contract
  asked for:
  - `text_emb = TextEmbedder()` — small `nomic-embed-text-v1.5` on the 780M
    iGPU via Vulkan llama-server (validated in goals 1 & 2).
  - `emb = Embedder()` — the existing big `Qwen3-VL-Embedding-2B` worker on
    the 7700S dGPU, now reserved ONLY for raster-image ingest.

## ingest.py changes
- `ext_kind image handling`: confirmed. `ext_kind()` now returns `"image"` for
  `.jpg/.jpeg/.png/.webp` (the same set as `server.IMAGE_EXTS`, duplicated
  with a "single source of truth" comment to avoid an `ingest<->server`
  import cycle). `supported_files()` picks them up automatically because it
  filters by `ext_kind`, and `watcher.py` already advertises them in its
  `SUPPORTED` set.
- `ingest_file dispatch signature` (the new function header, copied
  verbatim):
  ```python
  def ingest_file(path, text_emb=None, image_emb=None,
                  text_store=None, image_store=None,
                  ocr_pages=None, dpi=150, chunk=1200):
      """Embed one file and append to the appropriate store. Returns
      (n_chunks, modality). Dispatched by file kind:
        * pdf / text / epub -> text_emb.embed_text(...) -> text_store
          (PDF requires ocr_pages to already be supplied)
        * image              -> image_emb.embed_image([path]) -> image_store,
          appending 1 vector with meta {"source": path, "modality": "image"}
      Caller is responsible for remove_source() on BOTH stores before calling
      when re-ingesting. Pass only the embedder+store pair relevant to the
      kind; ingest_file raises ValueError if the other pair is missing."""
  ```
  Returned modality strings: `"ocr"` (PDF), `"text"` (txt/epub), `"image"`
  (new). PDF/text/epub paths now route via `text_emb.embed_text(...)`;
  image paths via `image_emb.embed_image([path])` then
  `image_store.append(vecs, [{"source": path, "modality": "image"}])`.
- `ingest.py` CLI `main()` rewritten to match: it builds both stores at
  `<index>/text` and `<index>/image`, splits the file list into text-y and
  image files, runs OCR prepass first (if any PDFs), then cold-starts
  `TextEmbedder()` for the text phase, then cold-starts `Embedder()` for
  the image phase. Each phase prints its own per-file timing line and the
  final summary distinguishes `text store: N vectors` from
  `image store: M vectors`. `--reset` now resets BOTH stores.

## watcher.py
Touches stores directly? **NO.** The full `watcher.py` was re-read end to
end. It only:
1. imports `rag_collections.load()` for the watched folder list,
2. talks to the server's `ingest_path` MCP tool over HTTP via
   `mcp.client.streamable_http.streamablehttp_client` + `ClientSession.call_tool`,
3. triggers reconciles via debounced watchdog events.

There are zero references to `IndexStore`, `stores`, `Embedder`, or any
embedding model from `watcher.py`. **No changes made to `watcher.py`.**

## __dualtest__ scratch smoke test
Full command(s) run:
```
/home/gabriel/venv/bin/python /home/gabriel/projects/rag-mcp/kimi_contracts/smoke_goal3_dualtest.py
```

Full output:
```
=== Refactor Goal 3: __dualtest__ smoke test ===
  added __dualtest__ to collections.json
  created /tmp/rag_dualtest/hello.txt (23 bytes)
  created /tmp/rag_dualtest/hello.png (99 bytes)

--- fw-dgpu status BEFORE ---
state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended

--- importing server (fresh) ---
  COLLECTIONS keys: ['EVO2', 'Personal', 'STATF401', '__dualtest__', '__edge1__', '__edge2__', 'workbook']
  '__dualtest__' in text_stores: True
  '__dualtest__' in image_stores: True
  text_stores['__dualtest__'].root = /home/gabriel/projects/rag-mcp/index/__dualtest__/text
  image_stores['__dualtest__'].root = /home/gabriel/projects/rag-mcp/index/__dualtest__/image
  INGEST_VERSION = 4

  text store count pre:  0
  image store count pre: 0

--- calling server.ingest_path(collection='__dualtest__') ---
  ingest_path returned in 21.70s
--- ingest_path output ---
ingest reconcile of /tmp/rag_dualtest -> collection '__dualtest__':
  added:   hello.png(1), hello.txt(1)
  updated: none
  removed: none
  unchanged: 0
  total vectors now: text=1 image=1
--- end ingest_path output ---

  text store count post:  1
  image store count post: 1
  text store sources: {'/tmp/rag_dualtest/hello.txt': 1}
  image store sources: {'/tmp/rag_dualtest/hello.png': 1}

  text count = 1 (expected 1): PASS
  image count = 1 (expected 1): PASS
  text vector shape=(1, 768), dtype=float16, L2-norm=1.0000
  image vector shape=(1, 2048), dtype=float16, L2-norm=0.9998

--- fw-dgpu status AFTER (embedder still warm) ---
state=on power=9.0 clients=4 dstate=D0 runtime=active

--- stopping both embedders ---

--- fw-dgpu status AFTER stop ---
state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended

--- cleanup ---
  removed __dualtest__ from collections.json

--- summary ---
  text store count  = 1 (expected 1):  PASS
  image store count = 1 (expected 1):  PASS
  OVERALL: PASS
```
- Text store vector count: **1** (one `.txt` chunk → 768-dim float16 L2-normalized, from the small `nomic-embed-text-v1.5` on 780M)
- Image store vector count: **1** (one `.png` → 2048-dim float16 L2-normalized, from the big VL-2B on 7700S)
- PASS/FAIL: **PASS**
- Cleanup confirmed (removed from `collections.json`): **YES** — the script atomically pops the entry before exit and prints confirmation. The `index/__dualtest__/` dir is left behind as expected (matches the existing `__smoke__*` precedent).

Notes on the wall time (21.7 s for `ingest_path`):
- ~15 s is the VL-2B cold-start on the 7700S (the `emb.embed_image()` worker
  loads 2B parameters via torch); that's the one the goal 2 / goal 1 budgets
  both called out.
- ~3 s is the `text_emb` cold-start on the 780M (validated in goal 1).
- The remainder is per-file encode + IndexStore atomic append.

The dGPU was D3cold BEFORE the test, awake DURING (expected — `emb` was
resident on the 7700S), and back to D3cold after both embedders were
stopped in the test's cleanup. This matches what goal 1/2 documented as
the expected pattern.

## import sanity check
Command run:
```
/home/gabriel/venv/bin/python -c "import sys; sys.path.insert(0, '/home/gabriel/projects/rag-mcp'); import server; \
  print('IMPORT OK'); \
  print('text_stores:', sorted(server.text_stores.keys())); \
  print('image_stores:', sorted(server.image_stores.keys())); \
  print('INGEST_VERSION:', server.INGEST_VERSION); \
  print('text_emb:', type(server.text_emb).__name__, 'device=', server.text_emb.device, 'dim=', server.text_emb.dim); \
  print('emb:', type(server.emb).__name__, 'device=', server.emb.device)"
```
Result:
```
IMPORT OK
text_stores: ['EVO2', 'Personal', 'STATF401', '__edge1__', '__edge2__', 'workbook']
image_stores: ['EVO2', 'Personal', 'STATF401', '__edge1__', '__edge2__', 'workbook']
INGEST_VERSION: 4
text_emb: TextEmbedder device= None dim= None
emb: Embedder device= None
```
(`device=None`/`dim=None` for both embedders is the expected lazy state — both
are unstarted until their first `.start()` call. The smoke test above
proved `.start()` and `.stop()` work end to end.)

## Anything else noticed but NOT touched (out of scope)
- The live `rag-mcp.service` / `rag-ingest.service` are NOT being restarted in
  this goal; the live server process (PID 2289) is still running the pre-refactor
  `server.py` it imported at startup. Goal 5 is the restart goal. Running
  the live server today against the new on-disk code would require restart
  (forbidden in this goal).
- `index/__edge1__/` and `index/__edge2__/` already exist on disk (throwaway
  test debris from prior work, per the contract). Importing the modified
  `server.py` creates `index/__edge1__/{text,image}/` and
  `index/__edge2__/{text,image}/` subdirectories because `IndexStore` does
  `os.makedirs(..., exist_ok=True)` in its constructor. These are empty (no
  `embeddings.npy`, no `meta.jsonl`) and harmless — they only matter when the
  watcher tries to ingest into those `__edge*__` collections, which it
  doesn't because those folders are `/tmp/x` and `/tmp/x2`. Listed as
  noticed, not acted on.
- `_idle_watchdog` still uses a single shared `_last_used` clock for both
  embedders. The contract explicitly said "leave the deeper split
  (independent watchdogs per backend) for goal 4" — the minimal change here
  tears down BOTH embedders together when the shared idle clock expires,
  which is safer than letting either run forever but is not yet per-backend.
- `index/__dualtest__/` debris (manifest.json + 2 vectors each in text/image
  subdirs) is intentionally left behind per the contract. If a future
  cleanup pass is desired, removing the directory is safe — nothing else
  references it (the `__dualtest__` entry was popped from `collections.json`).

## Blockers
None.