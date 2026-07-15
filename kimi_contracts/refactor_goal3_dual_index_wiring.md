# CONTRACT — Refactor Goal 3: wire dual-store (text + image) into ingest and the server

This is a binding scope contract. Read it fully before touching any file.
This is the biggest and riskiest goal of the 5 — it touches the code that a
LIVE, currently-running service (`rag-mcp.service`) executes. You will NOT
restart that service in this goal (that's goal 5) but you WILL be editing
the files it loads on next restart, so correctness matters.

## 0. Ground truth and context

Read `~/projects/rag-mcp/kimi_contracts/refactor_goal2_REPORT.md` FIRST —
it names the exact class (`TextEmbedder` in `text_embedder_client.py`) and
confirms its interface. If that report shows goal 2 failed/blocked, STOP
and report — do not try to build the client yourself here.

**Current architecture (verified by Claude before writing this contract; line
numbers may have drifted, `grep -n` to confirm before editing):**

- `server.py`: single `COLLECTIONS = rag_collections.load()` (~line 39),
  single `stores = {name: IndexStore(c["index_dir"]) for ...}` (~line 40),
  single `emb = Embedder()` (~line 42, the big VL-2B model on the 7700S).
  `INGEST_VERSION = 3` (~line 36) gates re-ingest: `ingest_path` (~lines
  193-296) skips a file if its manifest entry's `iv` matches
  `INGEST_VERSION` AND its hash is unchanged AND it's present in the store
  — bump `INGEST_VERSION` and every file is treated as needing re-ingest,
  NO manual manifest editing required (see the `fresh = (...)` check
  ~line 221).
- `index_store.py`: `IndexStore(root)` is fully generic — it just needs a
  directory (`embeddings.npy` + `meta.jsonl` inside it). It has NO idea
  what "collection" means. You can freely instantiate two of them per
  collection at different subdirectories; no changes to this file needed.
- `ingest.py`: `ext_kind(path)` (~lines 96-104) currently recognizes only
  `.pdf`/`.txt,.md,.markdown,.text`/`.epub`. Raster image files
  (`.jpg/.jpeg/.png/.webp`) are NOT recognized — `supported_files()`
  (~lines 107-115) silently skips them. `server.py` already defines
  `IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}` (~line 37) but ONLY
  uses it to detect "is this already-known source path an image file" in
  `get_book_image`, never for ingestion. `ingest_file()` (~lines 149-188)
  is where a file gets embedded and appended to a store — it takes a
  single `(path, emb, store, ...)`.
- `worker_client.py`'s `Embedder.embed_image(paths, captions=None,
  is_query=False)` (~lines 92-98) and `embedder_worker.py`'s `embed_image`
  op (~lines 71-83) ALREADY EXIST and work (they call the real VL-2B model)
  — but nothing in `ingest.py` or `server.py` ever calls `embed_image`
  today. It is dead code you are activating for the first time.
- `~/projects/rag-mcp/index/<name>/` currently holds a FLAT
  `embeddings.npy` + `meta.jsonl` (the old VL-2B text embeddings) plus
  `manifest.json` and `ocr_cache/`. The 4 REAL collections are `STATF401`,
  `workbook`, `EVO2`, `Personal` (see `~/projects/rag-mcp/collections.json`).
  There are also several throwaway test dirs (`__smoke__`, `__smoke3__`,
  `__smoke4__`, `__smoke5__`, `__edge1__`, `__edge2__`, `__edge3__`,
  `__frametest__`) — these are disposable debris from prior work, IGNORE
  them entirely, do not archive or touch them.
- A git checkpoint commit `0c24882` exists at the repo root capturing the
  pre-refactor working state of all tracked files (NOT `index/`, which is
  gitignored) — this is your rollback point if something goes badly wrong;
  you can `git diff 0c24882 -- <file>` to see your own changes at any time.

## 1. Sole authorized task

1. **Archive old per-collection flat index files** (the 4 REAL collections
   only — `STATF401`, `workbook`, `EVO2`, `Personal`): for each, move
   `index/<name>/embeddings.npy` and `index/<name>/meta.jsonl` (if present)
   to a new `index/<name>/_archive_pre_dualindex/` subfolder (`mkdir -p`
   then move, preserving filenames). Do **NOT** move or touch
   `manifest.json` or `ocr_cache/` — those stay exactly where they are and
   get reused (OCR text extraction is unaffected by this refactor; only the
   embedding step changes).
2. **Dual stores**: change collection/store setup (in `server.py`, and
   wherever `ingest.py`'s CLI `main()` builds a store, ~line 202) so each
   collection gets TWO `IndexStore` instances: one at
   `index/<name>/text/` and one at `index/<name>/image/`. Use clear names
   (e.g. `text_stores` / `image_stores` dicts in `server.py`, replacing the
   single `stores` dict).
3. **Two embedders**: keep `emb = Embedder()` (VL-2B, 7700S) but it is now
   used ONLY for images from this point forward. Add
   `text_emb = TextEmbedder()` (import from `text_embedder_client.py`,
   goal 2's module) for all text embedding.
4. **Bump `INGEST_VERSION`** from 3 to 4 in `server.py` so the existing
   manifest-version guard naturally forces every file through re-embedding
   on the next reconcile (no manual manifest surgery).
5. **`ingest.py` changes**:
   - Extend `ext_kind()` to recognize `.jpg/.jpeg/.png/.webp` (reuse
     `server.py`'s `IMAGE_EXTS` constant — import it or duplicate the exact
     same set with a comment pointing at the source of truth; your call,
     but don't let them drift apart) and return `"image"`.
   - Extend `ingest_file()` (or restructure its signature — your call, but
     every call site must be updated consistently) to accept BOTH
     embedders and BOTH stores, and dispatch: text/pdf/epub kinds go
     through `text_emb.embed_text(...)` -> the TEXT store exactly as
     before (just swap which embedder object is used); the new `"image"`
     kind calls `image_emb.embed_image([path])` -> appends 1 vector with
     meta `{"source": path, "modality": "image"}` to the IMAGE store,
     returns `(1, "image")`.
   - Update `ingest.py`'s CLI `main()` (~lines 191-249) to match the new
     dual-embedder/dual-store signature — this CLI path is used directly by
     goal 5's real reingest, so it must actually work, not just the
     server's `ingest_path` tool.
6. **`server.py`'s `ingest_path` tool** (~lines 193-296): update to route
   through `text_stores`/`image_stores` and `text_emb`/`emb` as
   appropriate. For "remove the old vector(s) before re-ingesting a
   changed file" (~lines 254-255, currently `store.remove_source(f)`): call
   `remove_source(f)` on BOTH the text store and the image store for every
   file being re-ingested — `IndexStore.remove_source` is already a safe
   no-op when the source isn't present (returns 0 immediately, see
   `index_store.py`), so this is safe regardless of which store the file
   actually lives in.
   - `list_collections` (~line 142) and `status` (~line 534) currently
     print `st.count()` from the single store — update to report text and
     image counts separately (e.g. "142 text chunks, 3 images") since
     that's more informative and these are now genuinely two different
     things.
7. **`search` tool** (~lines 147-190): for THIS goal only, make the MINIMAL
   change needed to keep it mechanically working against the new plumbing
   — swap `store` -> `text_store` (the resolved collection's text store)
   and `emb` -> `text_emb` in its body. Do **NOT** rename the tool, do
   **NOT** add a new `search_image` tool, do **NOT** touch
   `get_book_image`/`screenshot_video`/the idle-watchdog beyond what's
   needed to keep them importing/running — all of that is explicitly
   **goal 4's job**, not this one. If `search`'s current cold-start logic
   (`_ensure_warm`, `_lock`, `_idle_watchdog`) references the single
   `emb` in a way that's now ambiguous (text vs image), make the smallest
   change that keeps text search working correctly and unambiguously
   against `text_emb`; leave the deeper split (independent watchdogs per
   backend) for goal 4.
8. **`watcher.py`**: read it fully first. It should ONLY call the
   `ingest_path` MCP tool remotely over HTTP and hold no direct references
   to `IndexStore`/`stores`. If that's confirmed true, it needs NO changes
   — say so in the report. If it turns out to touch stores directly
   (contrary to what Claude's research found), do NOT silently patch it —
   report it as a blocker instead, since it wasn't in the original research
   scope for this contract.
9. **Goal-3-level smoke test** (NOT a full reingest of real collections —
   that's goal 5): add one throwaway scratch collection (e.g. `__dualtest__`)
   to `collections.json` pointing at a new empty scratch folder (e.g.
   `/tmp/rag_dualtest/`); put one tiny `.txt` file AND one tiny synthetic
   PNG (generate it yourself with PIL — a plain solid-color square is
   fine) in that folder; call `ingest_path` for that collection; confirm
   the text store gets exactly 1 vector and the image store gets exactly 1
   vector; then REMOVE the `__dualtest__` entry from `collections.json`
   again (leave the `index/__dualtest__/` dir — harmless debris, matches
   the existing `__smoke__`-style precedent) so it doesn't linger as a
   configured collection.

## 2. Explicit prohibitions

- Do NOT touch `search`'s tool NAME, do NOT add `search_image`, do NOT
  restructure the idle-watchdog into per-backend timers — all goal 4.
- Do NOT touch `get_book_image` or `screenshot_video` bodies at all.
- Do NOT delete the archived old-index files.
- Do NOT run a full reconcile/reingest of the 4 REAL collections in this
  goal — only the `__dualtest__` scratch smoke test above. Real reingest is
  goal 5, deliberately kept separate so it can be watched more carefully.
- Do NOT restart `rag-mcp.service` or `rag-ingest.service`.
- Do NOT git commit, push, or amend.
- Do NOT touch `minimaxkey.txt`, OCR remote config, or anything in
  `ocr_client.py`/`ocr_worker.py`/`ocr_cache.py`.
- If something about the current architecture doesn't match this contract's
  description (line numbers drifted is expected and fine — grep and adapt;
  but if a described BEHAVIOR is just wrong), STOP and describe the
  conflict in the report rather than guessing at intent and expanding scope.

## 3. Timeout / budget

Target completion within 90 minutes — this is the largest goal in the
series. If stuck on one specific sub-item past ~30 minutes, move to the
next sub-item and note the blocker; don't let one snag eat the whole
budget.

## 4. Verification before declaring done

- Re-read the full modified `ingest_file()`, `ingest_path()`, and the
  store/embedder init code once more after editing, end to end.
- Run `python -c "import server"` (or equivalent, from the repo's venv —
  check `rag-mcp.service`'s `ExecStart` line for which python, likely
  `/home/gabriel/venv/bin/python`) and confirm no import-time error.
- Run the `__dualtest__` scratch smoke test described in item 9 and paste
  its real output (vector counts confirmed for both stores).

## 5. Exact deliverable format

Write to `~/projects/rag-mcp/kimi_contracts/refactor_goal3_REPORT.md`:

```markdown
# Refactor Goal 3 report

## Files changed
`git diff --stat` output against commit 0c24882: ...

## Archive locations
STATF401: index/STATF401/_archive_pre_dualindex/ — files present: ...
workbook: ...
EVO2: ...
Personal: ...

## Dual-store wiring
text_stores/image_stores variable names used: ...
INGEST_VERSION new value: ...

## ingest.py changes
ext_kind image handling: confirmed / not done — why
ingest_file dispatch signature (paste the new function signature): ...

## watcher.py
Touches stores directly? YES/NO. If YES: <blocker description>. If NO: no changes made.

## __dualtest__ scratch smoke test
Full command(s) run: ...
Full output: ...
Text store vector count: ...
Image store vector count: ...
PASS/FAIL: ...
Cleanup confirmed (removed from collections.json): YES/NO

## import sanity check
Command run: ...
Result: ...

## Anything else noticed but NOT touched (out of scope)
<bullet list, or "none">

## Blockers
<bullet list, or "none">
```

## 6. Termination

If the `__dualtest__` smoke test PASSED (both counts correct) and the
import sanity check PASSED, immediately read
`~/projects/rag-mcp/kimi_contracts/refactor_goal4_split_tools.md` and begin
executing it. Otherwise STOP after writing this report and wait — do not
proceed to goal 4 on top of broken plumbing.
