# CONTRACT — Refactor Goal 4: split `search` into `search_text` + `search_image`, with per-backend idle watchdog

This is a binding scope contract. Read it fully before touching any file.
This goal is about the USER-FACING interface: it changes which tools the
LLM sees and how they're documented. Done right, it sharply lowers battery
cost (text queries never wake the 7700S) and keeps image search fully
available for when it's actually wanted.

## 0. Ground truth and context

Read `~/projects/rag-mcp/kimi_contracts/refactor_goal3_REPORT.md` FIRST —
it confirms what dual-store plumbing is now in place and which names
(`text_stores`/`image_stores`, `text_emb`/`emb`, `INGEST_VERSION=4`, etc.)
are actually in the code. Use THOSE exact names; don't re-derive them. If
goal 3 reported any blocker, STOP here and report — do not try to finish
the split on broken plumbing.

After goal 3:
- `search()` is currently a single FastMCP tool that always uses the TEXT
  store and TEXT embedder; it needs to become two tools.
- `_ensure_warm()` / `_idle_watchdog` in `server.py` currently only watch
  the single `emb`. After the split, the IMAGE backend (big VL-2B on the
  7700S) is the expensive one and MUST get its own tighter lifetime (cold
  + warm idle-watched, just like today) — but the TEXT backend (small
  model on the 780M iGPU via Vulkan `llama-server`, goal 1/2) has a
  different power profile: the iGPU is always on, it's cheap, and adding
  a separate watchdog buys nothing. You may keep one shared watchdog that
  shuts down BOTH backends when the second-to-last activity (across both)
  exceeds `RAG_IDLE`, OR you may wire text_emb to NEVER be torn down within
  the configured lifetime and only add a periodic upper bound — your call,
  but the report MUST justify the choice (which one you picked and the
  power/latency reasoning behind it).
- Per-image embedding currently exists in the codebase but is dead code;
  goal 3 activated it in ingest; this goal makes it user-facing.

Any line numbers cited below may have drifted. `grep -n` to confirm before
editing.

## 1. Sole authorized task

1. **Replace `search`** (current `server.py` ~lines 147-190, exposed via
   `@mcp.tool()`) with TWO FastMCP tools:
   - `search_text(query: str, collection: str = "", k: int = 5) -> str`
     — embeds the query with `text_emb` against the collection's
     `text_store`. MUST be cold-startable, MUST touch ZERO GPU resources on
     the 7700S (the whole point), MUST keep the existing output formatting
     and footer hints (video-on-disk + get_book_image hint) — those were
     already correct, just keep them on the text path.
   - `search_image(query: str, collection: str = "", k: int = 5,
     collection_fold: bool = False) -> str` — embeds the query with
     `emb` (VL-2B) against the collection's `image_store`. If the
     collection's image_store is empty, return a clear text-only message
     explaining the user can index images with `ingest_path` and listing
     which extensions are supported (`.jpg/.jpeg/.png/.webp`).
   Both must use `_collection(collection)` to resolve the name as today.
   Neither should add longer response texts than the current single
   `search` output — it's already borderline long. Do NOT change the
   downstream tool name references for `get_book_image` or
   `screenshot_video` — those stay unchanged.
2. **Update the system-prompt style "Available collections" hint** (the
   `list_collections` tool body, current `server.py` ~lines 133-144) so
   the LLM now sees both `search_text` and `search_image` as available
   tools and understands the split. The shortest correct way: in the
   tool's docstring, append a one-line note like "Use `search_text` for
   passages and `search_image` for screenshots/figures; default queries
   the text store."
3. **Update `status()`** (current `server.py` ~lines 520-536) to report
   BOTH backend states independently — text backend (its lifecycle state
   + device, which will be "780M/Vulkan0" or similar), image backend
   (warm/cold + device = "RX 7700S" or asleep). The current output is a
   single line; replace with two lines, one per backend.
4. **Idle watchdog / lifecycle**: choose between the two options described
   in section 0 — document your choice and reason in the report. Both
   backends must still get torn down cleanly after idle (the user's
   battery-life concern is exactly about killing the 7700S, which is the
   image backend — that's the load-bearing requirement here).
5. **Verify `watcher.py` is unaffected**: read it; confirm it only calls
   the `ingest_path` MCP tool and doesn't reference the removed
   `search` tool by name. If it does, update its docstring/comments but
   do NOT change its run semantics.

## 2. Explicit prohibitions

- Do NOT change filenames of the existing Python modules.
- Do NOT rename `text_embedder_client.py` / `Embedder` / `TextEmbedder`.
- Do NOT touch `get_book_image`, `screenshot_video`, `_open_doc`,
  `_locate_page`, `_video_manifest_for`, `_parse_timestamp`,
  `_resolve_video`, `_video_duration`, `_sub_filter_path` — they remain
  untouched. The whole point of this goal is the SEARCH-TIME split, not
  document rendering.
- Do NOT touch ingest-side code (it's already correctly dual from goal 3).
- Do NOT restart `rag-mcp.service` / `rag-ingest.service`. Service restart
  and the full reingest are goal 5.
- Do NOT git commit, push, or amend.
- Do NOT change `indexing/format` of the dual stores. Do NOT renumber
  `INGEST_VERSION` again.

## 3. Timeout / budget

Target completion within 45 minutes.

## 4. Verification before declaring done

- Re-read `server.py` top to bottom and confirm there is no remaining
  reference to a single `search` tool or a single `store` dict.
- `python -c "import server"` from the venv (`/home/gabriel/venv/bin/python`,
  the one `rag-mcp.service` uses) — must complete with no exception.
- Run BOTH tools through the live MCP over HTTP at
  `http://127.0.0.1:8077/mcp` — but DON'T restart the service to pick up
  changes yet. Instead, do this:
  - Spin up a *separate* test-only MCP server instance (e.g. on port
    8093; verify it's free first) by running `RAG_PORT=8093
    RAG_TRANSPORT=streamable-http /home/gabriel/venv/bin/python
    /home/gabriel/projects/rag-mcp/server.py` in the background
    (record its PID; kill it at the END of verification) — that picks up
    the new code WITHOUT touching the live service. Use an in-process
    MCP client (look at how `watcher.py` does it: it imports
    `mcp.client.streamable_http` + `ClientSession`) to call
    `search_text(query="cat", collection="EVO2", k=3)` AND
    `search_image(query="whiteboard", collection="EVO2", k=3)` and
    confirm both return without exception (image query will return the
    "no images indexed" message since `EVO2` has none, which is the
    correct behavior — include it in the report output as proof).
  - Run `sudo /usr/local/bin/fw-dgpu status` BEFORE and AFTER the
    `search_text` call — both MUST show the dGPU asleep (D3cold/off).
    This is the load-bearing battery-life assertion of the whole refactor;
    if it fails the watchdog/lifecycle wiring is broken.
  - Kill the test server PID at the end. Confirm `pgrep -af
    "RAG_PORT=8093"` (or whatever pattern you can use) returns empty.

## 5. Exact deliverable format

Write to `~/projects/rag-mcp/kimi_contracts/refactor_goal4_REPORT.md`:

```markdown
# Refactor Goal 4 report

## Tools added
search_text: exposed? docstring: ...
search_image: exposed? docstring: ...
search: removed? (must be YES)

## Lifecycle/watchdog design
Approach chosen: shared per-cumulative-idle watchdog / per-backend split / ...
Justification (one short paragraph): ...

## status() before vs after
Before (paste the old single-line format): ...
After (paste the new dual-backend format): ...

## Verification — test server on port 8093
Test server PID: ...
Live service (rag-mcp.service) untouched? YES/NO
search_text query run: "cat", collection="EVO2", k=3
  Output (first ~20 lines): ...
  dGPU status BEFORE call: ...
  dGPU status AFTER call: ...
search_image query run: "whiteboard", collection="EVO2", k=3
  Output (first ~10 lines): ...
import sanity check (`python -c "import server"`): PASS/FAIL
Test server killed cleanly and confirmed dead: YES/NO

## Files changed
git diff --stat output against 0c24882 (cumulative, including any from goal 3): ...

## Anything else noticed but NOT touched (out of scope)
<bullet list, or "none">

## Blockers
<bullet list, or "none">
```

## 6. Termination

If both tools respond successfully AND both `search_text` dGPU-status
checks show the 7700S asleep, immediately read
`~/projects/rag-mcp/kimi_contracts/refactor_goal5_full_reingest_and_restart.md`
and begin executing it. Otherwise STOP after writing this report and wait.
