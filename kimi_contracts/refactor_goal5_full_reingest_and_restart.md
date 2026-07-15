# CONTRACT — Refactor Goal 5: reingest the 4 real collections on the new dual-store, restart, validate live

This is the final goal in the refactor series. It puts the new system in
front of users for real: it orchestrates the full reingest of the 4 real
collections through the new text-on-780M pipeline, then restarts the live
services, then validates end-to-end on the real data. Treat every step as
something Gabriel will see outcomes from — correctness matters.

## 0. Ground truth and context

Read `~/projects/rag-mcp/kimi_contracts/refactor_goal4_REPORT.md` FIRST —
it confirms `search_text` is exposed and that a fresh test server could
call it without waking the 7700S. If goal 4 reported any blocker, STOP and
report — do not try to finish on broken wiring.

**State of the system after goal 4:**

- Live service `rag-mcp.service` (HTTP on 127.0.0.1:8077) is STILL running
  the OLD single-store, VL-2B-on-7700S code — it is unaffected by any
  edits so far. Today it answers `search`/`get_book_image`/`screenshot_video`
  as before. THAT IS THE ROLLBACK POINT — if goal 5 goes badly, the user
  can just NOT commit and the live service keeps working unchanged.
- Live service `rag-ingest.service` is similarly still running the old
  code; it calls `ingest_path` over HTTP which talks to the (old) live
  server. After the restart later in this goal it will pick up new code.
- The 4 REAL collections (`STATF401`, `workbook`, `EVO2`, `Personal`) have
  their OLD VL-2B vectors safely archived at
  `index/<name>/_archive_pre_dualindex/{embeddings.npy,meta.jsonl}`
  (goal 3 verified this). The OCR text extraction is precomputed in
  `index/<name>/ocr_cache/` (unchanged), and per-collection manifest is at
  `index/<name>/manifest.json` (unchanged).
- `INGEST_VERSION` was bumped to 4 (goal 3), so the very first call to
  `ingest_path` will treat every file as needing reingest — the user does
  NOT need to mutate the manifest by hand.
- Two user services exist: `~/.config/systemd/user/rag-mcp.service` (the
  HTTP server) and `~/.config/systemd/user/rag-ingest.service` (the file
  watcher). Both are `enabled` + `WantedBy=default.target` per the
  `rag-mcp-knowledge-base` memory file.
- A git checkpoint commit `0c24882` exists at the repo root — it's the
  pre-refactor working state (not the live service, that's the same code
  on disk that the old service was launched from). The current `master`
  branch has all of goals 1–4's edits uncommitted.

**Service-restart discipline:** the live services use `Restart=on-failure`
in their unit files — restarting them causes a clean new process that
reads the updated files. To restart `rag-mcp.service`:

```
systemctl --user restart rag-mcp.service
```

Each restart takes ~1–2 s; the server comes back on the same port
(127.0.0.1:8077). `watcher.py` retries on connection failure.

**Reingest cost reminder (from the project's own history):** full-reingest
of the textbook (~900 chunks) was ~143 s of GPU compute on the
VL-2B previously. With TEXT on the 780M iGPU (much smaller model), this
should be roughly comparable or faster per-query, but the iGPU runs in
steady-state, so wall time is unlikely to be the limiting factor — the
limiting factors are (a) PDF OCR prepass (still on VL-2B llama-server via
Surya, unchanged behavior — uses the existing remote-first / GPU-fallback
path) and (b) disk I/O. Plan for reingest to take ~10–30 minutes total
across all 4 collections; that's expected and not a blocker.

## 1. Sole authorized task

Execute the full reingest and go live, in this exact order. Read each
step BEFORE acting on it.

1. **Pre-flight**: verify the live service is responding (curl
   `http://127.0.0.1:8077/mcp` — note: MCP over HTTP won't return plain
   HTML, but `--max-time 5` will tell you it's reachable vs. a 60s timeout
   will tell you something's wrong; OR use the small in-process client
   pattern from `watcher.py` to invoke `list_collections` and confirm it
   returns). Confirm `sudo /usr/local/bin/fw-dgpu status` shows the
   dGPU asleep RIGHT NOW. If it shows awake, STOP and ask Gabriel — the
   whole goal assumes a cold 7700S to start with.
2. **Kick off reingest** as a background task so you can monitor
   concurrently: invoke the `ingest_path` MCP tool on the live server ONCE
   PER COLLECTION, with `path=""` (which reconcile-mode reings the watched
   folder for that collection). Put each call behind a tight `--timeout`,
   but DO use `run_in_background` (Bash with `run_in_background=true`) so
   you can monitor progress while the others run. Make sure the calls
   handle the `ingest_path` lock contention correctly (only ONE collection
   may be reingesting at a time on a single server — see the
   `_ingest_lock` in `server.py`). Sequence them — STATF401 first
   (largest/oldest), then workbook, then EVO2, then Personal. Do not
   parallelize the per-collection calls against the same server.
   Wait for the prior to finish before starting the next; you can check
   server log via `journalctl --user -u rag-mcp.service -f` for
   per-file progress lines.
3. **During reingest**, periodically (every ~5 min) verify:
   - The background reingest is making forward progress (look at the
     `added` / `updated` / `unchanged` numbers in each tool result; expect
     the `unchanged` count to stay high after the first collection
     finishes, since files that hash-match an unchanged-but-iv-bumped
     entry go through too — reingest-time should match the "iv bumped ->
     always reingest" code path).
   - `sudo /usr/local/bin/fw-dgpu status` — the 7700S will wake during
     OCR of any PDFs (Surya llama-server is the existing OCR path,
     unchanged), which is expected. Watch for TWO things going wrong:
     a) the dGPU stuck awake AFTER reingest finished (a leaked process
     holding it — use `pgrep` to look for `llama-server` or other GPU
     clients and kill strays; report what you found), b) the embedder
     subprocess left alive between collections (shouldn't be, but if it
     is it's just a missed idle-watchdog — note it).
4. **Restart the services** ONCE ALL 4 REINGESTS HAVE COMPLETED:
   ```
   systemctl --user restart rag-mcp.service
   systemctl --user restart rag-ingest.service
   ```
   Use `systemctl --user is-active` to confirm both come back to
   `active`. Wait ~10 s for the Python server to be ready on the HTTP
   port (`curl http://127.0.0.1:8077/mcp --max-time 5` should at least
   not timeout; or just trust `is-active` and the brief warm window).
5. **End-to-end validation on the live, restarted service**:
   - `list_collections` should now show 4 collections, each with BOTH a
     text count and an image count (per the goal-3 change to the
     `list_collections` output). Paste the full output into the report.
   - Run `search_text("regression", collection="STATF401", k=3)` via the
     MCP HTTP client. Confirm the top hits actually reference STATF401
     material — they will include the index-shifted sources because the
     metadata format is the same as before, but the VECTOR values are now
     from the small text model, not the old VL-2B. Compare the listed
     source basenames to `ls /home/gabriel/Documents/STATF401/Materials/`
     to be sure they are real materials.
   - Run `search_text("matrix multiplication", collection="Personal",
     k=3)` and verify the returns are `Personal` material, not random.
   - Run `search_image("figure", collection="EVO2", k=3)` — confirm the
     empty-collection fallback message comes back, since EVO2 has no image
     files in scope; if it doesn't, the dispatch was wired wrong.
   - Run `status()` and confirm both backends report independently, with
     the text backend showing the 780M/iGPU device and the image backend
     showing whatever (cold/asleep is fine).
6. **dGPU-sleep proof**: IMMEDIATELY after the 3 `search_text` calls above,
   run `sudo /usr/local/bin/fw-dgpu status` — it MUST show D3cold/off.
   Then run a single `search_image(...)` call against any collection with
   images (even an empty-result one); IMMEDIATELY after, run
   `sudo /usr/local/bin/fw-dgpu status` again — note it's now awake (the
   image query is expected to wake it). Then wait ~5 minutes (set a
   timer) without making any further calls and re-check — the watchdog
   should have killed the image backend and the dGPU should be back to
   asleep. This is the END-to-END proof that battery life will improve.
7. **Verify `get_book_image` and `screenshot_video` still work** with a
   tiny smoke call each on a known-good collection/source/timestamp —
   their code bodies were NOT modified by goals 3 or 4, but a unit-test
   from this end prevents confusing regressions if a wiring change
   accidentally broke an import path. Use a passage from an EVO2 search
   result above; pick a `[VIDEO on disk: ...]` reference if one appears,
   otherwise any one source PDF or EPUB. Paste the rendered/payload size
   into the report (don't paste actual image bytes).
8. **NO commit / NO push in this goal** — leave the working tree
   uncommitted. Gabriel may want to inspect the diff or roll back
   cleanly if any subtle issue comes up later. Mention the local branch
   state and uncommitted files in the report (e.g. `git status --short`
   output at the end).

## 2. Explicit prohibitions

- Do NOT modify any source code in this goal. This goal is operations:
  call the existing tools, restart services, observe.
- Do NOT touch `~/projects/rag-mcp/index/<name>/_archive_pre_dualindex/`
  files — those are the proven rollback if anything goes wrong in real
  use.
- Do NOT restart services out of order: ALL reingests MUST be complete
  before either `restart` runs (the new server uses dual stores; if you
  hit the new `ingest_path` mid-reingest it can produce incoherent state).
- Do NOT run git commit / push / amend.
- Do NOT edit `~/projects/rag-mcp/.gitignore` or `collections.json`.
- Do NOT call `sudo /usr/local/bin/fw-dgpu on` or `off` — status only.
  Status-only is read-only and safe.
- If reingest on any individual collection fails (raise, stuck, etc.),
  STOP and report — do not silently skip and proceed, because then a
  partial state is what users see.

## 3. Timeout / budget

Target completion within 90 minutes of wall time, with reingest
dominating. Check progress at 30 and 60 minutes; if any one collection's
reingest is stuck past 30 minutes (no forward progress, no new log lines
in `journalctl` for a clear block reason), STOP and report rather than
waiting indefinitely.

## 4. Verification before declaring done

- Confirm `git status --short` shows every goal-1-through-4 source
  modification as a single-line uncommitted change (or just a handful of
  them), no spurious files, no leftover scratch files outside the
  disposed-of `index/__dualtest__/` (which is meant to linger, matches
  existing precedent).
- Confirm `systemctl --user is-active rag-mcp.service` +
  `rag-ingest.service` are both `active`.
- Confirm both per-backend `status()` lines are reachable via the live
  HTTP server (one more curl after the restart).

## 5. Exact deliverable format

Write to `~/projects/rag-mcp/kimi_contracts/refactor_goal5_REPORT.md`:

```markdown
# Refactor Goal 5 report

## Pre-flight
Live service reachable: YES/NO
dGPU status at start: ...

## Reingest results
STATF401: started, finished, files added/updated/unchanged, total vectors now (text + image): ...
workbook:  ...
EVO2: ...
Personal: ...

## Watcher/embedder orphan check during reingest
dGPU anomalies observed (leaked processes, stragglers): ...
Embedder subprocess stragglers: ...

## Service restart
Active after restart: rag-mcp.service YES/NO, rag-ingest.service YES/NO
Reachable on 127.0.0.1:8077 (curl --max-time 5): ...

## E2E live validation
list_collections full output: ...
search_text("regression", STATF401, k=3) first result: source basename + score + first line of snippet
search_text("matrix multiplication", Personal, k=3) first result: source basename + score + first line of snippet
search_image("figure", EVO2, k=3) output: ...
status() full output: ...

## dGPU-sleep proof
After 3 search_text calls — dGPU status: ...
After search_image call — dGPU status: ...
After 5 min idle — dGPU status: ...
PASS/FAIL (image backend killed by watchdog within 5 min): ...

## get_book_image and screenshot_video smoke
Tool called: get_book_image / screenshot_video
Collection/source/timestamp used: ...
Payload-size or summary: ...
PASS/FAIL: ...

## Final repo state
git status --short output: ...
Files changed cumulatively vs commit 0c24882 (`git diff --stat 0c24882`): ...
NO commit was made (verified): ...

## Anything else noticed but NOT touched (out of scope)
<bullet list, or "none">

## Blockers
<bullet list, or "none">
```

## 6. Termination

After the report is written and the dGPU-sleep proof is either PASSED or
explicitly BLOCKED with a clearly-stated reason, the goal is COMPLETE —
the refactor is in production and Gabriel has the energy-cost outcome he
asked for. STOP and wait. Do not start any other work.

If at any point an unrecoverable problem surfaces (e.g. live service
unreachable after restart with no obvious cause, watchdog doesn't fire
even after >10 minutes idle, image-search cold-start spuriously wakes the
dGPU due to a wiring error), STOP IMMEDIATELY after the report and do
not try to roll back or fix further on your own — describe the exact
state precisely in the report so Gabriel can decide.
