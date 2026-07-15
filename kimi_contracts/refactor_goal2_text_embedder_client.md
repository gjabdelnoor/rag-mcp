# CONTRACT — Refactor Goal 2: build the reusable text-embedder worker + client

This is a binding scope contract. Read it fully before touching any file.

## 0. Ground truth and context

Read `~/projects/rag-mcp/kimi_contracts/refactor_goal1_REPORT.md` FIRST —
it names the exact model file, port, binary path, device-pin env value,
pooling mode, and prefix convention that goal 1 already validated. Use
THOSE exact choices here; don't re-derive them. If the report shows goal 1
FAILED or was blocked, STOP immediately and report that you cannot proceed
— do not try to re-solve goal 1's problem yourself inside this contract.

This repo already has a proven pattern for managing a long-lived subprocess
as a cold/warm resource with an idle lifecycle — TWO examples to mirror:
- `~/projects/rag-mcp/worker_client.py` — `Embedder` class (`.alive`,
  `.start()`, `.stop()`, `.embed_text(texts, is_query=False)` ->
  `np.ndarray` float32 (n, dim), L2-normalized). This is the exact
  interface shape your new class must match, so it's a drop-in swap in a
  later goal.
- `~/projects/rag-mcp/ocr_client.py` — `OCRWorker` class, especially
  `.stop()` (lines ~243-260, graceful shutdown + hard-kill fallback) and
  `_reap_llama_servers()` (lines ~262-278, a `pgrep`-based orphan-killer
  safety net) — llama-server processes can be left orphaned holding
  resources if a plain SIGTERM doesn't land; mirror this safety net for
  your own model file's process name.

Any line numbers above may have drifted since written — `grep -n` to
confirm before relying on them.

## 1. Sole authorized task

Create `~/projects/rag-mcp/text_embedder_client.py`, a NEW standalone module
(importing it must NOT pull in torch — it only manages a subprocess and
speaks HTTP to it, same "thin client" philosophy as `worker_client.py`'s own
docstring). It must expose a class (name it `TextEmbedder`) with:

- `.alive` (property) — True iff the subprocess is running.
- `.start(ready_timeout=120)` — launches `llama-server` with goal 1's
  validated model/binary/port/device-pin/pooling flags (via
  `subprocess.Popen`, following `OCRWorker.start()`'s pattern of waiting for
  a readiness signal before returning — for a plain llama-server this means
  polling its health/embedding endpoint until it responds, since it doesn't
  speak the JSON-lines stdin/stdout protocol the OTHER two worker classes
  use). On success set `.device` (e.g. `"780M (Vulkan0)"`) and `.dim` (from
  goal 1's report).
- `.stop()` — terminate the subprocess (SIGTERM, wait, SIGKILL fallback,
  mirroring `OCRWorker.stop()`), PLUS a `pgrep`-based safety net that kills
  any orphaned `llama-server` process still holding the specific model
  filename from goal 1 (mirror `_reap_llama_servers()`, adapted to grep for
  your model's gguf filename instead of `surya-2.gguf`).
- `.embed_text(texts: list[str], is_query: bool = False) -> np.ndarray`
  shape `(len(texts), dim)`, dtype float32, L2-normalized. If the model uses
  a query/document prefix convention (goal 1's report says whether it does),
  apply `search_query: ` when `is_query=True` and the document-side prefix
  otherwise — EXACTLY matching what goal 1 validated gives good semantic
  separation. If there's no prefix convention (e.g. bge-small), this is a
  no-op passthrough. This signature must match `Embedder.embed_text` in
  `worker_client.py` exactly (same argument names, same return shape/dtype
  convention) so a later goal can swap one for the other with no call-site
  changes beyond the variable name.
- Before binding, verify your chosen port is actually free (`ss -ltn` check
  or a bind-and-release probe) and raise a clear error if not, rather than
  silently colliding with something else.

Write a standalone test (either inline `if __name__ == "__main__":` in the
same file, or a separate `test_text_embedder_client.py` — your choice) that:
1. Instantiates `TextEmbedder`, calls `.start()`.
2. Runs the SAME 5-sentence semantic sanity battery as goal 1 through
   `.embed_text()`, confirms the same within/cross-group gap (>= 0.15).
3. Calls `.stop()`, confirms `.alive` is False and no orphan process remains
   (`pgrep` check).
4. Checks `sudo /usr/local/bin/fw-dgpu status` before `.start()` and after
   `.stop()` — both MUST show the dGPU still asleep (D3cold/off/suspended).

## 2. Explicit prohibitions

- Do NOT touch `server.py`, `ingest.py`, `worker_client.py`,
  `embedder_worker.py`, `index_store.py`, `rag_collections.py`, or
  `watcher.py`. Wiring this new client into the live server is goal 3, not
  this goal.
- Do NOT restart or stop `rag-mcp.service` / `rag-ingest.service`.
- Do NOT touch the 7700S / the existing VL-2B embedder.
- Do NOT git commit, push, or amend.
- Do NOT change the model/port/device-pin choices goal 1 validated — reuse
  them as-is. If you believe they need to change, STOP and explain why in
  the report instead of unilaterally picking something new.

## 3. Timeout / budget

Target completion within 45 minutes. If stuck past 25 minutes on one
specific issue (e.g. readiness-polling the HTTP endpoint), move on and note
the blocker.

## 4. Verification before declaring done

Run the standalone test FRESH right before writing the report (not reused
output from earlier iteration) and paste its real output, including the
fw-dgpu status lines, into the report.

## 5. Exact deliverable format

Write to `~/projects/rag-mcp/kimi_contracts/refactor_goal2_REPORT.md`:

```markdown
# Refactor Goal 2 report

## File created
Path: ~/projects/rag-mcp/text_embedder_client.py
Class name: TextEmbedder
Confirmed interface: .alive / .start() / .stop() / .embed_text(texts, is_query=False)

## Test run (fresh, this session)
Full stdout: ...
Semantic gap (within vs cross group): PASS/FAIL, numbers: ...
Orphan-process check after stop(): PASS/FAIL
dGPU status before start(): ...
dGPU status after stop(): ...
dGPU sleep PASS/FAIL: ...

## Deviations from goal 1's validated setup (if any, and why)
...

## Blockers
<bullet list, or "none">
```

## 6. Termination

If the test's semantic-gap check AND both dGPU-sleep checks PASS,
immediately read `~/projects/rag-mcp/kimi_contracts/refactor_goal3_dual_index_wiring.md`
and begin executing it. Otherwise STOP after writing this report and wait.
