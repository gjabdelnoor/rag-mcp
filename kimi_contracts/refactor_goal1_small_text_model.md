# CONTRACT — Refactor Goal 1: validate a small text embedding model on the 780M (Vulkan), standalone

This is a binding scope contract. Read it fully before touching any file.

## 0. Ground truth and context

`~/projects/rag-mcp` is a live, currently-running MCP knowledge-base server
(systemd services `rag-mcp.service` + `rag-ingest.service`, both `active`
right now — other tools depend on them, DO NOT restart or stop them in this
goal). Today, EVERY embedding call — including a single-word text search
query — loads the 2B-parameter `Qwen/Qwen3-VL-Embedding-2B` model onto the
discrete RX 7700S GPU (`embedder_worker.py`, hardcoded `cuda:0`) and wakes it
from D3cold sleep. That dGPU wake + fan ramp is what tanks Gabriel's battery
when he uses the RAG tool on the go. Goal of this whole 5-goal refactor:
split into TWO models — a tiny, cheap TEXT embedding model that runs on the
integrated 780M GPU (Vulkan, never wakes the dGPU) for everyday text search,
and keep the existing big VL-2B model on the 7700S reserved ONLY for a new,
separate, deliberate "image search" tool. This goal (1 of 5) only validates
the small-model-on-780M piece in isolation — no production code changes yet.

Proven device-pin recipe already used on THIS exact machine for a different
llama.cpp server (`~/qwen-vulkan-launcher/qwen-vulkan-launch.sh`): export
`GGML_VK_VISIBLE_DEVICES=<index>` where index is derived so that only
"Vulkan0" (= 780M iGPU, RADV PHOENIX) survives — Vulkan1 (RX 7700S, Navi33)
must NEVER be allocated (see lines ~124-134 of that script for the exact
pattern; it's a real working reference, read it). A working Vulkan
`llama-server` binary is already installed and used by this repo's OCR path:
`RAG_LLAMA_BIN=/home/gabriel/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-vulkan-avx2-2.22.0/llama-server`
(see `~/projects/rag-mcp/ocr_client.py:52-69`, `find_llama_server()`). Try
this binary first for `--embedding` serving; if it lacks embedding support
for the chosen model architecture, fall back to
`~/qwen-bench/llama.cpp/build-vulkan/bin/llama-server` (a newer upstream
build also present on this machine), and report which one worked.

There is a CLI to check/control the dGPU's sleep state, already installed,
NOPASSWD sudo: `sudo /usr/local/bin/fw-dgpu status` (also `on`/`off`, but you
should only ever need `status` in this goal — never call `on`/`off`). Expect
it to report D3cold/suspended/off both before and after everything you do in
this goal, since nothing here should touch the 7700S at all.

Landmine from this codebase's own history (see `qwen3_vl_embedding.py` and
prior notes): an embedding model can load and run with ZERO errors while
producing semantically-meaningless output (e.g. a checkpoint/architecture
mismatch silently random-initializes weights) — L2-normalized garbage still
looks like a valid unit vector. NEVER declare an embedding model "working"
without a semantic sanity check (below).

Any line numbers cited above may have drifted. `grep -n` to confirm before
relying on them.

## 1. Sole authorized task

Produce a STANDALONE validation script (does not import/touch any existing
rag-mcp production module) that proves a small (<300M parameter) text
embedding model can be served entirely on the 780M iGPU via Vulkan
`llama-server`, with sane semantic output, while never touching the 7700S.

1. **Pick and download** a small text-embedding GGUF model, in this
   preference order — try the first, and only fall back if it genuinely
   fails to load or serve embeddings after real troubleshooting:
   a. `nomic-ai/nomic-embed-text-v1.5-GGUF` (Q8_0 or F16 quant, ~150-550MB;
      note it uses `search_query: ` / `search_document: ` text prefixes —
      record this in the report, it matters for goal 2).
   b. A reputable `bge-small-en-v1.5` GGUF repo (e.g. `CompendiumLabs/bge-small-en-v1.5-gguf`),
      ~33M params, no special prefix, `--pooling cls`.
   c. A reputable `gte-small` GGUF repo, similar profile to (b).
   Save the downloaded file under `~/projects/rag-mcp/models/` (create the
   dir). Add `models/` to `~/projects/rag-mcp/.gitignore` (one line — this
   IS in scope, it's not a code refactor, just keeping a binary blob out of
   git).
2. **Serve it**: launch `llama-server --embedding --pooling <mean|cls per
   model card> -ngl 99 --device Vulkan0` (mirror the exact device-pin
   pattern from `qwen-vulkan-launch.sh`: `export GGML_VK_VISIBLE_DEVICES=<idx
   for the 780M>` before launching, so only Vulkan0 exists from the
   process's point of view). Pick a local port NOT already in use — ports
   8033, 8077, 8078, 8080 are already taken on this machine by other
   services; verify your chosen port is free with `ss -ltn` before binding
   (try 8091 or nearby, but verify, don't assume).
3. **Before** starting the server: run `sudo /usr/local/bin/fw-dgpu status`,
   record the output verbatim.
4. **Semantic sanity check**: query the running server's embedding endpoint
   (try `/embedding` and `/v1/embeddings`, whichever the binary actually
   serves) for these 5 sentences, apply the model's query/document prefix
   convention if it has one:
   - A1: "the cat sat on the mat"
   - A2: "a feline rested on the rug"
   - A3: "my dog is sleeping on the couch"
   - B1: "the stock market crashed today"
   - B2: "quarterly earnings missed expectations"
   Compute all pairwise cosine similarities. VERIFY within-group (A-A, B-B)
   similarity is clearly higher than cross-group (A-B) — require at least a
   0.15 gap between the mean within-group and mean cross-group similarity.
   If this fails, the model/serving setup is broken — do not paper over it;
   troubleshoot (wrong pooling mode is the most likely cause) or fall back
   to the next model candidate.
5. **After** the queries: run `sudo /usr/local/bin/fw-dgpu status` again.
   MUST still show D3cold/suspended/off. If it shows the dGPU awake, the
   device pin failed — this is a hard failure for this goal, fix the pin
   (check `GGML_VK_VISIBLE_DEVICES` value and the server's own startup log
   line reporting which Vulkan device it bound) before reporting success.
6. **Measure**: cold load time (process launch to first successful
   embedding), warm per-query latency (single short sentence, repeat 3x and
   average), embedding dimension.
7. Cleanly kill the `llama-server` process when done (and verify with
   `pgrep` that nothing is left running).

## 2. Explicit prohibitions

- Do NOT read, import, or modify `server.py`, `ingest.py`, `worker_client.py`,
  `embedder_worker.py`, `index_store.py`, `rag_collections.py`, or `watcher.py`.
  This goal is fully standalone.
- Do NOT restart, stop, or otherwise touch `rag-mcp.service` or
  `rag-ingest.service` (they are live and in use).
- Do NOT touch the 7700S / run the existing VL-2B embedder / call
  `sudo /usr/local/bin/fw-dgpu on` or `off` — `status` only.
- Do NOT git commit, push, or amend anything.
- Do NOT install new system packages. `pip install`/`uv pip install` of a
  Python HTTP client (e.g. `requests`, likely already available) is fine if
  actually needed; do not add heavier dependencies.
- If every model candidate genuinely fails after real effort, STOP and
  report the exact failure — do not fabricate a passing result.

## 3. Timeout / budget

Target completion within 60 minutes. If one model candidate is stuck (won't
load, won't serve, garbage embeddings unfixable) past ~20 minutes, move to
the next candidate and note the blocker — don't burn the whole budget on one.

## 4. Verification before declaring done

Re-run the finished validation script FRESH (not from memory/earlier partial
output) immediately before writing the report, and paste its real stdout
into the report — including the fw-dgpu status lines and the cosine
similarity numbers.

## 5. Exact deliverable format

Write to `~/projects/rag-mcp/kimi_contracts/refactor_goal1_REPORT.md`:

```markdown
# Refactor Goal 1 report

## Model chosen
HF repo + exact filename: ...
Size on disk: ...
Prefix convention (if any): ...
Pooling mode used: ...
Embedding dimension: ...

## Serving setup
llama-server binary path used: ...
Port: ...
GGML_VK_VISIBLE_DEVICES value used: ...
Server startup log line confirming device bound: ...

## dGPU sleep verification
Before: <verbatim fw-dgpu status output>
After: <verbatim fw-dgpu status output>
PASS/FAIL: ...

## Semantic sanity check
Full pairwise cosine similarity matrix (5x5): ...
Mean within-group similarity: ...
Mean cross-group similarity: ...
Gap: ...
PASS/FAIL (gap >= 0.15): ...

## Timing
Cold load time: ...
Warm per-query latency (avg of 3): ...

## Standalone script location
Path: ...

## Blockers
<bullet list, or "none">
```

## 6. Termination

If the dGPU-sleep verification AND the semantic sanity check both PASS,
immediately read `~/projects/rag-mcp/kimi_contracts/refactor_goal2_text_embedder_client.md`
and begin executing it. If either FAILED or you had to stop on an
unresolved blocker, STOP after writing this report and wait — do not
proceed to goal 2 on a broken foundation.
