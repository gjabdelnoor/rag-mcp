# rag-mcp — local multimodal RAG MCP server

A knowledge base over your study materials, exposed to a local LLM (e.g. Gemma)
as an MCP server. Material is organized into named **collections** ("notebooks"),
each a watched folder with its own vector index and a description. Embeddings are
multimodal — text, textbook chapters, and **OCR'd PDFs** — all on the **Radeon
RX 7700S** (dGPU).

| modality | how it's indexed |
|----------|------------------|
| `.txt` / `.md` | embedded as text (sliding-window chunks) |
| `.epub` | embedded as text, **one chunk per chapter-section** |
| `.pdf`  | **OCR'd with Surya-2 on the 7700S**, the OCR text embedded one chunk per page; the page image is rendered on demand by `get_book_image` so figures are still visible |

Everything embeds with **Qwen3-VL-Embedding-2B** into each collection's own
vector space.

## Idle profile (the whole point)
- The MCP server process is thin: **mcp + numpy only, no torch**.
- The embedding model runs in a **child subprocess** spawned *cold* on first use
  and **killed after 15 minutes idle** — that returns the GPU to 0% and frees
  VRAM/RAM.
- OCR runs in a **separate short-lived subprocess** only during ingest. The
  embedder and the OCR `llama-server` **never** hold the GPU at the same time:
  ingest runs OCR as a *prepass* (embedder stopped), then embeds (OCR server
  killed). On an 8 GB card neither alone is a problem; together they'd OOM.
- At idle: **0% GPU, ~0% CPU, no model resident**.

## Collections (notebooks)
Defined in `collections.json` (a default is written on first run; see
`collections.example.json`). Each entry is a folder + a description:

```json
{
  "STATF401": {"folder": "~/Documents/STATF401/Materials", "description": "…"},
  "workbook": {"folder": "~/Documents/Gabe's Workbook",    "description": "…"}
}
```
Each collection's index lives under `index/<name>/`. The LLM discovers them with
**`list_collections`** (returns name + description + counts) and then searches a
chosen one.

## Tools
| tool | purpose |
|------|---------|
| `list_collections()` | the notebooks you can search, with descriptions |
| `search(query, collection, k=5)` | top-k passages from a collection |
| `get_book_image(collection, source, section, text, page, dpi)` | render the page a passage is on, returned as an image for a multimodal model |
| `ingest_path(collection, path="", recursive=True)` | incremental (re)ingest (OCR prepass + embed) |
| `status()` | warm/cold, per-collection counts, device |

`collection` may be omitted only when a single collection exists.

## OCR on the 7700S (Surya-2 via Vulkan llama-server)
PDF recognition runs through Surya 0.20's **llamacpp backend**, which spawns a
Vulkan `llama-server` that does the VLM compute on the dGPU (~3 GB VRAM,
~1.7 s/page). The binary is auto-located (LM Studio's bundled Vulkan build) or
set via `RAG_LLAMA_BIN`. If the GPU spawn/inference fails it falls back to CPU
(`LLAMA_CPP_NGL=0`). OCR results are **cached by file hash** at
`index/<col>/ocr_cache/<sha1>.json`, so re-ingests and the watcher never re-OCR.

## Auto-ingest: drop a file in a folder, it gets indexed
The **`rag-ingest.service`** daemon (`watcher.py`) watches **every** collection's
folder with inotify and, on any add/change/delete, calls the server's
`ingest_path` for that collection:
- only **new or content-changed** files (by SHA-1, plus an ingest-version guard)
  are (re-)embedded; deleted files are dropped
- **debounced** + **catches up on startup** (reconciles every collection on boot)
- torch-free and tiny; the GPU model stays owned solely by the server

```bash
systemctl --user status rag-ingest.service
journalctl --user -u rag-ingest.service -f
```

## Ingest from the CLI
```bash
source ~/venv/bin/activate
python ingest.py "~/Documents/STATF401/Materials" --index index/STATF401 --reset
```
PDFs are OCR'd (prepass) then embedded; text/epub chunk at `--chunk 1200`.

## Runs as a persistent systemd daemon (survives reboots)
The server runs as a **user systemd service** over **streamable-HTTP** on
`127.0.0.1:8077`.
```bash
systemctl --user status rag-mcp.service
journalctl --user -u rag-mcp.service -f
```
- `~/.config/systemd/user/rag-mcp.service` sets `RAG_TRANSPORT=streamable-http`
- `WantedBy=default.target` + `loginctl enable-linger gabriel` → starts on boot
- Registered with Claude Code: `claude mcp add --scope user --transport http rag http://127.0.0.1:8077/mcp`

> The same `server.py` still runs as plain stdio if launched without
> `RAG_TRANSPORT`.

## Layout
| file | role |
|------|------|
| `server.py` | MCP server; tools + lazy load + idle watchdog + two-phase ingest |
| `embedder_worker.py` | subprocess that loads the embedder on the 7700S (only torch file) |
| `qwen3_vl_embedding.py` | vendored official Qwen3-VL embedder (correct weight loading) |
| `worker_client.py` | spawns/kills the embedder worker, JSON-lines protocol |
| `ocr_worker.py` | subprocess running Surya-2 OCR via the Vulkan llama-server |
| `ocr_client.py` | spawns/kills the OCR worker, GPU→CPU fallback, orphan reaper |
| `ocr_cache.py` | OCR results cached by file hash |
| `index_store.py` | flat vector store: `embeddings.npy` (float16) + `meta.jsonl` |
| `ingest.py` | CLI + reusable `ingest_file`/`ocr_pages_for`/`file_signature` |
| `rag_collections.py` | collection config loader |
| `watcher.py` | event-driven multi-folder ingester daemon |
| `index/<name>/` | per-collection index (vectors + meta + manifest + ocr_cache) |

## Env knobs
`RAG_IDLE` (idle seconds, default 900) · `RAG_INDEX_ROOT` (index parent dir) ·
`RAG_COLLECTIONS` (config path) · `RAG_MODEL` (HF id) · `RAG_MAXLEN` ·
`RAG_LLAMA_BIN` (Vulkan llama-server) · `RAG_OCR_NGL` (GPU layers, default 99).
