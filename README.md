# rag-mcp — local multimodal RAG MCP server

A **knowledge base** over your study materials, exposed to a local LLM (e.g.
Gemma, Qwen3) as an MCP server. Material is organized into named
**collections** ("notebooks"), each a watched folder with its own vector
index and a human description. The server is thin — `mcp + numpy only, no
torch` at idle — and runs the heavy models in cold-spawned child
subprocesses that are killed after 15 min idle.

Three embedding models, one shared lock — see `server.py`'s policy note for
why:

| embedder | model / device | role |
|----------|----------------|------|
| `text_emb` | `nomic-embed-text-v1.5` on the **780M iGPU** (Vulkan0) | search only — single-vector, latency-sensitive |
| `ingest_emb` | `nomic-embed-text-v1.5` on the **7700S dGPU** (Vulkan1, `xlarge` preset) | bulk text/pdf/epub ingest — batched |
| `emb` | `Qwen3-VL-Embedding-2B` on the **7700S dGPU** | raster-image ingest — shares the 7700S with `ingest_emb`, never both warm at once |

## Modalities

| modality | how it's indexed |
|----------|------------------|
| `.txt` / `.md` | embedded as text (sliding-window chunks) |
| `.epub` | embedded as text, **one chunk per chapter-section** |
| `.pdf` (short, ≤ `RAG_MAX_OCR_PAGES`=20 by default) | **OCR'd with Surya-2 on the 7700S**, then embedded one chunk per page; the page image is rendered on demand by `get_book_image` so figures are still visible |
| `.pdf` (long) | embedded text layer extracted with PyMuPDF (no GPU); same chunking as above |
| `.jpg` / `.jpeg` / `.png` / `.webp` | embedded as images via the VL-2B embedder, one vector per file |

Text chunks live in `<collection>/text/`, image vectors in `<collection>/image/`.

## Idle profile (the whole point)
- The MCP server process is thin: **mcp + numpy only, no torch** at idle.
- All embedders live in child subprocesses, **cold-spawned on first use** and
  **killed after `RAG_IDLE` seconds** (default 900 = 15 min). That returns the
  GPUs to 0% and frees VRAM/RAM.
- OCR runs in a **separate short-lived subprocess** only during ingest. The
  OCR llama-server and the 7700S embedders **never** overlap: ingest runs OCR
  as a *prepass* (7700S embedders stopped), then embeds text on the 7700S
  (OCR server killed), then embeds images on the 7700S (text embedder killed).
  On an 8 GB card neither alone is a problem; together they'd OOM.
- At idle: **0% GPU, ~0% CPU, no model resident**.

## Collections (notebooks)
Defined in `collections.json` (a default is written on first run; see
`collections.example.json`). Each entry is a folder + a description:

```json
{
  "STATF401": {"folder": "~/Documents/STATF401/Materials", "description": "…"},
  "workbook": {"folder": "~/Documents/Gabe's Workbook",   "description": "…"}
}
```
`collections.json` is gitignored — every install writes its own. Each
collection's index lives under `index/<name>/{text,image}/`. The LLM
discovers them with **`list_collections`** (returns name + description +
counts) and then searches a chosen one.

## Tools
| tool | purpose |
|------|---------|
| `list_collections()` | the notebooks you can search, with descriptions |
| `search(query, collection, k=5)` | top-k passages from a collection (text-only for now) |
| `get_book_image(collection, source, section, text, page, dpi)` | render the page a passage is on, returned as an image for a multimodal model |
| `screenshot_video(collection, video, timestamps)` | grab captioned stills from a saved video on disk (up to 10 per call, pure ffmpeg) |
| `ingest_path(collection, path="", recursive=True)` | incremental (re)ingest (OCR prepass + dual-store embed) |
| `status()` | warm/cold, per-collection counts, devices |

`collection` may be omitted only when a single collection exists.

## OCR on the 7700S (Surya-2 via Vulkan llama-server)
PDF recognition runs through Surya 0.20's **llamacpp backend**, which spawns
a Vulkan `llama-server` that does the VLM compute on the dGPU (~3 GB VRAM,
~1.7 s/page). The binary is auto-located (LM Studio's bundled Vulkan build)
or set via `RAG_LLAMA_BIN`. If the GPU spawn/inference fails it falls back
to CPU (`LLAMA_CPP_NGL=0`). OCR results are **cached by file hash** at
`index/<col>/ocr_cache/<sha1>.json`, so re-ingests and the watcher never
re-OCR. Long PDFs (`> RAG_MAX_OCR_PAGES`) skip OCR entirely and use their
embedded text layer.

A hosted OpenAI-compatible OCR endpoint can be preferred over local with
`RAG_OCR_REMOTE_URL` + `MINIMAX_API_KEY`. The remote path is tried first so
it doesn't contend with the user's other GPU workloads; local GPU/CPU is the
fallback. Remote failures rate-limit back off exponentially with jitter.

## Auto-ingest: drop a file in a folder, it gets indexed
The **`rag-ingest.service`** daemon (`watcher.py`) watches **every** collection's
folder with inotify and, on any add/change/delete, calls the server's
`ingest_path` for that collection:
- only **new or content-changed** files (by SHA-1, plus an ingest-version guard)
  are (re-)embedded; deleted files are dropped
- **debounced** + **catches up on startup** (reconciles every collection on boot)
- torch-free and tiny; the GPU models stay owned solely by the server

```bash
systemctl --user status rag-ingest.service
journalctl --user -u rag-ingest.service -f
```

## Ingest from the CLI
```bash
python ingest.py "~/Documents/STATF401/Materials" --index index/STATF401 --reset
```
Short PDFs are OCR'd (prepass) then embedded; long PDFs use the text layer;
text/epub chunk at `--chunk 1200`; images go through the VL-2B embedder.

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
| `server.py` | MCP server; tools + lazy load + idle watchdog + dual-phase ingest |
| `embedder_worker.py` | subprocess that loads the VL-2B on the 7700S (only torch file) |
| `qwen3_vl_embedding.py` | vendored official Qwen3-VL embedder (correct weight loading) |
| `worker_client.py` | spawns/kills the VL-2B worker, JSON-lines protocol |
| `text_embedder_client.py` | thin client for the small `nomic-embed` llama-server |
| `dgpu_embed_ctl.py` | CLI to start/stop/bench the 7700S `nomic` embedder; presets |
| `ocr_worker.py` | subprocess running Surya-2 OCR via the Vulkan llama-server |
| `ocr_client.py` | spawns/kills the OCR worker, remote→GPU→CPU fallback, orphan reaper |
| `ocr_cache.py` | OCR results cached by file hash |
| `index_store.py` | flat vector store: `embeddings.npy` (float16) + `meta.jsonl`, flock-locked |
| `ingest.py` | CLI + reusable `ingest_file`/`ocr_pages_for`/`file_signature` |
| `rag_collections.py` | collection config loader |
| `watcher.py` | event-driven multi-folder ingester daemon |
| `kimi_contracts/` | internal AI-contract docs for the dual-store refactor (kept for transparency; not user-facing) |
| `index/<name>/{text,image}/` | per-collection dual-store index (vectors + meta + manifest + ocr_cache) |

## Env knobs
`RAG_IDLE` (idle seconds, default 900) ·
`RAG_INDEX_ROOT` (index parent dir) ·
`RAG_COLLECTIONS` (config path) ·
`RAG_MODEL` (VL-2B HF id) ·
`RAG_MAXLEN` (VL-2B max length, default 8192) ·
`RAG_LLAMA_BIN` (Vulkan llama-server path; auto-located if unset) ·
`RAG_OCR_NGL` (OCR GPU layers, default 99) ·
`RAG_MAX_OCR_PAGES` (skip OCR for longer PDFs, default 20) ·
`RAG_OCR_REMOTE_URL` + `MINIMAX_API_KEY` + `RAG_OCR_REMOTE_MODEL` (hosted OCR) ·
`RAG_TEXT_EMBED_*` (text-embedder model/port/ctx/pooling/env overrides) ·
`RAG_BENCH_SAMPLE_DIR` (PDF dir for `dgpu_embed_ctl bench*`).

## License
MIT — see `LICENSE`.
