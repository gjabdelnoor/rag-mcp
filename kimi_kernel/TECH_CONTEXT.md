# TECH_CONTEXT — shared facts for all KIMI_CONTRACT_*.md in this folder

You (Kimi) are the coder. Claude is the reviewer/architect and will read your reports
every 30 minutes, review your code, repair breakage, and queue the next goal. Stay
strictly inside the authorized-task clause of whichever contract points here. Do not
freelance beyond it.

## 0. THE TWO INVIOLABLE LAWS (from the user, verbatim intent)

1. **DO NOT SACRIFICE FIDELITY.** Never reduce the *spatial* resolution of a frame to
   go faster. `max_pixels` / `min_pixels` for the still-image path must NOT be lowered
   below the current effective resolution. Speed must come from *temporal* token
   merging (the native video path), multi-GPU parallelism, and better kernels —
   NEVER from throwing away pixels. Any prototype that wins speed by downscaling
   frames is an automatic FAIL of the contract.
2. **The native video path is the headline optimization.** Qwen3-VL can ingest a frame
   sequence as a *video* (`pixel_values_videos`, `video_grid_thw`) with a temporal
   patch-merge (temporal patch size 2) so N frames cost far fewer than N× the tokens.
   This is throughput WITHOUT fidelity loss. Prefer it.

## 1. What we are optimizing

The RAG knowledge base embeds 1-fps video frames (a "video journal"). A ~45-min video
= ~2,700 frames. Current per-image embedding runs at ~16 frames/min (~3.75 s/frame) —
~2.8 hours per video. Unacceptable. Goal: cut wall-time by >5× with ZERO fidelity loss.

## 2. The model & code (DO NOT EDIT THESE IN PLACE)

- Model: `Qwen/Qwen3-VL-Embedding-2B` (multimodal, 2048-dim, last-token pooled, L2-norm).
- Embedder class: `/home/gabriel/projects/rag-mcp/qwen3_vl_embedding.py`
  (`Qwen3VLEmbedder.process(items)` where each item is `{"text":..,"image":..,"video":..}`).
  It ALREADY has a video path: `format_model_input(video=[frame paths])` →
  `process_vision_info` → `pixel_values_videos`. Constants: `FRAME_MAX_PIXELS=768*32*32`,
  `MAX_FRAMES=64`, `MAX_TOTAL_PIXELS=10*FRAME_MAX_PIXELS`, temporal merge via processor.
- Worker (only process that imports torch): `/home/gabriel/projects/rag-mcp/embedder_worker.py`
  (stdin/stdout JSON; ops `embed_text`, `embed_image`). Client:
  `/home/gabriel/projects/rag-mcp/worker_client.py`. Ingest: `ingest.py::ingest_frame_dir`.
- Frame units on disk: `<collection>/<slug>.frames/` with `frames.json`
  (`{video,title,fps,frames:[{file,t,cc}]}`). One live example is under
  `/home/gabriel/Documents/Personal/*.frames/` (~2,692 real 720p jpgs) — USE IT as bench data.

**Work in THIS folder** (`/home/gabriel/projects/rag-mcp/kimi_kernel/`). Import the
parent modules; do not modify `embedder_worker.py` / `qwen3_vl_embedding.py` /
`server.py` / `ingest.py` until Claude reviews and promotes your prototype.

## 3. Hardware & environment

- GPU[0] = AMD Radeon RX 7700S, gfx1102, 8 GB VRAM (ROCm `cuda:0`). Fast.
- GPU[1] = AMD Radeon 780M iGPU, gfx1103, shared system RAM (GTT). Slower but usable.
  **The 780M needs `HSA_OVERRIDE_GFX_VERSION=11.0.0`** in the subprocess env or ROCm
  GEMM fails (NOT_SUPPORTED). Select a GPU with `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`.
- Python: **`/home/gabriel/venv/bin/python`** (has torch-ROCm, transformers, qwen_vl_utils,
  pytorch-triton-rocm). pip installs, if truly needed, go into THIS venv only.

## 4. HARD PROHIBITIONS (violating any = contract FAIL, and may corrupt a LIVE run)

A real user embedding job is RUNNING RIGHT NOW on GPU[0] (7700S at ~98% VRAM: the live
embedder + a sleeping 35B llama-server). Therefore:

- **NEVER** kill, restart, or `systemctl` the following: `rag-mcp.service`,
  `rag-ingest.service`, any `llama-server` (the 35B, and it may be sleeping), any
  `whisper-server`, or the live `yt-dlpcc` process. Leave them ALL running.
- **NEVER** run a GPU benchmark that loads the 2B model on `cuda:0` while VRAM is tight.
  Before loading the model on ANY device, check free VRAM (`rocm-smi --showmeminfo vram`
  or `torch.cuda.mem_get_info()`); if free < 5 GB on the target device, DO NOT load —
  instead do CPU-correctness checks + synthetic micro-benchmarks and write
  "GPU timing DEFERRED — device busy" in your report. Do not force it, do not wait-loop
  forever, do not try to free VRAM by killing anything.
- Prefer benchmarking on GPU[1] (780M) when GPU[0] is busy, if it has ≥5 GB free.
- Do not touch `~/Documents/Personal/` except to READ frames as bench input.

## 5. Deliverable rules (every contract)

- Final deliverable is a timestamped markdown report at the exact path the contract
  names (e.g. `REPORT_1.md`). Include: what you built, how to run it, measured numbers
  (frames/sec, tokens/frame, per-vector cost) or an explicit "DEFERRED — GPU busy" note,
  a fidelity statement (proving you did NOT downscale), and open questions for Claude.
- Keep code self-contained and runnable via `/home/gabriel/venv/bin/python <script>`.
- Correctness gate: any new embedding path MUST produce vectors whose cosine similarity
  to the reference per-image embedding is sane (identical frame → same-ish vector);
  include a tiny correctness test proving retrieval still works.
