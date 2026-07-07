# KIMI_CONTRACT_1 — Native video-path frame embedding (HEADLINE)

Read `TECH_CONTEXT.md` in this folder first. The two inviolable laws apply.

## Clause 1 — Sole authorized task
Build a prototype, `kimi_kernel/video_embed.py`, that embeds a `<slug>.frames/`
directory by feeding the frame sequence through Qwen3-VL's **native video path**
(`format_model_input(video=[frame_paths...])` → `pixel_values_videos`,
`video_grid_thw`, temporal patch-merge) INSTEAD of one independent still image per
frame — WITHOUT lowering spatial resolution of any frame.

## Clause 2 — Coverage requirement (no silent frame-dropping)
The native path samples down to `MAX_FRAMES=64` per call by default. You MUST cover
the ENTIRE timeline: process the frames in consecutive **windows** of `W` frames
(default `W=32`, tunable via `--window`) so every frame is represented; produce ONE
embedding vector per window, tagged with the window's start/end timestamp and the
union of the windows' closed captions. No frame in the middle of the video may be
skipped. Emit metadata compatible with `ingest.py` (source=framedir, modality
"video-window", t_start, t_end, text=cc, image=representative middle frame path).

## Clause 3 — Fidelity proof (mandatory)
Do NOT reduce `FRAME_MAX_PIXELS`, `max_pixels`, or `min_pixels` below current values.
In the report, state the exact pixels/frame fed to the model and confirm it equals the
still-image path's spatial resolution. A speed win via spatial downscale = FAIL.

## Clause 4 — Correctness gate
Prove retrieval still works: embed a text query (e.g. a phrase from one window's
captions) with `embed_text` and show the matching window is the top hit over a handful
of windows (cosine). Include this micro-test in the script (`--selftest`).

## Clause 5 — Benchmark
Measure frames-of-footage embedded per second AND vectors produced, video path vs the
current per-image path, on the real `~/Documents/Personal/*.frames/` data. Respect the
VRAM guard in TECH_CONTEXT clause 4 — if GPU[0] is busy, bench on GPU[1] (780M, with
`HSA_OVERRIDE_GFX_VERSION=11.0.0`) or write "DEFERRED — GPU busy" with a token-count
analysis instead (tokens/window vs tokens/frame is a valid theoretical speedup proxy).

## Clause 6 — Explicit prohibitions
No editing parent-dir production files. No killing/restarting any service or server
(see TECH_CONTEXT clause 4). No spatial downscaling. No `pip install` unless a named
import is genuinely missing, and then only into `/home/gabriel/venv`.

## Clause 7 — Timeout & termination
Spend at most ~25 minutes of wall effort before writing the report. Then STOP and write
`kimi_kernel/REPORT_1.md` (timestamped). Do not start Contract 2 — Claude queues it.

## Clause 8 — Exact deliverables
1. `kimi_kernel/video_embed.py` (runnable via `/home/gabriel/venv/bin/python`,
   flags `--frames <dir> --window N --selftest --device cuda:0|cuda:1`).
2. `kimi_kernel/REPORT_1.md` per TECH_CONTEXT clause 5 (numbers or DEFERRED, fidelity
   statement, tokens/window vs tokens/frame, open questions for Claude).
