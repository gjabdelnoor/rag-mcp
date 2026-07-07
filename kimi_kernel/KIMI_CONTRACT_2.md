# KIMI_CONTRACT_2 — Multi-GPU data-parallel embedding

Read `TECH_CONTEXT.md` first. The two inviolable laws apply. This builds ON TOP of
Contract 1's `video_embed.py` (Claude reviewed it — CONTRACT 1 PASSED).

## AMENDMENT (Claude, loop tick 1) — READ THIS, IT OVERRIDES CONFLICTING CLAUSES BELOW
Reality checks since this contract was written:
- **The 780M is UNUSABLE for this model.** ROCm exposes only ~512 MB total on GPU[1]
  (nearly all already used). A 2B fp16 model (~4 GB) cannot live there. So the
  "shard across 7700S + 780M" idea is DEAD — do NOT pursue 780M data-parallel.
- **GPU[0] (7700S) is currently FREE** (~8.5 GB free; the live embedder + 35B unloaded).
  Contract 1's GPU benchmark was DEFERRED for VRAM — now you can actually measure it.

Revised task, in THIS order:
1. **Measure Contract 1 for real on the 7700S.** Run `video_embed.py` on the real
   `~/Documents/Personal/*.frames/` corpus on `cuda:0`: report frames/sec, per-window
   latency, and confirm the video-window vectors match the CPU selftest (cosine ≈ 1.0
   CPU-vs-GPU on a few windows). Validate or correct the 4.78× theoretical figure with
   a MEASURED number. Also sweep window size `W ∈ {8,16,32}` for the speed/granularity
   tradeoff (Claude will pick the default).
2. **Maximize SINGLE-GPU (7700S) throughput** → deliver `parallel_embedder.py` as a
   throughput-optimized pipeline: overlap CPU-side vision preprocessing with GPU compute
   (double-buffered producer/consumer), and/or batch multiple windows per forward, and/or
   concurrent CUDA streams — whatever measurably raises frames/sec on ONE 7700S. Document
   (but do not need to run) an N-process data-parallel design for a future 2nd real GPU.
3. **VRAM GUARD STILL APPLIES and is stricter now:** before loading the model on cuda:0,
   require >5 GB free AND `pgrep embedder_worker` empty AND the user's live `yt-dlpcc`
   ingest not actively embedding. If any fails, DEFER (design + document) — never collide
   with a reclaiming job, never kill anything.
Correctness gate (Clause 3 below) is unchanged and binding.

## Clause 1 — Sole authorized task
Build `kimi_kernel/parallel_embedder.py`: a data-parallel embedding pool that shards
frame-windows across BOTH GPUs — one worker subprocess pinned to GPU[0] (7700S) and one
to GPU[1] (780M) — and merges results in original order. Each worker is the existing
`embedder_worker.py`-style stdin/stdout JSON process, launched with the right
`HIP_VISIBLE_DEVICES` (and `HSA_OVERRIDE_GFX_VERSION=11.0.0` for the 780M).

## Clause 2 — Design requirements
- A dispatcher hands each worker a batch, collects `(index, vector)`, reassembles in
  order. Workers pull from a shared queue so the faster GPU (7700S) naturally does more
  (dynamic load balance, NOT a fixed 50/50 split).
- Degrade gracefully: if only one GPU has ≥5 GB free (VRAM guard, TECH_CONTEXT clause 4),
  run single-GPU and say so — never OOM a busy GPU, never kill anything to make room.
- Drop-in shape: expose `embed_windows(items) -> np.ndarray` matching the vector layout
  `worker_client.Embedder.embed_image` produces, so it can later replace that call.

## Clause 3 — Fidelity & correctness
No spatial downscale (TECH_CONTEXT law 1). Correctness: sharded output MUST be
bit-for-bit / cosine-≈1.0 identical (per vector) to the single-GPU output for the same
frames — include a `--verify` mode that asserts this on a small sample.

## Clause 4 — Benchmark
frames/sec: single-GPU (7700S) vs single-GPU (780M) vs dual-GPU pool. Report the
speedup and the realized load split. Respect the VRAM guard; if GPU[0] is live-busy,
bench 780M-only + document the dual-GPU design and defer the dual timing.

## Clause 5 — Prohibitions / timeout / deliverables
No editing parent production files. No service/server kills. ≤25 min, then STOP and
write `kimi_kernel/REPORT_2.md` (timestamped): frames/sec table, load split, correctness
result, VRAM-guard behavior observed, open questions. Deliverables:
`kimi_kernel/parallel_embedder.py` + `kimi_kernel/REPORT_2.md`. Do not start Contract 3.
