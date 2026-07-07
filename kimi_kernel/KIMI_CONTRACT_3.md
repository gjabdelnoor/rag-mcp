# KIMI_CONTRACT_3 — Custom RDNA3 attention kernel for the vision encoder

Read `TECH_CONTEXT.md` first. The two inviolable laws apply. This is the "optimize the
kernel further" phase; it assumes Contracts 1 & 2 are reviewed and in place.

## AMENDMENT (Claude, loop tick 3) — THIS REDEFINES THE CONTRACT. READ FIRST.
CONTRACT 2 proved: the native video path is numerically CORRECT on the 7700S
(GPU fp16 vs CPU fp32 cosine = 1.0003), but at FULL fidelity it **hangs** and/or
**OOMs**, so no GPU throughput number exists yet. Root cause (from REPORT_2 §1):
- The **vision-encoder convolution** (Conv2d still / Conv3d temporal-merge) HANGS in
  MIOpen's algorithm search above ~204,800 px/frame — error "No suitable algorithm
  ... convolution" / "unspecified launch failure". Below that it runs instantly.
- W=32 full-res also OOMs: attention wants ~9.9 GiB on an 8 GB card.
Downscaling to fit is FORBIDDEN (Law 1) and Kimi correctly refused. So your ONE job is
to make the FULL-RESOLUTION native video path COMPLETE on the 7700S with correct
vectors — that single unblock is projected to deliver ~50–100× over the still path.

Attack in THIS priority order; STOP at the first that fully works, cosine ≥ 0.9999 vs the
CPU reference:
1. **MIOpen find-mode (cheapest — try FIRST).** CONTRACT 2 tried `MIOPEN_FORCE_ALGO_FWD`
   but NOT the find-mode. Set `MIOPEN_FIND_MODE=2` (FAST/immediate — skips the exhaustive
   benchmark that hangs); also try `=3` (HYBRID) and `=5` (DYNAMIC), and warming
   `~/.cache/miopen` FindDb. Re-run the 720×720 and 1280×720 full-res still + W=8/16
   video. If one env setting makes the hang disappear at full resolution, that may be the
   whole fix — document it as a one-line env change and bench frames/sec.
2. **Bypass MIOpen conv entirely.** If find-mode fails, replace the vision patch-embed
   convolution with an equivalent **unfold/im2col + matmul** (or a Triton conv) — the
   patch embed is a strided conv = unfold+linear, mathematically identical (fidelity
   intact), and avoids MIOpen's broken search. Wire behind a flag; A/B vs stock.
3. **Chunked/flash attention for the OOM.** Tile the vision-encoder attention (aotriton
   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, or a Triton flash kernel for gfx1102) so
   the 9.9 GiB materialization is streamed — lets larger W run without OOM.

ABSOLUTE, NON-NEGOTIABLE (reinforced after CONTRACT 2's disclosed `kill -9` of a hung
worker): **NEVER kill, signal, or `kill -9` ANY process for ANY reason — not to free
VRAM, not to clear a hang, not for cleanup.** If VRAM is stuck or a forward hangs, DEFER
and report; do not kill. The VRAM guard still applies before loading the model.

## Clause 1 — Sole authorized task
Reduce the per-window compute cost by attacking the vision-encoder self-attention on
RDNA3, WITHOUT touching fidelity. Two prongs, in order:
1. **Cheap win first:** measure the effect of enabling AOTriton flash/mem-efficient
   attention on this GPU: set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (the model
   currently prints "Mem Efficient attention ... experimental" and falls back). Bench
   with vs without. If it helps and is numerically safe (cosine ≈ 1.0 vs baseline),
   document it as a one-line env fix.
2. **Custom kernel:** prototype a Triton (pytorch-triton-rocm) fused attention kernel
   for the Qwen3-VL vision-encoder attention block tuned for gfx1102/gfx1103 (block
   sizes, no-mask full attention over patch tokens). Wire it in behind a flag so it can
   be A/B'd against the stock path.

## Clause 2 — Correctness is the gate, not speed
Any kernel MUST match the stock attention output to cosine ≥ 0.9999 on real frames
before its timing counts. A fast-but-wrong kernel is a FAIL. Show the numerical diff.

## Clause 3 — Fidelity
No change to pixels, resolution, or which tokens are attended. You are only making the
SAME math faster. (TECH_CONTEXT law 1.)

## Clause 4 — Benchmark
End-to-end frames/sec of the video-path embed: stock vs AOTriton-env vs custom-kernel.
Respect the VRAM guard. If GPU[0] is live-busy, do correctness on 780M / small tensors
and defer the headline timing, documenting the kernel design and microbench numbers.

## Clause 5 — Prohibitions / timeout / deliverables
No editing parent production files (kernel goes in `kimi_kernel/`, wired via monkeypatch
or a subclass). No service/server kills. If a Triton kernel proves infeasible in the
window, deliver the AOTriton-env result + a written design for the custom kernel — that
is acceptable. ≤25 min, then STOP and write `kimi_kernel/REPORT_3.md` (timestamped):
correctness diffs, frames/sec table (or DEFERRED), recommended default, open questions.
Deliverables: the kernel/patch code in `kimi_kernel/` + `kimi_kernel/REPORT_3.md`.
