# rag-mcp first-time setup

rag-mcp runs four embedding tiers — `tiny`, `small`, `medium`, `large` — and the
right one depends on your hardware. `python setup.py` runs a one-time
diagnostic, recommends a tier, and tells you exactly what to start.

## Quick start

```bash
python setup.py                # probe + interactive picker + write setup.json
python setup.py --json         # probe-only, no writes (machine-readable)
python setup.py --preset tiny  # force a tier without the picker
python setup.py --preset large --yes   # bypass the `large` warning
```

After setup writes `setup.json`, follow the printed `Next steps` to start the
embedder (or just `python server.py` if you picked `medium`).

## The tiers

| preset | model | dim | needs | supports images |
|---|---|---|---|---|
| `tiny`  | sentence-transformers/all-MiniLM-L6-v2 (22M) | 384  | any CPU or GPU           | no  |
| `small` | nomic-embed-text-v1.5 (137M, GGUF)            | 768  | ~1 TFLOPs FP32, 4 GB VRAM | no  |
| `medium`| Qwen3-VL-Embedding-2B (2B)                   | 2048 | ~10 TFLOPs FP32, 6 GB     | yes |
| `large` | Qwen3-VL-Embedding-8B (8B, 8-bit)            | 4096 | ~13 TFLOPs FP32, 12 GB    | no  |

The "needs" column is the floor; the picker takes the highest tier the host
satisfies. `large` additionally prints a warning — its quality gain over
`medium` is **marginal** for typical knowledge bases (textbooks, papers,
notes) and only meaningful on very complex / domain-specific corpora.

## What the diagnostic does

`device_probe.probe()` is torch-free:

1. `nvidia-smi --query-gpu=name,memory.total` — NVIDIA GPUs.
2. `rocm-smi --showproductname --showmeminfo vram --json` — AMD GPUs.
3. `sysctl` + `system_profiler` — Apple Silicon (best-effort).
4. `lspci -mm` — last-resort fallback (no TFLOPs available).
5. A 2-second numpy matmul (1024³ FP32) for the CPU benchmark.

Each GPU is matched against a static device→TFLOPs table
(`device_probe.NVIDIA_TFLOPS_FP32`, `AMD_TFLOPS_FP32`, `APPLE_TFLOPS_FP32`).
Unknown devices get `tflops_fp32=None` and the picker conservatively drops
them to `small`.

The probe takes ~2 seconds total (the CPU benchmark dominates). On a headless
laptop with no GPU, you get a CPU TFLOPS estimate and the picker lands on
`tiny`.

## Per-tier startup

After picking, the onboarding prints the right command:

| preset | start command | notes |
|---|---|---|
| `tiny`  | `python tiny_embedder_client.py start` | sentence-transformers; CPU-friendly; text only |
| `small` | `python text_embedder_client.py start` | nomic GGUF on the iGPU; text only |
| `medium`| `python server.py`                    | the existing MCP server; text + images |
| `large` | `python large_embedder_client.py start` | VL-8B 8-bit on the dGPU; needs `pip install bitsandbytes`; text only |

Each tier also has a `shell` REPL for one-off embeddings:

```bash
python tiny_embedder_client.py shell
> the cat sat on the mat
  norm=1.0000  head=[-0.0234, 0.0156, ...]
```

## Switching presets (destructive)

Different tiers use different vector dims — different vector spaces. The
onboarding never silently switches. When you re-run `setup.py` with a
different `--preset`:

1. It writes the new `setup.json`.
2. It prints the `ingest.py --reset` commands you need to run.
3. Pass `--reset-after` to do the reset automatically (it shells into
   `ingest.py --reset` for every collection in `collections.json`).

To downgrade later: `python setup.py --preset medium --force` then
`python ingest.py --reset` for each collection.

## Override flags

| flag | effect |
|---|---|
| `--preset tiny|small|medium|large` | skip the picker |
| `--force` | overwrite an existing `setup.json` |
| `--yes` | skip the `large` warning confirmation |
| `--reset-after` | after switching, run `ingest.py --reset` for every collection (DESTRUCTIVE) |
| `--no-cpu-bench` | skip the numpy matmul (faster, less accurate) |
| `--bench-budget <s>` | change the CPU-bench budget (default 2 s) |
| `--json` | machine-readable output, no writes |

## Files involved

| file | role |
|---|---|
| `setup.py` | the onboarding CLI |
| `device_probe.py` | torch-free probe + TFLOPs table + picker (also runnable as `python device_probe.py`) |
| `setup.json` | gitignored; written by `setup.py`, schema-versioned |
| `setup.json.example` | skeleton for reference |
| `tiny_embedder_worker.py` / `tiny_embedder_client.py` | sentence-transformers backend |
| `large_embedder_worker.py` / `large_embedder_client.py` | VL-8B 8-bit backend (requires `bitsandbytes`) |
| `text_embedder_client.py` (existing) | nomic GGUF backend for `small` |
| `server.py` (existing) | the `medium`-tier MCP server |
| `SETUP.md` | this file |

## Re-running the diagnostic

```bash
python device_probe.py        # human-readable
python device_probe.py --json # machine-readable
python setup.py probe --json  # same data via the setup subcommand
```

Add new devices to `device_probe.py`'s static tables (NVIDIA_TFLOPS_FP32,
AMD_TFLOPS_FP32, APPLE_TFLOPS_FP32) when you encounter one we don't yet know.
