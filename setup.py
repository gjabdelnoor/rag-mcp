#!/usr/bin/env python3
"""rag-mcp first-time setup / onboarding.

Probes the host, picks an embedding preset, writes `setup.json`, and prints the
right command(s) to start a server at that preset. Does NOT touch `server.py` —
the existing server is the medium preset (small text + medium images) and stays
as-is. Tiny/small/large presets get standalone embedder CLIs that the
onboarding points you at.

Usage:
  python setup.py                   # interactive: probe, recommend, confirm
  python setup.py --json            # machine-readable probe output (no writes)
  python setup.py --preset tiny     # force a preset (skips interactive picker)
  python setup.py --preset large --yes   # bypass the `large` warning
  python setup.py --reset-after     # after switching presets, run ingest.py --reset
                                    # on every collection (destructive)

The probe is torch-free (nvidia-smi / rocm-smi / lspci + numpy CPU benchmark)
so the no-torch-at-idle invariant of the running server is preserved.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import textwrap

import device_probe as dp


SETUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.json")
SCHEMA = 1


def _load_setup() -> dict | None:
    if not os.path.isfile(SETUP_PATH):
        return None
    try:
        return json.load(open(SETUP_PATH))
    except Exception:                                # noqa: BLE001
        return None


def _save_setup(payload: dict) -> None:
    tmp = SETUP_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, SETUP_PATH)


def _humanize_gpu(detected: dict) -> str:
    gpus = detected.get("gpus") or []
    if not gpus:
        return "GPU: none detected"
    lines = []
    for g in gpus:
        lines.append(
            f"GPU: {g['name']} ({g['vendor']}) — {g['vram_gb']} GB VRAM, "
            f"{g.get('tflops_fp32') or 'unknown'} TFLOPS FP32"
        )
    return "\n".join(lines)


def _print_hardware(detected: dict) -> None:
    cpu = detected.get("cpu_tflops_fp32")
    print("Detected hardware:")
    print(f"  CPU:    ~{cpu} TFLOPS FP32 (numpy matmul, 2 s budget)")
    print(_humanize_gpu(detected))


def _format_preset_line(name: str, recommended: bool) -> str:
    p = dp.PRESETS[name]
    marker = "  ← recommended" if recommended else ""
    warn = "  ⚠ requires confirmation" if name == "large" else ""
    return (f"  [{name[0].upper()}] {name:<7} {p['summary']}{warn}{marker}")


def _interactive_pick(detected: dict) -> str:
    rec = dp.pick(detected)
    alts = dp.alternatives(detected)
    print()
    print("Choices:")
    for name in ("tiny", "small", "medium", "large"):
        print(_format_preset_line(name, recommended=(name == rec)))
    print()
    default_idx = {"tiny": 1, "small": 2, "medium": 3, "large": 4}[rec]
    while True:
        raw = input(f"Pick [1-4] (default {default_idx}): ").strip()
        if not raw:
            return rec
        if raw in ("1", "2", "3", "4"):
            chosen = ("tiny", "small", "medium", "large")[int(raw) - 1]
            if chosen not in alts and alts:
                print(f"⚠  {chosen!r} isn't supported by this hardware "
                      f"(allowed: {', '.join(alts)}). Pick again.")
                continue
            return chosen
        if raw in dp.VALID_PRESETS:
            if raw not in alts and alts:
                print(f"⚠  {raw!r} isn't supported by this hardware "
                      f"(allowed: {', '.join(alts)}). Pick again.")
                continue
            return raw
        print("  please enter 1, 2, 3, or 4.")


def _confirm_large(args) -> bool:
    if args.yes:
        return True
    print()
    print(textwrap.fill("⚠  " + dp.PRESETS["large"]["warning"], 78))
    print()
    while True:
        raw = input("Type 'yes' to confirm `large`, anything else to abort: ").strip().lower()
        if raw == "yes":
            return True
        return False


def _print_run_instructions(preset: str) -> None:
    """Tell the user exactly what to run for this preset."""
    print()
    print("Next steps:")
    if preset == "tiny":
        print("  The `tiny` preset uses sentence-transformers (CPU-friendly, text-only).")
        print("  Start the embedder:")
        print("    python tiny_embedder_client.py start")
        print("  Embed one-off text:")
        print('    python tiny_embedder_client.py embed "your text here"')
        print("  Drop the running tiny server:")
        print("    python tiny_embedder_client.py stop")
    elif preset == "small":
        print("  The `small` preset uses nomic-embed-text-v1.5 (GGUF, text-only).")
        print("  Start the embedder on the iGPU:")
        print("    python text_embedder_client.py start")
        print("  Embed one-off text:")
        print('    python text_embedder_client.py embed "your text here"')
        print("  Stop:")
        print("    python text_embedder_client.py stop")
    elif preset == "medium":
        print("  The `medium` preset is the default — no changes needed.")
        print("  Start the MCP server (the existing entry point):")
        print("    python server.py")
        print("  Ingest a folder:")
        print('    python ingest.py "~/Documents/..." --index index/<col> --reset')
    elif preset == "large":
        print("  The `large` preset uses Qwen3-VL-Embedding-8B (8-bit, text-only).")
        print("  Requires bitsandbytes:")
        print("    pip install bitsandbytes")
        print("  Start the embedder on the dGPU:")
        print("    python large_embedder_client.py start")
        print("  Embed one-off text:")
        print('    python large_embedder_client.py embed "your text here"')
        print("  Stop:")
        print("    python large_embedder_client.py stop")
    print()
    print("Re-run this onboarding any time:")
    print("  python setup.py [--preset tiny|small|medium|large]")


def _maybe_warn_preset_change(new: str, old_setup: dict | None) -> bool:
    """If a previous setup exists with a different preset, refuse silently.
    Returns True if a destructive change is requested."""
    if not old_setup:
        return False
    old = old_setup.get("preset")
    if old and old != new:
        print()
        print(f"⚠  Preset change: {old} → {new}")
        print(f"   The existing index uses dim={dp.PRESETS[old]['dim']} vectors.")
        print(f"   {new} uses dim={dp.PRESETS[new]['dim']} vectors — a different vector space.")
        print("   Re-ingest every collection before search results are meaningful:")
        for col in _collections_in_use():
            print(f"     python ingest.py \"{col}\" --reset")
        print("   Or re-run setup.py with --reset-after to do it automatically.")
        print()
        return True
    return False


def _collections_in_use() -> list[str]:
    """Return the `folder` paths of every collection in collections.json (best effort)."""
    try:
        import rag_collections
        cols = rag_collections.load()
        return [c.get("folder") for c in cols.values() if c.get("folder")]
    except Exception:                                # noqa: BLE001
        return []


def _run_reset_after() -> None:
    """Destructive: run ingest.py --reset for every collection folder."""
    import subprocess
    folders = _collections_in_use()
    if not folders:
        print("[reset-after] no collections found in collections.json — nothing to reset.")
        return
    print(f"[reset-after] resetting and re-ingesting {len(folders)} collection(s)...")
    for folder in folders:
        col = os.path.basename(os.path.normpath(folder)) or "default"
        idx = os.path.join("index", col)
        print(f"[reset-after] {folder} -> {idx}")
        try:
            subprocess.run(
                [sys.executable, "ingest.py", folder, "--index", idx, "--reset"],
                check=False,
            )
        except Exception as e:                       # noqa: BLE001
            print(f"[reset-after] FAILED for {folder}: {e!r}")


def cmd_probe(args) -> int:
    detected = dp.probe(include_cpu_bench=not args.no_cpu_bench,
                        cpu_bench_budget_s=args.bench_budget)
    detected["recommendation"] = dp.pick(detected)
    detected["alternatives"] = dp.alternatives(detected)
    if args.json:
        json.dump(detected, sys.stdout, indent=2)
        print()
    else:
        _print_hardware(detected)
        print(f"\nRecommendation: {detected['recommendation']}")
        print(f"Alternatives:   {', '.join(detected['alternatives']) or '(none)'}")
    return 0


def cmd_setup(args) -> int:
    old_setup = _load_setup()
    if old_setup and not args.force:
        if args.json:
            payload = dict(old_setup)
            payload["status"] = "exists"
            json.dump(payload, sys.stdout, indent=2)
            print()
        else:
            print(f"setup.json already exists (preset={old_setup.get('preset')!r}). "
                  "Pass --force to overwrite.")
        return 0

    detected = dp.probe(include_cpu_bench=not args.no_cpu_bench,
                        cpu_bench_budget_s=args.bench_budget)
    detected["recommendation"] = dp.pick(detected)
    detected["alternatives"] = dp.alternatives(detected)

    if args.json:
        # Machine-readable mode: dump the probe + recommendation and exit.
        # No picker, no writes, no confirmation prompts.
        json.dump(detected, sys.stdout, indent=2)
        print()
        return 0

    if args.preset:
        if args.preset not in dp.VALID_PRESETS:
            print(f"unknown preset {args.preset!r}; choose one of "
                  f"{', '.join(dp.VALID_PRESETS)}", file=sys.stderr)
            return 2
        # warn if --preset forces something the hardware can't support
        if args.preset not in dp.alternatives(detected) and dp.alternatives(detected):
            print(f"⚠  {args.preset!r} exceeds the highest preset this hardware "
                  f"can run ({dp.alternatives(detected)[0]}); proceeding anyway.")
        chosen = args.preset
    else:
        _print_hardware(detected)
        chosen = _interactive_pick(detected)

    if chosen == "large" and not _confirm_large(args):
        print("aborted.")
        return 1

    payload = {
        "schema": SCHEMA,
        "preset": chosen,
        "detected": {
            "cpu_tflops_fp32": detected.get("cpu_tflops_fp32"),
            "gpus": detected.get("gpus") or [],
            "best_gpu": detected.get("best_gpu"),
        },
        "warned_large": (chosen == "large"),
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    _save_setup(payload)
    print(f"\nwrote {SETUP_PATH}  (preset={chosen!r})")

    destructive = _maybe_warn_preset_change(chosen, old_setup)
    if destructive and args.reset_after:
        _run_reset_after()

    _print_run_instructions(chosen)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="rag-mcp first-time setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of pretty text")
    ap.add_argument("--preset", choices=dp.VALID_PRESETS,
                    help="skip the picker; force this preset")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing setup.json")
    ap.add_argument("--yes", action="store_true",
                    help="skip the `large`-preset confirmation prompt")
    ap.add_argument("--reset-after", action="store_true",
                    help="after switching presets, run ingest.py --reset on every "
                         "collection (DESTRUCTIVE)")
    ap.add_argument("--no-cpu-bench", action="store_true",
                    help="skip the numpy matmul CPU benchmark (faster)")
    ap.add_argument("--bench-budget", type=float, default=2.0,
                    help="seconds to spend on the CPU benchmark (default 2.0)")

    sub = ap.add_subparsers(dest="cmd", required=False)

    # Default subcommand is `setup` (the onboarding flow); `probe` is read-only.
    p_probe = sub.add_parser("probe", help="probe hardware and exit (no writes)")
    p_probe.set_defaults(_fn=cmd_probe)
    p_setup = sub.add_parser("setup", help="probe + interactive picker + write setup.json")
    p_setup.set_defaults(_fn=cmd_setup)

    args = ap.parse_args()
    fn = getattr(args, "_fn", None) or cmd_setup
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
