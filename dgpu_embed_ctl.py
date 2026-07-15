#!/usr/bin/env python3
"""Start/stop/bench the small text embedder on the 7700S dGPU (Vulkan1, BDF
0000:03:00.0) instead of its usual home on the 780M iGPU (Vulkan0). Exists so
we can find the fastest batch/ubatch/parallel combo for the 7700S's extra
VRAM and compute, independently of the always-on 780M embedder the server
uses, and run the two side by side for a 2x-GPU ingest.

State is a pidfile (not an in-process object) because start/stop/bench are
separate CLI invocations:

  python dgpu_embed_ctl.py start  [--preset large] [--port 8092]
  python dgpu_embed_ctl.py stop   [--port 8092]
  python dgpu_embed_ctl.py status [--port 8092]
  python dgpu_embed_ctl.py bench  [--preset large] [--port 8092] [--n 200]
  python dgpu_embed_ctl.py bench-all [--port 8092] [--n 200]   # try every preset
"""
import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_embedder_client import TextEmbedder, DEFAULT_MODEL_PATH, DEFAULT_POOLING

DGPU_VK_INDEX = "1"
DGPU_VK_BDF = "0000:03:00.0"          # 7700S / Navi33, confirmed via lspci
DEFAULT_PORT = 8092
PIDFILE_TMPL = "/tmp/dgpu_embed_{port}.json"

# ctx == batch == ubatch unless noted; parallel = physical batch "slots".
PRESETS = {
    "small":   dict(ctx=2048,  batch=2048,  ubatch=2048,  parallel=1),
    "medium":  dict(ctx=4096,  batch=4096,  ubatch=4096,  parallel=1),
    "large":   dict(ctx=8192,  batch=8192,  ubatch=8192,  parallel=1),
    "xlarge":  dict(ctx=8192,  batch=8192,  ubatch=8192,  parallel=4),
    "xxlarge": dict(ctx=16384, batch=16384, ubatch=16384, parallel=4),
}


def _pidfile(port):
    return PIDFILE_TMPL.format(port=port)


def _dgpu_env():
    env = dict(os.environ)
    env["GGML_VK_VISIBLE_DEVICES"] = DGPU_VK_INDEX
    env["VK_LOADER_DEVICE_SELECT"] = DGPU_VK_BDF
    return env


def _make_embedder(preset, port):
    p = PRESETS[preset]
    return TextEmbedder(
        model_path=DEFAULT_MODEL_PATH, port=port, pooling=DEFAULT_POOLING,
        ctx_size=p["ctx"], batch_size=p["batch"], ubatch_size=p["ubatch"],
        parallel=p["parallel"], env=_dgpu_env())


def _read_pidfile(port):
    path = _pidfile(port)
    if not os.path.isfile(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def _write_pidfile(port, pid, preset):
    json.dump({"pid": pid, "preset": preset, "port": port, "started": time.time()},
               open(_pidfile(port), "w"))


def dgpu_busy_pct():
    """% GPU-busy for the 7700S (rocm-smi GPU[0] on this machine), or None if
    rocm-smi is unavailable."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--json"], text=True,
            stderr=subprocess.DEVNULL, timeout=10)
        data = json.loads(out)
        for v in data.values():
            if isinstance(v, dict) and "GPU use (%)" in v:
                return float(v["GPU use (%)"])
    except Exception:
        pass
    return None


def wait_for_dgpu_idle(threshold=10.0, stable_checks=3, interval=2.0,
                        timeout=600.0, quiet=False):
    """Block until the 7700S has been below `threshold`% busy for
    `stable_checks` consecutive polls (a single low reading can just be a gap
    between batches of someone else's job). One heavy Vulkan workload at a
    time on this GPU — see fw16-dgpu-thermal-shared memory note. Raises
    TimeoutError if it never clears within `timeout`s."""
    deadline = time.time() + timeout
    consecutive = 0
    while time.time() < deadline:
        pct = dgpu_busy_pct()
        if pct is None or pct <= threshold:
            consecutive += 1
            if consecutive >= stable_checks:
                return
        else:
            if not quiet:
                print(f"  7700S busy ({pct:.0f}%) — waiting for it to free up ...")
            consecutive = 0
        time.sleep(interval)
    raise TimeoutError(f"7700S still busy after {timeout:.0f}s")


def _proc_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def cmd_start(args):
    info = _read_pidfile(args.port)
    if info and _proc_alive(info["pid"]):
        print(f"already running: preset={info['preset']} pid={info['pid']} "
              f"port={args.port}")
        return
    if not args.no_wait:
        wait_for_dgpu_idle()
    emb = _make_embedder(args.preset, args.port)
    print(f"starting 7700S text embedder: preset={args.preset} "
          f"({PRESETS[args.preset]}) port={args.port} ...")
    t0 = time.time()
    emb.start()
    _write_pidfile(args.port, emb.proc.pid, args.preset)
    print(f"ready in {time.time()-t0:.1f}s on {emb.device} (dim={emb.dim}), "
          f"pid={emb.proc.pid}")


def cmd_stop(args):
    info = _read_pidfile(args.port)
    if not info or not _proc_alive(info["pid"]):
        print(f"nothing running on port {args.port}")
        os.path.isfile(_pidfile(args.port)) and os.remove(_pidfile(args.port))
        return
    pid = info["pid"]
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.time() + 8
    while time.time() < deadline and _proc_alive(pid):
        time.sleep(0.2)
    if _proc_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    os.remove(_pidfile(args.port))
    print(f"stopped pid={pid} on port {args.port}")


def cmd_status(args):
    info = _read_pidfile(args.port)
    if not info or not _proc_alive(info["pid"]):
        print(f"port {args.port}: COLD")
        return
    idle = time.time() - info["started"]
    print(f"port {args.port}: WARM  preset={info['preset']} "
          f"{PRESETS[info['preset']]}  pid={info['pid']}  up {idle:.0f}s")


def _sample_texts(n, sample_dir=None):
    """Pull n real chunks from PDFs under `sample_dir` (falls back to synthetic
    text) so bench numbers reflect actual document content. `sample_dir` may be
    set via --sample-dir or $RAG_BENCH_SAMPLE_DIR; default = synthetic."""
    if not sample_dir:
        sample_dir = os.environ.get("RAG_BENCH_SAMPLE_DIR", "")
    if not sample_dir:
        return [f"synthetic benchmark chunk number {i} " * 20 for i in range(n)]
    try:
        import glob
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ingest as ing
        pdfs = glob.glob(os.path.join(sample_dir, "**", "*.pdf"), recursive=True)
        texts = []
        for pdf in pdfs:
            for pg in ing.pdf_text_pages(pdf):
                texts.extend(ing.chunk_text(pg["html"], 1200))
                if len(texts) >= n:
                    return texts[:n]
        if texts:
            return texts
    except Exception:
        pass
    return [f"synthetic benchmark chunk number {i} " * 20 for i in range(n)]


def _run_bench(preset, port, n, concurrency=1, sample_dir=None):
    """Embed n chunks in batches of 16, firing `concurrency` batches at once
    (a thread pool of persistent-connection clients) so multi-slot presets
    (parallel>1) actually get pipelined work instead of one request at a time."""
    emb = _make_embedder(preset, port)
    t0 = time.time()
    emb.start()
    boot_s = time.time() - t0
    texts = _sample_texts(n, sample_dir=sample_dir)
    B = 16
    batches = [texts[i:i + B] for i in range(0, len(texts), B)]

    t0 = time.time()
    if concurrency <= 1:
        for b in batches:
            emb.embed_text(b)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(ex.map(emb.embed_text, batches))
    elapsed = time.time() - t0
    rate = len(texts) / elapsed if elapsed > 0 else 0.0
    print(f"  preset={preset:8s} conc={concurrency:2d} {str(PRESETS[preset]):55s} "
          f"boot={boot_s:5.1f}s  {len(texts)} chunks in {elapsed:5.1f}s "
          f"-> {rate:6.1f} chunks/s")
    _write_pidfile(port, emb.proc.pid, preset)
    return rate


def cmd_bench(args):
    src = args.sample_dir or os.environ.get("RAG_BENCH_SAMPLE_DIR", "") or "synthetic"
    print(f"benchmarking preset={args.preset} conc={args.concurrency} with "
          f"{args.n} chunks from {src} on the 7700S (port {args.port}) ...")
    _run_bench(args.preset, args.port, args.n, args.concurrency,
               sample_dir=args.sample_dir)


def cmd_bench_all(args):
    src = args.sample_dir or os.environ.get("RAG_BENCH_SAMPLE_DIR", "") or "synthetic"
    print(f"benchmarking ALL presets x concurrency in {{1, preset parallel}} "
          f"with {args.n} chunks from {src} on the 7700S (port {args.port}); "
          f"stops+restarts the server per combo since batch/ubatch/parallel "
          f"are launch-time flags ...")
    results = {}
    for preset in PRESETS:
        for conc in sorted({1, PRESETS[preset]["parallel"]}):
            info = _read_pidfile(args.port)
            if info and _proc_alive(info["pid"]):
                cmd_stop(args)
            results[(preset, conc)] = _run_bench(preset, args.port, args.n, conc,
                                                 sample_dir=args.sample_dir)
    cmd_stop(args)
    best = max(results, key=results.get)
    print(f"\nbest: preset={best[0]} concurrency={best[1]}  "
          f"({results[best]:.1f} chunks/s)")
    print("(embedder left stopped; `start --preset "
          f"{best[0]}` + drive it with {best[1]} concurrent client threads "
          "for real ingest work)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--sample-dir", default=None,
                    help="optional directory of PDFs for `bench`/`bench-all` "
                         "to sample real text from (default: synthetic text)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start"); p.add_argument("--preset", default="large",
                                                 choices=PRESETS)
    p.add_argument("--no-wait", action="store_true",
                   help="skip waiting for the 7700S to be idle before launch")
    p.set_defaults(fn=cmd_start)
    p = sub.add_parser("stop"); p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("bench"); p.add_argument("--preset", default="large",
                                                 choices=PRESETS)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=1)
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("bench-all"); p.add_argument("--n", type=int, default=200)
    p.set_defaults(fn=cmd_bench_all)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
