#!/usr/bin/env python3
"""Large embedder client — spawns/kills the Qwen3-VL-Embedding-8B worker.

Same shape as worker_client.py and tiny_embedder_client.py. Importing this
module does NOT import torch; that lives only inside the child subprocess.

CLI:
  python large_embedder_client.py start      # spawn the worker, write pidfile
  python large_embedder_client.py stop       # SIGTERM, fallback SIGKILL
  python large_embedder_client.py status     # warm/cold + dim + quant
  python large_embedder_client.py embed "…"  # one-off embedding
  python large_embedder_client.py shell      # REPL: type text -> get vector
"""
from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "large_embedder_worker.py")
PIDFILE = "/tmp/large_embedder.json"
STARTUP_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 120.0


class LargeEmbedder:
    def __init__(self, python: str | None = None, env: dict | None = None):
        self.python = python or sys.executable
        self.env = dict(env if env is not None else os.environ)
        self.proc: subprocess.Popen | None = None
        self._id = 0
        self.dim: int | None = None
        self.model_name: str | None = None
        self.quant: str | None = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, ready_timeout: float = STARTUP_TIMEOUT_S) -> None:
        if self.alive:
            return
        log_path = "/tmp/large_embedder.log"
        log_fp = open(log_path, "wb")
        self.proc = subprocess.Popen(
            [self.python, WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log_fp, text=True, env=self.env, bufsize=1,
        )
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("large worker died during startup "
                                   "(see /tmp/large_embedder.log)")
            msg = json.loads(line)
            if msg.get("event") == "ready":
                self.dim = msg.get("dim")
                self.model_name = msg.get("model")
                self.quant = msg.get("quant")
                return
        raise TimeoutError("large worker did not become ready "
                           "(see /tmp/large_embedder.log)")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self._id += 1
            self.proc.stdin.write(json.dumps({"op": "shutdown", "id": self._id}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=15)
        except Exception:                            # noqa: BLE001
            try:
                self.proc.terminate()
                self.proc.wait(timeout=8)
            except Exception:                        # noqa: BLE001
                self.proc.kill()
        finally:
            self.proc = None

    def _rpc(self, req: dict, timeout: float = REQUEST_TIMEOUT_S) -> dict:
        if not self.alive:
            self.start()
        self._id += 1
        req["id"] = self._id
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("large worker died mid-request")
            msg = json.loads(line)
            if msg.get("id") == self._id:
                return msg
        raise TimeoutError("large worker request timed out")

    def embed_text(self, texts, is_query: bool = False) -> np.ndarray:
        resp = self._rpc({"op": "embed_text", "texts": list(texts),
                          "is_query": bool(is_query)})
        if not resp.get("ok"):
            raise RuntimeError("large embed failed: " + str(resp.get("error")))
        buf = base64.b64decode(resp["vecs_b64"])
        return np.frombuffer(buf, dtype=np.float32).reshape(resp["n"], resp["dim"])


def _proc_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def cmd_start(args):
    le = LargeEmbedder()
    if le.alive:
        print("already running")
        return
    print(f"starting large embedder worker "
          f"({os.environ.get('RAG_LARGE_MODEL', 'Qwen/Qwen3-VL-Embedding-8B')}, "
          f"quant={os.environ.get('RAG_LARGE_QUANT', '8bit')}) ...")
    t0 = time.time()
    le.start()
    print(f"ready in {time.time()-t0:.1f}s (dim={le.dim}, quant={le.quant})")
    payload = {"pid": le.proc.pid, "dim": le.dim, "model": le.model_name,
               "quant": le.quant, "started": time.time()}
    tmp = PIDFILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, PIDFILE)
    print(f"pidfile: {PIDFILE}")


def cmd_stop(args):
    if not os.path.isfile(PIDFILE):
        print("nothing running")
        return
    info = json.load(open(PIDFILE))
    pid = info.get("pid")
    if not pid or not _proc_alive(pid):
        print(f"stale pidfile (pid={pid}); removing")
        os.remove(PIDFILE)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.time() + 10
    while time.time() < deadline and _proc_alive(pid):
        time.sleep(0.2)
    if _proc_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os.remove(PIDFILE)
    print(f"stopped pid={pid}")


def cmd_status(args):
    if not os.path.isfile(PIDFILE):
        print("COLD")
        return
    info = json.load(open(PIDFILE))
    pid = info.get("pid")
    if not pid or not _proc_alive(pid):
        print("COLD (stale pidfile)")
        return
    age = time.time() - info.get("started", time.time())
    print(f"WARM  pid={pid}  dim={info.get('dim')}  "
          f"model={info.get('model')}  quant={info.get('quant')}  up {age:.0f}s")


def cmd_embed(args):
    le = LargeEmbedder()
    le.start()
    vecs = le.embed_text(args.texts, is_query=args.is_query)
    for text, v in zip(args.texts, vecs):
        preview = ", ".join(f"{x:.4f}" for x in v[:8])
        print(f"[{v.shape}] {text!r}")
        print(f"  norm={np.linalg.norm(v):.4f}  head=[{preview}, ...]")
        print(f"  full: {v.tolist()}")
    if not args.keep_alive:
        le.stop()


def cmd_shell(args):
    le = LargeEmbedder()
    le.start()
    print(f"# large embedder shell (dim={le.dim}, model={le.model_name}, "
          f"quant={le.quant})")
    print("# type text, hit enter -> get vector; ctrl-d to quit")
    try:
        while True:
            text = input("> ")
            if not text:
                continue
            v = le.embed_text([text], is_query=False)[0]
            print(f"  norm={np.linalg.norm(v):.4f}  head="
                  f"[{', '.join(f'{x:.4f}' for x in v[:8])}, ...]")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        le.stop()


def main():
    ap = argparse.ArgumentParser(description="large embedder client")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start"); p.set_defaults(fn=cmd_start)
    p = sub.add_parser("stop"); p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("embed")
    p.add_argument("texts", nargs="+")
    p.add_argument("--is-query", action="store_true")
    p.add_argument("--keep-alive", action="store_true",
                   help="don't stop the worker after embedding")
    p.set_defaults(fn=cmd_embed)
    p = sub.add_parser("shell"); p.set_defaults(fn=cmd_shell)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
