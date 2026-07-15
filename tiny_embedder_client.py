#!/usr/bin/env python3
"""Tiny embedder client — spawns/kills the sentence-transformers worker.

Same shape as worker_client.py (VL-2B) and text_embedder_client.py (nomic).
Importing this module does NOT import sentence_transformers; that lives only
inside the child subprocess.

CLI:
  python tiny_embedder_client.py start      # spawn the worker, write pidfile
  python tiny_embedder_client.py stop       # SIGTERM, fallback SIGKILL
  python tiny_embedder_client.py status     # warm/cold + dim
  python tiny_embedder_client.py embed "…"  # one-off embedding (prints numpy row)
  python tiny_embedder_client.py shell      # REPL: type text -> get vector
"""
from __future__ import annotations

import base64
import json
import os
import shlex
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "tiny_embedder_worker.py")
PIDFILE = "/tmp/tiny_embedder.json"
STARTUP_TIMEOUT_S = 60.0
REQUEST_TIMEOUT_S = 60.0


class TinyEmbedder:
    def __init__(self, python: str | None = None, env: dict | None = None):
        self.python = python or sys.executable
        self.env = dict(env if env is not None else os.environ)
        self.proc: subprocess.Popen | None = None
        self._id = 0
        self._lock_path = None
        self.dim: int | None = None
        self.model_name: str | None = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, ready_timeout: float = STARTUP_TIMEOUT_S) -> None:
        if self.alive:
            return
        log_path = "/tmp/tiny_embedder.log"
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
                raise RuntimeError("tiny worker died during startup")
            msg = json.loads(line)
            if msg.get("event") == "ready":
                self.dim = msg.get("dim")
                self.model_name = msg.get("model")
                return
        raise TimeoutError("tiny worker did not become ready")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self._id += 1
            self.proc.stdin.write(json.dumps({"op": "shutdown", "id": self._id}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:                            # noqa: BLE001
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
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
                raise RuntimeError("tiny worker died mid-request")
            msg = json.loads(line)
            if msg.get("id") == self._id:
                return msg
        raise TimeoutError("tiny worker request timed out")

    def embed_text(self, texts, is_query: bool = False) -> np.ndarray:
        resp = self._rpc({"op": "embed_text", "texts": list(texts),
                          "is_query": bool(is_query)})
        if not resp.get("ok"):
            raise RuntimeError("tiny embed failed: " + str(resp.get("error")))
        buf = base64.b64decode(resp["vecs_b64"])
        return np.frombuffer(buf, dtype=np.float32).reshape(resp["n"], resp["dim"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_pidfile(dim, model_name):
    payload = {"pid": TinyEmbedder().proc.pid if False else None,  # filled by caller
               "dim": dim, "model": model_name,
               "started": time.time()}
    import json as _json
    tmp = PIDFILE + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(payload, f, indent=2)
    os.replace(tmp, PIDFILE)


def _proc_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def cmd_start(args):
    te = TinyEmbedder()
    if te.alive:
        print("already running")
        return
    print(f"starting tiny embedder worker ({os.environ.get('RAG_TINY_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')}) ...")
    t0 = time.time()
    te.start()
    print(f"ready in {time.time()-t0:.1f}s (dim={te.dim}, model={te.model_name})")
    payload = {"pid": te.proc.pid, "dim": te.dim, "model": te.model_name,
               "started": time.time()}
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
    deadline = time.time() + 8
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
          f"model={info.get('model')}  up {age:.0f}s")


def cmd_embed(args):
    te = TinyEmbedder()
    te.start()
    vecs = te.embed_text(args.texts, is_query=args.is_query)
    for text, v in zip(args.texts, vecs):
        # human-readable: short summary (first 8 dims) + full row on stderr
        preview = ", ".join(f"{x:.4f}" for x in v[:8])
        print(f"[{v.shape}] {text!r}")
        print(f"  norm={np.linalg.norm(v):.4f}  head=[{preview}, ...]")
        print(f"  full: {v.tolist()}")
    if not args.keep_alive:
        te.stop()


def cmd_shell(args):
    te = TinyEmbedder()
    te.start()
    print(f"# tiny embedder shell (dim={te.dim}, model={te.model_name})")
    print("# type text, hit enter -> get vector; ctrl-d to quit")
    try:
        while True:
            text = input("> ")
            if not text:
                continue
            v = te.embed_text([text], is_query=False)[0]
            print(f"  norm={np.linalg.norm(v):.4f}  head="
                  f"[{', '.join(f'{x:.4f}' for x in v[:8])}, ...]")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        te.stop()


def main():
    ap = argparse.ArgumentParser(description="tiny embedder client")
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
