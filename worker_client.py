"""Thin client to manage the embedder subprocess and talk the JSON-lines protocol.

Used by BOTH the MCP server (lazy, query-time) and ingest.py. Importing this
module does NOT import torch — the heavy libs live only inside the subprocess.
"""
import subprocess, sys, os, json, base64, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "embedder_worker.py")


class Embedder:
    """Owns one embedder_worker subprocess. Spawns on first use (cold start)."""

    def __init__(self, python=None, env=None):
        self.python = python or sys.executable
        self.env = env or dict(os.environ)
        self.proc = None
        self._id = 0
        self._lock = threading.Lock()
        self.device = None
        self.dim = None

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, ready_timeout=600):
        """Cold start: launch the worker and block until the model is loaded."""
        if self.alive:
            return
        self.proc = subprocess.Popen(
            [self.python, WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, env=self.env, bufsize=1,
        )
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("worker died during startup")
            msg = json.loads(line)
            if msg.get("event") == "ready":
                self.device = msg.get("device")
                self.dim = msg.get("dim")
                return
        raise TimeoutError("worker did not become ready")

    def stop(self):
        """Kill the worker -> frees VRAM and RAM. This is the 'go cold' action."""
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
            self.proc = None

    def _rpc(self, req):
        with self._lock:
            if not self.alive:
                self.start()
            self._id += 1
            req["id"] = self._id
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("worker died mid-request")
                msg = json.loads(line)
                if msg.get("event") == "ready":
                    continue
                if msg.get("id") == self._id:
                    return msg

    def _decode(self, msg):
        if not msg.get("ok"):
            raise RuntimeError("embed failed: " + str(msg.get("error")))
        buf = base64.b64decode(msg["vecs_b64"])
        return np.frombuffer(buf, dtype=np.float32).reshape(msg["n"], msg["dim"])

    def embed_text(self, texts, is_query=False):
        return self._decode(self._rpc({"op": "embed_text", "texts": list(texts),
                                       "is_query": is_query}))

    def embed_image(self, paths, captions=None, is_query=False):
        """Embed images. `captions` (optional, parallel to `paths`) fuses each
        image with a short text — used for video-frame closed captions."""
        req = {"op": "embed_image", "paths": list(paths), "is_query": is_query}
        if captions is not None:
            req["captions"] = list(captions)
        return self._decode(self._rpc(req))
