"""Thin client for the small text-embedding llama-server on the 780M iGPU.

Mirrors the `Embedder` interface in `worker_client.py` so a later refactor can
swap the heavy 2B VL embedder for this small text-only one with no call-site
changes. Importing this module does NOT import torch — the model lives only in
the child `llama-server` subprocess we spawn.

Validated setup (see `kimi_contracts/refactor_goal1_REPORT.md`):
- model:       ~/projects/rag-mcp/models/nomic-embed-text-v1.5.Q8_0.gguf
- binary:      LM Studio's bundled Vulkan llama-server (see `find_llama_server`)
- port:        8091 (validated free on this machine)
- pooling:     mean
- dim:         768
- prefix:      "search_query: " if is_query else "search_document: "
- device pin:  GGML_VK_VISIBLE_DEVICES=0  +  VK_LOADER_DEVICE_SELECT=0000:c4:00.0
               (both required — see goal 1 report)
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Defaults — pinned by goal 1 validation. Override via env if needed.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

DEFAULT_MODEL_FILENAME = "nomic-embed-text-v1.5.Q8_0.gguf"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, DEFAULT_MODEL_FILENAME)

# 780M iGPU on this machine: Vulkan0 = RADV PHOENIX = PCI BDF 0000:c4:00.0.
# We expose both knobs so callers can override (e.g. for testing on a
# different machine). Goal 1 proved that BOTH are required: VK_VISIBLE_DEVICES
# alone binds compute to Vulkan0 but the Mesa loader still opens the dGPU
# render node and wakes the 7700S out of D3cold. Adding VK_LOADER_DEVICE_SELECT
# filters at the loader layer so the dGPU stays asleep for the whole run.
DEFAULT_VK_VISIBLE_DEVICES = "0"
DEFAULT_VK_LOADER_DEVICE_SELECT = "0000:c4:00.0"
DEFAULT_DEVICE_LABEL = "780M (Vulkan0)"

DEFAULT_POOLING = "mean"
# 2048 = nomic-embed-text-v1.5's trained context. Char-capped chunks (1200
# chars) can tokenize past 512 on math-dense PDF text-layer pages (seen up to
# ~1200 tokens in ESL), so 512 caused llama-server 500s during ingest.
DEFAULT_CTX_SIZE = 2048
DEFAULT_PORT = 8091
DEFAULT_READY_TIMEOUT_S = 120.0
DEFAULT_REQUEST_TIMEOUT_S = 60.0


def _find_llama_server() -> str:
    """Locate a Vulkan-capable `llama-server` (same precedence as ocr_client)."""
    env = os.environ.get("RAG_LLAMA_BIN")
    if env and os.path.exists(env):
        return env
    pats = [
        os.path.expanduser("~/.lmstudio/extensions/backends/*vulkan*/llama-server"),
        os.path.expanduser("~/.lmstudio/extensions/backends/**/llama-server"),
    ]
    import glob as _glob
    for pat in pats:
        hits = sorted(_glob.glob(pat, recursive=True))
        if hits:
            return hits[-1]
    found = shutil.which("llama-server")
    return found or "llama-server"


def _port_is_free(host: str, port: int) -> bool:
    """Bind-and-release probe: True iff we can grab the port and let it go."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
    except OSError:
        return False
    finally:
        s.close()
    # Also confirm no one is currently LISTENing there (a TIME_WAIT bind
    # could let us bind but then refuse connection attempts).
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return False  # something is accepting on it
    except (ConnectionRefusedError, socket.timeout, OSError):
        return True


def _http_get_status(url: str, timeout: float = 5.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.getcode()
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, ConnectionError, OSError):
        return 0


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _parse_embedding_response(resp: dict) -> List[float]:
    """Accept either llama-server's native /embedding shape or an OpenAI-compatible
    /v1/embeddings shape; return a flat list[float]."""
    if "embedding" in resp and isinstance(resp["embedding"], list):
        return [float(x) for x in resp["embedding"]]
    if "data" in resp and isinstance(resp["data"], list) and resp["data"]:
        item = resp["data"][0]
        if "embedding" in item:
            return [float(x) for x in item["embedding"]]
    raise RuntimeError(f"unrecognized embedding response: {list(resp.keys())}")


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize, returning a fresh float32 array. Zero rows stay zero."""
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (arr / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------
class TextEmbedder:
    """Owns one llama-server subprocess serving the small text embedder on the
    780M iGPU. Mirrors `worker_client.Embedder`'s public surface so this can be
    drop-in swapped in a later refactor."""

    def __init__(self, model_path: Optional[str] = None, port: Optional[int] = None,
                 pooling: Optional[str] = None, ctx_size: Optional[int] = None,
                 binary: Optional[str] = None, env: Optional[dict] = None,
                 batch_size: Optional[int] = None, ubatch_size: Optional[int] = None,
                 parallel: Optional[int] = None):
        self.model_path = model_path or os.environ.get(
            "RAG_TEXT_EMBED_MODEL", DEFAULT_MODEL_PATH)
        self.port = int(port if port is not None
                        else os.environ.get("RAG_TEXT_EMBED_PORT", DEFAULT_PORT))
        self.pooling = pooling or os.environ.get(
            "RAG_TEXT_EMBED_POOLING", DEFAULT_POOLING)
        self.ctx_size = int(ctx_size if ctx_size is not None
                            else os.environ.get("RAG_TEXT_EMBED_CTX", DEFAULT_CTX_SIZE))
        # batch/ubatch default to ctx_size (one big batch fits in context);
        # parallel is separate physical batch "slots" — raise it to pipeline
        # more concurrent embed requests on a GPU with VRAM/compute to spare
        # (the 7700S, not the 780M).
        self.batch_size = int(batch_size) if batch_size is not None else self.ctx_size
        self.ubatch_size = int(ubatch_size) if ubatch_size is not None else self.ctx_size
        self.parallel = int(parallel) if parallel is not None else 1
        self.binary = binary or _find_llama_server()
        self.env = dict(env if env is not None else os.environ)

        self.proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[str] = None
        self.device: Optional[str] = None
        self.dim: Optional[int] = None

    # ------------------------------------------------------------------ public
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, ready_timeout: float = DEFAULT_READY_TIMEOUT_S) -> None:
        """Cold start: launch llama-server, block until /health responds 200."""
        if self.alive:
            return
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"text-embed model not found: {self.model_path}")
        if not _port_is_free("127.0.0.1", self.port):
            raise RuntimeError(
                f"port {self.port} is already in use on 127.0.0.1; "
                "set RAG_TEXT_EMBED_PORT to a free port")

        # Spawn in its own process group so we can SIGTERM/SIGKILL the whole tree.
        # Inherit env (caller may want RAG_LLAMA_BIN etc.) and add the device-pin
        # knobs unless they're already set.
        env = dict(self.env)
        env.setdefault("GGML_VK_VISIBLE_DEVICES", DEFAULT_VK_VISIBLE_DEVICES)
        env.setdefault("VK_LOADER_DEVICE_SELECT", DEFAULT_VK_LOADER_DEVICE_SELECT)

        # The launch script does the heavy lifting (args, logging). We just
        # exec it through bash so the user's shell can expand anything that
        # needs expanding and we get a clean stdout/stderr stream.
        here = os.path.dirname(os.path.abspath(__file__))
        launch_script = os.path.join(here, "kimi_contracts",
                                     "launch_embed_server.sh")
        if not os.path.isfile(launch_script):
            raise FileNotFoundError(f"launch script missing: {launch_script}")
        cmd = ["bash", launch_script, self.model_path,
               str(self.port), self.pooling, str(self.ctx_size),
               str(self.batch_size), str(self.ubatch_size), str(self.parallel)]

        self._log_path = f"/tmp/text_embedder_{self.port}.log"
        log_fp = open(self._log_path, "wb")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=log_fp, stderr=subprocess.STDOUT,
            env=env, start_new_session=True, close_fds=True,
        )
        try:
            self._wait_ready(ready_timeout)
        except Exception:
            self.stop()
            raise

        # Probe the server to capture its reported device + dim.
        self.device, self.dim = self._probe_server()

    def stop(self) -> None:
        """SIGTERM, wait, SIGKILL fallback, then a pgrep safety net for any
        orphan llama-server still holding the model file."""
        if self.proc is not None:
            pid = self.proc.pid
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self.proc = None
        self._reap_orphans()

    def embed_text(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Embed a list of strings. Returns shape (len(texts), dim), float32,
        L2-normalized. Applies the model's search_query / search_document
        prefix convention (nomic-embed-text-v1.5)."""
        if not self.alive:
            self.start()
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)

        prefix = "search_query: " if is_query else "search_document: "
        prefixed = [prefix + t for t in texts]

        # llama-server's /embedding endpoint accepts {"content": "..."} for
        # single inputs. For batches we send {"content": [...]} — same key,
        # list value — which llama-server also understands (and returns a
        # list of embeddings under the "results" key). We try a couple of
        # shapes for resilience.
        urls = [f"http://127.0.0.1:{self.port}/embedding",
                f"http://127.0.0.1:{self.port}/v1/embeddings"]
        last_err: Optional[Exception] = None
        vecs: Optional[List[List[float]]] = None
        for url in urls:
            for body in ({"content": prefixed},
                         {"input": prefixed},
                         {"content": prefixed[0]} if len(prefixed) == 1 else None):
                if body is None:
                    continue
                try:
                    resp = _http_post_json(url, body, DEFAULT_REQUEST_TIMEOUT_S)
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    continue
                # Parse: could be a single embedding, a list of results, or
                # OAI-style {"data": [...]}.
                try:
                    if ("embedding" in resp and isinstance(resp["embedding"], list)
                            and resp["embedding"]
                            and isinstance(resp["embedding"][0], (int, float))):
                        # single-string case
                        vecs = [resp["embedding"]]
                        break
                    if "results" in resp and isinstance(resp["results"], list):
                        vecs = [r["embedding"] for r in resp["results"]]
                        break
                    if "data" in resp and isinstance(resp["data"], list):
                        vecs = [item["embedding"] for item in resp["data"]]
                        break
                except (KeyError, TypeError, IndexError) as e:
                    last_err = e
                    continue
            if vecs is not None:
                break

        if vecs is None:
            raise RuntimeError(f"all embedding endpoint shapes failed: {last_err!r}")

        arr = np.asarray(vecs, dtype=np.float32)
        return _l2_normalize_rows(arr)

    # ----------------------------------------------------------------- helpers
    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        health_url = f"http://127.0.0.1:{self.port}/health"
        last_err = ""
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                # process died; surface the log
                tail = ""
                if self._log_path and os.path.isfile(self._log_path):
                    with open(self._log_path, "r", errors="replace") as f:
                        tail = f.read()[-2000:]
                raise RuntimeError(
                    f"llama-server died during startup (exit={self.proc.returncode}); "
                    f"log tail:\n{tail}")
            code = _http_get_status(health_url, timeout=2.0)
            if code == 200:
                return
            last_err = f"http={code}"
            time.sleep(0.25)
        raise TimeoutError(
            f"llama-server did not become healthy in {timeout}s "
            f"(last status: {last_err}); log: {self._log_path}")

    def _probe_server(self) -> tuple:
        """Discover the device label and embedding dim from the running server."""
        device = DEFAULT_DEVICE_LABEL
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/pooling/info", timeout=5) as r:
                _ = json.loads(r.read().decode("utf-8"))
                # Server reports its pooling choice; that's a nice signal but
                # doesn't change device/dim.
        except Exception:
            pass
        # Embed a one-token probe to get the dim.
        try:
            v = self._embed_raw("probe")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"could not probe embedding dim: {e!r}")
        return device, len(v)

    def _embed_raw(self, text: str) -> List[float]:
        """Single-text embedding WITHOUT prefixing, NO L2 norm — used only for
        the dim probe at startup. Falls back through endpoint shapes."""
        for url in (f"http://127.0.0.1:{self.port}/embedding",
                    f"http://127.0.0.1:{self.port}/v1/embeddings"):
            for body in ({"content": text}, {"input": text}):
                try:
                    resp = _http_post_json(url, body, 30.0)
                except Exception:
                    continue
                try:
                    return _parse_embedding_response(resp)
                except Exception:
                    continue
        raise RuntimeError("dim probe failed on all endpoints")

    def _reap_orphans(self) -> None:
        """Kill any llama-server still serving our model file (mirror of
        ocr_client.OCRWorker._reap_llama_servers)."""
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", os.path.basename(self.model_path)],
                text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, OSError):
            return
        me = str(os.getpid())
        for pid in out.split():
            pid = pid.strip()
            if not pid or pid == me:
                continue
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, ValueError):
                pass


__all__ = ["TextEmbedder", "DEFAULT_DEVICE_LABEL"]