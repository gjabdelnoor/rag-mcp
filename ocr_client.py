"""Thin client to manage the OCR worker subprocess (Surya-2 on the 7700S).

Like worker_client.Embedder but for OCR. Importing this module does NOT import
torch/surya — those live only in the child. The client picks the Vulkan
`llama-server` binary, runs the worker with the llamacpp backend on the GPU, and
transparently falls back to CPU (LLAMA_CPP_NGL=0) if the GPU spawn/inference
fails. Killing the worker tears down its llama-server and frees VRAM.

Auto-batching: at spawn time we read the *current free VRAM* (via rocm-smi)
and pick ``SURYA_INFERENCE_PARALLEL`` so that the GPU is filled but not
overcommitted. The hardcoded max is **20** (anything beyond this saturates
the llama-server scheduler with no real gain); per-slot ctx is fixed at 4096
to keep KV-cache small enough that 20 slots still fit alongside the user's
other GPU users (e.g. Gemma llama-server). Empirically (60-page deck 1,
Vulkan, on the 7700S):
  * parallel=1  -> 5.7 s/page (baseline)
  * parallel=4  -> ~3.5 s/page  (~1.6× baseline, comfortable GPU footprint)
  * parallel=20 -> ~1.4 s/page  (3.9× baseline, ~3.5 GB VRAM)
Override with RAG_OCR_PARALLEL=<n> or RAG_OCR_DISABLE_AUTOBATCH=1.
"""
import subprocess, sys, os, json, glob, threading, time, random, concurrent.futures

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "ocr_worker.py")

# Auto-batching knobs (hardcoded; see module docstring).
OCR_PARALLEL_MAX = 20
OCR_PARALLEL_DEFAULT = 4                  # when VRAM can't be measured
# Approximate VRAM used per parallel slot at ctx=4096 (KV cache + state) — a
# safe overestimate so we never over-allocate. Measured ~150 MiB/slot.
VRAM_PER_SLOT_MIB = 200
# Reserve a safety margin so we never try to claim VRAM the OS hasn't released
# from a recently-killed model (HIP context teardown lags a few seconds).
VRAM_RESERVE_MIB = 512

# Remote (cloud) OCR parallelism + backoff knobs. Pages are OCR'd by up to
# RAG_OCR_REMOTE_WORKERS threads concurrently (default 10). To avoid a
# synchronous stampede at t=0 — which can trip the tenant's per-second rate
# limit even when the individual requests are small — the first batch of
# submissions is spread across OCR_REMOTE_STAGGER_WINDOW seconds with jitter.
# Per-call failures (HTTP 429/408/5xx, connection/timeout) back off
# exponentially with jitter, capped at OCR_REMOTE_MAX_BACKOFF, and honor any
# Retry-After header the server returns.
OCR_REMOTE_MAX_WORKERS = int(os.environ.get("RAG_OCR_REMOTE_WORKERS", "10"))
OCR_REMOTE_STAGGER_WINDOW = float(
    os.environ.get("RAG_OCR_REMOTE_STAGGER_WINDOW", "0.5"))
OCR_REMOTE_BASE_BACKOFF = float(os.environ.get("RAG_OCR_REMOTE_BASE_BACKOFF", "1.0"))
OCR_REMOTE_MAX_BACKOFF = float(os.environ.get("RAG_OCR_REMOTE_MAX_BACKOFF", "60.0"))
OCR_REMOTE_MAX_RETRIES = int(os.environ.get("RAG_OCR_REMOTE_MAX_RETRIES", "6"))


def find_llama_server():
    """Locate a Vulkan-capable `llama-server`. Prefer an explicit override, then
    LM Studio's bundled Vulkan build, then anything on PATH."""
    env = os.environ.get("RAG_LLAMA_BIN")
    if env and os.path.exists(env):
        return env
    pats = [
        os.path.expanduser("~/.lmstudio/extensions/backends/"
                            "*vulkan*/llama-server"),
        os.path.expanduser("~/.lmstudio/extensions/backends/"
                            "**/llama-server"),
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[-1]          # newest-sorted
    from shutil import which
    return which("llama-server") or "llama-server"


def _free_vram_mib():
    """GPU0 free VRAM in MiB, or None if rocm-smi can't read it."""
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"],
                                      text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    total = free = None
    for line in out.splitlines():
        if "GPU[0]" in line:
            if "Total" in line:
                total = int(line.split()[-1]) / (1 << 20)
            elif "Used" in line:
                used = int(line.split()[-1]) / (1 << 20)
                free = (total or 0) - used
    return free


def _strip_response_wrappers(raw):
    """Hosted chat models wrap their answers in <think>…</think> reasoning
    plus a fenced ```html … ``` code block. The cache only wants the structural
    HTML, so strip both wrappers and any trailing prose."""
    import re
    # drop the <think>…</think> block (greedy across newlines)
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # if there's a fenced code block, take its contents
    m = re.search(r"```(?:html)?\s*\n(.*?)```", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


def _autobatch_parallel():
    """Pick a parallel value for the spawned llama-server based on free VRAM.

    Reserves VRAM_RESERVE_MIB for other GPU users and the safety margin;
    divides the rest by VRAM_PER_SLOT_MIB to get a slot count; clamps to
    [1, OCR_PARALLEL_MAX].
    """
    free = _free_vram_mib()
    if free is None:
        return OCR_PARALLEL_DEFAULT
    usable = max(0, free - VRAM_RESERVE_MIB)
    n = int(usable // VRAM_PER_SLOT_MIB)
    return max(1, min(OCR_PARALLEL_MAX, n))


def _is_rate_limit_error(e):
    """True for errors worth retrying: HTTP 429/408, 5xx, or connection /
    timeout errors (OpenAI SDK raises APIConnectionError/APITimeoutError
    without a status_code). Other 4xx are treated as hard failures."""
    status = getattr(e, "status_code", None)
    if status == 429 or status == 408:
        return True
    if isinstance(status, int) and 500 <= status < 600:
        return True
    return type(e).__name__ in ("APIConnectionError", "APITimeoutError")


def _get_retry_after(e):
    """Pull Retry-After (seconds) from the SDK exception's response headers,
    or None if absent / unparseable."""
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    hdrs = getattr(resp, "headers", None) or {}
    ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
    if ra is None:
        return None
    try:
        return float(ra)
    except (TypeError, ValueError):
        return None


def _call_with_backoff(client, model, b64, attempt=0):
    """Single remote OCR call with exponential backoff + jitter on rate-limit
    or transient errors. Raises on hard failures or after
    OCR_REMOTE_MAX_RETRIES retries. Backoff is capped at
    OCR_REMOTE_MAX_BACKOFF and a Retry-After header (if the server sent one)
    overrides the computed delay."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",
                 "text": "Return the page as HTML, preserving structure."},
            ]}],
            timeout=600,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        if not _is_rate_limit_error(e) or attempt >= OCR_REMOTE_MAX_RETRIES:
            raise
        delay = min(OCR_REMOTE_MAX_BACKOFF,
                    OCR_REMOTE_BASE_BACKOFF * (2 ** attempt))
        jitter = random.uniform(0, delay * 0.5)
        sleep_for = delay + jitter
        ra = _get_retry_after(e)
        if ra is not None:
            sleep_for = max(sleep_for, ra + random.uniform(0, 1.0))
        print(f"[ocr] remote rate-limited ({type(e).__name__}, "
              f"status={getattr(e, 'status_code', '?')}); "
              f"backing off {sleep_for:.1f}s "
              f"(attempt {attempt + 1}/{OCR_REMOTE_MAX_RETRIES})",
              file=sys.stderr, flush=True)
        time.sleep(sleep_for)
        return _call_with_backoff(client, model, b64, attempt + 1)


class OCRWorker:
    """Owns one ocr_worker subprocess. GPU first, CPU fallback on failure."""

    def __init__(self, python=None, dpi=150):
        self.python = python or sys.executable
        self.proc = None
        self._id = 0
        self._lock = threading.Lock()
        self.dpi = dpi
        self.llama = find_llama_server()
        self.mode = None             # "gpu" | "cpu"

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _env(self, cpu):
        e = dict(os.environ)
        e["SURYA_INFERENCE_BACKEND"] = "llamacpp"
        e["LLAMA_CPP_BINARY"] = self.llama
        e["LLAMA_CPP_NGL"] = "0" if cpu else os.environ.get("RAG_OCR_NGL", "99")
        # Per-slot ctx 4096 is plenty for OCR (page+prompt+output < 2K tokens);
        # llama-server scales --ctx-size = parallel * per-slot automatically.
        e.setdefault("SURYA_INFERENCE_CTX_PER_SLOT", "4096")
        # Pick parallel from free VRAM, unless the user pinned a value
        if "SURYA_INFERENCE_PARALLEL" not in e:
            if "RAG_OCR_PARALLEL" in e:
                e["SURYA_INFERENCE_PARALLEL"] = e["RAG_OCR_PARALLEL"]
            elif e.get("RAG_OCR_DISABLE_AUTOBATCH") == "1":
                e["SURYA_INFERENCE_PARALLEL"] = str(OCR_PARALLEL_DEFAULT)
            else:
                n = _autobatch_parallel()
                free = _free_vram_mib()
                print(f"[ocr] autobatch: free VRAM ~{int(free or 0)} MiB "
                      f"-> parallel={n} (cap {OCR_PARALLEL_MAX}, "
                      f"{VRAM_PER_SLOT_MIB} MiB/slot, "
                      f"{VRAM_RESERVE_MIB} MiB reserve)",
                      file=sys.stderr, flush=True)
                e["SURYA_INFERENCE_PARALLEL"] = str(n)
        return e

    def start(self, cpu=False, ready_timeout=120):
        if self.alive:
            return
        self.mode = "cpu" if cpu else "gpu"
        self.proc = subprocess.Popen(
            [self.python, WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, env=self._env(cpu), bufsize=1,
        )
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("ocr worker died during startup")
            if json.loads(line).get("event") == "ready":
                return
        raise TimeoutError("ocr worker did not become ready")

    def stop(self):
        """Graceful shutdown so Surya's atexit kills the llama-server; then make
        sure neither the worker nor any stray llama-server survives."""
        if self.proc is not None:
            try:
                self._id += 1
                self.proc.stdin.write(json.dumps({"op": "shutdown",
                                                  "id": self._id}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.terminate(); self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
            self.proc = None
        self._reap_llama_servers()
        self.mode = None

    @staticmethod
    def _reap_llama_servers():
        """Safety net: Surya only tears down its llama-server on a CLEAN exit, so
        if the worker was SIGKILLed/crashed the server can be orphaned holding
        VRAM. Kill any llama-server still serving the Surya OCR model."""
        try:
            pids = subprocess.run(["pgrep", "-f", "surya-2.gguf"],
                                  capture_output=True, text=True).stdout.split()
        except Exception:
            return
        me = str(os.getpid())
        for pid in pids:
            if pid and pid != me:
                try:
                    os.kill(int(pid), 9)
                except Exception:
                    pass

    def _rpc(self, req, timeout=1800):
        self._id += 1
        req["id"] = self._id
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("ocr worker died mid-request")
            msg = json.loads(line)
            if msg.get("event") == "ready":
                continue
            if msg.get("id") == self._id:
                return msg
        raise TimeoutError("ocr request timed out")

    def ocr_pdf(self, path, dpi=None):
        """OCR every page of a PDF -> list of {"page", "html"}.

        Priority: remote MiniMax first (if RAG_OCR_REMOTE_URL is set), then
        local Vulkan GPU (Surya llama-server), then local CPU. The remote
        path is preferred because it doesn't contend with the user's other
        GPU workloads; the local fallback exists so OCR still works when the
        network is unreachable.
        """
        dpi = dpi or self.dpi
        with self._lock:
            pages = self._ocr_remote(path, dpi)
            if pages:
                return pages
            print(f"[ocr] remote OCR unavailable; falling back to local GPU",
                  file=sys.stderr, flush=True)
            for cpu in (False, True):
                try:
                    if not self.alive:
                        self.start(cpu=cpu)
                    msg = self._rpc({"op": "ocr", "path": path, "dpi": dpi})
                    if msg.get("ok"):
                        return msg["pages"]
                    raise RuntimeError(msg.get("error", "ocr failed"))
                except Exception as e:
                    if cpu:
                        print(f"[ocr] local CPU OCR failed ({e!r})",
                              file=sys.stderr, flush=True)
                        break
                    print(f"[ocr] GPU OCR failed ({e!r}); retrying on CPU",
                          file=sys.stderr, flush=True)
                    self.stop()
        return []

    def _ocr_remote(self, path, dpi):
        """OCR via a hosted OpenAI-compatible endpoint. Reads RAG_OCR_REMOTE_URL
        and MINIMAX_API_KEY from the environment. The image is base64-encoded
        and posted to /chat/completions with the configured model name. Returns
        the same {"page", "html"} shape as the local worker, so the cache and
        ingest paths don't care which path produced the text.

        Parallelism: pages are OCR'd concurrently by up to
        RAG_OCR_REMOTE_WORKERS threads (default 10). To avoid a synchronous
        stampede at t=0 — which can trip a tenant's per-second rate limit
        even when individual requests are small — the first batch of
        submissions is spread over OCR_REMOTE_STAGGER_WINDOW seconds with
        jitter. Per-call failures (HTTP 429/408/5xx, connection/timeout)
        back off exponentially with jitter and honor any Retry-After header
        the server returns.

        If the remote path returns no content for ANY page (or any call
        fails after retries), treat the whole call as failed — caching empty
        HTML would poison the index. The caller falls through to the local
        GPU path.
        """
        import base64, io
        from PIL import Image
        import fitz
        url = os.environ.get("RAG_OCR_REMOTE_URL")
        key = os.environ.get("MINIMAX_API_KEY")
        model = os.environ.get("RAG_OCR_REMOTE_MODEL", "minimax/surya-ocr")
        if not (url and key):
            return []
        n_workers = max(1, OCR_REMOTE_MAX_WORKERS)
        print(f"[ocr] trying remote OCR at {url} (model={model}, "
              f"workers={n_workers})",
              file=sys.stderr, flush=True)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=url)
            doc = fitz.open(path)
            # Render every page up-front: cheap, keeps the worker threads
            # busy only with network I/O.
            page_data = []
            for pno in range(doc.page_count):
                pix = doc.load_page(pno).get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                buf = io.BytesIO(); img.save(buf, format="PNG")
                page_data.append((pno,
                                  base64.b64encode(buf.getvalue()).decode("ascii")))
            doc.close()
            n = len(page_data)
            if n == 0:
                return []
            # Stagger only the first batch (the burst most likely to hit a
            # rate limit). After that, pages trickle in as workers free up
            # and a flat pace is fine.
            first_batch = min(n_workers, n)
            spacing = (OCR_REMOTE_STAGGER_WINDOW / first_batch
                       if first_batch > 1 else 0.0)
            out = [None] * n
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_workers) as ex:
                futures = []
                for i, (pno, b64) in enumerate(page_data):
                    if i < first_batch and i > 0:
                        time.sleep(spacing + random.uniform(0, spacing * 0.5))
                    futures.append((pno, ex.submit(
                        _call_with_backoff, client, model, b64)))
                for pno, fut in futures:
                    raw = fut.result()        # propagate per-page failures
                    out[pno] = {"page": pno + 1,
                                "html": _strip_response_wrappers(raw)}
            non_empty = sum(1 for p in out if p.get("html", "").strip())
            if non_empty == 0:
                print(f"[ocr] remote returned empty content for all {len(out)} "
                      f"pages; rejecting remote result and falling back to local",
                      file=sys.stderr, flush=True)
                return []
            if non_empty < len(out):
                print(f"[ocr] remote returned empty content for "
                      f"{len(out) - non_empty}/{len(out)} pages; rejecting remote "
                      f"result to avoid poisoning the cache",
                      file=sys.stderr, flush=True)
                return []
            return out
        except Exception as e:
            print(f"[ocr] remote OCR failed: {e!r}", file=sys.stderr, flush=True)
            return []
