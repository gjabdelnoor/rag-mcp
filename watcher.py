#!/usr/bin/env python3
"""Event-driven ingester. Watches every collection's folder and, when files are
added/changed/removed, tells the RAG server to reconcile that collection's index
by calling the server's `ingest_path` MCP tool over HTTP.

Why delegate instead of embedding here: the server is the single owner of the
GPU model, so this daemon stays torch-free and tiny. It reuses the server's
cold/warm policy and its OCR prepass automatically.

Robustness:
  * debounce — a copied file fires many events; wait for quiet.
  * write-stability — only trigger once pending files stop growing.
  * startup catch-up — reconcile every collection once on boot.
"""
import os, sys, time, asyncio, threading
from datetime import timedelta
import rag_collections

SERVER_URL = os.environ.get("RAG_URL", "http://127.0.0.1:8077/mcp")
DEBOUNCE = float(os.environ.get("RAG_DEBOUNCE", "4"))     # quiet seconds
SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".text", ".epub",
             ".jpg", ".jpeg", ".png", ".webp", ".json"}   # .json = frames.json unit

COLLECTIONS = rag_collections.load()
_pending = set()                 # collection names awaiting reconcile
_last_event = [0.0]
_state = threading.Lock()


def log(*a):
    print("[watcher]", *a, file=sys.stderr, flush=True)


def _is_supported(p):
    return os.path.splitext(p)[1].lower() in SUPPORTED


async def _reconcile(collection, folder):
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession
    async with streamablehttp_client(SERVER_URL) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool(
                "ingest_path",
                {"collection": collection, "path": folder, "recursive": True},
                read_timeout_seconds=timedelta(seconds=3600))
            return res.content[0].text if res.content else "(no output)"


def reconcile(collection):
    folder = COLLECTIONS[collection]["folder"]
    try:
        out = asyncio.run(_reconcile(collection, folder))
        log(f"reconcile[{collection}] ->\n" + out)
        return True
    except Exception as e:
        log(f"reconcile[{collection}] failed ({e!r}); will retry on next change")
        return False


def reconcile_all_with_retry(tries=20, delay=3):
    """Startup catch-up for every collection; keep trying until the server's up."""
    pending = set(COLLECTIONS)
    for _ in range(tries):
        for name in list(pending):
            if reconcile(name):
                pending.discard(name)
        if not pending:
            return
        time.sleep(delay)


def _trigger_loop():
    """Fire reconciles once events have been quiet for DEBOUNCE seconds."""
    while True:
        time.sleep(1)
        with _state:
            if not _pending or (time.time() - _last_event[0]) < DEBOUNCE:
                continue
            todo = list(_pending)
            _pending.clear()
        for name in todo:
            log(f"change settled in {name}; reconciling")
            reconcile(name)


def main():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class Handler(FileSystemEventHandler):
        def __init__(self, collection):
            self.collection = collection

        def _note(self, path):
            if path and _is_supported(path):
                with _state:
                    _pending.add(self.collection)
                    _last_event[0] = time.time()

        def on_created(self, e):  self._note(e.src_path)
        def on_modified(self, e): self._note(e.src_path)
        def on_moved(self, e):
            self._note(getattr(e, "dest_path", None)); self._note(e.src_path)
        def on_deleted(self, e):  self._note(e.src_path)

    obs = Observer()
    for name, c in COLLECTIONS.items():
        os.makedirs(c["folder"], exist_ok=True)
        obs.schedule(Handler(name), c["folder"], recursive=True)
        log(f"watching [{name}] {c['folder']}")
    obs.start()
    threading.Thread(target=_trigger_loop, daemon=True).start()

    reconcile_all_with_retry()               # startup catch-up (waits for server)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop(); obs.join()


if __name__ == "__main__":
    main()
