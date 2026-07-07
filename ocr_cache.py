"""Disk cache of OCR results, keyed by file content hash.

OCR is the expensive step (a VLM pass per page on the GPU), so we never want to
redo it for a file we've already read. Results are stored as JSON under a
collection's index dir at ``ocr_cache/<sha1>.json`` — keyed by the file's
content hash, so an unchanged file is a cache hit even if it moved/renamed, and
an edited file (new hash) misses and is re-OCR'd.
"""
import os, json

VERSION = 2   # bump to invalidate all cached OCR (e.g. model/prompt change)


def _dir(index_dir):
    return os.path.join(index_dir, "ocr_cache")


def path_for(index_dir, file_hash):
    return os.path.join(_dir(index_dir), f"{file_hash}.json")


def get(index_dir, file_hash):
    p = path_for(index_dir, file_hash)
    if not os.path.exists(p):
        return None
    try:
        data = json.load(open(p))
        if data.get("version") == VERSION:
            return data.get("pages")
    except Exception:
        return None
    return None


def put(index_dir, file_hash, pages, meta=None):
    d = _dir(index_dir)
    os.makedirs(d, exist_ok=True)
    p = path_for(index_dir, file_hash)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": VERSION, "pages": pages, **(meta or {})}, f)
    os.replace(tmp, p)
