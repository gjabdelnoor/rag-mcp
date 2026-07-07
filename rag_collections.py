"""Named vector stores ("collections" / "notebooks").

Each collection is a watched folder + its own on-disk index + a human
description. The config lives at ``collections.json`` (override with
``RAG_COLLECTIONS``); a sensible default is written on first run. Every
collection's index lives under ``RAG_INDEX_ROOT/<name>/`` so they never share
vectors.

A collection name is a short slug (``STATF401``); the description is what the
``list_collections`` tool shows the LLM so it can pick the right store.
"""
import os, json

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.environ.get("RAG_COLLECTIONS", os.path.join(ROOT, "collections.json"))
INDEX_ROOT = os.environ.get("RAG_INDEX_ROOT", os.path.join(ROOT, "index"))

DEFAULTS = {
    "STATF401": {
        "folder": "/home/gabriel/Documents/STATF401/Materials",
        "description": "STAT-F401 course materials: lecture transcripts, "
                       "lecture slides (OCR'd), and the Weisberg *Applied "
                       "Linear Regression* textbook. Ask here for anything "
                       "about the statistics/regression class.",
    },
    "workbook": {
        "folder": "/home/gabriel/Documents/Gabe's Workbook",
        "description": "Gabe's Workbook: general scientific papers, technical "
                       "notes, and scripts collected for reference. Ask here "
                       "for research papers and general science/engineering "
                       "material not tied to a specific course.",
    },
}


def _save(data):
    tmp = CONFIG + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, CONFIG)


def load():
    """Return {name: {folder, description, index_dir}} with paths normalized."""
    if os.path.exists(CONFIG):
        data = json.load(open(CONFIG))
    else:
        data = {k: dict(v) for k, v in DEFAULTS.items()}
        _save(data)
    out = {}
    for name, c in data.items():
        out[name] = {
            "folder": os.path.expanduser(c["folder"]),
            "description": c.get("description", ""),
            "index_dir": os.path.join(INDEX_ROOT, name),
        }
    return out


def folder_to_collection(collections, path):
    """Which collection's watched folder contains `path` (or None)."""
    ap = os.path.abspath(path)
    best = None
    for name, c in collections.items():
        f = os.path.abspath(c["folder"])
        if ap == f or ap.startswith(f + os.sep):
            if best is None or len(f) > len(collections[best]["folder"]):
                best = name
    return best
