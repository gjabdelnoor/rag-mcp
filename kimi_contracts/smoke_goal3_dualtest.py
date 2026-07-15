#!/usr/bin/env python3
"""Smoke test for goal 3 dual-store wiring.

Adds a throwaway `__dualtest__` collection to `collections.json`, points it at
`/tmp/rag_dualtest/`, drops one tiny .txt + one tiny PNG in there, calls the
new `server.ingest_path` programmatically (so we don't have to restart the
live `rag-mcp.service`), confirms the text store got exactly 1 vector AND the
image store got exactly 1 vector, then removes `__dualtest__` from
`collections.json` again (the `index/__dualtest__/` dir is left behind as
harmless debris — same precedent as the existing `__smoke__*` dirs).

Imports `server` fresh in this subprocess, so the live rag-mcp Python (which
still has the pre-refactor code loaded) is unaffected.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

REPO = "/home/gabriel/projects/rag-mcp"
COLLECTIONS_JSON = os.path.join(REPO, "collections.json")
TEST_FOLDER = "/tmp/rag_dualtest"
TEST_INDEX_DIR = os.path.join(REPO, "index", "__dualtest__")

TXT_PATH = os.path.join(TEST_FOLDER, "hello.txt")
PNG_PATH = os.path.join(TEST_FOLDER, "hello.png")

# How long we wait for the VL-2B image embedder cold start (15s is the goal's
# own budget hint from server.py's idle docstring).
IMAGE_EMB_READY_S = 90


def make_test_files():
    os.makedirs(TEST_FOLDER, exist_ok=True)
    if os.path.exists(TXT_PATH):
        os.remove(TXT_PATH)
    with open(TXT_PATH, "w") as f:
        f.write("the cat sat on the mat\n")
    # Remove any stale png
    if os.path.exists(PNG_PATH):
        os.remove(PNG_PATH)
    # Make a tiny synthetic PNG with PIL (a solid-color 32x32 square).
    from PIL import Image
    img = Image.new("RGB", (32, 32), color=(180, 60, 60))
    img.save(PNG_PATH, format="PNG")
    print(f"  created {TXT_PATH} ({os.path.getsize(TXT_PATH)} bytes)")
    print(f"  created {PNG_PATH} ({os.path.getsize(PNG_PATH)} bytes)")


def add_dualtest_to_collections_json():
    with open(COLLECTIONS_JSON, "r") as f:
        data = json.load(f)
    data["__dualtest__"] = {
        "folder": TEST_FOLDER,
        "description": "smoke test for goal 3 dual-store wiring",
    }
    tmp = COLLECTIONS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, COLLECTIONS_JSON)
    print("  added __dualtest__ to collections.json")


def remove_dualtest_from_collections_json():
    with open(COLLECTIONS_JSON, "r") as f:
        data = json.load(f)
    data.pop("__dualtest__", None)
    tmp = COLLECTIONS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, COLLECTIONS_JSON)
    print("  removed __dualtest__ from collections.json")


def reset_dualtest_index():
    """Clear any leftover embeddings so the count is deterministic. Also reset
    the in-memory stores below."""
    for sub in ("text", "image"):
        d = os.path.join(TEST_INDEX_DIR, sub)
        if not os.path.isdir(d):
            continue
        for fn in ("embeddings.npy", "meta.jsonl"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                os.remove(p)
    # Don't touch the top-level __dualtest__/manifest.json so an existing
    # manifest can prove INGEST_VERSION=4 was bumped. We'll wipe it so the
    # test is self-contained.
    if os.path.isdir(TEST_INDEX_DIR):
        mp = os.path.join(TEST_INDEX_DIR, "manifest.json")
        if os.path.exists(mp):
            os.remove(mp)


def fw_dgpu_status():
    return subprocess.check_output(
        ["sudo", "/usr/local/bin/fw-dgpu", "status"],
        text=True, stderr=subprocess.STDOUT).strip()


def main() -> int:
    print("=== Refactor Goal 3: __dualtest__ smoke test ===")

    # 0. Reset index dir for a clean baseline (manifest.json+stores)
    reset_dualtest_index()

    # 1. Add __dualtest__ entry, create test files
    add_dualtest_to_collections_json()
    make_test_files()

    # 2. dGPU BEFORE
    print("\n--- fw-dgpu status BEFORE ---")
    before = fw_dgpu_status()
    print(before)

    # 3. Import server.py (fresh COLLECTIONS dict from the modified
    #    collections.json)
    sys.path.insert(0, REPO)
    print("\n--- importing server (fresh) ---")
    import server
    print(f"  COLLECTIONS keys: {sorted(server.COLLECTIONS.keys())}")
    print(f"  '__dualtest__' in text_stores: "
          f"{'__dualtest__' in server.text_stores}")
    print(f"  '__dualtest__' in image_stores: "
          f"{'__dualtest__' in server.image_stores}")
    print(f"  text_stores['__dualtest__'].root = "
          f"{server.text_stores['__dualtest__'].root}")
    print(f"  image_stores['__dualtest__'].root = "
          f"{server.image_stores['__dualtest__'].root}")
    print(f"  INGEST_VERSION = {server.INGEST_VERSION}")

    # Pre-condition: both stores are empty.
    print(f"\n  text store count pre:  {server.text_stores['__dualtest__'].count()}")
    print(f"  image store count pre: {server.image_stores['__dualtest__'].count()}")

    # 4. Call ingest_path. This will cold-start text_emb (~3s on 780M) AND
    #    cold-start emb (~15s on 7700S, the big VL-2B).
    print("\n--- calling server.ingest_path(collection='__dualtest__') ---")
    t0 = time.monotonic()
    try:
        result = server.ingest_path(collection="__dualtest__", path=TEST_FOLDER,
                                    recursive=True)
    except Exception as e:
        print(f"ingest_path raised: {e!r}")
        result = f"<exception: {e!r}>"
    dt = time.monotonic() - t0
    print(f"  ingest_path returned in {dt:.2f}s")
    print("--- ingest_path output ---")
    print(result)
    print("--- end ingest_path output ---")

    # 5. Verify counts
    txt_n = server.text_stores["__dualtest__"].count()
    img_n = server.image_stores["__dualtest__"].count()
    print(f"\n  text store count post:  {txt_n}")
    print(f"  image store count post: {img_n}")
    print(f"  text store sources: {server.text_stores['__dualtest__'].sources()}")
    print(f"  image store sources: {server.image_stores['__dualtest__'].sources()}")

    expected_txt = 1
    expected_img = 1
    text_pass = (txt_n == expected_txt)
    image_pass = (img_n == expected_img)
    overall_pass = text_pass and image_pass
    print(f"\n  text count = {txt_n} (expected {expected_txt}): "
          f"{'PASS' if text_pass else 'FAIL'}")
    print(f"  image count = {img_n} (expected {expected_img}): "
          f"{'PASS' if image_pass else 'FAIL'}")

    # 6. Verify vector dimensions
    if txt_n == 1:
        import numpy as np
        v = np.load(server.text_stores["__dualtest__"].vecs_path)
        print(f"  text vector shape={v.shape}, dtype={v.dtype}, "
              f"L2-norm={float(np.linalg.norm(v[0].astype(np.float32))):.4f}")
    if img_n == 1:
        import numpy as np
        v = np.load(server.image_stores["__dualtest__"].vecs_path)
        print(f"  image vector shape={v.shape}, dtype={v.dtype}, "
              f"L2-norm={float(np.linalg.norm(v[0].astype(np.float32))):.4f}")

    # 7. dGPU AFTER (the VL-2B image embedder was warmed on the 7700S, so the
    #    dGPU WILL be awake after ingest_path; we just check it goes back
    #    after .stop()).
    print("\n--- fw-dgpu status AFTER (embedder still warm) ---")
    print(fw_dgpu_status())

    # 8. Clean shutdown of both embedders (so the live system is unaffected
    #    and the dGPU can go back to D3cold)
    print("\n--- stopping both embedders ---")
    if server.text_emb.alive:
        server.text_emb.stop()
    if server.emb.alive:
        server.emb.stop()
    time.sleep(8.0)  # let PCIe settle
    print("\n--- fw-dgpu status AFTER stop ---")
    print(fw_dgpu_status())

    # 9. Remove __dualtest__ from collections.json (cleanup)
    print("\n--- cleanup ---")
    remove_dualtest_from_collections_json()

    # 10. Final summary
    print("\n--- summary ---")
    print(f"  text store count  = {txt_n} (expected 1):  "
          f"{'PASS' if text_pass else 'FAIL'}")
    print(f"  image store count = {img_n} (expected 1):  "
          f"{'PASS' if image_pass else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())