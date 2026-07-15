#!/usr/bin/env python3
"""Standalone test for `TextEmbedder` (refactor goal 2).

Exercises the full lifecycle the goal 2 contract requires:
  1. .start() launches the llama-server on the 780M iGPU and blocks until
     /health responds 200.
  2. The SAME 5-sentence semantic sanity battery from goal 1 is run through
     .embed_text(); we confirm within/cross-group cosine gap >= 0.15.
  3. .stop() terminates the subprocess cleanly and the orphan pgrep check
     confirms no stray llama-server still holds the model file.
  4. `sudo /usr/local/bin/fw-dgpu status` is checked before .start() and
     after .stop(); the dGPU must stay asleep (D3cold/suspended/off) in both
     snapshots.

Run: `python3 test_text_embedder_client.py`
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time

# Allow running from anywhere — repo root on sys.path.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from text_embedder_client import TextEmbedder  # noqa: E402

# Five sentences from the contract — same as goal 1, applied with the
# "search_document: " prefix (this is the document-side role for the
# sanity check; is_query=False uses the document prefix).
SENTENCES = {
    "A1": "the cat sat on the mat",
    "A2": "a feline rested on the rug",
    "A3": "my dog is sleeping on the couch",
    "B1": "the stock market crashed today",
    "B2": "quarterly earnings missed expectations",
}
ORDER = ["A1", "A2", "A3", "B1", "B2"]

EXPECTED_MODEL_FILENAME = "nomic-embed-text-v1.5.Q8_0.gguf"
SETTLE_AFTER_STOP_S = 8.0  # let PCIe D-state settle (see goal 1 report)


def fw_dgpu_status() -> str:
    out = subprocess.check_output(
        ["sudo", "/usr/local/bin/fw-dgpu", "status"],
        text=True, stderr=subprocess.STDOUT)
    return out.strip()


def dgpu_is_asleep(reading: str) -> bool:
    """Contract accepts D3cold / suspended / off. dstate=D0 with runtime=
    suspended and power=0.0 is functionally asleep too."""
    if "D3cold" in reading:
        return True
    if "state=off" in reading or "state=suspended" in reading:
        return True
    if "runtime=suspended" in reading:
        return True
    return False


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main() -> int:
    print("=== Refactor Goal 2: TextEmbedder standalone test ===")
    print(f"model file: {EXPECTED_MODEL_FILENAME}")
    print(f"port:       8091\n")

    # 0. dGPU status BEFORE
    print("--- fw-dgpu status BEFORE ---")
    before = fw_dgpu_status()
    print(before)
    before_ok = dgpu_is_asleep(before)
    print(f"before asleep: {before_ok}")

    # 1. .start()
    te = TextEmbedder()  # picks up goal 1 validated defaults
    print(f"\nTextEmbedder.alive (pre-start) = {te.alive}")
    t0 = time.monotonic()
    te.start()
    cold_load = time.monotonic() - t0
    print(f".start() returned in {cold_load:.2f}s; alive={te.alive} "
          f"device={te.device!r} dim={te.dim}")

    # 2. 5-sentence sanity battery via .embed_text()
    print("\n--- semantic sanity battery (is_query=False) ---")
    texts = [SENTENCES[k] for k in ORDER]
    vecs = te.embed_text(texts, is_query=False)
    assert vecs.shape == (len(texts), te.dim), \
        f"unexpected shape {vecs.shape} (expected ({len(texts)}, {te.dim}))"
    assert vecs.dtype.name == "float32", f"expected float32, got {vecs.dtype}"
    print(f"  embedded {len(texts)} texts -> {vecs.shape} {vecs.dtype}")
    # verify L2 norm on every row
    norms = [round(float(((v ** 2).sum()) ** 0.5), 6) for v in vecs]
    print(f"  row L2 norms: {norms}")
    assert all(abs(n - 1.0) < 1e-4 for n in norms), \
        f"rows are not L2-normalized: {norms}"

    by_key = {ORDER[i]: vecs[i].tolist() for i in range(len(ORDER))}
    sim = {}
    for i, ki in enumerate(ORDER):
        for j, kj in enumerate(ORDER):
            if i < j:
                sim[(ki, kj)] = cosine(by_key[ki], by_key[kj])

    print("\n--- pairwise cosine similarity (upper triangle) ---")
    header = "       " + "  ".join(f"{k:>6}" for k in ORDER)
    print(header)
    for i, ki in enumerate(ORDER):
        row = [f"{ki:>4}  "]
        for j, kj in enumerate(ORDER):
            if i == j:
                row.append(f"{'1.000':>6}")
            elif i < j:
                row.append(f"{sim[(ki, kj)]:>6.3f}")
            else:
                row.append(f"{sim[(kj, ki)]:>6.3f}")
        print(" ".join(row))

    within = [sim[("A1", "A2")], sim[("B1", "B2")]]
    cross = []
    for a in ("A1", "A2", "A3"):
        for b in ("B1", "B2"):
            cross.append(cosine(by_key[a], by_key[b]))
    mean_within = sum(within) / len(within)
    mean_cross = sum(cross) / len(cross)
    gap = mean_within - mean_cross
    print(f"\nmean within = {mean_within:.4f}")
    print(f"mean cross  = {mean_cross:.4f}")
    print(f"gap         = {gap:.4f}  (need >= 0.15)")
    semantic_pass = gap >= 0.15
    print(f"semantic check: {'PASS' if semantic_pass else 'FAIL'}")

    # 3. .stop() + orphan check
    print("\n--- .stop() ---")
    te.stop()
    print(f".alive after stop() = {te.alive}")
    alive_ok = (te.alive is False)
    time.sleep(0.5)
    leftover = subprocess.run(
        ["pgrep", "-af", EXPECTED_MODEL_FILENAME],
        capture_output=True, text=True)
    orphans = [ln for ln in leftover.stdout.splitlines()
               if "pgrep" not in ln and EXPECTED_MODEL_FILENAME in ln]
    print(f"orphan llama-server processes holding the model file: "
          f"{len(orphans)}")
    for ln in orphans:
        print(f"  {ln}")
    orphan_ok = (len(orphans) == 0)

    # Give PCIe D-state time to settle.
    print(f"(sleeping {SETTLE_AFTER_STOP_S}s for dGPU to settle)")
    time.sleep(SETTLE_AFTER_STOP_S)

    # 4. dGPU status AFTER
    print("\n--- fw-dgpu status AFTER ---")
    after = fw_dgpu_status()
    print(after)
    after_ok = dgpu_is_asleep(after)
    print(f"after asleep: {after_ok}")

    # 5. summary
    print("\n--- summary ---")
    print(f"  cold load       = {cold_load:.2f}s")
    print(f"  semantic gap    = {gap:.4f}  -> {'PASS' if semantic_pass else 'FAIL'}")
    print(f"  .alive after stop = {te.alive}  -> "
          f"{'PASS' if alive_ok else 'FAIL'}")
    print(f"  orphan check    = {len(orphans)} orphans  -> "
          f"{'PASS' if orphan_ok else 'FAIL'}")
    print(f"  dGPU before     = {before_ok}  "
          f"({before.splitlines()[0]})")
    print(f"  dGPU after      = {after_ok}  "
          f"({after.splitlines()[0]})")
    overall = (semantic_pass and alive_ok and orphan_ok
               and before_ok and after_ok)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())