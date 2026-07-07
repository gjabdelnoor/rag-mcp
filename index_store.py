"""Barebones vector store: a float16 matrix (.npy) + a JSONL sidecar of metadata.

Small enough that the MCP server can mmap the matrix and do an exact cosine
search with numpy in the parent process — no torch, no GPU, tiny RAM. The store
holds L2-normalized vectors, so cosine == dot product.

Writes are atomic (temp file + os.replace) and guarded by an flock on
`<root>/.lock`, so a manual `ingest.py` run cannot corrupt the index while the
server is reading it, and vice versa.
"""
import os, json, fcntl, contextlib
import numpy as np


class IndexStore:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.vecs_path = os.path.join(root, "embeddings.npy")
        self.meta_path = os.path.join(root, "meta.jsonl")
        self.lock_path = os.path.join(root, ".lock")

    @contextlib.contextmanager
    def _lock(self, exclusive):
        f = open(self.lock_path, "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    def _save_atomic(self, vecs):
        tmp = self.vecs_path + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, np.ascontiguousarray(vecs, dtype=np.float16))
        os.replace(tmp, self.vecs_path)

    def _load_vecs(self):
        if not os.path.exists(self.vecs_path):
            return None
        return np.load(self.vecs_path)

    # ---- ingestion side (write) ----
    def append(self, vecs, metas):
        """vecs: (n, dim) float32 normalized. metas: list of n dicts."""
        vecs = np.ascontiguousarray(vecs, dtype=np.float16)
        with self._lock(exclusive=True):
            old = self._load_vecs()
            if old is not None:
                vecs = np.concatenate([old, vecs], axis=0)
            self._save_atomic(vecs)
            with open(self.meta_path, "a") as f:
                for m in metas:
                    f.write(json.dumps(m) + "\n")

    def remove_source(self, source):
        """Delete every vector+meta whose meta['source'] == source. Returns the
        number of rows removed. Rewrites both files atomically under the lock."""
        with self._lock(exclusive=True):
            if not os.path.exists(self.meta_path):
                return 0
            metas = [json.loads(l) for l in open(self.meta_path)]
            keep = [i for i, m in enumerate(metas) if m.get("source") != source]
            removed = len(metas) - len(keep)
            if removed == 0:
                return 0
            mat = self._load_vecs()
            if mat is not None:
                self._save_atomic(mat[keep])
            tmp = self.meta_path + ".tmp"
            with open(tmp, "w") as f:
                for i in keep:
                    f.write(json.dumps(metas[i]) + "\n")
            os.replace(tmp, self.meta_path)
            return removed

    def reset(self):
        with self._lock(exclusive=True):
            for p in (self.vecs_path, self.meta_path):
                if os.path.exists(p):
                    os.remove(p)

    # ---- query side (read) ----
    def count(self):
        if not os.path.exists(self.meta_path):
            return 0
        with self._lock(exclusive=False), open(self.meta_path) as f:
            return sum(1 for _ in f)

    def sources(self):
        """Map source-path -> number of indexed chunks."""
        out = {}
        if not os.path.exists(self.meta_path):
            return out
        with self._lock(exclusive=False), open(self.meta_path) as f:
            for line in f:
                s = json.loads(line).get("source")
                out[s] = out.get(s, 0) + 1
        return out

    def search(self, qvec, k=5):
        with self._lock(exclusive=False):
            mat = self._load_vecs()
            if mat is None or mat.shape[0] == 0:
                return []
            q = np.asarray(qvec, dtype=np.float32).reshape(-1)
            sims = (mat.astype(np.float32) @ q)                 # cosine (normalized)
            k = min(k, sims.shape[0])
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            metas = self._read_metas(set(top.tolist()))
        return [{"score": float(sims[i]), **metas[i]} for i in top]

    def _read_metas(self, idxs):
        out = {}
        with open(self.meta_path) as f:
            for i, line in enumerate(f):
                if i in idxs:
                    out[i] = json.loads(line)
        return out
