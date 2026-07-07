#!/usr/bin/env python3
"""Validate the embedder: weights load correctly AND embeddings are semantic."""
import subprocess, sys, json, base64, time, os
import numpy as np
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
img_path = "/tmp/rag_smoke.png"
Image.new("RGB", (64, 64), (120, 30, 200)).save(img_path)

p = subprocess.Popen([sys.executable, os.path.join(here, "embedder_worker.py")],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     text=True, env=dict(os.environ))
def send(o): p.stdin.write(json.dumps(o) + "\n"); p.stdin.flush()
def recv(): return json.loads(p.stdout.readline())
def vecs(r): return np.frombuffer(base64.b64decode(r["vecs_b64"]), np.float32).reshape(r["n"], r["dim"])

t0 = time.time(); print("READY", recv().get("device"), f"(load {time.time()-t0:.1f}s)")

texts = ["ordinary least squares linear regression",
         "fitting a straight line to data by minimizing squared residuals",
         "photosynthesis converts sunlight into chemical energy in plants"]
t0 = time.time(); send({"id": 1, "op": "embed_text", "texts": texts}); r = recv()
v = vecs(r); print(f"TEXT {v.shape} {time.time()-t0:.2f}s norm={np.linalg.norm(v[0]):.3f}")
sim01 = float(v[0] @ v[1]); sim02 = float(v[0] @ v[2])
print(f"  sim(regression, OLS-desc) = {sim01:.3f}   <-- should be HIGH")
print(f"  sim(regression, photosyn) = {sim02:.3f}   <-- should be LOWER")
print(f"  SEMANTIC OK: {sim01 > sim02 + 0.05}")

t0 = time.time(); send({"id": 2, "op": "embed_image", "paths": [img_path]}); r = recv()
print(f"IMAGE ok={r.get('ok')} dim={r.get('dim')} {time.time()-t0:.2f}s err={r.get('error')}")
p.stdin.close(); p.terminate()
