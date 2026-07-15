# Refactor Goal 2 report

## File created
- Path: `~/projects/rag-mcp/text_embedder_client.py`
- Class name: `TextEmbedder`
- Confirmed interface: `.alive` (property), `.start(ready_timeout=120)`,
  `.stop()`, `.embed_text(texts, is_query=False) -> np.ndarray`
  shape `(len(texts), dim)` dtype `float32` L2-normalized
- Standalone test: `~/projects/rag-mcp/test_text_embedder_client.py`
  (run with `python3 test_text_embedder_client.py`)
- Importing the module pulls in ZERO torch/surya modules — verified via
  `python3 -c 'import text_embedder_client; [m for m in sys.modules if "torch" in m]'`
  returning empty. Only stdlib (`urllib`, `subprocess`, `socket`, `signal`,
  `json`, `time`, `shutil`) + `numpy` (same as `worker_client.py`) are loaded.

## Test run (fresh, this session)
Full stdout:
```
=== Refactor Goal 2: TextEmbedder standalone test ===
model file: nomic-embed-text-v1.5.Q8_0.gguf
port:       8091

--- fw-dgpu status BEFORE ---
state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended
before asleep: True

TextEmbedder.alive (pre-start) = False
.start() returned in 2.30s; alive=True device='780M (Vulkan0)' dim=768

--- semantic sanity battery (is_query=False) ---
  embedded 5 texts -> (5, 768) float32
  row L2 norms: [1.0, 1.0, 1.0, 1.0, 1.0]

--- pairwise cosine similarity (upper triangle) ---
           A1      A2      A3      B1      B2
  A1    1.000  0.781  0.631  0.503  0.460
  A2    0.781  1.000  0.750  0.542  0.499
  A3    0.631  0.750  1.000  0.573  0.513
  B1    0.503  0.542  0.573  1.000  0.561
  B2    0.460  0.499  0.513  0.561  1.000

mean within = 0.6712
mean cross  = 0.5151
gap         = 0.1561  (need >= 0.15)
semantic check: PASS

--- .stop() ---
.alive after stop() = False
orphan llama-server processes holding the model file: 0
(sleeping 8.0s for dGPU to settle)

--- fw-dgpu status AFTER ---
state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended
after asleep: True

--- summary ---
  cold load       = 2.30s
  semantic gap    = 0.1561  -> PASS
  .alive after stop = False  -> PASS
  orphan check    = 0 orphans  -> PASS
  dGPU before     = True  (state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended)
  dGPU after      = True  (state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended)

OVERALL: PASS
```

- Semantic gap (within vs cross group): **PASS**, gap = **0.1561** (within mean 0.6712, cross mean 0.5151, ≥ 0.15)
- Orphan-process check after `.stop()`: **PASS** (0 orphans; `pgrep -af nomic-embed-text-v1.5.Q8_0.gguf` returned empty after `.stop()` and the 8s settle)
- dGPU status before `.start()`: `state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended`
- dGPU status after `.stop()`: `state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended`
- dGPU sleep PASS/FAIL: **PASS** (verbatim identical D3cold/suspended/off in both snapshots)

## Deviations from goal 1's validated setup (if any, and why)
None. The client reuses goal 1's validated choices as-is:
- model file: `~/projects/rag-mcp/models/nomic-embed-text-v1.5.Q8_0.gguf`
- binary: `~/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-vulkan-avx2-2.22.0/llama-server` (auto-resolved via `_find_llama_server()`, same precedence chain as `ocr_client.find_llama_server()`)
- port: `8091` (verified free at start time via bind-and-release probe; raises a clear `RuntimeError` if occupied instead of silently colliding)
- pooling: `mean`
- ctx_size: `512`
- device pin: `GGML_VK_VISIBLE_DEVICES=0` AND `VK_LOADER_DEVICE_SELECT=0000:c4:00.0` (both set via `env.setdefault` if not already in the environment, matching the validated recipe)
- prefix convention: `search_query: ` when `is_query=True`, `search_document: ` otherwise (nomic-embed-text-v1.5)

The launch is delegated to the existing `kimi_contracts/launch_embed_server.sh` script (created in goal 1), so all of goal 1's device-pin / port-handling / startup-log logic stays in one place.

Two small additions beyond the bare contract requirements:
1. **Port-free pre-check** (`_port_is_free`, bind-and-release + connect-probe) — the contract asked for this; it raises a clear `RuntimeError` if the port is taken rather than letting llama-server's own bind error propagate as a less helpful crash.
2. **Dim probe at startup** (`.embed_text` shape) — `_probe_server()` embeds a one-token probe after `/health` is 200 to discover `dim` from the running model, so the client's `self.dim` reflects what the model actually produces rather than a hardcoded guess.

## Blockers
None.