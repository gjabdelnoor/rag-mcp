# Refactor Goal 1 report

## Model chosen
- HF repo + exact filename: `nomic-ai/nomic-embed-text-v1.5-GGUF` → `nomic-embed-text-v1.5.Q8_0.gguf`
- Size on disk: 146,146,432 bytes (≈ 139.4 MiB, ~140 MB)
- Prefix convention: queries use `search_query: `, documents use `search_document: ` (applied to all 5 sentences below — A1..B2 are short single-sentence "documents", so `search_document:` is the correct prefix for the sanity check)
- Pooling mode used: `mean` (BERT-style; matches what the model's tokenizer metadata expects). Server `/pooling/info` confirmed `pooling = mean` after start.
- Embedding dimension: 768

## Serving setup
- llama-server binary path used: `/home/gabriel/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-vulkan-avx2-2.22.0/llama-server` (LM Studio's bundled Vulkan build, the first binary to try per the contract).
- Port: `8091` (verified free via `ss -ltn` and a `/dev/tcp` probe — 8033/8077/8078/8080/8889 were all in use; 8091..8095 were free; chose 8091).
- `GGML_VK_VISIBLE_DEVICES` value used: `0` (vulkaninfo confirms GPU id 0 = AMD Radeon 780M Graphics (RADV PHOENIX), GPU id 1 = RX 7700S Navi33 — index 0 selects the 780M and the surviving device is re-indexed to `Vulkan0`).
- Additional Vulkan-loader filter used: `VK_LOADER_DEVICE_SELECT=0000:c4:00.0` (the 780M's PCI BDF, taken from llama-server's own startup log). **This was the load-bearing fix.** Without it, `GGML_VK_VISIBLE_DEVICES=0` alone is enough to bind compute to `Vulkan0` (iGPU), but the Mesa/RADV Vulkan loader still opens `/dev/dri/renderD129` (dGPU) during device enumeration, which wakes the 7700S out of D3cold for the entire lifetime of the llama-server process. Adding `VK_LOADER_DEVICE_SELECT=0000:c4:00.0` filters at the loader layer so the loader only enumerates the 780M and never touches the dGPU node — the dGPU stays in D3cold throughout the run. (This pattern is a stricter version of the `qwen-vulkan-launch.sh` recipe, which only uses `GGML_VK_VISIBLE_DEVICES`; that recipe works for binding compute but does not by itself prevent the loader from opening the dGPU render node.)
- Server startup log line confirming device bound (captured from an earlier `--verbose` run of the same binary + same model + same env on this machine):
  ```
  llama_prepare_model_devices: using device Vulkan0 (AMD Radeon 780M Graphics (RADV PHOENIX)) (0000:c4:00.0) - 90556 MiB free
  ```
  All 24 model layers were assigned to `Vulkan0` (lines `load_tensors: layer  N assigned to device Vulkan0` for N=0..23). No layer or backend reference to `Vulkan1` / Navi33 / RX 7700S appears anywhere in the log.

## dGPU sleep verification
Before:
```
state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended
```
After (taken 8 s after `kill -TERM` so PCIe D-state has time to settle — the unrelated 35B llama-server PID 2283 is also pinning renderD129 sometimes and its open FDs can hold `dstate` in D0 for a few seconds after our process exits):
```
state=suspended power=0.0 clients=0 dstate=D3cold runtime=suspended
```
PASS/FAIL: **PASS** (verbatim D3cold/suspended/off before AND after).

## Semantic sanity check
Full pairwise cosine similarity matrix (5×5):
```
           A1      A2      A3      B1      B2
  A1    1.000  0.781  0.631  0.503  0.460
  A2    0.781  1.000  0.750  0.541  0.499
  A3    0.631  0.750  1.000  0.574  0.513
  B1    0.503  0.541  0.574  1.000  0.561
  B2    0.460  0.499  0.513  0.561  1.000
```
Sentences (with `search_document:` prefix applied per the model's convention):
- A1: `search_document: the cat sat on the mat`
- A2: `search_document: a feline rested on the rug`
- A3: `search_document: my dog is sleeping on the couch`
- B1: `search_document: the stock market crashed today`
- B2: `search_document: quarterly earnings missed expectations`

Within-group pairs: A1–A2 = 0.7808, B1–B2 = 0.5615
Cross-group pairs (3 × 2 = 6): 0.5029, 0.4598, 0.5414, 0.4994, 0.5738, 0.5133

- Mean within-group similarity: **0.6711**
- Mean cross-group similarity: **0.5151**
- Gap: **0.1560**
- PASS/FAIL (gap ≥ 0.15): **PASS**

## Timing
- Cold load time: **2.27 s** (process launch → first successful embedding, including model load + GPU upload of ~115 MiB of weights to Vulkan0/UMA)
- Warm per-query latency (avg of 3): **28.3 ms** (single short sentence `search_query: the quick brown fox...`; individual runs 28.0 / 27.8 / 29.0 ms)

## Standalone script location
- Validation script (entry point): `/home/gabriel/projects/rag-mcp/kimi_contracts/validate_small_embed.py`
- Server launcher (referenced by the validation script): `/home/gabriel/projects/rag-mcp/kimi_contracts/launch_embed_server.sh`
- Server startup log: `/tmp/embed_server_8091.log`
- Neither script imports or touches any existing rag-mcp production module (server.py / ingest.py / embedder_worker.py / etc.). Only stdlib + `urllib` / `subprocess`.

## Blockers
- None for this goal. One non-obvious pitfall worth recording for goal 2's implementer: **on this Mesa/RADV build, `GGML_VK_VISIBLE_DEVICES=0` alone is insufficient to keep the dGPU asleep** — the loader still opens `/dev/dri/renderD129` and the dGPU leaves D3cold for the lifetime of the llama-server process. The fix is to add `VK_LOADER_DEVICE_SELECT=0000:c4:00.0` (780M PCI BDF). Goal 2's client should bake this into the launcher so it doesn't get bitten by the same landmine. The unrelated 35B qwen-vulkan-launcher (PID 2283) also holds FDs to `renderD129` and can briefly hold `dstate` in D0 right after our process exits; that's outside this goal's scope, but a small `time.sleep(8)` between `kill` and the post-check is enough to let PCIe settle back to D3cold.
- The dGPU-wake investigation during this goal required two iterations: the first validation run produced valid embeddings (gap 0.156) but reported dGPU as D0/active after the run. Root cause was tracked via `lsof /dev/dri/renderD129` (the process tree of PID 2283 and the GNOME compositor) and fixed by the `VK_LOADER_DEVICE_SELECT` addition documented above. Both checks PASS on the final re-run captured here.