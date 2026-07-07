# PROMOTION PLAN — apply the embedding speedup to production (needs Gabriel's OK)

Phase A succeeded: the full-resolution native video path now runs on the 7700S at
**~8× the still-path baseline, zero fidelity loss**. Nothing below has been applied to
production yet — I'm holding it because it touches the live `rag-mcp` daemon and changes
embedding *semantics*. Two tiers, apply on approval:

## Tier 0 — CPU query-embedding (fixes RAG-search-times-out while the 35B is awake)
OBSERVED in the Aleph Chat Peer: `rag_search` timed out at 45s. Root cause is GPU
contention, not the app — with the 35B llama-server resident on the 7700S (8 GB), the
daemon can't load the 2B embedder to embed the *query*, so the search hangs → times out.
Fix (daemon-side, low risk): embed **queries** on CPU. A single text query embed is tiny
(~1–3 s on CPU) and never needs the GPU, so searches stop colliding with the resident LLM.
Keep image/ingest embeds on GPU (throughput path). Where: `embedder_worker.py` / the query
path in `server.py::search` — add a `device="cpu"` fast lane for text-query embeds, or a
tiny always-CPU text encoder. Verify: run a `rag search` while the 35B is loaded → returns
in seconds. This makes the whole KB (and Aleph) usable concurrently with the Peer.

## Tier 1 — the safe reliability fix (low risk, recommend applying first)
Root cause of the ORIGINAL journal-ingest stall (216/2692) was almost certainly MIOpen's
exhaustive conv search hanging at 720p. Fix = a single env var; no semantic change.
- Add to `rag-mcp.service` (drop-in `systemctl --user edit rag-mcp.service`), or to
  `embedder_worker.py` env at top:
  - `MIOPEN_FIND_MODE=2`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Then `systemctl --user restart rag-mcp.service` (brief KB downtime).
- Effect: the still-image embed path stops hanging; existing per-frame ingest becomes
  reliable. This alone lets the current ingest finish (slowly).

## Tier 2 — the 8× video-path speedup (semantic change → your call on W)
Changes how video-frame dirs embed: from one vector PER FRAME to one vector PER W-FRAME
WINDOW (temporal merge). Faster + coarser retrieval granularity. Recommended `W=16`
(8×, ~21 min for a 45-min video); `W=8` is finer (4×, ~41 min). YOUR CALL on W.
Files (all in `~/projects/rag-mcp/`), promote Kimi's `kimi_kernel/` prototypes:
1. `embedder_worker.py`: after model load, `install_patch_embed_kernel(embedder.model)`
   (from `patch_embed_kernel.py`); add an `embed_video_windows` op wrapping `video_embed.py`'s
   windowed path.
2. `worker_client.py`: add `embed_video_windows(frame_dir, W=16)`.
3. `ingest.py::ingest_frame_dir`: call the window path instead of per-image `embed_image`;
   write per-window metadata (t_start,t_end, union captions, representative frame).
4. `server.py` search: already handles image modality; confirm window records render.
Keep `patch_embed_kernel.py` etc. vendored in rag-mcp (not kimi_kernel) once promoted.

## Tier 3 — re-ingest the journal
After Tier 1+2: `remove_source` the Personal frames + re-ingest
`~/Documents/Personal/*.frames` → ~21 min at W=16. The stalled 216 partial vectors are
replaced. Then the journal video is fully queryable (text + frames).

## Verification after promotion
- `test_kernel.py` cos ≥ 0.9999 gate still passes in-daemon.
- A search for a known moment returns the right window + frame image.
- Watch `journalctl --user -u rag-mcp.service` for `MIOPEN_FIND_MODE` + no hang.
