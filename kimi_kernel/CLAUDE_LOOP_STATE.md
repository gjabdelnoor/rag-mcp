# CLAUDE_LOOP_STATE — what "Proceed." means on each 30-min loop fire

Role: Claude is delegator/reviewer. Kimi (local CLI, tmux session `kimi`) is the coder.
Project: make Qwen3-VL frame embedding >5× faster with ZERO fidelity loss, via the
native video (temporal-merge) path → multi-GPU parallelism → custom RDNA3 kernel.

## On each "Proceed." fire, do this:
1. `tmux capture-pane -t kimi -p -S -60` — see what Kimi is doing / its goal status bar
   (`[goal ● active · Ns · N turns]`), and whether the current goal completed.
2. Check for new deliverables: `kimi_kernel/REPORT_1.md`, `REPORT_2.md`, `REPORT_3.md`
   and the code (`video_embed.py`, `parallel_embedder.py`, kernel files).
3. If a REPORT exists → REVIEW its code (read the .py), sanity-check the two laws were
   honored (NO spatial downscale; fidelity intact; correctness test passed). Note issues.
4. If Kimi is stuck/broke something → repair or write a corrective contract and re-`/goal`.
5. If a goal finished cleanly → the next queued goal auto-starts; if the queue is empty,
   write the next escalating contract and `/goal next "<summary>, see contract at <path>"`.
6. Keep the live user embed run + all servers UNTOUCHED (see TECH_CONTEXT clause 4).
7. Then end the turn (the loop brings you back in 30 min). Don't poll in a tight wait.

## Full backlog (queue in this order, one `/goal next` at a time when Kimi is idle):
PHASE A — embedding optimization (folder `/home/gabriel/projects/rag-mcp/kimi_kernel/`):
- KIMI_CONTRACT_1.md — native video-path windowed embed (headline). REPORT_1.md.
- KIMI_CONTRACT_2.md — multi-GPU data-parallel pool. REPORT_2.md.
- KIMI_CONTRACT_3.md — custom RDNA3 attention kernel + AOTriton. REPORT_3.md.
PHASE B — Aleph desktop app (folder `/home/gabriel/projects/aleph/`, spec = APP_SPEC.md):
"integrate all my knowledge bases and MCPs into a custom desktop app as a .deb; surprise
me with the quality." Client of the existing daemons; PySide6; no service kills; no GPU.
- KIMI_CONTRACT_APP_1.md — foundation + multimodal RAG search view. REPORT_APP_1.md.
- KIMI_CONTRACT_APP_2.md — chat "Peer" with tool-calling into KBs/MCPs. REPORT_APP_2.md.
- KIMI_CONTRACT_APP_3.md — Sources browser + Journal timeline + polish. REPORT_APP_3.md.
- KIMI_CONTRACT_APP_4.md — package installable aleph_*.deb. REPORT_APP_4.md.
Backend map + collections + endpoints are VERIFIED inside APP_SPEC.md — trust it.

## Driving Kimi over tmux (learned gotchas):
- Run `tmux set -g extended-keys on` once; without it Enter becomes a newline and
  the message never submits (it just piles up in the input box).
- ONLY submit a `/goal` or `/goal next` when Kimi is IDLE at the prompt (no
  `[goal ● active]` spinner). Sending mid-run stuffs the input box but does NOT
  submit; a later stray Enter can then fire a malformed message.
- If text is stuck in the box, clear it with several `tmux send-keys -t kimi C-u`
  before doing anything else.
- Queue backlog status (update each tick):
  - CONTRACT 1 — DONE + reviewed PASS (video_embed.py + REPORT_1.md; fidelity kept,
    selftest passed, GPU bench was deferred-for-VRAM).
  - CONTRACT 2 — DONE (REPORT_2.md + parallel_embedder.py). KEY FINDING: the native video
    path is numerically CORRECT on the 7700S (cosine 1.0003 GPU-vs-CPU) but HANGS at full
    resolution (MIOpen conv algorithm-search "no suitable algorithm" above ~204,800 px/frame)
    and OOMs at W=32 (attention wants ~9.9 GiB on 8 GB). Kimi correctly REFUSED to downscale
    (Law 1). So no GPU throughput number yet — the whole optimization is now GATED on fixing
    the MIOpen hang. DISCLOSED VIOLATION: Kimi `kill -9`'d a hung transient embedder_worker to
    free VRAM — reviewed: NO lasting harm (daemon PID 603194 alive, NRestarts=0; it was a
    transient hung worker, not the service worker). Reinforced an ABSOLUTE no-kill clause in
    CONTRACT 3. NOTE: the user's stalled yt-dlpcc has now fully ended (gone) at 216/2692.
  - CONTRACT 3 — REDEFINED via AMENDMENT at top of KIMI_CONTRACT_3.md: it is now THE UNBLOCK
    (make full-res video path complete on 7700S). Priority: (1) MIOPEN_FIND_MODE=2/3/5 env
    (cheapest, untried), (2) replace vision patch-embed conv with unfold+matmul to bypass
    MIOpen, (3) chunked/flash attention for the W=32 OOM. Correctness gate cosine ≥ 0.9999.
    GPU-FREE WORK DONE + reviewed (REPORT_3.md): shipped patch_embed_kernel.py — an
    unfold+matmul replacement for the MIOpen Conv3d, CPU cos=1.000000 (bit-identical,
    fidelity intact), 2.12× faster on CPU; plus miopen_harness.py (MIOPEN_FIND_MODE sweep)
    + test/poll scripts. Honored no-kill perfectly (deferred, never signaled the daemon
    worker that held 6.31GB the whole poll window).
    GPU VALIDATION: at tick 5 the daemon worker had idle-died → GPU[0] FREE (27MB). Sent
    an URGENT follow-up telling Kimi to run miopen_harness.py + the patch_embed kernel on
    cuda:0 at full res (720x720 + W=8/16) NOW and APPEND results to REPORT_3.md under a
    "GPU VALIDATION" section. Kimi is running it. THIS IS THE PIVOTAL RESULT of the night:
    does the conv-bypass (or MIOPEN_FIND_MODE) make the full-res video path COMPLETE on
    the 7700S? If yes → ~50-100× over still, journal-embed works at full fidelity.
    ✅ SUCCESS (REPORT_3.md §11 GPU VALIDATION): MIOPEN_FIND_MODE=2 unblocks the hang
    (one-line env fix; modes 3/5 still hang) AND the matmul patch-embed kernel is GPU
    cos=1.000000 at full res. W=16+matmul = 2.1 fps = 8.0× over still baseline, ~21 min
    for the 2692-frame corpus, ZERO fidelity loss. 5× target EXCEEDED. All Kimi procs
    exited cleanly, daemon untouched.
  ==> PHASE A (embedding optimization) COMPLETE & SUCCESSFUL. Promotion into production is
      NOT done — held for the user's approval (touches live rag-mcp daemon + changes embed
      semantics per-frame→per-window). One-click plan written: kimi_kernel/PROMOTION_PLAN.md
      (Tier 1 = safe MIOPEN_FIND_MODE=2 env fix; Tier 2 = 8× video path, user picks W;
      Tier 3 = re-ingest Personal ~21min). WHEN USER WAKES: walk them through PROMOTION_PLAN.
  - PHASE B (Aleph app, /home/gabriel/projects/aleph/):
    - APP_1 — DONE + reviewed PASS. Polished PySide6 dark UI, nav rail, Search view wired to
      live RAG :8077 (cards + frame gallery + get_book_image), collections discovered (edges
      filtered), LLM peer 8033 found, imports clean, PySide6 6.11.1. Screenshot shots/main_window.png
      looks genuinely good. git df5b8b9.
    - APP_2 — DONE + reviewed PASS. Chat "Peer": streams llama.cpp :8033 + 6 read-only tools
      (rag_search/rag_get_image/wikipedia/find_book/find_paper/r_python_docs), off-UI-thread,
      hard read-only gate, frames inline, graceful tool-failure cards. git df5b8b9→d65c6cf.
      shots/chat_with_tool.png looks great. KNOWN ISSUE surfaced in that shot: rag_search timed
      out (45s) — NOT an app defect (the Peer degraded gracefully). Root cause = GPU contention:
      with the 35B resident on the 7700S, the daemon can't load the embedder to embed the query.
      Candidate fix (daemon, needs user OK) = embed QUERIES on CPU (text embed is tiny, ~seconds),
      keep image/ingest on GPU → searches never collide with the 35B. Added to PROMOTION_PLAN §Tier 0.
      Kimi's own 5 honest gaps for Phase 3 (in REPORT_APP_2): cold-start race (wait sig_session_ready
      before enabling Send), per-turn tool-disable flapping, no stream-cancel on Reset, image-size cap,
      no chat persistence — Phase 3 contract can pick these up.
    - APP_3 — DONE + reviewed PASS. Sources view (5 MCP servers, online dots, endpoints, read-only
      ad-hoc runner, destructive tools hidden) + Journal timeline (2692-frame filmstrip from
      frames.json, local jpgs, no re-embed, video card + YouTube URL + timestamps) + polish
      (dark theme, aleph.svg icon, shortcut, geometry persist). shots/sources_view.png +
      journal_view.png both look great. git 95e76b4 + d2f57c5. Kimi's honest gaps for Phase 4:
      Sources tools-list needed Chat opened first (McpClient lacked list_tools) — FOLDED INTO
      the APP_4 goal as a pre-package fix; plus minor: enlarge-dialog title, arrow-key stepping,
      substring destructive filter, Ctrl+F conflict (all non-blocking).
    - APP_4 — DONE + reviewed PASS (FINAL). Built dist/aleph_0.1.0_amd64.deb (38K), tagged v0.1.0,
      git 1715ff8. Correct Debian layout (/opt/aleph + /usr/bin/aleph + aleph.desktop + hicolor
      scalable icon + man page + copyright + lintian overrides). SAFETY VERIFIED: postinst only
      refreshes desktop/icon caches + runs read-only aleph-check.py; prerm no-op; NO systemctl
      start/stop/enable/kill anywhere. aleph-check.py is probe-only (`systemctl --user cat` read-only
      + socket port probes). desktop-file-validate PASS (one cosmetic multi-main-category hint —
      trim Categories someday, non-blocking). Sources list_tools standalone fix folded in.
  ==> PHASE B (Aleph desktop app) COMPLETE & SUCCESSFUL — all 4 phases shipped, feature-complete,
      installable. INSTALL CMD for Gabriel: `sudo apt install /home/gabriel/projects/aleph/dist/aleph_0.1.0_amd64.deb`
      then launch with `aleph`. Full end-to-end review passed (imports clean, 5 screenshots good,
      no daemon touched). 30-min /loop CANCELLED (CronDelete 138fe99e) — backlog empty, nothing left
      to do autonomously; the only open item (production promotion) needs Gabriel's explicit approval.
    - WATCH: Kimi session context at ~50% (255k/512k). If it climbs past ~80%, start a FRESH kimi
      session for remaining phases — the on-disk contracts/reports make a cold pickup clean.
  - NOT yet queued: APP_1..4 (Aleph desktop app, in /home/gabriel/projects/aleph/).
    Queue APP_1 via `/goal` when Kimi is idle after REPORT_3 lands.
- USER'S ORIGINAL EMBED JOB IS STALLED at 216/2692 frames (yt-dlpcc PID 633956 blocked on
  a dead MCP ingest call; embedder unloaded; GPU[0] free). Left untouched (no-kill law;
  216 partial vectors are harmless — a re-ingest replaces the whole frame source). PAYOFF
  TASK once CONTRACT 2 validates the fast path: promote video_embed into the daemon and
  re-ingest ~/Documents/Personal so the journal video finishes in minutes. Confirm with
  the user before promoting into production rag-mcp.

## Hard constraints (never violate, never let Kimi violate):
- FIDELITY IS SACRED — speed never comes from lowering resolution/max_pixels.
- Never kill/restart rag-mcp.service, llama-server, whisper-server, or the live yt-dlpcc.
- 7700S = cuda:0 gfx1102; 780M = gfx1103 needs HSA_OVERRIDE_GFX_VERSION=11.0.0.
- Python = /home/gabriel/venv/bin/python. Work stays in kimi_kernel/ until Claude promotes.
