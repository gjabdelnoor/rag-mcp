#!/usr/bin/env python3
"""
kb_tui.py — knowledge-base TUI. Two worlds, deliberately split:

  CLAUDE tiers  -> `claude -p` on your SUBSCRIPTION (Agent SDK credit).
                   No API key. No custom tools. Built-in Read/Grep/Glob only,
                   scoped to the folder when folder-scope is ON. Depth comes
                   from --max-turns + Claude's default adaptive thinking (there
                   is no effort flag on claude -p).
        default (ctrl-D): --max-turns 1   (quick)
        boost   (ctrl-T): --max-turns 3   (2 extra agentic turns)

  MINIMAX tier  -> MiniMax M3 on its own API key. Full custom tool loop:
                   grep_files, screenshot_book, consensus_search, pull_papers.
        cheap   (ctrl-E): 20-turn agentic budget. Primary workhorse.

Folder-scope toggle (gray -> rainbow, ctrl-F or click):
  - Claude tiers: adds Grep/Glob/Read to --allowedTools + --add-dir <scope>.
  - MiniMax tier: exposes grep_files + screenshot_book to the model.
  Off = model answers from KB context alone.

PAPER-FINDING FLOW (MiniMax tier):
  1. Model calls consensus_search(query, [filters]) -> 200M+ peer-reviewed
     papers via the Consensus MCP (https://mcp.consensus.app/mcp). Returns
     title, authors, abstract, journal, year, citation_count, url.
  2. Model calls pull_papers(query) -> Anna's Archive search + download of
     the top result, returning the local PDF path. The user can open it.

AUTH:
  Claude   : run `claude login` once. Do NOT set ANTHROPIC_API_KEY (that would
             bill pay-as-you-go instead of your subscription), and this app
             warns if it's set. Do NOT use --bare (it skips OAuth).
  MiniMax  : export MINIMAX_API_KEY=...
  Consensus: OAuth 2.0 + PKCE + DCR against https://consensus.app
             (the auth server, NOT the MCP host). First call opens your
             browser to consent; tokens are cached at
             ~/.config/kb_tui/consensus_tokens.json and refreshed silently.
             (Enterprise Bearer tokens via CONSENSUS_API_KEY still work too
             — the helper picks whichever path is set.)
  Anna's   : annas_archive_search.py + annas_archive_download.py expected at
             ~/annas_archive_{search,download}.py (override with
             ANNAS_SEARCH_SCRIPT / ANNAS_DOWNLOAD_SCRIPT). AA_API_KEY is
             baked into annas_archive_download.py.

SETUP:
  pip install textual openai mcp
  (need the `claude` CLI on PATH, logged in; ripgrep optional for MiniMax grep)
  python kb_tui.py --kb kb.txt --scope ./notes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual.widgets import Header, Footer, Input, RichLog, Static

# =============================================================================
# Tiers
# =============================================================================

MINIMAX_MODEL = "MiniMax-M3"           # exact string; the API rejects
                                       # anything else with 404 model_not_found.
MINIMAX_BASE_URL = "https://api.minimax.io/v1"


def _load_minimax_key() -> None:
    """Populate os.environ['MINIMAX_API_KEY'] from ~/minimaxkey.txt if not
    already set. The file is a one-line dotenv-style KEY=VALUE. This lets the
    cheap tier work without requiring `export MINIMAX_API_KEY=...` in the
    shell — the key lives in the user's home dir with mode 0600 and is
    loaded on demand."""
    if os.environ.get("MINIMAX_API_KEY"):
        return
    key_file = Path.home() / "minimaxkey.txt"
    if not key_file.is_file():
        return
    try:
        for line in key_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass

CONSENSUS_MCP_URL = "https://mcp.consensus.app/mcp"
CONSENSUS_AUTH_ISSUER = "https://consensus.app"
CONSENSUS_OAUTH_SCOPE = "search"
CONSENSUS_CALLBACK_HOST = "127.0.0.1"
CONSENSUS_CALLBACK_PORT = 8765
CONSENSUS_CALLBACK_URL = (
    f"http://{CONSENSUS_CALLBACK_HOST}:{CONSENSUS_CALLBACK_PORT}/callback"
)
CONSENSUS_TOKEN_STORE = Path.home() / ".config" / "kb_tui" / "consensus_tokens.json"
CONSENSUS_TIMEOUT_S = 30
CONSENSUS_READ_TIMEOUT_S = 120

# Default annas-script locations; override with env vars below.
DEFAULT_ANNAS_SEARCH = os.path.expanduser("~/annas_archive_search.py")
DEFAULT_ANNAS_DOWNLOAD = os.path.expanduser("~/annas_archive_download.py")


@dataclass(frozen=True)
class Tier:
    name: str
    backend: str          # "claude" | "minimax"
    max_turns: int


TIERS = {
    "default": Tier("default", "claude", 1),
    "boost":   Tier("boost",   "claude", 3),
    "cheap":   Tier("cheap",   "minimax", 20),
}

SYSTEM = (
    "You answer from the provided knowledge base and available tools. Prefer "
    "the KB; use tools to fill gaps. Cite filenames/sources you used. If you "
    "cannot answer, say so plainly.\n\n"
    "Tool workflow for finding NEW literature: call `consensus_search` first "
    "to get paper metadata (title, authors, year, url), then `pull_papers` "
    "with the paper's title to download the PDF via Anna's Archive. The "
    "local path is returned for the user to open."
)

# =============================================================================
# CLAUDE backend: subscription via `claude -p`
# =============================================================================


def _run_claude_subprocess(tier: Tier, prompt: str, scope_dir: Path,
                           folder_scope: bool, log) -> str:
    """Fallback path: shell out to `claude -p`. Used only if the Agent SDK
    is not importable (older setup, isolated venv)."""
    cmd = ["claude", "-p", "--output-format", "json",
           "--max-turns", str(tier.max_turns)]
    if folder_scope and scope_dir.exists():
        cmd += ["--allowedTools", "Read,Grep,Glob",
                "--add-dir", str(scope_dir)]
    log(f"[cyan]$ claude -p --max-turns {tier.max_turns}"
        f"{' +folder' if folder_scope and scope_dir.exists() else ''}[/]")
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True,
                           text=True, timeout=600)
    except FileNotFoundError:
        return "[claude error] `claude` CLI not found on PATH."
    except subprocess.TimeoutExpired:
        return "[claude error] timed out (600s)."
    if p.returncode != 0:
        return (f"[claude error] exit {p.returncode}: "
                f"{p.stderr.strip()[:400] or p.stdout.strip()[:400]}")
    try:
        data = json.loads(p.stdout)
        return data.get("result") or data.get("text") or json.dumps(data)[:800]
    except json.JSONDecodeError:
        return p.stdout.strip() or "[claude error] empty output"


async def _collect_sdk_output(async_iter) -> str:
    """Walk the Agent SDK message stream and return the final text.

    `ResultMessage` carries the final aggregated `result`; we prefer that.
    Otherwise we concatenate `AssistantMessage.content` text blocks."""
    text_parts: list[str] = []
    async for msg in async_iter:
        # ResultMessage is the terminal message with the full result.
        if type(msg).__name__ == "ResultMessage":
            result = getattr(msg, "result", None)
            if result:
                return result if isinstance(result, str) else str(result)
            continue
        # AssistantMessage: append text blocks.
        content = getattr(msg, "content", None)
        if not content:
            continue
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None) or (
                    block.get("text") if isinstance(block, dict) else None
                )
                if text:
                    text_parts.append(text)
    return "\n".join(text_parts) or "[no text returned]"


def run_claude(tier: Tier, kb: str, question: str, scope_dir: Path,
               folder_scope: bool, log) -> str:
    """Claude tier via the Anthropic Agent SDK (`claude_agent_sdk`).

    Uses OAuth/subscription (same auth as `claude -p`), supports `max_turns`,
    `effort`, and tool scoping. Falls back to a `claude -p` subprocess if
    the SDK isn't installed.

    Folder scope adds Read/Grep/Glob + `add_dirs` so Claude Code's built-in
    tools can read the notes folder. Without scope, the model answers from
    the piped KB alone (no tools)."""
    prompt = (
        f"{SYSTEM}\n\n"
        f"Knowledge base follows, then the question.\n\n"
        f"<kb>\n{kb}\n</kb>\n\nQuestion: {question}"
    )

    try:
        from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
    except ImportError:
        return _run_claude_subprocess(tier, prompt, scope_dir, folder_scope, log)

    options_kwargs: dict = {
        "max_turns": tier.max_turns,
        "system_prompt": SYSTEM,
    }
    if folder_scope and scope_dir.exists():
        options_kwargs["allowed_tools"] = ["Read", "Grep", "Glob"]
        options_kwargs["add_dirs"] = [str(scope_dir)]
    # Effort knob: only the boost tier (more turns + deeper reasoning) sets
    # it. The Agent SDK exposes the full effort range; `claude -p` did not.
    if tier.name == "boost":
        options_kwargs["effort"] = "high"

    log_extra = f" effort={options_kwargs['effort']}" if "effort" in options_kwargs else ""
    log(f"[cyan]$ claude (Agent SDK) max-turns={tier.max_turns}"
        f"{' +folder' if folder_scope and scope_dir.exists() else ''}"
        f"{log_extra}[/]")

    async def call():
        from claude_agent_sdk import query, ClaudeAgentOptions
        return await _collect_sdk_output(
            query(prompt=prompt, options=ClaudeAgentOptions(**options_kwargs))
        )

    try:
        return asyncio.run(call())
    except Exception as e:                           # noqa: BLE001
        return f"[claude error] {e}"


# =============================================================================
# MINIMAX backend: M3 with the full custom tool loop
# =============================================================================


def minimax_tool_specs(folder_scope: bool) -> list[dict]:
    """The tool surface exposed to the MiniMax-tier model.

    Folder-scoped tools (grep_files + screenshot_book) are prepended; the
    paper-finding tools (consensus_search + pull_papers) are always available
    so the model can chain them regardless of folder scope.
    """
    paper_tools = [
        {"type": "function", "function": {
            "name": "consensus_search",
            "description": (
                "Search 200M+ peer-reviewed papers via Consensus MCP. Returns "
                "title, authors, abstract, journal, year, citation_count, url, "
                "and (Pro+) study_type + takeaway. Use FIRST when looking for "
                "new literature. Optional filters: year_min/year_max, "
                "study_types (rct|meta-analysis|systematic review|...), "
                "sjr_max (1=Q1..4=Q4), human (true=human-only), sample_size_min, "
                "medical_mode, exclude_preprints, duration_min/duration_max."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "year_min": {"type": "integer"},
                "year_max": {"type": "integer"},
                "study_types": {"type": "array", "items": {"type": "string"}},
                "sjr_max": {"type": "integer"},
                "human": {"type": "boolean"},
                "sample_size_min": {"type": "integer"},
                "medical_mode": {"type": "boolean"},
                "exclude_preprints": {"type": "boolean"},
                "duration_min": {"type": "integer"},
                "duration_max": {"type": "integer"},
            }, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "pull_papers",
            "description": (
                "Find and download a paper via Anna's Archive (search -> "
                "download). The query is a paper title or DOI. Returns the "
                "local PDF path(s) on success. Use AFTER consensus_search once "
                "you know the title. Requires the AA account at "
                "https://annas-archive.pk/account (AA_API_KEY is baked into "
                "annas_archive_download.py)."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 1}},
                "required": ["query"]}}},
    ]

    scoped_tools = []
    if folder_scope:
        scoped_tools = [
            {"type": "function", "function": {
                "name": "grep_files",
                "description": (
                    "Text-only ripgrep over the scoped folder (md/txt/org/rst). "
                    "file:line matches. No PDFs/images."
                ),
                "parameters": {"type": "object", "properties": {
                    "pattern": {"type": "string"},
                    "max_matches": {"type": "integer", "default": 40}},
                    "required": ["pattern"]}}},
            {"type": "function", "function": {
                "name": "screenshot_book",
                "description": (
                    "Retrieve from the local RAG (rag-mcp): PDFs, scanned "
                    "pages, images. Returns top-k passages + image refs."
                ),
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5}},
                    "required": ["query"]}}},
        ]

    return scoped_tools + paper_tools


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "this", "that", "these",
    "those", "what", "which", "who", "when", "where", "why", "how",
    "and", "or", "but", "of", "to", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "its", "their", "there", "them", "they", "you",
    "your", "i", "me", "my", "we", "us", "our", "he", "she", "him",
    "her", "it", "is", "are", "was", "were", "be",
})


def _naive_retrieve(kb: str, question: str, max_chars: int = 6000,
                    chunk_size: int = 400) -> str:
    """Naive lexical retrieval (BM25-lite): split KB into ~chunk_size
    windows with stride=chunk_size//2, score each by overlap with the
    question's content words (stopwords stripped), keep the top windows
    until max_chars is reached, return in original document order.

    This is not semantic — but it's true retrieval: the model sees the
    parts of the KB most likely to answer the question, not the first
    6K chars (which might be an unrelated table of contents). Works
    without any vector index, so it functions on un-ingested collections
    like VCP.
    """
    import re
    if not kb or not question:
        return kb[:max_chars]
    words = set(re.findall(r"\w+", question.lower())) - _STOPWORDS
    if not words:
        return kb[:max_chars]
    stride = max(1, chunk_size // 2)
    chunks: list[tuple[int, str]] = []   # (start_offset, text)
    for start in range(0, max(len(kb) - chunk_size, 1) + 1, stride):
        chunks.append((start, kb[start:start + chunk_size]))
    if not chunks:                              # kb shorter than chunk_size
        chunks = [(0, kb)]
    scored = []
    for start, chunk in chunks:
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        overlap = len(words & chunk_words)
        scored.append((overlap, start, chunk))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked: list[tuple[int, str]] = []
    total = 0
    for score, start, chunk in scored:
        if score == 0:
            break                       # remaining chunks have no signal
        if total + len(chunk) > max_chars:
            continue                    # try smaller chunks instead
        picked.append((start, chunk))
        total += len(chunk)
        if total >= max_chars:
            break
    if not picked:
        # No overlap at all — fall back to head of doc, but flag it.
        head = kb[:max_chars]
        return head + ("" if len(kb) <= max_chars else
                       f"\n\n... [lexical retrieval found 0 keyword overlap; "
                       f"showing first {max_chars:,} of {len(kb):,} chars]")
    picked.sort(key=lambda t: t[0])
    out_parts = [c for _, c in picked]
    return ("\n\n[...]\n\n".join(out_parts) +
            f"\n\n[retrieved {len(picked)} chunk(s) of {len(kb):,} chars "
            f"by keyword overlap; this is NOT a semantic index]")


def run_minimax(tier: Tier, kb: str, question: str, tools: Tools,
                folder_scope: bool, log) -> str:
    """MiniMax M3 backend — full custom tool loop on its own API key.

    Runs an agentic loop with up to `tier.max_turns` turns: each turn the
    model can either respond or call one of the tools declared in
    `minimax_tool_specs(folder_scope)`. On the last turn we force a final
    answer (no further tool calls).

    The KB is passed via lexical retrieval (see `_naive_retrieve`) since
    MiniMax-M3's context window is ~2K tokens — pipe-only would lose 99%
    of any non-trivial KB. Retrieval returns the chunks most likely to
    contain the answer, not just the first N chars."""
    _load_minimax_key()
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key or api_key == "MINIMAX_API_KEY":
        return ("[minimax error] no real API key. Set MINIMAX_API_KEY in the "
                "environment, or write `MINIMAX_API_KEY=sk-...` to "
                "~/minimaxkey.txt.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=MINIMAX_BASE_URL)
    oa_tools = minimax_tool_specs(folder_scope)
    kb_for_model = _naive_retrieve(kb, question, max_chars=6000)
    if len(kb) > 6000:
        log(f"[cyan]ℹ MiniMax-M3 sees only the top 6,000 chars "
            f"(retrieved by keyword overlap from {len(kb):,}-char KB; "
            f"ingest this collection for semantic RAG instead).[/]")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"<kb>\n{kb_for_model}\n</kb>\n\nQuestion: {question}"},
    ]
    for turn in range(tier.max_turns):
        resp = client.chat.completions.create(
            model=MINIMAX_MODEL, messages=messages,
            tools=oa_tools, max_tokens=4096)
        msg = resp.choices[0].message
        if msg.content:
            log(f"[dim]{msg.content}[/]")
        calls = msg.tool_calls or []
        if not calls or turn == tier.max_turns - 1:
            return msg.content or "[no text returned]"
        messages.append(msg.model_dump())
        for c in calls:
            cargs = json.loads(c.function.arguments or "{}")
            log(f"[magenta]  → {c.function.name}({json.dumps(cargs)})[/]")
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": tools.dispatch(c.function.name, cargs)})
    return "[budget exhausted without final answer]"


# =============================================================================
# Consensus OAuth: token storage + redirect/callback handlers
# =============================================================================


class FileTokenStorage:
    """Persist OAuth tokens + registered client info to a single JSON file.

    Implements the `mcp.client.auth.oauth2.TokenStorage` protocol (all four
    methods are async). Used so a successful DCR + token exchange survives
    across runs — the user only consents once.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except Exception:                           # noqa: BLE001
            return {}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path)

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken
        data = self._load()
        raw = data.get("tokens")
        if not raw:
            return None
        try:
            return OAuthToken.model_validate(raw)
        except Exception:                           # noqa: BLE001
            return None

    async def set_tokens(self, tokens) -> None:
        data = self._load()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        self._save(data)

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull
        data = self._load()
        raw = data.get("client_info")
        if not raw:
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception:                           # noqa: BLE001
            return None

    async def set_client_info(self, client_info) -> None:
        data = self._load()
        data["client_info"] = client_info.model_dump(
            mode="json", exclude_none=True,
        )
        self._save(data)


async def _consensus_redirect(url: str) -> None:
    """Open the OAuth authorization URL in the user's default browser."""
    print(f"\n[consensus] opening browser to authorize:\n  {url}\n",
          file=sys.stderr, flush=True)
    webbrowser.open(url)


async def _consensus_callback() -> tuple[str, str | None]:
    """Run a tiny HTTP server on 127.0.0.1:8765 that captures the OAuth
    redirect. Returns (code, state). Resolves once the user clicks Allow
    in the browser and the auth server redirects back."""
    loop = asyncio.get_event_loop()
    result_future: asyncio.Future = loop.create_future()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                          # noqa: N802
            qs = parse_qs(urlparse(self.path).query)
            code = (qs.get("code") or [None])[0]
            state = (qs.get("state") or [None])[0]
            err = (qs.get("error") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if err:
                body = (f"<h2>Authorization failed: {err}</h2>"
                        "<p>Close this tab and check the TUI.</p>")
            else:
                body = "<h2>OK — you can close this tab.</h2>"
            self.wfile.write(body.encode("utf-8"))
            if not result_future.done():
                loop.call_soon_threadsafe(
                    result_future.set_result, (code, state, err),
                )

        def log_message(self, *_a, **_kw):          # noqa: N802
            return  # silence access log

    server = HTTPServer((CONSENSUS_CALLBACK_HOST, CONSENSUS_CALLBACK_PORT), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        code, state, err = await asyncio.wait_for(result_future, timeout=300)
    finally:
        server.shutdown()
        server.server_close()
    if err:
        raise RuntimeError(f"consensus OAuth error: {err}")
    if not code:
        raise RuntimeError("consensus OAuth callback returned no code")
    return code, state


class Tools:
    def __init__(self, scope_dir: Path):
        self.scope_dir = scope_dir

    def dispatch(self, name: str, args: dict) -> str:
        fn = getattr(self, f"_{name}", None)
        if fn is None:
            return f"[tool error] unknown tool: {name}"
        try:
            return fn(args)
        except Exception as e:
            return f"[tool error] {name}: {e}"

    # -- folder-scoped ---------------------------------------------------------

    def _grep_files(self, args: dict) -> str:
        if not self.scope_dir.exists():
            return f"[grep] scope dir missing: {self.scope_dir}"
        cmd = ["rg", "--no-heading", "-n", "-I",
               "--max-count", str(int(args.get("max_matches", 40))),
               "-g", "*.md", "-g", "*.txt", "-g", "*.org", "-g", "*.rst",
               args["pattern"], str(self.scope_dir)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            return "[grep] ripgrep (rg) not installed"
        if p.returncode not in (0, 1):
            return f"[grep] rg error: {p.stderr.strip() or p.returncode}"
        return p.stdout.strip() or "[grep] no matches"

    def _screenshot_book(self, args: dict) -> str:
        """TODO(gabe): wire to rag-mcp MCP at http://127.0.0.1:8077/mcp
        (call `search` then `get_book_image` for the top hits)."""
        return (f"[screenshot_book STUB] k={args.get('k', 5)} for "
                f"{args['query']!r}. Wire to rag-mcp.")

    # -- paper-finding ---------------------------------------------------------

    def _consensus_search(self, args: dict) -> str:
        """Search 200M+ peer-reviewed papers via the Consensus MCP.

        Auth: OAuth 2.0 + PKCE + DCR against https://consensus.app (NOT
        Bearer-token). First call opens your browser; tokens are cached at
        ~/.config/kb_tui/consensus_tokens.json and refreshed silently.
        Set CONSENSUS_API_KEY to use a Consensus enterprise Bearer instead
        (skips the OAuth dance; sets the Authorization header directly).
        """
        query = args["query"]
        search_args: dict = {"query": query}
        for k in ("year_min", "year_max", "study_types", "sjr_max", "human",
                  "sample_size_min", "medical_mode", "exclude_preprints",
                  "duration_min", "duration_max"):
            if k in args and args[k] is not None:
                search_args[k] = args[k]

        auth = self._consensus_auth()

        async def call():
            # Lazy imports so kb_tui.py still starts without `mcp` installed
            # (consensus_search just fails when called).
            from mcp.client.streamable_http import streamablehttp_client
            from mcp import ClientSession

            kwargs = {"timeout": CONSENSUS_TIMEOUT_S}
            if isinstance(auth, dict):
                kwargs["headers"] = auth
            else:
                kwargs["auth"] = auth  # OAuthClientProvider is httpx.Auth

            async with streamablehttp_client(CONSENSUS_MCP_URL, **kwargs) as (r, w, _):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    return await s.call_tool(
                        "search", search_args,
                        read_timeout_seconds=timedelta(seconds=CONSENSUS_READ_TIMEOUT_S),
                    )

        try:
            result = asyncio.run(call())
        except Exception as e:                       # noqa: BLE001
            return f"[consensus error] {e}"

        parts = []
        for part in (getattr(result, "content", None) or []):
            text = getattr(part, "text", None)
            parts.append(text if text is not None else str(part))
        return "\n".join(parts) if parts else "[consensus] empty result"

    def _consensus_auth(self):
        """Return either a Bearer-token dict (enterprise) or an
        OAuthClientProvider. Cached on the Tools instance so the OAuth
        dance only runs once per session."""
        # Enterprise short-circuit
        api_key = os.environ.get("CONSENSUS_API_KEY")
        if api_key:
            return {"Authorization": f"Bearer {api_key}"}

        if getattr(self, "_consensus_provider", None) is not None:
            return self._consensus_provider

        from mcp.client.auth.oauth2 import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata

        storage = FileTokenStorage(CONSENSUS_TOKEN_STORE)
        metadata = OAuthClientMetadata(
            redirect_uris=[CONSENSUS_CALLBACK_URL],
            scope=CONSENSUS_OAUTH_SCOPE,
            client_name="kb_tui",
            token_endpoint_auth_method="none",
        )
        provider = OAuthClientProvider(
            server_url=CONSENSUS_MCP_URL,
            client_metadata=metadata,
            storage=storage,
            redirect_handler=_consensus_redirect,
            callback_handler=_consensus_callback,
            timeout=300.0,
        )
        self._consensus_provider = provider
        return provider

    def _pull_papers(self, args: dict) -> str:
        """Find and download papers via Anna's Archive (search -> download)."""
        search_script = os.environ.get("ANNAS_SEARCH_SCRIPT", DEFAULT_ANNAS_SEARCH)
        download_script = os.environ.get(
            "ANNAS_DOWNLOAD_SCRIPT", DEFAULT_ANNAS_DOWNLOAD,
        )
        if not os.path.isfile(search_script):
            return (f"[annas] search script missing: {search_script}. "
                    "Set ANNAS_SEARCH_SCRIPT or install.")
        if not os.path.isfile(download_script):
            return (f"[annas] download script missing: {download_script}. "
                    "Set ANNAS_DOWNLOAD_SCRIPT or install.")

        query = args["query"]
        max_results = max(1, int(args.get("max_results", 1)))

        # Step 1: search annas (Playwright + cookies -> MD5 + title + url)
        try:
            proc = subprocess.run(
                [sys.executable, search_script, query],
                capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            return f"[annas search] timed out (180s) for {query!r}"
        except Exception as e:                       # noqa: BLE001
            return f"[annas search error] {e}"
        if proc.returncode != 0:
            err = proc.stderr.strip()[:400] or f"exit {proc.returncode}"
            return f"[annas search error] {err}"
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            return (f"[annas search error] bad json ({e}); "
                    f"raw stdout: {proc.stdout[:300]}")
        results = data.get("results") or []
        if not results:
            return f"[annas] no matches for {query!r}"

        # Step 2: hand the search JSON to the downloader (--from-file --indices)
        tmp_path = Path(tempfile.gettempdir()) / "kb_tui_annas_results.json"
        tmp_path.write_text(json.dumps(data))
        indices = list(range(1, min(max_results, len(results)) + 1))

        try:
            proc = subprocess.run(
                [sys.executable, download_script,
                 "--from-file", str(tmp_path),
                 "--indices", *map(str, indices)],
                capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            return f"[annas download] timed out (600s) for {query!r}"
        except Exception as e:                       # noqa: BLE001
            return f"[annas download error] {e}"
        if proc.returncode != 0:
            err = proc.stderr.strip()[:400] or f"exit {proc.returncode}"
            return f"[annas download error] {err}"

        # Parse the JSON tail (after "--- JSON OUTPUT ---") for machine-readable
        # results; fall back to raw stdout if the separator isn't present.
        out = proc.stdout
        json_section = (
            out.split("--- JSON OUTPUT ---", 1)[1].strip()
            if "--- JSON OUTPUT ---" in out else out
        )
        try:
            data = json.loads(json_section)
            ok = data.get("success_count", 0)
            total = data.get("total", 0)
            out_dir = data.get("output_directory", "?")
            lines = [f"Downloaded {ok}/{total} paper(s) to {out_dir}:"]
            for r in data.get("results", []):
                tag = "OK " if r.get("success") else "ERR"
                title = r.get("title", "?")
                if r.get("success"):
                    lines.append(f"  [{tag}] #{r['index']} {title} "
                                 f"-> {Path(r['filepath']).name}")
                else:
                    lines.append(f"  [{tag}] #{r['index']} {title} "
                                 f"FAILED: {r.get('error', '?')}")
            return "\n".join(lines)
        except (json.JSONDecodeError, KeyError):
            return out


# =============================================================================
# Dispatch
# =============================================================================


def answer(tier, kb, question, tools, scope_dir, folder_scope, log) -> str:
    try:
        if tier.backend == "claude":
            return run_claude(tier, kb, question, scope_dir, folder_scope, log)
        return run_minimax(tier, kb, question, tools, folder_scope, log)
    except Exception as e:
        return f"[error: {tier.backend} backend failed] {e}"


# =============================================================================
# TUI
# =============================================================================


import colorsys
import time as _time


# Draw a folder icon as a real inline SVG string. Textual's Image widget
# (or the `textual-image` package) can render this; we fall back to a
# block-character folder when SVG rendering is unavailable.
FOLDER_SVG_DARK = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="14" '
    'viewBox="0 0 20 14" fill="none">'
    '<path d="M1 3.5C1 2.67 1.67 2 2.5 2h4l1.5 2h9.5c.83 0 1.5.67 1.5 1.5v6c0 .83-.67 '
    '1.5-1.5 1.5h-15C1.67 12 1 11.33 1 10.5z" fill="#888" stroke="#aaa" '
    'stroke-width="0.7"/></svg>'
)
FOLDER_SVG_RAINBOW = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="14" '
    'viewBox="0 0 20 14" fill="none">'
    '<defs><linearGradient id="g" x1="0" x2="20" y1="0" y2="0" '
    'gradientUnits="userSpaceOnUse">'
    '<stop offset="0" stop-color="#ff5555"/>'
    '<stop offset=".16" stop-color="#ffaa00"/>'
    '<stop offset=".33" stop-color="#ffff55"/>'
    '<stop offset=".5" stop-color="#55ff55"/>'
    '<stop offset=".66" stop-color="#55ffff"/>'
    '<stop offset=".83" stop-color="#5555ff"/>'
    '<stop offset="1" stop-color="#ff55ff"/>'
    '</linearGradient></defs>'
    '<path d="M1 3.5C1 2.67 1.67 2 2.5 2h4l1.5 2h9.5c.83 0 1.5.67 1.5 1.5v6c0 .83-.67 '
    '1.5-1.5 1.5h-15C1.67 12 1 11.33 1 10.5z" fill="url(#g)" stroke="#fff" '
    'stroke-width="0.7"/></svg>'
)


def _rainbow_strip(text: str, phase: float) -> str:
    """Each character gets a hue along the rainbow, shifted by `phase`. Returns
    Rich markup. Empty input → empty output (no spurious markup)."""
    if not text:
        return ""
    out = []
    n = max(len(text), 1)
    for i, ch in enumerate(text):
        h = ((i / n) + phase) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        hex_ = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        out.append(f"[{hex_}]{ch}[/]")
    return "".join(out)


class FolderToggle(Static):
    """Folder-scope toggle.

    Off: a dim folder glyph + plain text.
    On : the same glyph + the label, every character of the label cycling
         through a moving rainbow hue (each character = one "LED" in an RGB
         strip). Animate via on_mount timer so the gradient flows.
    """

    active = reactive(False)

    FOLDER = "\U0001F4C1"          # fallback glyph if SVG render fails

    def render(self) -> str:
        phase = _time.monotonic() / 3.5   # full cycle ≈ 3.5 s
        # The icon stays put; only the label animates. Keeps the icon
        # legible across the whole rainbow (instead of an unreadable mid-hue
        # folder glyph).
        icon_color = "#ffaa00" if self.active else "#666666"
        if not self.active:
            return f"[{icon_color}]\U0001F4C1 folder-scope off[/]"
        label = " folder-scope ON"
        return f"[{icon_color}]\U0001F4C1[/]" + _rainbow_strip(label, phase)

    def on_mount(self) -> None:
        # Animate the rainbow ~12 fps; cheap (one hsv_to_rgb per char).
        self.set_interval(0.08, self.refresh)

    def on_click(self) -> None:
        self.app.action_toggle_scope()


TIER_ORDER = ("default", "boost", "cheap")
TIER_HELP = {
    "default": "Claude (subscription) · 1 turn",
    "boost":   "Claude (subscription) · 3 turns · effort=high",
    "cheap":   "MiniMax M3 (API) · 20-turn agentic loop",
}


def _consensus_available() -> bool:
    """Consensus MCP is usable when either an enterprise key is set or an
    OAuth token cache exists from a prior browser-consent flow."""
    if os.environ.get("CONSENSUS_API_KEY"):
        return True
    return (Path.home() / ".config" / "kb_tui" / "consensus_tokens.json").exists()


def _annas_available() -> bool:
    """Anna's Archive is usable when both helper scripts exist in $HOME
    (the same path the `kb` CLI uses)."""
    home = Path.home()
    return ((home / "annas_archive_search.py").exists()
            and (home / "annas_archive_download.py").exists())


class ToolIndicator(Static):
    """Status pill for an external tool.

    Same visual pattern as FolderToggle: lit (green) when the tool is
    available — env var present, OAuth token cached, helper script on disk —
    dim when not. Clickable to invoke directly (placeholder bell for now;
    future: open a query modal and run the tool on the current KB)."""

    def __init__(self, emoji: str, label: str, available: bool,
                 tooltip: str = "", **kwargs):
        super().__init__(**kwargs)
        self._emoji = emoji
        self._label = label
        self._available = available
        self._tooltip = tooltip

    def render(self) -> str:
        if self._available:
            return f"[bold green]{self._emoji} {self._label}[/]"
        return f"[dim]{self._emoji} {self._label}[/]"

    def on_click(self) -> None:
        # Future: open a query modal and fire the tool on the active KB.
        # For now: a single bell so the click is acknowledged.
        self.app.bell()


class KBApp(App):
    CSS = """
    RichLog { border: round $accent; padding: 0 1; }
    #bar { height: 1; }
    #tier { color: $warning; width: 60; }
    FolderToggle { width: auto; }
    ToolIndicator { width: auto; }
    Input { border: round $accent; }
    """

    BINDINGS = [
        ("ctrl+t", "cycle_tier", "Cycle tier"),
        ("ctrl+f", "toggle_scope", "Folder scope"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, kb_text: str, scope_dir: Path, kb_name: str = ""):
        super().__init__()
        self.kb = kb_text
        self.scope_dir = scope_dir
        self.kb_name = kb_name
        self.tier_key = "default"
        self.scope = False
        self.tools = Tools(scope_dir)

    def compose(self) -> ComposeResult:
        title = self.kb_name or "knowledge base"
        yield Header(show_clock=False)
        with Vertical():
            yield RichLog(highlight=True, markup=True, id="log",
                          wrap=True)
            with Horizontal(id="bar"):
                yield Static(self._tier_label(), id="tier")
                yield ToolIndicator(
                    "\U0001F9EA", "consensus", _consensus_available(),
                    tooltip="Consensus MCP — peer-reviewed paper search",
                    id="consensus-ind",
                )
                yield ToolIndicator(
                    "\U0001F50D", "annas", _annas_available(),
                    tooltip="Anna's Archive — search & download",
                    id="annas-ind",
                )
                yield FolderToggle(id="folder")
            yield Input(
                placeholder=f"Ask {title}…  (Enter to send · "
                             "Ctrl-T cycle tier · Ctrl-F folder scope)",
                id="q",
            )
        yield Footer()

    def _tier_label(self) -> str:
        return (f"tier: [bold]{self.tier_key}[/] · {TIER_HELP[self.tier_key]}")

    def _refresh(self):
        self.query_one("#tier", Static).update(self._tier_label())
        self.query_one("#folder", FolderToggle).active = self.scope

    def on_mount(self) -> None:
        # Auto-focus the input — user types without clicking.
        self.query_one("#q", Input).focus()

    def action_cycle_tier(self):
        idx = TIER_ORDER.index(self.tier_key)
        self.tier_key = TIER_ORDER[(idx + 1) % len(TIER_ORDER)]
        self._refresh()

    def action_toggle_scope(self):
        self.scope = not self.scope
        self._refresh()

    async def on_input_submitted(self, event: Input.Submitted):
        q = event.value.strip()
        if not q:
            return
        log = self.query_one("#log", RichLog)
        # Don't clear the input yet — keep the question visible while it
        # runs (clears on next render after the answer lands).
        self.query_one("#q", Input).value = ""
        tier = TIERS[self.tier_key]
        log.write(
            f"[bold green]› {q}[/]"
            f"  [dim]({tier.name}, scope={'on' if self.scope else 'off'})[/]"
        )
        result = await asyncio.get_event_loop().run_in_executor(
            None, answer, tier, self.kb, q, self.tools,
            self.scope_dir, self.scope, log.write,
        )
        log.write(result)
        log.write("")
        # Re-focus the input so the next keystroke goes straight in.
        self.query_one("#q", Input).focus()


def _load_kb_text_from_folder(folder: Path, max_bytes: int = 1_000_000) -> str:
    """Concatenate text-like files from a folder into one big KB blob, capped
    at ~1MB. Handles plain text (.md/.txt/.rst/.org), code (.py/.js/.ts/.sh),
    and PDFs (via PyMuPDF if available — each page becomes its own section
    so the model can cite page numbers)."""
    if not folder.exists():
        return f"(collection folder missing: {folder})"
    TEXT_EXTS = {".md", ".markdown", ".txt", ".text", ".rst", ".org"}
    CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".bash",
                 ".rs", ".go", ".c", ".cpp", ".h", ".hpp", ".java", ".rb"}
    PDF_EXTS = {".pdf"}
    texts: list[str] = []
    total = 0
    seen: set[str] = set()
    paths = sorted(
        (p for p in folder.glob("**/*") if p.is_file()),
        key=lambda p: (p.suffix.lower() not in TEXT_EXTS, str(p)),
    )
    for path in paths:
        if str(path) in seen:
            continue
        seen.add(str(path))
        ext = path.suffix.lower()
        try:
            rel = path.relative_to(folder)
        except ValueError:
            rel = path
        if ext in TEXT_EXTS or ext in CODE_EXTS:
            try:
                content = path.read_text(errors="ignore")
            except Exception:                           # noqa: BLE001
                continue
            chunk = f"# {rel}\n\n{content}"
        elif ext in PDF_EXTS:
            try:
                import fitz                                    # PyMuPDF
                doc = fitz.open(str(path))
                pages = []
                for pno in range(doc.page_count):
                    text = doc[pno].get_text().strip()
                    if text:
                        pages.append(f"## {rel} — page {pno + 1}\n\n{text}")
                doc.close()
                if not pages:
                    continue
                chunk = "\n\n".join(pages)
            except ImportError:
                continue                                      # skip PDFs silently
            except Exception:
                continue
        else:
            continue
        texts.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            texts.append(f"\n... (truncated at {max_bytes:,} bytes; "
                         f"the rest of this collection's files are NOT loaded)")
            break
    if not texts:
        return f"(collection folder has no readable text files: {folder})"
    return "\n\n".join(texts)


def _load_collection(name: str) -> tuple[str, Path]:
    """Resolve a collection name via kb_lib.load_collections(); return
    (kb_text, resolved_folder). kb_lib lives at
    ~/.local/share/annas-cli/kb_lib.py — same path the `kb` CLI uses."""
    try:
        sys.path.insert(0, str(Path.home() / ".local" / "share" / "annas-cli"))
        import kb_lib  # type: ignore
    except Exception as e:                          # noqa: BLE001
        raise RuntimeError(f"could not import kb_lib ({e}); install the "
                           "`kb` CLI or copy kb_lib.py next to kb_tui.py")
    cols = kb_lib.load_collections()
    if name not in cols:
        raise RuntimeError(f"unknown collection {name!r}; "
                           f"known: {', '.join(cols) or '(none)'}")
    col = cols[name]
    folder = col.resolved_folder
    return _load_kb_text_from_folder(folder), folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=None,
                    help="path to a flat KB text file (legacy; --collection is preferred)")
    ap.add_argument("--collection", default=None,
                    help="load KB from a named collection in "
                         "~/projects/rag-mcp/collections.json (via kb_lib)")
    ap.add_argument("--scope", default="./notes")
    args = ap.parse_args()

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is set — claude -p will bill "
              "pay-as-you-go, NOT your subscription. Unset it to use the "
              "subscription Agent SDK credit.")

    # Pull MINIMAX_API_KEY from ~/minimaxkey.txt if not already in the env.
    _load_minimax_key()
    if not os.environ.get("MINIMAX_API_KEY"):
        print("NOTE: MINIMAX_API_KEY not set — MiniMax tier (ctrl-E) will fail. "
              "Set it in the environment or write `MINIMAX_API_KEY=sk-...` "
              "to ~/minimaxkey.txt.")

    if not os.environ.get("CONSENSUS_API_KEY"):
        print("NOTE: no CONSENSUS_API_KEY — consensus_search will use OAuth "
              "and open your browser on the first call to consent. Tokens "
              "are cached at ~/.config/kb_tui/consensus_tokens.json. Set "
              "CONSENSUS_API_KEY to use an enterprise Bearer instead.")

    if args.collection:
        try:
            kb_text, folder = _load_collection(args.collection)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        print(f"loaded collection {args.collection!r} from {folder} "
              f"({len(kb_text):,} chars)", file=sys.stderr)
        KBApp(kb_text, folder, kb_name=args.collection).run()
        return

    kb_path = Path(args.kb) if args.kb else Path("kb.txt")
    kb_text = kb_path.read_text() if kb_path.exists() else "(empty KB)"
    KBApp(kb_text, Path(args.scope)).run()


if __name__ == "__main__":
    main()
