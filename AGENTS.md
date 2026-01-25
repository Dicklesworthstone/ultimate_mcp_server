# AGENTS.md - Ultimate MCP Server

> Agent guidelines for developing and maintaining this MCP server.

## RULE 1 - NO FILE DELETION
Do not delete any file or directory unless the user explicitly gives the exact command in this session.

## IRREVERSIBLE GIT & FILESYSTEM ACTIONS
Never run destructive commands (e.g., `git reset --hard`, `git clean -fd`, `rm -rf`) without explicit user approval and the exact command in the same message.

---

## Project Overview
Ultimate MCP Server is a FastAPI + FastMCP server that exposes a large catalog of MCP tools (LLM completions, filesystem, browser, SQL, OCR, vector/RAG, memory systems, etc.) through a unified Gateway. The CLI (`umcp`) runs the server in stdio/SSE/streamable-http modes and supports tool filtering and provider status checks.

## Key Directories
- `ultimate_mcp_server/cli/` - Typer CLI entrypoints and commands.
- `ultimate_mcp_server/core/` - Gateway, server wiring, providers, state store.
- `ultimate_mcp_server/core/providers/` - Provider adapters (OpenAI/Anthropic/Gemini/etc.).
- `ultimate_mcp_server/tools/` - MCP tool implementations and schemas.
- `ultimate_mcp_server/services/` - Cache, analytics, prompts, document, vector, knowledge base.
- `ultimate_mcp_server/utils/` - Logging, display, helpers.
- `tests/` - Unit/integration/manual tests.

---

## Safety First (MCP Server)
### Input Validation
- Always validate tool inputs with Pydantic schemas.
- Never trust agent/client data; sanitize paths, URLs, and query inputs.
- Avoid command injection when wrapping CLI tools.

### Filesystem Boundaries
- Require `filesystem.allowed_directories` (or env `FILESYSTEM__ALLOWED_DIRECTORIES`).
- Do not widen allowed dirs without clear justification.
- Respect path normalization and symlink checks in `tools/filesystem.py`.

### Secrets Handling
- Never hardcode or log API keys/tokens.
- Use `.env` or environment variables for secrets.
- Avoid leaking secrets in exception messages.

### Resource Protection
- Use timeouts for external calls (LLMs, browser, HTTP, DB).
- Be mindful of large downloads and OCR workloads.
- Avoid blocking the event loop; prefer async APIs.

### Network Exposure
- Default to `127.0.0.1` for local-only access.
- If exposing externally, require a reverse proxy, TLS, auth, and rate limiting.

---

## Development Workflow
### Run the Server
- Local stdio: `umcp run`
- Streamable HTTP: `umcp run -t shttp --host 127.0.0.1 --port 8013`
- Load all tools only when needed: `umcp run --load-all-tools`

### Add or Modify Tools
- Implement in `ultimate_mcp_server/tools/` with clear input/output schemas.
- Register tools via the Gateway registration flow in `core/server.py`.
- Ensure errors follow MCP response format and do not leak secrets.

---

## Quality Gates
- Environment: `uv sync` (or `uv sync --all-extras` for OCR/Playwright/etc.).
- Tests: `uv run pytest` (optionally scope to `tests/unit` or `tests/integration`).
- Lint/type checks (if configured):
  - `uv run ruff check ultimate_mcp_server`
  - `uv run mypy ultimate_mcp_server`

---

## Common Pitfalls
- Blocking the event loop with sync I/O in tool handlers.
- Overly broad filesystem allowlists.
- Inconsistent error shapes returned to MCP clients.
- Logging to stdout (should go to stderr to avoid MCP protocol interference).
- Exposing the server on 0.0.0.0 without a proxy/auth.

