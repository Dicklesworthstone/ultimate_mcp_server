# Ultimate MCP Server - AGENTS.md Research Summary

## Project Overview (1 paragraph)
Ultimate MCP Server is a Python 3.13+ FastAPI/FastMCP server that exposes a large catalog of MCP tools for AI agents (LLM completions, filesystem, browser automation, OCR, SQL, vector/RAG, memory systems, etc.). The core `Gateway` orchestrates provider clients (OpenAI/Anthropic/Gemini/etc.), tool registration, caching, analytics, and state persistence, and the CLI (`umcp`) starts the server in stdio/SSE/streamable-http modes. The design centers on MCP-native tool discovery and structured schemas, with optional tool filtering and a base-toolset default.

## Key Directories and Purposes
- `ultimate_mcp_server/` - Python package root.
  - `cli/` - Typer CLI entrypoints and commands (`umcp run`, provider status, etc.).
  - `core/` - `Gateway`, FastMCP server wiring, provider registry, state store, UMS API glue.
  - `core/providers/` - Provider adapters (openai, anthropic, gemini, grok, openrouter, etc.).
  - `tools/` - MCP tool implementations (filesystem, browser, SQL, OCR, memory, etc.).
  - `services/` - Shared services (cache, analytics, prompts, document, vector, knowledge_base).
  - `utils/` - Logging, display, helper utilities.
  - `config/` - Config examples and configuration models (`config.py`).
  - `clients/` - Provider-specific client wrappers.
- `tests/` - Unit/integration/manual tests.
- `examples/` - Usage examples and demo scripts.
- `storage/` - Runtime data (e.g., `smart_browser_internal`).
- Root scripts (e.g., `check_api_keys.py`, `run_all_demo_scripts_and_check_for_errors.py`) for diagnostics.

## Safety Rules to Include in AGENTS.md
- Configure `filesystem.allowed_directories` (or env `FILESYSTEM__ALLOWED_DIRECTORIES`) and avoid broad paths; filesystem tools enforce allowlists.
- Avoid exposing the server publicly without a reverse proxy, auth, and rate limiting; default bind should be `127.0.0.1`.
- Never hardcode or commit API keys; use `.env` or environment variables only.
- Treat SQL, CLI, and browser tools as high-risk surfaces; require input validation and avoid command injection patterns.
- Prefer least-privilege credentials for databases and external APIs.
- Be mindful of tool cost and resource usage (LLM calls, browser sessions, OCR) to avoid runaway spend.

## Workflow Recommendations
- Start server locally via `umcp run` (stdio) or `umcp run -t shttp` for HTTP clients.
- Use tool filtering (`--include-tools`, `--exclude-tools`) and default base toolset unless full catalog is required.
- Use `tools_list.json` or MCP `list_tools` for discovery before calling tools.
- Keep configuration in `gateway_config.yaml` or `~/.config/ultimate_mcp_server/config.yaml`; prefer `.env` for secrets.

## Quality Gates
- Environment: `uv sync` (or `uv sync --all-extras` when you need OCR/Playwright/etc.).
- Tests: `uv run pytest` (and optionally `uv run pytest tests/unit` / `tests/integration`).
- Lint/type checks (if used): `uv run ruff check ultimate_mcp_server` and `uv run mypy ultimate_mcp_server`.

