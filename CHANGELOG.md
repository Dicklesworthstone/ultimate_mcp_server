# Changelog

All notable changes to [Ultimate MCP Server](https://github.com/Dicklesworthstone/ultimate_mcp_server) are documented here.

This project has no formal release tags or GitHub Releases. The changelog below is reconstructed from the full git history, organized into logical milestone periods based on when major capabilities were introduced. Commit hashes link directly to GitHub.

---

## Milestone: Infrastructure and Governance (2026-01-14 through 2026-02-22)

Hardening pass: CI/CD, linting, licensing, and project metadata.

### CI/CD

- Add GitHub Actions CI workflow (ruff, mypy, pytest with coverage) and PyPI release workflow ([`1027820`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/10278201ff0f08081d43812e13c46c7121a758e3))

### Bug Fixes

- Restore missing `core/models` module with tournament and request Pydantic models, fixing `ModuleNotFoundError` on import ([`3b95914`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/3b959149017f9072d3485e9c531acfa1d842fded))
- Resolve ruff lint errors across codebase ([`ac0442d`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/ac0442dc3321034027bcc0d3786ced6c53fc2596))
- Apply ruff formatting and fix E741 ambiguous variable name ([`a2a425b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/a2a425bbd7e2b0853b13441ac3726bde3885e982))
- Ignore ruff star-import warnings in UMS API layer ([`a190b57`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/a190b5705b2a45e62e70ab4734d194c008ede363))

### Chores

- Update Python dependencies to latest stable versions ([`6e8066d`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/6e8066d9092e93497ce4970f5e262ee0399d256f))
- Add MIT License, later updated to MIT with OpenAI/Anthropic Rider ([`717399f`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/717399f7a10f96a2986c7af8f52a03433216df8b), [`95a8099`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/95a80998270a1fd258b826ef59a8051f54f1ac49))
- Update README license references ([`79e0aeb`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/79e0aeb2e5bbf19f13174505560dfc66621c945d))
- Add GitHub social preview image ([`3c3b57b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/3c3b57b2cb902b9fe7bfd130dd0427a470274009))
- Add AGENTS.md guidance ([`d681987`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/d68198794718cd99998290997fe2d2e70c218869))
- Add .gitignore patterns for NTM and compiled binaries ([`aa3e7bf`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/aa3e7bfca0500d5df2a7a6442bb3262d37ce9686))

---

## Milestone: Web Obstacle Detection and Dependency Overhaul (2025-09-19)

### Smart Browser

- Enhance web obstacle detection in smart browser tools (CAPTCHA, cookie walls, paywalls) ([`45d8baa`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/45d8baa9bf9b88062cafeb2af41a6dd0628c6f38))

### Dependencies

- Refactor dependency management in pyproject.toml; update uv.lock ([`45d8baa`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/45d8baa9bf9b88062cafeb2af41a6dd0628c6f38))

### Text Redline

- Significant revision of text redline tools ([`45d8baa`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/45d8baa9bf9b88062cafeb2af41a6dd0628c6f38))

---

## Milestone: Streaming-HTTP and UMS API Refactor (2025-06-13 through 2025-06-14)

Major transport upgrade and server architecture refactoring.

### Transport

- Add streaming-HTTP (`shttp`) transport support via latest FastMCP; server core rewritten (~11k-line diff) ([`c422335`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/c422335d2134ed5209864edefeb212dcf6c16a42))
- Add test clients for SSE, stdio, and HTTP transports ([`c422335`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/c422335d2134ed5209864edefeb212dcf6c16a42))

### Architecture

- Extract UMS API into dedicated module (`core/ums_api/`): endpoints, database, models, services -- removing ~5,400 lines from `server.py` ([`eb4efd9`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/eb4efd98d90b56efec1576962fe3d2b524a06bef))

### Unified Memory System

- Major rewrite of unified memory system tools ([`c422335`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/c422335d2134ed5209864edefeb212dcf6c16a42))

---

## Milestone: Unified Memory Expansion and Stabilization (2025-05-15 through 2025-05-30)

Focus on the unified memory system, new UMS tools, and cross-tool stability.

### Unified Memory System

- Add new UMS tools; major expansion of unified memory system (+6,170 lines) ([`49f47f3`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/49f47f3ca2ef55e56665527887c023213dc984ae))
- Refactoring of unified memory system with additional dependencies ([`3180fe5`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/3180fe5b6a4a2be29007497351e6e7a8dc1bf8ad))

### Stability

- Sustained bug-fix and improvement cycle across all tools ([`b5387a8`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b5387a8417a749be7e6d7aec4e1fd7040d72f5a0) through [`ecb646c`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/ecb646c47050a2a2a32ef3f00775bb0c2d39388a))

---

## Milestone: Eidentic Engine, Python Sandbox, and Provider Hardening (2025-05-02 through 2025-05-07)

### Python Sandbox

- Pyodide-based Python sandbox working end-to-end; add HTML boot template for browser-based execution ([`0744817`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/0744817f24e09c600168ae49100f53d72c6f0c9b))

### Memory System

- Refactoring for eidentic engine integration (+899 lines to unified memory system) ([`755c3c5`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/755c3c5c6405b5e74681ac92206b94695cb865bd))
- Fixes to memory system ([`75ea925`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/75ea925e60267df32435262d7e84afa1e8af8278))

### Context Optimization

- Shorten UMS docstrings to save context window space for agent callers ([`badfebf`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/badfebffd55737411a11bfd370a092e23763b9ce))

### Provider Hardening

- Major improvements to Anthropic, DeepSeek, Gemini, Grok, and Ollama providers ([`42e8e64`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/42e8e64821a8b7a178ba1f3a111922dddd757f3d))

### New Tools

- Add local text tools (ripgrep, awk, sed, jq as MCP tools) ([`42e8e64`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/42e8e64821a8b7a178ba1f3a111922dddd757f3d))

---

## Milestone: Big Restructuring (2025-04-26 through 2025-04-29)

Large-scale reorganization of the tools layer and smart browser rewrite.

### Architecture

- Big restructuring: reorganize tools directory, split `python_js_sandbox` into dedicated `python_sandbox` module, restructure SQL databases and smart browser tools ([`871e53e`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/871e53e743ab26c506ac7c934f5a4e3ee4689266))
- Move legacy browser automation and SQL tools to `tools/old/` ([`716f108`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/716f10804732fdef9de518ba17d00e02abf6bddb))

### Smart Browser

- Introduce `smart_browser.py` as replacement for legacy `browser_automation.py`; complete Playwright-based rewrite ([`716f108`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/716f10804732fdef9de518ba17d00e02abf6bddb))
- Further smart browser revisions and improvements ([`e90b729`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/e90b729ee5ce6399c83c6df66fbf1f5f464033e6), [`29fe7af`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/29fe7af0c94eefdce40219a5996bb25e1c5694ad))

### SQL Databases

- Introduce rewritten `sql_databases.py` replacing legacy SQL tools ([`716f108`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/716f10804732fdef9de518ba17d00e02abf6bddb))

### Python Sandbox

- Add initial `python_js_sandbox.py` with Python/JS code execution ([`716f108`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/716f10804732fdef9de518ba17d00e02abf6bddb))

---

## Milestone: Ollama Provider and Feature Expansion (2025-04-22 through 2025-04-23)

### Providers

- Add Ollama provider for local model support ([`4b175c9`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/4b175c9aed5edacda57d4f96c8eace5e191b1e5e), [`b65e8a9`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b65e8a97ea3fe5055c5303b55b5a981228711511))
- Add YAML config parsing for Ollama ([`4b175c9`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/4b175c9aed5edacda57d4f96c8eace5e191b1e5e))

### New Tools

- Add MCP tool context estimator for token counting ([`b65e8a9`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b65e8a97ea3fe5055c5303b55b5a981228711511))
- Add tool token counter utility ([`b65e8a9`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b65e8a97ea3fe5055c5303b55b5a981228711511))

### New Tools

- Add sentiment analysis tool ([`ad438bb`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/ad438bbda1a5116bce0afd9a6829716286fceaec))

### SSE Transport

- Fix SSE transport issues ([`1cd05cd`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/1cd05cdbcd3fed2ff6fee38440b9bbbfc24e572f))

### CLI

- Fixes to CLI commands ([`56ab344`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/56ab34434847495f61cbb4e069f0995ba42713d1))

---

## Milestone: Docstrings, Documentation, and Renaming (2025-04-17 through 2025-04-19)

### Project Identity

- Rename project from "LLM Gateway" to "Ultimate MCP Server"; rename all modules from `llm_gateway` to `ultimate_mcp_server` ([`979a9df`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/979a9df12d942ed5e27482647bd0ae9d3c977ea4))

### Documentation

- Dramatically improve docstrings across entire codebase; major README expansion (+529 lines) ([`5279590`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/52795901359b5da653254900ad295078d7f12357))
- Add technical analysis writeup of the unified memory system ([`1a5468a`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/1a5468a1bd3e55c5c0c48e8a7f44e58fd7f95ada))

### New Tools

- Add autonomous docstring refiner tool (2,806 lines); uses LLM ensembles to iteratively improve MCP tool documentation ([`6114c73`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/6114c736d05e60a4e58c38c5fecb5df81849199f))
- Add document refining tool ([`4c32d52`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/4c32d52738353946e73e25514c2f23fa7e229aa4))

---

## Milestone: BIG UPDATE -- Entity Graphs, Grok, OCR, Excel, and More (2025-04-15 through 2025-04-16)

The single largest capability expansion: six new tool modules in one push.

### Providers

- Add xAI Grok provider ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Expanded OpenRouter provider with enhanced model handling ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))

### New Tools

- Add entity relation graph tool -- entity extraction, relationship mapping, knowledge graph construction ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add OCR tools -- PDF/image text extraction with LLM enhancement ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add Excel spreadsheet automation -- creation, formatting, VBA macro generation ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add audio transcription tool -- Whisper-based speech-to-text ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add text classification tool ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add text redline tools -- HTML diff generation for document comparison ([`e674ef2`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/e674ef28412528a69f562f88a75a6c9398cf8653))
- Add HTML to Markdown conversion tool ([`b2cf714`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b2cf714362f893cd42767084bfd70e5ef19c52c2))
- Enhance audio transcription with expanded format support ([`403a596`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/403a596bc0d0d741aded7ffb54086ff86919c333))

### Benchmarking

- Add empirically measured model speed data and benchmarking tooling ([`b2a0ff3`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b2a0ff32d89e1f6a3f52e9e0aa91b9f9c0a43ab0))
- Expand model speed measurements with new providers ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))

### Examples

- Add Grok integration demo ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add meta API demo ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add research workflow demo ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add entity relation graph demo ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add text classification demo ([`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f))
- Add web automation instruction packs ([`403a596`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/403a596bc0d0d741aded7ffb54086ff86919c333))

---

## Milestone: Filesystem, SQL, Browser, and Memory System (2025-04-08 through 2025-04-13)

The platform expands from an LLM gateway into a comprehensive AI agent operating system.

### New Tools

- Add secure filesystem tools -- read, write, edit, search, directory operations with path validation and traversal protection ([`f9427eb`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/f9427ebcdfa41ed4ba6e00262270d1a65e3991e6), [`b011caa`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b011caa71d0acf02281a7fdc14759e98ac8cb493))
- Add SQL database interaction tools -- query execution, schema analysis, data exploration via SQLAlchemy ([`b011caa`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b011caa71d0acf02281a7fdc14759e98ac8cb493))
- Add Playwright-based browser automation tools -- navigation, interaction, scraping, screenshots, research ([`b011caa`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b011caa71d0acf02281a7fdc14759e98ac8cb493))
- Add cognitive and agent memory system -- working, episodic, semantic, procedural memory hierarchy; workflow tracking; memory consolidation and reflection ([`b011caa`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b011caa71d0acf02281a7fdc14759e98ac8cb493))

### Providers

- Add OpenRouter provider -- access diverse models via OpenRouter API key ([`02da0fb`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/02da0fbf569ee922456609c0b827c2344e977008))

### Claude Desktop

- Verified working integration with Claude Desktop ([`01356db`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/01356db06cf4546ce822e66147a6e928aa866374))

---

## Milestone: Massive Refactoring and Tournament Mode (2025-03-29 through 2025-04-01)

### New Tools

- Add multi-model compare and synthesize tool flow ([`f12829f`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/f12829f7960f2ff40e95169f1669d49e192b617b))
- Tournament mode: head-to-head model competitions for code and text generation ([`ee26f47`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/ee26f470108c0f159427f515c864936b0af684a6))

### Architecture

- Massive refactoring of provider and tool infrastructure ([`ee26f47`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/ee26f470108c0f159427f515c864936b0af684a6))
- Add code extraction testing ([`ee26f47`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/ee26f470108c0f159427f515c864936b0af684a6))

### Examples

- Add tournament code demo ([`ee26f47`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/ee26f470108c0f159427f515c864936b0af684a6))
- Add tournament text demo ([`ee26f47`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/ee26f470108c0f159427f515c864936b0af684a6))

---

## Milestone: Initial Release (2025-03-25 through 2025-03-26)

First public upload of the project (originally named "LLM Gateway").

### Core

- MCP-native server built on the Model Context Protocol via FastMCP ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- CLI (`umcp`) for server management and interaction ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Docker and docker-compose configuration ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))

### Providers (initial)

- OpenAI provider ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Anthropic (Claude) provider ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Google Gemini provider ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- DeepSeek provider ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))

### Tools (initial)

- Completion tools -- generate, chat, multi-completion, streaming ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Document tools -- smart chunking, summarization, batch processing ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Extraction tools -- JSON, table, key-value, semantic schema extraction ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Meta tools -- tool discovery, usage recommendations ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Optimization tools -- cost estimation, model comparison, workflow execution ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))

### Services (initial)

- Multi-level caching -- exact match, semantic similarity, task-aware; disk persistence ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Analytics and reporting -- usage tracking, cost monitoring ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Prompt template system -- Jinja2 templates, repository, versioning ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))
- Vector service -- embeddings, semantic search ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))

### Examples (initial)

- Basic completion, cache, Claude integration, cost optimization, document processing demos ([`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a))

---

## Cumulative Tool Inventory

For agent reference, the full set of tool modules present in the codebase as of [`79e0aeb`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/79e0aeb2e5bbf19f13174505560dfc66621c945d) (HEAD):

| Module | Status | First Appeared |
|--------|--------|----------------|
| `completion` | Active | 2025-03-25 |
| `document_conversion_and_processing` | Active | 2025-03-25 |
| `extraction` | Commented out | 2025-03-25 |
| `optimization` | Active | 2025-03-25 |
| `filesystem` | Active | 2025-04-13 |
| `sql_databases` | Commented out | 2025-04-13 / rewritten 2025-04-26 |
| `browser_automation` (legacy) | Replaced | 2025-04-13 |
| `cognitive_and_agent_memory` (legacy) | Replaced | 2025-04-13 |
| `entity_relation_graph` | Commented out | 2025-04-15 |
| `ocr_tools` | Active (via doc processing) | 2025-04-15 |
| `excel_spreadsheet_automation` | Active (Windows only) | 2025-04-15 |
| `audio_transcription` | Commented out | 2025-04-15 |
| `text_classification` | Commented out | 2025-04-15 |
| `text_redline_tools` | Commented out | 2025-04-15 |
| `html_to_markdown` | Active (via doc processing) | 2025-04-16 |
| `docstring_refiner` | Commented out | 2025-04-17 |
| `marqo_fused_search` | Commented out | 2025-04-18 |
| `rag` | Commented out | 2025-04-01 |
| `sentiment_analysis` | Active | 2025-04-22 |
| `smart_browser` | Active | 2025-04-26 |
| `python_sandbox` | Active | 2025-04-29 |
| `unified_memory_system` | Active | 2025-05-03 |
| `single_shot_synthesis` | Commented out | 2025-04-18 |
| `tournament` | Commented out | 2025-04-18 |
| `local_text_tools` | Active | 2025-05-04 |
| `meta_api_tool` | Registered dynamically | 2025-04-18 |
| `provider` | Active | 2025-03-25 |

## Provider Support Timeline

| Provider | First Appeared |
|----------|----------------|
| OpenAI | [`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a) (2025-03-25) |
| Anthropic | [`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a) (2025-03-25) |
| Google Gemini | [`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a) (2025-03-25) |
| DeepSeek | [`70a63cc`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/70a63cce51890a1358291418e5bba1ba39355b4a) (2025-03-25) |
| OpenRouter | [`02da0fb`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/02da0fbf569ee922456609c0b827c2344e977008) (2025-04-11) |
| xAI Grok | [`fb3207b`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/fb3207b9d2a984c658c5138b8697c18f31841f2f) (2025-04-15) |
| Ollama (local) | [`b65e8a9`](https://github.com/Dicklesworthstone/ultimate_mcp_server/commit/b65e8a97ea3fe5055c5303b55b5a981228711511) (2025-04-22) |
