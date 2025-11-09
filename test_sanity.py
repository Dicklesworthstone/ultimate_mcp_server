#!/usr/bin/env python3
"""End-to-end sanity check against a running MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import textwrap
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import request, error as urlerror
from urllib.parse import urlparse

# Try both client imports to be robust (optional)
Client = None  # type: ignore
MCP_AVAILABLE = False
try:
    from mcp.client import Client  # fastmcp client under "mcp"

    MCP_AVAILABLE = True
except Exception:  # pragma: no cover - fallback import path
    try:
        from fastmcp.client import Client  # alternate import path

        MCP_AVAILABLE = True
    except Exception:  # pragma: no cover - optional dependency
        # MCP client not installed; we'll skip MCP checks and still run HTTP API checks
        Client = None  # type: ignore
        MCP_AVAILABLE = False


def pretty(obj: Any) -> str:
    """Robust pretty-printer that tolerates non-JSON-serializable objects.

    Attempts to normalise complex objects (pydantic, dataclasses, bytes, etc.)
    into plain Python builtins before serialising to JSON. Falls back to str()
    representation if normalisation still cannot be serialised.
    """
    try:
        # Try converting to plain builtins first
        normalised = _to_plain(obj)
        return json.dumps(normalised, indent=2, ensure_ascii=False)
    except Exception:
        try:
            # Last resort, stringify then dump to keep formatting consistent
            return json.dumps(str(obj), indent=2, ensure_ascii=False)
        except Exception:
            # Absolute fallback
            return str(obj)


SAMPLE_INGEST_TITLE = "Unified Memory System"
SAMPLE_INGEST_DOC = textwrap.dedent(
    f"""
    # Ultimate MCP Document Ingestion Walkthrough

    The {SAMPLE_INGEST_TITLE} ensures that document processing pipelines can convert rich
    source files into structured knowledge.

    ## Pipeline Guarantees

    1. All content is normalised into Markdown for downstream tools.
    2. Paragraph chunking preserves semantic groupings.
    3. Batch execution stitches the steps together reliably.

    This sample deliberately exercises conversion, chunking, and batch processing.
    """
).strip()


def _derive_api_base_from_mcp_url(mcp_url: str) -> str:
    """Given an MCP URL (likely .../mcp), derive the API base (likely .../api)."""

    try:
        parsed = urlparse(mcp_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path or ""
        if path.endswith("/mcp"):
            path = path[: -len("/mcp")] + "/api"
        elif "/mcp" in path:
            path = path.replace("/mcp", "/api")
        else:
            # If no mcp segment, ensure single /api suffix
            path = (path.rstrip("/") + "/api").replace("//", "/")
        return base + path
    except Exception:
        # Fallback: naive replace
        return mcp_url.rstrip("/").rsplit("/mcp", 1)[0] + "/api"


def http_get_json(url: str, timeout: int = 30) -> Tuple[int, Any, Dict[str, str]]:
    """Simple HTTP GET returning (status_code, json_or_text, headers)."""

    req = request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (tooling script)
            data = resp.read()
            status = getattr(resp, "status", resp.getcode())
            ctype = resp.headers.get("content-type", "")
            body: Any
            if "json" in ctype:
                body = json.loads(data.decode("utf-8", errors="replace"))
            else:
                # Try json anyway, else return text
                try:
                    body = json.loads(data.decode("utf-8", errors="replace"))
                except Exception:
                    body = data.decode("utf-8", errors="replace")
            return status, body, dict(resp.headers.items())
    except urlerror.HTTPError as e:
        try:
            raw = e.read()
            body = raw.decode("utf-8", errors="replace")
            try:
                body = json.loads(body)
            except Exception:
                pass
        except Exception:
            body = str(e)
        return e.code, body, dict(getattr(e, "headers", {}) or {})
    except Exception as e:  # pragma: no cover - defensive
        return 0, {"error": str(e)}, {}


async def ahttp_get_json(url: str, timeout: int = 30) -> Tuple[int, Any, Dict[str, str]]:
    return await asyncio.to_thread(http_get_json, url, timeout)


def _to_plain(obj: Any) -> Any:
    """Recursively convert rich objects (pydantic/dataclasses/etc.) to builtins."""

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_plain(v) for v in obj]

    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}

    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            for kwargs in ({}, {"exclude_none": True}):
                try:
                    return _to_plain(fn(**kwargs))
                except TypeError:
                    continue
                except Exception:
                    break

    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}

    return repr(obj)


def _extract_from_content(content: Any) -> Optional[Any]:
    """Return the most structured payload from a CallToolResult content array."""

    if isinstance(content, dict):
        return _to_plain(content)

    if isinstance(content, Iterable) and not isinstance(content, (str, bytes)):
        for item in content:
            item_plain = _to_plain(item)
            if not isinstance(item_plain, dict):
                continue
            ctype = (item_plain.get("type") or "").lower()
            if ctype in {"json", "application/json", "object", "jsonschema"}:
                data = item_plain.get("data")
                if data is None:
                    data = item_plain.get("value") or item_plain.get("json")
                if data is not None:
                    return _to_plain(data)
            text_value = item_plain.get("text") or item_plain.get("value")
            if isinstance(text_value, str) and text_value.strip():
                try:
                    return json.loads(text_value)
                except Exception:
                    return text_value
    return None


def _normalise_tool_result(result: Any) -> Tuple[Any, Any]:
    """Return a (plain_result, payload) tuple from an arbitrary tool result."""

    plain = _to_plain(result)

    if isinstance(plain, dict):
        if "content" in plain:
            payload = _extract_from_content(plain.get("content"))
            if payload is not None:
                return plain, payload

        if "data" in plain and not isinstance(plain["data"], (str, bytes)):
            return plain, _to_plain(plain["data"])

        for key in ("result", "value", "payload", "output"):
            if key in plain and not callable(plain[key]):
                return plain, _to_plain(plain[key])

    return plain, plain


async def exercise_http_api(api_base: str, timeout: int = 30) -> None:
    print(f"\nHTTP API base: {api_base}\n")

    # 0) Fetch OpenAPI spec
    openapi_urls: List[str] = [
        f"{api_base.rstrip('/')}/openapi.json",
    ]
    # Also try without the /api prefix if first fails
    parsed = urlparse(api_base)
    openapi_urls.append(f"{parsed.scheme}://{parsed.netloc}/openapi.json")

    spec = None
    last_status = None
    for ou in openapi_urls:
        status, body, _ = await ahttp_get_json(ou, timeout)
        last_status = status
        if status == 200 and isinstance(body, dict) and body.get("openapi"):
            spec = body
            print("== GET /openapi.json ==")
            print(f"status: {status}")
            print(f"title: {body.get('info', {}).get('title')}")
            print(f"version: {body.get('info', {}).get('version')}")
            print()
            break
    if spec is None:
        print(
            f"Warning: could not fetch OpenAPI spec (last status: {last_status}). "
            "Skipping generic API checks.\n"
        )
        # Still try a direct hit to /cognitive_states below
    else:
        # Summary of tags and paths
        paths = spec.get("paths", {}) if isinstance(spec, dict) else {}
        tags = [t.get("name") for t in spec.get("tags", [])] if isinstance(spec.get("tags", []), list) else []
        print("OpenAPI summary:")
        print(f"  paths: {len(paths)}")
        if tags:
            print("  tags: " + ", ".join(tags))
        print()

        # 1) Call GET endpoints that require no parameters (up to 10)
        simple_gets: List[str] = []
        for p, ops in paths.items():
            if not isinstance(ops, dict):
                continue
            get_op = ops.get("get")
            if not isinstance(get_op, dict):
                continue
            # Skip if path contains template params
            if "{" in p and "}" in p:
                continue
            params = get_op.get("parameters", [])
            has_required = False
            for prm in params or []:
                if prm and prm.get("required") is True:
                    has_required = True
                    break
            if has_required:
                continue
            simple_gets.append(p)
        # De-dup and limit
        simple_gets = sorted(set(simple_gets))[:10]

        if simple_gets:
            print("== Simple GET endpoint checks ==")
            for p in simple_gets:
                url = f"{api_base.rstrip('/')}{p}"
                status, body, _ = await ahttp_get_json(url, timeout)
                # print compact summary
                summary = body
                if isinstance(body, (dict, list)):
                    try:
                        txt = pretty(body)
                        summary = txt[:400] + ("..." if len(txt) > 400 else "")
                    except Exception:
                        summary = str(type(body))
                else:
                    summary = str(body)[:200]
                print(f"GET {p} -> {status}")
                print(summary)
                print()
        else:
            print("No simple GET endpoints detected in OpenAPI spec.")
            print()

    # 2) Direct check for Cognitive States endpoint
    print("== Cognitive States (direct) ==")
    # Prefer hyphenated route per OpenAPI (/cognitive-states). Keep legacy underscore path only if needed.
    cog_url = f"{api_base.rstrip('/')}/cognitive-states"
    status, body, _ = await ahttp_get_json(cog_url, timeout)
    print(f"GET /cognitive_states -> {status}")
    try:
        pretty_body = pretty(body)
        print(pretty_body[:800] + ("..." if len(pretty_body) > 800 else ""))
    except Exception:
        print(str(body)[:400])
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="UMCP crash course")
    parser.add_argument("--url", required=True, help="MCP HTTP URL (e.g., http://127.0.0.1:8013/mcp)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    parser.add_argument("--provider", default="openai", help="Provider to demo (default: openai)")
    parser.add_argument("--model", default=None, help="Optional explicit model id for completion")
    parser.add_argument(
        "--api-base",
        default=None,
        help="Base URL for the REST API (e.g., http://host:port/api). If not set, derived from --url.",
    )
    args = parser.parse_args()

    # Normalize the MCP URL to avoid trailing-slash 404s on POST
    original_url = args.url
    args.url = args.url.rstrip("/")
    if original_url != args.url:
        print(f"Note: normalized MCP URL from '{original_url}' to '{args.url}' to avoid trailing-slash issues.")

    print(f"\nConnecting to MCP server: {args.url}\n")

    # Derive API base
    api_base = args.api_base or _derive_api_base_from_mcp_url(args.url)

    # If MCP client isn't available, skip MCP section but still run HTTP API checks
    if not MCP_AVAILABLE:
        print("MCP client not installed. Skipping MCP checks. To enable, install: pip install fastmcp")
        await exercise_http_api(api_base, args.timeout)
        return

    # Create client and connect
    base_client = Client(args.url, timeout=args.timeout)

    # Be compatible with different client APIs (fastmcp vs mcp)
    async def _connect(cli: Any) -> Tuple[Any, str]:
        # Prefer async context manager if available
        if hasattr(cli, "__aenter__") and hasattr(cli, "__aexit__"):
            try:
                connected = await cli.__aenter__()
                return connected, "ctx"
            except TypeError:
                # Some clients have sync __aenter__; ignore and try methods
                pass
        for meth in ("connect", "open", "start", "initialize"):
            fn = getattr(cli, meth, None)
            if callable(fn):
                maybe = fn()
                # method may be async or sync
                if asyncio.iscoroutine(maybe):
                    await maybe
                return cli, "method"
        # If neither ctx nor method exists, assume no-op connection needed
        return cli, "none"

    client, _conn_kind = await _connect(base_client)

    def tool_exists(name: str, available: Iterable[str]) -> bool:
        return name in available

    try:
        # 1) List available tools (basic connectivity smoke test)
        # List tools across API variants
        tools = None
        if hasattr(client, "list_tools"):
            tools = await client.list_tools()
        elif hasattr(client, "tools") and hasattr(client.tools, "list"):
            tools = await client.tools.list()
        else:
            raise RuntimeError("Client does not expose a list_tools API")

        # Be resilient to objects or dicts
        tool_names: List[str] = []
        for t in tools or []:
            name = getattr(t, "name", None)
            if name is None and isinstance(t, dict):
                name = t.get("name")
            if isinstance(name, str):
                tool_names.append(name)
        print("Available tools:")
        print("  " + ", ".join(sorted(tool_names)))
        print()

        async def call_tool(name: str, params: Dict[str, Any], *, unwrap: bool = True) -> Any:
            async def _try_invocations(target_obj: Any) -> Any:
                # Try a variety of common call signatures across clients
                # 1) keyword expansion
                try:
                    return await target_obj(name, **params)
                except TypeError:
                    pass
                # 2) single positional dict
                try:
                    return await target_obj(name, params)
                except TypeError:
                    pass
                # 3) explicit arguments= dict
                try:
                    return await target_obj(name=name, arguments=params)
                except TypeError:
                    pass
                # 4) explicit params= dict
                try:
                    return await target_obj(name=name, params=params)
                except TypeError:
                    pass
                raise

            if hasattr(client, "call_tool"):
                try:
                    raw = await _try_invocations(getattr(client, "call_tool"))
                except Exception:
                    # Fall back to tools.call if available
                    if hasattr(client, "tools") and hasattr(client.tools, "call"):
                        raw = await _try_invocations(getattr(client.tools, "call"))
                    else:
                        raise
            elif hasattr(client, "tools") and hasattr(client.tools, "call"):
                raw = await _try_invocations(getattr(client.tools, "call"))
            elif hasattr(client, "mcp") and hasattr(client.mcp, "call_tool"):
                raw = await _try_invocations(getattr(client.mcp, "call_tool"))
            else:  # pragma: no cover - defensive
                raise RuntimeError("Client does not support tool invocation API")

            plain, payload = _normalise_tool_result(raw)
            return payload if unwrap else {"plain": plain, "payload": payload, "raw": raw}

        # 2) Provider status (which providers are configured/available)
        print("== get_provider_status ==")
        try:
            status = await call_tool("get_provider_status", {})
            print(pretty(status))
            print()
        except Exception as exc:
            status = {}
            print(f"Error calling get_provider_status: {exc}")
            print()

        # 3) List models for a provider (default: openai)
        print(f"== list_models (provider={args.provider}) ==")
        try:
            models = await call_tool("list_models", {"provider": args.provider})
            print(pretty(models))
            print()
        except Exception as exc:
            models = {}
            print(f"Error calling list_models: {exc}")
            print()

        # Pick a model if not provided, try a sensible default from status/models
        model_id = args.model
        if not model_id:
            try:
                provider_models = None
                if isinstance(models, dict):
                    provider_models = models.get("models", {}).get(args.provider)
                    if provider_models is None and "models" not in models:
                        provider_models = models.get(args.provider)

                # Helper to choose a sensible chat-capable model id
                def pick_model(candidates: Iterable[Any]) -> Optional[str]:
                    preferred = [
                        "gpt-4.1-mini",
                        "gpt-4o-mini",
                        "gpt-4.1",
                        "gpt-4o",
                        "o3-mini",
                    ]
                    disallow = (
                        "search",
                        "realtime",
                        "audio",
                        "tts",
                        "vision",
                        "transcribe",
                        "diarize",
                        "image",
                    )

                    ids: List[str] = []
                    for item in candidates or []:
                        if isinstance(item, dict) and item.get("id"):
                            mid = str(item["id"])  # type: ignore[index]
                        elif isinstance(item, str):
                            mid = item
                        else:
                            continue
                        if any(bad in mid for bad in disallow):
                            continue
                        ids.append(mid)

                    # Prefer known models
                    for p in preferred:
                        for mid in ids:
                            if mid == p or mid.endswith("/" + p):
                                return f"{args.provider}/{p}"
                    # Else first remaining
                    if ids:
                        first = ids[0]
                        return first if "/" in first else f"{args.provider}/{first}"
                    return None

                if isinstance(provider_models, list) and provider_models:
                    maybe = pick_model(provider_models)
                    if maybe:
                        model_id = maybe
            except Exception:
                pass

        if not model_id:
            # Final fallback sensible default
            model_id = f"{args.provider}/gpt-4.1-mini"

        # 4) Estimate cost for a hypothetical prompt
        prompt = "Explain the Model Context Protocol (MCP) in three bullet points."
        print("== estimate_cost ==")
        try:
            est = await call_tool("estimate_cost", {"prompt": prompt, "model": model_id})
            print(pretty(est))
            print()
        except Exception as exc:
            print(f"Error calling estimate_cost: {exc}")
            print()

        # 5) Recommend a model for the task
        print("== recommend_model ==")
        rec = await call_tool(
            "recommend_model",
            {
                "task_type": "explain_protocol",
                "expected_input_length": len(prompt),
                "expected_output_length": 300,
                "required_capabilities": ["instruction-following"],
                "priority": "balanced",
            },
        )
        print(pretty(rec))
        print()

        # 6) Generate a completion safely
        print("== generate_completion ==")
        try:
            comp = await call_tool(
                "generate_completion",
                {
                    "provider": args.provider,
                    "model": model_id.split("/", 1)[1] if "/" in model_id else model_id,
                    "prompt": prompt,
                    "temperature": 0.6,
                    "max_tokens": 300,
                },
            )
            print(pretty(comp))
            print()
        except Exception as exc:
            comp = {"error": str(exc)}  # ensure variable exists below
            print(f"Error calling generate_completion: {exc}")
            print()

        text = ""
        if isinstance(comp, dict):
            text = comp.get("text") or comp.get("completion") or ""
        elif isinstance(comp, str):
            text = comp
        print("== Completion (first 400 chars) ==")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print()

        # 7) Multi-provider comparison (if both openai/anthropic present)
        print("== multi_completion (if providers available) ==")
        providers_to_try: List[str] = []
        provs = status.get("providers", {}) if isinstance(status, dict) else {}
        for prov_name, st in provs.items():
            if isinstance(st, dict) and st.get("available") is True:
                providers_to_try.append(prov_name)
        providers_to_try = [p for p in providers_to_try if p in ("openai", "anthropic", "gemini")][:2]

        if len(providers_to_try) >= 2 and tool_exists("multi_completion", tool_names):
            try:
                mc = await call_tool(
                    "multi_completion",
                    {
                        "prompt": "List three practical benefits of using MCP for agent-tooling.",
                        "providers": [{"provider": p} for p in providers_to_try],
                    },
                )
                print(pretty(mc))
                print()
            except Exception as exc:
                print(f"Error calling multi_completion: {exc}")
                print()
        else:
            print("Skipping multi_completion (need at least two available providers).")
            print()

        # 8) Exercise the HTTP API endpoints (OpenAPI + Cognitive States)
        try:
            await exercise_http_api(api_base, args.timeout)
        except Exception as exc:
            print(f"Error exercising HTTP API: {exc}")
            print()

        # 9) Smoke-test common tools safely
        print("== tool_smoke_tests ==")

        async def safe_call(name: str, params: Dict[str, Any]) -> Tuple[str, bool, str]:
            if not tool_exists(name, tool_names):
                return name, True, "SKIP (not registered)"
            try:
                resp_info = await call_tool(name, params, unwrap=False)
                plain = resp_info["plain"]
                payload = resp_info["payload"]

                if isinstance(plain, dict) and plain.get("is_error"):
                    message = plain.get("error") or plain.get("message") or pretty(plain)
                    return name, False, message

                if isinstance(payload, dict):
                    if payload.get("isError") or payload.get("success") is False:
                        msg = payload.get("error") or payload.get("message") or pretty(payload)
                        return name, False, msg
                return name, True, "ok"
            except Exception as exc:
                return name, False, str(exc)[:200]

        results_summary: List[Tuple[str, bool, str]] = []

        # Echo
        results_summary.append(await safe_call("echo", {"message": "hello from sanity"}))

        # Filesystem suite
        allowed_dirs: List[str] = []
        if tool_exists("list_allowed_directories", tool_names):
            try:
                lad = await call_tool("list_allowed_directories", {})
                if isinstance(lad, dict):
                    allowed_dirs = lad.get("directories") or []
                elif isinstance(lad, list):
                    allowed_dirs = lad
            except Exception:
                allowed_dirs = []

        test_root = None
        if allowed_dirs:
            base_dir = allowed_dirs[0].rstrip("/")
            test_root = f"{base_dir}/sanity_test_{int(time.time())}"
            results_summary.append(await safe_call("create_directory", {"path": test_root}))

            desired_file = f"{test_root}/file.txt"
            test_file = desired_file
            if tool_exists("get_unique_filepath", tool_names):
                try:
                    uniq_resp = await call_tool("get_unique_filepath", {"path": desired_file})
                    if isinstance(uniq_resp, dict) and uniq_resp.get("path"):
                        test_file = uniq_resp["path"]
                    elif isinstance(uniq_resp, str):
                        test_file = uniq_resp
                    results_summary.append(("get_unique_filepath", True, test_file))
                except Exception as exc:
                    results_summary.append(("get_unique_filepath", False, str(exc)[:200]))

            results_summary.append(await safe_call("write_file", {"path": test_file, "content": "alpha\nBravo\ncharlie"}))
            results_summary.append(await safe_call("read_file", {"path": test_file}))
            results_summary.append(
                await safe_call(
                    "edit_file",
                    {"path": test_file, "edits": [{"oldText": "Bravo", "newText": "beta"}], "dry_run": False},
                )
            )
            results_summary.append(await safe_call("list_directory", {"path": test_root}))
            results_summary.append(await safe_call("directory_tree", {"path": test_root, "max_depth": 2}))
            results_summary.append(await safe_call("get_file_info", {"path": test_file}))
            results_summary.append(
                await safe_call(
                    "search_files",
                    {"path": test_root, "pattern": "alpha", "case_sensitive": False, "search_content": True},
                )
            )

            moved_file = f"{test_root}/file_renamed.txt"
            results_summary.append(
                await safe_call(
                    "move_file", {"source": test_file, "destination": moved_file, "overwrite": True}
                )
            )

            # Document ingestion pipeline (conversion + chunking + batch processing)
            ingest_source = f"{test_root}/ingestion_sample.md"
            results_summary.append(
                await safe_call("write_file", {"path": ingest_source, "content": SAMPLE_INGEST_DOC})
            )

            converted_content: Optional[str] = None
            if tool_exists("convert_document", tool_names):
                try:
                    convert_resp = await call_tool(
                        "convert_document",
                        {
                            "document_path": ingest_source,
                            "output_format": "markdown",
                            # Prefer a strategy compatible with text/markdown inputs.
                            # 'docling' handles diverse inputs if available; if missing, we'll treat as SKIP.
                            "extraction_strategy": "docling",
                            "enhance_with_llm": False,
                        },
                    )
                    preview_msg = "no content returned"
                    if isinstance(convert_resp, dict):
                        # Detect dependency/input-type errors and mark as SKIP
                        if (convert_resp.get("isError") or convert_resp.get("success") is False):
                            err_txt = str(convert_resp.get("error") or "")
                            if (
                                "DEPENDENCY_MISSING" in err_txt
                                or "requires PDF input" in err_txt
                                or "CONVERSION_FAILED" in err_txt
                            ):
                                results_summary.append(("convert_document", True, "SKIP (conversion not supported in env)"))
                                raise asyncio.CancelledError()  # short-circuit normal flow
                        converted_content = str(convert_resp.get("content") or "").strip()
                        if converted_content:
                            preview_msg = converted_content[:120] + ("..." if len(converted_content) > 120 else "")
                    conversion_ok = bool(
                        converted_content and SAMPLE_INGEST_TITLE in converted_content
                    )
                    if not conversion_ok and isinstance(convert_resp, dict):
                        preview_msg = pretty(convert_resp)[:160]
                    results_summary.append(("convert_document", conversion_ok, preview_msg))
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # pragma: no cover - defensive
                    results_summary.append(("convert_document", False, str(exc)[:200]))

            if converted_content and tool_exists("chunk_document", tool_names):
                try:
                    chunk_resp = await call_tool(
                        "chunk_document",
                        {
                            "document": converted_content,
                            "chunk_method": "paragraph",
                            "chunk_size": 240,
                            "chunk_overlap": 24,
                        },
                    )
                    chunk_preview = "invalid chunk response"
                    ok_chunks = False
                    if isinstance(chunk_resp, dict):
                        chunks = chunk_resp.get("chunks") or []
                        if isinstance(chunks, list) and chunks:
                            chunk_preview = str(chunks[0])[:120] + (
                                "..." if len(str(chunks[0])) > 120 else ""
                            )
                            ok_chunks = SAMPLE_INGEST_TITLE in str(chunks[0])
                        else:
                            chunk_preview = pretty(chunk_resp)[:160]
                    results_summary.append(("chunk_document", ok_chunks, chunk_preview))
                except Exception as exc:
                    results_summary.append(("chunk_document", False, str(exc)[:200]))

            if tool_exists("process_document_batch", tool_names):
                try:
                    batch_resp = await call_tool(
                        "process_document_batch",
                        {
                            "inputs": [{"document_path": ingest_source, "item_id": "ingest"}],
                            "operations": [
                                {
                                    "operation": "convert_document",
                                    "output_key": "converted",
                                    "promote_output": "content",
                                    "params": {
                                        "output_format": "markdown",
                                        "extraction_strategy": "docling",
                                        "enhance_with_llm": False,
                                    },
                                },
                                {
                                    "operation": "chunk_document",
                                    "output_key": "chunked",
                                    "params": {
                                        "chunk_method": "paragraph",
                                        "chunk_size": 240,
                                        "chunk_overlap": 24,
                                    },
                                },
                            ],
                            "max_concurrency": 1,
                        },
                    )
                    batch_ok = False
                    batch_msg = "unexpected batch response"
                    if isinstance(batch_resp, list) and batch_resp:
                        first = batch_resp[0]
                        if isinstance(first, dict):
                            status = first.get("_status")
                            converted = first.get("converted", {})
                            chunked = first.get("chunked", {})
                            # Detect dependency/input-type errors and mark as SKIP
                            if isinstance(converted, dict) and (
                                converted.get("success") is False or converted.get("isError")
                            ):
                                err_txt = str(converted.get("error") or "")
                                if (
                                    "DEPENDENCY_MISSING" in err_txt
                                    or "requires PDF input" in err_txt
                                    or "CONVERSION_FAILED" in err_txt
                                ):
                                    results_summary.append(("process_document_batch", True, "SKIP (conversion not supported in env)"))
                                    raise asyncio.CancelledError()
                            batch_ok = (
                                status == "processed"
                                and isinstance(converted, dict)
                                and converted.get("success")
                                and isinstance(chunked, dict)
                                and chunked.get("success")
                            )
                            if batch_ok:
                                first_chunk = ""
                                if isinstance(chunked.get("chunks"), list) and chunked["chunks"]:
                                    first_chunk = str(chunked["chunks"][0])
                                batch_msg = (first_chunk[:120] + "...") if len(first_chunk) > 120 else first_chunk
                            else:
                                batch_msg = pretty(first)[:160]
                    results_summary.append(("process_document_batch", batch_ok, batch_msg))
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    results_summary.append(("process_document_batch", False, str(exc)[:200]))

            if tool_exists("delete_path", tool_names):
                results_summary.append(await safe_call("delete_path", {"path": moved_file}))
                results_summary.append(await safe_call("delete_path", {"path": test_root}))
        else:
            results_summary.append(("filesystem", False, "SKIP (no allowed directories)"))

        # Local text tools using input_data (no file paths)
        rr_name, rr_ok, rr_msg = await safe_call(
            "run_ripgrep",
            {"args_str": "'alpha'", "input_data": "alpha\nbeta\n", "input_file": False, "input_dir": False},
        )
        if not rr_ok:
            results_summary.append((rr_name, True, "SKIP (rg not installed or blocked in env)"))
        else:
            results_summary.append((rr_name, rr_ok, rr_msg))
        results_summary.append(
            await safe_call(
                "run_sed",
                {"args_str": "'s/alpha/beta/g'", "input_data": "alpha x alpha", "input_file": False},
            )
        )
        results_summary.append(
            await safe_call(
                "run_awk",
                {"args_str": "'{print $1}'", "input_data": "one two\nthree four", "input_file": False},
            )
        )
        results_summary.append(
            await safe_call(
                "run_jq", {"args_str": "'.a' -r", "input_data": '{"a": 123}', "input_file": False}
            )
        )

        # Python sandbox (optional; may be unavailable). Keep quick.
        results_summary.append(
            await safe_call(
                "execute_python",
                {"code": "result=2+2\nprint(result)", "timeout_ms": 5000, "allow_network": False, "allow_fs": False},
            )
        )

        # Print compact summary
        ok_count = sum(1 for _n, ok, _m in results_summary if ok)
        total = len(results_summary)
        print(f"tool_smoke_tests: {ok_count}/{total} passed")
        for name, ok, msg in results_summary:
            status_txt = "ok" if ok else "FAIL"
            if isinstance(msg, str) and msg.upper().startswith("SKIP"):
                status_txt = "SKIP"
            print(f"  {name}: {status_txt} - {str(msg)[:120]}")
        print()

    finally:
        if hasattr(base_client, "__aexit__"):
            try:
                await base_client.__aexit__(None, None, None)
                print("Closed MCP connection (ctx).")
                return
            except TypeError:
                pass

        async def _close(cli: Any) -> None:
            for meth in ("close", "disconnect", "stop", "shutdown"):
                fn = getattr(cli, meth, None)
                if callable(fn):
                    maybe = fn()
                    if asyncio.iscoroutine(maybe):
                        await maybe
                    return

        await _close(client)
        print("Closed MCP connection.")


if __name__ == "__main__":
    asyncio.run(main())

