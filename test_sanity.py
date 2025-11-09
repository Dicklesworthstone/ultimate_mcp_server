#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Tuple
from urllib import request, error as urlerror
from urllib.parse import urlparse

# Try both client imports to be robust (optional)
Client = None  # type: ignore
MCP_AVAILABLE = False
try:
    from mcp.client import Client  # fastmcp client under "mcp"
    MCP_AVAILABLE = True
except Exception:
    try:
        from fastmcp.client import Client  # alternate import path
        MCP_AVAILABLE = True
    except Exception:
        # MCP client not installed; we'll skip MCP checks and still run HTTP API checks
        Client = None  # type: ignore
        MCP_AVAILABLE = False

def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

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
        with request.urlopen(req, timeout=timeout) as resp:
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
    except Exception as e:
        return 0, {"error": str(e)}, {}

async def ahttp_get_json(url: str, timeout: int = 30) -> Tuple[int, Any, Dict[str, str]]:
    return await asyncio.to_thread(http_get_json, url, timeout)

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
        print(f"Warning: could not fetch OpenAPI spec (last status: {last_status}). Skipping generic API checks.\n")
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
    cog_url = f"{api_base.rstrip('/')}/cognitive_states"
    status, body, _ = await ahttp_get_json(cog_url, timeout)
    print(f"GET /cognitive_states -> {status}")
    try:
        print(pretty(body)[:800] + ("..." if isinstance(body, (dict, list)) and len(pretty(body)) > 800 else ""))
    except Exception:
        print(str(body)[:400])
    print()

async def main():
    parser = argparse.ArgumentParser(description="UMCP crash course")
    parser.add_argument("--url", required=True, help="MCP HTTP URL (e.g., http://127.0.0.1:8013/mcp)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    parser.add_argument("--provider", default="openai", help="Provider to demo (default: openai)")
    parser.add_argument("--model", default=None, help="Optional explicit model id for completion")
    parser.add_argument("--api-base", default=None, help="Base URL for the REST API (e.g., http://host:port/api). If not set, derived from --url.")
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
    async def _connect(cli):
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
        tool_names = [t.name for t in tools]
        print("Available tools:")
        print("  " + ", ".join(sorted(tool_names)))
        print()

        # Helper to call a tool by name (consistent across fastmcp versions)
        async def call_tool(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
            # Try common call patterns
            if hasattr(client, "call_tool"):
                return await client.call_tool(name, **params)
            # fastmcp: client.tools.call(name, **params)
            if hasattr(client, "tools") and hasattr(client.tools, "call"):
                return await client.tools.call(name, **params)
            # fastmcp: client.mcp.call_tool(name, **params)
            if hasattr(client, "mcp") and hasattr(client.mcp, "call_tool"):
                return await client.mcp.call_tool(name, **params)
            raise RuntimeError("Client does not support tool invocation API")

        # 2) Provider status (which providers are configured/available)
        print("== get_provider_status ==")
        status = await call_tool("get_provider_status", {})
        print(pretty(status))
        print()

        # 3) List models for a provider (default: openai)
        print(f"== list_models (provider={args.provider}) ==")
        models = await call_tool("list_models", {"provider": args.provider})
        print(pretty(models))
        print()

        # Pick a model if not provided, try a sensible default from status/models
        model_id = args.model
        if not model_id:
            # Try to pick the first model id from the response
            try:
                model_list = models.get("models", {}).get(args.provider, [])
                if model_list and isinstance(model_list, list):
                    # list_models may return list of dicts with "id" field or list of strings
                    first = model_list[0]
                    if isinstance(first, dict) and "id" in first:
                        model_id = f"{args.provider}/{first['id']}"
                    else:
                        # If the list already contains full ids with provider prefix
                        model_id = first if isinstance(first, str) else None
            except Exception:
                pass

        # Fallback if still unknown
        if not model_id:
            # Reasonable default if openai configured
            model_id = f"{args.provider}/gpt-4.1-mini"

        # 4) Estimate cost for a hypothetical prompt
        prompt = "Explain the Model Context Protocol (MCP) in three bullet points."
        print("== estimate_cost ==")
        est = await call_tool("estimate_cost", {"prompt": prompt, "model": model_id})
        print(pretty(est))
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

        text = comp.get("text") or comp.get("completion") or ""
        print("== Completion (first 400 chars) ==")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print()

        # 7) Multi-provider comparison (if both openai/anthropic present)
        print("== multi_completion (if providers available) ==")
        providers_to_try = []
        provs = status.get("providers", {}) if isinstance(status, dict) else {}
        for prov_name, st in provs.items():
            if isinstance(st, dict) and st.get("available") is True:
                providers_to_try.append(prov_name)
        # Keep to 2 providers to save time/cost
        providers_to_try = [p for p in providers_to_try if p in ("openai", "anthropic", "gemini")][:2]

        if len(providers_to_try) >= 2:
            mc = await call_tool(
                "multi_completion",
                {
                    "prompt": "List three practical benefits of using MCP for agent-tooling.",
                    "providers": [{"provider": p} for p in providers_to_try],
                    "temperature": 0.5,
                    "max_tokens": 200,
                },
            )
            print(pretty(mc))
            print()
        else:
            print("Skipping multi_completion (need at least two available providers).")
            print()

        # 8) Exercise the HTTP API endpoints (OpenAPI + Cognitive States)
        await exercise_http_api(api_base, args.timeout)

    finally:
        # Gracefully close using any supported method
        # Try context manager exit first
        if hasattr(base_client, "__aexit__"):
            try:
                await base_client.__aexit__(None, None, None)
                print("Closed MCP connection (ctx).")
                return
            except TypeError:
                pass
        async def _close(cli):
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