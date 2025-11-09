#!/usr/bin/env python3
"""Validate that all expected MCP tools are registered and callable."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import test_sanity

# Re-exported for type checking clarity
Client = test_sanity.Client  # type: ignore
MCP_AVAILABLE = test_sanity.MCP_AVAILABLE
SAMPLE_INGEST_DOC = test_sanity.SAMPLE_INGEST_DOC
SAMPLE_INGEST_TITLE = test_sanity.SAMPLE_INGEST_TITLE


def load_expected_tools(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Expected tool manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Tool manifest must contain a JSON list of tool names")
    return [str(item) for item in data]


async def connect_client(url: str, timeout: int) -> Tuple[Any, Any]:
    base_client = Client(url, timeout=timeout)

    async def _connect(cli: Any) -> Tuple[Any, Any]:
        if hasattr(cli, "__aenter__") and hasattr(cli, "__aexit__"):
            try:
                connected = await cli.__aenter__()
                return connected, cli
            except TypeError:
                pass
        for meth in ("connect", "open", "start", "initialize"):
            fn = getattr(cli, meth, None)
            if callable(fn):
                maybe = fn()
                if asyncio.iscoroutine(maybe):
                    await maybe
                return cli, cli
        return cli, cli

    return await _connect(base_client)


async def list_tools(client: Any) -> Iterable[Any]:
    if hasattr(client, "list_tools"):
        return await client.list_tools()
    if hasattr(client, "tools") and hasattr(client.tools, "list"):
        return await client.tools.list()
    raise RuntimeError("Client does not expose a list_tools API")


async def call_tool(client: Any, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(client, "call_tool"):
        raw = await client.call_tool(name, **params)
    elif hasattr(client, "tools") and hasattr(client.tools, "call"):
        raw = await client.tools.call(name, **params)
    elif hasattr(client, "mcp") and hasattr(client.mcp, "call_tool"):
        raw = await client.mcp.call_tool(name, **params)
    else:
        raise RuntimeError("Client does not support tool invocation API")

    plain, payload = test_sanity._normalise_tool_result(raw)
    if isinstance(plain, dict) and plain.get("is_error"):
        raise RuntimeError(plain.get("error") or plain.get("message") or test_sanity.pretty(plain))
    return payload if payload is not None else plain


async def graceful_close(client: Any, base_client: Any) -> None:
    if hasattr(base_client, "__aexit__"):
        try:
            await base_client.__aexit__(None, None, None)
            return
        except TypeError:
            pass
    for meth in ("close", "disconnect", "stop", "shutdown"):
        fn = getattr(client, meth, None)
        if callable(fn):
            maybe = fn()
            if asyncio.iscoroutine(maybe):
                await maybe
            return


async def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MCP tool registration")
    parser.add_argument("--url", required=True, help="MCP HTTP URL (e.g., http://127.0.0.1:8013/mcp)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    parser.add_argument("--expected", type=Path, default=Path(__file__).with_name("tools_list.json"))
    parser.add_argument("--provider", default="openai", help="Provider used for sample invocations")
    args = parser.parse_args()

    if not MCP_AVAILABLE:
        raise RuntimeError("MCP client not installed. Install fastmcp to run this test.")

    args.url = args.url.rstrip("/")
    print(f"Connecting to MCP server: {args.url}\n")

    expected_tools = load_expected_tools(args.expected)
    client, base_client = await connect_client(args.url, args.timeout)

    try:
        tool_infos = await list_tools(client)
        tool_names = sorted(t.name for t in tool_infos)
        print("Available tools:")
        print("  " + ", ".join(tool_names))
        print()

        missing = sorted(set(expected_tools) - set(tool_names))
        unexpected = sorted(set(tool_names) - set(expected_tools))

        for name in expected_tools:
            status = "✅" if name in tool_names else "❌"
            print(f"== {name} ==")
            print(f"{status} {name} is {'registered' if status == '✅' else 'missing'}")
            print()

        if missing:
            print("Missing tools:")
            for name in missing:
                print(f"  - {name}")
            print()

        if unexpected:
            print("Unexpected tools detected:")
            for name in unexpected:
                print(f"  - {name}")
            print()

        exercises: List[Tuple[str, Callable[[argparse.Namespace], Dict[str, Any]]]] = [
            ("echo", lambda _a: {"message": "toolset diagnostics"}),
            ("get_provider_status", lambda _a: {}),
            ("list_allowed_directories", lambda _a: {}),
            ("list_models", lambda a: {"provider": a.provider}),
        ]

        for tool_name, param_fn in exercises:
            print(f"== {tool_name} ==")
            if tool_name not in tool_names:
                print("SKIP (not registered)")
                print()
                continue
            try:
                payload = await call_tool(client, tool_name, param_fn(args))
            except Exception as exc:
                print(f"❌ exercising {tool_name} ... failed: {exc}")
                print()
                continue

            display = test_sanity.pretty(payload) if isinstance(payload, dict) else str(payload)
            print(f"✅ {tool_name} is registered")
            print(f"   ↪ payload: {display[:400]}" + ("..." if len(display) > 400 else ""))
            print()

        # Document ingestion smoke test (conversion + chunking + batch pipeline)
        print("== document_ingestion ==")
        doc_required = {"write_file", "convert_document", "chunk_document", "process_document_batch"}
        if not doc_required.issubset(set(tool_names)):
            missing = ", ".join(sorted(doc_required - set(tool_names)))
            print(f"SKIP (missing tools: {missing})")
            print()
        else:
            allowed_dirs: List[str] = []
            if "list_allowed_directories" in tool_names:
                try:
                    lad = await call_tool(client, "list_allowed_directories", {})
                    if isinstance(lad, dict):
                        allowed_dirs = lad.get("directories") or []
                    elif isinstance(lad, list):
                        allowed_dirs = lad
                except Exception as exc:
                    print(f"  ↪ failed to discover allowed directories: {exc}")
            if not allowed_dirs:
                print("SKIP (no allowed directories discovered)")
                print()
            else:
                base_dir = allowed_dirs[0].rstrip("/") or "/"
                timestamp = int(time.time())
                if "create_directory" in tool_names:
                    root_dir = f"{base_dir}/tool_ingest_{timestamp}"
                    doc_path = f"{root_dir}/ingest_sample.md"
                else:
                    root_dir = base_dir
                    doc_path = f"{base_dir}/tool_ingest_{timestamp}.md"
                cleanup_paths: List[str] = []
                try:
                    if "create_directory" in tool_names:
                        await call_tool(client, "create_directory", {"path": root_dir})
                        cleanup_paths.append(root_dir)
                    await call_tool(client, "write_file", {"path": doc_path, "content": SAMPLE_INGEST_DOC})
                    cleanup_paths.append(doc_path)

                    convert_resp = await call_tool(
                        client,
                        "convert_document",
                        {
                            "document_path": doc_path,
                            "output_format": "markdown",
                            "enhance_with_llm": False,
                        },
                    )
                    converted = convert_resp if isinstance(convert_resp, dict) else {}
                    content = str(converted.get("content") or "")
                    if not content:
                        raise RuntimeError("convert_document returned empty content")
                    if SAMPLE_INGEST_TITLE not in content:
                        raise RuntimeError(
                            "converted content missing expected marker"
                        )
                    print("  ↪ convert_document ok (content preserved)")

                    chunk_resp = await call_tool(
                        client,
                        "chunk_document",
                        {
                            "document": content,
                            "chunk_method": "paragraph",
                            "chunk_size": 240,
                            "chunk_overlap": 24,
                        },
                    )
                    chunks = []
                    if isinstance(chunk_resp, dict):
                        maybe_chunks = chunk_resp.get("chunks")
                        if isinstance(maybe_chunks, list):
                            chunks = maybe_chunks
                    if not chunks:
                        raise RuntimeError("chunk_document returned no chunks")
                    print(f"  ↪ chunk_document ok ({len(chunks)} chunks)")

                    batch_resp = await call_tool(
                        client,
                        "process_document_batch",
                        {
                            "inputs": [{"document_path": doc_path, "item_id": "ingest"}],
                            "operations": [
                                {
                                    "operation": "convert_document",
                                    "output_key": "converted",
                                    "promote_output": "content",
                                    "params": {
                                        "output_format": "markdown",
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
                    if not isinstance(batch_resp, list) or not batch_resp:
                        raise RuntimeError("process_document_batch returned no items")
                    first = batch_resp[0]
                    if not isinstance(first, dict):
                        raise RuntimeError("process_document_batch first item invalid")
                    if first.get("_status") != "processed":
                        raise RuntimeError(f"batch status {first.get('_status')}")
                    converted_payload = first.get("converted", {})
                    chunked_payload = first.get("chunked", {})
                    if not (isinstance(converted_payload, dict) and converted_payload.get("success")):
                        raise RuntimeError("batch convert step failed")
                    if not (isinstance(chunked_payload, dict) and chunked_payload.get("success")):
                        raise RuntimeError("batch chunk step failed")
                    print("  ↪ process_document_batch ok (convert + chunk)")
                    print()
                except Exception as exc:
                    print(f"❌ document_ingestion failed: {exc}")
                    print()
                finally:
                    if "delete_path" in tool_names:
                        for path in reversed(cleanup_paths):
                            try:
                                await call_tool(client, "delete_path", {"path": path})
                            except Exception:
                                pass

        if missing:
            raise SystemExit(1)

        print("🎉 All expected tools are registered")
    finally:
        await graceful_close(client, base_client)


if __name__ == "__main__":
    asyncio.run(main())

