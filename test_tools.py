#!/usr/bin/env python3
"""Sanity test script for Ultimate MCP Server tools.

This script connects to an MCP server and verifies that the expected
core toolset is registered.  It optionally exercises a handful of
lightweight tools to confirm basic functionality without requiring
external resources.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import List, Sequence, Set

from fastmcp import Client

# Core tools we expect to be registered on the server.
EXPECTED_TOOLS: Sequence[str] = [
    "analyze_pdf_structure",
    "autopilot",
    "batch_format_texts",
    "chat_completion",
    "chunk_document",
    "clean_and_format_text_as_markdown",
    "collect_documentation",
    "compare_models",
    "convert_document",
    "create_directory",
    "directory_tree",
    "download",
    "download_site_pdfs",
    "echo",
    "edit_file",
    "enhance_ocr_text",
    "estimate_cost",
    "execute_python",
    "extract_tables",
    "generate_completion",
    "generate_qa_pairs",
    "get_file_info",
    "get_provider_status",
    "get_unique_filepath",
    "list_allowed_directories",
    "list_directory",
    "list_models",
    "move_file",
    "multi_completion",
    "ocr_image",
    "optimize_markdown_formatting",
    "process_document_batch",
    "read_file",
    "read_multiple_files",
    "recommend_model",
    "repl_python",
    "run_awk",
    "run_jq",
    "run_macro",
    "run_ripgrep",
    "run_sed",
    "search",
    "search_files",
    "summarize_document",
    "write_file",
]

# Tools that we lightly exercise after verifying they are present.
# The inputs are intentionally simple to avoid heavy processing or
# dependencies.  Only a subset of tools are exercised because many of
# them require sizeable inputs (PDF files, OCR data, etc.).
LIGHTWEIGHT_TOOL_TESTS = {
    "echo": {"message": "Sanity check"},
    "get_provider_status": {},
    "list_models": {},
    "list_allowed_directories": {},
}


async def run_tool_check(client: Client, tool_name: str) -> bool:
    """Verify that ``tool_name`` is registered and optionally test it.

    Returns ``True`` when the tool is present (regardless of whether the
    optional call succeeds) and ``False`` if the tool is missing.
    """

    tools = await client.list_tools()
    available = {tool.name for tool in tools}

    if tool_name not in available:
        print(f"❌ {tool_name} is not registered")
        return False

    print(f"✅ {tool_name} is registered")

    if tool_name in LIGHTWEIGHT_TOOL_TESTS:
        print(f"   ↪ exercising {tool_name} ...", end=" ")
        payload = LIGHTWEIGHT_TOOL_TESTS[tool_name]
        try:
            result = await client.call_tool(tool_name, payload)
            if result:
                # Convert the first message to plain text for readability.
                text = result[0].text if hasattr(result[0], "text") else str(result[0])
                if tool_name == "get_provider_status":
                    try:
                        provider_summary = json.loads(text)
                        providers = provider_summary.get("providers", {})
                        print(f"ok ({len(providers)} providers)")
                    except json.JSONDecodeError:
                        print("ok (non-JSON response)")
                else:
                    preview = text.strip().splitlines()[0] if text else ""
                    preview = preview[:120]
                    print(f"ok ({preview})")
            else:
                print("ok (no response body)")
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"failed: {exc}")

    return True


async def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Sanity test for MCP tools")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8013/mcp",
        help="MCP server URL (default: %(default)s)",
    )

    args = parser.parse_args(argv)

    print(f"Connecting to MCP server: {args.url}")

    try:
        async with Client(args.url) as client:
            tools = await client.list_tools()
            tool_names = sorted(tool.name for tool in tools)

            print("\nAvailable tools:")
            if tool_names:
                print("  " + ", ".join(tool_names))
            else:
                print("  (none)")

            expected: Set[str] = set(EXPECTED_TOOLS)
            missing = sorted(expected - set(tool_names))
            unexpected = sorted(set(tool_names) - expected)

            if missing:
                print("\nMissing expected tools:")
                for name in missing:
                    print(f"  - {name}")

            if unexpected:
                print("\nAdditional tools present:")
                for name in unexpected:
                    print(f"  - {name}")

            print()
            results: List[bool] = []
            for tool_name in EXPECTED_TOOLS:
                print(f"== {tool_name} ==")
                result = await run_tool_check(client, tool_name)
                results.append(result)
                print()

            if missing:
                print("❌ Some expected tools are missing")
                return 1

            if not all(results):
                print("⚠️ Some tool checks failed — see output above")
                return 2

            print("🎉 All expected tools are registered")
            return 0

    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"❌ Failed to connect or run sanity checks: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
