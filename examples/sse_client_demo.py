#!/usr/bin/env python3
"""
Demo: Connect to Ultimate MCP Server in SSE mode using the official MCP Python SDK.

Requirements:
    pip install "mcp[cli]"

This script connects to a running Ultimate MCP Server server in SSE mode (default: http://127.0.0.1:8013/sse),
lists available tools, and calls the 'echo' tool if available.
"""

import asyncio
import sys
from typing import Optional

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError:
    print("[ERROR] You must install the MCP Python SDK: pip install 'mcp[cli]'")
    sys.exit(1)

DEFAULT_SSE_URL = "http://127.0.0.1:8013/sse"


async def main(sse_url: Optional[str] = None):
    sse_url = sse_url or DEFAULT_SSE_URL
    print(f"Connecting to MCP SSE server at: {sse_url}\n")
    try:
        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("[OK] Connected. Listing available tools...\n")
                tools = await session.list_tools()
                if not tools:
                    print("[ERROR] No tools available on the server.")
                    return 1
                print("Available tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
                # Try to call the 'echo' tool if available
                echo_tool = next((t for t in tools if t.name == "echo"), None)
                if not echo_tool:
                    print("\n[INFO] 'echo' tool not found. Demo will exit.")
                    return 0
                # Call the echo tool
                test_message = "Hello from SSE client demo!"
                print(f"\nCalling 'echo' tool with message: '{test_message}'...")
                result = await session.call_tool("echo", {"message": test_message})
                print(f"[RESULT] echo: {result}")
                return 0
    except Exception as e:
        print(f"[ERROR] Failed to connect or interact with server: {e}")
        return 1


if __name__ == "__main__":
    sse_url = sys.argv[1] if len(sys.argv) > 1 else None
    exit_code = asyncio.run(main(sse_url))
    sys.exit(exit_code)
