"""Command-line interface for Ultimate MCP Server."""

# Modern CLI implementation using typer
from ultimate_mcp_server.cli.typer_cli import app, cli

__all__ = ["app", "cli"]
