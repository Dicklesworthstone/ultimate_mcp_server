"""Utility functions for Ultimate MCP Server."""

from ultimate_mcp_server.utils.logging.console import console
from ultimate_mcp_server.utils.logging.logger import (
    critical,
    debug,
    error,
    get_logger,
    info,
    logger,
    section,
    success,
    warning,
)
from ultimate_mcp_server.utils.parsing import parse_result, process_mcp_result

__all__ = [
    # Logging utilities
    "logger",
    "console",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
    "section",
    "get_logger",
    # Parsing utilities
    "parse_result",
    "process_mcp_result",
]
