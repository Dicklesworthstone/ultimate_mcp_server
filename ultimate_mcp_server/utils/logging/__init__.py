"""
Gateway Logging Package.

This package provides enhanced logging capabilities with rich formatting,
progress tracking, and console output for the Gateway system.
"""

import logging
import logging.handlers
from typing import Any, Dict, List, Optional

# Import Rich-based console
# Adjusted imports to be relative within the new structure
from .console import (
    console,
    create_progress,
    live_display,
    print_json,
    print_panel,
    print_syntax,
    print_table,
    print_tree,
    status,
)

# Import emojis
from .emojis import (
    COMPLETED,
    CRITICAL,
    DEBUG,
    ERROR,
    FAILED,
    INFO,
    RUNNING,
    SUCCESS,
    WARNING,
    get_emoji,
)

# Import formatters and handlers
from .formatter import (
    DetailedLogFormatter,
    GatewayLogRecord,
    RichLoggingHandler,
    SimpleLogFormatter,
    create_rich_console_handler,  # Added missing import used in server.py LOGGING_CONFIG
)

# Import logger and related utilities
from .logger import (
    Logger,
    critical,
    debug,
    error,
    info,
    section,
    success,
    warning,
)

# Import panels
from .panels import (
    CodePanel,
    ErrorPanel,
    HeaderPanel,
    InfoPanel,
    ResultPanel,
    ToolOutputPanel,
    WarningPanel,
    display_code,
    display_error,
    display_header,
    display_info,
    display_results,
    display_tool_output,
    display_warning,
)

# Import progress tracking
from .progress import (
    GatewayProgress,
    track,
)

# Create a global logger instance for importing
logger = Logger("ultimate")

# Removed configure_root_logger, initialize_logging, set_log_level functions
# Logging is now configured via dictConfig in main.py (or server.py equivalent)


def get_logger(name: str) -> Logger:
    """
    Get or create a specialized Logger instance for a specific component.

    This function provides access to the enhanced logging system of the Ultimate MCP Server,
    returning a Logger instance that includes rich formatting, emoji support, and other
    advanced features beyond Python's standard logging.

    The returned Logger is configured with the project's logging settings and integrates
    with the rich console output system. It provides methods like success() and section()
    in addition to standard logging methods.

    Args:
        name: The logger name, typically the module or component name.
             Can use dot notation for hierarchy (e.g., "module.submodule").

    Returns:
        An enhanced Logger instance with rich formatting and emoji support

    Example:
        ```python
        # In a module file
        from ultimate_mcp_server.utils.logging import get_logger

        # Create logger with the module name
        logger = get_logger(__name__)

        # Use the enhanced logging methods
        logger.info("Server starting")                  # Basic info log
        logger.success("Operation completed")           # Success log (not in std logging)
        logger.warning("Resource low", resource="RAM")  # With additional context
        logger.error("Failed to connect", emoji_key="network")  # With custom emoji
        ```
    """
    # Use the new base name for sub-loggers if needed, or keep original logic
    # return Logger(f"ultimate_mcp_server.{name}") # Option 1: Prefix with base name
    return Logger(name)  # Option 2: Keep original name logic


def capture_logs(level: Optional[str] = None) -> "LogCapture":
    """
    Create a context manager to capture logs for testing or debugging.

    This function is a convenience wrapper around the LogCapture class, creating
    and returning a context manager that will capture logs at or above the specified
    level during its active scope.

    Use this function when you need to verify that certain log messages are emitted
    during tests, or when you want to collect logs for analysis without modifying
    the application's logging configuration.

    Args:
        level: Minimum log level to capture (e.g., "INFO", "WARNING", "ERROR").
               If None, all log levels are captured. Default: None

    Returns:
        A LogCapture context manager that will collect logs when active

    Example:
        ```python
        # Test that a function produces expected log messages
        def test_login_function():
            with capture_logs("WARNING") as logs:
                # Call function that should produce a warning log for invalid login
                result = login("invalid_user", "wrong_password")

                # Assert that the expected warning was logged
                assert logs.contains("Invalid login attempt")
                assert len(logs.get_logs()) == 1
        ```
    """
    return LogCapture(level)


# Log capturing for testing
class LogCapture:
    """
    Context manager for capturing and analyzing logs during execution.

    This class provides a way to intercept, store, and analyze logs emitted during
    a specific block of code execution. It's primarily useful for:

    - Testing: Verify that specific log messages were emitted during tests
    - Debugging: Collect logs for examination without changing logging configuration
    - Analysis: Gather statistics about logging patterns

    The LogCapture acts as a context manager, capturing logs only within its scope
    and providing methods to retrieve and analyze the captured logs after execution.

    Each captured log entry is stored as a dictionary with details including the
    message, level, timestamp, and source file/line information.

    Example usage:
        ```python
        # Capture all logs
        with LogCapture() as capture:
            # Code that generates logs
            perform_operation()

            # Check for specific log messages
            assert capture.contains("Database connected")
            assert not capture.contains("Error")

            # Get all captured logs
            all_logs = capture.get_logs()

            # Get only warning and error messages
            warnings = capture.get_logs(level="WARNING")
        ```
    """

    def __init__(self, level: Optional[str] = None):
        """Initialize the log capture.

        Args:
            level: Minimum log level to capture
        """
        self.level = level
        self.level_num = getattr(logging, self.level.upper(), 0) if self.level else 0
        self.logs: List[Dict[str, Any]] = []
        self.handler = self._create_handler()

    def _create_handler(self) -> logging.Handler:
        """Create a handler to capture logs.

        Returns:
            Log handler
        """

        class CaptureHandler(logging.Handler):
            def __init__(self, capture):
                super().__init__()
                self.capture = capture

            def emit(self, record):
                # Skip if record level is lower than minimum
                if record.levelno < self.capture.level_num:
                    return

                # Add log record to captured logs
                self.capture.logs.append(
                    {
                        "level": record.levelname,
                        "message": record.getMessage(),
                        "name": record.name,
                        "time": record.created,
                        "file": record.pathname,
                        "line": record.lineno,
                    }
                )

        return CaptureHandler(self)

    def __enter__(self) -> "LogCapture":
        """Enter the context manager.

        Returns:
            Self
        """
        # Add handler to root logger
        # Use the project's logger name
        logging.getLogger("ultimate").addHandler(self.handler)
        # Consider adding to the absolute root logger as well if needed
        # logging.getLogger().addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager.

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        # Remove handler from root logger
        logging.getLogger("ultimate").removeHandler(self.handler)
        # logging.getLogger().removeHandler(self.handler)

    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get captured logs, optionally filtered by level.

        Args:
            level: Filter logs by level

        Returns:
            List of log records
        """
        if not level:
            return self.logs

        level_num = getattr(logging, level.upper(), 0)
        return [log for log in self.logs if getattr(logging, log["level"], 0) >= level_num]

    def get_messages(self, level: Optional[str] = None) -> List[str]:
        """Get captured log messages, optionally filtered by level.

        Args:
            level: Filter logs by level

        Returns:
            List of log messages
        """
        return [log["message"] for log in self.get_logs(level)]

    def contains(self, text: str, level: Optional[str] = None) -> bool:
        """Check if any log message contains the given text.

        Args:
            text: Text to search for
            level: Optional level filter

        Returns:
            True if text is found in any message
        """
        return any(text in msg for msg in self.get_messages(level))


__all__ = [
    # Console
    "console",
    "create_progress",
    "status",
    "print_panel",
    "print_syntax",
    "print_table",
    "print_tree",
    "print_json",
    "live_display",
    # Logger and utilities
    "logger",
    "Logger",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
    "section",
    "get_logger",
    "capture_logs",
    "LogCapture",
    # Emojis
    "get_emoji",
    "INFO",
    "DEBUG",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "SUCCESS",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    # Panels
    "HeaderPanel",
    "ResultPanel",
    "InfoPanel",
    "WarningPanel",
    "ErrorPanel",
    "ToolOutputPanel",
    "CodePanel",
    "display_header",
    "display_results",
    "display_info",
    "display_warning",
    "display_error",
    "display_tool_output",
    "display_code",
    # Progress tracking
    "GatewayProgress",
    "track",
    # Formatters and handlers
    "GatewayLogRecord",
    "SimpleLogFormatter",
    "DetailedLogFormatter",
    "RichLoggingHandler",
    "create_rich_console_handler",
]
