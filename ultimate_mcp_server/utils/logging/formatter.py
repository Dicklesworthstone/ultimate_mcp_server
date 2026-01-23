"""
Log formatters for Gateway logging system.

This module provides formatters that convert log records into Rich renderables
with consistent styling and visual elements.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from rich.console import Console, ConsoleRenderable, Group
from rich.logging import RichHandler
from rich.style import Style
from rich.text import Text
from rich.traceback import Traceback

from .console import get_rich_console  # Import the console factory

# Use relative imports for utils within the same package
from .emojis import LEVEL_EMOJIS, get_emoji
from .themes import get_component_style, get_level_style


class GatewayLogRecord:
    """Enhanced log record simulation using standard LogRecord attributes.

    This class is mostly for documentation and conceptual clarity.
    The actual data comes from the standard logging.LogRecord,
    populated via the 'extra' dictionary in the Logger._log method.
    """

    def __init__(self, record: logging.LogRecord):
        """Initialize from a standard logging.LogRecord."""
        self.record = record

    @property
    def level(self) -> str:
        """Get the original Gateway log level name (e.g., 'success')."""
        return getattr(self.record, "gateway_level", self.record.levelname.lower())

    @property
    def message(self) -> str:
        """Get the log message."""
        return self.record.getMessage()

    @property
    def component(self) -> Optional[str]:
        """Get the Gateway component."""
        comp = getattr(self.record, "component", None)
        return comp.lower() if comp else None

    @property
    def operation(self) -> Optional[str]:
        """Get the Gateway operation."""
        op = getattr(self.record, "operation", None)
        return op.lower() if op else None

    @property
    def custom_emoji(self) -> Optional[str]:
        """Get the custom emoji override."""
        return getattr(self.record, "custom_emoji", None)

    @property
    def context(self) -> Optional[Dict[str, Any]]:
        """Get the additional context data."""
        return getattr(self.record, "log_context", None)

    @property
    def timestamp(self) -> float:
        """Get the log record creation time."""
        return self.record.created

    @property
    def exception_info(self) -> Optional[Tuple]:
        """Get the exception info tuple."""
        return self.record.exc_info

    @property
    def emoji(self) -> str:
        """Get the appropriate emoji for this log record."""
        if self.custom_emoji:
            return self.custom_emoji

        # Use operation emoji if available
        if self.operation:
            operation_emoji = get_emoji("operation", self.operation)
            if operation_emoji != "❓":  # If not unknown
                return operation_emoji

        # Fall back to level emoji (use gateway_level if available)
        return LEVEL_EMOJIS.get(self.level, "❓")

    @property
    def style(self) -> Style:
        """Get the appropriate style for this log record."""
        return get_level_style(self.level)

    @property
    def component_style(self) -> Style:
        """Get the style for this record's component."""
        if not self.component:
            return self.style
        return get_component_style(self.component)

    @property
    def format_time(self) -> str:
        """Format the timestamp for display."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S.%f")[:-3]  # Trim microseconds to milliseconds

    def has_exception(self) -> bool:
        """Check if this record contains exception information."""
        return self.record.exc_info is not None


class GatewayLogFormatter(logging.Formatter):
    """Base formatter for Gateway logs that converts to Rich renderables.
    Adapts standard Formatter for Rich output.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        show_time: bool = True,
        show_level: bool = True,
        show_component: bool = True,
        show_path: bool = False,
        **kwargs,
    ):
        """Initialize the formatter.

        Args:
            fmt: Format string (standard logging format)
            datefmt: Date format string
            style: Formatting style ('%', '{', '$')
            show_time: Whether to show timestamp in Rich output
            show_level: Whether to show log level in Rich output
            show_component: Whether to show component in Rich output
            show_path: Whether to show path/lineno in Rich output
            **kwargs: Additional args for base Formatter
        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, **kwargs)
        self.show_time = show_time
        self.show_level = show_level
        self.show_component = show_component
        self.show_path = show_path

    def format(self, record: logging.LogRecord) -> str:
        """Format the record into a string (for non-Rich handlers)."""
        # Use default formatting for file/non-rich output
        # Add custom fields to the record temporarily if needed
        record.gateway_component = getattr(record, "component", "")
        record.gateway_operation = getattr(record, "operation", "")
        # Use the standard Formatter implementation
        return super().format(record)

    def format_rich(self, record: logging.LogRecord) -> ConsoleRenderable:
        """Format a standard logging.LogRecord into a Rich renderable.

        Args:
            record: The log record to format

        Returns:
            A Rich renderable object
        """
        # Subclasses should implement this
        raise NotImplementedError("Subclasses must implement format_rich")


class SimpleLogFormatter(GatewayLogFormatter):
    """Simple single-line log formatter for Rich console output."""

    def format_rich(self, record: logging.LogRecord) -> Text:
        """Format a record as a single line of rich text.

        Args:
            record: The log record to format

        Returns:
            Formatted Text object
        """
        gateway_record = GatewayLogRecord(record)  # Wrap for easier access
        result = Text()

        # Add timestamp if requested
        if self.show_time:
            result.append(f"[{gateway_record.format_time}] ", style="timestamp")

        # Add emoji
        result.append(f"{gateway_record.emoji} ", style=gateway_record.style)

        # Add level if requested
        if self.show_level:
            level_text = f"[{gateway_record.level.upper()}] "
            result.append(level_text, style=gateway_record.style)

        # Add component if available and requested
        if self.show_component and gateway_record.component:
            component_text = f"[{gateway_record.component}] "
            result.append(component_text, style=gateway_record.component_style)

        # Add operation if available
        if gateway_record.operation:
            operation_text = f"{gateway_record.operation}: "
            result.append(operation_text, style="operation")

        # Add message
        result.append(gateway_record.message)

        # Add path/line number if requested
        if self.show_path:
            path_text = f" ({record.pathname}:{record.lineno})"
            result.append(path_text, style="dim")

        # Add Exception/Traceback if present (handled by RichHandler.render)

        return result


class DetailedLogFormatter(GatewayLogFormatter):
    """Multi-line formatter that can include context data (Placeholder)."""

    def format_rich(self, record: logging.LogRecord) -> ConsoleRenderable:
        """Format a record with potentially detailed information.

        Args:
            record: The log record to format

        Returns:
            Formatted Panel or Text object
        """
        # Fallback to simple formatting for now
        formatter = SimpleLogFormatter(
            show_time=self.show_time,
            show_level=self.show_level,
            show_component=self.show_component,
            show_path=self.show_path,
        )
        return formatter.format_rich(record)


class RichLoggingHandler(RichHandler):
    """Custom RichHandler that uses GatewayLogFormatter.

    Overrides render to use the custom formatter.
    """

    def __init__(
        self,
        level: int = logging.NOTSET,
        console: Optional[Console] = None,
        formatter: Optional[GatewayLogFormatter] = None,
        show_path: bool = False,  # Control path display via handler
        **kwargs,
    ):
        """Initialize the Rich handler.

        Args:
            level: Log level for this handler
            console: Rich console instance (uses global if None)
            formatter: Custom Gateway formatter (creates default if None)
            show_path: Whether to show path/lineno in the logs
            **kwargs: Additional args for RichHandler
        """
        # Use the provided console or the default from console.py
        effective_console = console or get_rich_console()

        super().__init__(level=level, console=effective_console, **kwargs)

        # Create a default SimpleLogFormatter if none is provided
        self.formatter = formatter or SimpleLogFormatter(show_path=show_path)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using Rich formatting."""
        try:
            # Let the custom formatter create the Rich renderable
            message_renderable = self.format_rich(record)

            # Get the traceback if there is one
            traceback_renderable = None
            if record.exc_info:
                traceback_renderable = Traceback.from_exception(
                    *record.exc_info,
                    width=self.console.width if self.console else None,  # Check if console exists
                    extra_lines=self.tracebacks_extra_lines,
                    theme=self.tracebacks_theme,
                    word_wrap=self.tracebacks_word_wrap,
                    show_locals=self.tracebacks_show_locals,
                    locals_max_length=self.locals_max_length,
                    locals_max_string=self.locals_max_string,
                    suppress=self.tracebacks_suppress,
                )

            # Use the render method to combine message and traceback
            renderable = self.render(
                record=record,
                traceback=traceback_renderable,  # Pass the Traceback instance
                message_renderable=message_renderable,
            )
            if self.console:
                self.console.print(renderable)
        except Exception:
            self.handleError(record)

    def format_rich(self, record: logging.LogRecord) -> ConsoleRenderable:
        """Format the record using the assigned GatewayLogFormatter."""
        # Ensure formatter is of the correct type before calling format_rich
        if isinstance(self.formatter, GatewayLogFormatter):
            # Indentation corrected: 4 spaces
            return self.formatter.format_rich(record)
        elif isinstance(self.formatter, logging.Formatter):
            # Indentation corrected: 4 spaces
            # Fallback for standard formatter (e.g., if assigned incorrectly)
            return Text(self.formatter.format(record))
        else:
            # Indentation corrected: 4 spaces
            # Fallback if formatter is None or unexpected type
            return Text(record.getMessage())

    def render(
        self,
        *,  # Make args keyword-only
        record: logging.LogRecord,
        traceback: Optional[Traceback],
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        """Renders log message and Traceback.
        Overridden to ensure our formatted message_renderable is used correctly.

        Args:
            record: logging Record.
            traceback: Traceback instance or None for no Traceback.
            message_renderable: Renderable representing log message.

        Returns:
            Renderable to be written to console.
        """
        # message_renderable is already formatted by format_rich
        # We just need to potentially append the traceback
        if traceback:
            # If the message is simple Text, append newline and traceback
            if isinstance(message_renderable, Text):
                # Check if message already ends with newline for cleaner separation
                if not str(message_renderable).endswith("\n"):
                    message_renderable = Text.assemble(
                        message_renderable, "\n"
                    )  # Use assemble for safety
                return Group(message_renderable, traceback)
            else:
                # For Panels or other renderables, group them
                return Group(message_renderable, traceback)
        else:
            return message_renderable


def create_rich_console_handler(**kwargs):
    """Factory function to create a RichLoggingHandler.
    Used in dictConfig.

    Args:
        **kwargs: Arguments passed from dictConfig, forwarded to RichLoggingHandler.
                  Includes level, formatter (if specified), show_path, etc.

    Returns:
        Instance of RichLoggingHandler.
    """
    # Ensure console is not passed directly if we want the shared one
    kwargs.pop("console", None)

    # Extract formatter config if provided (though unlikely needed with custom handler)
    # Pop it from kwargs to prevent it being passed to the handler
    kwargs.pop("formatter", None)
    # We expect the handler config to specify the formatter directly or rely on default

    # Extract level, default to NOTSET if not provided
    level_name = kwargs.pop("level", "NOTSET").upper()
    level = logging.getLevelName(level_name)

    # Extract show_path flag
    show_path = kwargs.pop("show_path", False)

    # Create the handler instance
    # Pass relevant args like show_path
    # Also pass RichHandler specific args if they exist in kwargs
    rich_handler_args = {
        k: v
        for k, v in kwargs.items()
        if k
        in (
            "show_time",
            "show_level",
            "markup",
            "rich_tracebacks",
            "tracebacks_width",
            "tracebacks_extra_lines",
            "tracebacks_theme",
            "tracebacks_word_wrap",
            "tracebacks_show_locals",
            "locals_max_length",
            "locals_max_string",
            "tracebacks_suppress",
        )
    }
    # Add show_path explicitly as it's specific to our handler/formatter logic here
    handler = RichLoggingHandler(level=level, show_path=show_path, **rich_handler_args)

    # Note: Setting a specific formatter via dictConfig for this custom handler
    # might require more complex logic here to instantiate the correct GatewayLogFormatter.
    # For now, it defaults to SimpleLogFormatter controlled by show_path.

    return handler
