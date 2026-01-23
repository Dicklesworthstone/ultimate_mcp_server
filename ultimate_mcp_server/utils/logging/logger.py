"""
Main Logger class for Gateway.

This module provides the central Logger class that integrates all Gateway logging
functionality with a beautiful, informative interface.
"""

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console

# Use relative imports for utils within the same package
from .console import console
from .emojis import get_emoji
from .formatter import (
    DetailedLogFormatter,
    RichLoggingHandler,
    SimpleLogFormatter,
)
from .panels import (
    CodePanel,
    ErrorPanel,
    HeaderPanel,
    InfoPanel,
    ResultPanel,
    ToolOutputPanel,
    WarningPanel,
)
from .progress import GatewayProgress

# Set up standard Python logging with our custom handler
# Logging configuration is handled externally via dictConfig


class Logger:
    """
    Advanced logging system with rich formatting, progress tracking, and structured output.

    The Logger class extends Python's standard logging system with enhanced features:

    Key Features:
    - Rich console output with color, emoji, and formatted panels
    - Component-based logging for better organization of log messages
    - Operation tracking with timing and progress visualization
    - Multi-level logging (debug, info, success, warning, error, critical)
    - Context data capture for more detailed debugging
    - Integrated progress bars and spinners for long-running operations
    - Special formatters for code blocks, results, errors, and warnings

    Integration with Python's logging:
    - Builds on top of the standard logging module
    - Compatible with external logging configuration (e.g., dictConfig)
    - Properly propagates logs to ensure they reach root handlers
    - Adds custom "extra" fields to standard LogRecord objects

    Usage Patterns:
    - Create loggers with get_logger() for consistent naming
    - Use component and operation parameters to organize related logs
    - Add context data as structured information with message
    - Use special display methods (code, warning_panel, etc.) for rich output
    - Track long operations with time_operation and progress tracking

    This logger is designed to make complex server operations more transparent,
    providing clear information for both developers and users of the Ultimate MCP Server.
    """

    def __init__(
        self,
        name: str = "ultimate",  # Default logger name changed
        console: Optional[Console] = None,
        level: str = "info",
        show_timestamps: bool = True,
        component: Optional[str] = None,
        capture_output: bool = False,
    ):
        """Initialize the logger.

        Args:
            name: Logger name
            console: Rich console to use
            level: Initial log level
            show_timestamps: Whether to show timestamps in logs
            component: Default component name
            capture_output: Whether to capture and store log output
        """
        self.name = name
        # Use provided console or get global console, defaulting to stderr console
        if console is not None:
            self.console = console
        else:
            global_console = globals().get("console")
            if global_console is not None:
                self.console = global_console
            else:
                self.console = Console(file=sys.stderr)

        self.level = level.lower()
        self.show_timestamps = show_timestamps
        self.component = component
        self.capture_output = capture_output

        # Create a standard Python logger
        self.python_logger = logging.getLogger(name)

        # Set up formatters
        self.simple_formatter = SimpleLogFormatter(
            show_time=show_timestamps, show_level=True, show_component=True
        )
        self.detailed_formatter = DetailedLogFormatter(
            show_time=show_timestamps, show_level=True, show_component=True
        )

        # Progress tracker
        self.progress = GatewayProgress(console=self.console)

        # Output capture if enabled
        self.captured_logs = [] if capture_output else None

        # Restore propagation to allow messages to reach root handlers
        # Make sure this is True so logs configured via dictConfig are passed up
        self.python_logger.propagate = True

        # Set initial log level on the Python logger instance
        # Note: The effective level will be determined by the handler/root config
        self.set_level(level)

    def set_level(self, level: str) -> None:
        """Set the log level.

        Args:
            level: Log level (debug, info, warning, error, critical)
        """
        level = level.lower()
        self.level = level  # Store the intended level for should_log checks

        # Map to Python logging levels
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        python_level = level_map.get(level, logging.INFO)
        # Set level on the logger itself. Handlers might have their own levels.
        self.python_logger.setLevel(python_level)

    def get_level(self) -> str:
        """Get the current log level.

        Returns:
            Current log level
        """
        # Return the Python logger's effective level
        effective_level_num = self.python_logger.getEffectiveLevel()
        level_map_rev = {
            logging.DEBUG: "debug",
            logging.INFO: "info",
            logging.WARNING: "warning",
            logging.ERROR: "error",
            logging.CRITICAL: "critical",
        }
        return level_map_rev.get(effective_level_num, "info")

    def should_log(self, level: str) -> bool:
        """Check if a message at the given level should be logged based on Python logger's effective level.

        Args:
            level: Log level to check

        Returns:
            Whether messages at this level should be logged
        """
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "success": logging.INFO,  # Map success to info for level check
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        message_level_num = level_map.get(level.lower(), logging.INFO)
        return self.python_logger.isEnabledFor(message_level_num)

    def _log(
        self,
        level: str,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        emoji: Optional[str] = None,
        emoji_key: Optional[str] = None,  # Add emoji_key parameter
        context: Optional[Dict[str, Any]] = None,
        use_detailed_formatter: bool = False,  # This arg seems unused now?
        exception_info: Optional[Union[bool, Tuple]] = None,
        stack_info: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Internal method to handle logging via the standard Python logging mechanism.

        Args:
            level: Log level
            message: Log message
            component: Gateway component (core, composite, analysis, etc.)
            operation: Operation being performed
            emoji: Custom emoji override
            emoji_key: Key to look up emoji from emoji map (alternative to emoji)
            context: Additional contextual data
            exception_info: Include exception info (True/False or tuple)
            stack_info: Include stack info
            extra: Dictionary passed as extra to logging framework
        """
        # Check if we should log at this level using standard Python logging check
        # No need for the custom should_log method here if using stdlib correctly

        # Map level name to Python level number
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "success": logging.INFO,  # Log success as INFO
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        level_num = level_map.get(level.lower(), logging.INFO)

        if not self.python_logger.isEnabledFor(level_num):
            return

        # Use default component if not provided
        component = component or self.component

        # If emoji_key is provided, use it to determine emoji
        if emoji_key and not emoji:
            emoji = get_emoji("operation", emoji_key)
            if emoji == "❓":  # If operation emoji not found
                # Try level emojis
                from .emojis import LEVEL_EMOJIS

                emoji = LEVEL_EMOJIS.get(emoji_key, "❓")

        # Prepare 'extra' dict for LogRecord
        log_extra = (
            {} if extra is None else extra.copy()
        )  # Create a copy to avoid modifying the original

        # Remove any keys that conflict with built-in LogRecord attributes
        for reserved_key in [
            "message",
            "asctime",
            "exc_info",
            "exc_text",
            "lineno",
            "funcName",
            "created",
            "levelname",
            "levelno",
        ]:
            if reserved_key in log_extra:
                del log_extra[reserved_key]

        # Add our custom keys
        log_extra["component"] = component
        log_extra["operation"] = operation
        log_extra["custom_emoji"] = emoji
        log_extra["log_context"] = context  # Use a different key to avoid collision
        log_extra["gateway_level"] = level  # Pass the original level name if needed by formatter

        # Handle exception info
        exc_info = None
        if exception_info:
            if isinstance(exception_info, bool):
                exc_info = sys.exc_info()
            else:
                exc_info = exception_info  # Assume it's a valid tuple

        # Log through Python's logging system
        self.python_logger.log(
            level=level_num, msg=message, exc_info=exc_info, stack_info=stack_info, extra=log_extra
        )

        # Capture if enabled
        if self.captured_logs is not None:
            self.captured_logs.append(
                {
                    "level": level,
                    "message": message,
                    "component": component,
                    "operation": operation,
                    "timestamp": datetime.now().isoformat(),
                    "context": context,
                }
            )

    # --- Standard Logging Methods ---

    def debug(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        emoji_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a debug message."""
        self._log(
            "debug",
            message,
            component,
            operation,
            context=context,
            emoji_key=emoji_key,
            extra=kwargs,
        )

    def info(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        emoji_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log an info message."""
        self._log(
            "info",
            message,
            component,
            operation,
            context=context,
            emoji_key=emoji_key,
            extra=kwargs,
        )

    def success(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        emoji_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a success message."""
        self._log(
            "success",
            message,
            component,
            operation,
            context=context,
            emoji_key=emoji_key,
            extra=kwargs,
        )

    def warning(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        emoji_key: Optional[str] = None,
        # details: Optional[List[str]] = None, # Details handled by panel methods
        **kwargs,
    ) -> None:
        """Log a warning message."""
        self._log(
            "warning",
            message,
            component,
            operation,
            context=context,
            emoji_key=emoji_key,
            extra=kwargs,
        )

    def error(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        emoji_key: Optional[str] = None,
        # error_code: Optional[str] = None,
        # resolution_steps: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Log an error message."""
        # Get the exception info tuple if an exception was provided
        exc_info = None
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)
        elif "exc_info" in kwargs:
            exc_info = kwargs.pop("exc_info")  # Remove from kwargs to prevent conflicts

        self._log(
            "error",
            message,
            component,
            operation,
            context=context,
            exception_info=exc_info,
            emoji_key=emoji_key,
            extra=kwargs,
        )

    def critical(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        emoji_key: Optional[str] = None,
        # error_code: Optional[str] = None, # Pass via context or kwargs
        **kwargs,
    ) -> None:
        """Log a critical message."""
        # Get the exception info tuple if an exception was provided
        exc_info = None
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)
        elif "exc_info" in kwargs:
            exc_info = kwargs.pop("exc_info")  # Remove from kwargs to prevent conflicts

        self._log(
            "critical",
            message,
            component,
            operation,
            context=context,
            exception_info=exc_info,
            emoji_key=emoji_key,
            extra=kwargs,
        )

    # --- Rich Display Methods ---
    # These methods use the console directly or generate renderables
    # They might bypass the standard logging flow, or log additionally

    def operation(
        self,
        operation: str,
        message: str,
        component: Optional[str] = None,
        level: str = "info",
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log an operation-specific message.

        Args:
            operation: Operation name
            message: Log message
            component: Gateway component
            level: Log level (default: info)
            context: Additional context
            **kwargs: Extra fields for logging
        """
        self._log(level, message, component, operation, context=context, extra=kwargs)

    def tool(
        self,
        tool: str,
        command: str,
        output: str,
        status: str = "success",
        duration: Optional[float] = None,
        component: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Display formatted output from a tool.

        Args:
            tool: Name of the tool
            command: Command executed
            output: Tool output
            status: Execution status (success, error)
            duration: Execution duration in seconds
            component: Gateway component
            **kwargs: Extra fields for logging
        """
        # Optionally log the event
        log_level = "error" if status == "error" else "debug"
        log_message = f"Tool '{tool}' finished (status: {status})"
        log_context = {
            "command": command,
            "output_preview": output[:100] + "..." if len(output) > 100 else output,
        }
        if duration is not None:
            log_context["duration_s"] = duration
        self._log(
            log_level,
            log_message,
            component,
            operation=f"tool.{tool}",
            context=log_context,
            extra=kwargs,
        )

        # Display the panel directly on the console
        panel = ToolOutputPanel(tool, command, output, status, duration)
        self.console.print(panel)

    def code(
        self,
        code: str,
        language: str = "python",
        title: Optional[str] = None,
        line_numbers: bool = True,
        highlight_lines: Optional[List[int]] = None,
        message: Optional[str] = None,
        component: Optional[str] = None,
        level: str = "debug",
        **kwargs,
    ) -> None:
        """Display a code block.

        Args:
            code: Code string
            language: Language for syntax highlighting
            title: Optional title for the panel
            line_numbers: Show line numbers
            highlight_lines: Lines to highlight
            message: Optional message to log alongside displaying the code
            component: Gateway component
            level: Log level for the optional message (default: debug)
            **kwargs: Extra fields for logging
        """
        if message:
            self._log(
                level,
                message,
                component,
                context={"code_preview": code[:100] + "..." if len(code) > 100 else code},
                extra=kwargs,
            )

        # Display the panel directly
        panel = CodePanel(code, language, title, line_numbers, highlight_lines)
        self.console.print(panel)

    def display_results(
        self,
        title: str,
        results: Union[List[Dict[str, Any]], Dict[str, Any]],
        status: str = "success",
        component: Optional[str] = None,
        show_count: bool = True,
        compact: bool = False,
        message: Optional[str] = None,
        level: str = "info",
        **kwargs,
    ) -> None:
        """Display results in a formatted panel.

        Args:
            title: Panel title
            results: Results data
            status: Status (success, warning, error)
            component: Gateway component
            show_count: Show count in title
            compact: Use compact format
            message: Optional message to log
            level: Log level for the optional message (default: info)
            **kwargs: Extra fields for logging
        """
        if message:
            self._log(
                level,
                message,
                component,
                context={
                    "result_count": len(results) if isinstance(results, list) else 1,
                    "status": status,
                },
                extra=kwargs,
            )

        # Display the panel directly
        panel = ResultPanel(title, results, status, component, show_count, compact)
        self.console.print(panel)

    def section(
        self,
        title: str,
        subtitle: Optional[str] = None,
        component: Optional[str] = None,
    ) -> None:
        """Display a section header.

        Args:
            title: Section title
            subtitle: Optional subtitle
            component: Gateway component
        """
        # This is purely presentational, doesn't log typically
        panel = HeaderPanel(title, subtitle, component=component)
        self.console.print(panel)

    def info_panel(
        self,
        title: str,
        content: Union[str, List[str], Dict[str, Any]],
        icon: Optional[str] = None,
        style: str = "info",
        component: Optional[str] = None,
    ) -> None:
        """Display an informational panel.

        Args:
            title: Panel title
            content: Panel content
            icon: Optional icon
            style: Panel style
            component: Gateway component
        """
        # Could log the title/content summary if desired
        # self._log("info", f"Displaying info panel: {title}", component)
        panel = InfoPanel(title, content, icon, style)
        self.console.print(panel)

    def warning_panel(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[List[str]] = None,
        component: Optional[str] = None,
    ) -> None:
        """Display a warning panel.

        Args:
            title: Optional panel title
            message: Warning message
            details: Optional list of detail strings
            component: Gateway component
        """
        # Log the warning separately
        log_title = title if title else "Warning"
        self.warning(f"{log_title}: {message}", component, context={"details": details})

        # Display the panel directly
        panel = WarningPanel(title, message, details)
        self.console.print(panel)

    def error_panel(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[str] = None,
        resolution_steps: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        component: Optional[str] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """Display an error panel.

        Args:
            title: Optional panel title
            message: Error message
            details: Optional detail string (e.g., traceback)
            resolution_steps: Optional list of resolution steps
            error_code: Optional error code
            component: Gateway component
            exception: Associated exception (for logging traceback)
        """
        # Log the error separately
        log_title = title if title else "Error"
        log_context = {
            "details": details,
            "resolution": resolution_steps,
            "error_code": error_code,
        }
        self.error(f"{log_title}: {message}", component, context=log_context, exception=exception)

        # Display the panel directly
        panel = ErrorPanel(title, message, details, resolution_steps, error_code)
        self.console.print(panel)

    # --- Context Managers & Decorators ---

    @contextmanager
    def time_operation(
        self,
        operation: str,
        component: Optional[str] = None,
        level: str = "info",
        start_message: Optional[str] = "Starting {operation}...",
        end_message: Optional[str] = "Finished {operation} in {duration:.2f}s",
        **kwargs,
    ):
        """
        Context manager that times an operation and logs its start and completion.

        This method provides a clean, standardized way to track and log the duration
        of operations, ensuring consistent timing measurement and log formatting.
        It automatically logs the start of an operation, executes the operation
        within the context, measures the exact duration, and logs the completion
        with timing information.

        The timing uses Python's monotonic clock for accurate duration measurement
        even if system time changes during execution. Both start and end messages
        support templating with format string syntax, allowing customization while
        maintaining consistency.

        Key features:
        - Precise operation timing with monotonic clock
        - Automatic logging at start and end of operations
        - Customizable message templates
        - Consistent log format and metadata
        - Exception-safe timing (duration is logged even if operation fails)
        - Hierarchical operation tracking when combined with component parameter

        Usage Examples:
        ```python
        # Basic usage
        with logger.time_operation("data_processing"):
            process_large_dataset()

        # Custom messages and different log level
        with logger.time_operation(
            operation="database_backup",
            component="storage",
            level="debug",
            start_message="Starting backup of {operation}...",
            end_message="Backup of {operation} completed in {duration:.3f}s"
        ):
            backup_database()

        # Timing nested operations with different components
        with logger.time_operation("parent_task", component="scheduler"):
            do_first_part()
            with logger.time_operation("child_task", component="worker"):
                do_second_part()
            finish_task()
        ```

        Args:
            operation: Name of the operation being timed
            component: Component performing the operation (uses logger default if None)
            level: Log level for start/end messages (default: "info")
            start_message: Template string for operation start message
                          (None to skip start logging)
            end_message: Template string for operation end message
                        (None to skip end logging)
            **kwargs: Additional fields to include in log entries

        Yields:
            None

        Note:
            This context manager is exception-safe: the end message with duration
            is logged even if an exception occurs within the context. Exceptions
            are re-raised normally after logging.
        """
        start_time = time.monotonic()
        if start_message:
            self._log(
                level, start_message.format(operation=operation), component, operation, extra=kwargs
            )

        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            if end_message:
                self._log(
                    level,
                    end_message.format(operation=operation, duration=duration),
                    component,
                    operation,
                    context={"duration_s": duration},
                    extra=kwargs,
                )

    def track(
        self,
        iterable: Any,
        description: str,
        name: Optional[str] = None,
        total: Optional[int] = None,
        parent: Optional[str] = None,
        # Removed component - handled by logger instance
    ) -> Any:
        """Track progress over an iterable using the logger's progress tracker.

        Args:
            iterable: Iterable to track
            description: Description of the task
            name: Optional task name (defaults to description)
            total: Optional total number of items
            parent: Optional parent task name

        Returns:
            The iterable wrapped with progress tracking
        """
        return self.progress.track(iterable, description, name, total, parent)

    @contextmanager
    def task(
        self,
        description: str,
        name: Optional[str] = None,
        total: int = 100,
        parent: Optional[str] = None,
        # Removed component - handled by logger instance
        autostart: bool = True,
    ):
        """
        Context manager for tracking and displaying progress of a task.

        This method creates a rich progress display for long-running tasks, providing
        visual feedback and real-time status updates. It integrates with rich's
        progress tracking to show animated spinners, completion percentage, and
        elapsed/remaining time.

        The task progress tracker is particularly useful for operations like:
        - File processing (uploads, downloads, parsing)
        - Batch database operations
        - Multi-step data processing pipelines
        - API calls with multiple sequential requests
        - Any operation where progress feedback improves user experience

        The progress display automatically adapts to terminal width and supports
        nested tasks with parent-child relationships, allowing for complex operation
        visualization. Progress can be updated manually within the context.

        Key Features:
        - Real-time progress visualization with percentage completion
        - Automatic elapsed and remaining time estimation
        - Support for nested tasks and task hierarchies
        - Customizable description and task identification
        - Thread-safe progress updates
        - Automatic completion on context exit

        Usage Examples:
        ```python
        # Basic usage - process 50 items
        with logger.task("Processing files", total=50) as task:
            for i, file in enumerate(files):
                process_file(file)
                task.update(advance=1)  # Increment progress by 1

        # Nested tasks with parent-child relationship
        with logger.task("Main import", total=100) as main_task:
            # Process users (contributes 30% to main task)
            with logger.task("Importing users", total=len(users), parent=main_task.id) as subtask:
                for user in users:
                    import_user(user)
                    subtask.update(advance=1)
                main_task.update(advance=30)  # Users complete = 30% of main task

            # Process products (contributes 70% to main task)
            with logger.task("Importing products", total=len(products), parent=main_task.id) as subtask:
                for product in products:
                    import_product(product)
                    subtask.update(advance=1)
                main_task.update(advance=70)  # Products complete = 70% of main task
        ```

        Args:
            description: Human-readable description of the task
            name: Unique identifier for the task (defaults to description if None)
            total: Total number of steps/work units for completion (100%)
            parent: ID of parent task (for nested task hierarchies)
            autostart: Automatically start displaying progress (default: True)

        Yields:
            GatewayProgress instance that can be used to update progress

        Notes:
            - The yielded progress object has methods like `update(advance=N)` to
              increment progress and `update(total=N)` to adjust the total units.
            - Tasks are automatically completed when the context exits, even if
              an exception occurs.
            - For tasks without a clear number of steps, you can use update with
              a percentage value: `task.update(completed=50)` for 50% complete.
        """
        with self.progress.task(description, name, total, parent, autostart) as task_context:
            yield task_context

    @contextmanager
    def catch_and_log(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        reraise: bool = True,
        level: str = "error",
        message: str = "An error occurred during {operation}",
    ):
        """
        Context manager that catches, logs, and optionally re-raises exceptions.

        This utility provides structured exception handling with automatic logging,
        allowing code to maintain a consistent error handling pattern while ensuring
        all exceptions are properly logged with relevant context information. It's
        particularly useful for operations where you want to ensure errors are always
        recorded, even if they'll be handled or suppressed at a higher level.

        The context manager wraps a block of code and:
        1. Executes the code normally
        2. Catches any exceptions that occur
        3. Logs the exception with configurable component, operation, and message
        4. Optionally re-raises the exception (controlled by the reraise parameter)

        This prevents "silent failures" and ensures consistent logging of all errors
        while preserving the original exception's traceback for debugging purposes.

        Key features:
        - Standardized error logging across the application
        - Configurable log level for different error severities
        - Component and operation tagging for error categorization
        - Template-based error messages with operation name substitution
        - Control over exception propagation behavior

        Usage Examples:
        ```python
        # Basic usage - catch, log, and re-raise
        with logger.catch_and_log(component="auth", operation="login"):
            user = authenticate_user(username, password)

        # Suppress exception after logging
        with logger.catch_and_log(
            component="email",
            operation="send_notification",
            reraise=False,
            level="warning",
            message="Failed to send notification email for {operation}"
        ):
            send_email(user.email, "Welcome!", template="welcome")

        # Use as a safety net around cleanup code
        try:
            # Main operation
            process_file(file_path)
        finally:
            # Always log errors in cleanup but don't let them mask the main exception
            with logger.catch_and_log(reraise=False, level="warning"):
                os.remove(temp_file)
        ```

        Args:
            component: Component name for error categorization (uses logger default if None)
            operation: Operation name for context (substituted in message template)
            reraise: Whether to re-raise the caught exception (default: True)
            level: Log level to use for the error message (default: "error")
            message: Template string for the error message, with {operation} placeholder

        Yields:
            None

        Note:
            When reraise=False, exceptions are completely suppressed after logging.
            This can be useful for non-critical operations like cleanup tasks,
            but should be used carefully to avoid hiding important errors.
        """
        component = component or self.component
        operation = operation or "operation"
        try:
            yield
        except Exception:
            log_msg = message.format(operation=operation)
            self._log(level, log_msg, component, operation, exception_info=True)
            if reraise:
                raise

    def log_call(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        level: str = "debug",
        log_args: bool = True,
        log_result: bool = False,
        log_exceptions: bool = True,
    ):
        """
        Decorator that logs function entries, exits, timing, and exceptions.

        This decorator provides automatic instrumentation for function calls,
        generating standardized log entries when functions are called and when they
        complete. It tracks execution time, captures function arguments and results,
        and properly handles and logs exceptions.

        When applied to a function, it will:
        1. Log when the function is entered, optionally including arguments
        2. Execute the function normally
        3. Track the exact execution time using a monotonic clock
        4. Log function completion with duration, optionally including the return value
        5. Catch, log, and re-raise any exceptions that occur

        This is particularly valuable for:
        - Debugging complex call flows and function interaction
        - Performance analysis and identifying slow function calls
        - Audit trails of function execution and parameters
        - Troubleshooting intermittent issues with full context
        - Standardizing logging across large codebases

        Configuration Options:
        - Logging level can be adjusted based on function importance
        - Function arguments can be optionally included or excluded (for privacy/size)
        - Return values can be optionally captured (for debugging/audit)
        - Exception handling can be customized
        - Component and operation names provide hierarchical organization

        Usage Examples:
        ```python
        # Basic usage - log entry and exit at debug level
        @logger.log_call()
        def process_data(item_id, options=None):
            # Function implementation...
            return result

        # Customized - log as info level, include specific operation name
        @logger.log_call(
            component="billing",
            operation="payment_processing",
            level="info"
        )
        def process_payment(payment_id, amount):
            # Process payment...
            return receipt_id

        # Capture return values but not arguments (e.g., for sensitive data)
        @logger.log_call(
            level="debug",
            log_args=False,
            log_result=True
        )
        def validate_credentials(username, password):
            # Validate credentials without logging the password
            return is_valid

        # Detailed debugging for critical components
        @logger.log_call(
            component="auth",
            operation="token_verification",
            level="debug",
            log_args=True,
            log_result=True,
            log_exceptions=True
        )
        def verify_auth_token(token):
            # Verify token with full logging
            return token_data
        ```

        Args:
            component: Component name for logs (defaults to logger's component)
            operation: Operation name for logs (defaults to function name)
            level: Log level for entry/exit messages (default: "debug")
            log_args: Whether to log function arguments (default: True)
            log_result: Whether to log function return value (default: False)
            log_exceptions: Whether to log exceptions (default: True)

        Returns:
            Decorated function that logs entry, exit, and timing information

        Notes:
            - For functions with large or sensitive arguments, set log_args=False
            - When log_result=True, be cautious with functions returning large data
              structures as they will be truncated but may still impact performance
            - This decorator preserves the original function's name, docstring,
              and signature for compatibility with introspection tools
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Determine operation name
                op_name = operation or func.__name__
                comp_name = component or self.component

                # Log entry
                entry_msg = f"Entering {op_name}..."
                context = {}
                if log_args:
                    # Be careful logging args, could contain sensitive info or be large
                    try:
                        arg_repr = f"args={args!r}, kwargs={kwargs!r}"
                        context["args"] = (
                            arg_repr[:200] + "..." if len(arg_repr) > 200 else arg_repr
                        )
                    except Exception:
                        context["args"] = "<Could not represent args>"

                self._log(level, entry_msg, comp_name, op_name, context=context)

                start_time = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                    duration = time.monotonic() - start_time

                    # Log exit
                    exit_msg = f"Exiting {op_name} (duration: {duration:.3f}s)"
                    exit_context = {"duration_s": duration}
                    if log_result:
                        try:
                            res_repr = repr(result)
                            exit_context["result"] = (
                                res_repr[:200] + "..." if len(res_repr) > 200 else res_repr
                            )
                        except Exception:
                            exit_context["result"] = "<Could not represent result>"

                    self._log(level, exit_msg, comp_name, op_name, context=exit_context)
                    return result

                except Exception as e:
                    duration = time.monotonic() - start_time
                    if log_exceptions:
                        exc_level = "error"  # Always log exceptions as error?
                        exc_msg = f"Exception in {op_name} after {duration:.3f}s: {e}"
                        exc_context = {"duration_s": duration}
                        if log_args:  # Include args context if available
                            exc_context.update(context)

                        self._log(
                            exc_level,
                            exc_msg,
                            comp_name,
                            op_name,
                            exception_info=True,
                            context=exc_context,
                        )
                    raise

            return wrapper

        return decorator

    # --- Startup/Shutdown Methods ---

    def startup(
        self,
        version: str,
        component: Optional[str] = None,
        mode: str = "standard",
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log server startup information.

        Args:
            version: Server version
            component: Component name (usually None for global startup)
            mode: Performance mode
            context: Additional startup context
            **kwargs: Extra fields for logging
        """
        message = f"Starting Server (Version: {version}, Mode: {mode})"
        emoji = get_emoji("system", "startup")
        self.info(message, component, operation="startup", emoji=emoji, context=context, **kwargs)

    def shutdown(
        self,
        component: Optional[str] = None,
        duration: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log server shutdown information.

        Args:
            component: Component name
            duration: Optional uptime duration
            context: Additional shutdown context
            **kwargs: Extra fields for logging
        """
        message = "Server Shutting Down"
        if duration is not None:
            message += f" (Uptime: {duration:.2f}s)"
        emoji = get_emoji("system", "shutdown")
        self.info(message, component, operation="shutdown", emoji=emoji, context=context, **kwargs)


# --- Global Convenience Functions ---
# These use the global 'logger' instance created in __init__.py

# At the global level, declare logger as None initially
logger = None


def get_logger(name: str) -> Logger:
    """
    Get or create a logger instance for a specific component or module.

    This function creates a properly named Logger instance following the application's
    logging hierarchy and naming conventions. It serves as the primary entry point
    for obtaining loggers throughout the application, ensuring consistent logger
    configuration and behavior.

    The function implements a pseudo-singleton pattern for the default logger:
    - The first call initializes a global default logger
    - Each subsequent call creates a new named logger instance
    - The name parameter establishes the logger's identity in the logging hierarchy

    Logger Naming Conventions:
    Logger names should follow Python's module path pattern, where dots separate
    hierarchy levels. The recommended practice is to use:
    - The module's __name__ variable in most cases
    - Explicit names for specific subsystems or components

    Examples:
    - "ultimate_mcp_server.core.state_store"
    - "ultimate_mcp_server.services.rag"
    - "ultimate_mcp_server.tools.local_text"

    Args:
        name: Logger name that identifies the component, module, or subsystem
              Usually set to the module's __name__ or a specific component identifier

    Returns:
        A configured Logger instance with the specified name

    Usage Examples:
    ```python
    # Standard usage in a module
    logger = get_logger(__name__)

    # Component-specific logger
    auth_logger = get_logger("ultimate_mcp_server.auth")

    # Usage with structured logging
    logger = get_logger("my_module")
    logger.info("User action",
                component="auth",
                operation="login",
                context={"user_id": user.id})
    ```

    Note:
        While each call returns a new Logger instance, they all share the underlying
        Python logging configuration and output destinations. This allows for
        centralized control of log levels, formatting, and output handlers through
        standard logging configuration.
    """
    # Initialize the global logger if needed
    global logger
    if logger is None:
        logger = Logger(name)

    # Return a new logger with the requested name
    return Logger(name)


# Helper functions for global usage
def debug(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    emoji_key: Optional[str] = None,
    **kwargs,
) -> None:
    """Forward to default logger's debug method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)

    logger.debug(message, component, operation, context, emoji_key=emoji_key, **kwargs)


def info(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    emoji_key: Optional[str] = None,
    **kwargs,
) -> None:
    """Forward to default logger's info method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)

    logger.info(message, component, operation, context, emoji_key=emoji_key, **kwargs)


def success(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    emoji_key: Optional[str] = None,
    **kwargs,
) -> None:
    """Forward to default logger's success method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)

    logger.success(message, component, operation, context, emoji_key=emoji_key, **kwargs)


def warning(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    emoji_key: Optional[str] = None,
    # details: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """Forward to default logger's warning method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)

    logger.warning(message, component, operation, context, emoji_key=emoji_key, **kwargs)


def error(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
    emoji_key: Optional[str] = None,
    # error_code: Optional[str] = None,
    # resolution_steps: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """Forward to default logger's error method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)

    # Handle exc_info specially to prevent conflicts
    exc_info = kwargs.pop("exc_info", None) if "exc_info" in kwargs else None

    logger.error(
        message,
        component,
        operation,
        context,
        exception=exception,
        emoji_key=emoji_key,
        **{**kwargs, "exc_info": exc_info} if exc_info is not None else kwargs,
    )


def critical(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
    emoji_key: Optional[str] = None,
    # error_code: Optional[str] = None,
    **kwargs,
) -> None:
    """Forward to default logger's critical method."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)

    # Handle exc_info specially to prevent conflicts
    exc_info = kwargs.pop("exc_info", None) if "exc_info" in kwargs else None

    logger.critical(
        message,
        component,
        operation,
        context,
        exception=exception,
        emoji_key=emoji_key,
        **{**kwargs, "exc_info": exc_info} if exc_info is not None else kwargs,
    )


def section(
    title: str,
    subtitle: Optional[str] = None,
    component: Optional[str] = None,
) -> None:
    """Display a section header using the global logger's console."""
    # Ensure logger is initialized
    global logger
    if logger is None:
        logger = Logger(__name__)

    logger.section(title, subtitle, component)


# Example Usage (if run directly)
if __name__ == "__main__":
    # Example of how the logger might be configured and used

    # Normally configuration happens via dictConfig in main entry point
    # For standalone testing, we can add a handler manually
    test_logger = Logger("test_logger", level="debug")  # Create instance
    test_logger.python_logger.addHandler(RichLoggingHandler(console=console))
    # Need to prevent propagation if manually adding handler here for test
    test_logger.python_logger.propagate = False

    test_logger.section("Initialization", "Setting up components")
    test_logger.startup(version="1.0.0", mode="test")

    test_logger.debug("This is a debug message", component="core", operation="setup")
    test_logger.info("This is an info message", component="api")
    test_logger.success(
        "Operation completed successfully", component="worker", operation="process_data"
    )
    test_logger.warning(
        "Something looks suspicious", component="cache", context={"key": "user:123"}
    )

    try:
        x = 1 / 0
    except ZeroDivisionError as e:
        test_logger.error("An error occurred", component="math", operation="divide", exception=e)

    test_logger.critical("System unstable!", component="core", context={"reason": "disk full"})

    test_logger.info_panel("Configuration", {"host": "localhost", "port": 8013}, component="core")
    test_logger.warning_panel(
        "Cache Alert",
        "Cache nearing capacity",
        details=["Size: 95MB", "Limit: 100MB"],
        component="cache",
    )
    test_logger.error_panel(
        "DB Connection Failed",
        "Could not connect to database",
        details="Connection timed out after 5s",
        resolution_steps=["Check DB server status", "Verify credentials"],
        error_code="DB500",
        component="db",
    )

    test_logger.tool(
        "grep",
        "grep 'error' log.txt",
        "line 1: error found\nline 5: error processing",
        status="success",
        duration=0.5,
        component="analysis",
    )
    test_logger.code(
        "def hello():\n  print('Hello')", language="python", title="Example Code", component="docs"
    )

    with test_logger.time_operation("long_process", component="worker"):
        time.sleep(0.5)

    with test_logger.task("Processing items", total=10) as p:
        for _i in range(10):
            time.sleep(0.05)
            p.update_task(p.current_task_id, advance=1)  # Assuming task context provides task_id

    @test_logger.log_call(component="utils", log_result=True)
    def add_numbers(a, b):
        return a + b

    add_numbers(5, 3)

    test_logger.shutdown(duration=123.45)

__all__ = [
    "critical",
    "debug",
    "error",
    "get_logger",
    "info",
    "logger",  # Add logger to exported names
    "warning",
]
