"""
Panel definitions for Ultimate MCP Server logging system.

This module provides specialized panels for different types of output like
headers, results, errors, warnings, etc.
"""

from typing import Any, Dict, List, Optional, Union

from rich.box import SIMPLE
from rich.columns import Columns
from rich.console import ConsoleRenderable
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ultimate_mcp_server.utils.logging.console import console
from ultimate_mcp_server.utils.logging.emojis import ERROR, INFO, SUCCESS, WARNING


class HeaderPanel:
    """Panel for section headers."""

    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        component: Optional[str] = None,
        style: str = "bright_blue",
    ):
        """Initialize a header panel.

        Args:
            title: Panel title
            subtitle: Optional subtitle
            component: Optional component name
            style: Panel style
        """
        self.title = title
        self.subtitle = subtitle
        self.component = component
        self.style = style

    def __rich__(self) -> ConsoleRenderable:
        """Render the panel."""
        # Create the title text
        title_text = Text()
        title_text.append("- ", style="bright_black")
        title_text.append(self.title, style="bold")
        title_text.append(" -", style="bright_black")

        # Create the content
        content = Text()

        if self.component:
            content.append(f"[{self.component}] ", style="component")

        if self.subtitle:
            content.append(self.subtitle)

        return Panel(
            content,
            title=title_text,
            title_align="center",
            border_style=self.style,
            expand=True,
            padding=(1, 2),
        )


class ResultPanel:
    """Panel for displaying operation results."""

    def __init__(
        self,
        title: str,
        results: Union[List[Dict[str, Any]], Dict[str, Any]],
        status: str = "success",
        component: Optional[str] = None,
        show_count: bool = True,
        compact: bool = False,
    ):
        """Initialize a result panel.

        Args:
            title: Panel title
            results: Results to display (list of dicts or single dict)
            status: Result status (success, warning, error)
            component: Optional component name
            show_count: Whether to show result count in title
            compact: Whether to use a compact display style
        """
        self.title = title
        self.results = results if isinstance(results, list) else [results]
        self.status = status.lower()
        self.component = component
        self.show_count = show_count
        self.compact = compact

    def __rich__(self) -> ConsoleRenderable:
        """Render the panel."""
        # Determine style and emoji based on status
        if self.status == "success":
            style = "result.success"
            emoji = SUCCESS
        elif self.status == "warning":
            style = "result.warning"
            emoji = WARNING
        elif self.status == "error":
            style = "result.error"
            emoji = ERROR
        else:
            style = "result.info"
            emoji = INFO

        # Create title
        title_text = Text()
        title_text.append(f"{emoji} ", style=style)
        title_text.append(self.title, style=f"bold {style}")

        if self.show_count and len(self.results) > 0:
            title_text.append(f" ({len(self.results)} items)", style="bright_black")

        # Create content
        if self.compact:
            # Compact mode - just show key/value list
            rows = []
            for item in self.results:
                for k, v in item.items():
                    rows.append(
                        {
                            "key": k,
                            "value": self._format_value(v),
                        }
                    )

            table = Table(box=None, expand=True, show_header=False)
            table.add_column("Key", style="data.key")
            table.add_column("Value", style="", overflow="fold")

            for row in rows:
                table.add_row(row["key"], row["value"])

            content = table
        else:
            # Full mode - create a table per result item
            tables = []

            for i, item in enumerate(self.results):
                if not item:  # Skip empty items
                    continue

                table = Table(
                    box=SIMPLE,
                    title=f"Item {i + 1}" if len(self.results) > 1 else None,
                    title_style="bright_black",
                    expand=True,
                    show_header=False,
                )
                table.add_column("Key", style="data.key")
                table.add_column("Value", style="", overflow="fold")

                for k, v in item.items():
                    table.add_row(k, self._format_value(v))

                tables.append(table)

            content = (
                Columns(tables) if len(tables) > 1 else tables[0] if tables else Text("No results")
            )

        # Return the panel
        return Panel(
            content,
            title=title_text,
            border_style=style,
            expand=True,
            padding=(1, 1),
        )

    def _format_value(self, value: Any) -> str:
        """Format a value for display.

        Args:
            value: Value to format

        Returns:
            Formatted string
        """
        if value is None:
            return "[dim]None[/dim]"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return ", ".join(self._format_value(v) for v in value[:5]) + (
                f" ... (+{len(value) - 5} more)" if len(value) > 5 else ""
            )
        elif isinstance(value, dict):
            if len(value) == 0:
                return "{}"
            else:
                return "{...}"  # Just indicate there's content
        else:
            return str(value)


class InfoPanel:
    """Panel for displaying information."""

    def __init__(
        self,
        title: str,
        content: Union[str, List[str], Dict[str, Any]],
        icon: Optional[str] = None,
        style: str = "info",
    ):
        """Initialize an information panel.

        Args:
            title: Panel title
            content: Content to display (string, list, or dict)
            icon: Emoji or icon character
            style: Style name to apply (from theme)
        """
        self.title = title
        self.content = content
        self.icon = icon or INFO
        self.style = style

    def __rich__(self) -> ConsoleRenderable:
        """Render the panel."""
        # Create title
        title_text = Text()
        title_text.append(f"{self.icon} ", style=self.style)
        title_text.append(self.title, style=f"bold {self.style}")

        # Format content based on type
        if isinstance(self.content, str):
            content = Text(self.content)
        elif isinstance(self.content, list):
            content = Text()
            for i, item in enumerate(self.content):
                if i > 0:
                    content.append("\n")
                content.append(f"• {item}")
        elif isinstance(self.content, dict):
            # Create a table for dict content
            table = Table(box=None, expand=True, show_header=False)
            table.add_column("Key", style="data.key")
            table.add_column("Value", style="", overflow="fold")

            for k, v in self.content.items():
                table.add_row(k, str(v))

            content = table
        else:
            content = Text(str(self.content))

        # Return the panel
        return Panel(
            content,
            title=title_text,
            border_style=self.style,
            expand=True,
            padding=(1, 2),
        )


class WarningPanel:
    """Panel for displaying warnings."""

    def __init__(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[List[str]] = None,
    ):
        """Initialize a warning panel.

        Args:
            title: Optional panel title
            message: Main warning message
            details: Optional list of detail points
        """
        self.title = title or "Warning"
        self.message = message
        self.details = details or []

    def __rich__(self) -> ConsoleRenderable:
        """Render the panel."""
        # Create title
        title_text = Text()
        title_text.append(f"{WARNING} ", style="warning")
        title_text.append(self.title, style="bold warning")

        # Create content
        content = Text()

        # Add message
        if self.message:
            content.append(self.message)

        # Add details if any
        if self.details and len(self.details) > 0:
            if self.message:
                content.append("\n\n")

            content.append("Details:", style="bold")
            content.append("\n")

            for i, detail in enumerate(self.details):
                if i > 0:
                    content.append("\n")
                content.append(f"• {detail}")

        # Return the panel
        return Panel(
            content,
            title=title_text,
            border_style="warning",
            expand=True,
            padding=(1, 2),
        )


class ErrorPanel:
    """Panel for displaying errors."""

    def __init__(
        self,
        title: Optional[str] = None,
        message: str = "",
        details: Optional[str] = None,
        resolution_steps: Optional[List[str]] = None,
        error_code: Optional[str] = None,
    ):
        """Initialize an error panel.

        Args:
            title: Optional panel title
            message: Main error message
            details: Optional error details
            resolution_steps: Optional list of steps to resolve the error
            error_code: Optional error code for reference
        """
        self.title = title or "Error"
        self.message = message
        self.details = details
        self.resolution_steps = resolution_steps or []
        self.error_code = error_code

    def __rich__(self) -> ConsoleRenderable:
        """Render the panel."""
        # Create title
        title_text = Text()
        title_text.append(f"{ERROR} ", style="error")
        title_text.append(self.title, style="bold error")

        if self.error_code:
            title_text.append(f" [{self.error_code}]", style="bright_black")

        # Create content
        content = Text()

        # Add message
        if self.message:
            content.append(self.message, style="bold")

        # Add details if any
        if self.details:
            if self.message:
                content.append("\n\n")

            content.append(self.details)

        # Add resolution steps if any
        if self.resolution_steps and len(self.resolution_steps) > 0:
            if self.message or self.details:
                content.append("\n\n")

            content.append("Resolution steps:", style="bold")
            content.append("\n")

            for i, step in enumerate(self.resolution_steps):
                if i > 0:
                    content.append("\n")
                content.append(f"{i + 1}. {step}")

        # Return the panel
        return Panel(
            content,
            title=title_text,
            border_style="error",
            expand=True,
            padding=(1, 2),
        )


class ToolOutputPanel:
    """Panel for displaying tool command output."""

    def __init__(
        self,
        tool: str,
        command: str,
        output: str,
        status: str = "success",
        duration: Optional[float] = None,
    ):
        """Initialize a tool output panel.

        Args:
            tool: Tool name (ripgrep, awk, jq, etc.)
            command: Command that was executed
            output: Command output text
            status: Execution status (success, error)
            duration: Optional execution duration in seconds
        """
        self.tool = tool
        self.command = command
        self.output = output
        self.status = status.lower()
        self.duration = duration

    def __rich__(self) -> ConsoleRenderable:
        """Render the panel."""
        # Determine style and emoji based on status
        if self.status == "success":
            style = "tool.success"
            emoji = SUCCESS
        else:
            style = "tool.error"
            emoji = ERROR

        # Create title
        title_text = Text()
        title_text.append(f"{emoji} ", style=style)
        title_text.append(f"{self.tool}", style=f"bold {style}")

        if self.duration is not None:
            title_text.append(f" ({self.duration:.2f}s)", style="tool.duration")

        # Create content
        content = Columns(
            [
                Panel(
                    Text(self.command, style="tool.command"),
                    title="Command",
                    title_style="bright_black",
                    border_style="tool.command",
                    padding=(1, 1),
                ),
                Panel(
                    Text(self.output, style="tool.output"),
                    title="Output",
                    title_style="bright_black",
                    border_style="bright_black",
                    padding=(1, 1),
                ),
            ],
            expand=True,
            padding=(0, 1),
        )

        # Return the panel
        return Panel(
            content,
            title=title_text,
            border_style=style,
            expand=True,
            padding=(1, 1),
        )


class CodePanel:
    """Panel for displaying code with syntax highlighting."""

    def __init__(
        self,
        code: str,
        language: str = "python",
        title: Optional[str] = None,
        line_numbers: bool = True,
        highlight_lines: Optional[List[int]] = None,
    ):
        """Initialize a code panel.

        Args:
            code: The code to display
            language: Programming language for syntax highlighting
            title: Optional panel title
            line_numbers: Whether to show line numbers
            highlight_lines: List of line numbers to highlight
        """
        self.code = code
        self.language = language
        self.title = title
        self.line_numbers = line_numbers
        self.highlight_lines = highlight_lines

    def __rich__(self) -> ConsoleRenderable:
        """Render the panel."""
        # Create syntax highlighting component
        syntax = Syntax(
            self.code,
            self.language,
            theme="monokai",
            line_numbers=self.line_numbers,
            highlight_lines=self.highlight_lines,
        )

        # Create title
        if self.title:
            title_text = Text(self.title)
        else:
            title_text = Text()
            title_text.append(self.language.capitalize(), style="bright_blue bold")
            title_text.append(" Code", style="bright_black")

        # Return the panel
        return Panel(
            syntax,
            title=title_text,
            border_style="bright_blue",
            expand=True,
            padding=(0, 0),
        )


# Helper functions for creating panels


def display_header(
    title: str,
    subtitle: Optional[str] = None,
    component: Optional[str] = None,
) -> None:
    """Display a section header.

    Args:
        title: Section title
        subtitle: Optional subtitle
        component: Optional component name
    """
    panel = HeaderPanel(title, subtitle, component)
    console.print(panel)


def display_results(
    title: str,
    results: Union[List[Dict[str, Any]], Dict[str, Any]],
    status: str = "success",
    component: Optional[str] = None,
    show_count: bool = True,
    compact: bool = False,
) -> None:
    """Display operation results.

    Args:
        title: Results title
        results: Results to display (list of dicts or single dict)
        status: Result status (success, warning, error)
        component: Optional component name
        show_count: Whether to show result count in title
        compact: Whether to use a compact display style
    """
    panel = ResultPanel(title, results, status, component, show_count, compact)
    console.print(panel)


def display_info(
    title: str,
    content: Union[str, List[str], Dict[str, Any]],
    icon: Optional[str] = None,
    style: str = "info",
) -> None:
    """Display an information panel.

    Args:
        title: Panel title
        content: Content to display (string, list, or dict)
        icon: Emoji or icon character
        style: Style name to apply (from theme)
    """
    panel = InfoPanel(title, content, icon, style)
    console.print(panel)


def display_warning(
    title: Optional[str] = None,
    message: str = "",
    details: Optional[List[str]] = None,
) -> None:
    """Display a warning panel.

    Args:
        title: Optional panel title
        message: Main warning message
        details: Optional list of detail points
    """
    panel = WarningPanel(title, message, details)
    console.print(panel)


def display_error(
    title: Optional[str] = None,
    message: str = "",
    details: Optional[str] = None,
    resolution_steps: Optional[List[str]] = None,
    error_code: Optional[str] = None,
) -> None:
    """Display an error panel.

    Args:
        title: Optional panel title
        message: Main error message
        details: Optional error details
        resolution_steps: Optional list of steps to resolve the error
        error_code: Optional error code for reference
    """
    panel = ErrorPanel(title, message, details, resolution_steps, error_code)
    console.print(panel)


def display_tool_output(
    tool: str,
    command: str,
    output: str,
    status: str = "success",
    duration: Optional[float] = None,
) -> None:
    """Display tool command output.

    Args:
        tool: Tool name (ripgrep, awk, jq, etc.)
        command: Command that was executed
        output: Command output text
        status: Execution status (success, error)
        duration: Optional execution duration in seconds
    """
    panel = ToolOutputPanel(tool, command, output, status, duration)
    console.print(panel)


def display_code(
    code: str,
    language: str = "python",
    title: Optional[str] = None,
    line_numbers: bool = True,
    highlight_lines: Optional[List[int]] = None,
) -> None:
    """Display code with syntax highlighting.

    Args:
        code: The code to display
        language: Programming language for syntax highlighting
        title: Optional panel title
        line_numbers: Whether to show line numbers
        highlight_lines: List of line numbers to highlight
    """
    panel = CodePanel(code, language, title, line_numbers, highlight_lines)
    console.print(panel)
