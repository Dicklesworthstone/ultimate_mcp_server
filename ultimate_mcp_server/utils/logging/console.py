"""
Rich console configuration for Gateway logging system.

This module provides a configured Rich console instance for beautiful terminal output,
along with utility functions for common console operations.
"""

import sys  # Add this import
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.box import ROUNDED, Box
from rich.console import Console, ConsoleRenderable
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.traceback import install as install_rich_traceback
from rich.tree import Tree

# Use relative import for theme
from .themes import RICH_THEME

# Configure global console with our theme
# Note: Recording might be useful for testing or specific scenarios
console = Console(
    theme=RICH_THEME,
    highlight=True,
    markup=True,
    emoji=True,
    record=False,  # Set to True to capture output for testing
    width=None,  # Auto-width, or set a fixed width if desired
    color_system="auto",  # "auto", "standard", "256", "truecolor"
    file=sys.stderr,  # Always use stderr to avoid interfering with JSON-RPC messages on stdout
)

# Install rich traceback handler for beautiful error tracebacks
# show_locals=True can be verbose, consider False for production
install_rich_traceback(console=console, show_locals=False)


# Custom progress bar setup
def create_progress(
    transient: bool = True, auto_refresh: bool = True, disable: bool = False, **kwargs
) -> Progress:
    """Create a customized Rich Progress instance.

    Args:
        transient: Whether to remove the progress bar after completion
        auto_refresh: Whether to auto-refresh the progress bar
        disable: Whether to disable the progress bar
        **kwargs: Additional arguments passed to Progress constructor

    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),  # Use theme style
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",  # Use theme style
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        auto_refresh=auto_refresh,
        disable=disable,
        **kwargs,
    )


@contextmanager
def status(message: str, spinner: str = "dots", **kwargs):
    """Context manager for displaying a status message during an operation.

    Args:
        message: The status message to display
        spinner: The spinner animation to use
        **kwargs: Additional arguments to pass to Status constructor

    Yields:
        Rich Status object that can be updated
    """
    with Status(message, console=console, spinner=spinner, **kwargs) as status_obj:
        yield status_obj


def print_panel(
    content: Union[str, Text, ConsoleRenderable],
    title: Optional[str] = None,
    style: Optional[str] = "info",  # Use theme styles by default
    box: Optional[Box] = ROUNDED,
    expand: bool = True,
    padding: Tuple[int, int] = (1, 2),
    **kwargs,
) -> None:
    """Print content in a styled panel.

    Args:
        content: The content to display in the panel
        title: Optional panel title
        style: Style name to apply (from theme)
        box: Box style to use
        expand: Whether the panel should expand to fill width
        padding: Panel padding (vertical, horizontal)
        **kwargs: Additional arguments to pass to Panel constructor
    """
    if isinstance(content, str):
        content = Text.from_markup(content)  # Allow markup in string content

    panel = Panel(
        content,
        title=title,
        style=style if style else "none",  # Pass style name directly
        border_style=style,  # Use same style for border unless overridden
        box=box,
        expand=expand,
        padding=padding,
        **kwargs,
    )
    console.print(panel)


def print_syntax(
    code: str,
    language: str = "python",
    line_numbers: bool = True,
    theme: str = "monokai",  # Standard Rich theme
    title: Optional[str] = None,
    background_color: Optional[str] = None,
    **kwargs,
) -> None:
    """Print syntax-highlighted code.

    Args:
        code: The code to highlight
        language: The programming language
        line_numbers: Whether to show line numbers
        theme: Syntax highlighting theme (e.g., 'monokai', 'native')
        title: Optional title for the code block (creates a panel)
        background_color: Optional background color
        **kwargs: Additional arguments to pass to Syntax constructor
    """
    syntax = Syntax(
        code,
        language,
        theme=theme,
        line_numbers=line_numbers,
        background_color=background_color,
        **kwargs,
    )

    if title:
        # Use a neutral panel style for code
        print_panel(syntax, title=title, style="none", padding=(0, 1))
    else:
        console.print(syntax)


def print_table(
    title: Optional[str] = None,
    columns: Optional[List[Union[str, Dict[str, Any]]]] = None,
    rows: Optional[List[List[Any]]] = None,
    box: Box = ROUNDED,
    show_header: bool = True,
    **kwargs,
) -> Table:
    """Create and print a Rich table.

    Args:
        title: Optional table title
        columns: List of column names or dicts for more control (e.g., {"header": "Name", "style": "bold"})
        rows: List of rows, each a list of values (will be converted to str)
        box: Box style to use
        show_header: Whether to show the table header
        **kwargs: Additional arguments to pass to Table constructor

    Returns:
        The created Table instance (in case further modification is needed)
    """
    table = Table(title=title, box=box, show_header=show_header, **kwargs)

    if columns:
        for column in columns:
            if isinstance(column, dict):
                table.add_column(**column)
            else:
                table.add_column(str(column))

    if rows:
        for row in rows:
            # Ensure all items are renderable (convert simple types to str)
            renderable_row = [
                item if isinstance(item, ConsoleRenderable) else str(item) for item in row
            ]
            table.add_row(*renderable_row)

    console.print(table)
    return table


def print_tree(
    name: str,
    data: Union[Dict[str, Any], List[Any]],
    guide_style: str = "bright_black",
    highlight: bool = True,
    **kwargs,
) -> None:
    """Print a hierarchical tree structure from nested data.

    Args:
        name: The root label of the tree
        data: Nested dictionary or list to render as a tree
        guide_style: Style for the tree guides
        highlight: Apply highlighting to the tree
        **kwargs: Additional arguments to pass to Tree constructor
    """
    tree = Tree(name, guide_style=guide_style, highlight=highlight, **kwargs)

    def build_tree(branch, node_data):
        """Recursively build the tree from nested data."""
        if isinstance(node_data, dict):
            for key, value in node_data.items():
                sub_branch = branch.add(str(key))
                build_tree(sub_branch, value)
        elif isinstance(node_data, list):
            for index, item in enumerate(node_data):
                # Use index as label or try to represent item briefly
                label = f"[{index}]"
                sub_branch = branch.add(label)
                build_tree(sub_branch, item)
        else:
            # Leaf node
            branch.add(Text(str(node_data)))

    build_tree(tree, data)
    console.print(tree)


def print_json(
    data: Any, title: Optional[str] = None, indent: int = 2, highlight: bool = True
) -> None:
    """Print data formatted as JSON with syntax highlighting.

    Args:
        data: The data to format as JSON.
        title: Optional title (creates a panel).
        indent: JSON indentation level.
        highlight: Apply syntax highlighting.
    """
    import json

    try:
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        if highlight:
            syntax = Syntax(json_str, "json", theme="native", word_wrap=True)
            if title:
                print_panel(syntax, title=title, style="none", padding=(0, 1))
            else:
                console.print(syntax)
        else:
            if title:
                print_panel(json_str, title=title, style="none", padding=(0, 1))
            else:
                console.print(json_str)
    except Exception as e:
        console.print(f"[error]Could not format data as JSON: {e}[/error]")


@contextmanager
def live_display(renderable: ConsoleRenderable, **kwargs):
    """Context manager for displaying a live-updating renderable.

    Args:
        renderable: The Rich renderable to display live.
        **kwargs: Additional arguments for the Live instance.

    Yields:
        The Live instance.
    """
    with Live(renderable, console=console, **kwargs) as live:
        yield live


def get_rich_console() -> Console:
    """Returns the shared Rich Console instance."""
    return console
