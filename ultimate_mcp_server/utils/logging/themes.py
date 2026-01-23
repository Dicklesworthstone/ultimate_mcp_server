"""
Color themes for Gateway logging system.

This module defines color schemes for different log levels, operations, and components
to provide visual consistency and improve readability of log output.
"""

from typing import Optional, Tuple

from rich.style import Style
from rich.theme import Theme

COLORS = {
    # Main colors
    "primary": "bright_blue",
    "secondary": "cyan",
    "accent": "magenta",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "critical": "bright_red",
    "info": "bright_blue",
    "debug": "bright_black",
    "trace": "bright_black",
    # Component-specific colors (Adapt as needed for ultimate)
    "core": "blue",
    "provider": "cyan",  # Example: Renamed 'composite' to 'provider'
    "router": "green",  # Example: Renamed 'analysis' to 'router'
    "cache": "bright_magenta",
    "api": "bright_yellow",
    "mcp": "bright_blue",  # Kept if relevant
    "utils": "magenta",  # Example: Added 'utils'
    "default_component": "bright_cyan",  # Fallback component color
    # Tool-specific colors (Keep or remove as needed)
    "ripgrep": "blue",
    "awk": "green",
    "jq": "yellow",
    "sqlite": "magenta",
    # Result/Status colors
    "high_confidence": "green",
    "medium_confidence": "yellow",
    "low_confidence": "red",
    # Misc
    "muted": "bright_black",
    "highlight": "bright_white",
    "timestamp": "bright_black",
    "path": "bright_blue",
    "code": "bright_cyan",
    "data": "bright_yellow",
    "data.key": "bright_black",  # Added for context tables
}

STYLES = {
    # Base styles for log levels
    "info": Style(color=COLORS["info"]),
    "debug": Style(color=COLORS["debug"]),
    "warning": Style(color=COLORS["warning"], bold=True),
    "error": Style(color=COLORS["error"], bold=True),
    "critical": Style(color=COLORS["critical"], bold=True, reverse=True),
    "success": Style(color=COLORS["success"], bold=True),
    "trace": Style(color=COLORS["trace"], dim=True),
    # Component styles (Matching adapted COLORS)
    "core": Style(color=COLORS["core"], bold=True),
    "provider": Style(color=COLORS["provider"], bold=True),
    "router": Style(color=COLORS["router"], bold=True),
    "cache": Style(color=COLORS["cache"], bold=True),
    "api": Style(color=COLORS["api"], bold=True),
    "mcp": Style(color=COLORS["mcp"], bold=True),
    "utils": Style(color=COLORS["utils"], bold=True),
    "default_component": Style(color=COLORS["default_component"], bold=True),
    # Operation styles
    "operation": Style(color=COLORS["accent"], bold=True),
    "startup": Style(color=COLORS["success"], bold=True),
    "shutdown": Style(color=COLORS["error"], bold=True),
    "request": Style(color=COLORS["primary"], bold=True),
    "response": Style(color=COLORS["secondary"], bold=True),
    # Confidence level styles
    "high_confidence": Style(color=COLORS["high_confidence"], bold=True),
    "medium_confidence": Style(color=COLORS["medium_confidence"], bold=True),
    "low_confidence": Style(color=COLORS["low_confidence"], bold=True),
    # Misc styles
    "timestamp": Style(color=COLORS["timestamp"], dim=True),
    "path": Style(color=COLORS["path"], underline=True),
    "code": Style(color=COLORS["code"], italic=True),
    "data": Style(color=COLORS["data"]),
    "data.key": Style(color=COLORS["data.key"], bold=True),
    "muted": Style(color=COLORS["muted"], dim=True),
    "highlight": Style(color=COLORS["highlight"], bold=True),
}

# Rich theme that can be used directly with Rich Console
RICH_THEME = Theme({name: style for name, style in STYLES.items()})


# Get the appropriate style for a log level
def get_level_style(level: str) -> Style:
    """Get the Rich style for a specific log level.

    Args:
        level: The log level (info, debug, warning, error, critical, success, trace)

    Returns:
        The corresponding Rich Style
    """
    level = level.lower()
    return STYLES.get(level, STYLES["info"])  # Default to info style


# Get style for a component
def get_component_style(component: str) -> Style:
    """Get the Rich style for a specific component.

    Args:
        component: The component name (core, provider, router, etc.)

    Returns:
        The corresponding Rich Style
    """
    component = component.lower()
    # Fallback to a default component style if specific one not found
    return STYLES.get(component, STYLES["default_component"])


# Get color by name
def get_color(name: str) -> str:
    """Get a color by name.

    Args:
        name: The color name

    Returns:
        The color string that can be used with Rich
    """
    return COLORS.get(name.lower(), COLORS["primary"])


# Apply style to text directly
def style_text(text: str, style_name: str) -> str:
    """Apply a named style to text (for use without Rich console).

    This is a utility function that doesn't depend on Rich, useful for
    simple terminal output or when Rich console isn't available.

    Args:
        text: The text to style
        style_name: The name of the style to apply

    Returns:
        Text with ANSI color codes applied (using Rich tags for simplicity)
    """
    # This uses Rich markup format for simplicity, assuming it will be printed
    # by a Rich console later or that the markup is acceptable.
    return f"[{style_name}]{text}[/{style_name}]"


# Get foreground and background colors for a specific context
def get_context_colors(context: str, component: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Get appropriate foreground and background colors for a log context.

    Args:
        context: The log context (e.g., 'request', 'response')
        component: Optional component name for further refinement

    Returns:
        Tuple of (foreground_color, background_color) or (color, None)
    """
    style = STYLES.get(context.lower()) or STYLES.get("default_component")

    if style and style.color:
        return (str(style.color.name), str(style.bgcolor.name) if style.bgcolor else None)
    else:
        # Fallback to basic colors
        fg = COLORS.get(context.lower(), COLORS["primary"])
        return (fg, None)
