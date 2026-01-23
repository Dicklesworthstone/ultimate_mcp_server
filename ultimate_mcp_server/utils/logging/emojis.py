"""
Emoji definitions for Gateway logging system.

This module contains constants for emojis used in logging to provide visual cues
about the type and severity of log messages.
"""

from typing import Dict

# Log level emojis
INFO = "ℹ️"
DEBUG = "🔍"
WARNING = "⚠️"
ERROR = "❌"
CRITICAL = "🚨"
SUCCESS = "✅"
TRACE = "📍"

# Status emojis
RUNNING = "🔄"
PENDING = "⏳"
COMPLETED = "🏁"
FAILED = "👎"
STARTING = "🚀"
STOPPING = "🛑"
RESTARTING = "🔁"
LOADING = "📥"
SAVING = "📤"
CANCELLED = "🚫"
TIMEOUT = "⏱️"
SKIPPED = "⏭️"

# Operation emojis (Adapt for ultimate)
REQUEST = "➡️"  # Example
RESPONSE = "⬅️"  # Example
PROCESS = "⚙️"  # Example
CACHE_HIT = "✅"  # Example
CACHE_MISS = "❌"  # Example
AUTHENTICATE = "🔒"  # Example
AUTHORIZE = "🔑"  # Example
VALIDATE = "✔️"
CONNECT = "🔌"
DISCONNECT = "🔌"
UPDATE = "📝"

# Component emojis (Adapt for ultimate)
CORE = "⚙️"
PROVIDER = "☁️"  # Example
ROUTER = "🔀"  # Example
CACHE = "📦"
API = "🌐"
MCP = "📡"  # Keep if relevant
UTILS = "🔧"  # Example

# Tool emojis (Keep/remove/add as needed)
# RIPGREP = "🔍"
# AWK = "🔧"
# JQ = "🧰"
# SQLITE = "🗃️"

# Result emojis
FOUND = "🎯"
NOT_FOUND = "🔍"
PARTIAL = "◐"
UNKNOWN = "❓"
HIGH_CONFIDENCE = "🔒"
MEDIUM_CONFIDENCE = "🔓"
LOW_CONFIDENCE = "🚪"

# System emojis
STARTUP = "🔆"
SHUTDOWN = "🔅"
CONFIG = "⚙️"
ERROR = "⛔"  # Distinct from level error
WARNING = "⚠️"  # Same as level warning
DEPENDENCY = "🧱"
VERSION = "🏷️"
UPDATE_AVAILABLE = "🆕"

# User interaction emojis (Keep if relevant)
INPUT = "⌨️"
OUTPUT = "📺"
HELP = "❓"
HINT = "💡"
EXAMPLE = "📋"
QUESTION = "❓"
ANSWER = "💬"

# Time emojis
TIMING = "⏱️"
SCHEDULED = "📅"
DELAYED = "⏰"
OVERTIME = "⌛"

# Convenience mapping for log levels
LEVEL_EMOJIS: Dict[str, str] = {
    "info": INFO,
    "debug": DEBUG,
    "warning": WARNING,
    "error": ERROR,
    "critical": CRITICAL,
    "success": SUCCESS,
    "trace": TRACE,
}

# Dictionary for mapping operation names to emojis
OPERATION_EMOJIS: Dict[str, str] = {
    "request": REQUEST,
    "response": RESPONSE,
    "process": PROCESS,
    "cache_hit": CACHE_HIT,
    "cache_miss": CACHE_MISS,
    "authenticate": AUTHENTICATE,
    "authorize": AUTHORIZE,
    "validate": VALIDATE,
    "connect": CONNECT,
    "disconnect": DISCONNECT,
    "update": UPDATE,
    # Add other common operations here
    "startup": STARTUP,
    "shutdown": SHUTDOWN,
    "config": CONFIG,
}

# Dictionary for mapping component names to emojis
COMPONENT_EMOJIS: Dict[str, str] = {
    "core": CORE,
    "provider": PROVIDER,
    "router": ROUTER,
    "cache": CACHE,
    "api": API,
    "mcp": MCP,
    "utils": UTILS,
    # Add other components here
}


# Get emoji by name function for more dynamic access
def get_emoji(category: str, name: str) -> str:
    """Get an emoji by category and name.

    Args:
        category: The category of emoji (e.g., 'level', 'status', 'operation', 'component')
        name: The name of the emoji within that category

    Returns:
        The emoji string or a default '?' if not found
    """
    category = category.lower()
    name_lower = name.lower()

    if category == "level":
        return LEVEL_EMOJIS.get(name_lower, "?")
    elif category == "operation":
        return OPERATION_EMOJIS.get(name_lower, "⚙️")  # Default to generic gear
    elif category == "component":
        return COMPONENT_EMOJIS.get(name_lower, "🧩")  # Default to puzzle piece

    # Fallback for other categories or direct constant lookup
    name_upper = name.upper()
    globals_dict = globals()
    if name_upper in globals_dict:
        return globals_dict[name_upper]

    # Default if nothing matches
    return "❓"
