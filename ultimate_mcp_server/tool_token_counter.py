import inspect
import json
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from ultimate_mcp_server.constants import COST_PER_MILLION_TOKENS
from ultimate_mcp_server.tools.base import _get_json_schema_type
from ultimate_mcp_server.utils.text import count_tokens


def extract_tool_info(func: Callable, tool_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract tool information from a function, similar to how MCP does it.

    Args:
        func: The function to extract information from
        tool_name: Optional custom name for the tool (defaults to function name)

    Returns:
        Dictionary containing the tool information
    """
    # Get function name and docstring
    name = tool_name or func.__name__
    description = func.__doc__ or f"Tool: {name}"

    # Get function parameters
    sig = inspect.signature(func)
    params = {}

    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter for class methods
        if param_name == "self":
            continue

        # Skip context parameter which is usually added by decorators
        if param_name == "ctx":
            continue

        # Also skip state management parameters
        if param_name in ["get_state", "set_state", "delete_state"]:
            continue

        # Get parameter type annotation and default value
        param_type = param.annotation
        param_default = param.default if param.default is not inspect.Parameter.empty else None

        # Convert Python type to JSON Schema
        if param_type is not inspect.Parameter.empty:
            param_schema = _get_json_schema_type(param_type)
        else:
            param_schema = {"type": "object"}  # Default to object for unknown types

        # Add default value if available
        if param_default is not None:
            param_schema["default"] = param_default

        # Add to parameters
        params[param_name] = param_schema

    # Construct input schema
    input_schema = {
        "type": "object",
        "properties": params,
        "required": [
            param_name
            for param_name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param_name not in ["self", "ctx", "get_state", "set_state", "delete_state"]
        ],
    }

    # Construct final tool info
    tool_info = {"name": name, "description": description, "inputSchema": input_schema}

    return tool_info


def count_tool_registration_tokens(tools: List[Callable], model: str = "gpt-4o") -> int:
    """
    Count the tokens that would be used to register the given tools with an LLM.

    Args:
        tools: List of tool functions
        model: The model to use for token counting (default: gpt-4o)

    Returns:
        Total number of tokens
    """
    # Extract tool info for each tool
    tool_infos = [extract_tool_info(tool) for tool in tools]

    # Convert to JSON string (similar to what MCP does when sending to LLM)
    tools_json = json.dumps({"tools": tool_infos}, ensure_ascii=False)

    # Count tokens
    token_count = count_tokens(tools_json, model)

    return token_count


def calculate_cost_per_provider(token_count: int) -> Dict[str, float]:
    """
    Calculate the cost of including the tokens as input for various API providers.

    Args:
        token_count: Number of tokens

    Returns:
        Dictionary mapping provider names to costs in USD
    """
    costs = {}

    try:
        # Make sure we can access the cost data structure
        if not isinstance(COST_PER_MILLION_TOKENS, dict):
            console = Console()
            console.print("[yellow]Warning: COST_PER_MILLION_TOKENS is not a dictionary[/yellow]")
            return costs

        for provider_name, provider_info in COST_PER_MILLION_TOKENS.items():
            # Skip if provider_info is not a dictionary
            if not isinstance(provider_info, dict):
                continue

            # Choose a reasonable default input cost if we can't determine from models
            default_input_cost = 0.01  # $0.01 per million tokens as a safe default
            input_cost_per_million = default_input_cost

            try:
                # Try to get cost from provider models if available
                if provider_info and len(provider_info) > 0:
                    # Try to find the most expensive model
                    max_cost = 0
                    for _model_name, model_costs in provider_info.items():
                        if isinstance(model_costs, dict) and "input" in model_costs:
                            cost = model_costs["input"]
                            if cost > max_cost:
                                max_cost = cost

                    if max_cost > 0:
                        input_cost_per_million = max_cost
            except Exception as e:
                # If any error occurs, use the default cost
                console = Console()
                console.print(
                    f"[yellow]Warning getting costs for {provider_name}: {str(e)}[/yellow]"
                )

            # Calculate cost for this token count
            cost = (token_count / 1_000_000) * input_cost_per_million

            # Store in results
            costs[provider_name] = cost
    except Exception as e:
        console = Console()
        console.print(f"[red]Error calculating costs: {str(e)}[/red]")

    return costs


def display_tool_token_usage(
    current_tools_info: List[Dict[str, Any]], all_tools_info: List[Dict[str, Any]]
):
    """
    Display token usage information for tools in a Rich table.

    Args:
        current_tools_info: List of tool info dictionaries for currently registered tools
        all_tools_info: List of tool info dictionaries for all available tools
    """
    # Convert to JSON and count tokens
    current_json = json.dumps({"tools": current_tools_info}, ensure_ascii=False)
    all_json = json.dumps({"tools": all_tools_info}, ensure_ascii=False)

    current_token_count = count_tokens(current_json)
    all_token_count = count_tokens(all_json)

    # Calculate size in KB
    current_kb = len(current_json) / 1024
    all_kb = len(all_json) / 1024

    # Calculate costs for each provider
    current_costs = calculate_cost_per_provider(current_token_count)
    all_costs = calculate_cost_per_provider(all_token_count)

    # Create Rich table
    console = Console()
    table = Table(title="Tool Registration Token Usage")

    # Add columns
    table.add_column("Metric", style="cyan")
    table.add_column("Current Tools", style="green")
    table.add_column("All Tools", style="yellow")
    table.add_column("Difference", style="magenta")

    # Add rows
    table.add_row(
        "Number of Tools",
        str(len(current_tools_info)),
        str(len(all_tools_info)),
        str(len(all_tools_info) - len(current_tools_info)),
    )

    table.add_row("Size (KB)", f"{current_kb:.2f}", f"{all_kb:.2f}", f"{all_kb - current_kb:.2f}")

    table.add_row(
        "Token Count",
        f"{current_token_count:,}",
        f"{all_token_count:,}",
        f"{all_token_count - current_token_count:,}",
    )

    # Add cost rows for each provider
    for provider_name in sorted(current_costs.keys()):
        current_cost = current_costs.get(provider_name, 0)
        all_cost = all_costs.get(provider_name, 0)

        table.add_row(
            f"Cost ({provider_name})",
            f"${current_cost:.4f}",
            f"${all_cost:.4f}",
            f"${all_cost - current_cost:.4f}",
        )

    # Print table
    console.print(table)

    return {
        "current_tools": {
            "count": len(current_tools_info),
            "size_kb": current_kb,
            "tokens": current_token_count,
            "costs": current_costs,
        },
        "all_tools": {
            "count": len(all_tools_info),
            "size_kb": all_kb,
            "tokens": all_token_count,
            "costs": all_costs,
        },
    }


async def count_registered_tools_tokens(mcp_server):
    """
    Count tokens for tools that are currently registered with the MCP server.

    Args:
        mcp_server: The MCP server instance

    Returns:
        Dictionary with token counts and costs
    """
    # Get registered tools info from the server
    # Since we might not have direct access to the function objects, extract tool info from the MCP API
    if hasattr(mcp_server, "tools") and hasattr(mcp_server.tools, "list"):
        # Try to get tool definitions directly
        current_tools_info = await mcp_server.tools.list()
    else:
        # Fallback if we can't access the tools directly
        current_tools_info = []
        console = Console()
        console.print(
            "[yellow]Warning: Could not directly access registered tools from MCP server[/yellow]"
        )

    try:
        # Import all available tools
        from ultimate_mcp_server.tools import STANDALONE_TOOL_FUNCTIONS

        # Extract full tool info for all available tools
        all_tools_info = [extract_tool_info(func) for func in STANDALONE_TOOL_FUNCTIONS]
    except ImportError:
        console = Console()
        console.print("[yellow]Warning: Could not import STANDALONE_TOOL_FUNCTIONS[/yellow]")
        all_tools_info = []

    # Display token usage
    result = display_tool_token_usage(current_tools_info, all_tools_info)

    return result
