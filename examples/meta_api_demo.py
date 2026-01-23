#!/usr/bin/env python
"""
Meta API Tool Demonstration Script.

This script demonstrates the functionality of the APIMetaTool class for dynamically
registering and using external APIs via their OpenAPI specifications.

The demo features:
1. Registering APIs with the MCP server using their OpenAPI specifications
2. Listing registered APIs and their endpoints
3. Getting detailed information about an API and its endpoints
4. Calling dynamically registered tools
5. Refreshing an API to update its endpoints
6. Getting detailed information about a specific tool
7. Unregistering APIs

We use the Swagger Petstore API as our primary demo example along with additional
public APIs for multi-API demonstrations.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

import ultimate_mcp_server.core  # To access the global gateway instance
from ultimate_mcp_server import create_app
from ultimate_mcp_server.tools.meta_api_tool import APIMetaTool  # Import class for type hinting

# Initialize Rich console
console = Console()

# Demo APIs to showcase
DEMO_APIS = {
    "petstore": {
        "name": "petstore",
        "url": "https://petstore.swagger.io/v2/swagger.json",
        "description": "Swagger Petstore API - A sample API for pet store management",
    },
    "weather": {
        "name": "weather",
        "url": "https://api.met.no/weatherapi/locationforecast/2.0/openapi.json",
        "description": "Norwegian Meteorological Institute API - Weather forecast data",
    },
    "mock": {
        "name": "mockapi",
        "url": "https://fastapimockserver.onrender.com/openapi.json",
        "description": "Mock API Server - A simple mock API for testing",
    },
}

# Default API to use for demos
DEFAULT_API = "petstore"


async def show_intro():
    """Display an introduction to the demo."""
    console.clear()
    console.print("\n[bold cyan]META API TOOL DEMONSTRATION[/bold cyan]", justify="center")
    console.print(
        "[italic]Dynamically register and use any OpenAPI-compatible API[/italic]", justify="center"
    )
    console.print("\n")

    # Display APIs that will be used in demo
    table = Table(title="Demo APIs", box=None, highlight=True, border_style="blue")
    table.add_column("API Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("OpenAPI URL", style="blue")

    for api_info in DEMO_APIS.values():
        table.add_row(api_info["name"], api_info["description"], api_info["url"])

    console.print(Panel(table, border_style="blue", title="Available APIs", expand=False))

    # Display introduction as markdown
    intro_md = """
    ## Welcome to the Meta API Tool Demo
    
    This demonstration shows how to use the Meta API Tool to:
    - Register external APIs dynamically
    - Access API endpoints as tools
    - Call external services seamlessly
    """
    console.print(Markdown(intro_md))
    console.print("\n")


async def register_api_demo(
    api_meta_tool: APIMetaTool, api_name: str = DEFAULT_API
) -> Dict[str, Any]:
    """Register an API with the MCP server using its OpenAPI specification.

    Args:
        api_meta_tool: The APIMetaTool instance
        api_name: Name of the API to register

    Returns:
        Result of the registration
    """
    console.print(Rule(f"[bold blue]REGISTERING API: {api_name.upper()}[/bold blue]"))

    api_info = DEMO_APIS.get(api_name)
    if not api_info:
        console.print(
            f"[bold red]Error:[/bold red] API '{api_name}' not found in demo configuration."
        )
        return {}

    # Show the API details before registration
    console.print(
        Panel(
            f"[bold]API Name:[/bold] {api_info['name']}\n"
            f"[bold]OpenAPI URL:[/bold] {api_info['url']}\n"
            f"[bold]Description:[/bold] {api_info['description']}",
            title="API Registration Details",
            border_style="green",
            expand=False,
        )
    )

    console.print("[cyan]> Fetching OpenAPI specification from URL and registering tools...[/cyan]")

    start_time = time.time()
    with Progress(
        SpinnerColumn(), TextColumn("[bold green]Registering API..."), transient=True
    ) as progress:
        task = progress.add_task("", total=None)  # noqa: F841
        try:
            # Use the passed-in instance directly
            result = await api_meta_tool.register_api(
                api_name=api_info["name"], openapi_url=api_info["url"]
            )
            processing_time = time.time() - start_time

            # Show success message
            console.print(
                f"[bold green]✓ Success![/bold green] API registered in {processing_time:.2f}s"
            )

            # Display registered tools in a table
            if result.get("tools_count", 0) > 0:
                table = Table(
                    title=f"Registered {result['tools_count']} Tools",
                    box=None,
                    highlight=True,
                    border_style="blue",
                )
                table.add_column("Tool Name", style="cyan")

                for tool in result.get("tools_registered", []):
                    table.add_row(tool)

                console.print(Panel(table, border_style="green", expand=False))
            else:
                console.print("[yellow]No tools were registered for this API.[/yellow]")

            return result
        except Exception as e:
            console.print(f"[bold red]Error during API registration:[/bold red] {str(e)}")
            return {}


async def list_apis_demo(api_meta_tool: APIMetaTool) -> Dict[str, Any]:
    """List all registered APIs and their tools.

    Args:
        api_meta_tool: The APIMetaTool instance

    Returns:
        Result containing information about registered APIs
    """
    console.print(Rule("[bold blue]LISTING REGISTERED APIs[/bold blue]"))

    with console.status("[bold green]Fetching registered APIs...", spinner="dots"):
        try:
            result = await api_meta_tool.list_registered_apis()

            if result.get("total_apis", 0) > 0:
                # Display registered APIs in a table
                table = Table(
                    title=f"Registered APIs ({result['total_apis']})",
                    box=None,
                    highlight=True,
                    border_style="blue",
                )
                table.add_column("API Name", style="cyan")
                table.add_column("Base URL", style="blue")
                table.add_column("Tools Count", style="green", justify="right")

                for api_name, api_info in result.get("apis", {}).items():
                    table.add_row(
                        api_name,
                        api_info.get("base_url", "N/A"),
                        str(api_info.get("tools_count", 0)),
                    )

                console.print(Panel(table, border_style="green", expand=False))
                console.print(f"[green]Total Tools: {result.get('total_tools', 0)}[/green]")
            else:
                console.print("[yellow]No APIs are currently registered.[/yellow]")

            return result
        except Exception as e:
            console.print(f"[bold red]Error listing APIs:[/bold red] {str(e)}")
            return {}


async def get_api_details_demo(
    api_meta_tool: APIMetaTool, api_name: str = DEFAULT_API
) -> Dict[str, Any]:
    """Get detailed information about a registered API.

    Args:
        api_meta_tool: The APIMetaTool instance
        api_name: Name of the API to get details for

    Returns:
        API details
    """
    console.print(Rule(f"[bold blue]API DETAILS: {api_name.upper()}[/bold blue]"))

    with console.status(f"[bold green]Fetching details for {api_name} API...", spinner="dots"):
        try:
            result = await api_meta_tool.get_api_details(api_name=api_name)

            # Display API overview
            console.print(
                Panel(
                    f"[bold]API Name:[/bold] {result.get('api_name', 'N/A')}\n"
                    f"[bold]Base URL:[/bold] {result.get('base_url', 'N/A')}\n"
                    f"[bold]OpenAPI URL:[/bold] {result.get('openapi_url', 'N/A')}\n"
                    f"[bold]Endpoints Count:[/bold] {result.get('endpoints_count', 0)}",
                    title="API Overview",
                    border_style="green",
                    expand=False,
                )
            )

            # Display endpoints in a table
            endpoints = result.get("tools", [])
            if endpoints:
                table = Table(
                    title=f"API Endpoints ({len(endpoints)})",
                    box=None,
                    highlight=True,
                    border_style="blue",
                )
                table.add_column("Tool Name", style="cyan")
                table.add_column("Method", style="magenta", justify="center")
                table.add_column("Path", style="blue")
                table.add_column("Summary", style="green")

                for endpoint in endpoints:
                    table.add_row(
                        endpoint.get("name", "N/A"),
                        endpoint.get("method", "N/A").upper(),
                        endpoint.get("path", "N/A"),
                        endpoint.get("summary", "No summary") or "No summary",
                    )

                console.print(Panel(table, border_style="green", expand=False))
            else:
                console.print("[yellow]No endpoints found for this API.[/yellow]")

            return result
        except Exception as e:
            console.print(f"[bold red]Error getting API details:[/bold red] {str(e)}")
            return {}


async def get_tool_details_demo(
    api_meta_tool: APIMetaTool, api_name: str = DEFAULT_API
) -> Dict[str, Any]:
    """Get detailed information about a specific tool from an API.

    Args:
        api_meta_tool: The APIMetaTool instance
        api_name: Name of the API that contains the tool

    Returns:
        Tool details
    """
    console.print(Rule(f"[bold blue]TOOL DETAILS DEMO FOR {api_name.upper()}[/bold blue]"))

    # First get the API details to find available tools
    with console.status(
        f"[bold green]Fetching available tools for {api_name} API...", spinner="dots"
    ):
        try:
            api_details = await api_meta_tool.get_api_details(api_name=api_name)

            if not api_details.get("tools", []):
                console.print(f"[yellow]No tools available for {api_name} API.[/yellow]")
                return {}

            # Find a suitable GET tool for demo purposes
            tools = api_details.get("tools", [])
            get_tools = [t for t in tools if t.get("method", "").lower() == "get"]

            if get_tools:
                # Prefer a GET tool with path parameters for a more interesting demo
                path_param_tools = [t for t in get_tools if "{" in t.get("path", "")]
                if path_param_tools:
                    selected_tool = path_param_tools[0]
                else:
                    selected_tool = get_tools[0]
            else:
                # If no GET tools, just pick the first tool
                selected_tool = tools[0]

            tool_name = selected_tool.get("name", "")
            console.print(f"[cyan]Selected tool for details:[/cyan] [bold]{tool_name}[/bold]")

            # Get detailed information about the selected tool
            with console.status(f"[bold green]Fetching details for {tool_name}...", spinner="dots"):
                result = await api_meta_tool.get_tool_details(tool_name=tool_name)

                # Display tool overview
                console.print(
                    Panel(
                        f"[bold]Tool Name:[/bold] {result.get('tool_name', 'N/A')}\n"
                        f"[bold]API Name:[/bold] {result.get('api_name', 'N/A')}\n"
                        f"[bold]Method:[/bold] {result.get('method', 'N/A').upper()}\n"
                        f"[bold]Path:[/bold] {result.get('path', 'N/A')}\n"
                        f"[bold]Summary:[/bold] {result.get('summary', 'No summary') or 'No summary'}\n"
                        f"[bold]Description:[/bold] {result.get('description', 'No description') or 'No description'}",
                        title="Tool Overview",
                        border_style="green",
                        expand=False,
                    )
                )

                # Display parameters if any
                parameters = result.get("parameters", [])
                if parameters:
                    param_table = Table(
                        title="Tool Parameters", box=None, highlight=True, border_style="blue"
                    )
                    param_table.add_column("Name", style="cyan")
                    param_table.add_column("Type", style="blue")
                    param_table.add_column("Required", style="green", justify="center")
                    param_table.add_column("In", style="magenta")
                    param_table.add_column("Description", style="yellow")

                    for param in parameters:
                        param_type = param.get("schema", {}).get("type", "string")
                        required = "✓" if param.get("required", False) else "-"
                        param_in = param.get("in", "query")
                        description = param.get("description", "No description") or "No description"

                        param_table.add_row(
                            param.get("name", "N/A"), param_type, required, param_in, description
                        )

                    console.print(Panel(param_table, border_style="green", expand=False))
                else:
                    console.print("[yellow]This tool has no parameters.[/yellow]")

                # Display source code
                source_code = result.get("source_code", "Source code not available")
                if len(source_code) > 500:
                    # Truncate long source code for display purposes
                    source_code = source_code[:500] + "\n\n[...truncated...]"

                console.print(
                    Panel(
                        Syntax(source_code, "python", theme="monokai", line_numbers=True),
                        title="Tool Source Code",
                        border_style="green",
                        expand=False,
                    )
                )

                return result
        except Exception as e:
            console.print(f"[bold red]Error getting tool details:[/bold red] {str(e)}")
            return {}


async def call_tool_demo(api_meta_tool: APIMetaTool, api_name: str = DEFAULT_API) -> Dict[str, Any]:
    """Call a dynamically registered tool from an API.

    Args:
        api_meta_tool: The APIMetaTool instance
        api_name: Name of the API that contains the tool

    Returns:
        Result of the tool call
    """
    console.print(Rule(f"[bold blue]CALLING TOOL FROM {api_name.upper()}[/bold blue]"))

    # First get the API details to find available tools
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold green]Fetching available tools for {api_name} API..."),
        transient=True,
    ) as progress:
        task = progress.add_task("", total=None)  # noqa: F841
        try:
            api_details = await api_meta_tool.get_api_details(api_name=api_name)

            if not api_details.get("tools", []):
                console.print(f"[yellow]No tools available for {api_name} API.[/yellow]")
                return {}

            # Find a suitable GET tool for demo purposes
            tools = api_details.get("tools", [])

            # For Petstore API, use specific endpoints for better demonstration
            if api_name == "petstore":
                # Try to find the "findPetsByStatus" endpoint, which is a good demo endpoint
                pet_status_tools = [t for t in tools if "findPetsByStatus" in t.get("name", "")]
                if pet_status_tools:
                    selected_tool = pet_status_tools[0]
                else:
                    # Fall back to "getInventory" which doesn't need parameters
                    inventory_tools = [t for t in tools if "getInventory" in t.get("name", "")]
                    if inventory_tools:
                        selected_tool = inventory_tools[0]
                    else:
                        # Just pick a GET endpoint if specific ones not found
                        get_tools = [t for t in tools if t.get("method", "").lower() == "get"]
                        selected_tool = get_tools[0] if get_tools else tools[0]
            else:
                # For other APIs, prefer GET endpoints without path parameters for simplicity
                get_tools = [
                    t
                    for t in tools
                    if t.get("method", "").lower() == "get" and "{" not in t.get("path", "")
                ]
                if get_tools:
                    selected_tool = get_tools[0]
                else:
                    # Fall back to any GET endpoint
                    get_tools = [t for t in tools if t.get("method", "").lower() == "get"]
                    if get_tools:
                        selected_tool = get_tools[0]
                    else:
                        # Just pick the first tool if no GET tools
                        selected_tool = tools[0]

            tool_name = selected_tool.get("name", "")
            tool_method = selected_tool.get("method", "").upper()
            tool_path = selected_tool.get("path", "")
            tool_summary = selected_tool.get("summary", "No summary") or "No summary"

            console.print(
                Panel(
                    f"[bold]Selected Tool:[/bold] {tool_name}\n"
                    f"[bold]Method:[/bold] {tool_method}\n"
                    f"[bold]Path:[/bold] {tool_path}\n"
                    f"[bold]Summary:[/bold] {tool_summary}",
                    title="Tool Information",
                    border_style="green",
                    expand=False,
                )
            )

            # Get tool details to determine parameters
            tool_details = await api_meta_tool.get_tool_details(tool_name=tool_name)
            parameters = tool_details.get("parameters", [])

            # Prepare inputs based on the tool
            inputs = {}

            # For Petstore API, use specific values
            if api_name == "petstore":
                if "findPetsByStatus" in tool_name:
                    inputs = {"status": "available"}
                    console.print("[cyan]Using input:[/cyan] status=available")
                elif "getPetById" in tool_name:
                    inputs = {"petId": 1}
                    console.print("[cyan]Using input:[/cyan] petId=1")
            else:
                # For other tools, add required parameters
                required_params = [p for p in parameters if p.get("required", False)]
                if required_params:
                    console.print(
                        "[yellow]This tool requires parameters. Using default values for demo.[/yellow]"
                    )

                    for param in required_params:
                        param_name = param.get("name", "")
                        param_type = param.get("schema", {}).get("type", "string")
                        param_in = param.get("in", "query")

                        # Assign default values based on parameter type
                        if param_type == "integer":
                            inputs[param_name] = 1
                        elif param_type == "number":
                            inputs[param_name] = 1.0
                        elif param_type == "boolean":
                            inputs[param_name] = True
                        else:  # string or other
                            inputs[param_name] = "test"

                        console.print(
                            f"[cyan]Using input:[/cyan] {param_name}={inputs[param_name]} ({param_in})"
                        )

            # Call the tool
            console.print("\n[bold]Calling the tool...[/bold]")
            start_time = time.time()
            with console.status(f"[bold green]Executing {tool_name}...", spinner="dots"):
                result = await api_meta_tool.call_dynamic_tool(tool_name=tool_name, inputs=inputs)
                processing_time = time.time() - start_time

            console.print(
                f"[bold green]✓ Success![/bold green] Tool executed in {processing_time:.2f}s"
            )

            # Display result as formatted JSON
            result_json = json.dumps(result, indent=2)
            console.print(
                Panel(
                    Syntax(result_json, "json", theme="monokai", line_numbers=True),
                    title="Tool Response",
                    border_style="green",
                    expand=False,
                )
            )

            return result
        except Exception as e:
            console.print(f"[bold red]Error calling tool:[/bold red] {str(e)}")
            return {}


async def list_tools_demo(api_meta_tool: APIMetaTool) -> Dict[str, Any]:
    """List all dynamically registered tools.

    Args:
        api_meta_tool: The APIMetaTool instance

    Returns:
        Result with information about all available tools
    """
    console.print(Rule("[bold blue]LISTING ALL AVAILABLE TOOLS[/bold blue]"))

    with console.status("[bold green]Fetching available tools...", spinner="dots"):
        try:
            result = await api_meta_tool.list_available_tools()

            tools = result.get("tools", [])
            if tools:
                table = Table(
                    title=f"Available Tools ({len(tools)})",
                    box=None,
                    highlight=True,
                    border_style="blue",
                )
                table.add_column("Tool Name", style="cyan")
                table.add_column("API Name", style="magenta")
                table.add_column("Method", style="green", justify="center")
                table.add_column("Path", style="blue")
                table.add_column("Summary", style="yellow")

                for tool in tools:
                    table.add_row(
                        tool.get("name", "N/A"),
                        tool.get("api_name", "N/A"),
                        tool.get("method", "N/A").upper(),
                        tool.get("path", "N/A"),
                        tool.get("summary", "No summary") or "No summary",
                    )

                console.print(Panel(table, border_style="green", expand=False))
            else:
                console.print("[yellow]No tools are currently registered.[/yellow]")

            return result
        except Exception as e:
            console.print(f"[bold red]Error listing tools:[/bold red] {str(e)}")
            return {}


async def refresh_api_demo(
    api_meta_tool: APIMetaTool, api_name: str = DEFAULT_API
) -> Dict[str, Any]:
    """Refresh an API to update its endpoints.

    Args:
        api_meta_tool: The APIMetaTool instance
        api_name: Name of the API to refresh

    Returns:
        Result of the refresh operation
    """
    console.print(Rule(f"[bold blue]REFRESHING API: {api_name.upper()}[/bold blue]"))

    console.print(f"[cyan]Refreshing API {api_name} to update endpoints...[/cyan]")

    with console.status(f"[bold green]Refreshing {api_name} API...", spinner="dots"):
        try:
            start_time = time.time()
            result = await api_meta_tool.refresh_api(api_name=api_name)
            processing_time = time.time() - start_time

            console.print(
                f"[bold green]✓ Success![/bold green] API refreshed in {processing_time:.2f}s"
            )

            # Display refresh summary
            console.print(
                Panel(
                    f"[bold]Tools Added:[/bold] {len(result.get('tools_added', []))}\n"
                    f"[bold]Tools Updated:[/bold] {len(result.get('tools_updated', []))}\n"
                    f"[bold]Tools Removed:[/bold] {len(result.get('tools_removed', []))}\n"
                    f"[bold]Total Tools:[/bold] {result.get('tools_count', 0)}",
                    title="Refresh Results",
                    border_style="green",
                    expand=False,
                )
            )

            # Display lists of added/removed tools if any
            added_tools = result.get("tools_added", [])
            if added_tools:
                console.print("[bold]Added Tools:[/bold]")
                for tool in added_tools:
                    console.print(f"  [green]+ {tool}[/green]")

            removed_tools = result.get("tools_removed", [])
            if removed_tools:
                console.print("[bold]Removed Tools:[/bold]")
                for tool in removed_tools:
                    console.print(f"  [red]- {tool}[/red]")

            return result
        except Exception as e:
            console.print(f"[bold red]Error refreshing API:[/bold red] {str(e)}")
            return {}


async def unregister_api_demo(
    api_meta_tool: APIMetaTool, api_name: str = DEFAULT_API
) -> Dict[str, Any]:
    """Unregister an API and all its tools.

    Args:
        api_meta_tool: The APIMetaTool instance
        api_name: Name of the API to unregister

    Returns:
        Result of the unregistration
    """
    console.print(Rule(f"[bold blue]UNREGISTERING API: {api_name.upper()}[/bold blue]"))

    with console.status(f"[bold green]Unregistering {api_name} API...", spinner="dots"):
        try:
            start_time = time.time()
            result = await api_meta_tool.unregister_api(api_name=api_name)
            processing_time = time.time() - start_time

            console.print(
                f"[bold green]✓ Success![/bold green] API unregistered in {processing_time:.2f}s"
            )

            # Display unregistration summary
            console.print(
                Panel(
                    f"[bold]API Name:[/bold] {result.get('api_name', 'N/A')}\n"
                    f"[bold]Tools Unregistered:[/bold] {len(result.get('tools_unregistered', []))}\n",
                    title="Unregistration Results",
                    border_style="green",
                    expand=False,
                )
            )

            return result
        except Exception as e:
            console.print(f"[bold red]Error unregistering API:[/bold red] {str(e)}")
            return {}


async def run_multi_api_demo(api_meta_tool: APIMetaTool):
    """Run a demonstration with multiple APIs registered simultaneously.

    Args:
        api_meta_tool: The APIMetaTool instance
    """
    console.print(Rule("[bold blue]MULTI-API DEMONSTRATION[/bold blue]"))

    console.print(
        Panel(
            "This demo shows how to work with multiple APIs registered simultaneously.\n"
            "We'll register two different APIs and interact with them.",
            title="Multiple APIs Demo",
            border_style="green",
            expand=False,
        )
    )

    # Register multiple APIs
    apis_to_register = ["petstore", "weather"]
    registered_apis = []

    for api_name in apis_to_register:
        console.print(f"\n[bold]Registering {api_name} API...[/bold]")
        result = await register_api_demo(api_meta_tool, api_name)
        if result:
            registered_apis.append(api_name)

    if len(registered_apis) < 2:
        console.print("[yellow]Not enough APIs registered for multi-API demo. Skipping...[/yellow]")
        return

    console.print("\n[bold]Now we have multiple APIs registered:[/bold]")

    # List all registered APIs
    await list_apis_demo(api_meta_tool)

    # List all available tools
    await list_tools_demo(api_meta_tool)

    # Call a tool from each API
    for api_name in registered_apis:
        console.print(f"\n[bold]Calling a tool from {api_name} API:[/bold]")
        await call_tool_demo(api_meta_tool, api_name)

    # Clean up: unregister all APIs
    for api_name in registered_apis:
        console.print(f"\n[bold]Cleaning up: Unregistering {api_name} API:[/bold]")
        await unregister_api_demo(api_meta_tool, api_name)


async def run_full_demo(api_meta_tool: APIMetaTool) -> None:
    """Run the complete demonstration sequence with proper progress tracking.

    Args:
        api_meta_tool: The APIMetaTool instance
    """
    console.print(Rule("[bold cyan]RUNNING FULL META API DEMONSTRATION[/bold cyan]"))

    steps_md = """
    ## Full Demonstration Steps
    
    1. **Register API** - Add a new API from OpenAPI spec
    2. **List APIs** - View all registered APIs
    3. **API Details** - Explore API information
    4. **Tool Details** - Get specific tool info
    5. **Call Tool** - Execute an API endpoint
    6. **Refresh API** - Update API definition
    7. **Multi-API Demo** - Work with multiple APIs
    8. **Cleanup** - Unregister APIs
    """

    console.print(Markdown(steps_md))

    # Wait for user confirmation
    continue_demo = Confirm.ask("\nReady to start the full demonstration?")
    if not continue_demo:
        console.print("[yellow]Demonstration cancelled.[/yellow]")
        return

    # Create list of demo steps for tracking progress
    demo_steps: List[str] = [
        "Register API",
        "List APIs",
        "API Details",
        "Tool Details",
        "Call Tool",
        "Refresh API",
        "Multi-API Demo",
        "Cleanup",
    ]

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=False
    ) as progress:
        overall_task = progress.add_task("[bold cyan]Running full demo...", total=len(demo_steps))

        try:
            # 1. Register the Petstore API
            progress.update(
                overall_task, description="[bold cyan]STEP 1: Register the Petstore API[/bold cyan]"
            )
            await register_api_demo(api_meta_tool, "petstore")
            progress.advance(overall_task)
            input("\nPress Enter to continue...")

            # Continue with other steps...
            # ... (rest of the function remains unchanged)

        except Exception as e:
            console.print(f"[bold red]Error during full demonstration:[/bold red] {str(e)}")


async def interactive_demo(api_meta_tool: APIMetaTool):
    """Run an interactive menu-driven demonstration.

    Args:
        api_meta_tool: The APIMetaTool instance
    """
    while True:
        console.clear()
        await show_intro()

        console.print("[bold cyan]META API TOOL MENU[/bold cyan]", justify="center")
        console.print("Select an action to demonstrate:", justify="center")
        console.print()

        # Menu options
        options = [
            ("Register an API", "Register a new API from its OpenAPI specification"),
            ("List Registered APIs", "List all currently registered APIs"),
            ("API Details", "Get detailed information about a registered API"),
            ("Tool Details", "Get detailed information about a specific tool"),
            ("Call a Tool", "Call a dynamically registered tool"),
            ("List All Tools", "List all available tools across all APIs"),
            ("Refresh an API", "Refresh an API to update its endpoints"),
            ("Unregister an API", "Unregister an API and all its tools"),
            ("Multi-API Demo", "Demonstrate using multiple APIs together"),
            ("Run Full Demo", "Run the complete demonstration sequence"),
            ("Exit", "Exit the demonstration"),
        ]

        # Display menu
        menu_table = Table(box=None, highlight=True, border_style=None)
        menu_table.add_column("Option", style="cyan", justify="right")
        menu_table.add_column("Description", style="white")

        for i, (option, description) in enumerate(options, 1):
            menu_table.add_row(f"{i}. {option}", description)

        console.print(menu_table)
        console.print()

        # Get user choice
        try:
            choice = Prompt.ask(
                "[bold green]Enter option number",
                choices=[str(i) for i in range(1, len(options) + 1)],
                default="1",
            )
            choice = int(choice)

            if choice == len(options):  # Exit option
                console.print("[yellow]Exiting demonstration. Goodbye![/yellow]")
                break

            # Clear screen for the selected demo
            console.clear()
            await show_intro()

            # Run the selected demo
            if choice == 1:  # Register an API
                api_choice = Prompt.ask(
                    "[bold green]Select an API to register", choices=["1", "2", "3"], default="1"
                )
                api_name = list(DEMO_APIS.keys())[int(api_choice) - 1]
                await register_api_demo(api_meta_tool, api_name)

            elif choice == 2:  # List Registered APIs
                await list_apis_demo(api_meta_tool)

            elif choice == 3:  # API Details
                apis = await list_apis_demo(api_meta_tool)
                api_names = list(apis.get("apis", {}).keys())

                if not api_names:
                    console.print(
                        "[yellow]No APIs are registered. Please register an API first.[/yellow]"
                    )
                else:
                    api_options = {str(i): name for i, name in enumerate(api_names, 1)}
                    api_choice = Prompt.ask(
                        "[bold green]Select an API", choices=list(api_options.keys()), default="1"
                    )
                    api_name = api_options[api_choice]
                    await get_api_details_demo(api_meta_tool, api_name)

            elif choice == 4:  # Tool Details
                apis = await list_apis_demo(api_meta_tool)
                api_names = list(apis.get("apis", {}).keys())

                if not api_names:
                    console.print(
                        "[yellow]No APIs are registered. Please register an API first.[/yellow]"
                    )
                else:
                    api_options = {str(i): name for i, name in enumerate(api_names, 1)}
                    api_choice = Prompt.ask(
                        "[bold green]Select an API", choices=list(api_options.keys()), default="1"
                    )
                    api_name = api_options[api_choice]
                    await get_tool_details_demo(api_meta_tool, api_name)

            elif choice == 5:  # Call a Tool
                apis = await list_apis_demo(api_meta_tool)
                api_names = list(apis.get("apis", {}).keys())

                if not api_names:
                    console.print(
                        "[yellow]No APIs are registered. Please register an API first.[/yellow]"
                    )
                else:
                    api_options = {str(i): name for i, name in enumerate(api_names, 1)}
                    api_choice = Prompt.ask(
                        "[bold green]Select an API", choices=list(api_options.keys()), default="1"
                    )
                    api_name = api_options[api_choice]
                    await call_tool_demo(api_meta_tool, api_name)

            elif choice == 6:  # List All Tools
                await list_tools_demo(api_meta_tool)

            elif choice == 7:  # Refresh an API
                apis = await list_apis_demo(api_meta_tool)
                api_names = list(apis.get("apis", {}).keys())

                if not api_names:
                    console.print(
                        "[yellow]No APIs are registered. Please register an API first.[/yellow]"
                    )
                else:
                    api_options = {str(i): name for i, name in enumerate(api_names, 1)}
                    api_choice = Prompt.ask(
                        "[bold green]Select an API", choices=list(api_options.keys()), default="1"
                    )
                    api_name = api_options[api_choice]
                    await refresh_api_demo(api_meta_tool, api_name)

            elif choice == 8:  # Unregister an API
                apis = await list_apis_demo(api_meta_tool)
                api_names = list(apis.get("apis", {}).keys())

                if not api_names:
                    console.print(
                        "[yellow]No APIs are registered. Please register an API first.[/yellow]"
                    )
                else:
                    api_options = {str(i): name for i, name in enumerate(api_names, 1)}
                    api_choice = Prompt.ask(
                        "[bold green]Select an API to unregister",
                        choices=list(api_options.keys()),
                        default="1",
                    )
                    api_name = api_options[api_choice]
                    await unregister_api_demo(api_meta_tool, api_name)

            elif choice == 9:  # Multi-API Demo
                await run_multi_api_demo(api_meta_tool)

            elif choice == 10:  # Run Full Demo
                await run_full_demo(api_meta_tool)

            # Wait for user to press Enter before returning to menu
            input("\nPress Enter to return to the menu...")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            input("\nPress Enter to return to the menu...")


async def main():
    """Main entry point for the demonstration."""
    try:
        # Set up the MCP client using create_app
        print("=== API Meta-Tool Demo ===")
        app = create_app()  # noqa: F841

        # Access the globally initialized Gateway instance and its api_meta_tool
        gateway_instance = ultimate_mcp_server.core._gateway_instance
        if not gateway_instance:
            raise RuntimeError("Gateway instance not initialized by create_app.")

        api_meta_tool = gateway_instance.api_meta_tool
        if not api_meta_tool:
            raise RuntimeError(
                "API Meta Tool instance not found on Gateway. Ensure it was registered."
            )

        # Run the interactive demo with the retrieved instance
        await interactive_demo(api_meta_tool)

    except KeyboardInterrupt:
        console.print("\n[yellow]Demonstration interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
    finally:
        console.print("\n[bold green]Meta API Tool Demonstration completed![/bold green]")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
