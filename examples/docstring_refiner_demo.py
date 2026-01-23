#!/usr/bin/env python
"""
Advanced Docstring Refiner Demo for Ultimate MCP Server.

This script demonstrates the autonomous documentation refinement tool that analyzes, tests, and improves
documentation (descriptions, schemas, examples) for MCP tools, enhancing their usability with LLM agents.
The demo showcases multiple refinement approaches, and visualization techniques, while providing
comprehensive performance metrics and cost analysis.

Features:
    - Single and multi-tool refinement demonstrations
    - Custom test generation strategy configuration
    - Provider fallbacks and model selection optimization
    - Visual diffs of documentation improvements
    - Cost estimation and optimization techniques
    - Schema-focused refinement capabilities
    - Model comparison and performance analysis
    - Practical testing with intentionally flawed tools
    - Adaptive refinement based on tool complexity

Command-line Arguments:
    --demo {all,single,multi,custom-testing,optimize,all-tools,schema-focus,practical,model-comparison}:
        Specific demo to run (default: all)

    --tool TOOL:
        Specify a specific tool to refine (bypasses automatic selection)

    --iterations N:
        Number of refinement iterations to run

    --model MODEL:
        Specify a model to use for refinement (e.g., gpt-4.1-mini, claude-3-5-haiku)

    --provider PROVIDER:
        Specify a provider to use for refinement (e.g., openai, anthropic)

    --visualize {minimal,standard,full}:
        Control visualization detail level (default: standard)

    --cost-limit FLOAT:
        Maximum cost limit in USD (default: 5.0)

    --output-dir DIR:
        Directory to save results

    --save-results:
        Save refinement results to files

    --verbose, -v:
        Increase output verbosity

    --create-flawed:
        Create flawed example tools for practical testing

Demo Modes:
    single:
        Demonstrates refining a single tool with detailed progress tracking
        and visualization of description, schema, and example improvements.

    multi:
        Demonstrates refining multiple tools simultaneously, showcasing parallel
        processing and cross-tool analysis of documentation patterns.

    custom-testing:
        Demonstrates advanced test generation strategies with fine-grained control
        over the types and quantities of test cases.

    optimize:
        Showcases cost optimization techniques for large-scale refinement,
        comparing standard and cost-optimized approaches.

    all-tools:
        Demonstrates the capability to refine all available tools in a single run,
        with resource management and prioritization features.

    schema-focus:
        Focuses specifically on schema improvements, with detailed visualization
        of JSON schema patches and validation improvements.

    practical:
        Creates and refines intentionally flawed example tools to demonstrate
        the system's ability to identify and fix common documentation issues.

    model-comparison:
        Compares the performance of different LLM models for refinement tasks,
        with detailed metrics on success rates, cost, and processing time.

Dependencies:
    - ultimate: Core framework for interfacing with LLMs and tools
    - rich: For beautiful console output and visualizations
    - asyncio: For asynchronous processing of refinement operations
    - Required API keys for providers (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

Usage Examples:
    # Run all demos with standard visualization
    python docstring_refiner_demo.py

    # Run just the single tool refinement demo with a specific tool
    python docstring_refiner_demo.py --demo single --tool generate_completion

    # Run the model comparison demo with full visualization and save results
    python docstring_refiner_demo.py --demo model-comparison --visualize full --save-results

    # Run the multi-tool demo with a specific model and cost limit
    python docstring_refiner_demo.py --demo multi --model gpt-4.1-mini --cost-limit 2.5

    # Create and test flawed example tools
    python docstring_refiner_demo.py --demo practical --create-flawed

Return Values:
    The script returns exit code 0 on successful completion, or exit code 1 if
    critical errors occur during execution.

Methods:
    The script contains various helper functions and demo methods:

    setup_gateway_and_tools(): Initializes the Gateway and ensures required tools are available

    get_suitable_tools(): Finds appropriate tools for demonstrations based on complexity

    display_refinement_progress(): Callback for tracking refinement progress events

    create_text_diff(), create_side_by_side_diff(): Generate visual diffs of documentation changes

    display_refinement_result(): Formats and displays refinement results with appropriate detail level

    create_flawed_example_tools(): Creates example tools with intentional documentation flaws

    Demo functions (demo_*): Implement specific demonstration scenarios

Implementation Notes:
    - The script uses the global MCP instance from the Gateway for all tool operations
    - Refinement operations are tracked through a CostTracker instance for budget management
    - All demonstrations include graceful fallbacks for providers and models
    - Progress updates are displayed using Rich's Progress components
    - Results can be saved to files for later analysis or integration

Author:
    Ultimate MCP Server Team

Version:
    1.0.0
"""

import argparse
import asyncio
import datetime
import difflib
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Rich for beautiful console output
from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

# Project imports
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.tools.base import with_error_handling
from ultimate_mcp_server.tools.docstring_refiner import (
    RefinementProgressEvent,
)
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger
logger = get_logger("example.docstring_refiner")

# Create a separate console for detailed output
detail_console = Console(highlight=False)

# Global MCP instance (will be populated from Gateway)
mcp = None

# Global settings that can be modified by command line args
SETTINGS = {
    "output_dir": None,
    "visualization_level": "standard",  # "minimal", "standard", "full"
    "cost_limit": 5.0,  # USD
    "preferred_providers": [Provider.OPENAI.value, Provider.ANTHROPIC.value, Provider.GEMINI.value],
    "fallback_providers": [Provider.DEEPSEEK.value, Provider.GROK.value],
    "save_results": False,
    "verbose": False,
}


def parse_arguments():
    """Parse command line arguments for the demo."""
    parser = argparse.ArgumentParser(
        description="Advanced Docstring Refiner Demo for Ultimate MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Available demos:
  all                   - Run all demos (default)
  single                - Single tool refinement
  multi                 - Multi-tool refinement
  custom-testing        - Custom test generation strategies
  optimize              - Cost optimization techniques
  all-tools             - Refine all available tools
  schema-focus          - Focus on schema improvements
  practical             - Practical testing with flawed tools
  model-comparison      - Compare different LLM models for refinement
""",
    )

    # Demo selection
    parser.add_argument(
        "--demo",
        default="all",
        choices=[
            "all",
            "single",
            "multi",
            "custom-testing",
            "optimize",
            "all-tools",
            "schema-focus",
            "practical",
            "model-comparison",
        ],
        help="Specific demo to run (default: all)",
    )

    # Tool selection
    parser.add_argument(
        "--tool", help="Specify a specific tool to refine (bypasses automatic selection)"
    )

    # Iteration control
    parser.add_argument(
        "--iterations", type=int, default=None, help="Number of refinement iterations to run"
    )

    # Model specification
    parser.add_argument(
        "--model",
        default=None,
        help="Specify a model to use for refinement (e.g., gpt-4.1-mini, claude-3-5-haiku)",
    )

    # Provider specification
    parser.add_argument(
        "--provider",
        default=None,
        help=f"Specify a provider to use for refinement (e.g., {Provider.OPENAI.value}, {Provider.ANTHROPIC.value})",
    )

    # Visualization options
    parser.add_argument(
        "--visualize",
        choices=["minimal", "standard", "full"],
        default="standard",
        help="Control visualization detail level",
    )

    # Cost limit
    parser.add_argument("--cost-limit", type=float, default=5.0, help="Maximum cost limit in USD")

    # Output directory
    parser.add_argument("--output-dir", help="Directory to save results")

    # Save results
    parser.add_argument(
        "--save-results", action="store_true", help="Save refinement results to files"
    )

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")

    # Create flawed tools for testing
    parser.add_argument(
        "--create-flawed",
        action="store_true",
        help="Create flawed example tools for practical testing",
    )

    args = parser.parse_args()

    # Update settings
    SETTINGS["visualization_level"] = args.visualize
    SETTINGS["cost_limit"] = args.cost_limit
    SETTINGS["save_results"] = args.save_results
    SETTINGS["verbose"] = args.verbose

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        SETTINGS["output_dir"] = output_dir

    return args


async def setup_gateway_and_tools(create_flawed_tools=False):
    """Set up the gateway and ensure docstring refiner tool is available."""
    global mcp
    logger.debug("Initializing Gateway for docstring refiner demo...")
    logger.info("Initializing Gateway for docstring refiner demo...", emoji_key="start")

    # Create Gateway instance with all tools
    logger.debug("Creating Gateway instance with all tools")
    gateway = Gateway(
        "docstring-refiner-demo", register_tools=True
    )  # Register all tools, not just minimal tools

    # Initialize providers (needed for the tool to function)
    try:
        logger.debug("Initializing providers...")
        await gateway._initialize_providers()
        logger.success("Successfully initialized providers", emoji_key="success")
        logger.debug("Successfully initialized providers")
    except Exception as e:
        logger.error(f"Error initializing providers: {e}", emoji_key="error", exc_info=True)
        logger.exception("Error initializing providers")
        console.print(
            Panel(
                f"Error initializing providers: {escape(str(e))}\n\n"
                "Check that your API keys are set correctly in environment variables:\n"
                "- OPENAI_API_KEY\n"
                "- ANTHROPIC_API_KEY\n"
                "- GEMINI_API_KEY\n",
                title="❌ Provider Initialization Failed",
                border_style="red",
                expand=False,
            )
        )
        # Continue anyway, as some providers might still work

    # Store the MCP server instance
    mcp = gateway.mcp
    logger.debug("Stored MCP server instance")

    # Display available providers with available models
    logger.debug("Getting provider information")
    provider_tree = Tree("[bold cyan]Available Providers & Models[/bold cyan]")
    provider_info = []

    for provider_name, provider in gateway.providers.items():
        if provider:
            try:
                models = await provider.list_models()
                provider_branch = provider_tree.add(f"[yellow]{provider_name}[/yellow]")

                # Group models by category/capability
                categorized_models = {}
                for model in models:
                    model_id = model.get("id", "unknown")
                    if "4" in model_id:
                        category = "GPT-4 Family"
                    elif "3" in model_id:
                        category = "GPT-3 Family"
                    elif "claude" in model_id.lower():
                        category = "Claude Family"
                    elif "gemini" in model_id.lower():
                        category = "Gemini Family"
                    elif "deepseek" in model_id.lower():
                        category = "DeepSeek Family"
                    else:
                        category = "Other Models"

                    if category not in categorized_models:
                        categorized_models[category] = []
                    categorized_models[category].append(model_id)

                # Add models to the tree by category
                for category, model_list in categorized_models.items():
                    category_branch = provider_branch.add(f"[cyan]{category}[/cyan]")
                    for model_id in sorted(model_list):
                        category_branch.add(f"[green]{model_id}[/green]")

                # Get default model for provider info
                default_model = provider.get_default_model()
                provider_info.append(f"{provider_name} (default: {default_model})")
            except Exception as e:
                logger.warning(
                    f"Could not get models for {provider_name}: {e}", emoji_key="warning"
                )
                logger.warning(f"Could not get models for {provider_name}: {e}")
                provider_info.append(f"{provider_name} (models unavailable)")
                provider_branch = provider_tree.add(f"[yellow]{provider_name}[/yellow]")
                provider_branch.add(f"[red]Error listing models: {escape(str(e))}[/red]")

    # Display provider info based on visualization level
    if SETTINGS["visualization_level"] == "full":
        console.print(Panel(provider_tree, border_style="dim cyan", padding=(1, 2)))
    else:
        console.print(
            Panel(
                f"Available providers: {', '.join(provider_info)}",
                title="Provider Configuration",
                border_style="cyan",
                expand=False,
            )
        )

    # Verify the docstring_refiner tool is available
    logger.debug("Checking for available tools")
    tool_list = await mcp.list_tools()
    available_tools = [t.name for t in tool_list]
    logger.debug(f"Available tools before registration: {available_tools}")

    # Display all available tools
    tool_tree = Tree("[bold cyan]Available MCP Tools[/bold cyan]")

    # Group tools by namespace for better visualization
    tool_namespaces = {}
    for tool_name in available_tools:
        if ":" in tool_name:
            namespace, name = tool_name.split(":", 1)
            if namespace not in tool_namespaces:
                tool_namespaces[namespace] = []
            tool_namespaces[namespace].append(name)
        else:
            if "root" not in tool_namespaces:
                tool_namespaces["root"] = []
            tool_namespaces["root"].append(tool_name)

    # Add tools to tree with proper grouping
    for namespace, tools in tool_namespaces.items():
        if namespace == "root":
            for tool in sorted(tools):
                tool_tree.add(f"[green]{tool}[/green]")
        else:
            ns_branch = tool_tree.add(f"[yellow]{namespace}[/yellow]")
            for tool in sorted(tools):
                ns_branch.add(f"[green]{tool}[/green]")

    # Display tool info based on visualization level
    if SETTINGS["visualization_level"] in ["standard", "full"]:
        console.print(Panel(tool_tree, border_style="dim cyan", padding=(1, 2)))
    else:
        console.print(f"[cyan]Tools available:[/cyan] {len(available_tools)}")

    # Check if refine_tool_documentation is available
    if "refine_tool_documentation" in available_tools:
        logger.success("refine_tool_documentation tool available.", emoji_key="success")
    else:
        logger.warning(
            "refine_tool_documentation tool not found in available tools list.", emoji_key="warning"
        )
        console.print(
            Panel(
                "The refine_tool_documentation tool is not registered automatically.\n"
                "This demo will attempt to register it manually as a fallback.",
                title="⚠️ Tool Availability Notice",
                border_style="yellow",
            )
        )

        # Manually register the refine_tool_documentation tool as a fallback
        # Note: This should no longer be necessary since the tool is now included in STANDALONE_TOOL_FUNCTIONS
        # in ultimate/tools/__init__.py, but we keep it as a fallback in case of issues
        try:
            print("Attempting to manually register refine_tool_documentation tool as fallback...")
            from ultimate_mcp_server.tools.docstring_refiner import refine_tool_documentation

            print("Imported refine_tool_documentation successfully")

            # Create a simplified wrapper to avoid Pydantic validation issues
            @with_error_handling
            async def docstring_refiner_wrapper(
                tool_names=None, refine_all_available=False, max_iterations=1, ctx=None
            ):
                """
                Refine the documentation of MCP tools.

                Args:
                    tool_names: List of tools to refine, or None to use refine_all_available
                    refine_all_available: Whether to refine all available tools
                    max_iterations: Maximum number of refinement iterations
                    ctx: MCP context

                Returns:
                    Refinement results
                """
                print(
                    f"Wrapper called with tool_names={tool_names}, refine_all_available={refine_all_available}"
                )
                # Simply pass through to the actual implementation
                return await refine_tool_documentation(
                    tool_names=tool_names,
                    refine_all_available=refine_all_available,
                    max_iterations=max_iterations,
                    ctx=ctx,
                )

            # Register our simplified wrapper instead
            mcp.tool(name="refine_tool_documentation")(docstring_refiner_wrapper)
            print("Registered fallback wrapper tool successfully")
            logger.success(
                "Successfully registered fallback wrapper for refine_tool_documentation tool",
                emoji_key="success",
            )
        except Exception as e:
            logger.error(
                f"Failed to register fallback refine_tool_documentation tool: {e}",
                emoji_key="error",
                exc_info=True,
            )
            print(f"Error registering fallback tool: {type(e).__name__}: {str(e)}")
            import traceback

            print("Stack trace:")
            traceback.print_exc()
            console.print(
                Panel(
                    f"Error registering the fallback refine_tool_documentation tool: {escape(str(e))}\n\n"
                    "This demo requires the docstring_refiner tool to be properly registered.",
                    title="❌ Registration Failed",
                    border_style="red",
                    expand=False,
                )
            )
            console.print(
                Panel(
                    "This demo requires the docstring_refiner tool to be properly registered.\n"
                    "Check that you have the correct version of the Ultimate MCP Server and dependencies installed.",
                    title="⚠️ Demo Requirements Not Met",
                    border_style="red",
                    expand=False,
                )
            )
            return gateway

    # Create flawed example tools if requested
    if create_flawed_tools:
        created_tools = await create_flawed_example_tools(mcp)
        if created_tools:
            console.print(
                Panel(
                    f"Created {len(created_tools)} flawed example tools for testing:\n"
                    + "\n".join([f"- [cyan]{name}[/cyan]" for name in created_tools]),
                    title="🛠️ Flawed Tools Created",
                    border_style="yellow",
                    expand=False,
                )
            )

    return gateway


async def create_flawed_example_tools(mcp_instance):
    """Create flawed example tools for demonstration purposes."""
    created_tools = []

    try:
        # Create a temporary directory to store any needed files
        temp_dir = tempfile.mkdtemp(prefix="docstring_refiner_flawed_tools_")
        logger.info(f"Created temporary directory for flawed tools: {temp_dir}", emoji_key="setup")

        # Define several flawed tools with various issues

        # Tool 1: Ambiguous Description
        @mcp_instance.tool()
        async def flawed_process_text(
            text: str, mode: str = "simple", include_metadata: bool = False
        ):
            """Process the given text.

            This tool does processing on text.

            Args:
                text: Text to process
                mode: Processing mode (simple, advanced, expert)
                include_metadata: Whether to include metadata in result
            """
            # Actual implementation doesn't matter for the demo
            result = {"processed": text[::-1]}  # Just reverse the text
            if include_metadata:
                result["metadata"] = {"length": len(text), "mode": mode}
            return result

        created_tools.append("flawed_process_text")

        # Tool 2: Missing Parameter Descriptions
        @mcp_instance.tool()
        async def flawed_scrape_website(url, depth=1, extract_links=True, timeout=30.0):
            """Website scraper tool.

            Extracts content from websites.
            """
            # Simulate scraping
            return {
                "title": f"Page at {url}",
                "content": f"Scraped content with depth {depth}",
                "links": ["https://example.com/1", "https://example.com/2"]
                if extract_links
                else [],
            }

        created_tools.append("flawed_scrape_website")

        # Tool 3: Confusing Schema & Inconsistent Description
        @mcp_instance.tool()
        async def flawed_data_processor(config, inputs, format="json"):
            """Processes data.

            The analyzer takes configuration and processes input data.
            The system allows different engine versions and parameters.
            """
            # Just return dummy data
            return {
                "outputs": [f"Processed: {i}" for i in inputs],
                "engine_used": config.get("engine", "v1"),
                "format": format,
            }

        created_tools.append("flawed_data_processor")

        # Tool 4: Misleading Examples in Description but no schema examples
        @mcp_instance.tool()
        async def flawed_product_search(query, filters=None, sort="rating", page=1, per_page=20):
            """Search for products in the database.

            Example usage:
            ```
            search_products("laptop", {"category": "electronics", "min_price": 500}, sort_by="newest")
            ```

            The search function allows querying for items along with filtering and sorting options.
            """
            # Return dummy results
            return {
                "results": [
                    {"id": i, "name": f"{query} product {i}", "price": random.randint(10, 1000)}
                    for i in range(1, 6)
                ],
                "total": 243,
                "page": page,
                "per_page": per_page,
            }

        created_tools.append("flawed_product_search")

        # Tool 5: Schema with type issues (number vs integer conflicts)
        @mcp_instance.tool()
        async def flawed_calculator(values, operation, precision=2, scale_factor=1.0):
            """Statistical calculator.

            Calculate statistics on a set of values. The operation determines which
            statistic to calculate. Valid operations are:

            - sum: Calculate the sum of all values
            - average: Calculate the mean of the values
            - max: Find the maximum value
            - min: Find the minimum value

            The precision parameter must be an integer between 0 and 10.

            After calculation, the result is multiplied by the scale_factor.
            """
            # Perform the calculation
            if operation == "sum":
                result = sum(values)
            elif operation == "average":
                result = sum(values) / len(values) if values else 0
            elif operation == "max":
                result = max(values) if values else None
            elif operation == "min":
                result = min(values) if values else None
            else:
                result = None

            # Apply scale and precision
            if result is not None:
                result = round(result * scale_factor, precision)

            return {"result": result}

        created_tools.append("flawed_calculator")

        logger.success(
            f"Successfully created {len(created_tools)} flawed example tools", emoji_key="success"
        )
        return created_tools

    except Exception as e:
        logger.error(f"Error creating flawed example tools: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error creating flawed example tools:[/bold red] {escape(str(e))}")
        return []


async def display_refinement_progress(event: RefinementProgressEvent):
    """Handle progress events from the refinement process."""
    # Create a formatted message based on the event type
    if event.stage == "starting_iteration":
        message = f"[bold cyan]Starting iteration {event.iteration}/{event.total_iterations} for {event.tool_name}[/bold cyan]"
    elif event.stage == "agent_simulation":
        message = f"[blue]Simulating agent usage for {event.tool_name}...[/blue]"
    elif event.stage == "test_generation":
        message = f"[blue]Generating test cases for {event.tool_name}...[/blue]"
    elif event.stage == "test_execution_start":
        message = f"[blue]Executing tests for {event.tool_name}...[/blue]"
    elif event.stage == "test_execution_progress":
        message = f"[blue]Test execution progress: {event.progress_pct:.1f}%[/blue]"
    elif event.stage == "test_execution_end":
        success_rate = event.details.get("success_rate") if event.details else None
        if success_rate is not None:
            message = f"[green]Tests completed for {event.tool_name} - Success rate: {success_rate:.1%}[/green]"
        else:
            message = f"[green]Tests completed for {event.tool_name}[/green]"
    elif event.stage == "analysis_start":
        message = f"[blue]Analyzing results for {event.tool_name}...[/blue]"
    elif event.stage == "analysis_end":
        message = f"[green]Analysis completed for {event.tool_name}[/green]"
    elif event.stage == "schema_patching":
        message = f"[blue]Applying schema patches for {event.tool_name}...[/blue]"
    elif event.stage == "winnowing":
        message = f"[blue]Optimizing documentation for {event.tool_name}...[/blue]"
    elif event.stage == "iteration_complete":
        message = (
            f"[bold green]Iteration {event.iteration} complete for {event.tool_name}[/bold green]"
        )
    elif event.stage == "tool_complete":
        message = f"[bold magenta]Refinement complete for {event.tool_name}[/bold magenta]"
    elif event.stage == "error":
        message = (
            f"[bold red]Error during refinement for {event.tool_name}: {event.message}[/bold red]"
        )
    else:
        message = f"[dim]{event.message}[/dim]"

    # Print the message
    detail_console.print(message)

    # Print additional details if in verbose mode
    if SETTINGS["verbose"] and event.details:
        try:
            detail_console.print(
                f"[dim cyan]Details: {json.dumps(event.details, default=str)}[/dim cyan]"
            )
        except Exception:
            detail_console.print(f"[dim cyan]Details: {event.details}[/dim cyan]")

    # Return True to confirm the callback was processed
    return True


def create_text_diff(original: str, improved: str) -> Panel:
    """Create a colorized diff between original and improved text."""
    diff = difflib.unified_diff(
        original.splitlines(),
        improved.splitlines(),
        lineterm="",
        n=3,  # Context lines
    )

    # Convert diff to rich text with colors
    rich_diff = []
    for line in diff:
        if line.startswith("+"):
            rich_diff.append(f"[green]{escape(line)}[/green]")
        elif line.startswith("-"):
            rich_diff.append(f"[red]{escape(line)}[/red]")
        elif line.startswith("@@"):
            rich_diff.append(f"[cyan]{escape(line)}[/cyan]")
        else:
            rich_diff.append(escape(line))

    # Return as panel
    if rich_diff:
        diff_panel = Panel(
            "\n".join(rich_diff),
            title="Documentation Changes (Diff)",
            border_style="yellow",
            expand=False,
        )
        return diff_panel
    else:
        return Panel(
            "[dim italic]No differences found[/dim italic]",
            title="Documentation Changes (Diff)",
            border_style="dim",
            expand=False,
        )


def create_side_by_side_diff(
    original: str, improved: str, title: str = "Documentation Comparison"
) -> Panel:
    """Create a side-by-side comparison of original and improved text."""
    # Wrap in panels with highlighting
    original_panel = Panel(escape(original), title="Original", border_style="dim red", expand=True)

    improved_panel = Panel(escape(improved), title="Improved", border_style="green", expand=True)

    # Create side-by-side group
    comparison = Group(Rule("Before / After"), Group(original_panel, improved_panel))

    return Panel(comparison, title=title, border_style="cyan", expand=False)


def display_refinement_result(
    result: Dict,
    console: Console = console,
    visualization_level: str = "standard",
    save_to_file: bool = False,
    output_dir: Optional[Path] = None,
):
    """Display the results of the docstring refinement process."""
    console.print(Rule("[bold green]Refinement Results[/bold green]", style="green"))

    # Summary statistics
    stats_table = Table(
        title="[bold]Summary Statistics[/bold]", box=box.ROUNDED, show_header=False, expand=False
    )
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="white")
    stats_table.add_row("Total Tools Refined", str(len(result.get("refined_tools", []))))
    stats_table.add_row("Total Iterations", str(result.get("total_iterations_run", 0)))
    stats_table.add_row("Total Tests Executed", str(result.get("total_test_calls_attempted", 0)))
    stats_table.add_row("Total Test Failures", str(result.get("total_test_calls_failed", 0)))
    stats_table.add_row(
        "Total Validation Failures", str(result.get("total_schema_validation_failures", 0))
    )
    stats_table.add_row("Total Processing Time", f"{result.get('total_processing_time', 0.0):.2f}s")
    stats_table.add_row("Total Cost", f"${result.get('total_refinement_cost', 0.0):.6f}")
    console.print(stats_table)

    # Save results to file if requested
    if save_to_file and output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"refinement_results_{timestamp}.json"
        try:
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"[green]Results saved to:[/green] {result_file}")
        except Exception as e:
            console.print(f"[red]Error saving results to file:[/red] {e}")

    # Tools refined
    refined_tools = result.get("refined_tools", [])
    if refined_tools:
        console.print("\n[bold]Tools Refined:[/bold]")

        # Results tallying
        total_description_improvements = 0
        total_schema_improvements = 0
        total_example_improvements = 0
        flaw_categories_observed = {}

        for i, tool in enumerate(refined_tools):
            tool_name = tool.get("tool_name", "Unknown tool")
            initial_success_rate = tool.get("initial_success_rate", 0.0)
            final_success_rate = tool.get("final_success_rate", 0.0)
            improvement_factor = tool.get("improvement_factor", 0.0)

            # Decide on panel color based on improvement
            if improvement_factor > 0.5:
                border_style = "green"
            elif improvement_factor > 0:
                border_style = "blue"
            else:
                border_style = "yellow"

            # Create a panel for each tool
            success_change = (final_success_rate - initial_success_rate) * 100
            success_change_str = (
                f"[green]+{success_change:.1f}%[/green]"
                if success_change > 0
                else f"[red]{success_change:.1f}%[/red]"
                if success_change < 0
                else "[yellow]No change[/yellow]"
            )

            tool_panel_content = [
                f"Initial Success Rate: [yellow]{initial_success_rate:.1%}[/yellow]",
                f"Final Success Rate: [green]{final_success_rate:.1%}[/green]",
                f"Change: {success_change_str}",
                f"Improvement Factor: [cyan]{improvement_factor:.2f}x[/cyan]",
            ]

            console.print(
                Panel(
                    Group(*tool_panel_content),
                    title=f"[bold]{i + 1}. {tool_name}[/bold]",
                    border_style=border_style,
                    expand=False,
                )
            )

            # Display the final proposed changes
            final_changes = tool.get("final_proposed_changes", {})
            iterations = tool.get("iterations", [])

            if final_changes:
                # Check if description was improved
                original_desc = None
                for iter_data in iterations:
                    if iter_data.get("iteration") == 1:
                        # Get the original description from the first iteration
                        original_desc = iter_data.get("documentation_used", {}).get(
                            "description", ""
                        )
                        break

                final_desc = final_changes.get("description", "")

                # Count this as an improvement if descriptions differ
                if original_desc and final_desc and original_desc != final_desc:
                    total_description_improvements += 1

                    # Display description changes based on visualization level
                    if visualization_level in ["standard", "full"]:
                        console.print("[bold cyan]Description Changes:[/bold cyan]")

                        if visualization_level == "full":
                            # Show diff view for detailed visualization
                            console.print(create_text_diff(original_desc, final_desc))

                        # Show side-by-side comparison
                        console.print(
                            create_side_by_side_diff(
                                original_desc, final_desc, title="Description Comparison"
                            )
                        )

                # Display schema patches if any
                schema_patches = tool.get("final_proposed_schema_patches", [])
                if schema_patches:
                    total_schema_improvements += 1

                    if visualization_level in ["standard", "full"]:
                        console.print("[bold cyan]Schema Patches Applied:[/bold cyan]")
                        console.print(
                            Panel(
                                Syntax(
                                    json.dumps(schema_patches, indent=2),
                                    "json",
                                    theme="default",
                                    line_numbers=False,
                                ),
                                title="JSON Patch Operations",
                                border_style="magenta",
                                expand=False,
                            )
                        )

                # Display examples
                examples = final_changes.get("examples", [])
                if examples:
                    total_example_improvements += len(examples)

                    if visualization_level in ["standard", "full"]:
                        console.print("[bold cyan]Generated Examples:[/bold cyan]")
                        examples_to_show = (
                            examples if visualization_level == "full" else examples[:3]
                        )

                        for j, example in enumerate(examples_to_show):
                            args = example.get("args", {})
                            comment = example.get("comment", "No description")
                            addresses_failure = example.get("addresses_failure_pattern", "")

                            # Add failure pattern as subtitle if present
                            subtitle = (
                                f"Addresses: {addresses_failure}" if addresses_failure else None
                            )

                            console.print(
                                Panel(
                                    Syntax(
                                        json.dumps(args, indent=2),
                                        "json",
                                        theme="default",
                                        line_numbers=False,
                                    ),
                                    title=f"Example {j + 1}: {comment}",
                                    subtitle=subtitle,
                                    border_style="dim green",
                                    expand=False,
                                )
                            )

                        if len(examples) > 3 and visualization_level == "standard":
                            console.print(f"[dim]...and {len(examples) - 3} more examples[/dim]")

            # Collect flaw categories if available
            for iter_data in iterations:
                analysis = iter_data.get("analysis", {})
                if analysis:
                    flaws = analysis.get("identified_flaw_categories", [])
                    for flaw in flaws:
                        if flaw not in flaw_categories_observed:
                            flaw_categories_observed[flaw] = 0
                        flaw_categories_observed[flaw] += 1

            console.print()  # Add spacing between tools

        # Display improvement summary
        console.print(Rule("[bold blue]Improvement Summary[/bold blue]", style="blue"))

        improvement_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        improvement_table.add_column("Improvement Type", style="blue")
        improvement_table.add_column("Count", style="cyan")
        improvement_table.add_column("Details", style="white")

        improvement_table.add_row(
            "Description Improvements",
            str(total_description_improvements),
            f"{total_description_improvements} of {len(refined_tools)} tools ({total_description_improvements / len(refined_tools) * 100:.0f}%)",
        )
        improvement_table.add_row(
            "Schema Improvements",
            str(total_schema_improvements),
            f"{total_schema_improvements} of {len(refined_tools)} tools ({total_schema_improvements / len(refined_tools) * 100:.0f}%)",
        )
        improvement_table.add_row(
            "Example Additions",
            str(total_example_improvements),
            f"Average {total_example_improvements / len(refined_tools):.1f} examples per tool",
        )

        console.print(improvement_table)

        # Display flaw categories if any were observed
        if flaw_categories_observed and visualization_level in ["standard", "full"]:
            console.print("\n[bold cyan]Documentation Flaws Identified:[/bold cyan]")

            flaws_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
            flaws_table.add_column("Flaw Category", style="magenta")
            flaws_table.add_column("Occurrences", style="cyan")
            flaws_table.add_column("Description", style="white")

            # Map flaw categories to descriptions
            flaw_descriptions = {
                "MISSING_DESCRIPTION": "Documentation is missing key information",
                "AMBIGUOUS_DESCRIPTION": "Description is unclear or can be interpreted in multiple ways",
                "INCORRECT_DESCRIPTION": "Description contains incorrect information",
                "MISSING_SCHEMA_CONSTRAINT": "Schema is missing important constraints",
                "INCORRECT_SCHEMA_CONSTRAINT": "Schema contains incorrect constraints",
                "OVERLY_RESTRICTIVE_SCHEMA": "Schema is unnecessarily restrictive",
                "TYPE_CONFUSION": "Parameter types are inconsistent or unclear",
                "MISSING_EXAMPLE": "Documentation lacks necessary examples",
                "MISLEADING_EXAMPLE": "Examples provided are incorrect or misleading",
                "INCOMPLETE_EXAMPLE": "Examples are present but insufficient",
                "PARAMETER_DEPENDENCY_UNCLEAR": "Dependencies between parameters are not explained",
                "CONFLICTING_CONSTRAINTS": "Schema contains contradictory constraints",
                "AGENT_FORMULATION_ERROR": "Documentation hinders LLM agent's ability to use the tool",
                "SCHEMA_PREVALIDATION_FAILURE": "Schema validation issues",
                "TOOL_EXECUTION_ERROR": "Issues with tool execution",
                "UNKNOWN": "Unspecified documentation issue",
            }

            # Sort flaws by occurrence count
            sorted_flaws = sorted(
                flaw_categories_observed.items(), key=lambda x: x[1], reverse=True
            )

            for flaw, count in sorted_flaws:
                flaws_table.add_row(
                    flaw, str(count), flaw_descriptions.get(flaw, "No description available")
                )

            console.print(flaws_table)

    # Error reporting
    errors = result.get("errors_during_refinement_process", [])
    if errors:
        console.print("[bold red]Errors During Refinement:[/bold red]")
        for error in errors:
            console.print(f"- [red]{escape(error)}[/red]")


async def get_suitable_tools(
    mcp_instance,
    count: int = 1,
    complexity: str = "medium",
    exclude_tools: Optional[List[str]] = None,
) -> List[str]:
    """
    Find suitable tools for refinement based on complexity.

    Args:
        mcp_instance: The MCP server instance
        count: Number of tools to return
        complexity: Desired complexity level ("simple", "medium", "complex")
        exclude_tools: List of tool names to exclude

    Returns:
        List of suitable tool names
    """
    exclude_tools = exclude_tools or []

    # Get all available tools
    tool_list = await mcp_instance.list_tools()

    # Filter out excluded tools and refine_tool_documentation itself
    available_tools = [
        t.name
        for t in tool_list
        if t.name not in exclude_tools and t.name != "refine_tool_documentation"
    ]

    if not available_tools:
        return []

    # Define complexity criteria based on schema properties
    if complexity == "simple":
        # Simple tools have few required parameters and a flat schema
        preferred_tools = []
        for tool_name in available_tools:
            try:
                tool_def = next((t for t in tool_list if t.name == tool_name), None)
                if not tool_def:
                    continue

                input_schema = getattr(tool_def, "inputSchema", {})
                if not input_schema:
                    continue

                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])

                # Simple tools have few properties and required fields
                if len(properties) <= 3 and len(required) <= 1:
                    # Check for nested objects which would increase complexity
                    has_nested = any(
                        isinstance(prop, dict) and prop.get("type") == "object"
                        for prop in properties.values()
                    )

                    if not has_nested:
                        preferred_tools.append(tool_name)
            except Exception:
                continue

    elif complexity == "complex":
        # Complex tools have deep nested structures and many required parameters
        preferred_tools = []
        for tool_name in available_tools:
            try:
                tool_def = next((t for t in tool_list if t.name == tool_name), None)
                if not tool_def:
                    continue

                input_schema = getattr(tool_def, "inputSchema", {})
                if not input_schema:
                    continue

                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])

                # Complex tools have many properties or required fields
                if len(properties) >= 5 or len(required) >= 3:
                    # Check for nested objects which would increase complexity
                    has_nested = any(
                        isinstance(prop, dict) and prop.get("type") == "object"
                        for prop in properties.values()
                    )

                    if has_nested:
                        preferred_tools.append(tool_name)
            except Exception:
                continue

    else:  # medium complexity (default)
        # Medium tools are somewhere in between
        preferred_tools = []
        for tool_name in available_tools:
            try:
                tool_def = next((t for t in tool_list if t.name == tool_name), None)
                if not tool_def:
                    continue

                input_schema = getattr(tool_def, "inputSchema", {})
                if not input_schema:
                    continue

                properties = input_schema.get("properties", {})

                # Medium tools have a moderate number of properties
                if 3 <= len(properties) <= 6:
                    preferred_tools.append(tool_name)
            except Exception:
                continue

    # If we couldn't find tools matching the complexity criteria, fall back to any available tool
    if not preferred_tools:
        preferred_tools = available_tools

    # Prioritize tools without namespaces (i.e., not "namespace:tool_name")
    prioritized_tools = [t for t in preferred_tools if ":" not in t]

    # If we still need more tools and have prioritized all we could, add namespace tools
    if len(prioritized_tools) < count:
        namespace_tools = [t for t in preferred_tools if ":" in t]
        prioritized_tools.extend(namespace_tools)

    # Return the requested number of tools (or fewer if not enough are available)
    return prioritized_tools[: min(count, len(prioritized_tools))]


async def demo_single_tool_refinement(
    gateway: Gateway,
    tracker: CostTracker,
    target_tool: Optional[str] = None,
    refinement_provider: Optional[str] = None,
    refinement_model: Optional[str] = None,
    max_iterations: Optional[int] = None,
):
    """Demonstrate refining documentation for a single tool."""
    console.print(Rule("[bold cyan]Single Tool Refinement[/bold cyan]", style="cyan"))

    # Use specified tool or find a suitable one
    selected_tool = None
    if target_tool:
        # Check if specified tool exists
        tool_list = await gateway.mcp.list_tools()
        available_tools = [t.name for t in tool_list]

        if target_tool in available_tools:
            selected_tool = target_tool
        else:
            logger.warning(f"Specified tool '{target_tool}' not found", emoji_key="warning")
            console.print(
                f"[yellow]Warning:[/yellow] Specified tool '{target_tool}' not found. Selecting automatically."
            )

    # Auto-select if needed
    if not selected_tool:
        suitable_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="medium")

        if suitable_tools:
            selected_tool = suitable_tools[0]
        else:
            logger.error("No suitable tools found for refinement demo", emoji_key="error")
            console.print(
                "[bold red]Error:[/bold red] No suitable tools found for refinement demo."
            )
            return

    console.print(f"Selected tool for refinement: [cyan]{selected_tool}[/cyan]")

    # Determine provider and model
    provider = refinement_provider or Provider.OPENAI.value

    # Find best available model if not specified
    if not refinement_model:
        try:
            if provider == Provider.OPENAI.value:
                model = "gpt-4.1"  # Prefer this for best results
                # Check if model is available
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    models = await provider_instance.list_models()
                    model_ids = [m.get("id") for m in models]
                    if model not in model_ids:
                        model = "gpt-4.1-mini"  # Fall back to mini
            elif provider == Provider.ANTHROPIC.value:
                model = "claude-3-5-sonnet"
            else:
                # Use default model for other providers
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    model = provider_instance.get_default_model()
                else:
                    model = None
        except Exception as e:
            logger.warning(f"Error determining model for {provider}: {e}", emoji_key="warning")
            model = None

        # If we still don't have a model, try a different provider
        if not model:
            for fallback_provider in SETTINGS["fallback_providers"]:
                try:
                    provider_instance = gateway.providers.get(fallback_provider)
                    if provider_instance:
                        model = provider_instance.get_default_model()
                        provider = fallback_provider
                        break
                except Exception:
                    continue

            # If still no model, use a reasonable default
            if not model:
                model = "gpt-4.1-mini"
                provider = Provider.OPENAI.value
    else:
        model = refinement_model

    # Define refinement parameters
    iterations = max_iterations or 2  # Default to 2 for demo

    params = {
        "tool_names": [selected_tool],
        "max_iterations": iterations,
        "refinement_model_config": {
            "provider": provider,
            "model": model,
            "temperature": 0.2,
        },
        "validation_level": "full",
        "enable_winnowing": True,
        "progress_callback": display_refinement_progress,
    }

    console.print(
        Panel(
            Syntax(
                json.dumps({k: v for k, v in params.items() if k != "progress_callback"}, indent=2),
                "json",
            ),
            title="Refinement Parameters",
            border_style="dim cyan",
            expand=False,
        )
    )

    # Create a progress display
    console.print("\n[bold cyan]Refinement Progress:[/bold cyan]")
    detail_console.print(f"\n[bold]Starting refinement for {selected_tool}...[/bold]")

    # Estimate cost
    estimated_cost = 0.03 * iterations  # Very rough estimate per iteration
    console.print(f"[cyan]Estimated cost:[/cyan] ${estimated_cost:.2f} USD")

    # Check if cost would exceed limit
    if estimated_cost > SETTINGS["cost_limit"]:
        console.print(
            Panel(
                f"Estimated cost (${estimated_cost:.2f}) exceeds the set limit (${SETTINGS['cost_limit']:.2f}).\n"
                "Adjusting iterations to stay within budget.",
                title="⚠️ Cost Limit Warning",
                border_style="yellow",
                expand=False,
            )
        )
        # Adjust iterations to stay under limit
        adjusted_iterations = max(1, int(SETTINGS["cost_limit"] / 0.03))
        params["max_iterations"] = adjusted_iterations
        console.print(
            f"[yellow]Reducing iterations from {iterations} to {adjusted_iterations}[/yellow]"
        )

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        task_id = progress.add_task("[cyan]Refining tool documentation...", total=100)

        # Execute the refinement
        start_time = time.time()
        try:
            result = await gateway.mcp.call_tool("refine_tool_documentation", params)

            # Simulate progress updates (since we can't hook into the actual progress)
            # The actual progress is displayed through display_refinement_progress
            elapsed = 0
            while progress.tasks[task_id].completed < 100 and elapsed < 60:
                progress.update(task_id, completed=min(95, elapsed * 1.5))
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time

            progress.update(task_id, completed=100)

            # Track cost if available
            if isinstance(result, dict) and "total_refinement_cost" in result:
                tracker.add_generic_cost(
                    cost=result.get("total_refinement_cost", 0.0),
                    description=f"Refinement of {selected_tool}",
                    provider=provider,
                    model=model,
                )

            # Display the results
            display_refinement_result(
                result,
                console=console,
                visualization_level=SETTINGS["visualization_level"],
                save_to_file=SETTINGS["save_results"],
                output_dir=SETTINGS["output_dir"],
            )

            return result

        except Exception as e:
            progress.update(task_id, completed=100, description="[bold red]Refinement failed!")
            logger.error(
                f"Error during single tool refinement: {e}", emoji_key="error", exc_info=True
            )
            console.print(f"[bold red]Error during refinement:[/bold red] {escape(str(e))}")
            return None


async def demo_multi_tool_refinement(
    gateway: Gateway,
    tracker: CostTracker,
    target_tools: Optional[List[str]] = None,
    refinement_provider: Optional[str] = None,
    refinement_model: Optional[str] = None,
    max_iterations: Optional[int] = None,
):
    """Demonstrate refining documentation for multiple tools simultaneously."""
    console.print(Rule("[bold cyan]Multi-Tool Refinement[/bold cyan]", style="cyan"))

    # Use specified tools or find suitable ones
    selected_tools = []

    if target_tools:
        # Check which specified tools exist
        tool_list = await gateway.mcp.list_tools()
        available_tools = [t.name for t in tool_list]

        for tool_name in target_tools:
            if tool_name in available_tools:
                selected_tools.append(tool_name)
            else:
                logger.warning(f"Specified tool '{tool_name}' not found", emoji_key="warning")
                console.print(
                    f"[yellow]Warning:[/yellow] Specified tool '{tool_name}' not found. Skipping."
                )

    # Auto-select if needed
    if not selected_tools:
        # Get various complexity levels for a diverse mix
        simple_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="simple")
        medium_tools = await get_suitable_tools(
            gateway.mcp, count=1, complexity="medium", exclude_tools=simple_tools
        )
        complex_tools = await get_suitable_tools(
            gateway.mcp, count=1, complexity="complex", exclude_tools=simple_tools + medium_tools
        )

        selected_tools = simple_tools + medium_tools + complex_tools

        if not selected_tools:
            # Fall back to any available tools
            selected_tools = await get_suitable_tools(gateway.mcp, count=3, complexity="medium")

    if not selected_tools:
        logger.error("No suitable tools found for multi-tool refinement demo", emoji_key="error")
        console.print(
            "[bold red]Error:[/bold red] No suitable tools found for multi-tool refinement demo."
        )
        return

    console.print(f"Selected tools for refinement: [cyan]{', '.join(selected_tools)}[/cyan]")

    # Determine provider and model
    provider = refinement_provider or Provider.OPENAI.value

    # Find best available model if not specified
    if not refinement_model:
        try:
            if provider == Provider.OPENAI.value:
                model = "gpt-4.1-mini"  # Use mini for multi-tool to save cost
                # Check if model is available
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    models = await provider_instance.list_models()
                    model_ids = [m.get("id") for m in models]
                    if model not in model_ids:
                        model = provider_instance.get_default_model()
            elif provider == Provider.ANTHROPIC.value:
                model = "claude-3-5-haiku"
            else:
                # Use default model for other providers
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    model = provider_instance.get_default_model()
                else:
                    model = None
        except Exception as e:
            logger.warning(f"Error determining model for {provider}: {e}", emoji_key="warning")
            model = None

        # If we still don't have a model, try a different provider
        if not model:
            for fallback_provider in SETTINGS["fallback_providers"]:
                try:
                    provider_instance = gateway.providers.get(fallback_provider)
                    if provider_instance:
                        model = provider_instance.get_default_model()
                        provider = fallback_provider
                        break
                except Exception:
                    continue

            # If still no model, use a reasonable default
            if not model:
                model = "gpt-4.1-mini"
                provider = Provider.OPENAI.value
    else:
        model = refinement_model

    # Define refinement parameters with variations from the first demo
    iterations = max_iterations or 1  # Default to 1 for multi-tool

    params = {
        "tool_names": selected_tools,
        "max_iterations": iterations,
        "refinement_model_config": {
            "provider": provider,
            "model": model,
            "temperature": 0.3,
        },
        # Add an ensemble for better analysis if using full visualization
        "analysis_ensemble_configs": [
            {
                "provider": Provider.ANTHROPIC.value
                if provider != Provider.ANTHROPIC.value
                else Provider.OPENAI.value,
                "model": "claude-3-5-haiku"
                if provider != Provider.ANTHROPIC.value
                else "gpt-4.1-mini",
                "temperature": 0.2,
            }
        ]
        if SETTINGS["visualization_level"] == "full"
        else None,
        "validation_level": "basic",  # Use basic validation for speed
        "enable_winnowing": False,  # Skip winnowing for demo speed
        "progress_callback": display_refinement_progress,
    }

    console.print(
        Panel(
            Syntax(
                json.dumps(
                    {
                        k: v
                        for k, v in params.items()
                        if k not in ["progress_callback", "analysis_ensemble_configs"]
                    },
                    indent=2,
                ),
                "json",
            ),
            title="Multi-Tool Refinement Parameters",
            border_style="dim cyan",
            expand=False,
        )
    )

    # Estimate cost - higher with multiple tools
    estimated_cost = 0.02 * iterations * len(selected_tools)
    console.print(f"[cyan]Estimated cost:[/cyan] ${estimated_cost:.2f} USD")

    # Check if cost would exceed limit
    if estimated_cost > SETTINGS["cost_limit"]:
        console.print(
            Panel(
                f"Estimated cost (${estimated_cost:.2f}) exceeds the set limit (${SETTINGS['cost_limit']:.2f}).\n"
                "Reducing tool count to stay within budget.",
                title="⚠️ Cost Limit Warning",
                border_style="yellow",
                expand=False,
            )
        )
        # Reduce the number of tools
        max_tools = max(1, int(SETTINGS["cost_limit"] / (0.02 * iterations)))
        selected_tools = selected_tools[:max_tools]
        params["tool_names"] = selected_tools
        console.print(f"[yellow]Reducing tools to: {', '.join(selected_tools)}[/yellow]")

    # Create a progress display
    console.print("\n[bold cyan]Multi-Tool Refinement Progress:[/bold cyan]")
    detail_console.print(f"\n[bold]Starting refinement for {len(selected_tools)} tools...[/bold]")

    # We'll create a task for each tool
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        # Create a task for overall progress
        overall_task = progress.add_task("[cyan]Overall progress...", total=100)

        # Execute the refinement
        start_time = time.time()
        try:
            result = await gateway.mcp.call_tool("refine_tool_documentation", params)

            # Simulate progress updates
            # The actual progress is displayed through display_refinement_progress
            elapsed = 0
            while progress.tasks[overall_task].completed < 100 and elapsed < 120:
                progress.update(overall_task, completed=min(95, elapsed * 0.8))
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time

            progress.update(overall_task, completed=100)

            # Track cost if available
            if isinstance(result, dict) and "total_refinement_cost" in result:
                tracker.add_generic_cost(
                    cost=result.get("total_refinement_cost", 0.0),
                    description=f"Multi-tool refinement ({len(selected_tools)} tools)",
                    provider=provider,
                    model=model,
                )

            # Display the results
            display_refinement_result(
                result,
                console=console,
                visualization_level=SETTINGS["visualization_level"],
                save_to_file=SETTINGS["save_results"],
                output_dir=SETTINGS["output_dir"],
            )

            return result

        except Exception as e:
            progress.update(overall_task, completed=100, description="[bold red]Refinement failed!")
            logger.error(
                f"Error during multi-tool refinement: {e}", emoji_key="error", exc_info=True
            )
            console.print(
                f"[bold red]Error during multi-tool refinement:[/bold red] {escape(str(e))}"
            )
            return None


async def demo_custom_test_generation(
    gateway: Gateway,
    tracker: CostTracker,
    target_tool: Optional[str] = None,
    refinement_provider: Optional[str] = None,
    refinement_model: Optional[str] = None,
    max_iterations: Optional[int] = None,
):
    """Demonstrate refinement with custom test generation strategies."""
    console.print(Rule("[bold cyan]Custom Test Generation Strategy[/bold cyan]", style="cyan"))

    # Choose a single tool to refine
    selected_tool = None

    if target_tool:
        # Check if specified tool exists
        tool_list = await gateway.mcp.list_tools()
        available_tools = [t.name for t in tool_list]

        if target_tool in available_tools:
            selected_tool = target_tool
        else:
            logger.warning(f"Specified tool '{target_tool}' not found", emoji_key="warning")
            console.print(
                f"[yellow]Warning:[/yellow] Specified tool '{target_tool}' not found. Selecting automatically."
            )

    # Auto-select if needed (prefer complex tools for custom test demo)
    if not selected_tool:
        complex_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="complex")

        if complex_tools:
            selected_tool = complex_tools[0]
        else:
            # Fall back to medium complexity
            medium_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="medium")

            if medium_tools:
                selected_tool = medium_tools[0]
            else:
                # Last resort - any tool
                simple_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="simple")

                if simple_tools:
                    selected_tool = simple_tools[0]

    if not selected_tool:
        logger.error("No suitable tools found for custom test generation demo", emoji_key="error")
        console.print(
            "[bold red]Error:[/bold red] No suitable tools found for custom test generation demo."
        )
        return

    console.print(f"Selected tool for custom test generation: [cyan]{selected_tool}[/cyan]")

    # Determine provider and model
    provider = refinement_provider or Provider.OPENAI.value

    # Find best available model if not specified
    if not refinement_model:
        try:
            if provider == Provider.OPENAI.value:
                model = "gpt-4.1"  # Prefer this for best results
                # Check if model is available
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    models = await provider_instance.list_models()
                    model_ids = [m.get("id") for m in models]
                    if model not in model_ids:
                        model = "gpt-4.1-mini"  # Fall back to mini
            elif provider == Provider.ANTHROPIC.value:
                model = "claude-3-5-sonnet"
            else:
                # Use default model for other providers
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    model = provider_instance.get_default_model()
                else:
                    model = None
        except Exception as e:
            logger.warning(f"Error determining model for {provider}: {e}", emoji_key="warning")
            model = None

        # If we still don't have a model, try a different provider
        if not model:
            for fallback_provider in SETTINGS["fallback_providers"]:
                try:
                    provider_instance = gateway.providers.get(fallback_provider)
                    if provider_instance:
                        model = provider_instance.get_default_model()
                        provider = fallback_provider
                        break
                except Exception:
                    continue

            # If still no model, use a reasonable default
            if not model:
                model = "gpt-4.1-mini"
                provider = Provider.OPENAI.value
    else:
        model = refinement_model

    # Define refinement parameters with custom test generation strategy
    iterations = max_iterations or 1

    params = {
        "tool_names": [selected_tool],
        "max_iterations": iterations,
        "refinement_model_config": {
            "provider": provider,
            "model": model,
            "temperature": 0.2,
        },
        # Custom test generation strategy
        "generation_config": {
            "positive_required_only": 3,  # More tests with just required params
            "positive_optional_mix": 5,  # More tests with mixed optional params
            "negative_type": 4,  # More type validation checks
            "negative_required": 3,  # More tests with missing required params
            "edge_boundary_min": 2,  # More tests with boundary values
            "edge_boundary_max": 2,
            "llm_realistic_combo": 5,  # More LLM-generated realistic tests
            "llm_ambiguity_probe": 3,  # More tests probing ambiguities
        },
        "validation_level": "full",
        "enable_winnowing": True,
        "progress_callback": display_refinement_progress,
    }

    console.print(
        Panel(
            Group(
                Syntax(
                    json.dumps(
                        {
                            k: v
                            for k, v in params.items()
                            if k not in ["progress_callback", "generation_config"]
                        },
                        indent=2,
                    ),
                    "json",
                ),
                "\n[bold cyan]Custom Test Generation Strategy:[/bold cyan]",
                Syntax(json.dumps(params["generation_config"], indent=2), "json"),
            ),
            title="Custom Test Generation Parameters",
            border_style="dim cyan",
            expand=False,
        )
    )

    # Estimate cost (higher due to more test cases)
    estimated_cost = 0.04 * iterations
    console.print(f"[cyan]Estimated cost:[/cyan] ${estimated_cost:.2f} USD")

    # Check if cost would exceed limit
    if estimated_cost > SETTINGS["cost_limit"]:
        console.print(
            Panel(
                f"Estimated cost (${estimated_cost:.2f}) exceeds the set limit (${SETTINGS['cost_limit']:.2f}).\n"
                "Reducing iterations to stay within budget.",
                title="⚠️ Cost Limit Warning",
                border_style="yellow",
                expand=False,
            )
        )
        # Adjust iterations to stay under limit
        params["max_iterations"] = 1

    # Create a progress display
    console.print("\n[bold cyan]Custom Test Generation Progress:[/bold cyan]")
    detail_console.print(
        f"\n[bold]Starting refinement with custom test strategy for {selected_tool}...[/bold]"
    )

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        task_id = progress.add_task("[cyan]Refining with custom test strategy...", total=100)

        # Execute the refinement
        start_time = time.time()
        try:
            result = await gateway.mcp.call_tool("refine_tool_documentation", params)

            # Simulate progress updates
            elapsed = 0
            while progress.tasks[task_id].completed < 100 and elapsed < 60:
                progress.update(task_id, completed=min(95, elapsed * 1.5))
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time

            progress.update(task_id, completed=100)

            # Track cost if available
            if isinstance(result, dict) and "total_refinement_cost" in result:
                tracker.add_generic_cost(
                    cost=result.get("total_refinement_cost", 0.0),
                    description=f"Custom test strategy for {selected_tool}",
                    provider=provider,
                    model=model,
                )

            # Display the results
            display_refinement_result(
                result,
                console=console,
                visualization_level=SETTINGS["visualization_level"],
                save_to_file=SETTINGS["save_results"],
                output_dir=SETTINGS["output_dir"],
            )

            return result

        except Exception as e:
            progress.update(task_id, completed=100, description="[bold red]Refinement failed!")
            logger.error(
                f"Error during custom test generation: {e}", emoji_key="error", exc_info=True
            )
            console.print(
                f"[bold red]Error during custom test generation:[/bold red] {escape(str(e))}"
            )
            return None


async def demo_all_tools_refinement(
    gateway: Gateway,
    tracker: CostTracker,
    refinement_provider: Optional[str] = None,
    refinement_model: Optional[str] = None,
    max_iterations: Optional[int] = None,
):
    """Demonstrate refining documentation for all available tools."""
    console.print(Rule("[bold cyan]All Tools Refinement[/bold cyan]", style="cyan"))

    # Get all available tools (excluding refine_tool_documentation itself)
    tool_list = await gateway.mcp.list_tools()
    available_tools = [t.name for t in tool_list if t.name != "refine_tool_documentation"]

    if not available_tools:
        logger.error("No tools available for refinement", emoji_key="error")
        console.print("[bold red]Error:[/bold red] No tools available for refinement.")
        return

    console.print(f"[cyan]Found {len(available_tools)} tools available for refinement[/cyan]")

    # Determine provider and model
    provider = refinement_provider or Provider.OPENAI.value

    # Find best available model if not specified
    if not refinement_model:
        try:
            if provider == Provider.OPENAI.value:
                model = "gpt-4.1-mini"  # Use mini for multi-tool to save cost
                # Check if model is available
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    models = await provider_instance.list_models()
                    model_ids = [m.get("id") for m in models]
                    if model not in model_ids:
                        model = provider_instance.get_default_model()
            elif provider == Provider.ANTHROPIC.value:
                model = "claude-3-5-haiku"
            else:
                # Use default model for other providers
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    model = provider_instance.get_default_model()
                else:
                    model = None
        except Exception as e:
            logger.warning(f"Error determining model for {provider}: {e}", emoji_key="warning")
            model = None

        # If we still don't have a model, try a different provider
        if not model:
            for fallback_provider in SETTINGS["fallback_providers"]:
                try:
                    provider_instance = gateway.providers.get(fallback_provider)
                    if provider_instance:
                        model = provider_instance.get_default_model()
                        provider = fallback_provider
                        break
                except Exception:
                    continue

            # If still no model, use a reasonable default
            if not model:
                model = "gpt-4.1-mini"
                provider = Provider.OPENAI.value
    else:
        model = refinement_model

    # Define refinement parameters
    iterations = max_iterations or 1  # Default to 1 for all-tools

    params = {
        "refine_all_available": True,  # This is the key difference for this demo
        "max_iterations": iterations,
        "refinement_model_config": {
            "provider": provider,
            "model": model,
            "temperature": 0.3,
        },
        "validation_level": "basic",  # Use basic validation for speed
        "enable_winnowing": False,  # Skip winnowing for demo speed
        "progress_callback": display_refinement_progress,
    }

    console.print(
        Panel(
            Syntax(
                json.dumps({k: v for k, v in params.items() if k != "progress_callback"}, indent=2),
                "json",
            ),
            title="All Tools Refinement Parameters",
            border_style="dim cyan",
            expand=False,
        )
    )

    # Estimate cost - higher with multiple tools
    estimated_cost = (
        0.01 * iterations * len(available_tools)
    )  # Lower per-tool cost with bulk processing
    console.print(f"[cyan]Estimated cost:[/cyan] ${estimated_cost:.2f} USD")

    # Check if cost would exceed limit
    if estimated_cost > SETTINGS["cost_limit"]:
        console.print(
            Panel(
                f"Estimated cost (${estimated_cost:.2f}) exceeds the set limit (${SETTINGS['cost_limit']:.2f}).\n"
                "Switching to targeted refinement to stay within budget.",
                title="⚠️ Cost Limit Warning",
                border_style="yellow",
                expand=False,
            )
        )

        # Switch to using targeted tool_names instead of refine_all_available
        max_tools = max(1, int(SETTINGS["cost_limit"] / (0.02 * iterations)))
        selected_tools = random.sample(available_tools, min(max_tools, len(available_tools)))

        params["refine_all_available"] = False
        params["tool_names"] = selected_tools

        console.print(f"[yellow]Reducing to {len(selected_tools)} randomly selected tools[/yellow]")

    # Create a progress display
    console.print("\n[bold cyan]All Tools Refinement Progress:[/bold cyan]")
    detail_console.print(
        f"\n[bold]Starting refinement for all {len(available_tools)} tools...[/bold]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    ) as progress:
        task_id = progress.add_task("[cyan]Refining all tools...", total=100)

        # Execute the refinement
        start_time = time.time()
        try:
            result = await gateway.mcp.call_tool("refine_tool_documentation", params)

            # Simulate progress updates
            elapsed = 0
            while (
                progress.tasks[task_id].completed < 100 and elapsed < 300
            ):  # Longer timeout for all tools
                progress.update(
                    task_id, completed=min(95, elapsed * 0.3)
                )  # Slower progress for more tools
                await asyncio.sleep(1.0)
                elapsed = time.time() - start_time

            progress.update(task_id, completed=100)

            # Track cost if available
            if isinstance(result, dict) and "total_refinement_cost" in result:
                tracker.add_generic_cost(
                    cost=result.get("total_refinement_cost", 0.0),
                    description=f"All tools refinement ({len(available_tools)} tools)",
                    provider=provider,
                    model=model,
                )

            # Display the results
            display_refinement_result(
                result,
                console=console,
                visualization_level=SETTINGS["visualization_level"],
                save_to_file=SETTINGS["save_results"],
                output_dir=SETTINGS["output_dir"],
            )

            return result

        except Exception as e:
            progress.update(task_id, completed=100, description="[bold red]Refinement failed!")
            logger.error(
                f"Error during all tools refinement: {e}", emoji_key="error", exc_info=True
            )
            console.print(
                f"[bold red]Error during all tools refinement:[/bold red] {escape(str(e))}"
            )
            return None


async def demo_schema_focused_refinement(
    gateway: Gateway,
    tracker: CostTracker,
    target_tool: Optional[str] = None,
    refinement_provider: Optional[str] = None,
    refinement_model: Optional[str] = None,
):
    """Demonstrate refinement focused specifically on schema improvements."""
    console.print(Rule("[bold cyan]Schema-Focused Refinement[/bold cyan]", style="cyan"))

    # Choose a complex tool to refine
    selected_tool = None

    if target_tool:
        # Check if specified tool exists
        tool_list = await gateway.mcp.list_tools()
        available_tools = [t.name for t in tool_list]

        if target_tool in available_tools:
            selected_tool = target_tool
        else:
            logger.warning(f"Specified tool '{target_tool}' not found", emoji_key="warning")
            console.print(
                f"[yellow]Warning:[/yellow] Specified tool '{target_tool}' not found. Selecting automatically."
            )

    # Auto-select if needed (prefer complex tools for schema refinement)
    if not selected_tool:
        complex_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="complex")

        if complex_tools:
            selected_tool = complex_tools[0]
        else:
            # Fall back to medium complexity
            medium_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="medium")

            if medium_tools:
                selected_tool = medium_tools[0]
            else:
                # Last resort - any tool
                simple_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="simple")

                if simple_tools:
                    selected_tool = simple_tools[0]

    if not selected_tool:
        logger.error(
            "No suitable tools found for schema-focused refinement demo", emoji_key="error"
        )
        console.print(
            "[bold red]Error:[/bold red] No suitable tools found for schema-focused refinement demo."
        )
        return

    console.print(f"Selected tool for schema-focused refinement: [cyan]{selected_tool}[/cyan]")

    # Get tool schema
    tool_list = await gateway.mcp.list_tools()
    tool_def = next((t for t in tool_list if t.name == selected_tool), None)

    if not tool_def or not hasattr(tool_def, "inputSchema"):
        logger.error(f"Could not get schema for tool {selected_tool}", emoji_key="error")
        console.print(f"[bold red]Error:[/bold red] Could not get schema for tool {selected_tool}.")
        return

    input_schema = getattr(tool_def, "inputSchema", {})

    # Display the original schema
    console.print("[bold cyan]Original Schema:[/bold cyan]")
    console.print(
        Panel(
            Syntax(json.dumps(input_schema, indent=2), "json", theme="default", line_numbers=False),
            title="Original Input Schema",
            border_style="dim cyan",
            expand=False,
        )
    )

    # Determine provider and model
    provider = refinement_provider or Provider.OPENAI.value

    # Find best available model if not specified
    if not refinement_model:
        try:
            if provider == Provider.OPENAI.value:
                model = "gpt-4.1"  # Prefer this for best schema analysis
                # Check if model is available
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    models = await provider_instance.list_models()
                    model_ids = [m.get("id") for m in models]
                    if model not in model_ids:
                        model = "gpt-4.1-mini"  # Fall back to mini
            elif provider == Provider.ANTHROPIC.value:
                model = "claude-3-5-sonnet"
            else:
                # Use default model for other providers
                provider_instance = gateway.providers.get(provider)
                if provider_instance:
                    model = provider_instance.get_default_model()
                else:
                    model = None
        except Exception as e:
            logger.warning(f"Error determining model for {provider}: {e}", emoji_key="warning")
            model = None

        # If we still don't have a model, try a different provider
        if not model:
            for fallback_provider in SETTINGS["fallback_providers"]:
                try:
                    provider_instance = gateway.providers.get(fallback_provider)
                    if provider_instance:
                        model = provider_instance.get_default_model()
                        provider = fallback_provider
                        break
                except Exception:
                    continue

            # If still no model, use a reasonable default
            if not model:
                model = "gpt-4.1-mini"
                provider = Provider.OPENAI.value
    else:
        model = refinement_model

    # Define refinement parameters focused on schema improvements
    params = {
        "tool_names": [selected_tool],
        "max_iterations": 1,  # Single iteration focused on schema
        "refinement_model_config": {
            "provider": provider,
            "model": model,
            "temperature": 0.2,
        },
        # Custom test generation strategy focused on schema edge cases
        "generation_config": {
            "positive_required_only": 2,
            "positive_optional_mix": 3,
            "negative_type": 4,  # More type validation checks
            "negative_required": 3,  # More tests with missing required params
            "negative_enum": 3,  # More enum testing
            "negative_format": 3,  # More format testing
            "negative_range": 3,  # More range testing
            "negative_length": 3,  # More length testing
            "negative_pattern": 3,  # More pattern testing
            "edge_boundary_min": 3,  # More tests with min boundary values
            "edge_boundary_max": 3,  # More tests with max boundary values
            "llm_ambiguity_probe": 2,  # Probe for ambiguities
        },
        "validation_level": "full",  # Strict validation
        "enable_winnowing": False,  # No winnowing needed
        "progress_callback": display_refinement_progress,
    }

    console.print(
        Panel(
            Syntax(
                json.dumps(
                    {
                        k: v
                        for k, v in params.items()
                        if k not in ["progress_callback", "generation_config"]
                    },
                    indent=2,
                ),
                "json",
            ),
            title="Schema-Focused Refinement Parameters",
            border_style="dim cyan",
            expand=False,
        )
    )

    # Estimate cost
    estimated_cost = 0.035  # Schema focus costs a bit more due to edge case testing
    console.print(f"[cyan]Estimated cost:[/cyan] ${estimated_cost:.2f} USD")

    # Create a progress display
    console.print("\n[bold cyan]Schema-Focused Refinement Progress:[/bold cyan]")
    detail_console.print(
        f"\n[bold]Starting schema-focused refinement for {selected_tool}...[/bold]"
    )

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        task_id = progress.add_task("[cyan]Refining schema...", total=100)

        # Execute the refinement
        start_time = time.time()
        try:
            result = await gateway.mcp.call_tool("refine_tool_documentation", params)

            # Simulate progress updates
            elapsed = 0
            while progress.tasks[task_id].completed < 100 and elapsed < 60:
                progress.update(task_id, completed=min(95, elapsed * 1.5))
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time

            progress.update(task_id, completed=100)

            # Track cost if available
            if isinstance(result, dict) and "total_refinement_cost" in result:
                tracker.add_generic_cost(
                    cost=result.get("total_refinement_cost", 0.0),
                    description=f"Schema-focused refinement of {selected_tool}",
                    provider=provider,
                    model=model,
                )

            # Extract schema patches from the result
            refined_tools = result.get("refined_tools", [])
            target_tool_result = next(
                (t for t in refined_tools if t.get("tool_name") == selected_tool), None
            )

            if target_tool_result and target_tool_result.get("final_proposed_schema_patches"):
                schema_patches = target_tool_result.get("final_proposed_schema_patches", [])
                patched_schema = target_tool_result.get("final_schema_after_patches", {})

                if schema_patches:
                    console.print("[bold green]Schema Refinement Results:[/bold green]")

                    console.print(
                        Panel(
                            Syntax(
                                json.dumps(schema_patches, indent=2),
                                "json",
                                theme="default",
                                line_numbers=False,
                            ),
                            title="Applied Schema Patches",
                            border_style="magenta",
                            expand=False,
                        )
                    )

                    if patched_schema:
                        console.print(
                            Panel(
                                Syntax(
                                    json.dumps(patched_schema, indent=2),
                                    "json",
                                    theme="default",
                                    line_numbers=False,
                                ),
                                title="Refined Schema",
                                border_style="green",
                                expand=False,
                            )
                        )

                    # Generate a side-by-side comparison
                    console.print(
                        create_side_by_side_diff(
                            json.dumps(input_schema, indent=2),
                            json.dumps(patched_schema, indent=2),
                            title="Schema Before/After Comparison",
                        )
                    )
                else:
                    console.print("[yellow]No schema patches were applied.[/yellow]")

            # Display the full results
            display_refinement_result(
                result,
                console=console,
                visualization_level=SETTINGS["visualization_level"],
                save_to_file=SETTINGS["save_results"],
                output_dir=SETTINGS["output_dir"],
            )

            return result

        except Exception as e:
            progress.update(task_id, completed=100, description="[bold red]Refinement failed!")
            logger.error(
                f"Error during schema-focused refinement: {e}", emoji_key="error", exc_info=True
            )
            console.print(
                f"[bold red]Error during schema-focused refinement:[/bold red] {escape(str(e))}"
            )
            return None


async def demo_model_comparison(
    gateway: Gateway, tracker: CostTracker, target_tool: Optional[str] = None
):
    """Demonstrate comparing different LLM models for refinement."""
    console.print(Rule("[bold cyan]Model Comparison for Refinement[/bold cyan]", style="cyan"))

    # Choose a single tool to refine
    selected_tool = None

    if target_tool:
        # Check if specified tool exists
        tool_list = await gateway.mcp.list_tools()
        available_tools = [t.name for t in tool_list]

        if target_tool in available_tools:
            selected_tool = target_tool
        else:
            logger.warning(f"Specified tool '{target_tool}' not found", emoji_key="warning")
            console.print(
                f"[yellow]Warning:[/yellow] Specified tool '{target_tool}' not found. Selecting automatically."
            )

    # Auto-select if needed
    if not selected_tool:
        medium_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="medium")

        if medium_tools:
            selected_tool = medium_tools[0]
        else:
            # Fall back to any available tool
            simple_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="simple")

            if simple_tools:
                selected_tool = simple_tools[0]

    if not selected_tool:
        logger.error("No suitable tools found for model comparison demo", emoji_key="error")
        console.print(
            "[bold red]Error:[/bold red] No suitable tools found for model comparison demo."
        )
        return

    console.print(f"Selected tool for model comparison: [cyan]{selected_tool}[/cyan]")

    # Define models to compare
    models_to_compare = []

    # Check which models are available
    for provider_name in SETTINGS["preferred_providers"] + SETTINGS["fallback_providers"]:
        provider_instance = gateway.providers.get(provider_name)
        if provider_instance:
            try:
                available_models = await provider_instance.list_models()
                model_ids = [m.get("id") for m in available_models]

                if provider_name == Provider.OPENAI.value:
                    if "gpt-4.1" in model_ids:
                        models_to_compare.append((provider_name, "gpt-4.1"))
                    if "gpt-4.1-mini" in model_ids:
                        models_to_compare.append((provider_name, "gpt-4.1-mini"))

                elif provider_name == Provider.ANTHROPIC.value:
                    if "claude-3-5-sonnet" in model_ids:
                        models_to_compare.append((provider_name, "claude-3-5-sonnet"))
                    if "claude-3-5-haiku" in model_ids:
                        models_to_compare.append((provider_name, "claude-3-5-haiku"))

                elif provider_name == Provider.GEMINI.value:
                    if "gemini-2.0-pro" in model_ids:
                        models_to_compare.append((provider_name, "gemini-2.0-pro"))

                elif provider_name == Provider.DEEPSEEK.value:
                    if "deepseek-chat" in model_ids:
                        models_to_compare.append((provider_name, "deepseek-chat"))

                # If we already have 3+ models, stop looking
                if len(models_to_compare) >= 3:
                    break

            except Exception as e:
                logger.warning(
                    f"Error listing models for {provider_name}: {e}", emoji_key="warning"
                )

    # If we don't have enough models, add some defaults that might work
    if len(models_to_compare) < 2:
        fallback_models = [
            (Provider.OPENAI.value, "gpt-4.1-mini"),
            (Provider.ANTHROPIC.value, "claude-3-5-haiku"),
            (Provider.GEMINI.value, "gemini-2.0-pro"),
        ]

        for provider, model in fallback_models:
            if (provider, model) not in models_to_compare:
                models_to_compare.append((provider, model))
                if len(models_to_compare) >= 3:
                    break

    # Limit to max 3 models for a reasonable comparison
    models_to_compare = models_to_compare[:3]

    if not models_to_compare:
        logger.error("No models available for comparison", emoji_key="error")
        console.print("[bold red]Error:[/bold red] No models available for comparison.")
        return

    console.print(
        f"Models being compared: [cyan]{', '.join([f'{p}/{m}' for p, m in models_to_compare])}[/cyan]"
    )

    # Estimate total cost
    estimated_cost = 0.03 * len(models_to_compare)
    console.print(f"[cyan]Estimated total cost:[/cyan] ${estimated_cost:.2f} USD")

    # Check if cost would exceed limit
    if estimated_cost > SETTINGS["cost_limit"]:
        console.print(
            Panel(
                f"Estimated cost (${estimated_cost:.2f}) exceeds the set limit (${SETTINGS['cost_limit']:.2f}).\n"
                "Reducing the number of models to compare.",
                title="⚠️ Cost Limit Warning",
                border_style="yellow",
                expand=False,
            )
        )
        max_models = max(2, int(SETTINGS["cost_limit"] / 0.03))
        models_to_compare = models_to_compare[:max_models]
        console.print(
            f"[yellow]Comparing only: {', '.join([f'{p}/{m}' for p, m in models_to_compare])}[/yellow]"
        )

    # Create a progress display
    console.print("\n[bold cyan]Model Comparison Progress:[/bold cyan]")

    # Results storage
    model_results = {}

    # Run refinement with each model
    for provider, model in models_to_compare:
        detail_console.print(f"\n[bold]Starting refinement with {provider}/{model}...[/bold]")

        params = {
            "tool_names": [selected_tool],
            "max_iterations": 1,
            "refinement_model_config": {
                "provider": provider,
                "model": model,
                "temperature": 0.2,
            },
            "validation_level": "basic",
            "enable_winnowing": False,
            "progress_callback": display_refinement_progress,
        }

        with Progress(
            TextColumn(f"[bold blue]Testing {provider}/{model}..."),
            BarColumn(complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        ) as progress:
            task_id = progress.add_task(f"[cyan]Refining with {model}...", total=100)

            # Execute the refinement
            start_time = time.time()
            try:
                result = await gateway.mcp.call_tool("refine_tool_documentation", params)

                # Simulate progress updates
                elapsed = 0
                while progress.tasks[task_id].completed < 100 and elapsed < 60:
                    progress.update(task_id, completed=min(95, elapsed * 1.5))
                    await asyncio.sleep(0.5)
                    elapsed = time.time() - start_time

                progress.update(task_id, completed=100)

                # Track cost if available
                if isinstance(result, dict) and "total_refinement_cost" in result:
                    tracker.add_generic_cost(
                        cost=result.get("total_refinement_cost", 0.0),
                        description=f"{provider}/{model} refinement of {selected_tool}",
                        provider=provider,
                        model=model,
                    )

                # Store result for comparison
                model_results[(provider, model)] = {
                    "result": result,
                    "processing_time": time.time() - start_time,
                    "cost": result.get("total_refinement_cost", 0.0)
                    if isinstance(result, dict)
                    else 0.0,
                }

            except Exception as e:
                progress.update(task_id, completed=100, description=f"[bold red]{model} failed!")
                logger.error(
                    f"Error during refinement with {provider}/{model}: {e}",
                    emoji_key="error",
                    exc_info=True,
                )
                console.print(
                    f"[bold red]Error during refinement with {provider}/{model}:[/bold red] {escape(str(e))}"
                )

    # Compare and display results
    if model_results:
        console.print(Rule("[bold blue]Model Comparison Results[/bold blue]", style="blue"))

        # Create comparison table
        comparison_table = Table(title="Model Performance Comparison", box=box.ROUNDED)
        comparison_table.add_column("Model", style="cyan")
        comparison_table.add_column("Initial Success", style="dim yellow")
        comparison_table.add_column("Final Success", style="green")
        comparison_table.add_column("Improvement", style="magenta")
        comparison_table.add_column("Processing Time", style="blue")
        comparison_table.add_column("Cost", style="red")

        for (provider, model), data in model_results.items():
            result = data["result"]
            refined_tools = result.get("refined_tools", [])

            # Find the specific tool result
            tool_result = next(
                (t for t in refined_tools if t.get("tool_name") == selected_tool), None
            )

            if tool_result:
                initial_success = tool_result.get("initial_success_rate", 0.0)
                final_success = tool_result.get("final_success_rate", 0.0)
                improvement = tool_result.get("improvement_factor", 0.0)

                comparison_table.add_row(
                    f"{provider}/{model}",
                    f"{initial_success:.1%}",
                    f"{final_success:.1%}",
                    f"{improvement:.2f}x",
                    f"{data['processing_time']:.2f}s",
                    f"${data['cost']:.6f}",
                )

        console.print(comparison_table)

        # Find the best model
        best_model = None
        best_improvement = -1

        for (provider, model), data in model_results.items():
            result = data["result"]
            refined_tools = result.get("refined_tools", [])
            tool_result = next(
                (t for t in refined_tools if t.get("tool_name") == selected_tool), None
            )

            if tool_result:
                improvement = tool_result.get("improvement_factor", 0.0)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_model = (provider, model)

        if best_model:
            console.print(
                f"[bold green]Best model:[/bold green] [cyan]{best_model[0]}/{best_model[1]}[/cyan] with {best_improvement:.2f}x improvement"
            )

            # Show detailed results for the best model
            best_data = model_results[best_model]
            console.print("\n[bold cyan]Detailed Results for Best Model:[/bold cyan]")

            display_refinement_result(
                best_data["result"],
                console=console,
                visualization_level=SETTINGS["visualization_level"],
                save_to_file=SETTINGS["save_results"],
                output_dir=SETTINGS["output_dir"],
            )

        return model_results
    else:
        console.print("[yellow]No results available for comparison.[/yellow]")
        return None


async def demo_cost_optimization(
    gateway: Gateway, tracker: CostTracker, target_tool: Optional[str] = None
):
    """Demonstrate cost optimization techniques for documentation refinement."""
    console.print(Rule("[bold cyan]Cost Optimization Techniques[/bold cyan]", style="cyan"))

    # Choose a single tool to refine
    selected_tool = None

    if target_tool:
        # Check if specified tool exists
        tool_list = await gateway.mcp.list_tools()
        available_tools = [t.name for t in tool_list]

        if target_tool in available_tools:
            selected_tool = target_tool
        else:
            logger.warning(f"Specified tool '{target_tool}' not found", emoji_key="warning")
            console.print(
                f"[yellow]Warning:[/yellow] Specified tool '{target_tool}' not found. Selecting automatically."
            )

    # Auto-select if needed
    if not selected_tool:
        medium_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="medium")

        if medium_tools:
            selected_tool = medium_tools[0]
        else:
            # Fall back to any available tool
            simple_tools = await get_suitable_tools(gateway.mcp, count=1, complexity="simple")

            if simple_tools:
                selected_tool = simple_tools[0]

    if not selected_tool:
        logger.error("No suitable tools found for cost optimization demo", emoji_key="error")
        console.print(
            "[bold red]Error:[/bold red] No suitable tools found for cost optimization demo."
        )
        return

    console.print(f"Selected tool for cost optimization: [cyan]{selected_tool}[/cyan]")

    # Create a table of optimization techniques
    optimization_table = Table(title="Cost Optimization Techniques", box=box.SIMPLE_HEAD)
    optimization_table.add_column("Technique", style="cyan")
    optimization_table.add_column("Description", style="white")
    optimization_table.add_column("Est. Savings", style="green")

    optimization_table.add_row(
        "Smaller Models",
        "Use smaller, faster models for initial iterations or simple tools",
        "50-80%",
    )
    optimization_table.add_row(
        "Reduced Iterations", "Single iteration can capture most improvements", "30-60%"
    )
    optimization_table.add_row(
        "Basic Validation", "Use 'basic' validation level instead of 'full'", "10-20%"
    )
    optimization_table.add_row(
        "Focused Strategies", "Custom test generation focused on important cases", "20-40%"
    )
    optimization_table.add_row("Bulk Processing", "Refine multiple related tools at once", "30-50%")
    optimization_table.add_row(
        "Skip Winnowing", "Disable winnowing for quick improvements", "5-10%"
    )

    console.print(optimization_table)

    # Define and display standard vs. optimized configurations
    standard_config = {
        "tool_names": [selected_tool],
        "max_iterations": 3,
        "refinement_model_config": {
            "provider": Provider.OPENAI.value,
            "model": "gpt-4.1",
            "temperature": 0.2,
        },
        "validation_level": "full",
        "enable_winnowing": True,
    }

    optimized_config = {
        "tool_names": [selected_tool],
        "max_iterations": 1,
        "refinement_model_config": {
            "provider": Provider.OPENAI.value,
            "model": "gpt-4.1-mini",
            "temperature": 0.3,
        },
        "validation_level": "basic",
        "enable_winnowing": False,
        # Focused test generation to save costs
        "generation_config": {
            "positive_required_only": 2,
            "positive_optional_mix": 2,
            "negative_type": 2,
            "negative_required": 1,
            "negative_enum": 0,
            "negative_format": 0,
            "negative_range": 0,
            "negative_length": 0,
            "negative_pattern": 0,
            "edge_empty": 0,
            "edge_null": 0,
            "edge_boundary_min": 0,
            "edge_boundary_max": 0,
            "llm_realistic_combo": 2,
            "llm_ambiguity_probe": 1,
            "llm_simulation_based": 0,
        },
    }

    # Compare costs
    standard_est_cost = 0.09  # 3 iterations with gpt-4.1
    optimized_est_cost = 0.015  # 1 iteration with gpt-4.1-mini and reduced tests
    savings_pct = ((standard_est_cost - optimized_est_cost) / standard_est_cost) * 100

    console.print(
        Panel(
            Group(
                "[bold]Standard Config:[/bold]",
                Syntax(
                    json.dumps(standard_config, indent=2),
                    "json",
                    theme="default",
                    line_numbers=False,
                ),
                f"[yellow]Estimated Cost: ${standard_est_cost:.3f}[/yellow]",
                "\n[bold]Optimized Config:[/bold]",
                Syntax(
                    json.dumps(optimized_config, indent=2),
                    "json",
                    theme="default",
                    line_numbers=False,
                ),
                f"[green]Estimated Cost: ${optimized_est_cost:.3f}[/green]",
                f"\n[bold cyan]Estimated Savings: {savings_pct:.1f}%[/bold cyan]",
            ),
            title="Cost Comparison",
            border_style="dim cyan",
            expand=False,
        )
    )

    # Run the optimized configuration
    console.print("\n[bold cyan]Running Cost-Optimized Refinement:[/bold cyan]")
    detail_console.print(
        f"\n[bold]Starting cost-optimized refinement for {selected_tool}...[/bold]"
    )

    # Add progress callback
    optimized_config["progress_callback"] = display_refinement_progress

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        task_id = progress.add_task("[cyan]Running cost-optimized refinement...", total=100)

        # Execute the refinement
        start_time = time.time()
        try:
            result = await gateway.mcp.call_tool("refine_tool_documentation", optimized_config)

            # Simulate progress updates
            elapsed = 0
            while progress.tasks[task_id].completed < 100 and elapsed < 30:
                progress.update(
                    task_id, completed=min(95, elapsed * 3)
                )  # Faster progress for optimized mode
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time

            progress.update(task_id, completed=100)

            # Track cost if available
            if isinstance(result, dict) and "total_refinement_cost" in result:
                actual_cost = result.get("total_refinement_cost", 0.0)
                tracker.add_generic_cost(
                    cost=actual_cost,
                    description=f"Cost-optimized refinement of {selected_tool}",
                    provider=optimized_config["refinement_model_config"]["provider"],
                    model=optimized_config["refinement_model_config"]["model"],
                )

                # Compare estimated vs actual cost
                console.print("[bold cyan]Cost Analysis:[/bold cyan]")
                console.print(f"Estimated Cost: ${optimized_est_cost:.3f}")
                console.print(f"Actual Cost: ${actual_cost:.3f}")
                console.print(
                    f"Actual Savings vs. Standard: {((standard_est_cost - actual_cost) / standard_est_cost) * 100:.1f}%"
                )

            # Display the results
            display_refinement_result(
                result,
                console=console,
                visualization_level=SETTINGS["visualization_level"],
                save_to_file=SETTINGS["save_results"],
                output_dir=SETTINGS["output_dir"],
            )

            return result

        except Exception as e:
            progress.update(task_id, completed=100, description="[bold red]Refinement failed!")
            logger.error(
                f"Error during cost-optimized refinement: {e}", emoji_key="error", exc_info=True
            )
            console.print(
                f"[bold red]Error during cost-optimized refinement:[/bold red] {escape(str(e))}"
            )
            return None


async def demo_practical_testing(gateway: Gateway, tracker: CostTracker):
    """Demonstrate practical testing with flawed examples."""
    console.print(Rule("[bold cyan]Practical Testing with Flawed Tools[/bold cyan]", style="cyan"))

    # Check if we have flawed example tools
    created_tools = await create_flawed_example_tools(gateway.mcp)

    if not created_tools:
        logger.error("Failed to create flawed example tools", emoji_key="error")
        console.print(
            "[bold red]Error:[/bold red] Failed to create flawed example tools for demonstration."
        )
        return

    console.print(
        f"Created {len(created_tools)} flawed example tools for practical testing:\n"
        + "\n".join([f"- [cyan]{name}[/cyan]" for name in created_tools])
    )

    # Get details on the intentional flaws
    flaws_table = Table(title="Intentional Documentation Flaws", box=box.ROUNDED)
    flaws_table.add_column("Tool", style="cyan")
    flaws_table.add_column("Flaw Type", style="yellow")
    flaws_table.add_column("Description", style="white")

    flaws_table.add_row(
        "flawed_process_text",
        "Ambiguous Description",
        "Description is vague and doesn't explain parameters.",
    )
    flaws_table.add_row(
        "flawed_scrape_website",
        "Missing Parameter Descriptions",
        "Parameters in schema have no descriptions.",
    )
    flaws_table.add_row(
        "flawed_data_processor",
        "Confusing Schema & Description Mismatch",
        "Description calls the tool 'analyzer' but name is 'processor'.",
    )
    flaws_table.add_row(
        "flawed_product_search",
        "Misleading Examples",
        "Example shows incorrect parameter name 'sort_by' vs schema 'sort'.",
    )
    flaws_table.add_row(
        "flawed_calculator",
        "Schema/Implementation Conflict",
        "Clear description but possible schema type confusion.",
    )

    console.print(flaws_table)

    # Select a flawed tool to demonstrate refinement
    selected_tool = created_tools[0]  # Start with the first one
    console.print(f"\nSelected tool for demonstration: [cyan]{selected_tool}[/cyan]")

    # Show the original flawed tool definition
    tool_list = await gateway.mcp.list_tools()
    tool_def = next((t for t in tool_list if t.name == selected_tool), None)

    if tool_def and hasattr(tool_def, "inputSchema") and hasattr(tool_def, "description"):
        input_schema = getattr(tool_def, "inputSchema", {})
        description = getattr(tool_def, "description", "")

        console.print("[bold cyan]Original Flawed Tool Definition:[/bold cyan]")

        console.print(
            Panel(
                escape(description),
                title="Original Description",
                border_style="dim red",
                expand=False,
            )
        )

        console.print(
            Panel(
                Syntax(
                    json.dumps(input_schema, indent=2), "json", theme="default", line_numbers=False
                ),
                title="Original Schema",
                border_style="dim red",
                expand=False,
            )
        )

    # Run refinement on the flawed tool
    console.print("\n[bold cyan]Running Refinement on Flawed Tool:[/bold cyan]")
    detail_console.print(f"\n[bold]Starting refinement for flawed tool {selected_tool}...[/bold]")

    params = {
        "tool_names": [selected_tool],
        "max_iterations": 2,
        "refinement_model_config": {
            "provider": Provider.OPENAI.value,
            "model": "gpt-4.1",  # Use the best model for these challenging cases
            "temperature": 0.2,
        },
        "validation_level": "full",
        "enable_winnowing": True,
        "progress_callback": display_refinement_progress,
    }

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        task_id = progress.add_task("[cyan]Refining flawed tool...", total=100)

        # Execute the refinement
        start_time = time.time()
        try:
            result = await gateway.mcp.call_tool("refine_tool_documentation", params)

            # Simulate progress updates
            elapsed = 0
            while progress.tasks[task_id].completed < 100 and elapsed < 60:
                progress.update(task_id, completed=min(95, elapsed * 1.5))
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time

            progress.update(task_id, completed=100)

            # Track cost if available
            if isinstance(result, dict) and "total_refinement_cost" in result:
                tracker.add_generic_cost(
                    cost=result.get("total_refinement_cost", 0.0),
                    description=f"Flawed tool refinement of {selected_tool}",
                    provider=Provider.OPENAI.value,
                    model="gpt-4.1",
                )

            # Display the results
            display_refinement_result(
                result,
                console=console,
                visualization_level=SETTINGS["visualization_level"],
                save_to_file=SETTINGS["save_results"],
                output_dir=SETTINGS["output_dir"],
            )

            # Highlight identified flaws
            refined_tools = result.get("refined_tools", [])
            target_tool_result = next(
                (t for t in refined_tools if t.get("tool_name") == selected_tool), None
            )

            if target_tool_result:
                identified_flaws = []
                for iter_data in target_tool_result.get("iterations", []):
                    analysis = iter_data.get("analysis", {})
                    if analysis:
                        flaws = analysis.get("identified_flaw_categories", [])
                        for flaw in flaws:
                            if flaw not in identified_flaws:
                                identified_flaws.append(flaw)

                if identified_flaws:
                    console.print("\n[bold cyan]Identified Documentation Flaws:[/bold cyan]")
                    flaw_details = {
                        "MISSING_DESCRIPTION": "Documentation is missing key information",
                        "AMBIGUOUS_DESCRIPTION": "Description is unclear or can be interpreted in multiple ways",
                        "INCORRECT_DESCRIPTION": "Description contains incorrect information",
                        "MISSING_SCHEMA_CONSTRAINT": "Schema is missing important constraints",
                        "INCORRECT_SCHEMA_CONSTRAINT": "Schema contains incorrect constraints",
                        "OVERLY_RESTRICTIVE_SCHEMA": "Schema is unnecessarily restrictive",
                        "TYPE_CONFUSION": "Parameter types are inconsistent or unclear",
                        "MISSING_EXAMPLE": "Documentation lacks necessary examples",
                        "MISLEADING_EXAMPLE": "Examples provided are incorrect or misleading",
                        "INCOMPLETE_EXAMPLE": "Examples are present but insufficient",
                        "PARAMETER_DEPENDENCY_UNCLEAR": "Dependencies between parameters are not explained",
                        "CONFLICTING_CONSTRAINTS": "Schema contains contradictory constraints",
                        "AGENT_FORMULATION_ERROR": "Documentation hinders LLM agent's ability to use the tool",
                        "SCHEMA_PREVALIDATION_FAILURE": "Schema validation issues",
                        "TOOL_EXECUTION_ERROR": "Issues with tool execution",
                        "UNKNOWN": "Unspecified documentation issue",
                    }

                    for flaw in identified_flaws:
                        console.print(
                            f"- [bold yellow]{flaw}[/bold yellow]: {flaw_details.get(flaw, 'No description available')}"
                        )

            return result

        except Exception as e:
            progress.update(task_id, completed=100, description="[bold red]Refinement failed!")
            logger.error(
                f"Error during flawed tool refinement: {e}", emoji_key="error", exc_info=True
            )
            console.print(
                f"[bold red]Error during flawed tool refinement:[/bold red] {escape(str(e))}"
            )
            return None


async def main():
    """Main entry point for the demo."""
    try:
        print("Starting demo...")
        logger.debug("Starting demo...")
        args = parse_arguments()
        print(f"Args parsed: {args}")
        logger.debug(f"Args parsed: {args}")

        # Set up gateway
        print("Setting up gateway...")
        gateway = await setup_gateway_and_tools(create_flawed_tools=args.create_flawed)  # noqa: F841
        print("Gateway setup complete")

        # Initialize cost tracker
        tracker = CostTracker(limit=SETTINGS["cost_limit"])

        # Check if the tool was successfully registered
        print("Checking if tool is registered...")
        tool_list = await mcp.list_tools()
        available_tools = [t.name for t in tool_list]
        print(f"Available tools: {available_tools}")

        if "refine_tool_documentation" in available_tools:
            print("Tool is available, proceeding with demo")
            logger.info("Tool successfully registered, proceeding with demo", emoji_key="success")

            # Run the selected demo based on CLI arguments
            print(f"Running demo: {args.demo}")

            # Select a demo based on specified arguments
            if args.demo == "single" or args.demo == "all":
                print("Running single tool refinement demo")
                result = await demo_single_tool_refinement(
                    gateway,
                    tracker,
                    target_tool=args.tool,
                    refinement_provider=args.provider,
                    refinement_model=args.model,
                    max_iterations=args.iterations,
                )
                if result:
                    logger.success("Single tool refinement demo completed", emoji_key="success")

            elif args.demo == "multi":
                print("Running multi-tool refinement demo")
                result = await demo_multi_tool_refinement(
                    gateway,
                    tracker,
                    target_tools=[args.tool] if args.tool else None,
                    refinement_provider=args.provider,
                    refinement_model=args.model,
                    max_iterations=args.iterations,
                )
                if result:
                    logger.success("Multi-tool refinement demo completed", emoji_key="success")

            elif args.demo == "custom-testing":
                print("Running custom test generation demo")
                result = await demo_custom_test_generation(
                    gateway,
                    tracker,
                    target_tool=args.tool,
                    refinement_provider=args.provider,
                    refinement_model=args.model,
                    max_iterations=args.iterations,
                )
                if result:
                    logger.success("Custom test generation demo completed", emoji_key="success")

            elif args.demo == "optimize":
                print("Running cost optimization demo")
                result = await demo_cost_optimization(gateway, tracker, target_tool=args.tool)
                if result:
                    logger.success("Cost optimization demo completed", emoji_key="success")

            elif args.demo == "all-tools":
                print("Running all-tools refinement demo")
                result = await demo_all_tools_refinement(
                    gateway,
                    tracker,
                    refinement_provider=args.provider,
                    refinement_model=args.model,
                    max_iterations=args.iterations,
                )
                if result:
                    logger.success("All-tools refinement demo completed", emoji_key="success")

            elif args.demo == "schema-focus":
                print("Running schema-focused refinement demo")
                result = await demo_schema_focused_refinement(
                    gateway,
                    tracker,
                    target_tool=args.tool,
                    refinement_provider=args.provider,
                    refinement_model=args.model,
                )
                if result:
                    logger.success("Schema-focused refinement demo completed", emoji_key="success")

            elif args.demo == "practical":
                print("Running practical testing demo")
                result = await demo_practical_testing(gateway, tracker)
                if result:
                    logger.success("Practical testing demo completed", emoji_key="success")

            elif args.demo == "model-comparison":
                print("Running model comparison demo")
                result = await demo_model_comparison(gateway, tracker, target_tool=args.tool)
                if result:
                    logger.success("Model comparison demo completed", emoji_key="success")

            elif args.demo == "all":
                print("Running all demos")
                console.print(
                    Panel(
                        "Running all demos in sequence. This may take some time.",
                        title="ℹ️ Running All Demos",
                        border_style="cyan",
                        expand=False,
                    )
                )

                # Run each demo in sequence
                demos = [
                    demo_single_tool_refinement(
                        gateway,
                        tracker,
                        target_tool=args.tool,
                        refinement_provider=args.provider,
                        refinement_model=args.model,
                        max_iterations=args.iterations,
                    ),
                    demo_multi_tool_refinement(
                        gateway,
                        tracker,
                        refinement_provider=args.provider,
                        refinement_model=args.model,
                        max_iterations=args.iterations,
                    ),
                    demo_custom_test_generation(
                        gateway,
                        tracker,
                        target_tool=args.tool,
                        refinement_provider=args.provider,
                        refinement_model=args.model,
                    ),
                    demo_cost_optimization(gateway, tracker, target_tool=args.tool),
                    demo_schema_focused_refinement(
                        gateway,
                        tracker,
                        target_tool=args.tool,
                        refinement_provider=args.provider,
                        refinement_model=args.model,
                    ),
                    demo_model_comparison(gateway, tracker, target_tool=args.tool),
                ]

                if args.create_flawed:
                    demos.append(demo_practical_testing(gateway, tracker))

                for demo_coro in demos:
                    try:
                        await demo_coro
                    except Exception as e:
                        logger.error(f"Error running demo: {e}", emoji_key="error", exc_info=True)
                        console.print(f"[bold red]Error running demo:[/bold red] {escape(str(e))}")

                logger.success("All demos completed", emoji_key="success")

            else:
                print("No valid demo specified")
                console.print(
                    Panel(
                        f"The specified demo '{args.demo}' is not recognized.\n"
                        "Available demos: all, single, multi, custom-testing, optimize, all-tools, schema-focus, practical, model-comparison",
                        title="⚠️ Invalid Demo Selection",
                        border_style="yellow",
                        expand=False,
                    )
                )
        else:
            print("Tool is not available")
            # Tool not available, show error message
            console.print(
                Panel(
                    "This demo requires the docstring_refiner tool to be properly registered.\n"
                    "Due to known issues with Pydantic definitions, the tool can't be registered in this demo.\n\n"
                    "Check that you have the correct version of the Ultimate MCP Server and dependencies installed.",
                    title="⚠️ Demo Requirements Not Met",
                    border_style="red",
                    expand=False,
                )
            )

        # Display cost summary
        console.print(Rule("[bold green]Total Demo Cost Summary[/bold green]", style="green"))
        tracker.display_costs(console=console)

        logger.info("Docstring Refiner Demo completed successfully", emoji_key="success")
        console.print(Rule("[bold green]Demo Complete[/bold green]", style="green"))
        print("Demo completed successfully")

    except Exception as e:
        print(f"Error in main: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
