#!/usr/bin/env python
"""Enhanced demo of the Advanced Response Comparator & Synthesizer Tool."""

import asyncio
import json
import sys
from collections import namedtuple  # Import namedtuple
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.server import Gateway  # Use Gateway to get MCP

# from ultimate_mcp_server.tools.meta import compare_and_synthesize  # Add correct import
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker  # Import CostTracker
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger
logger = get_logger("example.compare_synthesize_v2")

# Create a simple structure for cost tracking from dict (tokens might be missing)
TrackableResult = namedtuple(
    "TrackableResult",
    ["cost", "input_tokens", "output_tokens", "provider", "model", "processing_time"],
)

# Global MCP instance (will be populated from Gateway)
mcp = None


async def setup_gateway_and_tools():
    """Set up the gateway and register tools."""
    global mcp
    logger.info("Initializing Gateway and MetaTools for enhanced demo...", emoji_key="start")
    gateway = Gateway("compare-synthesize-demo-v2", register_tools=False)

    # Initialize providers (needed for the tool to function)
    try:
        await gateway._initialize_providers()
    except Exception as e:
        logger.critical(
            f"Failed to initialize providers: {e}. Check API keys.",
            emoji_key="critical",
            exc_info=True,
        )
        sys.exit(1)  # Exit if providers can't be initialized

    # REMOVE MetaTools instance
    # meta_tools = MetaTools(gateway) # Pass the gateway instance  # noqa: F841
    mcp = gateway.mcp  # Store the MCP server instance

    # Manually register the required tool
    # mcp.tool()(compare_and_synthesize)
    # logger.info("Manually registered compare_and_synthesize tool.")

    # Verify tool registration
    tool_list = await mcp.list_tools()
    tool_names = [t.name for t in tool_list]  # Access name attribute directly
    # Use console.print for tool list
    console.print(f"Registered tools: [cyan]{escape(str(tool_names))}[/cyan]")
    if "compare_and_synthesize" in tool_names:
        logger.success("compare_and_synthesize tool registered successfully.", emoji_key="success")
    else:
        logger.error("compare_and_synthesize tool FAILED to register.", emoji_key="error")
        sys.exit(1)  # Exit if the required tool isn't available

    logger.success("Setup complete.", emoji_key="success")


# Refactored print_result function using Rich
def print_result(title: str, result: dict):
    """Helper function to print results clearly using Rich components."""
    console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))

    # Handle potential list result format (from older tool versions?)
    if isinstance(result, list) and len(result) > 0:
        if hasattr(result[0], "text"):
            try:
                result = json.loads(result[0].text)
            except Exception:
                result = {"error": "Failed to parse result from list format"}
        else:
            result = result[0]  # Assume first item is the dict
    elif not isinstance(result, dict):
        result = {"error": f"Unexpected result format: {type(result)}"}

    if result.get("error"):
        error_content = f"[red]Error:[/red] {escape(result['error'])}"
        if "partial_results" in result and result["partial_results"]:
            try:
                partial_json = json.dumps(result["partial_results"], indent=2)
                error_content += "\n\n[yellow]Partial Results:[/yellow]"
                error_panel_content = Syntax(
                    partial_json, "json", theme="default", line_numbers=False, word_wrap=True
                )
            except Exception as json_err:
                error_panel_content = (
                    f"[red]Could not display partial results: {escape(str(json_err))}[/red]"
                )
        else:
            error_panel_content = error_content

        console.print(
            Panel(
                error_panel_content,
                title="[bold red]Tool Error[/bold red]",
                border_style="red",
                expand=False,
            )
        )

    else:
        # Display synthesis/analysis sections
        if "synthesis" in result:
            synthesis_data = result["synthesis"]
            if isinstance(synthesis_data, dict):
                if "best_response_text" in synthesis_data:
                    console.print(
                        Panel(
                            escape(synthesis_data["best_response_text"].strip()),
                            title="[bold green]Best Response Text[/bold green]",
                            border_style="green",
                            expand=False,
                        )
                    )

                if "synthesized_response" in synthesis_data:
                    console.print(
                        Panel(
                            escape(synthesis_data["synthesized_response"].strip()),
                            title="[bold magenta]Synthesized Response[/bold magenta]",
                            border_style="magenta",
                            expand=False,
                        )
                    )

                if synthesis_data.get("best_response", {}).get("reasoning"):
                    console.print(
                        Panel(
                            escape(synthesis_data["best_response"]["reasoning"].strip()),
                            title="[bold cyan]Best Response Reasoning[/bold cyan]",
                            border_style="dim cyan",
                            expand=False,
                        )
                    )

                if synthesis_data.get("synthesis_strategy"):
                    console.print(
                        Panel(
                            escape(synthesis_data["synthesis_strategy"].strip()),
                            title="[bold yellow]Synthesis Strategy Explanation[/bold yellow]",
                            border_style="dim yellow",
                            expand=False,
                        )
                    )

                if "ranking" in synthesis_data:
                    try:
                        ranking_json = json.dumps(synthesis_data["ranking"], indent=2)
                        console.print(
                            Panel(
                                Syntax(
                                    ranking_json,
                                    "json",
                                    theme="default",
                                    line_numbers=False,
                                    word_wrap=True,
                                ),
                                title="[bold]Ranking[/bold]",
                                border_style="dim blue",
                                expand=False,
                            )
                        )
                    except Exception as json_err:
                        console.print(
                            f"[red]Could not display ranking: {escape(str(json_err))}[/red]"
                        )

                if "comparative_analysis" in synthesis_data:
                    try:
                        analysis_json = json.dumps(synthesis_data["comparative_analysis"], indent=2)
                        console.print(
                            Panel(
                                Syntax(
                                    analysis_json,
                                    "json",
                                    theme="default",
                                    line_numbers=False,
                                    word_wrap=True,
                                ),
                                title="[bold]Comparative Analysis[/bold]",
                                border_style="dim blue",
                                expand=False,
                            )
                        )
                    except Exception as json_err:
                        console.print(
                            f"[red]Could not display comparative analysis: {escape(str(json_err))}[/red]"
                        )

            else:  # Handle case where synthesis data isn't a dict (e.g., raw text error)
                console.print(
                    Panel(
                        f"[yellow]Synthesis Output (raw/unexpected format):[/yellow]\n{escape(str(synthesis_data))}",
                        title="[bold yellow]Synthesis Data[/bold yellow]",
                        border_style="yellow",
                        expand=False,
                    )
                )

        # Display Stats Table
        stats_table = Table(
            title="[bold]Execution Stats[/bold]", box=box.ROUNDED, show_header=False, expand=False
        )
        stats_table.add_column("Metric", style="cyan", no_wrap=True)
        stats_table.add_column("Value", style="white")
        stats_table.add_row(
            "Eval/Synth Model",
            f"{escape(result.get('synthesis_provider', 'N/A'))}/{escape(result.get('synthesis_model', 'N/A'))}",
        )
        stats_table.add_row("Total Cost", f"${result.get('cost', {}).get('total_cost', 0.0):.6f}")
        stats_table.add_row("Processing Time", f"{result.get('processing_time', 0.0):.2f}s")
        console.print(stats_table)

    console.print()  # Add spacing after each result block


async def run_comparison_demo(tracker: CostTracker):
    """Demonstrate different modes of compare_and_synthesize."""
    if not mcp:
        logger.error("MCP server not initialized. Run setup first.", emoji_key="error")
        return

    prompt = "Explain the main benefits of using asynchronous programming in Python for a moderately technical audience. Provide 2-3 key advantages."

    # --- Configuration for initial responses ---
    console.print(Rule("[bold green]Configurations[/bold green]"))
    console.print(f"[cyan]Prompt:[/cyan] {escape(prompt)}")
    initial_configs = [
        {
            "provider": Provider.OPENAI.value,
            "model": "gpt-4.1-mini",
            "parameters": {"temperature": 0.6, "max_tokens": 150},
        },
        {
            "provider": Provider.ANTHROPIC.value,
            "model": "claude-3-5-haiku-20241022",
            "parameters": {"temperature": 0.5, "max_tokens": 150},
        },
        {
            "provider": Provider.GEMINI.value,
            "model": "gemini-2.0-flash-lite",
            "parameters": {"temperature": 0.7, "max_tokens": 150},
        },
        {
            "provider": Provider.DEEPSEEK.value,
            "model": "deepseek-chat",
            "parameters": {"temperature": 0.6, "max_tokens": 150},
        },
    ]
    console.print(
        f"[cyan]Initial Models:[/cyan] {[f'{c["provider"]}:{c["model"]}' for c in initial_configs]}"
    )

    # --- Evaluation Criteria ---
    criteria = [
        "Clarity: Is the explanation clear and easy to understand for the target audience?",
        "Accuracy: Are the stated benefits of async programming technically correct?",
        "Relevance: Does the response directly address the prompt and focus on key advantages?",
        "Conciseness: Is the explanation brief and to the point?",
        "Completeness: Does it mention 2-3 distinct and significant benefits?",
    ]
    console.print("[cyan]Evaluation Criteria:[/cyan]")
    for i, criterion in enumerate(criteria):
        console.print(f"  {i + 1}. {escape(criterion)}")

    # --- Criteria Weights (Optional) ---
    criteria_weights = {
        "Clarity: Is the explanation clear and easy to understand for the target audience?": 0.3,
        "Accuracy: Are the stated benefits of async programming technically correct?": 0.3,
        "Relevance: Does the response directly address the prompt and focus on key advantages?": 0.15,
        "Conciseness: Is the explanation brief and to the point?": 0.1,
        "Completeness: Does it mention 2-3 distinct and significant benefits?": 0.15,
    }
    console.print("[cyan]Criteria Weights (Optional):[/cyan]")
    # Create a small table for weights
    weights_table = Table(box=box.MINIMAL, show_header=False)
    weights_table.add_column("Criterion Snippet", style="dim")
    weights_table.add_column("Weight", style="green")
    for crit, weight in criteria_weights.items():
        weights_table.add_row(escape(crit.split(":")[0]), f"{weight:.2f}")
    console.print(weights_table)

    # --- Synthesis/Evaluation Model ---
    synthesis_model_config = {"provider": Provider.OPENAI.value, "model": "gpt-4.1"}
    console.print(
        f"[cyan]Synthesis/Evaluation Model:[/cyan] {escape(synthesis_model_config['provider'])}:{escape(synthesis_model_config['model'])}"
    )
    console.print()  # Spacing before demos start

    common_args = {
        "prompt": prompt,
        "configs": initial_configs,
        "criteria": criteria,
        "criteria_weights": criteria_weights,
    }

    # --- Demo 1: Select Best Response ---
    logger.info("Running format 'best'...", emoji_key="processing")
    try:
        result = await mcp.call_tool(
            "compare_and_synthesize",
            {
                **common_args,
                "response_format": "best",
                "include_reasoning": True,  # Show why it was selected
                "synthesis_model": synthesis_model_config,  # Explicitly specify model to avoid OpenRouter
            },
        )
        print_result("Response Format: 'best' (with reasoning)", result)
        # Track cost
        if (
            isinstance(result, dict)
            and "cost" in result
            and "synthesis_provider" in result
            and "synthesis_model" in result
        ):
            try:
                trackable = TrackableResult(
                    cost=result.get("cost", {}).get("total_cost", 0.0),
                    input_tokens=0,  # Tokens not typically aggregated in this tool's output
                    output_tokens=0,
                    provider=result.get("synthesis_provider", "unknown"),
                    model=result.get("synthesis_model", "compare_synthesize"),
                    processing_time=result.get("processing_time", 0.0),
                )
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(
                    f"Could not track cost for 'best' format: {track_err}", exc_info=False
                )
    except Exception as e:
        logger.error(f"Error during 'best' format demo: {e}", emoji_key="error", exc_info=True)

    # --- Demo 2: Synthesize Responses (Comprehensive Strategy) ---
    logger.info("Running format 'synthesis' (comprehensive)...", emoji_key="processing")
    try:
        result = await mcp.call_tool(
            "compare_and_synthesize",
            {
                **common_args,
                "response_format": "synthesis",
                "synthesis_strategy": "comprehensive",
                "synthesis_model": synthesis_model_config,  # Specify model for consistency
                "include_reasoning": True,
            },
        )
        print_result("Response Format: 'synthesis' (Comprehensive Strategy)", result)
        # Track cost
        if (
            isinstance(result, dict)
            and "cost" in result
            and "synthesis_provider" in result
            and "synthesis_model" in result
        ):
            try:
                trackable = TrackableResult(
                    cost=result.get("cost", {}).get("total_cost", 0.0),
                    input_tokens=0,  # Tokens not typically aggregated
                    output_tokens=0,
                    provider=result.get("synthesis_provider", "unknown"),
                    model=result.get("synthesis_model", "compare_synthesize"),
                    processing_time=result.get("processing_time", 0.0),
                )
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(
                    f"Could not track cost for 'synthesis comprehensive': {track_err}",
                    exc_info=False,
                )
    except Exception as e:
        logger.error(
            f"Error during 'synthesis comprehensive' demo: {e}", emoji_key="error", exc_info=True
        )

    # --- Demo 3: Synthesize Responses (Conservative Strategy, No Reasoning) ---
    logger.info(
        "Running format 'synthesis' (conservative, no reasoning)...", emoji_key="processing"
    )
    try:
        result = await mcp.call_tool(
            "compare_and_synthesize",
            {
                **common_args,
                "response_format": "synthesis",
                "synthesis_strategy": "conservative",
                "synthesis_model": synthesis_model_config,  # Explicitly specify
                "include_reasoning": False,  # Hide the synthesis strategy explanation
            },
        )
        print_result("Response Format: 'synthesis' (Conservative, No Reasoning)", result)
        # Track cost
        if (
            isinstance(result, dict)
            and "cost" in result
            and "synthesis_provider" in result
            and "synthesis_model" in result
        ):
            try:
                trackable = TrackableResult(
                    cost=result.get("cost", {}).get("total_cost", 0.0),
                    input_tokens=0,  # Tokens not typically aggregated
                    output_tokens=0,
                    provider=result.get("synthesis_provider", "unknown"),
                    model=result.get("synthesis_model", "compare_synthesize"),
                    processing_time=result.get("processing_time", 0.0),
                )
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(
                    f"Could not track cost for 'synthesis conservative': {track_err}",
                    exc_info=False,
                )
    except Exception as e:
        logger.error(
            f"Error during 'synthesis conservative' demo: {e}", emoji_key="error", exc_info=True
        )

    # --- Demo 4: Rank Responses ---
    logger.info("Running format 'ranked'...", emoji_key="processing")
    try:
        result = await mcp.call_tool(
            "compare_and_synthesize",
            {
                **common_args,
                "response_format": "ranked",
                "include_reasoning": True,  # Show reasoning for ranks
                "synthesis_model": synthesis_model_config,  # Explicitly specify
            },
        )
        print_result("Response Format: 'ranked' (with reasoning)", result)
        # Track cost
        if (
            isinstance(result, dict)
            and "cost" in result
            and "synthesis_provider" in result
            and "synthesis_model" in result
        ):
            try:
                trackable = TrackableResult(
                    cost=result.get("cost", {}).get("total_cost", 0.0),
                    input_tokens=0,  # Tokens not typically aggregated
                    output_tokens=0,
                    provider=result.get("synthesis_provider", "unknown"),
                    model=result.get("synthesis_model", "compare_synthesize"),
                    processing_time=result.get("processing_time", 0.0),
                )
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(
                    f"Could not track cost for 'ranked' format: {track_err}", exc_info=False
                )
    except Exception as e:
        logger.error(f"Error during 'ranked' format demo: {e}", emoji_key="error", exc_info=True)

    # --- Demo 5: Analyze Responses ---
    logger.info("Running format 'analysis'...", emoji_key="processing")
    try:
        result = await mcp.call_tool(
            "compare_and_synthesize",
            {
                **common_args,
                "response_format": "analysis",
                # No reasoning needed for analysis format, it's inherent
                "synthesis_model": synthesis_model_config,  # Explicitly specify
            },
        )
        print_result("Response Format: 'analysis'", result)
        # Track cost
        if (
            isinstance(result, dict)
            and "cost" in result
            and "synthesis_provider" in result
            and "synthesis_model" in result
        ):
            try:
                trackable = TrackableResult(
                    cost=result.get("cost", {}).get("total_cost", 0.0),
                    input_tokens=0,  # Tokens not typically aggregated
                    output_tokens=0,
                    provider=result.get("synthesis_provider", "unknown"),
                    model=result.get("synthesis_model", "compare_synthesize"),
                    processing_time=result.get("processing_time", 0.0),
                )
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(
                    f"Could not track cost for 'analysis' format: {track_err}", exc_info=False
                )
    except Exception as e:
        logger.error(f"Error during 'analysis' format demo: {e}", emoji_key="error", exc_info=True)

    # Display cost summary at the end
    tracker.display_summary(console)


async def main():
    """Run the enhanced compare_and_synthesize demo."""
    tracker = CostTracker()  # Instantiate tracker
    await setup_gateway_and_tools()
    await run_comparison_demo(tracker)  # Pass tracker
    # logger.info("Skipping run_comparison_demo() as the 'compare_and_synthesize' tool function is missing.") # Remove skip message


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Demo stopped by user.")
    except Exception as main_err:
        logger.critical(
            f"Demo failed with unexpected error: {main_err}", emoji_key="critical", exc_info=True
        )
