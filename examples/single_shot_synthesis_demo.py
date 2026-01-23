#!/usr/bin/env python3
"""
Single-Shot Synthesis Demo - Demonstrates the single_shot_synthesis tool

This script shows how to:
1. Define a prompt and a set of "expert" models.
2. Specify a "synthesizer" model.
3. Call the single_shot_synthesis tool to get a fused response.
4. Display the individual expert responses and the final synthesized output.

Usage:
  python examples/single_shot_synthesis_demo.py [--prompt "Your question here"] [--type text|code]

Options:
  --prompt TEXT    The prompt/question for the models.
  --name TEXT      A descriptive name for the synthesis task.
  --type TYPE      Type of synthesis: 'text' or 'code' (default: text).
  --expert-models MODEL [MODEL...]  List of expert model IDs.
  --synthesizer-model MODEL         Model ID for the synthesizer.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich import box
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.exceptions import ProviderError, ToolError  # For error handling

# from ultimate_mcp_server.tools.single_shot_synthesis import single_shot_synthesis # Called via gateway
from ultimate_mcp_server.utils import get_logger, process_mcp_result
from ultimate_mcp_server.utils.display import CostTracker  # Reusing CostTracker
from ultimate_mcp_server.utils.logging.console import console

logger = get_logger("example.single_shot_synthesis")
gateway: Optional[Gateway] = None

# --- Configuration ---
DEFAULT_EXPERT_MODEL_CONFIGS_SSS: List[Dict[str, Any]] = [  # SSS suffix for SingleShotSynthesis
    {"model_id": "openai/gpt-4o-mini", "temperature": 0.7},
    {"model_id": "anthropic/claude-3-5-haiku-20241022", "temperature": 0.65},
    # {"model_id": "google/gemini-1.5-flash-latest", "temperature": 0.7},
]

DEFAULT_SYNTHESIZER_MODEL_CONFIG_SSS: Dict[str, Any] = {
    "model_id": "anthropic/claude-3-7-sonnet-20250219",
    "temperature": 0.5,
    "max_tokens": 3000,  # Allow more tokens for comprehensive synthesis
}
# Fallback if preferred synthesizer isn't available
FALLBACK_SYNTHESIZER_MODEL_CONFIG_SSS: Dict[str, Any] = {
    "model_id": "anthropic/claude-3-7-sonnet-20250219",  # Fallback to Sonnet 3.5
    "temperature": 0.5,
    "max_tokens": 3000,
}

DEFAULT_SSS_PROMPT = "Compare and contrast the query optimization strategies used in PostgreSQL versus MySQL for complex analytical queries involving multiple joins and aggregations. Highlight key differences in their execution planners and indexing techniques."
DEFAULT_SSS_TASK_NAME = "DB Query Optimization Comparison"
DEFAULT_SSS_TYPE = "text"


def parse_arguments_sss():
    parser = argparse.ArgumentParser(description="Run a single-shot multi-model synthesis demo")
    parser.add_argument("--prompt", type=str, default=DEFAULT_SSS_PROMPT)
    parser.add_argument("--name", type=str, default=DEFAULT_SSS_TASK_NAME)
    parser.add_argument("--type", type=str, default=DEFAULT_SSS_TYPE, choices=["text", "code"])
    parser.add_argument(
        "--expert-models",
        type=str,
        nargs="+",
        default=[mc["model_id"] for mc in DEFAULT_EXPERT_MODEL_CONFIGS_SSS],
        help="List of expert model IDs.",
    )
    parser.add_argument(
        "--synthesizer-model",
        type=str,
        default=DEFAULT_SYNTHESIZER_MODEL_CONFIG_SSS["model_id"],
        help="Model ID for the synthesizer.",
    )
    return parser.parse_args()


async def setup_gateway_for_demo_sss():
    global gateway
    if gateway:
        return
    logger.info("Initializing gateway for single-shot synthesis demo...", emoji_key="rocket")
    try:
        gateway = Gateway(name="sss_demo_gateway", register_tools=True, load_all_tools=True)
        if not gateway.providers:
            await gateway._initialize_providers()
    except Exception as e:
        logger.critical(f"Failed to initialize Gateway: {e}", exc_info=True)
        raise

    mcp_tools = await gateway.mcp.list_tools()
    registered_tool_names = [t.name for t in mcp_tools]
    if "single_shot_synthesis" not in registered_tool_names:
        logger.error(
            "Gateway initialized, but 'single_shot_synthesis' tool is missing!", emoji_key="error"
        )
        raise RuntimeError("Required 'single_shot_synthesis' tool not registered.")
    logger.success(
        "Gateway for demo initialized and synthesis tool verified.", emoji_key="heavy_check_mark"
    )


def display_single_shot_synthesis_results(
    results_data: Dict[str, Any],
    original_prompt: str,  # Added to display the prompt
    console_instance,
):
    """Displays the results from the single_shot_synthesis tool."""
    console_instance.print(
        Rule(
            f"[bold magenta]Single-Shot Synthesis Task: {results_data.get('name', 'N/A')}[/bold magenta]"
        )
    )

    if not results_data or not isinstance(results_data, dict) or not results_data.get("request_id"):
        console_instance.print(
            Panel(
                "[bold red]No valid results data to display or essential fields missing.[/bold red]",
                border_style="red",
            )
        )
        if isinstance(results_data, dict) and results_data.get("error_message"):
            console_instance.print(
                f"[bold red]Error in results data:[/bold red] {escape(results_data['error_message'])}"
            )
        return

    console_instance.print(f"Request ID: [cyan]{results_data.get('request_id')}[/cyan]")
    status = results_data.get("status", "UNKNOWN")
    status_color = (
        "green" if status == "SUCCESS" else ("yellow" if status == "PARTIAL_SUCCESS" else "red")
    )
    console_instance.print(f"Status: [bold {status_color}]{status}[/bold {status_color}]")

    if results_data.get("error_message") and status in ["FAILED", "PARTIAL_SUCCESS"]:
        console_instance.print(
            Panel(
                f"[red]{escape(results_data.get('error_message'))}[/red]",
                title="[bold red]Error Message[/bold red]",
                border_style="red",
            )
        )

    storage_path = results_data.get("storage_path")
    if storage_path:
        console_instance.print(
            f"Artifacts Storage: [blue underline]{escape(storage_path)}[/blue underline]"
        )

    console_instance.print(
        Panel(
            escape(original_prompt),
            title="[bold]Original Prompt[/bold]",
            border_style="blue",
            expand=False,
        )
    )

    console_instance.print(Rule("[bold blue]Expert Model Responses[/bold blue]"))
    expert_responses = results_data.get("expert_responses", [])
    if expert_responses:
        for i, resp_dict in enumerate(expert_responses):
            model_id_display = resp_dict.get("model_id", "Unknown Model")
            has_error = bool(resp_dict.get("error"))
            status_icon = "❌" if has_error else "✔️"
            panel_title = f"{status_icon} Expert {i + 1}: {model_id_display}"
            current_border_style = "red" if has_error else "dim cyan"
            if has_error:
                panel_title += " [bold red](Failed)[/bold red]"

            content_table = Table(box=None, show_header=False, padding=(0, 1))
            content_table.add_column(style="dim")
            content_table.add_column()

            if resp_dict.get("error"):
                content_table.add_row(
                    "[bold red]Error[/bold red]", escape(resp_dict.get("error", ""))
                )

            text_content = resp_dict.get("response_text")
            # Assuming 'code' type experts are not used in this specific demo,
            # but adding for completeness if structure changes.
            # code_content = resp_dict.get("extracted_code")
            # if code_content:
            #     content_table.add_row("Extracted Code", Syntax(code_content, "python", theme="monokai", line_numbers=True, word_wrap=True))

            if text_content:
                content_table.add_row(
                    "Response Text",
                    escape(text_content[:1000] + ("..." if len(text_content) > 1000 else "")),
                )
            elif not resp_dict.get("error"):
                content_table.add_row(
                    "Response Text", "[italic]No content from this expert.[/italic]"
                )

            metrics = resp_dict.get("metrics", {})
            cost = metrics.get("cost", 0.0)
            api_latency = metrics.get("api_latency_ms", "N/A")
            total_task_time = metrics.get("total_task_time_ms", "N/A")
            input_tokens = metrics.get("input_tokens", "N/A")
            output_tokens = metrics.get("output_tokens", "N/A")

            metrics_table = Table(box=box.ROUNDED, show_header=False, title="Metrics")
            metrics_table.add_column(style="cyan")
            metrics_table.add_column(style="white")
            metrics_table.add_row("Cost", f"${cost:.6f}")
            metrics_table.add_row("Input Tokens", str(input_tokens))
            metrics_table.add_row("Output Tokens", str(output_tokens))
            metrics_table.add_row("API Latency", f"{api_latency} ms")
            metrics_table.add_row("Total Task Time", f"{total_task_time} ms")
            if (
                metrics.get("api_model_id_used")
                and metrics.get("api_model_id_used") != model_id_display
            ):
                metrics_table.add_row("API Model Used", str(metrics.get("api_model_id_used")))

            main_panel_content = [content_table, metrics_table]

            console_instance.print(
                Panel(
                    Group(*main_panel_content),
                    title=f"[bold cyan]{panel_title}[/bold cyan]",
                    border_style=current_border_style,
                    expand=False,
                )
            )
    else:
        console_instance.print("[italic]No expert responses available.[/italic]")

    console_instance.print(Rule("[bold green]Synthesized Response[/bold green]"))

    synthesizer_metrics = results_data.get("synthesizer_metrics", {})
    synthesizer_model_id_used_api = synthesizer_metrics.get("api_model_id_used")

    # Attempt to get configured synthesizer model from input if API one is not available (should be rare)
    # This requires passing synthesizer_config to this function or storing it in results_data
    # For now, we rely on api_model_id_used from metrics.
    # Example: configured_synth_model = results_data.get("synthesizer_model_config", {}).get("model_id", "N/A")
    # synthesizer_model_display = synthesizer_model_id_used_api or configured_synth_model

    if synthesizer_model_id_used_api:
        console_instance.print(
            f"Synthesizer Model Used (from API): [magenta]{synthesizer_model_id_used_api}[/magenta]"
        )
    else:
        # If not in metrics, try to infer from input or display N/A (needs input passed)
        console_instance.print(
            "Synthesizer Model: [magenta]N/A (configured model not directly in output, check logs or input config)[/magenta]"
        )

    thinking_process = results_data.get("synthesizer_thinking_process")
    if thinking_process:
        console_instance.print(
            Panel(
                escape(thinking_process),
                title="[bold]Synthesizer Thinking Process[/bold]",
                border_style="yellow",
                expand=False,
            )
        )

    final_text = results_data.get("synthesized_response_text")
    final_code = results_data.get("synthesized_extracted_code")
    # Determine if the original task was for code
    tournament_type = results_data.get(
        "tournament_type", "text"
    )  # Assuming this field might be added to output for context

    if tournament_type == "code" and final_code:
        console_instance.print(
            Panel(
                Syntax(final_code, "python", theme="monokai", line_numbers=True, word_wrap=True),
                title="[bold]Final Synthesized Code[/bold]",
                border_style="green",
            )
        )
    elif (
        final_text
    ):  # Also show text if it's a code tournament but no code was extracted, or if it's text type
        console_instance.print(
            Panel(
                escape(final_text),
                title="[bold]Final Synthesized Text[/bold]",
                border_style="green",
            )
        )
    else:
        console_instance.print(
            "[italic]No synthesized response generated (or it was empty).[/italic]"
        )

    if synthesizer_metrics:
        console_instance.print(Rule("[bold]Synthesizer Metrics[/bold]"))
        synth_metrics_table = Table(box=box.SIMPLE, show_header=False, title_justify="left")
        synth_metrics_table.add_column("Metric", style="cyan")
        synth_metrics_table.add_column("Value", style="white")
        synth_metrics_table.add_row("Cost", f"${synthesizer_metrics.get('cost', 0.0):.6f}")
        synth_metrics_table.add_row(
            "Input Tokens", str(synthesizer_metrics.get("input_tokens", "N/A"))
        )
        synth_metrics_table.add_row(
            "Output Tokens", str(synthesizer_metrics.get("output_tokens", "N/A"))
        )
        synth_metrics_table.add_row(
            "API Latency", f"{synthesizer_metrics.get('api_latency_ms', 'N/A')} ms"
        )
        synth_metrics_table.add_row(
            "API Model Used", str(synthesizer_metrics.get("api_model_id_used", "N/A"))
        )
        console_instance.print(synth_metrics_table)

    console_instance.print(Rule("[bold]Overall Metrics for Entire Operation[/bold]"))
    total_metrics = results_data.get("total_metrics", {})
    # Reuse Rich Table for overall metrics
    overall_metrics_table = Table(box=box.SIMPLE, show_header=False, title_justify="left")
    overall_metrics_table.add_column("Metric", style="cyan")
    overall_metrics_table.add_column("Value", style="white")
    overall_metrics_table.add_row("Total Cost", f"${total_metrics.get('total_cost', 0.0):.6f}")
    overall_metrics_table.add_row(
        "Total Input Tokens (All Calls)", str(total_metrics.get("total_input_tokens", "N/A"))
    )
    overall_metrics_table.add_row(
        "Total Output Tokens (All Calls)", str(total_metrics.get("total_output_tokens", "N/A"))
    )
    overall_metrics_table.add_row(
        "Overall Task Time", f"{total_metrics.get('overall_task_time_ms', 'N/A')} ms"
    )
    console_instance.print(overall_metrics_table)
    console_instance.print()

    # Display the full prompt sent to the synthesizer model
    if storage_path and results_data.get("status") in ["SUCCESS", "PARTIAL_SUCCESS"]:
        synthesis_prompt_file = Path(storage_path) / "synthesis_prompt.md"
        if synthesis_prompt_file.exists():
            try:
                synthesis_prompt_content = synthesis_prompt_file.read_text(encoding="utf-8")
                console_instance.print(
                    Rule("[bold yellow]Full Prompt to Synthesizer Model[/bold yellow]")
                )
                console_instance.print(
                    Panel(
                        Syntax(
                            synthesis_prompt_content,
                            "markdown",
                            theme="monokai",
                            line_numbers=True,
                            word_wrap=True,
                        ),
                        title="[bold]Synthesizer Input Prompt[/bold]",
                        border_style="yellow",
                        expand=False,  # Keep it collapsed by default as it can be long
                    )
                )
            except Exception as e:
                logger.warning(f"Could not read or display synthesis_prompt.md: {e}", exc_info=True)
        else:
            logger.info("synthesis_prompt.md not found, skipping display.")


async def run_single_shot_demo(tracker: CostTracker, args: argparse.Namespace):
    console.print(Rule(f"[bold blue]Single-Shot Synthesis Demo - Task: {args.name}[/bold blue]"))
    console.print(
        f"Prompt: [yellow]{escape(args.prompt[:100] + ('...' if len(args.prompt) > 100 else ''))}[/yellow]"
    )
    console.print(f"Task Type: [magenta]{args.type}[/magenta]")

    expert_configs_for_tool: List[Dict[str, Any]] = []
    for model_id_str in args.expert_models:
        default_mc = next(
            (mc for mc in DEFAULT_EXPERT_MODEL_CONFIGS_SSS if mc["model_id"] == model_id_str), None
        )
        if default_mc:
            expert_configs_for_tool.append(default_mc.copy())  # Use copy
        else:
            expert_configs_for_tool.append({"model_id": model_id_str})

    synthesizer_config_for_tool: Dict[str, Any] = {"model_id": args.synthesizer_model}
    if args.synthesizer_model == DEFAULT_SYNTHESIZER_MODEL_CONFIG_SSS["model_id"]:
        synthesizer_config_for_tool = DEFAULT_SYNTHESIZER_MODEL_CONFIG_SSS.copy()
    elif args.synthesizer_model == FALLBACK_SYNTHESIZER_MODEL_CONFIG_SSS["model_id"]:
        synthesizer_config_for_tool = FALLBACK_SYNTHESIZER_MODEL_CONFIG_SSS.copy()

    console.print(
        f"Expert Models: [cyan]{', '.join([mc['model_id'] for mc in expert_configs_for_tool])}[/cyan]"
    )
    console.print(f"Synthesizer Model: [cyan]{synthesizer_config_for_tool['model_id']}[/cyan]")

    # Tool expects "expert_models" and "synthesizer_model" as per Pydantic aliases
    synthesis_input_for_tool = {
        "name": args.name,
        "prompt": args.prompt,
        "expert_models": expert_configs_for_tool,
        "synthesizer_model": synthesizer_config_for_tool,
        "tournament_type": args.type,
        # "synthesis_instructions": "Please synthesize these for clarity and impact..."
    }

    try:
        logger.info(f"Calling single_shot_synthesis tool for task: {args.name}", emoji_key="gear")

        console.print(
            Panel(
                f"Initiating Single-Shot Synthesis task: '[bold]{escape(args.name)}[/bold]'.\\n"
                f"This involves parallel calls to [cyan]{len(expert_configs_for_tool)}[/cyan] expert model(s) "
                f"followed by the synthesizer model ([cyan]{synthesizer_config_for_tool['model_id']}[/cyan]).\\n"
                f"Prompt: '{escape(args.prompt[:150] + ('...' if len(args.prompt) > 150 else ''))}'\\n"
                "[italic]Please wait, this may take a few moments...[/italic]",
                title="[bold blue]🚀 Starting Synthesis Process[/bold blue]",
                border_style="blue",
                expand=False,
            )
        )

        synthesis_data_dict: Optional[Dict[str, Any]] = None
        with console.status(
            "[bold yellow]Processing synthesis request via single_shot_synthesis tool...",
            spinner="dots",
        ):
            # The tool 'single_shot_synthesis' is already registered with the gateway
            synthesis_result_raw = await gateway.mcp.call_tool(
                "single_shot_synthesis", synthesis_input_for_tool
            )

        # Process the result (moved out of the status context)
        if isinstance(synthesis_result_raw, dict):
            logger.info(
                "Tool call returned a dictionary directly. Using it as result.", emoji_key="package"
            )
            synthesis_data_dict = synthesis_result_raw
        elif isinstance(synthesis_result_raw, list):
            logger.info(
                "Tool call returned a list. Processing its first element.", emoji_key="package"
            )
            if synthesis_result_raw:
                first_element = synthesis_result_raw[0]
                if isinstance(first_element, dict):
                    synthesis_data_dict = first_element
                elif hasattr(first_element, "text") and isinstance(first_element.text, str):
                    logger.info(
                        "First element has a .text attribute (like TextContent). Attempting to parse its .text attribute as JSON.",
                        emoji_key="memo",
                    )
                    try:
                        synthesis_data_dict = json.loads(first_element.text)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"JSON parsing of .text attribute failed: {e}. Falling back to process_mcp_result with the .text content.",
                            emoji_key="warning",
                        )
                        synthesis_data_dict = await process_mcp_result(
                            first_element.text
                        )  # Pass the string for LLM repair
                else:
                    logger.warning(
                        f"Tool call returned a list, but its first element is not a dictionary or TextContent-like. Content: {synthesis_result_raw!r:.500}",
                        emoji_key="warning",
                    )
                    synthesis_data_dict = await process_mcp_result(
                        synthesis_result_raw
                    )  # Fallback with the whole list
            else:
                logger.warning(
                    "Tool call returned an empty list. Falling back to process_mcp_result.",
                    emoji_key="warning",
                )
                synthesis_data_dict = await process_mcp_result(synthesis_result_raw)  # Fallback
        elif isinstance(synthesis_result_raw, str):  # If it's a string, try to parse
            logger.info(
                "Tool call returned a string. Attempting to parse with process_mcp_result.",
                emoji_key="memo",
            )
            synthesis_data_dict = await process_mcp_result(synthesis_result_raw)
        else:  # If it's some other type, log and try process_mcp_result
            logger.warning(
                f"Tool call returned an unexpected type: {type(synthesis_result_raw)}. Attempting to process with process_mcp_result.",
                emoji_key="warning",
            )
            synthesis_data_dict = await process_mcp_result(synthesis_result_raw)

        # Check for errors from the tool call itself or if the synthesis_data_dict is problematic
        if (
            not synthesis_data_dict
            or not isinstance(synthesis_data_dict, dict)
            or synthesis_data_dict.get("success", True) is False
            or (
                synthesis_data_dict.get("status") == "FAILED"
                and synthesis_data_dict.get("error_message")
            )
        ):
            error_msg = "Unknown error or empty/invalid data from synthesis tool call."
            if synthesis_data_dict and isinstance(synthesis_data_dict, dict):
                error_msg = synthesis_data_dict.get(
                    "error_message", synthesis_data_dict.get("error", error_msg)
                )

            logger.error(
                f"Single-shot synthesis tool call failed or returned invalid data: {error_msg}",
                emoji_key="cross_mark",
            )
            console.print(
                f"[bold red]Error from synthesis tool call:[/bold red] {escape(error_msg)}"
            )
            # Still attempt to display partial data if the structure is somewhat intact
            if synthesis_data_dict and isinstance(synthesis_data_dict, dict):
                display_single_shot_synthesis_results(synthesis_data_dict, args.prompt, console)
            else:
                console.print(
                    Panel(
                        "[bold red]Received no usable data from the synthesis tool.[/bold red]",
                        border_style="red",
                    )
                )
            return 1

        console.print(Rule("[bold green]✔️ Synthesis Process Completed[/bold green]"))
        # Pass the original prompt (args.prompt) to the display function
        display_single_shot_synthesis_results(synthesis_data_dict, args.prompt, console)

        # Cost tracking
        total_metrics = synthesis_data_dict.get("total_metrics", {})
        cost = total_metrics.get("total_cost", 0.0)
        input_tokens = total_metrics.get("total_input_tokens", 0)
        output_tokens = total_metrics.get("total_output_tokens", 0)

        # For tracker, provider/model is ambiguous for the whole operation, use task name
        tracker.record_call(
            cost=cost,
            provider="synthesis_tool_operation",
            model=args.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    except (ToolError, ProviderError, Exception) as e:
        logger.error(
            f"An error occurred during the single-shot synthesis demo: {e}",
            exc_info=True,
            emoji_key="error",
        )
        console.print(f"[bold red]Demo Error:[/bold red] {escape(str(e))}")
        return 1
    finally:
        tracker.display_summary(console)
        logger.info("Single-shot synthesis demo finished.", emoji_key="party_popper")
    return 0


async def main_async_sss():
    args = parse_arguments_sss()
    tracker = CostTracker()
    exit_code = 1
    try:
        await setup_gateway_for_demo_sss()
        exit_code = await run_single_shot_demo(tracker, args)
    except Exception as e:
        console.print(
            f"[bold red]Critical error in demo setup or execution:[/bold red] {escape(str(e))}"
        )
        logger.critical(f"Demo main_async_sss failed: {e}", exc_info=True)
    finally:
        logger.info("Demo finished.")
    return exit_code


if __name__ == "__main__":
    try:
        final_exit_code = asyncio.run(main_async_sss())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Demo interrupted by user.[/bold yellow]")
        final_exit_code = 130
    sys.exit(final_exit_code)
