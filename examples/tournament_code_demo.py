#!/usr/bin/env python3
"""
Tournament Code Demo - Demonstrates running a code improvement tournament

This script shows how to:
1. Create a tournament with multiple models, including diversity and evaluators.
2. Track progress across multiple rounds.
3. Retrieve and analyze the improved code and evaluation scores.

The tournament task is to write and iteratively improve a Python function for
parsing messy CSV data, handling various edge cases.

Usage:
  python examples/tournament_code_demo.py [--task TASK] [--rounds N]

Options:
  --task TASK       Specify a coding task (default: parse_csv)
  --rounds N        Number of tournament rounds (default: 2)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional  # Added List

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax  # For displaying code

# Add these imports to fix undefined names
from ultimate_mcp_server.core.models.tournament import TournamentStatus

# Assuming Gateway, PromptTemplate, etc. are correctly located
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.exceptions import ProviderError, ToolError
from ultimate_mcp_server.services.prompts import PromptTemplate  # If used

# Import tournament tools for manual registration
from ultimate_mcp_server.tools.tournament import (
    create_tournament,
    get_tournament_results,
    get_tournament_status,
)
from ultimate_mcp_server.utils import (
    get_logger,
    process_mcp_result,
)  # process_mcp_result might need updates
from ultimate_mcp_server.utils.display import (  # Ensure these are updated for new TournamentData structure
    CostTracker,
    display_tournament_results,  # This will need significant updates
    display_tournament_status,  # This likely needs updates too
)
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger
logger = get_logger("example.tournament_code")

# Global gateway instance (initialized in setup_gateway)
gateway: Optional[Gateway] = None

# --- Configuration ---
DEFAULT_MODEL_CONFIGS: List[Dict[str, Any]] = [
    {
        "model_id": "openai/gpt-4o-mini",  # Example, use your actual model IDs
        "diversity_count": 2,  # Generate 2 variants from this model
        "temperature": 0.7,
    },
    {
        "model_id": "anthropic/claude-3-5-haiku-20241022",
        "diversity_count": 1,
        "temperature": 0.6,
    },
    # Add more models as available/desired
    # {
    #     "model_id": "google/gemini-1.5-flash-latest",
    #     "diversity_count": 1,
    # },
]
DEFAULT_NUM_ROUNDS = 2
DEFAULT_TOURNAMENT_NAME = "Advanced Code Improvement Tournament"

# Default Evaluators
DEFAULT_EVALUATORS: List[Dict[str, Any]] = [
    {
        "evaluator_id": "python_syntax_checker",
        "type": "regex_match",
        "params": {
            "patterns": [r"^\s*def\s+\w+\(.*\):|^\s*class\s+\w+:"],  # Basic check for def/class
            "target_field": "extracted_code",
            "match_mode": "any_can_match",
        },
        "weight": 0.2,
        "primary_metric": False,
    },
    {
        "evaluator_id": "code_length_penalty",  # Example: Penalize overly short/long code
        "type": "regex_match",  # Could be a custom evaluator
        "params": {
            # This regex means: content has between 5 and 500 lines (approx)
            "patterns": [r"^(?:[^\n]*\n){4,499}[^\n]*$"],
            "target_field": "extracted_code",
            "match_mode": "all_must_match",  # Must be within the line range
            "regex_flag_options": ["MULTILINE", "DOTALL"],
        },
        "weight": 0.1,
    },
    {
        "evaluator_id": "llm_code_grader",
        "type": "llm_grader",
        "params": {
            "model_id": "anthropic/claude-3-5-haiku-20241022",  # Use a cost-effective grader
            "rubric": (
                "Evaluate the provided Python code based on the original prompt. "
                "Score from 0-100 considering: \n"
                "1. Correctness & Robustness (does it likely solve the problem, handle edges?).\n"
                "2. Efficiency (algorithmic complexity, resource usage).\n"
                "3. Readability & Maintainability (clarity, comments, Pythonic style).\n"
                "4. Completeness (are all requirements addressed?).\n"
                "Provide a 'Score: XX' line and a brief justification."
            ),
        },
        "weight": 0.7,  # Main evaluator
        "primary_metric": True,
    },
]


# The generic code prompt template
TEMPLATE_CODE = """
# GENERIC CODE TOURNAMENT PROMPT TEMPLATE

Write a {{code_type}} that {{task_description}}.

{{context}}

Your solution should:
{% for requirement in requirements %}
{{ loop.index }}. {{requirement}}
{% endfor %}

{% if example_inputs %}
Example inputs:
```
{{example_inputs}}
```
{% endif %}

{% if example_outputs %}
Expected outputs:
```
{{example_outputs}}
```
{% endif %}

Provide ONLY the Python code for your solution, enclosed in triple backticks (```python ... ```).
No explanations before or after the code block, unless they are comments within the code itself.
"""

# Define predefined tasks
TASKS = {
    "parse_csv": {
        "code_type": "Python function",
        "task_description": "parses a CSV string that may use different delimiters and contains various edge cases",
        "context": "Your function should be robust enough to handle real-world messy CSV data.",
        "requirements": [
            "Implement `parse_csv_string(csv_data: str) -> list[dict]`",
            "Accept a string `csv_data` which might contain CSV data",
            "Automatically detect the delimiter (comma, semicolon, or tab)",
            "Handle quoted fields correctly, including escaped quotes within fields",
            "Treat the first row as the header",
            "Return a list of dictionaries, where each dictionary represents a row",
            "Handle errors gracefully by logging warnings and skipping problematic rows",
            "Return an empty list if the input is empty or only contains a header",
            "Include necessary imports (e.g., `csv`, `io`).",
            "Be efficient for moderately large inputs (e.g., up to 1MB).",
        ],
        "example_inputs": """name,age,city\n"Smith, John",42,New York\n"Doe, Jane";39;"Los Angeles, CA"\n"\\"Williams\\", Bob"\t65\t"Chicago" """,
        "example_outputs": """[\n    {"name": "Smith, John", "age": "42", "city": "New York"},\n    {"name": "Doe, Jane", "age": "39", "city": "Los Angeles, CA"},\n    {"name": "\\"Williams\\", Bob", "age": "65", "city": "Chicago"}\n]""",
    },
    # Add other tasks (calculator, string_util) here if needed, similar structure
}


def create_custom_task_variables(task_description_custom: str):
    return {
        "code_type": "Python function",
        "task_description": task_description_custom,
        "context": "Ensure your solution is well-documented and handles potential edge cases.",
        "requirements": [
            "Implement the solution as specified in the task description.",
            "Write clean, readable, and efficient Python code.",
            "Include type hints and comprehensive docstrings.",
            "Handle potential errors gracefully.",
            "Make sure all necessary imports are included.",
        ],
        "example_inputs": "# Provide relevant example inputs if applicable",
        "example_outputs": "# Provide expected outputs for the examples if applicable",
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a code improvement tournament demo")
    parser.add_argument(
        "--task",
        type=str,
        default="parse_csv",
        choices=list(TASKS.keys()) + ["custom"],
        help="Coding task (default: parse_csv)",
    )
    parser.add_argument(
        "--custom-task", type=str, help="Custom coding task description (used when --task=custom)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_NUM_ROUNDS,
        help=f"Number of tournament rounds (default: {DEFAULT_NUM_ROUNDS})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[mc["model_id"] for mc in DEFAULT_MODEL_CONFIGS],  # Pass only model_id strings
        help="List of model IDs to participate (e.g., 'openai/gpt-4o-mini'). Overrides default models.",
    )
    return parser.parse_args()


async def setup_gateway_for_demo():
    global gateway
    if gateway:
        return

    logger.info("Initializing gateway for code tournament demo...", emoji_key="rocket")
    # Assuming Gateway constructor and _initialize_providers are async
    # The actual Gateway might not need to be created this way if it's a singleton managed by server.py
    # For a standalone demo, this direct instantiation is okay.
    # For integration, you'd likely get the existing gateway instance.
    try:
        # This is a simplified setup. In a real server, Gateway might be a singleton.
        # Here, we create a new one for the demo.
        # Ensure your actual Gateway class can be instantiated and initialized like this.
        # from ultimate_mcp_server.core import get_gateway_instance, async_init_gateway
        # gateway = get_gateway_instance()
        # if not gateway:
        #     gateway = await async_init_gateway() # This sets the global instance

        # For a script, direct instantiation:
        gateway = Gateway(
            name="code_tournament_demo_gateway",
            register_tools=False,  # Changed: register_tools=False, removed load_all_tools
        )
        # In a script, you might need to manually initialize providers if not done by Gateway constructor
        if not gateway.providers:  # Check if providers are already initialized
            await gateway._initialize_providers()

        # Manually register tournament tools
        mcp = gateway.mcp
        mcp.tool()(create_tournament)
        mcp.tool()(get_tournament_status)
        mcp.tool()(get_tournament_results)
        logger.info("Manually registered tournament tools for the demo.")

    except Exception as e:
        logger.critical(f"Failed to initialize Gateway: {e}", exc_info=True)
        raise

    # Verify tournament tools are registered (they should be if register_tools=True and load_all_tools=True)
    # This check is more for sanity.
    mcp_tools = await gateway.mcp.list_tools()
    registered_tool_names = [t.name for t in mcp_tools]
    required_tournament_tools = [
        "create_tournament",
        "get_tournament_status",
        "get_tournament_results",
    ]
    missing_tools = [
        tool for tool in required_tournament_tools if tool not in registered_tool_names
    ]

    if missing_tools:
        logger.error(
            f"Gateway initialized, but required tournament tools are missing: {missing_tools}",
            emoji_key="error",
        )
        logger.info(f"Available tools: {registered_tool_names}")
        raise RuntimeError(f"Required tournament tools not registered: {missing_tools}")

    logger.success(
        "Gateway for demo initialized and tournament tools verified.", emoji_key="heavy_check_mark"
    )


async def poll_tournament_status_enhanced(
    tournament_id: str, storage_path: Optional[str] = None, interval: int = 10
) -> Optional[str]:
    logger.info(
        f"Polling status for tournament {tournament_id} (storage: {storage_path})...",
        emoji_key="hourglass",
    )
    final_states = [
        status.value
        for status in [
            TournamentStatus.COMPLETED,
            TournamentStatus.FAILED,
            TournamentStatus.CANCELLED,
        ]
    ]

    while True:
        status_input = {"tournament_id": tournament_id}
        status_result_raw = await gateway.mcp.call_tool("get_tournament_status", status_input)

        # Process MCP result to get the dictionary
        status_data_dict = await process_mcp_result(status_result_raw)  # Ensure this returns a dict

        if "error" in status_data_dict or not status_data_dict.get(
            "success", True
        ):  # Check for tool call error
            error_message = status_data_dict.get(
                "error_message", status_data_dict.get("error", "Unknown error fetching status")
            )
            if storage_path and "not found" in error_message.lower():
                # Fallback to direct file reading
                state_file = Path(storage_path) / "tournament_state.json"
                logger.debug(f"Tournament not found via API, trying direct read: {state_file}")
                if state_file.exists():
                    try:
                        direct_state = json.loads(state_file.read_text())
                        current_status = direct_state.get("status")
                        # Reconstruct a compatible status_data_dict for display
                        status_data_dict = {
                            "tournament_id": tournament_id,
                            "name": direct_state.get("name"),
                            "tournament_type": direct_state.get("config", {}).get(
                                "tournament_type"
                            ),
                            "status": current_status,
                            "current_round": direct_state.get("current_round", -1)
                            + 1,  # Adjust for display
                            "total_rounds": direct_state.get("config", {}).get("rounds", 0),
                            "progress_summary": f"Read from file. Round {direct_state.get('current_round', -1) + 1}.",
                            "created_at": direct_state.get("created_at"),
                            "updated_at": direct_state.get("updated_at"),
                            "error_message": direct_state.get("error_message"),
                        }
                        logger.info(f"Successfully read direct state from file: {current_status}")
                    except Exception as e:
                        logger.error(f"Error reading state file directly: {e}")
                        # Keep original error message
                else:
                    logger.warning(f"Fallback state file not found: {state_file}")
            else:  # Non-"not found" error or no storage path
                logger.error(
                    f"Error fetching tournament status: {error_message}", emoji_key="error"
                )
                return None  # Indicate polling error

        # Display status using the utility function (ensure it handles the new dict structure)
        display_tournament_status(status_data_dict)  # Expects a dict

        current_status_val = status_data_dict.get("status")
        if current_status_val in final_states:
            logger.success(
                f"Tournament {tournament_id} reached final state: {current_status_val}",
                emoji_key="heavy_check_mark",
            )
            return current_status_val

        await asyncio.sleep(interval)


# --- Robust result processing for demo ---
async def robust_process_mcp_result(result_raw, storage_path=None):
    from ultimate_mcp_server.utils import process_mcp_result

    try:
        processed = await process_mcp_result(result_raw)
        # If no error, or error is not about JSON, return as is
        if not processed.get("error") or "LLM repair" not in processed.get("error", ""):
            return processed
    except Exception as e:
        processed = {"error": f"Exception in process_mcp_result: {e}"}

    # Fallback: try to load from file if storage_path is provided
    if storage_path:
        state_file = Path(storage_path) / "tournament_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as file_e:
                return {"error": f"Failed to parse both API and file: {file_e}"}
    # Otherwise, return a clear error
    return {"error": f"API did not return JSON. Raw: {str(result_raw)[:200]}"}


async def run_code_tournament_demo(tracker: CostTracker, args: argparse.Namespace):
    if args.task == "custom":
        if not args.custom_task:
            console.print(
                "[bold red]Error:[/bold red] --custom-task description must be provided when --task=custom."
            )
            return 1
        task_name = "custom_task"
        task_vars = create_custom_task_variables(args.custom_task)
        task_description_log = args.custom_task
    elif args.task in TASKS:
        task_name = args.task
        task_vars = TASKS[task_name]
        task_description_log = task_vars["task_description"]
    else:  # Should not happen due to argparse choices
        console.print(f"[bold red]Error:[/bold red] Unknown task '{args.task}'.")
        return 1

    console.print(
        Rule(
            f"[bold blue]{DEFAULT_TOURNAMENT_NAME} - Task: {task_name.replace('_', ' ').title()}[/bold blue]"
        )
    )
    console.print(f"Task Description: [yellow]{escape(task_description_log)}[/yellow]")

    # Prepare model_configs based on CLI input or defaults
    # The tool now expects list of dicts for model_configs
    current_model_configs = []
    if args.models == [mc["model_id"] for mc in DEFAULT_MODEL_CONFIGS]:  # Default models used
        current_model_configs = DEFAULT_MODEL_CONFIGS
        console.print(
            f"Using default models: [cyan]{', '.join([mc['model_id'] for mc in current_model_configs])}[/cyan]"
        )
    else:  # Custom models from CLI
        for model_id_str in args.models:
            # Find if this model_id_str matches any default config to get its diversity/temp
            # This is a simple way; a more complex CLI could allow full ModelConfig dicts
            default_mc = next(
                (mc for mc in DEFAULT_MODEL_CONFIGS if mc["model_id"] == model_id_str), None
            )
            if default_mc:
                current_model_configs.append(default_mc)
            else:  # Model from CLI not in defaults, use basic config
                current_model_configs.append({"model_id": model_id_str, "diversity_count": 1})
        console.print(f"Using CLI specified models: [cyan]{', '.join(args.models)}[/cyan]")

    console.print(f"Rounds: [cyan]{args.rounds}[/cyan]")
    console.print(
        f"Evaluators: [cyan]{', '.join([e['evaluator_id'] for e in DEFAULT_EVALUATORS])}[/cyan]"
    )

    code_prompt_template = PromptTemplate(
        template=TEMPLATE_CODE,
        template_id="demo_code_prompt",
        required_vars=[
            "code_type",
            "task_description",
            "context",
            "requirements",
            "example_inputs",
            "example_outputs",
        ],
    )
    try:
        initial_prompt = code_prompt_template.render(task_vars)
    except Exception as e:
        logger.error(f"Failed to render prompt template: {e}", exc_info=True)
        return 1

    console.print(
        Panel(
            escape(initial_prompt[:500] + "..."),
            title="[bold]Initial Prompt Preview[/bold]",
            border_style="dim",
        )
    )

    create_input = {
        "name": f"{DEFAULT_TOURNAMENT_NAME} - {task_name.replace('_', ' ').title()}",
        "prompt": initial_prompt,
        "models": current_model_configs,  # This should be List[Dict] for the tool
        "rounds": args.rounds,
        "tournament_type": "code",
        "evaluators": DEFAULT_EVALUATORS,  # Pass evaluator configs
        # Add other new config params if desired, e.g., extraction_model_id
        "extraction_model_id": "anthropic/claude-3-5-haiku-20241022",
        "max_retries_per_model_call": 2,
        "max_concurrent_model_calls": 3,
    }

    tournament_id: Optional[str] = None
    storage_path: Optional[str] = None

    try:
        logger.info("Creating code tournament...", emoji_key="gear")
        create_result_raw = await gateway.mcp.call_tool("create_tournament", create_input)
        create_data = await process_mcp_result(
            create_result_raw
        )  # process_mcp_result must return dict

        # Corrected error handling, similar to tournament_text_demo.py
        if "error" in create_data:
            error_msg = create_data.get(
                "error_message", create_data.get("error", "Unknown error creating tournament")
            )
            logger.error(f"Failed to create tournament: {error_msg}", emoji_key="cross_mark")
            console.print(f"[bold red]Error creating tournament:[/bold red] {escape(error_msg)}")
            return 1

        tournament_id = create_data.get("tournament_id")
        storage_path = create_data.get("storage_path")  # Get storage_path

        if not tournament_id:
            logger.error(
                "No tournament ID returned from create_tournament call.", emoji_key="cross_mark"
            )
            console.print("[bold red]Error: No tournament ID returned.[/bold red]")
            return 1

        console.print(
            f"Tournament [bold green]'{create_input['name']}'[/bold green] created successfully!"
        )
        console.print(f"  ID: [yellow]{tournament_id}[/yellow]")
        console.print(f"  Status: [magenta]{create_data.get('status')}[/magenta]")
        if storage_path:
            console.print(f"  Storage Path: [blue underline]{storage_path}[/blue underline]")

        await asyncio.sleep(1)  # Brief pause for task scheduling

        final_status_val = await poll_tournament_status_enhanced(
            tournament_id, storage_path, interval=10
        )

        if final_status_val == TournamentStatus.COMPLETED.value:
            logger.info(
                f"Tournament {tournament_id} completed. Fetching final results...",
                emoji_key="sports_medal",
            )
            results_input = {"tournament_id": tournament_id}
            results_raw = await gateway.mcp.call_tool("get_tournament_results", results_input)
            processed_results_dict = await robust_process_mcp_result(results_raw, storage_path)

            results_data_dict = processed_results_dict
            workaround_applied_successfully = False

            # If process_mcp_result itself signals an error
            # (This will be true if JSON parsing failed and LLM repair also failed to produce valid JSON)
            if (
                "error" in processed_results_dict
            ):  # Simpler check for any error from process_mcp_result
                original_error_msg = processed_results_dict.get(
                    "error", "Unknown error processing results"
                )
                logger.warning(
                    f"Initial processing of 'get_tournament_results' failed with: {original_error_msg}"
                )

                # Attempt workaround if it's a code tournament, storage path is known,
                # AND the initial processing via MCP failed.
                current_tournament_type = create_input.get("tournament_type", "unknown")
                if current_tournament_type == "code" and storage_path:
                    logger.info(
                        f"Applying workaround for 'get_tournament_results' failure. "
                        f"Attempting to load results directly from storage: {storage_path}"
                    )
                    state_file_path = Path(storage_path) / "tournament_state.json"
                    if state_file_path.exists():
                        try:
                            with open(state_file_path, "r", encoding="utf-8") as f:
                                results_data_dict = json.load(f)  # Override with data from file
                            logger.success(
                                f"Workaround successful: Loaded results from {state_file_path}"
                            )
                            workaround_applied_successfully = True
                        except Exception as e:
                            logger.error(
                                f"Workaround failed: Could not load or parse {state_file_path}: {e}"
                            )
                            # results_data_dict remains processed_results_dict (the error dict from initial processing)
                    else:
                        logger.warning(
                            f"Workaround failed: State file not found at {state_file_path}"
                        )
                        # results_data_dict remains processed_results_dict (the error dict from initial processing)
                # If not a code tournament, or no storage path, or workaround failed,
                # results_data_dict is still the original error dict from processed_results_dict

            # Now, check the final results_data_dict (either from tool or successful workaround)
            # This outer check sees if results_data_dict *still* has an error after potential workaround
            if "error" in results_data_dict:
                # This block will be hit if:
                # 1. Original tool call failed AND it wasn't the specific known issue for the workaround.
                # 2. Original tool call failed with the known issue, BUT the workaround also failed (e.g., file not found, parse error).
                final_error_msg = results_data_dict.get(
                    "error_message", results_data_dict.get("error", "Unknown error")
                )
                logger.error(
                    f"Failed to get tournament results (workaround_applied_successfully={workaround_applied_successfully}): {final_error_msg}",
                    emoji_key="cross_mark",
                )
                console.print(
                    f"[bold red]Error fetching results:[/bold red] {escape(final_error_msg)}"
                )
            else:
                # Successfully got data, either from tool or workaround
                if workaround_applied_successfully:
                    console.print(
                        "[yellow i](Workaround applied: Results loaded directly from tournament_state.json)[/yellow i]"
                    )

                # Pass the full dictionary results_data_dict to display_tournament_results
                display_tournament_results(
                    results_data_dict, console
                )  # Ensure this function handles the new structure

                # Example of accessing overall best response
                overall_best_resp_data = results_data_dict.get("overall_best_response")
                if overall_best_resp_data:
                    console.print(
                        Rule("[bold green]Overall Best Response Across All Rounds[/bold green]")
                    )
                    best_variant_id = overall_best_resp_data.get("model_id_variant", "N/A")
                    best_score = overall_best_resp_data.get("overall_score", "N/A")
                    console.print(
                        f"Best Variant: [cyan]{best_variant_id}[/cyan] (Score: {best_score:.2f if isinstance(best_score, float) else 'N/A'})"
                    )
                    best_code = overall_best_resp_data.get("extracted_code")
                    if best_code:
                        console.print(
                            Panel(
                                Syntax(best_code, "python", theme="monokai", line_numbers=True),
                                title=f"Code from {best_variant_id}",
                                border_style="green",
                            )
                        )
                    else:
                        console.print(
                            "[yellow]No extracted code found for the overall best response.[/yellow]"
                        )

                # Try to find and mention the leaderboard file from the last round
                last_round_num = results_data_dict.get("config", {}).get("rounds", 0) - 1
                if last_round_num >= 0 and last_round_num < len(
                    results_data_dict.get("rounds_results", [])
                ):
                    last_round_data = results_data_dict["rounds_results"][last_round_num]
                    leaderboard_file = last_round_data.get("leaderboard_file_path")
                    if leaderboard_file:
                        console.print(
                            f"\nCheck the final leaderboard: [blue underline]{leaderboard_file}[/blue underline]"
                        )
                    comparison_file = last_round_data.get("comparison_file_path")
                    if comparison_file:
                        console.print(
                            f"Check the final round comparison: [blue underline]{comparison_file}[/blue underline]"
                        )

        elif final_status_val:  # FAILED or CANCELLED
            logger.warning(
                f"Tournament {tournament_id} ended with status: {final_status_val}",
                emoji_key="warning",
            )
            console.print(
                f"[bold yellow]Tournament ended with status: {final_status_val}[/bold yellow]"
            )
            # Optionally fetch results for FAILED tournaments to see partial data / error
            if final_status_val == TournamentStatus.FAILED.value:
                results_input = {"tournament_id": tournament_id}
                results_raw = await gateway.mcp.call_tool("get_tournament_results", results_input)
                results_data_dict = await robust_process_mcp_result(results_raw, storage_path)
                if results_data_dict and not results_data_dict.get(
                    "error_message"
                ):  # Check success of get_tournament_results
                    display_tournament_results(results_data_dict, console)  # Display what we have
        else:  # Polling failed
            logger.error(f"Polling failed for tournament {tournament_id}.", emoji_key="cross_mark")
            console.print(f"[bold red]Polling failed for tournament {tournament_id}.[/bold red]")

    except (ToolError, ProviderError, Exception) as e:  # Catch more general errors
        logger.error(
            f"An error occurred during the code tournament demo: {e}",
            exc_info=True,
            emoji_key="error",
        )
        console.print(f"[bold red]Demo Error:[/bold red] {escape(str(e))}")
        return 1
    finally:
        tracker.display_summary(console)
        logger.info("Code tournament demo finished.", emoji_key="party_popper")
    return 0


async def main_async():
    args = parse_arguments()
    tracker = CostTracker()
    exit_code = 1  # Default to error
    try:
        await setup_gateway_for_demo()
        exit_code = await run_code_tournament_demo(tracker, args)
    except Exception as e:
        console.print(
            f"[bold red]Critical error in demo setup or execution:[/bold red] {escape(str(e))}"
        )
        logger.critical(f"Demo main_async failed: {e}", exc_info=True)
    finally:
        # Simplified cleanup, similar to tournament_text_demo.py
        if gateway:
            # Perform any necessary general gateway cleanup if available in the future
            # For now, specific sandbox closing is removed as it caused issues and
            # repl_python is not explicitly registered/used by this demo with register_tools=False
            pass
        logger.info("Demo finished.")
    return exit_code


if __name__ == "__main__":
    try:
        final_exit_code = asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Demo interrupted by user.[/bold yellow]")
        final_exit_code = 130  # Standard exit code for Ctrl+C
    sys.exit(final_exit_code)
