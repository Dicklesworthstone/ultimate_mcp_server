#!/usr/bin/env python
import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


def _fmt_id(val: Any, length: int = 8) -> str:
    """Return a short id string safe for logs."""
    s = str(val) if val is not None else "?"
    # Ensure slicing doesn't go out of bounds if string is shorter than length
    return s[: min(length, len(s))]


# --- Project Setup ---
# Add project root to path for imports when running as script
# Adjust this path if your script location relative to the project root differs
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    # Navigate up until we find a directory likely containing the project modules
    PROJECT_ROOT = SCRIPT_DIR
    while (
        not (PROJECT_ROOT / "ultimate_mcp_server").is_dir()
        and not (PROJECT_ROOT / "ultimate_mcp_client").is_dir()
        and PROJECT_ROOT.parent != PROJECT_ROOT
    ):  # Prevent infinite loop
        PROJECT_ROOT = PROJECT_ROOT.parent

    if (
        not (PROJECT_ROOT / "ultimate_mcp_server").is_dir()
        and not (PROJECT_ROOT / "ultimate_mcp_client").is_dir()
    ):
        print(
            f"Error: Could not reliably determine project root from {SCRIPT_DIR}.", file=sys.stderr
        )
        # Fallback: Add script dir anyway, maybe it's flat structure
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
            print(
                f"Warning: Added script directory {SCRIPT_DIR} to path as fallback.",
                file=sys.stderr,
            )
        else:
            sys.exit(1)  # Give up if markers not found after traversing up

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

except Exception as e:
    print(f"Error setting up sys.path: {e}", file=sys.stderr)
    sys.exit(1)

from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.pretty import pretty_repr  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.traceback import install as install_rich_traceback  # noqa: E402

from ultimate_mcp_server.config import get_config  # noqa: E402

# Tools and related components from unified_memory
from ultimate_mcp_server.tools.unified_memory_system import (  # noqa: E402
    ActionStatus,
    ActionType,
    ArtifactType,
    # Utilities/Enums/Exceptions needed
    DBConnection,
    LinkType,
    MemoryLevel,
    MemoryType,
    ThoughtType,
    ToolError,
    ToolInputError,
    # Action Dependency Tools
    add_action_dependency,
    auto_update_focus,
    compute_memory_statistics,
    consolidate_memories,
    create_memory_link,
    # Thought
    create_thought_chain,
    # Workflow
    create_workflow,
    delete_expired_memories,
    focus_memory,
    generate_reflection,
    generate_workflow_report,
    get_action_dependencies,
    get_action_details,
    get_artifact_by_id,
    get_artifacts,
    get_linked_memories,
    get_memory_by_id,
    get_recent_actions,
    get_thought_chain,
    get_workflow_context,
    get_workflow_details,
    # Working Memory / State
    get_working_memory,
    hybrid_search_memories,
    # Initialization
    initialize_memory_system,
    list_workflows,
    load_cognitive_state,
    optimize_working_memory,
    promote_memory_level,
    query_memories,
    record_action_completion,
    # Action
    record_action_start,
    # Artifacts
    record_artifact,
    record_thought,
    save_cognitive_state,
    search_semantic_memories,
    # Core Memory
    store_memory,
    summarize_text,
    update_memory,
    update_workflow_status,
    visualize_memory_network,
    visualize_reasoning_chain,
)

# Utilities from the project
from ultimate_mcp_server.utils import get_logger  # noqa: E402

console = Console()
logger = get_logger("demo.unified_memory")
config = get_config()  # Load config to ensure provider keys might be found

install_rich_traceback(show_locals=False, width=console.width)

DEMO_DB_FILE: Optional[str] = config.agent_memory.db_path  # Global to hold the DB path being used


async def safe_tool_call(func, args: Dict, description: str, suppress_output: bool = False):
    """Helper to call a tool function, catch errors, and display results."""
    display_title = not suppress_output
    display_args = not suppress_output
    display_result_panel = not suppress_output

    if display_title:
        title = f"DEMO: {description}"
        console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]", style="blue"))
    if display_args:
        if args:
            console.print(f"[dim]Calling [bold cyan]{func.__name__}[/] with args:[/]")
            try:
                # Filter out db_path if it matches the global default for cleaner logs
                args_to_print = {
                    k: v for k, v in args.items() if k != "db_path" or v != DEMO_DB_FILE
                }
                args_repr = pretty_repr(args_to_print, max_length=120, max_string=100)
            except Exception:
                args_repr = str(args)[:300]
            console.print(args_repr)
        else:
            console.print(f"[dim]Calling [bold cyan]{func.__name__}[/] (no arguments)[/]")

    start_time = time.monotonic()
    result = None
    try:
        # Use the global DEMO_DB_FILE if db_path isn't explicitly in args
        if "db_path" not in args and DEMO_DB_FILE:
            args["db_path"] = DEMO_DB_FILE

        result = await func(**args)

        processing_time = time.monotonic() - start_time
        log_func = getattr(logger, "debug", print)
        log_func(f"Tool '{func.__name__}' execution time: {processing_time:.4f}s")

        if display_result_panel:
            success = isinstance(result, dict) and result.get("success", False)
            panel_title = f"[bold {'green' if success else 'yellow'}]Result: {func.__name__} {'✅' if success else '❔'}[/]"
            panel_border = "green" if success else "yellow"

            try:
                # Handle specific large/complex outputs
                if func.__name__ == "generate_workflow_report" and result.get("report"):
                    report_preview = str(result["report"])[:500] + (
                        "..." if len(str(result["report"])) > 500 else ""
                    )
                    result_repr = f"Report Format: {result.get('format')}\nStyle: {result.get('style_used')}\nPreview:\n---\n{report_preview}\n---"
                elif func.__name__ in [
                    "visualize_reasoning_chain",
                    "visualize_memory_network",
                ] and result.get("visualization"):
                    viz_preview = str(result["visualization"])[:500] + (
                        "..." if len(str(result["visualization"])) > 500 else ""
                    )
                    result_repr = f"Visualization Format: {result.get('format')}\nContent Preview:\n---\n{viz_preview}\n---"
                elif func.__name__ == "summarize_text" and result.get("summary"):
                    summary_preview = str(result["summary"])[:500] + (
                        "..." if len(str(result["summary"])) > 500 else ""
                    )
                    result_repr = f"Summary Preview:\n---\n{summary_preview}\n---"
                elif func.__name__ == "consolidate_memories" and result.get("consolidated_content"):
                    content_preview = str(result["consolidated_content"])[:500] + (
                        "..." if len(str(result["consolidated_content"])) > 500 else ""
                    )
                    result_repr = f"Consolidated Content Preview:\n---\n{content_preview}\n---"
                elif func.__name__ == "generate_reflection" and result.get("content"):
                    content_preview = str(result["content"])[:500] + (
                        "..." if len(str(result["content"])) > 500 else ""
                    )
                    result_repr = f"Reflection Content Preview:\n---\n{content_preview}\n---"
                else:
                    result_repr = pretty_repr(result, max_length=200, max_string=150)
            except Exception:
                result_repr = f"(Could not represent result of type {type(result)} fully)\n{str(result)[:500]}"

            console.print(
                Panel(
                    escape(result_repr), title=panel_title, border_style=panel_border, expand=False
                )
            )

        return result

    except (ToolInputError, ToolError) as e:
        processing_time = time.monotonic() - start_time
        log_func_error = getattr(logger, "error", print)
        log_func_error(f"Tool '{func.__name__}' failed: {e}", exc_info=False)
        if display_result_panel:
            error_title = f"[bold red]Error: {func.__name__} Failed ❌[/]"
            error_content = f"[bold red]{type(e).__name__}:[/] {escape(str(e))}"
            details = None
            if hasattr(e, "details") and e.details:
                details = e.details
            elif hasattr(e, "context") and e.context:
                details = e.context

            if details:
                try:
                    details_repr = pretty_repr(details)
                except Exception:
                    details_repr = str(details)
                error_content += f"\n\n[yellow]Details:[/]\n{escape(details_repr)}"
            console.print(Panel(error_content, title=error_title, border_style="red", expand=False))
        return {
            "success": False,
            "error": str(e),
            "error_code": getattr(e, "error_code", "TOOL_ERROR"),
            "error_type": type(e).__name__,
            "details": details or {},
            "isError": True,
        }
    except Exception as e:
        processing_time = time.monotonic() - start_time
        log_func_critical = getattr(logger, "critical", print)
        log_func_critical(f"Unexpected error calling '{func.__name__}': {e}", exc_info=True)
        if display_result_panel:
            console.print(f"\n[bold red]CRITICAL UNEXPECTED ERROR in {func.__name__}:[/bold red]")
            console.print_exception(show_locals=False)
        return {
            "success": False,
            "error": f"Unexpected: {str(e)}",
            "error_code": "UNEXPECTED_ERROR",
            "error_type": type(e).__name__,
            "details": {"traceback": traceback.format_exc()},
            "isError": True,
        }
    finally:
        if display_title:
            console.print()


# --- Demo Setup & Teardown ---


async def setup_demo_environment():
    """Initialize the memory system using the DEFAULT database file."""
    global DEMO_DB_FILE
    DEMO_DB_FILE = config.agent_memory.db_path
    log_func_info = getattr(logger, "info", print)
    log_func_info(f"Using default database for demo: {DEMO_DB_FILE}")
    console.print(
        Panel(
            f"Using default database: [cyan]{DEMO_DB_FILE}[/]\n"
            f"[yellow]NOTE:[/yellow] This demo will operate on the actual development database.",
            title="Demo Setup",
            border_style="yellow",
        )
    )

    init_result = await safe_tool_call(
        initialize_memory_system,
        {"db_path": DEMO_DB_FILE},
        "Initialize Memory System",
    )
    if not init_result or not init_result.get("success"):
        console.print(
            "[bold red]CRITICAL:[/bold red] Failed to initialize memory system. Aborting demo."
        )
        console.print(
            "[yellow]Check DB access and potentially API key configuration/network if init requires them.[/yellow]"
        )
        await cleanup_demo_environment()
        sys.exit(1)


async def cleanup_demo_environment():
    """Close DB connection."""
    global DEMO_DB_FILE
    log_func_info = getattr(logger, "info", print)
    log_func_warn = getattr(logger, "warning", print)

    try:
        await DBConnection.close_connection()
        log_func_info("Closed database connection.")
    except Exception as e:
        log_func_warn(f"Error closing DB connection during cleanup: {e}")

    if DEMO_DB_FILE:
        log_func_info(f"Demo finished using database: {DEMO_DB_FILE}")
        console.print(f"Demo finished using database: [dim]{DEMO_DB_FILE}[/dim]")
        DEMO_DB_FILE = None


# --- Individual Demo Sections ---


# (Keep existing sections 1-8 as they are)
async def demonstrate_basic_workflows():
    """Demonstrate basic workflow CRUD and listing operations."""
    console.print(Rule("[bold green]1. Basic Workflow Operations[/bold green]", style="green"))

    # Create
    create_args = {
        "title": "Enhanced WF Demo",
        "goal": "Demonstrate core workflow, action, artifact, and memory linking.",
        "tags": ["enhanced", "demo", "core"],
    }
    wf_result = await safe_tool_call(create_workflow, create_args, "Create Enhanced Workflow")
    wf_id = wf_result.get("workflow_id") if wf_result.get("success") else None

    if not wf_id:
        console.print(
            "[bold red]CRITICAL DEMO FAILURE:[/bold red] Failed to create workflow. Cannot continue basic workflow demo."
        )
        return None  # Return None to signal failure

    # Get Details
    await safe_tool_call(
        get_workflow_details, {"workflow_id": wf_id}, f"Get Workflow Details ({_fmt_id(wf_id)})"
    )

    # List (should show the one we created)
    await safe_tool_call(list_workflows, {"limit": 5}, "List Workflows (Limit 5)")

    # List Filtered by Tag
    await safe_tool_call(list_workflows, {"tag": "enhanced"}, "List Workflows Tagged 'enhanced'")

    # Update Status (to active for subsequent steps)
    await safe_tool_call(
        update_workflow_status,
        {"workflow_id": wf_id, "status": "active"},
        f"Ensure Workflow Status is Active ({_fmt_id(wf_id)})",
    )

    return wf_id


async def demonstrate_basic_actions(wf_id: Optional[str]):
    """Demonstrate basic action recording, completion, and retrieval."""
    console.print(Rule("[bold green]2. Basic Action Operations[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping action demo: No valid workflow ID provided.[/yellow]")
        return {}  # Return empty dict

    action_ids = {}

    # Record Action 1 Start (e.g., Planning)
    start_args_1 = {
        "workflow_id": wf_id,
        "action_type": ActionType.PLANNING.value,
        "reasoning": "Initial planning phase for the enhanced demo.",
        "title": "Plan Demo Steps",
        "tags": ["planning"],
    }
    action_res_1 = await safe_tool_call(
        record_action_start, start_args_1, "Record Action 1 Start (Planning)"
    )
    action_id_1 = (
        action_res_1.get("action_id") if action_res_1 and action_res_1.get("success") else None
    )  # More robust check
    if not action_id_1:
        console.print(
            "[bold red]CRITICAL DEMO FAILURE:[/bold red] Failed to record start for Action 1. Aborting action demo."
        )
        return {}  # Return empty dict
    action_ids["action1_id"] = action_id_1

    # Record Action 1 Completion (Needs action_id_1, which is now checked)
    complete_args_1 = {
        "action_id": action_id_1,
        "status": ActionStatus.COMPLETED.value,
        "summary": "Planning complete. Next step: data simulation.",
    }
    await safe_tool_call(
        record_action_completion,
        complete_args_1,
        f"Record Action 1 Completion ({_fmt_id(action_id_1)})",
    )

    # Record Action 2 Start (e.g., Tool Use - simulated)
    start_args_2 = {
        "workflow_id": wf_id,
        "action_type": ActionType.TOOL_USE.value,
        "reasoning": "Simulating data generation based on the plan.",
        "tool_name": "simulate_data",
        "tool_args": {"rows": 100, "type": "random"},
        "title": "Simulate Demo Data",
        "tags": ["data", "simulation"],
        "parent_action_id": action_id_1,  # Link to planning action
    }
    action_res_2 = await safe_tool_call(
        record_action_start, start_args_2, "Record Action 2 Start (Simulate Data)"
    )
    action_id_2 = (
        action_res_2.get("action_id") if action_res_2 and action_res_2.get("success") else None
    )  # More robust check
    if action_id_2:
        action_ids["action2_id"] = action_id_2
        # Moved inside the 'if action_id_2:' block:
        await safe_tool_call(
            get_action_details,
            {"action_ids": [action_id_1, action_id_2]},  # Both IDs are valid here
            "Get Action Details (Multiple Actions)",
        )
        complete_args_2 = {
            "action_id": action_id_2,  # Now guaranteed to be non-None
            "status": ActionStatus.FAILED.value,
            "summary": "Simulation failed due to resource limit.",
            "tool_result": {"error": "Timeout", "code": 504},
        }
        await safe_tool_call(
            record_action_completion,
            complete_args_2,
            f"Record Action 2 Completion (Failed - {_fmt_id(action_id_2)})",
        )
    else:
        # Keep the existing else block for handling Action 2 start failure
        console.print(
            "[bold red]Action 2 failed to start. Skipping completion and dependency tests involving Action 2.[/bold red]"
        )
        # Ensure action_id_2 is not added to the dict if it's None
        if "action2_id" in action_ids:
            del action_ids["action2_id"]
        # Potentially skip dependency demo if action2_id is needed? (The demo logic does skip if action2_id is missing)

    # Get Action Details (Only Action 1 if Action 2 failed) - Moved outside check block
    if action_id_1 and not action_id_2:  # Only fetch Action 1 if Action 2 failed
        await safe_tool_call(
            get_action_details,
            {"action_id": action_id_1},
            f"Get Action Details (Action 1 Only - {_fmt_id(action_id_1)})",
        )

    # Get Recent Actions (should show both)
    await safe_tool_call(
        get_recent_actions, {"workflow_id": wf_id, "limit": 5}, "Get Recent Actions"
    )

    return action_ids


async def demonstrate_action_dependencies(wf_id: Optional[str], action_ids: Dict):
    """Demonstrate adding and retrieving action dependencies."""
    console.print(Rule("[bold green]3. Action Dependency Operations[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping dependency demo: No valid workflow ID.[/yellow]")
        return
    action1_id = action_ids.get("action1_id")
    action2_id = action_ids.get("action2_id")
    if not action1_id or not action2_id:
        console.print(
            "[yellow]Skipping dependency demo: Need at least two valid action IDs.[/yellow]"
        )
        return

    # Add Dependency (Action 2 requires Action 1)
    await safe_tool_call(
        add_action_dependency,
        {
            "source_action_id": action2_id,
            "target_action_id": action1_id,
            "dependency_type": "requires",
        },
        f"Add Dependency ({_fmt_id(action2_id)} requires {_fmt_id(action1_id)})",
    )

    # Get Dependencies for Action 1 (Should show Action 2 depends on it - Downstream)
    await safe_tool_call(
        get_action_dependencies,
        {"action_id": action1_id, "direction": "downstream"},
        f"Get Dependencies (Downstream of Action 1 - {_fmt_id(action1_id)})",
    )

    # Get Dependencies for Action 2 (Should show it depends on Action 1 - Upstream)
    await safe_tool_call(
        get_action_dependencies,
        {"action_id": action2_id, "direction": "upstream", "include_details": True},
        f"Get Dependencies (Upstream of Action 2 - {_fmt_id(action2_id)}, with Details)",
    )

    # Get Action 1 Details (Include Dependencies)
    await safe_tool_call(
        get_action_details,
        {"action_id": action1_id, "include_dependencies": True},
        f"Get Action 1 Details ({_fmt_id(action1_id)}), Include Dependencies",
    )


async def demonstrate_artifacts(wf_id: Optional[str], action_ids: Dict):
    """Demonstrate artifact recording and retrieval."""
    console.print(Rule("[bold green]4. Artifact Operations[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping artifact demo: No valid workflow ID provided.[/yellow]")
        return {}  # Return empty dict
    action1_id = action_ids.get("action1_id")
    action2_id = action_ids.get("action2_id")  # May be None if Action 2 failed

    artifact_ids = {}

    # Record Artifact 1 (e.g., Plan document from Action 1)
    artifact_args_1 = {
        "workflow_id": wf_id,
        "action_id": action1_id,
        "name": "demo_plan.txt",
        "artifact_type": ArtifactType.FILE.value,  # Use enum value
        "description": "Initial plan for the demo steps.",
        "path": "/path/to/demo_plan.txt",
        "content": "Step 1: Plan\nStep 2: Simulate\nStep 3: Analyze",  # Small content example
        "tags": ["planning", "document"],
        "is_output": False,
    }
    art_res_1 = await safe_tool_call(
        record_artifact, artifact_args_1, "Record Artifact 1 (Plan Doc)"
    )
    art_id_1 = (
        art_res_1.get("artifact_id") if art_res_1 and art_res_1.get("success") else None
    )  # Robust check
    if not art_id_1:
        console.print(
            "[bold red]DEMO WARNING:[/bold red] Failed to record Artifact 1. Subsequent steps needing art1_id might fail."
        )
        # Don't abort, but warn
    else:
        artifact_ids["art1_id"] = art_id_1

    # Record Artifact 2 (e.g., Error log from Action 2)
    artifact_args_2 = {
        "workflow_id": wf_id,
        "action_id": action2_id,
        "name": "simulation_error.log",
        "artifact_type": ArtifactType.TEXT.value,
        "description": "Error log from the failed data simulation.",
        "content": "ERROR: Timeout waiting for resource. Code 504.",
        "tags": ["error", "log", "simulation"],
    }
    art_res_2 = await safe_tool_call(
        record_artifact, artifact_args_2, "Record Artifact 2 (Error Log)"
    )
    art_id_2 = art_res_2.get("artifact_id") if art_res_2.get("success") else None

    # --- ADD CHECK before recording Artifact 2 ---
    if not action2_id:
        console.print(
            "[yellow]Skipping Artifact 2 recording: Action 2 ID is not available (likely failed to start).[/yellow]"
        )
    else:
        # Proceed with recording Artifact 2 only if action2_id exists
        artifact_args_2["action_id"] = action2_id  # Assign the valid ID
        art_res_2 = await safe_tool_call(
            record_artifact, artifact_args_2, "Record Artifact 2 (Error Log)"
        )
        art_id_2 = art_res_2.get("artifact_id") if art_res_2 and art_res_2.get("success") else None
        if art_id_2:
            artifact_ids["art2_id"] = art_id_2
        else:
            console.print("[bold red]DEMO WARNING:[/bold red] Failed to record Artifact 2.")

    # Get Artifacts (List all for workflow)
    await safe_tool_call(
        get_artifacts, {"workflow_id": wf_id, "limit": 5}, "Get Artifacts (List for Workflow)"
    )

    # Get Artifacts (Filter by tag 'planning')
    await safe_tool_call(
        get_artifacts,
        {"workflow_id": wf_id, "tag": "planning"},
        "Get Artifacts (Filter by Tag 'planning')",
    )

    # Get Artifact by ID (Get the plan doc)
    if art_id_1:
        await safe_tool_call(
            get_artifact_by_id,
            {"artifact_id": art_id_1, "include_content": True},
            f"Get Artifact by ID ({_fmt_id(art_id_1)}, Include Content)",
        )
    else:
        console.print(
            "[yellow]Skipping 'Get Artifact by ID' for Artifact 1 as it failed to record.[/yellow]"
        )

    return artifact_ids


async def demonstrate_thoughts_and_linking(
    wf_id: Optional[str], action_ids: Dict, artifact_ids: Dict
):
    """Demonstrate thought chains, recording thoughts, and linking them."""
    console.print(Rule("[bold green]5. Thought Operations & Linking[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping thought demo: No valid workflow ID provided.[/yellow]")
        return None
    action1_id = action_ids.get("action1_id")  # noqa: F841
    action2_id = action_ids.get("action2_id")  # Might be None
    art1_id = artifact_ids.get("art1_id")  # Might be None if artifact demo failed

    # Check if prerequisite artifact exists before linking
    if not art1_id:
        console.print(
            "[yellow]Skipping thought demo: Planning artifact ID (art1_id) not available.[/yellow]"
        )
        return None

    # Create a new thought chain
    chain_args = {
        "workflow_id": wf_id,
        "title": "Analysis Thought Chain",
        "initial_thought": "Review the plan artifact.",
        "initial_thought_type": ThoughtType.PLAN.value,
    }
    chain_res = await safe_tool_call(
        create_thought_chain, chain_args, "Create New Thought Chain (Analysis)"
    )
    chain_id = (
        chain_res.get("thought_chain_id") if chain_res and chain_res.get("success") else None
    )  # Robust check

    if not chain_id:
        console.print(
            "[bold red]CRITICAL DEMO FAILURE:[/bold red] Failed to create thought chain. Aborting thought demo."
        )
        return None

    # Record a thought linked to the plan artifact
    thought_args_1 = {
        "workflow_id": wf_id,
        "thought_chain_id": chain_id,
        "content": "The plan seems straightforward but lacks detail on simulation parameters.",
        "thought_type": ThoughtType.CRITIQUE.value,
        "relevant_artifact_id": art1_id,  # Link to the plan artifact
    }
    thought_res_1 = await safe_tool_call(
        record_thought, thought_args_1, "Record Thought (Critique Linked to Artifact)"
    )
    thought1_id = thought_res_1.get("thought_id") if thought_res_1.get("success") else None

    if not thought1_id:
        console.print(
            "[bold red]DEMO WARNING:[/bold red] Failed to record thought 1. Subsequent linked thought might fail or be unlinked."
        )

    # Record second thought (depends on action2_id, thought1_id)
    if not action2_id:
        console.print("[yellow]Skipping recording thought 2: Action 2 ID is missing.[/yellow]")
    elif not thought1_id:
        console.print("[yellow]Skipping recording thought 2: Thought 1 ID is missing.[/yellow]")
        # Record thought 2 without parent link if action2_id exists but thought1_id doesn't?
        thought_args_2_no_parent = {
            "workflow_id": wf_id,
            "thought_chain_id": chain_id,
            "content": "The simulation failure needs investigation. Was it transient or configuration?",
            "thought_type": ThoughtType.QUESTION.value,
            "relevant_action_id": action2_id,  # Action 2 ID exists here
        }
        await safe_tool_call(
            record_thought,
            thought_args_2_no_parent,
            "Record Thought (Question Linked to Action, NO PARENT)",
        )
    else:
        # Record another thought linked to the failed action
        thought_args_2 = {
            "workflow_id": wf_id,
            "thought_chain_id": chain_id,
            "content": "The simulation failure needs investigation. Was it transient or configuration?",
            "thought_type": ThoughtType.QUESTION.value,
            "relevant_action_id": action_ids.get("action2_id"),  # Link to failed action
            "parent_thought_id": thought1_id,  # Link to previous thought
        }

    await safe_tool_call(
        record_thought, thought_args_2, "Record Thought (Question Linked to Action)"
    )

    # Get the thought chain details (should show linked thoughts)
    await safe_tool_call(
        get_thought_chain,
        {"thought_chain_id": chain_id},
        f"Get Analysis Thought Chain Details ({_fmt_id(chain_id)})",
    )

    return chain_id


async def demonstrate_memory_operations(wf_id: Optional[str], action_ids: Dict, thought_ids: Dict):
    """Demonstrate memory storage, querying, linking."""
    console.print(Rule("[bold green]6. Memory Operations & Querying[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping memory demo: No valid workflow ID provided.[/yellow]")
        return {}  # Return empty dict

    mem_ids = {}

    action1_id = action_ids.get("action1_id")  # Might be None  # noqa: F841
    action2_id = action_ids.get("action2_id")  # Might be None  # noqa: F841

    # Store Memory 1 (Related to Planning Action)
    store_args_1 = {
        "workflow_id": wf_id,
        "action_id": action_ids.get("action1_id"),
        "content": "The initial plan involves simulation and analysis.",
        "memory_type": MemoryType.SUMMARY.value,
        "memory_level": MemoryLevel.EPISODIC.value,
        "description": "Summary of initial plan",
        "tags": ["planning", "summary"],
        "generate_embedding": False,  # Set False explicitly for baseline
    }
    mem_res_1 = await safe_tool_call(store_memory, store_args_1, "Store Memory 1 (Plan Summary)")
    mem1_id = mem_res_1.get("memory_id") if mem_res_1.get("success") else None
    if mem1_id:
        mem_ids["mem1_id"] = mem1_id

    # Store Memory 2 (Related to Failed Action)
    store_args_2 = {
        "workflow_id": wf_id,
        "action_id": action_ids.get("action2_id"),
        "content": "Data simulation failed with a timeout error (Code 504).",
        "memory_type": MemoryType.OBSERVATION.value,
        "memory_level": MemoryLevel.EPISODIC.value,
        "description": "Simulation failure detail",
        "importance": 7.0,  # Failed actions might be important
        "tags": ["error", "simulation", "observation"],
        "generate_embedding": False,
    }
    mem_res_2 = await safe_tool_call(
        store_memory, store_args_2, "Store Memory 2 (Simulation Error)"
    )
    mem2_id = mem_res_2.get("memory_id") if mem_res_2.get("success") else None
    if mem2_id:
        mem_ids["mem2_id"] = mem2_id

    # Store Memory 3 (A more general fact)
    store_args_3 = {
        "workflow_id": wf_id,
        "content": "Timeout errors often indicate resource contention or configuration issues.",
        "memory_type": MemoryType.FACT.value,
        "memory_level": MemoryLevel.SEMANTIC.value,
        "description": "General knowledge about timeouts",
        "importance": 6.0,
        "confidence": 0.9,
        "tags": ["error", "knowledge", "fact"],
        "generate_embedding": False,
    }
    mem_res_3 = await safe_tool_call(store_memory, store_args_3, "Store Memory 3 (Timeout Fact)")
    mem3_id = mem_res_3.get("memory_id") if mem_res_3.get("success") else None
    if mem3_id:
        mem_ids["mem3_id"] = mem3_id

    # Link Memory 2 (Error) -> Memory 3 (Fact)
    if mem2_id and mem3_id:
        await safe_tool_call(
            create_memory_link,
            {
                "source_memory_id": mem2_id,
                "target_memory_id": mem3_id,
                "link_type": LinkType.REFERENCES.value,
                "description": "Error relates to general timeout knowledge",
            },
            f"Link Error ({_fmt_id(mem2_id)}) to Fact ({_fmt_id(mem3_id)})",
        )

        # Get Linked Memories for the Error Memory
        await safe_tool_call(
            get_linked_memories,
            {"memory_id": mem2_id, "direction": "outgoing", "include_memory_details": True},
            f"Get Outgoing Linked Memories for Error ({_fmt_id(mem2_id)})",
        )

    # Query Memories using FTS
    await safe_tool_call(
        query_memories,
        {"workflow_id": wf_id, "search_text": "simulation error timeout"},
        "Query Memories (FTS: 'simulation error timeout')",
    )

    # Query Memories by Importance Range
    await safe_tool_call(
        query_memories,
        {"workflow_id": wf_id, "min_importance": 6.5, "sort_by": "importance"},
        "Query Memories (Importance >= 6.5)",
    )

    # Query Memories by Memory Type
    await safe_tool_call(
        query_memories,
        {"workflow_id": wf_id, "memory_type": MemoryType.FACT.value},
        "Query Memories (Type: Fact)",
    )

    # Update Memory 1's tags
    if mem1_id:
        await safe_tool_call(
            update_memory,
            {"memory_id": mem1_id, "tags": ["planning", "summary", "initial_phase"]},
            f"Update Memory 1 Tags ({_fmt_id(mem1_id)})",
        )
        # Verify update
        await safe_tool_call(
            get_memory_by_id,
            {"memory_id": mem1_id},
            f"Get Memory 1 After Tag Update ({_fmt_id(mem1_id)})",
        )

    # Example: Record a thought linked to a memory
    if mem3_id and thought_ids:  # Assuming demonstrate_thoughts ran successfully
        thought_chain_id_str = thought_ids.get("main_chain_id")
        if not thought_chain_id_str:
            console.print(
                "[yellow]Skipping thought link to memory: main_chain_id not found in thought_ids dict.[/yellow]"
            )
        else:
            thought_args_link = {
                "workflow_id": wf_id,
                "thought_chain_id": thought_chain_id_str,  # Pass the string ID
                "content": "Based on the general knowledge about timeouts, need to check server logs.",
                "thought_type": ThoughtType.PLAN.value,
                "relevant_memory_id": mem3_id,  # Link to the Fact memory
            }
            await safe_tool_call(
                record_thought,
                thought_args_link,
                f"Record Thought Linked to Memory ({_fmt_id(mem3_id)})",
            )
    elif not thought_ids:
        console.print(
            "[yellow]Skipping thought link to memory: thought_ids dict is empty or None.[/yellow]"
        )

    return mem_ids


async def demonstrate_embedding_and_search(wf_id: Optional[str], mem_ids: Dict, thought_ids: Dict):
    """Demonstrate embedding generation and semantic/hybrid search."""
    console.print(Rule("[bold green]7. Embedding & Semantic Search[/bold green]", style="green"))
    if not wf_id:
        console.print("[yellow]Skipping embedding demo: No valid workflow ID.[/yellow]")
        return  # Return immediately if no workflow ID
    mem1_id = mem_ids.get("mem1_id")  # Plan summary
    mem2_id = mem_ids.get("mem2_id")  # Simulation error
    mem3_id = mem_ids.get("mem3_id")  # Timeout fact

    if not mem1_id or not mem2_id or not mem3_id:
        console.print(
            "[yellow]Skipping embedding demo: Missing required memory IDs from previous steps.[/yellow]"
        )
        return  # Return immediately if prerequisite memories are missing

    # 1. Update Memory 2 (Error) to generate embedding
    # This relies on the embedding service being functional (API key configured)
    console.print(
        "[yellow]Attempting to generate embeddings. Requires configured Embedding Service (e.g., OpenAI API key).[/yellow]"
    )
    update_res = await safe_tool_call(
        update_memory,
        {
            "memory_id": mem2_id,
            "regenerate_embedding": True,
        },
        f"Update Memory 2 ({_fmt_id(mem2_id)}) to Generate Embedding",
    )
    if not (update_res and update_res.get("success") and update_res.get("embedding_regenerated")):
        console.print(
            "[red]   -> Failed to generate embedding for Memory 2. Semantic/Hybrid search may not work as expected.[/red]"
        )

    # 2. Store a new memory WITH embedding generation enabled
    store_args_4 = {
        "workflow_id": wf_id,
        "content": "Investigating the root cause of the simulation timeout is the next priority.",
        "memory_type": MemoryType.PLAN.value,
        "memory_level": MemoryLevel.EPISODIC.value,
        "description": "Next step planning",
        "importance": 7.5,
        "tags": ["investigation", "planning", "error_handling"],
        "generate_embedding": True,  # Explicitly enable
    }
    mem_res_4 = await safe_tool_call(
        store_memory, store_args_4, "Store Memory 4 (Next Step Plan) with Embedding"
    )
    mem4_id = mem_res_4.get("memory_id") if mem_res_4.get("success") else None
    if mem4_id:
        mem_ids["mem4_id"] = mem4_id  # Add to our tracked IDs

    # Check if embedding was actually generated for Mem4
    if mem4_id:
        mem4_details = await safe_tool_call(
            get_memory_by_id,
            {"memory_id": mem4_id},
            f"Check Memory 4 Details ({_fmt_id(mem4_id)})",
            suppress_output=True,
        )
        if mem4_details and mem4_details.get("success") and mem4_details.get("embedding_id"):
            console.print(
                f"[green]   -> Embedding ID confirmed for Memory 4: {_fmt_id(mem4_details['embedding_id'])}[/green]"
            )
        else:
            console.print(
                "[yellow]   -> Warning: Embedding ID missing for Memory 4. Embedding generation likely failed.[/yellow]"
            )
            console.print("[dim]      (Semantic/Hybrid search results may be limited.)[/dim]")

    # 3. Semantic Search
    await safe_tool_call(
        search_semantic_memories,
        {
            "workflow_id": wf_id,
            "query": "problems with simulation performance",
            "limit": 3,
            "threshold": 0.5,
        },
        "Semantic Search: 'problems with simulation performance'",
    )
    await safe_tool_call(
        search_semantic_memories,
        {
            "workflow_id": wf_id,
            "query": "next actions to take",
            "limit": 2,
            "memory_level": MemoryLevel.EPISODIC.value,
        },
        "Semantic Search: 'next actions to take' (Episodic only)",
    )

    # 4. Hybrid Search
    await safe_tool_call(
        hybrid_search_memories,
        {
            "workflow_id": wf_id,
            "query": "investigate timeout simulation",
            "limit": 4,
            "semantic_weight": 0.6,
            "keyword_weight": 0.4,
            "tags": ["error"],
            "include_content": False,
        },
        "Hybrid Search: 'investigate timeout simulation' + tag 'error'",
    )

    # 5. Demonstrate link suggestions
    # Update Mem3 (Timeout fact) to generate embedding
    update_res_3 = await safe_tool_call(
        update_memory,
        {"memory_id": mem3_id, "regenerate_embedding": True},
        f"Update Memory 3 ({_fmt_id(mem3_id)}) to Generate Embedding",
    )
    if not (
        update_res_3 and update_res_3.get("success") and update_res_3.get("embedding_regenerated")
    ):
        console.print(
            "[red]   -> Failed to generate embedding for Memory 3. Link suggestion test might fail.[/red]"
        )

    # --- Store Memory 5 (Hypothesis) ---
    hypothesis_content = "Resource limits on the simulation server might be too low."
    thought_chain_id = thought_ids.get("main_chain_id") if isinstance(thought_ids, dict) else None
    hypothesis_thought_id = None
    if thought_chain_id:
        thought_args_hyp = {
            "workflow_id": wf_id,
            "thought_chain_id": thought_chain_id,
            "content": hypothesis_content,
            "thought_type": ThoughtType.HYPOTHESIS.value,
            "relevant_memory_id": mem3_id,
        }
        hyp_thought_res = await safe_tool_call(
            record_thought, thought_args_hyp, "Record Hypothesis Thought"
        )
        hypothesis_thought_id = (
            hyp_thought_res.get("thought_id") if hyp_thought_res.get("success") else None
        )
    else:
        console.print(
            "[yellow]Skipping hypothesis memory storage: Could not get thought chain ID.[/yellow]"
        )

    mem5_id = None
    mem_res_5 = None
    if hypothesis_thought_id:
        store_args_5 = {
            "workflow_id": wf_id,
            "thought_id": hypothesis_thought_id,
            "content": hypothesis_content,
            "memory_type": MemoryType.REASONING_STEP.value,
            "memory_level": MemoryLevel.SEMANTIC.value,
            "description": "Hypothesis on timeout cause",
            "importance": 6.5,
            "confidence": 0.6,
            "tags": ["hypothesis", "resource", "error", "reasoning_step"],
            "generate_embedding": True,
            "suggest_links": True,  # Explicitly ask for suggestions
            "max_suggested_links": 2,
        }
        mem_res_5 = await safe_tool_call(
            store_memory, store_args_5, "Store Memory 5 (Hypothesis Reasoning) - Suggest Links"
        )
        mem5_id = mem_res_5.get("memory_id") if mem_res_5.get("success") else None
        if mem5_id:
            mem_ids["mem5_id"] = mem5_id

        # Check suggestions result
        if mem_res_5 and mem_res_5.get("success") and mem_res_5.get("suggested_links"):
            console.print("[cyan]   -> Link suggestions received for Memory 5:[/]")
            console.print(pretty_repr(mem_res_5["suggested_links"]))
        elif mem_res_5 and mem_res_5.get("success"):
            console.print(
                "[dim]   -> No link suggestions returned for Memory 5 (or embedding failed).[/dim]"
            )
        elif mem_res_5 and not mem_res_5.get("success"):
            console.print(
                "[yellow]   -> Failed to store Memory 5, cannot check suggestions.[/yellow]"
            )
    else:
        console.print(
            "[yellow]Skipping Memory 5 storage: Hypothesis thought recording failed.[/yellow]"
        )


async def demonstrate_state_and_working_memory(
    wf_id: str,
    mem_ids_dict: Dict[str, str],
    action_ids_dict: Dict[str, str],
    thought_ids_dict: Dict[str, Any],
    state_ids_dict: Dict[str, str],
):
    """Demonstrate saving/loading state and working memory operations."""
    console.print(
        Rule("[bold green]8. Cognitive State & Working Memory[/bold green]", style="green")
    )

    # --- Retrieve necessary IDs from previous steps ---
    main_wf_id = wf_id
    main_chain_id = thought_ids_dict.get("main_chain_id")  # noqa: F841
    plan_action_id = action_ids_dict.get("action1_id")
    sim_action_id = action_ids_dict.get("action2_id")
    mem1_id = mem_ids_dict.get("mem1_id")
    mem2_id = mem_ids_dict.get("mem2_id")
    mem3_id = mem_ids_dict.get("mem3_id")
    mem4_id = mem_ids_dict.get("mem4_id")
    mem5_id = mem_ids_dict.get("mem5_id")

    hypothesis_thought_id = None
    if mem5_id and main_wf_id:
        mem5_details = await safe_tool_call(
            get_memory_by_id,
            {"memory_id": mem5_id},
            f"Get Memory 5 Details ({_fmt_id(mem5_id)}) for Thought ID",
            suppress_output=True,
        )
        if mem5_details and mem5_details.get("success"):
            hypothesis_thought_id = mem5_details.get("thought_id")
            if hypothesis_thought_id:
                console.print(
                    f"[cyan]   -> Retrieved Hypothesis Thought ID: {_fmt_id(hypothesis_thought_id)}[/cyan]"
                )
            else:
                console.print(
                    "[yellow]   -> Could not retrieve hypothesis thought ID from Memory 5 details.[/yellow]"
                )
        else:
            # Handle case where get_memory_by_id failed or didn't return success
            console.print(
                f"[yellow]   -> Failed to get details for Memory 5 ({_fmt_id(mem5_id)}) to find Thought ID.[/yellow]"
            )

    # --- Check if we have enough *critical* data to proceed ---
    # Hypothesis thought ID is critical for saving the intended state goals
    if not (
        main_wf_id
        and mem1_id
        and mem2_id
        and mem3_id
        and mem4_id
        and plan_action_id
        and hypothesis_thought_id  # Ensure this critical ID exists
    ):
        console.print(
            "[bold yellow]Skipping state/working memory demo:[/bold yellow] Missing critical IDs (workflow, mem1-4, plan_action, hypothesis_thought) from previous steps."
        )
        return  # Exit if critical IDs are missing

    # Prepare IDs for saving state - check individually for non-critical ones
    working_mems = [
        mem_id for mem_id in [mem2_id, mem3_id, mem4_id, mem5_id] if mem_id
    ]  # Filter None
    focus_mems = [mem4_id] if mem4_id else []  # Filter None
    context_actions = [
        action_id for action_id in [plan_action_id, sim_action_id] if action_id
    ]  # Filter None
    goal_thoughts = [hypothesis_thought_id]  # Already checked above

    # 1. Save Cognitive State
    save_args = {
        "workflow_id": wf_id,
        "title": "State after simulation failure and hypothesis",
        "working_memory_ids": working_mems,
        "focus_area_ids": focus_mems,
        "context_action_ids": context_actions,
        "current_goal_thought_ids": goal_thoughts,
    }
    state_res = await safe_tool_call(save_cognitive_state, save_args, "Save Cognitive State")
    state_id = (
        state_res.get("state_id") if state_res and state_res.get("success") else None
    )  # More robust check

    if state_id:
        state_ids_dict["saved_state_id"] = state_id
        console.print(
            f"[green]   -> Cognitive state saved successfully with ID: {_fmt_id(state_id)}[/green]"
        )
    else:
        console.print(
            "[bold red]CRITICAL DEMO FAILURE:[/bold red] Failed to save cognitive state. Cannot proceed with working memory tests."
        )
        return  # Exit if state saving failed

    # 2. Load Cognitive State (by ID) - Use the confirmed state_id
    await safe_tool_call(
        load_cognitive_state,
        {"workflow_id": wf_id, "state_id": state_id},  # Use state_id directly
        f"Load Cognitive State ({_fmt_id(state_id)}) by ID",
    )

    # 3. Load Cognitive State (Latest)
    await safe_tool_call(
        load_cognitive_state,
        {"workflow_id": wf_id},
        "Load Latest Cognitive State",
    )

    # --- Working Memory Operations using the saved state_id as the context_id ---
    # The variable 'state_id' now holds the context ID we need for the rest of this section.
    console.print(
        f"\n[dim]Using saved state ID '{_fmt_id(state_id)}' as context_id for working memory tests...[/dim]\n"
    )

    # 4. Focus Memory (Focus on the 'hypothesis' memory if it exists)
    focus_target_id = mem_ids_dict.get("mem5_id")  # Get mem5_id again here
    if focus_target_id:
        await safe_tool_call(
            focus_memory,
            {
                "memory_id": focus_target_id,
                "context_id": state_id,  # USE state_id
                "add_to_working": False,  # Assume it's already there from save_state
            },
            f"Focus Memory ({_fmt_id(focus_target_id)}) in Context ({_fmt_id(state_id)})",  # USE state_id
        )
    else:
        console.print(
            "[bold yellow]Skipping focus memory test: Hypothesis memory ID (mem5_id) not available.[/bold yellow]"
        )

    # 5. Get Working Memory (Should reflect the saved state initially)
    await safe_tool_call(
        get_working_memory,
        {
            "context_id": state_id,  # USE state_id
            "include_links": False,  # Keep output cleaner for this demo step
        },
        f"Get Working Memory for Context ({_fmt_id(state_id)})",  # USE state_id
    )

    # 6. Optimize Working Memory (Reduce size, using 'balanced' strategy)
    wm_details = await safe_tool_call(
        get_working_memory,
        {"context_id": state_id},  # USE state_id
        "Get WM Size Before Optimization",
        suppress_output=True,
    )
    current_wm_size = (
        len(wm_details.get("working_memories", []))
        if wm_details and wm_details.get("success")
        else 0
    )

    if current_wm_size > 2:  # Only optimize if we have more than 2 memories
        target_optimize_size = max(1, current_wm_size // 2)
        console.print(
            f"[cyan]   -> Optimizing working memory from {current_wm_size} down to {target_optimize_size}...[/cyan]"
        )
        await safe_tool_call(
            optimize_working_memory,
            {
                "context_id": state_id,  # USE state_id
                "target_size": target_optimize_size,
                "strategy": "balanced",
            },
            f"Optimize Working Memory (Context: {_fmt_id(state_id)}, Target: {target_optimize_size})",  # USE state_id
        )
        await safe_tool_call(
            get_working_memory,
            {"context_id": state_id, "include_links": False},  # USE state_id
            f"Get Working Memory After Optimization (Context: {_fmt_id(state_id)})",  # USE state_id
        )
    else:
        console.print(
            f"[dim]Skipping working memory optimization: Current size ({current_wm_size}) is too small.[/dim]"
        )


async def demonstrate_metacognition(wf_id: str, mem_ids: Dict, state_ids: Dict):
    """Demonstrate context retrieval, auto-focus, promotion, consolidation, reflection, summarization."""
    console.print(Rule("[bold green]9. Meta-Cognition & Summarization[/bold green]", style="green"))

    # 1. Get Workflow Context
    await safe_tool_call(get_workflow_context, {"workflow_id": wf_id}, "Get Full Workflow Context")

    # 2. Auto Update Focus
    context_id = state_ids.get("saved_state_id")
    if context_id:
        await safe_tool_call(
            auto_update_focus,
            {"context_id": context_id},
            f"Auto Update Focus for Context ({_fmt_id(context_id)})",
        )
    else:
        console.print("[yellow]Skipping auto-focus: No context_id (state_id) available.[/yellow]")

    # 3. Promote Memory Level
    mem1_id = mem_ids.get("mem1_id")  # Episodic summary
    mem3_id = mem_ids.get("mem3_id")  # Semantic fact
    if mem1_id:
        console.print(
            f"[cyan]   -> Manually increasing access_count for Memory 1 ({_fmt_id(mem1_id)}) to test promotion...[/cyan]"
        )
        try:
            async with DBConnection(DEMO_DB_FILE) as conn:
                await conn.execute(
                    "UPDATE memories SET access_count = 10, confidence = 0.9 WHERE memory_id = ?",
                    (mem1_id,),
                )
                await conn.commit()
            await safe_tool_call(
                promote_memory_level,
                {"memory_id": mem1_id},
                f"Attempt Promote Memory 1 ({_fmt_id(mem1_id)}) from Episodic",
            )
        except Exception as e:
            console.print(f"[red]   -> Error updating access count for promotion test: {e}[/red]")

    if mem3_id:
        await safe_tool_call(
            promote_memory_level,
            {"memory_id": mem3_id},
            f"Attempt Promote Memory 3 ({_fmt_id(mem3_id)}) from Semantic (Should Fail)",
        )

    # 4. Consolidate Memories (requires LLM)
    mem_ids_for_consolidation = [
        mid
        for mid in [mem_ids.get("mem1_id"), mem_ids.get("mem2_id"), mem_ids.get("mem3_id")]
        if mid
    ]
    if len(mem_ids_for_consolidation) >= 2:
        console.print(
            "[yellow]Attempting memory consolidation. Requires configured LLM provider (e.g., OpenAI API key).[/yellow]"
        )
        await safe_tool_call(
            consolidate_memories,
            {
                "workflow_id": wf_id,
                "target_memories": mem_ids_for_consolidation,
                "consolidation_type": "summary",
                "store_result": True,
                "provider": config.default_provider or "openai",
            },
            "Consolidate Memories (Summary)",
        )
    else:
        console.print(
            "[yellow]Skipping consolidation: Not enough source memories available.[/yellow]"
        )

    # 5. Generate Reflection (requires LLM)
    console.print(
        "[yellow]Attempting reflection generation. Requires configured LLM provider.[/yellow]"
    )
    await safe_tool_call(
        generate_reflection,
        {
            "workflow_id": wf_id,
            "reflection_type": "gaps",
            "provider": config.default_provider
            or "openai",  # Use configured default from GatewayConfig
        },
        "Generate Reflection (Gaps)",
    )

    # 6. Summarize Text (requires LLM)
    console.print(
        "[yellow]Attempting text summarization. Requires configured LLM provider.[/yellow]"
    )
    sample_text = """
    The Unified Memory System integrates several components for advanced agent cognition.
    It tracks workflows, actions, artifacts, and thoughts. A multi-level memory hierarchy
    (working, episodic, semantic, procedural) allows for different types of knowledge storage.
    Vector embeddings enable semantic search capabilities. Associative links connect related
    memory items. Cognitive states can be saved and loaded, preserving the agent's context.
    Maintenance tools help manage memory expiration and provide statistics. Reporting and
    visualization tools offer insights into the agent's processes. This system aims to provide
    a robust foundation for complex autonomous agents.
    """
    await safe_tool_call(
        summarize_text,
        {
            "text_to_summarize": sample_text,
            "target_tokens": 50,
            "record_summary": True,
            "workflow_id": wf_id,
            "provider": config.default_provider or "openai",
        },
        "Summarize Sample Text and Record Memory",
    )


async def demonstrate_maintenance_and_stats(wf_id: str):
    """Demonstrate memory deletion and statistics computation."""
    console.print(Rule("[bold green]10. Maintenance & Statistics[/bold green]", style="green"))

    # 1. Delete Expired Memories
    # Store a temporary memory with a short TTL
    console.print("[cyan]   -> Storing a temporary memory with TTL=1 second...[/cyan]")
    ttl_mem_args = {
        "workflow_id": wf_id,
        "content": "This memory should expire quickly.",
        "memory_type": "observation",
        "ttl": 1,  # 1 second TTL
    }
    ttl_mem_res = await safe_tool_call(
        store_memory,  # Pass the function object
        ttl_mem_args,  # Pass the arguments dictionary
        "Store Temporary Memory",
        suppress_output=True,
    )

    if ttl_mem_res and ttl_mem_res.get("success"):
        console.print("[cyan]   -> Waiting 2 seconds for memory to expire...[/cyan]")
        await asyncio.sleep(2)
        await safe_tool_call(
            delete_expired_memories, {}, "Delete Expired Memories (Should delete 1)"
        )
    else:
        console.print(
            "[yellow]   -> Failed to store temporary memory for expiration test.[/yellow]"
        )
        if ttl_mem_res:
            console.print(f"[yellow]      Error: {ttl_mem_res.get('error')}[/yellow]")

    # 2. Compute Statistics (Workflow Specific)
    await safe_tool_call(
        compute_memory_statistics,
        {"workflow_id": wf_id},
        f"Compute Statistics for Workflow ({_fmt_id(wf_id)})",
    )

    # 3. Compute Statistics (Global)
    await safe_tool_call(compute_memory_statistics, {}, "Compute Global Statistics")


async def demonstrate_reporting_and_viz(wf_id: str, thought_chain_id: str, mem_ids: Dict):
    """Demonstrate report generation and visualization."""
    console.print(Rule("[bold green]11. Reporting & Visualization[/bold green]", style="green"))

    # 1. Generate Workflow Reports
    await safe_tool_call(
        generate_workflow_report,
        {"workflow_id": wf_id, "report_format": "markdown", "style": "professional"},
        "Generate Workflow Report (Markdown, Professional)",
    )
    await safe_tool_call(
        generate_workflow_report,
        {"workflow_id": wf_id, "report_format": "html", "style": "concise"},
        "Generate Workflow Report (HTML, Concise)",
    )
    await safe_tool_call(
        generate_workflow_report,
        {"workflow_id": wf_id, "report_format": "json"},
        "Generate Workflow Report (JSON)",
    )
    await safe_tool_call(
        generate_workflow_report,
        {"workflow_id": wf_id, "report_format": "mermaid"},
        "Generate Workflow Report (Mermaid Diagram)",
    )

    # 2. Visualize Reasoning Chain
    if thought_chain_id:
        await safe_tool_call(
            visualize_reasoning_chain,
            {"thought_chain_id": thought_chain_id},
            f"Visualize Reasoning Chain ({_fmt_id(thought_chain_id)})",
        )
    else:
        console.print(
            "[yellow]Skipping reasoning visualization: No thought_chain_id available.[/yellow]"
        )

    # 3. Visualize Memory Network
    # Visualize around the 'error' memory
    center_mem_id = mem_ids.get("mem2_id")
    if center_mem_id:
        await safe_tool_call(
            visualize_memory_network,
            {"center_memory_id": center_mem_id, "depth": 1, "max_nodes": 15},
            f"Visualize Memory Network (Centered on Error Mem {_fmt_id(center_mem_id)}, Depth 1)",
        )
    else:
        console.print(
            "[yellow]Skipping centered memory visualization: Error memory ID not available.[/yellow]"
        )

    # Visualize top memories for the workflow
    await safe_tool_call(
        visualize_memory_network,
        {"workflow_id": wf_id, "max_nodes": 20},
        f"Visualize Memory Network (Workflow {_fmt_id(wf_id)}, Top 20 Relevant)",
    )


# --- Main Execution Logic ---
async def main():
    """Run the extended Unified Memory System demonstration suite."""
    console.print(
        Rule(
            "[bold magenta]Unified Memory System Tools Demo (Extended)[/bold magenta]",
            style="white",
        )
    )
    exit_code = 0
    # Dictionaries to store IDs created during the demo
    wf_ids = {}
    action_ids = {}
    artifact_ids = {}
    thought_ids = {}  # Store chain ID
    mem_ids = {}
    state_ids = {}  # Store state ID

    try:
        await setup_demo_environment()

        # --- Run Demo Sections in Order ---
        wf_id = await demonstrate_basic_workflows()
        if wf_id:
            wf_ids["main_wf_id"] = wf_id
        else:
            raise RuntimeError("Workflow creation failed, cannot continue demo.")

        action_ids = await demonstrate_basic_actions(wf_ids.get("main_wf_id"))
        await demonstrate_action_dependencies(wf_ids.get("main_wf_id"), action_ids)
        artifact_ids = await demonstrate_artifacts(wf_ids.get("main_wf_id"), action_ids)

        chain_id = await demonstrate_thoughts_and_linking(
            wf_ids.get("main_wf_id"), action_ids, artifact_ids
        )
        if chain_id:
            thought_ids["main_chain_id"] = chain_id

        mem_ids = await demonstrate_memory_operations(
            wf_ids.get("main_wf_id"), action_ids, thought_ids
        )  # Pass thought_ids dict
        await demonstrate_embedding_and_search(wf_ids.get("main_wf_id"), mem_ids, thought_ids)

        # State/Working Memory depends on previous steps creating IDs
        # Pass all collected ID dictionaries
        await demonstrate_state_and_working_memory(
            wf_id=wf_ids["main_wf_id"],
            mem_ids_dict=mem_ids,
            action_ids_dict=action_ids,
            thought_ids_dict=thought_ids,  # Contains chain_id and potentially specific thought IDs if needed later
            state_ids_dict=state_ids,  # Pass dict to store the created state_id
        )

        # --- Run NEW Advanced Demo Sections ---
        await demonstrate_metacognition(wf_ids["main_wf_id"], mem_ids, state_ids)
        await demonstrate_maintenance_and_stats(wf_ids["main_wf_id"])
        await demonstrate_reporting_and_viz(
            wf_ids["main_wf_id"], thought_ids.get("main_chain_id"), mem_ids
        )
        # --- End NEW Sections ---

        logger.success(
            "Unified Memory System Demo completed successfully!",
            emoji_key="complete",
        )
        console.print(Rule("[bold green]Demo Finished[/bold green]", style="green"))

    except Exception as e:
        logger.critical(f"Demo crashed unexpectedly: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"\n[bold red]CRITICAL ERROR:[/bold red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        exit_code = 1

    finally:
        # Clean up the demo environment
        console.print(Rule("Cleanup", style="dim"))
        await cleanup_demo_environment()

    return exit_code


if __name__ == "__main__":
    # Run the demo
    final_exit_code = asyncio.run(main())
    sys.exit(final_exit_code)
