# local_text_tools_demo.py
"""
Comprehensive demonstration script for the local_text_tools functions in Ultimate MCP Server.

This script showcases the usage of local command-line text processing utilities
(ripgrep, awk, sed, jq) through the secure, standalone functions provided by
ultimate_mcp_server.tools.local_text_tools.
It includes basic examples, advanced command-line techniques, security failure demos,
streaming examples, and interactive workflows demonstrating LLM-driven tool usage
on sample documents.

It uses sample files from the 'sample/' directory relative to this script.
NOTE: The LLM interactive demos require a configured LLM provider (e.g., OpenAI API key).

-------------------------------------------------------------------------------------
IMPORTANT: ABOUT ERROR INDICATORS AND "FAILURES" IN THIS DEMO
-------------------------------------------------------------------------------------

Many demonstrations in this script INTENTIONALLY trigger security features and error
handling. These appear as red ❌ boxes but are actually showing CORRECT BEHAVIOR.

Examples of intentional security demonstrations include:
- Invalid regex patterns (to show proper error reporting)
- AWK/SED script syntax errors (to show validation)
- Path traversal attempts (to demonstrate workspace confinement)
- Usage of forbidden flags like 'sed -i' (showing security limits)
- Redirection attempts (demonstrating shell character blocking)
- Command substitution (showing protection against command injection)

When you see "SECURITY CHECK PASSED" or "INTENTIONAL DEMONSTRATION" in the description,
this indicates a feature working correctly, not a bug in the tools.
-------------------------------------------------------------------------------------
"""

# --- Standard Library Imports ---
import asyncio
import inspect  # top-level import is fine too
import json
import os
import re
import shlex
import shutil
import sys
import time
from enum import Enum  # Import Enum from the enum module, not typing
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional

# --- Configuration & Path Setup ---
# Add project root to path for imports when running as script
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR
    # Try to find project root marker ('ultimate_mcp_server' dir or 'pyproject.toml')
    while (
        not (
            (PROJECT_ROOT / "ultimate_mcp_server").is_dir()
            or (PROJECT_ROOT / "pyproject.toml").is_file()
        )
        and PROJECT_ROOT.parent != PROJECT_ROOT
    ):
        PROJECT_ROOT = PROJECT_ROOT.parent

    # If marker found and path not added, add it
    if (PROJECT_ROOT / "ultimate_mcp_server").is_dir() or (
        PROJECT_ROOT / "pyproject.toml"
    ).is_file():
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    # Fallback if no clear marker found upwards
    elif SCRIPT_DIR.parent != PROJECT_ROOT and (SCRIPT_DIR.parent / "ultimate_mcp_server").is_dir():
        PROJECT_ROOT = SCRIPT_DIR.parent
        print(f"Warning: Assuming project root is {PROJECT_ROOT}", file=sys.stderr)
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
    # Final fallback: add script dir itself
    elif str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
        print(
            f"Warning: Could not reliably determine project root. Added script directory {SCRIPT_DIR} to path as fallback.",
            file=sys.stderr,
        )
    else:
        # If already in path, assume it's okay
        pass

    # Set MCP_TEXT_WORKSPACE environment variable to PROJECT_ROOT before importing local_text_tools
    os.environ["MCP_TEXT_WORKSPACE"] = str(PROJECT_ROOT)
    print(f"Set MCP_TEXT_WORKSPACE to: {os.environ['MCP_TEXT_WORKSPACE']}", file=sys.stderr)

except Exception as e:
    print(f"Error setting up sys.path: {e}", file=sys.stderr)
    sys.exit(1)


# --- Third-Party Imports ---
try:
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.pretty import pretty_repr
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.traceback import install as install_rich_traceback
except ImportError:
    print("Error: 'rich' library not found. Please install it: pip install rich", file=sys.stderr)
    sys.exit(1)


# --- Project-Specific Imports ---
# Import necessary tools and components
try:
    # Import specific functions and types
    from ultimate_mcp_server.config import get_config  # To check LLM provider config
    from ultimate_mcp_server.constants import Provider  # For LLM demo

    # Import specific exceptions
    from ultimate_mcp_server.exceptions import ToolExecutionError, ToolInputError
    from ultimate_mcp_server.tools.completion import chat_completion
    from ultimate_mcp_server.tools.local_text_tools import (
        ToolErrorCode,
        ToolResult,
        get_workspace_dir,  # Function to get configured workspace
        run_awk,
        run_awk_stream,
        run_jq,
        run_jq_stream,
        run_ripgrep,
        run_ripgrep_stream,
        run_sed,
        run_sed_stream,
    )
    from ultimate_mcp_server.utils import get_logger
except ImportError as import_err:
    print(f"Error: Failed to import necessary MCP Server components: {import_err}", file=sys.stderr)
    print(
        "Please ensure the script is run from within the correct environment, the package is installed (`pip install -e .`), and project structure is correct.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Initialization ---
console = Console()
logger = get_logger("demo.local_text_tools")
install_rich_traceback(show_locals=False, width=console.width)

# Define path to sample files relative to this script's location
SAMPLE_DIR = SCRIPT_DIR / "sample"
if not SAMPLE_DIR.is_dir():
    print(
        f"Error: Sample directory not found at expected location: {SCRIPT_DIR}/sample",
        file=sys.stderr,
    )
    # Try locating it relative to Project Root as fallback
    ALT_SAMPLE_DIR = PROJECT_ROOT / "examples" / "local_text_tools_demo" / "sample"
    if ALT_SAMPLE_DIR.is_dir():
        print(f"Found sample directory at alternate location: {ALT_SAMPLE_DIR}", file=sys.stderr)
        SAMPLE_DIR = ALT_SAMPLE_DIR
    else:
        print(
            f"Please ensure the 'sample' directory exists within {SCRIPT_DIR} or {ALT_SAMPLE_DIR}.",
            file=sys.stderr,
        )
        sys.exit(1)

# Store both absolute and relative paths for the samples
SAMPLE_DIR_ABS = SAMPLE_DIR
CLASSIFICATION_SAMPLES_DIR_ABS = SAMPLE_DIR / "text_classification_samples"

# Create relative paths for use with the tools - relative to PROJECT_ROOT
SAMPLE_DIR_REL = SAMPLE_DIR.relative_to(PROJECT_ROOT)
CLASSIFICATION_SAMPLES_DIR_REL = CLASSIFICATION_SAMPLES_DIR_ABS.relative_to(PROJECT_ROOT)

# Use relative paths for the tools
CONTRACT_FILE_PATH = str(SAMPLE_DIR_REL / "legal_contract.txt")  # Relative path
ARTICLE_FILE_PATH = str(SAMPLE_DIR_REL / "article.txt")
EMAIL_FILE_PATH = str(CLASSIFICATION_SAMPLES_DIR_REL / "email_classification.txt")
SCHEDULE_FILE_PATH = str(SAMPLE_DIR_REL / "SCHEDULE_1.2")  # Added for awk demo
JSON_SAMPLE_PATH = str(SAMPLE_DIR_REL / "sample_data.json")  # Added for jq file demo

# But for file operations (checking existence, etc.), use absolute paths
CONTRACT_FILE_PATH_ABS = str(SAMPLE_DIR_ABS / "legal_contract.txt")
ARTICLE_FILE_PATH_ABS = str(SAMPLE_DIR_ABS / "article.txt")
EMAIL_FILE_PATH_ABS = str(CLASSIFICATION_SAMPLES_DIR_ABS / "email_classification.txt")
SCHEDULE_FILE_PATH_ABS = str(SAMPLE_DIR_ABS / "SCHEDULE_1.2")
JSON_SAMPLE_PATH_ABS = str(SAMPLE_DIR_ABS / "sample_data.json")

# Create sample JSON file if it doesn't exist
if not Path(JSON_SAMPLE_PATH_ABS).exists():
    sample_json_content = """
[
  {"user": "Alice", "dept": "Sales", "region": "North", "value": 100, "tags": ["active", "pipeline"]},
  {"user": "Bob", "dept": "IT", "region": "South", "value": 150, "tags": ["active", "support"]},
  {"user": "Charlie", "dept": "Sales", "region": "North", "value": 120, "tags": ["inactive", "pipeline"]},
  {"user": "David", "dept": "IT", "region": "West", "value": 200, "tags": ["active", "admin"]}
]
"""
    try:
        # Make sure the directory exists
        Path(JSON_SAMPLE_PATH_ABS).parent.mkdir(parents=True, exist_ok=True)
        with open(JSON_SAMPLE_PATH_ABS, "w") as f:
            f.write(sample_json_content)
        logger.info(f"Created sample JSON file: {JSON_SAMPLE_PATH_ABS}")
    except OSError as e:
        logger.error(f"Failed to create sample JSON file {JSON_SAMPLE_PATH_ABS}: {e}")
        # Continue without it, jq file demos will fail gracefully

MAX_LLM_ITERATIONS = 5  # Limit for the interactive demo

# --- Helper Functions ---

ToolFunction = Callable[..., Coroutine[Any, Any, ToolResult]]
StreamFunction = Callable[..., Coroutine[Any, Any, AsyncIterator[str]]]


async def safe_tool_call(
    tool_func: ToolFunction,
    args: Dict[str, Any],
    description: str,
    display_input: bool = True,
    display_output: bool = True,
) -> ToolResult:
    """Helper to call a tool function, catch errors, and display results."""
    tool_func_name = getattr(tool_func, "__name__", "unknown_tool")

    if display_output:
        console.print(Rule(f"[bold blue]{escape(description)}[/bold blue]", style="blue"))

    if not callable(tool_func):
        console.print(
            f"[bold red]Error:[/bold red] Tool function '{tool_func_name}' is not callable."
        )
        return ToolResult(success=False, error=f"Function '{tool_func_name}' not callable.")

    if display_input and display_output:
        console.print(f"[dim]Calling [bold cyan]{tool_func_name}[/] with args:[/]")
        try:
            args_to_print = args.copy()
            # Truncate long input_data for display
            if "input_data" in args_to_print and isinstance(args_to_print["input_data"], str):
                if len(args_to_print["input_data"]) > 200:
                    args_to_print["input_data"] = args_to_print["input_data"][:200] + "[...]"
            args_repr = pretty_repr(args_to_print, max_length=120, max_string=200)
            console.print(args_repr)
        except Exception:
            console.print("(Could not represent args)")

    start_time = time.monotonic()
    result: ToolResult = ToolResult(
        success=False, error="Execution did not complete.", exit_code=None
    )  # Default error

    try:
        result = await tool_func(**args)  # Direct function call
        processing_time = time.monotonic() - start_time
        logger.debug(f"Tool '{tool_func_name}' execution time: {processing_time:.4f}s")

        if display_output:
            success = result.get("success", False)
            is_dry_run = result.get("dry_run_cmdline") is not None

            panel_title = f"[bold {'green' if success else 'red'}]Result: {tool_func_name} {'✅' if success else '❌'}{' (Dry Run)' if is_dry_run else ''}[/]"
            panel_border = "green" if success else "red"

            # Format output for display
            output_display = ""
            exit_code = result.get("exit_code", "N/A")
            output_display += f"[bold]Exit Code:[/bold] {exit_code}\n"
            duration = result.get("duration", 0.0)
            output_display += f"[bold]Duration:[/bold] {duration:.3f}s\n"
            cached = result.get("cached_result", False)
            output_display += f"[bold]Cached:[/bold] {'Yes' if cached else 'No'}\n"

            if is_dry_run:
                cmdline = result.get("dry_run_cmdline", [])
                output_display += f"\n[bold yellow]Dry Run Command:[/]\n{shlex.join(cmdline)}\n"
            elif success:
                stdout_str = result.get("stdout", "")
                stderr_str = result.get("stderr", "")
                stdout_trunc = result.get("stdout_truncated", False)
                stderr_trunc = result.get("stderr_truncated", False)

                if stdout_str:
                    output_display += f"\n[bold green]STDOUT ({len(stdout_str)} chars{', TRUNCATED' if stdout_trunc else ''}):[/]\n"
                    # Try syntax highlighting if stdout looks like JSON
                    stdout_str.strip().startswith(("{", "[")) and stdout_str.strip().endswith(
                        ("}", "]")
                    )
                    # Limit length for display
                    display_stdout = stdout_str[:3000] + ("..." if len(stdout_str) > 3000 else "")
                    # Just add the plain output text instead of the Syntax object
                    output_display += display_stdout
                else:
                    output_display += "[dim]STDOUT: (empty)[/]"

                if stderr_str:
                    header = f"[bold yellow]STDERR ({len(stderr_str)} chars{', TRUNCATED' if stderr_trunc else ''}):[/]"
                    output_display += f"\n\n{header}"
                    # Apply syntax highlighting for stderr too if it looks structured
                    is_stderr_json_like = stderr_str.strip().startswith(
                        ("{", "[")
                    ) and stderr_str.strip().endswith(("}", "]"))
                    if is_stderr_json_like:
                        stderr_display = stderr_str[:1000] + (
                            "..." if len(stderr_str) > 1000 else ""
                        )
                        Syntax(
                            stderr_display,
                            "json",
                            theme="monokai",
                            line_numbers=False,
                            word_wrap=True,
                        )
                        # We'll print this directly later
                    else:
                        output_display += "\n" + escape(
                            stderr_str[:1000] + ("..." if len(stderr_str) > 1000 else "")
                        )
                else:
                    output_display += "\n\n[dim]STDERR: (empty)[/]"

            # Create panel with the text content
            console.print(
                Panel(output_display, title=panel_title, border_style=panel_border, expand=False)
            )

    except (ToolInputError, ToolExecutionError) as e:  # Catch specific tool errors
        processing_time = time.monotonic() - start_time
        logger.error(f"Tool '{tool_func_name}' failed: {e}", exc_info=False)
        if display_output:
            error_title = f"[bold red]Error: {tool_func_name} Failed ❌[/]"
            error_code_val = getattr(e, "error_code", None)
            # Handle both enum and string error codes
            error_code_str = ""
            if error_code_val:
                if hasattr(error_code_val, "value"):
                    error_code_str = f" ({error_code_val.value})"
                else:
                    error_code_str = f" ({error_code_val})"
            error_content = f"[bold red]{type(e).__name__}{error_code_str}:[/] {escape(str(e))}"
            if hasattr(e, "details") and e.details:
                try:
                    details_repr = pretty_repr(e.details)
                except Exception:
                    details_repr = str(e.details)
                error_content += f"\n\n[yellow]Details:[/]\n{escape(details_repr)}"
            console.print(Panel(error_content, title=error_title, border_style="red", expand=False))
        # Ensure result dict structure on error
        result = ToolResult(
            success=False,
            error=str(e),
            error_code=getattr(e, "error_code", ToolErrorCode.UNEXPECTED_FAILURE),
            details=getattr(e, "details", {}),
            stdout=None,
            stderr=None,
            exit_code=None,
            duration=processing_time,
        )
    except Exception as e:
        processing_time = time.monotonic() - start_time
        logger.critical(f"Unexpected error calling '{tool_func_name}': {e}", exc_info=True)
        if display_output:
            console.print(f"\n[bold red]CRITICAL UNEXPECTED ERROR in {tool_func_name}:[/bold red]")
            console.print_exception(show_locals=False)
        result = ToolResult(
            success=False,
            error=f"Unexpected: {str(e)}",
            error_code=ToolErrorCode.UNEXPECTED_FAILURE,
            stdout=None,
            stderr=None,
            exit_code=None,
            duration=processing_time,
        )
    finally:
        if display_output:
            console.print()  # Add spacing

    # Ensure result is always a ToolResult-like dictionary before returning
    if not isinstance(result, dict):
        logger.error(
            f"Tool {tool_func_name} returned non-dict type {type(result)}. Returning error dict."
        )
        result = ToolResult(
            success=False,
            error=f"Tool returned unexpected type: {type(result).__name__}",
            error_code=ToolErrorCode.UNEXPECTED_FAILURE,
        )

    # Ensure basic keys exist even if tool failed unexpectedly before returning dict
    result.setdefault("success", False)
    result.setdefault("cached_result", False)

    return result


async def safe_tool_stream_call(
    stream_func: StreamFunction,
    args: Dict[str, Any],
    description: str,
) -> bool:
    """
    Call a run_*_stream wrapper, printing the stream as it arrives.
    Works whether the wrapper returns the iterator directly or returns it
    inside a coroutine (the current behaviour when decorators are applied).
    """
    tool_name = getattr(stream_func, "__name__", "unknown_stream_tool")
    console.print(
        Rule(f"[bold magenta]Streaming Demo: {escape(description)}[/bold magenta]", style="magenta")
    )
    console.print(f"[dim]Calling [bold cyan]{tool_name}[/] with args:[/]")
    console.print(pretty_repr(args, max_length=120, max_string=200))

    # ─── call the wrapper ────────────────────────────────────────────────────────
    stream_obj = stream_func(**args)  # do *not* await yet
    if inspect.iscoroutine(stream_obj):  # decorator returned coroutine
        stream_obj = await stream_obj  # now we have AsyncIterator

    if not hasattr(stream_obj, "__aiter__"):
        console.print(
            Panel(
                f"[red]Fatal: {tool_name} did not return an async iterator.[/red]",
                border_style="red",
            )
        )
        return False

    # ─── consume the stream  ─────────────────────────────────────────────────────
    start = time.monotonic()
    line_count, buffered = 0, ""
    console.print("[yellow]--- Streaming Output Start ---[/]")

    try:
        async for line in stream_obj:  # type: ignore[arg-type]
            line_count += 1
            buffered += line
            if len(buffered) > 2000 or "\n" in buffered:
                console.out(escape(buffered), end="")
                buffered = ""
        if buffered:
            console.out(escape(buffered), end="")

        status = "[green]Complete"
        ok = True
    except Exception:
        console.print_exception()
        status = "[red]Failed"
        ok = False

    console.print(
        f"\n[yellow]--- Streaming {status} ({line_count} lines in "
        f"{time.monotonic() - start:.3f}s) ---[/]\n"
    )
    return ok


# --- Demo Functions ---


async def demonstrate_ripgrep_basic():
    """Demonstrate basic usage of the run_ripgrep tool."""
    console.print(Rule("[bold green]1. Ripgrep (rg) Basic Examples[/bold green]", style="green"))

    classification_samples_str = str(CLASSIFICATION_SAMPLES_DIR_REL)
    article_file_quoted = shlex.quote(ARTICLE_FILE_PATH)
    class_dir_quoted = shlex.quote(classification_samples_str)

    # 1a: Basic search in a file
    await safe_tool_call(
        run_ripgrep,
        {
            "args_str": f"--threads=4 'Microsoft' {article_file_quoted}",
            "input_file": True,  # Indicate args_str contains the file target
        },
        "Search for 'Microsoft' in article.txt (with thread limiting)",
    )

    # 1b: Case-insensitive search with context
    await safe_tool_call(
        run_ripgrep,
        {
            "args_str": f"--threads=4 -i --context 2 'anthropic' {article_file_quoted}",
            "input_file": True,
        },
        "Case-insensitive search for 'anthropic' with context (-i -C 2, limited threads)",
    )

    # 1c: Search for lines NOT containing a pattern
    await safe_tool_call(
        run_ripgrep,
        {
            "args_str": f"--threads=4 --invert-match 'AI' {article_file_quoted}",
            "input_file": True,
        },
        "Find lines NOT containing 'AI' in article.txt (-v, limited threads)",
    )

    # 1d: Count matches per file in a directory
    await safe_tool_call(
        run_ripgrep,
        {
            "args_str": f"--threads=4 --count-matches 'Subject:' {class_dir_quoted}",
            "input_dir": True,  # Indicate args_str contains the dir target
        },
        "Count lines with 'Subject:' in classification samples dir (-c, limited threads)",
    )

    # 1e: Search within input_data
    sample_data = "Line one\nLine two with pattern\nLine three\nAnother pattern line"
    await safe_tool_call(
        run_ripgrep,
        {"args_str": "--threads=4 'pattern'", "input_data": sample_data},
        "Search for 'pattern' within input_data string (limited threads)",
    )

    # 1f: JSON output
    await safe_tool_call(
        run_ripgrep,
        {
            "args_str": f"--threads=4 --json 'acquisition' {article_file_quoted}",
            "input_file": True,
        },
        "Search for 'acquisition' with JSON output (--json, limited threads)",
    )

    # 1g: Error case - Invalid Regex Pattern (example)
    await safe_tool_call(
        run_ripgrep,
        {"args_str": f"--threads=4 '[' {article_file_quoted}", "input_file": True},
        "Search with potentially invalid regex pattern '[' (INTENTIONAL DEMONSTRATION: regex validation)",
    )


async def demonstrate_ripgrep_advanced():
    """Demonstrate advanced usage of the run_ripgrep tool."""
    console.print(
        Rule("[bold green]1b. Ripgrep (rg) Advanced Examples[/bold green]", style="green")
    )

    contract_file_quoted = shlex.quote(CONTRACT_FILE_PATH)
    class_dir_quoted = shlex.quote(str(CLASSIFICATION_SAMPLES_DIR_REL))

    # Adv 1a: Multiline search (simple example)
    await safe_tool_call(
        run_ripgrep,
        # Search for "ARTICLE I" followed by "Consideration" within 10 lines, case sensitive
        {
            "args_str": f"--threads=4 --multiline --multiline-dotall --context 1 'ARTICLE I.*?Consideration' {contract_file_quoted}",
            "input_file": True,
        },
        "Multiline search for 'ARTICLE I' then 'Consideration' within context (-U -C 1, limited threads)",
    )

    # Adv 1b: Search specific file types and replace output
    await safe_tool_call(
        run_ripgrep,
        # Search for 'Agreement' in .txt files, replace matching text with '***CONTRACT***'
        {
            "args_str": f"--threads=4 --replace '***CONTRACT***' 'Agreement' {contract_file_quoted}",
            "input_file": True,
        },
        "Search for 'Agreement' in contract file and replace in output (--replace, limited threads)",
    )

    # Adv 1c: Using Globs to include/exclude
    # Search for 'email' in classification samples, but exclude the news samples file
    exclude_pattern = shlex.quote(
        os.path.basename(CLASSIFICATION_SAMPLES_DIR_REL / "news_samples.txt")
    )
    await safe_tool_call(
        run_ripgrep,
        {
            "args_str": f"--threads=4 -i 'email' -g '!{exclude_pattern}' {class_dir_quoted}",
            "input_dir": True,
        },
        f"Search for 'email' in classification dir, excluding '{exclude_pattern}' (-g, limited threads)",
    )

    # Adv 1d: Print only matching part with line numbers and context
    await safe_tool_call(
        run_ripgrep,
        # Extract dates like YYYY-MM-DD
        {
            "args_str": f"--threads=4 --only-matching --line-number --context 1 '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}' {contract_file_quoted}",
            "input_file": True,
        },
        "Extract date patterns (YYYY-MM-DD) with line numbers and context (-o -n -C 1, limited threads)",
    )

    # Adv 1e: Follow symlinks (if applicable and symlinks were created in setup)
    # This depends on your setup having symlinks pointing into allowed directories
    # Example assumes a symlink named 'contract_link.txt' points to legal_contract.txt
    link_path = SAMPLE_DIR_ABS / "contract_link.txt"  # Absolute path for creation
    target_path = SAMPLE_DIR_ABS / "legal_contract.txt"  # Absolute path for file operations
    # Create link for demo if target exists
    if target_path.exists() and not link_path.exists():
        try:
            os.symlink(target_path.name, link_path)  # Relative link
            logger.info("Created symlink 'contract_link.txt' for demo.")
        except OSError as e:
            logger.warning(f"Could not create symlink for demo: {e}")

    # Use relative path for the tool
    link_path_rel = (
        link_path.relative_to(PROJECT_ROOT) if link_path.exists() else "nonexistent_link.txt"
    )
    link_path_quoted = shlex.quote(str(link_path_rel))
    await safe_tool_call(
        run_ripgrep,
        {"args_str": f"--threads=4 --follow 'Acquirer' {link_path_quoted}", "input_file": True},
        "Search for 'Acquirer' following symlinks (--follow, limited threads) (requires symlink setup)",
    )


async def demonstrate_awk_basic():
    """Demonstrate basic usage of the run_awk tool."""
    console.print(Rule("[bold green]2. AWK Basic Examples[/bold green]", style="green"))

    email_file_quoted = shlex.quote(EMAIL_FILE_PATH)

    # 2a: Print specific fields (e.g., Subject lines)
    await safe_tool_call(
        run_awk,
        # FS = ':' is the field separator, print second field ($2) if first field is 'Subject'
        {
            "args_str": f"-F ':' '/^Subject:/ {{ print $2 }}' {email_file_quoted}",
            "input_file": True,
        },
        "Extract Subject lines from email sample using AWK (-F ':')",
    )

    # 2b: Count lines containing a specific word using AWK logic
    await safe_tool_call(
        run_awk,
        # Increment count if line contains 'account', print total at the end
        {
            "args_str": f"'/account/ {{ count++ }} END {{ print \"Lines containing account:\", count }}' {email_file_quoted}",
            "input_file": True,
        },
        "Count lines containing 'account' in email sample using AWK",
    )

    # 2c: Process input_data - print first word of each line
    awk_input_data = "Apple Banana Cherry\nDog Elephant Fox\nOne Two Three"
    await safe_tool_call(
        run_awk,
        {"args_str": "'{ print $1 }'", "input_data": awk_input_data},
        "Print first word of each line from input_data using AWK",
    )

    # 2d: Error case - Syntax error in AWK script
    await safe_tool_call(
        run_awk,
        {"args_str": "'{ print $1 '", "input_data": awk_input_data},  # Missing closing brace
        "Run AWK with a syntax error in the script (INTENTIONAL DEMONSTRATION: script validation)",
    )


async def demonstrate_awk_advanced():
    """Demonstrate advanced usage of the run_awk tool."""
    console.print(Rule("[bold green]2b. AWK Advanced Examples[/bold green]", style="green"))

    contract_file_quoted = shlex.quote(CONTRACT_FILE_PATH)
    schedule_file_quoted = shlex.quote(SCHEDULE_FILE_PATH)

    # Adv 2a: Calculate sum based on a field (extracting amounts from contract)
    await safe_tool_call(
        run_awk,
        # Find lines with '$', extract the number after '$', sum them
        {
            "args_str": f'\'/[$]/ {{ gsub(/[,USD$]/, ""); for(i=1;i<=NF;i++) if ($i ~ /^[0-9.]+$/) sum+=$i }} END {{ printf "Total Value Mentioned: $%.2f\\n", sum }}\' {contract_file_quoted}',
            "input_file": True,
        },
        "Sum numeric values following '$' in contract using AWK",
    )

    # Adv 2b: Using BEGIN block and variables to extract definitions
    await safe_tool_call(
        run_awk,
        # Find lines defining terms like ("Acquirer"), print term and line number
        {
            "args_str": f'\'/^\\s*[A-Z][[:alpha:] ]+\\s+\\(.*"[[:alpha:]].*"\\)/ {{ if(match($0, /\\("([^"]+)"\\)/, arr)) {{ term=arr[1]; print "Term Defined: ", term, "(Line: " NR ")" }} }}\' {contract_file_quoted}',
            "input_file": True,
        },
        'Extract defined terms (e.g., ("Acquirer")) using AWK and NR',
    )

    # Adv 2c: Change output field separator and process specific sections
    await safe_tool_call(
        run_awk,
        # In ARTICLE I, print section number and title, comma separated
        {
            "args_str": f'\'BEGIN {{ OFS=","; print "Section,Title" }} /^## ARTICLE I/,/^## ARTICLE II/ {{ if (/^[0-9]\\.[0-9]+\\s/) {{ title=$0; sub(/^[0-9.]+s*/, "", title); print $1, title }} }}\' {contract_file_quoted}',
            "input_file": True,
        },
        "Extract section titles from ARTICLE I, CSV formatted (OFS)",
    )

    # Adv 2d: Associative arrays to count stockholder types from SCHEDULE_1.2 file
    if Path(SCHEDULE_FILE_PATH_ABS).exists():
        await safe_tool_call(
            run_awk,
            # Count occurrences based on text before '(' or '%'
            {
                "args_str": f'-F\'|\' \'/^\\| / && NF>2 {{ gsub(/^ +| +$/, "", $2); types[$2]++ }} END {{ print "Stockholder Counts:"; for (t in types) print t ":", types[t] }}\' {schedule_file_quoted}',
                "input_file": True,
            },
            "Use associative array in AWK to count stockholder types in Schedule 1.2",
        )
    else:
        logger.warning(f"Skipping AWK advanced demo 2d, file not found: {SCHEDULE_FILE_PATH_ABS}")


async def demonstrate_sed_basic():
    """Demonstrate basic usage of the run_sed tool."""
    console.print(Rule("[bold green]3. SED Basic Examples[/bold green]", style="green"))

    article_file_quoted = shlex.quote(ARTICLE_FILE_PATH)

    # 3a: Simple substitution
    await safe_tool_call(
        run_sed,
        {
            "args_str": f"'s/Microsoft/MegaCorp/g' {article_file_quoted}",
            "input_file": True,
        },
        "Replace 'Microsoft' with 'MegaCorp' in article.txt (global)",
    )

    # 3b: Delete lines containing a pattern
    await safe_tool_call(
        run_sed,
        {
            "args_str": f"'/Anthropic/d' {article_file_quoted}",
            "input_file": True,
        },
        "Delete lines containing 'Anthropic' from article.txt",
    )

    # 3c: Print only lines containing a specific pattern (-n + p)
    await safe_tool_call(
        run_sed,
        {
            "args_str": f"-n '/acquisition/p' {article_file_quoted}",
            "input_file": True,
        },
        "Print only lines containing 'acquisition' from article.txt",
    )

    # 3d: Process input_data - change 'line' to 'row'
    sed_input_data = "This is line one.\nThis is line two.\nAnother line."
    await safe_tool_call(
        run_sed,
        {"args_str": "'s/line/row/g'", "input_data": sed_input_data},
        "Replace 'line' with 'row' in input_data string",
    )

    # 3e: Demonstrate blocked in-place edit attempt (security feature)
    await safe_tool_call(
        run_sed,
        {
            "args_str": f"-i 's/AI/ArtificialIntelligence/g' {article_file_quoted}",
            "input_file": True,
        },
        "Attempt in-place edit with sed -i (SECURITY CHECK PASSED: forbidden flag blocked)",
    )

    # 3f: Error case - Unterminated substitute command
    await safe_tool_call(
        run_sed,
        {
            "args_str": "'s/AI/ArtificialIntelligence",
            "input_data": sed_input_data,
        },  # Missing closing quote and delimiter
        "Run SED with an unterminated 's' command (INTENTIONAL DEMONSTRATION: script validation)",
    )


async def demonstrate_sed_advanced():
    """Demonstrate advanced usage of the run_sed tool."""
    console.print(Rule("[bold green]3b. SED Advanced Examples[/bold green]", style="green"))

    contract_file_quoted = shlex.quote(CONTRACT_FILE_PATH)

    # Adv 3a: Multiple commands with -e
    await safe_tool_call(
        run_sed,
        # Command 1: Change 'Agreement' to 'CONTRACT'. Command 2: Delete lines with 'Exhibit'.
        {
            "args_str": f"-e 's/Agreement/CONTRACT/g' -e '/Exhibit/d' {contract_file_quoted}",
            "input_file": True,
        },
        "Use multiple SED commands (-e) for substitution and deletion",
    )

    # Adv 3b: Using address ranges (print ARTICLE III content)
    await safe_tool_call(
        run_sed,
        {
            "args_str": f"-n '/^## ARTICLE III/,/^## ARTICLE IV/p' {contract_file_quoted}",
            "input_file": True,
        },
        "Print content between '## ARTICLE III' and '## ARTICLE IV' using SED addresses",
    )

    # Adv 3c: Substitute only the first occurrence on a line
    await safe_tool_call(
        run_sed,
        # Change only the first 'Company' to 'Firm' on each line
        {
            "args_str": f"'s/Company/Firm/' {contract_file_quoted}",
            "input_file": True,
        },
        "Substitute only the first occurrence of 'Company' per line",
    )

    # Adv 3d: Using capture groups to reformat dates (MM/DD/YYYY -> YYYY-MM-DD)
    # Note: This regex is basic, might not handle all date formats in the text perfectly
    await safe_tool_call(
        run_sed,
        # Capture month, day, year and rearrange
        {
            "args_str": rf"-E 's|([0-9]{{1,2}})/([0-9]{{1,2}})/([0-9]{{4}})|\3-\1-\2|g' {contract_file_quoted}",
            "input_file": True,
        },
        "Rearrange date format (MM/DD/YYYY -> YYYY-MM-DD) using SED capture groups",
    )

    # Adv 3e: Insert text before lines matching a pattern
    await safe_tool_call(
        run_sed,
        # Insert 'IMPORTANT: ' before lines starting with '## ARTICLE'
        {
            "args_str": f"'/^## ARTICLE/i IMPORTANT: ' {contract_file_quoted}",
            "input_file": True,
        },
        "Insert text before lines matching a pattern using SED 'i' command",
    )


async def demonstrate_jq_basic():
    """Demonstrate basic usage of the run_jq tool."""
    console.print(Rule("[bold green]4. JQ Basic Examples[/bold green]", style="green"))

    # Using input_data for most basic examples
    jq_input_data = """
    {
      "id": "wf-123",
      "title": "Data Processing",
      "steps": [
        {"name": "load", "status": "completed", "duration": 5.2},
        {"name": "transform", "status": "running", "duration": null, "details": {"type": "pivot"}},
        {"name": "analyze", "status": "pending", "duration": null}
      ],
      "metadata": {
        "user": "admin",
        "priority": "high"
      }
    }
    """

    # 4a: Select a top-level field
    await safe_tool_call(
        run_jq,
        {"args_str": "'.title'", "input_data": jq_input_data},
        "Select the '.title' field using JQ",
    )

    # 4b: Select a nested field
    await safe_tool_call(
        run_jq,
        {"args_str": "'.metadata.priority'", "input_data": jq_input_data},
        "Select the nested '.metadata.priority' field using JQ",
    )

    # 4c: Select names from the steps array
    await safe_tool_call(
        run_jq,
        {"args_str": "'.steps[].name'", "input_data": jq_input_data},
        "Select all step names from the '.steps' array using JQ",
    )

    # 4d: Filter steps by status
    await safe_tool_call(
        run_jq,
        {"args_str": "'.steps[] | select(.status == \"completed\")'", "input_data": jq_input_data},
        "Filter steps where status is 'completed' using JQ",
    )

    # 4e: Create a new object structure
    await safe_tool_call(
        run_jq,
        # Create a new object with workflow id and number of steps
        {
            "args_str": "'{ workflow: .id, step_count: (.steps | length) }'",
            "input_data": jq_input_data,
        },
        "Create a new object structure using JQ '{ workflow: .id, step_count: .steps | length }'",
    )

    # 4f: Error case - Invalid JQ filter syntax
    await safe_tool_call(
        run_jq,
        {
            "args_str": "'.steps[] | select(.status =)'",
            "input_data": jq_input_data,
        },  # Incomplete select
        "Run JQ with invalid filter syntax (INTENTIONAL DEMONSTRATION: script validation)",
    )

    # 4g: Error case - Process non-JSON input (Input Validation)
    await safe_tool_call(
        run_jq,
        {"args_str": "'.'", "input_data": "This is not JSON."},
        "Run JQ on non-JSON input data (INTENTIONAL DEMONSTRATION: input validation)",
    )

    # 4h: Using a JSON file as input
    if Path(JSON_SAMPLE_PATH_ABS).exists():
        json_file_quoted = shlex.quote(JSON_SAMPLE_PATH)
        await safe_tool_call(
            run_jq,
            {
                "args_str": f"'.[] | select(.dept == \"IT\").user' {json_file_quoted}",
                "input_file": True,
            },
            "Select 'user' from IT department in sample_data.json",
        )
    else:
        logger.warning(f"Skipping JQ basic demo 4h, file not found: {JSON_SAMPLE_PATH_ABS}")


async def demonstrate_jq_advanced():
    """Demonstrate advanced usage of the run_jq tool."""
    console.print(Rule("[bold green]4b. JQ Advanced Examples[/bold green]", style="green"))

    # Using file input for advanced examples
    if not Path(JSON_SAMPLE_PATH_ABS).exists():
        logger.warning(f"Skipping JQ advanced demos, file not found: {JSON_SAMPLE_PATH_ABS}")
        return

    json_file_quoted = shlex.quote(JSON_SAMPLE_PATH)

    # Adv 4a: Map and filter combined (select users with 'active' tag)
    await safe_tool_call(
        run_jq,
        {
            "args_str": f"'.[] | select(.tags | contains([\"active\"])) | .user' {json_file_quoted}",
            "input_file": True,
        },
        "JQ: Select users with the 'active' tag using 'contains' from file",
    )

    # Adv 4b: Group by department and calculate average value
    # Note: jq 'group_by' produces nested arrays, requires map to process
    await safe_tool_call(
        run_jq,
        {
            "args_str": f"'group_by(.dept) | map({{department: .[0].dept, avg_value: (map(.value) | add / length)}})' {json_file_quoted}",
            "input_file": True,
        },
        "JQ: Group by 'dept' and calculate average 'value' from file",
    )

    # Adv 4c: Using variables and checking multiple conditions
    await safe_tool_call(
        run_jq,
        # Find IT users from South or West with value > 120
        {
            "args_str": f'\'map(select(.dept == "IT" and (.region == "South" or .region == "West") and .value > 120))\' {json_file_quoted}',
            "input_file": True,
        },
        "JQ: Complex select with multiple AND/OR conditions from file",
    )

    # Adv 4d: Raw output (-r) to get just text values
    await safe_tool_call(
        run_jq,
        # Output user names directly without JSON quotes
        {"args_str": f"-r '.[] | .user' {json_file_quoted}", "input_file": True},
        "JQ: Get raw string output using -r flag from file",
    )


async def demonstrate_security_features():
    """Demonstrate argument validation and security features."""
    console.print(Rule("[bold red]5. Security Feature Demonstrations[/bold red]", style="red"))

    target_file_quoted = shlex.quote(ARTICLE_FILE_PATH)
    workspace = get_workspace_dir()  # Get the actual workspace for context  # noqa: F841

    # Sec 1: Forbidden flag (-i for sed) - Already in sed_basic, ensure it's shown clearly
    console.print("[dim]--- Test: Forbidden Flag ---[/]")
    await safe_tool_call(
        run_sed,
        {
            "args_str": f"-i 's/AI/ArtificialIntelligence/g' {target_file_quoted}",
            "input_file": True,
        },
        "Attempt in-place edit with sed -i (SECURITY CHECK PASSED: forbidden flag blocked)",
    )

    # Sec 2: Forbidden characters (e.g., > for redirection)
    console.print("[dim]--- Test: Forbidden Characters ---[/]")
    await safe_tool_call(
        run_awk,
        {"args_str": "'{ print $1 > \"output.txt\" }'", "input_data": "hello world"},
        "Attempt redirection with awk '>' (SECURITY CHECK PASSED: forbidden operation blocked)",
    )

    # Sec 3: Command substitution attempt
    console.print("[dim]--- Test: Command Substitution ---[/]")
    await safe_tool_call(
        run_ripgrep,
        {
            "args_str": f"--threads=4 'pattern' `echo {target_file_quoted}`",
            "input_file": True,
            "input_dir": False,
        },  # Input from args only
        "Attempt command substitution with backticks `` (SECURITY CHECK PASSED: command injection blocked)",
    )
    await safe_tool_call(
        run_ripgrep,
        {
            "args_str": f"--threads=4 'pattern' $(basename {target_file_quoted})",
            "input_file": True,
            "input_dir": False,
        },
        "Attempt command substitution with $() (SECURITY CHECK PASSED: command injection blocked)",
    )

    # Sec 4: Path Traversal
    console.print("[dim]--- Test: Path Traversal ---[/]")
    # Choose a target likely outside the workspace
    traversal_path = (
        "../../etc/passwd"
        if sys.platform != "win32"
        else "..\\..\\Windows\\System32\\drivers\\etc\\hosts"
    )
    traversal_path_quoted = shlex.quote(traversal_path)
    await safe_tool_call(
        run_ripgrep,
        {"args_str": f"--threads=4 'root' {traversal_path_quoted}", "input_file": True},
        f"Attempt path traversal '{traversal_path}' (SECURITY CHECK PASSED: path traversal blocked)",
    )

    # Sec 5: Absolute Path
    console.print("[dim]--- Test: Absolute Path ---[/]")
    # Use a known absolute path
    abs_path = str(
        Path(target_file_quoted).resolve()
    )  # Should be inside workspace IF demo runs from there, but treat as example
    abs_path_quoted = shlex.quote(abs_path)  # noqa: F841
    # Let's try a known outside-workspace path if possible
    abs_outside_path = "/tmp/testfile" if sys.platform != "win32" else "C:\\Windows\\notepad.exe"
    abs_outside_path_quoted = shlex.quote(abs_outside_path)

    await safe_tool_call(
        run_ripgrep,
        {"args_str": f"--threads=4 'test' {abs_outside_path_quoted}", "input_file": True},
        f"Attempt absolute path '{abs_outside_path}' (SECURITY CHECK PASSED: absolute path blocked)",
    )

    # Sec 6: Dry Run
    console.print("[dim]--- Test: Dry Run ---[/]")
    await safe_tool_call(
        run_ripgrep,
        {
            "args_str": f"--json -i 'pattern' {target_file_quoted}",
            "input_file": True,
            "dry_run": True,
        },
        "Demonstrate dry run (--json -i 'pattern' <file>)",
    )


async def demonstrate_streaming():
    """Demonstrate the streaming capabilities."""
    console.print(Rule("[bold magenta]6. Streaming Examples[/bold magenta]", style="magenta"))

    # Use a file likely to produce multiple lines of output
    target_file_quoted = shlex.quote(CONTRACT_FILE_PATH)

    # Stream 1: Ripgrep stream for a common word
    await safe_tool_stream_call(
        run_ripgrep_stream,
        {"args_str": f"--threads=4 -i 'Agreement' {target_file_quoted}", "input_file": True},
        "Stream search results for 'Agreement' in contract (with thread limiting)",
    )

    # Stream 2: Sed stream to replace and print
    await safe_tool_stream_call(
        run_sed_stream,
        {"args_str": f"'s/Section/Clause/g' {target_file_quoted}", "input_file": True},
        "Stream sed output replacing 'Section' with 'Clause'",
    )

    # Stream 3: Awk stream to print fields
    await safe_tool_stream_call(
        run_awk_stream,
        {
            "args_str": f"'/^##/ {{print \"Found Section: \", $0}}' {target_file_quoted}",
            "input_file": True,
        },
        "Stream awk output printing lines starting with '##'",
    )

    # Stream 4: JQ stream on input data
    jq_stream_input = """
{"id": 1, "value": "alpha"}
{"id": 2, "value": "beta"}
{"id": 3, "value": "gamma"}
{"id": 4, "value": "delta"}
    """
    await safe_tool_stream_call(
        run_jq_stream,
        {"args_str": "'.value'", "input_data": jq_stream_input},
        "Stream jq extracting '.value' from multiple JSON objects",
    )


# --- LLM Interactive Workflow Section ---
# NOTE: run_llm_interactive_workflow helper remains largely the same,
#       but system prompts are updated below.


async def run_llm_interactive_workflow(
    goal: str,
    system_prompt: str,
    target_file: Optional[str] = None,
    initial_input_data: Optional[str] = None,
):
    """Runs an interactive workflow driven by an LLM using the text tool functions."""
    # --- LLM Config Check ---
    llm_provider_name = None
    llm_model_name = None
    try:
        config = get_config()
        # Use configured default provider or fallback
        llm_provider_name = config.default_provider or Provider.OPENAI.value
        provider_config = getattr(config.providers, llm_provider_name, None)
        if not provider_config or not provider_config.api_key:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] LLM provider '{llm_provider_name}' API key not configured."
            )
            console.print("Skipping this LLM interactive workflow demo.")
            return False  # Indicate skip
        llm_model_name = provider_config.default_model  # Use provider's default (can be None)
        if not llm_model_name:
            # Try a known default if provider default is missing
            if llm_provider_name == Provider.OPENAI.value:
                llm_model_name = "gpt-3.5-turbo"
            elif llm_provider_name == Provider.ANTHROPIC.value:
                llm_model_name = "claude-3-5-haiku-20241022"  # Use a valid model without comments
            # Add other provider fallbacks if needed
            else:
                llm_model_name = "default"  # Placeholder if truly unknown

            if llm_model_name != "default":
                logger.info(
                    f"No default model for provider '{llm_provider_name}', using fallback: {llm_model_name}"
                )
            else:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Could not determine default model for provider '{llm_provider_name}'. LLM calls might fail."
                )

    except Exception as e:
        console.print(f"[bold red]Error checking LLM configuration:[/bold red] {e}")
        console.print("Skipping this LLM interactive workflow demo.")
        return False  # Indicate skip

    # --- Workflow Setup ---
    console.print(
        Panel(f"[bold]Goal:[/bold]\n{escape(goal)}", title="LLM Task", border_style="blue")
    )
    messages = [{"role": "system", "content": system_prompt}]
    # Add initial content if provided
    if target_file:
        messages.append(
            {"role": "user", "content": f"The primary target file for operations is: {target_file}"}
        )
    elif initial_input_data:
        messages.append(
            {
                "role": "user",
                "content": f"The input data to process is:\n```\n{initial_input_data[:1000]}\n```",
            }
        )

    # --- Helper to call LLM ---
    async def run_llm_step(history: List[Dict]) -> Optional[Dict]:
        # (This helper remains largely the same as before, relying on imported chat_completion)
        try:
            llm_response = await chat_completion(
                provider=llm_provider_name,  # type: ignore
                model=llm_model_name,
                messages=history,
                temperature=0.1,
                max_tokens=600,  # Increased slightly for potentially complex plans
                additional_params={
                    "json_mode": True
                },  # Pass json_mode through additional_params instead
            )
            if not llm_response.get("success"):
                error_detail = llm_response.get("error", "Unknown error")
                console.print(f"[bold red]LLM call failed:[/bold red] {error_detail}")
                # Provide feedback to LLM about the failure
                history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "tool": "error",
                                "args": {"reason": f"LLM API call failed: {error_detail}"},
                            }
                        ),
                    }
                )
                history.append(
                    {
                        "role": "user",
                        "content": "Your previous response resulted in an API error. Please check your request and try again, ensuring valid JSON output.",
                    }
                )
                # Try one more time after feedback
                llm_response = await chat_completion(
                    provider=llm_provider_name,  # type: ignore
                    model=llm_model_name,
                    messages=history,
                    temperature=0.15,  # Slightly higher temp for retry
                    max_tokens=600,
                    additional_params={
                        "json_mode": True
                    },  # Pass json_mode through additional_params here too
                )
                if not llm_response.get("success"):
                    console.print(
                        f"[bold red]LLM call failed on retry:[/bold red] {llm_response.get('error')}"
                    )
                    return None  # Give up after retry

            llm_content = llm_response.get("message", {}).get("content", "").strip()

            # Attempt to parse the JSON directly
            try:
                # Handle potential ```json blocks if provider doesn't strip them in JSON mode
                if llm_content.startswith("```json"):
                    llm_content = re.sub(r"^```json\s*|\s*```$", "", llm_content, flags=re.DOTALL)

                parsed_action = json.loads(llm_content)
                if (
                    isinstance(parsed_action, dict)
                    and "tool" in parsed_action
                    and "args" in parsed_action
                ):
                    # Basic validation of args structure
                    if not isinstance(parsed_action["args"], dict):
                        raise ValueError("LLM 'args' field is not a dictionary.")
                    return parsed_action
                else:
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] LLM response is valid JSON but lacks 'tool' or 'args'. Raw:\n",
                        llm_content,
                    )
                    return {
                        "tool": "error",
                        "args": {
                            "reason": "LLM response structure invalid (expected top-level 'tool' and 'args' keys in JSON)."
                        },
                    }
            except (json.JSONDecodeError, ValueError) as json_err:
                console.print(
                    f"[bold red]Error:[/bold red] LLM response was not valid JSON ({json_err}). Raw response:\n",
                    llm_content,
                )
                # Try to find tool name even in broken JSON for feedback
                tool_match = re.search(r'"tool":\s*"(\w+)"', llm_content)
                reason = f"LLM response was not valid JSON ({json_err})."
                if tool_match:
                    reason += f" It mentioned tool '{tool_match.group(1)}'."
                return {"tool": "error", "args": {"reason": reason}}
        except Exception as e:
            console.print(f"[bold red]Error during LLM interaction:[/bold red] {e}")
            logger.error("LLM interaction error", exc_info=True)
            return None

    # Map tool names from LLM response to actual functions
    TOOL_FUNCTIONS = {
        "run_ripgrep": run_ripgrep,
        "run_awk": run_awk,
        "run_sed": run_sed,
        "run_jq": run_jq,
        # Add streaming if needed, but LLM needs careful prompting for stream handling
        # "run_ripgrep_stream": run_ripgrep_stream,
    }

    # --- Iteration Loop ---
    for i in range(MAX_LLM_ITERATIONS):
        console.print(Rule(f"[bold]LLM Iteration {i + 1}/{MAX_LLM_ITERATIONS}[/bold]"))

        llm_action = await run_llm_step(messages)
        if not llm_action:
            console.print("[bold red]Failed to get valid action from LLM. Stopping.[/bold red]")
            break

        # Append LLM's raw action choice to history BEFORE execution
        messages.append({"role": "assistant", "content": json.dumps(llm_action)})

        tool_name = llm_action.get("tool")
        tool_args = llm_action.get("args", {})  # Should be a dict if validation passed

        console.print(f"[magenta]LLM Planned Action:[/magenta] Tool = {tool_name}")
        console.print(f"[magenta]LLM Args:[/magenta] {pretty_repr(tool_args)}")

        if tool_name == "finish":
            console.print(Rule("[bold green]LLM Finished[/bold green]", style="green"))
            console.print("[bold green]Final Answer:[/bold green]")
            final_answer = tool_args.get("final_answer", "No final answer provided.")
            # Display potential JSON nicely
            try:
                # Attempt to parse if it looks like JSON, otherwise print escaped string
                if isinstance(final_answer, str) and final_answer.strip().startswith(("{", "[")):
                    parsed_answer = json.loads(final_answer)
                    console.print(
                        Syntax(json.dumps(parsed_answer, indent=2), "json", theme="monokai")
                    )
                else:
                    console.print(escape(str(final_answer)))  # Ensure it's a string
            except json.JSONDecodeError:
                console.print(escape(str(final_answer)))  # Print escaped string on parse fail
            break
        if tool_name == "error":
            console.print(Rule("[bold red]LLM Reported Error[/bold red]", style="red"))
            console.print(
                f"[bold red]Reason:[/bold red] {escape(tool_args.get('reason', 'No reason provided.'))}"
            )
            # Don't break immediately, let LLM try again based on this error feedback
            messages.append(
                {
                    "role": "user",
                    "content": f"Your previous step resulted in an error state: {tool_args.get('reason')}. Please analyze the issue and plan the next step or finish.",
                }
            )
            continue  # Allow LLM to react to its own error report

        tool_func_to_call = TOOL_FUNCTIONS.get(tool_name)

        if not tool_func_to_call:
            error_msg = f"LLM requested invalid or unsupported tool: '{tool_name}'. Allowed: {list(TOOL_FUNCTIONS.keys())}"
            console.print(f"[bold red]Error:[/bold red] {error_msg}")
            messages.append(
                {
                    "role": "user",
                    "content": f"Execution Error: {error_msg}. Please choose a valid tool from the allowed list.",
                }
            )
            continue

        # Basic validation of common args
        if "args_str" not in tool_args or not isinstance(tool_args["args_str"], str):
            error_msg = f"LLM tool call for '{tool_name}' is missing 'args_str' string argument."
            console.print(f"[bold red]Error:[/bold red] {error_msg}")
            messages.append({"role": "user", "content": f"Input Error: {error_msg}"})
            continue

        # Inject target file/data if not explicitly set by LLM but context suggests it
        # Less critical now LLM is prompted to include path in args_str and set flags
        if (
            "input_file" not in tool_args
            and "input_dir" not in tool_args
            and "input_data" not in tool_args
        ):
            # Simple heuristic: if target_file seems to be in args_str, set input_file=True
            if target_file and shlex.quote(target_file) in tool_args.get("args_str", ""):
                tool_args["input_file"] = True
                logger.debug(f"Injecting input_file=True based on args_str content: {target_file}")
            # Maybe inject input_data if available and no file/dir flags? Risky.
            # Let's rely on the LLM providing the flags or safe_tool_call catching errors.

        # Execute tool using the safe helper
        execution_result = await safe_tool_call(
            tool_func_to_call,
            tool_args,  # Pass the dict received from LLM
            f"Executing LLM Request: {tool_name}",
            display_input=False,  # Already printed LLM args
            display_output=False,  # Summarize below for LLM context
        )

        # Prepare result summary for LLM (Truncate long outputs)
        result_summary_for_llm = ""
        if isinstance(execution_result, dict):
            success = execution_result.get("success", False)
            stdout_preview = (execution_result.get("stdout", "") or "")[:1500]  # Limit length
            stderr_preview = (execution_result.get("stderr", "") or "")[:500]
            stdout_trunc = execution_result.get("stdout_truncated", False)
            stderr_trunc = execution_result.get("stderr_truncated", False)
            exit_code = execution_result.get("exit_code")
            error_msg = execution_result.get("error")
            error_code = execution_result.get("error_code")

            result_summary_for_llm = f"Tool Execution Result ({tool_name}):\n"
            result_summary_for_llm += f"Success: {success}\n"
            result_summary_for_llm += f"Exit Code: {exit_code}\n"
            if error_msg:
                result_summary_for_llm += f"Error: {error_msg}\n"
            if error_code:
                if isinstance(error_code, Enum):
                    error_code_repr = error_code.value
                else:
                    error_code_repr = str(error_code)
                result_summary_for_llm += f"Error Code: {error_code_repr}\n"

            stdout_info = f"STDOUT ({len(stdout_preview)} chars preview{' - TRUNCATED' if stdout_trunc else ''}):"
            result_summary_for_llm += f"{stdout_info}\n```\n{stdout_preview}\n```\n"

            if stderr_preview:
                stderr_info = f"STDERR ({len(stderr_preview)} chars preview{' - TRUNCATED' if stderr_trunc else ''}):"
                result_summary_for_llm += f"{stderr_info}\n```\n{stderr_preview}\n```\n"
            else:
                result_summary_for_llm += "STDERR: (empty)\n"
        else:  # Should not happen if safe_tool_call works
            result_summary_for_llm = (
                f"Tool Execution Error: Unexpected result format: {type(execution_result)}"
            )

        console.print(
            "[cyan]Execution Result Summary (for LLM):[/]", escape(result_summary_for_llm)
        )
        # Append the outcome back to the message history for the LLM's next turn
        messages.append({"role": "user", "content": result_summary_for_llm})

        if i == MAX_LLM_ITERATIONS - 1:
            console.print(Rule("[bold yellow]Max Iterations Reached[/bold yellow]", style="yellow"))
            console.print("Stopping LLM workflow.")
            break

    return True  # Indicate demo ran (or attempted to run)


async def demonstrate_llm_workflow_extract_contacts():
    """LLM Workflow: Extract email addresses and phone numbers from legal_contract.txt."""
    console.print(
        Rule("[bold cyan]7. LLM Workflow: Extract Contacts from Contract[/bold cyan]", style="cyan")
    )
    goal = f"Extract all unique email addresses and phone numbers (in standard format like XXX-XXX-XXXX or (XXX) XXX-XXXX) from the file: {CONTRACT_FILE_PATH}. Present the results clearly as two distinct lists (emails, phone numbers) in your final answer JSON."
    # Updated system prompt for standalone functions
    system_prompt = rf"""
You are an expert AI assistant tasked with extracting information from text using command-line tools accessed via functions.
Your goal is: {goal}
The primary target file is: {CONTRACT_FILE_PATH}

You have access to the following functions:
- `run_ripgrep(args_str: str, input_file: bool = False, input_data: Optional[str] = None, ...)`: For regex searching.
- `run_awk(args_str: str, input_file: bool = False, input_data: Optional[str] = None, ...)`: For text processing.
- `run_sed(args_str: str, input_file: bool = False, input_data: Optional[str] = None, ...)`: For text transformation.

To operate on the target file, you MUST:
1. Include the correctly quoted file path in the `args_str`. Use '{shlex.quote(CONTRACT_FILE_PATH)}'.
2. Set `input_file=True` in the arguments dictionary.

Example `run_ripgrep` call structure for a file:
{{
  "tool": "run_ripgrep",
  "args": {{
    "args_str": "-oN 'pattern' {shlex.quote(CONTRACT_FILE_PATH)}",
    "input_file": true
  }}
}}

Example `run_awk` call structure for stdin:
{{
  "tool": "run_awk",
  "args": {{
    "args_str": "'{{print $1}}'",
    "input_data": "some input data here"
  }}
}}

Plan your steps carefully:
1. Use `run_ripgrep` with appropriate regex patterns to find emails and phone numbers. Use flags like `-o` (only matching), `-N` (no line numbers), `--no-filename`.
2. You might need separate `run_ripgrep` calls for emails and phone numbers.
3. Consider using `run_awk` or `run_sed` on the output of `run_ripgrep` (passed via `input_data`) to normalize or unique sort the results, OR present the unique lists in your final answer. A simple approach is often best.
4. When finished, respond with `tool: "finish"` and provide the final answer in the specified format within `args: {{"final_answer": ...}}`.

Respond ONLY with a valid JSON object representing the next single action (tool and args) or the final answer. Do not add explanations outside the JSON.
"""
    await run_llm_interactive_workflow(goal, system_prompt, target_file=CONTRACT_FILE_PATH)


async def demonstrate_llm_workflow_financial_terms():
    """LLM Workflow: Extract key financial figures from legal_contract.txt."""
    console.print(
        Rule(
            "[bold cyan]8. LLM Workflow: Extract Financial Terms from Contract[/bold cyan]",
            style="cyan",
        )
    )
    goal = f"Extract the exact 'Transaction Value', 'Cash Consideration', and 'Stock Consideration' figures (including USD amounts) mentioned in ARTICLE I of the file: {CONTRACT_FILE_PATH}. Also find the 'Escrow Amount' percentage and the Escrow Agent's name. Structure the final answer as a JSON object."
    # Updated system prompt
    system_prompt = rf"""
You are an AI assistant specialized in analyzing legal documents using command-line tools accessed via functions.
Your goal is: {goal}
The target file is: {CONTRACT_FILE_PATH}

Available functions: `run_ripgrep`, `run_awk`, `run_sed`.
Remember to include the quoted file path '{shlex.quote(CONTRACT_FILE_PATH)}' in `args_str` and set `input_file=True` when operating on the file.

Plan your steps:
1. Use `run_ripgrep` to find relevant lines in ARTICLE I (e.g., search for 'Consideration', '$', 'USD', 'Escrow'). Use context flags like `-A`, `-C` to get surrounding lines if needed.
2. Use `run_ripgrep` again or `run_sed`/`run_awk` on the previous output (passed via `input_data`) or the original file to isolate the exact monetary figures (e.g., '$XXX,XXX,XXX USD') and the Escrow Agent name. Regex like `\$\d{{1,3}}(,\d{{3}})*(\.\d+)?\s*USD` might be useful. Be specific with your patterns.
3. Combine the extracted information into a JSON object for the `final_answer`.

Respond ONLY with a valid JSON object for the next action or the final answer (`tool: "finish"`).
"""
    await run_llm_interactive_workflow(goal, system_prompt, target_file=CONTRACT_FILE_PATH)


async def demonstrate_llm_workflow_defined_terms():
    """LLM Workflow: Extract defined terms like ("Acquirer") from legal_contract.txt."""
    console.print(
        Rule(
            "[bold cyan]9. LLM Workflow: Extract Defined Terms from Contract[/bold cyan]",
            style="cyan",
        )
    )
    goal = f'Find all defined terms enclosed in parentheses and quotes, like ("Acquirer"), in the file: {CONTRACT_FILE_PATH}. List the unique terms found in the final answer.'
    # Updated system prompt
    system_prompt = rf"""
You are an AI assistant skilled at extracting specific patterns from text using command-line tools accessed via functions.
Your goal is: {goal}
The target file is: {CONTRACT_FILE_PATH}

Available functions: `run_ripgrep`, `run_awk`, `run_sed`.
Remember to include the quoted file path '{shlex.quote(CONTRACT_FILE_PATH)}' in `args_str` and set `input_file=True` when operating on the file.

Plan your steps:
1. Use `run_ripgrep` with a regular expression to capture text inside `("...")`. The pattern should capture the content within the quotes. Use the `-o` flag for only matching parts, `-N` for no line numbers, `--no-filename`. Example regex: `\(\"([A-Za-z ]+)\"\)` (you might need to adjust escaping for rg's syntax within `args_str`).
2. Process the output to get unique terms. You could pipe the output of ripgrep into awk/sed using `input_data`, e.g., `run_awk` with `'!seen[$0]++'` to get unique lines, or just list unique terms in the final answer.
3. Respond ONLY with the JSON for the next action or the final answer (`tool: "finish"`).
"""
    await run_llm_interactive_workflow(goal, system_prompt, target_file=CONTRACT_FILE_PATH)


# --- Main Execution ---


async def main():
    """Run all LocalTextTools demonstrations."""
    console.print(
        Rule(
            "[bold magenta]Local Text Tools Demo (Standalone Functions)[/bold magenta]",
            style="white",
        )
    )

    # Check command availability (uses the new _COMMAND_METADATA if accessible, otherwise shutil.which)
    console.print("Checking availability of required command-line tools...")
    available_tools: Dict[str, bool] = {}
    missing_tools: List[str] = []
    commands_to_check = ["rg", "awk", "sed", "jq"]  # Commands used in demo
    try:
        # Try accessing the (internal) metadata if possible for accurate check
        from ultimate_mcp_server.tools.local_text_tools import _COMMAND_METADATA

        for cmd, meta in _COMMAND_METADATA.items():
            if cmd in commands_to_check:
                if meta.path and meta.path.exists():
                    available_tools[cmd] = True
                    console.print(f"[green]✓ {cmd} configured at: {meta.path}[/green]")
                else:
                    available_tools[cmd] = False
                    missing_tools.append(cmd)
                    status = "Not Found" if not meta.path else "Path Not Found"
                    console.print(f"[bold red]✗ {cmd} {status}[/bold red]")
        # Check any commands not in metadata via simple which
        for cmd in commands_to_check:
            if cmd not in available_tools:
                if shutil.which(cmd):
                    available_tools[cmd] = True
                    console.print(f"[green]✓ {cmd} found via shutil.which[/green]")
                else:
                    available_tools[cmd] = False
                    missing_tools.append(cmd)
                    console.print(f"[bold red]✗ {cmd} NOT FOUND[/bold red]")

    except ImportError:
        # Fallback to simple check if internal metadata not accessible
        logger.warning("Could not access internal _COMMAND_METADATA, using shutil.which fallback.")
        for cmd in commands_to_check:
            if shutil.which(cmd):
                available_tools[cmd] = True
                console.print(f"[green]✓ {cmd} found via shutil.which[/green]")
            else:
                available_tools[cmd] = False
                missing_tools.append(cmd)
                console.print(f"[bold red]✗ {cmd} NOT FOUND[/bold red]")

    if missing_tools:
        console.print(
            f"\n[bold yellow]Warning:[/bold yellow] The following tools seem missing or not configured: {', '.join(missing_tools)}"
        )
        console.print("Demonstrations requiring these tools will likely fail.")
        console.print("Please install them and ensure they are in your system's PATH.")
        console.print("-" * 30)

    # No instantiation needed for standalone functions

    # --- Basic Demos ---
    if available_tools.get("rg"):
        await demonstrate_ripgrep_basic()
    if available_tools.get("awk"):
        await demonstrate_awk_basic()
    if available_tools.get("sed"):
        await demonstrate_sed_basic()
    if available_tools.get("jq"):
        await demonstrate_jq_basic()

    # --- Advanced Demos ---
    if available_tools.get("rg"):
        await demonstrate_ripgrep_advanced()
    if available_tools.get("awk"):
        await demonstrate_awk_advanced()
    if available_tools.get("sed"):
        await demonstrate_sed_advanced()
    if available_tools.get("jq"):
        await demonstrate_jq_advanced()

    # --- Security Demos ---
    # These demos don't strictly require the tool to *succeed*, just to be called
    # Run them even if some tools might be missing, to show validation layer
    await demonstrate_security_features()

    # --- Streaming Demos ---
    if all(available_tools.get(cmd) for cmd in ["rg", "awk", "sed", "jq"]):
        await demonstrate_streaming()
    else:
        console.print(
            Rule(
                "[yellow]Skipping Streaming Demos (One or more tools missing)[/yellow]",
                style="yellow",
            )
        )

    # --- LLM Workflow Demos ---
    llm_available = False
    try:
        config = get_config()
        provider_key = config.default_provider or Provider.OPENAI.value  # Check default or fallback
        if (
            config.providers
            and getattr(config.providers, provider_key, None)
            and getattr(config.providers, provider_key).api_key
        ):
            llm_available = True
        else:
            logger.warning(f"LLM provider '{provider_key}' API key not configured.")
    except Exception as e:
        logger.warning(f"Could not verify LLM provider configuration: {e}")

    if llm_available and all(
        available_tools.get(cmd) for cmd in ["rg", "awk", "sed"]
    ):  # Check tools needed by LLM demos
        llm_demo_ran = await demonstrate_llm_workflow_extract_contacts()
        if llm_demo_ran:
            await demonstrate_llm_workflow_financial_terms()
        if llm_demo_ran:
            await demonstrate_llm_workflow_defined_terms()
    else:
        reason = (
            "LLM Provider Not Configured/Available"
            if not llm_available
            else "One or more required tools (rg, awk, sed) missing"
        )
        console.print(
            Rule(f"[yellow]Skipping LLM Workflow Demos ({reason})[/yellow]", style="yellow")
        )

    console.print(Rule("[bold green]Local Text Tools Demo Complete[/bold green]", style="green"))
    return 0


if __name__ == "__main__":
    # Run the demo
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Demo interrupted by user.[/bold yellow]")
        sys.exit(1)
