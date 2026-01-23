#!/usr/bin/env python
"""Comprehensive demonstration script for PythonSandbox tools in Ultimate MCP Server."""

# --- Standard Library Imports ---
import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# --- Configuration & Path Setup ---
# Add project root to path for imports when running as script
# Adjust this path if your script location relative to the project root differs
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if not (PROJECT_ROOT / "ultimate_mcp_server").is_dir():  # Check for the actual package dir
        # Fallback if running from a different structure (e.g., examples dir directly)
        PROJECT_ROOT = (
            Path(__file__).resolve().parent.parent.parent
        )  # Go up two levels if in examples
        if not (PROJECT_ROOT / "ultimate_mcp_server").is_dir():
            print(
                "Error: Could not reliably determine project root. Make sure 'ultimate_mcp_server' is importable.",
                file=sys.stderr,
            )
            sys.exit(1)
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"DEBUG: Added '{PROJECT_ROOT}' to sys.path")
except Exception as e:
    print(f"Error during initial path setup: {e}", file=sys.stderr)
    sys.exit(1)

# --- IMPORTANT: Playwright Check FIRST ---
# The sandbox relies heavily on Playwright. Check availability early.
try:
    import playwright.async_api as pw  # noqa F401 - Check import

    PLAYWRIGHT_AVAILABLE_DEMO = True
except ImportError:
    PLAYWRIGHT_AVAILABLE_DEMO = False
    print(
        "[ERROR] Playwright library not found. Please install it (`pip install playwright && playwright install chromium`).",
        file=sys.stderr,
    )
    # Exit immediately if Playwright is crucial for the demo's purpose
    sys.exit(1)

# --- Defer ultimate_mcp_server imports AFTER path setup ---
# Import Rich components
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.traceback import install as install_rich_traceback

# Import necessary tool functions and exceptions
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError
from ultimate_mcp_server.tools.python_sandbox import (
    _close_all_sandboxes,  # Import cleanup function
    display_sandbox_result,
    execute_python,
    repl_python,
)
from ultimate_mcp_server.utils import get_logger

# Use the generic display helper and make a sandbox-specific one
from ultimate_mcp_server.utils.display import safe_tool_call
from ultimate_mcp_server.utils.logging.console import console

# --- Logger and Constants ---
logger = get_logger("example.python_sandbox")
# Use a unique session ID for REPL tests
REPL_SESSION_HANDLE = f"demo-repl-{uuid.uuid4().hex[:8]}"

# Install rich tracebacks for better error display
install_rich_traceback(show_locals=False, width=console.width)

# --- Enhanced Display Helper (from older script) ---


# --- Argument Parsing ---
def parse_arguments():
    """Parse command line arguments for the demo."""
    parser = argparse.ArgumentParser(
        description="Python Sandbox Demo for Ultimate MCP Server Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Available demos:
  all           - Run all demos (default)
  basic         - Basic execution, stdout/stderr, result capture
  errors        - Syntax and Runtime error handling
  timeout       - Execution timeout handling
  packages      - Package loading (numpy, pandas)
  wheels        - Wheel loading via micropip (requires --allow-network)
  repl          - Persistent REPL state and reset functionality
  security      - Network and filesystem access controls
  visualization - Data visualization using matplotlib (requires package)
""",
    )

    parser.add_argument(
        "demo",
        nargs="?",
        default="all",
        choices=[
            "all",
            "basic",
            "errors",
            "timeout",
            "packages",
            "wheels",
            "repl",
            "security",
            "visualization",
        ],
        help="Specific demo to run (default: all)",
    )

    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Enable network access within the sandbox for demos requiring it (e.g., wheel loading)",
    )

    parser.add_argument(
        "--allow-fs",
        action="store_true",
        help="Enable filesystem access bridge (mcpfs) within the sandbox. Requires filesystem tool to be configured.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity (Note: internal tool logging is often DEBUG level)",
    )

    return parser.parse_args()


# --- Demo Functions ---


async def demonstrate_basic_execution(args):
    """Demonstrate basic code execution, I/O capture, results."""
    console.print(Rule("[bold cyan]1. Basic Execution & I/O[/bold cyan]", style="cyan"))
    logger.info("Demonstrating basic execution...", emoji_key="code")

    # --- Simple Execution with Result ---
    code_simple = """
result = 40 + 2
print("Calculation done.")
"""
    result = await safe_tool_call(
        execute_python,
        {"code": code_simple},
        description="Executing simple addition (result = 40 + 2)",
    )
    display_sandbox_result("Basic Addition", result, code_simple)

    # --- Stdout/Stderr Capture ---
    code_io = """
import sys
print("Hello to stdout!")
print("This is line 2 on stdout.")
print("Error message to stderr!", file=sys.stderr)
print("Another error line!", file=sys.stderr)
result = "IO test complete"
"""
    result = await safe_tool_call(
        execute_python, {"code": code_io}, description="Capturing stdout and stderr"
    )
    display_sandbox_result("Stdout/Stderr Capture", result, code_io)

    # --- No 'result' variable assigned ---
    code_no_result = """
x = 10
y = 20
print(f"x + y = {x+y}")
# 'result' variable is not assigned
"""
    result = await safe_tool_call(
        execute_python,
        {"code": code_no_result},
        description="Executing code without assigning to 'result'",
    )
    display_sandbox_result("No 'result' Variable Assigned", result, code_no_result)


async def demonstrate_error_handling(args):
    """Demonstrate handling of syntax and runtime errors."""
    console.print(Rule("[bold cyan]2. Error Handling Demo[/bold cyan]", style="cyan"))
    logger.info("Starting error handling demo")

    # --- Syntax Error ---
    code_syntax_error = "result = 1 +"  # Missing operand
    result = await safe_tool_call(
        execute_python,
        {"code": code_syntax_error},
        description="Executing code with SyntaxError (should fail)",
    )
    display_sandbox_result("Syntax Error Handling", result, code_syntax_error)

    # --- Runtime Error ---
    code_runtime_error = """
def divide(a, b):
    return a / b
result = divide(10, 0) # ZeroDivisionError
"""
    result = await safe_tool_call(
        execute_python,
        {"code": code_runtime_error},
        description="Executing code with ZeroDivisionError (should fail)",
    )
    display_sandbox_result("Runtime Error Handling (ZeroDivisionError)", result, code_runtime_error)

    # --- Name Error ---
    code_name_error = "result = undefined_variable + 5"
    result = await safe_tool_call(
        execute_python,
        {"code": code_name_error},
        description="Executing code with NameError (should fail)",
    )
    display_sandbox_result("Runtime Error Handling (NameError)", result, code_name_error)


async def demonstrate_timeout(args):
    """Demonstrate timeout handling."""
    console.print(Rule("[bold cyan]3. Timeout Handling Demo[/bold cyan]", style="cyan"))
    logger.info("Starting timeout handling demo")

    code_timeout = """
import time
print("Starting computation that will time out...")
time.sleep(5) # Sleep for 5 seconds
print("This line should not be reached due to timeout")
result = "Completed successfully despite timeout request?" # Should not happen
"""
    # Use a short timeout (3 seconds) to trigger the error
    result = await safe_tool_call(
        execute_python,
        {"code": code_timeout, "timeout_ms": 3000},
        description="Executing code that exceeds timeout (3s)",
    )
    display_sandbox_result("Timeout Handling (3s Timeout)", result, code_timeout)


async def demonstrate_packages(args):
    """Demonstrate loading Python packages."""
    console.print(
        Rule("[bold cyan]4. Package Loading Demo (NumPy & Pandas)[/bold cyan]", style="cyan")
    )
    logger.info("Starting package loading demo")

    # NumPy example
    numpy_code = """
import numpy as np
a = np.array([[1, 2], [3, 4]])
result = {
    'shape': a.shape,
    'mean': np.mean(a).item(), # Use .item() for scalar
    'determinant': np.linalg.det(a).item() if a.shape == (2, 2) else 'N/A'
}
print(f"Array:\\n{a}")
"""
    result = await safe_tool_call(
        execute_python,
        {"code": numpy_code, "packages": ["numpy"], "timeout_ms": 15000},
        description="Using numpy package",
    )
    display_sandbox_result("NumPy Package Demo", result, numpy_code)

    # Pandas example (depends on numpy)
    pandas_code = """
import pandas as pd
data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
df = pd.DataFrame(data)
print("DataFrame Head:")
print(df.head())
result = df.describe().to_dict() # Return summary stats as dict
"""
    result = await safe_tool_call(
        execute_python,
        {"code": pandas_code, "packages": ["pandas"], "timeout_ms": 20000},
        description="Using pandas package",
    )
    display_sandbox_result("Pandas Package Demo", result, pandas_code)


async def demonstrate_wheels(args):
    """Demonstrate loading wheels (requires network)."""
    console.print(Rule("[bold cyan]5. Wheel Loading Demo (httpx)[/bold cyan]", style="cyan"))
    logger.info("Starting wheel loading demo")

    if not args.allow_network:
        console.print(
            Panel(
                "Skipping wheel loading demo.\n"
                "Network access is required to install wheels from URLs or PyPI.\n"
                "Rerun with the [yellow]--allow-network[/yellow] flag to include this test.",
                title="Network Access Disabled",
                border_style="yellow",
                expand=False,
            )
        )
        return

    code_wheel = """
try:
    import httpx
    print(f"httpx version: {httpx.__version__}")
    # Make a simple request to test network access
    response = httpx.get('https://httpbin.org/get?demo=wheel', timeout=10)
    response.raise_for_status()
    data = response.json()
    result = f"Successfully fetched URL via httpx. Origin IP: {data.get('origin', 'Unknown')}"
except Exception as e:
    # Raising an exception shows up nicely in stderr display
    raise RuntimeError(f"Error using httpx: {e}") from e
"""
    # Specify package 'httpx'. Micropip should handle fetching it if not preloaded.
    result = await safe_tool_call(
        execute_python,
        {"code": code_wheel, "packages": ["httpx"], "allow_network": True, "timeout_ms": 25000},
        description="Loading 'httpx' package/wheel (requires network)",
    )
    display_sandbox_result("Wheel Loading Demo (httpx)", result, code_wheel)


async def demonstrate_repl(args):
    """Demonstrate persistent REPL sessions and reset."""
    console.print(Rule("[bold cyan]6. Persistent REPL Sessions[/bold cyan]", style="cyan"))
    logger.info("Demonstrating REPL functionality...", emoji_key="repl")

    repl_handle = REPL_SESSION_HANDLE  # Use a consistent handle for the demo

    # --- Call 1: Define Variable & Function ---
    code1 = """
x = 100
def double(val):
    return val * 2
print(f"Defined x = {x} and function double()")
result = "Setup complete"
"""
    result1 = await safe_tool_call(
        repl_python,
        {"code": code1, "handle": repl_handle},
        description=f"REPL Call 1 (Handle: {repl_handle[-8:]}): Define x and double()",
    )
    display_sandbox_result(f"REPL Step 1 (Handle: ...{repl_handle[-8:]})", result1, code1)
    if (
        not result1
        or not result1.get("success")
        or result1.get("result", {}).get("handle") != repl_handle
    ):
        console.print(
            "[bold red]Error:[/bold red] Failed to get handle from first REPL call. Aborting REPL demo."
        )
        return

    # --- Call 2: Use Variable & Function ---
    code2 = "result = double(x) # Uses x and double() from previous call"
    result2 = await safe_tool_call(
        repl_python,
        {"code": code2, "handle": repl_handle},
        description=f"REPL Call 2 (Handle: {repl_handle[-8:]}): Call double(x)",
    )
    display_sandbox_result(f"REPL Step 2 (Handle: ...{repl_handle[-8:]})", result2, code2)

    # --- Call 3: Import and Use ---
    code3 = """
import math
result = math.sqrt(x) # Use x again
print(f"Square root of x ({x}) is {result}")
"""
    result3 = await safe_tool_call(
        repl_python,
        {
            "code": code3,
            "handle": repl_handle,
            "packages": [],
        },
        description=f"REPL Call 3 (Handle: {repl_handle[-8:]}): Import math and use x",
    )
    display_sandbox_result(f"REPL Step 3 (Handle: ...{repl_handle[-8:]})", result3, code3)

    # --- Call 4: Reset Session ---
    # Code is empty, only resetting
    result4 = await safe_tool_call(
        repl_python,
        {"code": "", "handle": repl_handle, "reset": True},
        description=f"REPL Call 4 (Handle: {repl_handle[-8:]}): Resetting the session",
    )
    display_sandbox_result(
        f"REPL Step 4 - Reset (Handle: ...{repl_handle[-8:]})",
        result4,
        "# Resetting the REPL state",
    )

    # --- Call 5: Try Using Variable After Reset (should fail) ---
    code5 = """
try:
    result = double(x) # Should fail as x and double are gone
except NameError as e:
    print(f"Caught expected error: {e}")
    result = f"Caught expected NameError: {e}"
"""
    result5 = await safe_tool_call(
        repl_python,
        {"code": code5, "handle": repl_handle},
        description=f"REPL Call 5 (Handle: {repl_handle[-8:]}): Using state after reset (should fail/catch NameError)",
    )
    display_sandbox_result(
        f"REPL Step 5 - Post-Reset (Handle: ...{repl_handle[-8:]})", result5, code5
    )


async def demonstrate_security(args):
    """Demonstrate network and filesystem access controls."""
    console.print(Rule("[bold cyan]7. Security Controls[/bold cyan]", style="cyan"))
    logger.info("Demonstrating security controls...", emoji_key="security")

    # --- Network Access Control ---
    console.print(Rule("Network Access", style="dim"))
    code_network = """
import httpx
try:
    # Use httpx which was potentially loaded in wheel demo
    print("Attempting network request to httpbin...")
    response = httpx.get('https://httpbin.org/get?demo=network_security', timeout=5)
    response.raise_for_status()
    result = f"Network access successful. Status: {response.status_code}"
except Exception as e:
    # Use print instead of raise to see the output in the demo result
    print(f"Network request failed: {type(e).__name__}: {e}")
    result = f"Network request failed as expected (or httpx not loaded)."
"""
    # Attempt without network access (should fail within sandbox)
    console.print(
        Panel(
            "Attempting network access with [red]allow_network=False[/red] (expected failure or httpx import error)",
            title="Network Test 1",
        )
    )
    result_net_denied = await safe_tool_call(
        execute_python,
        {"code": code_network, "packages": ["httpx"], "allow_network": False},
        description="Network access with allow_network=False",
    )
    display_sandbox_result("Network Access Denied", result_net_denied, code_network)

    # Attempt with network access (should succeed IF network flag is passed)
    console.print(
        Panel(
            "Attempting network access with [green]allow_network=True[/green]",
            title="Network Test 2",
        )
    )
    if args.allow_network:
        result_net_allowed = await safe_tool_call(
            execute_python,
            {"code": code_network, "packages": ["httpx"], "allow_network": True},
            description="Network access with allow_network=True",
        )
        display_sandbox_result("Network Access Allowed", result_net_allowed, code_network)
    else:
        console.print(
            "[yellow]Skipped:[/yellow] Rerun demo with --allow-network flag to test allowed network access."
        )

    # --- Filesystem Access Control ---
    console.print(Rule("Filesystem Access (via mcpfs bridge)", style="dim"))
    code_fs_list = """
try:
    import mcpfs
    print("Attempting to list directory '.' via mcpfs...")
    # Note: Path inside sandbox needs to map to an allowed host path
    target_path = '.' # Represents the sandbox's current dir
    listing = mcpfs.listdir(target_path)
    result = f"Successfully listed '{target_path}': {len(listing)} entries found via mcpfs."
    print(f"Listing result: {listing}")
except ModuleNotFoundError:
    print("mcpfs module not available (allow_fs=False?)")
    result = "mcpfs module not found (expected failure)"
except Exception as e:
    print(f"Filesystem access failed: {type(e).__name__}: {e}")
    result = f"Filesystem access failed: {e}"
"""
    # Attempt without FS access (should fail - ModuleNotFoundError)
    console.print(
        Panel(
            "Attempting filesystem access with [red]allow_fs=False[/red] (expected ModuleNotFoundError)",
            title="Filesystem Test 1",
        )
    )
    result_fs_denied = await safe_tool_call(
        execute_python,
        {"code": code_fs_list, "allow_fs": False},
        description="Filesystem access with allow_fs=False",
    )
    display_sandbox_result("Filesystem Access Denied (mcpfs)", result_fs_denied, code_fs_list)

    # Attempt with FS access (should succeed IF FS flag is passed AND FS tool configured on host)
    console.print(
        Panel(
            "Attempting filesystem access with [green]allow_fs=True[/green]",
            title="Filesystem Test 2",
        )
    )
    if args.allow_fs:
        console.print(
            "[yellow]Note:[/yellow] Success requires the host Filesystem tool to be configured with allowed directories."
        )
        result_fs_allowed = await safe_tool_call(
            execute_python,
            {"code": code_fs_list, "allow_fs": True},
            description="Filesystem access with allow_fs=True",
        )
        display_sandbox_result("Filesystem Access Allowed (mcpfs)", result_fs_allowed, code_fs_list)
    else:
        console.print(
            "[yellow]Skipped:[/yellow] Rerun demo with --allow-fs flag to test allowed filesystem access bridge."
        )
        console.print(
            "[dim](Also ensure the host Filesystem tool is configured with allowed directories.)[/dim]"
        )


async def demonstrate_visualization(args):
    """Demonstrate data visualization capabilities."""
    console.print(
        Rule("[bold cyan]8. Data Visualization Demo (Matplotlib)[/bold cyan]", style="cyan")
    )
    logger.info("Starting data visualization demo")

    matplotlib_code = """
# Ensure backend is non-interactive
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

try:
    print("Generating plot...")
    # Generate data
    x = np.linspace(-np.pi, np.pi, 200)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5)) # Use fig, ax pattern
    ax.plot(x, y_sin, label='sin(x)')
    ax.plot(x, y_cos, label='cos(x)', linestyle='--')
    ax.set_title('Sine and Cosine Waves')
    ax.set_xlabel('Radians')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()
    plt.tight_layout() # Adjust layout

    # Save plot to base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=90) # Save the figure object
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free memory

    print(f"Generated plot as base64 string (Length: {len(img_str)} chars)")
    result = f"data:image/png;base64,{img_str}"
except Exception as e:
    print(f"Error during plotting: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc() # Print traceback to stderr for diagnosis
    result = f"Plot generation failed: {e}"

"""
    # Requires matplotlib and numpy packages
    result = await safe_tool_call(
        execute_python,
        {"code": matplotlib_code, "packages": ["numpy", "matplotlib"], "timeout_ms": 25000},
        description="Generating plot with Matplotlib",
    )

    # Display result, summarizing the base64 string
    result_display = result.copy()
    if result_display.get("success") and "result" in result_display.get("result", {}):
        res_value = result_display["result"]["result"]
        if isinstance(res_value, str) and res_value.startswith("data:image/png;base64,"):
            result_display["result"]["result"] = f"[Base64 image data - {len(res_value)} chars]"

    display_sandbox_result("Matplotlib Visualization", result_display, matplotlib_code)
    console.print(
        Panel(
            "[yellow]Note:[/] The 'result' contains base64 image data. In a web UI, this could be displayed using an `<img>` tag.",
            border_style="yellow",
        )
    )


async def main():
    """Run the Python Sandbox tools demonstration."""
    args = parse_arguments()
    exit_code = 0

    console.print(Rule("[bold magenta]Python Sandbox Tools Demo[/bold magenta]", style="white"))

    # Explicitly check for Playwright availability
    if not PLAYWRIGHT_AVAILABLE_DEMO:
        console.print(
            "[bold red]Error:[/bold red] Playwright is required for the Python Sandbox tool but is not installed or importable."
        )
        console.print(
            "Please install it via: [cyan]pip install playwright && playwright install chromium[/]"
        )
        return 1  # Exit if core dependency is missing

    logger.info("Starting Python Sandbox demonstration", emoji_key="start")

    try:
        # --- Display Demo Options ---
        if args.demo == "all":
            console.print(
                Panel(
                    "Running all demo sections.\n"
                    "Use command-line arguments to run specific sections (e.g., `python examples/python_sandbox_demo.py repl`).\n"
                    "Use `--allow-network` or `--allow-fs` to enable those features for relevant tests.",
                    title="Demo Options",
                    border_style="cyan",
                    expand=False,
                )
            )

        # --- Run Selected Demonstrations ---
        run_all = args.demo == "all"

        if run_all or args.demo == "basic":
            await demonstrate_basic_execution(args)
            console.print()

        if run_all or args.demo == "errors":
            await demonstrate_error_handling(args)
            console.print()

        if run_all or args.demo == "timeout":
            await demonstrate_timeout(args)
            console.print()

        if run_all or args.demo == "packages":
            await demonstrate_packages(args)
            console.print()

        if run_all or args.demo == "wheels":
            await demonstrate_wheels(args)
            console.print()

        if run_all or args.demo == "repl":
            await demonstrate_repl(args)
            console.print()

        if run_all or args.demo == "security":
            await demonstrate_security(args)
            console.print()

        if run_all or args.demo == "visualization":
            await demonstrate_visualization(args)
            console.print()

        logger.success(f"Python Sandbox Demo(s) completed: {args.demo}", emoji_key="complete")
        console.print(Rule("[bold green]Demo Complete[/bold green]", style="green"))

    except (ToolInputError, ToolError, ProviderError) as e:
        logger.error(f"Tool Error during demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"\n[bold red]TOOL ERROR:[/bold red] {escape(str(e))}")
        if hasattr(e, "details") and e.details:
            console.print("[bold]Details:[/bold]")
            console.print(escape(str(e.details)))
        exit_code = 1
    except Exception as e:
        logger.critical(f"Demo crashed unexpectedly: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"\n[bold red]CRITICAL ERROR:[/bold red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        exit_code = 1
    finally:
        # --- Cleanup ---
        console.print(Rule("Cleanup", style="dim"))
        try:
            # Explicitly call the sandbox cleanup function
            await _close_all_sandboxes()
            logger.info("Sandbox cleanup completed.", emoji_key="cleanup")
            console.print("Sandbox cleanup finished.")
        except Exception as e:
            logger.error(f"Error during sandbox cleanup: {e}", emoji_key="error")
            console.print(f"[bold red]Error during sandbox cleanup:[/bold red] {escape(str(e))}")

    return exit_code


if __name__ == "__main__":
    # Run the demo
    final_exit_code = asyncio.run(main())
    sys.exit(final_exit_code)
