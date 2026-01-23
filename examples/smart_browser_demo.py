#!/usr/bin/env python
"""
DETAILED Demonstration script for the Smart Browser Tools in Ultimate MCP Server,
showcasing browsing, interaction, search, download, macro, and autopilot features.
"""

import asyncio
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to path for imports when running as script
# Adjust this relative path if your script structure is different
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    print(f"INFO: Added {_PROJECT_ROOT} to sys.path")

# Rich imports for enhanced terminal UI
from rich import box, get_console  # noqa: E402
from rich.console import Group  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402
from rich.traceback import install as install_rich_traceback  # noqa: E402

# Initialize Rich console
console = get_console()


# Define a fallback logger in case the import fails
def create_fallback_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Import Gateway and MCP components
from ultimate_mcp_server.core.server import Gateway  # noqa: E402
from ultimate_mcp_server.exceptions import ToolError, ToolInputError  # noqa: E402

# Import smart browser tools directly
from ultimate_mcp_server.tools.smart_browser import (  # noqa: E402
    autopilot,
    browse,
    click,
    collect_documentation,
    download,
    download_site_pdfs,
    parallel,
    run_macro,
    search,
    shutdown,
    type_text,
)
from ultimate_mcp_server.utils import get_logger  # noqa: E402
from ultimate_mcp_server.utils.display import CostTracker  # noqa: E402

# Initialize logger
logger = get_logger("demo.smart_browser")

# Install rich tracebacks
install_rich_traceback(show_locals=True, width=console.width, extra_lines=2)

# --- Configuration ---
# Base directory for Smart Browser outputs
SMART_BROWSER_INTERNAL_BASE = "storage/smart_browser_internal"  # Relative path used by the tool
SMART_BROWSER_DOWNLOADS_BASE = "storage/smart_browser_downloads"  # Default download relative path
DEMO_OUTPUTS_DIR = Path(
    "./sb_demo_outputs"
)  # Local dir for demo-specific outputs like the test HTML

# Example URLs for demo
URL_EXAMPLE = "http://example.com"
URL_BOOKSTORE = "http://books.toscrape.com/"
URL_QUOTES = "http://quotes.toscrape.com/"
URL_PDF_SAMPLE = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
URL_GITHUB = "https://github.com/features/copilot"

# --- Demo Helper Functions (Unchanged from previous version) ---


def timestamp_str(short: bool = False) -> str:
    """Return a formatted timestamp string."""
    now = time.time()  # Use time.time for consistency
    dt_now = datetime.fromtimestamp(now)
    if short:
        return f"[dim]{dt_now.strftime('%H:%M:%S')}[/]"
    return f"[dim]{dt_now.strftime('%Y-%m-%d %H:%M:%S')}[/]"


def truncate_text_by_lines(text: str, max_lines: int = 50) -> str:
    """Truncates text to show first/last lines if too long."""
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    half_lines = max_lines // 2
    # Ensure half_lines is at least 1 if max_lines >= 2
    half_lines = max(1, half_lines)
    # Handle edge case where max_lines is 1
    if max_lines == 1:
        return lines[0] + "\n[...TRUNCATED...]"

    # Return first half, separator, and last half
    return "\n".join(lines[:half_lines] + ["[...TRUNCATED...]"] + lines[-half_lines:])


def format_value(key: str, value: Any, detail_level: int = 1) -> Any:
    """Format specific values for display, returning strings with markup."""
    if value is None:
        return "[dim]None[/]"  # Keep markup
    if isinstance(value, bool):
        return "[green]Yes[/]" if value else "[red]No[/]"  # Keep markup
    if isinstance(value, float):
        return f"{value:.3f}"  # Return simple string
    if key.lower().endswith("time_seconds") or key.lower() == "duration_ms":
        try:
            val_s = float(value) / 1000.0 if key.lower() == "duration_ms" else float(value)
            return f"[green]{val_s:.3f}s[/]"  # Keep markup
        except (ValueError, TypeError):
            return escape(str(value))  # Fallback for non-numeric time values
    if key.lower() == "size_bytes" and isinstance(value, int):
        if value < 0:
            return "[dim]N/A[/]"
        if value > 1024 * 1024:
            return f"{value / (1024 * 1024):.2f} MB"
        if value > 1024:
            return f"{value / 1024:.2f} KB"
        return f"{value} Bytes"  # Return simple string

    if isinstance(value, list):
        if not value:
            return "[dim]Empty List[/]"  # Keep markup
        list_len = len(value)
        preview_count = 3 if detail_level < 2 else 5
        suffix = (
            f" [dim]... ({list_len} items total)[/]" if list_len > preview_count else ""
        )  # Keep markup
        if detail_level >= 1:
            previews = [
                str(
                    format_value(f"{key}[{i}]", item, detail_level=0)
                )  # Recursive call returns string
                for i, item in enumerate(value[:preview_count])
            ]
            return f"[{', '.join(previews)}]{suffix}"  # Returns string with markup
        else:
            return f"[List with {list_len} items]"  # Keep markup

    if isinstance(value, dict):
        if not value:
            return "[dim]Empty Dict[/]"  # Keep markup
        dict_len = len(value)
        preview_count = 4 if detail_level < 2 else 8
        preview_keys = list(value.keys())[:preview_count]
        suffix = (
            f" [dim]... ({dict_len} keys total)[/]" if dict_len > preview_count else ""
        )  # Keep markup
        if detail_level >= 1:
            items_preview = [
                # Key repr for clarity, value formatted recursively
                f"{repr(k)}: {str(format_value(k, value[k], detail_level=0))}"
                for k in preview_keys
            ]
            return f"{{{'; '.join(items_preview)}}}{suffix}"  # Returns string with markup
        else:
            return f"[Dict with {dict_len} keys]"  # Keep markup

    if isinstance(value, str):
        value_truncated = truncate_text_by_lines(value, 30)  # Truncate by lines first
        preview_len = 300 if detail_level < 2 else 600
        suffix = ""
        # Check length after line truncation
        if len(value_truncated) > preview_len:
            value_display = value_truncated[:preview_len]
            suffix = "[dim]... (truncated)[/]"  # Keep markup
        else:
            value_display = value_truncated

        # Escape only if it doesn't look like it contains Rich markup
        if "[" in value_display and "]" in value_display and "/" in value_display:
            # Heuristic: Assume it might contain markup, don't escape
            return value_display + suffix
        else:
            # Safe to escape plain strings
            return escape(value_display) + suffix

    # Fallback: escape the string representation of other types
    return escape(str(value))


def display_page_state(state: Dict[str, Any], title: str = "Page State"):
    """Display the 'page_state' dictionary nicely."""
    panel_content = []
    url = state.get("url", "N/A")
    panel_content.append(
        Text.from_markup(f"[bold cyan]URL:[/bold cyan] [link={url}]{escape(url)}[/link]")
    )
    panel_content.append(
        Text.from_markup(f"[bold cyan]Title:[/bold cyan] {escape(state.get('title', 'N/A'))}")
    )

    main_text = state.get("main_text", "")
    if main_text:
        truncated_text = truncate_text_by_lines(main_text, 15)
        panel_content.append(Text.from_markup("\n[bold cyan]Main Text Summary:[/bold cyan]"))
        panel_content.append(Panel(escape(truncated_text), border_style="dim", padding=(0, 1)))

    elements = state.get("elements", [])
    if elements:
        elements_table = Table(
            title=Text.from_markup(f"Interactive Elements ({len(elements)} found)"),
            box=box.MINIMAL,
            show_header=True,
            padding=(0, 1),
            border_style="blue",
        )
        elements_table.add_column("ID", style="magenta", no_wrap=True)
        elements_table.add_column("Tag", style="cyan")
        elements_table.add_column("Role", style="yellow")
        elements_table.add_column("Text Preview", style="white", max_width=60)
        elements_table.add_column("BBox", style="dim")

        preview_count = 15
        for elem in elements[:preview_count]:
            elem_text_raw = elem.get("text", "")
            elem_text_preview = escape(
                elem_text_raw[:60] + ("..." if len(elem_text_raw) > 60 else "")
            )
            bbox = elem.get("bbox", [])
            if len(bbox) == 4:
                bbox_str = f"({bbox[0]}x{bbox[1]}, {bbox[2]}w{bbox[3]}h)"
            else:
                bbox_str = "[Invalid Bbox]"

            elements_table.add_row(
                str(elem.get("id", "?")),
                str(elem.get("tag", "?")),
                str(elem.get("role", "")),
                elem_text_preview,  # Pass escaped preview string
                bbox_str,
            )
        if len(elements) > preview_count:
            elements_table.add_row(
                "...",
                Text.from_markup(f"[dim]{len(elements) - preview_count} more...[/]"),
                "",
                "",
                "",
            )

        panel_content.append(Text.from_markup("\n[bold cyan]Elements:[/bold cyan]"))
        panel_content.append(elements_table)

    console.print(
        Panel(
            Group(*panel_content),
            title=Text.from_markup(title),
            border_style="blue",
            padding=(1, 2),
            expand=False,
        )
    )


def display_result(
    title: str, result: Dict[str, Any], display_options: Optional[Dict] = None
) -> None:
    """Display operation result with enhanced formatting using Rich."""
    display_options = display_options or {}
    console.print(
        Rule(
            Text.from_markup(f"[bold cyan]{escape(title)}[/] {timestamp_str(short=True)}"),
            style="cyan",
        )
    )

    success = result.get("success", False)
    detail_level = display_options.get("detail_level", 1)
    # Use _display_options from result if available, otherwise use passed options
    effective_display_options = result.get("_display_options", display_options)

    hide_keys_set = set(
        effective_display_options.get(
            "hide_keys",
            [
                "success",
                "page_state",
                "results",
                "steps",
                "download",
                "final_page_state",
                "documentation",
                "raw_response",
                "raw_llm_response",
                "_display_options",  # Also hide internal options
            ],
        )
    )

    # --- Status Panel ---
    status_panel_content = Text.from_markup(
        f"Status: {'[bold green]Success[/]' if success else '[bold red]Failed[/]'}\n"
    )
    if not success:
        error_code = result.get("error_code", "N/A")
        error_msg = result.get("error", "Unknown error")
        status_panel_content.append(
            Text.from_markup(f"Error Code: [yellow]{escape(str(error_code))}[/]\n")
        )
        status_panel_content.append(
            Text.from_markup(f"Message: [red]{escape(str(error_msg))}[/]\n")
        )
        console.print(
            Panel(
                status_panel_content,
                title="Operation Status",
                border_style="red",
                padding=(1, 2),
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                status_panel_content,
                title="Operation Status",
                border_style="green",
                padding=(0, 1),
                expand=False,
            )
        )

    # --- Top Level Details ---
    details_table = Table(
        title="Result Summary", box=box.MINIMAL, show_header=False, padding=(0, 1)
    )
    details_table.add_column("Key", style="cyan", justify="right", no_wrap=True)
    details_table.add_column("Value", style="white")
    has_details = False
    for key, value in result.items():
        if key in hide_keys_set or key.startswith("_"):
            continue
        formatted_value = format_value(key, value, detail_level=detail_level)
        details_table.add_row(
            escape(str(key)), formatted_value
        )  # formatted_value is already string/markup
        has_details = True
    if has_details:
        console.print(details_table)

    # --- Special Section Displays ---

    # Page State
    if "page_state" in result and isinstance(result["page_state"], dict):
        display_page_state(result["page_state"], title="Page State After Action")
    elif "final_page_state" in result and isinstance(result["final_page_state"], dict):
        display_page_state(result["final_page_state"], title="Final Page State")

    # Search Results
    if "results" in result and isinstance(result["results"], list) and "query" in result:
        search_results = result["results"]
        search_table = Table(
            title=Text.from_markup(
                f"Search Results for '{escape(result['query'])}' ({len(search_results)} found)"
            ),
            box=box.ROUNDED,
            show_header=True,
            padding=(0, 1),
        )
        search_table.add_column("#", style="dim")
        search_table.add_column("Title", style="cyan")
        search_table.add_column("URL", style="blue", no_wrap=False)
        search_table.add_column("Snippet", style="white", no_wrap=False)
        for i, item in enumerate(search_results, 1):
            title = truncate_text_by_lines(item.get("title", ""), 3)
            snippet = truncate_text_by_lines(item.get("snippet", ""), 5)
            url = item.get("url", "")
            search_table.add_row(
                str(i), escape(title), f"[link={url}]{escape(url)}[/link]", escape(snippet)
            )
        console.print(search_table)

    # Download Result
    if "download" in result and isinstance(result["download"], dict):
        dl_info = result["download"]
        dl_table = Table(
            title="Download Details", box=box.MINIMAL, show_header=False, padding=(0, 1)
        )
        dl_table.add_column("Metric", style="cyan", justify="right")
        dl_table.add_column("Value", style="white")
        dl_table.add_row("File Path", escape(dl_info.get("file_path", "N/A")))
        dl_table.add_row("File Name", escape(dl_info.get("file_name", "N/A")))
        dl_table.add_row("SHA256", escape(dl_info.get("sha256", "N/A")))
        dl_table.add_row("Size", format_value("size_bytes", dl_info.get("size_bytes", -1)))
        dl_table.add_row("Source URL", escape(dl_info.get("url", "N/A")))
        dl_table.add_row(
            "Tables Extracted",
            format_value("tables_extracted", dl_info.get("tables_extracted", False)),
        )
        if dl_info.get("tables"):
            # format_value handles potential markup in table preview string
            dl_table.add_row("Table Preview", format_value("tables", dl_info.get("tables")))
        console.print(
            Panel(dl_table, title="Download Result", border_style="green", padding=(1, 2))
        )

    # Macro/Autopilot Steps
    if "steps" in result and isinstance(result["steps"], list):
        steps = result["steps"]
        steps_table = Table(
            title=Text.from_markup(f"Macro/Autopilot Steps ({len(steps)} executed)"),
            box=box.ROUNDED,
            show_header=True,
            padding=(0, 1),
        )
        steps_table.add_column("#", style="dim")
        steps_table.add_column("Action/Tool", style="cyan")
        steps_table.add_column("Arguments/Hint", style="white", no_wrap=False)
        steps_table.add_column("Status", style="yellow")
        steps_table.add_column("Result/Error", style="white", no_wrap=False)

        for i, step in enumerate(steps, 1):
            action = step.get("action", step.get("tool", "?"))
            args = step.get("args")  # Check if 'args' exists
            if args is None:  # If no 'args', use the step itself excluding status keys
                args = {
                    k: v
                    for k, v in step.items()
                    if k
                    not in ["action", "tool", "success", "result", "error", "step", "duration_ms"]
                }

            args_preview = format_value("args", args, detail_level=0)  # format_value handles markup
            success_step = step.get("success", False)
            status = "[green]OK[/]" if success_step else "[red]FAIL[/]"  # Markup string
            outcome = step.get("result", step.get("error", ""))
            outcome_preview = format_value(
                "outcome", outcome, detail_level=0
            )  # format_value handles markup
            steps_table.add_row(str(i), escape(action), args_preview, status, outcome_preview)
        console.print(steps_table)

    # Documentation (assuming it's stored under 'file_path' key now)
    if (
        "file_path" in result and result.get("pages_collected") is not None
    ):  # Check for doc collection result structure
        doc_file_path = result.get("file_path")
        pages_collected = result.get("pages_collected")
        if doc_file_path and pages_collected > 0:
            content_to_display: Any = f"[dim]Documentation saved to: {escape(doc_file_path)}[/]"
            try:
                with open(doc_file_path, "r", encoding="utf-8") as f:
                    content = f.read(1500)  # Read preview
                content_to_display += f"\n\n[bold]File Preview ({len(content)} chars):[/]\n"
                content_to_display += escape(content) + "\n[dim]...[/]"
            except Exception as e:
                content_to_display += f"\n[yellow]Could not read file preview: {escape(str(e))}[/]"

            console.print(
                Panel(
                    Text.from_markup(content_to_display),
                    title=f"Collected Documentation ({pages_collected} pages)",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

    console.print()  # Add spacing


async def safe_tool_call(
    operation_name: str, tool_func: callable, *args, tracker: Optional[CostTracker] = None, **kwargs
) -> Tuple[bool, Dict[str, Any]]:
    """Safely call a tool function, handling exceptions and logging."""
    console.print(
        f"\n[cyan]Calling Tool:[/][bold] {escape(operation_name)}[/] {timestamp_str(short=True)}"
    )
    display_options = kwargs.pop("display_options", {})

    log_args_repr = {}
    MAX_ARG_LEN = 100
    for k, v in kwargs.items():
        try:
            if isinstance(v, (str, bytes)) and len(v) > MAX_ARG_LEN:
                log_args_repr[k] = f"{type(v).__name__}(len={len(v)})"
            elif isinstance(v, (list, dict)) and len(v) > 10:
                log_args_repr[k] = f"{type(v).__name__}(len={len(v)})"
            else:
                log_args_repr[k] = repr(v)
        except Exception:  # Handle potential errors during repr()
            log_args_repr[k] = f"<{type(v).__name__} repr_error>"

    logger.debug(f"Executing {operation_name} with args: {args}, kwargs: {log_args_repr}")

    try:
        # Call the tool function directly
        result = await tool_func(*args, **kwargs)
        if not isinstance(result, dict):
            logger.error(f"Tool '{operation_name}' returned non-dict type: {type(result)}")
            return False, {
                "success": False,
                "error": f"Tool returned unexpected type: {type(result).__name__}",
                "error_code": "INTERNAL_ERROR",
                "_display_options": display_options,
            }

        # Store display options within the result for the display function
        result["_display_options"] = display_options
        logger.debug(f"Tool '{operation_name}' completed.")
        # Add success=True if missing and no error key present (should usually be set by tool)
        if "success" not in result and "error" not in result:
            result["success"] = True
        return result.get("success", False), result  # Return success flag and the result dict
    except ToolInputError as e:
        logger.warning(f"Input error for {operation_name}: {e}")
        return False, {
            "success": False,
            "error": str(e),
            "error_code": getattr(e, "error_code", "INPUT_ERROR"),
            "_display_options": display_options,
        }
    except ToolError as e:
        logger.error(f"Tool error during {operation_name}: {e}", exc_info=True)
        return False, {
            "success": False,
            "error": str(e),
            "error_code": getattr(e, "error_code", "TOOL_ERROR"),
            "_display_options": display_options,
        }
    except Exception as e:
        logger.error(f"Unexpected error during {operation_name}: {e}", exc_info=True)
        tb_str = traceback.format_exc(limit=1)
        return False, {
            "success": False,
            "error": f"{type(e).__name__}: {e}\n{tb_str}",
            "error_type": type(e).__name__,
            "error_code": "UNEXPECTED_ERROR",
            "_display_options": display_options,
        }


# --- Demo Sections ---


async def demo_section_1_browse(gateway, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 1: Basic Browsing[/]", style="green"))
    logger.info("Starting Demo Section 1: Basic Browsing")

    # 1a: Browse Example.com
    success, result = await safe_tool_call(
        "Browse Example.com", browse, url=URL_EXAMPLE, tracker=tracker
    )
    display_result("Browse Example.com", result)

    # 1b: Browse Bookstore (wait for specific element)
    success, result = await safe_tool_call(
        "Browse Bookstore (wait for footer)",
        browse,
        url=URL_BOOKSTORE,
        wait_for_selector="footer.footer",
        tracker=tracker,
    )
    display_result("Browse Bookstore (Wait)", result)


async def demo_section_2_interaction(gateway, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 2: Page Interaction[/]", style="green"))
    logger.info("Starting Demo Section 2: Page Interaction")

    # 2a: Search on Bookstore
    console.print(f"--- Scenario: Search for 'Science' on {URL_BOOKSTORE} ---")
    success, initial_state_res = await safe_tool_call(
        "Load Bookstore Search Page",
        browse,
        url=URL_BOOKSTORE,
        tracker=tracker,
    )
    if not success:
        console.print("[red]Cannot proceed with interaction demo, failed to load page.[/]")
        return
    display_result("Bookstore Initial State", initial_state_res)

    # Fill the search form using task hints
    fields_to_type = [
        {"task_hint": "The search input field", "text": "Science", "enter": False},
    ]
    success, fill_res = await safe_tool_call(
        "Type into Bookstore Search Form",
        type_text,
        url=URL_BOOKSTORE,
        fields=fields_to_type,
        submit_hint="The search button",
        wait_after_submit_ms=1500,
        tracker=tracker,
    )
    display_result("Type into Bookstore Search Form", fill_res)

    # 2b: Click the first search result (if successful)
    if success:
        console.print("--- Scenario: Click the first search result ---")
        current_url = fill_res.get("page_state", {}).get("url", URL_BOOKSTORE)

        success, click_res = await safe_tool_call(
            "Click First Book Result",
            click,
            url=current_url,
            task_hint="The link for the first book shown in the results list",
            wait_ms=1000,
            tracker=tracker,
        )
        display_result("Click First Book Result", click_res)


async def demo_section_3_search(gateway, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 3: Web Search[/]", style="green"))
    logger.info("Starting Demo Section 3: Web Search")

    search_query = "latest advancements in large language models"

    # 3a: Search Bing
    success, result = await safe_tool_call(
        "Search Bing",
        search,
        query=search_query,
        engine="bing",
        max_results=5,
        tracker=tracker,
    )
    display_result(f"Search Bing: '{search_query}'", result)

    # 3b: Search DuckDuckGo
    success, result = await safe_tool_call(
        "Search DuckDuckGo",
        search,
        query=search_query,
        engine="duckduckgo",
        max_results=5,
        tracker=tracker,
    )
    display_result(f"Search DuckDuckGo: '{search_query}'", result)


async def demo_section_4_download(gateway, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 4: File Download[/]", style="green"))
    logger.info("Starting Demo Section 4: File Download")

    # Ensure local demo output dir exists
    DEMO_OUTPUTS_DIR_ABS = DEMO_OUTPUTS_DIR.resolve(
        strict=False
    )  # Resolve to absolute, allow non-existent
    DEMO_OUTPUTS_DIR_ABS.mkdir(parents=True, exist_ok=True)  # Ensure it exists after resolving

    # Create the parent directory for PDF downloads if it doesn't exist
    pdf_parent_dir = "storage/smart_browser_site_pdfs"
    console.print(f"[cyan]Creating parent directory for PDFs: {pdf_parent_dir}[/cyan]")
    from ultimate_mcp_server.tools.filesystem import create_directory

    parent_dir_result = await create_directory(path=pdf_parent_dir)
    if not parent_dir_result.get("success", False):
        console.print(
            f"[yellow]Warning: Could not create parent directory: {parent_dir_result.get('error', 'Unknown error')}[/yellow]"
        )
    else:
        console.print(f"[green]Successfully created parent directory: {pdf_parent_dir}[/green]")

    # 4a: Download PDFs from a site
    console.print("--- Scenario: Find and Download PDFs from Example.com ---")
    success, result = await safe_tool_call(
        "Download PDFs from Example.com",
        download_site_pdfs,
        start_url=URL_EXAMPLE,
        max_depth=1,
        max_pdfs=5,
        dest_subfolder="example_com_pdfs",
        tracker=tracker,
    )
    display_result("Download PDFs from Example.com", result)
    if result.get("pdf_count", 0) == 0:
        console.print("[yellow]Note: No PDFs found on example.com as expected.[/]")

    # 4b: Click-based download
    download_page_content = f"""
    <!DOCTYPE html>
    <html><head><title>Download Test</title></head>
    <body><h1>Download Page</h1>
    <p>Click the link to download a dummy PDF.</p>
    <a href="{URL_PDF_SAMPLE}" id="downloadLink">Download Dummy PDF Now</a>
    <p>Another paragraph.</p>
    </body></html>
    """
    download_page_path = DEMO_OUTPUTS_DIR_ABS / "download_test.html"
    try:
        download_page_path.write_text(download_page_content, encoding="utf-8")
        local_url = download_page_path.as_uri()

        console.print("\n--- Scenario: Click a link to download a file ---")
        success, result = await safe_tool_call(
            "Click to Download PDF",
            download,
            url=local_url,
            task_hint="The 'Download Dummy PDF Now' link",
            dest_dir="storage/sb_demo_outputs/clicked_downloads",  # Adjusted path
            tracker=tracker,
        )
        display_result("Click to Download PDF", result)
    except Exception as e:
        console.print(f"[red]Error setting up or running click-download demo: {e}[/]")
    finally:
        if download_page_path.exists():
            try:
                download_page_path.unlink()
            except OSError:
                pass


async def demo_section_5_macro(gateway, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 5: Execute Macro[/]", style="green"))
    logger.info("Starting Demo Section 5: Execute Macro")

    macro_task = f"Go to {URL_BOOKSTORE}, search for 'History', find the book 'Sapiens: A Brief History of Humankind', and click its link."
    console.print("--- Scenario: Execute Macro ---")
    console.print(f"[italic]Task:[/italic] {escape(macro_task)}")

    success, result = await safe_tool_call(
        "Execute Bookstore Search Macro",
        run_macro,
        url=URL_BOOKSTORE,
        task=macro_task,
        max_rounds=5,
        tracker=tracker,
    )
    display_result("Execute Bookstore Search Macro", result)


async def demo_section_6_autopilot(gateway, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 6: Autopilot[/]", style="green"))
    logger.info("Starting Demo Section 6: Autopilot")

    autopilot_task = "Search the web for the official documentation URL of the 'httpx' Python library, then browse that URL and summarize the main page content."
    console.print("--- Scenario: Autopilot ---")
    console.print(f"[italic]Task:[/italic] {escape(autopilot_task)}")

    success, result = await safe_tool_call(
        "Run Autopilot: Find httpx Docs",
        autopilot,
        task=autopilot_task,
        max_steps=8,
        scratch_subdir="autopilot_demo",
        tracker=tracker,
    )
    display_result("Run Autopilot: Find httpx Docs", result)
    if result.get("run_log"):
        console.print(f"[dim]Autopilot run log saved to: {result['run_log']}[/]")


async def demo_section_7_parallel(gateway, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 7: Parallel Processing[/]", style="green"))
    logger.info("Starting Demo Section 7: Parallel Processing")

    urls_to_process = [
        URL_EXAMPLE,
        URL_BOOKSTORE,
        URL_QUOTES,
        "http://httpbin.org/delay/1",
        "https://webscraper.io/test-sites/e-commerce/static",
    ]
    console.print("--- Scenario: Get Page State for Multiple URLs in Parallel ---")
    console.print(f"[dim]URLs:[/dim] {urls_to_process}")

    success, result = await safe_tool_call(
        "Parallel Get Page State",
        parallel,
        urls=urls_to_process,
        action="get_state",  # Only 'get_state' supported currently
        # max_tabs=3 # Can override default here if needed
        tracker=tracker,
    )

    # Custom display for parallel results (same logic as before)
    console.print(Rule("[bold cyan]Parallel Processing Results[/]", style="cyan"))
    if success:
        console.print(f"Total URLs Processed: {result.get('processed_count', 0)}")
        console.print(f"Successful: {result.get('successful_count', 0)}")
        console.print("-" * 20)
        for i, item_result in enumerate(result.get("results", [])):
            url = item_result.get("url", f"URL {i + 1}")
            item_success = item_result.get("success", False)
            panel_title = f"Result for: {escape(url)}"
            border = "green" if item_success else "red"
            content = ""
            if item_success:
                state = item_result.get("page_state", {})
                content = f"Title: {escape(state.get('title', 'N/A'))}\nElements Found: {len(state.get('elements', []))}"
            else:
                content = f"[red]Error:[/red] {escape(item_result.get('error', 'Unknown'))}"
            console.print(
                Panel(content, title=panel_title, border_style=border, padding=(0, 1), expand=False)
            )
    else:
        console.print(
            Panel(
                f"[red]Parallel processing tool call failed:[/red]\n{escape(result.get('error', '?'))}",
                border_style="red",
            )
        )
    console.print()


async def demo_section_8_docs(gateway, tracker: CostTracker) -> None:
    console.print(Rule("[bold green]Demo 8: Documentation Collection[/]", style="green"))
    logger.info("Starting Demo Section 8: Documentation Collection")

    package_name = "fastapi"  # Use a different package
    console.print(f"--- Scenario: Collect Documentation for '{package_name}' ---")

    success, result = await safe_tool_call(
        f"Collect Docs: {package_name}",
        collect_documentation,
        package=package_name,
        max_pages=15,
        rate_limit_rps=2.0,
        tracker=tracker,
    )
    # Use the updated display logic that looks for file_path and pages_collected
    display_result(f"Collect Docs: {package_name}", result)


# --- Main Function ---
async def main() -> int:
    """Run the SmartBrowser tools demo."""
    console.print(Rule("[bold magenta]Smart Browser Tools Demo[/bold magenta]"))

    exit_code = 0
    gateway = None

    # Ensure local demo output directory exists
    DEMO_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Demo-specific outputs will be saved in: {DEMO_OUTPUTS_DIR}[/]")

    try:
        # --- Initialize Gateway for providers only ---
        console.print("[cyan]Initializing MCP Gateway...[/]")
        gateway = Gateway("smart-browser-demo")
        console.print("[cyan]Initializing Providers (for LLM tools)...[/]")
        await gateway._initialize_providers()

        # --- Initialize Smart Browser module ---
        console.print("[cyan]Initializing Smart Browser tool...[/]")
        # await initialize()

        # Initialize CostTracker
        tracker = CostTracker()

        # Run Demo Sections (passing gateway and tracker)
        await demo_section_1_browse(gateway, tracker)
        await demo_section_2_interaction(gateway, tracker)
        await demo_section_3_search(gateway, tracker)
        await demo_section_4_download(gateway, tracker)
        await demo_section_5_macro(gateway, tracker)
        await demo_section_6_autopilot(gateway, tracker)  # Uncomment to run autopilot
        # console.print(
        #     "[yellow]Skipping Autopilot demo section (can be intensive). Uncomment to run.[/]"
        # )
        await demo_section_7_parallel(gateway, tracker)
        await demo_section_8_docs(gateway, tracker)

        console.print(Rule("[bold magenta]Demo Complete[/bold magenta]"))

    except Exception as e:
        logger.critical(f"Demo failed with critical error: {e}", exc_info=True)
        console.print("[bold red]CRITICAL ERROR DURING DEMO:[/]")
        console.print_exception(show_locals=True)
        exit_code = 1
    finally:
        # Shutdown Smart Browser
        console.print("[cyan]Shutting down Smart Browser tool...[/]")
        try:
            await shutdown()
        except Exception as e:
            logger.error(f"Error during Smart Browser shutdown: {e}")

    return exit_code


if __name__ == "__main__":
    # Ensure the script is run with asyncio
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user. Shutting down...[/]")
        # Try to run shutdown asynchronously even on keyboard interrupt
        try:
            asyncio.run(shutdown())
        except Exception as e:
            print(f"Error during emergency shutdown: {e}")
        sys.exit(1)
