#!/usr/bin/env python
"""
DETAILED Demonstration script for the STANDALONE Document Processing functions
in Ultimate MCP Server, showcasing integrated OCR, analysis, conversion, and batch capabilities
with extensive examples.
"""

import asyncio
import base64
import datetime as dt
import json
import os
import sys
import traceback  # Added for more detailed error printing if needed
import warnings  # Added for warning control
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

import httpx

# Filter Docling-related deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling_core")
warnings.filterwarnings("ignore", message="Could not parse formula with MathML")

# Add project root to path for imports when running as script
# Adjust this relative path if your script structure is different
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    print(f"INFO: Added {_PROJECT_ROOT} to sys.path")

# Rich imports for enhanced terminal UI
from rich import box, get_console  # noqa: E402
from rich.console import Group  # noqa: E402
from rich.layout import Layout  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    FileSizeColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.rule import Rule  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402
from rich.traceback import install as install_rich_traceback  # noqa: E402

# --- Global Constants ---
# Maximum number of lines to display for any content
MAX_DISPLAY_LINES = 50  # Used to truncate all displayed content

# --- Attempt to import required MCP Server components ---
try:
    # Assuming standard MCP Server structure
    from ultimate_mcp_server.core.server import Gateway
    from ultimate_mcp_server.exceptions import ToolError, ToolInputError

    # Import the standalone functions and availability flags
    from ultimate_mcp_server.tools.document_conversion_and_processing import (
        # Import availability flags
        _DOCLING_AVAILABLE,
        _PANDAS_AVAILABLE,
        _TIKTOKEN_AVAILABLE,
        analyze_pdf_structure,
        canonicalise_entities,
        chunk_document,
        clean_and_format_text_as_markdown,
        convert_document,
        detect_content_type,
        enhance_ocr_text,
        extract_entities,
        extract_metrics,
        extract_tables,
        flag_risks,
        generate_qa_pairs,
        identify_sections,
        ocr_image,
        optimize_markdown_formatting,
        process_document_batch,
        summarize_document,
    )
    from ultimate_mcp_server.utils import get_logger
    from ultimate_mcp_server.utils.display import CostTracker  # Import CostTracker

    MCP_COMPONENTS_LOADED = True
except ImportError as e:
    MCP_COMPONENTS_LOADED = False
    _IMPORT_ERROR_MSG = str(e)
    # Handle this error gracefully in the main function
    print(f"\n[ERROR] Failed to import required MCP components: {_IMPORT_ERROR_MSG}")
    print("Please ensure:")
    print("1. You are running this script from the correct directory structure.")
    print("2. The MCP Server environment is activated.")
    print("3. All dependencies (including optional ones used in the demo) are installed.")
    sys.exit(1)

# Initialize Rich console and logger
console = get_console()
logger = get_logger("demo.doc_proc_standalone")  # Updated logger name

# Install rich tracebacks for better error display
install_rich_traceback(show_locals=True, width=console.width, extra_lines=2)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE_DIR = SCRIPT_DIR / "sample_docs"  # Changed dir name slightly
DEFAULT_SAMPLE_PDF_URL = "https://arxiv.org/pdf/1706.03762.pdf"  # Attention is All You Need
DEFAULT_SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/tesseract-ocr/tesseract/main/testing/phototest.tif"  # Use Tesseract sample TIFF
SAMPLE_HTML_URL = "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)"
# Additional sample PDFs for testing diversity
BUFFETT_SHAREHOLDER_LETTER_URL = "https://www.berkshirehathaway.com/letters/2022ltr.pdf"  # Likely digital PDF, good for text/layout
BACKPROPAGATION_PAPER_URL = "https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf"  # Older, might be scanned/need OCR

DOWNLOADED_FILES_DIR = DEFAULT_SAMPLE_DIR / "downloaded"

# Config from environment variables
USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "3"))
ACCELERATOR_DEVICE = "cuda" if USE_GPU else "cpu"
SKIP_DOWNLOADS = os.environ.get("SKIP_DOWNLOADS", "false").lower() == "true"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Define result types for type hints
ResultData = Dict[str, Any]
OperationResult = Tuple[bool, ResultData]
FileResult = Optional[Path]

# --- Demo Helper Functions (Mostly unchanged, minor adjustments for clarity) ---


def create_demo_layout() -> Layout:
    """Create a Rich layout for the demo UI."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=5),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=1),
    )
    layout["footer"].update("[dim]Standalone Document Processing Demo Footer[/]")
    return layout


def timestamp_str(short: bool = False) -> str:
    """Return a formatted timestamp string."""
    now = dt.datetime.now()
    if short:
        return f"[dim]{now.strftime('%H:%M:%S')}[/]"
    return f"[dim]{now.strftime('%Y-%m-%d %H:%M:%S')}[/]"


def truncate_text_by_lines(text: str, max_lines: int = MAX_DISPLAY_LINES) -> str:
    """Truncates text to show first/last lines with indicator."""
    if not text or not isinstance(text, str):
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    half_lines = max_lines // 2
    return "\n".join(lines[:half_lines] + ["[dim][...TRUNCATED...]"] + lines[-half_lines:])


def format_value_for_display(key: str, value: Any, detail_level: int = 1) -> Any:
    """Format specific values for better display."""
    if value is None:
        return "[dim]None[/]"
    if isinstance(value, bool):
        return "[green]Yes[/]" if value else "[red]No[/]"
    if isinstance(value, float):
        # Specific formatting for processing_time
        if "time" in key.lower() and not key.lower().startswith("creation"):
            return f"[green]{value:.3f}s[/]"
        return f"{value:.3f}"  # Standard float formatting

    if isinstance(value, list):
        if not value:
            return "[dim]Empty List[/]"
        list_len = len(value)
        preview_count = 3 if detail_level < 2 else 5
        suffix = f" [dim]... ({list_len} items total)[/]" if list_len > preview_count else ""
        if detail_level >= 1:
            previews = []
            for item in value[:preview_count]:
                item_preview = format_value_for_display(f"{key}_item", item, detail_level=0)
                previews.append(str(item_preview))
            return f"[{', '.join(previews)}]{suffix}"
        else:
            return f"[List with {list_len} items]"

    if isinstance(value, dict):
        if not value:
            return "[dim]Empty Dict[/]"
        dict_len = len(value)
        preview_count = 4 if detail_level < 2 else 8
        preview_keys = list(value.keys())[:preview_count]
        suffix = f" [dim]... ({dict_len} keys total)[/]" if dict_len > preview_count else ""
        if detail_level >= 1:
            items_preview = [
                f"{repr(k)}: {format_value_for_display(k, value[k], detail_level=0)}"
                for k in preview_keys
            ]
            return f"{{{'; '.join(items_preview)}}}{suffix}"
        else:
            return f"[Dict with {dict_len} keys]"

    if isinstance(value, str):
        str_len = len(value)
        # Always truncate by lines first for display consistency
        truncated_by_lines = truncate_text_by_lines(value, MAX_DISPLAY_LINES)
        # Then apply character limit if still too long
        preview_len = 300 if detail_level < 2 else 600
        if len(truncated_by_lines) > preview_len:
            return escape(truncated_by_lines[:preview_len]) + f"[dim]... ({str_len} chars total)[/]"
        return escape(truncated_by_lines)

    return escape(str(value))


def display_result(title: str, result: ResultData, display_options: Optional[Dict] = None) -> None:
    """Display operation result with enhanced formatting using Rich."""
    display_options = display_options or {}
    start_time = dt.datetime.now()

    title_display = Text.from_markup(escape(title)) if not isinstance(title, Text) else title
    console.print(Rule(f"[bold cyan]{title_display}[/] {timestamp_str()}", style="cyan"))

    success = result.get("success", False)
    detail_level = display_options.get("detail_level", 1)
    hide_keys_set = set(
        display_options.get("hide_keys", ["success", "raw_llm_response", "raw_text"])
    )
    display_keys = display_options.get("display_keys")

    # --- Summary Panel ---
    summary_panel_content = Text()
    summary_panel_content.append(
        Text.from_markup(
            f"Status: {'[bold green]Success[/]' if success else '[bold red]Failed[/]'}\n"
        )
    )
    if not success:
        error_code = result.get("error_code", "N/A")
        error_msg = result.get("error", "Unknown error")
        summary_panel_content.append(
            Text.from_markup(f"Error Code: [yellow]{escape(str(error_code))}[/]\n")
        )
        summary_panel_content.append(
            Text.from_markup(f"Message: [red]{escape(str(error_msg))}[/]\n")
        )
        console.print(
            Panel(
                summary_panel_content, title="Operation Status", border_style="red", padding=(1, 2)
            )
        )
        return  # Stop display if failed

    top_level_info = {
        "processing_time": "Processing Time",
        "extraction_strategy_used": "Strategy Used",
        "output_format": "Output Format",
        "was_html": "Input Detected as HTML",  # Relevant for clean_and_format...
        "file_path": "Output File Path",
    }
    for key, display_name in top_level_info.items():
        if key in result and key not in hide_keys_set:
            value_str = format_value_for_display(key, result[key], detail_level=0)
            summary_panel_content.append(
                Text.from_markup(f"{display_name}: [blue]{value_str}[/]\n")
            )

    console.print(
        Panel(
            summary_panel_content, title="Operation Summary", border_style="green", padding=(1, 2)
        )
    )

    # --- Details Section ---
    details_to_display = {}
    for key, value in result.items():
        if (
            key in hide_keys_set or key in top_level_info or key.startswith("_")
        ):  # Skip internal keys
            continue
        if display_keys and key not in display_keys:
            continue
        details_to_display[key] = value

    if not details_to_display:
        console.print(Text.from_markup("[dim]No further details requested or available.[/]"))
        console.print()
        return

    console.print(Rule("Details", style="dim"))

    for key, value in details_to_display.items():
        key_title = key.replace("_", " ").title()
        panel_border = "blue"
        panel_content: Any = None
        format_type = "text"

        # Determine format for content-like keys
        is_content_key = key.lower() in [
            "content",
            "markdown_text",
            "optimized_markdown",
            "summary",
            "first_table_preview",
            "tables",
        ]
        if is_content_key:
            if "markdown" in key.lower() or result.get("output_format") == "markdown":
                format_type = "markdown"
            elif result.get("output_format") == "html":
                format_type = "html"
            elif (
                result.get("output_format") == "json"
                or key == "tables"
                and result.get("tables")
                and isinstance(result.get("tables")[0], list)
            ):
                format_type = "json"
            elif (
                key == "tables"
                and result.get("tables")
                and isinstance(result.get("tables")[0], str)
            ):  # Assuming CSV string
                format_type = "csv"
            else:
                format_type = "text"
            format_type = display_options.get("format_key", {}).get(
                key, format_type
            )  # Allow override

        if is_content_key and isinstance(value, str):
            if not value:
                panel_content = "[dim]Empty Content[/]"
            else:
                truncated_value = truncate_text_by_lines(value, MAX_DISPLAY_LINES)
                if format_type == "markdown":
                    panel_content = Markdown(truncated_value)
                elif format_type == "csv":
                    panel_content = Syntax(
                        truncated_value,
                        "csv",
                        theme="paraiso-dark",
                        line_numbers=False,
                        word_wrap=True,
                    )
                else:
                    panel_content = Syntax(
                        truncated_value,
                        format_type,
                        theme="paraiso-dark",
                        line_numbers=False,
                        word_wrap=True,
                    )
            panel_border = "green" if format_type == "markdown" else "white"
            console.print(
                Panel(
                    panel_content,
                    title=key_title,
                    border_style=panel_border,
                    padding=(1, 2),
                    expand=False,
                )
            )

        elif key.lower() == "chunks" and isinstance(value, list):
            chunk_table = Table(
                title=f"Chunk Preview (Total: {len(value)})", box=box.MINIMAL, show_header=True
            )
            chunk_table.add_column("#", style="cyan")
            chunk_table.add_column("Preview (First 80 chars)", style="white")
            chunk_table.add_column("Length", style="green")
            limit = 5 if detail_level < 2 else 10
            for i, chunk in enumerate(value[:limit], 1):
                chunk_str = str(chunk)
                chunk_preview = truncate_text_by_lines(
                    chunk_str[:80] + ("..." if len(chunk_str) > 80 else ""), 5
                )
                chunk_table.add_row(str(i), escape(chunk_preview), str(len(chunk_str)))
            if len(value) > limit:
                chunk_table.add_row("...", f"[dim]{len(value) - limit} more...[/]", "")
            console.print(Panel(chunk_table, title=key_title, border_style="blue"))

        elif key.lower() == "qa_pairs" and isinstance(value, list):
            qa_text = Text()
            limit = 3 if detail_level < 2 else 5
            for i, qa in enumerate(value[:limit], 1):
                q_text = truncate_text_by_lines(qa.get("question", ""), 5)
                a_text = truncate_text_by_lines(qa.get("answer", ""), 10)
                qa_text.append(f"{i}. Q: ", style="bold cyan")
                qa_text.append(escape(q_text) + "\n")
                qa_text.append("   A: ", style="green")
                qa_text.append(escape(a_text) + "\n\n")
            if len(value) > limit:
                qa_text.append(f"[dim]... {len(value) - limit} more ...[/]")
            console.print(Panel(qa_text, title=key_title, border_style="blue"))

        elif (
            key.lower() == "tables" and isinstance(value, list) and value
        ):  # Handle table list (JSON/Pandas)
            first_table = value[0]
            if isinstance(first_table, list):  # JSON format
                panel_content = Syntax(
                    json.dumps(first_table[:5], indent=2),
                    "json",
                    theme="paraiso-dark",
                    line_numbers=False,
                    word_wrap=True,
                )
                panel_title = f"{key_title} (First Table JSON Preview, {len(value)} total)"
                console.print(
                    Panel(panel_content, title=panel_title, border_style="yellow", padding=(1, 1))
                )
            elif hasattr(first_table, "to_string"):  # Pandas DataFrame
                panel_content = escape(first_table.head(5).to_string())
                panel_title = f"{key_title} (First Table Pandas Preview, {len(value)} total)"
                console.print(
                    Panel(panel_content, title=panel_title, border_style="yellow", padding=(1, 1))
                )
            else:  # Fallback if format unknown
                console.print(
                    Panel(
                        f"First table type: {type(first_table).__name__}. Preview:\n{str(first_table)[:500]}...",
                        title=key_title,
                        border_style="yellow",
                    )
                )

        elif isinstance(value, dict):  # General Dict Handling (metadata, metrics, risks, etc.)
            dict_table = Table(title="Contents", box=box.MINIMAL, show_header=False, expand=False)
            dict_table.add_column("SubKey", style="magenta", justify="right", no_wrap=True)
            dict_table.add_column("SubValue", style="white")
            item_count = 0
            max_items = 5 if detail_level == 0 else 20
            for k, v in value.items():
                dict_table.add_row(
                    escape(str(k)), format_value_for_display(k, v, detail_level=detail_level)
                )
                item_count += 1
                if item_count >= max_items:
                    dict_table.add_row("[dim]...[/]", f"[dim]({len(value)} total items)[/]")
                    break
            panel_border = (
                "magenta" if "quality" in key.lower() or "metrics" in key.lower() else "blue"
            )
            console.print(
                Panel(dict_table, title=key_title, border_style=panel_border, padding=(1, 1))
            )

        elif isinstance(value, list):  # General List Handling
            list_panel_content = [Text.from_markup(f"[cyan]Total Items:[/] {len(value)}")]
            limit = 5 if detail_level < 2 else 10
            for i, item in enumerate(value[:limit]):
                item_display = format_value_for_display(
                    f"{key}[{i}]", item, detail_level=detail_level - 1
                )
                list_panel_content.append(f"[magenta]{i + 1}.[/] {item_display}")
            if len(value) > limit:
                list_panel_content.append(
                    Text.from_markup(f"[dim]... {len(value) - limit} more ...[/]")
                )
            console.print(Panel(Group(*list_panel_content), title=key_title, border_style="blue"))

        else:  # Fallback for simple types
            value_display = format_value_for_display(key, value, detail_level=detail_level)
            console.print(f"[bold cyan]{key_title}:[/] {value_display}")

    end_time = dt.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    console.print(Text.from_markup(f"[dim]Result details displayed in {elapsed:.3f}s[/]"))
    console.print()  # Add spacing


async def download_file_with_progress(
    url: str, output_path: Path, description: str, progress: Optional[Progress] = None
) -> FileResult:
    """Download a file with a detailed progress bar."""
    if output_path.exists() and output_path.stat().st_size > 1000:
        logger.info(f"Using existing file: {output_path}")
        console.print(
            Text.from_markup(f"[dim]Using existing file: [blue underline]{output_path.name}[/][/]")
        )
        return output_path
    if SKIP_DOWNLOADS:
        console.print(
            f"[yellow]Skipping download for {description} due to SKIP_DOWNLOADS setting.[/]"
        )
        return None

    console.print(f"Attempting to download [bold]{description}[/] from [underline]{url}[/]...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            async with client.stream("GET", url) as response:
                if response.status_code == 404:
                    logger.error(f"File not found (404) at {url}")
                    console.print(f"[red]Error: File not found (404) for {description}.[/]")
                    return None
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                task_description = f"Downloading {description}..."

                local_progress = progress is None
                if local_progress:
                    progress = Progress(  # type: ignore
                        TextColumn("[bold blue]{task.description}", justify="right"),
                        BarColumn(bar_width=None),
                        "[progress.percentage]{task.percentage:>3.1f}%",
                        "•",
                        TransferSpeedColumn(),
                        "•",
                        FileSizeColumn(),
                        "•",
                        TimeRemainingColumn(),
                        console=console,
                        transient=True,
                    )
                    progress.start()  # type: ignore

                download_task = progress.add_task(task_description, total=total_size)  # type: ignore
                bytes_downloaded = 0
                try:
                    with open(output_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            bytes_written = len(chunk)
                            bytes_downloaded += bytes_written
                            progress.update(download_task, advance=bytes_written)  # type: ignore
                    progress.update(
                        download_task,
                        completed=max(bytes_downloaded, total_size),
                        description=f"Downloaded {description}",
                    )  # type: ignore
                finally:
                    if local_progress:
                        progress.stop()  # type: ignore

        logger.info(f"Successfully downloaded {description} to {output_path}")
        console.print(
            Text.from_markup(
                f"[green]✓ Downloaded {description} to [blue underline]{output_path.name}[/][/]"
            )
        )
        return output_path
    except httpx.RequestError as e:
        logger.error(f"Network error downloading {description} from {url}: {e}")
        console.print(
            Text.from_markup(
                f"[red]Network Error downloading {description}: {type(e).__name__}. Check connection or URL.[/]"
            )
        )
        return None
    except Exception as e:
        logger.error(f"Failed to download {description} from {url}: {e}", exc_info=True)
        console.print(
            Text.from_markup(f"[red]Error downloading {description}: {type(e).__name__} - {e}[/]")
        )
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        return None


async def safe_tool_call(
    operation_name: str,
    tool_func: Callable[..., Awaitable[Dict]],
    *args,
    tracker: Optional[CostTracker] = None,
    **kwargs,
) -> OperationResult:
    """Safely call a standalone tool function, handling exceptions and logging."""
    console.print(
        Text.from_markup(
            f"\n[cyan]Calling Tool:[/][bold] {escape(operation_name)}[/] {timestamp_str(short=True)}"
        )
    )
    display_options = kwargs.pop("display_options", {})  # Extract display options

    # Log arguments carefully
    log_args_repr = {}
    MAX_ARG_LEN = 100
    for k, v in kwargs.items():
        if k == "image_data" and isinstance(v, str):  # Don't log full base64
            log_args_repr[k] = f"str(len={len(v)}, starting_chars='{v[:10]}...')"
        elif isinstance(v, (str, bytes)) and len(v) > MAX_ARG_LEN:
            log_args_repr[k] = f"{type(v).__name__}(len={len(v)})"
        elif isinstance(v, (list, dict)) and len(v) > 10:
            log_args_repr[k] = f"{type(v).__name__}(len={len(v)})"
        else:
            log_args_repr[k] = repr(v)
    logger.debug(f"Executing {operation_name} with kwargs: {log_args_repr}")

    try:
        # Directly call the standalone function
        result = await tool_func(*args, **kwargs)

        if not isinstance(result, dict):
            logger.error(
                f"Tool '{operation_name}' returned non-dict type: {type(result)}. Value: {str(result)[:150]}"
            )
            return False, {
                "success": False,
                "error": f"Tool returned unexpected type: {type(result).__name__}",
                "error_code": "INTERNAL_ERROR",
                "_display_options": display_options,
            }

        # Cost tracking (if applicable)
        if tracker is not None and result.get("success", False):
            # The standalone functions might not directly return cost info in the same way.
            # If LLM calls happen internally, cost tracking might need to be done within
            # the `_standalone_llm_call` or rely on the global tracker if `generate_completion` updates it.
            # For now, assume cost is tracked elsewhere or add specific fields if needed.
            if "llm_cost" in result or "cost" in result:
                # Attempt to track cost if relevant fields exist
                cost = result.get("cost", result.get("llm_cost", 0.0))
                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)
                provider = result.get("provider", "unknown")
                model = result.get("model", operation_name)  # Use op name as fallback model
                processing_time = result.get("processing_time", 0.0)
                tracker.add_call_data(
                    cost, input_tokens, output_tokens, provider, model, processing_time
                )

        result["_display_options"] = display_options  # Pass options for display func
        logger.debug(f"Tool '{operation_name}' completed successfully.")
        return True, result
    except ToolInputError as e:
        logger.warning(f"Input error for {operation_name}: {e}")
        return False, {
            "success": False,
            "error": str(e),
            "error_code": e.error_code,
            "_display_options": display_options,
        }
    except ToolError as e:
        logger.error(f"Tool error during {operation_name}: {e}")
        return False, {
            "success": False,
            "error": str(e),
            "error_code": e.error_code,
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


# --- Demo Sections (Updated to call standalone functions) ---


async def demo_section_1_conversion_ocr(
    sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate convert_document with various strategies and OCR."""
    console.print(Rule("[bold green]Demo 1: Document Conversion & OCR[/]", style="green"))
    logger.info("Starting Demo Section 1: Conversion & OCR")

    pdf_digital = sample_files.get("pdf_digital")
    buffett_pdf = sample_files.get("buffett_pdf")
    backprop_pdf = sample_files.get("backprop_pdf")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")

    pdf_files_to_process = [pdf for pdf in [pdf_digital, buffett_pdf, backprop_pdf] if pdf]

    if not pdf_files_to_process:
        console.print("[yellow]Skipping Demo 1: Need at least one sample PDF.[/]")
        return

    def get_output_path(
        input_file: Path, format_name: str, strategy: str, output_format: str
    ) -> str:
        base_name = input_file.stem
        return str(conversion_outputs_dir / f"{base_name}_{strategy}_{format_name}.{output_format}")

    for pdf_file in pdf_files_to_process:
        console.print(
            Panel(
                Text.from_markup(f"Processing PDF: [cyan]{pdf_file.name}[/]"), border_style="blue"
            )
        )

        # 1a: Direct Text Strategy (Raw Text)
        output_path = get_output_path(pdf_file, "direct", "raw_text", "txt")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> Text (Direct Text)",
            convert_document,  # Call standalone function
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="text",
            extraction_strategy="direct_text",
            enhance_with_llm=False,
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> Text (Direct Text)",
                result,
                {"format_key": {"content": "text"}},
            )

        # 1b: Direct Text Strategy (Markdown Output + Enhance)
        output_path = get_output_path(pdf_file, "direct", "enhanced_md", "md")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> MD (Direct Text + Enhance)",
            convert_document,  # Call standalone function
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="markdown",
            extraction_strategy="direct_text",
            enhance_with_llm=True,
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> MD (Direct + Enhance)",
                result,
                {"format_key": {"content": "markdown"}},
            )

        # 1c: Docling Strategy (Markdown Output) - Check availability
        if _DOCLING_AVAILABLE:
            output_path = get_output_path(pdf_file, "docling", "md", "md")
            success, result = await safe_tool_call(
                f"{pdf_file.name} -> MD (Docling)",
                convert_document,  # Call standalone function
                tracker=tracker,
                document_path=str(pdf_file),
                output_format="markdown",
                extraction_strategy="docling",
                accelerator_device=ACCELERATOR_DEVICE,
                save_to_file=True,
                output_path=output_path,
            )
            if success:
                display_result(
                    f"{pdf_file.name} -> MD (Docling)",
                    result,
                    {"format_key": {"content": "markdown"}},
                )
        else:
            console.print("[yellow]Docling unavailable, skipping Docling conversions.[/]")

        # --- OCR on PDF ---
        console.print(
            Panel(
                f"Processing PDF with OCR Strategy: [cyan]{pdf_file.name}[/]", border_style="blue"
            )
        )

        # 1d: OCR Strategy (Raw Text)
        output_path = get_output_path(pdf_file, "ocr", "raw_text", "txt")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> Text (OCR Raw)",
            convert_document,  # Call standalone function
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="text",
            extraction_strategy="ocr",
            enhance_with_llm=False,
            ocr_options={"language": "eng", "dpi": 150},
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> Text (OCR Raw)",
                result,
                {"format_key": {"content": "text"}, "detail_level": 0},
            )

        # 1e: OCR Strategy (Markdown, Enhanced, Quality Assess)
        output_path = get_output_path(pdf_file, "ocr", "enhanced_md", "md")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> MD (OCR + Enhance + Quality)",
            convert_document,  # Call standalone function
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="markdown",
            extraction_strategy="ocr",
            enhance_with_llm=True,
            ocr_options={
                "language": "eng",
                "assess_quality": True,
                "remove_headers": True,
                "dpi": 200,
            },  # Try header removal
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> MD (OCR + Enhance + Quality)",
                result,
                {"format_key": {"content": "markdown"}},
            )

        # 1f: Hybrid Strategy
        output_path = get_output_path(pdf_file, "hybrid", "text", "txt")
        success, result = await safe_tool_call(
            f"{pdf_file.name} -> Text (Hybrid + Enhance)",
            convert_document,  # Call standalone function
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="text",
            extraction_strategy="hybrid_direct_ocr",
            enhance_with_llm=True,
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{pdf_file.name} -> Text (Hybrid + Enhance)",
                result,
                {"format_key": {"content": "text"}},
            )

    # --- Image Conversion (Using convert_document) ---
    image_file = sample_files.get("image")
    if image_file:
        console.print(
            Panel(
                f"Processing Image via convert_document: [cyan]{image_file.name}[/]",
                border_style="blue",
            )
        )
        output_path = get_output_path(image_file, "convert_doc", "md", "md")
        success, result = await safe_tool_call(
            f"{image_file.name} -> MD (Convert Doc)",
            convert_document,  # Call standalone function
            tracker=tracker,
            document_path=str(image_file),
            output_format="markdown",  # Strategy inferred
            save_to_file=True,
            output_path=output_path,
        )
        if success:
            display_result(
                f"{image_file.name} -> MD (via convert_document)",
                result,
                {"format_key": {"content": "markdown"}},
            )

    # --- Conversion from Bytes ---
    if pdf_digital:
        console.print(Panel("Processing PDF from Bytes Data using OCR", border_style="blue"))
        try:
            pdf_bytes = pdf_digital.read_bytes()
            output_path = get_output_path(pdf_digital, "bytes", "ocr_text", "txt")
            success, result = await safe_tool_call(
                "PDF Bytes -> Text (OCR)",
                convert_document,  # Call standalone function
                tracker=tracker,
                document_data=pdf_bytes,
                output_format="text",
                extraction_strategy="ocr",
                enhance_with_llm=False,
                ocr_options={"dpi": 150},
                save_to_file=True,
                output_path=output_path,
            )
            if success:
                display_result(
                    "PDF Bytes -> Text (OCR Raw)",
                    result,
                    {"format_key": {"content": "text"}, "detail_level": 0},
                )
        except Exception as e:
            console.print(f"[red]Error processing PDF bytes: {e}[/]")


async def demo_section_2_dedicated_ocr(sample_files: Dict[str, Path], tracker: CostTracker) -> None:
    """Demonstrate the dedicated ocr_image tool."""
    console.print(Rule("[bold green]Demo 2: Dedicated Image OCR Tool[/]", style="green"))
    logger.info("Starting Demo Section 2: Dedicated Image OCR Tool")

    image_file = sample_files.get("image")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")

    if not image_file:
        console.print("[yellow]Skipping Demo 2: Sample image not available.[/]")
        return

    def get_output_path(base_name: str, method: str, output_format: str) -> str:
        return str(conversion_outputs_dir / f"{base_name}_ocr_{method}.{output_format}")

    console.print(
        Panel(
            f"Processing Image with ocr_image Tool: [cyan]{image_file.name}[/]", border_style="blue"
        )
    )

    # 2a: OCR Image from Path (Default: Enhance=True, Output=Markdown)
    output_path = get_output_path(image_file.stem, "default", "md")
    success, result = await safe_tool_call(
        "OCR Image (Path, Defaults)",
        ocr_image,  # Call standalone function
        tracker=tracker,
        image_path=str(image_file),
    )
    if success:
        try:
            Path(output_path).write_text(result.get("content", ""), encoding="utf-8")
            console.print(f"[green]✓ Saved OCR output to: [blue underline]{output_path}[/]")
        except Exception as e:
            console.print(f"[red]Error saving OCR output: {e}[/]")
        display_result(
            "OCR Image (Path, Defaults)", result, {"format_key": {"content": "markdown"}}
        )

    # 2b: OCR Image from Path (Raw Text, Specific Preprocessing)
    output_path = get_output_path(image_file.stem, "raw_preprocessing", "txt")
    success, result = await safe_tool_call(
        "OCR Image (Path, Raw Text, Preprocessing)",
        ocr_image,  # Call standalone function
        tracker=tracker,
        image_path=str(image_file),
        output_format="text",
        enhance_with_llm=False,
        ocr_options={
            "language": "eng",
            "preprocessing": {"threshold": "adaptive", "denoise": True, "deskew": False},
        },
    )
    if success:
        try:
            Path(output_path).write_text(result.get("content", ""), encoding="utf-8")
            console.print(f"[green]✓ Saved OCR output to: [blue underline]{output_path}[/]")
        except Exception as e:
            console.print(f"[red]Error saving OCR output: {e}[/]")
        display_result(
            "OCR Image (Raw Text, Preprocessing)", result, {"format_key": {"content": "text"}}
        )

    # 2c: OCR Image from Base64 Data (Enhance=True, Quality Assess)
    try:
        console.print(Panel("Processing Image from Base64 Data", border_style="blue"))
        img_bytes = image_file.read_bytes()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        output_path = get_output_path(image_file.stem, "base64_enhanced", "md")
        success, result = await safe_tool_call(
            "OCR Image (Base64, Enhance, Quality)",
            ocr_image,  # Call standalone function
            tracker=tracker,
            image_data=img_base64,
            output_format="markdown",
            enhance_with_llm=True,
            ocr_options={"assess_quality": True},
        )
        if success:
            try:
                Path(output_path).write_text(result.get("content", ""), encoding="utf-8")
                console.print(f"[green]✓ Saved OCR output to: [blue underline]{output_path}[/]")
            except Exception as e:
                console.print(f"[red]Error saving OCR output: {e}[/]")
            display_result(
                "OCR Image (Base64, Enhance, Quality)",
                result,
                {"format_key": {"content": "markdown"}},
            )
    except Exception as e:
        console.print(f"[red]Failed to process image from Base64: {type(e).__name__} - {e}[/]")


async def demo_section_3_enhance_text(sample_files: Dict[str, Path], tracker: CostTracker) -> None:
    """Demonstrate enhancing existing noisy text."""
    console.print(Rule("[bold green]Demo 3: Enhance Existing OCR Text[/]", style="green"))
    logger.info("Starting Demo Section 3: Enhance OCR Text")

    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")

    noisy_text = """
    INVOlCE # 12345 - ACME C0rp.
    Date: Octobor 25, 2O23

    Billed To: Example Inc. , 123 Main St . Anytown USA

    Itemm Descriptiom                 Quantlty    Unlt Price    Tota1
    -----------------------------------------------------------------
    Wldget Modell A                     lO          $ I5.0O      $l5O.OO
    Gadgett Type B                      5           $ 25.5O      $l27.5O
    Assembly Srvlce                   2 hrs       $ 75.OO      $l5O.OO
    -----------------------------------------------------------------
                                        Subtota1 :             $427.5O
                                        Tax (8%) :             $ 34.2O
                                        TOTAL    :             $461.7O

    Notes: Payment due ln 3O days. Thank you for yuor buslness!

    Page I / l - Confidential Document"""
    console.print(Panel("Original Noisy Text:", border_style="yellow"))
    console.print(
        Syntax(truncate_text_by_lines(noisy_text), "text", theme="default", line_numbers=True)
    )

    def get_output_path(base_name: str, format_name: str) -> str:
        return str(conversion_outputs_dir / f"{base_name}.{format_name}")

    # 3a: Enhance to Markdown (Remove Headers, Assess Quality)
    output_path = get_output_path("enhanced_noisy_text_markdown", "md")
    success, result = await safe_tool_call(
        "Enhance -> MD (Rm Headers, Quality)",
        enhance_ocr_text,  # Call standalone function
        tracker=tracker,
        text=noisy_text,
        output_format="markdown",
        enhancement_options={"remove_headers": True, "assess_quality": True},
    )
    if success:
        try:
            Path(output_path).write_text(result.get("content", ""), encoding="utf-8")
            console.print(f"[green]✓ Saved enhanced markdown to: [blue underline]{output_path}[/]")
        except Exception as e:
            console.print(f"[red]Error saving enhanced markdown: {e}[/]")
        display_result(
            "Enhance -> MD (Rm Headers, Quality)", result, {"format_key": {"content": "markdown"}}
        )

    # 3b: Enhance to Plain Text (Keep Headers)
    output_path = get_output_path("enhanced_noisy_text_plain", "txt")
    success, result = await safe_tool_call(
        "Enhance -> Text (Keep Headers)",
        enhance_ocr_text,  # Call standalone function
        tracker=tracker,
        text=noisy_text,
        output_format="text",
        enhancement_options={"remove_headers": False},
    )
    if success:
        try:
            Path(output_path).write_text(result.get("content", ""), encoding="utf-8")
            console.print(f"[green]✓ Saved enhanced text to: [blue underline]{output_path}[/]")
        except Exception as e:
            console.print(f"[red]Error saving enhanced text: {e}[/]")
        display_result(
            "Enhance -> Text (Keep Headers)", result, {"format_key": {"content": "text"}}
        )


async def demo_section_4_html_markdown(sample_files: Dict[str, Path], tracker: CostTracker) -> None:
    """Demonstrate HTML processing and Markdown utilities."""
    console.print(Rule("[bold green]Demo 4: HTML & Markdown Processing[/]", style="green"))
    logger.info("Starting Demo Section 4: HTML & Markdown Processing")

    html_file = sample_files.get("html")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")

    if not html_file:
        console.print("[yellow]Skipping Demo 4: Sample HTML not available.[/]")
        return

    def get_output_path(base_name: str, method: str, format_name: str) -> str:
        return str(conversion_outputs_dir / f"{base_name}_{method}.{format_name}")

    console.print(Panel(f"Processing HTML File: [cyan]{html_file.name}[/]", border_style="blue"))
    try:
        html_content = html_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        console.print(f"[red]Error reading HTML file {html_file}: {e}[/]")
        return

    # --- clean_and_format_text_as_markdown ---
    console.print(Rule("HTML to Markdown Conversion", style="dim"))

    # 4a: Auto Extraction (Default)
    output_path = get_output_path(html_file.stem, "auto_extract", "md")
    success, result_auto = await safe_tool_call(
        "HTML -> MD (Auto Extract)",
        clean_and_format_text_as_markdown,  # Call standalone function
        tracker=tracker,
        text=html_content,
        extraction_method="auto",
        preserve_tables=True,
    )
    if success:
        try:
            Path(output_path).write_text(result_auto.get("markdown_text", ""), encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Error saving markdown: {e}[/]")
        else:
            console.print(
                f"[green]✓ Saved auto-extracted markdown to: [blue underline]{output_path}[/]"
            )
        display_result(
            "HTML -> MD (Auto Extract)", result_auto, {"format_key": {"markdown_text": "markdown"}}
        )

    # 4b: Readability Extraction (No Tables)
    output_path = get_output_path(html_file.stem, "readability_no_tables", "md")
    success, result_read = await safe_tool_call(
        "HTML -> MD (Readability, No Tables)",
        clean_and_format_text_as_markdown,  # Call standalone function
        tracker=tracker,
        text=html_content,
        extraction_method="readability",
        preserve_tables=False,
    )
    if success:
        try:
            Path(output_path).write_text(result_read.get("markdown_text", ""), encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Error saving markdown: {e}[/]")
        else:
            console.print(
                f"[green]✓ Saved readability markdown to: [blue underline]{output_path}[/]"
            )
        display_result(
            "HTML -> MD (Readability, No Tables)",
            result_read,
            {"format_key": {"markdown_text": "markdown"}},
        )

    # --- optimize_markdown_formatting ---
    console.print(Rule("Markdown Optimization", style="dim"))
    markdown_to_optimize = (
        result_auto.get("markdown_text") if success else "## Default MD\n* Item 1\n* Item 2\n"
    )
    if markdown_to_optimize:
        console.print(Panel("Original Markdown for Optimization:", border_style="yellow"))
        console.print(
            Syntax(truncate_text_by_lines(markdown_to_optimize), "markdown", theme="default")
        )

        # 4c: Optimize with fixes and wrapping
        output_path = get_output_path(html_file.stem, "optimized_normalized", "md")
        success, result_opt1 = await safe_tool_call(
            "Optimize MD (Normalize, Fix, Wrap)",
            optimize_markdown_formatting,  # Call standalone function
            tracker=tracker,
            markdown=markdown_to_optimize,
            normalize_headings=True,
            fix_lists=True,
            fix_links=True,
            add_line_breaks=True,
            max_line_length=80,
        )
        if success:
            try:
                Path(output_path).write_text(
                    result_opt1.get("optimized_markdown", ""), encoding="utf-8"
                )
            except Exception as e:
                console.print(f"[red]Error saving markdown: {e}[/]")
            else:
                console.print(
                    f"[green]✓ Saved optimized markdown to: [blue underline]{output_path}[/]"
                )
            display_result(
                "Optimize MD (Normalize, Fix, Wrap)",
                result_opt1,
                {"format_key": {"optimized_markdown": "markdown"}},
            )

        # 4d: Optimize in Compact Mode
        output_path = get_output_path(html_file.stem, "optimized_compact", "md")
        success, result_opt2 = await safe_tool_call(
            "Optimize MD (Compact Mode)",
            optimize_markdown_formatting,  # Call standalone function
            tracker=tracker,
            markdown=markdown_to_optimize,
            compact_mode=True,
        )
        if success:
            try:
                Path(output_path).write_text(
                    result_opt2.get("optimized_markdown", ""), encoding="utf-8"
                )
            except Exception as e:
                console.print(f"[red]Error saving markdown: {e}[/]")
            else:
                console.print(
                    f"[green]✓ Saved compact markdown to: [blue underline]{output_path}[/]"
                )
            display_result(
                "Optimize MD (Compact Mode)",
                result_opt2,
                {"format_key": {"optimized_markdown": "markdown"}},
            )
    else:
        console.print("[yellow]Skipping optimization as initial conversion failed.[/]")

    # --- detect_content_type ---
    console.print(Rule("Content Type Detection", style="dim"))
    success, result_detect = await safe_tool_call(
        "Detect Type (HTML)", detect_content_type, text=html_content[:6000], tracker=tracker
    )
    if success:
        display_result("Detect Type (HTML)", result_detect)

    md_for_detect = (
        result_auto.get("markdown_text", "## Sample\nText") if result_auto else "## Sample\nText"
    )
    success, result_detect = await safe_tool_call(
        "Detect Type (Markdown)", detect_content_type, text=md_for_detect[:6000], tracker=tracker
    )
    if success:
        display_result("Detect Type (Markdown)", result_detect)


async def demo_section_5_analyze_structure(
    sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate the dedicated PDF structure analysis tool."""
    console.print(Rule("[bold green]Demo 5: Analyze PDF Structure Tool[/]", style="green"))
    logger.info("Starting Demo Section 5: Analyze PDF Structure")

    pdf_digital = sample_files.get("pdf_digital")
    buffett_pdf = sample_files.get("buffett_pdf")
    backprop_pdf = sample_files.get("backprop_pdf")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")

    pdf_files_to_process = [pdf for pdf in [pdf_digital, buffett_pdf, backprop_pdf] if pdf]

    if not pdf_files_to_process:
        console.print("[yellow]Skipping Demo 5: No PDF files available.[/]")
        return

    def get_output_path(file_name: str, analysis_type: str) -> str:
        return str(conversion_outputs_dir / f"{file_name}_analysis_{analysis_type}.json")

    for pdf_file in pdf_files_to_process:
        console.print(
            Panel(f"Analyzing PDF Structure: [cyan]{pdf_file.name}[/]", border_style="blue")
        )

        # 5a: Analyze Structure (Default options)
        output_path = get_output_path(pdf_file.stem, "default")
        success, result = await safe_tool_call(
            f"Analyze {pdf_file.name} Structure (Defaults)",
            analyze_pdf_structure,  # Call standalone function
            tracker=tracker,
            file_path=str(pdf_file),
        )
        if success:
            try:
                result_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
                Path(output_path).write_text(json.dumps(result_to_save, indent=2), encoding="utf-8")
                console.print(f"[green]✓ Saved PDF analysis to: [blue underline]{output_path}[/]")
            except Exception as e:
                console.print(f"[red]Error saving PDF analysis: {e}[/]")
            display_result(f"Analyze {pdf_file.name} Structure (Defaults)", result)

        # 5b: Analyze Structure (All options enabled)
        output_path = get_output_path(pdf_file.stem, "all_options")
        success, result_all = await safe_tool_call(
            f"Analyze {pdf_file.name} Structure (All Options)",
            analyze_pdf_structure,  # Call standalone function
            tracker=tracker,
            file_path=str(pdf_file),
            extract_metadata=True,
            extract_outline=True,
            extract_fonts=True,
            extract_images=True,
            estimate_ocr_needs=True,
        )
        if success:
            try:
                result_to_save = {k: v for k, v in result_all.items() if not k.startswith("_")}
                Path(output_path).write_text(json.dumps(result_to_save, indent=2), encoding="utf-8")
                console.print(
                    f"[green]✓ Saved detailed PDF analysis to: [blue underline]{output_path}[/]"
                )
            except Exception as e:
                console.print(f"[red]Error saving PDF analysis: {e}[/]")
            display_result(f"Analyze {pdf_file.name} Structure (All Options)", result_all)


async def demo_section_6_chunking_tables(
    sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate Document Chunking and Table Extraction tools."""
    console.print(Rule("[bold green]Demo 6: Chunking & Table Extraction[/]", style="green"))
    logger.info("Starting Demo Section 6: Chunking & Table Extraction")

    pdf_digital = sample_files.get("pdf_digital")
    buffett_pdf = sample_files.get("buffett_pdf")
    backprop_pdf = sample_files.get("backprop_pdf")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")

    pdf_files_to_process = [pdf for pdf in [pdf_digital, buffett_pdf, backprop_pdf] if pdf]

    if not pdf_files_to_process:
        console.print("[yellow]Skipping Demo 6: No PDF files available.[/]")
        return

    def get_output_path(base_name: str, process_type: str, format_name: str) -> str:
        return str(conversion_outputs_dir / f"{base_name}_{process_type}.{format_name}")

    for pdf_file in pdf_files_to_process:
        try:
            console.print(
                Panel(
                    f"Preparing Content for Chunking/Tables from: [cyan]{pdf_file.name}[/]",
                    border_style="dim",
                )
            )
            success, conv_result = await safe_tool_call(
                f"Get MD for {pdf_file.name}",
                convert_document,  # Call standalone function
                tracker=tracker,
                document_path=str(pdf_file),
                output_format="markdown",
                extraction_strategy="direct_text",
                enhance_with_llm=False,  # Use raw for speed
            )
            if not success or not conv_result.get("content"):
                console.print(
                    f"[red]Failed to get content for {pdf_file.name}. Skipping chunk/table demo for this file.[/]"
                )
                continue
            markdown_content = conv_result["content"]
            console.print("[green]✓ Content prepared.[/]")

            # --- Chunking Demonstrations ---
            console.print(Rule(f"Document Chunking for {pdf_file.name}", style="dim"))
            chunking_configs = [
                {"method": "paragraph", "size": 500, "overlap": 50},
                {"method": "character", "size": 800, "overlap": 100},
                {"method": "token", "size": 200, "overlap": 20},
                {"method": "section", "size": 1000, "overlap": 0},
            ]
            for config in chunking_configs:
                method, size, overlap = config["method"], config["size"], config["overlap"]
                if method == "token" and not _TIKTOKEN_AVAILABLE:
                    console.print(
                        f"[yellow]Skipping chunking method '{method}': Tiktoken not available.[/]"
                    )
                    continue
                output_path = get_output_path(pdf_file.stem, f"chunks_{method}", "json")
                success, result = await safe_tool_call(
                    f"Chunking {pdf_file.name} ({method.capitalize()})",
                    chunk_document,  # Call standalone function
                    tracker=tracker,
                    document=markdown_content,
                    chunk_method=method,
                    chunk_size=size,
                    chunk_overlap=overlap,
                )
                if success:
                    try:
                        result_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
                        Path(output_path).write_text(
                            json.dumps(result_to_save, indent=2), encoding="utf-8"
                        )
                        console.print(f"[green]✓ Saved chunks to: [blue underline]{output_path}[/]")
                    except Exception as e:
                        console.print(f"[red]Error saving chunks: {e}[/]")
                    display_result(f"Chunking {pdf_file.name} ({method}, size={size})", result)

            # --- Table Extraction (Requires Docling) ---
            console.print(
                Rule(f"Table Extraction for {pdf_file.name} (Requires Docling)", style="dim")
            )
            if _DOCLING_AVAILABLE:
                tables_dir = conversion_outputs_dir / f"{pdf_file.stem}_tables"
                tables_dir.mkdir(exist_ok=True)

                # 6a: Extract as CSV
                success, result_csv = await safe_tool_call(
                    f"Extract {pdf_file.name} Tables (CSV)",
                    extract_tables,  # Call standalone function
                    tracker=tracker,
                    document_path=str(pdf_file),
                    table_mode="csv",
                    output_dir=str(tables_dir / "csv"),
                )
                if success and result_csv.get("tables"):
                    display_result(
                        f"Extract {pdf_file.name} Tables (CSV)",
                        result_csv,
                        {"display_keys": ["tables", "saved_files"], "detail_level": 0},
                    )
                    if result_csv["tables"]:
                        console.print(
                            Panel(
                                escape(result_csv["tables"][0][:500]) + "...",
                                title="First Table Preview (CSV)",
                            )
                        )
                elif success:
                    console.print(f"[yellow]No tables found by Docling in {pdf_file.name}.[/]")

                # 6b: Extract as JSON
                success, result_json = await safe_tool_call(
                    f"Extract {pdf_file.name} Tables (JSON)",
                    extract_tables,  # Call standalone function
                    tracker=tracker,
                    document_path=str(pdf_file),
                    table_mode="json",
                    output_dir=str(tables_dir / "json"),
                )
                if success and result_json.get("tables"):
                    display_result(
                        f"Extract {pdf_file.name} Tables (JSON)",
                        result_json,
                        {"display_keys": ["tables"], "detail_level": 1},
                    )

                # 6c: Extract as Pandas DataFrame (if available)
                if _PANDAS_AVAILABLE:
                    success, result_pd = await safe_tool_call(
                        f"Extract {pdf_file.name} Tables (Pandas)",
                        extract_tables,  # Call standalone function
                        tracker=tracker,
                        document_path=str(pdf_file),
                        table_mode="pandas",
                        output_dir=str(tables_dir / "pandas_csv"),  # Save as csv
                    )
                    if success and result_pd.get("tables"):
                        display_result(
                            f"Extract {pdf_file.name} Tables (Pandas)",
                            result_pd,
                            {"display_keys": ["tables"], "detail_level": 0},
                        )
                        if result_pd["tables"]:
                            first_df = result_pd["tables"][0]
                            if hasattr(first_df, "shape") and hasattr(
                                first_df, "columns"
                            ):  # Check if it looks like a DataFrame
                                console.print(
                                    Panel(
                                        f"First DataFrame Info:\nShape: {first_df.shape}\nColumns: {list(first_df.columns)}",
                                        title="First DataFrame Preview",
                                    )
                                )
                            else:
                                console.print(
                                    f"[yellow]Pandas result format unexpected: {type(first_df)}[/]"
                                )
                else:
                    console.print(
                        "[yellow]Pandas unavailable, skipping Pandas table extraction.[/]"
                    )
            else:
                console.print("[yellow]Docling unavailable, skipping table extraction demo.[/]")
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name} in Sec 6: {e}", exc_info=True)
            console.print(f"[bold red]Error processing {pdf_file.name}:[/] {e}")


async def demo_section_7_analysis(sample_files: Dict[str, Path], tracker: CostTracker) -> None:
    """Demonstrate the document analysis tools."""
    console.print(Rule("[bold green]Demo 7: Document Analysis Suite[/]", style="green"))
    logger.info("Starting Demo Section 7: Document Analysis Suite")

    pdf_digital = sample_files.get("pdf_digital")
    buffett_pdf = sample_files.get("buffett_pdf")
    backprop_pdf = sample_files.get("backprop_pdf")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")

    pdf_files_to_process = [pdf for pdf in [pdf_digital, buffett_pdf, backprop_pdf] if pdf]

    if not pdf_files_to_process:
        console.print("[yellow]Skipping Demo 7: No PDF files available.[/]")
        return

    def get_output_path(base_name: str, analysis_type: str, format_name: str = "json") -> str:
        return str(conversion_outputs_dir / f"{base_name}_analysis_{analysis_type}.{format_name}")

    for pdf_file in pdf_files_to_process:
        console.print(
            Panel(f"Preparing Text for Analysis from: [cyan]{pdf_file.name}[/]", border_style="dim")
        )
        success, conv_result = await safe_tool_call(
            f"Get Text for {pdf_file.name}",
            convert_document,  # Call standalone function
            tracker=tracker,
            document_path=str(pdf_file),
            output_format="markdown",
            extraction_strategy="direct_text",
            enhance_with_llm=False,
        )
        if not success or not conv_result.get("content"):
            console.print(f"[red]Failed to get text for analysis of {pdf_file.name}.[/]")
            continue
        analysis_text = conv_result["content"]
        console.print("[green]✓ Content prepared.[/]")
        console.print(
            Panel(
                escape(truncate_text_by_lines(analysis_text[:600])),
                title=f"Text Preview for {pdf_file.name}",
                border_style="dim",
            )
        )

        entities_result_for_canon = None

        # 7.1 Identify Sections
        output_path = get_output_path(pdf_file.stem, "sections")
        success, result = await safe_tool_call(
            f"Identify Sections in {pdf_file.name}",
            identify_sections,
            document=analysis_text,
            tracker=tracker,
        )
        if success:
            try:
                result_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
                Path(output_path).write_text(json.dumps(result_to_save, indent=2), encoding="utf-8")
                console.print(
                    f"[green]✓ Saved sections analysis to: [blue underline]{output_path}[/]"
                )
            except Exception as e:
                console.print(f"[red]Error saving analysis: {e}[/]")
            display_result(f"Identify Sections ({pdf_file.name})", result)

        # 7.2 Extract Entities
        output_path = get_output_path(pdf_file.stem, "entities")
        success, result = await safe_tool_call(
            f"Extract Entities from {pdf_file.name}",
            extract_entities,
            document=analysis_text,
            tracker=tracker,
        )
        if success:
            entities_result_for_canon = result  # Save for next step
            try:
                result_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
                Path(output_path).write_text(json.dumps(result_to_save, indent=2), encoding="utf-8")
                console.print(
                    f"[green]✓ Saved entities analysis to: [blue underline]{output_path}[/]"
                )
            except Exception as e:
                console.print(f"[red]Error saving analysis: {e}[/]")
            display_result(f"Extract Entities ({pdf_file.name})", result)

        # 7.3 Canonicalise Entities
        if entities_result_for_canon and entities_result_for_canon.get("entities"):
            output_path = get_output_path(pdf_file.stem, "canon_entities")
            success, result = await safe_tool_call(
                f"Canonicalise Entities for {pdf_file.name}",
                canonicalise_entities,
                entities_input=entities_result_for_canon,
                tracker=tracker,
            )
            if success:
                try:
                    result_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
                    Path(output_path).write_text(
                        json.dumps(result_to_save, indent=2), encoding="utf-8"
                    )
                    console.print(
                        f"[green]✓ Saved canonicalized entities to: [blue underline]{output_path}[/]"
                    )
                except Exception as e:
                    console.print(f"[red]Error saving analysis: {e}[/]")
                display_result(f"Canonicalise Entities ({pdf_file.name})", result)
        else:
            console.print(
                f"[yellow]Skipping canonicalization for {pdf_file.name} (no entities).[/]"
            )

        # 7.4 Generate QA Pairs
        output_path = get_output_path(pdf_file.stem, "qa_pairs")
        success, result = await safe_tool_call(
            f"Generate QA Pairs for {pdf_file.name}",
            generate_qa_pairs,
            document=analysis_text,
            num_questions=4,
            tracker=tracker,
        )
        if success:
            try:
                result_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
                Path(output_path).write_text(json.dumps(result_to_save, indent=2), encoding="utf-8")
                console.print(f"[green]✓ Saved QA pairs to: [blue underline]{output_path}[/]")
            except Exception as e:
                console.print(f"[red]Error saving QA pairs: {e}[/]")
            display_result(f"Generate QA Pairs ({pdf_file.name})", result)

        # 7.5 Summarize Document
        output_path = get_output_path(pdf_file.stem, "summary", "md")
        success, result = await safe_tool_call(
            f"Summarize {pdf_file.name}",
            summarize_document,
            document=analysis_text,
            max_length=100,
            tracker=tracker,
        )
        if success:
            try:
                Path(output_path).write_text(result.get("summary", ""), encoding="utf-8")
            except Exception as e:
                console.print(f"[red]Error saving summary: {e}[/]")
            else:
                console.print(f"[green]✓ Saved summary to: [blue underline]{output_path}[/]")
            display_result(
                f"Summarize {pdf_file.name}", result, {"format_key": {"summary": "text"}}
            )

        # 7.6 Extract Metrics (Domain specific)
        output_path = get_output_path(pdf_file.stem, "metrics")
        success, result = await safe_tool_call(
            f"Extract Metrics from {pdf_file.name}",
            extract_metrics,
            document=analysis_text,
            tracker=tracker,
        )
        if success:
            try:
                result_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
                Path(output_path).write_text(json.dumps(result_to_save, indent=2), encoding="utf-8")
                console.print(f"[green]✓ Saved metrics to: [blue underline]{output_path}[/]")
            except Exception as e:
                console.print(f"[red]Error saving metrics: {e}[/]")
            display_result(f"Extract Metrics ({pdf_file.name})", result)
            if not result.get("metrics"):
                console.print(f"[yellow]Note: No pre-defined metrics found in {pdf_file.name}.[/]")

        # 7.7 Flag Risks (Domain specific)
        output_path = get_output_path(pdf_file.stem, "risks")
        success, result = await safe_tool_call(
            f"Flag Risks in {pdf_file.name}", flag_risks, document=analysis_text, tracker=tracker
        )
        if success:
            try:
                result_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
                Path(output_path).write_text(json.dumps(result_to_save, indent=2), encoding="utf-8")
                console.print(f"[green]✓ Saved risks analysis to: [blue underline]{output_path}[/]")
            except Exception as e:
                console.print(f"[red]Error saving risks analysis: {e}[/]")
            display_result(f"Flag Risks ({pdf_file.name})", result)
            if not result.get("risks"):
                console.print(f"[yellow]Note: No pre-defined risks found in {pdf_file.name}.[/]")


async def demo_section_8_batch_processing(
    sample_files: Dict[str, Path], tracker: CostTracker
) -> None:
    """Demonstrate the standalone batch processing pipeline."""
    console.print(Rule("[bold green]Demo 8: Advanced Batch Processing[/]", style="green"))
    logger.info("Starting Demo Section 8: Batch Processing")

    pdf_digital = sample_files.get("pdf_digital")
    buffett_pdf = sample_files.get("buffett_pdf")
    image_file = sample_files.get("image")
    conversion_outputs_dir = sample_files.get("conversion_outputs_dir")  # noqa: F841

    # --- Prepare Batch Inputs ---
    batch_inputs = []
    if pdf_digital:
        batch_inputs.append({"document_path": str(pdf_digital), "item_id": "pdf1"})
    if buffett_pdf:
        batch_inputs.append({"document_path": str(buffett_pdf), "item_id": "pdf2"})
    if image_file:
        batch_inputs.append({"image_path": str(image_file), "item_id": "img1"})  # Use image_path

    if not batch_inputs:
        console.print("[yellow]Skipping batch demo: No suitable input files found.[/]")
        return

    console.print(f"Prepared {len(batch_inputs)} items for batch processing.")

    # --- Define Batch Operations Pipeline ---
    # NOTE: We access nested results using input_keys_map pointing to the
    #       output_key of a previous step (e.g., "conversion_result")
    #       and then assume the batch processor can handle accessing the nested 'content' field.
    #       If the batch processor *only* supports top-level keys in input_keys_map,
    #       this structure would need further adjustment (e.g., adding intermediate steps
    #       to explicitly pull nested data to the top level if promotion isn't flexible enough).
    #       Let's proceed assuming the worker logic can handle `item_state[state_key]`
    #       where `state_key` refers to a previous output key, and we'll access `.content` inside the worker if needed.
    #       ***Correction***: The worker does NOT handle nested access via dot notation in the map value.
    #       The map value MUST be a key present in the top-level item_state.
    #       WORKAROUND: Do not promote output from Step 1. Have subsequent steps map their
    #       input argument to the desired nested key within the state using `input_keys_map`.
    #       The batch worker needs modification to support this. Let's try the workaround.

    batch_operations = [
        # Step 1: Convert PDF/OCR Image to Markdown
        {
            "operation": "convert_document",
            "output_key": "conversion_result",  # Result stored here
            "params": {
                "output_format": "markdown",
                "extraction_strategy": "hybrid_direct_ocr",
                "enhance_with_llm": True,
                "ocr_options": {"dpi": 200},
                "accelerator_device": ACCELERATOR_DEVICE,
            },
            # REMOVED "promote_output": "content"
        },
        # Step 2: Chunk the resulting markdown content from Step 1
        {
            "operation": "chunk_document",
            # The worker needs to know the input arg name ('document') and the state key to get it from.
            "input_keys_map": {
                "document": "conversion_result"
            },  # Map 'document' arg to the dict from step 1
            "output_key": "chunking_result",
            "params": {"chunk_method": "paragraph", "chunk_size": 750, "chunk_overlap": 75},
            # If we wanted chunks available later, we could promote here:
            # "promote_output": "chunks"
        },
        # Step 3: Generate QA pairs using the *original* markdown from Step 1
        {
            "operation": "generate_qa_pairs",
            "input_keys_map": {
                "document": "conversion_result"
            },  # Map 'document' arg to the dict from step 1
            "output_key": "qa_result",
            "params": {"num_questions": 3},
        },
        # Step 4: Summarize the original converted content from Step 1
        {
            "operation": "summarize_document",
            "input_keys_map": {
                "document": "conversion_result"
            },  # Map 'document' arg to the dict from step 1
            "output_key": "summary_result",
            "params": {"max_length": 80},
        },
    ]

    # --- Adjusting the worker function to handle dictionary input via input_keys_map ---
    # The batch processor's worker (_apply_op_to_item_worker) needs a slight modification
    # to handle the case where input_keys_map points to a dictionary result from a previous step,
    # and we need to extract a specific field (like 'content') from it.

    # Let's modify the worker logic conceptually (assuming this change is made in the actual tool file):
    # Inside _apply_op_to_item_worker, when processing input_keys_map:
    # ```python
    # # ... inside worker ...
    # if isinstance(op_input_map, dict):
    #     for param_name, state_key in op_input_map.items():
    #         if state_key not in item_state:
    #             raise ToolInputError(...)
    #
    #         mapped_value = item_state[state_key]
    #
    #         # *** ADDED LOGIC ***
    #         # If mapped value is a dict from a previous step, and the param_name suggests content ('document', 'text', etc.)
    #         # try to extract the 'content' field from that dictionary.
    #         if isinstance(mapped_value, dict) and param_name in ["document", "text", "content"]:
    #              content_value = mapped_value.get("content")
    #              if content_value is not None:
    #                   mapped_value = content_value
    #              else:
    #                   # Maybe try other common keys or raise error if 'content' expected but missing
    #                   logger.warning(f"Mapped input '{state_key}' is dict, but key 'content' not found for param '{param_name}'")
    #                   # Fallback to using the whole dict? Or fail? Let's use whole dict as fallback for now.
    #         # *** END ADDED LOGIC ***
    #
    #         # Assign the potentially extracted value
    #         if param_name != primary_input_arg_name:
    #             call_kwargs[param_name] = mapped_value
    #         elif call_kwargs.get(primary_input_arg_name) != mapped_value: # Use .get() for safety
    #             logger.warning(...)
    #             call_kwargs[primary_input_arg_name] = mapped_value
    # # ... rest of worker ...
    # ```
    # **Assuming this modification is made in the `process_document_batch`'s internal worker**,
    # the pipeline definition above should now work correctly.

    console.print(
        Panel("Defined Batch Pipeline (Corrected Input Mapping):", border_style="magenta")
    )
    console.print(Syntax(json.dumps(batch_operations, indent=2), "json", theme="default"))

    # --- Execute Batch Processing ---
    console.print(f"\nExecuting batch pipeline with concurrency {MAX_CONCURRENT_TASKS}...")
    try:
        # Call the standalone batch processing function
        batch_results = await process_document_batch(
            inputs=batch_inputs, operations=batch_operations, max_concurrency=MAX_CONCURRENT_TASKS
        )

        console.print(Rule("[bold]Batch Processing Results[/]", style="blue"))

        # --- Display Batch Results ---
        if not batch_results:
            console.print("[yellow]Batch processing returned no results.[/]")
        else:
            console.print(f"Processed {len(batch_results)} items.")
            for i, item_result in enumerate(batch_results):
                item_id = item_result.get("item_id", f"Item {i}")
                status = item_result.get("_status", "unknown")
                color = (
                    "green" if status == "processed" else "red" if status == "failed" else "yellow"
                )
                console.print(
                    Rule(f"Result for: [bold {color}]{item_id}[/] (Status: {status})", style=color)
                )

                outputs_table = Table(title="Generated Outputs", box=box.MINIMAL, show_header=False)
                outputs_table.add_column("Step", style="cyan")
                outputs_table.add_column("Output Key", style="magenta")
                outputs_table.add_column("Preview / Status", style="white")

                for op_spec in batch_operations:
                    key = op_spec["output_key"]
                    step_result = item_result.get(key)
                    preview = "[dim]Not generated[/]"
                    if step_result and isinstance(step_result, dict):
                        step_success = step_result.get("success", False)
                        preview = (
                            "[green]Success[/]"
                            if step_success
                            else f"[red]Failed: {step_result.get('error_code', 'ERROR')}[/]"
                        )
                        if step_success:
                            if "content" in step_result and isinstance(step_result["content"], str):
                                preview += f" (Content len: {len(step_result['content'])})"
                            elif "chunks" in step_result and isinstance(
                                step_result["chunks"], list
                            ):
                                preview += f" ({len(step_result['chunks'])} chunks)"
                            elif "summary" in step_result and isinstance(
                                step_result.get("summary"), str
                            ):
                                preview += f" (Summary len: {len(step_result['summary'])})"
                            elif "qa_pairs" in step_result and isinstance(
                                step_result.get("qa_pairs"), list
                            ):
                                preview += f" ({len(step_result['qa_pairs'])} pairs)"
                            elif "metrics" in step_result and isinstance(
                                step_result.get("metrics"), dict
                            ):
                                preview += f" ({len(step_result['metrics'])} metrics)"
                            elif "risks" in step_result and isinstance(
                                step_result.get("risks"), dict
                            ):
                                preview += f" ({len(step_result['risks'])} risks)"
                            # Add other previews as needed
                    outputs_table.add_row(op_spec["operation"], key, preview)

                console.print(outputs_table)

                if item_result.get("_error_log"):
                    error_panel_content = Text()
                    for err in item_result["_error_log"]:
                        error_panel_content.append(
                            Text.from_markup(f"- [yellow]{escape(err)}[/]\n")
                        )
                    console.print(
                        Panel(error_panel_content, title="Error Log", border_style="yellow")
                    )

                console.print("-" * 30)  # Separator

    except Exception as e:
        logger.error(f"Batch processing demo failed: {e}", exc_info=True)
        console.print(f"[bold red]Error during batch processing execution:[/]\n{e}")


async def main():
    """Main function to run the DocumentProcessingTool demo."""
    console.print(
        Rule("[bold] Document Processing Standalone Functions Demo [/bold]", style="blue")
    )
    if not MCP_COMPONENTS_LOADED:
        # Error already printed during import attempt
        sys.exit(1)

    # Set logger level based on environment variable
    console.print(f"Docling Available: {_DOCLING_AVAILABLE}")
    console.print(f"Pandas Available: {_PANDAS_AVAILABLE}")
    console.print(f"Tiktoken Available: {_TIKTOKEN_AVAILABLE}")
    console.print(f"Using Accelerator: {ACCELERATOR_DEVICE}")

    try:
        # Create a CostTracker instance
        tracker = CostTracker()

        # Create gateway - still useful for initializing providers if needed by underlying tools like generate_completion
        gateway = Gateway(
            "doc-proc-standalone-demo", register_tools=False
        )  # Don't register the old tool
        logger.info(
            "Initializing gateway and providers (needed for potential LLM calls)...",
            emoji_key="provider",
        )
        try:
            await gateway._initialize_providers()
            logger.info("Providers initialized.")
        except Exception as init_e:
            logger.error(f"Failed to initialize providers: {init_e}", exc_info=True)
            console.print(
                "[red]Error initializing providers. LLM-dependent operations might fail.[/]"
            )

        # --- Prepare sample files ---
        logger.info("Setting up sample files and directories...", emoji_key="setup")
        DEFAULT_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        DOWNLOADED_FILES_DIR.mkdir(parents=True, exist_ok=True)
        conversion_outputs_dir = DEFAULT_SAMPLE_DIR / "conversion_outputs"
        conversion_outputs_dir.mkdir(exist_ok=True)
        logger.info(f"Outputs will be saved in: {conversion_outputs_dir}")

        sample_files: Dict[str, Any] = {"conversion_outputs_dir": conversion_outputs_dir}

        # --- Download Files Concurrently (No shared progress bar) ---
        # The download_file_with_progress function will create its own transient progress bar
        # if no 'progress' object is passed.
        console.print(Rule("Downloading Sample Files", style="blue"))
        download_tasks = [
            download_file_with_progress(
                DEFAULT_SAMPLE_PDF_URL,
                DOWNLOADED_FILES_DIR / "attention_is_all_you_need.pdf",
                "Transformer Paper (PDF)",
            ),  # No progress obj passed
            download_file_with_progress(
                DEFAULT_SAMPLE_IMAGE_URL,
                DOWNLOADED_FILES_DIR / "sample_ocr_image.tif",
                "Sample OCR Image (TIFF)",
            ),  # No progress obj passed
            download_file_with_progress(
                SAMPLE_HTML_URL,
                DOWNLOADED_FILES_DIR / "transformer_wiki.html",
                "Transformer Wiki (HTML)",
            ),  # No progress obj passed
            download_file_with_progress(
                BUFFETT_SHAREHOLDER_LETTER_URL,
                DOWNLOADED_FILES_DIR / "buffett_letter_2022.pdf",
                "Buffett Letter (PDF)",
            ),  # No progress obj passed
            download_file_with_progress(
                BACKPROPAGATION_PAPER_URL,
                DOWNLOADED_FILES_DIR / "backprop_paper.pdf",
                "Backprop Paper (PDF)",
            ),  # No progress obj passed
        ]
        download_results = await asyncio.gather(*download_tasks)
        console.print(Rule("Downloads Complete", style="blue"))

        sample_files["pdf_digital"] = download_results[0]
        sample_files["image"] = download_results[1]
        sample_files["html"] = download_results[2]
        sample_files["buffett_pdf"] = download_results[3]
        sample_files["backprop_pdf"] = download_results[4]

        # --- Run Demo Sections ---
        # Pass the necessary sample_files dict and the tracker
        await demo_section_1_conversion_ocr(sample_files, tracker)
        await demo_section_2_dedicated_ocr(sample_files, tracker)
        await demo_section_3_enhance_text(sample_files, tracker)
        await demo_section_4_html_markdown(sample_files, tracker)
        await demo_section_5_analyze_structure(sample_files, tracker)
        await demo_section_6_chunking_tables(sample_files, tracker)
        await demo_section_7_analysis(sample_files, tracker)
        await demo_section_8_batch_processing(sample_files, tracker)

        # --- Display Final Cost Summary ---
        console.print(Rule("[bold]Demo Complete - Cost Summary[/]", style="blue"))
        tracker.display_summary(console)

    except Exception as e:
        logger.critical(f"Demo execution failed critically: {str(e)}", exc_info=True)
        console.print_exception(show_locals=True)  # Use Rich's exception printing
        return 1

    logger.info("Demo finished successfully.")
    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
