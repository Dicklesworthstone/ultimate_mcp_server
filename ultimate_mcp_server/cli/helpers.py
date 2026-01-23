"""Helper functions for the Ultimate MCP Server CLI."""

import json
import sys
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ultimate_mcp_server.config import get_env
from ultimate_mcp_server.constants import COST_PER_MILLION_TOKENS, Provider
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)
console = Console(file=sys.stderr)


def print_cost_table() -> None:
    """Display pricing information for all supported LLM models.

    This function creates and prints a formatted table showing the cost per million tokens
    for various LLM models across all supported providers (OpenAI, Anthropic, DeepSeek, etc.).
    The table separates input token costs from output token costs, as these are typically
    billed at different rates.

    Models are grouped by provider and sorted alphabetically for easy reference.
    This information is useful for cost planning, provider comparison, and
    understanding the financial implications of different model choices.
    """
    # Create table
    table = Table(title="Model Cost Per Million Tokens")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="blue")
    table.add_column("Input ($/M)", style="green")
    table.add_column("Output ($/M)", style="yellow")

    # Group models by provider
    models_by_provider = {}
    for model, costs in COST_PER_MILLION_TOKENS.items():
        # Determine provider
        provider = None
        if "gpt" in model:
            provider = Provider.OPENAI.value
        elif "claude" in model:
            provider = Provider.ANTHROPIC.value
        elif "deepseek" in model:
            provider = Provider.DEEPSEEK.value
        elif "gemini" in model:
            provider = Provider.GEMINI.value
        else:
            provider = "other"

        if provider not in models_by_provider:
            models_by_provider[provider] = []

        models_by_provider[provider].append((model, costs))

    # Add rows for each provider's models
    for provider in sorted(models_by_provider.keys()):
        models = sorted(models_by_provider[provider], key=lambda x: x[0])

        for model, costs in models:
            table.add_row(provider, model, f"${costs['input']:.3f}", f"${costs['output']:.3f}")

    # Print table
    console.print(table)


def format_tokens(tokens: int) -> str:
    """Format token count with thousands separator for better readability.

    Converts raw token counts (e.g., 1234567) into a more human-readable format
    with commas as thousand separators (e.g., "1,234,567"). This improves
    the readability of token usage statistics in CLI outputs and reports.

    Args:
        tokens: Raw token count as an integer

    Returns:
        Formatted string with thousand separators (e.g., "1,234,567")
    """
    return f"{tokens:,}"


def format_duration(seconds: float) -> str:
    """Format time duration in a human-friendly, adaptive format.

    Converts raw seconds into a more readable format, automatically selecting
    the appropriate unit based on the magnitude:
    - Milliseconds for durations under 0.1 seconds
    - Seconds with decimal precision for durations under 60 seconds
    - Minutes and seconds for longer durations

    This provides intuitive time displays in benchmarks and performance reports.

    Args:
        seconds: Duration in seconds (can be fractional)

    Returns:
        Formatted string like "50ms", "2.45s", or "1m 30.5s" depending on duration
    """
    if seconds < 0.1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def save_output_to_file(text: str, file_path: str, mode: str = "w") -> bool:
    """Write text content to a file with error handling and user feedback.

    This utility function safely writes text to a file, handling encoding
    and providing user feedback on success or failure. It's commonly used
    to save LLM outputs, generated code, or other text data for later use.

    Args:
        text: The string content to write to the file
        file_path: Target file path (absolute or relative to current directory)
        mode: File open mode - "w" for overwrite or "a" for append to existing content

    Returns:
        Boolean indicating success (True) or failure (False)
    """
    try:
        with open(file_path, mode, encoding="utf-8") as f:
            f.write(text)

        console.print(f"[green]Output saved to {file_path}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error saving output: {str(e)}[/red]")
        return False


def load_file_content(file_path: str) -> Optional[str]:
    """Read and return the entire contents of a text file.

    This utility function safely reads text from a file with proper UTF-8 encoding,
    handling any errors that may occur during the process. It's useful for loading
    prompts, templates, or other text files needed for LLM operations.

    Args:
        file_path: Path to the file to read (absolute or relative to current directory)

    Returns:
        The file's contents as a string if successful, or None if an error occurred
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        console.print(f"[red]Error loading file: {str(e)}[/red]")
        return None


def print_markdown(markdown_text: str) -> None:
    """Display Markdown content with proper formatting and styling.

    Renders Markdown text with appropriate styling (headings, bold, italic,
    lists, code blocks, etc.) in the terminal using Rich's Markdown renderer.
    This provides a more readable and visually appealing output for
    documentation, examples, or LLM responses that use Markdown formatting.

    Args:
        markdown_text: Raw Markdown-formatted text to render
    """
    md = Markdown(markdown_text)
    console.print(md)


def print_json(json_data: Union[Dict, List]) -> None:
    """Display JSON data with syntax highlighting and proper formatting.

    Converts a Python dictionary or list into a properly indented JSON string
    and displays it with syntax highlighting for improved readability.
    This is useful for displaying API responses, configuration data,
    or other structured data in a human-friendly format.

    Args:
        json_data: Python dictionary or list to be displayed as formatted JSON
    """
    json_str = json.dumps(json_data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)
    console.print(syntax)


def print_code(code: str, language: str = "python") -> None:
    """Display source code with syntax highlighting and line numbers.

    Renders code with proper syntax highlighting based on the specified language,
    along with line numbers for easier reference. This improves readability
    when displaying code examples, LLM-generated code, or code snippets
    from files.

    Args:
        code: Source code text to display
        language: Programming language for syntax highlighting (e.g., "python",
                 "javascript", "rust", "sql", etc.)
    """
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)


def print_model_comparison(provider: str, models: List[str], metrics: List[Dict[str, Any]]) -> None:
    """Display a side-by-side comparison of multiple models from the same provider.

    Creates a formatted table comparing performance metrics for different models
    from the same LLM provider. This is useful for identifying the optimal model
    for specific use cases based on response time, throughput, and cost metrics.

    The comparison includes:
    - Response time (formatted appropriately for the magnitude)
    - Processing speed (tokens per second)
    - Cost per request
    - Total token usage

    Args:
        provider: Name of the LLM provider (e.g., "openai", "anthropic")
        models: List of model identifiers to compare
        metrics: List of dictionaries containing performance metrics for each model,
                with keys like "time", "tokens_per_second", "cost", "total_tokens"
    """
    # Create table
    table = Table(title=f"{provider.capitalize()} Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Response Time", style="green")
    table.add_column("Tokens/Sec", style="yellow")
    table.add_column("Cost", style="magenta")
    table.add_column("Total Tokens", style="dim")

    # Add rows for each model
    for model, metric in zip(models, metrics, strict=False):
        table.add_row(
            model,
            format_duration(metric.get("time", 0)),
            f"{metric.get('tokens_per_second', 0):.1f}",
            f"${metric.get('cost', 0):.6f}",
            format_tokens(metric.get("total_tokens", 0)),
        )

    # Print table
    console.print(table)


def print_environment_info() -> None:
    """Display current environment configuration for diagnostics.

    Creates a formatted table showing important environment variables and their
    current values, with a focus on API keys, logging configuration, and cache settings.
    This is useful for troubleshooting and verifying that the environment is
    configured correctly before running the server or other commands.

    The output includes:
    - Status of API keys for each supported provider (set or not set)
    - Logging level configuration
    - Cache settings
    - Other relevant environment variables
    """
    # Create table
    table = Table(title="Environment Information")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Add API key info
    for provider in [p.value for p in Provider]:
        env_var = f"{provider.upper()}_API_KEY"
        has_key = bool(get_env(env_var))
        table.add_row(env_var, "✅ Set" if has_key else "❌ Not set")

    # Add other environment variables
    for var in ["LOG_LEVEL", "CACHE_ENABLED", "CACHE_DIR"]:
        value = get_env(var, "Not set")
        table.add_row(var, value)

    # Print table
    console.print(table)


def print_examples() -> None:
    """Display common usage examples for the CLI commands.

    Shows a set of syntax-highlighted example commands demonstrating how to use
    the most common features of the Ultimate MCP Server CLI. This helps users
    quickly learn the command patterns and options available without having to
    consult the full documentation.

    Examples cover:
    - Starting the server
    - Listing and testing providers
    - Generating completions (with and without streaming)
    - Running benchmarks
    - Managing the cache
    """
    examples = """
# Run the server
ultimate-mcp-server run --host 0.0.0.0 --port 8013

# List available providers
ultimate-mcp-server providers --check

# Test a provider
ultimate-mcp-server test openai --model gpt-4.1-mini --prompt "Hello, world!"

# Generate a completion
ultimate-mcp-server complete --provider anthropic --model claude-3-5-haiku-20241022 --prompt "Explain quantum computing"

# Stream a completion
ultimate-mcp-server complete --provider openai --stream --prompt "Write a poem about AI"

# Run benchmarks
ultimate-mcp-server benchmark --providers openai anthropic --runs 3

# Check cache status
ultimate-mcp-server cache --status

# Clear cache
ultimate-mcp-server cache --clear
"""

    syntax = Syntax(examples, "bash", theme="monokai", word_wrap=True)
    console.print(Panel(syntax, title="CLI Examples", border_style="cyan"))


def confirm_action(message: str, default: bool = False) -> bool:
    """Prompt the user for confirmation before performing a potentially destructive action.

    Displays a yes/no prompt with the specified message and waits for user input.
    This is used to confirm potentially destructive operations like clearing the cache
    or deleting files to prevent accidental data loss.

    Args:
        message: The question or confirmation message to display to the user
        default: The default response if the user just presses Enter without typing
                 anything (True for yes, False for no)

    Returns:
        Boolean indicating whether the user confirmed (True) or canceled (False) the action
    """
    default_str = "Y/n" if default else "y/N"
    response = input(f"{message} [{default_str}]: ")

    if not response:
        return default

    return response.lower() in ["y", "yes"]
