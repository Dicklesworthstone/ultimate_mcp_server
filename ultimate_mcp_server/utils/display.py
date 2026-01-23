"""Display utilities for the Ultimate MCP Server.

This module contains reusable display functions for formatting and
presenting results from Ultimate MCP Server operations using Rich.
"""

import json
import os
import time

# --- Filesystem Tool Display Helper ---
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich import box
from rich.console import (
    Capture,  # For capturing table output
    Console,
)
from rich.markup import escape
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

# Import the console for consistent styling
from ultimate_mcp_server.utils.logging.console import console

from ..exceptions import ToolError, ToolInputError

# Restore direct tool import
from ..tools.filesystem import list_directory
from .logging.logger import get_logger  # <-- ADD specific import

try:
    # Import only the exception needed by safe_tool_call
    from ..tools.filesystem import ProtectionTriggeredError
except ImportError:
    # Handle case where filesystem tools might not be installed/available
    # Define a dummy exception class if ProtectionTriggeredError cannot be imported
    class ProtectionTriggeredError(Exception):
        def __init__(self, message, context=None):
            super().__init__(message)
            self.context = context if context is not None else {}


def extract_and_parse_content(result: Any) -> Dict[str, Any]:
    """
    Extract content from various result formats and parse JSON if present.
    This handles TextContent objects, lists of TextContent, and other formats.

    Args:
        result: Result object that might be TextContent, list, dict, etc.

    Returns:
        Dictionary with parsed data or error information
    """
    # Handle list of objects (common in MCP responses)
    if isinstance(result, list):
        if not result:
            return {"error": "Empty result list"}
        # Just use the first item for now (we could process all in the future)
        result = result[0]

    # Extract text from TextContent object
    text_content = ""
    if hasattr(result, "text"):
        text_content = result.text
    elif isinstance(result, str):
        text_content = result
    elif isinstance(result, dict):
        return result  # Already a dict, no need to parse
    else:
        # Convert other types to string representation
        text_content = str(result)

    # Try to parse as JSON
    if text_content:
        try:
            parsed_data = json.loads(text_content)
            return parsed_data
        except json.JSONDecodeError:
            # Not JSON, return as raw text
            return {"raw_text": text_content, "error": "Not valid JSON"}

    # Empty content
    return {"error": "Empty content"}


def display_text_content_result(
    title: str, result: Any, console_instance: Optional[Console] = None
):
    """
    Display results from TextContent objects more reliably, which is useful for demos.
    This function is more forgiving with different formats and provides better handling
    for TextContent objects that might contain JSON strings.

    Args:
        title: Title to display for this result section
        result: Result object from an Ultimate MCP Server tool call (often a TextContent)
        console_instance: Optional console instance to use (defaults to shared console)
    """
    # Use provided console or default to shared console
    output = console_instance or console

    # Display section title
    output.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))

    # Extract and parse content
    parsed_data = extract_and_parse_content(result)

    # Check for extraction errors
    if "error" in parsed_data and "raw_text" in parsed_data:
        # Error parsing JSON, display as text
        output.print(
            Panel(
                escape(parsed_data["raw_text"]),
                title="[bold]Result Text[/bold]",
                border_style="green",
            )
        )
        return
    elif "error" in parsed_data and "raw_text" not in parsed_data:
        # Other error
        output.print(f"[red]{escape(parsed_data['error'])}[/red]")
        return

    # Display based on content type
    if isinstance(parsed_data, dict):
        # Special handling for QA pairs
        if "qa_pairs" in parsed_data and isinstance(parsed_data["qa_pairs"], list):
            qa_pairs = parsed_data["qa_pairs"]
            output.print(
                Panel(
                    "\n".join(
                        [
                            f"[bold]Q{i + 1}:[/bold] {escape(pair.get('question', 'N/A'))}\n[bold]A{i + 1}:[/bold] {escape(pair.get('answer', 'N/A'))}"
                            for i, pair in enumerate(qa_pairs)
                        ]
                    ),
                    title="[bold]Q&A Pairs[/bold]",
                    border_style="blue",
                )
            )
        # Special handling for entities
        elif "entities" in parsed_data:
            entities_data = parsed_data["entities"]
            if isinstance(entities_data, dict):
                # If it's a dict with entity types as keys
                entity_count = 0
                entity_table = Table(box=box.ROUNDED)
                entity_table.add_column("Type", style="cyan")
                entity_table.add_column("Entity", style="white")

                for entity_type, entities in entities_data.items():
                    if entities:
                        for entity in entities:
                            entity_text = (
                                entity
                                if isinstance(entity, str)
                                else entity.get("text", str(entity))
                            )
                            entity_table.add_row(entity_type, escape(entity_text))
                            entity_count += 1

                if entity_count > 0:
                    output.print(entity_table)
                else:
                    output.print("[yellow]No entities found in the document.[/yellow]")
            else:
                # If it's some other format, just show the raw data
                output.print(
                    Panel(
                        escape(json.dumps(entities_data, indent=2)),
                        title="[bold]Entities Data[/bold]",
                        border_style="blue",
                    )
                )
        # Summary
        elif "summary" in parsed_data and isinstance(parsed_data["summary"], str):
            output.print(
                Panel(
                    escape(parsed_data["summary"]),
                    title="[bold]Generated Summary[/bold]",
                    border_style="green",
                )
            )
        # Generic JSON display for other data
        else:
            # Filter out stats fields for cleaner display
            display_data = {
                k: v
                for k, v in parsed_data.items()
                if k not in ["model", "provider", "cost", "tokens", "processing_time"]
            }

            # Only show JSON panel if we have data to display
            if display_data:
                output.print(
                    Panel(
                        escape(json.dumps(display_data, indent=2)),
                        title="[bold]Result Data[/bold]",
                        border_style="blue",
                    )
                )

        # Display stats if available
        if any(
            k in parsed_data for k in ["model", "provider", "cost", "tokens", "processing_time"]
        ):
            _display_stats(parsed_data, output)
    else:
        # For other types (arrays, etc.)
        output.print(
            Panel(
                escape(json.dumps(parsed_data, indent=2)),
                title="[bold]Result Data[/bold]",
                border_style="blue",
            )
        )


def _display_input_data(input_data: Dict, output: Console):
    """
    Display input data with consistent formatting.

    This function formats and displays various types of input data using the Rich
    console library. It handles text content, JSON schemas, search queries, and
    embedding vectors, adjusting the display format appropriately for each type.

    Args:
        input_data: Dictionary containing input data to display. May include keys
                   like 'text', 'json_schema', 'query', and 'embeddings'.
        output: Rich Console instance to use for printing formatted output.
    """
    # Display input text if available
    if "text" in input_data:
        text_snippet = input_data["text"][:500] + ("..." if len(input_data["text"]) > 500 else "")
        output.print(
            Panel(
                escape(text_snippet),
                title="[cyan]Input Text Snippet[/cyan]",
                border_style="dim blue",
            )
        )

    # Display schema if available
    if "json_schema" in input_data and input_data["json_schema"]:
        try:
            schema_json = json.dumps(input_data["json_schema"], indent=2)
            output.print(
                Panel(
                    Syntax(schema_json, "json", theme="default", line_numbers=False),
                    title="[cyan]Input Schema[/cyan]",
                    border_style="dim blue",
                )
            )
        except Exception as e:
            output.print(f"[red]Could not display schema: {escape(str(e))}[/red]")

    # Display query if available (for search results)
    if "query" in input_data:
        output.print(
            Panel(
                escape(input_data["query"]),
                title="[cyan]Search Query[/cyan]",
                border_style="dim blue",
            )
        )

    # Display embeddings/vectors if available
    if "embeddings" in input_data:
        if isinstance(input_data["embeddings"], list) and len(input_data["embeddings"]) > 0:
            sample = input_data["embeddings"][0]
            dims = len(sample) if isinstance(sample, (list, tuple)) else "unknown"
            sample_str = (
                str(sample[:3]) + "..." if isinstance(sample, (list, tuple)) else str(sample)
            )
            output.print(
                Panel(
                    f"[cyan]Dimensions:[/cyan] {dims}\n[cyan]Sample:[/cyan] {escape(sample_str)}",
                    title="[cyan]Embedding Sample[/cyan]",
                    border_style="dim blue",
                )
            )


def _parse_and_display_output(result: Any, output: Console):
    """
    Parse result object and display appropriate visualizations.

    This function examines the structure of a result object and determines the best
    way to display it based on its content type. It automatically extracts and formats
    different data types like JSON data, vector search results, tables, key-value pairs,
    entity data, and embeddings.

    The function serves as an intelligent formatter that routes different content types
    to specialized display handlers that can present each type with appropriate
    rich formatting and visualization.

    Args:
        result: The result object to parse and display. Can be a list, dict, object
               with a 'text' attribute, or other structures.
        output: Rich Console instance to use for displaying the formatted content.

    Note:
        This is an internal utility function used by higher-level display functions
        to handle the details of content extraction and formatting.
    """
    # Extract result content
    parsed_result = {}
    raw_text = None

    # Handle list results (take first item)
    if isinstance(result, list) and result:
        result = result[0]

    # Handle object with text attribute
    if hasattr(result, "text"):
        raw_text = result.text
        try:
            parsed_result = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed_result = {"error": "Failed to parse JSON", "raw_text": raw_text}

    # Handle dictionary result
    elif isinstance(result, dict):
        parsed_result = result

    # Handle unknown result type
    else:
        parsed_result = {"error": f"Unexpected result type: {type(result)}"}

    # Display results based on content
    _display_result_content(parsed_result, output)


def _display_result_content(parsed_result: Dict, output: Console):
    """
    Display the content of results with appropriate formatting.

    This function intelligently selects appropriate display handlers for different
    types of result content. It checks for various data types (JSON data, vector search
    results, tables, key-value pairs, entities, embeddings, etc.) and routes the content
    to specialized display functions.

    Args:
        parsed_result: Dictionary containing parsed result data with various possible
                      content types to display.
        output: Rich Console instance to use for printing formatted output.
    """
    # Check for errors first
    if parsed_result.get("error"):
        _display_error(parsed_result, output)
        return

    # Display different result types

    # JSON Data
    if "data" in parsed_result and parsed_result["data"] is not None:
        _display_json_data(parsed_result["data"], "Extracted JSON Data", output)

    # Vector Search Results
    if "results" in parsed_result and isinstance(parsed_result["results"], list):
        _display_vector_results(parsed_result["results"], output)

    # Tables
    if "tables" in parsed_result and parsed_result["tables"]:
        _display_tables(parsed_result["tables"], output)

    # Key-Value Pairs
    if "key_value_pairs" in parsed_result or "pairs" in parsed_result:
        pairs = parsed_result.get("key_value_pairs", parsed_result.get("pairs", {}))
        _display_key_value_pairs(pairs, output)

    # Semantic Schema
    if "schema" in parsed_result and parsed_result["schema"]:
        _display_json_data(parsed_result["schema"], "Inferred Semantic Schema", output)

    # Entities
    if "entities" in parsed_result and parsed_result["entities"]:
        _display_entities(parsed_result["entities"], output)

    # Embeddings
    if "embeddings" in parsed_result and parsed_result["embeddings"]:
        _display_embeddings_info(
            parsed_result["embeddings"], parsed_result.get("model", "unknown"), output
        )

    # Display execution stats if available
    _display_stats(parsed_result, output)


def _display_error(result: Dict, output: Console):
    """
    Display error information.

    This function formats and displays error information in a visually distinct way.
    It creates a red-bordered panel containing the error message and optional raw text
    output for debugging purposes.

    Args:
        result: Dictionary containing error information. Should include an 'error' key
               and optionally a 'raw_text' key with the original output.
        output: Rich Console instance to use for printing formatted output.
    """
    error_content = f"[red]Error:[/red] {escape(result['error'])}"
    if result.get("raw_text"):
        error_content += f"\n\n[yellow]Raw Text Output:[/yellow]\n{escape(result['raw_text'])}"
    output.print(Panel(error_content, title="[bold red]Tool Error[/bold red]", border_style="red"))


def _display_json_data(data: Any, title: str, output: Console):
    """
    Display JSON data with proper formatting.

    This function formats and displays JSON data with syntax highlighting and proper
    indentation. It handles JSON serialization errors gracefully and displays the data
    in a visually appealing panel with a descriptive title.

    Args:
        data: Any data structure that can be serialized to JSON.
        title: Title string to display above the JSON content.
        output: Rich Console instance to use for printing formatted output.
    """
    try:
        data_json = json.dumps(data, indent=2)
        output.print(
            Panel(
                Syntax(data_json, "json", theme="default", line_numbers=True, word_wrap=True),
                title=f"[bold green]{title}[/bold green]",
                border_style="green",
            )
        )
    except Exception as e:
        output.print(f"[red]Could not display JSON data: {escape(str(e))}[/red]")


def _display_vector_results(results: List[Dict], output: Console):
    """
    Display vector search results.

    This function creates and displays a formatted table showing vector search results,
    including IDs, similarity scores, metadata, and text snippets. It automatically
    adapts the table columns based on the structure of the first result item, handling
    various metadata fields dynamically.

    Args:
        results: List of dictionaries containing vector search results. Each dictionary
                typically includes 'id', 'similarity' or 'score', optional 'metadata',
                and 'text' fields.
        output: Rich Console instance to use for printing formatted output.
    """
    results_table = Table(title="[bold green]Vector Search Results[/bold green]", box=box.ROUNDED)

    # Determine columns based on first result
    if not results:
        output.print("[yellow]No vector search results to display[/yellow]")
        return

    first_result = results[0]

    # Add standard columns
    results_table.add_column("ID", style="cyan")
    results_table.add_column("Score", style="green", justify="right")

    # Add metadata columns if available
    metadata_keys = []
    if "metadata" in first_result and isinstance(first_result["metadata"], dict):
        metadata_keys = list(first_result["metadata"].keys())
        for key in metadata_keys:
            results_table.add_column(key.capitalize(), style="magenta")

    # Add text column
    results_table.add_column("Text", style="white")

    # Add rows
    for item in results:
        row = [
            escape(str(item.get("id", ""))),
            f"{item.get('similarity', item.get('score', 0.0)):.4f}",
        ]

        # Add metadata values
        if metadata_keys:
            metadata = item.get("metadata", {})
            for key in metadata_keys:
                row.append(escape(str(metadata.get(key, ""))))

        # Add text
        text = item.get("text", "")
        text_snippet = text[:80] + ("..." if len(text) > 80 else "")
        row.append(escape(text_snippet))

        results_table.add_row(*row)

    output.print(results_table)


def _display_tables(tables: List[Dict], output: Console):
    """
    Display extracted tables.

    This function formats and displays extracted table data in multiple formats
    (JSON, Markdown) with appropriate syntax highlighting. It includes table titles
    and associated metadata when available.

    Args:
        tables: List of dictionaries containing table information. Each dictionary may
               include 'title', 'json', 'markdown', and 'metadata' fields.
        output: Rich Console instance to use for printing formatted output.
    """
    for i, table_info in enumerate(tables):
        table_title = table_info.get("title", f"Table {i + 1}")
        output.print(Rule(f"[green]Extracted: {escape(table_title)}[/green]"))

        # JSON format
        if table_info.get("json"):
            try:
                table_json = json.dumps(table_info["json"], indent=2)
                output.print(
                    Panel(
                        Syntax(
                            table_json, "json", theme="default", line_numbers=False, word_wrap=True
                        ),
                        title="[bold]JSON Format[/bold]",
                        border_style="dim green",
                    )
                )
            except Exception as e:
                output.print(f"[red]Could not display table JSON: {escape(str(e))}[/red]")

        # Markdown format
        if table_info.get("markdown"):
            output.print(
                Panel(
                    Syntax(table_info["markdown"], "markdown", theme="default"),
                    title="[bold]Markdown Format[/bold]",
                    border_style="dim green",
                )
            )

        # Metadata
        if table_info.get("metadata"):
            try:
                meta_json = json.dumps(table_info["metadata"], indent=2)
                output.print(
                    Panel(
                        Syntax(meta_json, "json", theme="default", line_numbers=False),
                        title="[bold]Metadata[/bold]",
                        border_style="dim green",
                    )
                )
            except Exception as e:
                output.print(f"[red]Could not display metadata: {escape(str(e))}[/red]")


def _display_key_value_pairs(pairs: Union[Dict, List], output: Console):
    """
    Display key-value pairs in a table.

    This function creates and displays a formatted table showing key-value pairs.
    It handles both dictionary and list inputs, adapting the display format
    appropriately for each case.

    Args:
        pairs: Dictionary of key-value pairs or list of dictionaries containing
              key-value pairs to display.
        output: Rich Console instance to use for printing formatted output.
    """
    kv_table = Table(title="[bold green]Extracted Key-Value Pairs[/bold green]", box=box.ROUNDED)
    kv_table.add_column("Key", style="magenta")
    kv_table.add_column("Value", style="white")

    if isinstance(pairs, dict):
        for k, v in pairs.items():
            kv_table.add_row(escape(str(k)), escape(str(v)))
    elif isinstance(pairs, list):
        for item in pairs:
            if isinstance(item, dict):
                for k, v in item.items():
                    kv_table.add_row(escape(str(k)), escape(str(v)))

    if kv_table.row_count > 0:
        output.print(kv_table)


def _display_entities(entities: List[Dict], output: Console):
    """
    Display extracted entities.

    This function creates and displays a formatted table showing extracted entities,
    including their type, text, context snippet, and confidence score. It's optimized
    for displaying named entity recognition (NER) results.

    Args:
        entities: List of dictionaries containing entity information. Each dictionary
                 typically includes 'type', 'text', 'context', and 'score' fields.
        output: Rich Console instance to use for printing formatted output.
    """
    entity_table = Table(title="[bold green]Extracted Entities[/bold green]", box=box.ROUNDED)
    entity_table.add_column("Type", style="cyan")
    entity_table.add_column("Text", style="white")
    entity_table.add_column("Context", style="dim")
    entity_table.add_column("Score", style="green", justify="right")

    for entity in entities:
        context_snippet = entity.get("context", "")[:50] + (
            "..." if len(entity.get("context", "")) > 50 else ""
        )
        score_str = f"{entity.get('score', 0.0):.2f}" if entity.get("score") is not None else "N/A"

        entity_table.add_row(
            escape(entity.get("type", "N/A")),
            escape(entity.get("text", "N/A")),
            escape(context_snippet),
            score_str,
        )

    output.print(entity_table)


def _display_embeddings_info(embeddings: List, model: str, output: Console):
    """
    Display information about embeddings.

    This function creates and displays a summary table of embedding information,
    including model name, embedding count, dimensions, and sample values. It handles
    edge cases like empty embedding lists and non-numeric embedding values.

    Args:
        embeddings: List of embedding vectors. Each vector is typically a list of
                  floating-point numbers.
        model: Name of the embedding model used to generate the embeddings.
        output: Rich Console instance to use for printing formatted output.
    """
    if not isinstance(embeddings, list) or len(embeddings) == 0:
        return

    # Just display summary info about the embeddings
    sample = embeddings[0]
    dims = len(sample) if isinstance(sample, (list, tuple)) else "unknown"

    embed_table = Table(title="[bold green]Embedding Information[/bold green]", box=box.MINIMAL)
    embed_table.add_column("Property", style="cyan")
    embed_table.add_column("Value", style="white")

    embed_table.add_row("Model", escape(model))
    embed_table.add_row("Count", str(len(embeddings)))
    embed_table.add_row("Dimensions", str(dims))

    # Show a few values from first embedding
    if isinstance(sample, (list, tuple)) and len(sample) > 0:
        sample_values = sample[:3]
        try:
            # Try to round values if they're numeric
            rounded_values = [round(x, 6) for x in sample_values]
            sample_str = str(rounded_values) + "..."
        except (TypeError, ValueError):
            sample_str = str(sample_values) + "..."
        embed_table.add_row("Sample Values", escape(sample_str))

    output.print(embed_table)


def _display_stats(result: Dict, output: Console):
    """
    Display execution statistics.

    This function creates and displays a summary table of execution statistics,
    including provider, model, cost, token usage, and processing time. It only
    displays statistics that are actually present in the input data.

    Args:
        result: Dictionary containing execution statistics. May include keys like
               'provider', 'model', 'cost', 'tokens', and 'processing_time'.
        output: Rich Console instance to use for printing formatted output.
    """
    # Check if we have stats data
    has_stats = any(k in result for k in ["model", "provider", "cost", "tokens", "processing_time"])
    if not has_stats:
        return

    stats_table = Table(title="Execution Stats", box=box.MINIMAL, show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")

    if "provider" in result:
        stats_table.add_row("Provider", escape(result.get("provider", "N/A")))

    if "model" in result:
        stats_table.add_row("Model", escape(result.get("model", "N/A")))

    if "cost" in result:
        stats_table.add_row("Cost", f"${result.get('cost', 0.0):.6f}")

    if "tokens" in result:
        tokens = result.get("tokens", {})
        if isinstance(tokens, dict):
            stats_table.add_row(
                "Tokens (In/Out/Total)",
                f"{tokens.get('input', 0)} / {tokens.get('output', 0)} / {tokens.get('total', 0)}",
            )

    if "processing_time" in result:
        stats_table.add_row("Processing Time", f"{result.get('processing_time', 0.0):.3f}s")

    if stats_table.row_count > 0:
        output.print(stats_table)

    # Add a blank line after stats
    output.print()


# Specialized display functions for different demo types


def display_embedding_generation_results(results_data: Dict, output: Optional[Console] = None):
    """
    Display embedding generation results in a formatted table.

    This function creates a rich, formatted table visualization of embedding generation
    results from multiple models. It organizes and presents key information including
    model names, embedding dimensions, generation times, costs, sample values, and
    success status for each embedding model.

    The visualization is designed to help users compare embedding results across different
    models and providers at a glance, making it easier to evaluate performance, cost,
    and quality differences between embedding options.

    Args:
        results_data: Dictionary containing embedding generation results. Expected to
                     contain a 'models' key with a list of model result dictionaries.
                     Each model result should include fields like 'name', 'dimensions',
                     'time', 'cost', 'embedding_sample', and 'success'.
        output: Optional Rich Console instance to use for display. If not provided,
                uses the default shared console.

    Note:
        If the results_data dictionary doesn't contain a 'models' key or the list is empty,
        the function will display a warning message instead of a table.
    """
    display = output or console

    if not results_data.get("models"):
        display.print("[yellow]No embedding results to display[/yellow]")
        return

    results_table = Table(title="Embedding Generation Results", box=box.ROUNDED, show_header=True)
    results_table.add_column("Model", style="magenta")
    results_table.add_column("Dimensions", style="cyan", justify="right")
    results_table.add_column("Gen Time (s)", style="yellow", justify="right")
    results_table.add_column("Cost ($)", style="green", justify="right")
    results_table.add_column("Sample Values", style="dim")
    results_table.add_column("Status", style="white")

    for model_info in results_data["models"]:
        status_str = "[green]Success[/green]" if model_info.get("success") else "[red]Failed[/red]"

        # Format sample values if available
        sample_str = "N/A"
        if model_info.get("embedding_sample") is not None:
            sample_str = escape(str(model_info["embedding_sample"]) + "...")

        results_table.add_row(
            escape(model_info.get("name", "Unknown")),
            str(model_info.get("dimensions", "-")),
            f"{model_info.get('time', 0.0):.3f}",
            f"{model_info.get('cost', 0.0):.6f}",
            sample_str,
            status_str,
        )

    display.print(results_table)
    display.print()


def display_vector_similarity_results(similarity_data: Dict, output: Optional[Console] = None):
    """
    Display semantic similarity scores between text pairs in a formatted table.

    This function creates a rich, visually appealing table visualization of semantic
    similarity results between text pairs. It extracts and presents information about
    each compared text pair and their corresponding similarity score, making it easy
    to see which text segments are semantically related.

    The table includes columns for text snippets from each pair (truncated if too long)
    and their corresponding similarity score. This visualization is particularly useful
    for comparing multiple text pairs at once and identifying patterns of semantic
    relatedness across a dataset.

    Args:
        similarity_data: Dictionary containing semantic similarity results. Expected
                        to contain a 'pairs' key with a list of comparison result
                        dictionaries. Each pair should include 'text1', 'text2', and
                        'score' fields.
        output: Optional Rich Console instance to use for display. If not provided,
                uses the default shared console.

    Note:
        If the similarity_data dictionary doesn't contain valid pairs data or the list
        is empty, the function will display a warning message instead of a table.
        Similarity scores are displayed with 4 decimal places of precision.
    """
    display = output or console

    pairs = similarity_data.get("pairs", [])
    if not pairs or not isinstance(pairs, list) or len(pairs) == 0:
        display.print("[yellow]No similarity data to display[/yellow]")
        return

    similarity_table = Table(title="Semantic Similarity Scores", box=box.ROUNDED, show_header=True)
    similarity_table.add_column("Text 1 Snippet", style="white")
    similarity_table.add_column("Text 2 Snippet", style="white")
    similarity_table.add_column("Similarity Score", style="green", justify="right")

    for pair in pairs:
        text1 = pair.get("text1", "")[:50] + ("..." if len(pair.get("text1", "")) > 50 else "")
        text2 = pair.get("text2", "")[:50] + ("..." if len(pair.get("text2", "")) > 50 else "")
        score = pair.get("score", 0.0)

        # If score is a numpy array, convert to scalar
        try:
            if hasattr(score, "item"):  # Check if it's potentially a numpy scalar
                score = score.item()
        except (AttributeError, TypeError):
            pass

        similarity_table.add_row(escape(text1), escape(text2), f"{score:.4f}")

    display.print(similarity_table)
    display.print()


def display_analytics_metrics(metrics_data: Dict, output: Optional[Console] = None):
    """
    Display analytics metrics in an attractive format.

    This function takes a dictionary of analytics metrics data and displays it in a
    formatted table using the Rich library. The metrics are grouped by category,
    and each category is displayed as a separate table.

    Args:
        metrics_data: Dictionary containing analytics metrics data
        output: Optional Rich Console instance to use for display. If not provided,
                the default console will be used.
    """
    # Use provided console or default
    output = output or console

    # Check required data
    if not metrics_data or not isinstance(metrics_data, dict):
        output.print("[yellow]No analytics metrics data to display[/yellow]")
        return

    # Display section header
    output.print(Rule("[bold blue]Analytics Metrics[/bold blue]"))

    # Create metrics table
    metrics_table = Table(title="[bold]Metrics Overview[/bold]", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Count", style="green", justify="right")
    metrics_table.add_column("Details", style="dim")

    # Process data
    if "request_counts" in metrics_data:
        for metric, count in metrics_data["request_counts"].items():
            metrics_table.add_row(metric.replace("_", " ").title(), str(count), "")

    # Display table
    output.print(metrics_table)

    # Display any grouped metrics
    if "request_distributions" in metrics_data:
        for group_name, distribution in metrics_data["request_distributions"].items():
            distribution_table = Table(
                title=f"[bold]{group_name.replace('_', ' ').title()} Distribution[/bold]",
                box=box.SIMPLE,
            )
            distribution_table.add_column("Category", style="cyan")
            distribution_table.add_column("Count", style="green", justify="right")
            distribution_table.add_column("Percentage", style="yellow", justify="right")

            total = sum(distribution.values())
            for category, count in distribution.items():
                percentage = (count / total) * 100 if total > 0 else 0
                distribution_table.add_row(category, str(count), f"{percentage:.1f}%")

            output.print(distribution_table)


# --- Tournament Display Functions ---


def display_tournament_status(status_data: Dict[str, Any], output: Optional[Console] = None):
    """
    Display tournament status with better formatting using Rich.

    This function takes a dictionary containing tournament status information
    and displays it in a formatted table using the Rich library. The table
    includes the tournament status, current round, total rounds, progress
    percentage, and timestamps if available.

    Args:
        status_data: Dictionary with tournament status information
        output: Optional console to use (defaults to shared console)
    """
    # Use provided console or default
    display = output or console

    # Extract status information
    status = status_data.get("status", "UNKNOWN")
    current_round = status_data.get("current_round", 0)
    total_rounds = status_data.get("total_rounds", 0)

    # Calculate progress percentage
    if total_rounds > 0:
        progress = (current_round / total_rounds) * 100
    else:
        progress = 0

    # Create status table with improved formatting
    status_table = Table(box=box.SIMPLE, show_header=False, expand=False)
    status_table.add_column("Metric", style="cyan")
    status_table.add_column("Value", style="white")

    # Add status row with color based on status value
    status_color = "green" if status == "COMPLETED" else "yellow" if status == "RUNNING" else "red"
    status_table.add_row("Status", f"[bold {status_color}]{status}[/bold {status_color}]")

    # Add rounds progress
    status_table.add_row("Round", f"{current_round}/{total_rounds}")

    # Add progress percentage
    status_table.add_row("Progress", f"[green]{progress:.1f}%[/green]")

    # Add timestamps if available
    if "created_at" in status_data:
        status_table.add_row(
            "Created", status_data.get("created_at", "N/A").replace("T", " ").split(".")[0]
        )
    if "updated_at" in status_data:
        status_table.add_row(
            "Last Updated", status_data.get("updated_at", "N/A").replace("T", " ").split(".")[0]
        )

    display.print(status_table)

    # Add progress bar visual for better UX
    if status == "RUNNING":
        from rich.progress import BarColumn, Progress, TextColumn

        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )

        with progress_bar:
            task = progress_bar.add_task("Tournament Progress", total=100, completed=progress)  # noqa: F841
            # Just show the bar visualization, don't actually wait/update


def display_tournament_results(results_data: Dict[str, Any], output: Optional[Console] = None):
    """
    Display tournament results with better formatting using Rich.

    This function takes a dictionary containing tournament results and displays
    it in a formatted table using the Rich library. The table includes the
    tournament name, type, final status, total rounds, storage path, models
    used, and execution stats if available.

    Args:
        results_data: Dictionary with tournament results
        output: Optional console to use (defaults to shared console)
    """
    # Use provided console or default
    display = output or console

    # Display section title
    display.print(Rule("[bold blue]Tournament Results[/bold blue]"))

    # Create summary table
    summary_table = Table(
        title="[bold green]Final Results Summary[/bold green]",
        box=box.ROUNDED,
        show_header=False,
        expand=False,
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="white")

    # Add tournament information
    summary_table.add_row(
        "Tournament Name", escape(results_data.get("config", {}).get("name", "N/A"))
    )
    summary_table.add_row(
        "Tournament Type", escape(results_data.get("config", {}).get("tournament_type", "N/A"))
    )
    summary_table.add_row(
        "Final Status", f"[bold green]{escape(results_data.get('status', 'N/A'))}[/bold green]"
    )
    summary_table.add_row("Total Rounds", str(results_data.get("config", {}).get("rounds", "N/A")))

    # Add storage path if available
    storage_path = results_data.get("storage_path")
    summary_table.add_row(
        "Storage Path", escape(storage_path) if storage_path else "[dim]Not available[/dim]"
    )

    # Display summary table
    display.print(summary_table)

    # Display models used in tournament
    models = results_data.get("config", {}).get("models", [])
    if models:
        model_table = Table(title="[bold]Models Used[/bold]", box=box.SIMPLE, show_header=True)
        model_table.add_column("Provider", style="magenta")
        model_table.add_column("Model", style="blue")

        for model_config in models:
            model_id = model_config.get("model_id", "N/A")
            if ":" in model_id:
                provider, model = model_id.split(":", 1)
                model_table.add_row(provider, model)
            else:
                model_table.add_row("Unknown", model_id)

        display.print(model_table)

    # Display execution stats if available
    if any(key in results_data for key in ["processing_time", "cost", "tokens"]):
        _display_stats(results_data, display)


def display_completion_result(console: Console, result: Any, title: str = "Completion Result"):
    """
    Display a completion result with stats.

    This function takes a completion result and displays it in a formatted panel
    using the Rich library. The panel includes the completion text and various
    stats such as input tokens, output tokens, total tokens, cost, and processing
    time if available.

    Args:
        console: Rich console to print to
        result: Completion result to display
        title: Title for the result panel
    """
    # Display the completion text
    console.print(Panel(result.text.strip(), title=title, border_style="green", expand=False))

    # Display stats
    stats_table = Table(title="Completion Stats", show_header=False, box=None)
    stats_table.add_column("Metric", style="green")
    stats_table.add_column("Value", style="white")

    # Add standard metrics if they exist
    if hasattr(result, "input_tokens"):
        stats_table.add_row("Input Tokens", str(result.input_tokens))
    if hasattr(result, "output_tokens"):
        stats_table.add_row("Output Tokens", str(result.output_tokens))
    if hasattr(result, "total_tokens"):
        stats_table.add_row("Total Tokens", str(result.total_tokens))
    if hasattr(result, "cost"):
        stats_table.add_row("Cost", f"${result.cost:.6f}")
    if hasattr(result, "processing_time"):
        stats_table.add_row("Processing Time", f"{result.processing_time:.3f}s")

    console.print(stats_table)


def display_cache_stats(
    stats: Dict[str, Any],
    stats_log: Optional[Dict[int, Dict[str, int]]] = None,
    console: Optional[Console] = None,
):
    """
    Display cache statistics in a formatted table.

    This function takes a dictionary of cache statistics and displays it in a
    formatted table using the Rich library. The table includes information such
    as cache enabled status, persistence, hit rate, total gets, cache hits,
    cache misses, total sets, and estimated savings if available.

    Args:
        stats: Cache statistics dictionary
        stats_log: Optional log of statistics at different stages
        console: Rich console to print to (creates one if None)
    """
    if console is None:
        from ultimate_mcp_server.utils.logging.console import console

    # Create the stats table
    stats_table = Table(title="Cache Statistics", box=box.SIMPLE)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")

    # Add enabled state
    stats_table.add_row(
        "Cache Enabled", "[green]Yes[/green]" if stats.get("enabled", False) else "[red]No[/red]"
    )

    # Add persistence information
    stats_table.add_row(
        "Persistence",
        "[green]Enabled[/green]"
        if stats.get("persistence", False)
        else "[yellow]Disabled[/yellow]",
    )

    # Add hit and miss counts
    cache_stats = stats.get("stats", {})
    stats_table.add_row("Total Gets", str(cache_stats.get("get_count", 0)))
    stats_table.add_row("Cache Hits", str(cache_stats.get("hit_count", 0)))
    stats_table.add_row("Cache Misses", str(cache_stats.get("miss_count", 0)))
    stats_table.add_row("Total Sets", str(cache_stats.get("set_count", 0)))

    # Calculate hit rate
    gets = cache_stats.get("get_count", 0)
    hits = cache_stats.get("hit_count", 0)
    hit_rate = (hits / gets) * 100 if gets > 0 else 0
    stats_table.add_row("Hit Rate", f"{hit_rate:.1f}%")

    # Add estimated savings if available
    if "savings" in stats:
        savings = stats["savings"]
        if isinstance(savings, dict) and "cost" in savings:
            stats_table.add_row("Cost Savings", f"${savings['cost']:.6f}")
        if isinstance(savings, dict) and "time" in savings:
            stats_table.add_row("Time Savings", f"{savings['time']:.3f}s")

    console.print(stats_table)

    # Display changes over time if stats_log is provided
    if stats_log and len(stats_log) > 1:
        changes_table = Table(title="Cache Changes During Demo", box=box.SIMPLE)
        changes_table.add_column("Stage", style="cyan")
        changes_table.add_column("Gets", style="white")
        changes_table.add_column("Hits", style="green")
        changes_table.add_column("Misses", style="yellow")
        changes_table.add_column("Sets", style="blue")

        for stage, stage_stats in sorted(stats_log.items()):
            changes_table.add_row(
                f"Step {stage}",
                str(stage_stats.get("get_count", 0)),
                str(stage_stats.get("hit_count", 0)),
                str(stage_stats.get("miss_count", 0)),
                str(stage_stats.get("set_count", 0)),
            )

        console.print(changes_table)


def parse_and_display_result(
    title: str, input_data: Dict, result: Any, console: Optional[Console] = None
):
    """
    Parse and display extraction results.

    This function takes a title, input data, and extraction result, and displays
    the extracted data in a formatted panel using the Rich library. The function
    supports various extraction formats such as JSON, tables, and entity data.

    Args:
        title: Title for the display
        input_data: Input data used for the extraction
        result: Extraction result
        console: Rich console to print to (creates one if None)
    """
    if console is None:
        from ultimate_mcp_server.utils.logging.console import console

    console.print(Rule(f"[bold blue]{title}[/bold blue]"))

    # Check for errors first
    if "error" in result and result["error"]:
        console.print(f"[bold red]Error:[/bold red] {result['error']}")
        if "raw_text" in result:
            console.print(Panel(result["raw_text"], title="Raw Response", border_style="red"))
        return

    # Display the extracted data based on expected keys for different demos
    extracted_data_displayed = False

    # 1. JSON Extraction (expects 'json' key)
    if "json" in result and isinstance(result["json"], dict):
        data = result["json"]
        json_str = json.dumps(data, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Extracted JSON Data", border_style="green"))
        extracted_data_displayed = True

    # 2. Table Extraction (expects 'formats' and 'metadata')
    elif "formats" in result and isinstance(result["formats"], dict):
        formats = result["formats"]
        if "json" in formats and formats["json"]:
            try:
                _display_json_data(formats["json"], "Extracted Table (JSON)", console)
                extracted_data_displayed = True
            except Exception as e:
                console.print(f"[red]Error displaying table JSON: {e}[/red]")
        if "markdown" in formats and formats["markdown"]:
            try:
                console.print(
                    Panel(
                        Syntax(formats["markdown"], "markdown", theme="default"),
                        title="Extracted Table (Markdown)",
                        border_style="dim green",
                    )
                )
                extracted_data_displayed = True  # Even if JSON fails, MD might succeed
            except Exception as e:
                console.print(f"[red]Error displaying table Markdown: {e}[/red]")
        if "metadata" in result and result["metadata"]:
            try:
                _display_json_data(result["metadata"], "Table Metadata", console)
            except Exception as e:
                console.print(f"[red]Error displaying table metadata: {e}[/red]")

    # 3. Schema Inference / Entity Extraction (expects 'extracted_data')
    elif "extracted_data" in result:
        data = result["extracted_data"]
        # Check if it looks like entity data (dict with list values)
        is_entity_data = False
        if isinstance(data, dict):
            is_entity_data = all(isinstance(v, list) for v in data.values())

        if is_entity_data:
            # Simplified entity display for this function
            entity_table = Table(
                title="[bold green]Extracted Entities[/bold green]", box=box.ROUNDED
            )
            entity_table.add_column("Category", style="cyan")
            entity_table.add_column("Value", style="white")
            for category, items in data.items():
                for item in items:
                    entity_text = (
                        str(item.get("name", item)) if isinstance(item, dict) else str(item)
                    )
                    entity_table.add_row(category, escape(entity_text))
            if entity_table.row_count > 0:
                console.print(entity_table)
                extracted_data_displayed = True
            else:
                console.print("[yellow]No entities found.[/yellow]")
                extracted_data_displayed = True  # Still counts as displayed
        else:
            # Assume other 'extracted_data' is generic JSON
            try:
                _display_json_data(data, "Extracted Data", console)
                extracted_data_displayed = True
            except Exception as e:
                console.print(f"[red]Error displaying extracted data: {e}[/red]")

    # Fallback if no specific keys matched
    if not extracted_data_displayed:
        console.print(
            "[yellow]Could not find expected data keys (json, formats, extracted_data) in result.[/yellow]"
        )
        # Optionally display the whole result as JSON for debugging
        try:
            full_result_json = json.dumps(
                result, indent=2, default=str
            )  # Use default=str for non-serializable items
            console.print(
                Panel(
                    Syntax(full_result_json, "json", theme="monokai", line_numbers=False),
                    title="[dim]Full Result Object[/dim]",
                    border_style="dim",
                )
            )
        except Exception:
            pass  # Ignore if full result can't be serialized

    # Display performance metrics
    if any(k in result for k in ["tokens", "cost", "processing_time"]):
        metrics_table = Table(title="Performance Metrics", box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")

        # Add provider and model info
        if "provider" in result:
            metrics_table.add_row("Provider", result["provider"])
        if "model" in result:
            metrics_table.add_row("Model", result["model"])

        # Add token usage
        if "tokens" in result:
            tokens = result["tokens"]
            if isinstance(tokens, dict):
                for token_type, count in tokens.items():
                    metrics_table.add_row(f"{token_type.title()} Tokens", str(count))
            else:
                metrics_table.add_row("Total Tokens", str(tokens))

        # Add cost and timing
        if "cost" in result:
            metrics_table.add_row("Cost", f"${result['cost']:.6f}")
        if "processing_time" in result:
            metrics_table.add_row("Processing Time", f"{result['processing_time']:.3f}s")

        console.print(metrics_table)


def display_table_data(table_data: List[Dict], console: Console):
    """
    Display tabular data extracted from text.

    This function takes a list of dictionaries representing table rows and
    displays it in a formatted table using the Rich library. The table is also
    displayed as JSON for reference.

    Args:
        table_data: List of dictionaries representing table rows
        console: Rich console to print to
    """
    if not table_data:
        console.print("[yellow]No table data found[/yellow]")
        return

    # Create a Rich table from the data
    rich_table = Table(box=box.SIMPLE)

    # Add columns from the first row's keys
    columns = list(table_data[0].keys())
    for column in columns:
        rich_table.add_column(str(column), style="cyan")

    # Add rows
    for row in table_data:
        rich_table.add_row(*[str(row.get(col, "")) for col in columns])

    console.print(rich_table)

    # Also display as JSON for reference
    json_str = json.dumps(table_data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Table Data (JSON)", border_style="blue"))


def display_key_value_pairs(pairs: List[Dict], console: Console):
    """
    Display key-value pairs extracted from text.

    This function takes a list of dictionaries with 'key' and 'value' fields
    and displays it in a formatted table using the Rich library.

    Args:
        pairs: List of dictionaries with 'key' and 'value' fields
        console: Rich console to print to
    """
    if not pairs:
        console.print("[yellow]No key-value pairs found[/yellow]")
        return

    # Create a Rich table for the key-value pairs
    kv_table = Table(box=None)
    kv_table.add_column("Key", style="green")
    kv_table.add_column("Value", style="white")

    for pair in pairs:
        kv_table.add_row(pair.get("key", ""), pair.get("value", ""))

    console.print(Panel(kv_table, title="Extracted Key-Value Pairs", border_style="green"))


logger = get_logger(__name__)  # Initialize logger for this module


async def safe_tool_call(tool_func, args_dict, description=""):
    """
    Helper function to safely call an async tool function and display results/errors.

    This function wraps an async tool function call and handles common error patterns
    (ToolError, ProtectionTriggeredError, generic exceptions) and formats successful
    outputs for various common tool result structures using Rich.

    Args:
        tool_func: The asynchronous tool function to call.
        args_dict: A dictionary of arguments to pass to the tool function.
        description: A description of the tool call for display purposes.

    Returns:
        A dictionary containing:
        - 'success': Boolean indicating if the call was successful (no errors/protection).
        - 'result': The raw result from the tool function.
        - 'error' (optional): Error message if an error occurred.
        - 'details' (optional): Additional error details.
        - 'protection_triggered' (optional): Boolean, true if deletion protection was triggered.
        - 'context' (optional): Context dictionary from ProtectionTriggeredError.
    """
    tool_name = tool_func.__name__
    call_desc = description or f"Calling [bold magenta]{tool_name}[/bold magenta]"
    # Use pretty_repr for args for better complex type display
    args_str = ", ".join(f"{k}=[yellow]{pretty_repr(v)}[/yellow]" for k, v in args_dict.items())
    console.print(
        Panel(
            f"{call_desc}\nArgs: {args_str}", title="Tool Call", border_style="blue", expand=False
        )
    )

    start_time = time.monotonic()
    try:
        # Directly await the function
        result = await tool_func(**args_dict)
        duration = time.monotonic() - start_time

        # Check for error/protection structure (often returned by @with_error_handling)
        is_error = isinstance(result, dict) and (
            result.get("error") is not None or result.get("isError") is True
        )
        is_protection_triggered = (
            isinstance(result, dict) and result.get("protectionTriggered") is True
        )

        if is_protection_triggered:
            error_msg = result.get("error", "Protection triggered, reason unspecified.")
            context = result.get("details", {}).get("context", {})  # Context might be nested
            console.print(
                Panel(
                    f"[bold yellow]🛡️ Protection Triggered![/bold yellow]\n"
                    f"Message: {escape(error_msg)}\n"
                    f"Context: {pretty_repr(context)}",
                    title=f"Result: {tool_name} (Blocked)",
                    border_style="yellow",
                    subtitle=f"Duration: {duration:.3f}s",
                )
            )
            return {
                "success": False,
                "protection_triggered": True,
                "result": result,
                "error": error_msg,
                "context": context,
            }

        elif is_error:
            error_msg = result.get("error", "Unknown error occurred.")
            error_code = result.get("error_code", "UNKNOWN_ERROR")
            error_type = result.get("error_type", "ERROR")
            details = result.get("details", None)

            logger.debug(f"Error response structure from {tool_name}: {pretty_repr(result)}")

            error_content = f"[bold red]Error ({error_code})[/bold red]\n"
            error_content += f"Type: {error_type}\n"
            error_content += f"Message: {escape(str(error_msg))}"
            if details:
                error_content += f"\nDetails:\n{pretty_repr(details)}"
            else:
                error_content += "\nDetails: N/A"

            console.print(
                Panel(
                    error_content,
                    title=f"Result: {tool_name} (Failed)",
                    border_style="red",
                    subtitle=f"Duration: {duration:.3f}s",
                )
            )
            return {
                "success": False,
                "error": error_msg,
                "details": details,
                "result": result,
                "error_code": error_code,
            }

        else:
            # Successful result - display nicely
            output_content = ""
            if isinstance(result, dict):
                # Common success patterns
                if "message" in result:
                    output_content += f"Message: [green]{escape(result['message'])}[/green]\n"
                if "path" in result:
                    output_content += f"Path: [cyan]{escape(str(result['path']))}[/cyan]\n"
                if "size" in result and not any(
                    k in result for k in ["content", "files"]
                ):  # Avoid printing size if content/files also present
                    # Only print size if it's the primary info (like in write_file result)
                    output_content += f"Size: [yellow]{result['size']}[/yellow] bytes\n"
                if "created" in result and isinstance(result["created"], bool):
                    output_content += (
                        f"Created: {'Yes' if result['created'] else 'No (already existed)'}\n"
                    )
                # Handle 'diff' from edit_file
                if "diff" in result and result.get("diff") not in [
                    "No changes detected after applying edits.",
                    "No edits provided.",
                    None,
                    "",
                ]:
                    diff_content = result["diff"]
                    output_content += f"Diff:\n{Syntax(diff_content, 'diff', theme='monokai', background_color='default')}\n"
                # Handle 'matches' from search_files
                if "matches" in result and "pattern" in result:
                    output_content += f"Search Matches ({len(result['matches'])} for pattern '{result['pattern']}'):\n"
                    rel_base = Path(result.get("path", "."))
                    output_content += "\n".join(
                        f"- [cyan]{escape(os.path.relpath(m, rel_base))}[/cyan]"
                        for m in result["matches"][:20]
                    )
                    if len(result["matches"]) > 20:
                        output_content += "\n- ... (more matches)"
                    if result.get("warnings"):
                        output_content += (
                            "\n[yellow]Warnings:[/yellow]\n"
                            + "\n".join(f"- {escape(w)}" for w in result["warnings"])
                            + "\n"
                        )
                # Handle 'entries' from list_directory
                elif "entries" in result and "path" in result:
                    output_content += (
                        f"Directory Listing for [cyan]{escape(str(result['path']))}[/cyan]:\n"
                    )
                    table = Table(show_header=True, header_style="bold magenta", box=None)
                    table.add_column("Name", style="cyan", no_wrap=True)
                    table.add_column("Type", style="green")
                    table.add_column("Info", style="yellow")
                    for entry in result.get("entries", []):
                        name = entry.get("name", "?")
                        etype = entry.get("type", "unknown")
                        info_str = ""
                        if etype == "file" and "size" in entry:
                            info_str += f"{entry['size']} bytes"
                        elif etype == "symlink" and "symlink_target" in entry:
                            info_str += f"-> {escape(str(entry['symlink_target']))}"
                        if "error" in entry:
                            info_str += f" [red](Error: {escape(entry['error'])})[/red]"
                        icon = (
                            "📄"
                            if etype == "file"
                            else "📁"
                            if etype == "directory"
                            else "🔗"
                            if etype == "symlink"
                            else "❓"
                        )
                        table.add_row(f"{icon} {escape(name)}", etype, info_str)
                    with Capture(console) as capture:  # Use Capture from rich.console
                        console.print(table)
                    output_content += capture.get()
                    if result.get("warnings"):
                        output_content += (
                            "\n[yellow]Warnings:[/yellow]\n"
                            + "\n".join(f"- {escape(w)}" for w in result["warnings"])
                            + "\n"
                        )
                # Handle 'tree' from directory_tree
                elif "tree" in result and "path" in result:
                    output_content += (
                        f"Directory Tree for [cyan]{escape(str(result['path']))}[/cyan]:\n"
                    )

                    # Local helper function to build the rich tree recursively
                    def build_rich_tree_display(parent_node, children):
                        for item in children:
                            name = item.get("name", "?")
                            item_type = item.get("type", "unknown")
                            info = ""
                            if "size" in item:
                                size_bytes = item["size"]
                                if size_bytes < 1024:
                                    info += f" ({size_bytes}b)"
                                elif size_bytes < 1024 * 1024:
                                    info += f" ({size_bytes / 1024:.1f}KB)"
                                else:
                                    info += f" ({size_bytes / (1024 * 1024):.1f}MB)"
                            if "target" in item:
                                info += f" → {escape(item['target'])}"
                            if "error" in item:
                                info += f" [red](Error: {escape(item['error'])})[/red]"

                            if item_type == "directory":
                                node = parent_node.add(
                                    f"📁 [bold cyan]{escape(name)}[/bold cyan]{info}"
                                )
                                if "children" in item:
                                    build_rich_tree_display(node, item["children"])
                            elif item_type == "file":
                                icon = "📄"  # Default icon
                                ext = os.path.splitext(name)[1].lower()
                                if ext in [".jpg", ".png", ".gif", ".bmp", ".jpeg", ".svg"]:
                                    icon = "🖼️"
                                elif ext in [".mp3", ".wav", ".ogg", ".flac"]:
                                    icon = "🎵"
                                elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                                    icon = "🎬"
                                elif ext in [".py", ".js", ".java", ".c", ".cpp", ".go", ".rs"]:
                                    icon = "📜"
                                elif ext in [".json", ".xml", ".yaml", ".yml"]:
                                    icon = "📋"
                                elif ext in [".zip", ".tar", ".gz", ".7z", ".rar"]:
                                    icon = "📦"
                                elif ext in [".md", ".txt", ".doc", ".docx", ".pdf"]:
                                    icon = "📝"
                                parent_node.add(f"{icon} [green]{escape(name)}[/green]{info}")
                            elif item_type == "symlink":
                                parent_node.add(f"🔗 [magenta]{escape(name)}[/magenta]{info}")
                            elif item_type == "info":
                                parent_node.add(f"ℹ️ [dim]{escape(name)}[/dim]")
                            elif item_type == "error":
                                parent_node.add(f"❌ [red]{escape(name)}[/red]{info}")
                            else:
                                parent_node.add(f"❓ [yellow]{escape(name)}[/yellow]{info}")

                    rich_tree_root = Tree(
                        f"📁 [bold cyan]{escape(os.path.basename(result['path']))}[/bold cyan]"
                    )
                    build_rich_tree_display(rich_tree_root, result["tree"])
                    with Capture(console) as capture:  # Use Capture from rich.console
                        console.print(rich_tree_root)
                    output_content += capture.get()
                # Handle 'directories' from list_allowed_directories
                elif "directories" in result and "count" in result:
                    output_content += f"Allowed Directories ({result['count']}):\n"
                    output_content += (
                        "\n".join(f"- [green]{escape(d)}[/green]" for d in result["directories"])
                        + "\n"
                    )
                # Handle 'files' from read_multiple_files
                elif "files" in result and "succeeded" in result:
                    output_content += f"Read Results: [green]{result['succeeded']} succeeded[/green], [red]{result['failed']} failed[/red]\n"
                    for file_res in result.get("files", []):
                        path_str = escape(str(file_res.get("path", "N/A")))
                        if file_res.get("success"):
                            size_info = (
                                f" ({file_res.get('size', 'N/A')}b)" if "size" in file_res else ""
                            )
                            # Use preview if available, else content snippet
                            content_display = file_res.get("preview", file_res.get("content", ""))
                            output_content += f"- [green]Success[/green]: [cyan]{path_str}[/cyan]{size_info}\n  Content: '{escape(str(content_display))}'\n"
                        else:
                            output_content += f"- [red]Failed[/red]: [cyan]{path_str}[/cyan]\n  Error: {escape(str(file_res.get('error', 'Unknown')))}\n"
                # Handle 'content' block (from read_file)
                elif "content" in result and "path" in result:  # More specific check for read_file
                    # Check if content is list of blocks (MCP format) or simple string/bytes
                    content_data = result["content"]
                    preview_content = ""
                    if (
                        isinstance(content_data, list)
                        and content_data
                        and "text" in content_data[0]
                    ):
                        # Assumes MCP text block format
                        preview_content = "\n".join(
                            [
                                escape(block.get("text", ""))
                                for block in content_data
                                if block.get("type") == "text"
                            ]
                        )
                    elif isinstance(content_data, str):
                        # Simple string content
                        preview_content = escape(
                            content_data[:1000] + ("..." if len(content_data) > 1000 else "")
                        )  # Limit preview
                    elif isinstance(content_data, bytes):
                        # Handle bytes (e.g., hex preview)
                        try:
                            import binascii

                            hex_preview = binascii.hexlify(content_data[:64]).decode(
                                "ascii"
                            )  # Preview first 64 bytes
                            preview_content = f"[dim]Binary Content (Hex Preview):[/dim]\n{hex_preview}{'...' if len(content_data) > 64 else ''}"
                        except Exception:
                            preview_content = "[dim]Binary Content (Preview unavailable)[/dim]"

                    if preview_content:  # Only add if we have something to show
                        output_content += (
                            f"Content ({result.get('size', 'N/A')} bytes):\n{preview_content}\n"
                        )
                    elif "size" in result:  # If no content preview but size exists
                        output_content += f"Size: [yellow]{result['size']}[/yellow] bytes\n"

                # Handle 'modified' from get_file_info
                elif "name" in result and "modified" in result:
                    output_content += f"File Info for [cyan]{escape(result['name'])}[/cyan]:\n"
                    info_table = Table(show_header=False, box=None)
                    info_table.add_column("Property", style="blue")
                    info_table.add_column("Value", style="yellow")
                    skip_keys = {"success", "message", "path", "name"}
                    for k, v in result.items():
                        if k not in skip_keys:
                            info_table.add_row(escape(k), pretty_repr(v))
                    with Capture(console) as capture:  # Use Capture from rich.console
                        console.print(info_table)
                    output_content += capture.get()

                # Fallback for other dictionaries
                else:
                    excluded_keys = {
                        "content",
                        "tree",
                        "entries",
                        "matches",
                        "files",
                        "success",
                        "message",
                    }
                    display_dict = {k: v for k, v in result.items() if k not in excluded_keys}
                    if display_dict:
                        output_content += "Result Data:\n" + pretty_repr(display_dict) + "\n"
                    elif not output_content:  # If nothing else was printed
                        output_content = "[dim](Tool executed successfully, no specific output format matched)[/dim]"

            # Handle non-dict results (should be rare)
            else:
                output_content = escape(str(result))

            console.print(
                Panel(
                    output_content,
                    title=f"Result: {tool_name} (Success)",
                    border_style="green",
                    subtitle=f"Duration: {duration:.3f}s",
                )
            )
            return {"success": True, "result": result}

    except ProtectionTriggeredError as pte:
        duration = time.monotonic() - start_time
        logger.warning(
            f"Protection triggered calling {tool_name}: {pte}", exc_info=True
        )  # Use logger from display.py
        console.print(
            Panel(
                f"[bold yellow]🛡️ Protection Triggered![/bold yellow]\n"
                f"Message: {escape(str(pte))}\n"
                f"Context: {pretty_repr(pte.context)}",
                title=f"Result: {tool_name} (Blocked)",
                border_style="yellow",
                subtitle=f"Duration: {duration:.3f}s",
            )
        )
        return {
            "success": False,
            "protection_triggered": True,
            "error": str(pte),
            "context": pte.context,
            "result": None,
        }
    except (ToolInputError, ToolError) as tool_err:
        duration = time.monotonic() - start_time
        error_code = getattr(tool_err, "error_code", "TOOL_ERROR")
        details = getattr(tool_err, "details", None)  # Use getattr with default None
        logger.error(
            f"Tool Error calling {tool_name}: {tool_err} ({error_code})",
            exc_info=True,
            extra={"details": details},
        )  # Use logger from display.py

        error_content = f"[bold red]{type(tool_err).__name__} ({error_code})[/bold red]\n"
        error_content += f"Message: {escape(str(tool_err))}"
        if details:
            error_content += f"\nDetails:\n{pretty_repr(details)}"
        else:
            error_content += "\nDetails: N/A"

        error_content += f"\n\nFunction: [yellow]{tool_name}[/yellow]"
        error_content += (
            f"\nArguments: [dim]{pretty_repr(args_dict)}[/dim]"  # Use pretty_repr for args
        )

        console.print(
            Panel(
                error_content,
                title=f"Result: {tool_name} (Failed)",
                border_style="red",
                subtitle=f"Duration: {duration:.3f}s",
            )
        )
        return {
            "success": False,
            "error": str(tool_err),
            "details": details,
            "error_code": error_code,
            "result": None,
        }
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.critical(
            f"Unexpected Exception calling {tool_name}: {e}", exc_info=True
        )  # Use logger from display.py
        console.print(
            Panel(
                f"[bold red]Unexpected Error ({type(e).__name__})[/bold red]\n{escape(str(e))}",
                title=f"Result: {tool_name} (Critical Failure)",
                border_style="red",
                subtitle=f"Duration: {duration:.3f}s",
            )
        )
        # Include basic args in details for unexpected errors too
        return {
            "success": False,
            "error": f"Unexpected: {str(e)}",
            "details": {"type": type(e).__name__, "args": args_dict},
            "result": None,
        }


# --- Async Rich Directory Tree Builder ---
# RESTORED ASYNC VERSION


async def _build_rich_directory_tree_recursive(
    path: Path, tree_node: Tree, depth: int, max_depth: int
):
    """
    Recursive helper to build a Rich Tree using async list_directory.

    This function is a recursive helper for generating a Rich Tree representation
    of a directory structure using the async list_directory tool. It traverses the
    directory tree and adds nodes for each file, directory, and symlink encountered,
    with appropriate icons and styling based on file types.

    Args:
        path: The current directory path (Path object) to display.
        tree_node: The parent Tree node to add child nodes to.
        depth: The current recursion depth in the directory tree.
        max_depth: The maximum depth to traverse, preventing excessive recursion.

    Note:
        This function uses different icons for different file types and includes
        size information for files and target information for symlinks when available.
    """
    if depth >= max_depth:
        tree_node.add("📁 [dim]...(max depth reached)[/dim]")
        return

    try:
        # Call the async list_directory tool
        list_result = await list_directory(path=str(path))

        # Handle potential errors from the list_directory call itself
        if isinstance(list_result, dict) and list_result.get("error"):
            error_msg = list_result.get("error", "Unknown listing error")
            tree_node.add(f"❌ [red]Error listing: {escape(error_msg)}[/red]")
            return

        # Ensure result structure is as expected
        if not isinstance(list_result, dict) or "entries" not in list_result:
            tree_node.add(
                f"❓ [yellow]Unexpected result format from list_directory for {escape(str(path))}[/yellow]"
            )
            logger.warning(f"Unexpected list_directory result for {path}: {list_result}")
            return

        entries = sorted(list_result.get("entries", []), key=lambda x: x.get("name", ""))

        for item in entries:
            name = item.get("name", "?")
            item_type = item.get("type", "unknown")
            entry_error = item.get("error")
            item_path = path / name

            # Skip hidden files/dirs (same logic as demo)
            # if name.startswith('.') and name != '.gitignore':
            #     continue

            # Handle entry-specific errors
            if entry_error:
                tree_node.add(f"❌ [red]{escape(name)} - Error: {escape(entry_error)}[/red]")
                continue

            info = ""
            # Use size reported by list_directory
            if "size" in item and item_type == "file":
                size_bytes = item["size"]
                if size_bytes < 1024:
                    info += f" ({size_bytes}b)"
                elif size_bytes < 1024 * 1024:
                    info += f" ({size_bytes / 1024:.1f}KB)"
                else:
                    info += f" ({size_bytes / (1024 * 1024):.1f}MB)"
            # Use symlink_target reported by list_directory
            if item_type == "symlink" and "symlink_target" in item:
                info += f" → {escape(str(item['symlink_target']))}"

            if item_type == "directory":
                # Add node for directory
                dir_node = tree_node.add(f"📁 [bold cyan]{escape(name)}[/bold cyan]{info}")
                # Recurse into subdirectory
                await _build_rich_directory_tree_recursive(
                    item_path, dir_node, depth + 1, max_depth
                )
            elif item_type == "file":
                # Icon logic copied from demo
                icon = "📄"
                ext = os.path.splitext(name)[1].lower()
                if ext in [".jpg", ".png", ".gif", ".bmp", ".jpeg", ".svg"]:
                    icon = "🖼️"
                elif ext in [".mp3", ".wav", ".ogg", ".flac"]:
                    icon = "🎵"
                elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    icon = "🎬"
                elif ext in [".py", ".js", ".java", ".c", ".cpp", ".go", ".rs"]:
                    icon = "📜"
                elif ext in [".json", ".xml", ".yaml", ".yml"]:
                    icon = "📋"
                elif ext in [".zip", ".tar", ".gz", ".7z", ".rar"]:
                    icon = "📦"
                elif ext in [".md", ".txt", ".doc", ".docx", ".pdf"]:
                    icon = "📝"
                tree_node.add(f"{icon} [green]{escape(name)}[/green]{info}")
            elif item_type == "symlink":
                tree_node.add(f"🔗 [magenta]{escape(name)}[/magenta]{info}")
            # Handle potential info/error items from list_directory (though less common than directory_tree)
            elif item_type == "info":
                tree_node.add(f"ℹ️ [dim]{escape(name)}[/dim]")
            elif item_type == "error":
                tree_node.add(f"❌ [red]{escape(name)}[/red]{info}")
            else:  # Handle unknown type
                tree_node.add(f"❓ [yellow]{escape(name)}[/yellow]{info}")

    except Exception as e:
        # Catch unexpected errors during the process for this path
        logger.error(f"Unexpected error building tree for {path}: {e}", exc_info=True)
        tree_node.add(
            f"❌ [bold red]Failed to process: {escape(str(path))} ({type(e).__name__})[/bold red]"
        )


async def generate_rich_directory_tree(path: Union[str, Path], max_depth: int = 3) -> Tree:
    """
    Generates a `rich.Tree` visualization of a directory using async filesystem tools.

    This function generates a Rich Tree representation of a directory structure
    using the async filesystem tools provided by the Ultimate MCP Server. It
    supports traversing the directory tree up to a specified maximum depth.

    Args:
        path: The starting directory path (string or Path object).
        max_depth: The maximum depth to traverse.

    Returns:
        A `rich.Tree` object representing the directory structure.
    """
    start_path = Path(path)
    tree_root = Tree(f"📁 [bold cyan]{escape(start_path.name)}[/bold cyan]")

    # Check if the root path exists and is a directory before starting recursion
    try:
        # Use list_directory for the initial check
        initial_check = await list_directory(path=str(start_path))
        if isinstance(initial_check, dict) and initial_check.get("error"):
            # Check if the error is because it's not a directory or doesn't exist
            error_msg = initial_check["error"]
            if "Not a directory" in error_msg or "No such file or directory" in error_msg:
                tree_root.add(f"❌ [red]{escape(error_msg)}: {escape(str(start_path))}[/red]")
            else:
                tree_root.add(f"❌ [red]Error accessing root path: {escape(error_msg)}[/red]")
            return tree_root  # Return tree with only the error
        # We assume if list_directory didn't error, it's a directory.
    except Exception as e:
        logger.error(f"Error during initial check for {start_path}: {e}", exc_info=True)
        tree_root.add(
            f"❌ [bold red]Failed initial check: {escape(str(start_path))} ({type(e).__name__})[/bold red]"
        )
        return tree_root

    # Start the recursive build if initial check seems okay
    await _build_rich_directory_tree_recursive(start_path, tree_root, depth=0, max_depth=max_depth)
    return tree_root


# --- Cost Tracking Utility ---


class CostTracker:
    """
    Tracks and aggregates API call costs and token usage across multiple LLM operations.

    The CostTracker provides a centralized mechanism for monitoring API usage costs
    and token consumption across multiple calls to language model providers. It maintains
    a structured record of costs organized by provider and model, supporting various
    result formats from different API calls.

    The tracker can extract cost and token information from both object attributes
    (like CompletionResult objects) and dictionary structures (like tool results),
    making it versatile for different API response formats.

    Features:
    - Detailed tracking by provider and model
    - Support for input and output token counts
    - Optional cost limit monitoring
    - Rich console visualization of cost summaries
    - Aggregation of calls, tokens, and costs

    Usage example:
    ```python
    # Initialize tracker
    tracker = CostTracker(limit=5.0)  # Set $5.00 cost limit

    # Track costs from various API calls
    tracker.add_call(completion_result)
    tracker.add_call(summarization_result)

    # Check if limit exceeded
    if tracker.exceeds_limit():
        print("Cost limit exceeded!")

    # Display summary in console
    tracker.display_summary(console)
    ```

    Attributes:
        data: Nested dictionary storing cost and token data organized by provider and model
        limit: Optional cost limit in USD to monitor usage against
    """

    def __init__(self, limit: Optional[float] = None):
        """
        Initialize a new cost tracker with an optional spending limit.

        Args:
            limit: Optional maximum cost limit in USD. If provided, the tracker
                  can report when costs exceed this threshold using exceeds_limit().
        """
        self.data: Dict[
            str, Dict[str, Dict[str, Any]]
        ] = {}  # {provider: {model: {cost, tokens..., calls}}}
        self.limit: Optional[float] = limit  # Cost limit in USD

    @property
    def total_cost(self) -> float:
        """
        Get the total accumulated cost across all providers and models.

        Returns:
            float: The sum of all tracked costs in USD
        """
        total = 0.0
        for provider_data in self.data.values():
            for model_data in provider_data.values():
                total += model_data.get("cost", 0.0)
        return total

    def exceeds_limit(self) -> bool:
        """
        Check if the current total cost exceeds the specified limit.

        Returns:
            bool: True if a limit is set and the total cost exceeds it, False otherwise
        """
        if self.limit is None:
            return False
        return self.total_cost >= self.limit

    def add_call(self, result: Any, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Add cost and token data from an API call result to the tracker.

        This method extracts cost and token information from various result formats,
        including structured objects with attributes (like CompletionResult) or
        dictionaries (like tool results). It intelligently identifies the relevant
        data fields and updates the tracking statistics.

        Args:
            result: The API call result containing cost and token information.
                   Can be an object with attributes or a dictionary.
            provider: Optional provider name override. If not specified, will be
                     extracted from the result if available.
            model: Optional model name override. If not specified, will be
                  extracted from the result if available.

        Example:
            ```python
            # Track a direct API call result
            summarization_result = await summarize_document(...)
            tracker.add_call(summarization_result)

            # Track with explicit provider/model specification
            tracker.add_call(
                custom_result,
                provider="openai",
                model="gpt-4o"
            )
            ```
        """
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        # Try extracting from object attributes (e.g., CompletionResult)
        if hasattr(result, "cost") and result.cost is not None:
            cost = float(result.cost)
        if hasattr(result, "provider") and result.provider:
            provider = result.provider
        if hasattr(result, "model") and result.model:
            model = result.model
        if hasattr(result, "input_tokens") and result.input_tokens is not None:
            input_tokens = int(result.input_tokens)
        if hasattr(result, "output_tokens") and result.output_tokens is not None:
            output_tokens = int(result.output_tokens)
        if hasattr(result, "total_tokens") and result.total_tokens is not None:
            total_tokens = int(result.total_tokens)
        elif input_tokens > 0 or output_tokens > 0:
            total_tokens = input_tokens + output_tokens  # Calculate if not present

        # Try extracting from dictionary keys (e.g., tool results, stats dicts)
        elif isinstance(result, dict):
            cost = float(result.get("cost", 0.0))
            provider = result.get("provider", provider)  # Use existing if key not found
            model = result.get("model", model)  # Use existing if key not found
            tokens_data = result.get("tokens", {})
            if isinstance(tokens_data, dict):
                input_tokens = int(tokens_data.get("input", 0))
                output_tokens = int(tokens_data.get("output", 0))
                total_tokens = int(tokens_data.get("total", 0))
                if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
                    total_tokens = input_tokens + output_tokens
            elif isinstance(
                tokens_data, (int, float)
            ):  # Handle case where 'tokens' is just a total number
                total_tokens = int(tokens_data)

        # --- Fallback / Defaulting ---
        # If provider/model couldn't be determined, use defaults
        provider = provider or "UnknownProvider"
        model = model or "UnknownModel"

        # --- Update Tracking Data ---
        if provider not in self.data:
            self.data[provider] = {}
        if model not in self.data[provider]:
            self.data[provider][model] = {
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }

        self.data[provider][model]["cost"] += cost
        self.data[provider][model]["input_tokens"] += input_tokens
        self.data[provider][model]["output_tokens"] += output_tokens
        self.data[provider][model]["total_tokens"] += total_tokens
        self.data[provider][model]["calls"] += 1

    def record_call(
        self, provider: str, model: str, input_tokens: int, output_tokens: int, cost: float
    ):
        """
        Directly record a call with explicit token counts and cost.

        This method allows manual tracking of API calls with explicit parameter values,
        useful when the cost information isn't available in a structured result object.

        Args:
            provider: The provider name (e.g., "openai", "anthropic")
            model: The model name (e.g., "gpt-4", "claude-3-opus")
            input_tokens: Number of input (prompt) tokens
            output_tokens: Number of output (completion) tokens
            cost: The cost of the API call in USD

        Example:
            ```python
            tracker.record_call(
                provider="openai",
                model="gpt-4o",
                input_tokens=1500,
                output_tokens=350,
                cost=0.03
            )
            ```
        """
        if provider not in self.data:
            self.data[provider] = {}
        if model not in self.data[provider]:
            self.data[provider][model] = {
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }

        total_tokens = input_tokens + output_tokens

        self.data[provider][model]["cost"] += cost
        self.data[provider][model]["input_tokens"] += input_tokens
        self.data[provider][model]["output_tokens"] += output_tokens
        self.data[provider][model]["total_tokens"] += total_tokens
        self.data[provider][model]["calls"] += 1

    def add_custom_cost(
        self,
        description: str,
        provider: str,
        model: str,
        cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """
        Add a custom cost entry with an optional description.

        This method is useful for tracking costs that aren't directly tied to a specific
        API call, such as batch processing fees, infrastructure costs, or estimated costs.

        Args:
            description: A descriptive label for this cost entry (e.g., "Batch Processing")
            provider: The provider name or service category
            model: The model name or service type
            cost: The cost amount in USD
            input_tokens: Optional input token count (default: 0)
            output_tokens: Optional output token count (default: 0)

        Example:
            ```python
            tracker.add_custom_cost(
                "Batch Processing",
                "openai",
                "gpt-4-turbo",
                0.25,
                input_tokens=5000,
                output_tokens=1200
            )
            ```
        """
        # Format the model name to include the description
        custom_model = f"{model} ({description})"

        if provider not in self.data:
            self.data[provider] = {}
        if custom_model not in self.data[provider]:
            self.data[provider][custom_model] = {
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }

        total_tokens = input_tokens + output_tokens

        self.data[provider][custom_model]["cost"] += cost
        self.data[provider][custom_model]["input_tokens"] += input_tokens
        self.data[provider][custom_model]["output_tokens"] += output_tokens
        self.data[provider][custom_model]["total_tokens"] += total_tokens
        self.data[provider][custom_model]["calls"] += 1

    def display_summary(
        self, console_instance: Optional[Console] = None, title: str = "Total Demo Cost Summary"
    ):
        """
        Display a formatted summary of all tracked costs and token usage in a Rich console table.

        This method generates a detailed tabular report showing:
        - Costs broken down by provider and model
        - Number of calls made to each model
        - Input, output, and total token counts
        - Subtotals by provider (when multiple models are used)
        - Grand totals across all providers and models
        - Progress against cost limit (if set)

        The report is formatted using Rich tables with color coding for readability.

        Args:
            console_instance: Optional Rich Console instance to use for display.
                             If not provided, uses the default console.
            title: Custom title for the summary report.
                  Defaults to "Total Demo Cost Summary".

        Example:
            ```python
            # Display with default settings
            tracker.display_summary()

            # Display with custom title and console
            from rich.console import Console
            custom_console = Console(width=100)
            tracker.display_summary(
                console_instance=custom_console,
                title="AI Generation Cost Report"
            )
            ```
        """
        output = console_instance or console  # Use provided or default console

        output.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))

        if not self.data:
            output.print("[yellow]No cost data tracked.[/yellow]")
            return

        summary_table = Table(
            title="[bold]API Call Costs & Tokens[/bold]",
            box=box.ROUNDED,
            show_footer=True,
            footer_style="bold",
        )
        summary_table.add_column("Provider", style="cyan", footer="Grand Total")
        summary_table.add_column("Model", style="magenta")
        summary_table.add_column(
            "Calls", style="blue", justify="right", footer=" "
        )  # Placeholder footer
        summary_table.add_column("Input Tokens", style="yellow", justify="right", footer=" ")
        summary_table.add_column("Output Tokens", style="yellow", justify="right", footer=" ")
        summary_table.add_column("Total Tokens", style="bold yellow", justify="right", footer=" ")
        summary_table.add_column("Total Cost ($)", style="bold green", justify="right", footer=" ")

        grand_total_cost = 0.0
        grand_total_calls = 0
        grand_total_input = 0
        grand_total_output = 0
        grand_total_tokens = 0

        sorted_providers = sorted(self.data.keys())
        for provider in sorted_providers:
            provider_total_cost = 0.0
            provider_total_calls = 0
            provider_total_input = 0
            provider_total_output = 0
            provider_total_tokens = 0

            sorted_models = sorted(self.data[provider].keys())
            num_models = len(sorted_models)

            for i, model in enumerate(sorted_models):
                stats = self.data[provider][model]
                provider_total_cost += stats["cost"]
                provider_total_calls += stats["calls"]
                provider_total_input += stats["input_tokens"]
                provider_total_output += stats["output_tokens"]
                provider_total_tokens += stats["total_tokens"]

                # Display provider only on the first row for that provider
                provider_display = escape(provider) if i == 0 else ""

                summary_table.add_row(
                    provider_display,
                    escape(model),
                    str(stats["calls"]),
                    f"{stats['input_tokens']:,}",
                    f"{stats['output_tokens']:,}",
                    f"{stats['total_tokens']:,}",
                    f"{stats['cost']:.6f}",
                )

            # Add provider subtotal row if more than one model for the provider
            if num_models > 1:
                summary_table.add_row(
                    "[dim]Subtotal[/dim]",
                    f"[dim]{provider}[/dim]",
                    f"[dim]{provider_total_calls:,}[/dim]",
                    f"[dim]{provider_total_input:,}[/dim]",
                    f"[dim]{provider_total_output:,}[/dim]",
                    f"[dim]{provider_total_tokens:,}[/dim]",
                    f"[dim]{provider_total_cost:.6f}[/dim]",
                    style="dim",
                    end_section=(
                        provider != sorted_providers[-1]
                    ),  # Add separator line unless it's the last provider
                )
            elif provider != sorted_providers[-1]:
                # Add separator if only one model but not the last provider
                summary_table.add_row(end_section=True)

            grand_total_cost += provider_total_cost
            grand_total_calls += provider_total_calls
            grand_total_input += provider_total_input
            grand_total_output += provider_total_output
            grand_total_tokens += provider_total_tokens

        # Update footer values (need to re-assign list for footer update)
        summary_table.columns[2].footer = f"{grand_total_calls:,}"
        summary_table.columns[3].footer = f"{grand_total_input:,}"
        summary_table.columns[4].footer = f"{grand_total_output:,}"
        summary_table.columns[5].footer = f"{grand_total_tokens:,}"
        summary_table.columns[6].footer = f"{grand_total_cost:.6f}"

        output.print(summary_table)

        # Display cost limit information if set
        if self.limit is not None:
            limit_color = "green" if self.total_cost < self.limit else "red"
            output.print(
                f"[{limit_color}]Cost limit: ${self.limit:.2f} | Current usage: ${self.total_cost:.2f} ({(self.total_cost / self.limit * 100):.1f}%)[/{limit_color}]"
            )

        output.print()  # Add a blank line after the table

    def display_costs(
        self, console: Optional[Console] = None, title: str = "Total Demo Cost Summary"
    ):
        """
        Alias for display_summary for backward compatibility.

        This method provides a backward-compatible interface for legacy code
        that might be calling display_costs() instead of display_summary().

        Args:
            console: Console instance to use for display
            title: Custom title for the summary report

        Returns:
            Same as display_summary()
        """
        return self.display_summary(console_instance=console, title=title)
