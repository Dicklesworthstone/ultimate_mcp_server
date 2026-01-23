#!/usr/bin/env python3
"""Demo script showcasing the marqo_fused_search tool."""

import asyncio
import json
import os
import sys
import time  # Add time import
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# Add Rich imports
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from ultimate_mcp_server.tools.marqo_fused_search import DateRange, marqo_fused_search  # noqa: E402
from ultimate_mcp_server.utils.logging import logger  # noqa: E402

# Initialize Rich Console
console = Console()

# --- Configuration ---
CONFIG_FILE_PATH = os.path.join(project_root, "marqo_index_config.json")


def load_marqo_config() -> Dict[str, Any]:
    """Loads Marqo configuration from the JSON file."""
    try:
        with open(CONFIG_FILE_PATH, "r") as f:
            config = json.load(f)
            logger.info(f"Loaded Marqo config from {CONFIG_FILE_PATH}")
            return config
    except FileNotFoundError:
        logger.error(
            f"Marqo config file not found at {CONFIG_FILE_PATH}. Cannot run dynamic examples."
        )
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding Marqo config file {CONFIG_FILE_PATH}: {e}")
        return {}


def find_schema_field(schema: Dict[str, Any], required_properties: Dict[str, Any]) -> Optional[str]:
    """
    Finds the first field name in the schema that matches all required properties.
    Handles nested properties like 'type'.
    """
    if not schema or "fields" not in schema:
        return None

    for field_name, properties in schema["fields"].items():
        match = True
        for req_prop, req_value in required_properties.items():
            # Allow checking properties like 'type', 'filterable', 'sortable', 'role', 'searchable'
            if properties.get(req_prop) != req_value:
                match = False
                break
        if match:
            # Avoid returning internal fields like _id unless specifically requested
            if field_name == "_id" and required_properties.get("role") != "internal":
                continue
            return field_name
    return None


# --- Helper Function ---
async def run_search_example(example_name: str, **kwargs):
    """Runs a single search example and prints the results using Rich."""
    console.print(Rule(f"[bold cyan]{example_name}[/bold cyan]"))

    # Display parameters using a panel
    param_str_parts = []
    for key, value in kwargs.items():
        # Format DateRange nicely
        if isinstance(value, DateRange):
            start_str = value.start_date.strftime("%Y-%m-%d") if value.start_date else "N/A"
            end_str = value.end_date.strftime("%Y-%m-%d") if value.end_date else "N/A"
            param_str_parts.append(
                f"  [green]{key}[/green]: Start=[yellow]{start_str}[/yellow], End=[yellow]{end_str}[/yellow]"
            )
        else:
            param_str_parts.append(f"  [green]{key}[/green]: [yellow]{escape(str(value))}[/yellow]")
    param_str = "\n".join(param_str_parts)
    console.print(Panel(param_str, title="Search Parameters", border_style="blue", expand=False))

    try:
        start_time = time.time()  # Use time for accurate timing
        results = await marqo_fused_search(**kwargs)
        processing_time = time.time() - start_time

        logger.debug(f"Raw results for '{example_name}': {results}")  # Keep debug log

        if results.get("success"):
            logger.success(
                f"Search successful for '{example_name}'! ({processing_time:.3f}s)",
                emoji_key="success",
            )

            # Display results using Rich Syntax for JSON
            results_json = json.dumps(results, indent=2, default=str)
            syntax = Syntax(results_json, "json", theme="default", line_numbers=True)
            console.print(Panel(syntax, title="Marqo Search Results", border_style="green"))

        else:
            # Display error nicely if success is False but no exception was raised
            error_msg = results.get("error", "Unknown error")
            error_code = results.get("error_code", "UNKNOWN_CODE")
            logger.error(
                f"Search failed for '{example_name}': {error_code} - {error_msg}", emoji_key="error"
            )
            console.print(
                Panel(
                    f"[bold red]Error ({error_code}):[/bold red]\n{escape(error_msg)}",
                    title="Search Failed",
                    border_style="red",
                )
            )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"An exception occurred during '{example_name}' ({processing_time:.3f}s): {e}",
            emoji_key="critical",
            exc_info=True,
        )
        # Display exception using Rich traceback
        console.print_exception(show_locals=False)
        console.print(
            Panel(
                f"[bold red]Exception:[/bold red]\n{escape(str(e))}",
                title="Execution Error",
                border_style="red",
            )
        )

    console.print()  # Add space after each example


# --- Main Demo Function ---
async def main():
    """Runs various demonstrations of the marqo_fused_search tool."""

    # Load Marqo configuration and schema
    marqo_config = load_marqo_config()
    if not marqo_config:
        logger.error("Exiting demo as Marqo config could not be loaded.")
        return

    schema = marqo_config.get("default_schema", {})
    tensor_field = schema.get("tensor_field")
    # content_field = schema.get("default_content_field", "content") # Not directly used in examples
    date_field = schema.get("default_date_field")  # Used for date range

    # --- Find suitable fields dynamically ---
    # For filter examples (keyword preferred)
    filter_field = find_schema_field(
        schema, {"filterable": True, "type": "keyword"}
    ) or find_schema_field(schema, {"filterable": True})  # Fallback to any filterable

    # For lexical search (requires searchable='lexical')
    lexical_field_1 = find_schema_field(schema, {"searchable": "lexical"})
    lexical_field_2 = (
        find_schema_field(schema, {"searchable": "lexical", "field_name_not": lexical_field_1})
        or lexical_field_1
    )  # Find a second one if possible

    # For hybrid search (need tensor + lexical)
    hybrid_tensor_field = tensor_field  # Use the main tensor field
    hybrid_lexical_field_1 = lexical_field_1
    hybrid_lexical_field_2 = lexical_field_2

    # For explicit tensor search (need tensor field)
    explicit_tensor_field = tensor_field

    logger.info("Dynamically determined fields for examples:")
    logger.info(f"  Filter Field: '{filter_field}'")
    logger.info(f"  Lexical Fields: '{lexical_field_1}', '{lexical_field_2}'")
    logger.info(f"  Tensor Field (for hybrid/explicit): '{hybrid_tensor_field}'")
    logger.info(f"  Date Field (for range): '{date_field}'")

    # --- Run Examples ---

    # --- Example 1: Basic Semantic Search --- (No specific fields needed)
    await run_search_example("Basic Semantic Search", query="impact of AI on software development")

    # --- Example 2: Search with Metadata Filter ---
    if filter_field:
        # Use a plausible value; specific value might not exist in data
        example_filter_value = "10-K" if filter_field == "form_type" else "example_value"
        await run_search_example(
            "Search with Metadata Filter",
            query="latest advancements in renewable energy",
            filters={filter_field: example_filter_value},
        )
    else:
        logger.warning("Skipping Example 2: No suitable filterable field found in schema.")

    # --- Example 3: Search with Multiple Filter Values (OR condition) ---
    if filter_field:
        # Use plausible values
        example_filter_values = (
            ["10-K", "10-Q"] if filter_field == "form_type" else ["value1", "value2"]
        )
        await run_search_example(
            "Search with Multiple Filter Values (OR)",
            query="financial report analysis",
            filters={filter_field: example_filter_values},
        )
    else:
        logger.warning("Skipping Example 3: No suitable filterable field found in schema.")

    # --- Example 4: Search with Date Range ---
    if date_field and find_schema_field(schema, {"name": date_field, "type": "timestamp"}):
        start_date = datetime.now() - timedelta(days=900)
        end_date = datetime.now() - timedelta(days=30)
        await run_search_example(
            "Search with Date Range",
            query="market trends",
            date_range=DateRange(start_date=start_date, end_date=end_date),
        )
    else:
        logger.warning(
            f"Skipping Example 4: No sortable timestamp field named '{date_field}' (default_date_field) found in schema."
        )

    # --- Example 5: Pure Lexical Search --- (Relies on schema having lexical fields)
    # The tool will auto-detect lexical fields if not specified, but this tests the weight
    await run_search_example(
        "Pure Lexical Search",
        query="exact sciences",  # Query likely to hit company name etc.
        semantic_weight=0.0,
    )

    # --- Example 6: Hybrid Search with Custom Weight --- (Relies on schema having both)
    await run_search_example(
        "Hybrid Search with Custom Weight",
        query="balancing innovation and regulation",
        semantic_weight=0.5,  # Equal weight
    )

    # --- Example 7: Pagination (Limit and Offset) --- (No specific fields needed)
    await run_search_example(
        "Pagination (Limit and Offset)", query="common programming paradigms", limit=10, offset=10
    )

    # --- Example 8: Explicit Searchable Attributes (Tensor Search) ---
    if explicit_tensor_field:
        await run_search_example(
            "Explicit Tensor Searchable Attributes",
            query="neural network architectures",
            searchable_attributes=[explicit_tensor_field],
            semantic_weight=1.0,  # Ensure tensor search is used
        )
    else:
        logger.warning("Skipping Example 8: No tensor field found in schema.")

    # --- Example 9: Explicit Hybrid Search Attributes ---
    if hybrid_tensor_field and hybrid_lexical_field_1:
        lexical_fields = [hybrid_lexical_field_1]
        if hybrid_lexical_field_2 and hybrid_lexical_field_1 != hybrid_lexical_field_2:
            lexical_fields.append(hybrid_lexical_field_2)
        await run_search_example(
            "Explicit Hybrid Search Attributes",
            query="machine learning applications in healthcare",
            hybrid_search_attributes={"tensor": [hybrid_tensor_field], "lexical": lexical_fields},
            semantic_weight=0.6,  # Specify hybrid search balance
        )
    else:
        logger.warning("Skipping Example 9: Need both tensor and lexical fields defined in schema.")

    # --- Example 12: Overriding Marqo URL and Index Name --- (Keep commented out)
    # ... rest of the code ...
    console.print(Rule("[bold magenta]Marqo Fused Search Demo Complete[/bold magenta]"))


if __name__ == "__main__":
    console.print(Rule("[bold magenta]Starting Marqo Fused Search Demo[/bold magenta]"))
    # logger.info("Starting Marqo Fused Search Demo...") # Replaced by Rich rule
    asyncio.run(main())
    # logger.info("Marqo Fused Search Demo finished.") # Replaced by Rich rule
