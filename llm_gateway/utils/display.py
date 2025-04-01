"""Display utilities for rich output formatting in demos and applications."""
import json
from typing import Any, Dict, List, Optional, Union

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

# Import the console for consistent styling
from llm_gateway.utils.logging.console import console


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
    if hasattr(result, 'text'):
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
    title: str, 
    result: Any, 
    console_instance: Optional[Console] = None
):
    """
    Display results from TextContent objects more reliably, which is useful for demos.
    This function is more forgiving with different formats and provides better handling
    for TextContent objects that might contain JSON strings.
    
    Args:
        title: Title to display for this result section
        result: Result object from an LLM Gateway tool call (often a TextContent)
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
        output.print(Panel(
            escape(parsed_data["raw_text"]),
            title="[bold]Result Text[/bold]",
            border_style="green"
        ))
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
            output.print(Panel(
                "\n".join([f"[bold]Q{i+1}:[/bold] {escape(pair.get('question', 'N/A'))}\n[bold]A{i+1}:[/bold] {escape(pair.get('answer', 'N/A'))}" 
                        for i, pair in enumerate(qa_pairs)]),
                title="[bold]Q&A Pairs[/bold]", 
                border_style="blue"
            ))
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
                            entity_text = entity if isinstance(entity, str) else entity.get('text', str(entity))
                            entity_table.add_row(entity_type, escape(entity_text))
                            entity_count += 1
                
                if entity_count > 0:
                    output.print(entity_table)
                else:
                    output.print("[yellow]No entities found in the document.[/yellow]")
            else:
                # If it's some other format, just show the raw data
                output.print(Panel(
                    escape(json.dumps(entities_data, indent=2)),
                    title="[bold]Entities Data[/bold]",
                    border_style="blue"
                ))
        # Summary
        elif "summary" in parsed_data and isinstance(parsed_data["summary"], str):
            output.print(Panel(
                escape(parsed_data["summary"]),
                title="[bold]Generated Summary[/bold]",
                border_style="green"
            ))
        # Generic JSON display for other data
        else:
            # Filter out stats fields for cleaner display
            display_data = {k: v for k, v in parsed_data.items() 
                           if k not in ["model", "provider", "cost", "tokens", "processing_time"]}
            
            # Only show JSON panel if we have data to display
            if display_data:
                output.print(Panel(
                    escape(json.dumps(display_data, indent=2)),
                    title="[bold]Result Data[/bold]",
                    border_style="blue"
                ))
        
        # Display stats if available
        if any(k in parsed_data for k in ["model", "provider", "cost", "tokens", "processing_time"]):
            _display_stats(parsed_data, output)
    else:
        # For other types (arrays, etc.)
        output.print(Panel(
            escape(json.dumps(parsed_data, indent=2)),
            title="[bold]Result Data[/bold]",
            border_style="blue"
        ))


def parse_and_display_result(
    title: str, 
    input_data: Optional[Dict] = None, 
    result: Any = None, 
    display_input: bool = True,
    console_instance: Optional[Console] = None
):
    """
    Parse and display results using Rich with consistent formatting across demos.
    
    Args:
        title: Title to display for this result section
        input_data: Dictionary containing input data (e.g., text, schema)
        result: Result object from an LLM Gateway tool call
        display_input: Whether to display the input data
        console_instance: Optional console instance to use (defaults to shared console)
    """
    # Use provided console or default to shared console
    output = console_instance or console
    
    # Display section title
    output.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))

    # Display input if requested
    if display_input and input_data:
        _display_input_data(input_data, output)

    # Handle different result formats
    if result is not None:
        # For TextContent, use the specialized function
        if isinstance(result, list) and result and hasattr(result[0], 'text'):
            # Use extract_and_parse_content and display appropriately
            parsed_data = extract_and_parse_content(result)
            _display_result_content(parsed_data, output)
        elif hasattr(result, 'text'):
            # Single TextContent object
            parsed_data = extract_and_parse_content(result)
            _display_result_content(parsed_data, output)
        else:
            # Use existing handler for other types
            _parse_and_display_output(result, output)


def _display_input_data(input_data: Dict, output: Console):
    """Display input data with consistent formatting."""
    # Display input text if available
    if "text" in input_data:
        text_snippet = input_data["text"][:500] + ("..." if len(input_data["text"]) > 500 else "")
        output.print(Panel(
            escape(text_snippet), 
            title="[cyan]Input Text Snippet[/cyan]", 
            border_style="dim blue"
        ))
    
    # Display schema if available
    if "json_schema" in input_data and input_data["json_schema"]:
        try:
            schema_json = json.dumps(input_data["json_schema"], indent=2)
            output.print(Panel(
                Syntax(schema_json, "json", theme="default", line_numbers=False), 
                title="[cyan]Input Schema[/cyan]", 
                border_style="dim blue"
            ))
        except Exception as e:
            output.print(f"[red]Could not display schema: {escape(str(e))}[/red]")
    
    # Display query if available (for search results)
    if "query" in input_data:
        output.print(Panel(
            escape(input_data["query"]), 
            title="[cyan]Search Query[/cyan]", 
            border_style="dim blue"
        ))
        
    # Display embeddings/vectors if available
    if "embeddings" in input_data:
        if isinstance(input_data["embeddings"], list) and len(input_data["embeddings"]) > 0:
            sample = input_data["embeddings"][0]
            dims = len(sample) if isinstance(sample, (list, tuple)) else "unknown"
            sample_str = str(sample[:3]) + "..." if isinstance(sample, (list, tuple)) else str(sample)
            output.print(Panel(
                f"[cyan]Dimensions:[/cyan] {dims}\n[cyan]Sample:[/cyan] {escape(sample_str)}", 
                title="[cyan]Embedding Sample[/cyan]", 
                border_style="dim blue"
            ))


def _parse_and_display_output(result: Any, output: Console):
    """Parse result object and display appropriate visualizations."""
    # Extract result content
    parsed_result = {}
    raw_text = None
    
    # Handle list results (take first item)
    if isinstance(result, list) and result:
        result = result[0]
        
    # Handle object with text attribute
    if hasattr(result, 'text'):
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
    """Display the content of results with appropriate formatting."""
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
        _display_embeddings_info(parsed_result["embeddings"], 
                                parsed_result.get("model", "unknown"),
                                output)
    
    # Display execution stats if available
    _display_stats(parsed_result, output)


def _display_error(result: Dict, output: Console):
    """Display error information."""
    error_content = f"[red]Error:[/red] {escape(result['error'])}"
    if result.get("raw_text"):
        error_content += f"\n\n[yellow]Raw Text Output:[/yellow]\n{escape(result['raw_text'])}"
    output.print(Panel(
        error_content, 
        title="[bold red]Tool Error[/bold red]", 
        border_style="red"
    ))


def _display_json_data(data: Any, title: str, output: Console):
    """Display JSON data with proper formatting."""
    try:
        data_json = json.dumps(data, indent=2)
        output.print(Panel(
            Syntax(data_json, "json", theme="default", line_numbers=True, word_wrap=True),
            title=f"[bold green]{title}[/bold green]",
            border_style="green"
        ))
    except Exception as e:
        output.print(f"[red]Could not display JSON data: {escape(str(e))}[/red]")


def _display_vector_results(results: List[Dict], output: Console):
    """Display vector search results."""
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
            f"{item.get('similarity', item.get('score', 0.0)):.4f}"
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
    """Display extracted tables."""
    for i, table_info in enumerate(tables):
        table_title = table_info.get('title', f'Table {i+1}')
        output.print(Rule(f"[green]Extracted: {escape(table_title)}[/green]"))
        
        # JSON format
        if table_info.get("json"):
            try:
                table_json = json.dumps(table_info["json"], indent=2)
                output.print(Panel(
                    Syntax(table_json, "json", theme="default", line_numbers=False, word_wrap=True),
                    title="[bold]JSON Format[/bold]",
                    border_style="dim green"
                ))
            except Exception as e:
                output.print(f"[red]Could not display table JSON: {escape(str(e))}[/red]")
        
        # Markdown format
        if table_info.get("markdown"):
            output.print(Panel(
                Syntax(table_info["markdown"], "markdown", theme="default"),
                title="[bold]Markdown Format[/bold]",
                border_style="dim green"
            ))
        
        # Metadata
        if table_info.get("metadata"):
            try:
                meta_json = json.dumps(table_info["metadata"], indent=2)
                output.print(Panel(
                    Syntax(meta_json, "json", theme="default", line_numbers=False),
                    title="[bold]Metadata[/bold]",
                    border_style="dim green"
                ))
            except Exception as e:
                output.print(f"[red]Could not display metadata: {escape(str(e))}[/red]")


def _display_key_value_pairs(pairs: Union[Dict, List], output: Console):
    """Display key-value pairs in a table."""
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
    """Display extracted entities."""
    entity_table = Table(title="[bold green]Extracted Entities[/bold green]", box=box.ROUNDED)
    entity_table.add_column("Type", style="cyan")
    entity_table.add_column("Text", style="white")
    entity_table.add_column("Context", style="dim")
    entity_table.add_column("Score", style="green", justify="right")
    
    for entity in entities:
        context_snippet = entity.get("context", "")[:50] + ("..." if len(entity.get("context", "")) > 50 else "")
        score_str = f"{entity.get('score', 0.0):.2f}" if entity.get('score') is not None else "N/A"
        
        entity_table.add_row(
            escape(entity.get("type", "N/A")),
            escape(entity.get("text", "N/A")),
            escape(context_snippet),
            score_str
        )
    
    output.print(entity_table)


def _display_embeddings_info(embeddings: List, model: str, output: Console):
    """Display information about embeddings."""
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
    """Display execution statistics."""
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
                f"{tokens.get('input', 0)} / {tokens.get('output', 0)} / {tokens.get('total', 0)}"
            )
    
    if "processing_time" in result:
        stats_table.add_row("Processing Time", f"{result.get('processing_time', 0.0):.3f}s")
    
    if stats_table.row_count > 0:
        output.print(stats_table)
    
    # Add a blank line after stats
    output.print()


# Specialized display functions for different demo types

def display_embedding_generation_results(results_data: Dict, output: Optional[Console] = None):
    """Display embedding generation results in a table."""
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
            status_str
        )
    
    display.print(results_table)
    display.print()


def display_vector_similarity_results(similarity_data: Dict, output: Optional[Console] = None):
    """Display semantic similarity scores between text pairs."""
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
            if hasattr(score, 'item'):  # Check if it's potentially a numpy scalar
                score = score.item()
        except (AttributeError, TypeError):
            pass
            
        similarity_table.add_row(
            escape(text1),
            escape(text2),
            f"{score:.4f}"
        )
    
    display.print(similarity_table)
    display.print()


def display_analytics_metrics(metrics_data: Dict, output: Optional[Console] = None):
    """Display analytics metrics in an attractive format."""
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
            metrics_table.add_row(
                metric.replace("_", " ").title(),
                str(count),
                ""
            )
    
    # Display table
    output.print(metrics_table)
    
    # Display any grouped metrics
    if "request_distributions" in metrics_data:
        for group_name, distribution in metrics_data["request_distributions"].items():
            distribution_table = Table(
                title=f"[bold]{group_name.replace('_', ' ').title()} Distribution[/bold]",
                box=box.SIMPLE
            )
            distribution_table.add_column("Category", style="cyan")
            distribution_table.add_column("Count", style="green", justify="right")
            distribution_table.add_column("Percentage", style="yellow", justify="right")
            
            total = sum(distribution.values())
            for category, count in distribution.items():
                percentage = (count / total) * 100 if total > 0 else 0
                distribution_table.add_row(
                    category,
                    str(count),
                    f"{percentage:.1f}%"
                )
            
            output.print(distribution_table)

def display_cache_stats(stats: Dict, progression_stats: Optional[Dict[int, Dict]] = None, output: Optional[Console] = None):
    """Display cache statistics in an attractive format.
    
    Args:
        stats: Cache statistics dictionary
        progression_stats: Optional dictionary of stats progression through test runs
        output: Optional console to output to
    """
    # Use provided console or default
    output = output or console
    
    # Display section header
    output.print(Rule("[bold green]Cache Statistics[/bold green]"))
    
    # Create stats table
    stats_table = Table(box=box.ROUNDED, show_header=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white", justify="right")
    
    actual_stats = stats.get("stats", {})
    
    hits = actual_stats.get('hits', 0)
    misses = actual_stats.get('misses', 0)
    total = hits + misses
    hit_ratio = hits / total if total > 0 else 0
    saved_tokens = actual_stats.get('total_saved_tokens', 0)
    saved_cost = actual_stats.get('estimated_cost_savings', 0.0)

    # Display configuration stats
    stats_table.add_row("Cache Enabled", "✅" if stats.get("enabled", False) else "❌")
    stats_table.add_row("TTL", f"{stats.get('ttl', 0):,} seconds")
    stats_table.add_row("Max Entries", f"{stats.get('max_size', 0):,}")
    stats_table.add_row("Current Size", f"{stats.get('size', 0):,}")
    
    # Add a separator
    stats_table.add_section()
    
    # Add performance stats
    stats_table.add_row("Cache Hits", f"{hits:,}")
    stats_table.add_row("Cache Misses", f"{misses:,}")
    stats_table.add_row("Total Lookups", f"{total:,}")
    stats_table.add_row("Hit Ratio", f"{hit_ratio:.2%}")
    stats_table.add_row("Total Saved Tokens", f"{saved_tokens:,}")
    stats_table.add_row("Estimated Cost Savings", f"${saved_cost:.6f}")
    
    output.print(stats_table)
    
    # Display stats progression if provided
    if progression_stats and len(progression_stats) >= 2:
        output.print()
        prog_table = Table(title="Cache Hit Progression", box=box.MINIMAL)
        prog_table.add_column("Stage")
        prog_table.add_column("Hits", justify="right")
        prog_table.add_column("Hit Ratio", justify="right")
        
        for stage, stats_data in sorted(progression_stats.items()):
            stage_hits = stats_data.get('hits', 0)
            stage_misses = stats_data.get('misses', 0)
            stage_total = stage_hits + stage_misses
            stage_ratio = stage_hits / stage_total if stage_total > 0 else 0
            
            if stage == 1:
                stage_name = "After 1st (Miss)"
            elif stage == 2:
                stage_name = "After 2nd (Hit)"
            elif stage == 3:
                stage_name = "After 3rd (Bypass)"
            elif stage == 4:
                stage_name = "After 4th (Hit)"
            else:
                stage_name = f"Stage {stage}"
                
            prog_table.add_row(
                stage_name,
                str(stage_hits),
                f"{stage_ratio:.2%}"
            )
            
        output.print(prog_table)


# --- Tournament Display Functions ---

def display_tournament_status(status_data: Dict[str, Any], output: Optional[Console] = None):
    """Display tournament status with better formatting using Rich.
    
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
        status_table.add_row("Created", status_data.get("created_at", "N/A").replace("T", " ").split(".")[0])
    if "updated_at" in status_data:
        status_table.add_row("Last Updated", status_data.get("updated_at", "N/A").replace("T", " ").split(".")[0])
    
    display.print(status_table)
    
    # Add progress bar visual for better UX
    if status == "RUNNING":
        from rich.progress import BarColumn, Progress, TextColumn
        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
        )
        
        with progress_bar:
            task = progress_bar.add_task("Tournament Progress", total=100, completed=progress)  # noqa: F841
            # Just show the bar visualization, don't actually wait/update

def display_tournament_results(results_data: Dict[str, Any], output: Optional[Console] = None):
    """Display tournament results with better formatting using Rich.
    
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
        expand=False
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="white")

    # Add tournament information
    summary_table.add_row("Tournament Name", escape(results_data.get('config', {}).get('name', 'N/A')))
    summary_table.add_row("Tournament Type", escape(results_data.get('config', {}).get('tournament_type', 'N/A')))
    summary_table.add_row("Final Status", f"[bold green]{escape(results_data.get('status', 'N/A'))}[/bold green]")
    summary_table.add_row("Total Rounds", str(results_data.get('config', {}).get('rounds', 'N/A')))
    
    # Add storage path if available
    storage_path = results_data.get("storage_path")
    summary_table.add_row("Storage Path", escape(storage_path) if storage_path else "[dim]Not available[/dim]")
    
    # Display summary table
    display.print(summary_table)
    
    # Display models used in tournament
    models = results_data.get('config', {}).get('models', [])
    if models:
        model_table = Table(title="[bold]Models Used[/bold]", box=box.SIMPLE, show_header=True)
        model_table.add_column("Provider", style="magenta")
        model_table.add_column("Model", style="blue")
        
        for model_config in models:
            model_id = model_config.get('model_id', 'N/A')
            if ':' in model_id:
                provider, model = model_id.split(':', 1)
                model_table.add_row(provider, model)
            else:
                model_table.add_row("Unknown", model_id)
        
        display.print(model_table)
    
    # Display execution stats if available
    if any(key in results_data for key in ["processing_time", "cost", "tokens"]):
        _display_stats(results_data, display) 