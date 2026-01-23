#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entity relationship graph extraction and visualization demo using Ultimate MCP Server (New Version)."""

import asyncio
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from ultimate_mcp_server.constants import Provider
    from ultimate_mcp_server.tools.entity_relation_graph import (
        COMMON_SCHEMAS,
        HAS_NETWORKX,
        HAS_VISUALIZATION_LIBS,
        GraphStrategy,
        OutputFormat,
        VisualizationFormat,
        extract_entity_graph,
    )
    from ultimate_mcp_server.utils import get_logger
    from ultimate_mcp_server.utils.logging.console import console
except ImportError as e:
    print(f"Error importing Ultimate MCP Server modules: {e}")
    print(
        "Please ensure the script is run from the correct directory or the project path is set correctly."
    )
    sys.exit(1)

# Initialize logger
logger = get_logger("example.entity_graph")  # Updated logger name

# Setup Directories
SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = SCRIPT_DIR / "sample"
OUTPUT_DIR = SCRIPT_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TextDomain(Enum):
    """Domain types for demonstration examples."""

    BUSINESS = "business"
    ACADEMIC = "academic"
    LEGAL = "legal"
    MEDICAL = "medical"
    GENERAL = "general"  # Added for cases without a specific domain schema


# Console instances
main_console = console
# Keep detail_console if needed, but wasn't used in original provided script
# detail_console = Console(width=100, highlight=True)


def display_header(title: str) -> None:
    """Display a section header."""
    main_console.print()
    main_console.print(Rule(f"[bold blue]{title}[/bold blue]"))
    main_console.print()


def display_dataset_info(dataset_path: Path, title: str) -> None:
    """Display information about a dataset."""
    if not dataset_path.exists():
        main_console.print(f"[bold red]Error:[/bold red] Dataset file not found: {dataset_path}")
        return

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Count entities/characters for display
        char_count = len(content)
        word_count = len(content.split())
        # Simple sentence count (approximate)
        sentence_count = content.count(".") + content.count("?") + content.count("!")
        if sentence_count == 0 and char_count > 0:
            sentence_count = 1  # At least one sentence if there's text

        # Preview of the content (first 300 chars)
        preview = escape(content[:300] + "..." if len(content) > 300 else content)

        main_console.print(
            Panel(
                f"[bold cyan]Dataset:[/bold cyan] {dataset_path.name}\n"
                f"[bold cyan]Size:[/bold cyan] {char_count:,} characters | {word_count:,} words | ~{sentence_count:,} sentences\n\n"
                f"[bold cyan]Preview:[/bold cyan]\n{preview}",
                title=title,
                border_style="cyan",
                expand=False,
            )
        )
    except Exception as e:
        main_console.print(
            f"[bold red]Error reading dataset file {dataset_path.name}:[/bold red] {e}"
        )


def display_extraction_params(params: Dict[str, Any]) -> None:
    """Display extraction parameters passed to the tool."""
    param_table = Table(
        title="Extraction Parameters",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    param_table.add_column("Parameter", style="cyan", no_wrap=True)
    param_table.add_column("Value", style="green")

    # Filter parameters to display relevant ones
    display_keys = [
        "provider",
        "model",
        "strategy",
        "domain",
        "output_format",
        "visualization_format",
        "include_evidence",
        "include_attributes",
        "include_positions",
        "include_temporal_info",
        "normalize_entities",
        "max_entities",
        "max_relations",
        "min_confidence",
        "enable_reasoning",
        "language",  # Added language if used
    ]

    for key in display_keys:
        if key in params:
            value = params[key]
            # Format enums and lists nicely
            if isinstance(value, Enum):
                value_str = value.value
            elif isinstance(value, list):
                value_str = (
                    escape(", ".join(str(v) for v in value))
                    if value
                    else "[dim italic]Empty List[/dim italic]"
                )
            elif isinstance(value, bool):
                value_str = "[green]Yes[/green]" if value else "[red]No[/red]"
            elif value is None:
                value_str = "[dim italic]None[/dim italic]"
            else:
                value_str = escape(str(value))

            param_table.add_row(key, value_str)

    main_console.print(param_table)


def display_entity_stats(result: Dict[str, Any]) -> None:
    """Display statistics about extracted entities and relationships."""
    metadata = result.get("metadata", {})
    entities = result.get(
        "entities", []
    )  # Get entities directly for type counting if metadata missing
    relationships = result.get("relationships", [])

    entity_count = metadata.get("entity_count", len(entities))
    relationship_count = metadata.get("relationship_count", len(relationships))

    if entity_count == 0:
        main_console.print("[yellow]No entities found in extraction result.[/yellow]")
        return

    # Use metadata if available, otherwise count manually
    entity_types_meta = metadata.get("entity_types")
    rel_types_meta = metadata.get("relation_types")

    entity_type_counts = {}
    if entity_types_meta:
        for etype in entity_types_meta:
            # Count occurrences in the actual entity list for accuracy
            entity_type_counts[etype] = sum(1 for e in entities if e.get("type") == etype)
    else:  # Fallback if metadata key is missing
        for entity in entities:
            ent_type = entity.get("type", "Unknown")
            entity_type_counts[ent_type] = entity_type_counts.get(ent_type, 0) + 1

    rel_type_counts = {}
    if rel_types_meta:
        for rtype in rel_types_meta:
            rel_type_counts[rtype] = sum(1 for r in relationships if r.get("type") == rtype)
    else:  # Fallback
        for rel in relationships:
            rel_type = rel.get("type", "Unknown")
            rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1

    # Create entity stats table
    stats_table = Table(
        title="Extraction Statistics", box=box.ROUNDED, show_header=True, header_style="bold blue"
    )
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Count", style="green", justify="right")

    stats_table.add_row("Total Entities", str(entity_count))
    stats_table.add_row("Total Relationships", str(relationship_count))

    # Add entity type counts
    stats_table.add_section()
    for ent_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        stats_table.add_row(f"Entity Type: [italic]{escape(ent_type)}[/italic]", str(count))

    # Add relationship type counts (top 5)
    if rel_type_counts:
        stats_table.add_section()
        for rel_type, count in sorted(rel_type_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]:
            stats_table.add_row(
                f"Relationship Type: [italic]{escape(rel_type)}[/italic]", str(count)
            )
        if len(rel_type_counts) > 5:
            stats_table.add_row("[dim]... (other relationship types)[/dim]", "")

    main_console.print(stats_table)


def display_graph_metrics(result: Dict[str, Any]) -> None:
    """Display graph metrics if available in metadata."""
    # Metrics are now nested under metadata
    metrics = result.get("metadata", {}).get("metrics", {})
    if not metrics:
        # Check the top level as a fallback for older structure compatibility if needed
        metrics = result.get("metrics", {})
        if not metrics:
            main_console.print("[dim]No graph metrics calculated (requires networkx).[/dim]")
            return

    metrics_table = Table(
        title="Graph Metrics", box=box.ROUNDED, show_header=True, header_style="bold blue"
    )
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green", justify="right")

    # Add metrics to table
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Improved formatting
            if isinstance(value, float):
                if 0.0001 < abs(value) < 10000:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.3e}"  # Scientific notation for very small/large
            else:
                formatted_value = f"{value:,}"  # Add commas for integers
            metrics_table.add_row(key.replace("_", " ").title(), formatted_value)
        elif value is not None:
            metrics_table.add_row(key.replace("_", " ").title(), escape(str(value)))

    main_console.print(metrics_table)


def display_entities_table(result: Dict[str, Any], limit: int = 10) -> None:
    """Display extracted entities in a table, sorted appropriately."""
    entities = result.get("entities", [])
    if not entities:
        main_console.print("[yellow]No entities found to display.[/yellow]")
        return

    # Sorting based on available metrics from _add_graph_metrics
    # The new tool adds 'degree' and 'centrality'
    sort_key = "name"  # Default sort
    if entities and isinstance(entities[0], dict):
        if "centrality" in entities[0] and any(e.get("centrality", 0) > 0 for e in entities):
            entities.sort(key=lambda x: x.get("centrality", 0.0), reverse=True)
            sort_key = "centrality"
        elif "degree" in entities[0] and any(e.get("degree", 0) > 0 for e in entities):
            entities.sort(key=lambda x: x.get("degree", 0.0), reverse=True)
            sort_key = "degree"
        elif "mentions" in entities[0] and any(e.get("mentions") for e in entities):
            entities.sort(key=lambda x: len(x.get("mentions", [])), reverse=True)
            sort_key = "mentions"
        else:
            entities.sort(key=lambda x: x.get("name", "").lower())  # Fallback to name

    # Limit to top entities
    display_entities = entities[:limit]

    title = f"Top {limit} Entities"
    if sort_key != "name":
        title += f" (Sorted by {sort_key.capitalize()})"

    entity_table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold blue")
    entity_table.add_column("ID", style="dim", width=8)
    entity_table.add_column("Name", style="cyan", max_width=40)
    entity_table.add_column("Type", style="green", max_width=20)

    # Add columns for additional information if available
    has_degree = any(e.get("degree", 0) > 0 for e in display_entities)
    has_centrality = any(e.get("centrality", 0) > 0 for e in display_entities)
    has_mentions = any(e.get("mentions") for e in display_entities)
    has_attributes = any(e.get("attributes") for e in display_entities)

    if has_degree:
        entity_table.add_column("Degree", style="magenta", justify="right", width=8)
    if has_centrality:
        entity_table.add_column("Centrality", style="magenta", justify="right", width=10)
    if has_mentions:
        entity_table.add_column("Mentions", style="yellow", justify="right", width=8)
    if has_attributes:
        entity_table.add_column("Attributes", style="blue", max_width=50)

    # Add rows for each entity
    for entity in display_entities:
        row = [
            escape(entity.get("id", "")),
            escape(entity.get("name", "")),
            escape(entity.get("type", "Unknown")),
        ]

        if has_degree:
            degree = entity.get("degree", 0.0)
            row.append(f"{degree:.3f}")
        if has_centrality:
            centrality = entity.get("centrality", 0.0)
            row.append(f"{centrality:.4f}")
        if has_mentions:
            mentions_count = len(entity.get("mentions", []))
            row.append(str(mentions_count))
        if has_attributes:
            attributes = entity.get("attributes", {})
            # Format attributes more readably
            attr_str = "; ".join(
                f"{k}={v}" for k, v in attributes.items() if v
            )  # Ignore empty values
            row.append(escape(attr_str[:45] + ("..." if len(attr_str) > 45 else "")))

        entity_table.add_row(*row)

    main_console.print(entity_table)

    if len(entities) > limit:
        main_console.print(f"[dim italic]...and {len(entities) - limit} more entities[/dim italic]")


def display_relationships_table(result: Dict[str, Any], limit: int = 10) -> None:
    """Display extracted relationships in a table, sorted by confidence."""
    relationships = result.get("relationships", [])
    # Create entity map for quick name lookups
    entity_map = {
        entity["id"]: entity
        for entity in result.get("entities", [])
        if isinstance(entity, dict) and "id" in entity
    }

    if not relationships:
        main_console.print("[yellow]No relationships found to display.[/yellow]")
        return

    # Sort relationships by confidence (new tool ensures confidence exists)
    relationships.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    # Limit to top relationships
    display_relationships = relationships[:limit]

    rel_table = Table(
        title=f"Top {limit} Relationships (Sorted by Confidence)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue",
    )
    rel_table.add_column("Source", style="cyan", max_width=30)
    rel_table.add_column("Type", style="green", max_width=25)
    rel_table.add_column("Target", style="cyan", max_width=30)
    rel_table.add_column("Conf.", style="magenta", justify="right", width=6)

    # Check if we have evidence or temporal info
    has_evidence = any(r.get("evidence") for r in display_relationships)
    has_temporal = any(r.get("temporal") for r in display_relationships)

    if has_evidence:
        rel_table.add_column("Evidence", style="yellow", max_width=40)
    if has_temporal:
        rel_table.add_column("Temporal", style="blue", max_width=20)

    # Add rows for each relationship
    for rel in display_relationships:
        source_id = rel.get("source", "")
        target_id = rel.get("target", "")

        # Get entity names if available, fallback to ID
        source_name = entity_map.get(source_id, {}).get("name", source_id)
        target_name = entity_map.get(target_id, {}).get("name", target_id)

        row = [
            escape(source_name),
            escape(rel.get("type", "Unknown")),
            escape(target_name),
            f"{rel.get('confidence', 0.0):.2f}",
        ]

        if has_evidence:
            evidence = rel.get("evidence", "")
            row.append(escape(evidence[:35] + ("..." if len(evidence) > 35 else "")))
        if has_temporal:
            temporal = rel.get("temporal", {})
            temp_str = "; ".join(f"{k}={v}" for k, v in temporal.items())
            row.append(escape(temp_str[:18] + ("..." if len(temp_str) > 18 else "")))

        rel_table.add_row(*row)

    main_console.print(rel_table)

    if len(relationships) > limit:
        main_console.print(
            f"[dim italic]...and {len(relationships) - limit} more relationships[/dim italic]"
        )


def display_entity_graph_tree(
    result: Dict[str, Any], max_depth: int = 2, max_children: int = 5
) -> None:
    """Display a tree representation of the entity graph, starting from the most central node."""
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])
    entity_map = {
        entity["id"]: entity for entity in entities if isinstance(entity, dict) and "id" in entity
    }

    if (
        not entities or not relationships or not HAS_NETWORKX
    ):  # Tree view less useful without sorting/structure
        if not HAS_NETWORKX:
            main_console.print(
                "[yellow]Cannot display graph tree: NetworkX library not available for centrality sorting.[/yellow]"
            )
        else:
            main_console.print("[yellow]Cannot display graph tree: insufficient data.[/yellow]")
        return

    # Sort entities by centrality (assuming metrics were calculated)
    entities.sort(key=lambda x: x.get("centrality", 0.0), reverse=True)

    # Get most central entity as root
    if not entities:
        return  # Should not happen if check above passed, but safety
    root_entity = entities[0]
    root_id = root_entity["id"]

    # Create rich Tree
    tree = Tree(
        f"[bold cyan]{escape(root_entity.get('name', root_id))}[/bold cyan] "
        f"([dim italic]ID: {escape(root_id)}, Type: {escape(root_entity.get('type', 'Unknown'))}[/dim italic])"
    )

    # Keep track of edges explored to represent the tree structure
    explored_edges = set()

    # Recursively build tree using BFS approach for levels
    queue = [(root_id, tree, 0)]  # (entity_id, parent_tree_node, current_depth)
    visited_nodes_in_tree = {root_id}  # Prevent cycles *within this tree rendering*

    while queue:
        current_id, parent_node, depth = queue.pop(0)

        if depth >= max_depth:
            continue

        # Find outgoing relationships
        outgoing_rels = [
            r
            for r in relationships
            if r.get("source") == current_id
            and (current_id, r.get("target"), r.get("type")) not in explored_edges
        ]
        outgoing_rels.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

        # Find incoming relationships
        incoming_rels = [
            r
            for r in relationships
            if r.get("target") == current_id
            and (r.get("source"), current_id, r.get("type")) not in explored_edges
        ]
        incoming_rels.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

        # Add outgoing children
        children_count = 0
        for rel in outgoing_rels:
            if children_count >= max_children:
                parent_node.add("[dim italic]... (more outgoing)[/dim]")
                break

            target_id = rel.get("target")
            if target_id and target_id not in visited_nodes_in_tree:  # Avoid cycles in display
                target_entity = entity_map.get(target_id)
                if target_entity:
                    edge_sig = (current_id, target_id, rel.get("type"))
                    explored_edges.add(edge_sig)
                    visited_nodes_in_tree.add(target_id)

                    rel_type = escape(rel.get("type", "related to"))
                    conf = rel.get("confidence", 0.0)
                    target_name = escape(target_entity.get("name", target_id))
                    target_type = escape(target_entity.get("type", "Unknown"))

                    branch_text = (
                        f"-[[green]{rel_type}[/green] ({conf:.1f})]-> "
                        f"[cyan]{target_name}[/cyan] ([dim italic]{target_type}[/dim italic])"
                    )
                    branch = parent_node.add(branch_text)
                    queue.append((target_id, branch, depth + 1))
                    children_count += 1

        # Add incoming children (optional, can make tree busy)
        # Comment out this block if you only want outgoing relationships in the tree
        # children_count = 0
        # for rel in incoming_rels:
        #     if children_count >= max_children // 2: # Show fewer incoming
        #         parent_node.add("[dim italic]... (more incoming)[/dim]")
        #         break
        #     source_id = rel.get("source")
        #     if source_id and source_id not in visited_nodes_in_tree:
        #          source_entity = entity_map.get(source_id)
        #          if source_entity:
        #              edge_sig = (source_id, current_id, rel.get("type"))
        #              explored_edges.add(edge_sig)
        #              visited_nodes_in_tree.add(source_id)
        #
        #              rel_type = escape(rel.get("type", "related to"))
        #              conf = rel.get("confidence", 0.0)
        #              source_name = escape(source_entity.get("name", source_id))
        #              source_type = escape(source_entity.get("type", "Unknown"))
        #
        #              branch_text = (
        #                  f"<-[[red]{rel_type}[/red] ({conf:.1f})]- "
        #                  f"[magenta]{source_name}[/magenta] ([dim italic]{source_type}[/dim italic])"
        #              )
        #              branch = parent_node.add(branch_text)
        #              queue.append((source_id, branch, depth + 1))
        #              children_count += 1

    main_console.print(
        Panel(
            tree,
            title=f"Entity Graph Tree View (Root: {escape(root_entity.get('name', ''))})",
            border_style="blue",
        )
    )


def display_extraction_summary(result: Dict[str, Any]) -> None:
    """Display a summary of the extraction performance and cost."""
    metadata = result.get("metadata", {})
    provider = result.get("provider", "Unknown")
    model = result.get("model", "Unknown")
    tokens = result.get("tokens", {})
    cost = result.get("cost", 0.0)  # Cost is now float
    processing_time = result.get("processing_time", 0.0)  # Time is now float
    strategy = metadata.get("processing_strategy", "Unknown")
    schema_used = metadata.get("schema_used", "Unknown")

    summary_table = Table(box=box.ROUNDED, show_header=False, title="Extraction Summary")
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Provider", escape(provider))
    summary_table.add_row("Model", escape(model))
    summary_table.add_row("Strategy", escape(strategy))
    summary_table.add_row("Schema Used", escape(schema_used))
    summary_table.add_row("Input Tokens", f"{tokens.get('input', 0):,}")
    summary_table.add_row("Output Tokens", f"{tokens.get('output', 0):,}")
    summary_table.add_row("Total Tokens", f"{tokens.get('total', 0):,}")
    summary_table.add_row("Cost", f"${cost:.6f}")
    summary_table.add_row("Processing Time", f"{processing_time:.2f} seconds")

    main_console.print(summary_table)


def save_visualization(
    result: Dict[str, Any], domain: str, strategy: str, output_dir: Path
) -> Optional[str]:
    """Save visualization file based on the format present in the result."""
    visualization = result.get("visualization")  # Visualization data is now under this key
    if not visualization or not isinstance(visualization, dict):
        main_console.print("[dim]No visualization data found in the result.[/dim]")
        return None

    content = None
    extension = None
    file_path = None

    # Check for different visualization formats
    if "html" in visualization:
        content = visualization["html"]
        extension = "html"
    elif "svg" in visualization:
        content = visualization["svg"]
        extension = "svg"
    elif "png_url" in visualization:  # Assuming PNG might save file directly and return URL
        file_path = visualization["png_url"].replace("file://", "")
        extension = "png"
    elif "dot" in visualization:
        content = visualization["dot"]
        extension = "dot"

    if file_path:  # If path was returned directly (like maybe for PNG)
        if Path(file_path).exists():
            return file_path
        else:
            main_console.print(
                f"[red]Visualization file path provided but not found: {file_path}[/red]"
            )
            return None

    if content and extension:
        timestamp = int(time.time())
        # Sanitize domain and strategy for filename
        safe_domain = domain.replace(" ", "_").lower()
        safe_strategy = strategy.replace(" ", "_").lower()
        output_path = output_dir / f"graph_{safe_domain}_{safe_strategy}_{timestamp}.{extension}"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            return str(output_path)
        except Exception as e:
            main_console.print(
                f"[bold red]Error saving visualization file {output_path}:[/bold red] {e}"
            )
            return None
    elif "error" in visualization:
        main_console.print(
            f"[yellow]Visualization generation failed:[/yellow] {visualization['error']}"
        )
        return None
    else:
        main_console.print("[dim]Unsupported or missing visualization format in result.[/dim]")
        return None


async def run_entity_extraction(
    text: str,
    domain: TextDomain,
    strategy: GraphStrategy,
    model: str,
    output_format: OutputFormat = OutputFormat.JSON,
    visualization_format: VisualizationFormat = VisualizationFormat.HTML,  # Keep HTML for demo vis
    provider: str = Provider.ANTHROPIC.value,  # Example provider
) -> Optional[Dict[str, Any]]:
    """Run entity graph extraction with progress indicator and display params."""
    # Setup extraction parameters
    params = {
        "text": text,
        "provider": provider,  # Pass provider name string
        "model": model,
        "strategy": strategy,  # Pass enum directly
        "output_format": output_format,  # Pass enum directly
        "visualization_format": visualization_format,  # Pass enum directly
        # --- Include Flags (consider new defaults) ---
        "include_evidence": True,  # Explicitly keep True for demo
        "include_attributes": True,  # Explicitly keep True for demo
        "include_positions": False,  # Change to False to match new default (saves tokens)
        "include_temporal_info": True,  # Explicitly keep True for demo
        # --- Control Flags ---
        "normalize_entities": True,  # Keep True (new default)
        "enable_reasoning": False,  # Keep False for demo speed (can be enabled)
        # --- Limits ---
        "max_entities": 75,  # Adjusted limits for demo
        "max_relations": 150,
        "min_confidence": 0.55,  # Slightly higher confidence
        # --- Optional ---
        "language": None,  # Specify if needed, e.g., "Spanish"
        "domain": None,  # Set below if applicable
        # "custom_prompt": None, # Add if testing custom prompts
        # "system_prompt": None, # Add if testing system prompts
        # "additional_params": {"temperature": 0.2} # Example
    }

    # Add domain value string if applicable and not GENERAL
    if domain != TextDomain.GENERAL and domain.value in COMMON_SCHEMAS:
        params["domain"] = domain.value

    # Display parameters being used
    display_extraction_params(params)

    # Run extraction with progress spinner
    result = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=main_console,
        transient=False,  # Keep progress visible after completion
    ) as progress:
        task_desc = f"Extracting graph ({domain.value}/{strategy.value}/{model})..."
        task = progress.add_task(task_desc, total=None)

        try:
            start_time = time.monotonic()
            # Pass enums directly now, the tool handles conversion if needed
            result = await extract_entity_graph(**params)  # type: ignore
            end_time = time.monotonic()
            duration = end_time - start_time
            progress.update(
                task,
                completed=100,
                description=f"[green]Extraction complete ({duration:.2f}s)[/green]",
            )
            return result
        except Exception as e:
            logger.error(f"Extraction failed during run_entity_extraction: {e}", exc_info=True)
            progress.update(
                task,
                completed=100,
                description=f"[bold red]Extraction failed: {escape(str(e))}[/bold red]",
            )
            # Optionally re-raise or just return None
            # raise # Uncomment to stop the demo on failure
            return None  # Allow demo to continue


# --- Demonstration Functions (Updated calls to run_entity_extraction) ---


async def demonstrate_domain_extraction(
    domain: TextDomain,
    sample_file: str,
    strategy: GraphStrategy,
    model: str = "claude-3-5-haiku-20241022",  # Default model for demos
    provider: str = Provider.ANTHROPIC.value,
):
    """Helper function to demonstrate extraction for a specific domain."""
    domain_name = domain.value.capitalize()
    display_header(f"{domain_name} Domain Entity Graph Extraction ({strategy.value} strategy)")

    sample_path = SAMPLE_DIR / sample_file
    display_dataset_info(sample_path, f"{domain_name} Sample Text")

    if not sample_path.exists():
        return  # Skip if file missing

    with open(sample_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    try:
        result = await run_entity_extraction(
            text=text_content,
            domain=domain,
            strategy=strategy,
            model=model,
            provider=provider,
            visualization_format=VisualizationFormat.HTML,  # Request HTML for viewing
        )

        if result and result.get("success", False):
            # Display results
            display_entity_stats(result)
            if HAS_NETWORKX:  # Only display metrics if networkx is installed
                display_graph_metrics(result)
            display_entities_table(result)
            display_relationships_table(result)
            if HAS_NETWORKX:  # Tree view also requires networkx
                display_entity_graph_tree(result)

            # Save visualization if available
            vis_path = save_visualization(result, domain.value, strategy.value, OUTPUT_DIR)
            if vis_path:
                main_console.print(
                    f"\n[green]✓[/green] Visualization saved to: [blue link=file://{vis_path}]{vis_path}[/blue]"
                )
            elif "visualization" in result and "error" in result["visualization"]:
                main_console.print(
                    f"[yellow]Visualization generation failed: {result['visualization']['error']}[/yellow]"
                )

            # Display summary
            display_extraction_summary(result)
        elif result:
            main_console.print(
                f"[bold red]Extraction reported failure:[/bold red] {result.get('error', 'Unknown error')}"
            )
        # If result is None, run_entity_extraction already printed the error

    except Exception as e:
        # Catch errors not caught by run_entity_extraction's try/except
        main_console.print(
            f"[bold red]Error during {domain_name} demonstration:[/bold red] {escape(str(e))}"
        )
        logger.error(f"Unhandled error in {domain_name} demo: {e}", exc_info=True)


async def demonstrate_strategy_comparison():
    """Compare different extraction strategies on the same text."""
    display_header("Strategy Comparison")

    # Load business article for comparison
    comparison_file = "article.txt"
    comparison_path = SAMPLE_DIR / comparison_file
    display_dataset_info(comparison_path, f"{comparison_file} (For Strategy Comparison)")

    if not comparison_path.exists():
        return

    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison_text = f.read()

    # Define strategies to compare
    strategies_to_compare = [
        (GraphStrategy.STANDARD, "Standard"),
        (GraphStrategy.MULTISTAGE, "Multistage"),
        (GraphStrategy.CHUNKED, "Chunked"),  # Will process full text if short, or chunk if long
        (GraphStrategy.STRUCTURED, "Structured"),  # Needs examples from domain
        (GraphStrategy.STRICT_SCHEMA, "Strict Schema"),  # Needs domain
    ]

    # Setup comparison table
    comparison_table = Table(
        title="Strategy Comparison Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    comparison_table.add_column("Strategy", style="cyan")
    comparison_table.add_column("Entities", style="green", justify="right")
    comparison_table.add_column("Rels", style="green", justify="right")
    comparison_table.add_column("Time (s)", style="yellow", justify="right")
    comparison_table.add_column("Tokens", style="magenta", justify="right")
    comparison_table.add_column("Cost ($)", style="blue", justify="right")

    # Use a slightly smaller model for faster comparison if needed
    comparison_model = "gpt-4.1-mini"
    comparison_provider = Provider.OPENAI.value
    comparison_domain = TextDomain.BUSINESS

    # Compare each strategy
    for strategy, desc in strategies_to_compare:
        main_console.print(f"\n[bold underline]Running {desc} Strategy[/bold underline]")

        # Use full text for chunking demo, maybe excerpt for others if needed for speed?
        # Let's use full text for all to see chunking effect properly
        text_to_use = comparison_text

        result = None  # Define result outside try block
        try:
            result = await run_entity_extraction(
                text=text_to_use,
                domain=comparison_domain,  # Business domain has examples/schema
                strategy=strategy,
                model=comparison_model,
                provider=comparison_provider,
                visualization_format=VisualizationFormat.NONE,  # Skip visualization for comparison
            )

            if result and result.get("success", False):
                # Extract metrics for comparison
                entity_count = result.get("metadata", {}).get("entity_count", 0)
                rel_count = result.get("metadata", {}).get("relationship_count", 0)
                processing_time = result.get("processing_time", 0.0)
                token_count = result.get("tokens", {}).get("total", 0)
                cost = result.get("cost", 0.0)

                # Add to comparison table
                comparison_table.add_row(
                    desc,
                    str(entity_count),
                    str(rel_count),
                    f"{processing_time:.2f}",
                    f"{token_count:,}",
                    f"{cost:.6f}",
                )
                # Display brief stats for this strategy
                display_entity_stats(result)
            else:
                error_msg = (
                    result.get("error", "Extraction failed")
                    if result
                    else "Extraction returned None"
                )
                main_console.print(
                    f"[bold red]Error with {desc} strategy:[/bold red] {escape(error_msg)}"
                )
                comparison_table.add_row(
                    desc, "[red]ERR[/red]", "[red]ERR[/red]", "N/A", "N/A", "N/A"
                )

        except Exception as e:
            logger.error(f"Unhandled error comparing strategy {desc}: {e}", exc_info=True)
            main_console.print(
                f"[bold red]Unhandled Error with {desc} strategy:[/bold red] {escape(str(e))}"
            )
            comparison_table.add_row(
                desc, "[red]CRASH[/red]", "[red]CRASH[/red]", "N/A", "N/A", "N/A"
            )

    # Display final comparison table
    main_console.print(Rule("Comparison Summary"))
    main_console.print(comparison_table)


async def demonstrate_output_formats():
    """Demonstrate different output formats using a sample text."""
    display_header("Output Format Demonstration")

    # Load academic paper for output format demo
    format_file = "research_paper.txt"
    format_path = SAMPLE_DIR / format_file
    display_dataset_info(format_path, f"{format_file} (For Output Formats)")

    if not format_path.exists():
        return

    with open(format_path, "r", encoding="utf-8") as f:
        format_text = f.read()

    # Define output formats to demonstrate
    # Exclude NetworkX if library not installed
    formats_to_demonstrate = [
        (OutputFormat.JSON, "Standard JSON"),
        (OutputFormat.CYTOSCAPE, "Cytoscape.js"),
        (OutputFormat.NEO4J, "Neo4j Cypher"),
        (OutputFormat.RDF, "RDF Triples"),
        (OutputFormat.D3, "D3.js nodes/links"),
    ]
    if HAS_NETWORKX:
        formats_to_demonstrate.insert(1, (OutputFormat.NETWORKX, "NetworkX Object"))

    main_console.print(
        "[bold yellow]Note:[/bold yellow] This demonstrates how extracted data can be formatted."
    )

    # Use a short excerpt for speed
    text_excerpt = format_text[:2000]
    base_model = "gpt-4.1-mini"  # Faster model for formats
    base_provider = Provider.OPENAI.value
    base_domain = TextDomain.ACADEMIC

    # Extract with each output format
    for fmt, desc in formats_to_demonstrate:
        main_console.print(f"\n[bold underline]Demonstrating {desc} Output Format[/bold underline]")

        result = None  # Define outside try block
        try:
            result = await run_entity_extraction(
                text=text_excerpt,
                domain=base_domain,
                strategy=GraphStrategy.STANDARD,  # Use standard strategy
                model=base_model,
                provider=base_provider,
                output_format=fmt,  # Specify the output format
                visualization_format=VisualizationFormat.NONE,  # No viz needed here
            )

            if not result or not result.get("success", False):
                error_msg = (
                    result.get("error", "Extraction failed")
                    if result
                    else "Extraction returned None"
                )
                main_console.print(
                    f"[bold red]Error extracting data for {desc} format:[/bold red] {escape(error_msg)}"
                )
                continue  # Skip displaying output for this format

            # Display format-specific output key
            output_key = fmt.value  # Default key matches enum value
            if fmt == OutputFormat.NETWORKX:
                output_key = "graph"
            if fmt == OutputFormat.NEO4J:
                output_key = "neo4j_queries"
            if fmt == OutputFormat.RDF:
                output_key = "rdf_triples"
            if fmt == OutputFormat.D3:
                output_key = "d3"
            if fmt == OutputFormat.CYTOSCAPE:
                output_key = "cytoscape"

            if output_key in result:
                data_to_display = result[output_key]
                display_title = f"Sample of {desc} Output (`{output_key}` key)"

                if fmt == OutputFormat.JSON:
                    # Display a subset of the standard JSON keys
                    json_subset = {
                        "entities": result.get("entities", [])[:2],
                        "relationships": result.get("relationships", [])[:2],
                        "metadata": {
                            k: v
                            for k, v in result.get("metadata", {}).items()
                            if k in ["entity_count", "relationship_count", "processing_strategy"]
                        },
                    }
                    output_content = Syntax(
                        json.dumps(json_subset, indent=2),
                        "json",
                        theme="default",
                        line_numbers=True,
                        word_wrap=True,
                    )
                elif fmt == OutputFormat.NETWORKX:
                    graph_obj = result["graph"]
                    info = (
                        f"[green]✓[/green] NetworkX graph object created: {isinstance(graph_obj, nx.DiGraph)}\n"
                        f"Nodes: {graph_obj.number_of_nodes()}, Edges: {graph_obj.number_of_edges()}\n\n"
                        "[italic]Allows graph algorithms (centrality, paths, etc.)[/italic]"
                    )
                    output_content = info  # Simple text panel
                elif fmt == OutputFormat.CYTOSCAPE:
                    sample = {
                        "nodes": data_to_display.get("nodes", [])[:2],
                        "edges": data_to_display.get("edges", [])[:2],
                    }
                    output_content = Syntax(
                        json.dumps(sample, indent=2),
                        "json",
                        theme="default",
                        line_numbers=True,
                        word_wrap=True,
                    )
                elif fmt == OutputFormat.NEO4J:
                    queries = data_to_display
                    sample_queries = queries[:3]  # Show first few queries
                    output_content = Syntax(
                        "\n\n".join(sample_queries) + "\n...",
                        "cypher",
                        theme="default",
                        line_numbers=True,
                        word_wrap=True,
                    )
                elif fmt == OutputFormat.RDF:
                    triples = data_to_display
                    sample_triples = [
                        '("{}", "{}", "{}")'.format(*t) for t in triples[:5]
                    ]  # Format first few
                    output_content = Syntax(
                        "\n".join(sample_triples) + "\n...",
                        "turtle",
                        theme="default",
                        line_numbers=True,
                        word_wrap=True,
                    )  # Turtle isn't perfect but ok
                elif fmt == OutputFormat.D3:
                    sample = {
                        "nodes": data_to_display.get("nodes", [])[:2],
                        "links": data_to_display.get("links", [])[:2],
                    }
                    output_content = Syntax(
                        json.dumps(sample, indent=2),
                        "json",
                        theme="default",
                        line_numbers=True,
                        word_wrap=True,
                    )
                else:
                    # Fallback for unexpected formats
                    output_content = escape(str(data_to_display)[:500] + "...")

                main_console.print(
                    Panel(output_content, title=display_title, border_style="green", expand=False)
                )

            else:
                main_console.print(
                    f"[yellow]Output key '{output_key}' not found in result for {desc} format.[/yellow]"
                )

        except Exception as e:
            logger.error(f"Unhandled error demonstrating format {desc}: {e}", exc_info=True)
            main_console.print(
                f"[bold red]Unhandled Error with {desc} format:[/bold red] {escape(str(e))}"
            )


async def main():
    """Run entity relation graph extraction demonstrations."""
    try:
        # Display welcome message
        main_console.print(
            Rule("[bold magenta]Entity Relationship Graph Extraction Demo (v2 Tool)[/bold magenta]")
        )
        main_console.print(
            "[bold]This demonstrates the refactored `entity_graph` tool for extracting and visualizing "
            "knowledge graphs from text across different domains and using various strategies.[/bold]\n"
        )

        # Check for dependencies needed by the demo display itself
        if not HAS_VISUALIZATION_LIBS:
            main_console.print(
                "[yellow]Warning:[/yellow] `networkx`, `pyvis`, or `matplotlib` not installed."
            )
            main_console.print(
                "Graph metrics, tree view, and some visualizations may be unavailable."
            )
        if not HAS_NETWORKX:
            main_console.print(
                "[yellow]Warning:[/yellow] `networkx` not installed. Graph metrics and tree view disabled."
            )

        # Initialize the Gateway (optional, depends if Gateway context is needed for config/logging)
        # If the tool functions standalone, this might not be strictly necessary for the demo.
        # gateway = Gateway("entity-graph-demo", register_tools=True)
        # logger.info("Ultimate MCP Server initialized (optional for demo).")

        # Check if sample directory exists
        if not SAMPLE_DIR.exists() or not any(SAMPLE_DIR.iterdir()):
            main_console.print(
                f"[bold red]Error:[/bold red] Sample directory '{SAMPLE_DIR}' not found or is empty!"
            )
            main_console.print(
                "Please create the 'sample' directory next to this script and add text files (e.g., article.txt)."
            )
            return 1

        # --- Run Demonstrations ---
        # Define models to use - maybe select based on availability or speed
        # Using Sonnet as a balance, Haiku for comparisons/formats
        default_model = "gpt-4.1-mini"
        default_provider = Provider.OPENAI.value  # String like "anthropic"

        # 1. Domain Examples (using appropriate strategies)
        await demonstrate_domain_extraction(
            TextDomain.BUSINESS,
            "article.txt",
            GraphStrategy.STANDARD,
            model=default_model,
            provider=default_provider,
        )
        await demonstrate_domain_extraction(
            TextDomain.ACADEMIC,
            "research_paper.txt",
            GraphStrategy.MULTISTAGE,
            model=default_model,
            provider=default_provider,
        )
        await demonstrate_domain_extraction(
            TextDomain.LEGAL,
            "legal_contract.txt",
            GraphStrategy.STRUCTURED,
            model=default_model,
            provider=default_provider,
        )
        await demonstrate_domain_extraction(
            TextDomain.MEDICAL,
            "medical_case.txt",
            GraphStrategy.STRICT_SCHEMA,
            model=default_model,
            provider=default_provider,
        )

        # 2. Strategy Comparison
        await demonstrate_strategy_comparison()

        # 3. Output Format Demonstration
        await demonstrate_output_formats()

        main_console.print(
            Rule("[bold green]Entity Relationship Graph Extraction Demo Complete[/bold green]")
        )
        # Split the print into two simpler statements to avoid rich markup issues
        main_console.print(
            f"\n[bold]Visualizations and outputs have been saved to:[/bold] {OUTPUT_DIR}"
        )
        main_console.print("Open any HTML files in a web browser to explore interactive graphs.")

        return 0

    except Exception as e:
        # Catch-all for unexpected errors during setup or top-level execution
        logger.critical(f"Demo failed catastrophically: {e}", exc_info=True)
        main_console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
