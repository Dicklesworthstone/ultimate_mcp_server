#!/usr/bin/env python
"""Demonstration script for SQLTool in Ultimate MCP Server."""

import asyncio
import datetime as dt
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Rich imports for nice UI
import pandas as pd
import pandera as pa
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.traceback import install as install_rich_traceback
from rich.tree import Tree

from ultimate_mcp_server.core.server import Gateway  # Import the actual Gateway
from ultimate_mcp_server.exceptions import ToolError, ToolInputError

# Import the SQLTool class from our module
from ultimate_mcp_server.tools.sql_databases import SQLTool
from ultimate_mcp_server.utils import get_logger

# Initialize Rich console and logger
console = Console()
logger = get_logger("demo.sql_tool")

# Install rich tracebacks for better error display
install_rich_traceback(show_locals=False, width=console.width)

# --- Configuration ---
DEFAULT_CONNECTION_STRING = "sqlite:///:memory:"  # In-memory SQLite for demo
# You can replace with a connection string like:
# "postgresql://username:password@localhost:5432/demo_db"
# "mysql+pymysql://username:password@localhost:3306/demo_db"
# "mssql+pyodbc://username:password@localhost:1433/demo_db?driver=ODBC+Driver+17+for+SQL+Server"

# --- Demo Helper Functions ---


def display_result(title: str, result: Dict[str, Any], query_str: Optional[str] = None) -> None:
    """Display query result with enhanced formatting."""
    console.print(Rule(f"[bold cyan]{escape(title)}[/bold cyan]"))

    if query_str:
        console.print(
            Panel(
                Syntax(
                    query_str.strip(), "sql", theme="default", line_numbers=False, word_wrap=True
                ),
                title="Executed Query",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        console.print(
            Panel(
                f"[bold red]:x: Operation Failed:[/]\n{escape(error_msg)}",
                title="Error",
                border_style="red",
                padding=(1, 2),
                expand=False,
            )
        )
        return

    # Handle different result types based on content
    if "rows" in result:
        # Query result with rows
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        row_count = result.get("row_count", len(rows))

        if not rows:
            console.print(
                Panel(
                    "[yellow]No results returned for this operation.",
                    padding=(0, 1),
                    border_style="yellow",
                )
            )
            return

        table_title = f"Results ({row_count} row{'s' if row_count != 1 else ''} returned)"
        if "pagination" in result:
            pagination = result["pagination"]
            table_title += f" - Page {pagination.get('page', '?')}"

        table = Table(
            title=table_title,
            box=box.ROUNDED,
            show_header=True,
            padding=(0, 1),
            border_style="bright_blue",
        )

        # Add columns
        for name in columns:
            justify = (
                "right"
                if any(
                    k in name.lower()
                    for k in ["id", "count", "price", "amount", "quantity", "total"]
                )
                else "left"
            )
            style = "cyan" if justify == "left" else "magenta"
            table.add_column(name, style=style, justify=justify, header_style=f"bold {style}")

        # Add data rows
        for row in rows:
            table.add_row(*[escape(str(row.get(col_name, ""))) for col_name in columns])

        console.print(table)

        # Display pagination info if available
        if "pagination" in result:
            pagination = result["pagination"]
            pagination_info = Table(
                title="Pagination Info", show_header=False, box=box.SIMPLE, padding=(0, 1)
            )
            pagination_info.add_column("Metric", style="cyan", justify="right")
            pagination_info.add_column("Value", style="white")
            pagination_info.add_row("Page", str(pagination.get("page")))
            pagination_info.add_row("Page Size", str(pagination.get("page_size")))
            pagination_info.add_row(
                "Has Next",
                "[green]:heavy_check_mark:[/]"
                if pagination.get("has_next_page")
                else "[dim]:x:[/]",
            )
            pagination_info.add_row(
                "Has Previous",
                "[green]:heavy_check_mark:[/]"
                if pagination.get("has_previous_page")
                else "[dim]:x:[/]",
            )
            console.print(pagination_info)

        # Show if truncated
        if result.get("truncated"):
            console.print("[yellow]⚠ Results truncated (reached max_rows limit)[/yellow]")

    elif "documentation" in result:
        # Documentation result
        doc_content = result.get("documentation", "")
        format_type = result.get("format", "markdown")

        console.print(
            Panel(
                Syntax(
                    doc_content, format_type, theme="default", line_numbers=False, word_wrap=True
                ),
                title=f"Documentation ({format_type.upper()})",
                border_style="magenta",
                padding=(1, 2),
            )
        )

    else:
        # Generic success result, display as is
        console.print(
            Panel(
                "\n".join(
                    [f"[cyan]{k}:[/] {escape(str(v))}" for k, v in result.items() if k != "success"]
                ),
                title="Operation Result",
                border_style="green",
                padding=(1, 2),
            )
        )

    console.print()  # Add spacing


# Add setup functionality directly to avoid import issues
def init_demo_database(db_path):
    """Set up a demo database with sample tables and data."""
    logger.info(f"Setting up demo database at: {db_path}")

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    setup_queries = [
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            signup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT CHECK(status IN ('active', 'inactive', 'pending')) DEFAULT 'pending',
            ssn TEXT,
            credit_card TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            price DECIMAL(10,2) NOT NULL,
            category TEXT,
            in_stock BOOLEAN DEFAULT 1
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_amount DECIMAL(10,2) NOT NULL,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            price_per_unit DECIMAL(10,2) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
        """,
    ]

    # Insert sample data
    sample_data_queries = [
        # Insert customers with PII data already included
        """
        INSERT INTO customers (customer_id, name, email, status, ssn, credit_card) VALUES
            (1, 'Alice Johnson', 'alice.j@example.com', 'active', '123-45-6789', '4111-1111-1111-1111'),
            (2, 'Bob Smith', 'bob.smith@example.net', 'active', '234-56-7890', '4222-2222-2222-2222'),
            (3, 'Charlie Davis', 'charlie.d@example.org', 'inactive', '345-67-8901', '4333-3333-3333-3333'),
            (4, 'Diana Miller', 'diana.m@example.com', 'active', '456-78-9012', '4444-4444-4444-4444'),
            (5, 'Ethan Garcia', 'ethan.g@sample.net', 'pending', '567-89-0123', '4555-5555-5555-5555')
        """,
        # Insert products
        """
        INSERT INTO products (product_id, name, description, price, category, in_stock) VALUES
            (1, 'Laptop Pro X', 'High-performance laptop with 16GB RAM', 1499.99, 'Electronics', 1),
            (2, 'Smartphone Z', 'Latest flagship smartphone', 999.99, 'Electronics', 1),
            (3, 'Wireless Earbuds', 'Noise-cancelling earbuds', 179.99, 'Audio', 1),
            (4, 'Smart Coffee Maker', 'WiFi-enabled coffee machine', 119.99, 'Kitchen', 0),
            (5, 'Fitness Tracker', 'Waterproof fitness band with GPS', 79.99, 'Wearables', 1)
        """,
        # Insert orders
        """
        INSERT INTO orders (order_id, customer_id, total_amount, status) VALUES
            (1, 1, 1499.98, 'completed'),
            (2, 2, 89.99, 'processing'),
            (3, 1, 249.99, 'completed'),
            (4, 3, 1099.98, 'completed'),
            (5, 4, 49.99, 'processing')
        """,
        # Insert order items
        """
        INSERT INTO order_items (item_id, order_id, product_id, quantity, price_per_unit) VALUES
            (1, 1, 1, 1, 1499.99),
            (2, 2, 5, 1, 79.99),
            (3, 3, 3, 1, 179.99),
            (4, 3, 4, 1, 119.99),
            (5, 4, 2, 1, 999.99),
            (6, 4, 5, 1, 79.99),
            (7, 5, 4, 1, 119.99)
        """,
    ]

    try:
        # Execute each query to set up schema
        for query in setup_queries:
            cursor.execute(query)
            logger.info(f"Created table: {query.strip().split()[2]}")

        # Execute each query to insert data
        for query in sample_data_queries:
            cursor.execute(query)
            table_name = query.strip().split()[2]
            row_count = cursor.rowcount
            logger.info(f"Inserted {row_count} rows into {table_name}")

        # Commit the changes
        conn.commit()
        logger.info("Database setup complete")

    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

    return db_path


# --- Demo Functions ---


async def connection_demo(sql_tool: SQLTool, conn_string: Optional[str] = None) -> Optional[str]:
    """Demonstrate database connection and status checking."""
    console.print(Rule("[bold green]1. Database Connection Demo[/bold green]", style="green"))
    logger.info("Starting database connection demo")

    connection_id = None
    connection_string = conn_string or DEFAULT_CONNECTION_STRING

    with console.status("[bold cyan]Connecting to database...", spinner="earth"):
        try:
            # Connect to database
            connection_result = await sql_tool.manage_database(
                action="connect",
                connection_string=connection_string,
                echo=False,  # Disable SQLAlchemy logging for cleaner output
            )

            if connection_result.get("success"):
                connection_id = connection_result.get("connection_id")
                db_type = connection_result.get("database_type", "Unknown")

                logger.success(f"Connected to database with ID: {connection_id}")
                console.print(
                    Panel(
                        f"Connection ID: [bold cyan]{escape(connection_id)}[/]\n"
                        f"Database Type: [blue]{escape(db_type)}[/]",
                        title="[bold green]:link: Connected[/]",
                        border_style="green",
                        padding=(1, 2),
                        expand=False,
                    )
                )

                # Test the connection
                console.print("[cyan]Testing connection health...[/]")
                test_result = await sql_tool.manage_database(
                    action="test", connection_id=connection_id
                )

                if test_result.get("success"):
                    resp_time = test_result.get("response_time_seconds", 0)
                    version = test_result.get("version", "N/A")
                    console.print(
                        Panel(
                            f"[green]:heavy_check_mark: Connection test OK\n"
                            f"Response time: {resp_time:.4f}s\n"
                            f"DB Version: {version}",
                            border_style="green",
                            padding=(1, 2),
                        )
                    )
                else:
                    console.print(
                        Panel(
                            f"[bold red]:x: Connection test failed:[/]\n{escape(test_result.get('error', 'Unknown error'))}",
                            border_style="red",
                            padding=(1, 2),
                        )
                    )

                # Get connection status
                console.print("[cyan]Fetching database status...[/]")
                status_result = await sql_tool.manage_database(
                    action="status", connection_id=connection_id
                )

                if status_result.get("success"):
                    status_table = Table(
                        title="Active Connections",
                        box=box.HEAVY,
                        padding=(0, 1),
                        border_style="blue",
                    )
                    status_table.add_column("Connection ID", style="cyan")
                    status_table.add_column("Database", style="blue")
                    status_table.add_column("Last Accessed", style="dim")
                    status_table.add_column("Idle Time", style="yellow")

                    connections = status_result.get("connections", {})
                    for conn_id, conn_info in connections.items():
                        status_table.add_row(
                            conn_id,
                            conn_info.get("dialect", "unknown"),
                            conn_info.get("last_accessed", "N/A"),
                            f"{conn_info.get('idle_time_seconds', 0):.1f}s",
                        )

                    console.print(status_table)
                else:
                    console.print(
                        Panel(
                            f"[bold red]:x: Failed to get database status:[/]\n{escape(status_result.get('error', 'Unknown error'))}",
                            border_style="red",
                            padding=(1, 2),
                        )
                    )
            else:
                error_msg = connection_result.get("error", "Unknown error")
                logger.error(f"Failed to connect to database: {error_msg}")
                console.print(
                    Panel(
                        f"[bold red]:x: Connection failed:[/]\n{escape(error_msg)}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )

        except Exception as e:
            logger.error(f"Unexpected error in connection demo: {e}")
            console.print(f"[bold red]:x: Unexpected Error:[/]\n{escape(str(e))}")

    console.print()  # Spacing
    return connection_id


async def schema_discovery_demo(sql_tool: SQLTool, connection_id: str) -> None:
    """Demonstrate database schema discovery."""
    console.print(Rule("[bold green]2. Schema Discovery Demo[/bold green]", style="green"))
    logger.info("Starting schema discovery demo")

    with console.status("[bold cyan]Discovering database schema...", spinner="dots"):
        try:
            schema_result = await sql_tool.explore_database(
                connection_id=connection_id,
                action="schema",
                include_indexes=True,
                include_foreign_keys=True,
                detailed=True,
            )

            if schema_result.get("success"):
                tables = schema_result.get("tables", [])
                views = schema_result.get("views", [])
                relationships = schema_result.get("relationships", [])

                logger.success(
                    f"Schema discovered: {len(tables)} tables, {len(views)} views, {len(relationships)} relationships"
                )

                # Create a tree visualization
                tree = Tree(
                    f"[bold bright_blue]:database: Database Schema ({len(tables)} Tables, {len(views)} Views)[/]",
                    guide_style="bright_blue",
                )

                # Add Tables branch
                if tables:
                    tables_branch = tree.add("[bold cyan]:page_facing_up: Tables[/]")
                    for table in tables:
                        table_name = table.get("name", "Unknown")
                        table_node = tables_branch.add(f"[cyan]{escape(table_name)}[/]")

                        # Add columns
                        cols = table.get("columns", [])
                        if cols:
                            cols_branch = table_node.add(
                                "[bold yellow]:heavy_minus_sign: Columns[/]"
                            )
                            for col in cols:
                                col_name = col.get("name", "?")
                                col_type = col.get("type", "?")
                                is_pk = col.get("primary_key", False)
                                is_nullable = col.get("nullable", True)
                                pk_str = " [bold magenta](PK)[/]" if is_pk else ""
                                null_str = "" if is_nullable else " [dim]NOT NULL[/]"
                                cols_branch.add(
                                    f"[yellow]{escape(col_name)}[/]: {escape(col_type)}{pk_str}{null_str}"
                                )

                        # Add foreign keys
                        fks = table.get("foreign_keys", [])
                        if fks:
                            fks_branch = table_node.add("[bold blue]:link: Foreign Keys[/]")
                            for fk in fks:
                                ref_table = fk.get("referred_table", "?")
                                con_cols = ", ".join(fk.get("constrained_columns", []))
                                ref_cols = ", ".join(fk.get("referred_columns", []))
                                fks_branch.add(
                                    f"[blue]({escape(con_cols)})[/] -> [cyan]{escape(ref_table)}[/]({escape(ref_cols)})"
                                )

                # Add Views branch
                if views:
                    views_branch = tree.add("[bold magenta]:scroll: Views[/]")
                    for view in views:
                        view_name = view.get("name", "Unknown")
                        views_branch.add(f"[magenta]{escape(view_name)}[/]")

                console.print(
                    Panel(tree, title="Schema Overview", border_style="bright_blue", padding=(1, 2))
                )

                # Show schema hash if available
                if schema_hash := schema_result.get("schema_hash"):
                    console.print(f"[dim]Schema Hash: {schema_hash}[/dim]")
            else:
                error_msg = schema_result.get("error", "Unknown error")
                logger.error(f"Failed to discover schema: {error_msg}")
                console.print(
                    Panel(
                        f"[bold red]:x: Schema discovery failed:[/]\n{escape(error_msg)}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )

        except Exception as e:
            logger.error(f"Unexpected error in schema discovery demo: {e}")
            console.print(f"[bold red]:x: Unexpected Error:[/]\n{escape(str(e))}")

    console.print()  # Spacing


async def table_details_demo(sql_tool: SQLTool, connection_id: str, table_name: str) -> None:
    """Demonstrate getting detailed information about a specific table."""
    console.print(
        Rule(
            f"[bold green]3. Table Details: [cyan]{escape(table_name)}[/cyan][/bold green]",
            style="green",
        )
    )
    logger.info(f"Getting details for table: {table_name}")

    try:
        table_result = await sql_tool.explore_database(
            connection_id=connection_id,
            action="table",
            table_name=table_name,
            include_sample_data=True,
            sample_size=3,
            include_statistics=True,
        )

        if table_result.get("success"):
            logger.success(f"Successfully retrieved details for table: {table_name}")
            console.print(
                Panel(
                    f"[green]:heavy_check_mark: Details retrieved for [cyan]{escape(table_name)}[/]",
                    border_style="green",
                    padding=(0, 1),
                )
            )

            # Display columns
            columns = table_result.get("columns", [])
            if columns:
                cols_table = Table(
                    title="Columns",
                    box=box.ROUNDED,
                    show_header=True,
                    padding=(0, 1),
                    border_style="yellow",
                )
                cols_table.add_column("Name", style="yellow", header_style="bold yellow")
                cols_table.add_column("Type", style="white")
                cols_table.add_column("Nullable", style="dim")
                cols_table.add_column("PK", style="magenta")
                cols_table.add_column("Default", style="dim")

                for column in columns:
                    cols_table.add_row(
                        escape(column.get("name", "?")),
                        escape(column.get("type", "?")),
                        ":heavy_check_mark:" if column.get("nullable", False) else ":x:",
                        "[bold magenta]:key:[/]" if column.get("primary_key", False) else "",
                        escape(str(column.get("default", ""))),
                    )
                console.print(cols_table)

            # Display sample data
            sample_data = table_result.get("sample_data", {})
            sample_rows = sample_data.get("rows", [])
            sample_cols = sample_data.get("columns", [])

            if sample_rows:
                sample_table = Table(
                    title="Sample Data (first 3 rows)",
                    box=box.ROUNDED,
                    show_header=True,
                    padding=(0, 1),
                    border_style="green",
                )
                for col_name in sample_cols:
                    sample_table.add_column(col_name, style="dim cyan", header_style="bold cyan")

                for row in sample_rows:
                    sample_table.add_row(*[escape(str(row.get(col, ""))) for col in sample_cols])

                console.print(sample_table)

            # Display row count
            row_count = table_result.get("row_count", "N/A")
            console.print(f"[cyan]Total Rows:[/] [yellow]{row_count}[/yellow]")

            # Display statistics if available
            statistics = table_result.get("statistics", {})
            if statistics:
                stats_table = Table(
                    title="Column Statistics",
                    box=box.SIMPLE,
                    show_header=True,
                    padding=(0, 1),
                    border_style="magenta",
                )
                stats_table.add_column("Column", style="cyan")
                stats_table.add_column("Null Count", style="yellow", justify="right")
                stats_table.add_column("Distinct Count", style="blue", justify="right")

                for col_name, stats in statistics.items():
                    if isinstance(stats, dict) and "error" not in stats:
                        null_count = stats.get("null_count", "N/A")
                        distinct_count = stats.get("distinct_count", "N/A")
                        stats_table.add_row(escape(col_name), str(null_count), str(distinct_count))

                console.print(stats_table)
        else:
            error_msg = table_result.get("error", "Unknown error")
            logger.error(f"Failed to get table details: {error_msg}")
            console.print(
                Panel(
                    f"[bold red]:x: Failed to get table details:[/]\n{escape(error_msg)}",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    except Exception as e:
        logger.error(f"Unexpected error in table details demo: {e}")
        console.print(f"[bold red]:x: Unexpected Error:[/]\n{escape(str(e))}")

    console.print()  # Spacing


async def find_related_tables_demo(sql_tool: SQLTool, connection_id: str, table_name: str) -> None:
    """Demonstrate finding tables related to a specific table."""
    console.print(
        Rule(
            f"[bold green]4. Related Tables: [cyan]{escape(table_name)}[/cyan][/bold green]",
            style="green",
        )
    )
    logger.info(f"Finding tables related to {table_name}")

    try:
        relations_result = await sql_tool.explore_database(
            connection_id=connection_id,
            action="relationships",
            table_name=table_name,
            depth=2,  # Explore relationships to depth 2
        )

        if relations_result.get("success"):
            rel_graph = relations_result.get("relationship_graph", {})
            parents = rel_graph.get("parents", [])
            children = rel_graph.get("children", [])

            if parents or children:
                logger.success(f"Found relationships for table: {table_name}")

                # Create tree visualization
                rel_tree = Tree(
                    f"[bold blue]:link: Relationships for [cyan]{escape(table_name)}[/][/]",
                    guide_style="blue",
                )

                # Add parent relationships (tables referenced by this table)
                if parents:
                    parent_branch = rel_tree.add("[bold green]:arrow_up: References (Parents)[/]")
                    for parent in parents:
                        relationship = parent.get("relationship", "")
                        target = parent.get("target", {})
                        target_table = target.get("table", "?")
                        parent_branch.add(
                            f"[blue]{escape(relationship)}[/] -> [green]{escape(target_table)}[/]"
                        )

                # Add child relationships (tables that reference this table)
                if children:
                    child_branch = rel_tree.add(
                        "[bold magenta]:arrow_down: Referenced By (Children)[/]"
                    )
                    for child in children:
                        relationship = child.get("relationship", "")
                        source = child.get("source", {})
                        source_table = source.get("table", "?")
                        child_branch.add(
                            f"[magenta]{escape(source_table)}[/] -> [blue]{escape(relationship)}[/]"
                        )

                console.print(
                    Panel(
                        rel_tree, title="Table Relationships", border_style="blue", padding=(1, 2)
                    )
                )
            else:
                logger.info(f"No direct relationships found for {table_name}")
                console.print(
                    Panel(
                        f"[yellow]No direct relationships found for '{escape(table_name)}'",
                        border_style="yellow",
                        padding=(0, 1),
                    )
                )
        else:
            error_msg = relations_result.get("error", "Unknown error")
            logger.error(f"Failed to find relationships: {error_msg}")
            console.print(
                Panel(
                    f"[bold red]:x: Failed to find relationships:[/]\n{escape(error_msg)}",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    except Exception as e:
        logger.error(f"Unexpected error in relationship discovery demo: {e}")
        console.print(f"[bold red]:x: Unexpected Error:[/]\n{escape(str(e))}")

    console.print()  # Spacing


async def column_statistics_demo(
    sql_tool: SQLTool, connection_id: str, table_name: str, column_name: str
) -> None:
    """Demonstrate detailed column statistics."""
    console.print(
        Rule(
            f"[bold green]5. Column Statistics: [cyan]{escape(table_name)}.[yellow]{escape(column_name)}[/yellow][/cyan][/bold green]",
            style="green",
        )
    )
    logger.info(f"Analyzing statistics for column {table_name}.{column_name}")

    try:
        stats_result = await sql_tool.explore_database(
            connection_id=connection_id,
            action="column",
            table_name=table_name,
            column_name=column_name,
            histogram=True,
            num_buckets=8,
        )

        if stats_result.get("success"):
            logger.success(f"Successfully analyzed statistics for {table_name}.{column_name}")

            # Display basic statistics
            statistics = stats_result.get("statistics", {})
            if statistics:
                stats_table = Table(
                    title=f"Statistics for {column_name}",
                    box=box.ROUNDED,
                    show_header=False,
                    padding=(1, 1),
                    border_style="cyan",
                )
                stats_table.add_column("Metric", style="cyan", justify="right")
                stats_table.add_column("Value", style="white")

                for key, value in statistics.items():
                    stats_table.add_row(key.replace("_", " ").title(), str(value))

                console.print(stats_table)

            # Display histogram if available
            histogram = stats_result.get("histogram", {})
            buckets = histogram.get("buckets", [])

            if buckets:
                console.print("[bold cyan]Value Distribution:[/]")

                # Find the max count for scaling
                max_count = max(bucket.get("count", 0) for bucket in buckets)

                # Create a progress bar visualization for the histogram
                progress = Progress(
                    TextColumn("[cyan]{task.description}", justify="right"),
                    BarColumn(bar_width=40),
                    TextColumn("[magenta]{task.fields[count]} ({task.percentage:>3.1f}%)"),
                )

                with progress:
                    for bucket in buckets:
                        label = bucket.get("range", "?")
                        count = bucket.get("count", 0)
                        percentage = (count / max_count) * 100 if max_count > 0 else 0

                        # Add a task for this bucket
                        progress.add_task(
                            description=escape(str(label)),
                            total=100,
                            completed=percentage,
                            count=count,
                        )
        else:
            error_msg = stats_result.get("error", "Unknown error")
            logger.error(f"Failed to analyze column statistics: {error_msg}")
            console.print(
                Panel(
                    f"[bold red]:x: Failed to analyze column statistics:[/]\n{escape(error_msg)}",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    except Exception as e:
        logger.error(f"Unexpected error in column statistics demo: {e}")
        console.print(f"[bold red]:x: Unexpected Error:[/]\n{escape(str(e))}")

    console.print()  # Spacing


async def query_execution_demo(sql_tool: SQLTool, connection_id: str) -> None:
    """Demonstrate query execution capabilities."""
    console.print(Rule("[bold green]6. Query Execution Demo[/bold green]", style="green"))
    logger.info("Demonstrating query execution capabilities")

    try:
        # Simple SELECT query
        simple_query = (
            "SELECT customer_id, name, email, status FROM customers WHERE status = 'active'"
        )
        logger.info("Executing simple query...")

        with console.status("[cyan]Running simple query...[/]"):
            query_result = await sql_tool.execute_sql(
                connection_id=connection_id, query=simple_query, read_only=True, max_rows=10
            )

        display_result("Simple Query: Active Customers", query_result, query_str=simple_query)

        # Parameterized query
        param_query = "SELECT product_id, name, price FROM products WHERE category = :category AND price < :max_price ORDER BY price DESC"
        params = {"category": "Electronics", "max_price": 1000.00}

        logger.info(f"Executing parameterized query with params: {params}")

        with console.status("[cyan]Running parameterized query...[/]"):
            param_result = await sql_tool.execute_sql(
                connection_id=connection_id, query=param_query, parameters=params, read_only=True
            )

        display_result(
            "Parameterized Query: Electronics under $1000", param_result, query_str=param_query
        )

        # Pagination query
        pagination_query = (
            "SELECT product_id, name, category, price FROM products ORDER BY price DESC"
        )
        logger.info("Executing query with pagination (Page 1)")

        with console.status("[cyan]Running paginated query (Page 1)...[/]"):
            pagination_result_p1 = await sql_tool.execute_sql(
                connection_id=connection_id,
                query=pagination_query,
                pagination={"page": 1, "page_size": 2},
                read_only=True,
            )

        display_result(
            "Paginated Query: Products by Price (Page 1)",
            pagination_result_p1,
            query_str=pagination_query,
        )

        # Pagination page 2
        logger.info("Executing query with pagination (Page 2)")

        with console.status("[cyan]Running paginated query (Page 2)...[/]"):
            pagination_result_p2 = await sql_tool.execute_sql(
                connection_id=connection_id,
                query=pagination_query,
                pagination={"page": 2, "page_size": 2},
                read_only=True,
            )

        display_result("Paginated Query: Products by Price (Page 2)", pagination_result_p2)

        # Join query with multiple tables
        join_query = """
        SELECT c.name AS customer_name, o.order_id, o.order_date, o.total_amount, o.status
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE c.status = 'active'
        ORDER BY o.order_date DESC
        """

        logger.info("Executing join query")

        with console.status("[cyan]Running join query...[/]"):
            join_result = await sql_tool.execute_sql(
                connection_id=connection_id, query=join_query, read_only=True
            )

        display_result("Join Query: Orders by Active Customers", join_result, query_str=join_query)

    except Exception as e:
        logger.error(f"Unexpected error in query execution demo: {e}")
        console.print(f"[bold red]:x: Unexpected Error:[/]\n{escape(str(e))}")

    console.print()  # Spacing


async def nl_to_sql_demo(sql_tool: SQLTool, connection_id: str) -> None:
    """Demonstrate natural language to SQL conversion."""
    console.print(Rule("[bold green]7. Natural Language to SQL Demo[/bold green]", style="green"))
    logger.info("Demonstrating natural language to SQL conversion")

    try:
        # Example NL query
        natural_language = "Show me all active customers and their total order value"

        logger.info(f"Converting natural language to SQL: '{natural_language}'")

        with console.status("[cyan]Converting natural language to SQL...[/]"):
            nl_result = await sql_tool.execute_sql(
                connection_id=connection_id, natural_language=natural_language, read_only=True
            )

        if nl_result.get("success"):
            generated_sql = nl_result.get("generated_sql", "")
            confidence = nl_result.get("confidence", 0.0)

            # Display the generated SQL and confidence
            console.print(
                Panel(
                    Syntax(
                        generated_sql, "sql", theme="default", line_numbers=False, word_wrap=True
                    ),
                    title=f"Generated SQL (Confidence: {confidence:.2f})",
                    border_style="green",
                    padding=(1, 2),
                )
            )

            # Display the query results
            display_result("Natural Language Query Results", nl_result)
        else:
            error_msg = nl_result.get("error", "Unknown error")
            logger.error(f"Failed to convert natural language to SQL: {error_msg}")
            console.print(
                Panel(
                    f"[bold red]:x: Natural language conversion failed:[/]\n{escape(error_msg)}",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # Try another more complex example
        complex_nl = "What's the average price of products by category?"

        logger.info(f"Converting complex natural language to SQL: '{complex_nl}'")

        with console.status("[cyan]Converting complex natural language to SQL...[/]"):
            complex_result = await sql_tool.execute_sql(
                connection_id=connection_id, natural_language=complex_nl, read_only=True
            )

        if complex_result.get("success"):
            generated_sql = complex_result.get("generated_sql", "")
            confidence = complex_result.get("confidence", 0.0)

            console.print(
                Panel(
                    Syntax(
                        generated_sql, "sql", theme="default", line_numbers=False, word_wrap=True
                    ),
                    title=f"Generated SQL for complex query (Confidence: {confidence:.2f})",
                    border_style="green",
                    padding=(1, 2),
                )
            )

            display_result("Complex Natural Language Query Results", complex_result)
        else:
            error_msg = complex_result.get("error", "Unknown error")
            logger.error(f"Failed to convert complex natural language to SQL: {error_msg}")
            console.print(
                Panel(
                    f"[bold red]:x: Complex natural language conversion failed:[/]\n{escape(error_msg)}",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    except Exception as e:
        logger.error(f"Unexpected error in NL to SQL demo: {e}")
        console.print(f"[bold red]:x: Unexpected Error:[/]\n{escape(str(e))}")

    console.print()  # Spacing


async def documentation_demo(sql_tool: SQLTool, connection_id: str) -> None:
    """Demonstrate database documentation generation."""
    console.print(Rule("[bold green]8. Database Documentation Demo[/bold green]", style="green"))
    logger.info("Demonstrating database documentation generation")

    try:
        # Generate database documentation
        logger.info("Generating database documentation")

        with console.status("[cyan]Generating database documentation...[/]"):
            doc_result = await sql_tool.explore_database(
                connection_id=connection_id, action="documentation", output_format="markdown"
            )

        if doc_result.get("success"):
            logger.success("Successfully generated database documentation")

            # Display the documentation
            display_result("Database Documentation", doc_result)

            # Optionally save to file
            documentation = doc_result.get("documentation", "")
            if documentation:
                # Create a temporary file to save the documentation
                fd, doc_path = tempfile.mkstemp(suffix=".md", prefix="db_doc_")
                os.close(fd)

                with open(doc_path, "w") as f:
                    f.write(documentation)

                console.print(f"[green]Documentation saved to: [cyan]{doc_path}[/cyan][/green]")
        else:
            error_msg = doc_result.get("error", "Unknown error")
            logger.error(f"Failed to generate documentation: {error_msg}")
            console.print(
                Panel(
                    f"[bold red]:x: Documentation generation failed:[/]\n{escape(error_msg)}",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    except Exception as e:
        logger.error(f"Unexpected error in documentation demo: {e}")
        console.print(f"[bold red]:x: Unexpected Error:[/]\n{escape(str(e))}")

    console.print()  # Spacing


async def security_features_demo(sql_tool: SQLTool, connection_id: str) -> None:
    """Demonstrate security features of the SQLTool."""
    console.print(Rule("[bold green]10. Security Features Demo[/bold green]", style="green"))
    logger.info("Demonstrating security features")

    # --- PII MASKING DEMO ---
    console.print(Rule("[bold blue]10.1 PII Data Masking[/bold blue]", style="blue"))
    logger.info("Demonstrating PII data masking")

    try:
        console.print("[green]PII test data added successfully.[/]")
        # Now run a query to show masked PII data
        pii_select_query = """
        SELECT customer_id, name, email, ssn, credit_card 
        FROM customers 
        ORDER BY customer_id
        """
        with console.status("[cyan]Executing query with PII data...[/]"):
            pii_result = await sql_tool.execute_sql(
                connection_id=connection_id, query=pii_select_query, read_only=True
            )

        display_result(
            "PII Masking Demo: Automatically Masked Sensitive Data", pii_result, pii_select_query
        )

        console.print(
            Panel(
                "Notice how the [bold]SSN[/bold], [bold]credit card numbers[/bold], and [bold]email addresses[/bold] are "
                "automatically masked according to SQLTool's masking rules, protecting sensitive information.",
                title="PII Masking Explanation",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # --- PROHIBITED STATEMENT DETECTION DEMO ---
        console.print(
            Rule("[bold blue]10.2 Prohibited Statement Detection[/bold blue]", style="blue")
        )
        logger.info("Demonstrating prohibited statement detection")

        # List of prohibited statements to test
        prohibited_queries = [
            "DROP TABLE customers",
            "DELETE FROM products",
            "TRUNCATE TABLE orders",
            "ALTER TABLE customers DROP COLUMN name",
            "GRANT ALL PRIVILEGES ON products TO user",
            "CREATE USER hacker WITH PASSWORD 'password'",
        ]

        prohibited_table = Table(
            title="Prohibited Statement Detection",
            box=box.ROUNDED,
            show_header=True,
            padding=(0, 1),
            border_style="red",
        )
        prohibited_table.add_column("Prohibited SQL", style="yellow")
        prohibited_table.add_column("Result", style="green")

        for query in prohibited_queries:
            try:
                with console.status(f"[cyan]Testing: {query}[/]"):
                    await sql_tool.execute_sql(
                        connection_id=connection_id, query=query, read_only=True
                    )
                # If we get here, protection failed (no exception was raised)
                prohibited_table.add_row(query, "[red]FAILED - Statement was allowed![/]")
            except ToolError as e:
                # This is expected behavior - statement should be blocked
                prohibited_table.add_row(
                    query, f"[green]SUCCESS - Blocked: {str(e).split(':')[0]}[/]"
                )
            except Exception as e:
                prohibited_table.add_row(query, f"[yellow]ERROR: {str(e)[:50]}...[/]")

        console.print(prohibited_table)

        # --- ACL CONTROLS DEMO ---
        console.print(Rule("[bold blue]10.3 Access Control Lists (ACL)[/bold blue]", style="blue"))
        logger.info("Demonstrating ACL controls")

        # Set up ACL restrictions
        console.print("[cyan]Setting up ACL restrictions...[/]")
        # We'll restrict access to the 'customers' table and the 'credit_card' column
        sql_tool.update_acl(tables=["customers"], columns=["credit_card", "ssn"])

        console.print(
            Panel(
                "Access control lists configured:\n"
                "- Restricted tables: [red]customers[/]\n"
                "- Restricted columns: [red]credit_card, ssn[/]",
                title="ACL Configuration",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        # Try to access restricted table
        restricted_table_query = "SELECT * FROM customers"
        console.print("\n[cyan]Attempting to query restricted table:[/]")
        console.print(Syntax(restricted_table_query, "sql", theme="default"))

        try:
            with console.status("[cyan]Executing query on restricted table...[/]"):
                await sql_tool.execute_sql(
                    connection_id=connection_id, query=restricted_table_query, read_only=True
                )
            console.print("[red]ACL FAILURE: Query was allowed on restricted table![/]")
        except ToolError as e:
            console.print(
                Panel(
                    f"[green]✅ ACL WORKING: Access denied as expected:[/]\n{escape(str(e))}",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        # Try to access restricted column
        restricted_column_query = (
            "SELECT customer_id, name, credit_card FROM products JOIN customers USING(customer_id)"
        )
        console.print("\n[cyan]Attempting to query restricted column:[/]")
        console.print(Syntax(restricted_column_query, "sql", theme="default"))

        try:
            with console.status("[cyan]Executing query with restricted column...[/]"):
                await sql_tool.execute_sql(
                    connection_id=connection_id, query=restricted_column_query, read_only=True
                )
            console.print("[red]ACL FAILURE: Query was allowed with restricted column![/]")
        except ToolError as e:
            console.print(
                Panel(
                    f"[green]✅ ACL WORKING: Access denied as expected:[/]\n{escape(str(e))}",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        # Clear ACL restrictions for further demos
        sql_tool.update_acl(tables=[], columns=[])
        console.print("[cyan]ACL restrictions cleared for following demos.[/]")

        # --- SCHEMA DRIFT DETECTION ---
        console.print(Rule("[bold blue]10.4 Schema Drift Detection[/bold blue]", style="blue"))
        logger.info("Demonstrating schema drift detection")

        # First run schema discovery to capture initial state
        console.print("[cyan]Capturing initial schema state...[/]")

        with console.status("[cyan]Performing initial schema discovery...[/]"):
            initial_schema = await sql_tool.explore_database(
                connection_id=connection_id,
                action="schema",
                include_indexes=True,
                include_foreign_keys=True,
            )

        initial_hash = initial_schema.get("schema_hash", "unknown")
        console.print(
            f"[green]Initial schema captured with hash: [bold]{initial_hash[:16]}...[/][/]"
        )

        # Now make a schema change
        schema_change_query = "ALTER TABLE products ADD COLUMN last_updated TIMESTAMP"

        console.print("[cyan]Making a schema change...[/]")
        console.print(Syntax(schema_change_query, "sql", theme="default"))

        with console.status("[cyan]Executing schema change...[/]"):
            # Execute the schema change
            await sql_tool.execute_sql(
                connection_id=connection_id,
                query=schema_change_query,
                read_only=False,  # Need to disable read-only for ALTER TABLE
            )

        # Now run schema discovery again to detect the change
        with console.status("[cyan]Performing follow-up schema discovery to detect changes...[/]"):
            new_schema = await sql_tool.explore_database(
                connection_id=connection_id,
                action="schema",
                include_indexes=True,
                include_foreign_keys=True,
            )

        new_hash = new_schema.get("schema_hash", "unknown")
        schema_changed = new_schema.get("schema_change_detected", False)

        if initial_hash != new_hash:
            console.print(
                Panel(
                    f"[green]✅ SCHEMA DRIFT DETECTED:[/]\n"
                    f"- Initial hash: [dim]{initial_hash[:16]}...[/]\n"
                    f"- New hash: [bold]{new_hash[:16]}...[/]\n"
                    f"- Change detected by system: {'[green]Yes[/]' if schema_changed else '[red]No[/]'}",
                    title="Schema Drift Detection Result",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel(
                    "[red]Schema drift detection did not identify a change in hash even though schema was modified.[/]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    except Exception as e:
        logger.error(f"Error in security features demo: {e}", exc_info=True)
        console.print(
            Panel(
                f"[bold red]Error in security features demo:[/]\n{escape(str(e))}",
                border_style="red",
                padding=(1, 2),
            )
        )

    console.print()  # Spacing


async def advanced_export_demo(sql_tool: SQLTool, connection_id: str) -> None:
    """Demonstrate advanced export options."""
    console.print(Rule("[bold green]11. Advanced Export Options Demo[/bold green]", style="green"))
    logger.info("Demonstrating advanced export options")

    # Query to export
    export_query = """
    SELECT p.product_id, p.name AS product_name, p.category, p.price,
           SUM(oi.quantity) AS units_sold,
           SUM(oi.quantity * oi.price_per_unit) AS total_revenue
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.product_id, p.name, p.category, p.price
    ORDER BY total_revenue DESC
    """

    try:
        # --- PANDAS DATAFRAME EXPORT ---
        console.print(Rule("[bold blue]11.1 Pandas DataFrame Export[/bold blue]", style="blue"))
        logger.info("Demonstrating Pandas DataFrame export")

        console.print(Syntax(export_query, "sql", theme="default", line_numbers=False))

        with console.status("[cyan]Executing query and exporting to Pandas DataFrame...[/]"):
            df_result = await sql_tool.execute_sql(
                connection_id=connection_id,
                query=export_query,
                read_only=True,
                export={"format": "pandas"},
            )

        if df_result.get("success") and "dataframe" in df_result:
            df = df_result["dataframe"]

            # Display DataFrame info
            df_info = [
                f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
                f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                f"Column dtypes: {', '.join([f'{col}: {dtype}' for col, dtype in df.dtypes.items()])}",
            ]

            console.print(
                Panel(
                    "\n".join(df_info),
                    title="Pandas DataFrame Export Result",
                    border_style="green",
                    padding=(1, 2),
                )
            )

            # Show DataFrame operations
            console.print("[cyan]Demonstrating DataFrame operations:[/]")

            # Create a summary statistics table
            stats_table = Table(
                title="DataFrame Statistics", box=box.ROUNDED, padding=(0, 1), border_style="blue"
            )
            stats_table.add_column("Statistic", style="cyan")
            stats_table.add_column("Value", style="yellow")

            # Add some sample statistics
            stats_table.add_row("Average Price", f"${df['price'].mean():.2f}")
            stats_table.add_row("Max Price", f"${df['price'].max():.2f}")
            stats_table.add_row("Min Price", f"${df['price'].min():.2f}")
            stats_table.add_row("Total Revenue", f"${df['total_revenue'].sum():.2f}")
            stats_table.add_row(
                "Highest Revenue Product", df.loc[df["total_revenue"].idxmax()]["product_name"]
            )

            console.print(stats_table)

            # Create a simple DataFrame transformation
            console.print(
                "\n[cyan]Demonstrating DataFrame transformation - Adding discount column:[/]"
            )
            df["discount_price"] = df["price"] * 0.9

            # Display the first few rows of the transformed DataFrame
            table = Table(
                title="Transformed DataFrame (First 3 Rows)", box=box.ROUNDED, show_header=True
            )

            # Add columns based on the DataFrame
            for col in df.columns:
                justify = "right" if df[col].dtype.kind in "ifc" else "left"
                table.add_column(col, style="cyan", justify=justify)

            # Add the first 3 rows
            for _, row in df.head(3).iterrows():
                # Format numeric values nicely
                formatted_row = []
                for col in df.columns:
                    val = row[col]
                    if pd.api.types.is_numeric_dtype(
                        df[col].dtype
                    ):  # Check column dtype, not row value
                        if "price" in col or "revenue" in col:
                            formatted_row.append(f"${val:.2f}")
                        else:
                            formatted_row.append(
                                f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
                            )
                    else:
                        formatted_row.append(str(val))

                table.add_row(*formatted_row)

            console.print(table)
        else:
            console.print(
                Panel(
                    f"[red]Failed to export to DataFrame: {df_result.get('error', 'Unknown error')}[/]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # --- EXCEL EXPORT WITH FORMATTING ---
        console.print(
            Rule("[bold blue]11.2 Excel Export with Formatting[/bold blue]", style="blue")
        )
        logger.info("Demonstrating Excel export with formatting")

        excel_fd, excel_path = tempfile.mkstemp(suffix=".xlsx", prefix="sql_demo_export_")
        os.close(excel_fd)  # Close file descriptor, as we only need the path

        with console.status("[cyan]Executing query and exporting to formatted Excel...[/]"):
            excel_result = await sql_tool.execute_sql(
                connection_id=connection_id,
                query=export_query,
                read_only=True,
                export={
                    "format": "excel",
                    "path": excel_path,
                    # Note: Additional formatting options might be available in your implementation
                },
            )

        if excel_result.get("success") and "excel_path" in excel_result:
            export_path = excel_result["excel_path"]
            file_size = os.path.getsize(export_path) / 1024  # Size in KB

            console.print(
                Panel(
                    f"[green]✅ Successfully exported to Excel:[/]\n"
                    f"Path: [cyan]{export_path}[/]\n"
                    f"Size: [yellow]{file_size:.2f} KB[/]",
                    title="Excel Export Result",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel(
                    f"[red]Failed to export to Excel: {excel_result.get('error', 'Unknown error')}[/]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # --- CUSTOM EXPORT PATH (CSV) ---
        console.print(Rule("[bold blue]11.3 Custom Export Path (CSV)[/bold blue]", style="blue"))
        logger.info("Demonstrating custom export path")

        # Create a custom path in the user's home directory
        user_home = os.path.expanduser("~")
        custom_dir = os.path.join(user_home, "sql_demo_exports")
        os.makedirs(custom_dir, exist_ok=True)

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_path = os.path.join(custom_dir, f"product_sales_{timestamp}.csv")

        console.print(f"[cyan]Exporting to custom path: [/][yellow]{custom_path}[/]")

        with console.status("[cyan]Executing query and exporting to custom CSV path...[/]"):
            csv_result = await sql_tool.execute_sql(
                connection_id=connection_id,
                query=export_query,
                read_only=True,
                export={"format": "csv", "path": custom_path},
            )

        if csv_result.get("success") and "csv_path" in csv_result:
            export_path = csv_result["csv_path"]
            file_size = os.path.getsize(export_path) / 1024  # Size in KB

            # Read first few lines to show content
            with open(export_path, "r") as f:
                first_lines = [next(f) for _ in range(3)]

            console.print(
                Panel(
                    f"[green]✅ Successfully exported to custom CSV path:[/]\n"
                    f"Path: [cyan]{export_path}[/]\n"
                    f"Size: [yellow]{file_size:.2f} KB[/]\n\n"
                    f"[dim]Preview (first 3 lines):[/]\n"
                    f"[white]{escape(''.join(first_lines))}[/]",
                    title="Custom CSV Export Result",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel(
                    f"[red]Failed to export to custom CSV path: {csv_result.get('error', 'Unknown error')}[/]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    except Exception as e:
        logger.error(f"Error in advanced export demo: {e}", exc_info=True)
        console.print(
            Panel(
                f"[bold red]Error in advanced export demo:[/]\n{escape(str(e))}",
                border_style="red",
                padding=(1, 2),
            )
        )

    console.print()  # Spacing


async def schema_validation_demo(sql_tool: SQLTool, connection_id: str) -> None:
    """Demonstrate Pandera schema validation for query results."""
    console.print(Rule("[bold green]12. Schema Validation Demo[/bold green]", style="green"))
    logger.info("Demonstrating Pandera schema validation")

    try:
        # Query to validate
        validation_query = """
        SELECT 
            product_id,
            name AS product_name,
            price,
            category,
            in_stock
        FROM products
        """

        console.print("[cyan]We'll validate that query results conform to a specified schema:[/]")
        console.print(Syntax(validation_query, "sql", theme="default"))

        # Define a Pandera schema
        schema_code = """
        # Define a Pandera schema for validation using DataFrameSchema
        product_schema = pa.DataFrameSchema({
            "product_id": pa.Column(int, checks=pa.Check.greater_than(0)),
            "product_name": pa.Column(str, nullable=False),
            "price": pa.Column(float, checks=[
                pa.Check.greater_than(0, error="price must be positive"),
                pa.Check.less_than(2000.0, error="price must be under $2000")
            ]),
            "category": pa.Column(
                str,
                checks=pa.Check.isin(["Electronics", "Audio", "Kitchen", "Wearables"]),
                nullable=False
            ),
            "in_stock": pa.Column(bool)
        })
        """

        console.print(
            Panel(
                Syntax(schema_code, "python", theme="default"),
                title="Pandera Validation Schema",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Check pandera version
        version = getattr(pa, "__version__", "unknown")
        console.print(f"[dim]Using pandera version: {version}")

        # Define the actual schema
        product_schema = pa.DataFrameSchema(
            {
                "product_id": pa.Column(int, checks=pa.Check.greater_than(0)),
                "product_name": pa.Column(str, nullable=False),
                "price": pa.Column(
                    float,
                    checks=[
                        pa.Check.greater_than(0, error="price must be positive"),
                        pa.Check.less_than(2000.0, error="price must be under $2000"),
                    ],
                ),
                "category": pa.Column(
                    str,
                    checks=pa.Check.isin(["Electronics", "Audio", "Kitchen", "Wearables"]),
                    nullable=False,
                ),
                "in_stock": pa.Column(bool),
            }
        )

        # WORKAROUND: Instead of using built-in validation (which has an error),
        # we'll fetch the data first, then validate it manually
        console.print("[cyan]Executing query to fetch data...[/]")

        with console.status("[cyan]Running query...[/]"):
            query_result = await sql_tool.execute_sql(
                connection_id=connection_id, query=validation_query, read_only=True
            )

        if query_result.get("success"):
            # Show the data
            display_result("Data Retrieved for Validation", query_result)

            # Now manually validate with Pandera
            console.print("[cyan]Now validating results with Pandera...[/]")

            if pd is not None:
                try:
                    # Create DataFrame from results
                    df = pd.DataFrame(
                        query_result.get("rows", []), columns=query_result.get("columns", [])
                    )

                    # Fix type issues - convert in_stock to boolean if needed
                    if "in_stock" in df.columns and df["in_stock"].dtype != bool:
                        df["in_stock"] = df["in_stock"].astype(bool)

                    console.print(f"[dim]Created DataFrame with shape {df.shape} for validation")

                    # Validate the data
                    with console.status("[cyan]Validating against schema...[/]"):
                        try:
                            product_schema.validate(df)
                            console.print(
                                Panel(
                                    "[green]✅ Schema validation passed![/]\n"
                                    "All data meets the requirements defined in the schema.",
                                    title="Validation Result",
                                    border_style="green",
                                    padding=(1, 2),
                                )
                            )
                        except Exception as val_err:
                            console.print(
                                Panel(
                                    f"[yellow]⚠ Schema validation failed![/]\n"
                                    f"Error: {str(val_err)}",
                                    title="Validation Result",
                                    border_style="yellow",
                                    padding=(1, 2),
                                )
                            )
                except Exception as df_err:
                    console.print(f"[red]Error creating DataFrame: {df_err}[/]")
            else:
                console.print("[yellow]Pandas is not available, cannot perform validation.[/]")
        else:
            console.print(
                Panel(
                    f"[red]Failed to execute query: {query_result.get('error', 'Unknown error')}[/]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # Simulate a failing validation case
        console.print("\n[cyan]Simulating validation failure with invalid data...[/]")

        if pd is not None:
            # Create a DataFrame with valid and invalid data
            test_data = [
                # Valid data
                {
                    "product_id": 1,
                    "product_name": "Laptop Pro X",
                    "price": 1499.99,
                    "category": "Electronics",
                    "in_stock": True,
                },
                {
                    "product_id": 2,
                    "product_name": "Smartphone Z",
                    "price": 999.99,
                    "category": "Electronics",
                    "in_stock": True,
                },
                # Invalid data (negative price)
                {
                    "product_id": 6,
                    "product_name": "Invalid Product",
                    "price": -10.0,
                    "category": "Electronics",
                    "in_stock": True,
                },
                # Invalid data (unknown category)
                {
                    "product_id": 7,
                    "product_name": "Test Product",
                    "price": 50.0,
                    "category": "Invalid Category",
                    "in_stock": True,
                },
            ]

            test_df = pd.DataFrame(test_data)

            # Display the test data
            test_table = Table(
                title="Test Data for Validation",
                box=box.ROUNDED,
                show_header=True,
                padding=(0, 1),
                border_style="yellow",
            )
            for col in test_df.columns:
                test_table.add_column(str(col), style="cyan")

            for _, row in test_df.iterrows():
                test_table.add_row(*[str(val) for val in row])

            console.print(test_table)

            # Test validation
            console.print("[cyan]Attempting to validate this data...[/]")
            try:
                # Try to validate the DataFrame directly
                product_schema.validate(test_df, lazy=True)
                console.print(
                    Panel(
                        "[red]Unexpected result: Validation passed when it should have failed![/]",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
            except Exception as val_err:
                console.print(
                    Panel(
                        f"[green]✅ Validation correctly failed as expected![/]\n"
                        f"Error: {str(val_err)}",
                        title="Expected Validation Failure (Simulated)",
                        border_style="green",
                        padding=(1, 2),
                    )
                )
        else:
            console.print("[yellow]Pandas not available, cannot demonstrate validation failure.[/]")

    except Exception as e:
        logger.error(f"Error in schema validation demo: {e}", exc_info=True)
        console.print(
            Panel(
                f"[bold red]Error in schema validation demo:[/]\n{escape(str(e))}",
                border_style="red",
                padding=(1, 2),
            )
        )

    console.print()  # Spacing


async def audit_log_demo(sql_tool: SQLTool, connection_id: str) -> None:
    """Demonstrate audit log functionality."""
    console.print(Rule("[bold green]9. Audit Log Demo[/bold green]", style="green"))
    logger.info("Demonstrating audit log functionality")

    try:
        # View the audit log
        logger.info("Viewing audit log")

        with console.status("[cyan]Retrieving audit log...[/]"):
            audit_result = await sql_tool.access_audit_log(action="view", limit=10)

        if audit_result.get("success"):
            logger.success("Successfully retrieved audit log")

            records = audit_result.get("records", [])
            if records:
                audit_table = Table(
                    title="Audit Log",
                    box=box.ROUNDED,
                    show_header=True,
                    padding=(0, 1),
                    border_style="blue",
                )
                audit_table.add_column("ID", style="dim")
                audit_table.add_column("Timestamp", style="cyan")
                audit_table.add_column("Tool", style="green")
                audit_table.add_column("Action", style="yellow")
                audit_table.add_column("Connection ID", style="magenta")
                audit_table.add_column("Success", style="cyan")

                for record in records:
                    audit_table.add_row(
                        record.get("audit_id", "?"),
                        record.get("timestamp", "?"),
                        record.get("tool_name", "?"),
                        record.get("action", "?"),
                        record.get("connection_id", "?"),
                        "[green]:heavy_check_mark:[/]" if record.get("success") else "[red]:x:[/]",
                    )

                console.print(audit_table)

                # Show details of one specific audit record
                if records:
                    sample_record = records[0]
                    console.print(
                        Panel(
                            "\n".join(
                                [
                                    f"[cyan]{k}:[/] {escape(str(v))}"
                                    for k, v in sample_record.items()
                                    if k
                                    not in [
                                        "audit_id",
                                        "timestamp",
                                        "tool_name",
                                        "action",
                                        "connection_id",
                                        "success",
                                    ]
                                ]
                            ),
                            title=f"Audit Record Details: {sample_record.get('audit_id', '?')}",
                            border_style="dim",
                            padding=(1, 2),
                        )
                    )
            else:
                console.print(
                    Panel("[yellow]No audit records found.", border_style="yellow", padding=(0, 1))
                )
        else:
            error_msg = audit_result.get("error", "Unknown error")
            logger.error(f"Failed to retrieve audit log: {error_msg}")
            console.print(
                Panel(
                    f"[bold red]:x: Audit log retrieval failed:[/]\n{escape(error_msg)}",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # Export the audit log
        logger.info("Exporting audit log")

        with console.status("[cyan]Exporting audit log to CSV...[/]"):
            export_result = await sql_tool.access_audit_log(action="export", export_format="csv")

        if export_result.get("success"):
            export_path = export_result.get("path", "")
            record_count = export_result.get("record_count", 0)
            logger.success(f"Successfully exported {record_count} audit records to CSV")

            console.print(
                Panel(
                    f"[green]:heavy_check_mark: Exported {record_count} audit records to:[/]\n[cyan]{export_path}[/]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            error_msg = export_result.get("error", "Unknown error")
            logger.error(f"Failed to export audit log: {error_msg}")
            console.print(
                Panel(
                    f"[bold red]:x: Audit log export failed:[/]\n{escape(error_msg)}",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    except Exception as e:
        logger.error(f"Unexpected error in audit log demo: {e}")
        console.print(f"[bold red]:x: Unexpected Error:[/]\n{escape(str(e))}")

    console.print()  # Spacing


async def cleanup_demo(sql_tool: SQLTool, connection_id: str) -> None:
    """Demonstrate disconnecting from the database."""
    console.print(
        Rule("[bold green]Database Cleanup and Disconnection[/bold green]", style="green")
    )
    logger.info("Disconnecting from database")

    try:
        # Disconnect from the database
        disconnect_result = await sql_tool.manage_database(
            action="disconnect", connection_id=connection_id
        )

        if disconnect_result.get("success"):
            logger.success(f"Successfully disconnected from database (ID: {connection_id})")
            console.print(
                Panel(
                    f"[green]:heavy_check_mark: Successfully disconnected from database. Connection ID: [dim]{connection_id}[/dim][/]",
                    border_style="green",
                    padding=(0, 1),
                )
            )
        else:
            logger.error(f"Failed to disconnect: {disconnect_result.get('error')}")
            console.print(
                Panel(
                    f"[bold red]:x: Failed to disconnect:[/]\n{escape(disconnect_result.get('error', 'Unknown error'))}",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    except Exception as e:
        logger.error(f"Error in cleanup demo: {e}")
        console.print(f"[bold red]:x: Error in cleanup:[/]\n{escape(str(e))}")

    console.print()


async def verify_demo_database(sql_tool, connection_id: str) -> None:
    """Verify the demo database has been set up correctly."""
    logger.info("Verifying database setup...")

    # For consistency, we'll still display the setup status
    console.print(
        Panel(
            "[green]:heavy_check_mark: Using prepared sample database.",
            padding=(0, 1),
            border_style="green",
        )
    )

    # Check the tables to ensure the database was set up correctly
    try:
        # Execute a simple query to check if the tables have data
        result = await sql_tool.execute_sql(
            connection_id=connection_id,
            query="SELECT COUNT(*) as count FROM customers",
            read_only=True,
        )

        count = result.get("rows", [{}])[0].get("count", 0)
        if count > 0:
            logger.info(f"Verified database setup: {count} customers found")
            console.print(
                Panel(
                    f"[green]:heavy_check_mark: Sample database verified with {count} customer records.",
                    padding=(0, 1),
                    border_style="green",
                )
            )
        else:
            logger.warning("Database tables found but they appear to be empty")
            console.print(
                Panel(
                    "[yellow]⚠ Database tables found but they appear to be empty.",
                    padding=(0, 1),
                    border_style="yellow",
                )
            )

    except (ToolError, ToolInputError) as e:
        logger.error(f"Error checking database setup: {e}")
        console.print(
            Panel(
                f"[bold red]:x: Database Setup Error:[/]\n{escape(str(e))}",
                padding=(1, 2),
                border_style="red",
            )
        )


# --- Main Function ---


async def main() -> int:
    """Run the SQL database tools demo."""
    console.print(Rule("[bold magenta]SQL Database Tools Demo[/bold magenta]"))

    exit_code = 0
    connection_id = None

    # Get path to the pre-initialized database
    db_file = os.path.join(os.path.dirname(__file__), "demo.db")

    # Force recreate the demo database
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            console.print("[yellow]Removed existing database file to ensure correct schema.[/]")
        except OSError as e:
            console.print(f"[yellow]Warning: Could not remove existing database: {e}[/]")

    # Check if the demo database exists, and create it if not
    if not os.path.exists(db_file):
        console.print("[yellow]Demo database not found. Creating it now...[/]")
        try:
            # Initialize the database directly
            init_demo_database(db_file)
            console.print("[green]Demo database created successfully.[/]")
        except Exception as e:
            console.print(f"[red]Failed to create demo database: {e}[/]")
            return 1

    gateway = Gateway("sql-database-demo", register_tools=False)

    # Connection string for file-based SQLite database instead of memory
    file_connection_string = f"sqlite:///{db_file}"

    # Create an instance of the SQLTool
    try:
        sql_tool = SQLTool(gateway)

        # Run the demonstrations
        connection_id = await connection_demo(sql_tool, file_connection_string)

        if connection_id:
            await verify_demo_database(sql_tool, connection_id)
            await schema_discovery_demo(sql_tool, connection_id)
            await table_details_demo(sql_tool, connection_id, "customers")
            await find_related_tables_demo(sql_tool, connection_id, "orders")
            await column_statistics_demo(sql_tool, connection_id, "products", "price")
            await query_execution_demo(sql_tool, connection_id)
            await nl_to_sql_demo(sql_tool, connection_id)
            await documentation_demo(sql_tool, connection_id)
            await audit_log_demo(sql_tool, connection_id)

            # Add the new demos
            await security_features_demo(sql_tool, connection_id)
            await advanced_export_demo(sql_tool, connection_id)
            await schema_validation_demo(sql_tool, connection_id)

            await cleanup_demo(sql_tool, connection_id)
        else:
            logger.error("Skipping demonstrations due to connection failure")
            exit_code = 1

    except Exception as e:
        logger.critical(f"Demo failed with unexpected error: {e}")
        console.print(f"[bold red]CRITICAL ERROR: {escape(str(e))}[/]")
        exit_code = 1
    finally:
        # Ensure we shutdown the SQLTool if it was created
        if "sql_tool" in locals():
            try:
                await sql_tool.shutdown()
                logger.info("SQLTool shut down successfully")
            except Exception as shutdown_err:
                logger.error(f"Error during SQLTool shutdown: {shutdown_err}")
        # Clean up the demo database file
        try:
            if os.path.exists(db_file) and "sql_demo_export" in db_file:
                os.remove(db_file)
                logger.info(f"Cleaned up demo database file: {db_file}")
        except Exception as clean_err:
            logger.warning(f"Could not clean up demo database: {clean_err}")

    return exit_code


if __name__ == "__main__":
    # Setup logging
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
