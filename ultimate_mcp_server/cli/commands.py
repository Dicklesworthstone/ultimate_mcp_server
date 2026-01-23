"""Command implementations for the Ultimate MCP Server CLI."""

import asyncio
import inspect
import os
import sys
import time
from typing import List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.constants import BASE_TOOLSET_CATEGORIES, Provider
from ultimate_mcp_server.core.providers.base import get_provider
from ultimate_mcp_server.core.server import Gateway, start_server
from ultimate_mcp_server.graceful_shutdown import enable_quiet_shutdown
from ultimate_mcp_server.services.cache import get_cache_service
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)
console = Console(file=sys.stderr)  # Use stderr to avoid interfering with MCP protocol


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    log_level: Optional[str] = None,
    transport_mode: str = "sse",
    include_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
    load_all_tools: bool = False,
) -> None:
    """Start the Ultimate MCP Server with specified configuration.

    This function initializes and launches the MCP-compliant server that provides
    an API for AI agents to use various tools and capabilities. The server uses
    FastAPI under the hood and can be customized through configuration overrides.
    Server settings from command-line arguments take precedence over .env and config file values.

    Args:
        host: Network interface to bind to (e.g., '127.0.0.1' for local, '0.0.0.0' for all)
        port: TCP port to listen on (default: 8013 from config)
        workers: Number of parallel worker processes (more workers = higher concurrency)
        log_level: Logging verbosity (debug, info, warning, error, critical)
        transport_mode: Communication protocol mode ('sse' for Server-Sent Events streaming
                        or 'stdio' for standard input/output in certain environments)
        include_tools: Allowlist of specific tool names to register (if not specified, all
                       available tools are registered)
        exclude_tools: Blocklist of specific tool names to exclude (takes precedence over
                       include_tools if both are specified)
        load_all_tools: If True, load all available tools. If False (default), load only
                       the Base Toolset (completion, filesystem, optimization, provider,
                       local_text, meta, search).
    """
    # Set up graceful shutdown handling
    enable_quiet_shutdown()

    # Get the current config
    cfg = get_config()

    # Override config with provided values
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    if workers:
        cfg.server.workers = workers

    # Update tool registration config
    if include_tools or exclude_tools:
        cfg.tool_registration.filter_enabled = True

    if include_tools:
        cfg.tool_registration.included_tools = include_tools

    if exclude_tools:
        cfg.tool_registration.excluded_tools = exclude_tools

    # Determine effective log level
    effective_log_level = log_level or getattr(cfg, "log_level", "info")

    # Print server info
    console.print("[bold blue]Starting Ultimate MCP Server server[/bold blue]")
    console.print(f"Host: [cyan]{cfg.server.host}[/cyan]")
    console.print(f"Port: [cyan]{cfg.server.port}[/cyan]")
    console.print(f"Workers: [cyan]{cfg.server.workers}[/cyan]")
    console.print(f"Log level: [cyan]{effective_log_level.upper()}[/cyan]")
    console.print(f"Transport mode: [cyan]{transport_mode}[/cyan]")

    # Tool Loading Status
    if load_all_tools:
        console.print("Tool Loading: [yellow]All Available Tools[/yellow]")
    else:
        console.print(
            "Tool Loading: [yellow]Base Toolset Only[/yellow] (Use --load-all-tools to load all)"
        )
        # Format the categories for display
        console.print("  [bold]Includes:[/bold]")
        for category, tools in BASE_TOOLSET_CATEGORIES.items():
            console.print(f"    [cyan]{category}[/cyan]: {', '.join(tools)}")

    # Print tool filtering info if enabled
    if cfg.tool_registration.filter_enabled:
        if cfg.tool_registration.included_tools:
            console.print(
                f"Including tools: [cyan]{', '.join(cfg.tool_registration.included_tools)}[/cyan]"
            )
        if cfg.tool_registration.excluded_tools:
            console.print(
                f"Excluding tools: [red]{', '.join(cfg.tool_registration.excluded_tools)}[/red]"
            )

    console.print()

    # Set environment variables for the tool context estimator to use
    os.environ["MCP_SERVER_HOST"] = cfg.server.host
    os.environ["MCP_SERVER_PORT"] = str(cfg.server.port)

    # Just run the server directly - no threading, no token analysis
    start_server(
        host=cfg.server.host,
        port=cfg.server.port,
        workers=cfg.server.workers,
        log_level=effective_log_level,
        transport_mode=transport_mode,
        include_tools=cfg.tool_registration.included_tools
        if cfg.tool_registration.filter_enabled
        else None,
        exclude_tools=cfg.tool_registration.excluded_tools
        if cfg.tool_registration.filter_enabled
        else None,
        load_all_tools=load_all_tools,
    )


async def list_providers(check_keys: bool = False, list_models: bool = False) -> None:
    """Display information about configured LLM providers and their status.

    This function queries all supported LLM providers (OpenAI, Anthropic, Gemini, etc.),
    displays their configuration status, and can optionally verify API key validity
    and list available models. This is useful for diagnosing configuration issues
    or exploring what models are accessible through each provider.

    The basic output shows provider names, enabled status, API key presence,
    and default model configuration. Additional information is available through
    the optional parameters.

    Args:
        check_keys: When True, tests each configured API key against the respective
                   provider's authentication endpoint to verify it's valid and active
        list_models: When True, queries each provider's model list endpoint and displays
                    all models available for use with your current credentials
    """
    # Get the current config
    cfg = get_config()

    # Create provider table
    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Enabled", style="green")
    table.add_column("API Key", style="yellow")
    table.add_column("Default Model", style="blue")

    # Add spinner during provider initialization
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Initializing providers...[/bold blue]"),
        transient=True,
    ) as progress:
        progress.add_task("init", total=None)

        # Create Gateway instance (initializes all providers)
        gateway = Gateway()
        while not hasattr(gateway, "provider_status") or not gateway.provider_status:
            await asyncio.sleep(0.1)

    # Get provider status
    provider_status = gateway.provider_status

    # Add rows to table
    for provider_name in [p.value for p in Provider]:
        status = provider_status.get(provider_name, None)

        if status:
            enabled = "✅" if status.enabled else "❌"
            api_key = "✅" if status.api_key_configured else "❌"

            # Get default model
            provider_cfg = getattr(cfg, "providers", {}).get(provider_name, None)
            default_model = provider_cfg.default_model if provider_cfg else "N/A"

            table.add_row(provider_name, enabled, api_key, default_model)

    # Print table
    console.print(table)

    # Check API keys if requested
    if check_keys:
        console.print("\n[bold]API Key Check:[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Checking API keys...[/bold blue]"),
            transient=True,
        ) as progress:
            progress.add_task("check", total=None)

            for provider_name in [p.value for p in Provider]:
                console.print(Rule(f"[bold cyan]{provider_name}[/bold cyan]", align="center"))
                status = provider_status.get(provider_name, None)

                if status and status.api_key_configured:
                    try:
                        # Get provider instance
                        provider = get_provider(provider_name)

                        # Check API key
                        valid = await provider.check_api_key()

                        status_text = "[green]valid[/green]" if valid else "[red]invalid[/red]"
                        console.print(f"API Key Status: {status_text}")

                    except Exception as e:
                        console.print(f"[red]Error checking API key: {str(e)}[/red]")
                else:
                    if status:
                        console.print("API Key Status: [yellow]not configured[/yellow]")
                    else:
                        console.print("API Key Status: [dim]Provider not configured/enabled[/dim]")
                console.print()  # Add spacing after each provider's check

    # List models if requested
    if list_models:
        console.print("\n[bold]Available Models:[/bold]")

        with Progress(
            SpinnerColumn(), TextColumn("[bold blue]Loading models...[/bold blue]"), transient=True
        ) as progress:
            progress.add_task("load", total=None)

            for provider_name in [p.value for p in Provider]:
                status = provider_status.get(provider_name, None)

                if status and status.available:
                    provider_instance = gateway.providers.get(provider_name)

                    if provider_instance:
                        console.print(
                            Rule(f"[bold cyan]{provider_name} Models[/bold cyan]", align="center")
                        )
                        try:
                            # Get models
                            models = await provider_instance.list_models()

                            # Create model table
                            model_table = Table(
                                box=None, show_header=True, header_style="bold blue"
                            )
                            model_table.add_column("Model ID", style="cyan")
                            model_table.add_column("Description", style="green")

                            if models:
                                for model in models:
                                    model_table.add_row(
                                        model["id"], model.get("description", "[dim]N/A[/dim]")
                                    )
                                console.print(model_table)
                            else:
                                console.print(
                                    "[yellow]No models found or returned by provider.[/yellow]"
                                )

                        except Exception as e:
                            console.print(
                                f"[red]Error listing models for {provider_name}: {str(e)}[/red]"
                            )
                else:
                    if status:
                        console.print(Rule(f"[bold dim]{provider_name}[/bold dim]", align="center"))
                        console.print("[yellow]Provider not available or configured.[/yellow]")
                console.print()  # Add spacing after each provider's models


async def test_provider(
    provider: str, model: Optional[str] = None, prompt: str = "Hello, world!"
) -> None:
    """Test an LLM provider's functionality with a complete request-response cycle.

    This function conducts an end-to-end test of a specified LLM provider by
    sending a sample prompt and displaying the resulting completion. It measures
    performance metrics such as response time, token usage, and estimated cost.

    The test verifies:
    - API key validity and configuration
    - Connection to the provider's endpoint
    - Model availability and functionality
    - Response generation capabilities
    - Cost estimation correctness

    Results include the generated text, model used, token counts (input/output),
    estimated cost, and response time, providing a comprehensive health check.

    Args:
        provider: Provider identifier (e.g., 'openai', 'anthropic', 'gemini')
        model: Specific model to use for the test (if None, uses the provider's default model)
        prompt: Text prompt to send to the model (defaults to a simple greeting)
    """
    console.print(f"[bold]Testing provider:[/bold] [cyan]{provider}[/cyan]")
    console.print(f"[bold]Model:[/bold] [cyan]{model or 'default'}[/cyan]")
    console.print(f'[bold]Prompt:[/bold] [green]"{prompt}"[/green]')
    console.print()

    try:
        # Get provider instance
        provider_instance = get_provider(provider)

        # Initialize provider
        await provider_instance.initialize()

        # Show spinner during generation
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating completion...[/bold blue]"),
            transient=True,
        ) as progress:
            progress.add_task("generate", total=None)

            # Generate completion
            start_time = time.time()
            result = await provider_instance.generate_completion(
                prompt=prompt, model=model, temperature=0.7
            )
            elapsed_time = time.time() - start_time

        # Print result
        console.print("[bold cyan]Generated text:[/bold cyan]")
        console.print(result.text)
        console.print()

        # Print metrics
        console.print(f"[bold]Model used:[/bold] [cyan]{result.model}[/cyan]")
        console.print(
            f"[bold]Tokens:[/bold] [yellow]{result.input_tokens}[/yellow] input, [yellow]{result.output_tokens}[/yellow] output, [yellow]{result.total_tokens}[/yellow] total"
        )
        console.print(f"[bold]Cost:[/bold] [green]${result.cost:.6f}[/green]")
        console.print(f"[bold]Time:[/bold] [blue]{elapsed_time:.2f}s[/blue]")

    except Exception as e:
        console.print(f"[bold red]Error testing provider:[/bold red] {str(e)}")


async def generate_completion(
    provider: str,
    model: Optional[str] = None,
    prompt: str = "",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system: Optional[str] = None,
    stream: bool = False,
) -> None:
    """Generate text from an LLM provider directly through the CLI.

    This function provides direct access to LLM text generation capabilities
    without requiring an MCP server or client. It supports both synchronous
    (wait for full response) and streaming (token-by-token) output modes.

    For providers that natively support system prompts (like Anthropic),
    the system parameter is passed directly to the API. For other providers,
    the system message is prepended to the prompt with appropriate formatting.

    The function displays:
    - Generated text (with real-time streaming if requested)
    - Model information
    - Token usage statistics (input/output)
    - Cost estimate
    - Response time

    Args:
        provider: LLM provider identifier (e.g., 'openai', 'anthropic', 'gemini')
        model: Specific model ID to use (if None, uses provider's default model)
        prompt: The text prompt to send to the model
        temperature: Sampling temperature (0.0-2.0) controlling output randomness
                     (lower = more deterministic, higher = more creative)
        max_tokens: Maximum number of tokens to generate in the response
                    (if None, uses provider's default maximum)
        system: Optional system message for setting context/behavior
                (handled differently depending on provider capabilities)
        stream: When True, displays generated tokens as they arrive rather than
                waiting for the complete response (may affect metric reporting)
    """
    try:
        # Get provider instance
        provider_instance = get_provider(provider)

        # Initialize provider
        await provider_instance.initialize()

        # Set extra parameters based on provider
        kwargs = {}
        if system:
            if provider == Provider.ANTHROPIC.value:
                kwargs["system"] = system
            else:
                # For other providers, prepend system message to prompt
                prompt = f"System: {system}\n\nUser: {prompt}"

        # Show progress for non-streaming generation
        if not stream:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Generating completion...[/bold blue]"),
                transient=True,
            ) as progress:
                progress.add_task("generate", total=None)

                # Generate completion
                start_time = time.time()
                result = await provider_instance.generate_completion(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                elapsed_time = time.time() - start_time

            # Print result
            console.print(f"[cyan]{result.text}[/cyan]")
            console.print()

            # Print metrics
            console.print(f"[bold]Model:[/bold] [blue]{result.model}[/blue]")
            console.print(
                f"[bold]Tokens:[/bold] [yellow]{result.input_tokens}[/yellow] input, [yellow]{result.output_tokens}[/yellow] output"
            )
            console.print(f"[bold]Cost:[/bold] [green]${result.cost:.6f}[/green]")
            console.print(f"[bold]Time:[/bold] [blue]{elapsed_time:.2f}s[/blue]")

        else:
            # Streaming generation
            console.print("[bold blue]Generating completion (streaming)...[/bold blue]")

            # Generate streaming completion
            start_time = time.time()
            stream = provider_instance.generate_completion_stream(
                prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
            )

            # Process stream
            full_text = ""
            async for chunk, metadata in stream:  # noqa: B007
                console.print(chunk, end="")
                sys.stderr.flush()  # Use stderr instead of stdout
                full_text += chunk

            elapsed_time = time.time() - start_time

            # Print metrics
            console.print("\n")
            console.print(
                f"[bold]Model:[/bold] [blue]{metadata.get('model', model or 'unknown')}[/blue]"
            )
            console.print(f"[bold]Time:[/bold] [blue]{elapsed_time:.2f}s[/blue]")

    except Exception as e:
        console.print(f"[bold red]Error generating completion:[/bold red] {str(e)}")


async def check_cache(show_status: bool = True, clear: bool = False) -> None:
    """View cache statistics and optionally clear the LLM response cache.

    The server employs a caching system that stores LLM responses to avoid
    redundant API calls when the same or similar requests are made repeatedly.
    This significantly reduces API costs and improves response times.

    This function displays comprehensive cache information including:
    - Basic configuration (enabled status, TTL, max size)
    - Storage details (persistence, directory location)
    - Performance metrics (hit/miss counts, hit ratio percentage)
    - Efficiency data (estimated cost savings, total tokens saved)

    Clearing the cache removes all stored responses, which might be useful
    when testing changes, addressing potential staleness, or reclaiming disk space.

    Args:
        show_status: When True, displays detailed cache statistics and configuration
        clear: When True, purges all entries from the cache (requires confirmation)
    """
    # Get cache service
    cache_service = get_cache_service()

    if clear:
        # Clear cache
        console.print("[bold]Clearing cache...[/bold]")
        cache_service.clear()
        console.print("[green]Cache cleared successfully[/green]")

    if show_status:
        # Get cache stats
        stats = cache_service.get_stats()

        # Create status table
        table = Table(title="Cache Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        # Add rows
        table.add_row("Enabled", "✅" if stats["enabled"] else "❌")
        table.add_row("Size", f"{stats['size']} / {stats['max_size']} entries")
        table.add_row("TTL", f"{stats['ttl']} seconds")
        table.add_row("Persistence", "✅" if stats["persistence"]["enabled"] else "❌")
        table.add_row("Cache Directory", stats["persistence"]["cache_dir"])
        table.add_row("Fuzzy Matching", "✅" if stats["fuzzy_matching"] else "❌")

        # Print table
        console.print(table)

        # Create stats table
        stats_table = Table(title="Cache Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        # Add rows
        cache_stats = stats["stats"]
        stats_table.add_row("Hits", str(cache_stats["hits"]))
        stats_table.add_row("Misses", str(cache_stats["misses"]))
        stats_table.add_row("Hit Ratio", f"{cache_stats['hit_ratio']:.2%}")
        stats_table.add_row("Stores", str(cache_stats["stores"]))
        stats_table.add_row("Evictions", str(cache_stats["evictions"]))
        stats_table.add_row("Total Saved Tokens", f"{cache_stats['total_saved_tokens']:,}")
        stats_table.add_row(
            "Estimated Cost Savings", f"${cache_stats['estimated_cost_savings']:.6f}"
        )

        # Print table
        console.print(stats_table)


async def benchmark_providers(
    providers: List[str] = None,
    models: List[str] = None,
    prompt: Optional[str] = None,
    runs: int = 3,
) -> None:
    """Compare performance metrics across different LLM providers and models.

    This benchmark utility sends identical prompts to multiple provider/model
    combinations and measures various performance characteristics. It runs
    each test multiple times to establish average metrics for more reliable
    comparison.

    Benchmarks measure:
    - Response time: How long it takes to receive a complete response
    - Processing speed: Tokens per second throughput
    - Token efficiency: Input/output token ratio
    - Cost: Estimated price per request

    Results are presented in a table format for easy comparison. This helps
    identify the optimal provider/model for specific needs based on speed,
    cost, or quality considerations.

    Args:
        providers: List of provider identifiers to benchmark (e.g., ['openai', 'anthropic'])
                  If None, benchmarks all available and configured providers
        models: List of specific model IDs to test (e.g., ['gpt-4o', 'claude-3-5-haiku'])
               If None, uses each provider's default model
        prompt: Text prompt to use for testing (should be identical across all providers)
                If None, uses a default explanation prompt
        runs: Number of test iterations to run for each provider/model combination
              (higher values produce more reliable averages but take longer)
    """
    # Use default providers if not specified
    if not providers:
        providers = [p.value for p in Provider]

    # Set default prompt if not provided
    if not prompt:
        prompt = "Explain the concept of quantum computing in simple terms that a high school student would understand."

    console.print(f"[bold]Running benchmark with {runs} runs per provider/model[/bold]")
    console.print(f'[bold]Prompt:[/bold] [green]"{prompt}"[/green]')
    console.print()

    # Create results table
    table = Table(title="Benchmark Results")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="blue")
    table.add_column("Avg. Time (s)", style="green")
    table.add_column("Tokens/Sec", style="yellow")
    table.add_column("Avg. Cost ($)", style="magenta")
    table.add_column("Input Tokens", style="dim")
    table.add_column("Output Tokens", style="dim")

    # Track benchmarks for progress bar
    total_benchmarks = 0
    for provider_name in providers:
        try:
            provider_instance = get_provider(provider_name)
            await provider_instance.initialize()

            # Get available models
            available_models = await provider_instance.list_models()

            # Filter models if specified
            if models:
                available_models = [m for m in available_models if m["id"] in models]
            else:
                # Use default model if no models specified
                default_model = provider_instance.get_default_model()
                available_models = [m for m in available_models if m["id"] == default_model]

            total_benchmarks += len(available_models)

        except Exception:
            # Skip providers that can't be initialized
            pass

    # Run benchmarks with progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        benchmark_task = progress.add_task(
            "[bold blue]Running benchmarks...", total=total_benchmarks * runs
        )

        for provider_name in providers:
            try:
                # Get provider instance
                provider_instance = get_provider(provider_name)
                await provider_instance.initialize()

                # Get available models
                available_models = await provider_instance.list_models()

                # Filter models if specified
                if models:
                    available_models = [m for m in available_models if m["id"] in models]
                else:
                    # Use default model if no models specified
                    default_model = provider_instance.get_default_model()
                    available_models = [m for m in available_models if m["id"] == default_model]

                for model_info in available_models:
                    model_id = model_info["id"]

                    # Run benchmark for this model
                    total_time = 0.0
                    total_cost = 0.0
                    total_input_tokens = 0
                    total_output_tokens = 0
                    total_tokens = 0

                    for run in range(runs):
                        try:
                            # Update progress description
                            progress.update(
                                benchmark_task,
                                description=f"[bold blue]Benchmarking {provider_name}/{model_id} (Run {run + 1}/{runs})",
                            )

                            # Run benchmark
                            start_time = time.time()
                            result = await provider_instance.generate_completion(
                                prompt=prompt, model=model_id, temperature=0.7
                            )
                            run_time = time.time() - start_time

                            # Record metrics
                            total_time += run_time
                            total_cost += result.cost
                            total_input_tokens += result.input_tokens
                            total_output_tokens += result.output_tokens
                            total_tokens += result.total_tokens

                            # Update progress
                            progress.advance(benchmark_task)

                        except Exception as e:
                            console.print(
                                f"[red]Error in run {run + 1} for {provider_name}/{model_id}: {str(e)}[/red]"
                            )
                            # Still advance progress
                            progress.advance(benchmark_task)

                    # Calculate averages
                    avg_time = total_time / max(1, runs)
                    avg_cost = total_cost / max(1, runs)
                    avg_input_tokens = total_input_tokens // max(1, runs)
                    avg_output_tokens = total_output_tokens // max(1, runs)

                    # Calculate tokens per second
                    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

                    # Add to results table
                    table.add_row(
                        provider_name,
                        model_id,
                        f"{avg_time:.2f}",
                        f"{tokens_per_second:.1f}",
                        f"{avg_cost:.6f}",
                        str(avg_input_tokens),
                        str(avg_output_tokens),
                    )

            except Exception as e:
                console.print(f"[red]Error benchmarking provider {provider_name}: {str(e)}[/red]")

    # Print results
    console.print(table)


async def list_tools(category: Optional[str] = None) -> None:
    """Display all available MCP tools registered in the server.

    This function provides a comprehensive listing of tools that can be called
    through the Model Context Protocol (MCP) interface. Each tool represents
    a specific capability that AI agents can access, from text generation to
    filesystem operations, browser automation, database access, and more.

    Tools are organized into functional categories such as:
    - completion: Text generation capabilities
    - document: Document processing and analysis
    - extraction: Structured data extraction from text
    - filesystem: File and directory operations
    - browser: Web browsing and automation
    - rag: Retrieval-augmented generation
    - database: SQL database interactions
    - meta: Self-reflection and tool discovery

    The listing includes tool names, categories, and brief descriptions.
    The output also provides usage hints for filtering which tools are
    enabled when starting the server.

    Args:
        category: When specified, only shows tools belonging to the given
                 category (e.g., 'filesystem', 'document', 'browser')
    """
    # Import tools module to get the list of available tools
    from ultimate_mcp_server.tools import STANDALONE_TOOL_FUNCTIONS

    # Create tools table
    table = Table(title="Available Ultimate MCP Server Tools")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description", style="yellow")

    # Define tool categories
    categories = {
        "completion": [
            "generate_completion",
            "stream_completion",
            "chat_completion",
            "multi_completion",
        ],
        "provider": ["get_provider_status", "list_models"],
        "tournament": [
            "create_tournament",
            "get_tournament_status",
            "list_tournaments",
            "get_tournament_results",
            "cancel_tournament",
        ],
        "document": [
            "chunk_document",
            "summarize_document",
            "extract_entities",
            "generate_qa_pairs",
            "process_document_batch",
        ],
        "extraction": [
            "extract_json",
            "extract_table",
            "extract_key_value_pairs",
            "extract_semantic_schema",
            "extract_entity_graph",
            "extract_code_from_response",
        ],
        "filesystem": [
            "read_file",
            "read_multiple_files",
            "write_file",
            "edit_file",
            "create_directory",
            "list_directory",
            "directory_tree",
            "move_file",
            "search_files",
            "get_file_info",
            "list_allowed_directories",
        ],
        "rag": [
            "create_knowledge_base",
            "list_knowledge_bases",
            "delete_knowledge_base",
            "add_documents",
            "retrieve_context",
            "generate_with_rag",
        ],
        "meta": [
            "get_tool_info",
            "get_llm_instructions",
            "get_tool_recommendations",
            "register_api_meta_tools",
        ],
        "search": ["marqo_fused_search"],
        "ocr": [
            "extract_text_from_pdf",
            "process_image_ocr",
            "enhance_ocr_text",
            "analyze_pdf_structure",
            "batch_process_documents",
        ],
        "optimization": [
            "estimate_cost",
            "compare_models",
            "recommend_model",
            "execute_optimized_workflow",
        ],
        "database": [
            "connect_to_database",
            "disconnect_from_database",
            "discover_database_schema",
            "execute_query",
            "generate_database_documentation",
            "get_table_details",
            "find_related_tables",
            "analyze_column_statistics",
            "execute_parameterized_query",
            "create_database_view",
            "create_database_index",
            "test_connection",
            "execute_transaction",
            "execute_query_with_pagination",
            "get_database_status",
        ],
        "audio": [
            "transcribe_audio",
            "extract_audio_transcript_key_points",
            "chat_with_transcript",
        ],
        "browser": [
            "browser_init",
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_screenshot",
            "browser_close",
            "browser_select",
            "browser_checkbox",
            "browser_get_text",
            "browser_get_attributes",
            "browser_execute_javascript",
            "browser_wait",
            "execute_web_workflow",
            "extract_structured_data_from_pages",
            "find_and_download_pdfs",
            "multi_engine_search_summary",
        ],
        "classification": ["text_classification"],
    }

    # Find category for each tool
    tool_categories = {}
    for cat_name, tools in categories.items():
        for tool in tools:
            tool_categories[tool] = cat_name

    # Add rows to table
    for tool_func in STANDALONE_TOOL_FUNCTIONS:
        if callable(tool_func):
            tool_name = getattr(tool_func, "__name__", str(tool_func))
            tool_category = tool_categories.get(tool_name, "other")

            # Skip if category filter is provided and doesn't match
            if category and category.lower() != tool_category.lower():
                continue

            # Get docstring (first line only for description)
            docstring = inspect.getdoc(tool_func) or ""
            description = docstring.split("\n")[0] if docstring else ""

            table.add_row(tool_name, tool_category, description)

    # Add the special meta tool registrars
    if not category or category.lower() in ["meta", "other"]:
        if not category or category.lower() == "meta":
            table.add_row("register_api_meta_tools", "meta", "Register Meta API tools")

    # Sort table by category and tool name
    console.print(table)

    # Print usage hint
    console.print("\n[bold]Usage with tool filtering:[/bold]")
    console.print("To include only specific tools:")
    console.print("  umcp run --include-tools tool1 tool2 tool3")
    console.print("\nTo exclude specific tools:")
    console.print("  umcp run --exclude-tools tool1 tool2 tool3")
    console.print("\nTo include tools by category:")
    console.print("  umcp tools --category filesystem  # List filesystem tools")
    console.print(
        "  umcp run --include-tools read_file write_file edit_file  # Include only these filesystem tools"
    )
