#!/usr/bin/env python
"""Analytics and reporting demonstration for Ultimate MCP Server."""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.live import Live
from rich.markup import escape
from rich.rule import Rule
from rich.table import Table

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import get_provider
from ultimate_mcp_server.services.analytics.metrics import get_metrics_tracker
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker, display_analytics_metrics

# --- Add Rich Imports ---
from ultimate_mcp_server.utils.logging.console import console

# ----------------------

# Initialize logger
logger = get_logger("example.analytics_reporting")


async def simulate_llm_usage(tracker: CostTracker = None):
    """Simulate various LLM API calls to generate analytics data."""
    console.print(Rule("[bold blue]Simulating LLM Usage[/bold blue]"))
    logger.info("Simulating LLM usage to generate analytics data", emoji_key="start")

    metrics = get_metrics_tracker()
    providers_info = []

    # Setup providers - Get keys from loaded config via get_provider
    # REMOVED provider_configs dict and direct decouple calls

    # Iterate through defined Provider enums
    for provider_enum in Provider:
        # Skip if provider doesn't make sense to simulate here, e.g., OpenRouter might need extra config
        if provider_enum == Provider.OPENROUTER:
            logger.info(f"Skipping {provider_enum.value} in simulation.", emoji_key="skip")
            continue

        try:
            # Let get_provider handle config/key lookup internally
            provider = await get_provider(provider_enum.value)  # REMOVED api_key argument
            if provider:
                # Ensure initialization (get_provider might not initialize)
                # await provider.initialize() # Initialization might be done by get_provider or completion call
                providers_info.append((provider_enum.value, provider))
                logger.info(
                    f"Provider instance obtained for: {provider_enum.value}", emoji_key="provider"
                )
            else:
                logger.info(
                    f"Provider {provider_enum.value} not configured or key missing, skipping simulation.",
                    emoji_key="skip",
                )
        except Exception as e:
            logger.warning(
                f"Failed to get/initialize {provider_enum.value}: {e}", emoji_key="warning"
            )

    if not providers_info:
        logger.error("No providers could be initialized. Cannot simulate usage.", emoji_key="error")
        console.print(
            "[bold red]Error:[/bold red] No LLM providers could be initialized. Please check your API keys."
        )
        return metrics  # Return empty metrics

    console.print(f"Simulating usage with [cyan]{len(providers_info)}[/cyan] providers.")

    prompts = [
        "What is machine learning?",
        "Explain the concept of neural networks in simple terms.",
        "Write a short story about a robot that learns to love.",
        "Summarize the key innovations in artificial intelligence over the past decade.",
        "What are the ethical considerations in developing advanced AI systems?",
    ]

    total_calls = len(providers_info) * len(prompts)
    call_count = 0

    for provider_name, provider in providers_info:
        # Use default model unless specific logic requires otherwise
        model_to_use = provider.get_default_model()
        if not model_to_use:
            logger.warning(
                f"No default model found for {provider_name}, skipping provider.",
                emoji_key="warning",
            )
            continue  # Skip this provider if no default model

        for prompt in prompts:
            call_count += 1
            logger.info(
                f"Simulating call ({call_count}/{total_calls}) for {provider_name}",
                emoji_key="processing",
            )

            try:
                start_time = time.time()
                result = await provider.generate_completion(
                    prompt=prompt,
                    model=model_to_use,  # Use determined model
                    temperature=0.7,
                    max_tokens=150,
                )
                completion_time = time.time() - start_time

                # Track costs if tracker provided
                if tracker:
                    tracker.add_call(result)

                # Record metrics using the actual model returned in the result
                metrics.record_request(
                    provider=provider_name,
                    model=result.model,  # Use model from result
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost=result.cost,
                    duration=completion_time,
                    success=True,
                )

                # Log less verbosely during simulation
                # logger.success("Completion generated", emoji_key="success", provider=provider_name, model=result.model)

                await asyncio.sleep(0.2)  # Shorter delay

            except Exception as e:
                logger.error(
                    f"Error simulating completion for {provider_name}: {str(e)}", emoji_key="error"
                )
                metrics.record_request(
                    provider=provider_name,
                    model=model_to_use,  # Log error against intended model
                    input_tokens=0,  # Assume 0 tokens on error for simplicity
                    output_tokens=0,
                    cost=0.0,
                    duration=time.time() - start_time,  # Log duration even on error
                    success=False,  # Mark as failed
                )

    logger.info("Finished simulating LLM usage", emoji_key="complete")
    return metrics


async def demonstrate_metrics_tracking(tracker: CostTracker = None):
    """Demonstrate metrics tracking functionality using Rich."""
    console.print(Rule("[bold blue]Metrics Tracking Demonstration[/bold blue]"))
    logger.info("Starting metrics tracking demonstration", emoji_key="start")

    metrics = get_metrics_tracker(reset_on_start=True)
    await simulate_llm_usage(tracker)
    stats = metrics.get_stats()

    # Use the standardized display utility instead of custom code
    display_analytics_metrics(stats)

    return stats


async def demonstrate_analytics_reporting():
    """Demonstrate analytics reporting functionality."""
    console.print(Rule("[bold blue]Analytics Reporting Demonstration[/bold blue]"))
    logger.info("Starting analytics reporting demonstration", emoji_key="start")

    metrics = get_metrics_tracker()
    stats = metrics.get_stats()
    if stats["general"]["requests_total"] == 0:
        logger.warning("No metrics data found. Running simulation first.", emoji_key="warning")
        await simulate_llm_usage()
        stats = metrics.get_stats()

    # --- Perform calculations directly from stats ---
    general_stats = stats.get("general", {})
    provider_stats = stats.get("providers", {})
    model_stats = stats.get("models", {})
    daily_usage_stats = stats.get("daily_usage", [])
    total_cost = general_stats.get("cost_total", 0.0)
    total_tokens = general_stats.get("tokens_total", 0)

    # Calculate cost by provider
    cost_by_provider = []
    if total_cost > 0:
        cost_by_provider = [
            {
                "name": provider,
                "value": data.get("cost", 0.0),
                "percentage": data.get("cost", 0.0) / total_cost * 100 if total_cost > 0 else 0,
            }
            for provider, data in provider_stats.items()
        ]
        cost_by_provider.sort(key=lambda x: x["value"], reverse=True)

    # Calculate cost by model
    cost_by_model = []
    if total_cost > 0:
        cost_by_model = [
            {
                "name": model,
                "value": data.get("cost", 0.0),
                "percentage": data.get("cost", 0.0) / total_cost * 100 if total_cost > 0 else 0,
            }
            for model, data in model_stats.items()
        ]
        cost_by_model.sort(key=lambda x: x["value"], reverse=True)

    # Calculate tokens by provider
    tokens_by_provider = []
    if total_tokens > 0:
        tokens_by_provider = [
            {
                "name": provider,
                "value": data.get("tokens", 0),
                "percentage": data.get("tokens", 0) / total_tokens * 100 if total_tokens > 0 else 0,
            }
            for provider, data in provider_stats.items()
        ]
        tokens_by_provider.sort(key=lambda x: x["value"], reverse=True)

    # Calculate tokens by model
    tokens_by_model = []
    if total_tokens > 0:
        tokens_by_model = [
            {
                "name": model,
                "value": data.get("tokens", 0),
                "percentage": data.get("tokens", 0) / total_tokens * 100 if total_tokens > 0 else 0,
            }
            for model, data in model_stats.items()
        ]
        tokens_by_model.sort(key=lambda x: x["value"], reverse=True)

    # Calculate daily cost trend (simplified: just show daily cost, no % change)
    daily_cost_trend = [
        {"date": day.get("date"), "cost": day.get("cost", 0.0)} for day in daily_usage_stats
    ]
    daily_cost_trend.sort(key=lambda x: x["date"])  # Sort by date
    # --------------------------------------------------

    # Display reports using tables (using the calculated data)
    # Provider cost report
    if cost_by_provider:
        provider_cost_table = Table(
            title="[bold green]Cost by Provider[/bold green]", box=box.ROUNDED
        )
        provider_cost_table.add_column("Provider", style="magenta")
        provider_cost_table.add_column("Cost", style="green", justify="right")
        provider_cost_table.add_column("Percentage", style="cyan", justify="right")

        for item in cost_by_provider:
            provider_cost_table.add_row(
                escape(item["name"]), f"${item['value']:.6f}", f"{item['percentage']:.1f}%"
            )
        console.print(provider_cost_table)
        console.print()

    # Model cost report
    if cost_by_model:
        model_cost_table = Table(title="[bold green]Cost by Model[/bold green]", box=box.ROUNDED)
        model_cost_table.add_column("Model", style="blue")
        model_cost_table.add_column("Cost", style="green", justify="right")
        model_cost_table.add_column("Percentage", style="cyan", justify="right")

        for item in cost_by_model:
            model_cost_table.add_row(
                escape(item["name"]), f"${item['value']:.6f}", f"{item['percentage']:.1f}%"
            )
        console.print(model_cost_table)
        console.print()

    # Tokens by provider report
    if tokens_by_provider:
        tokens_provider_table = Table(
            title="[bold green]Tokens by Provider[/bold green]", box=box.ROUNDED
        )
        tokens_provider_table.add_column("Provider", style="magenta")
        tokens_provider_table.add_column("Tokens", style="white", justify="right")
        tokens_provider_table.add_column("Percentage", style="cyan", justify="right")

        for item in tokens_by_provider:
            tokens_provider_table.add_row(
                escape(item["name"]), f"{item['value']:,}", f"{item['percentage']:.1f}%"
            )
        console.print(tokens_provider_table)
        console.print()

    # Tokens by model report
    if tokens_by_model:
        tokens_model_table = Table(
            title="[bold green]Tokens by Model[/bold green]", box=box.ROUNDED
        )
        tokens_model_table.add_column("Model", style="blue")
        tokens_model_table.add_column("Tokens", style="white", justify="right")
        tokens_model_table.add_column("Percentage", style="cyan", justify="right")

        for item in tokens_by_model:
            tokens_model_table.add_row(
                escape(item["name"]), f"{item['value']:,}", f"{item['percentage']:.1f}%"
            )
        console.print(tokens_model_table)
        console.print()

    # Daily cost trend report
    if daily_cost_trend:
        daily_trend_table = Table(
            title="[bold green]Daily Cost Trend[/bold green]", box=box.ROUNDED
        )
        daily_trend_table.add_column("Date", style="yellow")
        daily_trend_table.add_column("Cost", style="green", justify="right")
        # daily_trend_table.add_column("Change", style="cyan", justify="right") # Removed change calculation for simplicity

        for item in daily_cost_trend:
            # change_str = f"{item.get('change', 0):.1f}%" if 'change' in item else "N/A"
            # change_style = ""
            # if 'change' in item:
            #     if item['change'] > 0:
            #         change_style = "[red]+"
            #     elif item['change'] < 0:
            #         change_style = "[green]"

            daily_trend_table.add_row(
                item["date"],
                f"${item['cost']:.6f}",
                # f"{change_style}{change_str}[/]" if change_style else change_str
            )
        console.print(daily_trend_table)
        console.print()

    # Return the calculated data instead of None
    return {
        "cost_by_provider": cost_by_provider,
        "cost_by_model": cost_by_model,
        "tokens_by_provider": tokens_by_provider,
        "tokens_by_model": tokens_by_model,
        "daily_cost_trend": daily_cost_trend,
    }


async def demonstrate_real_time_monitoring():
    """Demonstrate real-time metrics monitoring using Rich Live."""
    console.print(Rule("[bold blue]Real-Time Monitoring Demonstration[/bold blue]"))
    logger.info("Starting real-time monitoring (updates every 2s for 10s)", emoji_key="start")

    metrics = get_metrics_tracker()  # Use existing tracker

    def generate_stats_table() -> Table:
        """Generates a Rich Table with current stats."""
        stats = metrics.get_stats()["general"]
        table = Table(title="Live LLM Usage Stats", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white", justify="right")
        table.add_row("Total Requests", f"{stats['requests_total']:,}")
        table.add_row("Total Tokens", f"{stats['tokens_total']:,}")
        table.add_row("Total Cost", f"${stats['cost_total']:.6f}")
        table.add_row("Total Errors", f"{stats['errors_total']:,}")
        return table

    try:
        with Live(generate_stats_table(), refresh_per_second=0.5, console=console) as live:
            # Simulate some activity in the background while monitoring
            # We could run simulate_llm_usage again, but let's just wait for demo purposes
            end_time = time.time() + 10  # Monitor for 10 seconds
            while time.time() < end_time:
                # In a real app, other tasks would be modifying metrics here
                live.update(generate_stats_table())
                await asyncio.sleep(2)  # Update display every 2 seconds

            # Final update
            live.update(generate_stats_table())

    except Exception as e:
        logger.error(f"Error during live monitoring: {e}", emoji_key="error", exc_info=True)

    logger.info("Finished real-time monitoring demonstration", emoji_key="complete")
    console.print()


async def main():
    """Run all analytics and reporting demonstrations."""
    tracker = CostTracker()  # Create cost tracker instance
    try:
        # Demonstrate metrics tracking (includes simulation)
        await demonstrate_metrics_tracking(tracker)

        # Demonstrate report generation
        await demonstrate_analytics_reporting()

        # Demonstrate real-time monitoring
        await demonstrate_real_time_monitoring()

        # Display final cost summary
        tracker.display_summary(console)

    except Exception as e:
        logger.critical(f"Analytics demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1

    logger.success("Analytics & Reporting Demo Finished Successfully!", emoji_key="complete")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
