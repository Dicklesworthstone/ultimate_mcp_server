#!/usr/bin/env python
"""Multi-provider completion demo using Ultimate MCP Server."""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
# These imports need to be below sys.path modification, which is why they have noqa comments
from rich import box  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

# Project imports
from ultimate_mcp_server.constants import Provider  # noqa: E402
from ultimate_mcp_server.core.server import Gateway  # noqa: E402
from ultimate_mcp_server.utils import get_logger  # noqa: E402
from ultimate_mcp_server.utils.display import CostTracker  # Import CostTracker
from ultimate_mcp_server.utils.logging.console import console  # noqa: E402

# Initialize logger and console
logger = get_logger("example.multi_provider")


async def run_provider_comparison(tracker: CostTracker):
    """Run a comparison of completions across multiple providers using Rich."""
    console.print(Rule("[bold blue]Multi-Provider Completion Comparison[/bold blue]"))
    logger.info("Starting multi-provider comparison demo", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway("multi-provider-demo", register_tools=False)

    # Initialize providers
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()

    prompt = "Explain the advantages of quantum computing in 3-4 sentences."
    console.print(f"[cyan]Prompt:[/cyan] {escape(prompt)}")

    # Use model names directly if providers are inferred or handled by get_provider
    configs = [
        {"provider": Provider.OPENAI.value, "model": "gpt-4.1-mini"},
        {"provider": Provider.ANTHROPIC.value, "model": "claude-3-5-haiku-20241022"},
        {"provider": Provider.GEMINI.value, "model": "gemini-2.0-flash-lite"},
        {"provider": Provider.DEEPSEEK.value, "model": "deepseek-chat"},
        {"provider": Provider.GROK.value, "model": "grok-3-mini-latest"},
        {"provider": Provider.OPENROUTER.value, "model": "mistralai/mistral-nemo"},
        {"provider": Provider.OLLAMA.value, "model": "llama3.2"},
    ]

    results_data = []

    for config in configs:
        provider_name = config["provider"]
        model_name = config["model"]

        provider = gateway.providers.get(provider_name)
        if not provider:
            logger.warning(
                f"Provider {provider_name} not available or initialized, skipping.",
                emoji_key="warning",
            )
            continue

        try:
            logger.info(
                f"Generating completion with {provider_name}/{model_name}...",
                emoji_key="processing",
            )
            result = await provider.generate_completion(
                prompt=prompt, model=model_name, temperature=0.7, max_tokens=150
            )

            # Track the cost
            tracker.add_call(result)

            results_data.append(
                {
                    "provider": provider_name,
                    "model": model_name,
                    "text": result.text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cost": result.cost,
                    "processing_time": result.processing_time,
                }
            )
            logger.success(
                f"Completion from {provider_name}/{model_name} successful.", emoji_key="success"
            )

        except Exception as e:
            logger.error(
                f"Error with {provider_name}/{model_name}: {e}", emoji_key="error", exc_info=True
            )
            # Optionally store error result
            results_data.append(
                {
                    "provider": provider_name,
                    "model": model_name,
                    "text": f"[red]Error: {escape(str(e))}[/red]",
                    "cost": 0.0,
                    "processing_time": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )

    # Print comparison results using Rich Panels
    console.print(Rule("[bold green]Comparison Results[/bold green]"))
    for result in results_data:
        stats_line = (
            f"Cost: [green]${result['cost']:.6f}[/green] | "
            f"Time: [yellow]{result['processing_time']:.2f}s[/yellow] | "
            f"Tokens: [cyan]{result['input_tokens']} in, {result['output_tokens']} out[/cyan]"
        )
        console.print(
            Panel(
                escape(result["text"].strip()),
                title=f"[bold magenta]{escape(result['provider'])} / {escape(result['model'])}[/bold magenta]",
                subtitle=stats_line,
                border_style="blue",
                expand=False,
            )
        )

    # Filter out error results before calculating summary stats
    valid_results = [r for r in results_data if "Error" not in r["text"]]

    if valid_results:
        summary_table = Table(title="Comparison Summary", box=box.ROUNDED, show_header=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        try:
            fastest = min(valid_results, key=lambda r: r["processing_time"])
            summary_table.add_row(
                "⚡ Fastest",
                f"{escape(fastest['provider'])}/{escape(fastest['model'])} ({fastest['processing_time']:.2f}s)",
            )
        except ValueError:
            pass  # Handle empty list

        try:
            cheapest = min(valid_results, key=lambda r: r["cost"])
            summary_table.add_row(
                "💰 Cheapest",
                f"{escape(cheapest['provider'])}/{escape(cheapest['model'])} (${cheapest['cost']:.6f})",
            )
        except ValueError:
            pass

        try:
            most_tokens = max(valid_results, key=lambda r: r["output_tokens"])
            summary_table.add_row(
                "📄 Most Tokens",
                f"{escape(most_tokens['provider'])}/{escape(most_tokens['model'])} ({most_tokens['output_tokens']} tokens)",
            )
        except ValueError:
            pass

        if summary_table.row_count > 0:
            console.print(summary_table)
    else:
        console.print("[yellow]No valid results to generate summary.[/yellow]")

    # Display final summary
    tracker.display_summary(console)  # Display summary at the end

    console.print()  # Final spacing
    return 0


async def main():
    """Run the demo."""
    tracker = CostTracker()  # Instantiate tracker
    try:
        return await run_provider_comparison(tracker)  # Pass tracker
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical")
        return 1


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
