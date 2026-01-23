#!/usr/bin/env python
"""Claude integration demonstration using Ultimate MCP Server."""

import asyncio
import sys
import time
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

# Initialize logger
logger = get_logger("example.claude_integration_demo")


async def compare_claude_models(tracker: CostTracker):
    """Compare different Claude models."""
    console.print(Rule("[bold blue]Claude Model Comparison[/bold blue]"))
    logger.info("Starting Claude models comparison", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway("claude-demo", register_tools=False)

    # Initialize providers
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()

    provider_name = Provider.ANTHROPIC.value
    try:
        # Get the provider from the gateway
        provider = gateway.providers.get(provider_name)
        if not provider:
            logger.error(
                f"Provider {provider_name} not available or initialized", emoji_key="error"
            )
            return

        logger.info(f"Using provider: {provider_name}", emoji_key="provider")

        models = await provider.list_models()
        model_names = [m["id"] for m in models]  # Extract names from model dictionaries
        console.print(
            f"Found {len(model_names)} Claude models: [cyan]{escape(str(model_names))}[/cyan]"
        )

        # Select specific models to compare (Ensure these are valid and available)
        claude_models = [
            "anthropic/claude-3-7-sonnet-20250219",
            "anthropic/claude-3-5-haiku-20241022",
        ]
        # Filter based on available models
        models_to_compare = [m for m in claude_models if m in model_names]
        if not models_to_compare:
            logger.error(
                "None of the selected models for comparison are available. Exiting comparison.",
                emoji_key="error",
            )
            console.print("[red]Selected models not found in available list.[/red]")
            return
        console.print(f"Comparing models: [yellow]{escape(str(models_to_compare))}[/yellow]")

        prompt = """
        Explain the concept of quantum entanglement in a way that a high school student would understand.
        Keep your response brief and accessible.
        """
        console.print(f"[cyan]Using Prompt:[/cyan] {escape(prompt.strip())[:100]}...")

        results_data = []

        for model_name in models_to_compare:
            try:
                logger.info(f"Testing model: {model_name}", emoji_key="model")
                start_time = time.time()
                result = await provider.generate_completion(
                    prompt=prompt, model=model_name, temperature=0.3, max_tokens=300
                )
                processing_time = time.time() - start_time

                # Track the cost
                tracker.add_call(result)

                results_data.append(
                    {
                        "model": model_name,
                        "text": result.text,
                        "tokens": {
                            "input": result.input_tokens,
                            "output": result.output_tokens,
                            "total": result.total_tokens,
                        },
                        "cost": result.cost,
                        "time": processing_time,
                    }
                )

                logger.success(
                    f"Completion for {model_name} successful",
                    emoji_key="success",
                    # Tokens/cost/time logged implicitly by storing in results_data
                )

            except Exception as e:
                logger.error(
                    f"Error testing model {model_name}: {str(e)}", emoji_key="error", exc_info=True
                )
                # Optionally add an error entry to results_data if needed

        # Display comparison results using Rich
        if results_data:
            console.print(Rule("[bold green]Comparison Results[/bold green]"))

            for result_item in results_data:
                model = result_item["model"]
                time_s = result_item["time"]
                tokens = result_item.get("tokens", {}).get("total", 0)
                tokens_per_second = tokens / time_s if time_s > 0 else 0
                cost = result_item.get("cost", 0.0)
                text = result_item.get("text", "[red]Error generating response[/red]").strip()

                stats_line = (
                    f"Time: [yellow]{time_s:.2f}s[/yellow] | "
                    f"Tokens: [cyan]{tokens}[/cyan] | "
                    f"Speed: [blue]{tokens_per_second:.1f} tok/s[/blue] | "
                    f"Cost: [green]${cost:.6f}[/green]"
                )

                console.print(
                    Panel(
                        escape(text),
                        title=f"[bold magenta]{escape(model)}[/bold magenta]",
                        subtitle=stats_line,
                        border_style="blue",
                        expand=False,
                    )
                )
            console.print()

    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}", emoji_key="error", exc_info=True)
        # Optionally re-raise or handle differently


async def demonstrate_system_prompt(tracker: CostTracker):
    """Demonstrate Claude with system prompts."""
    console.print(Rule("[bold blue]Claude System Prompt Demonstration[/bold blue]"))
    logger.info("Demonstrating Claude with system prompts", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway("claude-demo", register_tools=False)

    # Initialize providers
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()

    provider_name = Provider.ANTHROPIC.value
    try:
        # Get the provider from the gateway
        provider = gateway.providers.get(provider_name)
        if not provider:
            logger.error(
                f"Provider {provider_name} not available or initialized", emoji_key="error"
            )
            return

        # Use a fast Claude model (ensure it's available)
        model = "anthropic/claude-3-5-haiku-20241022"
        available_models = await provider.list_models()
        if model not in [m["id"] for m in available_models]:
            logger.warning(
                f"Model {model} not available, falling back to default.", emoji_key="warning"
            )
            model = provider.get_default_model()
            if not model:
                logger.error(
                    "No suitable Claude model found for system prompt demo.", emoji_key="error"
                )
                return
        logger.info(f"Using model: {model}", emoji_key="model")

        system_prompt = """
You are a helpful assistant with expertise in physics.
Keep all explanations accurate but very concise.
Always provide real-world examples to illustrate concepts.
"""
        user_prompt = "Explain the concept of gravity."

        logger.info("Generating completion with system prompt", emoji_key="processing")

        result = await provider.generate_completion(
            prompt=user_prompt,
            model=model,
            temperature=0.7,
            system=system_prompt,
            max_tokens=1000,  # Increased max_tokens
        )

        # Track the cost
        tracker.add_call(result)

        logger.success("Completion with system prompt successful", emoji_key="success")

        # Display result using Rich Panels
        console.print(
            Panel(
                escape(system_prompt.strip()),
                title="[bold cyan]System Prompt[/bold cyan]",
                border_style="dim cyan",
                expand=False,
            )
        )
        console.print(
            Panel(
                escape(user_prompt.strip()),
                title="[bold yellow]User Prompt[/bold yellow]",
                border_style="dim yellow",
                expand=False,
            )
        )
        console.print(
            Panel(
                escape(result.text.strip()),
                title="[bold green]Claude Response[/bold green]",
                border_style="green",
                expand=False,
            )
        )

        # Display stats in a small table
        stats_table = Table(
            title="Execution Stats", show_header=False, box=box.MINIMAL, expand=False
        )
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Input Tokens", str(result.input_tokens))
        stats_table.add_row("Output Tokens", str(result.output_tokens))
        stats_table.add_row("Cost", f"${result.cost:.6f}")
        stats_table.add_row("Processing Time", f"{result.processing_time:.3f}s")
        console.print(stats_table)
        console.print()

    except Exception as e:
        logger.error(
            f"Error in system prompt demonstration: {str(e)}", emoji_key="error", exc_info=True
        )
        # Optionally re-raise or handle differently


async def explore_claude_models():
    """Display available Claude models."""
    console.print(Rule("[bold cyan]Available Claude Models[/bold cyan]"))

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway("claude-demo", register_tools=False)

    # Initialize providers
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()

    # Get provider from the gateway
    provider = gateway.providers.get(Provider.ANTHROPIC.value)
    if not provider:
        logger.error(
            f"Provider {Provider.ANTHROPIC.value} not available or initialized", emoji_key="error"
        )
        return

    # Get list of available models
    models = await provider.list_models()
    model_names = [m["id"] for m in models]  # Extract names from model dictionaries
    console.print(
        f"Found {len(model_names)} Claude models: [cyan]{escape(str(model_names))}[/cyan]"
    )


async def main():
    """Run Claude integration examples."""
    tracker = CostTracker()  # Instantiate tracker here
    try:
        # Run model comparison
        await compare_claude_models(tracker)  # Pass tracker

        console.print()  # Add space between sections

        # Run system prompt demonstration
        await demonstrate_system_prompt(tracker)  # Pass tracker

        # Run explore Claude models
        await explore_claude_models()

        # Display final summary
        tracker.display_summary(console)  # Display summary at the end

    except Exception as e:
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1

    logger.success("Claude Integration Demo Finished Successfully!", emoji_key="complete")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
