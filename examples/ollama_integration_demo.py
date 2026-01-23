#!/usr/bin/env python
"""Ollama integration demonstration using Ultimate MCP Server."""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration system instead of using update_ollama_env
from ultimate_mcp_server.config import get_config

# Load the config to ensure environment variables from .env are read
config = get_config()

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
from ultimate_mcp_server.utils.display import CostTracker  # Import CostTracker  # noqa: E402
from ultimate_mcp_server.utils.logging.console import console  # noqa: E402

# Initialize logger
logger = get_logger("example.ollama_integration_demo")


async def compare_ollama_models(tracker: CostTracker):
    """Compare different Ollama models."""
    console.print(Rule("[bold blue]🦙 Ollama Model Comparison[/bold blue]"))
    logger.info("Starting Ollama models comparison", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway("ollama-demo", register_tools=False)

    try:
        # Initialize providers
        logger.info("Initializing providers...", emoji_key="provider")
        await gateway._initialize_providers()

        provider_name = Provider.OLLAMA.value
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
                f"Found {len(model_names)} Ollama models: [cyan]{escape(str(model_names))}[/cyan]"
            )

            # Select specific models to compare (adjust these based on what you have installed locally)
            ollama_models = [
                "mix_77/gemma3-qat-tools:27b",
                "JollyLlama/GLM-Z1-32B-0414-Q4_K_M:latest",
                "llama3.2-vision:latest",
            ]
            # Filter based on available models
            models_to_compare = [m for m in ollama_models if m in model_names]
            if not models_to_compare:
                logger.error(
                    "None of the selected models for comparison are available. Please use 'ollama pull MODEL' to download models first.",
                    emoji_key="error",
                )
                console.print(
                    "[red]Selected models not found. Use 'ollama pull mix_77/gemma3-qat-tools:27b' to download models first.[/red]"
                )
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
                        f"Error testing model {model_name}: {str(e)}",
                        emoji_key="error",
                        exc_info=True,
                    )
                    # Optionally add an error entry to results_data if needed

            # Display comparison results using Rich
            if results_data:
                console.print(Rule("[bold green]Comparison Results[/bold green]"))

                for result_item in results_data:
                    model = result_item["model"]
                    time_s = result_item["time"]
                    tokens = result_item.get("tokens", {}).get("total", 0)

                    # If tokens is 0 but we have text, estimate based on text length
                    text = result_item.get("text", "").strip()
                    if tokens == 0 and text:
                        # Rough estimate: ~1.3 tokens per word plus some for punctuation
                        tokens = len(text.split()) + len(text) // 10
                        # Update the result item for cost tracking
                        result_item["tokens"]["output"] = tokens
                        result_item["tokens"]["total"] = tokens + result_item["tokens"]["input"]

                    tokens_per_second = tokens / time_s if time_s > 0 and tokens > 0 else 0
                    cost = result_item.get("cost", 0.0)

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
    finally:
        # Ensure resources are cleaned up
        if hasattr(gateway, "shutdown"):
            await gateway.shutdown()


async def demonstrate_system_prompt(tracker: CostTracker):
    """Demonstrate Ollama with system prompts."""
    console.print(Rule("[bold blue]🦙 Ollama System Prompt Demonstration[/bold blue]"))
    logger.info("Demonstrating Ollama with system prompts", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway("ollama-demo", register_tools=False)

    try:
        # Initialize providers
        logger.info("Initializing providers...", emoji_key="provider")
        await gateway._initialize_providers()

        provider_name = Provider.OLLAMA.value
        try:
            # Get the provider from the gateway
            provider = gateway.providers.get(provider_name)
            if not provider:
                logger.error(
                    f"Provider {provider_name} not available or initialized", emoji_key="error"
                )
                return

            # Use mix_77/gemma3-qat-tools:27b (ensure it's available)
            model = "mix_77/gemma3-qat-tools:27b"
            available_models = await provider.list_models()
            model_names = [m["id"] for m in available_models]

            if model not in model_names:
                logger.warning(
                    f"Model {model} not available, please run 'ollama pull mix_77/gemma3-qat-tools:27b'",
                    emoji_key="warning",
                )
                console.print(
                    "[yellow]Model not found. Please run 'ollama pull mix_77/gemma3-qat-tools:27b' to download it.[/yellow]"
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
                    title="[bold green]Ollama Response[/bold green]",
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
    finally:
        # Ensure resources are cleaned up
        if hasattr(gateway, "shutdown"):
            await gateway.shutdown()


async def demonstrate_streaming(tracker: CostTracker):
    """Demonstrate Ollama streaming capabilities."""
    console.print(Rule("[bold blue]🦙 Ollama Streaming Demonstration[/bold blue]"))
    logger.info("Demonstrating Ollama streaming capabilities", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway("ollama-demo", register_tools=False)

    try:
        # Initialize providers
        logger.info("Initializing providers...", emoji_key="provider")
        await gateway._initialize_providers()

        provider_name = Provider.OLLAMA.value
        try:
            # Get the provider from the gateway
            provider = gateway.providers.get(provider_name)
            if not provider:
                logger.error(
                    f"Provider {provider_name} not available or initialized", emoji_key="error"
                )
                return

            # Use any available Ollama model
            model = provider.get_default_model()
            logger.info(f"Using model: {model}", emoji_key="model")

            prompt = "Write a short poem about programming with AI"
            console.print(
                Panel(
                    escape(prompt.strip()),
                    title="[bold yellow]Prompt[/bold yellow]",
                    border_style="dim yellow",
                    expand=False,
                )
            )

            logger.info("Generating streaming completion", emoji_key="processing")

            console.print("[bold green]Streaming response:[/bold green]")

            # Stream completion and display tokens as they arrive
            output_text = ""
            token_count = 0
            start_time = time.time()

            stream = provider.generate_completion_stream(
                prompt=prompt, model=model, temperature=0.7, max_tokens=200
            )

            final_metadata = None
            async for chunk, metadata in stream:
                # Display the streaming chunk
                console.print(chunk, end="", highlight=False)
                output_text += chunk
                token_count += 1
                final_metadata = metadata

            # Newline after streaming is complete
            console.print()
            console.print()

            processing_time = time.time() - start_time

            if final_metadata:
                # Track the cost at the end
                if tracker:
                    # Create a simple object with the necessary attributes for the tracker
                    class StreamingCall:
                        def __init__(self, metadata):
                            self.model = metadata.get("model", "")
                            self.provider = metadata.get("provider", "")
                            self.input_tokens = metadata.get("input_tokens", 0)
                            self.output_tokens = metadata.get("output_tokens", 0)
                            self.total_tokens = metadata.get("total_tokens", 0)
                            self.cost = metadata.get("cost", 0.0)
                            self.processing_time = metadata.get("processing_time", 0.0)

                    tracker.add_call(StreamingCall(final_metadata))

                # Display stats
                stats_table = Table(
                    title="Streaming Stats", show_header=False, box=box.MINIMAL, expand=False
                )
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="white")
                stats_table.add_row("Input Tokens", str(final_metadata.get("input_tokens", 0)))
                stats_table.add_row("Output Tokens", str(final_metadata.get("output_tokens", 0)))
                stats_table.add_row("Cost", f"${final_metadata.get('cost', 0.0):.6f}")
                stats_table.add_row("Processing Time", f"{processing_time:.3f}s")
                stats_table.add_row(
                    "Tokens per Second",
                    f"{final_metadata.get('output_tokens', 0) / processing_time if processing_time > 0 else 0:.1f}",
                )
                console.print(stats_table)

            logger.success("Streaming demonstration completed", emoji_key="success")

        except Exception as e:
            logger.error(
                f"Error in streaming demonstration: {str(e)}", emoji_key="error", exc_info=True
            )
    finally:
        # Ensure resources are cleaned up
        if hasattr(gateway, "shutdown"):
            await gateway.shutdown()


async def explore_ollama_models():
    """Display available Ollama models."""
    console.print(Rule("[bold cyan]🦙 Available Ollama Models[/bold cyan]"))

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway("ollama-demo", register_tools=False)

    try:
        # Initialize providers
        logger.info("Initializing providers...", emoji_key="provider")
        await gateway._initialize_providers()

        # Get provider from the gateway
        provider = gateway.providers.get(Provider.OLLAMA.value)
        if not provider:
            logger.error(
                f"Provider {Provider.OLLAMA.value} not available or initialized", emoji_key="error"
            )
            console.print(
                "[red]Ollama provider not available. Make sure Ollama is installed and running on your machine.[/red]"
            )
            console.print(
                "[yellow]Visit https://ollama.com/download for installation instructions.[/yellow]"
            )
            return

        # Get list of available models
        try:
            models = await provider.list_models()

            if not models:
                console.print(
                    "[yellow]No Ollama models found. Use 'ollama pull MODEL' to download models.[/yellow]"
                )
                console.print("Example: [green]ollama pull mix_77/gemma3-qat-tools:27b[/green]")
                return

            # Create a table to display model information
            table = Table(title="Local Ollama Models")
            table.add_column("Model ID", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Size", style="yellow")

            for model in models:
                # Extract size from description if available
                size_str = "Unknown"
                description = model.get("description", "")

                # Check if size information is in the description (format: "... (X.XX GB)")
                import re

                size_match = re.search(r"\((\d+\.\d+) GB\)", description)
                if size_match:
                    size_gb = float(size_match.group(1))
                    size_str = f"{size_gb:.2f} GB"

                table.add_row(model["id"], description, size_str)

            console.print(table)
            console.print("\n[dim]Note: To add more models, use 'ollama pull MODEL_NAME'[/dim]")
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}", emoji_key="error")
            console.print(f"[red]Failed to list Ollama models: {str(e)}[/red]")
            console.print(
                "[yellow]Make sure Ollama is installed and running on your machine.[/yellow]"
            )
            console.print(
                "[yellow]Visit https://ollama.com/download for installation instructions.[/yellow]"
            )
    finally:
        # Ensure resources are cleaned up
        if hasattr(gateway, "shutdown"):
            await gateway.shutdown()


async def main():
    """Run Ollama integration examples."""
    console.print(
        Panel(
            "[bold]This demonstration shows how to use Ollama with the Ultimate MCP Server.[/bold]\n"
            "Ollama allows you to run LLMs locally on your own machine without sending data to external services.\n\n"
            "[yellow]Make sure you have Ollama installed and running:[/yellow]\n"
            "- Download from [link]https://ollama.com/download[/link]\n"
            "- Pull models with [green]ollama pull mix_77/gemma3-qat-tools:27b[/green] or similar commands",
            title="[bold blue]🦙 Ollama Integration Demo[/bold blue]",
            border_style="blue",
            expand=False,
        )
    )

    tracker = CostTracker()  # Instantiate tracker here
    try:
        # First show available models
        await explore_ollama_models()

        console.print()  # Add space between sections

        # Run model comparison
        await compare_ollama_models(tracker)  # Pass tracker

        console.print()  # Add space between sections

        # Run system prompt demonstration
        await demonstrate_system_prompt(tracker)  # Pass tracker

        console.print()  # Add space between sections

        # Run streaming demonstration
        await demonstrate_streaming(tracker)  # Pass tracker

        # Display final summary
        tracker.display_summary(console)  # Display summary at the end

    except Exception as e:
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    finally:
        # Clean up any remaining aiohttp resources by forcing garbage collection
        import gc

        gc.collect()

    logger.success("Ollama Integration Demo Finished Successfully!", emoji_key="complete")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
