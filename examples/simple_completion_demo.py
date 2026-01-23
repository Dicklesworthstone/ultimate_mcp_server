#!/usr/bin/env python
"""
Simple completion demo using Ultimate MCP Server's direct provider functionality.

This example demonstrates how to:
1. Initialize the Ultimate MCP Server Gateway
2. Connect directly to an LLM provider (OpenAI)
3. Generate a text completion with a specific model
4. Track and display token usage and costs

The demo bypasses the MCP tool interface and interacts directly with provider APIs,
which is useful for understanding the underlying provider connections or when you need
lower-level access to provider-specific features. It also showcases the CostTracker
utility for monitoring token usage and associated costs across multiple requests.

This script can be run as a standalone Python module and serves as a minimal example of
direct provider integration with the Ultimate MCP Server framework.

Usage:
    python examples/simple_completion_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger
logger = get_logger("example.simple_completion")


async def run_model_demo(tracker: CostTracker):
    """
    Run a simple completion demo using direct provider access to LLM APIs.

    This function demonstrates the complete workflow for generating text completions
    using the Ultimate MCP Server framework with direct provider access:

    1. Initialize a Gateway instance without registering tools
    2. Initialize the LLM providers from configuration
    3. Access a specific provider (OpenAI in this case)
    4. Generate a completion with a specific prompt and model
    5. Display the completion result with Rich formatting
    6. Track and display token usage and cost metrics

    Direct provider access (vs. using MCP tools) offers more control over provider-specific
    parameters and is useful for applications that need to customize provider interactions
    beyond what the standard MCP tools offer.

    Args:
        tracker: CostTracker instance to record token usage and costs for this operation.
                The tracker will be updated with the completion results.

    Returns:
        int: Exit code - 0 for success, 1 for failure

    Raises:
        Various exceptions may be raised by the provider initialization or completion
        generation process, but these are logged and contained within this function.
    """
    logger.info("Starting simple completion demo", emoji_key="start")
    # Use Rich Rule for title
    console.print(Rule("[bold blue]Simple Completion Demo[/bold blue]"))

    # Create Gateway instance
    gateway = Gateway("simple-demo", register_tools=False)

    # Initialize providers
    logger.info("Initializing providers", emoji_key="provider")
    await gateway._initialize_providers()

    # Get provider (OpenAI)
    provider_name = Provider.OPENAI.value
    provider = gateway.providers.get(provider_name)

    if not provider:
        logger.error(f"Provider {provider_name} not available", emoji_key="error")
        return 1

    logger.success(f"Provider {provider_name} initialized", emoji_key="success")

    # List available models
    models = await provider.list_models()
    logger.info(f"Available models: {len(models)}", emoji_key="model")

    # Pick a valid model from the provider
    model = "gpt-4.1-mini"  # A valid model from constants.py

    # Generate a completion
    prompt = "Explain quantum computing in simple terms."

    logger.info(f"Generating completion with {model}", emoji_key="processing")
    result = await provider.generate_completion(
        prompt=prompt, model=model, temperature=0.7, max_tokens=150
    )

    # Print the result using Rich Panel
    logger.success("Completion generated successfully!", emoji_key="success")
    console.print(
        Panel(
            result.text.strip(),
            title=f"Quantum Computing Explanation ({model})",
            subtitle=f"Prompt: {prompt}",
            border_style="green",
            expand=False,
        )
    )

    # Print stats using Rich Table
    stats_table = Table(title="Completion Stats", show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    stats_table.add_row("Input Tokens", str(result.input_tokens))
    stats_table.add_row("Output Tokens", str(result.output_tokens))
    stats_table.add_row("Cost", f"${result.cost:.6f}")
    stats_table.add_row("Processing Time", f"{result.processing_time:.2f}s")
    console.print(stats_table)

    # Track the call
    tracker.add_call(result)

    # Display cost summary
    tracker.display_summary(console)

    return 0


async def main():
    """
    Entry point function that sets up the demo environment and error handling.

    This function:
    1. Creates a CostTracker instance to monitor token usage and costs
    2. Calls the run_model_demo function within a try-except block
    3. Handles and logs any uncaught exceptions
    4. Returns an appropriate exit code based on execution success/failure

    The separation between main() and run_model_demo() allows for clean error handling
    and resource management at the top level while keeping the demo logic organized
    in its own function.

    Returns:
        int: Exit code - 0 for success, 1 for failure
    """
    tracker = CostTracker()
    try:
        return await run_model_demo(tracker)
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical")
        return 1


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
