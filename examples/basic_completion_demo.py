#!/usr/bin/env python
"""Basic completion example using Ultimate MCP Server."""

import argparse  # Add argparse import
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
# These imports need to be below sys.path modification, which is why they have noqa comments
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

# Project imports
from ultimate_mcp_server.constants import Provider  # noqa: E402
from ultimate_mcp_server.core.providers.base import ModelResponse  # noqa: E402
from ultimate_mcp_server.core.server import Gateway  # noqa: E402
from ultimate_mcp_server.utils import get_logger  # noqa: E402
from ultimate_mcp_server.utils.display import (  # Import CostTracker
    CostTracker,
    display_completion_result,
)
from ultimate_mcp_server.utils.logging.console import console  # noqa: E402

# Initialize logger
logger = get_logger("example.basic_completion")


# Parse command-line arguments
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run completion examples.")
    parser.add_argument("--json-only", action="store_true", help="Run only the JSON mode demos")
    return parser.parse_args()


async def run_basic_completion(gateway, tracker: CostTracker):
    """Run a basic completion example."""
    logger.info("Starting basic completion example", emoji_key="start")
    console.print(Rule("[bold blue]Basic Completion[/bold blue]"))

    # Prompt to complete
    prompt = "Explain the concept of federated learning in simple terms."

    try:
        # Get OpenAI provider from gateway
        provider = gateway.providers.get(Provider.OPENAI.value)
        if not provider:
            logger.error(
                f"Provider {Provider.OPENAI.value} not available or initialized", emoji_key="error"
            )
            return

        # Generate completion using OpenAI
        logger.info("Generating completion...", emoji_key="processing")
        result = await provider.generate_completion(prompt=prompt, temperature=0.7, max_tokens=200)

        # Log simple success message
        logger.success("Completion generated successfully!", emoji_key="success")

        # Display results using the utility function
        display_completion_result(
            console=console,
            result=result,  # Pass the original result object
            title="Federated Learning Explanation",
        )

        # Track cost
        tracker.add_call(result)

    except Exception as e:
        # Use logger for errors, as DetailedLogFormatter handles error panels well
        logger.error(f"Error generating completion: {str(e)}", emoji_key="error", exc_info=True)
        raise


async def run_chat_completion(gateway, tracker: CostTracker):
    """Run a chat completion example."""
    logger.info("Starting chat completion example", emoji_key="start")
    console.print(Rule("[bold blue]Chat Completion[/bold blue]"))

    # Test standard chat completion with OpenAI first as a basic example
    try:
        # Get OpenAI provider from gateway
        provider = gateway.providers.get(Provider.OPENAI.value)
        if not provider:
            logger.warning(
                f"Provider {Provider.OPENAI.value} not available or initialized, skipping standard example",
                emoji_key="warning",
            )
        else:
            # Define chat messages for regular chat completion
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise answers.",
                },
                {
                    "role": "user",
                    "content": "What is the difference between deep learning and machine learning?",
                },
            ]

            # Generate standard chat completion using OpenAI
            logger.info(
                "Generating standard chat completion with OpenAI...", emoji_key="processing"
            )
            result = await provider.generate_completion(
                messages=messages, temperature=0.7, max_tokens=200
            )

            # Log simple success message
            logger.success("Standard chat completion generated successfully!", emoji_key="success")

            # Display results using the utility function
            display_completion_result(
                console=console, result=result, title="Deep Learning vs Machine Learning"
            )

            # Track cost
            tracker.add_call(result)
    except Exception as e:
        logger.error(f"Error generating standard chat completion: {str(e)}", emoji_key="error")

    # Now test JSON mode with ALL providers
    console.print(
        "\n[bold yellow]Testing chat completion with json_mode=True across all providers[/bold yellow]"
    )

    # Define providers to test
    providers_to_try = [
        Provider.OPENAI.value,
        Provider.ANTHROPIC.value,
        Provider.GEMINI.value,
        Provider.OLLAMA.value,
        Provider.DEEPSEEK.value,
    ]

    # Define chat messages for JSON response
    json_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides information in JSON format.",
        },
        {
            "role": "user",
            "content": "List the top 3 differences between deep learning and machine learning as a JSON array with 'difference' and 'explanation' fields.",
        },
    ]

    # Track statistics
    json_successes = 0
    json_failures = 0
    valid_json_count = 0

    # Create a table for results
    results_table = Table(title="JSON Mode Chat Completion Results", show_header=True)
    results_table.add_column("Provider", style="cyan")
    results_table.add_column("Success", style="green")
    results_table.add_column("Valid JSON", style="blue")
    results_table.add_column("Tokens", style="yellow")
    results_table.add_column("Time (s)", style="magenta")

    for provider_name in providers_to_try:
        console.print(f"\n[bold]Testing JSON chat completion with provider: {provider_name}[/bold]")

        try:
            # Get provider from gateway
            provider = gateway.providers.get(provider_name)
            if not provider:
                logger.warning(
                    f"Provider {provider_name} not available or initialized, skipping",
                    emoji_key="warning",
                )
                continue

            # Generate chat completion with json_mode=True
            logger.info(
                f"Generating chat completion with json_mode=True for {provider_name}...",
                emoji_key="processing",
            )
            json_result = await provider.generate_completion(
                messages=json_messages, temperature=0.7, max_tokens=300, json_mode=True
            )

            # Log success message
            logger.success(
                f"{provider_name} JSON chat completion generated successfully!", emoji_key="success"
            )
            json_successes += 1

            # Check if result is valid JSON
            is_valid_json = False
            try:
                parsed_json = json.loads(json_result.text)
                is_valid_json = True
                valid_json_count += 1
                logger.info(f"{provider_name} returned valid JSON", emoji_key="success")
            except json.JSONDecodeError:
                # Try custom extraction for Anthropic-like responses
                if provider_name == Provider.ANTHROPIC.value:
                    try:
                        import re

                        code_block_match = re.search(
                            r"```(?:json)?\s*([\s\S]*?)```", json_result.text
                        )
                        if code_block_match:
                            code_content = code_block_match.group(1).strip()
                            parsed_json = json.loads(code_content)  # noqa: F841
                            is_valid_json = True
                            valid_json_count += 1
                            logger.info(
                                f"{provider_name} returned valid JSON inside code block",
                                emoji_key="success",
                            )
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        is_valid_json = False
                        logger.warning(
                            f"{provider_name} did not return valid JSON", emoji_key="warning"
                        )
                else:
                    logger.warning(
                        f"{provider_name} did not return valid JSON", emoji_key="warning"
                    )

            # Add to results table
            results_table.add_row(
                provider_name,
                "✓",
                "✓" if is_valid_json else "✗",
                f"{json_result.input_tokens}/{json_result.output_tokens}",
                f"{json_result.processing_time:.3f}",
            )

            # Create a custom display for the JSON result
            json_panel = Panel(
                json_result.text[:800] + ("..." if len(json_result.text) > 800 else ""),
                title=f"[cyan]{provider_name}[/cyan] JSON Chat Response [{'✓ Valid' if is_valid_json else '✗ Invalid'} JSON]",
                border_style="green" if is_valid_json else "red",
            )
            console.print(json_panel)

            # Track cost
            tracker.add_call(json_result)

        except Exception as e:
            logger.error(
                f"Error with {provider_name} JSON chat completion: {str(e)}", emoji_key="error"
            )
            json_failures += 1
            results_table.add_row(provider_name, "✗", "✗", "N/A", "N/A")

    # Display summary table
    console.print(results_table)

    # Display summary stats
    summary = Table(title="JSON Mode Chat Completion Summary", show_header=True)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Providers Tested", str(len(providers_to_try)))
    summary.add_row("Successful", str(json_successes))
    summary.add_row("Failed", str(json_failures))
    summary.add_row("Valid JSON", str(valid_json_count))
    console.print(summary)


async def run_streaming_completion(gateway):
    """Run a streaming completion example."""
    logger.info("Starting streaming completion example", emoji_key="start")
    console.print(Rule("[bold blue]Streaming Completion[/bold blue]"))

    # Prompt to complete
    prompt = "Write a short poem about artificial intelligence."

    try:
        # Get OpenAI provider from gateway
        provider = gateway.providers.get(Provider.OPENAI.value)
        if not provider:
            logger.error(
                f"Provider {Provider.OPENAI.value} not available or initialized", emoji_key="error"
            )
            return

        logger.info("Generating streaming completion...", emoji_key="processing")

        # Use Panel for streaming output presentation
        output_panel = Panel("", title="AI Poem (Streaming)", border_style="cyan", expand=False)

        # Start timer
        start_time = time.time()

        full_text = ""
        token_count = 0

        # Use Live display for the streaming output panel
        with Live(output_panel, console=console, refresh_per_second=4) as live:  # noqa: F841
            # Get stream from the provider directly
            stream = provider.generate_completion_stream(
                prompt=prompt, temperature=0.7, max_tokens=200
            )

            async for chunk, _metadata in stream:
                full_text += chunk
                token_count += 1
                # Update the panel content
                output_panel.renderable = full_text

        # Calculate processing time
        processing_time = time.time() - start_time

        # Log simple success message
        logger.success("Streaming completion generated successfully!", emoji_key="success")

        # Display stats using Rich Table
        stats_table = Table(title="Streaming Stats", show_header=False, box=None)
        stats_table.add_column("Metric", style="green")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Chunks Received", str(token_count))
        stats_table.add_row("Processing Time", f"{processing_time:.3f}s")
        console.print(stats_table)

    except Exception as e:
        # Use logger for errors
        logger.error(
            f"Error generating streaming completion: {str(e)}", emoji_key="error", exc_info=True
        )
        raise


async def run_cached_completion(gateway, tracker: CostTracker):
    """Run a completion with caching.

    Note: Since we're not using CompletionClient which has built-in caching,
    this example will make two separate calls to the provider.
    """
    logger.info("Starting cached completion example", emoji_key="start")
    console.print(Rule("[bold blue]Cached Completion Demo[/bold blue]"))

    # Prompt to complete
    prompt = "Explain the concept of federated learning in simple terms."

    try:
        # Get OpenAI provider from gateway
        provider = gateway.providers.get(Provider.OPENAI.value)
        if not provider:
            logger.error(
                f"Provider {Provider.OPENAI.value} not available or initialized", emoji_key="error"
            )
            return

        # First request
        logger.info("First request...", emoji_key="processing")
        start_time1 = time.time()
        result1 = await provider.generate_completion(prompt=prompt, temperature=0.7, max_tokens=200)
        processing_time1 = time.time() - start_time1

        # Track first call
        tracker.add_call(result1)

        # Note: We don't actually have caching here since we're not using CompletionClient
        # So instead we'll just make another call and compare times
        logger.info("Second request...", emoji_key="processing")
        start_time2 = time.time()
        result2 = await provider.generate_completion(  # noqa: F841
            prompt=prompt, temperature=0.7, max_tokens=200
        )
        processing_time2 = time.time() - start_time2

        # Track second call
        tracker.add_call(result2)

        # Log timing comparison
        processing_ratio = processing_time1 / processing_time2 if processing_time2 > 0 else 1.0
        logger.info(
            f"Time comparison - First call: {processing_time1:.3f}s, Second call: {processing_time2:.3f}s",
            emoji_key="processing",
        )
        logger.info(f"Speed ratio: {processing_ratio:.1f}x", emoji_key="info")

        console.print(
            "[yellow]Note: This example doesn't use actual caching since we're bypassing CompletionClient.[/yellow]"
        )

        # Display results
        display_completion_result(
            console=console,
            result=result1,  # Pass the original result object
            title="Federated Learning Explanation",
        )

    except Exception as e:
        logger.error(
            f"Error with cached completion demo: {str(e)}", emoji_key="error", exc_info=True
        )
        raise


async def run_multi_provider(gateway, tracker: CostTracker):
    """Run completion with multiple providers."""
    logger.info("Starting multi-provider example", emoji_key="start")
    console.print(Rule("[bold blue]Multi-Provider Completion[/bold blue]"))

    # Prompt to complete
    prompt = "List 3 benefits of quantum computing."

    providers_to_try = [Provider.OPENAI.value, Provider.ANTHROPIC.value, Provider.GEMINI.value]

    result_obj = None

    try:
        # Try providers in sequence
        logger.info("Trying multiple providers in sequence...", emoji_key="processing")

        for provider_name in providers_to_try:
            try:
                logger.info(f"Trying provider: {provider_name}", emoji_key="processing")

                # Get provider from gateway
                provider = gateway.providers.get(provider_name)
                if not provider:
                    logger.warning(
                        f"Provider {provider_name} not available or initialized, skipping",
                        emoji_key="warning",
                    )
                    continue

                # Generate completion
                result_obj = await provider.generate_completion(
                    prompt=prompt, temperature=0.7, max_tokens=200
                )

                # Track cost
                tracker.add_call(result_obj)

                logger.success(f"Successfully used provider: {provider_name}", emoji_key="success")
                break  # Exit loop on success

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {str(e)}", emoji_key="warning")
                # Continue to next provider

        if result_obj:
            # Display results
            display_completion_result(
                console=console,
                result=result_obj,  # Pass result_obj directly
                title=f"Response from {result_obj.provider}",  # Use result_obj.provider
            )
        else:
            logger.error("All providers failed. No results available.", emoji_key="error")

    except Exception as e:
        logger.error(
            f"Error with multi-provider completion: {str(e)}", emoji_key="error", exc_info=True
        )
        raise


async def run_json_mode_test(gateway, tracker: CostTracker):
    """Test the json_mode feature across multiple providers."""
    logger.info("Starting JSON mode test example", emoji_key="start")
    console.print(Rule("[bold blue]JSON Mode Test[/bold blue]"))

    # Create one prompt for regular completion and one for chat completion
    prompt = "Create a JSON array containing 3 countries with their name, capital, and population."

    # Create chat messages for testing with messages format
    chat_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides information in JSON format.",
        },
        {
            "role": "user",
            "content": "Create a JSON array containing 3 countries with their name, capital, and population.",
        },
    ]

    providers_to_try = [
        Provider.OPENAI.value,
        Provider.ANTHROPIC.value,
        Provider.GEMINI.value,
        Provider.OLLAMA.value,  # Test local Ollama models too
        Provider.DEEPSEEK.value,
    ]

    # Track statistics
    successes_completion = 0
    successes_chat = 0
    failures_completion = 0
    failures_chat = 0
    json_valid_completion = 0
    json_valid_chat = 0

    try:
        for provider_name in providers_to_try:
            try:
                logger.info(
                    f"Testing JSON mode with provider: {provider_name}", emoji_key="processing"
                )

                # Get provider from gateway
                provider = gateway.providers.get(provider_name)
                if not provider:
                    logger.warning(
                        f"Provider {provider_name} not available or initialized, skipping",
                        emoji_key="warning",
                    )
                    continue

                # --- TEST 1: REGULAR COMPLETION WITH JSON_MODE ---
                console.print(
                    f"\n[bold yellow]Testing regular completion with json_mode for {provider_name}:[/bold yellow]"
                )

                # Generate completion with json_mode=True
                result_completion = await provider.generate_completion(
                    prompt=prompt, temperature=0.7, max_tokens=300, json_mode=True
                )

                # Track cost
                tracker.add_call(result_completion)

                # Check if output is valid JSON
                is_valid_json_completion = False
                try:
                    # Try to parse the JSON to validate it
                    parsed_json = json.loads(result_completion.text)  # noqa: F841
                    is_valid_json_completion = True
                    json_valid_completion += 1
                except json.JSONDecodeError:
                    # Try custom extraction for Anthropic-like responses
                    if provider_name == Provider.ANTHROPIC.value:
                        try:
                            # This simple extraction handles the most common case where Anthropic
                            # wraps JSON in code blocks
                            import re

                            code_block_match = re.search(
                                r"```(?:json)?\s*([\s\S]*?)```", result_completion.text
                            )
                            if code_block_match:
                                code_content = code_block_match.group(1).strip()
                                parsed_json = json.loads(code_content)  # noqa: F841
                                is_valid_json_completion = True
                                json_valid_completion += 1
                        except (json.JSONDecodeError, TypeError, AttributeError):
                            is_valid_json_completion = False
                    else:
                        is_valid_json_completion = False

                # Display results for completion
                panel_title = f"[green]Regular Completion JSON Response from {provider_name}"
                if is_valid_json_completion:
                    panel_title += " ✓[/green]"
                    successes_completion += 1
                else:
                    panel_title += " ✗[/green]"
                    failures_completion += 1

                if result_completion.metadata.get("error"):
                    panel_title = f"[red]Error with {provider_name} (completion)[/red]"

                # Create a panel for the JSON response
                panel = Panel(
                    result_completion.text[:800]
                    + ("..." if len(result_completion.text) > 800 else ""),
                    title=panel_title,
                    border_style="cyan" if is_valid_json_completion else "red",
                )
                console.print(panel)

                # --- TEST 2: CHAT COMPLETION WITH JSON_MODE ---
                console.print(
                    f"\n[bold magenta]Testing chat completion with json_mode for {provider_name}:[/bold magenta]"
                )

                # Generate chat completion with json_mode=True
                result_chat = await provider.generate_completion(
                    messages=chat_messages, temperature=0.7, max_tokens=300, json_mode=True
                )

                # Track cost
                tracker.add_call(result_chat)

                # Check if output is valid JSON
                is_valid_json_chat = False
                try:
                    # Try to parse the JSON to validate it
                    parsed_json = json.loads(result_chat.text)  # noqa: F841
                    is_valid_json_chat = True
                    json_valid_chat += 1
                except json.JSONDecodeError:
                    # Try custom extraction for Anthropic-like responses
                    if provider_name == Provider.ANTHROPIC.value:
                        try:
                            import re

                            code_block_match = re.search(
                                r"```(?:json)?\s*([\s\S]*?)```", result_chat.text
                            )
                            if code_block_match:
                                code_content = code_block_match.group(1).strip()
                                parsed_json = json.loads(code_content)  # noqa: F841
                                is_valid_json_chat = True
                                json_valid_chat += 1
                        except (json.JSONDecodeError, TypeError, AttributeError):
                            is_valid_json_chat = False
                    else:
                        is_valid_json_chat = False

                # Display results for chat completion
                panel_title = f"[blue]Chat Completion JSON Response from {provider_name}"
                if is_valid_json_chat:
                    panel_title += " ✓[/blue]"
                    successes_chat += 1
                else:
                    panel_title += " ✗[/blue]"
                    failures_chat += 1

                if result_chat.metadata.get("error"):
                    panel_title = f"[red]Error with {provider_name} (chat)[/red]"

                # Create a panel for the JSON response
                panel = Panel(
                    result_chat.text[:800] + ("..." if len(result_chat.text) > 800 else ""),
                    title=panel_title,
                    border_style="green" if is_valid_json_chat else "red",
                )
                console.print(panel)

                # Add a small gap between providers
                console.print()

            except Exception as e:
                logger.error(
                    f"Provider {provider_name} failed with JSON mode: {str(e)}", emoji_key="error"
                )
                failures_completion += 1
                failures_chat += 1

        # Print summary
        summary = Table(title="JSON Mode Test Summary", show_header=True)
        summary.add_column("Test Type", style="cyan")
        summary.add_column("Providers Tested", style="white")
        summary.add_column("Successful", style="green")
        summary.add_column("Failed", style="red")
        summary.add_column("Valid JSON", style="blue")

        summary.add_row(
            "Regular Completion",
            str(len(providers_to_try)),
            str(successes_completion),
            str(failures_completion),
            str(json_valid_completion),
        )

        summary.add_row(
            "Chat Completion",
            str(len(providers_to_try)),
            str(successes_chat),
            str(failures_chat),
            str(json_valid_chat),
        )

        console.print(summary)

    except Exception as e:
        logger.error(f"Error in JSON mode test: {str(e)}", emoji_key="error", exc_info=True)
        raise


async def run_json_mode_streaming_test(gateway, tracker: CostTracker):
    """Test streaming with json_mode feature across multiple providers."""
    logger.info("Starting JSON mode streaming test", emoji_key="start")
    console.print(Rule("[bold blue]JSON Mode Streaming Test[/bold blue]"))

    # Prompt that naturally calls for a structured JSON response
    prompt = "Generate a JSON object with 5 recommended books, including title, author, and year published."

    # Chat messages for the streaming test
    chat_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that returns accurate information in JSON format.",
        },
        {
            "role": "user",
            "content": "Generate a JSON object with 5 recommended books, including title, author, and year published.",
        },
    ]

    # Use the same providers as in the regular JSON mode test
    providers_to_try = [
        Provider.OPENAI.value,
        Provider.ANTHROPIC.value,
        Provider.GEMINI.value,
        Provider.OLLAMA.value,
        Provider.DEEPSEEK.value,
    ]

    # Track statistics
    prompt_streaming_successes = 0
    chat_streaming_successes = 0
    prompt_json_valid = 0
    chat_json_valid = 0

    # Results comparison table
    comparison = Table(title="JSON Streaming Comparison By Provider", show_header=True)
    comparison.add_column("Provider", style="cyan")
    comparison.add_column("Method", style="blue")
    comparison.add_column("Valid JSON", style="green")
    comparison.add_column("Chunks", style="white")
    comparison.add_column("Time (s)", style="yellow")

    for provider_name in providers_to_try:
        console.print(f"\n[bold]Testing JSON mode streaming with provider: {provider_name}[/bold]")

        try:
            # Get provider from gateway
            provider = gateway.providers.get(provider_name)
            if not provider:
                logger.warning(
                    f"Provider {provider_name} not available or initialized, skipping",
                    emoji_key="warning",
                )
                continue

            # --- PART 1: TEST STREAMING WITH PROMPT ---
            console.print(
                f"[bold yellow]Testing prompt-based JSON streaming for {provider_name}:[/bold yellow]"
            )
            logger.info(
                f"Generating streaming JSON response with {provider_name} using prompt...",
                emoji_key="processing",
            )

            # Use Panel for streaming output presentation
            output_panel = Panel(
                "",
                title=f"{provider_name}: JSON Books (Prompt Streaming)",
                border_style="cyan",
                expand=False,
            )

            # Start timer
            start_time = time.time()

            full_text_prompt = ""
            token_count_prompt = 0

            # Use Live display for the streaming output panel
            with Live(output_panel, console=console, refresh_per_second=4):
                try:
                    # Get stream from the provider directly
                    stream = provider.generate_completion_stream(
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=500,
                        json_mode=True,  # Enable JSON mode for streaming
                    )

                    async for chunk, _metadata in stream:
                        full_text_prompt += chunk
                        token_count_prompt += 1
                        # Update the panel content
                        output_panel.renderable = full_text_prompt

                except Exception as e:
                    logger.error(
                        f"Error in prompt streaming for {provider_name}: {str(e)}",
                        emoji_key="error",
                    )
                    full_text_prompt = f"Error: {str(e)}"
                    output_panel.renderable = full_text_prompt

            # Calculate processing time
            processing_time_prompt = time.time() - start_time

            # Check if the final output is valid JSON
            is_valid_json_prompt = False
            try:
                if full_text_prompt and not full_text_prompt.startswith("Error:"):
                    parsed_json = json.loads(full_text_prompt)  # noqa: F841
                    is_valid_json_prompt = True
                    prompt_json_valid += 1
                    prompt_streaming_successes += 1
                    logger.success(
                        f"{provider_name} prompt JSON stream is valid!", emoji_key="success"
                    )
            except json.JSONDecodeError:
                # Try custom extraction for Anthropic-like responses
                if provider_name == Provider.ANTHROPIC.value:
                    try:
                        import re

                        code_block_match = re.search(
                            r"```(?:json)?\s*([\s\S]*?)```", full_text_prompt
                        )
                        if code_block_match:
                            code_content = code_block_match.group(1).strip()
                            parsed_json = json.loads(code_content)  # noqa: F841
                            is_valid_json_prompt = True
                            prompt_json_valid += 1
                            prompt_streaming_successes += 1
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        is_valid_json_prompt = False

            # Add to comparison table
            comparison.add_row(
                provider_name,
                "Prompt-based",
                "✓ Yes" if is_valid_json_prompt else "✗ No",
                str(token_count_prompt),
                f"{processing_time_prompt:.3f}",
            )

            # Track cost if stream was successful
            if full_text_prompt and not full_text_prompt.startswith("Error:"):
                est_input_tokens_prompt = len(prompt) // 4
                est_output_tokens_prompt = len(full_text_prompt) // 4
                est_result_prompt = ModelResponse(
                    text=full_text_prompt,
                    model=f"{provider_name}/default",
                    provider=provider_name,
                    input_tokens=est_input_tokens_prompt,
                    output_tokens=est_output_tokens_prompt,
                    total_tokens=est_input_tokens_prompt + est_output_tokens_prompt,
                    processing_time=processing_time_prompt,
                )
                tracker.add_call(est_result_prompt)

            # Show truncated output
            prompt_panel = Panel(
                full_text_prompt[:500] + ("..." if len(full_text_prompt) > 500 else ""),
                title=f"[cyan]{provider_name}[/cyan] Prompt JSON: [{'green' if is_valid_json_prompt else 'red'}]{'Valid' if is_valid_json_prompt else 'Invalid'}[/]",
                border_style="green" if is_valid_json_prompt else "red",
            )
            console.print(prompt_panel)

            # --- PART 2: TEST STREAMING WITH CHAT MESSAGES ---
            console.print(
                f"[bold magenta]Testing chat-based JSON streaming for {provider_name}:[/bold magenta]"
            )
            logger.info(
                f"Generating streaming JSON response with {provider_name} using chat messages...",
                emoji_key="processing",
            )

            # Use Panel for streaming output presentation
            chat_output_panel = Panel(
                "",
                title=f"{provider_name}: JSON Books (Chat Streaming)",
                border_style="blue",
                expand=False,
            )

            # Start timer
            start_time_chat = time.time()

            full_text_chat = ""
            token_count_chat = 0

            # Use Live display for the streaming output panel
            with Live(chat_output_panel, console=console, refresh_per_second=4):
                try:
                    # Get stream from the provider directly
                    chat_stream = provider.generate_completion_stream(
                        messages=chat_messages,  # Use messages instead of prompt
                        temperature=0.7,
                        max_tokens=500,
                        json_mode=True,  # Enable JSON mode for streaming
                    )

                    async for chunk, _metadata in chat_stream:
                        full_text_chat += chunk
                        token_count_chat += 1
                        # Update the panel content
                        chat_output_panel.renderable = full_text_chat
                except Exception as e:
                    logger.error(
                        f"Error in chat streaming for {provider_name}: {str(e)}", emoji_key="error"
                    )
                    full_text_chat = f"Error: {str(e)}"
                    chat_output_panel.renderable = full_text_chat

            # Calculate processing time
            processing_time_chat = time.time() - start_time_chat

            # Check if the final output is valid JSON
            is_valid_json_chat = False
            try:
                if full_text_chat and not full_text_chat.startswith("Error:"):
                    parsed_json_chat = json.loads(full_text_chat)  # noqa: F841
                    is_valid_json_chat = True
                    chat_json_valid += 1
                    chat_streaming_successes += 1
                    logger.success(
                        f"{provider_name} chat JSON stream is valid!", emoji_key="success"
                    )
            except json.JSONDecodeError:
                # Try custom extraction for Anthropic-like responses
                if provider_name == Provider.ANTHROPIC.value:
                    try:
                        import re

                        code_block_match = re.search(
                            r"```(?:json)?\s*([\s\S]*?)```", full_text_chat
                        )
                        if code_block_match:
                            code_content = code_block_match.group(1).strip()
                            parsed_json_chat = json.loads(code_content)  # noqa: F841
                            is_valid_json_chat = True
                            chat_json_valid += 1
                            chat_streaming_successes += 1
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        is_valid_json_chat = False

            # Add to comparison table
            comparison.add_row(
                provider_name,
                "Chat-based",
                "✓ Yes" if is_valid_json_chat else "✗ No",
                str(token_count_chat),
                f"{processing_time_chat:.3f}",
            )

            # Track cost if stream was successful
            if full_text_chat and not full_text_chat.startswith("Error:"):
                est_input_tokens_chat = sum(len(m["content"]) for m in chat_messages) // 4
                est_output_tokens_chat = len(full_text_chat) // 4
                est_result_chat = ModelResponse(
                    text=full_text_chat,
                    model=f"{provider_name}/default",
                    provider=provider_name,
                    input_tokens=est_input_tokens_chat,
                    output_tokens=est_output_tokens_chat,
                    total_tokens=est_input_tokens_chat + est_output_tokens_chat,
                    processing_time=processing_time_chat,
                )
                tracker.add_call(est_result_chat)

            # Show truncated output
            chat_panel = Panel(
                full_text_chat[:500] + ("..." if len(full_text_chat) > 500 else ""),
                title=f"[cyan]{provider_name}[/cyan] Chat JSON: [{'green' if is_valid_json_chat else 'red'}]{'Valid' if is_valid_json_chat else 'Invalid'}[/]",
                border_style="green" if is_valid_json_chat else "red",
            )
            console.print(chat_panel)

        except Exception as e:
            logger.error(
                f"Provider {provider_name} failed completely in JSON streaming test: {str(e)}",
                emoji_key="error",
            )

    # Print comparison table
    console.print(comparison)

    # Print summary
    summary = Table(title="JSON Streaming Test Summary", show_header=True)
    summary.add_column("Method", style="cyan")
    summary.add_column("Providers", style="white")
    summary.add_column("Successful", style="green")
    summary.add_column("Valid JSON", style="blue")

    summary.add_row(
        "Prompt-based",
        str(len(providers_to_try)),
        str(prompt_streaming_successes),
        str(prompt_json_valid),
    )

    summary.add_row(
        "Chat-based",
        str(len(providers_to_try)),
        str(chat_streaming_successes),
        str(chat_json_valid),
    )

    console.print(summary)


async def main():
    """Run completion examples."""
    # Parse command-line arguments
    args = parse_args()

    tracker = CostTracker()  # Instantiate tracker
    try:
        # Create a gateway instance for all examples to share
        gateway = Gateway("basic-completion-demo", register_tools=False)

        # Initialize providers
        logger.info("Initializing providers...", emoji_key="provider")
        await gateway._initialize_providers()

        if not args.json_only:
            # Run basic completion
            await run_basic_completion(gateway, tracker)

            console.print()  # Add space

            # Run chat completion
            await run_chat_completion(gateway, tracker)

            console.print()  # Add space

            # Run streaming completion
            await run_streaming_completion(gateway)

            console.print()  # Add space

            # Run cached completion
            await run_cached_completion(gateway, tracker)

            console.print()  # Add space

            # Run multi-provider completion
            await run_multi_provider(gateway, tracker)

            console.print()  # Add space

        # Run JSON mode test across providers
        await run_json_mode_test(gateway, tracker)

        console.print()  # Add space

        # Run JSON mode streaming test
        await run_json_mode_streaming_test(gateway, tracker)

        # Display cost summary at the end
        tracker.display_summary(console)

    except Exception as e:
        # Use logger for critical errors
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    # Run the examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
