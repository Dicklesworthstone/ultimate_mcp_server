#!/usr/bin/env python
"""Grok integration demonstration using Ultimate MCP Server."""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Project imports
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger
logger = get_logger("example.grok_integration")

# Create a separate console for detailed debugging output
debug_console = Console(stderr=True, highlight=False)


async def compare_grok_models(tracker: CostTracker):
    """Compare different Grok models."""
    console.print(Rule("[bold cyan]⚡ Grok Model Comparison [/bold cyan]", style="bold blue"))
    logger.info("Starting Grok models comparison", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway(
        "grok-demo", register_tools=False, provider_exclusions=[Provider.OPENROUTER.value]
    )

    # Initialize providers
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()

    provider_name = Provider.GROK
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

        # Display available models in a tree structure with consistent padding
        model_tree = Tree("[bold cyan]Available Grok Models[/bold cyan]")
        for model in model_names:
            # Only display grok-3 models
            if not model.startswith("grok-3"):
                continue

            if "fast" in model:
                model_tree.add(
                    f"[bold yellow]{model}[/bold yellow] [dim](optimized for speed)[/dim]"
                )
            elif "mini" in model:
                model_tree.add(
                    f"[bold green]{model}[/bold green] [dim](optimized for reasoning)[/dim]"
                )
            else:
                model_tree.add(f"[bold magenta]{model}[/bold magenta] [dim](general purpose)[/dim]")

        # Add padding around the tree
        console.print(Panel(model_tree, border_style="dim cyan", padding=(1, 2)))

        # Select specific models to compare
        grok_models = ["grok-3-latest", "grok-3-mini-latest"]

        # Filter based on available models
        models_to_compare = [m for m in grok_models if m in model_names]
        if not models_to_compare:
            # Only use grok-3 models
            models_to_compare = [m for m in model_names if m.startswith("grok-3")][:2]

        if not models_to_compare:
            logger.warning("No grok-3 models available for comparison.", emoji_key="warning")
            return

        # Consistent panel styling
        console.print(
            Panel(
                f"Comparing models: [yellow]{escape(', '.join(models_to_compare))}[/yellow]",
                title="[bold]Comparison Setup[/bold]",
                border_style="blue",  # Use blue for info
                padding=(1, 2),
            )
        )

        prompt = """
        Explain the concept of quantum entanglement in a way that a high school student would understand.
        Keep your response brief and accessible.
        """

        # Consistent panel styling for prompt
        console.print(
            Panel(
                escape(prompt.strip()),
                title="[bold]Test Prompt[/bold]",
                border_style="yellow",  # Yellow for prompts
                expand=False,
                padding=(1, 2),
            )
        )

        results_data = []

        # Create progress display with TaskProgressColumn
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("[green]{task.completed} of {task.total}"),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        ) as progress:
            task_id = progress.add_task("[cyan]Testing models...", total=len(models_to_compare))

            for model_name in models_to_compare:
                progress.update(
                    task_id, description=f"[cyan]Testing model: [bold]{model_name}[/bold]"
                )

                try:
                    logger.info(f"Testing model: {model_name}", emoji_key="model")
                    start_time = time.time()
                    result = await provider.generate_completion(
                        prompt=prompt, model=model_name, temperature=0.3, max_tokens=300
                    )
                    processing_time = time.time() - start_time

                    # Track the cost
                    tracker.add_call(result)

                    # Log detailed timing info to debug console
                    debug_console.print(f"[dim]Model {model_name} processing details:[/dim]")
                    debug_console.print(
                        f"[dim]Time: {processing_time:.2f}s | Tokens: {result.total_tokens}[/dim]"
                    )

                    # Check if model is a mini model with reasoning output
                    reasoning_content = None
                    reasoning_tokens = None
                    if "mini" in model_name and result.metadata:
                        reasoning_content = result.metadata.get("reasoning_content")
                        reasoning_tokens = result.metadata.get("reasoning_tokens")

                    results_data.append(
                        {
                            "model": model_name,
                            "text": result.text,
                            "tokens": {
                                "input": result.input_tokens,
                                "output": result.output_tokens,
                                "total": result.total_tokens,
                            },
                            "reasoning_content": reasoning_content,
                            "reasoning_tokens": reasoning_tokens,
                            "cost": result.cost,
                            "time": processing_time,
                        }
                    )

                    logger.success(
                        f"Completion for {model_name} successful",
                        emoji_key="success",
                    )

                except Exception as e:
                    logger.error(
                        f"Error testing model {model_name}: {str(e)}",
                        emoji_key="error",
                        exc_info=True,
                    )
                    debug_console.print_exception()
                    results_data.append({"model": model_name, "error": str(e)})

                progress.advance(task_id)

        # Display comparison results using Rich
        if results_data:
            # Bolder rule style
            console.print(
                Rule("[bold green]⚡ Comparison Results [/bold green]", style="bold green")
            )

            # Store panels for potential column layout
            comparison_panels = []

            for result_item in results_data:
                model = result_item["model"]

                if "error" in result_item:
                    # Handle error case with consistent styling
                    error_panel = Panel(
                        f"[red]{escape(result_item['error'])}[/red]",
                        title=f"[bold red]{escape(model)} - ERROR[/bold red]",
                        border_style="red",  # Red for errors
                        expand=False,
                        padding=(1, 2),
                    )
                    comparison_panels.append(error_panel)
                    continue

                time_s = result_item["time"]
                tokens = result_item.get("tokens", {})
                input_tokens = tokens.get("input", 0)
                output_tokens = tokens.get("output", 0)
                total_tokens = tokens.get("total", 0)

                tokens_per_second = total_tokens / time_s if time_s > 0 else 0
                cost = result_item.get("cost", 0.0)
                text = result_item.get("text", "[red]Error generating response[/red]").strip()

                # Determine border color based on model type (consistent scheme)
                border_style = "magenta"  # Magenta for general models
                if "mini" in model:
                    border_style = "green"  # Green for reasoning
                elif "fast" in model:
                    border_style = "yellow"  # Yellow for speed

                # Create the panel for this model's output
                model_panel = Panel(
                    escape(text),
                    title=f"[bold {border_style}]{escape(model)}[/bold {border_style}]",  # Use border color in title
                    subtitle="[dim]Response Text[/dim]",
                    border_style=border_style,
                    expand=True,
                    # height=len(text.splitlines()) + 4, # Adjust height dynamically based on padding
                    padding=(1, 2),  # Standard padding
                )

                # Create beautiful stats table with a slightly different box
                stats_table = Table(box=box.MINIMAL, show_header=False, expand=True, padding=0)
                stats_table.add_column("Metric", style="dim cyan", width=15)
                stats_table.add_column("Value", style="white")
                stats_table.add_row("Input Tokens", f"[yellow]{input_tokens}[/yellow]")
                stats_table.add_row("Output Tokens", f"[green]{output_tokens}[/green]")
                stats_table.add_row("Total Tokens", f"[bold cyan]{total_tokens}[/bold cyan]")
                stats_table.add_row("Time", f"[yellow]{time_s:.2f}s[/yellow]")
                stats_table.add_row("Speed", f"[blue]{tokens_per_second:.1f} tok/s[/blue]")
                stats_table.add_row("Cost", f"[green]${cost:.6f}[/green]")

                # Combine as a single compact panel
                combined_panel = Panel(
                    Group(model_panel, Align.center(stats_table)),
                    border_style=border_style,
                    padding=(1, 1),  # Reduced padding for combined view
                    title=f"[bold]Response from {escape(model)}[/bold]",
                )

                # If there's reasoning content, show it directly
                reasoning_content = result_item.get("reasoning_content")
                reasoning_tokens = result_item.get("reasoning_tokens")

                if reasoning_content:
                    reasoning_panel = Panel(
                        escape(reasoning_content),
                        title="[bold cyan]Reasoning Process[/bold cyan]",
                        subtitle=f"[dim]Reasoning Tokens: {reasoning_tokens}[/dim]",
                        border_style="cyan",  # Cyan for reasoning/tools
                        expand=True,
                        # height=len(reasoning_content.splitlines()) + 4, # Adjust height
                        padding=(1, 2),  # Standard padding
                    )
                    # Group main result and reasoning
                    comparison_panels.append(Group(combined_panel, reasoning_panel))
                else:
                    comparison_panels.append(combined_panel)

            # Use Columns layout if exactly two results (and no errors resulted in fewer panels)
            if len(comparison_panels) == 2 and len(comparison_panels) == len(results_data):
                console.print(Columns(comparison_panels, equal=True, expand=True))
            else:
                # Otherwise, print panels sequentially
                for panel in comparison_panels:
                    console.print(panel)

    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}", emoji_key="error", exc_info=True)


async def demonstrate_reasoning(tracker: CostTracker):
    """Demonstrate Grok-mini reasoning capabilities."""
    console.print(
        Rule("[bold cyan]⚡ Grok Reasoning Demonstration [/bold cyan]", style="bold blue")
    )
    logger.info("Demonstrating Grok-mini reasoning capabilities", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway(
        "grok-demo", register_tools=False, provider_exclusions=[Provider.OPENROUTER.value]
    )

    # Initialize providers
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()

    provider_name = Provider.GROK
    try:
        # Get the provider from the gateway
        provider = gateway.providers.get(provider_name)
        if not provider:
            logger.error(
                f"Provider {provider_name} not available or initialized", emoji_key="error"
            )
            return

        # Use a Grok mini model (ensure it's available)
        model = "grok-3-mini-latest"
        available_models = await provider.list_models()
        model_names = [m["id"] for m in available_models]

        if model not in model_names:
            # Find any mini model
            for m in model_names:
                if "mini" in m:
                    model = m
                    break
            else:
                logger.warning(
                    "No mini model available for reasoning demo. Using default model.",
                    emoji_key="warning",
                )
                model = provider.get_default_model()

        logger.info(f"Using model: {model}", emoji_key="model")

        # Problem requiring reasoning
        problem = """
        A cylindrical water tank has a radius of 3 meters and a height of 4 meters.
        If water flows in at a rate of 2 cubic meters per minute, how long will it take to fill the tank?
        Show your work step by step.
        """

        # Consistent panel styling for prompt
        console.print(
            Panel(
                escape(problem.strip()),
                title="[bold yellow]Math Problem[/bold yellow]",
                border_style="yellow",  # Yellow for prompts
                expand=False,
                padding=(1, 2),  # Standard padding
            )
        )

        with Progress(
            TextColumn("[bold blue]Status:"),
            BarColumn(complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        ) as progress:
            task = progress.add_task("[cyan]Thinking...", total=1)

            logger.info("Generating solution with reasoning", emoji_key="processing")

            result = await provider.generate_completion(
                prompt=problem,
                model=model,
                temperature=0.3,
                reasoning_effort="high",  # Use high reasoning effort
                max_tokens=1000,
            )

            # Track the cost
            tracker.add_call(result)

            progress.update(task, description="Complete!", completed=1)

        logger.success("Reasoning solution completed", emoji_key="success")

        # Extract reasoning content
        reasoning_content = None
        reasoning_tokens = None
        if result.metadata:
            reasoning_content = result.metadata.get("reasoning_content")
            reasoning_tokens = result.metadata.get("reasoning_tokens")

        # Create a more compact layout for reasoning demo
        if reasoning_content:
            reasoning_panel = Panel(
                escape(reasoning_content),
                title="[bold cyan]Thinking Process[/bold cyan]",
                subtitle=f"[dim]Reasoning Tokens: {reasoning_tokens}[/dim]",
                border_style="cyan",  # Cyan for reasoning/tools
                expand=True,
                # height=len(reasoning_content.splitlines()) + 4, # Adjust height
                padding=(1, 2),  # Standard padding
            )
        else:
            reasoning_panel = Panel(
                "[italic]No explicit reasoning process available[/italic]",
                title="[bold cyan]Thinking Process[/bold cyan]",
                border_style="cyan",  # Cyan for reasoning/tools
                expand=True,
                padding=(1, 2),  # Standard padding
            )

        # Format the answer
        answer_panel = Panel(
            escape(result.text.strip()),
            title="[bold green]Final Solution[/bold green]",
            subtitle=f"[dim]Tokens: {result.input_tokens} in, {result.output_tokens} out | Cost: ${result.cost:.6f} | Time: {result.processing_time:.2f}s[/dim]",
            border_style="green",  # Green for success/final result
            expand=True,
            # height=len(result.text.strip().splitlines()) + 4, # Adjust height
            padding=(1, 2),  # Standard padding
        )

        # Use Group for better vertical spacing control than grid
        console.print(Group(reasoning_panel, answer_panel))

    except Exception as e:
        logger.error(
            f"Error in reasoning demonstration: {str(e)}", emoji_key="error", exc_info=True
        )


async def demonstrate_function_calling(tracker: CostTracker):
    """Demonstrate Grok function calling capabilities."""
    console.print(
        Rule("[bold cyan]⚡ Grok Function Calling Demonstration [/bold cyan]", style="bold blue")
    )
    logger.info("Demonstrating Grok function calling capabilities", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway(
        "grok-demo", register_tools=False, provider_exclusions=[Provider.OPENROUTER.value]
    )

    # Initialize providers
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()

    provider_name = Provider.GROK
    try:
        # Get the provider from the gateway
        provider = gateway.providers.get(provider_name)
        if not provider:
            logger.error(
                f"Provider {provider_name} not available or initialized", emoji_key="error"
            )
            return

        # Use default Grok model
        model = provider.get_default_model()
        logger.info(f"Using model: {model}", emoji_key="model")

        # Define tools for the model to use
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The unit of temperature to use",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_flight_info",
                    "description": "Get flight information between two cities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "departure_city": {
                                "type": "string",
                                "description": "The departure city",
                            },
                            "arrival_city": {"type": "string", "description": "The arrival city"},
                            "date": {
                                "type": "string",
                                "description": "The date of travel in YYYY-MM-DD format",
                            },
                        },
                        "required": ["departure_city", "arrival_city"],
                    },
                },
            },
        ]

        # Display tools in a Panel for consistency
        tools_table = Table(
            title="[bold cyan]Available Tools[/bold cyan]",
            box=box.MINIMAL,
            show_header=True,
            header_style="bold magenta",
        )
        tools_table.add_column("Tool Name", style="cyan", no_wrap=True)
        tools_table.add_column("Description", style="white")
        tools_table.add_column("Parameters", style="green")

        for tool in tools:
            function = tool["function"]
            name = function["name"]
            description = function["description"]
            params = ", ".join([p for p in function["parameters"]["properties"]])
            tools_table.add_row(name, description, params)

        console.print(Panel(tools_table, border_style="cyan", padding=(1, 2)))  # Cyan for tools

        # User query
        user_query = "I'm planning a trip from New York to Los Angeles next week. What's the weather like in LA, and can you help me find flight information?"

        # Consistent panel styling for prompt
        console.print(
            Panel(
                escape(user_query),
                title="[bold yellow]User Query[/bold yellow]",
                border_style="yellow",  # Yellow for prompts
                expand=False,
                padding=(1, 2),  # Standard padding
            )
        )

        with Progress(
            TextColumn("[bold blue]Status:"),
            BarColumn(complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        ) as progress:
            task = progress.add_task("[cyan]Processing...", total=1)

            logger.info("Generating completion with function calling", emoji_key="processing")

            result = await provider.generate_completion(
                prompt=user_query, model=model, temperature=0.7, tools=tools, tool_choice="auto"
            )

            # Track the cost
            tracker.add_call(result)

            progress.update(task, description="Complete!", completed=1)

        logger.success("Function calling completed", emoji_key="success")

        # Check if there are tool calls in the response
        tool_calls = None
        if (
            hasattr(result.raw_response.choices[0].message, "tool_calls")
            and result.raw_response.choices[0].message.tool_calls
        ):
            tool_calls = result.raw_response.choices[0].message.tool_calls

        if tool_calls:
            # Format the model response
            response_text = (
                escape(result.text.strip())
                if result.text
                else "[italic dim]No direct text response, only tool calls.[/italic dim]"
            )
            response_info = f"[dim]Input Tokens: {result.input_tokens} | Output Tokens: {result.output_tokens} | Cost: ${result.cost:.6f}[/dim]"

            model_response_panel = Panel(
                response_text,
                title="[bold green]Model Response[/bold green]",
                subtitle=response_info,
                padding=(1, 2),  # Standard padding
                border_style="green",  # Green for success/results
            )

            # Prepare panels for tool calls
            tool_panels = []
            for tool_call in tool_calls:
                # Parse JSON arguments
                try:
                    args = json.loads(tool_call.function.arguments)
                    args_formatted = f"[json]{escape(json.dumps(args, indent=2))}[/json]"
                except Exception:
                    args_formatted = escape(tool_call.function.arguments)

                # Create compact tool call display content
                call_content_lines = [
                    f"[bold cyan]Function:[/bold cyan] [magenta]{tool_call.function.name}[/magenta]",
                    f"[bold cyan]Arguments:[/bold cyan]\n{args_formatted}",
                ]

                # Add mock function result if available
                result_data = None
                if tool_call.function.name == "get_weather":
                    location = args.get("location", "Unknown")
                    unit = args.get("unit", "fahrenheit")
                    temp = 75 if unit == "fahrenheit" else 24
                    result_data = {
                        "location": location,
                        "temperature": temp,
                        "unit": unit,
                        "condition": "Sunny",
                        "humidity": 65,
                    }
                elif tool_call.function.name == "get_flight_info":
                    departure = args.get("departure_city", "Unknown")
                    arrival = args.get("arrival_city", "Unknown")
                    date = args.get("date", "2025-04-20")  # noqa: F841
                    result_data = {
                        "flights": [
                            {
                                "airline": "Delta",
                                "flight": "DL1234",
                                "departure": f"{departure} 08:30 AM",
                                "arrival": f"{arrival} 11:45 AM",
                                "price": "$349.99",
                            },
                            {
                                "airline": "United",
                                "flight": "UA567",
                                "departure": f"{departure} 10:15 AM",
                                "arrival": f"{arrival} 1:30 PM",
                                "price": "$289.99",
                            },
                        ]
                    }

                if result_data:
                    result_formatted = f"[json]{escape(json.dumps(result_data, indent=2))}[/json]"
                    call_content_lines.append(
                        f"\n[bold blue]Mock Result:[/bold blue]\n{result_formatted}"
                    )

                # Join content lines for the panel
                call_content = "\n".join(call_content_lines)

                tool_panel = Panel(
                    call_content,
                    title=f"[bold magenta]Tool Call: {tool_call.function.name}[/bold magenta]",
                    subtitle=f"[dim]ID: {tool_call.id}[/dim]",
                    border_style="magenta",  # Magenta for specific tool calls
                    padding=(1, 2),  # Standard padding
                )
                tool_panels.append(tool_panel)

            # Use Columns for horizontal layout if multiple tool calls
            if len(tool_panels) > 1:
                tool_call_display = Columns(tool_panels, equal=True, expand=True)
            elif tool_panels:
                tool_call_display = tool_panels[0]
            else:  # Should not happen if tool_calls is true, but handle defensively
                tool_call_display = Text("No tool calls processed.", style="dim")

            # Create combined panel with response and tool calls
            combined_panel = Panel(
                Group(model_response_panel, tool_call_display),
                title="[bold green]Function Calling Results[/bold green]",
                border_style="green",  # Green for overall success
                padding=(1, 1),  # Slightly reduced outer padding
            )

            console.print(combined_panel)
        else:
            # No tool calls, just display the response with consistent styling
            console.print(
                Panel(
                    escape(result.text.strip()),
                    title="[bold green]Model Response (No Tool Calls)[/bold green]",
                    subtitle=f"[dim]Tokens: {result.input_tokens} in, {result.output_tokens} out | Cost: ${result.cost:.6f}[/dim]",
                    border_style="green",  # Green for success/result
                    padding=(1, 2),  # Standard padding
                )
            )

        console.print()  # Keep spacing

    except Exception as e:
        logger.error(
            f"Error in function calling demonstration: {str(e)}", emoji_key="error", exc_info=True
        )


async def streaming_example(tracker: CostTracker):
    """Demonstrate Grok streaming capabilities."""
    console.print(
        Rule("[bold cyan]⚡ Grok Streaming Demonstration [/bold cyan]", style="bold blue")
    )
    logger.info("Demonstrating Grok streaming capabilities", emoji_key="start")

    # Create Gateway instance - this handles provider initialization
    gateway = Gateway(
        "grok-demo", register_tools=False, provider_exclusions=[Provider.OPENROUTER.value]
    )

    # Initialize providers
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()

    provider_name = Provider.GROK
    try:
        # Get the provider from the gateway
        provider = gateway.providers.get(provider_name)
        if not provider:
            logger.error(
                f"Provider {provider_name} not available or initialized", emoji_key="error"
            )
            return

        # Use default Grok model
        model = provider.get_default_model()
        logger.info(f"Using model: {model}", emoji_key="model")

        # Create prompt for streaming
        prompt = "Write a short story about an AI that discovers emotions for the first time."

        # Consistent panel styling for prompt
        console.print(
            Panel(
                escape(prompt),
                title="[bold yellow]Streaming Prompt[/bold yellow]",
                border_style="yellow",  # Yellow for prompts
                expand=False,
                padding=(1, 2),  # Standard padding
            )
        )

        # Create streaming panel with consistent styling
        stream_panel = Panel(
            "",
            title=f"[bold green]Streaming Output from {model}[/bold green]",
            subtitle="[dim]Live output...[/dim]",
            border_style="green",  # Green for results
            expand=True,
            height=15,  # Slightly increased height for better visibility
            padding=(1, 2),  # Standard padding
        )

        # Setup for streaming
        logger.info("Starting stream", emoji_key="processing")
        stream = provider.generate_completion_stream(
            prompt=prompt, model=model, temperature=0.7, max_tokens=500
        )

        full_text = ""
        chunk_count = 0
        start_time = time.time()

        # Display streaming content with Rich Live display
        with Live(
            stream_panel, console=console, refresh_per_second=10, vertical_overflow="visible"
        ) as live:
            async for content, _metadata in stream:
                chunk_count += 1
                full_text += content

                # Update the live display
                # Ensure renderable is Text for better control if needed, though escape works
                stream_panel.renderable = Text(escape(full_text))
                stream_panel.subtitle = f"[dim]Received {chunk_count} chunks...[/dim]"
                live.update(stream_panel)  # No need to pass stream_panel again

        # Final update to show completion
        stream_panel.subtitle = f"[bold green]Stream Complete ({chunk_count} chunks)[/bold green]"
        # Update the panel content one last time outside the live context
        console.print(stream_panel)

        # Calculate stats
        processing_time = time.time() - start_time
        # More accurate token estimation might involve encoding, but keep simple for demo
        estimated_tokens = len(full_text.split()) * 1.3  # Rough estimate
        tokens_per_second = estimated_tokens / processing_time if processing_time > 0 else 0

        # Display final stats in a Panel with a Table
        stats_table = Table(
            title="[bold blue]Streaming Stats[/bold blue]",
            box=box.MINIMAL,
            padding=(0, 1),
            show_header=False,
        )
        stats_table.add_column("Metric", style="dim cyan")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Total Time", f"[yellow]{processing_time:.2f}s[/yellow]")
        stats_table.add_row("Chunks Received", f"[green]{chunk_count}[/green]")
        stats_table.add_row("Est. Output Tokens", f"[cyan]~{int(estimated_tokens)}[/cyan]")
        stats_table.add_row("Est. Speed", f"[blue]{tokens_per_second:.1f} tok/s[/blue]")

        console.print(
            Panel(stats_table, border_style="blue", padding=(1, 2))
        )  # Blue for info/stats
        logger.success("Streaming completed", emoji_key="success")

    except Exception as e:
        logger.error(
            f"Error in streaming demonstration: {str(e)}", emoji_key="error", exc_info=True
        )


async def main():
    """Run Grok integration examples."""
    tracker = CostTracker()
    try:
        # Create title with padding
        title = Text("⚡ Grok Integration Showcase ⚡", style="bold white on blue")
        title.justify = "center"
        # Add padding to the main title panel
        console.print(Panel(title, box=box.DOUBLE_EDGE, padding=(1, 0)))  # Vertical padding

        debug_console.print("[dim]Starting Grok integration demo in debug mode[/dim]")

        # Run model comparison
        await compare_grok_models(tracker)

        console.print()  # Add space between sections

        # Run reasoning demonstration
        await demonstrate_reasoning(tracker)

        console.print()  # Add space between sections

        # Run function calling demonstration
        await demonstrate_function_calling(tracker)

        console.print()  # Add space between sections

        # Run streaming example
        await streaming_example(tracker)

        # Display final summary
        tracker.display_summary(console)

    except Exception as e:
        logger.critical(f"Example failed: {str(e)}", emoji_key="critical", exc_info=True)
        debug_console.print_exception(show_locals=True)
        return 1

    logger.success("Grok Integration Demo Finished Successfully!", emoji_key="complete")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
