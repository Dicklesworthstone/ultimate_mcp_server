#!/usr/bin/env python
"""Prompt templates and repository demonstration for Ultimate MCP Server."""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.services.prompts import PromptTemplate, get_prompt_repository
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker, display_text_content_result

# --- Add Rich Imports ---
from ultimate_mcp_server.utils.logging.console import console

# ----------------------

# Initialize logger
logger = get_logger("example.prompt_templates")


async def demonstrate_prompt_templates():
    """Demonstrate prompt template creation and rendering."""
    # Use Rich Rule for title
    console.print(Rule("[bold blue]Prompt Template Demonstration[/bold blue]"))
    logger.info("Starting prompt template demonstration", emoji_key="start")

    # Simple prompt template
    template_text = """
You are an expert in {{field}}. 
Please explain {{concept}} in simple terms that a {{audience}} could understand.
"""

    # Create a prompt template
    template = PromptTemplate(
        template=template_text,
        template_id="simple_explanation",
        description="A template for generating simple explanations of concepts",
    )

    logger.info(f"Created prompt template: {template.template_id}", emoji_key="template")

    # Render the template with variables
    variables = {
        "field": "artificial intelligence",
        "concept": "neural networks",
        "audience": "high school student",
    }

    rendered_prompt = template.render(variables)

    logger.info(
        "Template rendered successfully", emoji_key="success", variables=list(variables.keys())
    )

    # Display rendered template using Rich
    console.print(Rule("[cyan]Simple Template Rendering[/cyan]"))
    console.print(
        Panel(
            Syntax(template.template, "jinja2", theme="default", line_numbers=False),
            title="[bold]Template Source[/bold]",
            border_style="dim blue",
            expand=False,
        )
    )
    vars_table = Table(title="[bold]Variables[/bold]", box=box.MINIMAL, show_header=False)
    vars_table.add_column("Key", style="magenta")
    vars_table.add_column("Value", style="white")
    for key, value in variables.items():
        vars_table.add_row(escape(key), escape(value))
    console.print(vars_table)
    console.print(
        Panel(
            escape(rendered_prompt.strip()),
            title="[bold green]Rendered Prompt[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    console.print()

    # Create a more complex template with conditional blocks
    complex_template_text = """
{% if system_message %}
{{system_message}}
{% else %}
You are a helpful assistant that provides accurate information.
{% endif %}

{% if context %}
Here is some context to help you answer:
{{context}}
{% endif %}

USER: {{query}}

Please respond with:
{% for item in response_items %}
- {{item}}
{% endfor %}
"""

    complex_template_obj = PromptTemplate(
        template=complex_template_text,  # Use the text variable
        template_id="complex_assistant",
        description="A complex assistant template with conditionals and loops",
        required_vars=["system_message", "query", "response_items", "context"],
    )

    # Complex variables
    complex_variables = {
        "system_message": "You are an expert in climate science who explains concepts clearly and objectively.",
        "query": "What are the main causes of climate change?",
        "context": """
Recent data shows that global temperatures have risen by about 1.1°C since pre-industrial times.
The IPCC Sixth Assessment Report (2021) states that human activities are unequivocally the main driver
of climate change, primarily through greenhouse gas emissions. CO2 levels have increased by 48% since 
the industrial revolution, reaching levels not seen in at least 800,000 years.
""",
        "response_items": [
            "A summary of the main causes based on scientific consensus",
            "The role of greenhouse gases (CO2, methane, etc.) in climate change",
            "Human activities that contribute most significantly to emissions",
            "Natural vs anthropogenic factors and their relative impact",
            "Regional variations in climate change impacts",
        ],
    }

    complex_rendered = complex_template_obj.render(complex_variables)

    logger.info(
        "Complex template rendered successfully",
        emoji_key="success",
        template_id=complex_template_obj.template_id,
    )

    # Display complex template rendering using Rich
    console.print(Rule("[cyan]Complex Template Rendering[/cyan]"))
    console.print(
        Panel(
            Syntax(complex_template_obj.template, "jinja2", theme="default", line_numbers=False),
            title="[bold]Template Source[/bold]",
            border_style="dim blue",
            expand=False,
        )
    )
    complex_vars_table = Table(title="[bold]Variables[/bold]", box=box.MINIMAL, show_header=False)
    complex_vars_table.add_column("Key", style="magenta")
    complex_vars_table.add_column("Value", style="white")
    for key, value in complex_variables.items():
        # Truncate long context for display
        display_value = escape(str(value))
        if key == "context" and len(display_value) > 150:
            display_value = display_value[:150] + "..."
        elif isinstance(value, list):
            display_value = escape(
                str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            )  # Truncate lists too
        complex_vars_table.add_row(escape(key), display_value)
    console.print(complex_vars_table)
    console.print(
        Panel(
            escape(complex_rendered.strip()),
            title="[bold green]Rendered Prompt[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    console.print()

    # Demonstrate rendering with missing variables (handled by Jinja's default behavior or errors)
    console.print(Rule("[cyan]Template with Missing Variables[/cyan]"))
    missing_variables = {
        "query": "How can individuals reduce their carbon footprint?",
        "response_items": [
            "Daily lifestyle changes with significant impact",
            "Transportation choices and alternatives",
            "Home energy consumption reduction strategies",
        ],
        # system_message and context are intentionally missing
    }

    try:
        missing_rendered = complex_template_obj.render(missing_variables)
        logger.info(
            "Template rendered with missing optional variables (using defaults)",
            emoji_key="info",
            missing=["system_message", "context"],
        )
        console.print(
            Panel(
                escape(missing_rendered.strip()),
                title="[bold yellow]Rendered with Defaults[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )
    except Exception as e:  # Catch Jinja exceptions or others
        logger.warning(f"Could not render with missing variables: {str(e)}", emoji_key="warning")
        console.print(
            Panel(
                f"[red]Error rendering template:[/red]\n{escape(str(e))}",
                title="[bold red]Rendering Error[/bold red]",
                border_style="red",
                expand=False,
            )
        )
    console.print()

    return template, complex_template_obj


async def demonstrate_prompt_repository():
    """Demonstrate saving and retrieving templates from repository."""
    # Use Rich Rule
    console.print(Rule("[bold blue]Prompt Repository Demonstration[/bold blue]"))
    logger.info("Starting prompt repository demonstration", emoji_key="start")

    # Get repository
    repo = get_prompt_repository()

    # Check repository path
    logger.info(f"Prompt repository path: {repo.base_dir}", emoji_key="info")

    # List existing prompts (if any)
    prompts = await repo.list_prompts()
    if prompts:
        logger.info(
            f"Found {len(prompts)} existing prompts: {', '.join(prompts)}", emoji_key="info"
        )
    else:
        logger.info("No existing prompts found in repository", emoji_key="info")

    # Create a new prompt template for saving
    translation_template = """
Translate the following {{source_language}} text into {{target_language}}:

TEXT: {{text}}

The translation should be:
- Accurate and faithful to the original
- Natural in the target language
- Preserve the tone and style of the original

TRANSLATION:
"""

    template = PromptTemplate(
        template=translation_template,
        template_id="translation_prompt",
        description="A template for translation tasks",
        metadata={
            "author": "Ultimate MCP Server",
            "version": "1.0",
            "supported_languages": ["English", "Spanish", "French", "German", "Japanese"],
        },
    )

    # Save to repository
    template_dict = template.to_dict()

    logger.info(
        f"Saving template '{template.template_id}' to repository",
        emoji_key="save",
        metadata=template.metadata,
    )

    save_result = await repo.save_prompt(template.template_id, template_dict)

    if save_result:
        logger.success(f"Template '{template.template_id}' saved successfully", emoji_key="success")
    else:
        logger.error(f"Failed to save template '{template.template_id}'", emoji_key="error")
        return

    # Retrieve the saved template
    logger.info(
        f"Retrieving template '{template.template_id}' from repository", emoji_key="loading"
    )

    retrieved_dict = await repo.get_prompt(template.template_id)

    if retrieved_dict:
        # Convert back to PromptTemplate object
        retrieved_template = PromptTemplate.from_dict(retrieved_dict)

        logger.success(
            f"Retrieved template '{retrieved_template.template_id}' successfully",
            emoji_key="success",
            metadata=retrieved_template.metadata,
        )

        # Display retrieved template details using Rich
        retrieved_table = Table(
            title=f"[bold]Retrieved Template: {escape(retrieved_template.template_id)}[/bold]",
            box=box.ROUNDED,
            show_header=False,
        )
        retrieved_table.add_column("Attribute", style="cyan")
        retrieved_table.add_column("Value", style="white")
        retrieved_table.add_row("Description", escape(retrieved_template.description))
        retrieved_table.add_row("Metadata", escape(str(retrieved_template.metadata)))
        console.print(retrieved_table)
        console.print(
            Panel(
                Syntax(retrieved_template.template, "jinja2", theme="default", line_numbers=False),
                title="[bold]Template Source[/bold]",
                border_style="dim blue",
                expand=False,
            )
        )
        console.print()

    else:
        logger.error(f"Failed to retrieve template '{template.template_id}'", emoji_key="error")

    # List prompts again to confirm addition
    updated_prompts = await repo.list_prompts()
    logger.info(
        f"Repository now contains {len(updated_prompts)} prompts: {', '.join(updated_prompts)}",
        emoji_key="info",
    )

    # Comment out the deletion to keep the template for the LLM demo
    # Uncommenting the below would delete the template
    """
    delete_result = await repo.delete_prompt(template.template_id)
    if delete_result:
        logger.info(
            f"Deleted template '{template.template_id}' from repository",
            emoji_key="cleaning"
        )
    """

    return retrieved_template


async def demonstrate_llm_with_templates(tracker: CostTracker):
    """Demonstrate using a template from the repository with an LLM."""
    # Use Rich Rule
    console.print(Rule("[bold blue]LLM with Template Demonstration[/bold blue]"))
    logger.info("Starting LLM with template demonstration", emoji_key="start")

    # Retrieve the translation template saved earlier
    repo = get_prompt_repository()
    template_id = "translation_prompt"
    template_dict = await repo.get_prompt(template_id)

    if not template_dict:
        console.print(f"Prompt '{template_id}' not found")
        logger.error(f"Template '{template_id}' not found. Skipping LLM demo.", emoji_key="error")
        return

    template = PromptTemplate.from_dict(template_dict)
    logger.info(f"Retrieved template '{template_id}' for LLM use", emoji_key="template")

    # Variables for translation
    translation_vars = {
        "source_language": "English",
        "target_language": "French",
        "text": "The quick brown fox jumps over the lazy dog.",
    }

    # Render the prompt
    try:
        rendered_prompt = template.render(translation_vars)
        logger.info("Translation prompt rendered", emoji_key="success")

        # Display the rendered prompt for clarity
        console.print(
            Panel(
                escape(rendered_prompt.strip()),
                title="[bold]Rendered Translation Prompt[/bold]",
                border_style="blue",
                expand=False,
            )
        )

    except Exception as e:
        logger.error(
            f"Error rendering translation prompt: {str(e)}", emoji_key="error", exc_info=True
        )
        return

    # Initialize gateway with providers
    gateway = Gateway("prompt-templates-demo", register_tools=False)
    logger.info("Initializing providers...", emoji_key="provider")
    await gateway._initialize_providers()

    # Providers to try in order of preference
    providers_to_try = [
        Provider.OPENAI.value,
        Provider.ANTHROPIC.value,
        Provider.GEMINI.value,
        Provider.DEEPSEEK.value,
    ]

    # Find an available provider
    provider = None
    provider_name = None

    for p_name in providers_to_try:
        if p_name in gateway.providers:
            provider = gateway.providers[p_name]
            provider_name = p_name
            logger.info(f"Using provider {p_name}", emoji_key="provider")
            break

    try:
        model = provider.get_default_model()
        logger.info(f"Using provider {provider_name} with model {model}", emoji_key="provider")

        # Generate completion using the rendered prompt
        logger.info("Generating translation...", emoji_key="processing")
        start_time = time.time()
        result = await provider.generate_completion(
            prompt=rendered_prompt, model=model, temperature=0.5, max_tokens=150
        )
        processing_time = time.time() - start_time

        logger.success("Translation generated successfully!", emoji_key="success")

        # Use display.py function for better visualization
        display_text_content_result(
            f"Translation Result ({escape(provider_name)}/{escape(model)})",
            result,
            console_instance=console,
        )

        # Track cost
        tracker.add_call(result)

        # Display additional stats with standard rich components
        stats_table = Table(title="Translation Stats", show_header=False, box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Provider", provider_name)
        stats_table.add_row("Model", model)
        stats_table.add_row("Input Tokens", str(result.input_tokens))
        stats_table.add_row("Output Tokens", str(result.output_tokens))
        stats_table.add_row("Cost", f"${result.cost:.6f}")
        stats_table.add_row("Processing Time", f"{processing_time:.3f}s")
        console.print(stats_table)

    except Exception as e:
        logger.error(f"Error during LLM completion: {str(e)}", emoji_key="error", exc_info=True)
        # Fall back to mock response
        console.print(
            Panel(
                "[yellow]Failed to generate real translation. Here's a mock response:[/yellow]\n"
                + "Le renard brun rapide saute par-dessus le chien paresseux.",
                title="[bold yellow]Mock Translation (After Error)[/bold yellow]",
                border_style="yellow",
            )
        )

    # Display cost summary at the end of this demo section
    tracker.display_summary(console)


async def main():
    """Run all demonstrations."""
    try:
        # Demonstrate template creation and rendering
        template1, template2 = await demonstrate_prompt_templates()
        console.print()  # Add space

        # Demonstrate repository usage
        retrieved_template = await demonstrate_prompt_repository()  # noqa: F841
        console.print()

        # Demonstrate using a template with LLM - no longer check for retrieved_template
        # as it should always be available since we commented out the deletion
        tracker = CostTracker()  # Instantiate tracker here
        await demonstrate_llm_with_templates(tracker)

    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1

    # Clean up after demo is complete - optionally delete the template
    try:
        # After demo is complete, we can clean up by deleting the template
        repo = get_prompt_repository()
        await repo.delete_prompt("translation_prompt")
        logger.info("Deleted demonstration template", emoji_key="cleaning")
    except Exception as e:
        logger.warning(f"Cleanup error: {str(e)}", emoji_key="warning")

    logger.success("Prompt Template Demo Finished Successfully!", emoji_key="complete")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
