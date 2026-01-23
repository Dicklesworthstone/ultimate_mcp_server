#!/usr/bin/env python
"""Script to check API key configurations for Ultimate MCP Server using rich formatting."""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.utils import get_logger

# Initialize rich console
console = Console()

logger = get_logger("api_key_checker")

# Map provider names to the corresponding environment variable names
# Used for informational display only
PROVIDER_ENV_VAR_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


async def check_api_keys():
    """
    Check API key configurations and display a comprehensive report.

    This async function:
    1. Loads the current configuration settings from all sources (environment variables,
       .env file, configuration files)
    2. Initializes a minimal Gateway instance to access provider configurations
    3. Checks if API keys are properly configured for all supported providers
    4. Displays formatted results using rich tables and panels, including:
       - Provider-by-provider API key status
       - Configuration loading priority information
       - How to set API keys properly
       - Example .env file content

    The function checks keys for all providers defined in the Provider enum,
    including OpenAI, Anthropic, DeepSeek, Gemini, and OpenRouter.

    Returns:
        int: Exit code (0 for success)
    """
    # Force load config to ensure we get the latest resolved settings
    cfg = get_config()

    # Create Gateway with minimal initialization (no tools) - kept for potential future checks
    gateway = Gateway(name="api-key-checker", register_tools=False)  # noqa: F841

    console.print(
        Panel(
            "Checking API Key Configuration based on loaded settings",
            title="[bold cyan]Ultimate MCP Server API Key Check[/bold cyan]",
            expand=False,
            border_style="blue",
        )
    )

    # Create table for results
    table = Table(title="Provider API Key Status", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="dim", width=12)
    table.add_column("API Key Status", style="cyan")
    table.add_column("Relevant Env Var", style="yellow")
    table.add_column("Status", style="bold")

    # Check each provider based on the loaded configuration
    for provider_name in [p.value for p in Provider]:
        # Get provider config from the loaded GatewayConfig object
        provider_config = getattr(cfg.providers, provider_name, None)

        # Check if key exists in the loaded config
        # This key would have been resolved from .env, env vars, or config file by get_config()
        config_key = provider_config.api_key if provider_config else None

        # Format key for display (if present)
        key_display = Text("Not set in config", style="dim yellow")
        status_text = Text("NOT CONFIGURED", style="red")
        status_icon = "❌"

        if config_key:
            if len(config_key) > 8:
                key_display = Text(f"{config_key[:4]}...{config_key[-4:]}", style="green")
            else:
                key_display = Text("[INVALID KEY FORMAT]", style="bold red")
            status_text = Text("CONFIGURED", style="green")
            status_icon = "✅"

        # Get the corresponding environment variable name for informational purposes
        env_var_name = PROVIDER_ENV_VAR_MAP.get(provider_name, "N/A")

        # Add row to table
        table.add_row(
            provider_name.capitalize(),
            key_display,
            env_var_name,
            f"[{status_text.style}]{status_icon} {status_text}[/]",
        )

    # Print the table
    console.print(table)

    # Configuration Loading Info Panel
    config_info = Text.assemble(
        ("1. ", "bold blue"),
        ("Environment Variables", "cyan"),
        (" (e.g., ", "dim"),
        ("GATEWAY_PROVIDERS__OPENAI__API_KEY=...", "yellow"),
        (")\n", "dim"),
        ("2. ", "bold blue"),
        ("Values in a ", "cyan"),
        (".env", "yellow"),
        (" file in the project root\n", "cyan"),
        ("3. ", "bold blue"),
        ("Values in a config file", "cyan"),
        (" (e.g., ", "dim"),
        ("gateway_config.yaml", "yellow"),
        (")\n", "dim"),
        ("4. ", "bold blue"),
        ("Default values defined in the configuration models", "cyan"),
    )
    console.print(
        Panel(config_info, title="[bold]Configuration Loading Priority[/]", border_style="blue")
    )

    # How to Set Keys Panel
    set_keys_info = Text.assemble(
        ("Ensure API keys are available via one of the methods above,\n", "white"),
        ("preferably using ", "white"),
        ("environment variables", "cyan"),
        (" or a ", "white"),
        (".env", "yellow"),
        (" file.", "white"),
    )
    console.print(Panel(set_keys_info, title="[bold]How to Set API Keys[/]", border_style="green"))

    # Example .env Panel
    env_example_lines = []
    for env_var in PROVIDER_ENV_VAR_MAP.values():
        env_example_lines.append(
            Text.assemble(
                (env_var, "yellow"),
                "=",
                ("your_", "dim"),
                (env_var.lower(), "dim cyan"),
                ("_here", "dim"),
            )
        )
    env_example_content = Text("\n").join(env_example_lines)
    console.print(
        Panel(
            env_example_content,
            title="[bold dim]Example .env file content[/]",
            border_style="yellow",
        )
    )

    console.print(
        "[bold green]Run your example scripts or the main server after setting the API keys.[/bold green]"
    )
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(check_api_keys())
    sys.exit(exit_code)
