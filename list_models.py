#!/usr/bin/env python3
import asyncio

from ultimate_mcp_server.core.server import Gateway


async def list_models():
    """
    List all available models from each configured LLM provider.

    This function initializes the MCP Gateway with all providers and queries
    each provider for its available models. It then prints a formatted list
    of all models grouped by provider, making it easy to see which models
    are accessible with the current configuration.

    The function performs the following steps:
    1. Initialize the Gateway, which loads configuration for all providers
    2. Initialize each provider, which may include API key validation
    3. Query each provider for its available models
    4. Print the provider name followed by a list of its available models

    Each model is displayed with the provider name as a prefix (e.g., "openai:gpt-4o")
    for clear identification. Models that don't have an 'id' field will use their
    'name' field instead, and those without either will be labeled as 'unknown'.

    This function is useful for:
    - Verifying that API keys are working correctly
    - Checking which models are available for use
    - Debugging provider configuration issues
    - Getting the correct model identifiers for use in applications

    Returns:
        None - Results are printed to the console

    Notes:
        - Requires valid API keys for each provider to be configured
        - Some providers may have rate limits on model listing operations
        - This function will fail if any provider's initialization fails
    """
    gateway = Gateway()
    await gateway._initialize_providers()
    print("Initialized providers")

    # Get the models from each provider
    for provider_name, provider in gateway.providers.items():
        print(f"\nProvider: {provider_name}")
        models = await provider.list_models()
        for model in models:
            print(f"  - {provider_name}:{model.get('id', model.get('name', 'unknown'))}")


if __name__ == "__main__":
    asyncio.run(list_models())
