#!/usr/bin/env python
"""Test script for checking Ollama connectivity."""

import asyncio
import sys

import aiohttp


async def test_ollama():
    """Test connection to Ollama API."""
    print("Testing Ollama API connectivity...")

    urls_to_try = [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ]

    for base_url in urls_to_try:
        print(f"\nTrying URL: {base_url}")
        try:
            # Create a session with a short timeout
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    # Try to connect to the tags endpoint
                    url = f"{base_url}/api/tags"
                    print(f"Connecting to: {url}")

                    async with session.get(url) as response:
                        status = response.status
                        print(f"Status code: {status}")

                        if status == 200:
                            data = await response.json()
                            models = data.get("models", [])
                            print(f"Success! Found {len(models)} models.")
                            if models:
                                print("Model names:")
                                for model in models:
                                    print(f"  - {model.get('name')}")
                        else:
                            text = await response.text()
                            print(f"Error response: {text[:200]}")
                except aiohttp.ClientConnectionError as e:
                    print(f"Connection error: {type(e).__name__} - {str(e)}")
                except asyncio.TimeoutError:
                    print("Connection timed out after 5 seconds")
                except Exception as e:
                    print(f"Unexpected error: {type(e).__name__} - {str(e)}")
        except Exception as e:
            print(f"Session creation error: {type(e).__name__} - {str(e)}")

    # Also try through the library's provider interface
    try:
        print("\nTesting through Ultimate MCP Server classes...")
        # Import the OllamaProvider class
        from ultimate_mcp_server.core.providers.ollama import OllamaProvider

        # Create an instance
        provider = OllamaProvider()
        print(f"Provider created with URL: {provider.config.api_url}")

        # Initialize the provider
        initialized = await provider.initialize()
        print(f"Provider initialized: {initialized}")

        if initialized:
            # Try to list models
            models = await provider.list_models()
            print(f"Models found through provider: {len(models)}")
            if models:
                print("Model IDs:")
                for model in models:
                    print(f"  - {model['id']}")

        # Make sure to shut down properly
        await provider.shutdown()
    except Exception as e:
        print(f"Provider test error: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"aiohttp version: {aiohttp.__version__}")
    asyncio.run(test_ollama())
