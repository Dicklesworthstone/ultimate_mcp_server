"""Provider module for Ultimate MCP Server.

This module provides access to LLM providers and provider-specific functionality.
"""

from typing import Dict, Type

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.anthropic import AnthropicProvider
from ultimate_mcp_server.core.providers.base import BaseProvider
from ultimate_mcp_server.core.providers.deepseek import DeepSeekProvider
from ultimate_mcp_server.core.providers.gemini import GeminiProvider
from ultimate_mcp_server.core.providers.grok import GrokProvider
from ultimate_mcp_server.core.providers.ollama import OllamaProvider
from ultimate_mcp_server.core.providers.openai import OpenAIProvider
from ultimate_mcp_server.core.providers.openrouter import OpenRouterProvider

# Provider registry
PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
    Provider.OPENAI.value: OpenAIProvider,
    Provider.ANTHROPIC.value: AnthropicProvider,
    Provider.DEEPSEEK.value: DeepSeekProvider,
    Provider.GEMINI.value: GeminiProvider,
    Provider.OPENROUTER.value: OpenRouterProvider,
    Provider.GROK.value: GrokProvider,
    Provider.OLLAMA.value: OllamaProvider,
}

__all__ = ["PROVIDER_REGISTRY"]
