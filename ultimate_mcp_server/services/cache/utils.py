"""Cache utility functions for Ultimate MCP Server.

This module provides utility functions for working with the cache service
that were previously defined in example scripts but are now part of the library.
"""

import hashlib

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import get_provider
from ultimate_mcp_server.services.cache import get_cache_service
from ultimate_mcp_server.utils import get_logger

# Initialize logger
logger = get_logger("ultimate_mcp_server.services.cache.utils")


async def run_completion_with_cache(
    prompt: str,
    provider_name: str = Provider.OPENAI.value,
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = None,
    use_cache: bool = True,
    ttl: int = 3600,  # Default 1 hour cache TTL
    api_key: str = None,
):
    """Run a completion with automatic caching.

    This utility function handles provider initialization, cache key generation,
    cache lookups, and caching results automatically.

    Args:
        prompt: Text prompt for completion
        provider_name: Provider to use (default: OpenAI)
        model: Model name (optional, uses provider default if not specified)
        temperature: Temperature for generation (default: 0.1)
        max_tokens: Maximum tokens to generate (optional)
        use_cache: Whether to use cache (default: True)
        ttl: Cache TTL in seconds (default: 3600/1 hour)
        api_key: Provider API key (optional, falls back to internal provider system)

    Returns:
        Completion result with additional processing_time attribute
    """
    try:
        # Let the provider system handle API keys if none provided
        provider = await get_provider(provider_name, api_key=api_key)
        await provider.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize provider {provider_name}: {e}", emoji_key="error")
        raise  # Re-raise exception to stop execution if provider fails

    cache_service = get_cache_service()

    # Create a more robust cache key using all relevant parameters
    model_id = model or provider.get_default_model()  # Ensure we have a model id

    # Create consistent hash of parameters that affect the result
    params_str = f"{prompt}:{temperature}:{max_tokens if max_tokens else 'default'}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()

    cache_key = f"completion:{provider_name}:{model_id}:{params_hash}"

    if use_cache and cache_service.enabled:
        cached_result = await cache_service.get(cache_key)
        if cached_result is not None:
            logger.success("Cache hit! Using cached result", emoji_key="cache")
            # Set processing time for cache retrieval (negligible)
            cached_result.processing_time = 0.001
            return cached_result

    # Generate completion if not cached or cache disabled
    if use_cache:
        logger.info("Cache miss. Generating new completion...", emoji_key="processing")
    else:
        logger.info(
            "Cache disabled by request. Generating new completion...", emoji_key="processing"
        )

    # Use the determined model_id and pass through other parameters
    result = await provider.generate_completion(
        prompt=prompt, model=model_id, temperature=temperature, max_tokens=max_tokens
    )

    # Save to cache if enabled
    if use_cache and cache_service.enabled:
        await cache_service.set(key=cache_key, value=result, ttl=ttl)
        logger.info(f"Result saved to cache (key: ...{cache_key[-10:]})", emoji_key="cache")

    return result
