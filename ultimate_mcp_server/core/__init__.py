"""Core functionality for Ultimate MCP Server."""

import asyncio
from typing import Optional

from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)

# Add a provider manager getter function
_gateway_instance = None


async def async_init_gateway():
    """
    Asynchronously initialize the global gateway instance.

    This function creates and initializes the Gateway singleton instance that manages
    provider connections and serves as the central access point for LLM capabilities.
    It ensures the gateway is properly initialized only once, maintaining a global
    instance that can be used across the application.

    The initialization process includes:
    1. Creating a Gateway instance if none exists
    2. Initializing all configured providers asynchronously
    3. Setting up the provider connections and validating configurations

    Returns:
        The initialized Gateway instance

    Note:
        This function is designed to be called from async code. For synchronous
        contexts, use get_gateway_instance() which handles event loop management.
    """
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = Gateway("provider-manager")
        await _gateway_instance._initialize_providers()
    return _gateway_instance


def get_provider_manager():
    """Get the provider manager from the Gateway instance.

    Returns:
        Provider manager with initialized providers
    """
    global _gateway_instance

    if _gateway_instance is None:
        try:
            # Try to run in current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task in the current event loop
                asyncio.create_task(async_init_gateway())
                logger.warning("Gateway instance requested before async init completed.")
                return {}
            else:
                # Run in a new event loop (blocks)
                logger.info("Synchronously initializing gateway for get_provider_manager.")
                _gateway_instance = Gateway("provider-manager")
                loop.run_until_complete(_gateway_instance._initialize_providers())
        except RuntimeError:
            # No event loop running, create one (blocks)
            logger.info("Synchronously initializing gateway for get_provider_manager (new loop).")
            _gateway_instance = Gateway("provider-manager")
            asyncio.run(_gateway_instance._initialize_providers())

    # Return the providers dictionary as a "manager"
    return _gateway_instance.providers if _gateway_instance else {}


def get_gateway_instance() -> Optional[Gateway]:
    """Synchronously get the initialized gateway instance.

    Returns:
        The Gateway instance or None if it hasn't been initialized yet.
    """
    global _gateway_instance
    if _gateway_instance is None:
        logger.warning("get_gateway_instance() called before instance was initialized.")
    return _gateway_instance


__all__ = ["Gateway", "get_provider_manager"]
