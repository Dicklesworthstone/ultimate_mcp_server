"""Caching service for Ultimate MCP Server."""

from ultimate_mcp_server.services.cache.cache_service import (
    CacheService,
    CacheStats,
    get_cache_service,
    with_cache,
)
from ultimate_mcp_server.services.cache.persistence import CachePersistence
from ultimate_mcp_server.services.cache.strategies import (
    CacheStrategy,
    ExactMatchStrategy,
    SemanticMatchStrategy,
    TaskBasedStrategy,
    get_strategy,
)
from ultimate_mcp_server.services.cache.utils import run_completion_with_cache

__all__ = [
    "CacheService",
    "CacheStats",
    "get_cache_service",
    "with_cache",
    "CachePersistence",
    "CacheStrategy",
    "ExactMatchStrategy",
    "SemanticMatchStrategy",
    "TaskBasedStrategy",
    "get_strategy",
    "run_completion_with_cache",
]
