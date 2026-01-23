"""Tests for the cache service."""

import asyncio
from pathlib import Path

import pytest

from ultimate_mcp_server.services.cache import (
    CacheService,
    with_cache,
)
from ultimate_mcp_server.services.cache.strategies import (
    ExactMatchStrategy,
    SemanticMatchStrategy,
    TaskBasedStrategy,
)
from ultimate_mcp_server.utils import get_logger

logger = get_logger("test.cache")


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture
def cache_service(temp_cache_dir: Path) -> CacheService:
    """Get a cache service instance with a temporary directory."""
    return CacheService(
        enabled=True,
        ttl=60,  # Short TTL for testing
        max_entries=10,
        enable_persistence=True,
        cache_dir=str(temp_cache_dir),
        enable_fuzzy_matching=True,
    )


class TestCacheService:
    """Tests for the cache service."""

    async def test_init(self, cache_service: CacheService):
        """Test cache service initialization."""
        logger.info("Testing cache service initialization", emoji_key="test")

        assert cache_service.enabled
        assert cache_service.ttl == 60
        assert cache_service.max_entries == 10
        assert cache_service.enable_persistence
        assert cache_service.enable_fuzzy_matching

    async def test_get_set(self, cache_service: CacheService):
        """Test basic get and set operations."""
        logger.info("Testing cache get/set operations", emoji_key="test")

        # Set a value
        key = "test-key"
        value = {"text": "Test value", "metadata": {"test": True}}
        await cache_service.set(key, value)

        # Get the value back
        result = await cache_service.get(key)
        assert result == value

        # Check cache stats
        assert cache_service.metrics.hits == 1
        assert cache_service.metrics.misses == 0
        assert cache_service.metrics.stores == 1

    async def test_cache_miss(self, cache_service: CacheService):
        """Test cache miss."""
        logger.info("Testing cache miss", emoji_key="test")

        # Get a non-existent key
        result = await cache_service.get("non-existent-key")
        assert result is None

        # Check cache stats
        assert cache_service.metrics.hits == 0
        assert cache_service.metrics.misses == 1

    async def test_cache_expiry(self, cache_service: CacheService):
        """Test cache entry expiry."""
        logger.info("Testing cache expiry", emoji_key="test")

        # Set a value with short TTL
        key = "expiring-key"
        value = {"text": "Expiring value"}
        await cache_service.set(key, value, ttl=1)  # 1 second TTL

        # Get immediately (should hit)
        result = await cache_service.get(key)
        assert result == value

        # Wait for expiry
        await asyncio.sleep(1.5)

        # Get again (should miss)
        result = await cache_service.get(key)
        assert result is None

        # Check stats
        assert cache_service.metrics.hits == 1
        assert cache_service.metrics.misses == 1

    async def test_cache_eviction(self, cache_service: CacheService):
        """Test cache eviction when max size is reached."""
        logger.info("Testing cache eviction", emoji_key="test")

        # Set max_entries + 1 values
        for i in range(cache_service.max_entries + 5):
            key = f"key-{i}"
            value = {"text": f"Value {i}"}
            await cache_service.set(key, value)

        # Check size - should be at most max_entries
        assert len(cache_service.cache) <= cache_service.max_entries

        # Check stats
        assert cache_service.metrics.evictions > 0

    async def test_fuzzy_matching(self, cache_service: CacheService):
        """Test fuzzy matching of cache keys."""
        logger.info("Testing fuzzy matching", emoji_key="test")

        # Set a value with a prompt that would generate a fuzzy key
        request_params = {
            "prompt": "What is the capital of France?",
            "model": "test-model",
            "temperature": 0.7,
        }

        key = cache_service.generate_cache_key(request_params)
        fuzzy_key = cache_service.generate_fuzzy_key(request_params)

        value = {"text": "The capital of France is Paris."}
        await cache_service.set(key, value, fuzzy_key=fuzzy_key, request_params=request_params)

        # Create a similar request that should match via fuzzy lookup
        similar_request = {
            "prompt": "What is the capital of France? Tell me about it.",
            "model": "different-model",
            "temperature": 0.5,
        }

        similar_key = cache_service.generate_cache_key(similar_request)
        similar_fuzzy = cache_service.generate_fuzzy_key(similar_request)  # noqa: F841

        # Should still find the original value
        result = await cache_service.get(similar_key, fuzzy=True)
        assert result == value

    async def test_cache_decorator(self):
        """Test the cache decorator."""
        logger.info("Testing cache decorator", emoji_key="test")

        call_count = 0

        @with_cache(ttl=60)
        async def test_function(arg1, arg2=None):
            nonlocal call_count
            call_count += 1
            return {"result": arg1 + str(arg2)}

        # First call should execute the function
        result1 = await test_function("test", arg2="123")
        assert result1 == {"result": "test123"}
        assert call_count == 1

        # Second call with same args should use cache
        result2 = await test_function("test", arg2="123")
        assert result2 == {"result": "test123"}
        assert call_count == 1  # Still 1

        # Call with different args should execute function again
        result3 = await test_function("test", arg2="456")
        assert result3 == {"result": "test456"}
        assert call_count == 2


class TestCacheStrategies:
    """Tests for cache strategies."""

    def test_exact_match_strategy(self):
        """Test exact match strategy."""
        logger.info("Testing exact match strategy", emoji_key="test")

        strategy = ExactMatchStrategy()

        # Generate key for a request
        request = {"prompt": "Test prompt", "model": "test-model", "temperature": 0.7}

        key = strategy.generate_key(request)
        assert key.startswith("exact:")

        # Should cache most requests
        assert strategy.should_cache(request, {"text": "Test response"})

        # Shouldn't cache streaming requests
        streaming_request = request.copy()
        streaming_request["stream"] = True
        assert not strategy.should_cache(streaming_request, {"text": "Test response"})

    def test_semantic_match_strategy(self):
        """Test semantic match strategy."""
        logger.info("Testing semantic match strategy", emoji_key="test")

        strategy = SemanticMatchStrategy()

        # Generate key for a request
        request = {
            "prompt": "What is the capital of France?",
            "model": "test-model",
            "temperature": 0.7,
        }

        key = strategy.generate_key(request)
        assert key.startswith("exact:")  # Primary key is still exact

        semantic_key = strategy.generate_semantic_key(request)
        assert semantic_key.startswith("semantic:")

        # Should generate similar semantic keys for similar prompts
        similar_request = {
            "prompt": "Tell me the capital city of France?",
            "model": "test-model",
            "temperature": 0.7,
        }

        similar_semantic_key = strategy.generate_semantic_key(similar_request)
        assert similar_semantic_key.startswith("semantic:")

        # The two semantic keys should share many common words
        # This is a bit harder to test deterministically, so we'll skip detailed assertions

    def test_task_based_strategy(self):
        """Test task-based strategy."""
        logger.info("Testing task-based strategy", emoji_key="test")

        strategy = TaskBasedStrategy()

        # Test different task types
        summarization_request = {
            "prompt": "Summarize this document: Lorem ipsum...",
            "model": "test-model",
            "task_type": "summarization",
        }

        extraction_request = {
            "prompt": "Extract entities from this text: John Smith...",
            "model": "test-model",
            "task_type": "extraction",
        }

        # Generate keys
        summary_key = strategy.generate_key(summarization_request)
        extraction_key = strategy.generate_key(extraction_request)

        # Keys should include task type
        assert "summarization" in summary_key
        assert "extraction" in extraction_key

        # Task-specific TTL
        summary_ttl = strategy.get_ttl(summarization_request, None)
        extraction_ttl = strategy.get_ttl(extraction_request, None)

        # Summarization should have longer TTL than extraction (typically)
        assert summary_ttl is not None
        assert extraction_ttl is not None
        assert summary_ttl > extraction_ttl
