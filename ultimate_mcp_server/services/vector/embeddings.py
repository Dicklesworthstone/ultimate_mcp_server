"""Embedding generation service for vector operations."""

import asyncio
import hashlib
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)

# Global dictionary to store embedding instances (optional)
embedding_instances = {}


class EmbeddingCache:
    """Cache for embeddings to avoid repeated API calls."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cache files
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".ultimate" / "embeddings"

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.cache = {}

        logger.info(
            f"Embeddings cache initialized (directory: {self.cache_dir})", emoji_key="cache"
        )

    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate a cache key for text and model.

        Args:
            text: Text to embed
            model: Embedding model name

        Returns:
            Cache key
        """
        # Create a hash based on text and model
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{model}_{text_hash}"

    def _get_cache_file_path(self, key: str) -> Path:
        """Get cache file path for a key.

        Args:
            key: Cache key

        Returns:
            Cache file path
        """
        return self.cache_dir / f"{key}.npy"

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get embedding from cache.

        Args:
            text: Text to embed
            model: Embedding model name

        Returns:
            Cached embedding or None if not found
        """
        key = self._get_cache_key(text, model)

        # Check in-memory cache first
        if key in self.cache:
            return self.cache[key]

        # Check disk cache
        cache_file = self._get_cache_file_path(key)
        if cache_file.exists():
            try:
                embedding = np.load(str(cache_file))
                # Add to in-memory cache
                self.cache[key] = embedding
                return embedding
            except Exception as e:
                logger.error(f"Failed to load embedding from cache: {str(e)}", emoji_key="error")

        return None

    def set(self, text: str, model: str, embedding: np.ndarray) -> None:
        """Set embedding in cache.

        Args:
            text: Text to embed
            model: Embedding model name
            embedding: Embedding vector
        """
        key = self._get_cache_key(text, model)

        # Add to in-memory cache
        self.cache[key] = embedding

        # Save to disk
        cache_file = self._get_cache_file_path(key)
        try:
            np.save(str(cache_file), embedding)
        except Exception as e:
            logger.error(f"Failed to save embedding to cache: {str(e)}", emoji_key="error")

    def clear(self) -> None:
        """Clear the embedding cache."""
        # Clear in-memory cache
        self.cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(
                    f"Failed to delete cache file {cache_file}: {str(e)}", emoji_key="error"
                )

        logger.info("Embeddings cache cleared", emoji_key="cache")


class EmbeddingService:
    """Generic service to create embeddings using different providers."""

    def __init__(
        self,
        provider_type: str = "openai",
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the embedding service.

        Args:
            provider_type: The type of embedding provider (e.g., 'openai').
            model_name: The specific embedding model to use.
            api_key: Optional API key. If not provided, attempts to load from config.
            **kwargs: Additional provider-specific arguments.
        """
        self.provider_type = provider_type.lower()
        self.model_name = model_name
        self.client = None
        self.api_key = api_key
        self.kwargs = kwargs

        try:
            config = get_config()
            if self.provider_type == "openai":
                provider_config = config.providers.openai
                # Use provided key first, then config key
                self.api_key = self.api_key or provider_config.api_key
                if not self.api_key:
                    raise ValueError("OpenAI API key not provided or found in configuration.")
                # Pass base_url and organization from config if available
                openai_kwargs = {
                    "api_key": self.api_key,
                    "base_url": provider_config.base_url or self.kwargs.get("base_url"),
                    "organization": provider_config.organization or self.kwargs.get("organization"),
                    "timeout": provider_config.timeout or self.kwargs.get("timeout"),
                }
                # Filter out None values before passing to OpenAI client
                openai_kwargs = {k: v for k, v in openai_kwargs.items() if v is not None}

                # Always use AsyncOpenAI
                self.client = AsyncOpenAI(**openai_kwargs)
                logger.info(
                    f"Initialized AsyncOpenAI embedding client for model: {self.model_name}"
                )

            else:
                raise ValueError(f"Unsupported embedding provider type: {self.provider_type}")

        except Exception as e:
            logger.error(
                f"Failed to initialize embedding service for provider {self.provider_type}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Embedding service initialization failed: {e}") from e

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors (each a list of floats).

        Raises:
            ValueError: If the provider type is unsupported or embedding fails.
            RuntimeError: If the client is not initialized.
        """
        if self.client is None:
            raise RuntimeError("Embedding client is not initialized.")

        try:
            if self.provider_type == "openai":
                response = await self.client.embeddings.create(input=texts, model=self.model_name)
                # Extract the embedding data
                embeddings = [item.embedding for item in response.data]
                logger.debug(
                    f"Successfully created {len(embeddings)} embeddings using {self.model_name}."
                )
                return embeddings

            else:
                raise ValueError(f"Unsupported provider type: {self.provider_type}")

        except Exception as e:
            logger.error(
                f"Failed to create embeddings using {self.provider_type} model {self.model_name}: {e}",
                exc_info=True,
            )
            # Re-raise the error or return an empty list/handle appropriately
            raise ValueError(f"Embedding creation failed: {e}") from e


def get_embedding_service(
    provider_type: str = "openai", model_name: str = "text-embedding-3-small", **kwargs
) -> EmbeddingService:
    """Factory function to get or create an EmbeddingService instance.

    Args:
        provider_type: The type of embedding provider.
        model_name: The specific embedding model.
        **kwargs: Additional arguments passed to the EmbeddingService constructor.

    Returns:
        An initialized EmbeddingService instance.
    """
    # Optional: Implement caching/singleton pattern for instances if desired
    instance_key = (provider_type, model_name)
    if instance_key in embedding_instances:
        # TODO: Check if kwargs match cached instance? For now, assume they do.
        logger.debug(
            f"Returning cached embedding service instance for {provider_type}/{model_name}"
        )
        return embedding_instances[instance_key]
    else:
        logger.debug(f"Creating new embedding service instance for {provider_type}/{model_name}")
        instance = EmbeddingService(provider_type=provider_type, model_name=model_name, **kwargs)
        embedding_instances[instance_key] = instance
        return instance


# Example usage (for testing)
async def main():
    # setup_logging(log_level="DEBUG") # Removed as logging is configured centrally
    # Make sure OPENAI_API_KEY is set in your .env file or environment
    os.environ["GATEWAY_FORCE_CONFIG_RELOAD"] = "true"  # Ensure latest config

    try:
        # Get the default OpenAI service
        openai_service = get_embedding_service()

        texts_to_embed = [
            "The quick brown fox jumps over the lazy dog.",
            "Quantum computing leverages quantum mechanics.",
            "Paris is the capital of France.",
        ]

        embeddings = await openai_service.create_embeddings(texts_to_embed)
        print(f"Generated {len(embeddings)} embeddings.")
        print(f"Dimension of first embedding: {len(embeddings[0])}")
        # print(f"First embedding (preview): {embeddings[0][:10]}...")

        # Example of specifying a different model (if available and configured)
        # try:
        #     ada_service = get_embedding_service(model_name='text-embedding-ada-002')
        #     ada_embeddings = await ada_service.create_embeddings(["Test with Ada model"])
        #     print(\"\nSuccessfully used Ada model.\")
        # except Exception as e:
        #     print(f\"\nCould not use Ada model (may need different API key/config): {e}\")

    except Exception as e:
        print(f"An error occurred during the example: {e}")
    finally:
        if "GATEWAY_FORCE_CONFIG_RELOAD" in os.environ:
            del os.environ["GATEWAY_FORCE_CONFIG_RELOAD"]


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
