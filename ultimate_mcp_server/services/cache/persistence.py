"""Cache persistence mechanisms."""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles

from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


class CachePersistence:
    """Handles cache persistence operations."""

    def __init__(self, cache_dir: Path):
        """Initialize the cache persistence handler.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "cache.pkl"
        self.metadata_file = cache_dir / "metadata.json"

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def save_cache(
        self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save cache data to disk.

        Args:
            data: Cache data to save
            metadata: Optional metadata about the cache

        Returns:
            True if successful
        """
        try:
            # Save cache data
            temp_file = f"{self.cache_file}.tmp"
            async with aiofiles.open(temp_file, "wb") as f:
                await f.write(pickle.dumps(data))

            # Rename temp file to cache file (atomic operation)
            os.replace(temp_file, self.cache_file)

            # Save metadata if provided
            if metadata:
                await self.save_metadata(metadata)

            logger.debug(f"Saved cache data to {self.cache_file}", emoji_key="cache")
            return True

        except Exception as e:
            logger.error(f"Failed to save cache data: {str(e)}", emoji_key="error")
            return False

    async def load_cache(self) -> Optional[Dict[str, Any]]:
        """Load cache data from disk.

        Returns:
            Cache data or None if file doesn't exist or error occurs
        """
        if not self.cache_file.exists():
            return None

        try:
            async with aiofiles.open(self.cache_file, "rb") as f:
                data = await f.read()

            cache_data = pickle.loads(data)

            logger.debug(f"Loaded cache data from {self.cache_file}", emoji_key="cache")
            return cache_data

        except Exception as e:
            logger.error(f"Failed to load cache data: {str(e)}", emoji_key="error")
            return None

    async def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Save cache metadata to disk.

        Args:
            metadata: Metadata to save

        Returns:
            True if successful
        """
        try:
            # Save metadata
            temp_file = f"{self.metadata_file}.tmp"
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(metadata, indent=2))

            # Rename temp file to metadata file (atomic operation)
            os.replace(temp_file, self.metadata_file)

            return True

        except Exception as e:
            logger.error(f"Failed to save cache metadata: {str(e)}", emoji_key="error")
            return False

    async def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load cache metadata from disk.

        Returns:
            Metadata or None if file doesn't exist or error occurs
        """
        if not self.metadata_file.exists():
            return None

        try:
            async with aiofiles.open(self.metadata_file, "r") as f:
                data = await f.read()

            metadata = json.loads(data)

            return metadata

        except Exception as e:
            logger.error(f"Failed to load cache metadata: {str(e)}", emoji_key="error")
            return None

    async def cleanup_old_cache_files(self, max_age_days: int = 30) -> int:
        """Clean up old cache files.

        Args:
            max_age_days: Maximum age of cache files in days

        Returns:
            Number of files deleted
        """
        import time

        now = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        deleted_count = 0

        try:
            # Find all cache files
            cache_files = list(self.cache_dir.glob("*.tmp"))

            # Delete old files
            for file_path in cache_files:
                mtime = file_path.stat().st_mtime
                age = now - mtime

                if age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old cache files", emoji_key="cache")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clean up old cache files: {str(e)}", emoji_key="error")
            return deleted_count
