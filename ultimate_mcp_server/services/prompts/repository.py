"""Prompt repository for managing and accessing prompts."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles

from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


class PromptRepository:
    """Repository for managing and accessing prompts."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton implementation for prompt repository."""
        if cls._instance is None:
            cls._instance = super(PromptRepository, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """Initialize the prompt repository.

        Args:
            base_dir: Base directory for prompt storage
        """
        # Only initialize once for singleton
        if self._initialized:
            return

        # Set base directory
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Default to project directory / prompts
            self.base_dir = Path.home() / ".ultimate" / "prompts"

        # Create directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Cache for prompts
        self.prompts = {}

        # Flag as initialized
        self._initialized = True

        logger.info(
            f"Prompt repository initialized (base_dir: {self.base_dir})", emoji_key="provider"
        )

    async def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a prompt by ID.

        Args:
            prompt_id: Prompt identifier

        Returns:
            Prompt data or None if not found
        """
        # Check cache first
        if prompt_id in self.prompts:
            return self.prompts[prompt_id]

        # Try to load from file
        prompt_path = self.base_dir / f"{prompt_id}.json"
        if not prompt_path.exists():
            logger.warning(f"Prompt '{prompt_id}' not found", emoji_key="warning")
            return None

        try:
            # Load prompt from file
            async with asyncio.Lock():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)

            # Cache for future use
            self.prompts[prompt_id] = prompt_data

            return prompt_data
        except Exception as e:
            logger.error(f"Error loading prompt '{prompt_id}': {str(e)}", emoji_key="error")
            return None

    async def save_prompt(self, prompt_id: str, prompt_data: Dict[str, Any]) -> bool:
        """Save a prompt.

        Args:
            prompt_id: Prompt identifier
            prompt_data: Prompt data to save

        Returns:
            True if successful
        """
        # Validate prompt data
        if not isinstance(prompt_data, dict) or "template" not in prompt_data:
            logger.error(f"Invalid prompt data for '{prompt_id}'", emoji_key="error")
            return False

        try:
            # Save to cache
            self.prompts[prompt_id] = prompt_data

            # Save to file
            prompt_path = self.base_dir / f"{prompt_id}.json"
            async with asyncio.Lock():
                async with aiofiles.open(prompt_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(prompt_data, indent=2))

            logger.info(f"Saved prompt '{prompt_id}'", emoji_key="success")
            return True
        except Exception as e:
            logger.error(f"Error saving prompt '{prompt_id}': {str(e)}", emoji_key="error")
            return False

    async def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt.

        Args:
            prompt_id: Prompt identifier

        Returns:
            True if successful
        """
        # Remove from cache
        if prompt_id in self.prompts:
            del self.prompts[prompt_id]

        # Remove file if exists
        prompt_path = self.base_dir / f"{prompt_id}.json"
        if prompt_path.exists():
            try:
                os.remove(prompt_path)
                logger.info(f"Deleted prompt '{prompt_id}'", emoji_key="success")
                return True
            except Exception as e:
                logger.error(f"Error deleting prompt '{prompt_id}': {str(e)}", emoji_key="error")
                return False

        return False

    async def list_prompts(self) -> List[str]:
        """List available prompts.

        Returns:
            List of prompt IDs
        """
        try:
            # Get prompt files
            prompt_files = list(self.base_dir.glob("*.json"))

            # Extract IDs from filenames
            prompt_ids = [f.stem for f in prompt_files]

            return prompt_ids
        except Exception as e:
            logger.error(f"Error listing prompts: {str(e)}", emoji_key="error")
            return []


def get_prompt_repository(base_dir: Optional[Union[str, Path]] = None) -> PromptRepository:
    """Get the prompt repository singleton instance.

    Args:
        base_dir: Base directory for prompt storage

    Returns:
        PromptRepository singleton instance
    """
    return PromptRepository(base_dir)
