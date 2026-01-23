"""YAML parsing utilities for Ultimate MCP Server."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def find_config_file() -> Optional[Path]:
    """Find the configuration file in standard locations.

    Looks for config.yaml in:
    1. Current directory
    2. ~/.config/umcp/config.yaml

    Returns:
        Path to config file if found, None otherwise
    """
    # Check current directory
    local_config = Path.cwd() / "config.yaml"
    if local_config.exists():
        return local_config

    # Check user's config directory
    home_config = Path.home() / ".config" / "umcp" / "config.yaml"
    if home_config.exists():
        return home_config

    return None


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML configuration from file.

    Args:
        config_path: Path to config file. If None, will try to find in standard locations.

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file has invalid YAML
    """
    if config_path is None:
        config_path = find_config_file()

    if not config_path or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e


def get_provider_config(
    provider_name: str, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get configuration for a specific provider.

    Args:
        provider_name: Name of the provider (e.g., "ollama")
        config: Configuration dictionary. If None, will load from file.

    Returns:
        Dictionary with provider configuration or empty dict if not found
    """
    if config is None:
        config = load_yaml_config()

    providers = config.get("providers", {})
    return providers.get(provider_name, {})
