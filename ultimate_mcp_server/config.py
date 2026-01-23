"""
Configuration management for Ultimate MCP Server.

Handles loading, validation, and access to configuration settings
from environment variables and config files.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from decouple import Config as DecoupleConfig
from decouple import RepositoryEnv, UndefinedValueError
from pydantic import BaseModel, Field, ValidationError, field_validator

# from pydantic_settings import BaseSettings, SettingsConfigDict # Removed BaseSettings

# --- Decouple Config Instance ---
# This will read from .env file and environment variables
decouple_config = DecoupleConfig(RepositoryEnv(".env"))
# --------------------------------

# Default configuration file paths (Adapt as needed)
DEFAULT_CONFIG_PATHS = [
    "./gateway_config.yaml",
    "./gateway_config.yml",
    "./gateway_config.json",
    "~/.config/ultimate_mcp_server/config.yaml",
    "~/.ultimate_mcp_server.yaml",
]

# Environment variable prefix (still potentially useful for non-secret env vars)
ENV_PREFIX = "GATEWAY_"

# Global configuration instance
_config = None

# Basic logger for config loading issues before full logging is set up
config_logger = logging.getLogger("ultimate_mcp_server.config")
handler = logging.StreamHandler(sys.stderr)
if not config_logger.hasHandlers():
    config_logger.addHandler(handler)
    config_logger.setLevel(logging.INFO)


class ServerConfig(BaseModel):
    """
    HTTP server configuration settings for the Ultimate MCP Server.

    This configuration class defines the core server parameters including network binding,
    performance settings, debugging options, and server identity information. It controls
    how the Ultimate MCP Server presents itself on the network and manages HTTP connections,
    especially when running in SSE (Server-Sent Events) mode.

    Settings defined here affect:
    - Where and how the server listens for connections (host, port)
    - How many concurrent workers are spawned to handle requests
    - Cross-origin resource sharing (CORS) for web clients
    - Logging verbosity level
    - Debug capabilities for development

    Most of these settings can be overridden at startup using environment variables
    or command-line arguments when launching the server.

    All values have sensible defaults suitable for local development. For production
    deployments, it's recommended to adjust host, port, workers, and CORS settings
    based on your specific requirements.
    """

    name: str = Field("Ultimate MCP Server", description="Name of the server")
    host: str = Field("127.0.0.1", description="Host to bind the server to")
    port: int = Field(8013, description="Port to bind the server to")  # Default port changed
    workers: int = Field(1, description="Number of worker processes")
    debug: bool = Field(False, description="Enable debug mode (affects reload)")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"], description="CORS allowed origins"
    )  # Use default_factory for mutable defaults
    log_level: str = Field(
        "info", description="Logging level (debug, info, warning, error, critical)"
    )
    version: str = Field("0.1.0", description="Server version (from config, not package)")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """
        Validate and normalize the log level configuration value.

        This validator ensures that the log_level field contains a valid logging level string.
        It performs two key functions:

        1. Validation: Checks that the provided value is one of the allowed logging levels
           (debug, info, warning, error, critical). If the value is invalid, it raises a
           ValidationError with a clear message listing the allowed values.

        2. Normalization: Converts the input to lowercase to ensure consistent handling
           regardless of how the value was specified in configuration sources. This allows
           users to specify the level in any case (e.g., "INFO", "info", "Info") and have
           it properly normalized.

        Args:
            v: The raw log_level value from the configuration source (file, env var, etc.)

        Returns:
            str: The validated and normalized (lowercase) log level string

        Raises:
            ValueError: If the provided value is not one of the allowed logging levels

        Example:
            >>> ServerConfig.validate_log_level("INFO")
            'info'
            >>> ServerConfig.validate_log_level("warning")
            'warning'
            >>> ServerConfig.validate_log_level("invalid")
            ValueError: Log level must be one of ['debug', 'info', 'warning', 'error', 'critical']
        """
        allowed = ["debug", "info", "warning", "error", "critical"]
        level_lower = v.lower()
        if level_lower not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return level_lower


class CacheConfig(BaseModel):
    """
    Caching system configuration for the Ultimate MCP Server.

    This configuration class defines parameters for the server's caching infrastructure,
    which is used to store and retrieve frequently accessed data like LLM completions.
    Effective caching significantly reduces API costs, improves response times, and
    decreases load on provider APIs.

    The caching system supports:
    - In-memory caching with configurable entry limits
    - Time-based expiration of cached entries
    - Optional persistence to disk
    - Fuzzy matching for similar but non-identical requests

    When enabled, the caching layer sits between tool calls and provider APIs,
    intercepting duplicate requests and returning cached results when appropriate.
    This is especially valuable for expensive operations like complex LLM completions
    that may be called multiple times with identical parameters.

    Proper cache configuration can dramatically reduce operating costs in production
    environments while improving response times for end users. The default settings
    provide a reasonable balance for most use cases, but may need adjustment based
    on traffic patterns and memory constraints.
    """

    enabled: bool = Field(True, description="Whether caching is enabled")
    ttl: int = Field(3600, description="Time-to-live for cache entries in seconds")
    max_entries: int = Field(10000, description="Maximum number of entries to store in cache")
    directory: Optional[str] = Field(None, description="Directory for cache persistence")
    fuzzy_match: bool = Field(True, description="Whether to use fuzzy matching for cache keys")


class ProviderConfig(BaseModel):
    """
    Configuration for an individual LLM provider connection.

    This class encapsulates all settings needed to establish and maintain a connection
    to a specific LLM provider service, such as OpenAI, Anthropic, or Gemini. Each provider
    instance in the system has its own configuration derived from this class, allowing for
    precise control over connection parameters, model selection, and authentication.

    The configuration supports:
    - Authentication via API keys (typically loaded from environment variables)
    - Custom API endpoints via base_url overrides
    - Organization-specific routing (for multi-tenant API services)
    - Default model selection for when no model is explicitly specified
    - Request timeout and token limit management
    - Provider-specific parameters via the additional_params dictionary

    Most provider settings can be loaded from either configuration files or environment
    variables, with environment variables taking precedence. This allows for secure
    management of sensitive credentials outside of versioned configuration files.

    For security best practices:
    - API keys should be specified via environment variables, not in configuration files
    - Custom API endpoints with private deployments should use HTTPS
    - Timeout values should be set appropriately to prevent hung connections

    Each provider has its own instance of this configuration class, allowing for
    independent configuration of multiple providers within the same server.
    """

    enabled: bool = Field(True, description="Whether the provider is enabled")
    api_key: Optional[str] = Field(
        None, description="API key for the provider (loaded via decouple)"
    )  # Updated description
    base_url: Optional[str] = Field(
        None, description="Base URL for API requests (loaded via decouple/file)"
    )  # Updated description
    organization: Optional[str] = Field(
        None, description="Organization identifier (loaded via decouple/file)"
    )  # Updated description
    default_model: Optional[str] = Field(
        None, description="Default model to use (loaded via decouple/file)"
    )  # Updated description
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for completions")
    timeout: Optional[float] = Field(30.0, description="Timeout for API requests in seconds")
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters (loaded via decouple/file)",
    )  # Updated description


class ProvidersConfig(BaseModel):
    """
    Centralized configuration for all supported LLM providers in the Ultimate MCP Server.

    This class serves as a container for individual provider configurations, organizing
    all supported provider settings in a structured hierarchy. It acts as the central
    registry of provider configurations, making it easy to:

    1. Access configuration for specific providers by name as attributes
    2. Iterate over all provider configurations for initialization and status checks
    3. Update provider settings through a consistent interface
    4. Add new providers to the system in a structured way

    Each provider has its own ProviderConfig instance as an attribute, named after the
    provider (e.g., openai, anthropic, gemini). This allows for dot-notation access
    to specific provider settings, providing a clean and intuitive API for configuration.

    The available providers are pre-defined based on the supported integrations in the
    system. Each provider's configuration follows the same structure but may have
    different default values or additional parameters based on provider-specific needs.

    When the configuration system loads settings from files or environment variables,
    it updates these provider configurations directly, making them the definitive source
    of provider settings throughout the application.
    """

    openai: ProviderConfig = Field(
        default_factory=ProviderConfig, description="OpenAI provider configuration"
    )
    anthropic: ProviderConfig = Field(
        default_factory=ProviderConfig, description="Anthropic provider configuration"
    )
    deepseek: ProviderConfig = Field(
        default_factory=ProviderConfig, description="DeepSeek provider configuration"
    )
    gemini: ProviderConfig = Field(
        default_factory=ProviderConfig, description="Gemini provider configuration"
    )
    openrouter: ProviderConfig = Field(
        default_factory=ProviderConfig, description="OpenRouter provider configuration"
    )
    grok: ProviderConfig = Field(
        default_factory=ProviderConfig, description="Grok (xAI) provider configuration"
    )
    ollama: ProviderConfig = Field(
        default_factory=ProviderConfig, description="Ollama provider configuration"
    )


class FilesystemProtectionConfig(BaseModel):
    """Configuration for filesystem protection heuristics."""

    enabled: bool = Field(True, description="Enable protection checks for this operation")
    max_files_threshold: int = Field(50, description="Trigger detailed check above this many files")
    datetime_stddev_threshold_sec: float = Field(
        60 * 60 * 24 * 30, description="Timestamp variance threshold (seconds)"
    )
    file_type_variance_threshold: int = Field(5, description="File extension variance threshold")
    max_stat_errors_pct: float = Field(
        10.0, description="Max percentage of failed stat calls allowed during check"
    )


class FilesystemConfig(BaseModel):
    """Configuration for filesystem tools."""

    allowed_directories: List[str] = Field(
        default_factory=list, description="List of absolute paths allowed for access"
    )
    file_deletion_protection: FilesystemProtectionConfig = Field(
        default_factory=FilesystemProtectionConfig,
        description="Settings for deletion protection heuristics",
    )
    file_modification_protection: FilesystemProtectionConfig = Field(
        default_factory=FilesystemProtectionConfig,
        description="Settings for modification protection heuristics (placeholder)",
    )
    default_encoding: str = Field("utf-8", description="Default encoding for text file operations")
    max_read_size_bytes: int = Field(
        100 * 1024 * 1024, description="Maximum size for reading files"
    )  # 100MB example


class AgentMemoryConfig(BaseModel):
    """Configuration for Cognitive and Agent Memory tool."""

    db_path: str = Field(
        "unified_agent_memory.db", description="Path to the agent memory SQLite database"
    )
    max_text_length: int = Field(
        64000, description="Maximum length for text fields (e.g., content, reasoning)"
    )
    connection_timeout: float = Field(10.0, description="Database connection timeout in seconds")
    max_working_memory_size: int = Field(
        20, description="Maximum number of items in working memory"
    )
    memory_decay_rate: float = Field(0.01, description="Decay rate for memory relevance per hour")
    importance_boost_factor: float = Field(
        1.5, description="Multiplier for explicitly marked important memories"
    )
    similarity_threshold: float = Field(
        0.75, description="Default threshold for semantic similarity search"
    )
    max_semantic_candidates: int = Field(
        500, description="Maximum candidates to consider in semantic search before scoring"
    )

    # TTLs per level (in seconds)
    ttl_working: int = Field(60 * 30, description="Default TTL for working memories (seconds)")
    ttl_episodic: int = Field(
        60 * 60 * 24 * 7, description="Default TTL for episodic memories (seconds)"
    )
    ttl_semantic: int = Field(
        60 * 60 * 24 * 30, description="Default TTL for semantic memories (seconds)"
    )
    ttl_procedural: int = Field(
        60 * 60 * 24 * 90, description="Default TTL for procedural memories (seconds)"
    )

    # Embedding related (primarily for reference/defaults, service might override)
    default_embedding_model: str = Field(
        "text-embedding-3-small", description="Default embedding model identifier"
    )
    embedding_dimension: int = Field(
        1536, description="Expected dimension for the default embedding model"
    )

    # Multi-tool support (for agents that make multiple UMS calls per turn)
    enable_batched_operations: bool = Field(
        True, description="Allow multiple tool calls per agent turn"
    )
    max_tools_per_batch: int = Field(
        20, description="Maximum number of tools that can be called in a single batch"
    )

    # SQLite Optimizations (Defined here, not env vars by default)
    sqlite_pragmas: List[str] = Field(
        default_factory=lambda: [
            "PRAGMA journal_mode=DELETE",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA foreign_keys=ON",
            "PRAGMA temp_store=MEMORY",
            "PRAGMA cache_size=-32000",  # ~32MB cache
            "PRAGMA mmap_size=2147483647",  # Max mmap size
            "PRAGMA busy_timeout=30000",  # 30 seconds busy timeout
        ],
        description="List of SQLite PRAGMA statements for optimization",
    )


class ToolRegistrationConfig(BaseModel):
    """Configuration for tool registration."""

    filter_enabled: bool = Field(False, description="Whether to filter which tools are registered")
    included_tools: List[str] = Field(
        default_factory=list, description="List of tool names to include (empty means include all)"
    )
    excluded_tools: List[str] = Field(
        default_factory=list,
        description="List of tool names to exclude (takes precedence over included_tools)",
    )


class SmartBrowserConfig(BaseModel):
    """Configuration specific to the Smart Browser tool."""

    sb_state_key_b64: Optional[str] = Field(
        None,
        description="Base64 encoded AES key for state encryption (e.g., 'openssl rand -base64 32')",
    )
    sb_max_tabs: int = Field(5, description="Max concurrent tabs in the pool")
    sb_tab_timeout: int = Field(300, description="Timeout for operations within a tab (seconds)")
    sb_inactivity_timeout: int = Field(
        600, description="Browser inactivity shutdown timeout (seconds)"
    )
    headless_mode: bool = Field(True, description="Run browser in headless mode")
    vnc_enabled: bool = Field(False, description="Enable VNC server for headful mode")
    vnc_password: Optional[str] = Field(
        None, description="Password for VNC server (required if vnc_enabled=True)"
    )
    proxy_pool_str: str = Field(
        "",
        description="Semicolon-separated list of proxy URLs (e.g., 'http://user:pass@host:port;socks5://host2:port2')",
    )
    proxy_allowed_domains_str: str = Field(
        "*",
        description="Comma-separated domains allowed for proxy (e.g., '.google.com,.example.com', '*' for all)",
    )
    vault_allowed_paths_str: str = Field(
        "secret/data/,kv/data/",
        description="Comma-separated allowed Vault path prefixes (e.g., 'kv/data/myapp/,secret/data/shared/')",
    )

    # Enhanced Locator Tunables
    max_widgets: int = Field(300, description="Max interactive elements extracted for page map")
    max_section_chars: int = Field(5000, description="Max chars for main text summary in page map")
    dom_fp_limit: int = Field(20000, description="Max chars used for DOM fingerprint calculation")
    llm_model_locator: str = Field("gpt-4o", description="LLM model used for locator fallback")
    retry_after_fail: int = Field(
        1, description="Number of LLM locator retries after initial failure"
    )
    seq_cutoff: float = Field(
        0.72, description="SequenceMatcher cutoff for heuristic locator match"
    )
    area_min: int = Field(
        400, description="Minimum pixel area (width*height) for elements in page map"
    )
    high_risk_domains_set: Set[str] = Field(  # Use set for direct comparison
        default_factory=lambda: {  # Use factory for mutable default
            ".google.com",
            ".facebook.com",
            ".linkedin.com",
            ".glassdoor.com",
            ".instagram.com",
            ".twitter.com",
            ".x.com",
            ".reddit.com",
            ".amazon.com",
            ".ebay.com",
            ".ticketmaster.com",
            ".cloudflare.com",
            ".datadome.co",
            ".perimeterx.net",
            ".recaptcha.net",
            ".hcaptcha.com",
        },
        description="Set of domains considered high-risk for bot detection (influences jitter timing)",
    )

    # Validator for high_risk_domains_set (ensures leading dot)
    @field_validator("high_risk_domains_set", mode="before")
    @classmethod
    def normalize_high_risk_domains(cls, v):
        if isinstance(v, str):  # Allow comma-separated string input from env/file
            domains = {d.strip().lower() for d in v.split(",") if d.strip()}
        elif isinstance(v, (list, set)):
            domains = {str(d).strip().lower() for d in v if str(d).strip()}
        else:
            raise ValueError("high_risk_domains_set must be a list, set, or comma-separated string")

        # Ensure leading dot for all domains
        normalized_domains = {d if d.startswith(".") else "." + d for d in domains}
        return normalized_domains


class GatewayConfig(BaseModel):  # Inherit from BaseModel now
    """
    Root configuration model for the entire Ultimate MCP Server system.

    This class serves as the top-level configuration container, bringing together
    all component-specific configurations into a unified structure. It represents the
    complete configuration state of the Ultimate MCP Server and is the primary interface
    for accessing configuration settings throughout the application.

    The configuration is hierarchically organized into logical sections:
    - server: Network, HTTP, and core server settings
    - providers: LLM provider connections and credentials
    - cache: Response caching behavior and persistence
    - filesystem: Safe filesystem access rules and protection
    - agent_memory: Settings for the agent memory and cognitive systems
    - tool_registration: Controls for which tools are enabled

    Additionally, it includes several top-level settings for paths and directories
    that are used across multiple components of the system.

    This configuration model is loaded through the config module's functions, which
    handle merging settings from:
    1. Default values defined in the model
    2. Configuration files (YAML/JSON)
    3. Environment variables
    4. Command-line arguments (where applicable)

    Throughout the application, this configuration is accessed through the get_config()
    function, which returns a singleton instance of this class with all settings
    properly loaded and validated.

    Usage example:
        ```python
        from ultimate_mcp_server.config import get_config

        config = get_config()

        # Access configuration sections
        server_port = config.server.port
        openai_api_key = config.providers.openai.api_key

        # Access top-level settings
        logs_dir = config.log_directory
        ```
    """

    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)
    agent_memory: AgentMemoryConfig = Field(default_factory=AgentMemoryConfig)  # Added agent memory
    tool_registration: ToolRegistrationConfig = Field(
        default_factory=ToolRegistrationConfig
    )  # Added tool registration config
    smart_browser: SmartBrowserConfig = Field(default_factory=SmartBrowserConfig)
    default_provider: str = Field(
        "openai",
        description="Default LLM provider to use if unspecified (e.g., 'openai', 'anthropic')",
    )

    storage_directory: str = Field("./storage", description="Directory for persistent storage")
    log_directory: str = Field("./logs", description="Directory for log files")
    prompt_templates_directory: str = Field(
        "./prompt_templates", description="Directory containing prompt templates"
    )  # Added prompt dir


def expand_path(path: str) -> str:
    """
    Expand a path string to resolve user home directories and environment variables.

    This utility function takes a potentially relative path string that may contain
    user home directory references (e.g., "~/logs") or environment variables
    (e.g., "$HOME/data") and expands it to an absolute path.

    The expansion process:
    1. Expands user home directory (e.g., "~" → "/home/username")
    2. Expands environment variables (e.g., "$VAR" → "value")
    3. Converts to an absolute path (resolving relative paths)

    Args:
        path: A path string that may contain "~" or environment variables

    Returns:
        The expanded absolute path as a string

    Example:
        >>> expand_path("~/logs")
        '/home/username/logs'
        >>> expand_path("$DATA_DIR/cache")
        '/var/data/cache'  # Assuming $DATA_DIR is set to "/var/data"
    """
    expanded = os.path.expanduser(path)
    expanded = os.path.expandvars(expanded)
    return os.path.abspath(expanded)


def find_config_file() -> Optional[str]:
    """
    Find the first available configuration file from the list of default paths.

    This function searches for configuration files in standard locations, following
    a predefined priority order. It checks each potential location sequentially and
    returns the path of the first valid configuration file found.

    The search locations (defined in DEFAULT_CONFIG_PATHS) typically include:
    - Current directory (e.g., "./gateway_config.yaml")
    - User config directory (e.g., "~/.config/ultimate_mcp_server/config.yaml")
    - User home directory (e.g., "~/.ultimate_mcp_server.yaml")

    Each path is expanded using expand_path() before checking if it exists.

    Returns:
        The path to the first found configuration file, or None if no files exist

    Note:
        This function only verifies that the files exist, not that they have
        valid content or format. Content validation happens during actual loading.
    """
    for path in DEFAULT_CONFIG_PATHS:
        try:
            expanded_path = expand_path(path)
            if os.path.isfile(expanded_path):
                config_logger.debug(f"Found config file: {expanded_path}")
                return expanded_path
        except Exception as e:
            config_logger.debug(f"Could not check path {path}: {e}")
    config_logger.debug("No default config file found.")
    return None


def load_config_from_file(path: str) -> Dict[str, Any]:
    """
    Load configuration data from a YAML or JSON file.

    This function reads and parses a configuration file into a Python dictionary.
    It automatically detects the file format based on the file extension:
    - .yaml/.yml: Parsed as YAML using PyYAML
    - .json: Parsed as JSON using Python's built-in json module

    The function performs several steps:
    1. Expands the path to resolve any home directory (~/...) or environment variables
    2. Verifies that the file exists
    3. Determines the appropriate parser based on file extension
    4. Reads and parses the file content
    5. Returns the parsed configuration as a dictionary

    Args:
        path: Path to the configuration file (can be relative or use ~/... or $VAR/...)

    Returns:
        Dictionary containing the parsed configuration data

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the file has an unsupported format or contains invalid syntax
        RuntimeError: If there are other errors reading the file

    Note:
        If the file is empty or contains "null" in YAML, an empty dictionary is
        returned rather than None, ensuring consistent return type.
    """
    path = expand_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config_logger.debug(f"Loading configuration from file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                config_data = yaml.safe_load(f)
            elif path.endswith(".json"):
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path}")
            return config_data if config_data is not None else {}
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid format in {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading {path}: {e}") from e


def load_config(
    config_file_path: Optional[str] = None,
    load_default_files: bool = True,
) -> GatewayConfig:
    """
    Load, merge, and validate configuration from multiple sources with priority handling.

    This function implements the complete configuration loading process, combining settings
    from multiple sources according to their priority. It also handles path expansion,
    directory creation, and validation of the resulting configuration.

    Configuration Sources (in order of decreasing priority):
    1. Environment variables (via decouple) - Use GATEWAY_* prefix or provider-specific vars
    2. .env file variables (via decouple) - Same naming as environment variables
    3. YAML/JSON configuration file - If explicitly specified or found in default locations
    4. Default values defined in Pydantic models - Fallback when no other source specifies a value

    Special handling:
    - Provider API keys: Loaded from provider-specific environment variables
      (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
    - Directory paths: Automatically expanded and created if they don't exist
    - Validation: All configuration values are validated against their Pydantic models

    Args:
        config_file_path: Optional explicit path to a configuration file to load.
                         If provided, this file must exist and be valid YAML/JSON.
        load_default_files: Whether to search for configuration files in default locations
                           if config_file_path is not provided. Default: True

    Returns:
        GatewayConfig: A fully loaded and validated configuration object

    Raises:
        FileNotFoundError: If an explicitly specified config file doesn't exist
        ValueError: If the config file has invalid format or content
        RuntimeError: If other errors occur during loading

    Example:
        ```python
        # Load with defaults and environment variables
        config = load_config()

        # Load from a specific config file
        config = load_config(config_file_path="path/to/custom_config.yaml")

        # Load only from environment variables, ignoring config files
        config = load_config(load_default_files=False)
        ```
    """
    global _config
    file_config_data = {}

    # 1. Find and load config file (if specified or found)
    chosen_file_path = None
    if config_file_path:
        chosen_file_path = expand_path(config_file_path)
    elif load_default_files:
        chosen_file_path = find_config_file()

    if chosen_file_path and os.path.isfile(chosen_file_path):
        try:
            file_config_data = load_config_from_file(chosen_file_path)
            config_logger.info(f"Loaded base configuration from: {chosen_file_path}")
        except Exception as e:
            config_logger.warning(f"Could not load config file {chosen_file_path}: {e}")
            if config_file_path:
                raise ValueError(f"Failed to load specified config: {chosen_file_path}") from e
    elif config_file_path:
        raise FileNotFoundError(f"Specified configuration file not found: {config_file_path}")

    # 2. Initialize GatewayConfig from Pydantic defaults and file data
    try:
        # Ensure nested keys exist before validation if loading from potentially incomplete file
        file_config_data.setdefault("server", {})
        file_config_data.setdefault("providers", {})
        file_config_data.setdefault("cache", {})
        file_config_data.setdefault("filesystem", {})
        file_config_data.setdefault("agent_memory", {})
        file_config_data.setdefault("tool_registration", {})
        file_config_data.setdefault("smart_browser", {})
        loaded_config = GatewayConfig.model_validate(file_config_data)
    except ValidationError as e:
        config_logger.error("Configuration validation failed during file/default loading:")
        config_logger.error(str(e))
        config_logger.warning("Falling back to default configuration before applying env vars.")
        loaded_config = GatewayConfig()  # Fallback to defaults

    # 3. Use decouple to load/override settings from .env/environment variables
    #    Decouple handles checking env vars and .env file automatically.

    # --- Load Provider API Keys ---
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "grok": "GROK_API_KEY",
    }
    for provider_name, env_var in provider_key_map.items():
        provider_conf = getattr(loaded_config.providers, provider_name, None)
        if provider_conf:
            api_key_from_env = decouple_config.get(env_var, default=None)
            if api_key_from_env:
                if provider_conf.api_key and provider_conf.api_key != api_key_from_env:
                    config_logger.debug(f"Overriding API key for {provider_name} from env/'.env'.")
                elif not provider_conf.api_key:
                    config_logger.debug(f"Setting API key for {provider_name} from env/'.env'.")
                provider_conf.api_key = api_key_from_env

    try:
        # Use the default defined in GatewayConfig as the fallback if env/file doesn't specify
        loaded_config.default_provider = decouple_config(
            "DEFAULT_PROVIDER", default=loaded_config.default_provider
        )
        config_logger.debug(f"Set default provider: {loaded_config.default_provider}")
    except Exception as e:
        config_logger.warning(
            f"Could not load default provider from env: {e}. Using default '{loaded_config.default_provider}'."
        )

    # --- Load other Provider settings (base_url, default_model, org, specific headers) ---
    # Example for OpenRouter specific headers
    openrouter_conf = loaded_config.providers.openrouter
    try:
        # Use get() to avoid UndefinedValueError if not set
        http_referer = decouple_config.get("OPENROUTER_HTTP_REFERER", default=None)
        x_title = decouple_config.get("OPENROUTER_X_TITLE", default=None)
        if http_referer:
            openrouter_conf.additional_params["http_referer"] = http_referer
            config_logger.debug("Setting OpenRouter http_referer from env/'.env'.")
        if x_title:
            openrouter_conf.additional_params["x_title"] = x_title
            config_logger.debug("Setting OpenRouter x_title from env/'.env'.")
    except Exception as e:  # Catch potential decouple issues
        config_logger.warning(f"Could not load optional OpenRouter headers from env: {e}")

    # --- Load Ollama Provider Settings ---
    ollama_conf = loaded_config.providers.ollama
    try:
        enabled_env = decouple_config.get("OLLAMA_ENABLED", default=None)
        if enabled_env is not None:
            ollama_conf.enabled = enabled_env.lower() == "true"
            config_logger.debug(f"Setting Ollama enabled from env/'.env': {ollama_conf.enabled}")

        api_url_env = decouple_config.get("OLLAMA_API_URL", default=None)
        if api_url_env:
            ollama_conf.base_url = api_url_env
            config_logger.debug(f"Setting Ollama base_url from env/'.env': {ollama_conf.base_url}")

        default_model_env = decouple_config.get("OLLAMA_DEFAULT_MODEL", default=None)
        if default_model_env:
            ollama_conf.default_model = default_model_env
            config_logger.debug(
                f"Setting Ollama default_model from env/'.env': {ollama_conf.default_model}"
            )

        request_timeout_env = decouple_config.get("OLLAMA_REQUEST_TIMEOUT", default=None)
        if request_timeout_env is not None:
            ollama_conf.timeout = int(request_timeout_env)
            config_logger.debug(f"Setting Ollama timeout from env/'.env': {ollama_conf.timeout}")
    except Exception as e:
        config_logger.warning(f"Could not load optional Ollama settings from env: {e}")

    # Example for generic provider settings like base_url, default_model, organization
    for provider_name in [
        "openai",
        "anthropic",
        "deepseek",
        "gemini",
        "openrouter",
        "grok",
        "ollama",
    ]:
        provider_conf = getattr(loaded_config.providers, provider_name, None)
        if provider_conf:
            p_name_upper = provider_name.upper()
            try:
                base_url_env = decouple_config.get(f"{p_name_upper}_BASE_URL", default=None)
                if base_url_env:
                    provider_conf.base_url = base_url_env
                    config_logger.debug(f"Setting {provider_name} base_url from env/'.env'.")

                default_model_env = decouple_config.get(
                    f"{p_name_upper}_DEFAULT_MODEL", default=None
                )
                if default_model_env:
                    provider_conf.default_model = default_model_env
                    config_logger.debug(f"Setting {provider_name} default_model from env/'.env'.")

                org_env = decouple_config.get(f"{p_name_upper}_ORGANIZATION", default=None)
                if org_env:
                    provider_conf.organization = org_env
                    config_logger.debug(f"Setting {provider_name} organization from env/'.env'.")

            except Exception as e:
                config_logger.warning(
                    f"Could not load optional settings for provider {provider_name} from env: {e}"
                )

    # --- Load Server Port ---
    try:
        server_port_env = decouple_config.get("GATEWAY_SERVER_PORT", default=None)
        if server_port_env is not None:
            loaded_config.server.port = decouple_config("GATEWAY_SERVER_PORT", cast=int)
            config_logger.debug(f"Overriding server port from env: {loaded_config.server.port}")
    except (ValueError, UndefinedValueError) as e:
        config_logger.warning(
            f"Invalid or missing GATEWAY_SERVER_PORT env var: {e}. Using default/file value."
        )

    # --- Load Filesystem Allowed Directories ---
    allowed_dirs_env_var = "FILESYSTEM__ALLOWED_DIRECTORIES"
    try:
        allowed_dirs_env_value_str = decouple_config.get(allowed_dirs_env_var, default=None)
        if allowed_dirs_env_value_str is not None:
            try:
                allowed_dirs_from_env = json.loads(allowed_dirs_env_value_str)
                if isinstance(allowed_dirs_from_env, list):
                    if loaded_config.filesystem.allowed_directories:
                        config_logger.debug(
                            f"Overriding filesystem.allowed_directories from env var {allowed_dirs_env_var}."
                        )
                    else:
                        config_logger.debug(
                            f"Setting filesystem.allowed_directories from env var {allowed_dirs_env_var}."
                        )
                    loaded_config.filesystem.allowed_directories = allowed_dirs_from_env
                else:
                    config_logger.warning(
                        f"Env var {allowed_dirs_env_var} did not contain a valid JSON list. Value ignored."
                    )
            except json.JSONDecodeError:
                config_logger.warning(
                    f"Failed to parse JSON from env var {allowed_dirs_env_var}. Value: '{allowed_dirs_env_value_str}'. Ignoring env var."
                )
    except Exception as e:
        config_logger.error(f"Error processing env var {allowed_dirs_env_var}: {e}", exc_info=True)

    # --- Load Agent Memory Settings ---
    agent_mem_conf = loaded_config.agent_memory  # Get the config object
    try:
        agent_mem_conf.db_path = decouple_config(
            "AGENT_MEMORY_DB_PATH", default=agent_mem_conf.db_path
        )
        agent_mem_conf.max_text_length = decouple_config(
            "AGENT_MEMORY_MAX_TEXT_LENGTH", default=agent_mem_conf.max_text_length, cast=int
        )
        agent_mem_conf.connection_timeout = decouple_config(
            "AGENT_MEMORY_CONNECTION_TIMEOUT", default=agent_mem_conf.connection_timeout, cast=float
        )
        agent_mem_conf.max_working_memory_size = decouple_config(
            "AGENT_MEMORY_MAX_WORKING_SIZE",
            default=agent_mem_conf.max_working_memory_size,
            cast=int,
        )
        # Load TTLs
        agent_mem_conf.ttl_working = decouple_config(
            "AGENT_MEMORY_TTL_WORKING", default=agent_mem_conf.ttl_working, cast=int
        )
        agent_mem_conf.ttl_episodic = decouple_config(
            "AGENT_MEMORY_TTL_EPISODIC", default=agent_mem_conf.ttl_episodic, cast=int
        )
        agent_mem_conf.ttl_semantic = decouple_config(
            "AGENT_MEMORY_TTL_SEMANTIC", default=agent_mem_conf.ttl_semantic, cast=int
        )
        agent_mem_conf.ttl_procedural = decouple_config(
            "AGENT_MEMORY_TTL_PROCEDURAL", default=agent_mem_conf.ttl_procedural, cast=int
        )
        # Load other parameters
        agent_mem_conf.memory_decay_rate = decouple_config(
            "AGENT_MEMORY_DECAY_RATE", default=agent_mem_conf.memory_decay_rate, cast=float
        )
        agent_mem_conf.importance_boost_factor = decouple_config(
            "AGENT_MEMORY_IMPORTANCE_BOOST",
            default=agent_mem_conf.importance_boost_factor,
            cast=float,
        )
        agent_mem_conf.similarity_threshold = decouple_config(
            "AGENT_MEMORY_SIMILARITY_THRESHOLD",
            default=agent_mem_conf.similarity_threshold,
            cast=float,
        )
        agent_mem_conf.max_semantic_candidates = decouple_config(
            "AGENT_MEMORY_MAX_SEMANTIC_CANDIDATES",
            default=agent_mem_conf.max_semantic_candidates,
            cast=int,
        )
        # Load embedding defaults (mainly for reference)
        agent_mem_conf.default_embedding_model = decouple_config(
            "AGENT_MEMORY_DEFAULT_EMBEDDING_MODEL", default=agent_mem_conf.default_embedding_model
        )
        agent_mem_conf.embedding_dimension = decouple_config(
            "AGENT_MEMORY_EMBEDDING_DIMENSION", default=agent_mem_conf.embedding_dimension, cast=int
        )

        # Load multi-tool support settings
        def _cast_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)

        agent_mem_conf.enable_batched_operations = decouple_config(
            "AGENT_MEMORY_ENABLE_BATCHED_OPERATIONS",
            default=agent_mem_conf.enable_batched_operations,
            cast=_cast_bool,
        )
        agent_mem_conf.max_tools_per_batch = decouple_config(
            "AGENT_MEMORY_MAX_TOOLS_PER_BATCH", default=agent_mem_conf.max_tools_per_batch, cast=int
        )

        config_logger.debug("Loaded agent memory settings from env/'.env' or defaults.")
    except (ValueError, UndefinedValueError) as e:
        config_logger.warning(
            f"Issue loading agent memory settings from env: {e}. Using Pydantic defaults."
        )
    except Exception as e:
        config_logger.error(f"Unexpected error loading agent memory settings: {e}", exc_info=True)

    # --- Load Prompt Templates Directory ---
    try:
        loaded_config.prompt_templates_directory = decouple_config(
            "GATEWAY_PROMPT_TEMPLATES_DIR", default=loaded_config.prompt_templates_directory
        )
        config_logger.debug(
            f"Set prompt templates directory: {loaded_config.prompt_templates_directory}"
        )
    except Exception as e:
        config_logger.warning(f"Could not load prompt templates directory from env: {e}")

    # --- Load Cache Directory ---
    try:
        cache_dir_env = decouple_config(
            "GATEWAY_CACHE_DIR", default=None
        )  # Changed env var name for clarity
        if cache_dir_env:
            loaded_config.cache.directory = cache_dir_env
            config_logger.debug(f"Set cache directory from env: {loaded_config.cache.directory}")
    except Exception as e:
        config_logger.warning(f"Could not load cache directory from env: {e}")

    sb_conf = loaded_config.smart_browser  # Get the config object
    try:
        # State Key (already added previously)
        sb_conf.sb_state_key_b64 = decouple_config("SB_STATE_KEY", default=sb_conf.sb_state_key_b64)
        if sb_conf.sb_state_key_b64:
            config_logger.debug("Loaded SB_STATE_KEY from env/'.env' or file.")
        else:
            config_logger.info("Smart Browser state encryption disabled (SB_STATE_KEY not found).")

        # Other SB settings
        sb_conf.sb_max_tabs = decouple_config("SB_MAX_TABS", default=sb_conf.sb_max_tabs, cast=int)
        sb_conf.sb_tab_timeout = decouple_config(
            "SB_TAB_TIMEOUT", default=sb_conf.sb_tab_timeout, cast=int
        )
        sb_conf.sb_inactivity_timeout = decouple_config(
            "SB_INACTIVITY_TIMEOUT", default=sb_conf.sb_inactivity_timeout, cast=int
        )
        sb_conf.headless_mode = decouple_config(
            "SB_HEADLESS_MODE", default=sb_conf.headless_mode, cast=bool
        )  # Use SB_ prefix
        sb_conf.vnc_enabled = decouple_config(
            "SB_VNC_ENABLED", default=sb_conf.vnc_enabled, cast=bool
        )  # Use SB_ prefix
        sb_conf.vnc_password = decouple_config(
            "SB_VNC_PASSWORD", default=sb_conf.vnc_password
        )  # Use SB_ prefix
        sb_conf.proxy_pool_str = decouple_config(
            "SB_PROXY_POOL", default=sb_conf.proxy_pool_str
        )  # Use SB_ prefix
        sb_conf.proxy_allowed_domains_str = decouple_config(
            "SB_PROXY_ALLOWED_DOMAINS", default=sb_conf.proxy_allowed_domains_str
        )  # Use SB_ prefix
        sb_conf.vault_allowed_paths_str = decouple_config(
            "SB_VAULT_ALLOWED_PATHS", default=sb_conf.vault_allowed_paths_str
        )  # Use SB_ prefix

        # Locator Tunables
        sb_conf.max_widgets = decouple_config(
            "SB_MAX_WIDGETS", default=sb_conf.max_widgets, cast=int
        )
        sb_conf.max_section_chars = decouple_config(
            "SB_MAX_SECTION_CHARS", default=sb_conf.max_section_chars, cast=int
        )
        sb_conf.dom_fp_limit = decouple_config(
            "SB_DOM_FP_LIMIT", default=sb_conf.dom_fp_limit, cast=int
        )
        sb_conf.llm_model_locator = decouple_config(
            "SB_LLM_MODEL_LOCATOR", default=sb_conf.llm_model_locator
        )
        sb_conf.retry_after_fail = decouple_config(
            "SB_RETRY_AFTER_FAIL", default=sb_conf.retry_after_fail, cast=int
        )
        sb_conf.seq_cutoff = decouple_config(
            "SB_SEQ_CUTOFF", default=sb_conf.seq_cutoff, cast=float
        )
        sb_conf.area_min = decouple_config("SB_AREA_MIN", default=sb_conf.area_min, cast=int)

        # High Risk Domains (Load as string, validator handles conversion)
        high_risk_domains_env = decouple_config("SB_HIGH_RISK_DOMAINS", default=None)
        if high_risk_domains_env is not None:
            # Let the validator handle parsing and normalization
            sb_conf.high_risk_domains_set = high_risk_domains_env  # Pass the raw string

        config_logger.debug("Loaded Smart Browser settings from env/'.env' or defaults.")

    except (ValueError, UndefinedValueError) as e:
        config_logger.warning(
            f"Issue loading Smart Browser settings from env: {e}. Using defaults/file values."
        )
    except Exception as e:
        config_logger.error(f"Unexpected error loading Smart Browser settings: {e}", exc_info=True)

    # --- Expand paths ---
    try:
        # Expand core directories
        loaded_config.storage_directory = expand_path(loaded_config.storage_directory)
        loaded_config.log_directory = expand_path(loaded_config.log_directory)
        loaded_config.prompt_templates_directory = expand_path(
            loaded_config.prompt_templates_directory
        )  # Expand new dir

        # Expand cache directory if set
        if loaded_config.cache.directory:
            loaded_config.cache.directory = expand_path(loaded_config.cache.directory)

        # Expand agent memory DB path (assuming it's a relative path)
        # Check if it's already absolute to avoid issues
        if not os.path.isabs(loaded_config.agent_memory.db_path):
            # Place it relative to storage_directory by default? Or workspace root? Let's choose storage.
            db_in_storage = (
                Path(loaded_config.storage_directory) / loaded_config.agent_memory.db_path
            )
            loaded_config.agent_memory.db_path = str(db_in_storage.resolve())
            config_logger.debug(
                f"Expanded agent memory db path to: {loaded_config.agent_memory.db_path}"
            )

        # Expand allowed filesystem directories
        expanded_allowed_dirs = []
        for d in loaded_config.filesystem.allowed_directories:
            if isinstance(d, str):
                expanded_allowed_dirs.append(expand_path(d))
            else:
                config_logger.warning(f"Ignoring non-string entry in allowed_directories: {d!r}")
        loaded_config.filesystem.allowed_directories = expanded_allowed_dirs
    except Exception as e:
        config_logger.error(f"Error expanding configured paths: {e}", exc_info=True)

    # --- Ensure critical directories exist ---
    try:
        # Use pathlib for consistency
        Path(loaded_config.storage_directory).mkdir(parents=True, exist_ok=True)
        Path(loaded_config.log_directory).mkdir(parents=True, exist_ok=True)
        Path(loaded_config.prompt_templates_directory).mkdir(
            parents=True, exist_ok=True
        )  # Ensure prompt dir exists

        if loaded_config.cache.enabled and loaded_config.cache.directory:
            Path(loaded_config.cache.directory).mkdir(parents=True, exist_ok=True)

        # Ensure Agent Memory DB directory exists
        db_dir = Path(loaded_config.agent_memory.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    except OSError as e:
        config_logger.error(f"Failed to create necessary directories: {e}")

    _config = loaded_config
    config_logger.debug(
        f"Effective allowed directories: {loaded_config.filesystem.allowed_directories}"
    )
    config_logger.debug(f"Effective Agent Memory DB path: {loaded_config.agent_memory.db_path}")
    config_logger.debug(
        f"Effective Prompt Templates directory: {loaded_config.prompt_templates_directory}"
    )
    return _config


def get_config() -> GatewayConfig:
    """
    Retrieve the globally cached configuration instance or load a new one if needed.

    This function serves as the primary entry point for accessing the server's configuration
    throughout the application. It implements a singleton pattern with on-demand loading and
    optional forced reloading to ensure consistent configuration access with minimal overhead.

    Key behaviors:
    - CACHING: Returns a previously loaded configuration instance when available
    - LAZY LOADING: Loads configuration on first access rather than at import time
    - FORCE RELOAD: Supports reloading via the GATEWAY_FORCE_CONFIG_RELOAD environment variable
    - COMPLETE: Includes settings from environment variables, config files, and defaults
    - VALIDATED: Uses Pydantic models to ensure all configuration values are valid

    The configuration loading follows this priority order:
    1. Environment variables (highest priority)
    2. .env file values
    3. Configuration file settings
    4. Pydantic default values (lowest priority)

    Returns:
        GatewayConfig: The validated configuration instance with all settings applied.

    Raises:
        RuntimeError: If configuration loading fails for any reason (invalid settings,
                     missing required values, inaccessible files, etc.)

    Example usage:
        ```python
        from ultimate_mcp_server.config import get_config

        # Access server configuration
        config = get_config()
        server_port = config.server.port

        # Access provider API keys
        openai_api_key = config.providers.openai.api_key

        # Check if a feature is enabled
        if config.cache.enabled:
            # Use caching functionality
            pass
        ```
    """
    global _config
    # Use decouple directly here for the reload flag check
    force_reload = (
        decouple_config.get("GATEWAY_FORCE_CONFIG_RELOAD", default="false").lower() == "true"
    )

    if _config is None or force_reload:
        try:
            _config = load_config()  # load_config now handles internal state update
        except Exception as e:
            config_logger.critical(f"Failed to load configuration: {e}", exc_info=True)
            raise RuntimeError("Configuration could not be loaded.") from e

    if _config is None:  # Should not happen if load_config succeeded or raised
        raise RuntimeError("Configuration is None after loading attempt.")

    return _config


def get_config_as_dict() -> Dict[str, Any]:
    """
    Convert the current configuration to a plain Python dictionary.

    This function retrieves the current configuration using get_config() and
    converts the Pydantic model instance to a standard Python dictionary. This is
    useful for situations where you need a serializable representation of the
    configuration, such as:

    - Sending configuration over an API
    - Logging configuration values
    - Debugging configuration state
    - Comparing configurations

    The conversion preserves the full nested structure of the configuration,
    with all Pydantic models converted to their dictionary representations.

    Returns:
        A nested dictionary containing all configuration values

    Raises:
        Any exceptions that might be raised by get_config()

    Example:
        ```python
        # Get dictionary representation of config for logging
        config_dict = get_config_as_dict()
        logger.debug(f"Current server configuration: {config_dict['server']}")

        # Use with JSON serialization
        import json
        config_json = json.dumps(get_config_as_dict())
        ```
    """
    config_obj = get_config()
    return config_obj.model_dump()
