"""Ollama provider implementation for the Ultimate MCP Server.

This module implements the Ollama provider, enabling interaction with locally running
Ollama models through a standard interface. Ollama is an open-source framework for
running LLMs locally with minimal setup.

The implementation supports:
- Text completion (generate) and chat completations
- Streaming responses
- Model listing and information retrieval
- Embeddings generation
- Cost tracking (estimated since Ollama is free to use locally)

Ollama must be installed and running locally (by default on localhost:11434)
for this provider to work properly.
"""

import asyncio
import json
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import httpx
from pydantic import BaseModel

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.constants import COST_PER_MILLION_TOKENS, Provider
from ultimate_mcp_server.core.providers.base import (
    BaseProvider,
    ModelResponse,
)
from ultimate_mcp_server.exceptions import ProviderError
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.providers.ollama")


# Define the Model class locally since it's not available in base.py
class Model(dict):
    """Model information returned by providers."""

    def __init__(self, id: str, name: str, description: str, provider: str, **kwargs):
        """Initialize a model info dictionary.

        Args:
            id: Model identifier (e.g., "llama3.2")
            name: Human-readable model name
            description: Longer description of the model
            provider: Provider name
            **kwargs: Additional model metadata
        """
        super().__init__(id=id, name=name, description=description, provider=provider, **kwargs)


# Define ProviderFeatures locally since it's not available in base.py
class ProviderFeatures:
    """Features supported by a provider."""

    def __init__(
        self,
        supports_chat_completions: bool = False,
        supports_streaming: bool = False,
        supports_function_calling: bool = False,
        supports_multiple_functions: bool = False,
        supports_embeddings: bool = False,
        supports_json_mode: bool = False,
        max_retries: int = 3,
    ):
        """Initialize provider features.

        Args:
            supports_chat_completions: Whether the provider supports chat completions
            supports_streaming: Whether the provider supports streaming responses
            supports_function_calling: Whether the provider supports function calling
            supports_multiple_functions: Whether the provider supports multiple functions
            supports_embeddings: Whether the provider supports embeddings
            supports_json_mode: Whether the provider supports JSON mode
            max_retries: Maximum number of retries for failed requests
        """
        self.supports_chat_completions = supports_chat_completions
        self.supports_streaming = supports_streaming
        self.supports_function_calling = supports_function_calling
        self.supports_multiple_functions = supports_multiple_functions
        self.supports_embeddings = supports_embeddings
        self.supports_json_mode = supports_json_mode
        self.max_retries = max_retries


# Define ProviderStatus locally since it's not available in base.py
class ProviderStatus:
    """Status information for a provider."""

    def __init__(
        self,
        name: str,
        enabled: bool = False,
        available: bool = False,
        api_key_configured: bool = False,
        features: Optional[ProviderFeatures] = None,
        default_model: Optional[str] = None,
    ):
        """Initialize provider status.

        Args:
            name: Provider name
            enabled: Whether the provider is enabled
            available: Whether the provider is available
            api_key_configured: Whether an API key is configured
            features: Provider features
            default_model: Default model for the provider
        """
        self.name = name
        self.enabled = enabled
        self.available = available
        self.api_key_configured = api_key_configured
        self.features = features
        self.default_model = default_model


class OllamaConfig(BaseModel):
    """Configuration for the Ollama provider."""

    # API endpoint (default is localhost:11434)
    api_url: str = "http://127.0.0.1:11434"

    # Default model to use if none specified
    default_model: str = "llama3.2"

    # Timeout settings
    request_timeout: int = 300

    # Whether this provider is enabled
    enabled: bool = True


class OllamaProvider(BaseProvider):
    """
    Provider implementation for Ollama.

    Ollama allows running open-source language models locally with minimal setup.
    This provider implementation connects to a locally running Ollama instance and
    provides a standard interface for generating completions and embeddings.

    Unlike cloud providers, Ollama runs models locally, so:
    - No API key is required
    - Costs are estimated (since running locally is free)
    - Model availability depends on what models have been downloaded locally

    The Ollama provider supports both chat completions and text completions,
    as well as streaming responses. It requires that the Ollama service is
    running and accessible at the configured endpoint.
    """

    provider_name = Provider.OLLAMA

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Ollama provider.

        Args:
            api_key: Not used by Ollama, included for API compatibility with other providers
            **kwargs: Additional provider-specific options
        """
        # Skip API key, it's not used by Ollama but we accept it for compatibility
        super().__init__()
        self.logger = get_logger(f"provider.{Provider.OLLAMA}")
        self.logger.info("Initializing Ollama provider...")
        self.config = self._load_config()
        self.logger.info(
            f"Loaded config: API URL={self.config.api_url}, default_model={self.config.default_model}, enabled={self.config.enabled}"
        )

        # Initialize session to None, we'll create it when needed
        self._session = None

        self.client_session_params = {
            "timeout": aiohttp.ClientTimeout(total=self.config.request_timeout)
        }

        # Unlike other providers, Ollama doesn't require an API key
        # But we'll still set this flag to True for consistency
        self._api_key_configured = True
        self._initialized = False

        # Set feature flags
        self.features = ProviderFeatures(
            supports_chat_completions=True,
            supports_streaming=True,
            supports_function_calling=False,  # Ollama doesn't support function calling natively
            supports_multiple_functions=False,
            supports_embeddings=True,
            supports_json_mode=True,  # Now supported via prompt engineering and format parameter
            max_retries=3,
        )

        # Set default costs for Ollama models (very low estimated costs)
        # Since Ollama runs locally, the actual cost is hardware usage/electricity
        # We'll use very low values for tracking purposes
        self._default_token_cost = {
            "input": 0.0001,  # $0.0001 per 1M tokens (effectively free)
            "output": 0.0001,  # $0.0001 per 1M tokens (effectively free)
        }
        self.logger.info("Ollama provider initialization completed")

    @property
    async def session(self) -> aiohttp.ClientSession:
        """Get the current session or create a new one if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(**self.client_session_params)
        return self._session

    async def __aenter__(self):
        """Enter async context, initializing the provider."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context, ensuring proper shutdown."""
        await self.shutdown()

    async def initialize(self) -> bool:
        """Initialize the provider, creating a new HTTP session.

        This method handles the initialization of the connection to Ollama.
        If Ollama isn't available (not installed or not running),
        it will gracefully report the issue without spamming errors.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Create a temporary session with a short timeout for the initial check
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as check_session:
                # Try to connect to Ollama and check if it's running
                self.logger.info(
                    f"Attempting to connect to Ollama at {self.config.api_url}/api/tags",
                    emoji_key="provider",
                )

                # First try the configured URL
                try:
                    async with check_session.get(
                        f"{self.config.api_url}/api/tags", timeout=5.0
                    ) as response:
                        if response.status == 200:
                            # Ollama is running, we'll create the main session when needed later
                            self.logger.info(
                                "Ollama service is available and running", emoji_key="provider"
                            )
                            self._initialized = True
                            return True
                        else:
                            self.logger.warning(
                                f"Ollama service responded with status {response.status}. "
                                "The service might be misconfigured.",
                                emoji_key="warning",
                            )
                except aiohttp.ClientConnectionError:
                    # Try alternate localhost format (127.0.0.1 instead of localhost or vice versa)
                    alternate_url = (
                        self.config.api_url.replace("localhost", "127.0.0.1")
                        if "localhost" in self.config.api_url
                        else self.config.api_url.replace("127.0.0.1", "localhost")
                    )
                    self.logger.info(
                        f"Connection failed, trying alternate URL: {alternate_url}",
                        emoji_key="provider",
                    )

                    try:
                        async with check_session.get(
                            f"{alternate_url}/api/tags", timeout=5.0
                        ) as response:
                            if response.status == 200:
                                # Update the config to use the working URL
                                self.logger.info(
                                    f"Connected successfully using alternate URL: {alternate_url}",
                                    emoji_key="provider",
                                )
                                self.config.api_url = alternate_url
                                self._initialized = True
                                return True
                            else:
                                self.logger.warning(
                                    f"Ollama service at alternate URL responded with status {response.status}. "
                                    "The service might be misconfigured.",
                                    emoji_key="warning",
                                )
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        self.logger.warning(
                            f"Could not connect to alternate URL: {str(e)}. "
                            "Make sure Ollama is installed and running: https://ollama.com/download",
                            emoji_key="warning",
                        )
                except aiohttp.ClientError as e:
                    # Other client errors
                    self.logger.warning(
                        f"Could not connect to Ollama service: {str(e)}. "
                        "Make sure Ollama is installed and running: https://ollama.com/download",
                        emoji_key="warning",
                    )
                except asyncio.TimeoutError:
                    # Timeout indicates Ollama is likely not responding
                    self.logger.warning(
                        "Connection to Ollama service timed out. "
                        "Make sure Ollama is installed and running: https://ollama.com/download",
                        emoji_key="warning",
                    )

            # If we got here, Ollama is not available
            self._initialized = False
            return False

        except Exception as e:
            # Catch any other exceptions to avoid spamming errors
            self.logger.error(
                f"Unexpected error initializing Ollama provider: {str(e)}", emoji_key="error"
            )
            self._initialized = False
            return False

    async def shutdown(self) -> None:
        """Shutdown the provider, closing the HTTP session."""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
        except Exception as e:
            self.logger.warning(
                f"Error closing Ollama session during shutdown: {str(e)}", emoji_key="warning"
            )
        finally:
            self._initialized = False

    def _load_config(self) -> OllamaConfig:
        """Load Ollama configuration from app configuration."""
        try:
            self.logger.info("Loading Ollama config from app configuration")
            config = get_config()
            # Print entire config for debugging
            self.logger.debug(f"Full config: {config}")

            if not hasattr(config, "providers"):
                self.logger.warning("Config doesn't have 'providers' attribute")
                return OllamaConfig()

            if not hasattr(config.providers, Provider.OLLAMA):
                self.logger.warning(f"Config doesn't have '{Provider.OLLAMA}' provider configured")
                return OllamaConfig()

            provider_config = getattr(config.providers, Provider.OLLAMA, {})
            self.logger.info(f"Found provider config: {provider_config}")

            if hasattr(provider_config, "dict"):
                self.logger.info("Provider config has 'dict' method, using it")
                return OllamaConfig(**provider_config.dict())
            else:
                self.logger.warning(
                    "Provider config doesn't have 'dict' method, attempting direct conversion"
                )
                # Try to convert to dict directly
                config_dict = {}

                # Define mapping from ProviderConfig field names to OllamaConfig field names
                field_mapping = {
                    "base_url": "api_url",  # ProviderConfig -> OllamaConfig
                    "default_model": "default_model",
                    "timeout": "request_timeout",
                    "enabled": "enabled",
                }

                # Map fields from provider_config to OllamaConfig's expected field names
                for provider_key, ollama_key in field_mapping.items():
                    if hasattr(provider_config, provider_key):
                        config_dict[ollama_key] = getattr(provider_config, provider_key)
                        self.logger.info(
                            f"Mapped {provider_key} to {ollama_key}: {getattr(provider_config, provider_key)}"
                        )

                self.logger.info(f"Created config dict: {config_dict}")
                return OllamaConfig(**config_dict)
        except Exception as e:
            self.logger.error(f"Error loading Ollama config: {e}", exc_info=True)
            return OllamaConfig()

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return self.config.default_model

    def get_status(self) -> ProviderStatus:
        """Get the current status of this provider."""
        return ProviderStatus(
            name=self.provider_name,
            enabled=self.config.enabled,
            available=self._initialized,
            api_key_configured=self._api_key_configured,
            features=self.features,
            default_model=self.get_default_model(),
        )

    async def check_api_key(self) -> bool:
        """
        Check if the Ollama service is accessible.

        Since Ollama doesn't use API keys, this just checks if the service is running.
        This check is designed to fail gracefully if Ollama is not installed or running,
        without causing cascading errors in the system.

        Returns:
            bool: True if Ollama service is running and accessible, False otherwise
        """
        if not self._initialized:
            try:
                # Attempt to initialize with a short timeout
                return await self.initialize()
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize Ollama during service check: {str(e)}",
                    emoji_key="warning",
                )
                return False

        try:
            # Use a dedicated session with short timeout for health check
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3.0)) as session:
                try:
                    async with session.get(f"{self.config.api_url}/api/tags") as response:
                        return response.status == 200
                except (aiohttp.ClientConnectionError, asyncio.TimeoutError, Exception) as e:
                    self.logger.warning(
                        f"Ollama service check failed: {str(e)}", emoji_key="warning"
                    )
                    return False
        except Exception as e:
            self.logger.warning(
                f"Failed to create session for Ollama check: {str(e)}", emoji_key="warning"
            )
            return False

    def _build_api_url(self, endpoint: str) -> str:
        """Build the full API URL for a given endpoint."""
        return f"{self.config.api_url}/api/{endpoint}"

    def _estimate_token_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost of a completion based on token counts.

        Since Ollama runs locally, the costs are just estimates and very low.
        """
        # Try to get model-specific costs if available
        model_costs = COST_PER_MILLION_TOKENS.get(model, self._default_token_cost)

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * model_costs.get(
            "input", self._default_token_cost["input"]
        )
        output_cost = (output_tokens / 1_000_000) * model_costs.get(
            "output", self._default_token_cost["output"]
        )

        return input_cost + output_cost

    async def list_models(self) -> List[Model]:
        """
        List all available models from Ollama.

        This method attempts to list all locally available Ollama models.
        If Ollama is not available or cannot be reached, it will return
        an empty list instead of raising an exception.

        Returns:
            List of available Ollama models, or empty list if Ollama is not available
        """
        if not self._initialized:
            try:
                initialized = await self.initialize()
                if not initialized:
                    self.logger.warning(
                        "Cannot list Ollama models because the service is not available",
                        emoji_key="warning",
                    )
                    return []
            except Exception:
                self.logger.warning(
                    "Cannot list Ollama models because initialization failed", emoji_key="warning"
                )
                return []

        try:
            # Create a dedicated session for this operation to avoid shared session issues
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session:
                return await self._fetch_models(session)
        except Exception as e:
            self.logger.warning(
                f"Error listing Ollama models: {str(e)}. The service may not be running.",
                emoji_key="warning",
            )
            return []

    async def _fetch_models(self, session: aiohttp.ClientSession) -> List[Model]:
        """Fetch models using the provided session."""
        try:
            async with session.get(self._build_api_url("tags")) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to list Ollama models: {response.status}")
                    return []

                data = await response.json()
                models = []

                # Process the response
                for model_info in data.get("models", []):
                    model_id = model_info.get("name", "")

                    # Extract additional info if available
                    description = f"Ollama model: {model_id}"
                    model_size = model_info.get("size", 0)
                    size_gb = None

                    if model_size:
                        # Convert to GB for readability if size is provided in bytes
                        size_gb = model_size / (1024 * 1024 * 1024)
                        description += f" ({size_gb:.2f} GB)"

                    models.append(
                        Model(
                            id=model_id,
                            name=model_id,
                            description=description,
                            provider=self.provider_name,
                            size=f"{size_gb:.2f} GB" if size_gb else "Unknown",
                        )
                    )

                return models
        except aiohttp.ClientConnectionError:
            self.logger.warning(
                "Connection refused while listing Ollama models", emoji_key="warning"
            )
            return []
        except asyncio.TimeoutError:
            self.logger.warning("Timeout while listing Ollama models", emoji_key="warning")
            return []
        except Exception as e:
            self.logger.warning(f"Error fetching Ollama models: {str(e)}", emoji_key="warning")
            return []

    async def generate_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        mirostat: Optional[int] = None,
        mirostat_tau: Optional[float] = None,
        mirostat_eta: Optional[float] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> ModelResponse:
        """Generate a completion from Ollama.

        Args:
            prompt: Text prompt to send to Ollama (optional if messages provided)
            messages: List of message dictionaries (optional if prompt provided)
            model: Ollama model name (e.g., "llama2:13b")
            max_tokens: Maximum tokens to generate
            temperature: Controls randomness (0.0-1.0)
            stop: List of strings that stop generation when encountered
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            mirostat: Mirostat sampling algorithm (0, 1, or 2)
            mirostat_tau: Target entropy for mirostat
            mirostat_eta: Learning rate for mirostat
            json_mode: Request JSON-formatted response
            **kwargs: Additional parameters

        Returns:
            ModelResponse object with completion result
        """
        if not self.config.api_url:
            raise ValueError("Ollama API URL not configured")

        # Verify we have either prompt or messages
        if prompt is None and not messages:
            raise ValueError("Either prompt or messages must be provided to generate a completion")

        # If model is None, use configured default
        model = model or self.get_default_model()

        # Only strip provider prefix if it's our provider name, keep organization prefixes
        if "/" in model and model.startswith(f"{self.provider_name}/"):
            model = model.split("/", 1)[1]

        # If JSON mode is enabled, use the streaming implementation internally
        # since Ollama's non-streaming JSON mode is inconsistent
        if json_mode:
            self.logger.debug(
                "JSON mode requested, using streaming implementation internally for reliability"
            )
            return await self._generate_completion_via_streaming(
                prompt=prompt,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                mirostat=mirostat,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                json_mode=True,  # Ensure json_mode is passed through
                **kwargs,
            )

        # Log request start
        self.logger.info(
            f"Generating Ollama completion (generate) with model {model}",
            emoji_key=self.provider_name,
        )

        # Convert messages to prompt if messages provided
        using_messages = False
        if messages and not prompt:
            using_messages = True
            # Convert messages to Ollama's chat format
            chat_params = {"messages": []}

            # Process messages into Ollama format
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")

                # Map roles to Ollama's expected format
                if role == "system":
                    ollama_role = "system"
                elif role == "user":
                    ollama_role = "user"
                elif role == "assistant":
                    ollama_role = "assistant"
                else:
                    # Default unknown roles to user
                    self.logger.warning(f"Unknown message role '{role}', treating as 'user'")
                    ollama_role = "user"

                chat_params["messages"].append({"role": ollama_role, "content": content})

            # Add model and parameters to chat_params
            chat_params["model"] = model

            # Add optional parameters if provided
            if temperature is not None and temperature != 0.7:
                chat_params["options"] = chat_params.get("options", {})
                chat_params["options"]["temperature"] = temperature

            if max_tokens is not None:
                chat_params["options"] = chat_params.get("options", {})
                chat_params["options"]["num_predict"] = max_tokens

            if stop:
                chat_params["options"] = chat_params.get("options", {})
                chat_params["options"]["stop"] = stop

            # Add other parameters if provided
            for param_name, param_value in [
                ("top_p", top_p),
                ("top_k", top_k),
                ("frequency_penalty", frequency_penalty),
                ("presence_penalty", presence_penalty),
                ("mirostat", mirostat),
                ("mirostat_tau", mirostat_tau),
                ("mirostat_eta", mirostat_eta),
            ]:
                if param_value is not None:
                    chat_params["options"] = chat_params.get("options", {})
                    chat_params["options"][param_name] = param_value

            # Add json_mode if requested (as format option)
            if json_mode:
                chat_params["options"] = chat_params.get("options", {})
                chat_params["options"]["format"] = "json"

                # For Ollama non-streaming completions, we need to force the system message
                # because the format param alone isn't reliable
                kwargs["add_json_instructions"] = True

                # Only add system message instruction as a fallback if explicitly requested
                add_json_instructions = kwargs.pop("add_json_instructions", False)

                # Add system message for json_mode only if requested
                if add_json_instructions:
                    has_system = any(msg.get("role", "").lower() == "system" for msg in messages)
                    if not has_system:
                        # Add JSON instruction as a system message
                        chat_params["messages"].insert(
                            0,
                            {
                                "role": "system",
                                "content": "You must respond with valid JSON. Format your entire response as a JSON object with properly quoted keys and values.",
                            },
                        )
                        self.logger.debug("Added JSON system instructions for chat_params")

            # Add any additional kwargs as options
            if kwargs:
                chat_params["options"] = chat_params.get("options", {})
                chat_params["options"].update(kwargs)

            # Use chat endpoint
            api_endpoint = self._build_api_url("chat")
            response_type = "chat"
        else:
            # Using generate endpoint with prompt
            # Prepare generate parameters
            generate_params = {"model": model, "prompt": prompt}

            # Add optional parameters if provided
            if temperature is not None and temperature != 0.7:
                generate_params["options"] = generate_params.get("options", {})
                generate_params["options"]["temperature"] = temperature

            if max_tokens is not None:
                generate_params["options"] = generate_params.get("options", {})
                generate_params["options"]["num_predict"] = max_tokens

            if stop:
                generate_params["options"] = generate_params.get("options", {})
                generate_params["options"]["stop"] = stop

            # Add other parameters if provided
            for param_name, param_value in [
                ("top_p", top_p),
                ("top_k", top_k),
                ("frequency_penalty", frequency_penalty),
                ("presence_penalty", presence_penalty),
                ("mirostat", mirostat),
                ("mirostat_tau", mirostat_tau),
                ("mirostat_eta", mirostat_eta),
            ]:
                if param_value is not None:
                    generate_params["options"] = generate_params.get("options", {})
                    generate_params["options"][param_name] = param_value

            # Add json_mode if requested (as format option)
            if json_mode:
                generate_params["options"] = generate_params.get("options", {})
                generate_params["options"]["format"] = "json"

                # For Ollama non-streaming completions, we need to force the JSON instructions
                # because the format param alone isn't reliable
                kwargs["add_json_instructions"] = True

                # Only enhance prompt with JSON instructions if explicitly requested
                add_json_instructions = kwargs.pop("add_json_instructions", False)
                if add_json_instructions:
                    # Enhance prompt with JSON instructions for better compliance
                    generate_params["prompt"] = (
                        f"Please respond with valid JSON only. {prompt}\nEnsure your entire response is a valid, parseable JSON object with properly quoted keys and values."
                    )
                    self.logger.debug("Enhanced prompt with JSON instructions for generate_params")

            # Add any additional kwargs as options
            if kwargs:
                generate_params["options"] = generate_params.get("options", {})
                generate_params["options"].update(kwargs)

            # Use generate endpoint
            api_endpoint = self._build_api_url("generate")
            response_type = "generate"  # noqa: F841

        # Start timer for tracking
        start_time = time.time()

        try:
            # Make HTTP request to Ollama
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                if using_messages:
                    # Using chat endpoint
                    response = await client.post(api_endpoint, json=chat_params)
                else:
                    # Using generate endpoint
                    response = await client.post(api_endpoint, json=generate_params)

                # Check for HTTP errors
                response.raise_for_status()

                # Parse response - handle multi-line JSON data which can happen with json_mode
                try:
                    # First try regular JSON parsing
                    result = response.json()
                except json.JSONDecodeError as e:
                    # If that fails, try parsing line by line and concatenate responses
                    self.logger.debug(
                        "Response contains multiple JSON objects, parsing line by line"
                    )
                    content = response.text
                    lines = content.strip().split("\n")

                    # If we have multiple JSON objects
                    if len(lines) > 1:
                        # For multiple objects, take the last one which should have the final response
                        # This happens in some Ollama versions when using format=json
                        try:
                            result = json.loads(
                                lines[-1]
                            )  # Use the last line, which typically has the complete response

                            # Verify result has response/message field, if not try the first line
                            if using_messages and "message" not in result:
                                result = json.loads(lines[0])
                            elif not using_messages and "response" not in result:
                                result = json.loads(lines[0])

                        except json.JSONDecodeError as e:
                            raise RuntimeError(
                                f"Failed to parse Ollama JSON response: {str(e)}. Response: {content[:200]}..."
                            ) from e
                    else:
                        # If we only have one line but still got a JSON error
                        raise RuntimeError(
                            f"Invalid JSON in Ollama response: {content[:200]}..."
                        ) from e

                # Calculate processing time
                processing_time = time.time() - start_time

                # Extract response text based on endpoint
                if using_messages:
                    # Extract from chat endpoint
                    completion_text = result.get("message", {}).get("content", "")
                else:
                    # Extract from generate endpoint
                    completion_text = result.get("response", "")

                # Log the raw response for debugging
                self.logger.debug(f"Raw Ollama response: {result}")
                self.logger.debug(f"Extracted completion text: {completion_text[:500]}...")

                # For JSON mode, ensure the completion text is properly formatted JSON
                if json_mode and completion_text:
                    # Always use add_json_instructions for this model since it seems to need it
                    if "gemma" in model.lower():
                        # Force adding instructions for gemma models specifically
                        kwargs["add_json_instructions"] = True

                    try:
                        # First try to extract JSON using our comprehensive method
                        extracted_json = self._extract_json_from_text(completion_text)
                        self.logger.debug(f"Extracted JSON: {extracted_json[:500]}...")

                        # If we found valid JSON, parse and format it
                        json_data = json.loads(extracted_json)

                        # If successful, format it nicely with indentation
                        if isinstance(json_data, (dict, list)):
                            completion_text = json.dumps(json_data, indent=2)
                            self.logger.debug("Successfully parsed and formatted JSON response")
                        else:
                            self.logger.warning(
                                f"JSON response is not a dict or list: {type(json_data)}"
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.warning(
                            f"Failed to extract valid JSON from response: {str(e)[:100]}..."
                        )

                # Calculate token usage
                prompt_tokens = result.get("prompt_eval_count", 0)
                completion_tokens = result.get("eval_count", 0)

                # Format the standardized response
                model_response = ModelResponse(
                    text=completion_text,
                    model=f"{self.provider_name}/{model}",
                    provider=self.provider_name,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    processing_time=processing_time,
                    raw_response=result,
                )

                # Add message field for chat_completion compatibility
                model_response.message = {"role": "assistant", "content": completion_text}

                # Ensure there's always a value returned for JSON mode to prevent empty displays
                if json_mode and (not completion_text or not completion_text.strip()):
                    # If we got an empty response, create a default one
                    default_json = {
                        "response": "No content was returned by the model",
                        "error": "Empty response with json_mode enabled",
                    }
                    completion_text = json.dumps(default_json, indent=2)
                    model_response.text = completion_text
                    model_response.message["content"] = completion_text
                    self.logger.warning(
                        "Empty response with JSON mode, returning default JSON structure"
                    )

                # Log success
                self.logger.success(
                    f"Ollama completion successful with model {model}",
                    emoji_key="completion_success",
                    tokens={"input": prompt_tokens, "output": completion_tokens},
                    time=processing_time,
                    model=model,
                )

                return model_response

        except httpx.HTTPStatusError as http_err:
            # Handle HTTP errors
            processing_time = time.time() - start_time
            try:
                error_json = http_err.response.json()
                error_msg = error_json.get("error", str(http_err))
            except (json.JSONDecodeError, KeyError):
                error_msg = (
                    f"HTTP error: {http_err.response.status_code} - {http_err.response.text}"
                )

            self.logger.error(
                f"Ollama API error: {error_msg}",
                emoji_key="error",
                status_code=http_err.response.status_code,
                model=model,
            )

            raise ConnectionError(f"Ollama API error: {error_msg}") from http_err

        except httpx.RequestError as req_err:
            # Handle request errors (e.g., connection issues)
            processing_time = time.time() - start_time
            error_msg = f"Request error: {str(req_err)}"

            self.logger.error(f"Ollama request error: {error_msg}", emoji_key="error", model=model)

            raise ConnectionError(f"Ollama request error: {error_msg}") from req_err

        except Exception as e:
            # Handle other unexpected errors
            processing_time = time.time() - start_time

            self.logger.error(
                f"Unexpected error calling Ollama: {str(e)}",
                emoji_key="error",
                model=model,
                exc_info=True,
            )

            raise RuntimeError(f"Unexpected error calling Ollama: {str(e)}") from e

    async def generate_completion_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        # This is the main try block for the whole function - needs exception handling
        try:
            # Verify we have either prompt or messages
            if prompt is None and not messages:
                raise ValueError(
                    "Either prompt or messages must be provided to generate a streaming completion"
                )

            # Check if provider is initialized before attempting to generate
            if not self._initialized:
                try:
                    initialized = await self.initialize()
                    if not initialized:
                        # Yield an error message and immediately terminate
                        error_metadata = {
                            "model": f"{self.provider_name}/{model or self.get_default_model()}",
                            "provider": self.provider_name,
                            "error": "Ollama service is not available. Make sure Ollama is installed and running: https://ollama.com/download",
                            "finish_reason": "error",
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "processing_time": 0.0,
                        }
                        yield "", error_metadata
                        return
                except Exception as e:
                    # Yield an error message and immediately terminate
                    error_metadata = {
                        "model": f"{self.provider_name}/{model or self.get_default_model()}",
                        "provider": self.provider_name,
                        "error": f"Failed to initialize Ollama provider: {str(e)}. Make sure Ollama is installed and running: https://ollama.com/download",
                        "finish_reason": "error",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "processing_time": 0.0,
                    }
                    yield "", error_metadata
                    return

            # Use default model if none specified
            model_id = model or self.get_default_model()

            # Only remove our provider prefix if present, keep organization prefixes
            if "/" in model_id and model_id.startswith(f"{self.provider_name}/"):
                model_id = model_id.split("/", 1)[1]

            # Check for json_mode flag and remove it from kwargs
            json_mode = kwargs.pop("json_mode", False)
            format_param = None

            if json_mode:
                # Ollama supports structured output via 'format' parameter at the ROOT level
                # This can be either "json" for basic JSON mode or a JSON schema for structured output
                format_param = "json"  # Use simple "json" string for basic JSON mode
                self.logger.debug("Setting format='json' for Ollama streaming")

                # Note: Format parameter may be less reliable with streaming
                # due to how content is chunked, but Ollama should handle this.

            # Flag to track if we're using messages format
            using_messages = False

            # Prepare the payload based on input type (messages or prompt)
            if messages:
                using_messages = True  # noqa: F841
                # Convert messages to Ollama's expected format
                ollama_messages = []

                # Process messages
                for msg in messages:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")

                    # Map roles to Ollama's expected format
                    if role == "system":
                        ollama_role = "system"
                    elif role == "user":
                        ollama_role = "user"
                    elif role == "assistant":
                        ollama_role = "assistant"
                    else:
                        # Default unknown roles to user
                        self.logger.warning(f"Unknown message role '{role}', treating as 'user'")
                        ollama_role = "user"

                    ollama_messages.append({"role": ollama_role, "content": content})

                # Build chat payload
                payload = {
                    "model": model_id,
                    "messages": ollama_messages,
                    "stream": True,
                    "options": {  # Ollama options go inside an 'options' dict
                        "temperature": temperature,
                    },
                }

                # Use chat endpoint
                api_endpoint = "chat"

            elif system is not None or model_id.startswith(
                ("llama", "gpt", "claude", "phi", "mistral")
            ):
                # Use chat endpoint with system message (if provided) and prompt
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": model_id,
                    "messages": messages,
                    "stream": True,
                    "options": {  # Ollama options go inside an 'options' dict
                        "temperature": temperature,
                    },
                }

                # Use chat endpoint
                api_endpoint = "chat"

            else:
                # Use generate endpoint with prompt
                payload = {
                    "model": model_id,
                    "prompt": prompt,
                    "stream": True,
                    "options": {  # Ollama options go inside an 'options' dict
                        "temperature": temperature,
                    },
                }

                # Use generate endpoint
                api_endpoint = "generate"

            # Add common optional parameters
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            if stop:
                payload["options"]["stop"] = stop

            # Add format parameter at the root level if JSON mode is enabled
            if format_param:
                payload["format"] = format_param

            # Add any additional supported parameters from kwargs into options
            for key, value in kwargs.items():
                if key in ["seed", "top_k", "top_p", "num_ctx"]:
                    payload["options"][key] = value

            # Log request including JSON mode status
            content_length = 0
            if messages:
                content_length = sum(len(m.get("content", "")) for m in messages)
            elif prompt:
                content_length = len(prompt)

            self.logger.info(
                f"Generating Ollama streaming completion ({api_endpoint}) with model {model_id}",
                emoji_key=self.provider_name,
                prompt_length=content_length,
                json_mode_requested=json_mode,
            )

            start_time = time.time()
            input_tokens = 0
            output_tokens = 0
            finish_reason = None
            final_error = None

            async with aiohttp.ClientSession(**self.client_session_params) as streaming_session:
                async with streaming_session.post(
                    self._build_api_url(api_endpoint), json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        final_error = (
                            f"Ollama streaming API error: {response.status} - {error_text}"
                        )
                        # Yield error and stop
                        yield (
                            "",
                            {
                                "error": final_error,
                                "finished": True,
                                "provider": self.provider_name,
                                "model": model_id,
                            },
                        )
                        return

                    buffer = ""
                    chunk_index = 0
                    async for line in response.content:
                        if not line.strip():
                            continue
                        buffer += line.decode("utf-8")

                        # Process complete JSON objects in the buffer
                        while "\n" in buffer:
                            json_str, buffer = buffer.split("\n", 1)
                            if not json_str.strip():
                                continue

                            try:
                                data = json.loads(json_str)
                                chunk_index += 1

                                # Extract content based on endpoint
                                if api_endpoint == "chat":
                                    text_chunk = data.get("message", {}).get("content", "")
                                else:  # generate endpoint
                                    text_chunk = data.get("response", "")

                                # Check if this is the final summary chunk
                                if data.get("done", False):
                                    input_tokens = data.get("prompt_eval_count", input_tokens)
                                    output_tokens = data.get("eval_count", output_tokens)
                                    finish_reason = data.get(
                                        "done_reason", "stop"
                                    )  # Get finish reason if available
                                    # Yield the final text chunk if any, then break to yield summary
                                    if text_chunk:
                                        metadata = {
                                            "provider": self.provider_name,
                                            "model": model_id,
                                            "chunk_index": chunk_index,
                                            "finished": False,
                                        }
                                        yield text_chunk, metadata
                                    break  # Exit inner loop after processing final chunk

                                # Yield regular chunk
                                if text_chunk:
                                    metadata = {
                                        "provider": self.provider_name,
                                        "model": model_id,
                                        "chunk_index": chunk_index,
                                        "finished": False,
                                    }
                                    yield text_chunk, metadata

                            except json.JSONDecodeError:
                                self.logger.warning(
                                    f"Could not decode JSON line: {json_str[:100]}..."
                                )
                                # Continue, maybe it's part of a larger object split across lines
                            except Exception as parse_error:
                                self.logger.warning(f"Error processing stream chunk: {parse_error}")
                                final_error = f"Error processing stream: {parse_error}"
                                break  # Stop processing on unexpected error

                        if final_error:
                            break  # Exit outer loop if error occurred

            # --- Final Chunk ---
            processing_time = time.time() - start_time
            total_tokens = input_tokens + output_tokens
            cost = self._estimate_token_cost(model_id, input_tokens, output_tokens)

            final_metadata = {
                "model": f"{self.provider_name}/{model_id}",
                "provider": self.provider_name,
                "finished": True,
                "finish_reason": finish_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "processing_time": processing_time,
                "error": final_error,
            }
            yield "", final_metadata  # Yield empty chunk with final stats

        except aiohttp.ClientConnectionError as e:
            # Yield connection error
            yield (
                "",
                {
                    "error": f"Connection to Ollama failed: {str(e)}",
                    "finished": True,
                    "provider": self.provider_name,
                    "model": model_id,
                },
            )
        except asyncio.TimeoutError:
            # Yield timeout error
            yield (
                "",
                {
                    "error": "Connection to Ollama timed out",
                    "finished": True,
                    "provider": self.provider_name,
                    "model": model_id,
                },
            )
        except Exception as e:
            # Yield generic error
            if isinstance(e, ProviderError):
                raise
            yield (
                "",
                {
                    "error": f"Error generating streaming completion: {str(e)}",
                    "finished": True,
                    "provider": self.provider_name,
                    "model": model_id,
                },
            )

    async def create_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate embeddings for a list of texts using the Ollama API.

        Args:
            texts: List of texts to generate embeddings for.
            model: The model ID to use (defaults to provider's default).
            **kwargs: Additional parameters to pass to the API.

        Returns:
            An ModelResponse object with the embeddings and metadata.
            If Ollama is not available, returns an error in the metadata.
        """
        # Check if provider is initialized before attempting to generate
        if not self._initialized:
            try:
                initialized = await self.initialize()
                if not initialized:
                    # Return a clear error without raising an exception
                    return ModelResponse(
                        text="",
                        model=f"{self.provider_name}/{model or self.get_default_model()}",
                        provider=self.provider_name,
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        processing_time=0.0,
                        metadata={
                            "error": "Ollama service is not available. Make sure Ollama is installed and running: https://ollama.com/download",
                            "embeddings": [],
                        },
                    )
            except Exception as e:
                # Return a clear error without raising an exception
                return ModelResponse(
                    text="",
                    model=f"{self.provider_name}/{model or self.get_default_model()}",
                    provider=self.provider_name,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    processing_time=0.0,
                    metadata={
                        "error": f"Failed to initialize Ollama provider: {str(e)}. Make sure Ollama is installed and running: https://ollama.com/download",
                        "embeddings": [],
                    },
                )

        # Use default model if none specified
        model_id = model or self.get_default_model()

        # Only remove our provider prefix if present, keep organization prefixes
        if "/" in model_id and model_id.startswith(f"{self.provider_name}/"):
            model_id = model_id.split("/", 1)[1]

        # Get total number of tokens in all texts
        # This is an estimation since Ollama doesn't provide token counts for embeddings
        total_tokens = sum(len(text.split()) for text in texts)

        # Prepare the result
        result_embeddings = []
        errors = []
        all_dimensions = None

        try:
            start_time = time.time()

            # Create a dedicated session for this embeddings request
            async with aiohttp.ClientSession(**self.client_session_params) as session:
                # Process each text individually (Ollama supports batching but we'll use same pattern as other providers)
                for text in texts:
                    payload = {
                        "model": model_id,
                        "prompt": text,
                    }

                    # Add any additional parameters
                    for key, value in kwargs.items():
                        if key not in payload and value is not None:
                            payload[key] = value

                    try:
                        async with session.post(
                            self._build_api_url("embeddings"), json=payload, timeout=30.0
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                errors.append(f"Ollama API error: {response.status} - {error_text}")
                                # Continue with the next text
                                continue

                            data = await response.json()

                            # Extract embeddings
                            embedding = data.get("embedding", [])

                            if not embedding:
                                errors.append(f"No embedding returned for text: {text[:50]}...")
                                continue

                            # Store the embedding
                            result_embeddings.append(embedding)

                            # Check dimensions for consistency
                            dimensions = len(embedding)
                            if all_dimensions is None:
                                all_dimensions = dimensions
                            elif dimensions != all_dimensions:
                                errors.append(
                                    f"Inconsistent embedding dimensions: got {dimensions}, expected {all_dimensions}"
                                )
                    except aiohttp.ClientConnectionError as e:
                        errors.append(
                            f"Connection to Ollama failed: {str(e)}. Make sure Ollama is running and accessible."
                        )
                        break
                    except asyncio.TimeoutError:
                        errors.append(
                            "Connection to Ollama timed out. Check if the service is overloaded."
                        )
                        break
                    except Exception as e:
                        errors.append(f"Error generating embedding: {str(e)}")
                        continue

            # Calculate processing time
            processing_time = time.time() - start_time

            # Calculate cost (estimated)
            estimated_cost = (total_tokens / 1_000_000) * 0.0001  # Very low cost estimation

            # Create response model with embeddings in metadata
            return ModelResponse(
                text="",  # Embeddings don't have text content
                model=f"{self.provider_name}/{model_id}",
                provider=self.provider_name,
                input_tokens=total_tokens,  # Use total tokens as input tokens for embeddings
                output_tokens=0,  # No output tokens for embeddings
                total_tokens=total_tokens,
                processing_time=processing_time,
                metadata={
                    "embeddings": result_embeddings,
                    "dimensions": all_dimensions or 0,
                    "errors": errors if errors else None,
                    "cost": estimated_cost,
                },
            )

        except aiohttp.ClientConnectionError as e:
            # Return a clear error without raising an exception
            return ModelResponse(
                text="",
                model=f"{self.provider_name}/{model_id}",
                provider=self.provider_name,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                processing_time=0.0,
                metadata={
                    "error": f"Connection to Ollama failed: {str(e)}. Make sure Ollama is running and accessible.",
                    "embeddings": [],
                    "cost": 0.0,
                },
            )
        except Exception as e:
            # Return a clear error without raising an exception
            if isinstance(e, ProviderError):
                raise
            return ModelResponse(
                text="",
                model=f"{self.provider_name}/{model_id}",
                provider=self.provider_name,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                processing_time=0.0,
                metadata={
                    "error": f"Error generating embeddings: {str(e)}",
                    "embeddings": result_embeddings,
                    "cost": 0.0,
                },
            )

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON content from text that might include markdown code blocks or explanatory text.

        Args:
            text: The raw text response that might contain JSON

        Returns:
            Cleaned JSON content
        """

        # First check if the text is already valid JSON
        try:
            json.loads(text)
            return text  # Already valid JSON
        except json.JSONDecodeError:
            pass  # Continue with extraction

        # Extract JSON from code blocks - common pattern
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block_match:
            code_content = code_block_match.group(1).strip()
            try:
                json.loads(code_content)
                return code_content
            except json.JSONDecodeError:
                # Try to fix common JSON syntax issues like trailing commas
                fixed_content = re.sub(r",\s*([}\]])", r"\1", code_content)
                try:
                    json.loads(fixed_content)
                    return fixed_content
                except json.JSONDecodeError:
                    pass  # Continue with other extraction methods

        # Look for JSON array or object patterns in the content
        # Find the first [ or { and the matching closing ] or }
        stripped = text.strip()

        # Try to extract array
        if "[" in stripped and "]" in stripped:
            start = stripped.find("[")
            # Find the matching closing bracket
            end = -1
            depth = 0
            for i in range(start, len(stripped)):
                if stripped[i] == "[":
                    depth += 1
                elif stripped[i] == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end > start:
                array_content = stripped[start:end]
                try:
                    json.loads(array_content)
                    return array_content
                except json.JSONDecodeError:
                    pass  # Try other methods

        # Try to extract object
        if "{" in stripped and "}" in stripped:
            start = stripped.find("{")
            # Find the matching closing bracket
            end = -1
            depth = 0
            for i in range(start, len(stripped)):
                if stripped[i] == "{":
                    depth += 1
                elif stripped[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end > start:
                object_content = stripped[start:end]
                try:
                    json.loads(object_content)
                    return object_content
                except json.JSONDecodeError:
                    pass  # Try other methods

        # If all else fails, return the original text
        return text

    async def _generate_completion_via_streaming(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        mirostat: Optional[int] = None,
        mirostat_tau: Optional[float] = None,
        mirostat_eta: Optional[float] = None,
        system: Optional[str] = None,
        json_mode: bool = False,  # Add json_mode parameter to pass it through to streaming method
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a completion via streaming and collect the results.

        This is a workaround for Ollama's inconsistent behavior with JSON mode
        in non-streaming completions. It uses the streaming API which works reliably
        with JSON mode, and collects all chunks into a single result.

        Args:
            Same as generate_completion and generate_completion_stream

        Returns:
            ModelResponse: The complete response
        """
        self.logger.debug("Using streaming method internally to handle JSON mode reliably")

        # Start the streaming generator
        stream_gen = self.generate_completion_stream(
            prompt=prompt,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            mirostat=mirostat,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            system=system,
            json_mode=json_mode,
            **kwargs,
        )

        # Collect all text chunks
        combined_text = ""
        metadata = {}
        input_tokens = 0
        output_tokens = 0
        processing_time = 0

        try:
            async for chunk, chunk_metadata in stream_gen:
                if chunk_metadata.get("error"):
                    # If there's an error, raise it
                    raise RuntimeError(chunk_metadata["error"])

                # Add current chunk to result
                combined_text += chunk

                # If this is the final chunk with stats, save the metadata
                if chunk_metadata.get("finished", False):
                    metadata = chunk_metadata
                    input_tokens = chunk_metadata.get("input_tokens", 0)
                    output_tokens = chunk_metadata.get("output_tokens", 0)
                    processing_time = chunk_metadata.get("processing_time", 0)
        except Exception as e:
            # If streaming fails, re-raise the exception
            raise RuntimeError(f"Error in streaming completion: {str(e)}") from e

        # Create a ModelResponse with the combined text
        result = ModelResponse(
            text=combined_text,
            model=metadata.get(
                "model", f"{self.provider_name}/{model or self.get_default_model()}"
            ),
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            processing_time=processing_time,
            raw_response={"streaming_source": True, "metadata": metadata},
        )

        # Add message field for chat_completion compatibility
        result.message = {"role": "assistant", "content": combined_text}

        return result
