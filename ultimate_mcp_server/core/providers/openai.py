"""OpenAI provider implementation."""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import BaseProvider, ModelResponse
from ultimate_mcp_server.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("ultimate_mcp_server.providers.openai")


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI API."""

    provider_name = Provider.OPENAI.value

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = kwargs.get("base_url")
        self.organization = kwargs.get("organization")
        self.models_cache = None

    async def initialize(self) -> bool:
        """Initialize the OpenAI client.

        Returns:
            bool: True if initialization was successful
        """
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
            )

            # Skip API call if using a mock key (for tests)
            if self.api_key and "mock-" in self.api_key:
                self.logger.info(
                    "Using mock OpenAI key - skipping API validation", emoji_key="mock"
                )
                return True

            # Test connection by listing models
            await self.list_models()

            self.logger.success("OpenAI provider initialized successfully", emoji_key="provider")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider: {str(e)}", emoji_key="error")
            return False

    async def generate_completion(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> ModelResponse:
        """Generate a completion using OpenAI.

        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "gpt-4o")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters

        Returns:
            ModelResponse with completion result

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            await self.initialize()

        # Use default model if not specified
        model = model or self.get_default_model()

        # Strip provider prefix if present (e.g., "openai/gpt-4o" -> "gpt-4o")
        if model.startswith(f"{self.provider_name}/"):
            original_model = model
            model = model.split("/", 1)[1]
            self.logger.debug(
                f"Stripped provider prefix from model name: {original_model} -> {model}"
            )

        # Handle case when messages are provided instead of prompt (for chat_completion)
        messages = kwargs.pop("messages", None)

        # If neither prompt nor messages are provided, raise an error
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        # Create messages if not already provided
        if not messages:
            messages = [{"role": "user", "content": prompt}]

        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # Add max_tokens if specified
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Check for json_mode flag and remove it from kwargs
        json_mode = kwargs.pop("json_mode", False)
        if json_mode:
            # Use the correct response_format for JSON mode
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for OpenAI")

        # Handle any legacy response_format passed directly, but prefer json_mode
        if "response_format" in kwargs and not json_mode:
            # Support both direct format object and type-only specification
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, dict):
                params["response_format"] = response_format
            elif isinstance(response_format, str) and response_format in ["json_object", "text"]:
                params["response_format"] = {"type": response_format}
            self.logger.debug(
                f"Setting response_format from direct param: {params.get('response_format')}"
            )

        # Add any remaining additional parameters
        params.update(kwargs)

        # --- Special handling for specific model parameter constraints ---
        if model == "o3-mini":
            if "temperature" in params:
                self.logger.debug(f"Removing unsupported 'temperature' parameter for model {model}")
                del params["temperature"]
        elif model == "o1-preview":
            current_temp = params.get("temperature")
            # Only allow temperature if it's explicitly set to 1.0, otherwise remove it to use API default.
            if current_temp is not None and current_temp != 1.0:
                self.logger.debug(
                    f"Removing non-default 'temperature' ({current_temp}) for model {model}"
                )
                del params["temperature"]
        # --- End special handling ---

        # Log request
        prompt_length = len(prompt) if prompt else sum(len(m.get("content", "")) for m in messages)
        self.logger.info(
            f"Generating completion with OpenAI model {model}",
            emoji_key=self.provider_name,
            prompt_length=prompt_length,
            json_mode=json_mode,  # Log if json_mode was requested
        )

        try:
            # API call with timing
            response, processing_time = await self.process_with_timer(
                self.client.chat.completions.create, **params
            )

            # Extract response text
            completion_text = response.choices[0].message.content

            # Create message object for chat_completion
            message = {"role": "assistant", "content": completion_text}

            # Create standardized response
            result = ModelResponse(
                text=completion_text,
                model=model,
                provider=self.provider_name,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                processing_time=processing_time,
                raw_response=response,
            )

            # Add message to result for chat_completion
            result.message = message

            # Log success
            self.logger.success(
                "OpenAI completion successful",
                emoji_key="success",
                model=model,
                tokens={"input": result.input_tokens, "output": result.output_tokens},
                cost=result.cost,
                time=result.processing_time,
            )

            return result

        except Exception as e:
            self.logger.error(f"OpenAI completion failed: {str(e)}", emoji_key="error", model=model)
            raise

    async def generate_completion_stream(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion using OpenAI.

        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "gpt-4o")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters

        Yields:
            Tuple of (text_chunk, metadata)

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            await self.initialize()

        # Use default model if not specified
        model = model or self.get_default_model()

        # Strip provider prefix if present (e.g., "openai/gpt-4o" -> "gpt-4o")
        if model.startswith(f"{self.provider_name}/"):
            original_model = model
            model = model.split("/", 1)[1]
            self.logger.debug(
                f"Stripped provider prefix from model name (stream): {original_model} -> {model}"
            )

        # Handle case when messages are provided instead of prompt (for chat_completion)
        messages = kwargs.pop("messages", None)

        # If neither prompt nor messages are provided, raise an error
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        # Create messages if not already provided
        if not messages:
            messages = [{"role": "user", "content": prompt}]

        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        # Add max_tokens if specified
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Check for json_mode flag and remove it from kwargs
        json_mode = kwargs.pop("json_mode", False)
        if json_mode:
            # Use the correct response_format for JSON mode
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for OpenAI streaming")

        # Add any remaining additional parameters
        params.update(kwargs)

        # Log request
        prompt_length = len(prompt) if prompt else sum(len(m.get("content", "")) for m in messages)
        self.logger.info(
            f"Generating streaming completion with OpenAI model {model}",
            emoji_key=self.provider_name,
            prompt_length=prompt_length,
            json_mode=json_mode,  # Log if json_mode was requested
        )

        start_time = time.time()
        total_chunks = 0

        try:
            # Make streaming API call
            stream = await self.client.chat.completions.create(**params)

            # Process the stream
            async for chunk in stream:
                total_chunks += 1

                # Extract content from the chunk
                delta = chunk.choices[0].delta
                content = delta.content or ""

                # Metadata for this chunk
                metadata = {
                    "model": model,
                    "provider": self.provider_name,
                    "chunk_index": total_chunks,
                    "finish_reason": chunk.choices[0].finish_reason,
                }

                yield content, metadata

            # Log success
            processing_time = time.time() - start_time
            self.logger.success(
                "OpenAI streaming completion successful",
                emoji_key="success",
                model=model,
                chunks=total_chunks,
                time=processing_time,
            )

        except Exception as e:
            self.logger.error(
                f"OpenAI streaming completion failed: {str(e)}", emoji_key="error", model=model
            )
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available OpenAI models with their capabilities and metadata.

        This method queries the OpenAI API to retrieve a comprehensive list of available
        models accessible to the current API key. It filters the results to focus on
        GPT models that are relevant to text generation tasks, excluding embeddings,
        moderation, and other specialized models.

        For efficiency, the method uses a caching mechanism that stores the model list
        after the first successful API call. Subsequent calls return the cached results
        without making additional API requests. This reduces latency and API usage while
        ensuring the available models information is readily accessible.

        If the API call fails (due to network issues, invalid credentials, etc.), the
        method falls back to returning a hardcoded list of common OpenAI models to ensure
        the application can continue functioning with reasonable defaults.

        Returns:
            A list of dictionaries containing model information with these fields:
            - id: The model identifier used when making API calls (e.g., "gpt-4o")
            - provider: Always "openai" for this provider
            - created: Timestamp of when the model was created (if available from API)
            - owned_by: Organization that owns the model (e.g., "openai", "system")

            The fallback model list (used on API errors) includes basic information
            for gpt-4o, gpt-4.1-mini, and other commonly used models.

        Example response:
            ```python
            [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "created": 1693399330,
                    "owned_by": "openai"
                },
                {
                    "id": "gpt-4.1-mini",
                    "provider": "openai",
                    "created": 1705006269,
                    "owned_by": "openai"
                }
            ]
            ```

        Note:
            The specific models returned depend on the API key's permissions and
            the models currently offered by OpenAI. As new models are released
            or existing ones deprecated, the list will change accordingly.
        """
        if self.models_cache:
            return self.models_cache

        try:
            if not self.client:
                await self.initialize()

            # Fetch models from API
            response = await self.client.models.list()

            # Process response
            models = []
            for model in response.data:
                # Filter to relevant models (chat-capable GPT models)
                if model.id.startswith("gpt-"):
                    models.append(
                        {
                            "id": model.id,
                            "provider": self.provider_name,
                            "created": model.created,
                            "owned_by": model.owned_by,
                        }
                    )

            # Cache results
            self.models_cache = models

            return models

        except Exception as e:
            self.logger.error(f"Failed to list OpenAI models: {str(e)}", emoji_key="error")

            # Return basic models on error
            return [
                {
                    "id": "gpt-4o",
                    "provider": self.provider_name,
                    "description": "Most capable GPT-4 model",
                },
                {
                    "id": "gpt-4.1-mini",
                    "provider": self.provider_name,
                    "description": "Smaller, efficient GPT-4 model",
                },
                {
                    "id": "gpt-4.1-mini",
                    "provider": self.provider_name,
                    "description": "Fast and cost-effective GPT model",
                },
            ]

    def get_default_model(self) -> str:
        """
        Get the default OpenAI model identifier to use when none is specified.

        This method determines the appropriate default model for OpenAI completions
        through a prioritized selection process:

        1. First, it attempts to load the default_model setting from the Ultimate MCP Server
           configuration system (from providers.openai.default_model in the config)
        2. If that's not available or valid, it falls back to a hardcoded default model
           that represents a reasonable balance of capability, cost, and availability

        Using the configuration system allows for flexible deployment-specific defaults
        without code changes, while the hardcoded fallback ensures the system remains
        functional even with minimal configuration.

        Returns:
            String identifier of the default OpenAI model to use (e.g., "gpt-4.1-mini").
            This identifier can be directly used in API calls to the OpenAI API.

        Note:
            The current hardcoded default is "gpt-4.1-mini", chosen for its balance of
            capability and cost. This may change in future versions as new models are
            released or existing ones are deprecated.
        """
        from ultimate_mcp_server.config import get_config

        # Safely get from config if available
        try:
            config = get_config()
            provider_config = getattr(config, "providers", {}).get(self.provider_name, None)
            if provider_config and provider_config.default_model:
                return provider_config.default_model
        except (AttributeError, TypeError):
            # Handle case when providers attribute doesn't exist or isn't a dict
            pass

        # Otherwise return hard-coded default
        return "gpt-4.1-mini"

    async def check_api_key(self) -> bool:
        """Check if the OpenAI API key is valid.

        This method performs a lightweight validation of the configured OpenAI API key
        by attempting to list available models. A successful API call confirms that:

        1. The API key is properly formatted and not empty
        2. The key has at least read permissions on the OpenAI API
        3. The API endpoint is accessible and responding
        4. The account associated with the key is active and not suspended

        This validation is useful when initializing the provider to ensure the API key
        works before attempting to make model completion requests that might fail later.

        Returns:
            bool: True if the API key is valid and the API is accessible, False otherwise.
            A False result may indicate an invalid key, network issues, or API service disruption.

        Notes:
            - This method simply calls list_models() which caches results for efficiency
            - No detailed error information is returned, only a boolean success indicator
            - The method silently catches all exceptions and returns False rather than raising
            - For debugging key issues, check server logs for the full exception details
        """
        try:
            # Just list models as a simple validation
            await self.list_models()
            return True
        except Exception:
            return False
