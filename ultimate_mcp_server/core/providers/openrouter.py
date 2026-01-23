# ultimate/core/providers/openrouter.py
"""OpenRouter provider implementation."""

import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from openai import AsyncOpenAI

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.constants import DEFAULT_MODELS, Provider
from ultimate_mcp_server.core.providers.base import BaseProvider, ModelResponse
from ultimate_mcp_server.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("ultimate_mcp_server.providers.openrouter")

# Default OpenRouter Base URL (can be overridden by config)
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseProvider):
    """Provider implementation for OpenRouter API (using OpenAI-compatible interface)."""

    provider_name = Provider.OPENROUTER.value

    def __init__(self, **kwargs):
        """Initialize the OpenRouter provider.

        Args:
            **kwargs: Additional options:
                - base_url (str): Override the default OpenRouter API base URL.
                - http_referer (str): Optional HTTP-Referer header.
                - x_title (str): Optional X-Title header.
        """
        config = get_config().providers.openrouter
        super().__init__(**kwargs)
        self.name = "openrouter"

        # Use config default first, then fallback to constants
        self.default_model = config.default_model or DEFAULT_MODELS.get(Provider.OPENROUTER)
        if not config.default_model:
            logger.debug(
                f"No default model set in config for OpenRouter, using fallback from constants: {self.default_model}"
            )

        # Get base_url from config, fallback to kwargs, then constant
        self.base_url = config.base_url or kwargs.get("base_url", DEFAULT_OPENROUTER_BASE_URL)

        # Get additional headers from config's additional_params
        self.http_referer = config.additional_params.get("http_referer") or kwargs.get(
            "http_referer"
        )
        self.x_title = config.additional_params.get("x_title") or kwargs.get("x_title")

        # We'll create the client in initialize() instead
        self.client = None
        self.available_models = []

    async def initialize(self) -> bool:
        """Initialize the OpenRouter client.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Create headers dictionary
            headers = {}
            if self.http_referer:
                headers["HTTP-Referer"] = self.http_referer
            if self.x_title:
                headers["X-Title"] = self.x_title

            # Get timeout from config
            config = get_config().providers.openrouter
            timeout = config.timeout or 30.0  # Default timeout 30s

            # Check if API key is available
            if not self.api_key:
                logger.warning(
                    f"{self.name} API key not found in configuration. Provider will be unavailable."
                )
                return False

            # Create the client
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers=headers,
                timeout=timeout,
            )

            # Pre-fetch available models
            try:
                self.available_models = await self.list_models()
                logger.info(f"Loaded {len(self.available_models)} models from OpenRouter")
            except Exception as model_err:
                logger.warning(f"Failed to fetch models from OpenRouter: {str(model_err)}")
                # Use hardcoded fallback models
                self.available_models = self._get_fallback_models()

            logger.success("OpenRouter provider initialized successfully", emoji_key="provider")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter provider: {str(e)}", emoji_key="error")
            return False

    def _initialize_client(self, **kwargs):
        """Initialize the OpenAI async client with OpenRouter specifics."""
        # This method is now deprecated - use initialize() instead
        logger.warning("_initialize_client() is deprecated, use initialize() instead")
        return False

    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> ModelResponse:
        """Generate a completion using OpenRouter.

        Args:
            prompt: Text prompt to send to the model
            model: Model name (e.g., "openai/gpt-4.1-mini", "google/gemini-flash-1.5")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters, including:
                - extra_headers (Dict): Additional headers for this specific call.
                - extra_body (Dict): OpenRouter-specific arguments.

        Returns:
            ModelResponse with completion result

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            initialized = await self._initialize_client()
            if not initialized:
                raise RuntimeError(f"{self.provider_name} provider not initialized.")

        # Use default model if not specified
        model = model or self.default_model

        # Ensure we have a model name before proceeding
        if model is None:
            logger.error(
                "Completion failed: No model specified and no default model configured for OpenRouter."
            )
            raise ValueError("No model specified and no default model configured for OpenRouter.")

        # Strip provider prefix only if it matches OUR provider name
        if model.startswith(f"{self.provider_name}:"):
            model = model.split(":", 1)[1]
            logger.debug(f"Stripped provider prefix from model name: {model}")
        # Note: Keep prefixes like 'openai/' or 'google/' as OpenRouter uses them.
        # DO NOT strip other provider prefixes as they're needed for OpenRouter routing

        # Create messages
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]

        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Extract OpenRouter specific args from kwargs
        extra_headers = kwargs.pop("extra_headers", {})
        extra_body = kwargs.pop("extra_body", {})

        json_mode = kwargs.pop("json_mode", False)
        if json_mode:
            # OpenRouter uses OpenAI-compatible API
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for OpenRouter")

        # Add any remaining kwargs to the main params (standard OpenAI args)
        params.update(kwargs)

        self.logger.info(
            f"Generating completion with {self.provider_name} model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt),
            json_mode_requested=json_mode,
        )

        try:
            # Make API call with timing
            response, processing_time = await self.process_with_timer(
                self.client.chat.completions.create,
                **params,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )

            # Extract response text
            completion_text = response.choices[0].message.content

            # Create standardized response
            result = ModelResponse(
                text=completion_text,
                model=response.model,  # Use model returned by API
                provider=self.provider_name,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                processing_time=processing_time,
                raw_response=response,
            )

            self.logger.success(
                f"{self.provider_name} completion successful",
                emoji_key="success",
                model=result.model,
                tokens={"input": result.input_tokens, "output": result.output_tokens},
                cost=result.cost,  # Will be calculated by ModelResponse
                time=result.processing_time,
            )

            return result

        except Exception as e:
            self.logger.error(
                f"{self.provider_name} completion failed for model {model}: {str(e)}",
                emoji_key="error",
                model=model,
            )
            raise

    async def generate_completion_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion using OpenRouter.

        Args:
            prompt: Text prompt to send to the model
            model: Model name (e.g., "openai/gpt-4.1-mini")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters, including:
                - extra_headers (Dict): Additional headers for this specific call.
                - extra_body (Dict): OpenRouter-specific arguments.

        Yields:
            Tuple of (text_chunk, metadata)

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            initialized = await self._initialize_client()
            if not initialized:
                raise RuntimeError(f"{self.provider_name} provider not initialized.")

        model = model or self.default_model
        if model.startswith(f"{self.provider_name}:"):
            model = model.split(":", 1)[1]
        # DO NOT strip other provider prefixes as they're needed for OpenRouter routing

        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]

        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        extra_headers = kwargs.pop("extra_headers", {})
        extra_body = kwargs.pop("extra_body", {})

        json_mode = kwargs.pop("json_mode", False)
        if json_mode:
            # OpenRouter uses OpenAI-compatible API
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for OpenRouter streaming")

        params.update(kwargs)

        self.logger.info(
            f"Generating streaming completion with {self.provider_name} model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt),
            json_mode_requested=json_mode,
        )

        start_time = time.time()
        total_chunks = 0
        final_model_name = model  # Store initially requested model

        try:
            stream = await self.client.chat.completions.create(
                **params, extra_headers=extra_headers, extra_body=extra_body
            )

            async for chunk in stream:
                total_chunks += 1
                delta = chunk.choices[0].delta
                content = delta.content or ""

                # Try to get model name from the chunk if available (some providers include it)
                if chunk.model:
                    final_model_name = chunk.model

                metadata = {
                    "model": final_model_name,
                    "provider": self.provider_name,
                    "chunk_index": total_chunks,
                    "finish_reason": chunk.choices[0].finish_reason,
                }

                yield content, metadata

            processing_time = time.time() - start_time
            self.logger.success(
                f"{self.provider_name} streaming completion successful",
                emoji_key="success",
                model=final_model_name,
                chunks=total_chunks,
                time=processing_time,
            )

        except Exception as e:
            self.logger.error(
                f"{self.provider_name} streaming completion failed for model {model}: {str(e)}",
                emoji_key="error",
                model=model,
            )
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenRouter models (provides examples, not exhaustive).

        OpenRouter offers a vast number of models. This list provides common examples.
        Refer to OpenRouter documentation for the full list.

        Returns:
            List of example model information dictionaries
        """
        if self.available_models:
            return self.available_models
        models = self._get_fallback_models()
        return models

    def get_default_model(self) -> str:
        """Get the default OpenRouter model.

        Returns:
            Default model name (e.g., "openai/gpt-4.1-mini")
        """
        # Allow override via environment variable
        default_model_env = os.environ.get("OPENROUTER_DEFAULT_MODEL")
        if default_model_env:
            return default_model_env

        # Fallback to constants
        return DEFAULT_MODELS.get(self.provider_name, "openai/gpt-4.1-mini")

    async def check_api_key(self) -> bool:
        """Check if the OpenRouter API key is valid by attempting a small request."""
        if not self.client:
            # Try to initialize if not already done
            if not await self._initialize_client():
                return False  # Initialization failed

        try:
            # Attempt a simple, low-cost operation, e.g., list models (even if it returns 404/permission error, it validates the key/URL)
            # Or use a very small completion request
            await self.client.chat.completions.create(
                model=self.get_default_model(),
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0,
            )
            return True
        except Exception as e:
            logger.warning(
                f"API key check failed for {self.provider_name}: {str(e)}", emoji_key="warning"
            )
            return False

    def get_available_models(self) -> List[str]:
        """Return a list of available model names."""
        return [model["id"] for model in self.available_models]

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        available_model_ids = [model["id"] for model in self.available_models]
        return model_name in available_model_ids

    async def create_completion(
        self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Create a completion using the specified model."""
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized (likely missing API key).")

        # Check if model is available
        if not self.is_model_available(model):
            # Fallback to default if provided model isn't listed? Or raise error?
            # Let's try the default model if the requested one isn't confirmed available.
            if self.default_model and self.is_model_available(self.default_model):
                logger.warning(
                    f"Model '{model}' not found in available list. Falling back to default '{self.default_model}'."
                )
                model = self.default_model
            else:
                # If even the default isn't available or set, raise error
                raise ValueError(
                    f"Model '{model}' is not available via OpenRouter according to fetched list, and no valid default model is set."
                )

        merged_kwargs = {**kwargs}
        # OpenRouter uses standard OpenAI params like max_tokens, temperature, etc.
        # Ensure essential params are passed
        if "max_tokens" not in merged_kwargs:
            merged_kwargs["max_tokens"] = (
                get_config().providers.openrouter.max_tokens or 1024
            )  # Use config or default

        if stream:
            logger.debug(f"Creating stream completion: Model={model}, Params={merged_kwargs}")
            return self._stream_completion_generator(model, messages, **merged_kwargs)
        else:
            logger.debug(f"Creating completion: Model={model}, Params={merged_kwargs}")
            try:
                response = await self.client.chat.completions.create(
                    model=model, messages=messages, stream=False, **merged_kwargs
                )
                # Extract content based on OpenAI library version
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        return (
                            choice.message.content or ""
                        )  # Return empty string if content is None
                    elif hasattr(choice, "delta") and hasattr(
                        choice.delta, "content"
                    ):  # Should not happen for stream=False but check
                        return choice.delta.content or ""
                logger.warning("Could not extract content from OpenRouter response.")
                return ""  # Return empty string if no content found
            except Exception as e:
                logger.error(f"OpenRouter completion failed: {e}", exc_info=True)
                raise RuntimeError(f"OpenRouter API call failed: {e}") from e

    async def _stream_completion_generator(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async generator for streaming completions."""
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized (likely missing API key).")
        try:
            stream = await self.client.chat.completions.create(
                model=model, messages=messages, stream=True, **kwargs
            )
            async for chunk in stream:
                # Extract content based on OpenAI library version
                content = ""
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        content = choice.delta.content
                    elif hasattr(choice, "message") and hasattr(
                        choice.message, "content"
                    ):  # Should not happen for stream=True
                        content = choice.message.content

                if content:
                    yield content
        except Exception as e:
            logger.error(f"OpenRouter stream completion failed: {e}", exc_info=True)
            # Depending on desired behavior, either raise or yield an error message
            # yield f"Error during stream: {e}"
            raise RuntimeError(f"OpenRouter API stream failed: {e}") from e

    # --- Cost Calculation (Needs OpenRouter Specific Data) ---
    def get_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculate the cost of a request based on OpenRouter pricing.

        Note: Requires loading detailed model pricing info, which is not
              done by default in fetch_available_models.
              This is a placeholder and needs enhancement.
        """
        # Placeholder: Need to fetch and store detailed pricing from OpenRouter
        # Example structure (needs actual data):
        openrouter_pricing = {
            # "model_id": {"prompt_cost_per_mtok": X, "completion_cost_per_mtok": Y},
            "openai/gpt-4o": {"prompt_cost_per_mtok": 5.0, "completion_cost_per_mtok": 15.0},
            "google/gemini-pro-1.5": {
                "prompt_cost_per_mtok": 3.5,
                "completion_cost_per_mtok": 10.5,
            },
            "anthropic/claude-3-opus": {
                "prompt_cost_per_mtok": 15.0,
                "completion_cost_per_mtok": 75.0,
            },
            # ... add more model costs from openrouter.ai/docs#models ...
        }

        model_cost = openrouter_pricing.get(model)
        if model_cost:
            prompt_cost = (prompt_tokens / 1_000_000) * model_cost.get("prompt_cost_per_mtok", 0)
            completion_cost = (completion_tokens / 1_000_000) * model_cost.get(
                "completion_cost_per_mtok", 0
            )
            return prompt_cost + completion_cost
        else:
            logger.warning(f"Cost calculation not available for OpenRouter model: {model}")
            # Return None if cost cannot be calculated
            return None

    # --- Prompt Formatting --- #
    def format_prompt(self, messages: List[Dict[str, str]]) -> Any:
        """Use standard list of dictionaries format for OpenRouter (like OpenAI)."""
        # OpenRouter generally uses the same format as OpenAI
        return messages

    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Return a list of fallback models when API is not accessible."""
        return [
            {
                "id": "mistralai/mistral-large",
                "provider": self.provider_name,
                "description": "Mistral: Strong open-weight model.",
            },
            {
                "id": "mistralai/mistral-nemo",
                "provider": self.provider_name,
                "description": "Mistral: Strong open-weight model.",
            },
            {
                "id": "meta-llama/llama-3-70b-instruct",
                "provider": self.provider_name,
                "description": "Meta: Powerful open-source instruction-tuned model.",
            },
        ]


# Make available via discovery
__all__ = ["OpenRouterProvider"]
