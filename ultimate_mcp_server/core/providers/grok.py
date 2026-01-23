"""Grok (xAI) provider implementation."""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import BaseProvider, ModelResponse
from ultimate_mcp_server.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("ultimate_mcp_server.providers.grok")


class GrokProvider(BaseProvider):
    """Provider implementation for xAI's Grok API."""

    provider_name = Provider.GROK.value

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Grok provider.

        Args:
            api_key: xAI API key
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = kwargs.get("base_url", "https://api.x.ai/v1")
        self.models_cache = None

    async def initialize(self) -> bool:
        """Initialize the Grok client.

        Returns:
            bool: True if initialization was successful
        """
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            # Skip API call if using a mock key (for tests)
            if self.api_key and "mock-" in self.api_key:
                self.logger.info("Using mock Grok key - skipping API validation", emoji_key="mock")
                return True

            # Test connection by listing models
            await self.list_models()

            self.logger.success("Grok provider initialized successfully", emoji_key="provider")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Grok provider: {str(e)}", emoji_key="error")
            return False

    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> ModelResponse:
        """Generate a completion using Grok.

        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "grok-3-latest")
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

        # Strip provider prefix if present (e.g., "grok/grok-3" -> "grok-3")
        if model.startswith(f"{self.provider_name}/"):
            original_model = model
            model = model.split("/", 1)[1]
            self.logger.debug(
                f"Stripped provider prefix from model name: {original_model} -> {model}"
            )

        # Create messages
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]

        # Get system message if provided
        system_message = kwargs.pop("system", None)
        if system_message and not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_message})

        # Handle tool support (function calling)
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)

        # Handle reasoning effort for grok-3-mini models
        reasoning_effort = None
        if model.startswith("grok-3-mini") and "reasoning_effort" in kwargs:
            reasoning_effort = kwargs.pop("reasoning_effort")

        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # Add max_tokens if specified
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Add tools and tool_choice if specified
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        # Handle JSON mode
        json_mode = kwargs.pop("json_mode", False)
        if json_mode:
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for Grok")

        # Add reasoning_effort for mini models if specified
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort

        # Add any additional parameters
        params.update(kwargs)

        # Log request
        self.logger.info(
            f"Generating completion with Grok model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt),
            json_mode_requested=json_mode,
        )

        try:
            # API call with timing
            response, processing_time = await self.process_with_timer(
                self.client.chat.completions.create, **params
            )

            # Extract response text
            completion_text = response.choices[0].message.content

            # Extract reasoning content for grok-3-mini models if available
            reasoning_content = None
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning_content = response.choices[0].message.reasoning_content

            # Get usage statistics
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Extract reasoning tokens if available
            reasoning_tokens = None
            if hasattr(response.usage, "completion_tokens_details") and hasattr(
                response.usage.completion_tokens_details, "reasoning_tokens"
            ):
                reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens

            # Create metadata with reasoning information
            metadata = {}
            if reasoning_content:
                metadata["reasoning_content"] = reasoning_content
            if reasoning_tokens:
                metadata["reasoning_tokens"] = reasoning_tokens

            # Create standardized response
            result = ModelResponse(
                text=completion_text,
                model=model,
                provider=self.provider_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                processing_time=processing_time,
                raw_response=response,
                metadata=metadata,
            )

            # Log success
            self.logger.success(
                "Grok completion successful",
                emoji_key="success",
                model=model,
                tokens={"input": result.input_tokens, "output": result.output_tokens},
                cost=result.cost,
                time=result.processing_time,
            )

            return result

        except Exception as e:
            self.logger.error(f"Grok completion failed: {str(e)}", emoji_key="error", model=model)
            raise

    async def generate_completion_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion using Grok.

        Args:
            prompt: Text prompt to send to the model
            model: Model name to use (e.g., "grok-3-latest")
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

        # Strip provider prefix if present (e.g., "grok/grok-3" -> "grok-3")
        if model.startswith(f"{self.provider_name}/"):
            original_model = model
            model = model.split("/", 1)[1]
            self.logger.debug(
                f"Stripped provider prefix from model name (stream): {original_model} -> {model}"
            )

        # Create messages
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]

        # Get system message if provided
        system_message = kwargs.pop("system", None)
        if system_message and not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_message})

        # Handle reasoning effort for grok-3-mini models
        reasoning_effort = None
        if model.startswith("grok-3-mini") and "reasoning_effort" in kwargs:
            reasoning_effort = kwargs.pop("reasoning_effort")

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

        # Handle JSON mode
        json_mode = kwargs.pop("json_mode", False)
        if json_mode:
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for Grok streaming")

        # Add reasoning_effort for mini models if specified
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort

        # Add any additional parameters
        params.update(kwargs)

        # Log request
        self.logger.info(
            f"Generating streaming completion with Grok model {model}",
            emoji_key=self.provider_name,
            prompt_length=len(prompt),
            json_mode_requested=json_mode,
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

                # Extract reasoning content for grok-3-mini models if available
                reasoning_content = None
                if hasattr(delta, "reasoning_content"):
                    reasoning_content = delta.reasoning_content

                # Metadata for this chunk
                metadata = {
                    "model": model,
                    "provider": self.provider_name,
                    "chunk_index": total_chunks,
                    "finish_reason": chunk.choices[0].finish_reason,
                }

                # Add reasoning content to metadata if available
                if reasoning_content:
                    metadata["reasoning_content"] = reasoning_content

                yield content, metadata

            # Log success
            processing_time = time.time() - start_time
            self.logger.success(
                "Grok streaming completion successful",
                emoji_key="success",
                model=model,
                chunks=total_chunks,
                time=processing_time,
            )

        except Exception as e:
            self.logger.error(
                f"Grok streaming completion failed: {str(e)}", emoji_key="error", model=model
            )
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Grok models.

        Returns:
            List of model information dictionaries
        """
        if self.models_cache:
            return self.models_cache

        try:
            if not self.client:
                await self.initialize()

            # Fetch models from API (Grok API uses the same endpoint as OpenAI)
            response = await self.client.models.list()

            # Process response
            models = []
            for model in response.data:
                # Filter to only include grok-3 models
                if model.id.startswith("grok-3"):
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
            self.logger.error(f"Failed to list Grok models: {str(e)}", emoji_key="error")

            # Return basic grok-3 models on error based on documentation
            return [
                {
                    "id": "grok-3-latest",
                    "provider": self.provider_name,
                    "description": "Flagship model for enterprise tasks (latest version)",
                },
                {
                    "id": "grok-3-beta",
                    "provider": self.provider_name,
                    "description": "Flagship model that excels at enterprise tasks, domain knowledge",
                },
                {
                    "id": "grok-3-fast-latest",
                    "provider": self.provider_name,
                    "description": "Fast version of grok-3, same quality with higher cost",
                },
                {
                    "id": "grok-3-mini-latest",
                    "provider": self.provider_name,
                    "description": "Lightweight model with thinking capabilities",
                },
                {
                    "id": "grok-3-mini-fast-latest",
                    "provider": self.provider_name,
                    "description": "Fast version of grok-3-mini, same quality with higher cost",
                },
            ]

    def get_default_model(self) -> str:
        """Get the default Grok model.

        Returns:
            Default model name
        """
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
        return "grok-3-beta"

    async def check_api_key(self) -> bool:
        """Check if the Grok API key is valid.

        Returns:
            bool: True if API key is valid
        """
        try:
            # Just list models as a simple validation
            await self.list_models()
            return True
        except Exception:
            return False
