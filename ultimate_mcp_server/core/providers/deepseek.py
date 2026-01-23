"""DeepSeek provider implementation."""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import BaseProvider, ModelResponse
from ultimate_mcp_server.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("ultimate_mcp_server.providers.deepseek")


class DeepSeekProvider(BaseProvider):
    """Provider implementation for DeepSeek API (using OpenAI-compatible interface)."""

    provider_name = Provider.DEEPSEEK.value

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the DeepSeek provider.

        Args:
            api_key: DeepSeek API key
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = kwargs.get("base_url", "https://api.deepseek.com")
        self.models_cache = None

    async def initialize(self) -> bool:
        """Initialize the DeepSeek client.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # DeepSeek uses OpenAI-compatible API
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            self.logger.success("DeepSeek provider initialized successfully", emoji_key="provider")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to initialize DeepSeek provider: {str(e)}", emoji_key="error"
            )
            return False

    async def generate_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        json_mode: bool = False,
        **kwargs,
    ) -> ModelResponse:
        """Generate a completion using DeepSeek's API.

        Args:
            prompt: Text prompt to send to the model (optional if messages provided)
            messages: List of message dictionaries (optional if prompt provided)
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            json_mode: If True, attempt to generate JSON output
            **kwargs: Additional parameters

        Returns:
            ModelResponse with the completion result
        """
        if not self.client:
            await self.initialize()

        # Verify we have either prompt or messages
        if prompt is None and not messages:
            raise ValueError("Either prompt or messages must be provided")

        # Use default model if not specified
        model = model or self.get_default_model()

        # Prepare API parameters
        if messages:
            # Using chat completion with messages
            params = {"model": model, "messages": messages, "temperature": temperature}
        else:
            # Using completion with prompt
            # Convert prompt to messages format for DeepSeek
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }

        # Add max_tokens if provided
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Handle JSON mode via response_format for compatible models
        if json_mode:
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for DeepSeek")

        # Add any remaining parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        # Log request parameters
        prompt_length = len(prompt) if prompt else sum(len(m.get("content", "")) for m in messages)
        self.logger.info(
            f"Generating completion with DeepSeek model {model}",
            emoji_key=self.provider_name,
            prompt_length=prompt_length,
            json_mode=json_mode,
        )

        try:
            # Start timer
            start_time = time.time()

            # Make API call
            response = await self.client.chat.completions.create(**params)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Extract text from response
            completion_text = response.choices[0].message.content

            # Create ModelResponse
            result = ModelResponse(
                text=completion_text,
                model=f"{self.provider_name}/{model}",
                provider=self.provider_name,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                processing_time=processing_time,
                raw_response=response,
            )

            # Add message for compatibility with chat_completion
            result.message = {"role": "assistant", "content": completion_text}

            # Log success
            self.logger.success(
                "DeepSeek completion successful",
                emoji_key="success",
                model=model,
                tokens={"input": result.input_tokens, "output": result.output_tokens},
                cost=result.cost,
                time=processing_time,
            )

            return result

        except Exception as e:
            # Log error
            self.logger.error(
                f"DeepSeek completion failed: {str(e)}", emoji_key="error", model=model
            )
            raise

    async def generate_completion_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        json_mode: bool = False,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion using DeepSeek.

        Args:
            prompt: Text prompt to send to the model (optional if messages provided)
            messages: List of message dictionaries (optional if prompt provided)
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            json_mode: If True, attempt to generate JSON output
            **kwargs: Additional parameters

        Yields:
            Tuples of (text_chunk, metadata)
        """
        if not self.client:
            await self.initialize()

        # Verify we have either prompt or messages
        if prompt is None and not messages:
            raise ValueError("Either prompt or messages must be provided")

        # Use default model if not specified
        model = model or self.get_default_model()

        # Prepare API parameters
        if messages:
            # Using chat completion with messages
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
        else:
            # Using completion with prompt
            # Convert prompt to messages format for DeepSeek
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "stream": True,
            }

        # Add max_tokens if provided
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Handle JSON mode via response_format for compatible models
        if json_mode:
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for DeepSeek streaming")

        # Add any remaining parameters
        for key, value in kwargs.items():
            if key not in params and key != "stream":  # Don't allow overriding stream
                params[key] = value

        # Log request parameters
        prompt_length = len(prompt) if prompt else sum(len(m.get("content", "")) for m in messages)
        self.logger.info(
            f"Generating streaming completion with DeepSeek model {model}",
            emoji_key=self.provider_name,
            prompt_length=prompt_length,
            json_mode=json_mode,
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
                    "model": f"{self.provider_name}/{model}",
                    "provider": self.provider_name,
                    "chunk_index": total_chunks,
                    "finish_reason": chunk.choices[0].finish_reason,
                }

                yield content, metadata

            # Log success
            processing_time = time.time() - start_time
            self.logger.success(
                "DeepSeek streaming completion successful",
                emoji_key="success",
                model=model,
                chunks=total_chunks,
                time=processing_time,
            )

            # Yield final metadata chunk
            final_metadata = {
                "model": f"{self.provider_name}/{model}",
                "provider": self.provider_name,
                "chunk_index": total_chunks + 1,
                "processing_time": processing_time,
                "finish_reason": "stop",
            }
            yield "", final_metadata

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"DeepSeek streaming completion failed: {str(e)}", emoji_key="error", model=model
            )

            # Yield error metadata
            error_metadata = {
                "model": f"{self.provider_name}/{model}",
                "provider": self.provider_name,
                "chunk_index": total_chunks + 1,
                "error": f"{type(e).__name__}: {str(e)}",
                "processing_time": processing_time,
                "finish_reason": "error",
            }
            yield "", error_metadata

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available DeepSeek models.

        Returns:
            List of model information dictionaries
        """
        # DeepSeek doesn't have a comprehensive models endpoint, so we return a static list
        if self.models_cache:
            return self.models_cache

        models = [
            {
                "id": "deepseek-chat",
                "provider": self.provider_name,
                "description": "General-purpose chat model",
            },
            {
                "id": "deepseek-reasoner",
                "provider": self.provider_name,
                "description": "Enhanced reasoning capabilities",
            },
        ]

        # Cache results
        self.models_cache = models

        return models

    def get_default_model(self) -> str:
        """Get the default DeepSeek model.

        Returns:
            Default model name
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
        return "deepseek-chat"

    async def check_api_key(self) -> bool:
        """Check if the DeepSeek API key is valid.

        Returns:
            bool: True if API key is valid
        """
        try:
            # Try a simple completion to validate the API key
            await self.client.chat.completions.create(
                model=self.get_default_model(),
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
