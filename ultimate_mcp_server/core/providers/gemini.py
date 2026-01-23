"""Google Gemini provider implementation."""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from google import genai

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import BaseProvider, ModelResponse
from ultimate_mcp_server.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("ultimate_mcp_server.providers.gemini")


class GeminiProvider(BaseProvider):
    """Provider implementation for Google Gemini API."""

    provider_name = Provider.GEMINI.value

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Gemini provider.

        Args:
            api_key: Google API key
            **kwargs: Additional options
        """
        super().__init__(api_key=api_key, **kwargs)
        self.models_cache = None

    async def initialize(self) -> bool:
        """Initialize the Gemini client.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Skip real API calls if using mock key for tests
            if self.api_key and "mock-" in self.api_key:
                self.logger.info(
                    "Using mock Gemini key - skipping API initialization", emoji_key="mock"
                )
                self.client = {"mock_client": True}
                return True

            # Create a client instance instead of configuring globally
            self.client = genai.Client(
                api_key=self.api_key, http_options={"api_version": "v1alpha"}
            )

            self.logger.success("Gemini provider initialized successfully", emoji_key="provider")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini provider: {str(e)}", emoji_key="error")
            return False

    async def generate_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> ModelResponse:
        """Generate a completion using Google Gemini.

        Args:
            prompt: Text prompt to send to the model (or None if messages provided)
            messages: List of message dictionaries (alternative to prompt)
            model: Model name to use (e.g., "gemini-2.0-flash-lite")
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (0.0-1.0)
            **kwargs: Additional model-specific parameters

        Returns:
            ModelResponse: Standardized response

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            await self.initialize()

        # Use default model if not specified
        model = model or self.get_default_model()

        # Strip provider prefix if present (e.g., "gemini:gemini-2.0-pro" -> "gemini-2.0-pro")
        if ":" in model:
            original_model = model
            model = model.split(":", 1)[1]
            self.logger.debug(
                f"Stripped provider prefix from model name: {original_model} -> {model}"
            )

        # Validate that either prompt or messages is provided
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        # Prepare generation config and API call kwargs
        config = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens  # Gemini uses max_output_tokens

        # Pop json_mode flag
        json_mode = kwargs.pop("json_mode", False)

        # Set up JSON mode in config dict per Gemini API docs
        if json_mode:
            # For Gemini, JSON mode is set via response_mime_type in the config dict
            config["response_mime_type"] = "application/json"
            self.logger.debug("Setting response_mime_type to application/json for Gemini in config")

        # Add remaining kwargs to config
        for key in list(kwargs.keys()):
            if key in ["top_p", "top_k", "candidate_count", "stop_sequences"]:
                config[key] = kwargs.pop(key)

        # Store other kwargs that might need to be passed directly
        request_params = {}
        for key in list(kwargs.keys()):
            if key in ["safety_settings", "tools", "system"]:
                request_params[key] = kwargs.pop(key)

        # Prepare content based on input type (prompt or messages)
        content = None
        if prompt:
            content = prompt
            log_input_size = len(prompt)
        elif messages:
            # Convert messages to Gemini format
            content = []
            log_input_size = 0
            for msg in messages:
                role = msg.get("role", "").lower()
                text = msg.get("content", "")
                log_input_size += len(text)

                # Map roles to Gemini's expectations
                if role == "system":
                    # For system messages, prepend to user input or add as user message
                    system_text = text
                    # Find the next user message to prepend to
                    for _i, future_msg in enumerate(
                        messages[messages.index(msg) + 1 :], messages.index(msg) + 1
                    ):
                        if future_msg.get("role", "").lower() == "user":
                            # Leave this system message to be handled when we reach the user message
                            # Just track its content for now
                            break
                    else:
                        # No user message found after system, add as separate user message
                        content.append({"role": "user", "parts": [{"text": system_text}]})
                    continue

                elif role == "user":
                    # Check if previous message was a system message
                    prev_system_text = ""
                    if messages.index(msg) > 0:
                        prev_msg = messages[messages.index(msg) - 1]
                        if prev_msg.get("role", "").lower() == "system":
                            prev_system_text = prev_msg.get("content", "")

                    # If there was a system message before, prepend it to the user message
                    if prev_system_text:
                        gemini_role = "user"
                        gemini_text = f"{prev_system_text}\n\n{text}"
                    else:
                        gemini_role = "user"
                        gemini_text = text

                elif role == "assistant":
                    gemini_role = "model"
                    gemini_text = text
                else:
                    self.logger.warning(f"Unsupported message role '{role}', treating as user")
                    gemini_role = "user"
                    gemini_text = text

                content.append({"role": gemini_role, "parts": [{"text": gemini_text}]})

        # Log request
        self.logger.info(
            f"Generating completion with Gemini model {model}",
            emoji_key=self.provider_name,
            prompt_length=log_input_size,
            json_mode_requested=json_mode,
        )

        start_time = time.time()

        try:
            # Check if we're using a mock client for testing
            if isinstance(self.client, dict) and self.client.get("mock_client"):
                # Return mock response for tests
                completion_text = "Mock Gemini response for testing"
                processing_time = 0.1
                response = None
            else:
                # Pass everything in the correct structure according to the API
                if isinstance(content, list):  # messages format
                    response = self.client.models.generate_content(
                        model=model,
                        contents=content,
                        config=config,  # Pass config dict containing temperature, max_output_tokens, etc.
                        **request_params,  # Pass other params directly if needed
                    )
                else:  # prompt format (string)
                    response = self.client.models.generate_content(
                        model=model,
                        contents=content,
                        config=config,  # Pass config dict containing temperature, max_output_tokens, etc.
                        **request_params,  # Pass other params directly if needed
                    )

                processing_time = time.time() - start_time

                # Extract response text
                completion_text = response.text

            # Estimate token usage (Gemini doesn't provide token counts)
            # Roughly 4 characters per token as a crude approximation
            char_to_token_ratio = 4.0
            estimated_input_tokens = log_input_size / char_to_token_ratio
            estimated_output_tokens = len(completion_text) / char_to_token_ratio

            # Create standardized response
            result = ModelResponse(
                text=completion_text,
                model=model,
                provider=self.provider_name,
                input_tokens=int(estimated_input_tokens),
                output_tokens=int(estimated_output_tokens),
                processing_time=processing_time,
                raw_response=None,  # Don't need raw response for tests
                metadata={"token_count_estimated": True},
            )

            # Add message for consistency with other providers
            result.message = {"role": "assistant", "content": completion_text}

            # Log success
            self.logger.success(
                "Gemini completion successful",
                emoji_key="success",
                model=model,
                tokens={"input": result.input_tokens, "output": result.output_tokens},
                cost=result.cost,
                time=result.processing_time,
            )

            return result

        except Exception as e:
            self.logger.error(f"Gemini completion failed: {str(e)}", emoji_key="error", model=model)
            raise

    async def generate_completion_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion using Google Gemini.

        Args:
            prompt: Text prompt to send to the model (or None if messages provided)
            messages: List of message dictionaries (alternative to prompt)
            model: Model name to use (e.g., "gemini-2.0-flash-lite")
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

        # Strip provider prefix if present (e.g., "gemini:gemini-2.0-pro" -> "gemini-2.0-pro")
        if ":" in model:
            original_model = model
            model = model.split(":", 1)[1]
            self.logger.debug(
                f"Stripped provider prefix from model name (stream): {original_model} -> {model}"
            )

        # Validate that either prompt or messages is provided
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        # Prepare config dict per Gemini API
        config = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        # Pop json_mode flag
        json_mode = kwargs.pop("json_mode", False)

        # Set up JSON mode in config dict
        if json_mode:
            # For Gemini, JSON mode is set via response_mime_type in the config dict
            config["response_mime_type"] = "application/json"
            self.logger.debug(
                "Setting response_mime_type to application/json for Gemini streaming in config"
            )

        # Add remaining kwargs to config
        for key in list(kwargs.keys()):
            if key in ["top_p", "top_k", "candidate_count", "stop_sequences"]:
                config[key] = kwargs.pop(key)

        # Store other kwargs that might need to be passed directly
        request_params = {}
        for key in list(kwargs.keys()):
            if key in ["safety_settings", "tools", "system"]:
                request_params[key] = kwargs.pop(key)

        # Prepare content based on input type (prompt or messages)
        content = None
        if prompt:
            content = prompt
            log_input_size = len(prompt)
        elif messages:
            # Convert messages to Gemini format
            content = []
            log_input_size = 0
            for msg in messages:
                role = msg.get("role", "").lower()
                text = msg.get("content", "")
                log_input_size += len(text)

                # Map roles to Gemini's expectations
                if role == "system":
                    # For system messages, prepend to user input or add as user message
                    system_text = text
                    # Find the next user message to prepend to
                    for _i, future_msg in enumerate(
                        messages[messages.index(msg) + 1 :], messages.index(msg) + 1
                    ):
                        if future_msg.get("role", "").lower() == "user":
                            # Leave this system message to be handled when we reach the user message
                            # Just track its content for now
                            break
                    else:
                        # No user message found after system, add as separate user message
                        content.append({"role": "user", "parts": [{"text": system_text}]})
                    continue

                elif role == "user":
                    # Check if previous message was a system message
                    prev_system_text = ""
                    if messages.index(msg) > 0:
                        prev_msg = messages[messages.index(msg) - 1]
                        if prev_msg.get("role", "").lower() == "system":
                            prev_system_text = prev_msg.get("content", "")

                    # If there was a system message before, prepend it to the user message
                    if prev_system_text:
                        gemini_role = "user"
                        gemini_text = f"{prev_system_text}\n\n{text}"
                    else:
                        gemini_role = "user"
                        gemini_text = text

                elif role == "assistant":
                    gemini_role = "model"
                    gemini_text = text
                else:
                    self.logger.warning(f"Unsupported message role '{role}', treating as user")
                    gemini_role = "user"
                    gemini_text = text

                content.append({"role": gemini_role, "parts": [{"text": gemini_text}]})

        # Log request
        self.logger.info(
            f"Generating streaming completion with Gemini model {model}",
            emoji_key=self.provider_name,
            input_type=f"{'prompt' if prompt else 'messages'} ({log_input_size} chars)",
            json_mode_requested=json_mode,
        )

        start_time = time.time()
        total_chunks = 0

        try:
            # Use the dedicated streaming method as per Google's documentation
            try:
                if isinstance(content, list):  # messages format
                    stream_response = self.client.models.generate_content_stream(
                        model=model, contents=content, config=config, **request_params
                    )
                else:  # prompt format (string)
                    stream_response = self.client.models.generate_content_stream(
                        model=model, contents=content, config=config, **request_params
                    )

                # Process the stream - iterating over chunks
                async def iterate_response():
                    # Convert sync iterator to async
                    for chunk in stream_response:
                        yield chunk

                async for chunk in iterate_response():
                    total_chunks += 1

                    # Extract text from the chunk
                    chunk_text = ""
                    if hasattr(chunk, "text"):
                        chunk_text = chunk.text
                    elif hasattr(chunk, "candidates") and chunk.candidates:
                        if hasattr(chunk.candidates[0], "content") and chunk.candidates[0].content:
                            if (
                                hasattr(chunk.candidates[0].content, "parts")
                                and chunk.candidates[0].content.parts
                            ):
                                chunk_text = chunk.candidates[0].content.parts[0].text

                    # Metadata for this chunk
                    metadata = {
                        "model": model,
                        "provider": self.provider_name,
                        "chunk_index": total_chunks,
                    }

                    yield chunk_text, metadata

                # Log success
                processing_time = time.time() - start_time
                self.logger.success(
                    "Gemini streaming completion successful",
                    emoji_key="success",
                    model=model,
                    chunks=total_chunks,
                    time=processing_time,
                )

                # Yield final metadata chunk
                yield (
                    "",
                    {
                        "model": model,
                        "provider": self.provider_name,
                        "chunk_index": total_chunks + 1,
                        "processing_time": processing_time,
                        "finish_reason": "stop",  # Gemini doesn't provide this directly
                    },
                )

            except (AttributeError, TypeError) as e:
                # If streaming isn't supported, fall back to non-streaming
                self.logger.warning(
                    f"Streaming not supported for current Gemini API: {e}. Falling back to non-streaming."
                )

                # Call generate_completion and yield the entire result as one chunk
                completion = await self.generate_completion(
                    prompt=prompt,
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=json_mode,
                    **kwargs,
                )

                yield (
                    completion.text,
                    {
                        "model": model,
                        "provider": self.provider_name,
                        "chunk_index": 1,
                        "is_fallback": True,
                    },
                )
                total_chunks = 1

                # Skip the rest of the streaming logic
                raise StopAsyncIteration() from e

        except Exception as e:
            self.logger.error(
                f"Gemini streaming completion failed: {str(e)}", emoji_key="error", model=model
            )
            # Yield error info in final chunk
            yield (
                "",
                {
                    "model": model,
                    "provider": self.provider_name,
                    "chunk_index": total_chunks + 1,
                    "error": f"{type(e).__name__}: {str(e)}",
                    "processing_time": time.time() - start_time,
                    "finish_reason": "error",
                },
            )

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Gemini models.

        Returns:
            List of model information dictionaries
        """
        # Gemini doesn't have a comprehensive models endpoint, so we return a static list
        if self.models_cache:
            return self.models_cache

        models = [
            {
                "id": "gemini-2.0-flash-lite",
                "provider": self.provider_name,
                "description": "Fastest and most efficient Gemini model",
            },
            {
                "id": "gemini-2.0-flash",
                "provider": self.provider_name,
                "description": "Fast Gemini model with good quality",
            },
            {
                "id": "gemini-2.0-pro",
                "provider": self.provider_name,
                "description": "More capable Gemini model",
            },
            {
                "id": "gemini-2.5-pro-preview-03-25",
                "provider": self.provider_name,
                "description": "Most capable Gemini model",
            },
        ]

        # Cache results
        self.models_cache = models

        return models

    def get_default_model(self) -> str:
        """Get the default Gemini model.

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
        return "gemini-2.0-flash-lite"

    async def check_api_key(self) -> bool:
        """Check if the Gemini API key is valid.

        Returns:
            bool: True if API key is valid
        """
        try:
            # Try listing models to validate the API key
            # Use the client's models API to check if API key is valid
            self.client.models.list()
            return True
        except Exception:
            return False
