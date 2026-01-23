# ultimate_mcp_server/providers/anthropic.py
"""Anthropic (Claude) provider implementation."""

import json
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from ultimate_mcp_server.constants import Provider, TaskType  # Import TaskType for logging
from ultimate_mcp_server.core.providers.base import (
    BaseProvider,
    ModelResponse,
)
from ultimate_mcp_server.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("ultimate_mcp_server.providers.anthropic")


class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic (Claude) API."""

    provider_name = Provider.ANTHROPIC.value

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            **kwargs: Additional options (e.g., base_url)
        """
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = kwargs.get("base_url")
        self.models_cache = None
        self.client: Optional[AsyncAnthropic] = None  # Initialize client attribute

    async def initialize(self) -> bool:
        """Initialize the Anthropic client.

        Returns:
            bool: True if initialization was successful
        """
        if not self.api_key:
            self.logger.error("Anthropic API key is not configured.", emoji_key="error")
            return False

        try:
            self.client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            # Skip API call if using a mock key (for tests)
            if "mock-" in self.api_key:
                self.logger.info(
                    "Using mock Anthropic key - skipping API validation", emoji_key="mock"
                )
                # Assume mock initialization is always successful for testing purposes
                self.is_initialized = True
                return True

            # Optional: Add a quick check_api_key() call here if desired,
            # but initialize might succeed even if key is invalid later.
            # is_valid = await self.check_api_key() # This makes initialize slower
            # if not is_valid:
            #     self.logger.error("Anthropic API key appears invalid.", emoji_key="error")
            #     return False

            self.logger.success("Anthropic provider initialized successfully", emoji_key="provider")
            self.is_initialized = True  # Mark as initialized
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to initialize Anthropic provider: {str(e)}",
                emoji_key="error",
                exc_info=True,  # Log traceback for debugging
            )
            self.is_initialized = False
            return False

    async def generate_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = 1024,  # Signature default
        temperature: float = 0.7,
        json_mode: bool = False,
        **kwargs,
    ) -> ModelResponse:
        """Generate a single non-chat completion using Anthropic Claude.

        Args:
            prompt: Text prompt to send to the model.
            messages: List of message dictionaries, alternative to prompt.
            model: Model name to use (e.g., "claude-3-opus-20240229").
            max_tokens: Maximum tokens to generate. Defaults to 1024.
            temperature: Temperature parameter (0.0-1.0).
            json_mode: If True, attempt to guide model towards JSON output (via prompting).
            **kwargs: Additional model-specific parameters (e.g., top_p, system).

        Returns:
            ModelResponse object.
        """
        if not self.client:
            if not await self.initialize():
                raise ConnectionError("Anthropic provider failed to initialize.")

        model = model or self.get_default_model()
        actual_model_name = self.strip_provider_prefix(model)

        # Original logic: Validate that either prompt or messages is provided
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        # Original logic: If messages are provided, use the chat_completion function
        if messages:
            # Ensure all necessary parameters are passed to generate_chat_completion
            # This includes system_prompt if it's in kwargs
            return await self.generate_chat_completion(
                messages=messages,
                model=model,  # Pass original model ID
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,  # Pass json_mode
                **kwargs,  # Pass other kwargs like system, top_p etc.
            )

        # Original logic: Prepare message list for the API from prompt
        # This path is taken if only 'prompt' is provided (and not 'messages')
        current_api_messages = [{"role": "user", "content": prompt}]

        # Original logic: Handle system prompt if passed in kwargs for the simple prompt case
        system_prompt = kwargs.pop("system", None)

        # Original logic: Handle JSON mode for simple prompt case
        if json_mode:
            self.logger.debug(
                "json_mode=True requested for completion (simple prompt), modifying user message for Anthropic."
            )
            # Modify the user message content in current_api_messages
            user_message_idx = -1
            for i, msg in enumerate(current_api_messages):
                if msg["role"] == "user":
                    user_message_idx = i
                    break

            if user_message_idx != -1:
                original_content = current_api_messages[user_message_idx]["content"]
                if (
                    isinstance(original_content, str)
                    and "Please respond with valid JSON" not in original_content
                ):
                    current_api_messages[user_message_idx]["content"] = (
                        f"{original_content}\\nPlease respond ONLY with valid JSON matching the expected schema. Do not include explanations or markdown formatting."
                    )
            else:
                # This case should ideally not happen if prompt is always user role.
                # If it could, one might append a new user message asking for JSON,
                # or include it in system prompt if system_prompt is being constructed here.
                self.logger.warning(
                    "Could not find user message to append JSON instruction for simple prompt case."
                )

        # Prepare API call parameters using max_tokens directly from signature
        api_params = {
            "messages": current_api_messages,
            "model": actual_model_name,
            "max_tokens": max_tokens,  # Uses max_tokens from signature (which defaults to 1024 if not passed)
            "temperature": temperature,
            **kwargs,  # Pass remaining kwargs (like top_p, etc.) that were not popped
        }
        if system_prompt:  # Add system prompt if it was extracted
            api_params["system"] = system_prompt

        # Logging before API call (original style)
        self.logger.info(
            f"Generating completion with Anthropic model {actual_model_name}",
            emoji_key=TaskType.COMPLETION.value,
            prompt_length=len(prompt) if prompt else 0,  # length of prompt if provided
            json_mode_requested=json_mode,
        )

        try:
            response, processing_time = await self.process_with_timer(
                self.client.messages.create, **api_params
            )
        except Exception as e:
            error_message = f"Anthropic API error during completion for model {actual_model_name}: {type(e).__name__}: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            raise ConnectionError(error_message) from e

        if (
            not response.content
            or not isinstance(response.content, list)
            or not hasattr(response.content[0], "text")
        ):
            raise ValueError(f"Unexpected response format from Anthropic API: {response}")
        completion_text = response.content[0].text

        # Post-process if JSON mode was requested (for simple prompt case) - best effort extraction
        if json_mode:  # This json_mode is the original parameter
            original_text_for_json_check = completion_text
            completion_text = self._extract_json_from_text(completion_text)
            if original_text_for_json_check != completion_text:
                self.logger.debug(
                    "Extracted JSON content from Anthropic response post-processing (simple prompt case)."
                )

        result = ModelResponse(
            text=completion_text,
            model=f"{self.provider_name}/{actual_model_name}",
            provider=self.provider_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            processing_time=processing_time,
            raw_response=response.model_dump(),
        )
        result.message = {"role": "assistant", "content": completion_text}

        self.logger.success(
            "Anthropic completion successful",
            emoji_key="success",
            model=result.model,
            tokens={"input": result.input_tokens, "output": result.output_tokens},
            cost=result.cost,
            time=result.processing_time,
        )
        return result

    # --- NEW METHOD ---
    async def generate_chat_completion(
        self,
        messages: List[
            Dict[str, Any]
        ],  # Use Dict for broader compatibility, or specific MessageParam type
        model: Optional[str] = None,
        max_tokens: Optional[int] = 1024,  # Provide a default
        temperature: float = 0.7,
        json_mode: bool = False,  # Add json_mode parameter
        **kwargs,
    ) -> ModelResponse:
        """Generate a chat completion using Anthropic Claude.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
                      Should conform to Anthropic's expected format.
            model: Model name to use (e.g., "claude-3-opus-20240229").
            max_tokens: Maximum tokens to generate. Defaults to 1024.
            temperature: Temperature parameter (0.0-1.0).
            json_mode: If True, guide the model to generate JSON output (via prompt engineering).
            **kwargs: Additional model-specific parameters (e.g., top_p, system).

        Returns:
            ModelResponse object containing the assistant's message.
        """
        if not self.client:
            if not await self.initialize():
                raise ConnectionError("Anthropic provider failed to initialize.")

        model = model or self.get_default_model()
        actual_model_name = self.strip_provider_prefix(model)

        # Handle system prompt extraction
        system_prompt = kwargs.pop("system", None)

        # Process the messages to extract system message and convert to Anthropic format
        processed_messages = []
        extracted_system = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Extract system message if present
            if role == "system":
                if extracted_system is None:  # Take the first system message
                    extracted_system = content
                # Don't add system messages to the processed_messages list
                continue
            elif role in ("user", "assistant"):
                # Keep user and assistant messages
                processed_messages.append({"role": role, "content": content})
            else:
                self.logger.warning(f"Ignoring unsupported message role: {role}")

        # If we found a system message, use it (overrides any system in kwargs)
        if extracted_system is not None:
            system_prompt = extracted_system

        # Process json_mode by modifying system prompt or last user message
        json_mode_requested = json_mode

        if json_mode_requested:
            self.logger.debug(
                "json_mode=True requested for chat completion, implementing via prompt engineering for Anthropic"
            )

            # If we have a system prompt, update it to include JSON instructions
            if system_prompt:
                system_prompt = f"{system_prompt}\n\nIMPORTANT: You must respond ONLY with valid JSON matching the expected schema. Do not include explanations or markdown formatting."
            # Otherwise, if there's at least one user message, modify the last one
            elif processed_messages and any(m.get("role") == "user" for m in processed_messages):
                # Find last user message
                for i in range(len(processed_messages) - 1, -1, -1):
                    if processed_messages[i].get("role") == "user":
                        user_content = processed_messages[i].get("content", "")
                        # Only add JSON instruction if not already present
                        if (
                            "respond with JSON" not in user_content
                            and "respond in JSON" not in user_content
                        ):
                            processed_messages[i]["content"] = (
                                f"{user_content}\n\nPlease respond ONLY with valid JSON. Do not include explanations or markdown formatting."
                            )
                        break
            # If neither system prompt nor user messages to modify, add a system prompt
            else:
                system_prompt = "You must respond ONLY with valid JSON. Do not include explanations or markdown formatting."

        # Prepare API call parameters
        api_params = {
            "messages": processed_messages,
            "model": actual_model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,  # Pass remaining kwargs (like top_p, etc.)
        }
        if system_prompt:
            api_params["system"] = system_prompt

        self.logger.info(
            f"Generating chat completion with Anthropic model {actual_model_name}",
            emoji_key=TaskType.CHAT.value,  # Use enum value
            message_count=len(processed_messages),
            json_mode_requested=json_mode_requested,  # Log if it was requested
        )

        try:
            response, processing_time = await self.process_with_timer(
                self.client.messages.create, **api_params
            )
        except Exception as e:
            error_message = f"Anthropic API error during chat completion for model {actual_model_name}: {type(e).__name__}: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            raise ConnectionError(error_message) from e

        # Extract response content
        if (
            not response.content
            or not isinstance(response.content, list)
            or not hasattr(response.content[0], "text")
        ):
            raise ValueError(f"Unexpected response format from Anthropic API: {response}")
        assistant_content = response.content[0].text

        # Create standardized response including the assistant message
        result = ModelResponse(
            text=assistant_content,  # Keep raw text accessible
            model=f"{self.provider_name}/{actual_model_name}",  # Return prefixed model ID
            provider=self.provider_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            processing_time=processing_time,
            raw_response=response.model_dump(),  # Use model_dump() if Pydantic
        )

        # Add message to result for chat_completion
        result.message = {"role": "assistant", "content": assistant_content}

        # Log success
        self.logger.success(
            "Anthropic chat completion successful",
            emoji_key="success",
            model=result.model,
            tokens={"input": result.input_tokens, "output": result.output_tokens},
            cost=result.cost,
            time=result.processing_time,
        )

        return result

    # --- END NEW METHOD ---

    async def generate_completion_stream(
        self,
        # Keep existing signature: accepts prompt primarily, but also messages/system in kwargs
        prompt: Optional[str] = None,  # Make prompt optional if messages are primary input
        messages: Optional[List[Dict[str, Any]]] = None,  # Allow messages directly
        model: Optional[str] = None,
        max_tokens: Optional[int] = 1024,  # Default max_tokens
        temperature: float = 0.7,
        json_mode: bool = False,  # Accept json_mode flag
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Generate a streaming completion using Anthropic Claude. Handles both prompt and message inputs.

        Args:
            prompt: (Optional) Text prompt (if messages not provided).
            messages: (Optional) List of message dictionaries. Takes precedence over prompt.
            model: Model name to use.
            max_tokens: Maximum tokens to generate. Defaults to 1024.
            temperature: Temperature parameter.
            json_mode: If True, guides model towards JSON (via prompting if using prompt input).
            **kwargs: Additional parameters (system, top_p, etc.).

        Yields:
            Tuple of (text_chunk, metadata).

        Raises:
            ConnectionError: If provider initialization fails or API call fails.
            ValueError: If neither prompt nor messages are provided.
        """
        if not self.client:
            if not await self.initialize():
                raise ConnectionError("Anthropic provider failed to initialize.")

        model = model or self.get_default_model()
        actual_model_name = self.strip_provider_prefix(model)

        # Prepare system prompt if provided in kwargs
        system_prompt = kwargs.pop("system", None)

        # Determine input messages: Use 'messages' if provided, otherwise construct from 'prompt'
        if messages:
            # Process the messages to extract system message and convert to Anthropic format
            processed_messages = []
            extracted_system = None

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                # Extract system message if present
                if role == "system":
                    if extracted_system is None:  # Take the first system message
                        extracted_system = content
                    # Don't add system messages to the processed_messages list
                    continue
                elif role in ("user", "assistant"):
                    # Keep user and assistant messages
                    processed_messages.append({"role": role, "content": content})
                else:
                    self.logger.warning(f"Ignoring unsupported message role in streaming: {role}")

            # If we found a system message, use it (overrides any system in kwargs)
            if extracted_system is not None:
                system_prompt = extracted_system

            input_desc = f"{len(processed_messages)} messages"
        elif prompt:
            # Construct messages from prompt
            processed_messages = [{"role": "user", "content": prompt}]
            input_desc = f"prompt ({len(prompt)} chars)"

            # Apply JSON mode prompt modification ONLY if using prompt input
            if json_mode:
                self.logger.debug(
                    "json_mode=True requested for stream completion, modifying prompt for Anthropic."
                )
                user_message = processed_messages[-1]
                original_content = user_message["content"]
                if "Please respond with valid JSON" not in original_content:
                    user_message["content"] = (
                        f"{original_content}\nPlease respond ONLY with valid JSON matching the expected schema. Do not include explanations or markdown formatting."
                    )
        else:
            raise ValueError(
                "Either 'prompt' or 'messages' must be provided for generate_completion_stream"
            )

        # Apply JSON mode to system prompt if using messages input and json_mode is True
        json_mode_requested = kwargs.pop("json_mode", json_mode)  # Keep track if it was requested
        if json_mode_requested and messages:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\nIMPORTANT: You must respond ONLY with valid JSON matching the expected schema. Do not include explanations or markdown formatting."
            else:
                system_prompt = "You must respond ONLY with valid JSON matching the expected schema. Do not include explanations or markdown formatting."

        # Prepare API call parameters
        params = {
            "model": actual_model_name,
            "messages": processed_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,  # Use the default or provided value
            **kwargs,  # Pass remaining kwargs
        }
        if system_prompt:
            params["system"] = system_prompt

        self.logger.info(
            f"Generating streaming completion with Anthropic model {actual_model_name}",
            emoji_key=self.provider_name,
            input_type=input_desc,
            json_mode_requested=json_mode_requested,
        )

        start_time = time.time()
        total_chunks = 0
        final_input_tokens = 0
        final_output_tokens = 0
        finish_reason = None  # Track finish reason

        try:
            async with self.client.messages.stream(**params) as stream:
                async for chunk in stream:
                    # Extract text delta
                    if chunk.type == "content_block_delta":
                        content = chunk.delta.text
                        total_chunks += 1
                        metadata = {
                            "model": f"{self.provider_name}/{actual_model_name}",
                            "provider": self.provider_name,
                            "chunk_index": total_chunks,
                            "finish_reason": None,  # Not final yet
                        }
                        yield content, metadata

                    # Don't attempt to capture usage from delta chunks - wait for final message

                # Important: Get final tokens from the final message state
                try:
                    final_message = await stream.get_final_message()
                    final_input_tokens = (
                        final_message.usage.input_tokens if hasattr(final_message, "usage") else 0
                    )
                    final_output_tokens = (
                        final_message.usage.output_tokens if hasattr(final_message, "usage") else 0
                    )
                    # Ensure finish_reason is captured from the final message
                    finish_reason = (
                        final_message.stop_reason
                        if hasattr(final_message, "stop_reason")
                        else "unknown"
                    )
                except Exception as e:
                    # If we can't get the final message for any reason, log it but continue
                    self.logger.warning(f"Couldn't get final message stats: {e}")
                    # Estimate token counts based on total characters / avg chars per token
                    char_count = sum(len(m.get("content", "")) for m in processed_messages)
                    final_input_tokens = char_count // 4  # Rough estimate
                    final_output_tokens = total_chunks * 5  # Very rough estimate

            processing_time = time.time() - start_time
            self.logger.success(
                "Anthropic streaming completion successful",
                emoji_key="success",
                model=f"{self.provider_name}/{actual_model_name}",
                chunks=total_chunks,
                tokens={"input": final_input_tokens, "output": final_output_tokens},
                time=processing_time,
                finish_reason=finish_reason,
            )

            # Yield a final empty chunk with aggregated metadata
            final_metadata = {
                "model": f"{self.provider_name}/{actual_model_name}",
                "provider": self.provider_name,
                "chunk_index": total_chunks + 1,
                "input_tokens": final_input_tokens,
                "output_tokens": final_output_tokens,
                "total_tokens": final_input_tokens + final_output_tokens,
                "processing_time": processing_time,
                "finish_reason": finish_reason,
            }
            yield "", final_metadata

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"Anthropic streaming completion failed after {processing_time:.2f}s: {str(e)}",
                emoji_key="error",
                model=f"{self.provider_name}/{actual_model_name}",
                exc_info=True,
            )
            # Yield a final error chunk
            error_metadata = {
                "model": f"{self.provider_name}/{actual_model_name}",
                "provider": self.provider_name,
                "chunk_index": total_chunks + 1,
                "error": f"{type(e).__name__}: {str(e)}",
                "finish_reason": "error",
                "processing_time": processing_time,
            }
            yield "", error_metadata
            # Don't re-raise here, let the caller handle the error chunk

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Anthropic Claude models.

        Returns:
            List of model information dictionaries including the provider prefix.
        """
        # Anthropic doesn't have a list models endpoint, return static list WITH prefix
        # Based on the models defined in constants.py
        static_models = [
            # Define with the full ID including provider prefix
            {
                "id": f"{self.provider_name}/claude-3-7-sonnet-20250219",
                "name": "Claude 3.7 Sonnet",
                "context_window": 200000,
                "input_cost_pmt": 3.0,
                "output_cost_pmt": 15.0,
                "features": ["chat", "completion", "vision", "tool_use"],
            },
            {
                "id": f"{self.provider_name}/claude-3-5-haiku-20241022",
                "name": "Claude 3.5 Haiku",
                "context_window": 200000,
                "input_cost_pmt": 0.80,
                "output_cost_pmt": 4.0,
                "features": ["chat", "completion", "vision"],
            },
            {
                "id": f"{self.provider_name}/claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "context_window": 200000,
                "input_cost_pmt": 15.0,
                "output_cost_pmt": 75.0,
                "features": ["chat", "completion", "vision"],
            },
        ]

        # Simple caching (optional, as list is static)
        if not self.models_cache:
            self.models_cache = static_models
        return self.models_cache

    def get_default_model(self) -> str:
        """Get the default Anthropic model ID (including provider prefix).

        Returns:
            Default model ID string (e.g., "anthropic/claude-3-5-haiku-20241022").
        """
        # Try getting from config first
        from ultimate_mcp_server.config import get_config

        default_model_id = f"{self.provider_name}/claude-3-5-haiku-20241022"  # Hardcoded default

        try:
            config = get_config()
            # Access nested provider config safely
            provider_config = config.providers.get(self.provider_name) if config.providers else None
            if provider_config and provider_config.default_model:
                # Ensure the configured default includes the prefix
                configured_default = provider_config.default_model
                if not configured_default.startswith(f"{self.provider_name}/"):
                    self.logger.warning(
                        f"Configured default model '{configured_default}' for Anthropic is missing the provider prefix. Using hardcoded default: {default_model_id}"
                    )
                    return default_model_id
                else:
                    return configured_default
        except (ImportError, AttributeError, TypeError) as e:
            self.logger.debug(
                f"Could not retrieve default model from config ({e}), using hardcoded default."
            )

        return default_model_id

    async def check_api_key(self) -> bool:
        """Check if the Anthropic API key is valid by making a minimal request.

        Returns:
            bool: True if API key allows a basic request.
        """
        if not self.client:
            self.logger.warning("Cannot check API key: Anthropic client not initialized.")
            # Attempt initialization first
            if not await self.initialize():
                return False  # Initialization failed, key likely invalid or other issue
            # If initialize succeeded but client still None (e.g., mock key path)
            if not self.client:
                return True  # Assume mock key is 'valid' for testing

        try:
            # Use the *unprefixed* default model name for the check
            default_model_unprefixed = self.strip_provider_prefix(self.get_default_model())
            await self.client.messages.create(
                model=default_model_unprefixed,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1,
            )
            self.logger.info("Anthropic API key validation successful.")
            return True
        except Exception as e:
            self.logger.warning(f"Anthropic API key validation failed: {type(e).__name__}")
            return False

    def strip_provider_prefix(self, model_id: str) -> str:
        """Removes the provider prefix (e.g., 'anthropic/') from a model ID."""
        prefix = f"{self.provider_name}/"
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
        # Handle ':' separator as well for backward compatibility if needed
        alt_prefix = f"{self.provider_name}:"
        if model_id.startswith(alt_prefix):
            return model_id[len(alt_prefix) :]
        return model_id  # Return original if no prefix found

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

        # Extract JSON from code blocks - most common Anthropic pattern
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

    async def process_with_timer(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Helper to time an async function call."""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
