"""Text completion tools for Ultimate MCP Server."""

import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from ultimate_mcp_server.constants import Provider, TaskType
from ultimate_mcp_server.core.providers.base import get_provider, parse_model_string
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError
from ultimate_mcp_server.services.cache import with_cache
from ultimate_mcp_server.tools.base import with_error_handling, with_retry, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.completion")

# --- Tool Functions (Standalone, Decorated) ---


@with_tool_metrics
@with_error_handling
async def generate_completion(
    prompt: str,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    stream: bool = False,
    json_mode: bool = False,
    additional_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates a single, complete text response for a given prompt (non-streaming).

    Use this tool for single-turn tasks where the entire response is needed at once,
    such as answering a question, summarizing text, translating, or classifying content.
    It waits for the full response from the LLM before returning.

    If you need the response to appear incrementally (e.g., for user interfaces or long generations),
    use the `stream_completion` tool instead.

    Args:
        prompt: The input text prompt for the LLM.
        provider: The name of the LLM provider (e.g., "openai", "anthropic", "gemini"). Defaults to "openai".
                  Use `list_models` or `get_provider_status` to see available providers.
        model: The specific model ID (e.g., "openai/gpt-4.1-mini", "anthropic/claude-3-5-haiku-20241022").
               If None, the provider's default model is used. Use `list_models` to find available IDs.
        max_tokens: (Optional) Maximum number of tokens to generate in the response.
        temperature: (Optional) Controls response randomness (0.0=deterministic, 1.0=creative). Default 0.7.
        stream: Must be False for this tool. Set to True to trigger an error directing to `stream_completion`.
        json_mode: (Optional) When True, instructs the model to return a valid JSON response. Default False.
                   Note: Support and behavior varies by provider.
        additional_params: (Optional) Dictionary of additional provider-specific parameters (e.g., `{"top_p": 0.9}`).

    Returns:
        A dictionary containing the full completion and metadata:
        {
            "text": "The generated completion text...",
            "model": "provider/model-used",
            "provider": "provider-name",
            "tokens": {
                "input": 15,
                "output": 150,
                "total": 165
            },
            "cost": 0.000123, # Estimated cost in USD
            "processing_time": 1.23, # Execution time in seconds
            "success": true
        }

    Raises:
        ToolInputError: If `stream` is set to True.
        ProviderError: If the provider is unavailable or the LLM request fails.
        ToolError: For other internal errors.
    """
    # Streaming not supported for this endpoint
    if stream:
        raise ToolInputError(
            "Streaming is not supported for `generate_completion`. Use the `stream_completion` tool instead.",
            param_name="stream",
            provided_value=stream,
        )

    start_time = time.time()

    # Check if model contains a provider prefix (e.g., "openai/gpt-4.1-mini")
    if model:
        extracted_provider, extracted_model = parse_model_string(model)
        if extracted_provider:
            provider = extracted_provider  # Override provider with the one from the model string
            model = extracted_model  # Use the model name without the provider prefix
            logger.debug(
                f"Using provider '{provider}' and model '{model}' extracted from model string"
            )

    # Get provider instance
    try:
        # Use provider name directly, get_provider handles splitting if needed
        provider_instance = await get_provider(provider)
    except Exception as e:
        raise ProviderError(
            f"Failed to initialize provider '{provider}': {str(e)}", provider=provider, cause=e
        ) from e

    # Set default additional params
    additional_params = additional_params or {}

    # Conditionally construct parameters for the provider call
    params_for_provider = {
        "prompt": prompt,
        "model": model,  # model here is already stripped of provider prefix if applicable
        "temperature": temperature,
        "json_mode": json_mode,
        # messages will be handled by chat_completion, this is for simple completion
    }
    if max_tokens is not None:
        params_for_provider["max_tokens"] = max_tokens

    # Merge any other additional_params, ensuring they don't overwrite core params already set
    # or ensuring that additional_params are provider-specific and don't conflict.
    # A safer merge would be params_for_provider.update({k: v for k, v in additional_params.items() if k not in params_for_provider})
    # However, the current **additional_params likely intends to override if keys match, so we keep that behavior for now
    # but apply it to the conditionally built dict.
    final_provider_params = {**params_for_provider, **additional_params}

    try:
        # Generate completion
        result = await provider_instance.generate_completion(**final_provider_params)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Log success
        logger.success(
            f"Completion generated successfully with {provider}/{result.model}",
            emoji_key=TaskType.COMPLETION.value,
            tokens={"input": result.input_tokens, "output": result.output_tokens},
            cost=result.cost,
            time=processing_time,
        )

        # Return standardized result
        return {
            "text": result.text,
            "model": result.model,  # Return the actual model used (might differ from input if default)
            "provider": provider,
            "tokens": {
                "input": result.input_tokens,
                "output": result.output_tokens,
                "total": result.total_tokens,
            },
            "cost": result.cost,
            "processing_time": processing_time,
            "success": True,
        }

    except Exception as e:
        # Convert to provider error
        # Use the potentially prefixed model name in the error context
        error_model = model or f"{provider}/default"
        raise ProviderError(
            f"Completion generation failed for model '{error_model}': {str(e)}",
            provider=provider,
            model=error_model,
            cause=e,
        ) from e


@with_tool_metrics
@with_error_handling
async def stream_completion(
    prompt: str,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    json_mode: bool = False,
    additional_params: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Generates a text completion for a prompt and streams the response chunk by chunk.

    Use this tool when you need to display the LLM's response progressively as it's generated,
    improving perceived responsiveness for users, especially for longer outputs.

    If you need the entire response at once, use `generate_completion`.

    Args:
        prompt: The input text prompt for the LLM.
        provider: The name of the LLM provider (e.g., "openai", "anthropic", "gemini"). Defaults to "openai".
                  Use `list_models` or `get_provider_status` to see available providers.
        model: The specific model ID (e.g., "openai/gpt-4.1-mini", "anthropic/claude-3-5-haiku-20241022").
               If None, the provider's default model is used. Use `list_models` to find available IDs.
        max_tokens: (Optional) Maximum number of tokens to generate in the response.
        temperature: (Optional) Controls response randomness (0.0=deterministic, 1.0=creative). Default 0.7.
        json_mode: (Optional) When True, instructs the model to return a valid JSON response. Default False.
        additional_params: (Optional) Dictionary of additional provider-specific parameters (e.g., `{"top_p": 0.9}`).

    Yields:
        A stream of dictionary chunks. Each chunk contains:
        {
            "text": "The incremental piece of generated text...",
            "chunk_index": 1,          # Sequence number of this chunk (starts at 1)
            "provider": "provider-name",
            "model": "provider/model-used",
            "finish_reason": null,    # Reason generation stopped (e.g., "stop", "length"), null until the end
            "finished": false         # True only for the very last yielded dictionary
        }
        The *final* yielded dictionary will have `finished: true` and may also contain aggregate metadata:
        {
            "text": "",               # Final chunk might be empty
            "chunk_index": 10,
            "provider": "provider-name",
            "model": "provider/model-used",
            "finish_reason": "stop",
            "finished": true,
            "full_text": "The complete generated text...", # Full response concatenated
            "processing_time": 5.67,                   # Total time in seconds
            "tokens": { "input": ..., "output": ..., "total": ... }, # Final token counts
            "cost": 0.000543                          # Final estimated cost
            "error": null                             # Error message if one occurred during streaming
        }

    Raises:
        ProviderError: If the provider is unavailable or the LLM stream request fails initially.
                       Errors during the stream yield an error message in the final chunk.
    """
    start_time = time.time()

    # Add MCP annotations for audience and priority
    # annotations = {  # noqa: F841
    #     "audience": ["assistant", "user"],  # Useful for both assistant and user
    #     "priority": 0.8  # High priority but not required (generate_completion is the primary tool)
    # }

    # Check if model contains a provider prefix (e.g., "openai/gpt-4.1-mini")
    if model:
        extracted_provider, extracted_model = parse_model_string(model)
        if extracted_provider:
            provider = extracted_provider  # Override provider with the one from the model string
            model = extracted_model  # Use the model name without the provider prefix
            logger.debug(
                f"Using provider '{provider}' and model '{model}' extracted from model string"
            )

    # Get provider instance
    try:
        provider_instance = await get_provider(provider)
    except Exception as e:
        logger.error(
            f"Failed to initialize provider '{provider}': {str(e)}",
            emoji_key="error",
            provider=provider,
        )
        # Yield a single error chunk if provider init fails
        yield {
            "error": f"Failed to initialize provider '{provider}': {str(e)}",
            "text": None,
            "finished": True,
            "provider": provider,
            "model": model,
        }
        return

    # Set default additional params
    additional_params = additional_params or {}

    logger.info(
        f"Starting streaming completion with {provider}",
        emoji_key=TaskType.COMPLETION.value,
        prompt_length=len(prompt),
        json_mode_requested=json_mode,  # Log the request
    )

    chunk_count = 0
    full_text = ""
    final_metadata = {}  # To store final metadata like model, cost etc.
    error_during_stream = None
    actual_model_used = model  # Keep track of the actual model used

    try:
        # Get stream, passing json_mode directly
        stream = provider_instance.generate_completion_stream(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,  # Pass the flag here
            **additional_params,
        )

        async for chunk, metadata in stream:
            chunk_count += 1
            full_text += chunk
            final_metadata.update(metadata)  # Keep track of latest metadata
            actual_model_used = metadata.get(
                "model", actual_model_used
            )  # Update if metadata provides it

            # Yield chunk with metadata
            yield {
                "text": chunk,
                "chunk_index": chunk_count,
                "provider": provider,
                "model": actual_model_used,
                "finish_reason": metadata.get("finish_reason"),
                "finished": False,
            }

    except Exception as e:
        error_during_stream = (
            f"Error during streaming after {chunk_count} chunks: {type(e).__name__}: {str(e)}"
        )
        logger.error(
            f"Error during streaming completion with {provider}/{actual_model_used or 'default'}: {error_during_stream}",
            emoji_key="error",
        )
        # Don't return yet, yield the final chunk with the error

    # --- Final Chunk ---
    processing_time = time.time() - start_time

    # Log completion (success or failure based on error_during_stream)
    log_level = logger.error if error_during_stream else logger.success
    log_message = (
        f"Streaming completion finished ({chunk_count} chunks)"
        if not error_during_stream
        else f"Streaming completion failed after {chunk_count} chunks"
    )

    log_level(
        log_message,
        emoji_key="error" if error_during_stream else "success",
        provider=provider,
        model=actual_model_used,
        tokens={
            "input": final_metadata.get("input_tokens"),
            "output": final_metadata.get("output_tokens"),
        },
        cost=final_metadata.get("cost"),
        time=processing_time,
        error=error_during_stream,
    )

    # Yield the final aggregated chunk
    yield {
        "text": "",  # No new text in the final summary chunk
        "chunk_index": chunk_count + 1,
        "provider": provider,
        "model": actual_model_used,
        "finish_reason": final_metadata.get("finish_reason"),
        "finished": True,
        "full_text": full_text,
        "processing_time": processing_time,
        "tokens": {
            "input": final_metadata.get("input_tokens"),
            "output": final_metadata.get("output_tokens"),
            "total": final_metadata.get("total_tokens"),
        },
        "cost": final_metadata.get("cost"),
        "error": error_during_stream,
    }


@with_tool_metrics
@with_error_handling
async def generate_completion_stream(
    prompt: str,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    json_mode: bool = False,
    additional_params: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Generates a text response in a streaming fashion for a given prompt.

    Use this tool when you want to display the response as it's being generated
    without waiting for the entire response to be completed. It yields chunks of text
    as they become available, allowing for more interactive user experiences.

    Args:
        prompt: The text prompt to send to the LLM.
        provider: The LLM provider to use (default: "openai").
        model: The specific model to use (if None, uses provider's default).
        max_tokens: Maximum tokens to generate in the response.
        temperature: Controls randomness in the output (0.0-1.0).
        json_mode: Whether to request JSON formatted output from the model.
        additional_params: Additional provider-specific parameters.

    Yields:
        Dictionary containing the generated text chunk and metadata:
        {
            "text": str,           # The text chunk
            "metadata": {...},     # Additional information about the generation
            "done": bool           # Whether this is the final chunk
        }

    Raises:
        ToolError: If an error occurs during text generation.
    """
    # Initialize variables to track metrics
    start_time = time.time()

    try:
        # Get provider instance
        provider_instance = await get_provider(provider)
        if not provider_instance:
            raise ValueError(f"Invalid provider: {provider}")

        # Add json_mode to additional_params if specified
        params = additional_params.copy() if additional_params else {}
        if json_mode:
            params["json_mode"] = True

        # Stream the completion
        async for chunk, metadata in provider_instance.generate_completion_stream(
            prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature, **params
        ):
            # Calculate elapsed time for each chunk
            elapsed_time = time.time() - start_time

            # Include additional metadata with each chunk
            response = {
                "text": chunk,
                "metadata": {
                    **metadata,
                    "elapsed_time": elapsed_time,
                },
                "done": metadata.get("finish_reason") is not None,
            }

            yield response

    except Exception as e:
        logger.error(f"Error in generate_completion_stream: {str(e)}", exc_info=True)
        raise ToolError(f"Failed to generate streaming completion: {str(e)}") from e


@with_cache(ttl=24 * 60 * 60)  # Cache results for 24 hours
@with_tool_metrics
@with_retry(max_retries=2, retry_delay=1.0)  # Retry up to 2 times on failure
@with_error_handling
async def chat_completion(
    messages: List[Dict[str, Any]],
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
    additional_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates a response within a conversational context (multi-turn chat).

    Use this tool for chatbot interactions, instruction following, or any task requiring
    the LLM to consider previous turns in a conversation. It takes a list of messages
    (user, assistant, system roles) as input.

    This tool automatically retries on transient failures and caches results for identical requests
    (based on messages, model, etc.) for 24 hours to save costs and time.
    Streaming is NOT supported; this tool returns the complete chat response at once.

    Args:
        messages: A list of message dictionaries representing the conversation history.
                  Each dictionary must have "role" ("user", "assistant", or "system") and "content" (string).
                  Example: `[{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]`
        provider: The name of the LLM provider (e.g., "openai", "anthropic", "gemini"). Defaults to "openai".
        model: The specific model ID (e.g., "openai/gpt-4o", "anthropic/claude-3-7-sonnet-20250219").
               If None, the provider's default model is used. Use `list_models` to find available IDs.
        max_tokens: (Optional) Maximum number of tokens for the *assistant's* response.
        temperature: (Optional) Controls response randomness (0.0=deterministic, 1.0=creative). Default 0.7.
        system_prompt: (Optional) An initial system message to guide the model's behavior (e.g., persona, instructions).
                       If provided, it's effectively prepended to the `messages` list as a system message.
        json_mode: (Optional) Request structured JSON output from the LLM. Default False.
        additional_params: (Optional) Dictionary of additional provider-specific parameters (e.g., `{"top_p": 0.9}`).

    Returns:
        A dictionary containing the assistant's response message and metadata:
        {
            "message": {
                "role": "assistant",
                "content": "The assistant's generated response..."
            },
            "model": "provider/model-used",
            "provider": "provider-name",
            "tokens": {
                "input": 55,  # Includes all input messages
                "output": 120, # Assistant's response only
                "total": 175
            },
            "cost": 0.000150, # Estimated cost in USD
            "processing_time": 2.50, # Execution time in seconds
            "cached_result": false, # True if the result was served from cache
            "success": true
        }

    Raises:
        ToolInputError: If the `messages` format is invalid.
        ProviderError: If the provider is unavailable or the LLM request fails (after retries).
        ToolError: For other internal errors.
    """
    start_time = time.time()

    # Validate messages format
    if not isinstance(messages, list) or not all(
        isinstance(m, dict) and "role" in m and "content" in m for m in messages
    ):
        raise ToolInputError(
            "Invalid messages format. Must be a list of dictionaries, each with 'role' and 'content'.",
            param_name="messages",
            provided_value=messages,
        )

    # Prepend system prompt if provided
    if system_prompt:
        # Avoid modifying the original list if called multiple times
        processed_messages = [{"role": "system", "content": system_prompt}] + messages
    else:
        processed_messages = messages

    # Check if model contains a provider prefix (e.g., "openai/gpt-4.1-mini")
    if model:
        extracted_provider, extracted_model = parse_model_string(model)
        if extracted_provider:
            provider = extracted_provider  # Override provider with the one from the model string
            model = extracted_model  # Use the model name without the provider prefix
            logger.debug(
                f"Using provider '{provider}' and model '{model}' extracted from model string"
            )

    # Get provider instance
    try:
        provider_instance = await get_provider(provider)
    except Exception as e:
        raise ProviderError(
            f"Failed to initialize provider '{provider}': {str(e)}", provider=provider, cause=e
        ) from e

    additional_params = additional_params or {}
    # Add json_mode to additional params if specified
    if json_mode:
        additional_params["json_mode"] = True

    try:
        result = await provider_instance.generate_completion(
            messages=processed_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **additional_params,
        )

        processing_time = time.time() - start_time

        logger.success(
            f"Chat completion generated successfully with {provider}/{result.model}",
            emoji_key=TaskType.CHAT.value,
            tokens={"input": result.input_tokens, "output": result.output_tokens},
            cost=result.cost,
            time=processing_time,
        )

        return {
            "message": result.message.dict()
            if hasattr(result.message, "dict")
            else result.message,  # Return message as dict
            "model": result.model,
            "provider": provider,
            "tokens": {
                "input": result.input_tokens,
                "output": result.output_tokens,
                "total": result.total_tokens,
            },
            "cost": result.cost,
            "processing_time": processing_time,
            # Note: cached_result is automatically added by the @with_cache decorator if applicable
            "success": True,
        }

    except Exception as e:
        error_model = model or f"{provider}/default"
        # Check if the exception has the model attribute, otherwise use the determined error_model
        error_model_from_exception = getattr(e, "model", None)
        final_error_model = error_model_from_exception or error_model

        raise ProviderError(
            f"Chat completion generation failed for model '{final_error_model}': {str(e)}",
            provider=provider,
            model=final_error_model,
            cause=e,
        ) from e


@with_cache(ttl=7 * 24 * 60 * 60)  # Cache results for 7 days
@with_tool_metrics
@with_error_handling  # Error handling should be used
async def multi_completion(
    prompt: str,
    providers: List[Dict[str, Any]],
    max_concurrency: int = 3,
    timeout: Optional[float] = 30.0,
) -> Dict[str, Any]:
    """Generates completions for the same prompt from multiple LLM providers/models concurrently.

    Use this tool to compare responses, latency, or cost across different models for a specific prompt.
    It runs requests in parallel up to `max_concurrency`.

    Results are cached for 7 days based on the prompt and provider configurations.

    Args:
        prompt: The input text prompt to send to all specified models.
        providers: A list of dictionaries, each specifying a provider and model configuration.
                   Example: `[{"provider": "openai", "model": "gpt-4.1-mini"}, {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "max_tokens": 50}]`
                   Each dict must contain at least "provider". "model" is optional (uses provider default).
                   Other valid parameters for `generate_completion` (like `max_tokens`, `temperature`) can be included.
        max_concurrency: (Optional) Maximum number of provider requests to run in parallel. Default 3.
        timeout: (Optional) Maximum time in seconds to wait for each individual provider request. Default 30.0.
                 Requests exceeding this time will result in a timeout error for that specific provider.

    Returns:
        A dictionary containing results from each provider, along with aggregate statistics:
        {
            "results": {
                "openai/gpt-4.1-mini": {        # Keyed by provider/model
                    "text": "Response from OpenAI...",
                    "model": "openai/gpt-4.1-mini",
                    "provider": "openai",
                    "tokens": { ... },
                    "cost": 0.000123,
                    "processing_time": 1.5,
                    "success": true,
                    "error": null
                },
                "anthropic/claude-3-5-haiku-20241022": {
                    "text": null,
                    "model": "anthropic/claude-3-5-haiku-20241022",
                    "provider": "anthropic",
                    "tokens": null,
                    "cost": 0.0,
                    "processing_time": 30.0,
                    "success": false,
                    "error": "Request timed out after 30.0 seconds"
                },
                ...
            },
            "aggregate_stats": {
                "total_requests": 2,
                "successful_requests": 1,
                "failed_requests": 1,
                "total_cost": 0.000123,
                "total_processing_time": 30.1, # Total wall time for the concurrent execution
                "average_processing_time": 1.5 # Average time for successful requests
            },
            "cached_result": false # True if the entire multi_completion result was cached
        }

    Raises:
        ToolInputError: If the `providers` list format is invalid.
        ToolError: For other internal errors during setup.
                 Individual provider errors are captured within the "results" dictionary.
    """
    start_time = time.time()

    # Validate providers format
    if not isinstance(providers, list) or not all(
        isinstance(p, dict) and "provider" in p for p in providers
    ):
        raise ToolInputError(
            "Invalid providers format. Must be a list of dictionaries, each with at least a 'provider' key.",
            param_name="providers",
            provided_value=providers,
        )

    results = {}
    tasks = []
    semaphore = asyncio.Semaphore(max_concurrency)
    total_cost = 0.0
    successful_requests = 0
    failed_requests = 0
    successful_times = []

    async def process_provider(provider_config):
        nonlocal total_cost, successful_requests, failed_requests, successful_times
        provider_name = provider_config.get("provider")
        model_name = provider_config.get("model")
        # Create a unique key for results dictionary, handling cases where model might be None initially
        result_key = f"{provider_name}/{model_name or 'default'}"

        async with semaphore:
            provider_start_time = time.time()
            error_message = None
            result_data = None
            actual_model_used = model_name  # Store the actual model reported by the result

            try:
                # Extract specific params for generate_completion
                completion_params = {
                    k: v for k, v in provider_config.items() if k not in ["provider"]
                }
                completion_params["prompt"] = prompt  # Add the common prompt

                logger.debug(
                    f"Calling generate_completion for {provider_name} / {model_name or 'default'}..."
                )

                # Call generate_completion with timeout
                completion_task = generate_completion(provider=provider_name, **completion_params)
                result_data = await asyncio.wait_for(completion_task, timeout=timeout)

                provider_processing_time = time.time() - provider_start_time

                if result_data and result_data.get("success"):
                    cost = result_data.get("cost", 0.0)
                    total_cost += cost
                    successful_requests += 1
                    successful_times.append(provider_processing_time)
                    actual_model_used = result_data.get("model")  # Get the actual model used
                    logger.info(f"Success from {result_key} in {provider_processing_time:.2f}s")
                else:
                    failed_requests += 1
                    error_message = (
                        result_data.get("error", "Unknown error during completion")
                        if isinstance(result_data, dict)
                        else "Invalid result format"
                    )
                    logger.warning(f"Failure from {result_key}: {error_message}")

            except asyncio.TimeoutError:
                provider_processing_time = time.time() - provider_start_time
                failed_requests += 1
                error_message = f"Request timed out after {timeout:.1f} seconds"
                logger.warning(f"Timeout for {result_key} after {timeout:.1f}s")
            except ProviderError as pe:
                provider_processing_time = time.time() - provider_start_time
                failed_requests += 1
                error_message = f"ProviderError: {str(pe)}"
                logger.warning(f"ProviderError for {result_key}: {str(pe)}")
                actual_model_used = pe.model  # Get model from exception if available
            except Exception as e:
                provider_processing_time = time.time() - provider_start_time
                failed_requests += 1
                error_message = f"Unexpected error: {type(e).__name__}: {str(e)}"
                logger.error(f"Unexpected error for {result_key}: {e}", exc_info=True)

            # Store result or error
            # Use the potentially updated result_key
            results[result_key] = {
                "text": result_data.get("text") if result_data else None,
                "model": actual_model_used,  # Use the actual model name from result or exception
                "provider": provider_name,
                "tokens": result_data.get("tokens") if result_data else None,
                "cost": result_data.get("cost", 0.0) if result_data else 0.0,
                "processing_time": provider_processing_time,
                "success": error_message is None,
                "error": error_message,
            }

    # Create tasks
    for config in providers:
        task = asyncio.create_task(process_provider(config))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    total_processing_time = time.time() - start_time
    average_processing_time = (
        sum(successful_times) / len(successful_times) if successful_times else 0.0
    )

    logger.info(
        f"Multi-completion finished. Success: {successful_requests}, Failed: {failed_requests}, Total Cost: ${total_cost:.6f}, Total Time: {total_processing_time:.2f}s",
        emoji_key="info",
    )

    return {
        "results": results,
        "aggregate_stats": {
            "total_requests": len(providers),
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "total_cost": total_cost,
            "total_processing_time": total_processing_time,
            "average_processing_time": average_processing_time,
        },
        # Note: cached_result is added by decorator
    }
