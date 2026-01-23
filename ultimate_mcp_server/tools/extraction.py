"""Advanced extraction tools for Ultimate MCP Server.

This module provides tools for extracting structured data (JSON, tables, key-value pairs, code)
from unstructured or semi-structured text using LLMs.
"""

import asyncio
import json
import re  # Added for code extraction
import time
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

from ultimate_mcp_server.constants import Provider

# Removed CompletionRequest import as not directly used by standalone functions
from ultimate_mcp_server.exceptions import ProviderError, ToolInputError
from ultimate_mcp_server.tools.base import BaseTool, with_error_handling, with_tool_metrics

# Import the standardized completion tool
from ultimate_mcp_server.tools.completion import generate_completion
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.extraction")


def _extract_and_parse_json(text: str) -> Tuple[Any, bool, Optional[str]]:
    """
    Robust utility to extract and parse JSON from text, handling various formats and edge cases.

    Args:
        text: The text that may contain JSON.

    Returns:
        Tuple of (parsed_data, success_flag, error_message)
    """
    # Start with a clean slate
    text = text.strip()
    error_message = None

    # Try a series of increasingly aggressive extraction techniques
    extraction_methods = [
        # Method 1: Direct parsing if it's already valid JSON
        lambda t: json.loads(t),
        # Method 2: Extract JSON using regex for common patterns
        lambda t: json.loads(
            re.search(r"(?s)(?:```(?:json)?\s*)?({[\s\S]*?}|\[[\s\S]*?\])(?:\s*```)?", t)
            .group(1)
            .strip()
        ),
        # Method 3: Remove markdown fences and try again
        lambda t: json.loads(re.sub(r"```(?:json)?\s*|\s*```", "", t).strip()),
        # Method 4: Fix common JSON syntax errors and try again
        lambda t: json.loads(_fix_common_json_errors(t)),
        # Method 5: Use ast.literal_eval as a fallback for Python literals
        lambda t: _safe_literal_eval(t),
    ]

    # Try each method in sequence until one works
    for i, method in enumerate(extraction_methods):
        try:
            result = method(text)
            logger.debug(f"Successfully parsed JSON using method {i + 1}")
            return result, True, None
        except Exception as e:
            # Continue to next method on failure
            if i == len(extraction_methods) - 1:  # Last method
                error_message = f"All JSON parsing methods failed. Last error: {str(e)}"

    return None, False, error_message


def _fix_common_json_errors(json_str: str) -> str:
    """
    Fix common JSON syntax errors found in LLM outputs.

    Args:
        json_str: The JSON string to fix

    Returns:
        Corrected JSON string
    """
    # Remove any text before the first '{' or '['
    json_str = re.sub(r"^.*?([{\[])", r"\1", json_str, flags=re.DOTALL)

    # Remove any text after the last '}' or ']'
    json_str = re.sub(r"([}\]])[^}\]]*$", r"\1", json_str, flags=re.DOTALL)

    # Fix missing quotes around keys
    json_str = re.sub(r"(\s*)(\w+)(\s*):", r'\1"\2"\3:', json_str)

    # Fix trailing commas in arrays
    json_str = re.sub(r",(\s*[\]}])", r"\1", json_str)

    # Fix missing commas between elements
    json_str = re.sub(r'(["}\]])(\s*)(["{\[])', r"\1,\2\3", json_str)

    return json_str


def _safe_literal_eval(text: str) -> Any:
    """
    Safely evaluate a string containing a Python literal.

    Args:
        text: The text containing a Python literal

    Returns:
        The evaluated Python object

    Raises:
        SyntaxError: If the text cannot be parsed as a Python literal
    """
    import ast

    # Remove any text before the first '{' or '['
    text = re.sub(r"^.*?([{\[])", r"\1", text, flags=re.DOTALL)

    # Remove any text after the last '}' or ']'
    text = re.sub(r"([}\]])[^}\]]*$", r"\1", text, flags=re.DOTALL)

    return ast.literal_eval(text)


@with_tool_metrics
@with_error_handling
async def extract_json(
    text: str,
    json_schema: Optional[Dict] = None,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    validate_output: bool = True,
    # Removed ctx=None
) -> Dict[str, Any]:
    """Extracts structured data formatted as JSON from within a larger text body.

    Use this tool when the input text contains a JSON object or list (potentially embedded
    within other text or markdown code fences) that needs to be isolated and parsed.
    Optionally validates the extracted JSON against a provided schema.

    Args:
        text: The input text potentially containing an embedded JSON object or list.
        json_schema: (Optional) A JSON schema (as a Python dictionary) to validate the extracted
                     JSON against. If validation fails, the error is included in the result.
        provider: The name of the LLM provider (e.g., "openai"). Defaults to "openai".
                  Providers supporting JSON mode (like OpenAI) are recommended for reliability.
        model: The specific model ID (e.g., "openai/gpt-4.1-mini"). Uses provider default if None.
        validate_output: (Optional) If True (default) and `json_schema` is provided, validates
                         the extracted data against the schema.

    Returns:
        A dictionary containing the extraction results:
        {
            "data": { ... } | [ ... ] | null, # The extracted JSON data (or null if extraction/parsing failed).
            "validation_result": {             # Included if json_schema provided & validate_output=True
                "valid": true | false,
                "errors": [ "Validation error message..." ] # List of errors if not valid
            } | null,
            "raw_text": "...",                # Included if JSON parsing failed
            "model": "provider/model-used",
            "provider": "provider-name",
            "tokens": { ... },
            "cost": 0.000045,
            "processing_time": 1.8,
            "success": true | false,
            "error": "Error message if success is false"
        }

    Raises:
        ProviderError: If the provider/LLM fails.
        ToolError: For other internal errors.
    """
    start_time = time.time()

    if not text or not isinstance(text, str):
        raise ToolInputError(
            "Input 'text' must be a non-empty string.", param_name="text", provided_value=text
        )

    try:
        # Check if there's already valid JSON in the input text
        extracted_data, success, error_message = _extract_and_parse_json(text)

        # If we found valid JSON in the input, return it right away
        if success:
            logger.info("Found and extracted valid JSON directly from input text")
            return {
                "data": extracted_data,
                "validation_result": None,  # No validation done for direct extraction
                "raw_text": None,
                "model": "direct-extraction",  # No model used
                "provider": "direct-extraction",  # No provider used
                "tokens": {"input": len(text), "output": 0, "total": len(text)},
                "cost": 0.0,  # No cost for direct extraction
                "processing_time": time.time() - start_time,
                "success": True,
                "error": None,
            }

        # Prepare model ID based on provider format
        effective_model = model
        # Ensure model ID includes provider prefix if not already present
        if model and provider not in model:
            effective_model = f"{provider}/{model}"

        schema_description = (
            f"The extracted JSON should conform to this JSON schema:\n```json\n{json.dumps(json_schema, indent=2)}\n```\n"
            if json_schema
            else ""
        )
        # Improved prompt asking the LLM to identify and extract the JSON
        prompt = (
            f"Identify and extract the primary JSON object or list embedded within the following text. "
            f"{schema_description}Focus on extracting only the JSON data structure itself, removing any surrounding text or markdown fences. "
            f"Text:\n```\n{text}\n```\nExtracted JSON:"
        )

        # Use JSON mode if supported by the provider (e.g., OpenAI)
        additional_params = {}
        if provider == Provider.OPENAI.value:
            additional_params["response_format"] = {"type": "json_object"}

        # Use the standardized completion tool instead of direct provider call
        completion_result = await generate_completion(
            prompt=prompt,
            model=effective_model,
            provider=provider,
            temperature=0.0,  # Low temp for precise extraction
            max_tokens=4000,  # Allow for large JSON objects
            additional_params=additional_params,
        )

        # Extract data from the standardized result format
        processing_time = completion_result.get("processing_time", time.time() - start_time)
        actual_model_used = completion_result.get("model", effective_model)
        raw_text_output = completion_result.get("text", "").strip()
        token_info = completion_result.get("tokens", {})
        cost = completion_result.get("cost", 0.0)
        tool_success = completion_result.get("success", False)

        # If the tool call failed, propagate the error
        if not tool_success:
            error_message = completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"JSON extraction failed: {error_message}",
                provider=provider,
                model=actual_model_used,
            )

        # Use our robust parsing function
        extracted_data, success, error_message = _extract_and_parse_json(raw_text_output)
        validation_result = None

        # Validate against schema if requested and extraction succeeded
        if success and json_schema and validate_output:
            validation_result = {"valid": True, "errors": []}
            try:
                jsonschema.validate(instance=extracted_data, schema=json_schema)
                logger.debug("JSON validated successfully against schema.")
            except jsonschema.exceptions.ValidationError as e:
                validation_result = {"valid": False, "errors": [str(e)]}
                logger.warning(f"JSON validation failed: {e}")
                # Keep success=True as extraction worked, but validation failed

        logger.info(
            f"JSON extraction attempt complete. Success: {success}, Validated: {validation_result.get('valid') if validation_result else 'N/A'}. Time: {processing_time:.2f}s"
        )
        return {
            "data": extracted_data,
            "validation_result": validation_result,
            "raw_text": raw_text_output
            if not success
            else None,  # Include raw only on parse failure
            "model": actual_model_used,
            "provider": provider,
            "tokens": token_info,
            "cost": cost,
            "processing_time": processing_time,
            "success": success,
            "error": error_message,
        }

    except Exception as e:
        error_model = model or f"{provider}/default"
        if isinstance(e, ProviderError):
            raise  # Re-raise
        else:
            raise ProviderError(
                f"JSON extraction failed: {str(e)}", provider=provider, model=error_model, cause=e
            ) from e


@with_tool_metrics
@with_error_handling
async def extract_table(
    text: str,
    headers: Optional[List[str]] = None,
    return_formats: Optional[List[str]] = None,  # Renamed from 'formats'
    extract_metadata: bool = False,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    # Removed ctx=None
) -> Dict[str, Any]:
    """Extracts tabular data found within text content.

    Identifies table structures in the input text and extracts the data, attempting
    to return it in specified formats (e.g., JSON list of objects, Markdown table).

    Args:
        text: The input text potentially containing one or more tables.
        headers: (Optional) A list of expected header strings. Providing headers helps the LLM
                 identify the correct table and map columns accurately.
        return_formats: (Optional) List of desired output formats. Supported: "json", "markdown".
                        Defaults to ["json"]. The result dictionary will contain keys matching these formats.
        extract_metadata: (Optional) If True, attempts to extract contextual metadata about the table,
                          such as a title, surrounding notes, or source information. Default False.
        provider: The name of the LLM provider (e.g., "openai"). Defaults to "openai".
        model: The specific model ID (e.g., "openai/gpt-4.1-mini"). Uses provider default if None.

    Returns:
        A dictionary containing the extracted table data and metadata:
        {
            "data": {                           # Dictionary containing requested formats
                "json": [ { "Header1": "Row1Val1", "Header2": "Row1Val2" }, ... ],
                "markdown": "| Header1 | Header2 |\n|---|---|\n| Row1Val1 | Row1Val2 |\n...",
                "metadata": { "title": "Table Title...", "notes": "..." } # If extract_metadata=True
            } | null, # Null if extraction fails
            "model": "provider/model-used",
            "provider": "provider-name",
            "tokens": { ... },
            "cost": 0.000180,
            "processing_time": 3.5,
            "success": true | false,
            "error": "Error message if success is false"
        }

    Raises:
        ProviderError: If the provider/LLM fails.
        ToolError: For other internal errors, including failure to parse the LLM response.
    """
    return_formats = return_formats or ["json"]
    start_time = time.time()

    if not text or not isinstance(text, str):
        raise ToolInputError(
            "Input 'text' must be a non-empty string.", param_name="text", provided_value=text
        )

    try:
        # Prepare model ID based on provider format
        effective_model = model
        # Ensure model ID includes provider prefix if not already present
        if model and provider not in model:
            effective_model = f"{provider}/{model}"

        headers_guidance = (
            f"The table likely has headers similar to: {', '.join(headers)}.\n"
            if headers
            else "Attempt to identify table headers automatically.\n"
        )
        metadata_guidance = (
            "Also extract any surrounding metadata like a table title, caption, or source notes.\n"
            if extract_metadata
            else ""
        )
        formats_guidance = (
            f"Return the extracted table data in these formats: {', '.join(return_formats)}."
        )

        # Improved prompt asking for specific formats in a JSON structure
        prompt = (
            f"Identify and extract the primary data table from the following text. "
            f"{headers_guidance}{metadata_guidance}{formats_guidance}"
            f"Format the output as a single JSON object containing keys for each requested format ({', '.join(return_formats)}) "
            f"and optionally a 'metadata' key if requested. Ensure the values are the table represented in that format."
            f"\n\nText:\n```\n{text}\n```\nResult JSON:"
        )

        # Use JSON mode if supported by the provider
        additional_params = {}
        if provider == Provider.OPENAI.value:
            additional_params["response_format"] = {"type": "json_object"}

        # Use the standardized completion tool instead of direct provider call
        completion_result = await generate_completion(
            prompt=prompt,
            model=effective_model,
            provider=provider,
            temperature=0.0,  # Low temp for precise extraction
            max_tokens=4000,
            additional_params=additional_params,
        )

        # Extract data from the standardized result format
        processing_time = completion_result.get("processing_time", time.time() - start_time)
        actual_model_used = completion_result.get("model", effective_model)
        raw_text_output = completion_result.get("text", "").strip()
        token_info = completion_result.get("tokens", {})
        cost = completion_result.get("cost", 0.0)
        tool_success = completion_result.get("success", False)

        # If the tool call failed, propagate the error
        if not tool_success:
            error_message = completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"Table extraction failed: {error_message}",
                provider=provider,
                model=actual_model_used,
            )

        # Use our robust parsing function
        extraction_result, success, error_message = _extract_and_parse_json(raw_text_output)

        # Basic validation if extraction succeeded
        if success and (
            not isinstance(extraction_result, dict)
            or not any(fmt in extraction_result for fmt in return_formats)
        ):
            logger.warning(
                f"Table extraction JSON result missing expected structure or formats ({return_formats}). Result: {extraction_result}"
            )
            # Allow partial success if it's a dict, but log warning
            if isinstance(extraction_result, dict):
                error_message = (
                    f"Warning: LLM output did not contain all requested formats ({return_formats})."
                )
            else:
                error_message = "Expected a JSON object with format keys"
                success = False
                extraction_result = None

        logger.info(
            f"Table extraction attempt complete. Success: {success}. Time: {processing_time:.2f}s"
        )
        return {
            "data": extraction_result,
            "raw_text": raw_text_output
            if not success
            else None,  # Include raw only on parse failure
            "model": actual_model_used,
            "provider": provider,
            "tokens": token_info,
            "cost": cost,
            "processing_time": processing_time,
            "success": success,
            "error": error_message,
        }

    except Exception as e:
        error_model = model or f"{provider}/default"
        if isinstance(e, ProviderError):
            raise
        else:
            raise ProviderError(
                f"Table extraction failed: {str(e)}", provider=provider, model=error_model, cause=e
            ) from e


@with_tool_metrics
@with_error_handling
async def extract_key_value_pairs(
    text: str,
    keys: Optional[List[str]] = None,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    # Removed ctx=None
) -> Dict[str, Any]:
    """Extracts key-value pairs from text, optionally targeting specific keys.

    Use this tool to pull out data points that appear in a "Key: Value" or similar format
    within unstructured text (e.g., fields from a form, details from a description).

    Args:
        text: The input text containing key-value pairs.
        keys: (Optional) A list of specific key names to look for and extract. If omitted,
              the tool attempts to extract all identifiable key-value pairs.
        provider: The name of the LLM provider (e.g., "openai"). Defaults to "openai".
        model: The specific model ID (e.g., "openai/gpt-4.1-mini"). Uses provider default if None.

    Returns:
        A dictionary containing the extracted key-value data and metadata:
        {
            "data": {             # Dictionary of extracted key-value pairs
                "Name": "Alice",
                "Role": "Engineer",
                "Location": "Remote", ...
            } | null,           # Null if extraction fails
            "model": "provider/model-used",
            "provider": "provider-name",
            "tokens": { ... },
            "cost": 0.000070,
            "processing_time": 2.1,
            "success": true | false,
            "error": "Error message if success is false"
        }

    Raises:
        ProviderError: If the provider/LLM fails.
        ToolError: For other internal errors, including failure to parse the LLM JSON response.
    """
    start_time = time.time()

    if not text or not isinstance(text, str):
        raise ToolInputError(
            "Input 'text' must be a non-empty string.", param_name="text", provided_value=text
        )

    try:
        # Prepare model ID based on provider format
        effective_model = model
        # Ensure model ID includes provider prefix if not already present
        if model and provider not in model:
            effective_model = f"{provider}/{model}"

        keys_guidance = (
            f"Extract the values for these specific keys: {', '.join(keys)}.\n"
            if keys
            else "Identify and extract all distinct key-value pairs present in the text.\n"
        )
        prompt = (
            f"Analyze the following text and extract key-value pairs. {keys_guidance}"
            f"Format the output as a single, flat JSON object mapping the extracted keys (as strings) to their corresponding values (as strings or appropriate simple types). "
            f"Infer the value associated with each key from the text context. Ignore keys not present in the text.\n\n"
            f"Text:\n```\n{text}\n```\nResult JSON object:"
        )

        # Use JSON mode if supported by the provider
        additional_params = {}
        if provider == Provider.OPENAI.value:
            additional_params["response_format"] = {"type": "json_object"}

        # Use the standardized completion tool instead of direct provider call
        completion_result = await generate_completion(
            prompt=prompt,
            model=effective_model,
            provider=provider,
            temperature=0.0,  # Low temp for precise extraction
            max_tokens=2000,
            additional_params=additional_params,
        )

        # Extract data from the standardized result format
        processing_time = completion_result.get("processing_time", time.time() - start_time)
        actual_model_used = completion_result.get("model", effective_model)
        raw_text_output = completion_result.get("text", "").strip()
        token_info = completion_result.get("tokens", {})
        cost = completion_result.get("cost", 0.0)
        tool_success = completion_result.get("success", False)

        # If the tool call failed, propagate the error
        if not tool_success:
            error_message = completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"Key-value pair extraction failed: {error_message}",
                provider=provider,
                model=actual_model_used,
            )

        # Use our robust parsing function
        kv_data, success, error_message = _extract_and_parse_json(raw_text_output)

        # Validate it's a dictionary if extraction succeeded
        if success and not isinstance(kv_data, dict):
            error_message = "Extracted data is not a valid key-value dictionary"
            logger.warning(error_message)
            success = False
            kv_data = None

        logger.info(
            f"Key-Value pair extraction attempt complete. Success: {success}. Time: {processing_time:.2f}s"
        )
        return {
            "data": kv_data,
            "raw_text": raw_text_output if not success else None,
            "model": actual_model_used,
            "provider": provider,
            "tokens": token_info,
            "cost": cost,
            "processing_time": processing_time,
            "success": success,
            "error": error_message,
        }

    except Exception as e:
        error_model = model or f"{provider}/default"
        if isinstance(e, ProviderError):
            raise
        else:
            raise ProviderError(
                f"Key-value pair extraction failed: {str(e)}",
                provider=provider,
                model=error_model,
                cause=e,
            ) from e


@with_tool_metrics
@with_error_handling
async def extract_semantic_schema(
    text: str,
    # Schema should ideally be passed as a structured dict, not within the prompt
    semantic_schema: Dict[str, Any],  # Changed from embedding prompt
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    # Removed ctx=None
) -> Dict[str, Any]:
    """Extracts information from text matching a specified semantic structure (schema).

    Use this tool when you need to populate a predefined JSON structure with information
    found or inferred from the input text. Unlike `extract_json`, the target JSON structure
    is *defined by you* (via `semantic_schema`), not expected to be present in the input text.

    Args:
        text: The input text containing information to extract.
        semantic_schema: A Python dictionary representing the desired JSON schema for the output.
                         Use JSON Schema conventions (e.g., {"type": "object", "properties": { ... }}).
                         This guides the LLM on what fields to extract and their expected types.
        provider: The name of the LLM provider (e.g., "openai"). Defaults to "openai".
                  Providers supporting JSON mode or strong instruction following are recommended.
        model: The specific model ID (e.g., "openai/gpt-4o"). Uses provider default if None.

    Returns:
        A dictionary containing the extracted data conforming to the schema and metadata:
        {
            "data": { ... }, # The extracted data, structured according to semantic_schema
            "model": "provider/model-used",
            "provider": "provider-name",
            "tokens": { ... },
            "cost": 0.000210,
            "processing_time": 4.1,
            "success": true | false,
            "error": "Error message if success is false"
        }

    Raises:
        ToolInputError: If `semantic_schema` is not a valid dictionary.
        ProviderError: If the provider/LLM fails.
        ToolError: For other internal errors, including failure to parse the LLM JSON response.
    """
    start_time = time.time()

    if not text or not isinstance(text, str):
        raise ToolInputError(
            "Input 'text' must be a non-empty string.", param_name="text", provided_value=text
        )
    if not semantic_schema or not isinstance(semantic_schema, dict):
        raise ToolInputError(
            "Input 'semantic_schema' must be a non-empty dictionary representing a JSON schema.",
            param_name="semantic_schema",
            provided_value=semantic_schema,
        )

    try:
        # Prepare model ID based on provider format
        effective_model = model
        # Ensure model ID includes provider prefix if not already present
        if model and provider not in model:
            effective_model = f"{provider}/{model}"

        # Create a clear prompt explaining the task and providing the schema
        schema_str = json.dumps(semantic_schema, indent=2)
        prompt = (
            f"Analyze the following text and extract information that conforms to the provided JSON schema. "
            f"Populate the fields in the schema based *only* on information present in the text. "
            f"If information for a field is not found, omit the field or use a null value as appropriate according to the schema. "
            f"Return ONLY the populated JSON object conforming to the schema.\n\n"
            f"JSON Schema:\n```json\n{schema_str}\n```\n\n"
            f"Text:\n```\n{text}\n```\nPopulated JSON object:"
        )

        # Use JSON mode if supported by the provider
        additional_params = {}
        if provider == Provider.OPENAI.value:
            additional_params["response_format"] = {"type": "json_object"}

        # Use the standardized completion tool instead of direct provider call
        completion_result = await generate_completion(
            prompt=prompt,
            model=effective_model,
            provider=provider,
            temperature=0.0,  # Low temp for precise extraction
            max_tokens=4000,
            additional_params=additional_params,
        )

        # Extract data from the standardized result format
        processing_time = completion_result.get("processing_time", time.time() - start_time)
        actual_model_used = completion_result.get("model", effective_model)
        raw_text_output = completion_result.get("text", "").strip()
        token_info = completion_result.get("tokens", {})
        cost = completion_result.get("cost", 0.0)
        tool_success = completion_result.get("success", False)

        # If the tool call failed, propagate the error
        if not tool_success:
            error_message = completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"Semantic schema extraction failed: {error_message}",
                provider=provider,
                model=actual_model_used,
            )

        # Use our robust parsing function
        extracted_data, success, error_message = _extract_and_parse_json(raw_text_output)

        # Validate against the provided schema if extraction succeeded
        if success:
            try:
                jsonschema.validate(instance=extracted_data, schema=semantic_schema)
                logger.debug("Successfully parsed and validated semantic schema JSON.")
            except jsonschema.exceptions.ValidationError as e:
                error_message = f"Warning: LLM output did not strictly conform to schema: {str(e)}"
                logger.warning(f"{error_message}. Data: {extracted_data}")
                # Still consider extraction successful if parsable

        logger.info(
            f"Semantic schema extraction attempt complete. Success: {success}. Time: {processing_time:.2f}s"
        )
        return {
            "data": extracted_data,
            "raw_text": raw_text_output if not success else None,
            "model": actual_model_used,
            "provider": provider,
            "tokens": token_info,
            "cost": cost,
            "processing_time": processing_time,
            "success": success,
            "error": error_message,
        }

    except Exception as e:
        error_model = model or f"{provider}/default"
        if isinstance(e, ProviderError):
            raise
        else:
            raise ProviderError(
                f"Semantic schema extraction failed: {str(e)}",
                provider=provider,
                model=error_model,
                cause=e,
            ) from e


# Note: This is a utility function, not typically exposed as a direct tool,
# but kept here as it relates to extraction from LLM *responses*.
# No standard decorators applied.
async def extract_code_from_response(
    response_text: str,
    model: str = "openai/gpt-4.1-mini",
    timeout: int = 15,
    tracker: Optional[
        Any
    ] = None,  # Add optional tracker (use Any for now to avoid circular import)
) -> str:
    """Extracts code blocks from LLM response text, using an LLM for complex cases.

    Primarily designed to clean up responses from code generation tasks.
    It first tries simple regex matching for markdown code fences. If that fails,
    it uses a specified LLM to identify and extract the code.

    Args:
        response_text: The text potentially containing code blocks.
        model: The specific model ID to use for LLM-based extraction if regex fails.
               Defaults to "openai/gpt-4.1-mini".
        timeout: Timeout in seconds for the LLM extraction call. Default 15.
        tracker: (Optional) An instance of a CostTracker for tracking cost and metrics.

    Returns:
        The extracted code block as a string, or the original text if no code is found or extraction fails.
    """
    if not response_text or not isinstance(response_text, str):
        return ""  # Return empty if no input

    # Try simple regex extraction first (common markdown format)
    code_blocks = re.findall(r"```(?:[a-zA-Z0-9\-_]*\n)?(.*?)\n?```", response_text, re.DOTALL)

    if code_blocks:
        # Return the content of the first code block found
        logger.debug("Extracted code using regex.")
        return code_blocks[0].strip()

    # If regex fails, use LLM for more robust extraction
    logger.debug("Regex failed, attempting LLM-based code extraction.")
    try:
        # Parse provider from model string if it contains a slash
        provider_id = model.split("/")[0] if "/" in model else Provider.OPENAI.value
        effective_model = model  # Use the full model string as provided

        prompt = f"Extract only the main code block from the following text. Return just the code itself, without any explanations or markdown fences.\n\nText:\n```\n{response_text}\n```\n\nCode:"

        # Set a timeout using asyncio.wait_for
        completion_task = generate_completion(
            prompt=prompt,
            model=effective_model,
            provider=provider_id,
            temperature=0.0,
            max_tokens=len(response_text),  # Allow enough tokens, approx original length
        )

        # Use asyncio.wait_for to implement timeout
        completion_result = await asyncio.wait_for(completion_task, timeout=timeout)

        # Check if completion succeeded
        if not completion_result.get("success", False):
            logger.warning(
                f"LLM code extraction failed: {completion_result.get('error', 'Unknown error')}. Returning original text."
            )
            return response_text

        # Track cost if tracker is provided
        if tracker:
            try:
                # Use getattr to safely access attributes, provide defaults
                # Create a temporary object for tracking as CostTracker expects attributes
                class Trackable:
                    pass

                trackable = Trackable()
                trackable.cost = completion_result.get("cost", 0.0)
                trackable.input_tokens = completion_result.get("tokens", {}).get("input", 0)
                trackable.output_tokens = completion_result.get("tokens", {}).get("output", 0)
                trackable.provider = provider_id
                trackable.model = completion_result.get("model", effective_model)
                trackable.processing_time = completion_result.get("processing_time", 0.0)
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(
                    f"Could not track cost for LLM code extraction: {track_err}", exc_info=False
                )

        extracted_code = completion_result.get("text", "").strip()
        logger.info(f"Extracted code using LLM ({effective_model}).")
        return extracted_code

    except asyncio.TimeoutError:
        logger.warning(f"LLM code extraction timed out after {timeout}s. Returning original text.")
        return response_text  # Fallback to original on timeout
    except Exception as e:
        logger.error(
            f"LLM code extraction failed: {str(e)}. Returning original text.", exc_info=False
        )
        return response_text  # Fallback to original on error


class ExtractionTools(BaseTool):
    """Tools for extracting structured data from unstructured text."""

    tool_name = "extraction"
    description = "Tools for extracting structured data from unstructured text, including JSON, tables, and key-value pairs."

    def __init__(self, gateway):
        """Initialize extraction tools.

        Args:
            gateway: Gateway or MCP server instance
        """
        super().__init__(gateway)
        self._register_tools()

    def _register_tools(self):
        """Register extraction tools with MCP server."""
        # Register the extraction functions as tools
        self.mcp.tool(name="extract_json")(extract_json)
        self.mcp.tool(name="extract_table")(extract_table)
        self.mcp.tool(name="extract_key_value_pairs")(extract_key_value_pairs)
        self.mcp.tool(name="extract_semantic_schema")(extract_semantic_schema)
        self.logger.info("Registered extraction tools", emoji_key="success")
