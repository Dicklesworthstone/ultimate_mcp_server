"""Parsing utilities for Ultimate MCP Server.

This module provides utility functions for parsing and processing
results from Ultimate MCP Server operations that were previously defined in
example scripts but are now part of the library.
"""

import json
import re
from typing import Any, Dict

from ultimate_mcp_server.utils import get_logger

# Initialize logger
logger = get_logger("ultimate_mcp_server.utils.parsing")


def extract_json_from_markdown(text: str) -> str:
    """Extracts a JSON string embedded within markdown code fences.

    Handles various markdown code block formats and edge cases:
    - Complete code blocks: ```json ... ``` or ``` ... ```
    - Alternative fence styles: ~~~json ... ~~~
    - Incomplete/truncated blocks with only opening fence
    - Multiple code blocks (chooses the first valid JSON)
    - Extensive JSON repair for common LLM output issues:
        - Unterminated strings
        - Trailing commas
        - Missing closing brackets
        - Unquoted keys
        - Truncated content

    If no valid JSON-like content is found in fences, returns the original string.

    Args:
        text: The input string possibly containing markdown-fenced JSON.

    Returns:
        The extracted JSON string or the stripped original string.
    """
    if not text:
        return ""

    cleaned_text = text.strip()
    possible_json_candidates = []

    # Try to find JSON inside complete code blocks with various fence styles
    # Look for backtick fences (most common)
    backtick_matches = re.finditer(
        r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE
    )
    for match in backtick_matches:
        possible_json_candidates.append(match.group(1).strip())

    # Look for tilde fences (less common but valid in some markdown)
    tilde_matches = re.finditer(
        r"~~~(?:json)?\s*(.*?)\s*~~~", cleaned_text, re.DOTALL | re.IGNORECASE
    )
    for match in tilde_matches:
        possible_json_candidates.append(match.group(1).strip())

    # If no complete blocks found, check for blocks with only opening fence
    if not possible_json_candidates:
        # Try backtick opening fence
        backtick_start = re.search(r"```(?:json)?\s*", cleaned_text, re.IGNORECASE)
        if backtick_start:
            content_after_fence = cleaned_text[backtick_start.end() :].strip()
            possible_json_candidates.append(content_after_fence)

        # Try tilde opening fence
        tilde_start = re.search(r"~~~(?:json)?\s*", cleaned_text, re.IGNORECASE)
        if tilde_start:
            content_after_fence = cleaned_text[tilde_start.end() :].strip()
            possible_json_candidates.append(content_after_fence)

    # If still no candidates, add the original text as last resort
    if not possible_json_candidates:
        possible_json_candidates.append(cleaned_text)

    # Try each candidate, returning the first one that looks like valid JSON
    for candidate in possible_json_candidates:
        # Apply advanced JSON repair
        repaired = _repair_json(candidate)
        try:
            # Validate if it's actually parseable JSON
            json.loads(repaired)
            return repaired  # Return the first valid JSON
        except json.JSONDecodeError:
            # If repair didn't work, continue to the next candidate
            continue

    # If no candidate worked with regular repair, try more aggressive repair on the first candidate
    if possible_json_candidates:
        aggressive_repair = _repair_json(possible_json_candidates[0], aggressive=True)
        try:
            json.loads(aggressive_repair)
            return aggressive_repair
        except json.JSONDecodeError:
            # Return the best we can - the first candidate with basic cleaning
            # This will still fail in json.loads, but at least we tried
            return possible_json_candidates[0]

    # Absolute fallback - return the original text
    return cleaned_text


def _repair_json(text: str, aggressive=False) -> str:
    """
    Repair common JSON formatting issues in LLM-generated output.

    This internal utility function applies a series of transformations to fix common
    JSON formatting problems that frequently occur in LLM outputs. It can operate in
    two modes: standard and aggressive.

    In standard mode (aggressive=False), it applies basic repairs like:
    - Removing trailing commas before closing brackets/braces
    - Ensuring property names are properly quoted
    - Basic structure validation

    In aggressive mode (aggressive=True), it applies more extensive repairs:
    - Fixing unterminated string literals by adding missing quotes
    - Balancing unmatched brackets and braces
    - Adding missing values for dangling properties
    - Handling truncated JSON at the end of strings
    - Attempting to recover partial JSON structures

    The aggressive repairs are particularly useful when dealing with outputs from
    models that have been truncated mid-generation or contain structural errors
    that would normally make the JSON unparseable.

    Args:
        text: The JSON-like string to repair, potentially containing formatting errors
        aggressive: Whether to apply more extensive repair techniques beyond basic
                   formatting fixes. Default is False (basic repairs only).

    Returns:
        A repaired JSON string that is more likely to be parseable. Note that even
        with aggressive repairs, the function cannot guarantee valid JSON for
        severely corrupted inputs.

    Note:
        This function is intended for internal use by extract_json_from_markdown.
        While it attempts to fix common issues, it may not address all possible
        JSON formatting problems, especially in severely malformed inputs.
    """
    if not text:
        return text

    # Step 1: Basic cleanup
    result = text.strip()

    # Quick check if it even remotely looks like JSON
    if not (result.startswith("{") or result.startswith("[")):
        return result

    # Step 2: Fix common issues

    # Fix trailing commas before closing brackets
    result = re.sub(r",\s*([\}\]])", r"\1", result)

    # Ensure property names are quoted
    result = re.sub(r"([{,]\s*)([a-zA-Z0-9_$]+)(\s*:)", r'\1"\2"\3', result)

    # If we're not in aggressive mode, return after basic fixes
    if not aggressive:
        return result

    # Step 3: Aggressive repairs for truncated/malformed JSON

    # Track opening/closing brackets to detect imbalance
    open_braces = result.count("{")
    close_braces = result.count("}")
    open_brackets = result.count("[")
    close_brackets = result.count("]")

    # Count quotes to check if we have an odd number (indicating unterminated strings)
    quote_count = result.count('"')
    if quote_count % 2 != 0:
        # We have an odd number of quotes, meaning at least one string is unterminated
        # This is a much more aggressive approach to fix strings

        # First, try to find all strings that are properly terminated
        proper_strings = []
        pos = 0
        in_string = False
        string_start = 0

        # This helps track properly formed strings and identify problematic ones
        while pos < len(result):
            if result[pos] == '"' and (pos == 0 or result[pos - 1] != "\\"):
                if not in_string:
                    # Start of a string
                    in_string = True
                    string_start = pos
                else:
                    # End of a string
                    in_string = False
                    proper_strings.append((string_start, pos))
            pos += 1

        # If we're still in a string at the end, we found an unterminated string
        if in_string:
            # Force terminate it at the end
            result += '"'

    # Even more aggressive string fixing
    # This regexp looks for a quote followed by any characters not containing a quote
    # followed by a comma, closing brace, or bracket, without a quote in between
    # This indicates an unterminated string
    result = re.sub(r'"([^"]*?)(?=,|\s*[\]}]|$)', r'"\1"', result)

    # Fix cases where value might be truncated mid-word just before closing quote
    # If we find something that looks like it's in the middle of a string, terminate it
    result = re.sub(
        r'"([^"]+)(\s*[\]}]|,|$)',
        lambda m: f'"{m.group(1)}"{"" if m.group(2).startswith(",") or m.group(2) in "]}," else m.group(2)}',
        result,
    )

    # Fix dangling quotes at the end of the string - these usually indicate a truncated string
    if result.rstrip().endswith('"'):
        # Add closing quote and appropriate structure depending on context
        result = result.rstrip() + '"'

        # Look at the previous few characters to determine if we need a comma or not
        context = result[-20:] if len(result) > 20 else result
        # If string ends with x": " it's likely a property name
        if re.search(r'"\s*:\s*"$', context):
            # Add a placeholder value and closing structure for the property
            result += "unknown"

    # Check for dangling property (property name with colon but no value)
    result = re.sub(r'"([^"]+)"\s*:(?!\s*["{[\w-])', r'"\1": null', result)

    # Add missing closing brackets/braces if needed
    if open_braces > close_braces:
        result += "}" * (open_braces - close_braces)
    if open_brackets > close_brackets:
        result += "]" * (open_brackets - close_brackets)

    # Handle truncated JSON structure - look for incomplete objects at the end
    # This is complex, but we'll try some common patterns

    # If JSON ends with a property name and colon but no value
    if re.search(r'"[^"]+"\s*:\s*$', result):
        result += "null"

    # If JSON ends with a comma, it needs another value - add a null
    if re.search(r",\s*$", result):
        result += "null"

    # If the JSON structure is fundamentally corrupted at the end (common in truncation)
    # Close any unclosed objects or arrays
    if not (result.endswith("}") or result.endswith("]") or result.endswith('"')):
        # Count unmatched opening brackets
        stack = []
        for char in result:
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack and (
                    (stack[-1] == "{" and char == "}") or (stack[-1] == "[" and char == "]")
                ):
                    stack.pop()

        # Close any unclosed structures
        for bracket in reversed(stack):
            if bracket == "{":
                result += "}"
            elif bracket == "[":
                result += "]"

    # As a final safeguard, try to eval the JSON with a permissive parser
    # This won't fix deep structural issues but catches cases our regexes missed
    try:
        import simplejson

        simplejson.loads(result, parse_constant=lambda x: x)
    except (ImportError, simplejson.JSONDecodeError):
        try:
            # Try one last time with the more permissive custom JSON parser
            _scan_once = json.scanner.py_make_scanner(json.JSONDecoder())
            try:
                _scan_once(result, 0)
            except StopIteration:
                # Likely unterminated JSON - do one final pass of common fixups

                # Check for unterminated strings of various forms one more time
                if re.search(r'(?<!")"(?:[^"\\]|\\.)*[^"\\](?!")(?=,|\s*[\]}]|$)', result):
                    # Even more aggressive fixes, replacing with generic values
                    result = re.sub(
                        r'(?<!")"(?:[^"\\]|\\.)*[^"\\](?!")(?=,|\s*[\]}]|$)',
                        r'"invalid_string"',
                        result,
                    )

                # Ensure valid JSON-like structure
                if not (result.endswith("}") or result.endswith("]")):
                    if result.count("{") > result.count("}"):
                        result += "}"
                    if result.count("[") > result.count("]"):
                        result += "]"
            except Exception:
                # Something else is wrong, but we've tried our best
                pass
        except Exception:
            # We've done all we reasonably can
            pass

    return result


async def parse_result(result: Any) -> Dict[str, Any]:
    """Parse the result from a tool call into a usable dictionary.

    Handles various return types from MCP tools, including TextContent objects,
    list results, and direct dictionaries. Attempts to extract JSON from
    markdown code fences if present.

    Args:
        result: Result from an MCP tool call or provider operation

    Returns:
        Parsed dictionary containing the result data
    """
    try:
        text_to_parse = None
        # Handle TextContent object (which has a .text attribute)
        if hasattr(result, "text"):
            text_to_parse = result.text

        # Handle list result
        elif isinstance(result, list):
            if result:
                first_item = result[0]
                if hasattr(first_item, "text"):
                    text_to_parse = first_item.text
                elif isinstance(first_item, dict):
                    # NEW: Check if it's an MCP-style text content dict
                    if first_item.get("type") == "text" and "text" in first_item:
                        text_to_parse = first_item["text"]
                    else:
                        # It's some other dictionary, return it directly
                        return first_item
                elif isinstance(first_item, str):
                    text_to_parse = first_item
                else:
                    logger.warning(f"List item type not directly parseable: {type(first_item)}")
                    return {
                        "error": f"List item type not directly parseable: {type(first_item)}",
                        "original_result_type": str(type(result)),
                    }
            else:  # Empty list
                return {}  # Or perhaps an error/warning? For now, empty dict.

        # Handle dictionary directly
        elif isinstance(result, dict):
            return result

        # Handle string directly
        elif isinstance(result, str):
            text_to_parse = result

        # If text_to_parse is still None or is empty/whitespace after potential assignments
        if text_to_parse is None or not text_to_parse.strip():
            logger.warning(
                f"No parsable text content found in result (type: {type(result)}, content preview: '{str(text_to_parse)[:100]}...')."
            )
            return {
                "error": "No parsable text content found in result",
                "result_type": str(type(result)),
                "content_preview": str(text_to_parse)[:100] if text_to_parse else None,
            }

        # At this point, text_to_parse should be a non-empty string.
        # Attempt to extract JSON from markdown (if any)
        # If no markdown, or extraction fails, json_to_parse will be text_to_parse itself.
        json_to_parse = extract_json_from_markdown(text_to_parse)

        if (
            not json_to_parse.strip()
        ):  # If extraction resulted in an empty string (e.g. from "``` ```")
            logger.warning(
                f"JSON extraction from text_to_parse yielded an empty string. Original text_to_parse: '{text_to_parse[:200]}...'"
            )
            # Fallback to trying the original text_to_parse if extraction gave nothing useful
            # This covers cases where text_to_parse might be pure JSON without fences.
            if text_to_parse.strip():  # Ensure original text_to_parse wasn't also empty
                json_to_parse = text_to_parse
            else:  # Should have been caught by the earlier check, but as a safeguard:
                return {
                    "error": "Content became empty after attempting JSON extraction",
                    "original_text_to_parse": text_to_parse,
                }

        # Now, json_to_parse should be the best candidate string for JSON parsing.
        # Only attempt to parse if it's not empty/whitespace.
        if not json_to_parse.strip():
            logger.warning(
                f"Final string to parse is empty. Original text_to_parse: '{text_to_parse[:200]}...'"
            )
            return {
                "error": "Final string for JSON parsing is empty",
                "original_text_to_parse": text_to_parse,
            }

        try:
            return json.loads(json_to_parse)
        except json.JSONDecodeError as e:
            problematic_text_for_repair = json_to_parse  # This is the string that failed json.loads
            logger.warning(
                f"Initial JSON parsing failed for: '{problematic_text_for_repair[:200]}...' Error: {e}. Attempting LLM repair...",
                emoji_key="warning",
            )
            try:
                from ultimate_mcp_server.tools.completion import generate_completion

                system_message_content = "You are an expert JSON repair assistant. Your goal is to return only valid JSON."
                # Prepend system instruction to the main prompt for completion models
                # (as generate_completion with openai provider doesn't natively use a separate system_prompt field in its current design)
                user_repair_request = (
                    f"The following text is supposed to be valid JSON but failed parsing. "
                    f"Please correct it and return *only* the raw, valid JSON string. "
                    f"Do not include any explanations or markdown formatting. "
                    f"If it's impossible to extract or repair to valid JSON, return an empty JSON object {{}}. "
                    f"Problematic text:\n\n```text\n{problematic_text_for_repair}\n```"
                )
                combined_prompt = f"{system_message_content}\n\n{user_repair_request}"

                llm_repair_result = await generate_completion(
                    prompt=combined_prompt,  # Use the combined prompt
                    provider="openai",
                    model="gpt-4.1-mini",
                    temperature=0.0,
                    additional_params={},  # Remove system_prompt from here
                )

                text_from_llm_repair = llm_repair_result.get("text", "")
                if not text_from_llm_repair.strip():
                    logger.error("LLM repair attempt returned empty string.")
                    return {
                        "error": "LLM repair returned empty string",
                        "original_text": problematic_text_for_repair,
                    }

                # Re-extract from LLM response, as it might add fences
                final_repaired_json_str = extract_json_from_markdown(text_from_llm_repair)

                if not final_repaired_json_str.strip():
                    logger.error(
                        f"LLM repair extracted an empty JSON string from LLM response: {text_from_llm_repair[:200]}..."
                    )
                    return {
                        "error": "LLM repair extracted empty JSON",
                        "llm_response": text_from_llm_repair,
                        "original_text": problematic_text_for_repair,
                    }

                try:
                    logger.debug(
                        f"Attempting to parse LLM-repaired JSON: {final_repaired_json_str[:200]}..."
                    )
                    parsed_llm_result = json.loads(final_repaired_json_str)
                    logger.success("LLM JSON repair successful.", emoji_key="success")
                    return parsed_llm_result
                except json.JSONDecodeError as llm_e:
                    logger.error(
                        f"LLM repair attempt failed. LLM response could not be parsed as JSON: {llm_e}. LLM response (after extraction): '{final_repaired_json_str[:200]}' Original LLM text: '{text_from_llm_repair[:500]}...'"
                    )
                    return {
                        "error": "LLM repair failed to produce valid JSON",
                        "detail": str(llm_e),
                        "llm_response_extracted": final_repaired_json_str,
                        "llm_response_raw": text_from_llm_repair,
                        "original_text": problematic_text_for_repair,
                    }
            except Exception as repair_ex:
                logger.error(f"Exception during LLM repair process: {repair_ex}", exc_info=True)
                return {
                    "error": "Exception during LLM repair",
                    "detail": str(repair_ex),
                    "original_text": problematic_text_for_repair,
                }

    except Exception as e:  # General error in parse_result
        logger.error(f"Critical error in parse_result: {e}", exc_info=True)
        return {"error": "Critical error during result parsing", "detail": str(e)}


async def process_mcp_result(result: Any) -> Dict[str, Any]:
    """
    Process and normalize results from MCP tool calls into a consistent dictionary format.

    This function serves as a user-friendly interface for handling and normalizing
    the various types of results that can be returned from MCP tools and provider operations.
    It acts as a bridge between the raw MCP tool outputs and downstream application code
    that expects a consistent dictionary structure.

    The function handles multiple return formats:
    - TextContent objects with a .text attribute
    - List results containing TextContent objects or dictionaries
    - Direct dictionary returns
    - JSON-like strings embedded in markdown code blocks

    Key features:
    - Automatic extraction of JSON from markdown code fences
    - JSON repair for malformed or truncated LLM outputs
    - Fallback to LLM-based repair for difficult parsing cases
    - Consistent error handling and reporting

    This function is especially useful in:
    - Handling results from completion tools where LLMs may return JSON in various formats
    - Processing tool responses that contain structured data embedded in text
    - Creating a consistent interface for downstream processing of MCP tool results
    - Simplifying error handling in client applications

    Args:
        result: The raw result from an MCP tool call or provider operation, which could
               be a TextContent object, a list, a dictionary, or another structure

    Returns:
        A dictionary containing either:
        - The successfully parsed result data
        - An error description with diagnostic information if parsing failed

    Example:
        ```python
        result = await some_mcp_tool()
        processed_data = await process_mcp_result(result)

        # Check for errors in the processed result
        if "error" in processed_data:
            print(f"Error processing result: {processed_data['error']}")
        else:
            # Use the normalized data
            print(f"Processed data: {processed_data}")
        ```
    """
    return await parse_result(result)
