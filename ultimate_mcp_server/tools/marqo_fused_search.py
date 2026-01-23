"""MCP tool for flexible searching of Marqo indices."""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx  # Add import for httpx
import marqo
from pydantic import BaseModel, Field, field_validator

from ultimate_mcp_server.clients import CompletionClient
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.exceptions import ToolExecutionError, ToolInputError
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.marqo_fused_search")

# --- Configuration Loading ---

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "marqo_index_config.json")


def load_marqo_config() -> Dict[str, Any]:
    """Loads Marqo configuration from the JSON file.

    Returns:
        A dictionary containing the loaded configuration.

    Raises:
        ToolExecutionError: If the config file cannot be found, parsed, or is invalid.
    """
    try:
        with open(CONFIG_FILE_PATH, "r") as f:
            config = json.load(f)
            logger.info(f"Loaded Marqo config from {CONFIG_FILE_PATH}")
            return config
    except FileNotFoundError:
        logger.warning(
            f"Marqo config file not found at {CONFIG_FILE_PATH}. Using minimal hardcoded defaults."
        )
        # Fallback to minimal, generic defaults if file not found
        return {
            "default_marqo_url": "http://localhost:8882",
            "default_index_name": "my_marqo_index",  # Generic name
            "default_schema": {  # Minimal fallback schema
                "fields": {
                    # Define only essential fields or placeholders if file is missing
                    "content": {"type": "text", "role": "content"},
                    "embedding": {"type": "tensor", "role": "tensor_vector"},
                    "_id": {"type": "keyword", "role": "internal"},
                },
                "tensor_field": "embedding",
                "default_content_field": "content",
                "default_date_field": None,  # No default date field assumed
            },
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding Marqo config file {CONFIG_FILE_PATH}: {e}", exc_info=True)
        raise ToolExecutionError(
            f"Failed to load or parse Marqo config file: {CONFIG_FILE_PATH}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error loading Marqo config: {e}", exc_info=True)
        raise ToolExecutionError("Unexpected error loading Marqo config") from e


MARQO_CONFIG = load_marqo_config()
DEFAULT_MARQO_URL = MARQO_CONFIG.get("default_marqo_url", "http://localhost:8882")
DEFAULT_INDEX_NAME = MARQO_CONFIG.get("default_index_name", "my_marqo_index")  # Use generic default
DEFAULT_INDEX_SCHEMA = MARQO_CONFIG.get("default_schema", {})

# --- Define cache file path relative to config file ---
CACHE_FILE_DIR = os.path.dirname(CONFIG_FILE_PATH)
CACHE_FILE_PATH = os.path.join(CACHE_FILE_DIR, "marqo_docstring_cache.json")

# --- LLM Client for Doc Generation ---


async def _call_llm_for_doc_generation(prompt: str) -> Optional[str]:
    """
    Calls an LLM to generate dynamic documentation for the Marqo search tool.

    This function uses the Ultimate MCP Server's CompletionClient to send a prompt to an
    LLM (preferring Gemini) and retrieve generated content for enhancing the tool's documentation.
    It handles the entire process including client initialization, LLM API call configuration,
    and error handling.

    The function is designed with low temperature settings for predictable, factual outputs
    and enables caching to avoid redundant LLM calls for the same configuration.

    Args:
        prompt: A detailed prompt string containing information about the Marqo index
               configuration and instructions for generating appropriate documentation.

    Returns:
        str: The generated documentation text if successful.
        None: If the LLM call fails or returns empty content.

    Notes:
        - Uses the Gemini provider by default, but will fall back to other providers if needed.
        - Sets temperature to 0.3 for consistent, deterministic outputs.
        - Limits output to 400 tokens, which is sufficient for documentation purposes.
        - Enables caching to improve performance for repeated calls.
    """
    try:
        # Instantiate the client - assumes necessary env vars/config are set for the gateway
        client = CompletionClient()

        logger.info("Calling LLM to generate dynamic docstring augmentation...")
        # Use generate_completion (can also use try_providers if preferred)
        result = await client.generate_completion(
            prompt=prompt,
            provider=Provider.GEMINI.value,  # Prioritize Gemini, adjust if needed
            temperature=0.3,  # Lower temperature for more factual/consistent doc generation
            max_tokens=400,  # Allow sufficient length for the documentation section
            use_cache=True,  # Cache the generated doc string for a given config
        )

        # --- FIX: Check for successful result (no exception) and non-empty text ---
        # if result.error is None and result.text:
        if result and result.text:  # Exception handled by the outer try/except
            logger.success(
                f"Successfully generated doc augmentation via {result.provider}. Length: {len(result.text)}"
            )
            return result.text.strip()
        else:
            # This case might be less likely if exceptions are used for errors,
            # but handles cases where generation succeeds but returns empty text or None result unexpectedly.
            provider_name = result.provider if result else "Unknown"
            logger.error(
                f"LLM call for doc generation succeeded but returned no text. Provider: {provider_name}"
            )
            return None

    except Exception as e:
        logger.error(f"Error during LLM call for doc generation: {e}", exc_info=True)
        return None


# --- Docstring Augmentation Generation ---


async def _generate_docstring_augmentation_from_config(config: Dict[str, Any]) -> str:
    """Generates dynamic documentation augmentation by calling an LLM with the config."""
    augmentation = ""
    try:
        index_name = config.get("default_index_name", "(Not specified)")
        schema = config.get("default_schema", {})
        schema_fields = schema.get("fields", {})
        date_field = schema.get("default_date_field")

        # Basic check: Don't call LLM for clearly minimal/default schemas
        if index_name == "my_marqo_index" and len(schema_fields) <= 3:  # Heuristic
            logger.info("Skipping LLM doc generation for minimal default config.")
            return ""

        # Format schema fields for the prompt
        formatted_schema = []
        for name, props in schema_fields.items():
            details = [f"type: {props.get('type')}"]
            if props.get("role"):
                details.append(f"role: {props.get('role')}")
            if props.get("filterable"):
                details.append("filterable")
            if props.get("sortable"):
                details.append("sortable")
            if props.get("searchable"):
                details.append(f"searchable: {props.get('searchable')}")
            formatted_schema.append(f"  - {name}: ({', '.join(details)})")
        schema_str = "\n".join(formatted_schema)

        # Construct the prompt for the LLM
        prompt = f"""
        Analyze the following Marqo index configuration and generate a concise markdown documentation section for a search tool using this index. Your goal is to help a user understand what kind of data they can search and how to use the tool effectively.

        Instructions:
        1.  **Infer Domain:** Based *only* on the index name and field names/types/roles, what is the likely subject matter or domain of the documents in this index (e.g., financial reports, product catalogs, medical articles, general documents)? State this clearly.
        2.  **Purpose:** Briefly explain the primary purpose of using a search tool with this index.
        3.  **Keywords:** Suggest 3-5 relevant keywords a user might include in their search queries for this specific index.
        4.  **Example Queries:** Provide 1-2 diverse example queries demonstrating typical use cases.
        5.  **Filtering:** Explain how the 'filters' parameter can be used, mentioning 1-2 specific filterable fields from the schema with examples (e.g., `filters={{"field_name": "value"}}`).
        6.  **Date Filtering:** If a default date field is specified (`{date_field}`), mention that the `date_range` parameter can be used with it.
        7.  **Format:** Output *only* the generated markdown section, starting with `---` and `**Configuration-Specific Notes:**`.

        Configuration Details:
        ----------------------
        Index Name: {index_name}
        Default Date Field: {date_field or "Not specified"}

        Schema Fields:
        {schema_str}
        ----------------------

        Generated Documentation Section (Markdown):
        """

        logger.info(
            f"Attempting to generate docstring augmentation via LLM for index: {index_name}"
        )
        logger.debug(f"LLM Prompt for doc generation:\n{prompt}")

        # Call the LLM
        generated_text = await _call_llm_for_doc_generation(prompt)

        if generated_text:
            augmentation = "\n\n" + generated_text  # Add separators
        else:
            logger.warning("LLM call failed or returned no content. No dynamic augmentation added.")

    except Exception as e:
        logger.error(f"Error preparing data or prompt for LLM doc generation: {e}", exc_info=True)

    return augmentation


# --- Health Check ---


async def check_marqo_availability(url: str, timeout_seconds: int = 5) -> bool:
    """Checks if the Marqo instance is available and responding via HTTP GET using httpx."""
    logger.info(f"Checking Marqo availability at {url} using httpx...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout_seconds)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Optionally, check the content if needed
            try:
                data = response.json()
                if (
                    isinstance(data, dict)
                    and "message" in data
                    and "Welcome to Marqo" in data["message"]
                ):
                    logger.success(
                        f"Marqo instance at {url} is available and responded successfully (v{data.get('version', 'unknown')})."
                    )
                    return True
                else:
                    logger.warning(
                        f"Marqo instance at {url} responded, but content was unexpected: {data}"
                    )
                    return True  # Assuming reachability is sufficient
            except json.JSONDecodeError:  # httpx raises json.JSONDecodeError
                logger.warning(
                    f"Marqo instance at {url} responded, but response was not valid JSON."
                )
                return True  # Assuming reachability is sufficient

    except httpx.TimeoutException:
        logger.error(f"Marqo check timed out after {timeout_seconds} seconds for URL: {url}")
        return False
    except httpx.RequestError as e:
        # Catches connection errors, HTTP errors (if raise_for_status is not used/fails), etc.
        logger.error(f"Marqo instance at {url} is unavailable or check failed: {e}")
        return False
    except httpx.HTTPStatusError as e:
        # Explicitly catch status errors after raise_for_status()
        logger.error(
            f"Marqo instance at {url} check failed with HTTP status {e.response.status_code}: {e}"
        )
        return False
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during Marqo health check for {url}: {e}", exc_info=True)
        return False


# --- Pydantic Models ---


class DateRange(BaseModel):
    """Date range for filtering."""

    start_date: Optional[datetime] = Field(None, description="Start date (inclusive).")
    end_date: Optional[datetime] = Field(None, description="End date (inclusive).")

    @field_validator("end_date")
    @classmethod
    def end_date_must_be_after_start_date(cls, v, info):
        """Validate that end_date is after start_date if both are provided"""
        if (
            v
            and info.data
            and "start_date" in info.data
            and info.data["start_date"]
            and v < info.data["start_date"]
        ):
            raise ValueError("end_date must be after start_date")
        return v


class MarqoSearchResult(BaseModel):
    """Individual search result from Marqo."""

    content: Optional[str] = Field(
        None,
        description="Main document content snippet from the hit, based on the schema's 'default_content_field'.",
    )
    score: float = Field(..., description="Relevance score assigned by Marqo.")
    highlights: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of highlighted text snippets matching the query, if requested."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of metadata fields associated with the hit document.",
    )
    # We won't include 'filing' here as that was specific to the smartedgar example's DB lookup


class MarqoSearchResponse(BaseModel):
    """Standardized response structure for Marqo search results."""

    results: List[MarqoSearchResult] = Field(
        default_factory=list, description="List of search result documents."
    )
    total_hits: int = Field(
        0,
        description="Total number of matching documents found by Marqo before applying limit/offset.",
    )
    limit: int = Field(0, description="The maximum number of results requested.")
    offset: int = Field(0, description="The starting offset used for pagination.")
    processing_time_ms: int = Field(
        0, description="Time taken by Marqo to process the search query, in milliseconds."
    )
    query: str = Field("", description="The original query string submitted.")
    error: Optional[str] = Field(None, description="Error message if the search operation failed.")
    success: bool = Field(
        True, description="Indicates whether the search operation was successful."
    )


# --- Helper Functions ---


def _quote_marqo_value(value: Any) -> str:
    """Formats a Python value into a string suitable for a Marqo filter query.

    Handles strings (quoting if necessary), booleans, numbers, and datetimes (converting to timestamp).

    Args:
        value: The value to format.

    Returns:
        A string representation of the value formatted for Marqo filtering.
    """
    if isinstance(value, str):
        # Escape backticks and colons within the string if needed, though Marqo is generally tolerant
        # If the string contains spaces or special characters Marqo might use for syntax, quote it.
        # Marqo's syntax is flexible, but explicit quotes can help. Using simple quotes here.
        if " " in value or ":" in value or "`" in value or "(" in value or ")" in value:
            # Basic escaping of quotes within the string
            escaped_value = value.replace("'", "\\'")
            return f"'{escaped_value}'"
        return value  # No quotes needed for simple strings
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, datetime):
        # Convert datetime to timestamp for filtering
        return str(int(value.timestamp()))
    else:
        # Default to string representation, quoted
        escaped_value = str(value).replace("'", "\\'")
        return f"'{escaped_value}'"


def _build_marqo_filter_string(
    filters: Optional[Dict[str, Any]] = None,
    date_range: Optional[DateRange] = None,
    schema: Dict[str, Any] = DEFAULT_INDEX_SCHEMA,
) -> Optional[str]:
    """Builds a Marqo filter string from various filter components based on the schema.

    Constructs a filter string compatible with Marqo's filtering syntax, handling
    date ranges and dictionary-based filters with validation against the provided schema.
    Ensures that only fields marked as 'filterable' in the schema are used.

    Args:
        filters: Dictionary where keys are schema field names and values are filter criteria
                 (single value or list for OR conditions).
        date_range: Optional DateRange object for time-based filtering.
        schema: The index schema dictionary used to validate filter fields and types.

    Returns:
        A Marqo-compatible filter string, or None if no valid filters are applicable.
    """
    filter_parts = []
    schema_fields = schema.get("fields", {})
    date_field_name = schema.get("default_date_field")

    # Add date range filter using the schema-defined date field
    if date_range and date_field_name and date_field_name in schema_fields:
        if schema_fields[date_field_name].get("type") == "timestamp":
            if date_range.start_date:
                start_ts = int(date_range.start_date.timestamp())
                filter_parts.append(f"{date_field_name}:[{start_ts} TO *]")
            if date_range.end_date:
                end_ts = int(date_range.end_date.timestamp())
                # Ensure the range includes the end date timestamp
                if date_range.start_date:
                    # Modify the start range part to include the end
                    filter_parts[-1] = f"{date_field_name}:[{start_ts} TO {end_ts}]"
                else:
                    filter_parts.append(f"{date_field_name}:[* TO {end_ts}]")
        else:
            logger.warning(
                f"Date range filtering requested, but schema field '{date_field_name}' is not type 'timestamp'. Skipping."
            )

    # Add other filters
    if filters:
        for key, value in filters.items():
            if value is None:
                continue

            # Validate field exists in schema and is filterable
            if key not in schema_fields:
                logger.warning(f"Filter key '{key}' not found in index schema. Skipping.")
                continue
            if not schema_fields[key].get("filterable", False):
                logger.warning(
                    f"Filter key '{key}' is not marked as filterable in schema. Skipping."
                )
                continue

            # Handle list values (OR condition within the key)
            if isinstance(value, list):
                if value:  # Only if list is not empty
                    # Quote each value appropriately
                    quoted_values = [_quote_marqo_value(v) for v in value]
                    # Create OR parts like field:(val1 OR val2 OR val3)
                    or_condition = " OR ".join(quoted_values)
                    filter_parts.append(f"{key}:({or_condition})")
            else:
                # Simple equality: field:value
                filter_parts.append(f"{key}:{_quote_marqo_value(value)}")

    return " AND ".join(filter_parts) if filter_parts else None


# --- Main Tool Function ---


@with_tool_metrics
@with_error_handling
async def marqo_fused_search(
    query: str,
    marqo_url: Optional[str] = None,
    index_name: Optional[str] = None,
    index_schema: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    date_range: Optional[DateRange] = None,
    semantic_weight: float = 0.7,
    limit: int = 10,
    offset: int = 0,
    highlighting: bool = True,
    rerank: bool = True,
    searchable_attributes: Optional[List[str]] = None,
    hybrid_search_attributes: Optional[
        Dict[str, List[str]]
    ] = None,  # e.g., {"tensor": ["field1"], "lexical": ["field2"]}
    client_type: str = "human",  # Could influence which 'file_type' is filtered if schema includes it
) -> Dict[str, Any]:
    """Performs a hybrid semantic and keyword search on a configured Marqo index.

    This tool automatically combines **lexical (keyword)** and **semantic (meaning-based)** search capabilities.
    You can provide simple, direct search terms. For specific phrases or keywords, the tool's lexical
    search component will find exact matches. For broader concepts or related ideas, the semantic
    search component will find relevant results based on meaning, even if the exact words aren't present.
    **Therefore, you generally do not need to formulate overly complex queries with many variations;
    trust the hybrid search to find relevant matches.**

    This tool allows searching a Marqo index with flexible filtering based on a
    provided or default index schema. It supports hybrid search, date ranges,
    metadata filtering, sorting, and faceting.

    The connection details (URL, index name) and index structure (schema) default to
    values loaded from `marqo_index_config.json` but can be overridden via parameters.

    Args:
        query: The search query string.
        marqo_url: (Optional) URL of the Marqo instance. Overrides the default from config.
        index_name: (Optional) Name of the Marqo index to search. Overrides the default from config.
        index_schema: (Optional) A dictionary describing the Marqo index structure (fields, types, roles).
                      Overrides the default from config. The schema dictates how filters,
                      sorting, and searchable attributes are interpreted.
                      Example Schema Structure:
                      `{
                          "fields": {
                              "title": {"type": "text", "role": "metadata", "searchable": "lexical"},
                              "body": {"type": "text", "role": "content"},
                              "embedding": {"type": "tensor", "role": "tensor_vector"},
                              "category": {"type": "keyword", "role": "metadata", "filterable": True},
                              "created_at": {"type": "timestamp", "role": "date", "filterable": True, "sortable": True}
                          },
                          "tensor_field": "embedding",
                          "default_content_field": "body",
                          "default_date_field": "created_at"
                      }`
        filters: (Optional) Dictionary of filters. Keys must be field names in the `index_schema` marked
                 as `"filterable": True`. Values can be single values (e.g., `"category": "news"`) or lists for
                 OR conditions (e.g., `"year": [2023, 2024]`).
        date_range: (Optional) Date range object with `start_date` and/or `end_date`. Applied to the field
                    specified by `default_date_field` in the schema.
                    **To filter by time, first inspect the available fields in the `index_schema`
                    (or the 'Configuration-Specific Notes' section below if available) to find the appropriate
                    date/timestamp field, then use this parameter.**
        semantic_weight: (Optional) Weight for semantic vs. lexical search in hybrid mode (0.0 to 1.0).
                         0.0 = pure lexical, 1.0 = pure semantic. Requires both tensor and lexical
                         fields defined in schema or `hybrid_search_attributes`. Default 0.7.
        limit: (Optional) Maximum number of results. Default 10.
        offset: (Optional) Starting offset for pagination. Default 0.
        highlighting: (Optional) Request highlights from Marqo. Default True.
        rerank: (Optional) Enable Marqo's reranking (if supported by the chosen search method/version).
                Default True.
        searchable_attributes: (Optional) Explicitly provide a list of schema field names to search for
                               TENSOR or LEXICAL search modes. Overrides auto-detection from schema.
        hybrid_search_attributes: (Optional) Explicitly provide fields for HYBRID search.
                                  Example: `{"tensor": ["embedding"], "lexical": ["title", "body"]}`.
                                  Overrides auto-detection from schema.
        client_type: (Optional) Identifier ('human', 'ai'). Can potentially be used with filters if the
                     schema includes a field like `file_type`. Default 'human'.

    Returns:
        A dictionary conforming to `MarqoSearchResponse`, containing:
        - `results`: List of `MarqoSearchResult` objects.
        - `total_hits`: Estimated total number of matching documents.
        - `limit`, `offset`, `processing_time_ms`, `query`: Search metadata.
        - `error`: Error message string if the search failed.
        - `success`: Boolean indicating success or failure.
    Example Successful Return:
        `{
            "results": [
                {
                    "content": "Example document content...",
                    "score": 0.85,
                    "highlights": [{"matched_text": "content snippet..."}],
                    "metadata": {"category": "news", "created_at": 1678886400, ...}
                }
            ],
            "total_hits": 150,
            "limit": 10,
            "offset": 0,
            "processing_time_ms": 55,
            "query": "search query text",
            "error": null,
            "success": true
        }`

    Raises:
        ToolInputError: For invalid parameters like negative limit, bad weight, unknown filter/sort fields,
                      or incompatible schema/parameter combinations.
        ToolExecutionError: If connection to Marqo fails or the search query itself fails execution on Marqo.
    """
    start_time_perf = time.perf_counter()

    # --- Use loaded config defaults or provided overrides ---
    final_marqo_url = marqo_url or DEFAULT_MARQO_URL
    final_index_name = index_name or DEFAULT_INDEX_NAME
    final_index_schema = index_schema or DEFAULT_INDEX_SCHEMA

    # --- Input Validation ---
    if not query:
        raise ToolInputError("Query cannot be empty.")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ToolInputError("semantic_weight must be between 0.0 and 1.0.")
    if limit <= 0:
        raise ToolInputError("Limit must be positive.")
    if offset < 0:
        raise ToolInputError("Offset cannot be negative.")

    # Validate schema basics
    if not isinstance(final_index_schema, dict) or "fields" not in final_index_schema:
        raise ToolInputError("Invalid index_schema format. Must be a dict with a 'fields' key.")
    schema_fields = final_index_schema.get("fields", {})

    # Validate searchable_attributes if provided
    if searchable_attributes:
        for field in searchable_attributes:
            if field not in schema_fields:
                raise ToolInputError(f"Searchable attribute '{field}' not found in index schema.")

    # Validate hybrid_search_attributes if provided
    if hybrid_search_attributes:
        for role in ["tensor", "lexical"]:
            if role in hybrid_search_attributes:
                for field in hybrid_search_attributes[role]:
                    if field not in schema_fields:
                        raise ToolInputError(
                            f"Hybrid searchable attribute '{field}' (role: {role}) not found in index schema."
                        )

    # --- Prepare Marqo Request ---
    try:
        mq = marqo.Client(url=final_marqo_url)
        marqo_index = mq.index(final_index_name)
    except Exception as e:
        raise ToolExecutionError(
            f"Failed to connect to Marqo at {final_marqo_url}: {e}", cause=e
        ) from e

    # Build filter string dynamically using the final schema
    filter_str = _build_marqo_filter_string(filters, date_range, final_index_schema)

    # Determine search method and parameters
    search_method = "TENSOR"  # Default to semantic
    marqo_search_params: Dict[str, Any] = {
        "q": query,
        "limit": limit,
        "offset": offset,
        "show_highlights": highlighting,
        # Rerank parameter might vary based on Marqo version and method
        # "re_ranker": "ms-marco-MiniLM-L-12-v2" if rerank else None
    }

    # Add filter string if generated
    if filter_str:
        marqo_search_params["filter_string"] = filter_str

    # Determine searchable attributes if not explicitly provided
    final_searchable_attributes = searchable_attributes
    final_hybrid_attributes = hybrid_search_attributes

    if not final_searchable_attributes and not final_hybrid_attributes:
        # Auto-detect attributes based on schema roles and types if not explicitly provided.
        logger.debug("Attempting to auto-detect searchable attributes from schema...")

        # 1. Identify potential tensor fields
        auto_tensor_fields = [
            name
            for name, props in schema_fields.items()
            if props.get("role") == "tensor_vector" or props.get("type") == "tensor"
        ]
        if not auto_tensor_fields:
            # Fallback: use the explicitly named top-level tensor field if roles aren't set
            tensor_field_name = final_index_schema.get("tensor_field")
            if tensor_field_name and tensor_field_name in schema_fields:
                logger.debug(f"Using schema-defined tensor_field: {tensor_field_name}")
                auto_tensor_fields = [tensor_field_name]
            else:
                logger.debug("No tensor fields identified via role or top-level schema key.")

        # 2. Identify potential lexical fields
        auto_lexical_fields = [
            name for name, props in schema_fields.items() if props.get("searchable") == "lexical"
        ]
        logger.debug(f"Auto-detected lexical fields: {auto_lexical_fields}")

        # 3. Decide configuration based on detected fields
        if auto_tensor_fields and auto_lexical_fields:
            # Both tensor and lexical fields found -> configure for HYBRID
            final_hybrid_attributes = {"tensor": auto_tensor_fields, "lexical": auto_lexical_fields}
            logger.debug(
                f"Configuring for HYBRID search with attributes: {final_hybrid_attributes}"
            )
        elif auto_tensor_fields:
            # Only tensor fields found -> configure for TENSOR
            final_searchable_attributes = auto_tensor_fields
            logger.debug(
                f"Configuring for TENSOR search with attributes: {final_searchable_attributes}"
            )
        elif auto_lexical_fields:
            # Only lexical fields found -> configure for LEXICAL
            final_searchable_attributes = auto_lexical_fields
            logger.debug(
                f"Configuring for LEXICAL search with attributes: {final_searchable_attributes}"
            )
        else:
            # Last resort: No specific searchable fields identified.
            # Default to searching the schema's 'default_content_field' lexically.
            default_content = final_index_schema.get("default_content_field")
            if default_content and default_content in schema_fields:
                final_searchable_attributes = [default_content]
                logger.warning(
                    f"No tensor or lexical fields marked in schema/params. Defaulting to LEXICAL search on field: '{default_content}'"
                )
            else:
                # Critical fallback failure - cannot determine what to search.
                raise ToolInputError(
                    "Could not determine searchable attributes from schema. Ensure schema defines roles/searchable flags, or provide explicit attributes."
                )

    # Configure Hybrid Search based on semantic_weight and determined attributes
    if final_hybrid_attributes and 0.0 < semantic_weight < 1.0:
        search_method = "HYBRID"
        marqo_search_params["search_method"] = search_method
        marqo_search_params["hybrid_parameters"] = {
            "alpha": semantic_weight,
            "searchableAttributesTensor": final_hybrid_attributes.get("tensor", []),
            "searchableAttributesLexical": final_hybrid_attributes.get("lexical", []),
            # Add other hybrid params like retrievalMethod, rankingMethod if needed/supported
            "retrievalMethod": "disjunction",  # Example
            "rankingMethod": "rrf",  # Example
        }
    elif semantic_weight == 0.0:
        search_method = "LEXICAL"
        marqo_search_params["search_method"] = search_method
        # Ensure we use lexical attributes if hybrid wasn't configured
        if final_searchable_attributes:
            marqo_search_params["searchable_attributes"] = final_searchable_attributes
        elif final_hybrid_attributes and "lexical" in final_hybrid_attributes:
            marqo_search_params["searchable_attributes"] = final_hybrid_attributes["lexical"]
        else:
            raise ToolInputError(
                "Lexical search selected (weight=0.0) but no lexical fields defined or provided."
            )

    else:  # semantic_weight == 1.0 or hybrid attributes not set for hybrid
        search_method = "TENSOR"
        marqo_search_params["search_method"] = search_method
        # Ensure we use tensor attributes
        if final_searchable_attributes:
            marqo_search_params["searchable_attributes"] = final_searchable_attributes
        elif final_hybrid_attributes and "tensor" in final_hybrid_attributes:
            marqo_search_params["searchable_attributes"] = final_hybrid_attributes["tensor"]
        else:
            # Try the schema's main tensor field
            main_tensor_field = final_index_schema.get("tensor_field")
            if main_tensor_field:
                marqo_search_params["searchable_attributes"] = [main_tensor_field]
            else:
                raise ToolInputError(
                    "Tensor search selected (weight=1.0) but no tensor fields defined or provided."
                )

    # --- Execute Search ---
    logger.info(
        f"Executing Marqo search on index '{final_index_name}' with method '{search_method}'"
    )
    logger.debug(f"Marqo search parameters: {marqo_search_params}")

    try:
        response = marqo_index.search(**marqo_search_params)
        logger.debug("Marqo response received.")
    except Exception as e:
        logger.error(f"Marqo search failed: {e}", exc_info=True)
        raise ToolExecutionError(
            f"Marqo search failed on index '{final_index_name}': {str(e)}", cause=e
        ) from e

    # --- Process Response ---
    results_list = []
    default_content_field = final_index_schema.get("default_content_field", "content")  # Fallback

    for hit in response.get("hits", []):
        metadata = {
            k: v
            for k, v in hit.items()
            if k not in ["_score", "_highlights", "_id"] and not k.startswith("__vector")
        }
        # Try to extract content from the default field, otherwise None
        content_value = hit.get(default_content_field)

        results_list.append(
            MarqoSearchResult(
                content=str(content_value) if content_value is not None else None,
                score=hit.get("_score", 0.0),
                highlights=hit.get("_highlights"),
                metadata=metadata,
            )
        )

    processing_time_ms = int(
        response.get("processingTimeMs", (time.perf_counter() - start_time_perf) * 1000)
    )

    final_response = MarqoSearchResponse(
        results=results_list,
        total_hits=response.get("nbHits", 0),  # Or use 'estimatedTotalHits' if available/preferred
        limit=response.get("limit", limit),
        offset=response.get("offset", offset),
        processing_time_ms=processing_time_ms,
        query=response.get("query", query),
        error=None,
        success=True,
    )

    return final_response.dict()


# --- Dynamically Augment Docstring ---
# Logic to generate and apply dynamic documentation based on MARQO_CONFIG via LLM call.

_docstring_augmentation_result: Optional[str] = None  # Store the generated string
_docstring_generation_done: bool = False  # Flag to ensure generation/loading happens only once


async def trigger_dynamic_docstring_generation():
    """
    Dynamically enhances the Marqo search tool docstring with index-specific documentation.

    This function uses an LLM to analyze the Marqo index configuration and generate custom
    documentation explaining the specific data domain, available filters, and example queries
    for the configured index. The resulting documentation is appended to the marqo_fused_search
    function's docstring.

    The function implements a caching mechanism:
    1. First checks for a cached docstring in the CACHE_FILE_PATH
    2. Validates cache freshness by comparing the modification time of the config file
    3. If cache is invalid or missing, calls an LLM to generate a new docstring
    4. Saves the new docstring to cache for future use

    This function should be called once during application startup, before any documentation
    is accessed. It is designed for async environments like FastAPI's startup events or
    any async initialization code.

    Dependencies:
    - Requires marqo_index_config.json to be properly configured
    - Uses CompletionClient to communicate with LLMs, requiring valid API keys
    - Needs write access to the cache directory for saving generated docstrings

    Returns:
        None. The result is applied directly to the marqo_fused_search.__doc__ attribute.
    """
    global _docstring_augmentation_result, _docstring_generation_done
    if _docstring_generation_done:
        return  # Already done

    logger.info("Checking cache and potentially triggering dynamic docstring generation...")
    cached_data = None
    current_config_mtime = 0.0

    # 1. Get config file modification time
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            current_config_mtime = os.path.getmtime(CONFIG_FILE_PATH)
        else:
            logger.warning(f"Marqo config file not found at {CONFIG_FILE_PATH} for mtime check.")
    except Exception as e:
        logger.error(f"Error getting modification time for {CONFIG_FILE_PATH}: {e}", exc_info=True)

    # 2. Try to load cache
    try:
        if os.path.exists(CACHE_FILE_PATH):
            with open(CACHE_FILE_PATH, "r") as f:
                cached_data = json.load(f)
                logger.info(f"Loaded docstring augmentation cache from {CACHE_FILE_PATH}")
    except Exception as e:
        logger.warning(
            f"Could not load or parse cache file {CACHE_FILE_PATH}: {e}. Will regenerate.",
            exc_info=True,
        )
        cached_data = None  # Ensure regeneration if cache is corrupt

    # 3. Check cache validity
    if (
        cached_data
        and isinstance(cached_data, dict)
        and "timestamp" in cached_data
        and "augmentation" in cached_data
        and current_config_mtime > 0  # Ensure we got a valid mtime for the config
        and abs(cached_data["timestamp"] - current_config_mtime)
        < 1e-6  # Compare timestamps (allowing for float precision)
    ):
        logger.info("Cache is valid. Using cached docstring augmentation.")
        _docstring_augmentation_result = cached_data["augmentation"]
    else:
        logger.info(
            "Cache invalid, missing, or config file updated. Regenerating docstring augmentation via LLM..."
        )
        # Call the async function that constructs prompt and calls LLM
        generated_augmentation = await _generate_docstring_augmentation_from_config(MARQO_CONFIG)

        if generated_augmentation:
            _docstring_augmentation_result = generated_augmentation
            # Save to cache if successful
            try:
                cache_content = {
                    "timestamp": current_config_mtime,
                    "augmentation": _docstring_augmentation_result,
                }
                with open(CACHE_FILE_PATH, "w") as f:
                    json.dump(cache_content, f, indent=2)
                logger.info(f"Saved new docstring augmentation to cache: {CACHE_FILE_PATH}")
            except Exception as e:
                logger.error(
                    f"Failed to save docstring augmentation to cache file {CACHE_FILE_PATH}: {e}",
                    exc_info=True,
                )
        else:
            _docstring_augmentation_result = ""  # Ensure it's a string even if generation fails
            logger.error("LLM generation failed. Docstring will not be augmented.")
            # Optional: Consider removing the cache file if generation fails to force retry next time?
            # try:
            #     if os.path.exists(CACHE_FILE_PATH):
            #         os.remove(CACHE_FILE_PATH)
            # except Exception as e_rem:
            #     logger.error(f"Failed to remove potentially stale cache file {CACHE_FILE_PATH}: {e_rem}")

    _docstring_generation_done = True
    logger.info("Dynamic docstring generation/loading process complete.")
    # Now apply the result (either cached or newly generated)
    _apply_generated_docstring()


def _apply_generated_docstring():
    """
    Applies the dynamically generated documentation to the marqo_fused_search function's docstring.

    This function takes the content from _docstring_augmentation_result (generated either via LLM
    or loaded from cache) and appends it to the existing docstring of the marqo_fused_search function.
    The function checks for the presence of a marker ("Configuration-Specific Notes:") to avoid
    applying the same augmentation multiple times.

    This function is called automatically at the end of trigger_dynamic_docstring_generation()
    and should not typically be called directly. It's designed as a separate function to allow
    for potential manual application in specialized scenarios.

    The function accesses the global variable _docstring_augmentation_result, which must be set
    prior to calling this function.

    Side Effects:
        Modifies marqo_fused_search.__doc__ by appending the dynamically generated content.
    """
    global _docstring_augmentation_result

    # Check if augmentation was successful and produced content
    if _docstring_augmentation_result:
        if marqo_fused_search.__doc__:
            base_doc = marqo_fused_search.__doc__.strip()
            # Avoid appending if already present (simple check)
            if "Configuration-Specific Notes:" not in base_doc:
                marqo_fused_search.__doc__ = base_doc + _docstring_augmentation_result
                logger.info(
                    f"Dynamically generated docstring augmentation applied. New length: {len(marqo_fused_search.__doc__)}"
                )
        else:
            logger.warning(
                "marqo_fused_search function is missing a base docstring. Augmentation skipped."
            )


# IMPORTANT:
# The async function `trigger_dynamic_docstring_generation()`
# must be called from your main application's async setup code
# (e.g., FastAPI startup event) *before* the tool documentation is needed.


# Example usage (for testing locally, if needed)
async def _run_test():
    # Example: Ensure dynamic docstring is generated before running the test search
    # In a real app, this trigger would happen during startup.
    await trigger_dynamic_docstring_generation()
    print("--- Current Docstring ---")
    print(marqo_fused_search.__doc__)
    print("-----------------------")

    test_query = "revenue growth"
    logger.info(f"Running test search with query: '{test_query}'")
    try:
        # Assuming MARQO_CONFIG points to the financial index for this test
        results = await marqo_fused_search(
            query=test_query,
            limit=5,
            filters={"form_type": "10-K"},  # Example filter using default schema
            # date_range=DateRange(start_date=datetime(2023, 1, 1)) # Example date range
        )
        import json

        print(json.dumps(results, indent=2))
        logger.info(f"Test search successful. Found {results['total_hits']} hits.")
    except Exception as e:
        logger.error(f"Test search failed: {e}", exc_info=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_run_test())
