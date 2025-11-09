"""Tools for LLM cost estimation, model comparison, recommendation, and workflow execution.

Provides utilities to help manage LLM usage costs and select appropriate models.
"""
import asyncio
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from ultimate_mcp_server.constants import COST_PER_MILLION_TOKENS
from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.tools.completion import chat_completion
from ultimate_mcp_server.tools.document_conversion_and_processing import (
    chunk_document,
    summarize_document,
)
from ultimate_mcp_server.tools.extraction import extract_json
from ultimate_mcp_server.tools.rag import (
    add_documents,
    create_knowledge_base,
    generate_with_rag,
    retrieve_context,
)
from ultimate_mcp_server.tools.text_classification import text_classification
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.text import count_tokens

logger = get_logger("ultimate_mcp_server.tools.optimization")

# --- Constants for Speed Score Mapping ---
# Define bins for mapping tokens/second to a 1-5 score (lower is faster)
# Adjust these thresholds based on observed performance and desired sensitivity
SPEED_SCORE_BINS = [
    (200, 1),  # > 200 tokens/s -> Score 1 (Fastest)
    (100, 2),  # 100-200 tokens/s -> Score 2
    (50, 3),   # 50-100 tokens/s -> Score 3
    (20, 4),   # 20-50 tokens/s -> Score 4
    (0, 5),    # 0-20 tokens/s -> Score 5 (Slowest)
]
DEFAULT_SPEED_SCORE = 3 # Fallback score if measurement is missing/invalid or hardcoded value is missing

def _map_tok_per_sec_to_score(tokens_per_sec: float) -> int:
    """Maps measured tokens/second to a 1-5 speed score (lower is faster)."""
    if tokens_per_sec is None or not isinstance(tokens_per_sec, (int, float)) or tokens_per_sec < 0:
        return DEFAULT_SPEED_SCORE # Return default for invalid input
    for threshold, score in SPEED_SCORE_BINS:
        if tokens_per_sec >= threshold:
            return score
    return SPEED_SCORE_BINS[-1][1] # Should hit the 0 threshold if positive

@with_tool_metrics
@with_error_handling
async def estimate_cost(
    prompt: str,
    model: str, # Can be full 'provider/model_name' or just 'model_name' if unique
    max_tokens: Optional[int] = None,
    include_output: bool = True
) -> Dict[str, Any]:
    """Estimates the monetary cost of an LLM request without executing it.

    Calculates cost based on input prompt tokens and estimated/specified output tokens
    using predefined cost rates for the specified model.

    Args:
        prompt: The text prompt that would be sent to the model.
        model: The model identifier (e.g., "openai/gpt-4.1-mini", "gpt-4.1-mini",
               "anthropic/claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022").
               Cost data must be available for the resolved model name in `COST_PER_MILLION_TOKENS`.
        max_tokens: (Optional) The maximum number of tokens expected in the output. If None,
                      output tokens are estimated as roughly half the input prompt tokens.
        include_output: (Optional) If False, calculates cost based only on input tokens, ignoring
                        `max_tokens` or output estimation. Defaults to True.

    Returns:
        A dictionary containing the cost estimate and token breakdown:
        {
            "cost": 0.000150, # Total estimated cost in USD
            "breakdown": {
                "input_cost": 0.000100,
                "output_cost": 0.000050
            },
            "tokens": {
                "input": 200,   # Tokens counted from the prompt
                "output": 100,  # Estimated or provided max_tokens
                "total": 300
            },
            "rate": {         # Cost per million tokens for this model
                "input": 0.50,
                "output": 1.50
            },
            "model": "gpt-4.1-mini", # Returns the original model string passed as input
            "resolved_model_key": "gpt-4.1-mini", # The key used for cost lookup
            "is_estimate": true
        }

    Raises:
        ToolInputError: If prompt or model format is invalid.
        ToolError: If the specified `model` cannot be resolved to cost data.
        ValueError: If token counting fails for the given model and prompt.
    """
    # Input validation
    if not prompt or not isinstance(prompt, str):
        raise ToolInputError("Prompt must be a non-empty string.")
    if not model or not isinstance(model, str):
        raise ToolInputError("Model must be a non-empty string.")

    # Flexible Cost Data Lookup
    cost_data = COST_PER_MILLION_TOKENS.get(model)
    resolved_model_key = model # Assume direct match first
    model_name_only = model # Use input model for token counting initially

    if not cost_data and '/' in model:
        # If direct lookup fails and it looks like a prefixed name, try stripping prefix
        potential_short_key = model.split('/')[-1]
        cost_data = COST_PER_MILLION_TOKENS.get(potential_short_key)
        if cost_data:
            resolved_model_key = potential_short_key
            model_name_only = potential_short_key # Use short name for token count
        # If short key also fails, cost_data remains None

    if not cost_data:
        error_message = f"Unknown model or cost data unavailable for: {model}"
        raise ToolError(error_message, error_code="MODEL_NOT_FOUND", details={"model": model})

    # Token Counting (use model_name_only derived from successful cost key)
    try:
        input_tokens = count_tokens(prompt, model=model_name_only)
    except ValueError as e:
        # Log warning with the original model input for clarity
        logger.warning(f"Could not count tokens for model '{model}' (using '{model_name_only}' for tiktoken): {e}. Using rough estimate.")
        input_tokens = len(prompt) // 4 # Fallback estimate

    # Estimate output tokens if needed
    estimated_output_tokens = 0
    if include_output:
        if max_tokens is not None:
            estimated_output_tokens = max_tokens
        else:
            estimated_output_tokens = input_tokens // 2
            logger.debug(f"max_tokens not provided, estimating output tokens as {estimated_output_tokens}")
    else:
         estimated_output_tokens = 0

    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * cost_data["input"]
    output_cost = (estimated_output_tokens / 1_000_000) * cost_data["output"]
    total_cost = input_cost + output_cost

    logger.info(f"Estimated cost for model '{model}' (using key '{resolved_model_key}'): ${total_cost:.6f} (In: {input_tokens} tokens, Out: {estimated_output_tokens} tokens)")
    return {
        "cost": total_cost,
        "breakdown": {
            "input_cost": input_cost,
            "output_cost": output_cost
        },
        "tokens": {
            "input": input_tokens,
            "output": estimated_output_tokens,
            "total": input_tokens + estimated_output_tokens
        },
        "rate": {
            "input": cost_data["input"],
            "output": cost_data["output"]
        },
        "model": model, # Return original input model string
        "resolved_model_key": resolved_model_key, # Key used for cost lookup
        "is_estimate": True
    }

@with_tool_metrics
@with_error_handling
async def compare_models(
    prompt: str,
    models: List[str], # List of model IDs (can be short or full names)
    max_tokens: Optional[int] = None,
    include_output: bool = True
) -> Dict[str, Any]:
    """Compares the estimated cost of running a prompt across multiple specified models.

    Uses the `estimate_cost` tool for each model in the list concurrently.

    Args:
        prompt: The text prompt to use for cost comparison.
        models: A list of model identifiers (e.g., ["openai/gpt-4.1-mini", "gpt-4.1-mini", "claude-3-5-haiku-20241022"]).
                `estimate_cost` will handle resolving these to cost data.
        max_tokens: (Optional) Maximum output tokens to assume for cost estimation across all models.
                      If None, output is estimated individually per model based on input.
        include_output: (Optional) Whether to include estimated output costs in the comparison. Defaults to True.

    Returns:
        A dictionary containing the cost comparison results:
        {
            "models": {
                "openai/gpt-4.1-mini": { # Uses the input model name as key
                    "cost": 0.000150,
                    "tokens": { "input": 200, "output": 100, "total": 300 }
                },
                "claude-3-5-haiku-20241022": {
                    "cost": 0.000087,
                    "tokens": { "input": 200, "output": 100, "total": 300 }
                },
                "some-unknown-model": { # Example of an error during estimation
                    "error": "Unknown model or cost data unavailable for: some-unknown-model"
                }
            },
            "ranking": [ # List of input model names ordered by cost (cheapest first), errors excluded
                "claude-3-5-haiku-20241022",
                "openai/gpt-4.1-mini"
            ],
            "cheapest": "claude-3-5-haiku-20241022", # Input model name with the lowest cost
            "most_expensive": "openai/gpt-4.1-mini", # Input model name with the highest cost
            "prompt_length_chars": 512,
            "max_tokens_assumed": 100
        }

    Raises:
        ToolInputError: If the `models` list is empty.
    """
    if not models or not isinstance(models, list):
        raise ToolInputError("'models' must be a non-empty list of model identifiers.")
    # Removed the check for '/' in model names - estimate_cost will handle resolution

    results = {}
    estimated_output_for_summary = None

    async def get_estimate(model_input_name): # Use a distinct variable name
        nonlocal estimated_output_for_summary
        try:
            estimate = await estimate_cost(
                prompt=prompt,
                model=model_input_name, # Pass the potentially short/full name
                max_tokens=max_tokens,
                include_output=include_output
            )
            # Use the original input name as the key in results
            results[model_input_name] = {
                "cost": estimate["cost"],
                "tokens": estimate["tokens"],
            }
            if estimated_output_for_summary is None:
                estimated_output_for_summary = estimate["tokens"]["output"]
        except ToolError as e:
            logger.warning(f"Could not estimate cost for model {model_input_name}: {e.detail}")
            results[model_input_name] = {"error": e.detail} # Store error under original name
        except Exception as e:
            logger.error(f"Unexpected error estimating cost for model {model_input_name}: {e}", exc_info=True)
            results[model_input_name] = {"error": f"Unexpected error: {str(e)}"}

    await asyncio.gather(*(get_estimate(model_name) for model_name in models))

    successful_estimates = {m: r for m, r in results.items() if "error" not in r}
    sorted_models = sorted(successful_estimates.items(), key=lambda item: item[1]["cost"])

    output_tokens_summary = estimated_output_for_summary if max_tokens is None else max_tokens
    if not include_output:
         output_tokens_summary = 0

    cheapest_model = sorted_models[0][0] if sorted_models else None
    most_expensive_model = sorted_models[-1][0] if sorted_models else None
    logger.info(f"Compared models: {list(results.keys())}. Cheapest: {cheapest_model or 'N/A'}")

    return {
        "models": results,
        "ranking": [m for m, _ in sorted_models], # Ranking uses original input names
        "cheapest": cheapest_model,
        "most_expensive": most_expensive_model,
        "prompt_length_chars": len(prompt),
        "max_tokens_assumed": output_tokens_summary,
    }

@with_tool_metrics
@with_error_handling
async def recommend_model(
    task_type: str,
    expected_input_length: int, # In characters
    expected_output_length: Optional[int] = None, # In characters
    required_capabilities: Optional[List[str]] = None,
    max_cost: Optional[float] = None,
    priority: str = "balanced" # Options: "cost", "quality", "speed", "balanced"
) -> Dict[str, Any]:
    """Recommends suitable LLM models based on task requirements and optimization priority.

    Evaluates known models against criteria like task type suitability (inferred),
    estimated cost (based on expected lengths), required capabilities,
    measured speed (tokens/sec if available), and quality metrics.

    Args:
        task_type: A description of the task (e.g., "summarization", "code generation", "entity extraction",
                   "customer support chat", "complex reasoning question"). Used loosely for capability checks.
        expected_input_length: Estimated length of the input text in characters.
        expected_output_length: (Optional) Estimated length of the output text in characters.
                                If None, it's roughly estimated based on input length.
        required_capabilities: (Optional) A list of specific capabilities the model MUST possess.
                               Current known capabilities include: "reasoning", "coding", "knowledge",
                               "instruction-following", "math". Check model metadata for supported values.
                               Example: ["coding", "instruction-following"]
        max_cost: (Optional) The maximum acceptable estimated cost (in USD) for a single run
                  with the expected input/output lengths. Models exceeding this are excluded.
        priority: (Optional) The primary factor for ranking suitable models.
                  Options:
                  - "cost": Prioritize the cheapest models.
                  - "quality": Prioritize models with the highest quality score.
                  - "speed": Prioritize models with the highest measured speed (tokens/sec).
                  - "balanced": (Default) Attempt to find a good mix of cost, quality, and speed.

    Returns:
        A dictionary containing model recommendations:
        {
            "recommendations": [
                {
                    "model": "anthropic/claude-3-5-haiku-20241022",
                    "estimated_cost": 0.000087,
                    "quality_score": 7,
                    "measured_speed_tps": 50.63, # Tokens per second
                    "capabilities": ["knowledge", "instruction-following"],
                    "reason": "Good balance of cost and speed, meets requirements."
                },
                {
                    "model": "openai/gpt-4.1-mini",
                    "estimated_cost": 0.000150,
                    "quality_score": 7,
                    "measured_speed_tps": 112.06,
                    "capabilities": ["reasoning", "coding", ...],
                    "reason": "Higher cost, but good quality/speed."
                }
                # ... other suitable models
            ],
            "parameters": { # Input parameters for context
                "task_type": "summarization",
                "expected_input_length": 2000,
                "expected_output_length": 500,
                "required_capabilities": [],
                "max_cost": 0.001,
                "priority": "balanced"
            },
            "excluded_models": { # Models evaluated but excluded, with reasons
                 "anthropic/claude-3-opus-20240229": "Exceeds max cost ($0.0015 > $0.001)",
                 "some-other-model": "Missing required capabilities: ['coding']"
            }
        }

    Raises:
        ToolInputError: If priority is invalid or lengths are non-positive.
    """
    if expected_input_length <= 0:
        raise ToolInputError("expected_input_length must be positive.")
    if expected_output_length is not None and expected_output_length <= 0:
        raise ToolInputError("expected_output_length must be positive if provided.")
    if priority not in ["cost", "quality", "speed", "balanced"]:
        raise ToolInputError(f"Invalid priority: '{priority}'. Must be cost, quality, speed, or balanced.")

    # --- Load Measured Speed Data ---
    measured_speeds: Dict[str, Any] = {}
    measured_speeds_file = "empirically_measured_model_speeds.json"
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filepath = os.path.join(project_root, measured_speeds_file)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                measured_speeds = json.load(f)
            logger.info(f"Successfully loaded measured speed data from {filepath}")
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load or parse measured speed data from {filepath}: {e}. Speed data will be 0.", exc_info=True)
            measured_speeds = {}
    else:
        logger.info(f"Measured speed file not found at {filepath}. Speed data will be 0.")
    # --- End Load Measured Speed Data ---

    # --- Model Metadata (Updated based on provided images) ---
    model_capabilities = {
        # OpenAI models
        "openai/gpt-4o": ["reasoning", "coding", "knowledge", "instruction-following", "math", "multimodal"], # Assuming multimodal based on general knowledge
        "openai/gpt-4o-mini": ["reasoning", "knowledge", "instruction-following"],
        "openai/gpt-4.1": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
        "openai/gpt-4.1-mini": ["reasoning", "coding", "knowledge", "instruction-following"],
        "openai/gpt-4.1-nano": ["reasoning", "knowledge", "instruction-following"], # Added reasoning
        "openai/o1-preview": ["reasoning", "coding", "knowledge", "instruction-following", "math"],
        "openai/o1": ["reasoning", "coding", "knowledge", "instruction-following", "math"], # Keep guess
        "openai/o3-mini": ["reasoning", "knowledge", "instruction-following"],

        # Anthropic models
        "anthropic/claude-3-opus-20240229": ["reasoning", "coding", "knowledge", "instruction-following", "math", "multimodal"],
        "anthropic/claude-3-sonnet-20240229": ["reasoning", "coding", "knowledge", "instruction-following", "math", "multimodal"], # Previous Sonnet version
        "anthropic/claude-3-5-haiku-20241022": ["knowledge", "instruction-following", "multimodal"], # Based on 3.5 Haiku column
        "anthropic/claude-3-5-sonnet-20241022": ["reasoning", "coding", "knowledge", "instruction-following", "math", "multimodal"], # Based on 3.5 Sonnet column
        "anthropic/claude-3-7-sonnet-20250219": ["reasoning", "coding", "knowledge", "instruction-following", "math", "multimodal"], # Based on 3.7 Sonnet column

        # DeepSeek models
        "deepseek/deepseek-chat": ["coding", "knowledge", "instruction-following"],
        "deepseek/deepseek-reasoner": ["reasoning", "math", "instruction-following"],

        # Gemini models
        "gemini/gemini-2.0-flash-lite": ["knowledge", "instruction-following"],
        "gemini/gemini-2.0-flash": ["knowledge", "instruction-following", "multimodal"],
        "gemini/gemini-2.0-flash-thinking-exp-01-21": ["reasoning", "coding", "knowledge", "instruction-following", "multimodal"],
        "gemini/gemini-2.5-pro-preview-03-25": ["reasoning", "coding", "knowledge", "instruction-following", "math", "multimodal"], # Map from gemini-2.5-pro-preview-03-25

        # Grok models (Estimates)
        "grok/grok-3-latest": ["reasoning", "knowledge", "instruction-following", "math"],
        "grok/grok-3-fast-latest": ["reasoning", "knowledge", "instruction-following"],
        "grok/grok-3-mini-latest": ["knowledge", "instruction-following"],
        "grok/grok-3-mini-fast-latest": ["knowledge", "instruction-following"],

        # OpenRouter models
        # Note: Capabilities depend heavily on the underlying model proxied by OpenRouter.
        # This is a generic entry for the one model listed in constants.py.
        "openrouter/mistralai/mistral-nemo": ["knowledge", "instruction-following", "coding"] # Estimate based on Mistral family
    }

    model_speed_fallback = {}

    model_quality = {
        "openai/gpt-4o": 8, # Updated
        "openai/gpt-4.1-mini": 7,
        "openai/gpt-4o-mini": 6,
        "openai/gpt-4.1": 8,
        "openai/gpt-4.1-nano": 5,
        "openai/o1-preview": 10,
        "openai/o3-mini": 7,

        "anthropic/claude-3-opus-20240229": 10,
        "anthropic/claude-3-sonnet-20240229": 8,
        "anthropic/claude-3-5-haiku-20241022": 7,
        "anthropic/claude-3-5-sonnet-20241022": 9,
        "anthropic/claude-3-7-sonnet-20250219": 10,

        "deepseek/deepseek-chat": 7,
        "deepseek/deepseek-reasoner": 8,

        "gemini/gemini-2.0-flash-lite": 5,
        "gemini/gemini-2.0-flash": 6,
        "gemini/gemini-2.0-flash-thinking-exp-01-21": 6,
        "gemini/gemini-2.5-pro-preview-03-25": 9,

        # Grok models (Estimates: 1-10 scale)
        "grok/grok-3-latest": 9,
        "grok/grok-3-fast-latest": 8,
        "grok/grok-3-mini-latest": 6,
        "grok/grok-3-mini-fast-latest": 6,

        # OpenRouter models (Estimates: 1-10 scale)
        "openrouter/mistralai/mistral-nemo": 7 # Estimate based on Mistral family
    }
    # --- End Model Metadata --- 

    # --- Pre-calculate model metadata lookups ---
    # Combine all known prefixed model names from metadata sources
    all_prefixed_metadata_keys = set(model_capabilities.keys()) | set(model_speed_fallback.keys()) | set(model_quality.keys())

    # Create a map from short names (e.g., "gpt-4.1-mini") to prefixed names (e.g., "openai/gpt-4.1-mini")
    # Handle potential ambiguities (same short name from different providers)
    short_to_prefixed_map: Dict[str, Optional[str]] = {}
    ambiguous_short_names = set()

    for key in all_prefixed_metadata_keys:
        if '/' in key:
            short_name = key.split('/')[-1]
            if short_name in short_to_prefixed_map:
                # Ambiguity detected
                if short_name not in ambiguous_short_names:
                     logger.warning(f"Ambiguous short model name '{short_name}' found. Maps to '{short_to_prefixed_map[short_name]}' and '{key}'. Will require full name for this model.")
                     short_to_prefixed_map[short_name] = None # Mark as ambiguous
                     ambiguous_short_names.add(short_name)
            elif short_name not in ambiguous_short_names:
                 short_to_prefixed_map[short_name] = key # Store unique mapping

    # Helper function to find the prefixed name for a cost key (using pre-calculated map)
    _prefixed_name_cache = {}
    def _get_prefixed_name_for_cost_key(cost_key: str) -> Optional[str]:
        if cost_key in _prefixed_name_cache:
            return _prefixed_name_cache[cost_key]

        # If the key is already prefixed, use it directly
        if '/' in cost_key:
             if cost_key in all_prefixed_metadata_keys:
                 _prefixed_name_cache[cost_key] = cost_key
                 return cost_key
             else:
                  # Even if prefixed, if it's not in our known metadata, treat as unknown for consistency
                  logger.warning(f"Prefixed cost key '{cost_key}' not found in any known metadata (capabilities, quality, speed).")
                  _prefixed_name_cache[cost_key] = None
                  return None

        # Look up the short name in the pre-calculated map
        prefixed_name = short_to_prefixed_map.get(cost_key)

        if prefixed_name is not None: # Found unique mapping
            _prefixed_name_cache[cost_key] = prefixed_name
            return prefixed_name
        elif cost_key in ambiguous_short_names: # Known ambiguous name
            logger.warning(f"Cannot resolve ambiguous short name '{cost_key}'. Please use the full 'provider/model_name' identifier.")
            _prefixed_name_cache[cost_key] = None
            return None
        else: # Short name not found in any metadata
             logger.warning(f"Short name cost key '{cost_key}' not found in any known model metadata. Cannot determine provider/full name.")
             _prefixed_name_cache[cost_key] = None
             return None
    # --- End Pre-calculation ---

    # Use a simple placeholder text based on length for cost estimation
    sample_text = "a" * expected_input_length
    required_capabilities = required_capabilities or []

    # Rough estimate for output length if not provided
    if expected_output_length is None:
        # Adjust this heuristic as needed (e.g., summarization shortens, generation might lengthen)
        estimated_output_length_chars = expected_input_length // 4
    else:
         estimated_output_length_chars = expected_output_length
    # Estimate max_tokens based on character length (very rough)
    estimated_max_tokens = estimated_output_length_chars // 3

    candidate_models_data = []
    excluded_models_reasons = {}
    all_cost_keys = list(COST_PER_MILLION_TOKENS.keys())

    async def evaluate_model(cost_key: str):
        # 1. Find prefixed name
        prefixed_model_name = _get_prefixed_name_for_cost_key(cost_key)
        if not prefixed_model_name:
             excluded_models_reasons[cost_key] = "Could not reliably determine provider/full name for metadata lookup."
             return

        # 2. Check capabilities
        capabilities = model_capabilities.get(prefixed_model_name, [])
        missing_caps = [cap for cap in required_capabilities if cap not in capabilities]
        if missing_caps:
            excluded_models_reasons[prefixed_model_name] = f"Missing required capabilities: {missing_caps}"
            return

        # 3. Estimate cost
        try:
            cost_estimate = await estimate_cost(
                prompt=sample_text,
                model=cost_key, # Use the key from COST_PER_MILLION_TOKENS
                max_tokens=estimated_max_tokens,
                include_output=True
            )
            estimated_cost_value = cost_estimate["cost"]
        except ToolError as e:
            excluded_models_reasons[prefixed_model_name] = f"Cost estimation failed: {e.detail}"
            return
        except Exception as e:
            logger.error(f"Unexpected error estimating cost for {cost_key} (prefixed: {prefixed_model_name}) in recommendation: {e}", exc_info=True)
            excluded_models_reasons[prefixed_model_name] = f"Cost estimation failed unexpectedly: {str(e)}"
            return

        # 4. Check max cost constraint
        if max_cost is not None and estimated_cost_value > max_cost:
            excluded_models_reasons[prefixed_model_name] = f"Exceeds max cost (${estimated_cost_value:.6f} > ${max_cost:.6f})"
            return

        # --- 5. Get Measured Speed (Tokens/Second) ---
        measured_tps = 0.0 # Default to 0.0 if no data
        speed_source = "unavailable"

        measured_data = measured_speeds.get(prefixed_model_name) or measured_speeds.get(cost_key)

        if measured_data and isinstance(measured_data, dict) and "error" not in measured_data:
            tokens_per_sec = measured_data.get("output_tokens_per_second")
            if tokens_per_sec is not None and isinstance(tokens_per_sec, (int, float)) and tokens_per_sec >= 0:
                measured_tps = float(tokens_per_sec)
                speed_source = f"measured ({measured_tps:.1f} t/s)"
            else:
                 speed_source = "no t/s in measurement"
        elif measured_data and "error" in measured_data:
                 speed_source = "measurement error"

        logger.debug(f"Speed for {prefixed_model_name}: {measured_tps:.1f} t/s (Source: {speed_source})")
        # --- End Get Measured Speed ---

        # 6. Gather data for scoring
        candidate_models_data.append({
            "model": prefixed_model_name,
            "cost_key": cost_key,
            "cost": estimated_cost_value,
            "quality": model_quality.get(prefixed_model_name, 5),
            "measured_speed_tps": measured_tps, # Store raw TPS
            "capabilities": capabilities,
            "speed_source": speed_source # Store source for potential debugging/output
        })

    # Evaluate all models
    await asyncio.gather(*(evaluate_model(key) for key in all_cost_keys))

    # --- Scoring Logic (Updated for raw TPS) ---
    def calculate_score(model_data, min_cost, cost_range, min_tps, tps_range):
        cost = model_data['cost']
        quality = model_data['quality']
        measured_tps = model_data['measured_speed_tps']

        # Normalize cost (1 is cheapest, 0 is most expensive)
        norm_cost_score = 1.0 - ((cost - min_cost) / cost_range) if cost_range > 0 else 1.0

        # Normalize quality (scale 1-10)
        norm_quality_score = quality / 10.0

        # Normalize speed (measured TPS - higher is better)
        # (1 is fastest, 0 is slowest/0)
        norm_speed_score_tps = (measured_tps - min_tps) / tps_range if tps_range > 0 else 0.0

        # Calculate final score based on priority
        if priority == "cost":
            # Lower weight for speed if using TPS, as cost is main driver
            score = norm_cost_score * 0.7 + norm_quality_score * 0.2 + norm_speed_score_tps * 0.1
        elif priority == "quality":
            score = norm_cost_score * 0.15 + norm_quality_score * 0.7 + norm_speed_score_tps * 0.15
        elif priority == "speed":
            score = norm_cost_score * 0.1 + norm_quality_score * 0.2 + norm_speed_score_tps * 0.7
        else: # balanced
            score = norm_cost_score * 0.34 + norm_quality_score * 0.33 + norm_speed_score_tps * 0.33

        return score
    # --- End Scoring Logic ---

    # Calculate scores for all candidates
    if not candidate_models_data:
        logger.warning("No candidate models found after filtering.")
    else:
        # Get min/max for normalization *before* scoring loop
        all_costs = [m['cost'] for m in candidate_models_data if m['cost'] > 0]
        min_cost = min(all_costs) if all_costs else 0.000001
        max_cost_found = max(all_costs) if all_costs else 0.000001
        cost_range = max_cost_found - min_cost

        all_tps = [m['measured_speed_tps'] for m in candidate_models_data]
        min_tps = min(all_tps) if all_tps else 0.0
        max_tps_found = max(all_tps) if all_tps else 0.0
        tps_range = max_tps_found - min_tps

        for model_data in candidate_models_data:
            # Pass normalization ranges to scoring function
            model_data['score'] = calculate_score(model_data, min_cost, cost_range, min_tps, tps_range)

    # Sort candidates by score (highest first)
    sorted_candidates = sorted(candidate_models_data, key=lambda x: x.get('score', 0), reverse=True)

    # Format recommendations
    recommendations_list = []
    if candidate_models_data:
        # Get min/max across candidates *after* filtering
        min_candidate_cost = min(m['cost'] for m in candidate_models_data)
        max_candidate_quality = max(m['quality'] for m in candidate_models_data)
        max_candidate_tps = max(m['measured_speed_tps'] for m in candidate_models_data)

        def _score_label(val: float) -> str:
            try:
                v = float(val)
            except Exception:
                v = 0.0
            if v >= 0.80:
                return "High overall score"
            if v >= 0.60:
                return "Good overall score"
            if v >= 0.40:
                return "Moderate overall score"
            return "Low overall score"

        for cand in sorted_candidates:
            reason = f"{_score_label(cand['score'])} ({cand['score']:.2f}) according to '{priority}' priority."
            # Adjust reason phrasing for TPS
            if priority == 'cost' and cand['cost'] <= min_candidate_cost:
                reason = f"Lowest estimated cost (${cand['cost']:.6f}) and meets requirements."
            elif priority == 'quality' and cand['quality'] >= max_candidate_quality:
                 reason = f"Highest quality score ({cand['quality']}/10) and meets requirements."
            elif priority == 'speed' and cand['measured_speed_tps'] >= max_candidate_tps:
                 reason = f"Fastest measured speed ({cand['measured_speed_tps']:.1f} t/s) and meets requirements."

            recommendations_list.append({
                "model": cand['model'],
                "estimated_cost": cand['cost'],
                "quality_score": cand['quality'],
                "measured_speed_tps": cand['measured_speed_tps'], # Add raw TPS
                "capabilities": cand['capabilities'],
                "reason": reason
            })

    logger.info(f"Recommended models (priority: {priority}): {[r['model'] for r in recommendations_list]}")
    return {
        "recommendations": recommendations_list,
        "parameters": { # Include input parameters for context
             "task_type": task_type,
             "expected_input_length": expected_input_length,
             "expected_output_length": estimated_output_length_chars,
             "required_capabilities": required_capabilities,
             "max_cost": max_cost,
             "priority": priority
         },
        "excluded_models": excluded_models_reasons
    }

@with_tool_metrics
@with_error_handling
async def execute_optimized_workflow(
    documents: Optional[List[str]] = None, # Make documents optional, workflow might not need them
    workflow: List[Dict[str, Any]] = None, # Require workflow definition
    max_concurrency: int = 5
) -> Dict[str, Any]:
    """Executes a predefined workflow consisting of multiple tool calls.

    Processes a list of documents (optional) through a sequence of stages defined in the workflow.
    Handles dependencies between stages (output of one stage as input to another) and allows
    for concurrent execution of independent stages or document processing within stages.

    Args:
        documents: (Optional) A list of input document strings. Required if the workflow references
                   'documents' as input for any stage.
        workflow: A list of dictionaries, where each dictionary defines a stage (a tool call).
                  Required keys per stage:
                  - `stage_id`: A unique identifier for this stage (e.g., "summarize_chunks").
                  - `tool_name`: The name of the tool function to call (e.g., "summarize_document").
                  - `params`: A dictionary of parameters to pass to the tool function.
                     Parameter values can be literal values (strings, numbers, lists) or references
                     to outputs from previous stages using the format `"${stage_id}.output_key"`
                     (e.g., `{"text": "${chunk_stage}.chunks"}`).
                     Special inputs: `"${documents}"` refers to the input `documents` list.
                  Optional keys per stage:
                  - `depends_on`: A list of `stage_id`s that must complete before this stage starts.
                  - `iterate_on`: The key from a previous stage's output list over which this stage
                                  should iterate (e.g., `"${chunk_stage}.chunks"`). The tool will be
                                  called once for each item in the list.
                  - `optimization_hints`: (Future use) Hints for model selection or cost saving for this stage.
        max_concurrency: (Optional) The maximum number of concurrent tasks (tool calls) to run.
                         Defaults to 5.

    Returns:
        A dictionary containing the results of all successful workflow stages:
        {
            "success": true,
            "results": {
                "chunk_stage": { "output": { "chunks": ["chunk1...", "chunk2..."] } },
                "summarize_chunks": { # Example of an iterated stage
                     "output": [
                         { "summary": "Summary of chunk 1..." },
                         { "summary": "Summary of chunk 2..." }
                     ]
                },
                "final_summary": { "output": { "summary": "Overall summary..." } }
            },
            "status": "Workflow completed successfully.",
            "total_processing_time": 15.8
        }
        or an error dictionary if the workflow fails:
        {
            "success": false,
            "results": { ... }, # Results up to the point of failure
            "status": "Workflow failed at stage 'stage_id'.",
            "error": "Error details from the failed stage...",
            "total_processing_time": 8.2
        }

    Raises:
        ToolInputError: If the workflow definition is invalid (missing keys, bad references,
                        circular dependencies - basic checks).
        ToolError: If a tool call within the workflow fails.
        Exception: For unexpected errors during workflow execution.
    """
    start_time = time.time()
    if not workflow or not isinstance(workflow, list):
        raise ToolInputError("'workflow' must be a non-empty list of stage dictionaries.")

    # --- Tool Mapping --- (Dynamically import or map tool names to functions)
    # Ensure all tools listed in workflows are mapped here correctly.
    
    try:
        api_meta_tool = None # Placeholder - this needs to be the actual instance
        
        if api_meta_tool: # Only add if instance is available
             meta_api_tools = {
                 "register_api": api_meta_tool.register_api,
                 "list_registered_apis": api_meta_tool.list_registered_apis,
                 "get_api_details": api_meta_tool.get_api_details,
                 "unregister_api": api_meta_tool.unregister_api,
                 "call_dynamic_tool": api_meta_tool.call_dynamic_tool,
                 "refresh_api": api_meta_tool.refresh_api,
                 "get_tool_details": api_meta_tool.get_tool_details,
                 "list_available_tools": api_meta_tool.list_available_tools,
             }
        else:
            logger.warning("APIMetaTool instance not available in execute_optimized_workflow. Meta API tools will not be callable in workflows.")
            meta_api_tools = {}
    except ImportError:
        logger.warning("APIMetaTool not found (meta_api_tool.py). Meta API tools cannot be used in workflows.")
        meta_api_tools = {}
        
    # Import extract_entity_graph lazily to avoid circular imports
    try:
        from ultimate_mcp_server.tools.entity_relation_graph import extract_entity_graph
    except ImportError:
        logger.warning("entity_relation_graph module not found. extract_entity_graph will not be available in workflows.")
        extract_entity_graph = None
        
    tool_functions = {
        # Core Gateway Tools
        "estimate_cost": estimate_cost,
        "compare_models": compare_models,
        "recommend_model": recommend_model,
        "chat_completion": chat_completion,
        "chunk_document": chunk_document,
        "summarize_document": summarize_document,
        "extract_json": extract_json,
        # Add extract_entity_graph conditionally
        **({"extract_entity_graph": extract_entity_graph} if extract_entity_graph else {}),
        # RAG Tools
        "create_knowledge_base": create_knowledge_base,
        "add_documents": add_documents,
        "retrieve_context": retrieve_context,
        "generate_with_rag": generate_with_rag,
        # Classification tools
        "text_classification": text_classification,
        
        # Merge Meta API tools
        **meta_api_tools,
        
        # Add other tools as needed...
    }

    # --- Advanced Workflow Validation Using NetworkX ---
    # Build directed graph from workflow
    dag = nx.DiGraph()
    
    # Add all stages as nodes
    for i, stage in enumerate(workflow):
        # Validate required keys
        if not all(k in stage for k in ["stage_id", "tool_name", "params"]):
            raise ToolInputError(f"Workflow stage {i} missing required keys (stage_id, tool_name, params).")
        
        stage_id = stage["stage_id"]
        
        # Validate params is a dictionary
        if not isinstance(stage["params"], dict):
            raise ToolInputError(f"Stage '{stage_id}' params must be a dictionary.")
        
        # Check for duplicate stage IDs
        if stage_id in dag:
            raise ToolInputError(f"Duplicate stage_id found: '{stage_id}'.")
        
        # Validate tool exists
        tool_name = stage["tool_name"]
        if tool_name not in tool_functions:
            raise ToolInputError(f"Unknown tool '{tool_name}' specified in stage '{stage_id}'.")
        
        # Validate depends_on is a list
        depends_on = stage.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise ToolInputError(f"Stage '{stage_id}' depends_on must be a list.")
        
        # Add node with full stage data
        dag.add_node(stage_id, stage=stage)
    
    # Add dependency edges
    for stage in workflow:
        stage_id = stage["stage_id"]
        depends_on = stage.get("depends_on", [])
        
        for dep_id in depends_on:
            if dep_id not in dag:
                raise ToolInputError(f"Stage '{stage_id}' depends on non-existent stage '{dep_id}'.")
            dag.add_edge(dep_id, stage_id)
    
    # Detect circular dependencies
    try:
        cycles = list(nx.simple_cycles(dag))
        if cycles:
            cycle_str = " -> ".join(cycles[0]) + " -> " + cycles[0][0]
            raise ToolInputError(f"Circular dependency detected in workflow: {cycle_str}")
    except nx.NetworkXNoCycle:
        # No cycles found, this is good
        pass
    
    # Dictionary to store results of each stage
    stage_results: Dict[str, Any] = {}
    # Set to keep track of completed stages
    completed_stages: Set[str] = set()
    # Dictionary to hold active tasks
    active_tasks: Dict[str, asyncio.Task] = {}  # noqa: F841
    # Semaphore to control concurrency
    concurrency_semaphore = asyncio.Semaphore(max_concurrency)
    
    # --- Workflow Execution Logic with NetworkX ---
    async def execute_stage(stage_id: str) -> None:
        """Execute a single workflow stage."""
        async with concurrency_semaphore:
            # Get stage definition
            stage = dag.nodes[stage_id]["stage"]
            tool_name = stage["tool_name"]
            params = stage["params"]
            iterate_on_ref = stage.get("iterate_on")
            
            logger.info(f"Starting workflow stage '{stage_id}' (Tool: {tool_name})")
            
            tool_func = tool_functions[tool_name]
            
            try:
                # Resolve parameters and handle iteration
                resolved_params, is_iteration, iteration_list = _resolve_params(
                    stage_id, params, iterate_on_ref, stage_results, documents
                )
                
                # Execute tool function(s)
                if is_iteration:
                    # Handle iteration case
                    iteration_tasks = []
                    
                    for i, item in enumerate(iteration_list):
                        # Create a new semaphore release for each iteration to allow other stages to run
                        # while keeping track of total concurrency
                        async def run_iteration(item_idx, item_value):
                            async with concurrency_semaphore:
                                iter_params = _inject_iteration_item(resolved_params, item_value)
                                try:
                                    result = await tool_func(**iter_params)
                                    return result
                                except Exception as e:
                                    # Capture exception details for individual iteration
                                    error_msg = f"Iteration {item_idx} failed: {type(e).__name__}: {str(e)}"
                                    logger.error(error_msg, exc_info=True)
                                    raise  # Re-raise to be caught by gather
                        
                        task = asyncio.create_task(run_iteration(i, item))
                        iteration_tasks.append(task)
                    
                    # Gather all iteration results (may raise if any iteration fails)
                    results = await asyncio.gather(*iteration_tasks)
                    stage_results[stage_id] = {"output": results}
                else:
                    # Single execution case
                    result = await tool_func(**resolved_params)
                    stage_results[stage_id] = {"output": result}
                
                # Mark stage as completed
                completed_stages.add(stage_id)
                logger.info(f"Workflow stage '{stage_id}' completed successfully")
                
            except Exception as e:
                error_msg = f"Workflow failed at stage '{stage_id}'. Error: {type(e).__name__}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                stage_results[stage_id] = {
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                }
                # Re-raise to signal failure to main execution loop
                raise
    
    async def execute_dag() -> Dict[str, Any]:
        """Execute the entire workflow DAG with proper dependency handling."""
        try:
            # Start with a topological sort to get execution order respecting dependencies
            try:
                execution_order = list(nx.topological_sort(dag))
                logger.debug(f"Workflow execution order (respecting dependencies): {execution_order}")
            except nx.NetworkXUnfeasible as e:
                # Should never happen as we already checked for cycles
                raise ToolInputError("Workflow contains circular dependencies that were not detected earlier.") from e
            
            # Process stages in waves of parallelizable tasks
            while len(completed_stages) < len(dag):
                # Find stages ready to execute (all dependencies satisfied)
                ready_stages = [
                    stage_id for stage_id in execution_order
                    if (stage_id not in completed_stages and 
                        all(pred in completed_stages for pred in dag.predecessors(stage_id)))
                ]
                
                if not ready_stages:
                    if len(completed_stages) < len(dag):
                        # This should never happen with a valid DAG that was topologically sorted
                        unfinished = set(execution_order) - completed_stages
                        logger.error(f"Workflow execution stalled. Unfinished stages: {unfinished}")
                        raise ToolError("Workflow execution stalled due to unresolvable dependencies.")
                    break
                
                # Launch tasks for all ready stages
                tasks = [execute_stage(stage_id) for stage_id in ready_stages]
                
                # Wait for all tasks to complete or for the first error
                try:
                    await asyncio.gather(*tasks)
                except Exception as e:
                    # Any stage failure will be caught here
                    # The specific error details are already in stage_results
                    logger.error(f"Workflow wave execution failed: {str(e)}")
                    
                    # Find the first failed stage for error reporting
                    failed_stage = next(
                        (s for s in ready_stages if s in stage_results and "error" in stage_results[s]),
                        ready_stages[0]  # Fallback if we can't identify the specific failed stage
                    )
                    
                    error_info = stage_results.get(failed_stage, {}).get("error", f"Unknown error in stage '{failed_stage}'")
                    
                    return {
                        "success": False,
                        "results": stage_results,
                        "status": f"Workflow failed at stage '{failed_stage}'.",
                        "error": error_info,
                        "total_processing_time": time.time() - start_time
                    }
                
                # If we reach here, all stages in this wave completed successfully
            
            # All stages completed successfully
            return {
                "success": True,
                "results": stage_results,
                "status": "Workflow completed successfully.",
                "total_processing_time": time.time() - start_time
            }
            
        except Exception as e:
            # Catch any unexpected errors in the main execution loop
            error_msg = f"Unexpected error in workflow execution: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "results": stage_results,
                "status": "Workflow failed with an unexpected error.",
                "error": error_msg,
                "total_processing_time": time.time() - start_time
            }
    
    # Execute the workflow DAG
    result = await execute_dag()
    
    total_time = time.time() - start_time
    if result["success"]:
        logger.info(f"Workflow completed successfully in {total_time:.2f}s")
    else:
        logger.error(f"Workflow failed after {total_time:.2f}s: {result.get('error', 'Unknown error')}")
    
    return result

# --- Helper functions for workflow execution --- 
# These need careful implementation for robustness

def _resolve_params(stage_id: str, params: Dict, iterate_on_ref: Optional[str], stage_results: Dict, documents: Optional[List[str]]) -> tuple[Dict, bool, Optional[List]]:
    """Resolves parameter values, handling references and iteration.
    Returns resolved_params, is_iteration, iteration_list.
    Raises ValueError on resolution errors.
    """
    resolved = {}
    is_iteration = False
    iteration_list = None
    iteration_param_name = None

    # Check for iteration first
    if iterate_on_ref:
         if not iterate_on_ref.startswith("${") or not iterate_on_ref.endswith("}"):
              raise ValueError(f"Invalid iterate_on reference format: '{iterate_on_ref}'")
         ref_key = iterate_on_ref[2:-1]
         
         if ref_key == "documents":
              if documents is None:
                   raise ValueError(f"Stage '{stage_id}' iterates on documents, but no documents were provided.")
              iteration_list = documents
         else:
              dep_stage_id, output_key = _parse_ref(ref_key)
              if dep_stage_id not in stage_results or "output" not in stage_results[dep_stage_id]:
                   raise ValueError(f"Dependency '{dep_stage_id}' for iteration not found or failed.")
              dep_output = stage_results[dep_stage_id]["output"]
              if not isinstance(dep_output, dict) or output_key not in dep_output:
                   raise ValueError(f"Output key '{output_key}' not found in dependency '{dep_stage_id}' for iteration.")
              iteration_list = dep_output[output_key]
              if not isinstance(iteration_list, list):
                  raise ValueError(f"Iteration target '{ref_key}' is not a list.")
         
         is_iteration = True
         # We still resolve other params, the iteration item is injected later
         logger.debug(f"Stage '{stage_id}' will iterate over {len(iteration_list)} items from '{iterate_on_ref}'")

    # Resolve individual parameters
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            ref_key = value[2:-1]
            if ref_key == "documents":
                 if documents is None:
                      raise ValueError(f"Parameter '{key}' references documents, but no documents provided.")
                 resolved[key] = documents
            else:
                dep_stage_id, output_key = _parse_ref(ref_key)
                if dep_stage_id not in stage_results or "output" not in stage_results[dep_stage_id]:
                    raise ValueError(f"Dependency '{dep_stage_id}' for parameter '{key}' not found or failed.")
                dep_output = stage_results[dep_stage_id]["output"]
                # Handle potential nested keys in output_key later if needed
                if not isinstance(dep_output, dict) or output_key not in dep_output:
                    raise ValueError(f"Output key '{output_key}' not found in dependency '{dep_stage_id}' for parameter '{key}'. Available keys: {list(dep_output.keys()) if isinstance(dep_output, dict) else 'N/A'}")
                resolved[key] = dep_output[output_key]
                # If this resolved param is the one we iterate on, store its name
                if is_iteration and iterate_on_ref == value:
                     iteration_param_name = key
        else:
            resolved[key] = value # Literal value
            
    # Validation: If iterating, one parameter must match the iterate_on reference
    if is_iteration and iteration_param_name is None:
         # This means iterate_on pointed to something not used directly as a param value
         # We need a convention here, e.g., assume the tool takes a list or find the param name
         # For now, let's assume the tool expects the *list* if iterate_on isn't directly a param value.
         # This might need refinement based on tool behavior. A clearer workflow definition could help.
         # Alternative: Raise error if iterate_on target isn't explicitly mapped to a param. 
         # logger.warning(f"Iteration target '{iterate_on_ref}' not directly mapped to a parameter in stage '{stage_id}'. Tool must handle list input.")
         # Let's require the iteration target to be mapped for clarity:
          raise ValueError(f"Iteration target '{iterate_on_ref}' must correspond to a parameter value in stage '{stage_id}'.")

    # Remove the iteration parameter itself from the base resolved params if iterating
    # It will be injected per-item later
    if is_iteration and iteration_param_name in resolved:
        del resolved[iteration_param_name] 
        resolved["_iteration_param_name"] = iteration_param_name # Store the name for injection

    return resolved, is_iteration, iteration_list

def _parse_ref(ref_key: str) -> tuple[str, str]:
    """Parses a reference like 'stage_id.output_key'"""
    parts = ref_key.split('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid reference format: '{ref_key}'. Expected 'stage_id.output_key'.")
    return parts[0], parts[1]

def _inject_iteration_item(base_params: Dict, item: Any) -> Dict:
     """Injects the current iteration item into the parameter dict."""
     injected_params = base_params.copy()
     iter_param_name = injected_params.pop("_iteration_param_name", None)
     if iter_param_name:
          injected_params[iter_param_name] = item
     else:
          # This case should be prevented by validation in _resolve_params
          logger.error("Cannot inject iteration item: Iteration parameter name not found in resolved params.")
          # Handle error appropriately, maybe raise
     return injected_params

async def _gather_iteration_results(stage_id: str, tasks: List[asyncio.Task]) -> List[Any]:
     """Gathers results from iteration sub-tasks. Raises exception if any sub-task failed."""
     results = []
     try:
          raw_results = await asyncio.gather(*tasks)
          # Assume each task returns the direct output dictionary
          results = list(raw_results) # gather preserves order
          logger.debug(f"Iteration stage '{stage_id}' completed with {len(results)} results.")
          return results
     except Exception:
          # If any sub-task failed, gather will raise the first exception
          logger.error(f"Iteration stage '{stage_id}' failed: One or more sub-tasks raised an error.", exc_info=True)
          # Cancel any remaining tasks in this iteration group if needed (gather might do this)
          for task in tasks:
               if not task.done(): 
                   task.cancel()
          raise # Re-raise the exception to fail the main workflow stage