"""Classification tools for Ultimate MCP Server."""

import json
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import get_provider
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError
from ultimate_mcp_server.services.cache import with_cache
from ultimate_mcp_server.tools.base import (
    with_error_handling,
    with_retry,
    with_tool_metrics,
)
from ultimate_mcp_server.tools.completion import generate_completion
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.text import preprocess_text

logger = get_logger("ultimate_mcp_server.tools.classification")


class ClassificationStrategy(Enum):
    """Strategies for text classification."""

    ZERO_SHOT = "zero_shot"  # Pure zero-shot classification
    FEW_SHOT = "few_shot"  # Few-shot examples included
    STRUCTURED = "structured"  # Structured output with reasoning
    ENSEMBLE = "ensemble"  # Combine multiple providers/models
    SEMANTIC = "semantic"  # Use semantic similarity


@with_cache(ttl=24 * 60 * 60)  # Cache results for 24 hours
@with_tool_metrics
@with_retry(max_retries=2, retry_delay=1.0)
@with_error_handling
async def text_classification(
    text: str,
    categories: Union[List[str], Dict[str, List[str]]],  # Simple list or hierarchical dict
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    multi_label: bool = False,
    confidence_threshold: float = 0.5,
    strategy: Union[str, ClassificationStrategy] = ClassificationStrategy.STRUCTURED,
    examples: Optional[List[Dict[str, Any]]] = None,
    custom_prompt_template: Optional[str] = None,
    max_results: int = 5,
    explanation_detail: str = "brief",  # "none", "brief", "detailed"
    preprocessing: bool = True,
    ensemble_config: Optional[List[Dict[str, Any]]] = None,
    taxonomy_description: Optional[str] = None,
    output_format: str = "json",  # "json", "text", "markdown"
    additional_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classifies text into one or more predefined categories using an LLM.

    Provides powerful text classification capabilities, including hierarchical categories,
    example-based few-shot learning, ensemble classification, custom prompting,
    and detailed explanations.

    Args:
        text: The input text to classify.
        categories: Either a list of category strings OR a dictionary mapping parent categories
                  to lists of subcategories (for hierarchical classification).
                  Example dict: {"Animals": ["Dog", "Cat"], "Vehicles": ["Car", "Boat"]}
        provider: The LLM provider (e.g., "openai", "anthropic", "gemini"). Defaults to "openai".
        model: The specific model ID. If None, the provider's default model is used.
        multi_label: If True, allows classification into multiple categories. Default False.
        confidence_threshold: Minimum confidence score (0.0-1.0) for a category to be included. Default 0.5.
        strategy: Classification approach to use. Options:
                - "zero_shot": Pure zero-shot classification
                - "few_shot": Use provided examples to demonstrate the task
                - "structured": (Default) Generate structured output with reasoning
                - "ensemble": Combine results from multiple models
                - "semantic": Use semantic similarity to match categories
        examples: Optional list of example classifications for few-shot learning.
                Each example should be a dict with "text" and "categories" keys.
                Example: [{"text": "I love my dog", "categories": ["Animals", "Pets"]}]
        custom_prompt_template: Optional custom prompt template with placeholders:
                {categories}, {format_instruction}, {confidence_threshold},
                {examples}, {taxonomy_description}, {text}
        max_results: Maximum number of categories to return (only affects multi_label=True). Default 5.
        explanation_detail: Level of explanation to include: "none", "brief" (default), or "detailed"
        preprocessing: If True, performs text cleanup and normalization before classification. Default True.
        ensemble_config: For strategy="ensemble", list of provider/model configurations.
                Example: [{"provider": "openai", "model": "gpt-4.1-mini", "weight": 0.7},
                          {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "weight": 0.3}]
        taxonomy_description: Optional description of the classification taxonomy to help guide the model.
        output_format: Format for classification result: "json" (default), "text", or "markdown"
        additional_params: Additional provider-specific parameters.

    Returns:
        A dictionary containing:
        {
            "classifications": [
                {
                    "category": "category_name",  # Or hierarchical: "parent_category/subcategory"
                    "confidence": 0.95,
                    "explanation": "Explanation for this classification"
                },
                ...
            ],
            "dominant_category": "most_confident_category",  # Only present if multi_label=True
            "provider": "provider-name",
            "model": "model-used",
            "tokens": {
                "input": 150,
                "output": 80,
                "total": 230
            },
            "cost": 0.000345,
            "processing_time": 1.23,
            "cached_result": false,  # Added by cache decorator
            "success": true
        }

    Raises:
        ToolInputError: If input parameters are invalid.
        ProviderError: If the provider is unavailable or classification fails.
        ToolError: For other errors during classification processing.
    """
    start_time = time.time()

    # --- Input Validation ---
    if not text or not isinstance(text, str):
        raise ToolInputError("Text must be a non-empty string.")

    # Validate categories format
    is_hierarchical = isinstance(categories, dict)

    if is_hierarchical:
        if not all(
            isinstance(parent, str)
            and isinstance(subcats, list)
            and all(isinstance(sub, str) for sub in subcats)
            for parent, subcats in categories.items()
        ):
            raise ToolInputError(
                "Hierarchical categories must be a dictionary mapping string keys to lists of string values."
            )
        # Create flattened list of all categories for validation later
        flat_categories = []
        for parent, subcats in categories.items():
            flat_categories.append(parent)  # Add parent itself as a category
            for sub in subcats:
                flat_categories.append(f"{parent}/{sub}")  # Add hierarchical path
    else:
        if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories):
            raise ToolInputError("Categories must be a non-empty list of strings.")
        flat_categories = categories

    if not flat_categories:
        raise ToolInputError("At least one category must be provided.")

    # Validate confidence threshold
    if (
        not isinstance(confidence_threshold, (int, float))
        or confidence_threshold < 0.0
        or confidence_threshold > 1.0
    ):
        raise ToolInputError(
            "Confidence threshold must be between 0.0 and 1.0.",
            param_name="confidence_threshold",
            provided_value=confidence_threshold,
        )

    # Validate strategy
    if isinstance(strategy, str):
        try:
            strategy = ClassificationStrategy(strategy)
        except ValueError as e:
            valid_strategies = [s.value for s in ClassificationStrategy]
            raise ToolInputError(
                f"Invalid strategy: '{strategy}'. Valid options are: {', '.join(valid_strategies)}",
                param_name="strategy",
                provided_value=strategy,
            ) from e
    elif not isinstance(strategy, ClassificationStrategy):
        raise ToolInputError("Strategy must be a string or ClassificationStrategy enum value.")

    # Validate examples for few-shot learning
    if examples is not None:
        if not isinstance(examples, list):
            raise ToolInputError("Examples must be a list of dictionaries.")
        for i, ex in enumerate(examples):
            if not isinstance(ex, dict) or "text" not in ex or "categories" not in ex:
                raise ToolInputError(
                    f"Example at index {i} must be a dictionary with 'text' and 'categories' keys."
                )

    # Validate ensemble configuration
    if strategy == ClassificationStrategy.ENSEMBLE:
        if not ensemble_config or not isinstance(ensemble_config, list):
            raise ToolInputError(
                "For ensemble strategy, ensemble_config must be a non-empty list of provider configurations."
            )
        for i, config in enumerate(ensemble_config):
            if not isinstance(config, dict) or "provider" not in config:
                raise ToolInputError(
                    f"Ensemble config at index {i} must be a dictionary with at least a 'provider' key."
                )

    # Validate explanation detail
    if explanation_detail not in ["none", "brief", "detailed"]:
        raise ToolInputError(
            f"Invalid explanation_detail: '{explanation_detail}'. Valid options are: none, brief, detailed.",
            param_name="explanation_detail",
            provided_value=explanation_detail,
        )

    # Validate output format
    if output_format not in ["json", "text", "markdown"]:
        raise ToolInputError(
            f"Invalid output_format: '{output_format}'. Valid options are: json, text, markdown.",
            param_name="output_format",
            provided_value=output_format,
        )

    # --- Text Preprocessing ---
    if preprocessing:
        # Assume preprocess_text exists in ultimate_mcp_server.utils.text
        original_length = len(text)
        text = preprocess_text(text)
        logger.debug(f"Preprocessed text from {original_length} to {len(text)} characters.")

    # --- Classification Strategy Execution ---
    if strategy == ClassificationStrategy.ENSEMBLE:
        # Handle ensemble classification
        result = await _perform_ensemble_classification(
            text,
            categories,
            is_hierarchical,
            multi_label,
            confidence_threshold,
            max_results,
            explanation_detail,
            ensemble_config,
            taxonomy_description,
            output_format,
            additional_params,
        )
    elif strategy == ClassificationStrategy.SEMANTIC:
        # Handle semantic similarity classification
        result = await _perform_semantic_classification(
            text,
            categories,
            is_hierarchical,
            multi_label,
            confidence_threshold,
            max_results,
            explanation_detail,
            provider,
            model,
            additional_params,
        )
    else:
        # Handle standard LLM classification (zero-shot, few-shot, structured)
        result = await _perform_standard_classification(
            text,
            categories,
            is_hierarchical,
            multi_label,
            confidence_threshold,
            max_results,
            explanation_detail,
            examples,
            custom_prompt_template,
            taxonomy_description,
            output_format,
            strategy.value,
            provider,
            model,
            additional_params,
            flat_categories,
        )

    # --- Post-processing ---
    # Calculate processing time
    processing_time = time.time() - start_time

    # Add processing time to result
    result["processing_time"] = processing_time

    # Log success
    logger.success(
        f"Text classification completed successfully using {strategy.value} strategy with {result['provider']}/{result['model']}",
        emoji_key="classification",  # Using string directly instead of enum
        tokens=result.get("tokens", {}),
        cost=result.get("cost", 0.0),
        time=processing_time,
        categories_found=len(result.get("classifications", [])),
    )

    return result


# --- Strategy Implementation Functions ---


async def _perform_standard_classification(
    text: str,
    categories: Union[List[str], Dict[str, List[str]]],
    is_hierarchical: bool,
    multi_label: bool,
    confidence_threshold: float,
    max_results: int,
    explanation_detail: str,
    examples: Optional[List[Dict[str, Any]]],
    custom_prompt_template: Optional[str],
    taxonomy_description: Optional[str],
    output_format: str,
    strategy: str,
    provider: str,
    model: Optional[str],
    additional_params: Optional[Dict[str, Any]],
    flat_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Performs classification using a single LLM with standard prompting."""
    # Get provider instance
    try:
        provider_instance = await get_provider(provider)  # noqa: F841
    except Exception as e:
        raise ProviderError(
            f"Failed to initialize provider '{provider}': {str(e)}", provider=provider, cause=e
        ) from e

    # Set default additional params
    additional_params = additional_params or {}

    # --- Build Classification Prompt ---

    # Format the categories list/hierarchy
    if is_hierarchical:
        categories_text = ""
        for parent, subcategories in categories.items():
            categories_text += f"- {parent}\n"
            for sub in subcategories:
                categories_text += f"  - {parent}/{sub}\n"
    else:
        categories_text = "\n".join([f"- {category}" for category in categories])

    # Determine format instruction based on strategy and parameters
    if multi_label:
        classification_type = "one or more categories"
    else:
        classification_type = "exactly one category"

    # Explanation detail instruction
    if explanation_detail == "none":
        explanation_instruction = "No explanation needed."
    elif explanation_detail == "detailed":
        explanation_instruction = """Include a detailed explanation for each classification, covering:
- Specific evidence from the text
- How this evidence relates to the category
- Any potential ambiguities or edge cases considered"""
    else:  # brief
        explanation_instruction = "Include a brief explanation justifying each classification."

    # Format instruction for output
    if output_format == "json":
        format_instruction = f"""For each matching category, include:
1. The category name (exactly as provided)
2. A confidence score between 0.0 and 1.0
3. {explanation_instruction}

Format your response as valid JSON with the following structure:
{{
  "classifications": [
    {{
      "category": "category_name",
      "confidence": 0.95,
      "explanation": "Justification for this classification"
    }}
    // More categories if multi-label is true and multiple categories match
  ]
}}

Only include categories with confidence scores above {confidence_threshold}.
{"Limit your response to the top " + str(max_results) + " most confident categories." if multi_label else ""}"""
    elif output_format == "markdown":
        format_instruction = f"""For each matching category, include:
1. The category name (exactly as provided)
2. A confidence score between 0.0 and 1.0
3. {explanation_instruction}

Format your response using markdown:
## Classifications
{
            '''
- **Category**: category_name
  - **Confidence**: 0.95
  - **Explanation**: Justification for this classification
'''
            if explanation_detail != "none"
            else '''
- **Category**: category_name
  - **Confidence**: 0.95
'''
        }

Only include categories with confidence scores above {confidence_threshold}.
{
            "Limit your response to the top " + str(max_results) + " most confident categories."
            if multi_label
            else ""
        }"""
    else:  # text
        format_instruction = f"""For each matching category, include:
1. The category name (exactly as provided)
2. A confidence score between 0.0 and 1.0
3. {explanation_instruction}

Format your response as plain text:
CATEGORY: category_name
CONFIDENCE: 0.95
{"EXPLANATION: Justification for this classification" if explanation_detail != "none" else ""}

Only include categories with confidence scores above {confidence_threshold}.
{"Limit your response to the top " + str(max_results) + " most confident categories." if multi_label else ""}"""

    # Add few-shot examples if provided
    examples_text = ""
    if examples and strategy == "few_shot":
        examples_text = "\n\nEXAMPLES:\n"
        for i, ex in enumerate(examples):
            examples_text += f"\nExample {i + 1}:\nText: {ex['text']}\n"
            if isinstance(ex["categories"], list):
                examples_text += f"Categories: {', '.join(ex['categories'])}\n"
            else:
                examples_text += f"Category: {ex['categories']}\n"

    # Add taxonomy description if provided
    taxonomy_text = ""
    if taxonomy_description:
        taxonomy_text = f"\nTAXONOMY DESCRIPTION:\n{taxonomy_description}\n"

    # Build the final prompt, using custom template if provided
    if custom_prompt_template:
        # Replace placeholders in custom template
        prompt = custom_prompt_template
        replacements = {
            "{categories}": categories_text,
            "{format_instruction}": format_instruction,
            "{confidence_threshold}": str(confidence_threshold),
            "{examples}": examples_text,
            "{taxonomy_description}": taxonomy_text,
            "{text}": text,
        }
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)
    else:
        # Use the standard prompt structure
        prompt = f"""Classify the following text into {classification_type} from this list:
{categories_text}{taxonomy_text}{examples_text}

{format_instruction}

Text to classify:
{text}
"""

    # --- Execute Classification Request ---
    try:
        # Use low temperature for more deterministic results
        temperature = additional_params.pop("temperature", 0.1)

        # Use the standardized completion tool
        completion_result = await generate_completion(
            prompt=prompt,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=1000,  # Generous token limit for detailed explanations
            additional_params=additional_params,
        )

        # Check if completion was successful
        if not completion_result.get("success", False):
            error_message = completion_result.get("error", "Unknown error during completion")
            raise ProviderError(
                f"Text classification failed: {error_message}",
                provider=provider,
                model=model or "default",
            )

        # --- Parse Response Based on Format ---
        classifications = []

        if output_format == "json":
            classifications = _parse_json_response(completion_result["text"], confidence_threshold)
        elif output_format == "markdown":
            classifications = _parse_markdown_response(
                completion_result["text"], confidence_threshold
            )
        else:  # text
            classifications = _parse_text_response(completion_result["text"], confidence_threshold)

        # Validate classifications against provided categories
        categories_to_validate = flat_categories if flat_categories is not None else categories
        _validate_classifications(classifications, categories_to_validate)

        # Sort by confidence and limit to max_results
        classifications = sorted(
            classifications, key=lambda x: x.get("confidence", 0), reverse=True
        )
        if multi_label and len(classifications) > max_results:
            classifications = classifications[:max_results]
        elif not multi_label and len(classifications) > 1:
            # For single-label, take only the highest confidence one
            classifications = classifications[:1]

        # Determine dominant category if multi-label
        dominant_category = None
        if multi_label and classifications:
            dominant_category = classifications[0]["category"]

        # --- Build Result ---
        classification_result = {
            "classifications": classifications,
            "provider": provider,
            "model": completion_result["model"],
            "tokens": completion_result["tokens"],
            "cost": completion_result["cost"],
            "success": True,
        }

        # Add dominant category if multi-label
        if multi_label:
            classification_result["dominant_category"] = dominant_category

        return classification_result

    except Exception as e:
        # Handle errors
        error_model = model or f"{provider}/default"
        raise ProviderError(
            f"Text classification failed for model '{error_model}': {str(e)}",
            provider=provider,
            model=error_model,
            cause=e,
        ) from e


async def _perform_ensemble_classification(
    text: str,
    categories: Union[List[str], Dict[str, List[str]]],
    is_hierarchical: bool,
    multi_label: bool,
    confidence_threshold: float,
    max_results: int,
    explanation_detail: str,
    ensemble_config: List[Dict[str, Any]],
    taxonomy_description: Optional[str],
    output_format: str,
    additional_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Performs ensemble classification using multiple models and aggregates the results."""
    # Track total tokens and cost
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    # Start with equal weights if not specified
    normalized_configs = []
    total_weight = 0.0

    for config in ensemble_config:
        weight = config.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or weight <= 0:
            weight = 1.0
        total_weight += weight
        normalized_configs.append({**config, "weight": weight})

    # Normalize weights
    for config in normalized_configs:
        config["weight"] = config["weight"] / total_weight

    # Execute classification with each model in parallel
    classification_tasks = []

    for config in normalized_configs:
        model_provider = config.get("provider")
        model_name = config.get("model")
        model_params = config.get("params", {})

        # Combine with global additional_params
        combined_params = {**(additional_params or {}), **model_params}

        # Create task for this model's classification
        task = _perform_standard_classification(
            text=text,
            categories=categories,
            is_hierarchical=is_hierarchical,
            multi_label=True,  # Always use multi-label for ensemble components
            confidence_threshold=0.0,  # Get all results for ensemble aggregation
            max_results=100,  # High limit to get comprehensive results
            explanation_detail="brief",  # Simplify for ensemble components
            examples=None,
            custom_prompt_template=None,
            taxonomy_description=taxonomy_description,
            output_format="json",  # Always use JSON for easy aggregation
            strategy="structured",
            provider=model_provider,
            model=model_name,
            additional_params=combined_params,
        )

        classification_tasks.append((config, task))

    # Collect all model results
    model_results = {}
    provider_model_used = "ensemble"

    for config, task in classification_tasks:
        try:
            result = await task
            model_id = f"{config['provider']}/{result['model']}"
            model_results[model_id] = {
                "classifications": result["classifications"],
                "weight": config["weight"],
                "tokens": result.get("tokens", {}),
                "cost": result.get("cost", 0.0),
            }

            # Accumulate tokens and cost
            total_input_tokens += result.get("tokens", {}).get("input", 0)
            total_output_tokens += result.get("tokens", {}).get("output", 0)
            total_cost += result.get("cost", 0.0)

            # Just use the first successful model as the "provider" for the result
            if provider_model_used == "ensemble":
                provider_model_used = model_id

        except Exception as e:
            logger.warning(
                f"Ensemble model {config['provider']}/{config.get('model', 'default')} failed: {str(e)}"
            )
            # Continue with other models

    if not model_results:
        raise ToolError(
            "All models in the ensemble failed to produce classifications.",
            error_code="ENSEMBLE_FAILURE",
        )

    # --- Aggregate Results ---
    # Create a map of category -> aggregated confidence and explanations
    aggregated = {}

    for model_id, result in model_results.items():
        model_weight = result["weight"]

        for cls in result["classifications"]:
            category = cls["category"]
            conf = cls.get("confidence", 0.0)
            expl = cls.get("explanation", "")

            weighted_conf = conf * model_weight

            if category not in aggregated:
                aggregated[category] = {
                    "category": category,
                    "confidence": weighted_conf,
                    "total_weight": model_weight,
                    "explanations": [],
                    "models": [],
                }
            else:
                aggregated[category]["confidence"] += weighted_conf
                aggregated[category]["total_weight"] += model_weight

            # Store explanation with model attribution
            if expl:
                aggregated[category]["explanations"].append(f"({model_id}): {expl}")

            # Track which models classified this category
            aggregated[category]["models"].append(model_id)

    # Finalize aggregation
    final_classifications = []

    for category, agg in aggregated.items():
        # Normalize confidence by total weight that contributed to this category
        if agg["total_weight"] > 0:
            normalized_confidence = agg["confidence"] / agg["total_weight"]
        else:
            normalized_confidence = 0.0

        # Only keep categories above threshold
        if normalized_confidence >= confidence_threshold:
            # Generate combined explanation based on detail level
            if explanation_detail == "none":
                combined_explanation = ""
            elif explanation_detail == "brief":
                model_count = len(agg["models"])
                combined_explanation = f"Classified by {model_count} model{'s' if model_count != 1 else ''} with average confidence {normalized_confidence:.2f}"
            else:  # detailed
                combined_explanation = "Classified by models: " + ", ".join(agg["models"]) + "\n"
                combined_explanation += "\n".join(agg["explanations"])

            final_classifications.append(
                {
                    "category": category,
                    "confidence": normalized_confidence,
                    "explanation": combined_explanation,
                    "contributing_models": len(agg["models"]),
                }
            )

    # Sort by confidence and limit results
    final_classifications = sorted(
        final_classifications, key=lambda x: x["confidence"], reverse=True
    )

    if not multi_label:
        # For single-label, take only the highest confidence
        if final_classifications:
            final_classifications = [final_classifications[0]]
    elif len(final_classifications) > max_results:
        # For multi-label, limit to max_results
        final_classifications = final_classifications[:max_results]

    # Determine dominant category if multi-label
    dominant_category = None
    if multi_label and final_classifications:
        dominant_category = final_classifications[0]["category"]

    # Build final result
    ensemble_result = {
        "classifications": final_classifications,
        "provider": "ensemble",
        "model": provider_model_used,  # Use the first successful model as identifier
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_input_tokens + total_output_tokens,
        },
        "cost": total_cost,
        "ensemble_models": list(model_results.keys()),
        "success": True,
    }

    # Add dominant category if multi-label
    if multi_label:
        ensemble_result["dominant_category"] = dominant_category

    return ensemble_result


async def _perform_semantic_classification(
    text: str,
    categories: Union[List[str], Dict[str, List[str]]],
    is_hierarchical: bool,
    multi_label: bool,
    confidence_threshold: float,
    max_results: int,
    explanation_detail: str,
    provider: str,
    model: Optional[str],
    additional_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Performs classification using semantic similarity between embeddings of text and categories.
    This is a fallback method when LLM-based classification is not ideal.
    """
    # This would need to be implemented using embedding functionality
    # For now, we'll create a placeholder implementation that delegates to standard classification
    # In a real implementation, we would:
    # 1. Generate embeddings for the input text
    # 2. Generate embeddings for each category (possibly with descriptions)
    # 3. Calculate cosine similarity scores
    # 4. Use scores as confidence values

    logger.info(
        "Semantic classification strategy requested. Using structured classification as fallback."
    )

    # Delegate to standard classification
    return await _perform_standard_classification(
        text=text,
        categories=categories,
        is_hierarchical=is_hierarchical,
        multi_label=multi_label,
        confidence_threshold=confidence_threshold,
        max_results=max_results,
        explanation_detail=explanation_detail,
        examples=None,
        custom_prompt_template=None,
        taxonomy_description="Please classify using semantic similarity between the input text and categories.",
        output_format="json",
        strategy="structured",
        provider=provider,
        model=model,
        additional_params=additional_params,
    )


# --- Response Parsing Functions ---


def _parse_json_response(response_text: str, confidence_threshold: float) -> List[Dict[str, Any]]:
    """Parses a JSON-formatted classification response with robust error handling."""
    # Try to find JSON in the response
    json_pattern = r"(\{.*?\})"

    # Strategy 1: Try to find the most complete JSON object with explicit classifications array
    matches = re.findall(r'(\{"classifications":\s?\[.*?\]\})', response_text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                data = json.loads(match)
                if "classifications" in data and isinstance(data["classifications"], list):
                    return data["classifications"]
            except json.JSONDecodeError:
                continue

    # Strategy 2: Look for any JSON object and check if it contains classifications
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                data = json.loads(match)
                if "classifications" in data and isinstance(data["classifications"], list):
                    return data["classifications"]
            except json.JSONDecodeError:
                continue

    # Strategy 3: Try to find a JSON array directly
    array_matches = re.findall(r"(\[.*?\])", response_text, re.DOTALL)
    if array_matches:
        for match in array_matches:
            try:
                array_data = json.loads(match)
                if isinstance(array_data, list) and all(
                    isinstance(item, dict) for item in array_data
                ):
                    # Check if these look like classification objects
                    if all("category" in item for item in array_data):
                        return array_data
            except json.JSONDecodeError:
                continue

    # Strategy 4: Fall back to regex-based extraction for common formats
    classifications = []

    # Look for category/confidence patterns
    category_patterns = [
        r'"category":\s*"([^"]+)".*?"confidence":\s*([\d.]+)',
        r'category:\s*"([^"]+)".*?confidence:\s*([\d.]+)',
        r'Category:\s*"?([^",\n]+)"?.*?Confidence:\s*([\d.]+)',
    ]

    for pattern in category_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
        for category, confidence_str in matches:
            try:
                confidence = float(confidence_str)
                if confidence >= confidence_threshold:
                    # Look for explanation
                    explanation = ""
                    expl_match = re.search(
                        r'"?explanation"?:\s*"([^"]+)"',
                        response_text[response_text.find(category) :],
                        re.IGNORECASE,
                    )
                    if expl_match:
                        explanation = expl_match.group(1)

                    classifications.append(
                        {
                            "category": category.strip(),
                            "confidence": confidence,
                            "explanation": explanation,
                        }
                    )
            except ValueError:
                continue

    if classifications:
        return classifications

    # If all strategies fail, raise error
    raise ToolError(
        "Failed to parse classification result. Could not find valid JSON or extract classifications.",
        error_code="PARSING_ERROR",
        details={"response_text": response_text},
    )


def _parse_markdown_response(
    response_text: str, confidence_threshold: float
) -> List[Dict[str, Any]]:
    """Parses a Markdown-formatted classification response."""
    classifications = []

    # Look for markdown category patterns
    category_pattern = r"\*\*Category\*\*:\s*([^\n]+)"
    confidence_pattern = r"\*\*Confidence\*\*:\s*([\d.]+)"
    explanation_pattern = r"\*\*Explanation\*\*:\s*([^\n]+(?:\n[^\*]+)*)"

    # Find all category blocks
    category_matches = re.finditer(category_pattern, response_text, re.IGNORECASE)

    for category_match in category_matches:
        category = category_match.group(1).strip()
        section_start = category_match.start()
        section_end = response_text.find("-", section_start + 1)
        if section_end == -1:  # Last section
            section_end = len(response_text)
        section = response_text[section_start:section_end]

        # Find confidence
        confidence_match = re.search(confidence_pattern, section, re.IGNORECASE)
        if not confidence_match:
            continue

        try:
            confidence = float(confidence_match.group(1))
            if confidence < confidence_threshold:
                continue

            # Find explanation
            explanation = ""
            explanation_match = re.search(explanation_pattern, section, re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()

            classifications.append(
                {"category": category, "confidence": confidence, "explanation": explanation}
            )
        except ValueError:
            continue

    # If no matches, try simpler pattern
    if not classifications:
        simpler_pattern = r"- \*\*(.*?)\*\*.*?(\d+\.\d+)"
        matches = re.findall(simpler_pattern, response_text)
        for match in matches:
            try:
                category = match[0].strip()
                confidence = float(match[1])
                if confidence >= confidence_threshold:
                    classifications.append(
                        {"category": category, "confidence": confidence, "explanation": ""}
                    )
            except (ValueError, IndexError):
                continue

    if classifications:
        return classifications

    # If all strategies fail, raise error
    raise ToolError(
        "Failed to parse markdown classification result.",
        error_code="PARSING_ERROR",
        details={"response_text": response_text},
    )


def _parse_text_response(response_text: str, confidence_threshold: float) -> List[Dict[str, Any]]:
    """Parses a plain text formatted classification response."""
    classifications = []

    # Define patterns for different text formats
    patterns = [
        # Standard format
        r"CATEGORY:\s*([^\n]+)[\s\n]+CONFIDENCE:\s*([\d.]+)(?:[\s\n]+EXPLANATION:\s*([^\n]+))?",
        # Alternative formats
        r"Category:\s*([^\n]+)[\s\n]+Confidence:\s*([\d.]+)(?:[\s\n]+Explanation:\s*([^\n]+))?",
        r"([^:]+):\s*(\d+\.\d+)(?:\s+(.+?))?(?:\n|$)",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        for match in matches:
            try:
                category = match.group(1).strip()
                confidence = float(match.group(2))

                if confidence < confidence_threshold:
                    continue

                explanation = ""
                if len(match.groups()) >= 3 and match.group(3):
                    explanation = match.group(3).strip()

                classifications.append(
                    {"category": category, "confidence": confidence, "explanation": explanation}
                )
            except (ValueError, IndexError):
                continue

    if classifications:
        return classifications

    # Fall back to less structured pattern matching
    lines = response_text.split("\n")
    current_category = None
    current_confidence = None
    current_explanation = ""

    for line in lines:
        if ":" not in line:
            if current_category and current_explanation:
                current_explanation += " " + line.strip()
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key in ["category", "class", "label"]:
            # Start a new category
            if current_category and current_confidence is not None:
                if current_confidence >= confidence_threshold:
                    classifications.append(
                        {
                            "category": current_category,
                            "confidence": current_confidence,
                            "explanation": current_explanation,
                        }
                    )

            current_category = value
            current_confidence = None
            current_explanation = ""

        elif key in ["confidence", "score", "probability"]:
            try:
                current_confidence = float(value.rstrip("%"))
                # Handle percentage values
                if current_confidence > 1 and current_confidence <= 100:
                    current_confidence /= 100
            except ValueError:
                current_confidence = None

        elif key in ["explanation", "reason", "justification"]:
            current_explanation = value

    # Don't forget the last category
    if current_category and current_confidence is not None:
        if current_confidence >= confidence_threshold:
            classifications.append(
                {
                    "category": current_category,
                    "confidence": current_confidence,
                    "explanation": current_explanation,
                }
            )

    if classifications:
        return classifications

    # If all strategies fail, raise error
    raise ToolError(
        "Failed to parse text classification result.",
        error_code="PARSING_ERROR",
        details={"response_text": response_text},
    )


def _validate_classifications(
    classifications: List[Dict[str, Any]], valid_categories: List[str]
) -> None:
    """Validates classification results against provided categories."""
    valid_categories_lower = [c.lower() for c in valid_categories]

    for i, cls in enumerate(classifications):
        category = cls.get("category", "")
        # Make case-insensitive comparison
        if category.lower() not in valid_categories_lower:
            # Try to fix common issues
            # 1. Check if category has extra quotes
            stripped_category = category.strip("\"'")
            if stripped_category.lower() in valid_categories_lower:
                cls["category"] = stripped_category
                continue

            # 2. Find closest match
            closest_match = None
            closest_distance = float("inf")

            for valid_cat in valid_categories:
                # Simple Levenshtein distance approximation for minor typos
                distance = sum(
                    a != b for a, b in zip(category.lower(), valid_cat.lower(), strict=False)
                ) + abs(len(category) - len(valid_cat))

                if (
                    distance < closest_distance and distance <= len(valid_cat) * 0.3
                ):  # Allow 30% error
                    closest_match = valid_cat
                    closest_distance = distance

            if closest_match:
                # Replace with closest match
                cls["category"] = closest_match
                # Note the correction in explanation
                if "explanation" in cls:
                    cls["explanation"] += (
                        f" (Note: Category corrected from '{category}' to '{closest_match}')"
                    )
                else:
                    cls["explanation"] = (
                        f"Category corrected from '{category}' to '{closest_match}'"
                    )
            else:
                # Invalid category with no close match - remove from results
                classifications[i] = None

    # Remove None entries (invalid categories that couldn't be fixed)
    while None in classifications:
        classifications.remove(None)
