"""Business-focused sentiment analysis tools for Ultimate MCP Server."""

import json
import time
from typing import Any, Dict, List, Optional

from ultimate_mcp_server.constants import Provider, TaskType
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.tools.completion import generate_completion
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.business_sentiment")


@with_tool_metrics
@with_error_handling
async def analyze_business_sentiment(
    text: str,
    industry: Optional[str] = None,
    analysis_mode: str = "standard",
    entity_extraction: bool = False,
    aspect_based: bool = False,
    competitive_analysis: bool = False,
    intent_detection: bool = False,
    risk_assessment: bool = False,
    language: str = "english",
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    threshold_config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Performs comprehensive business-oriented sentiment analysis for commercial applications.

    This enterprise-grade tool analyzes customer feedback, reviews, support tickets, survey responses,
    social media mentions and other business text data to extract actionable insights. It provides
    customizable analysis for specific industries and use cases with options for deep-dive analysis.

    Args:
        text: The business text to analyze (feedback, review, survey response, etc.).
        industry: Optional industry context to tailor analysis (e.g., "retail", "financial_services",
                 "healthcare", "hospitality", "technology", "telecommunications", "manufacturing").
                 Improves accuracy by applying industry-specific terminology and benchmarks.
        analysis_mode: Type of analysis to perform:
                     - "standard": Basic business sentiment with key indicators
                     - "comprehensive": Detailed analysis with all available metrics
                     - "customer_experience": Focus on satisfaction, loyalty, and effort
                     - "product_feedback": Focus on feature sentiment and product improvement
                     - "brand_perception": Focus on brand attributes and competitive positioning
                     - "support_ticket": Optimized for support ticket prioritization and resolution
                     - "sales_opportunity": Focus on purchase intent and sales readiness
        entity_extraction: Whether to identify and extract mentioned products, services, features,
                          and business entities. Useful for pinpointing what customers are discussing.
        aspect_based: Whether to break down sentiment by specific aspects/features mentioned.
                     Helps identify which specific product/service elements drive sentiment.
        competitive_analysis: Whether to identify competitor mentions and comparative sentiment.
                             Useful for competitive intelligence and benchmarking.
        intent_detection: Whether to detect customer intents (e.g., purchase interest, cancellation
                         risk, support request, recommendation intent, complaint, praise).
        risk_assessment: Whether to evaluate potential business risks (e.g., churn risk, PR risk,
                        legal/compliance issues, potential escalation) based on the text.
        language: Language of the input text (supports multiple languages for global businesses).
        provider: The name of the LLM provider (e.g., "openai", "anthropic", "gemini").
        model: The specific model ID. If None, uses the provider's default model.
        threshold_config: Optional dictionary of threshold values for various metrics to customize
                         sensitivity levels (e.g., {"churn_risk": 0.7, "urgency": 0.8}).

    Returns:
        A dictionary containing comprehensive business sentiment analysis:
        {
            "core_metrics": {
                "primary_sentiment": "positive",  # Overall business sentiment
                "sentiment_score": 0.75,          # Normalized score (-1.0 to 1.0)
                "confidence": 0.92,               # Confidence in the assessment
                "satisfaction_score": 4.2,        # Estimated satisfaction (1-5 scale)
                "nps_category": "promoter",       # Predicted NPS category: detractor/passive/promoter
                "urgency": "low",                 # Action urgency assessment: low/medium/high/critical
                "actionability": 0.35            # How actionable the feedback is (0.0-1.0)
            },
            "business_dimensions": {              # Business-specific metrics
                "customer_satisfaction": 0.82,    # Satisfaction indicator (0.0-1.0)
                "product_sentiment": 0.75,        # Product sentiment (0.0-1.0)
                "value_perception": 0.68,         # Price-to-value perception (0.0-1.0)
                "ease_of_use": 0.90,              # Usability perception when relevant (0.0-1.0)
                "customer_effort_score": 2.1,     # Estimated CES (1-7 scale, lower is better)
                "loyalty_indicators": 0.85,       # Loyalty/retention indicators (0.0-1.0)
                "recommendation_likelihood": 0.87 # Likelihood to recommend (0.0-1.0)
            },
            "intent_analysis": {                  # Only if intent_detection=True
                "purchase_intent": 0.15,          # Purchase interest level (0.0-1.0)
                "churn_risk": 0.08,               # Risk of customer churn (0.0-1.0)
                "support_needed": 0.75,           # Likelihood customer needs support (0.0-1.0)
                "feedback_type": "suggestion",    # Type: complaint/praise/question/suggestion
                "information_request": false      # Whether customer is requesting information
            },
            "aspect_sentiment": {                 # Only if aspect_based=True
                "product_quality": 0.85,
                "customer_service": 0.92,
                "shipping_speed": 0.45,
                "return_process": 0.30,
                "website_usability": 0.78
            },
            "entity_extraction": {                # Only if entity_extraction=True
                "products": ["Product X Pro", "Legacy Model"],
                "features": ["battery life", "touchscreen responsiveness"],
                "services": ["customer support", "technical assistance"],
                "mentioned_departments": ["billing", "technical support"]
            },
            "competitive_insights": {             # Only if competitive_analysis=True
                "competitor_mentions": ["Competitor A", "Competitor B"],
                "comparative_sentiment": {
                    "Competitor A": -0.2,         # Negative comparison to competitor
                    "Competitor B": 0.3           # Positive comparison to competitor
                },
                "perceived_advantages": ["price", "features"],
                "perceived_disadvantages": ["support response time"]
            },
            "risk_assessment": {                  # Only if risk_assessment=True
                "churn_probability": 0.32,
                "response_urgency": "medium",
                "pr_risk": "low",
                "legal_compliance_flags": ["data privacy concern"],
                "escalation_probability": 0.15
            },
            "message_characteristics": {
                "key_topics": ["product quality", "customer service", "pricing"],
                "key_phrases": ["extremely satisfied", "quick resolution"],
                "tone_indicators": ["appreciative", "constructive"],
                "clarity": 0.9,                  # How clear/specific the feedback is (0.0-1.0)
                "subjectivity": 0.4,             # Subjective vs. objective content (0.0-1.0)
                "emotional_intensity": 0.65      # Intensity of emotion expressed (0.0-1.0)
            },
            "industry_specific_insights": {},    # Varies based on 'industry' parameter
            "recommended_actions": [             # Business action recommendations
                "Follow up regarding mentioned technical issue",
                "Highlight positive experience in success stories"
            ],
            "meta": {                           # Metadata about the analysis
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "analysis_mode": "comprehensive",
                "language_detected": "english",
                "tokens": { "input": 350, "output": 820, "total": 1170 },
                "cost": 0.000843,
                "processing_time": 1.25,
                "version": "2.4.0"
            },
            "success": true
        }

    Raises:
        ToolInputError: If parameters are invalid or incompatible.
        ProviderError: If the provider is unavailable or the LLM request fails.
        ToolError: For other errors during processing.
    """
    start_time = time.time()

    # Parameter validation
    if not text or not isinstance(text, str):
        raise ToolInputError(
            "Input text must be a non-empty string.", param_name="text", provided_value=text
        )

    valid_analysis_modes = [
        "standard",
        "comprehensive",
        "customer_experience",
        "product_feedback",
        "brand_perception",
        "support_ticket",
        "sales_opportunity",
    ]
    if analysis_mode not in valid_analysis_modes:
        raise ToolInputError(
            f"Invalid analysis_mode. Must be one of: {', '.join(valid_analysis_modes)}",
            param_name="analysis_mode",
            provided_value=analysis_mode,
        )

    # Construct the analysis prompt based on parameters
    system_prompt = _build_sentiment_system_prompt(
        industry=industry,
        analysis_mode=analysis_mode,
        entity_extraction=entity_extraction,
        aspect_based=aspect_based,
        competitive_analysis=competitive_analysis,
        intent_detection=intent_detection,
        risk_assessment=risk_assessment,
        language=language,
        threshold_config=threshold_config,
    )

    user_prompt = f"""
    Analyze the following business text according to the specified parameters:
    
    Text to analyze:
    ```
    {text}
    ```
    
    Provide a detailed JSON response according to the format specified in the system instructions.
    """

    # Combined prompt for all providers
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        # Consistently use generate_completion for all providers
        completion_result = await generate_completion(
            prompt=combined_prompt,
            provider=provider,
            model=model,
            temperature=0.2,
            max_tokens=2000,
            additional_params={"response_format": {"type": "json_object"}}
            if provider.lower() == "openai"
            else None,
        )

        # Extract response text from the completion result
        response_text = completion_result["text"].strip()

        # Extract JSON response
        try:
            # Try to extract JSON if wrapped in code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                if json_end > json_start:
                    response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                if json_end > json_start:
                    response_text = response_text[json_start:json_end].strip()

            # Parse and validate JSON
            analysis_data = json.loads(response_text)

            # Validate minimum required fields
            if "core_metrics" not in analysis_data:
                logger.warning("Missing 'core_metrics' in response, adding empty object")
                analysis_data["core_metrics"] = {}

            # Ensure core metrics contains primary sentiment
            core_metrics = analysis_data["core_metrics"]
            if "primary_sentiment" not in core_metrics:
                sentiment_score = core_metrics.get("sentiment_score", 0.0)
                if sentiment_score > 0.2:
                    primary_sentiment = "positive"
                elif sentiment_score < -0.2:
                    primary_sentiment = "negative"
                else:
                    primary_sentiment = "neutral"
                core_metrics["primary_sentiment"] = primary_sentiment
                logger.debug(f"Added missing primary_sentiment: {primary_sentiment}")

            # Populate metadata
            processing_time = time.time() - start_time

            # Extract provider and model info from completion result
            result_provider = completion_result.get("provider", provider)
            result_model = completion_result.get("model", model)
            input_tokens = completion_result.get("tokens", {}).get("input", 0)
            output_tokens = completion_result.get("tokens", {}).get("output", 0)
            total_tokens = completion_result.get("tokens", {}).get("total", 0)
            cost = completion_result.get("cost", 0.0)

            meta = {
                "provider": result_provider,
                "model": result_model,
                "analysis_mode": analysis_mode,
                "language_detected": language,  # Actual detection would need more logic
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens,
                },
                "cost": cost,
                "processing_time": processing_time,
                "version": "2.4.0",  # Tool version
            }

            # Include metadata in the final response
            analysis_data["meta"] = meta
            analysis_data["success"] = True

            # Log successful completion
            logger.success(
                f"Business sentiment analysis completed successfully with {result_provider}/{result_model}",
                emoji_key=TaskType.CLASSIFICATION.value,
                analysis_mode=analysis_mode,
                sentiment=core_metrics.get("primary_sentiment", "unknown"),
                tokens={"input": input_tokens, "output": output_tokens},
                cost=cost,
                time=processing_time,
            )

            return analysis_data

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response: {e}",
                emoji_key="error",
                raw_response=response_text[:500],  # Log partial response for debugging
            )
            raise ToolError(
                f"Failed to parse business sentiment analysis response: {e}",
                error_code="invalid_response_format",
                details={"raw_response": response_text[:500]},
            ) from e

    except Exception as e:
        raise ProviderError(
            f"Business sentiment analysis failed: {str(e)}", provider=provider, model=model, cause=e
        ) from e


def _build_sentiment_system_prompt(
    industry: Optional[str],
    analysis_mode: str,
    entity_extraction: bool,
    aspect_based: bool,
    competitive_analysis: bool,
    intent_detection: bool,
    risk_assessment: bool,
    language: str,
    threshold_config: Optional[Dict[str, float]],
) -> str:
    """Builds a comprehensive system prompt for business sentiment analysis based on parameters."""

    # Base prompt with core instructions
    base_prompt = """
    You are an enterprise-grade business sentiment analysis system designed to extract actionable insights from customer and stakeholder feedback. Your analysis should be precise, nuanced, and tailored to business decision-making.
    
    Provide analysis in a structured JSON format with the following core sections:
    1. core_metrics: Essential sentiment indicators
    2. business_dimensions: Business-specific metrics like satisfaction and loyalty
    3. message_characteristics: Content properties like topics and tone
    
    All numerical scores should be consistent (higher is better unless otherwise specified) and normalized within their specified ranges.
    """

    # Industry-specific tailoring
    industry_prompt = ""
    if industry:
        industry_mappings = {
            "retail": "Retail and e-commerce context: Focus on product quality, shopping experience, delivery, returns, and customer service. Include retail-specific metrics like purchase satisfaction and repeat purchase intent.",
            "financial_services": "Financial services context: Focus on trust, security, transparency, and service quality. Include financial-specific metrics like perceived financial benefit, trust indicator, and financial confidence impact.",
            "healthcare": "Healthcare context: Focus on care quality, staff interactions, facility experience, and outcomes. Include healthcare-specific metrics like perceived care quality, staff empathy, and outcome satisfaction.",
            "hospitality": "Hospitality context: Focus on accommodations, amenities, staff service, and overall experience. Include hospitality-specific metrics like comfort rating, staff attentiveness, and value perception.",
            "technology": "Technology/SaaS context: Focus on software/product functionality, reliability, ease of use, and technical support. Include tech-specific metrics like feature satisfaction, reliability perception, and technical resolution satisfaction.",
            "telecommunications": "Telecommunications context: Focus on service reliability, coverage, customer support, and value. Include telecom-specific metrics like service reliability rating, coverage satisfaction, and value perception.",
            "manufacturing": "Manufacturing context: Focus on product quality, durability, specifications adherence, and support. Include manufacturing-specific metrics like quality rating, durability perception, and technical specification satisfaction.",
        }

        if industry.lower() in industry_mappings:
            industry_prompt = f"\nINDUSTRY CONTEXT: {industry_mappings[industry.lower()]}\n"
            industry_prompt += "\nInclude an 'industry_specific_insights' section with metrics and insights specific to this industry."
        else:
            industry_prompt = f"\nINDUSTRY CONTEXT: {industry} - Apply industry-specific terminology and standards.\n"

    # Analysis mode specification
    mode_prompt = "\nANALYSIS MODE: "
    if analysis_mode == "standard":
        mode_prompt += "Standard business sentiment with core metrics and key indicators."
    elif analysis_mode == "comprehensive":
        mode_prompt += "Comprehensive analysis with all available metrics and maximum detail."
    elif analysis_mode == "customer_experience":
        mode_prompt += "Customer experience focus: Emphasize satisfaction, loyalty, and effort metrics. Pay special attention to service interactions, pain points, and moments of delight."
    elif analysis_mode == "product_feedback":
        mode_prompt += "Product feedback focus: Emphasize feature sentiment, product quality, and improvement suggestions. Identify specific product components mentioned and their sentiment."
    elif analysis_mode == "brand_perception":
        mode_prompt += "Brand perception focus: Emphasize brand attributes, positioning, and emotional connections. Analyze brand promise fulfillment and competitive positioning."
    elif analysis_mode == "support_ticket":
        mode_prompt += "Support ticket focus: Emphasize issue categorization, severity, urgency, and resolution path. Detect technical terms and problem indicators."
    elif analysis_mode == "sales_opportunity":
        mode_prompt += "Sales opportunity focus: Emphasize purchase intent, objections, and decision factors. Analyze buying signals and sales readiness indicators."

    # Optional analysis components
    optional_components = []

    if entity_extraction:
        optional_components.append("""
        ENTITY EXTRACTION: Extract and categorize business entities mentioned in the text.
        - Include an 'entity_extraction' section with arrays of identified products, features, services, departments, locations, etc.
        - Normalize entity names when variations of the same entity are mentioned.
        - Exclude generic mentions and focus on specific named entities.
        """)

    if aspect_based:
        optional_components.append("""
        ASPECT-BASED SENTIMENT: Break down sentiment by specific aspects or features mentioned.
        - Include an 'aspect_sentiment' section with sentiment scores for each identified aspect.
        - Aspects should be specific (e.g., 'website_usability', 'checkout_process', 'product_quality').
        - Only include aspects explicitly mentioned or strongly implied in the text.
        - Score each aspect from -1.0 (extremely negative) to 1.0 (extremely positive).
        """)

    if competitive_analysis:
        optional_components.append("""
        COMPETITIVE ANALYSIS: Identify and analyze competitor mentions and comparisons.
        - Include a 'competitive_insights' section with competitor names and comparative sentiment.
        - Capture explicit and implicit comparisons to competitors.
        - Identify perceived advantages and disadvantages relative to competitors.
        - Score comparative sentiment from -1.0 (negative comparison) to 1.0 (positive comparison).
        """)

    if intent_detection:
        optional_components.append("""
        INTENT DETECTION: Identify customer intentions and likely next actions.
        - Include an 'intent_analysis' section with probabilities for purchase intent, churn risk, etc.
        - Classify the feedback type (complaint, praise, question, suggestion).
        - Detect specific intents like information requests, cancellation warnings, escalation threats.
        - Score intent probabilities from 0.0 (no indication) to 1.0 (strong indication).
        """)

    if risk_assessment:
        optional_components.append("""
        RISK ASSESSMENT: Evaluate potential business risks in the feedback.
        - Include a 'risk_assessment' section with probabilities and categories of identified risks.
        - Assess churn probability, PR/reputation risk, legal/compliance concerns, etc.
        - Provide an escalation probability and urgency level.
        - Flag sensitive content that may require special attention.
        """)

    # Language specification
    language_prompt = f"\nLANGUAGE: Analyze text in {language}. Ensure all scores and categorizations are correctly interpreted within cultural and linguistic context."

    # Threshold configurations if provided
    threshold_prompt = ""
    if threshold_config and isinstance(threshold_config, dict):
        threshold_prompt = "\nTHRESHOLD CONFIGURATION:"
        for metric, value in threshold_config.items():
            threshold_prompt += f"\n- {metric}: {value}"

    # Combine all prompt components
    full_prompt = base_prompt + industry_prompt + mode_prompt + language_prompt + threshold_prompt

    if optional_components:
        full_prompt += "\n\nADDITIONAL ANALYSIS COMPONENTS:"
        full_prompt += "\n".join(optional_components)

    # Output format specification
    output_format = """
    RESPONSE FORMAT: Respond only with a valid JSON object containing all applicable sections based on the analysis parameters.
    
    Always include these core sections:
    - core_metrics: Overall sentiment, scores, and primary indicators
    - business_dimensions: Business-specific satisfaction and perception metrics
    - message_characteristics: Content properties, topics, and expression styles
    - recommended_actions: 1-3 specific business actions based on the analysis
    - meta: Will be populated with metadata about the analysis
    
    Add optional sections as specified by the analysis parameters.
    
    Ensure all numerical values are normalized to their specified ranges and all categorical values use consistent terminology.
    """

    full_prompt += output_format

    return full_prompt


@with_tool_metrics
@with_error_handling
async def analyze_business_text_batch(
    texts: List[str],
    analysis_config: Dict[str, Any],
    aggregate_results: bool = True,
    max_concurrency: int = 3,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Processes a batch of business texts for sentiment analysis with aggregated insights.

    Designed for analyzing large volumes of business feedback (reviews, surveys, tickets)
    efficiently with detailed individual analyses and optional aggregated metrics. Ideal for
    business intelligence, customer experience programs, and trend identification.

    Args:
        texts: List of text items to analyze (reviews, feedback, etc.).
        analysis_config: Configuration dictionary for analyze_business_sentiment.
                        Example: {"analysis_mode": "standard", "entity_extraction": True}
                        All parameters from analyze_business_sentiment except text, provider, model.
        aggregate_results: Whether to generate aggregated insights across all analyzed texts.
                         Includes trend detection, sentiment distribution, and pattern identification.
        max_concurrency: Maximum number of parallel analyses to run.
        provider: The name of the LLM provider to use.
        model: The specific model ID. If None, uses the provider's default.

    Returns:
        A dictionary containing individual and aggregated results:
        {
            "individual_results": [
                {
                    "text_id": 0,
                    "text_preview": "First 50 characters of text...",
                    "analysis": { /* Complete analysis result for this text */ }
                },
                // Additional individual results...
            ],
            "aggregate_insights": {  // Only if aggregate_results=True
                "sentiment_distribution": {
                    "positive": 0.65,  // 65% positive
                    "neutral": 0.20,   // 20% neutral
                    "negative": 0.15   // 15% negative
                },
                "average_metrics": {
                    "sentiment_score": 0.42,
                    "satisfaction_score": 3.8,
                    // Other averaged metrics...
                },
                "top_aspects": [
                    {"name": "customer_service", "avg_sentiment": 0.75, "mention_count": 42},
                    {"name": "product_quality", "avg_sentiment": 0.62, "mention_count": 38},
                    // Additional aspects...
                ],
                "key_topics": [
                    {"topic": "shipping delays", "mention_count": 35, "avg_sentiment": -0.3},
                    {"topic": "easy checkout", "mention_count": 28, "avg_sentiment": 0.8},
                    // Additional topics...
                ],
                "entity_mention_frequencies": {
                    "products": {"Product X": 45, "Product Y": 23},
                    "features": {"user interface": 38, "reliability": 27}
                },
                "emerging_patterns": [
                    "Increasing mentions of mobile app usability",
                    "Growing negative sentiment about recent policy change"
                ],
                "risk_indicators": [
                    {"issue": "shipping delays", "severity": "medium", "trend": "increasing"},
                    {"issue": "billing confusion", "severity": "low", "trend": "stable"}
                ]
            },
            "meta": {
                "batch_size": 250,
                "success_count": 248,
                "error_count": 2,
                "processing_time": 128.5,
                "total_cost": 4.87,
                "timestamp": "2025-04-21T14:30:00Z"
            },
            "success": true
        }

    Raises:
        ToolInputError: If input parameters are invalid.
        ProviderError: If the provider service fails.
        ToolError: For other processing errors.
    """
    start_time = time.time()
    total_cost = 0.0
    success_count = 0
    error_count = 0

    # Validate inputs
    if not texts or not isinstance(texts, list):
        raise ToolInputError(
            "The 'texts' parameter must be a non-empty list of strings.",
            param_name="texts",
            provided_value=texts,
        )

    if not analysis_config or not isinstance(analysis_config, dict):
        raise ToolInputError(
            "The 'analysis_config' parameter must be a dictionary of configuration options.",
            param_name="analysis_config",
            provided_value=analysis_config,
        )

    # Process texts with concurrency control
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrency)
    individual_results = []
    all_analyses = []

    async def process_text(idx: int, text: str):
        nonlocal total_cost, success_count, error_count

        async with semaphore:
            text_preview = text[:50] + ("..." if len(text) > 50 else "")
            logger.debug(f"Processing text {idx + 1}/{len(texts)}: {text_preview}")

            try:
                # Create a copy of analysis_config to avoid modifying the original
                config = analysis_config.copy()

                # Add provider and model to config
                config["provider"] = provider
                config["model"] = model

                # Process the individual text using our refactored analyze_business_sentiment
                result = await analyze_business_sentiment(text=text, **config)

                # Update metrics
                total_cost += result.get("meta", {}).get("cost", 0.0)
                success_count += 1

                # Record result
                individual_results.append(
                    {"text_id": idx, "text_preview": text_preview, "analysis": result}
                )

                # Store for aggregation
                all_analyses.append(result)

                return result

            except Exception as e:
                logger.error(f"Error analyzing text {idx}: {str(e)}", exc_info=True)
                error_count += 1

                # Record error
                individual_results.append(
                    {"text_id": idx, "text_preview": text_preview, "error": str(e)}
                )

                return None

    # Create and run tasks
    tasks = [process_text(i, text) for i, text in enumerate(texts)]
    await asyncio.gather(*tasks)

    # Sort results by text_id to maintain original order
    individual_results.sort(key=lambda x: x["text_id"])

    # Build response
    result = {
        "individual_results": individual_results,
        "meta": {
            "batch_size": len(texts),
            "success_count": success_count,
            "error_count": error_count,
            "processing_time": time.time() - start_time,
            "total_cost": total_cost,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "success": True,
    }

    # Calculate aggregate insights if requested and we have successful analyses
    if aggregate_results and all_analyses:
        try:
            aggregate_insights = _calculate_aggregate_insights(all_analyses)
            result["aggregate_insights"] = aggregate_insights
        except Exception as e:
            logger.error(f"Error calculating aggregate insights: {str(e)}", exc_info=True)
            result["aggregate_insights_error"] = str(e)

    return result


def _calculate_aggregate_insights(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates aggregate insights across multiple business sentiment analyses."""

    # Initialize aggregation containers
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    sentiment_scores = []
    satisfaction_scores = []
    loyalty_indicators = []
    aspect_sentiments = {}
    topics = {}
    mentioned_entities = {"products": {}, "features": {}, "services": {}}

    # Process each analysis
    for analysis in analyses:
        # Skip any analyses without core_metrics
        if "core_metrics" not in analysis:
            continue

        core = analysis.get("core_metrics", {})
        business = analysis.get("business_dimensions", {})

        # Sentiment distribution
        sentiment = core.get("primary_sentiment", "neutral").lower()
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

        # Collect numerical metrics
        if "sentiment_score" in core:
            sentiment_scores.append(core["sentiment_score"])

        if "satisfaction_score" in business:
            satisfaction_scores.append(business["satisfaction_score"])

        if "loyalty_indicators" in business:
            loyalty_indicators.append(business["loyalty_indicators"])

        # Aspect sentiments
        for aspect, score in analysis.get("aspect_sentiment", {}).items():
            if aspect not in aspect_sentiments:
                aspect_sentiments[aspect] = {"scores": [], "count": 0}

            aspect_sentiments[aspect]["scores"].append(score)
            aspect_sentiments[aspect]["count"] += 1

        # Topics
        for topic in analysis.get("message_characteristics", {}).get("key_topics", []):
            if topic not in topics:
                topics[topic] = 0
            topics[topic] += 1

        # Entity mentions
        for entity_type, entities in analysis.get("entity_extraction", {}).items():
            if entity_type in mentioned_entities and isinstance(entities, list):
                for entity in entities:
                    if entity not in mentioned_entities[entity_type]:
                        mentioned_entities[entity_type][entity] = 0
                    mentioned_entities[entity_type][entity] += 1

    # Calculate distributions as percentages
    total_sentiments = sum(sentiment_counts.values())
    sentiment_distribution = {
        k: round(v / total_sentiments, 2) if total_sentiments else 0
        for k, v in sentiment_counts.items()
    }

    # Calculate average metrics
    average_metrics = {}
    if sentiment_scores:
        average_metrics["sentiment_score"] = sum(sentiment_scores) / len(sentiment_scores)

    if satisfaction_scores:
        average_metrics["satisfaction_score"] = sum(satisfaction_scores) / len(satisfaction_scores)

    if loyalty_indicators:
        average_metrics["loyalty_indicators"] = sum(loyalty_indicators) / len(loyalty_indicators)

    # Process aspect sentiments
    top_aspects = []
    for aspect, data in aspect_sentiments.items():
        avg_sentiment = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        top_aspects.append(
            {
                "name": aspect,
                "avg_sentiment": round(avg_sentiment, 2),
                "mention_count": data["count"],
            }
        )

    # Sort aspects by mention count
    top_aspects.sort(key=lambda x: x["mention_count"], reverse=True)

    # Process topics
    key_topics = [{"topic": k, "mention_count": v} for k, v in topics.items()]
    key_topics.sort(key=lambda x: x["mention_count"], reverse=True)

    # Build aggregated insights
    aggregate_insights = {
        "sentiment_distribution": sentiment_distribution,
        "average_metrics": average_metrics,
        "top_aspects": top_aspects[:10],  # Limit to top 10
        "key_topics": key_topics[:10],  # Limit to top 10
        "entity_mention_frequencies": mentioned_entities,
    }

    return aggregate_insights
