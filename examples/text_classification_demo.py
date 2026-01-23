#!/usr/bin/env python
"""
Text classification demonstration for Ultimate MCP Server.
This example showcases the comprehensive capabilities of the text_classification tool,
demonstrating various classification strategies, multi-label vs. single-label,
hierarchical categories, and more.
"""

import asyncio
import json
import os
import sys
import time
from collections import namedtuple  # Import namedtuple
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.tools.text_classification import (
    ClassificationStrategy,
    text_classification,
)
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker  # Import CostTracker
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger
logger = get_logger("example.text_classification")

# Create a separate debug console for detailed logging
debug_console = Console(stderr=True, highlight=False)

# Get configuration from centralized config system
gateway_config = get_config()
EXPORT_RESULTS = gateway_config.server.debug  # Using server.debug as a proxy for export results
RESULTS_DIR = os.path.join(gateway_config.storage_directory, "classification_results")
DEMO_TIMEOUT = 120  # Hard-coded default timeout for demo

# Cache for demonstration purposes
DEMO_RESULTS_CACHE = {}

# File paths for sample data
SAMPLE_DIR = Path(__file__).parent / "sample" / "text_classification_samples"
NEWS_SAMPLES_PATH = SAMPLE_DIR / "news_samples.txt"
PRODUCT_REVIEWS_PATH = SAMPLE_DIR / "product_reviews.txt"
SUPPORT_TICKETS_PATH = SAMPLE_DIR / "support_tickets.txt"
EMAIL_SAMPLES_PATH = SAMPLE_DIR / "email_classification.txt"

# Create a simple structure for cost tracking from dict
TrackableResult = namedtuple(
    "TrackableResult",
    ["cost", "input_tokens", "output_tokens", "provider", "model", "processing_time"],
)


# Helper Functions
def extract_samples_from_file(file_path):
    """Extract labeled samples from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    samples = {}
    current_label = None
    current_content = []

    for line in content.split("\n"):
        if (
            line.strip().endswith("SAMPLE:")
            or line.strip().endswith("EMAIL:")
            or line.strip().endswith("REVIEW:")
            or line.strip().endswith("ISSUE:")
            or line.strip().endswith("REPORT:")
            or line.strip().endswith("REQUEST:")
            or line.strip().endswith("QUESTION:")
        ):
            # Save previous sample
            if current_label and current_content:
                samples[current_label] = "\n".join(current_content).strip()

            # Start new sample
            current_label = line.strip().rstrip(":")
            current_content = []
        elif line.strip() and current_label is not None:
            current_content.append(line)

    # Add the last sample
    if current_label and current_content:
        samples[current_label] = "\n".join(current_content).strip()

    return samples


def display_classification_result(result, title, text_sample=None, categories=None):
    """Display classification results in a rich formatted table."""
    # Create main table for classification results
    results_table = Table(title=title, box=box.ROUNDED, show_header=True, expand=True)
    results_table.add_column("Category", style="cyan", no_wrap=True)
    results_table.add_column("Confidence", style="green", justify="right")
    results_table.add_column("Explanation", style="white")

    for classification in result.get("classifications", []):
        confidence = classification.get("confidence", 0.0)
        confidence_str = f"{confidence:.4f}"
        confidence_color = "green" if confidence > 0.8 else "yellow" if confidence > 0.6 else "red"

        results_table.add_row(
            classification.get("category", "Unknown"),
            f"[{confidence_color}]{confidence_str}[/{confidence_color}]",
            escape(classification.get("explanation", ""))[:100]
            + ("..." if len(classification.get("explanation", "")) > 100 else ""),
        )

    # Create metadata table
    meta_table = Table(show_header=False, box=None, expand=False)
    meta_table.add_column("Metric", style="cyan")
    meta_table.add_column("Value", style="white")
    meta_table.add_row("Provider", result.get("provider", "unknown"))
    meta_table.add_row("Model", result.get("model", "unknown"))
    meta_table.add_row("Processing Time", f"{result.get('processing_time', 0.0):.3f}s")
    meta_table.add_row("Input Tokens", str(result.get("tokens", {}).get("input", 0)))
    meta_table.add_row("Output Tokens", str(result.get("tokens", {}).get("output", 0)))
    meta_table.add_row("Cost", f"${result.get('cost', 0.0):.6f}")

    if "dominant_category" in result:
        meta_table.add_row("Dominant Category", result["dominant_category"])

    if "ensemble_models" in result:
        meta_table.add_row("Ensemble Models", ", ".join(result["ensemble_models"]))

    # Display text sample if provided
    if text_sample:
        text_panel = Panel(
            escape(text_sample[:300] + ("..." if len(text_sample) > 300 else "")),
            title="Sample Text",
            border_style="blue",
            expand=False,
        )
        console.print(text_panel)

    # Display categories if provided
    if categories:
        cat_display = ""
        if isinstance(categories, dict):
            for parent, subcats in categories.items():
                cat_display += f"- {parent}\n"
                for sub in subcats:
                    cat_display += f"  - {parent}/{sub}\n"
        else:
            for cat in categories:
                cat_display += f"- {cat}\n"

        cat_panel = Panel(
            cat_display.strip(),
            title="Classification Categories",
            border_style="green",
            expand=False,
        )
        console.print(cat_panel)

    # Display results and metadata
    console.print(results_table)
    console.print(meta_table)


async def demo_basic_classification(tracker: CostTracker):  # Add tracker
    """Demonstrate basic single-label classification with zero-shot."""
    console.print(Rule("[bold blue]Basic Text Classification Demo[/bold blue]"))
    logger.info("Starting basic classification demo", emoji_key="start")

    # Load news samples
    news_samples = extract_samples_from_file(NEWS_SAMPLES_PATH)

    # Simple categories for news classification
    categories = [
        "Technology",
        "Sports",
        "Politics",
        "Health",
        "Entertainment",
        "Science",
        "Business",
        "Education",
    ]

    # Select a sample
    sample_key = "TECH NEWS SAMPLE"
    sample_text = news_samples[sample_key]

    logger.info(f"Classifying a {sample_key} with zero-shot strategy", emoji_key="processing")

    # Run classification
    start_time = time.time()
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Classifying text...", total=None)
        result = await text_classification(
            text=sample_text,
            categories=categories,
            provider=Provider.OPENAI.value,
            model="gpt-3.5-turbo",  # Using a simpler model for basic demo
            multi_label=False,
            confidence_threshold=0.5,
            strategy=ClassificationStrategy.ZERO_SHOT,
            explanation_detail="brief",
        )

    # Track cost if possible
    if all(k in result for k in ["cost", "provider", "model"]) and "tokens" in result:
        try:
            trackable = TrackableResult(
                cost=result.get("cost", 0.0),
                input_tokens=result.get("tokens", {}).get("input", 0),
                output_tokens=result.get("tokens", {}).get("output", 0),
                provider=result.get("provider", "unknown"),
                model=result.get("model", "unknown"),
                processing_time=result.get("processing_time", 0.0),
            )
            tracker.add_call(trackable)
        except Exception as track_err:
            logger.warning(
                f"Could not track cost for basic classification: {track_err}", exc_info=False
            )

    # Record actual time (may differ from model reported time)
    elapsed_time = time.time() - start_time
    result["actual_processing_time"] = elapsed_time

    # Cache result for comparison
    DEMO_RESULTS_CACHE["basic"] = result

    # Export result if enabled
    if EXPORT_RESULTS:
        export_result("basic_classification", result, sample_text, categories)

    # Display result
    logger.success(f"Basic classification completed in {elapsed_time:.3f}s", emoji_key="success")
    display_classification_result(
        result,
        "Basic Single-Label Classification (Zero-Shot)",
        text_sample=sample_text,
        categories=categories,
    )
    console.print()
    return True


async def demo_multi_label_classification(tracker: CostTracker):  # Add tracker
    """Demonstrate multi-label classification."""
    console.print(Rule("[bold blue]Multi-Label Classification Demo[/bold blue]"))
    logger.info("Starting multi-label classification demo", emoji_key="start")

    # Load support ticket samples
    ticket_samples = extract_samples_from_file(SUPPORT_TICKETS_PATH)

    # Select a complex sample that might have multiple labels
    sample_key = "BUG REPORT"
    sample_text = ticket_samples[sample_key]

    # Categories for support tickets
    categories = [
        "Bug Report",
        "Feature Request",
        "Account Issue",
        "Billing Question",
        "Technical Question",
        "Security Issue",
        "Performance Problem",
        "UI/UX Feedback",
    ]

    logger.info("Classifying support ticket with multi-label strategy", emoji_key="processing")

    # Run classification
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Classifying with multiple labels...", total=None)
        result = await text_classification(
            text=sample_text,
            categories=categories,
            provider=Provider.OPENAI.value,
            model="gpt-4-mini",  # Using a better model for nuanced classification
            multi_label=True,
            confidence_threshold=0.3,  # Lower threshold to catch secondary categories
            strategy=ClassificationStrategy.STRUCTURED,
            explanation_detail="brief",
            max_results=3,  # Get top 3 matching categories
        )

    # Track cost if possible
    if all(k in result for k in ["cost", "provider", "model"]) and "tokens" in result:
        try:
            trackable = TrackableResult(
                cost=result.get("cost", 0.0),
                input_tokens=result.get("tokens", {}).get("input", 0),
                output_tokens=result.get("tokens", {}).get("output", 0),
                provider=result.get("provider", "unknown"),
                model=result.get("model", "unknown"),
                processing_time=result.get("processing_time", 0.0),
            )
            tracker.add_call(trackable)
        except Exception as track_err:
            logger.warning(
                f"Could not track cost for multi-label classification: {track_err}", exc_info=False
            )

    # Cache result for comparison
    DEMO_RESULTS_CACHE["multi_label"] = result

    # Display result
    logger.success("Multi-label classification completed", emoji_key="success")
    display_classification_result(
        result,
        "Multi-Label Classification (Structured)",
        text_sample=sample_text,
        categories=categories,
    )
    console.print()
    return True


async def demo_hierarchical_classification(tracker: CostTracker):  # Add tracker
    """Demonstrate hierarchical category classification."""
    console.print(Rule("[bold blue]Hierarchical Classification Demo[/bold blue]"))
    logger.info("Starting hierarchical classification demo", emoji_key="start")

    # Load product review samples
    review_samples = extract_samples_from_file(PRODUCT_REVIEWS_PATH)

    # Select a sample
    sample_key = "POSITIVE REVIEW"
    sample_text = review_samples[sample_key]

    # Hierarchical categories for product reviews
    categories = {
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Product Type": ["Electronics", "Appliance", "Clothing", "Software"],
        "Aspect": ["Performance", "Quality", "Price", "Customer Service", "Design", "Usability"],
    }

    logger.info("Classifying product review with hierarchical categories", emoji_key="processing")

    # Run classification
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Classifying with hierarchical categories...", total=None)
        result = await text_classification(
            text=sample_text,
            categories=categories,
            provider=Provider.OPENAI.value,
            model="gpt-4-mini",
            multi_label=True,  # Allow selecting one from each hierarchy
            confidence_threshold=0.6,
            strategy=ClassificationStrategy.STRUCTURED,
            explanation_detail="brief",
            taxonomy_description=(
                "This taxonomy categorizes product reviews across multiple dimensions: "
                "the sentiment (overall positivity/negativity), the type of product being discussed, "
                "and the specific aspects of the product mentioned in the review."
            ),
        )

    # Track cost if possible
    if all(k in result for k in ["cost", "provider", "model"]) and "tokens" in result:
        try:
            trackable = TrackableResult(
                cost=result.get("cost", 0.0),
                input_tokens=result.get("tokens", {}).get("input", 0),
                output_tokens=result.get("tokens", {}).get("output", 0),
                provider=result.get("provider", "unknown"),
                model=result.get("model", "unknown"),
                processing_time=result.get("processing_time", 0.0),
            )
            tracker.add_call(trackable)
        except Exception as track_err:
            logger.warning(
                f"Could not track cost for hierarchical classification: {track_err}", exc_info=False
            )

    # Cache result for comparison
    DEMO_RESULTS_CACHE["hierarchical"] = result

    # Display result
    logger.success("Hierarchical classification completed", emoji_key="success")
    display_classification_result(
        result,
        "Hierarchical Multi-Label Classification",
        text_sample=sample_text,
        categories=categories,
    )
    console.print()
    return True


async def demo_few_shot_classification(tracker: CostTracker):  # Add tracker
    """Demonstrate few-shot learning classification."""
    console.print(Rule("[bold blue]Few-Shot Classification Demo[/bold blue]"))
    logger.info("Starting few-shot classification demo", emoji_key="start")

    # Load email samples
    email_samples = extract_samples_from_file(EMAIL_SAMPLES_PATH)

    # Select a sample to classify
    sample_key = "PHISHING EMAIL"
    sample_text = email_samples[sample_key]

    # Categories for email classification
    categories = [
        "Spam",
        "Phishing",
        "Promotional",
        "Informational",
        "Urgent",
        "Personal",
        "Transactional",
    ]

    # Create example data for few-shot learning
    examples = [
        {"text": email_samples["SPAM EMAIL"], "categories": ["Spam"]},
        {"text": email_samples["PROMOTIONAL EMAIL"], "categories": ["Promotional"]},
        {"text": email_samples["PERSONAL EMAIL"], "categories": ["Personal"]},
    ]

    logger.info("Classifying email with few-shot learning", emoji_key="processing")

    # Run classification
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Classifying with few-shot examples...", total=None)
        result = await text_classification(
            text=sample_text,
            categories=categories,
            provider=Provider.OPENAI.value,
            model="gpt-3.5-turbo",  # Few-shot works well with simpler models
            multi_label=False,
            confidence_threshold=0.5,
            strategy=ClassificationStrategy.FEW_SHOT,
            examples=examples,
            explanation_detail="detailed",  # More detailed explanation
        )

    # Track cost if possible
    if all(k in result for k in ["cost", "provider", "model"]) and "tokens" in result:
        try:
            trackable = TrackableResult(
                cost=result.get("cost", 0.0),
                input_tokens=result.get("tokens", {}).get("input", 0),
                output_tokens=result.get("tokens", {}).get("output", 0),
                provider=result.get("provider", "unknown"),
                model=result.get("model", "unknown"),
                processing_time=result.get("processing_time", 0.0),
            )
            tracker.add_call(trackable)
        except Exception as track_err:
            logger.warning(
                f"Could not track cost for few-shot classification: {track_err}", exc_info=False
            )

    # Cache result for comparison
    DEMO_RESULTS_CACHE["few_shot"] = result

    # Display examples provided
    example_table = Table(title="Few-Shot Examples Provided", box=box.SIMPLE)
    example_table.add_column("Example", style="cyan")
    example_table.add_column("Category", style="green")
    example_table.add_column("Text Sample", style="white", max_width=60)

    for i, example in enumerate(examples):
        example_table.add_row(
            f"Example {i + 1}",
            ", ".join(example["categories"]),
            escape(example["text"][:100] + "..."),
        )

    console.print(example_table)
    console.print()

    # Display result
    logger.success("Few-shot classification completed", emoji_key="success")
    display_classification_result(
        result, "Few-Shot Classification", text_sample=sample_text, categories=categories
    )
    console.print()
    return True


async def demo_ensemble_classification(tracker: CostTracker):  # Add tracker
    """Demonstrate ensemble classification using multiple providers."""
    console.print(Rule("[bold blue]Ensemble Classification Demo[/bold blue]"))
    logger.info("Starting ensemble classification demo", emoji_key="start")

    # Load support ticket samples again but use a different one
    ticket_samples = extract_samples_from_file(SUPPORT_TICKETS_PATH)

    # Select a complex sample
    sample_key = "FEATURE REQUEST"
    sample_text = ticket_samples[sample_key]

    # Categories for support tickets (same as before)
    categories = [
        "Bug Report",
        "Feature Request",
        "Account Issue",
        "Billing Question",
        "Technical Question",
        "Security Issue",
        "Performance Problem",
        "UI/UX Feedback",
    ]

    # Configure ensemble with multiple models
    ensemble_config = [
        {
            "provider": Provider.OPENAI.value,
            "model": "gpt-3.5-turbo",
            "weight": 0.3,
            "params": {"temperature": 0.1},
        },
        {
            "provider": Provider.OPENAI.value,
            "model": "gpt-4-mini",
            "weight": 0.7,
            "params": {"temperature": 0.1},
        },
        # In a real-world scenario, you might include models from different providers
    ]

    logger.info("Classifying support ticket with ensemble strategy", emoji_key="processing")

    # Run classification
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Classifying with multiple models...", total=None)
        result = await text_classification(
            text=sample_text,
            categories=categories,
            provider=Provider.OPENAI.value,  # Base provider (though ensemble will use multiple)
            multi_label=True,
            confidence_threshold=0.4,
            strategy=ClassificationStrategy.ENSEMBLE,
            explanation_detail="brief",
            ensemble_config=ensemble_config,
            allow_abstain=True,
            # abstention_threshold=0.4 # Optionally set abstention threshold
        )

    # Track cost (The tool result should contain aggregated cost/tokens)
    if all(k in result for k in ["cost", "provider", "model"]) and "tokens" in result:
        try:
            trackable = TrackableResult(
                cost=result.get("cost", 0.0),
                input_tokens=result.get("tokens", {}).get("input", 0),
                output_tokens=result.get("tokens", {}).get("output", 0),
                provider=result.get("provider", "ensemble"),  # Provider is 'ensemble'
                model=result.get("model", "ensemble"),  # Model is 'ensemble'
                processing_time=result.get("processing_time", 0.0),
            )
            tracker.add_call(trackable)
        except Exception as track_err:
            logger.warning(
                f"Could not track cost for ensemble classification: {track_err}", exc_info=False
            )

    # Cache result for comparison
    DEMO_RESULTS_CACHE["ensemble"] = result

    # Display ensemble config
    ensemble_table = Table(title="Ensemble Configuration", box=box.SIMPLE)
    ensemble_table.add_column("Provider", style="cyan")
    ensemble_table.add_column("Model", style="green")
    ensemble_table.add_column("Weight", style="yellow")

    for config in ensemble_config:
        ensemble_table.add_row(config["provider"], config["model"], f"{config['weight']:.2f}")

    console.print(ensemble_table)
    console.print()

    # Display result
    logger.success("Ensemble classification completed", emoji_key="success")
    display_classification_result(
        result, "Ensemble Classification", text_sample=sample_text, categories=categories
    )
    console.print()
    return True


async def demo_custom_prompt_template(tracker: CostTracker):  # Add tracker
    """Demonstrate classification with a custom prompt template."""
    console.print(Rule("[bold blue]Custom Prompt Template Demo[/bold blue]"))
    logger.info("Starting custom prompt template demo", emoji_key="start")

    # Load news samples again but use a different one
    news_samples = extract_samples_from_file(NEWS_SAMPLES_PATH)

    # Select a different sample
    sample_key = "SCIENCE NEWS SAMPLE"
    sample_text = news_samples[sample_key]

    # Simple categories for news classification
    categories = [
        "Technology",
        "Sports",
        "Politics",
        "Health",
        "Entertainment",
        "Science",
        "Business",
        "Education",
    ]

    # Create a custom prompt template
    custom_template = """
You are a highly specialized news classification assistant.

I need you to analyze the following text and determine which category it belongs to:
{categories}

When classifying, consider:
- The topic and subject matter
- The terminology and jargon used
- The intended audience
- The writing style and tone

CLASSIFICATION FORMAT:
{format_instruction}

TEXT TO CLASSIFY:
```
{text}
```

Please provide your analysis now.
"""

    logger.info("Classifying news with custom prompt template", emoji_key="processing")

    # Run classification
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Classifying with custom prompt...", total=None)
        result = await text_classification(
            text=sample_text,
            categories=categories,
            provider=Provider.OPENAI.value,
            model="gpt-4-mini",
            multi_label=False,
            confidence_threshold=0.5,
            strategy=ClassificationStrategy.STRUCTURED,
            explanation_detail="detailed",
            custom_prompt_template=custom_template,
        )

    # Track cost if possible
    if all(k in result for k in ["cost", "provider", "model"]) and "tokens" in result:
        try:
            trackable = TrackableResult(
                cost=result.get("cost", 0.0),
                input_tokens=result.get("tokens", {}).get("input", 0),
                output_tokens=result.get("tokens", {}).get("output", 0),
                provider=result.get("provider", "unknown"),
                model=result.get("model", "unknown"),
                processing_time=result.get("processing_time", 0.0),
            )
            tracker.add_call(trackable)
        except Exception as track_err:
            logger.warning(
                f"Could not track cost for custom prompt classification: {track_err}",
                exc_info=False,
            )

    # Cache result for comparison
    DEMO_RESULTS_CACHE["custom_prompt"] = result

    # Display custom prompt template
    prompt_panel = Panel(
        escape(custom_template),
        title="Custom Prompt Template",
        border_style="magenta",
        expand=False,
    )
    console.print(prompt_panel)
    console.print()

    # Display result
    logger.success("Custom prompt classification completed", emoji_key="success")
    display_classification_result(
        result, "Classification with Custom Prompt", text_sample=sample_text, categories=categories
    )
    console.print()
    return True


def export_result(name, result, text_sample, categories):
    """Export classification result to JSON file."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Create timestamp for filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{RESULTS_DIR}/{name}_{timestamp}.json"

    # Prepare data to export
    export_data = {
        "timestamp": time.time(),
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
        "sample_text": text_sample,
        "categories": categories,
    }

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    logger.info(f"Exported results to {filename}", emoji_key="save")


async def demo_comparison(tracker: CostTracker):
    """Compare different classification strategies."""
    console.print(Rule("[bold blue]Classification Strategies Comparison[/bold blue]"))
    logger.info("Comparing classification strategies", emoji_key="analytics")

    # Check if we have all cached results
    required_demos = [
        "basic",
        "multi_label",
        "hierarchical",
        "few_shot",
        "ensemble",
        "custom_prompt",
    ]
    missing = [demo for demo in required_demos if demo not in DEMO_RESULTS_CACHE]

    if missing:
        logger.warning(f"Missing results for comparison: {', '.join(missing)}", emoji_key="warning")
        console.print(
            "[yellow]Some demo results are missing for comparison. Run all demos first.[/yellow]"
        )
        return False

    # Create a comparison table
    comparison = Table(title="Classification Strategies Comparison", box=box.ROUNDED)
    comparison.add_column("Strategy", style="cyan")
    comparison.add_column("Provider/Model", style="green")
    comparison.add_column("Tokens", style="yellow")
    comparison.add_column("Processing Time", style="magenta")
    comparison.add_column("Cost", style="red")
    comparison.add_column("Notes", style="white")

    for strategy, result in DEMO_RESULTS_CACHE.items():
        provider_model = f"{result.get('provider', 'unknown')}/{result.get('model', 'unknown')}"
        tokens = result.get("tokens", {}).get("total", 0)
        time_taken = f"{result.get('processing_time', 0.0):.3f}s"
        cost = f"${result.get('cost', 0.0):.6f}"

        # Add strategy-specific notes
        notes = ""
        if strategy == "basic":
            notes = "Simple and efficient for clear categories"
        elif strategy == "multi_label":
            notes = f"Found {len(result.get('classifications', []))} categories"
        elif strategy == "hierarchical":
            notes = "Effective for multi-dimensional taxonomies"
        elif strategy == "few_shot":
            notes = "Improved accuracy with example learning"
        elif strategy == "ensemble":
            notes = f"Aggregated {len(result.get('ensemble_models', []))} models"
        elif strategy == "custom_prompt":
            notes = "Tailored instruction for specific domain"

        # Format strategy name for display
        strategy_display = strategy.replace("_", " ").title()

        comparison.add_row(strategy_display, provider_model, str(tokens), time_taken, cost, notes)

    console.print(comparison)
    console.print()

    # Generate chart data for cost comparison
    costs = {k: v.get("cost", 0.0) for k, v in DEMO_RESULTS_CACHE.items()}
    tokens = {k: v.get("tokens", {}).get("total", 0) for k, v in DEMO_RESULTS_CACHE.items()}
    times = {k: v.get("processing_time", 0.0) for k, v in DEMO_RESULTS_CACHE.items()}

    # Create visual dashboard of results
    display_visual_dashboard(costs, tokens, times)

    # Export comparison data if enabled
    if EXPORT_RESULTS:
        comparison_data = {
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "costs": costs,
            "tokens": tokens,
            "times": times,
            "full_results": DEMO_RESULTS_CACHE,
        }
        with open(f"{RESULTS_DIR}/comparison_{time.strftime('%Y%m%d-%H%M%S')}.json", "w") as f:
            json.dump(comparison_data, f, indent=2)
        logger.info(f"Exported comparison data to {RESULTS_DIR}", emoji_key="save")

    # Display conclusion
    conclusion_panel = Panel(
        "Classification strategies comparison shows tradeoffs between accuracy, cost, and performance.\n\n"
        "- Zero-shot: Fastest and cheapest, good for simple categories\n"
        "- Few-shot: Better accuracy with examples, moderate cost increase\n"
        "- Hierarchical: Excellent for complex taxonomies, higher token usage\n"
        "- Ensemble: Highest accuracy but also highest cost and processing time\n"
        "- Custom prompt: Tailored for specific domains, good balance of accuracy and efficiency",
        title="Strategy Selection Guidelines",
        border_style="green",
        expand=False,
    )
    console.print(conclusion_panel)

    return True


def display_visual_dashboard(costs, tokens, times):
    """Display a visual dashboard of classification metrics using Rich Layout."""
    # Create layout
    layout = Layout()

    # Split into sections
    layout.split(Layout(name="header", size=3), Layout(name="main", ratio=1))

    # Split main section into columns
    layout["main"].split_row(
        Layout(name="costs", ratio=1), Layout(name="tokens", ratio=1), Layout(name="times", ratio=1)
    )

    # Create header
    header = Panel(Text("Classification Strategy Metrics", style="bold magenta"), box=box.ROUNDED)

    # Create visualization panels
    costs_panel = create_metric_panel(costs, "Classification Costs ($)", "red")
    tokens_panel = create_metric_panel(tokens, "Token Usage", "yellow")
    times_panel = create_metric_panel(times, "Processing Time (s)", "green")

    # Update layout
    layout["header"] = header
    layout["main"]["costs"] = costs_panel
    layout["main"]["tokens"] = tokens_panel
    layout["main"]["times"] = times_panel

    # Display dashboard
    console.print(layout)


def create_metric_panel(data, title, color):
    """Create a panel with visualization of metric data."""
    # Find max value for scaling
    max_value = max(data.values()) if data else 1
    scale_factor = 20  # Bar length scaling

    # Generate content
    content = ""
    for strategy, value in data.items():
        bar_length = int((value / max_value) * scale_factor) if max_value > 0 else 0
        bar = "█" * bar_length
        strategy_display = strategy.replace("_", " ").title()
        content += f"{strategy_display.ljust(15)} │ [{color}]{bar}[/{color}] {value:.4f}\n"

    return Panel(content, title=title, border_style=color)


async def run_all_demos(tracker: CostTracker):  # Add tracker
    """Run all classification demos in sequence."""
    console.print(Rule("[bold magenta]Text Classification Comprehensive Demo[/bold magenta]"))
    logger.info("Starting comprehensive text classification demo", emoji_key="start")

    start_time = time.time()
    success = True

    # Create results directory if exporting is enabled
    if EXPORT_RESULTS and not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        logger.info(f"Created results directory at {RESULTS_DIR}", emoji_key="folder")

    # Setup live display for overall progress
    overall_progress = Table.grid(expand=True)
    overall_progress.add_column()
    overall_progress.add_row("[bold blue]Running Text Classification Demo Suite...[/bold blue]")
    overall_progress.add_row("[cyan]Press Ctrl+C to abort[/cyan]")

    try:
        # Create a live display that updates during the demo
        with Live(overall_progress, refresh_per_second=4, console=console):
            # Run demos with timeout protection
            demo_tasks = [
                asyncio.create_task(demo_basic_classification(tracker)),  # Pass tracker
                asyncio.create_task(demo_multi_label_classification(tracker)),  # Pass tracker
                asyncio.create_task(demo_hierarchical_classification(tracker)),  # Pass tracker
                asyncio.create_task(demo_few_shot_classification(tracker)),  # Pass tracker
                asyncio.create_task(demo_ensemble_classification(tracker)),  # Pass tracker
                asyncio.create_task(demo_custom_prompt_template(tracker)),  # Pass tracker
            ]

            # Run all demos with timeout
            completed, pending = await asyncio.wait(
                demo_tasks, timeout=DEMO_TIMEOUT, return_when=asyncio.ALL_COMPLETED
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                overall_progress.add_row(f"[yellow]Demo timed out after {DEMO_TIMEOUT}s[/yellow]")

            # Compare results if we have enough demos completed
            if len(completed) >= 3:  # Require at least 3 demos for comparison
                await demo_comparison(tracker)

    except asyncio.CancelledError:
        logger.warning("Demo was cancelled by user", emoji_key="cancel")
        success = False
    except Exception as e:
        logger.critical(
            f"Text classification demo failed: {str(e)}", emoji_key="critical", exc_info=True
        )
        console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        success = False

    # Calculate total time
    total_time = time.time() - start_time

    # Display cost summary
    tracker.display_summary(console)

    if success:
        logger.success(
            f"Text Classification Demo Completed Successfully in {total_time:.2f}s!",
            emoji_key="complete",
        )
        console.print(
            Rule(
                f"[bold magenta]Text Classification Demo Complete ({total_time:.2f}s)[/bold magenta]"
            )
        )
        return 0
    else:
        logger.error(f"Text classification demo failed after {total_time:.2f}s", emoji_key="error")
        console.print(Rule("[bold red]Text Classification Demo Failed[/bold red]"))
        return 1


async def main():
    """Run the full text classification demo suite."""
    tracker = CostTracker()  # Instantiate tracker
    try:
        return await run_all_demos(tracker)  # Pass tracker
    except Exception as e:
        logger.critical(f"Demo failed unexpectedly: {str(e)}", emoji_key="critical", exc_info=True)
        return 1


if __name__ == "__main__":
    # Check for environment variables and display configuration
    if EXPORT_RESULTS:
        console.print(f"[blue]Results will be exported to: {RESULTS_DIR}[/blue]")

    # Check if sample files exist
    if not all(
        path.exists()
        for path in [
            NEWS_SAMPLES_PATH,
            PRODUCT_REVIEWS_PATH,
            SUPPORT_TICKETS_PATH,
            EMAIL_SAMPLES_PATH,
        ]
    ):
        console.print(
            "[bold red]Error:[/bold red] Sample data files not found. Please ensure all sample files exist in examples/sample/text_classification_samples/"
        )
        sys.exit(1)

    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
