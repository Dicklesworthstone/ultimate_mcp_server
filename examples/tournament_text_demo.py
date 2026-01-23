#!/usr/bin/env python3
"""
Tournament Text Demo - Demonstrates running a text improvement tournament

This script shows how to:
1. Create a tournament with multiple models focused on text refinement
2. Track progress across multiple rounds
3. Retrieve and analyze the improved essay/text

The tournament task is to refine and improve a comparative essay on
transformer vs. diffusion model architectures, demonstrating how
the tournament system can be used for general text refinement tasks.

Usage:
  python examples/tournament_text_demo.py [--topic TOPIC]

Options:
  --topic TOPIC    Specify a different essay topic (default: transformers vs diffusion models)
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from ultimate_mcp_server.core.models.requests import CompletionRequest
from ultimate_mcp_server.core.providers.base import get_provider
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.services.prompts import PromptTemplate
from ultimate_mcp_server.tools.tournament import (
    create_tournament,
    get_tournament_results,
    get_tournament_status,
)
from ultimate_mcp_server.utils import get_logger, process_mcp_result
from ultimate_mcp_server.utils.display import (
    CostTracker,
    display_tournament_results,
    display_tournament_status,
)
from ultimate_mcp_server.utils.logging.console import console

DEFAULT_MODEL_CONFIGS_TEXT: List[Dict[str, Any]] = [
    {
        "model_id": "openai/gpt-4o-mini",
        "diversity_count": 1,
        "temperature": 0.75,
    },
    {
        "model_id": "anthropic/claude-3-5-haiku-20241022",
        "diversity_count": 1,
        "temperature": 0.7,
    },
]
DEFAULT_NUM_ROUNDS_TEXT = 2
DEFAULT_TOURNAMENT_NAME_TEXT = "Advanced Text Refinement Tournament"


def parse_arguments_text():
    parser = argparse.ArgumentParser(description="Run a text refinement tournament demo")
    parser.add_argument(
        "--topic",
        type=str,
        default="transformer_vs_diffusion",
        choices=list(TOPICS.keys()) + ["custom"],
        help="Essay topic (default: transformer_vs_diffusion)",
    )
    parser.add_argument(
        "--custom-topic", type=str, help="Custom essay topic (used when --topic=custom)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_NUM_ROUNDS_TEXT,
        help=f"Number of tournament rounds (default: {DEFAULT_NUM_ROUNDS_TEXT})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[mc["model_id"] for mc in DEFAULT_MODEL_CONFIGS_TEXT],
        help="List of model IDs to participate.",
    )
    return parser.parse_args()


# Initialize logger using get_logger
logger = get_logger("example.tournament_text")

# Create a simple structure for cost tracking from dict (tokens might be missing)
TrackableResult = namedtuple(
    "TrackableResult",
    ["cost", "input_tokens", "output_tokens", "provider", "model", "processing_time"],
)

# Initialize global gateway
gateway: Optional[Gateway] = None

# --- Configuration ---
# Adjust model IDs based on your configured providers
MODEL_IDS = ["openai:gpt-4.1-mini", "deepseek:deepseek-chat", "gemini:gemini-2.5-pro-preview-03-25"]
NUM_ROUNDS = 2  # Changed from 3 to 2 for faster execution and debugging
TOURNAMENT_NAME = "Text Refinement Tournament Demo"  # More generic name

# The generic essay prompt template
TEMPLATE_TEXT = """
# GENERIC TEXT TOURNAMENT PROMPT TEMPLATE

Please write a high-quality, comprehensive {{content_type}} on the topic of: "{{topic}}".

{{context}}

Your {{content_type}} should thoroughly explore the following sections and subtopics:
{% for section in sections %}
## {{section.title}}
{% for subtopic in section.subtopics %}
- {{subtopic}}
{% endfor %}
{% endfor %}

Adhere to the following style and content requirements:
{{style_requirements}}

Please provide only the {{content_type}} text. If you have meta-comments or a thinking process,
enclose it in <thinking>...</thinking> tags at the very beginning of your response.
"""

# Define predefined topics
TOPICS = {
    "transformer_vs_diffusion": {
        "content_type": "technical essay",
        "topic": "comparing transformer architecture and diffusion models",
        "context": "Focus on their underlying mechanisms, common applications, strengths, weaknesses, and future potential in AI.",
        "sections": [
            {
                "title": "Core Principles",
                "subtopics": [
                    "Transformer self-attention, positional encoding",
                    "Diffusion forward/reverse processes, noise schedules",
                ],
            },
            {
                "title": "Applications & Performance",
                "subtopics": [
                    "Typical tasks for transformers (NLP, vision)",
                    "Typical tasks for diffusion models (image/audio generation)",
                    "Comparative performance benchmarks or known strengths",
                ],
            },
            {
                "title": "Limitations & Challenges",
                "subtopics": [
                    "Computational costs, data requirements",
                    "Interpretability, controllability, known failure modes for each",
                ],
            },
            {
                "title": "Future Outlook",
                "subtopics": [
                    "Potential for hybridization",
                    "Scaling frontiers",
                    "Impact on AGI research",
                ],
            },
        ],
        "style_requirements": "Write in a clear, objective, and technically precise manner suitable for an audience with a machine learning background. Aim for around 800-1200 words.",
    },
    "llm_vs_traditional_ai": {
        "content_type": "comparative analysis",
        "topic": "comparing large language models to traditional AI approaches",
        "context": "The rise of large language models has shifted the AI landscape significantly.",
        "sections": [
            {
                "title": "Fundamental Differences",
                "subtopics": [
                    "How LLMs differ architecturally from traditional ML/AI systems",
                    "Data requirements and training approaches",
                ],
            },
            {
                "title": "Capabilities and Limitations",
                "subtopics": [
                    "Tasks where LLMs excel compared to traditional approaches",
                    "Situations where traditional AI methods remain superior",
                    "Emergent capabilities unique to large language models",
                ],
            },
            {
                "title": "Real-world Applications",
                "subtopics": [
                    "Industries being transformed by LLMs",
                    "Where traditional AI approaches continue to dominate",
                    "Examples of hybrid systems combining both approaches",
                ],
            },
            {
                "title": "Future Outlook",
                "subtopics": [
                    "Projected evolution of both paradigms",
                    "Potential convergence or further divergence",
                    "Research frontiers for each approach",
                ],
            },
        ],
        "style_requirements": "Present a balanced analysis that acknowledges the strengths and weaknesses of both paradigms. Support claims with specific examples where possible.",
    },
}


# Create custom topic template
def create_custom_topic_variables(topic_description):
    """Create a simple custom topic with standard sections"""
    return {
        "content_type": "essay",
        "topic": topic_description,
        "context": "",
        "sections": [
            {
                "title": "Background and Key Concepts",
                "subtopics": [
                    "Define and explain the core elements of the topic",
                    "Provide necessary historical or theoretical context",
                ],
            },
            {
                "title": "Analysis of Main Aspects",
                "subtopics": [
                    "Examine the primary dimensions or elements of the topic",
                    "Discuss relationships between different aspects",
                    "Identify patterns or trends relevant to the topic",
                ],
            },
            {
                "title": "Practical Implications",
                "subtopics": [
                    "Real-world applications or impacts",
                    "How this topic affects related fields or domains",
                ],
            },
            {
                "title": "Future Perspectives",
                "subtopics": [
                    "Emerging trends or developments",
                    "Potential challenges and opportunities",
                    "Areas requiring further research or exploration",
                ],
            },
        ],
        "style_requirements": "Present a comprehensive and well-structured analysis with clear reasoning and specific examples where appropriate.",
    }


# Create the prompt template object
essay_template = PromptTemplate(
    template=TEMPLATE_TEXT,
    template_id="text_tournament_template",
    description="A template for text tournament prompts",
    required_vars=["content_type", "topic", "context", "sections", "style_requirements"],
)


# --- Helper Functions ---
def parse_result(result):
    """Parse the result from a tool call into a usable dictionary.

    Handles various return types from MCP tools.
    """
    try:
        # Handle TextContent object (which has a .text attribute)
        if hasattr(result, "text"):
            try:
                # Try to parse the text as JSON
                return json.loads(result.text)
            except json.JSONDecodeError:
                # Return the raw text if not JSON
                return {"text": result.text}

        # Handle list result
        if isinstance(result, list):
            if result:
                first_item = result[0]
                if hasattr(first_item, "text"):
                    try:
                        return json.loads(first_item.text)
                    except json.JSONDecodeError:
                        return {"text": first_item.text}
                else:
                    return first_item
            return {}

        # Handle dictionary directly
        if isinstance(result, dict):
            return result

        # Handle other potential types or return error
        else:
            return {"error": f"Unexpected result type: {type(result)}"}

    except Exception as e:
        return {"error": f"Error parsing result: {str(e)}"}


async def setup_gateway():
    """Set up the gateway for demonstration."""
    global gateway

    # Create gateway instance
    logger.info("Initializing gateway for demonstration", emoji_key="start")
    gateway = Gateway("text-tournament-demo", register_tools=False)

    # Initialize the server with all providers and built-in tools
    await gateway._initialize_providers()

    # Manually register tournament tools
    mcp = gateway.mcp
    mcp.tool()(create_tournament)
    mcp.tool()(get_tournament_status)
    mcp.tool()(get_tournament_results)
    logger.info("Manually registered tournament tools.")

    # Verify tools are registered
    tools = await gateway.mcp.list_tools()
    tournament_tools = [
        t.name for t in tools if t.name.startswith("tournament") or "tournament" in t.name
    ]
    logger.info(f"Registered tournament tools: {tournament_tools}", emoji_key="info")

    if not any("tournament" in t.lower() for t in [t.name for t in tools]):
        logger.warning(
            "No tournament tools found. Make sure tournament plugins are registered.",
            emoji_key="warning",
        )

    logger.success("Gateway initialized", emoji_key="success")


async def poll_tournament_status(
    tournament_id: str, storage_path: Optional[str] = None, interval: int = 5
) -> Optional[str]:
    """Poll the tournament status until it reaches a final state.

    Args:
        tournament_id: ID of the tournament to poll
        storage_path: Optional storage path to avoid tournament not found issues
        interval: Time between status checks in seconds
    """
    logger.info(f"Polling status for tournament {tournament_id}...", emoji_key="poll")
    final_states = ["COMPLETED", "FAILED", "CANCELLED"]

    # Add direct file polling capability to handle case where tournament manager can't find the tournament
    if storage_path:
        storage_dir = Path(storage_path)
        state_file = storage_dir / "tournament_state.json"
        logger.debug(f"Will check tournament state file directly at: {state_file}")

    while True:
        status_input = {"tournament_id": tournament_id}
        status_result = await gateway.mcp.call_tool("get_tournament_status", status_input)
        status_data = await process_mcp_result(status_result)

        if "error" in status_data:
            # If tournament manager couldn't find the tournament but we have the storage path,
            # try to read the state file directly (this is a fallback mechanism)
            if storage_path and "not found" in status_data.get("error", "").lower():
                try:
                    logger.debug(f"Attempting to read tournament state directly from: {state_file}")
                    if state_file.exists():
                        with open(state_file, "r", encoding="utf-8") as f:
                            direct_status_data = json.load(f)
                            status = direct_status_data.get("status")
                            current_round = direct_status_data.get("current_round", 0)
                            total_rounds = direct_status_data.get("config", {}).get("rounds", 0)

                            # Create a status object compatible with our display function
                            status_data = {
                                "tournament_id": tournament_id,
                                "status": status,
                                "current_round": current_round,
                                "total_rounds": total_rounds,
                                "storage_path": storage_path,
                            }
                            logger.debug(f"Successfully read direct state: {status}")
                    else:
                        logger.warning(f"State file not found at: {state_file}")
                except Exception as e:
                    logger.error(f"Error reading state file directly: {e}")
                    logger.error(
                        f"Error fetching status: {status_data['error']}", emoji_key="error"
                    )
                    return None  # Indicate error during polling
            else:
                # Standard error case
                logger.error(f"Error fetching status: {status_data['error']}", emoji_key="error")
                return None  # Indicate error during polling

        # Display improved status using the imported function
        display_tournament_status(status_data)

        status = status_data.get("status")
        if status in final_states:
            logger.success(f"Tournament reached final state: {status}", emoji_key="success")
            return status

        await asyncio.sleep(interval)


def extract_thinking(text: str) -> str:
    """Extract <thinking> tags content (simple version)."""
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def analyze_text_quality(text: str) -> Dict[str, Any]:
    """Basic text quality analysis."""
    word_count = len(text.split())
    # Add more metrics later (readability, sentiment, etc.)
    return {"word_count": word_count}


async def evaluate_essays(
    essays_by_model: Dict[str, str], tracker: CostTracker = None
) -> Dict[str, Any]:
    """Use LLM to evaluate which essay is the best.

    Args:
        essays_by_model: Dictionary mapping model IDs to their essay texts
        tracker: Optional CostTracker to track API call costs

    Returns:
        Dictionary with evaluation results
    """
    if not essays_by_model or len(essays_by_model) < 2:
        return {"error": "Not enough essays to compare"}

    eval_cost = 0.0  # Initialize evaluation cost

    try:
        # Format the essays for evaluation
        evaluation_prompt = "# Essay Evaluation\n\nPlease analyze the following essays on the same topic and determine which one is the best. "
        evaluation_prompt += "Consider factors such as technical accuracy, clarity, organization, depth of analysis, and overall quality.\n\n"

        # Add each essay
        for i, (model_id, essay) in enumerate(essays_by_model.items(), 1):
            display_model = model_id.split(":")[-1] if ":" in model_id else model_id
            # Limit each essay to 3000 chars to fit context windows
            truncated_essay = essay[:3000]
            if len(essay) > 3000:
                truncated_essay += "..."
            evaluation_prompt += f"## Essay {i} (by {display_model})\n\n{truncated_essay}\n\n"

        evaluation_prompt += "\n# Your Evaluation Task\n\n"
        evaluation_prompt += "1. Rank the essays from best to worst\n"
        evaluation_prompt += "2. Explain your reasoning for the ranking\n"
        evaluation_prompt += "3. Highlight specific strengths of the best essay\n"
        evaluation_prompt += "4. Suggest one improvement for each essay\n"

        # Use a more capable model for evaluation
        model_to_use = "gemini:gemini-2.5-pro-preview-03-25"

        logger.info(f"Evaluating essays using {model_to_use}...", emoji_key="evaluate")

        # Get the provider
        provider_id = model_to_use.split(":")[0]
        provider = await get_provider(provider_id)

        if not provider:
            return {
                "error": f"Provider {provider_id} not available for evaluation",
                "model_used": model_to_use,
                "eval_prompt": evaluation_prompt,
                "cost": 0.0,
            }

        # Generate completion for evaluation with timeout
        try:
            request = CompletionRequest(prompt=evaluation_prompt, model=model_to_use)

            # Set a timeout for the completion request
            completion_task = provider.generate_completion(
                prompt=request.prompt, model=request.model
            )

            # 45 second timeout for evaluation
            completion_result = await asyncio.wait_for(completion_task, timeout=45)

            # Track API call if tracker provided
            if tracker:
                tracker.add_call(completion_result)

            # Accumulate cost
            if hasattr(completion_result, "cost"):
                eval_cost = completion_result.cost
            elif hasattr(completion_result, "metrics") and isinstance(
                completion_result.metrics, dict
            ):
                eval_cost = completion_result.metrics.get("cost", 0.0)

            # Prepare result dict
            result = {
                "evaluation": completion_result.text,
                "model_used": model_to_use,
                "eval_prompt": evaluation_prompt,
                "cost": eval_cost,  # Return the cost
            }
        except asyncio.TimeoutError:
            logger.warning(
                f"Evaluation with {model_to_use} timed out after 45 seconds", emoji_key="warning"
            )
            return {
                "error": "Evaluation timed out after 45 seconds",
                "model_used": model_to_use,
                "eval_prompt": evaluation_prompt,
                "cost": 0.0,
            }
        except Exception as request_error:
            logger.error(f"Error during model request: {str(request_error)}", emoji_key="error")
            return {
                "error": f"Error during model request: {str(request_error)}",
                "model_used": model_to_use,
                "eval_prompt": evaluation_prompt,
                "cost": 0.0,
            }

    except Exception as e:
        logger.error(f"Essay evaluation failed: {str(e)}", emoji_key="error", exc_info=True)
        return {
            "error": str(e),
            "model_used": model_to_use if "model_to_use" in locals() else "unknown",
            "eval_prompt": evaluation_prompt
            if "evaluation_prompt" in locals()
            else "Error generating prompt",
            "cost": 0.0,
        }

    return result


async def calculate_tournament_costs(rounds_results, evaluation_cost=None):
    """Calculate total costs of the tournament by model and grand total.

    Args:
        rounds_results: List of round results data from tournament results
        evaluation_cost: Optional cost of the final evaluation step

    Returns:
        Dictionary with cost information
    """
    model_costs = {}
    total_cost = 0.0

    # Process costs for each round
    for _round_idx, round_data in enumerate(rounds_results):
        responses = round_data.get("responses", {})
        for model_id, response in responses.items():
            metrics = response.get("metrics", {})
            cost = metrics.get("cost", 0.0)

            # Convert to float if it's a string
            if isinstance(cost, str):
                try:
                    cost = float(cost.replace("$", ""))
                except (ValueError, TypeError):
                    cost = 0.0

            # Initialize model if not present
            if model_id not in model_costs:
                model_costs[model_id] = 0.0

            # Add to model total and grand total
            model_costs[model_id] += cost
            total_cost += cost

    # Add evaluation cost if provided
    if evaluation_cost:
        total_cost += evaluation_cost
        model_costs["evaluation"] = evaluation_cost

    return {"model_costs": model_costs, "total_cost": total_cost}


# --- Main Script Logic ---
async def run_tournament_demo(tracker: CostTracker):
    """Run the text tournament demo."""
    # Parse command line arguments
    args = parse_arguments_text()

    # Determine which topic to use
    if args.topic == "custom" and args.custom_topic:
        # Custom topic provided via command line
        topic_name = "custom"
        essay_variables = create_custom_topic_variables(args.custom_topic)
        topic_description = args.custom_topic
        log_topic_info = f"Using custom topic: [yellow]{escape(topic_description)}[/yellow]"
    elif args.topic in TOPICS:
        # Use one of the predefined topics
        topic_name = args.topic
        essay_variables = TOPICS[args.topic]
        topic_description = essay_variables["topic"]
        log_topic_info = f"Using predefined topic: [yellow]{escape(topic_description)}[/yellow]"
    else:
        # Default to transformer vs diffusion if topic not recognized
        topic_name = "transformer_vs_diffusion"
        essay_variables = TOPICS[topic_name]
        topic_description = essay_variables["topic"]
        log_topic_info = f"Using default topic: [yellow]{escape(topic_description)}[/yellow]"

    # Use Rich Rule for title
    console.print(
        Rule(f"[bold blue]{TOURNAMENT_NAME} - {topic_name.replace('_', ' ').title()}[/bold blue]")
    )
    console.print(log_topic_info)
    console.print(f"Models: [cyan]{', '.join(MODEL_IDS)}[/cyan]")
    console.print(f"Rounds: [cyan]{NUM_ROUNDS}[/cyan]")

    # Render the template
    try:
        rendered_prompt = essay_template.render(essay_variables)
        logger.info(f"Template rendered for topic: {topic_name}", emoji_key="template")

        # Show prompt preview in a Panel
        prompt_preview = rendered_prompt.split("\n")[:10]  # Show more lines
        preview_text = "\n".join(prompt_preview) + "\n..."
        console.print(
            Panel(
                escape(preview_text),
                title="[bold]Rendered Prompt Preview[/bold]",
                border_style="dim blue",
                expand=False,
            )
        )

    except Exception as e:
        logger.error(f"Template rendering failed: {str(e)}", emoji_key="error", exc_info=True)
        # Log template and variables for debugging using logger
        logger.debug(f"Template: {TEMPLATE_TEXT}")
        logger.debug(
            f"Variables: {escape(str(essay_variables))}"
        )  # Escape potentially complex vars
        return 1

    # 1. Create the tournament
    # Prepare model configurations
    # Default temperature from DEFAULT_MODEL_CONFIGS_TEXT, assuming it's a common parameter.
    # The create_tournament tool itself will parse these against InputModelConfig.
    model_configs = [
        {"model_id": mid, "diversity_count": 1, "temperature": 0.7} for mid in MODEL_IDS
    ]

    create_input = {
        "name": f"{TOURNAMENT_NAME} - {topic_name.replace('_', ' ').title()}",
        "prompt": rendered_prompt,
        "models": model_configs,  # Changed from model_ids to models
        "rounds": NUM_ROUNDS,
        "tournament_type": "text",
    }

    try:
        logger.info("Creating tournament...", emoji_key="processing")
        create_result = await gateway.mcp.call_tool("create_tournament", create_input)
        create_data = await process_mcp_result(create_result)

        if "error" in create_data:
            error_msg = create_data.get("error", "Unknown error")
            logger.error(f"Failed to create tournament: {error_msg}. Exiting.", emoji_key="error")
            return 1

        tournament_id = create_data.get("tournament_id")
        if not tournament_id:
            logger.error("No tournament ID returned. Exiting.", emoji_key="error")
            return 1

        # Extract storage path for reference
        storage_path = create_data.get("storage_path")
        logger.info(f"Tournament created with ID: {tournament_id}", emoji_key="tournament")
        if storage_path:
            logger.info(f"Tournament storage path: {storage_path}", emoji_key="path")

        # Add a small delay to ensure the tournament state is saved before proceeding
        await asyncio.sleep(2)

        # 2. Poll for status
        final_status = await poll_tournament_status(tournament_id, storage_path)

        # 3. Fetch and display final results
        if final_status == "COMPLETED":
            logger.info("Fetching final results...", emoji_key="results")
            results_input = {"tournament_id": tournament_id}
            final_results = await gateway.mcp.call_tool("get_tournament_results", results_input)
            results_data = await process_mcp_result(final_results)

            if "error" not in results_data:
                # Use the imported display function for tournament results
                display_tournament_results(results_data)

                # Track aggregated tournament cost (excluding separate evaluation)
                if isinstance(results_data, dict) and "cost" in results_data:
                    try:
                        total_cost = results_data.get("cost", {}).get("total_cost", 0.0)
                        processing_time = results_data.get("total_processing_time", 0.0)
                        trackable = TrackableResult(
                            cost=total_cost,
                            input_tokens=0,
                            output_tokens=0,
                            provider="tournament",
                            model="text_tournament",
                            processing_time=processing_time,
                        )
                        tracker.add_call(trackable)
                        logger.info(f"Tracked tournament cost: ${total_cost:.6f}", emoji_key="cost")
                    except Exception as track_err:
                        logger.warning(
                            f"Could not track tournament cost: {track_err}", exc_info=False
                        )

                # Analyze round progression if available
                rounds_results = results_data.get("rounds_results", [])
                if rounds_results:
                    console.print(Rule("[bold blue]Essay Evolution Analysis[/bold blue]"))

                    for round_idx, round_data in enumerate(rounds_results):
                        console.print(f"[bold]Round {round_idx} Analysis:[/bold]")
                        responses = round_data.get("responses", {})

                        round_table = Table(box=box.MINIMAL, show_header=True, expand=False)
                        round_table.add_column("Model", style="magenta")
                        round_table.add_column("Word Count", style="green", justify="right")

                        has_responses = False
                        for model_id, response in responses.items():
                            display_model = escape(model_id.split(":")[-1])
                            response_text = response.get("response_text", "")

                            if response_text:
                                has_responses = True
                                metrics = analyze_text_quality(response_text)
                                round_table.add_row(display_model, str(metrics["word_count"]))

                        if has_responses:
                            console.print(round_table)
                        else:
                            console.print("[dim]No valid responses recorded for this round.[/dim]")
                        console.print()  # Add space between rounds

                    # Evaluate final essays using LLM
                    final_round = rounds_results[-1]
                    final_responses = final_round.get("responses", {})

                    # Track evaluation cost
                    evaluation_cost = 0.0

                    if final_responses:
                        console.print(Rule("[bold blue]AI Evaluation of Essays[/bold blue]"))
                        console.print("[bold]Evaluating final essays...[/bold]")

                        essays_by_model = {}
                        for model_id, response in final_responses.items():
                            essays_by_model[model_id] = response.get("response_text", "")

                        evaluation_result = await evaluate_essays(essays_by_model, tracker)

                        if "error" not in evaluation_result:
                            console.print(
                                Panel(
                                    escape(evaluation_result["evaluation"]),
                                    title=f"[bold]Essay Evaluation (by {evaluation_result['model_used'].split(':')[-1]})[/bold]",
                                    border_style="green",
                                    expand=False,
                                )
                            )

                            # Track evaluation cost separately
                            if evaluation_cost > 0:
                                try:
                                    trackable_eval = TrackableResult(
                                        cost=evaluation_cost,
                                        input_tokens=0,  # Tokens for eval not easily available here
                                        output_tokens=0,
                                        provider=evaluation_result["model_used"].split(":")[0],
                                        model=evaluation_result["model_used"].split(":")[-1],
                                        processing_time=0,  # Eval time not tracked here
                                    )
                                    tracker.add_call(trackable_eval)
                                except Exception as track_err:
                                    logger.warning(
                                        f"Could not track evaluation cost: {track_err}",
                                        exc_info=False,
                                    )

                            # Save evaluation result to a file in the tournament directory
                            if storage_path:
                                try:
                                    evaluation_file = os.path.join(
                                        storage_path, "essay_evaluation.md"
                                    )
                                    with open(evaluation_file, "w", encoding="utf-8") as f:
                                        f.write(
                                            f"# Essay Evaluation by {evaluation_result['model_used']}\n\n"
                                        )
                                        f.write(evaluation_result["evaluation"])

                                    logger.info(
                                        f"Evaluation saved to {evaluation_file}", emoji_key="save"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Could not save evaluation to file: {str(e)}",
                                        emoji_key="warning",
                                    )

                            # Track evaluation cost if available
                            evaluation_cost = evaluation_result.get("cost", 0.0)
                            logger.info(
                                f"Evaluation cost: ${evaluation_cost:.6f}", emoji_key="cost"
                            )
                        else:
                            console.print(
                                f"[yellow]Could not evaluate essays: {evaluation_result.get('error')}[/yellow]"
                            )
                            # Try with fallback model if Gemini fails
                            if "gemini" in evaluation_result.get("model_used", ""):
                                console.print(
                                    "[bold]Trying evaluation with fallback model (gpt-4.1-mini)...[/bold]"
                                )
                                # Switch to OpenAI model as backup
                                essays_by_model_limited = {}
                                # Limit content size to avoid token limits
                                for model_id, essay in essays_by_model.items():
                                    essays_by_model_limited[model_id] = essay[
                                        :5000
                                    ]  # Shorter excerpt to fit in context

                                fallback_evaluation = {
                                    "model_used": "openai:gpt-4.1-mini",
                                    "eval_prompt": evaluation_result.get(
                                        "eval_prompt", "Evaluation failed"
                                    ),
                                }

                                try:
                                    provider_id = "openai"
                                    provider = await get_provider(provider_id)

                                    if provider:
                                        # Create a shorter, simplified prompt
                                        simple_prompt = "Compare these essays and rank them from best to worst:\n\n"
                                        for i, (model_id, essay) in enumerate(
                                            essays_by_model_limited.items(), 1
                                        ):
                                            display_model = (
                                                model_id.split(":")[-1]
                                                if ":" in model_id
                                                else model_id
                                            )
                                            simple_prompt += f"Essay {i} ({display_model}):\n{essay[:2000]}...\n\n"

                                        request = CompletionRequest(
                                            prompt=simple_prompt, model="openai:gpt-4.1-mini"
                                        )
                                        completion_result = await provider.generate_completion(
                                            prompt=request.prompt, model=request.model
                                        )

                                        fallback_evaluation["evaluation"] = completion_result.text

                                        # Track fallback evaluation cost
                                        if completion_result.cost > 0:
                                            try:
                                                trackable_fallback = TrackableResult(
                                                    cost=completion_result.cost,
                                                    input_tokens=0,
                                                    output_tokens=0,
                                                    provider="openai",
                                                    model="gpt-4.1-mini",
                                                    processing_time=0,  # Eval time not tracked
                                                )
                                                tracker.add_call(trackable_fallback)
                                            except Exception as track_err:
                                                logger.warning(
                                                    f"Could not track fallback evaluation cost: {track_err}",
                                                    exc_info=False,
                                                )

                                        logger.info(
                                            f"Fallback evaluation cost: ${completion_result.cost:.6f}",
                                            emoji_key="cost",
                                        )

                                        console.print(
                                            Panel(
                                                escape(fallback_evaluation["evaluation"]),
                                                title="[bold]Fallback Evaluation (by gpt-4.1-mini)[/bold]",
                                                border_style="yellow",
                                                expand=False,
                                            )
                                        )

                                        # Save fallback evaluation to file
                                        if storage_path:
                                            try:
                                                fallback_eval_file = os.path.join(
                                                    storage_path, "fallback_evaluation.md"
                                                )
                                                with open(
                                                    fallback_eval_file, "w", encoding="utf-8"
                                                ) as f:
                                                    f.write(
                                                        "# Fallback Essay Evaluation by gpt-4.1-mini\n\n"
                                                    )
                                                    f.write(fallback_evaluation["evaluation"])

                                                logger.info(
                                                    f"Fallback evaluation saved to {fallback_eval_file}",
                                                    emoji_key="save",
                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    f"Could not save fallback evaluation: {str(e)}",
                                                    emoji_key="warning",
                                                )
                                    else:
                                        console.print("[red]Fallback model unavailable[/red]")
                                except Exception as fallback_error:
                                    console.print(
                                        f"[red]Fallback evaluation failed: {str(fallback_error)}[/red]"
                                    )

                    # Find and highlight comparison file for final round
                    comparison_file = final_round.get("comparison_file_path")
                    if comparison_file:
                        console.print(
                            Panel(
                                f"Check the final comparison file for the full essay text and detailed round comparisons:\n[bold yellow]{escape(comparison_file)}[/bold yellow]",
                                title="[bold]Final Comparison File[/bold]",
                                border_style="yellow",
                                expand=False,
                            )
                        )
                    else:
                        logger.warning(
                            "Could not find path to final comparison file in results",
                            emoji_key="warning",
                        )

                    # Display cost summary
                    costs = await calculate_tournament_costs(rounds_results, evaluation_cost)
                    model_costs = costs.get("model_costs", {})
                    total_cost = costs.get("total_cost", 0.0)

                    console.print(Rule("[bold blue]Tournament Cost Summary[/bold blue]"))

                    cost_table = Table(box=box.MINIMAL, show_header=True, expand=False)
                    cost_table.add_column("Model", style="magenta")
                    cost_table.add_column("Total Cost", style="green", justify="right")

                    # Add model costs to table
                    for model_id, cost in sorted(model_costs.items()):
                        if model_id == "evaluation":
                            display_model = "Evaluation"
                        else:
                            display_model = model_id.split(":")[-1] if ":" in model_id else model_id

                        cost_table.add_row(display_model, f"${cost:.6f}")

                    # Add grand total
                    cost_table.add_row(
                        "[bold]GRAND TOTAL[/bold]", f"[bold]${total_cost:.6f}[/bold]"
                    )

                    console.print(cost_table)

                    # Save cost summary to file
                    if storage_path:
                        try:
                            cost_file = os.path.join(storage_path, "cost_summary.md")
                            with open(cost_file, "w", encoding="utf-8") as f:
                                f.write("# Tournament Cost Summary\n\n")
                                f.write("## Per-Model Costs\n\n")

                                for model_id, cost in sorted(model_costs.items()):
                                    if model_id == "evaluation":
                                        display_model = "Evaluation"
                                    else:
                                        display_model = (
                                            model_id.split(":")[-1] if ":" in model_id else model_id
                                        )

                                    f.write(f"- **{display_model}**: ${cost:.6f}\n")

                                f.write("\n## Grand Total\n\n")
                                f.write(f"**TOTAL COST**: ${total_cost:.6f}\n")

                            logger.info(f"Cost summary saved to {cost_file}", emoji_key="save")
                        except Exception as e:
                            logger.warning(
                                f"Could not save cost summary: {str(e)}", emoji_key="warning"
                            )
            else:
                logger.error(
                    f"Could not fetch final results: {results_data.get('error', 'Unknown error')}",
                    emoji_key="error",
                )
        elif final_status:
            logger.warning(
                f"Tournament ended with status {final_status}. Check logs or status details for more info.",
                emoji_key="warning",
            )

    except Exception as e:
        logger.error(f"Error in tournament demo: {str(e)}", emoji_key="error", exc_info=True)
        return 1

    # Display cost summary at the end
    tracker.display_summary(console)

    logger.success("Text Tournament Demo Finished", emoji_key="complete")
    console.print(
        Panel(
            "To view full essays and detailed comparisons, check the storage directory indicated in the results summary.",
            title="[bold]Next Steps[/bold]",
            border_style="dim green",
            expand=False,
        )
    )
    return 0


async def main():
    """Run the tournament demo."""
    tracker = CostTracker()  # Instantiate tracker
    try:
        # Set up gateway
        await setup_gateway()

        # Run the demo
        return await run_tournament_demo(tracker)  # Pass tracker
    except Exception as e:
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    finally:
        # Clean up
        if gateway:
            pass  # No cleanup needed for Gateway instance


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
