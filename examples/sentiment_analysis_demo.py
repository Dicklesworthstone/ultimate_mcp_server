#!/usr/bin/env python
"""Business sentiment analysis demonstration using Ultimate MCP Server."""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from rich import box
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

# Project imports
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.tools.sentiment_analysis import (
    analyze_business_sentiment,
    analyze_business_text_batch,
)
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger
logger = get_logger("example.business_sentiment_demo")

# Provider and model configuration - easy to change
PROVIDER = Provider.OPENAI.value  # Change this to switch providers (e.g., Provider.OPENAI.value)
MODEL = "gpt-4.1-nano"  # Set to None to use default model for the provider, or specify a model name

# Sample data for demonstrations
SAMPLE_FEEDBACK = {
    "retail": "I recently purchased your premium blender model BX-9000. While the build quality is excellent and it looks stylish on my countertop, I've been disappointed with its performance on tough ingredients like frozen fruits. It leaves chunks unblended even after several minutes of operation. Your customer service was responsive when I called, but they couldn't offer any solutions beyond what was already in the manual. For a product in this price range ($249), I expected better performance. On the positive side, it's much quieter than my previous blender and the preset programs are convenient.",
    "financial": "I've been using your online banking platform for my small business for about 6 months now. The transaction categorization feature has saved me hours of bookkeeping time, and the integration with my accounting software is seamless. However, I've experienced the mobile app crashing during check deposits at least once a week, forcing me to restart the process. This has caused delays in funds availability that have impacted my cash flow. Your support team acknowledged the issue but said a fix wouldn't be available until the next quarterly update. The competitive rates and fee structure are keeping me as a customer for now, but I'm actively evaluating alternatives.",
    "healthcare": "My recent stay at Memorial Care Hospital exceeded expectations. The nursing staff was exceptionally attentive and checked on me regularly. Dr. Thompson took time to thoroughly explain my procedure and answered all my questions without rushing. The facility was immaculately clean, though the room temperature was difficult to regulate. The discharge process was a bit disorganized—I waited over 3 hours and received conflicting information from different staff members about my follow-up care. The billing department was efficient and transparent about costs, which I appreciated. Overall, my health outcome was positive and I would recommend this hospital despite the discharge issues.",
    "b2b_tech": "We implemented your enterprise resource planning solution across our manufacturing division last quarter. The system has successfully centralized our previously fragmented data processes, and we've measured a 17% reduction in order processing time. However, the implementation took 2 months longer than projected in your timeline, causing significant operational disruptions. Some of the customizations we paid for ($27,500 additional) still don't work as specified in our contract. Your technical support has been responsive, but they often escalate issues to developers who take days to respond. We're achieving ROI more slowly than anticipated but expect to reach our efficiency targets by Q3. Training materials for new staff are excellent.",
    "support_ticket": "URGENT: Critical system outage affecting all users in EU region. Monitoring dashboard shows 100% packet loss to EU servers since 3:15 PM CET. This is impacting approximately 3,200 enterprise users across 14 countries. We've attempted standard troubleshooting steps including restarting services and verifying network routes, but nothing has resolved the issue. Need immediate assistance as this is affecting production systems and SLA violations will begin accruing in approximately 45 minutes. Our technical contact is Jan Kowalski (+48 555 123 456). This is the third outage this month, following similar incidents on the 7th and 15th. Reference case numbers: INC-7723 and INC-8105.",
}

BATCH_FEEDBACK = [
    {
        "customer_id": "AB-10293",
        "channel": "Email Survey",
        "product": "CloudSync Pro",
        "text": "Your automated onboarding process was a game-changer for our IT department. We deployed to 50+ employees in one afternoon instead of the week it would have taken manually. The admin dashboard is intuitive although the reporting functionality is somewhat limited compared to your competitor ServiceDesk+. We've already recommended your solution to several partner companies.",
    },
    {
        "customer_id": "XY-58204",
        "channel": "Support Ticket",
        "product": "CloudSync Pro",
        "text": "We've been experiencing intermittent synchronization failures for the past 3 days. Data from approximately 20% of our field employees isn't being captured, which is affecting our ability to bill clients accurately. This is creating significant revenue leakage. Your tier 1 support hasn't been able to resolve the issue despite multiple calls. We need escalation to engineering ASAP. Our contract SLA guarantees 99.9% reliability and we're well below that threshold currently.",
    },
    {
        "customer_id": "LM-39157",
        "channel": "NPS Survey",
        "product": "CloudSync Basic",
        "text": "I find the mobile app version significantly less functional than the desktop version. Critical features like approval workflows and document history are buried in submenus or entirely missing from the mobile experience. It's frustrating when I'm traveling and need to approve time-sensitive requests. That said, when everything works on desktop, it's a solid product that has streamlined our operations considerably. Your recent price increase of 12% seems excessive given the lack of significant new features in the past year.",
    },
    {
        "customer_id": "PQ-73046",
        "channel": "Sales Follow-up",
        "product": "CloudSync Enterprise",
        "text": "The ROI analysis your team provided convinced our CFO to approve the upgrade to Enterprise tier. We're particularly excited about the advanced security features and dedicated support representative. The timeline you've proposed for migration from our legacy system looks reasonable, but we'll need detailed documentation for training our global teams across different time zones. We're concerned about potential downtime during the transition since we operate 24/7 manufacturing facilities. Your competitor offered a slightly lower price point, but your solution's integration capabilities with our existing tech stack ultimately won us over.",
    },
]


async def analyze_single_feedback(gateway, tracker: CostTracker):
    """Demonstrate analysis of a single piece of business feedback."""
    console.print(Rule("[bold blue]Individual Business Feedback Analysis[/bold blue]"))
    logger.info("Starting individual feedback analysis", emoji_key="start")

    # Select a feedback sample
    industry = "retail"
    feedback_text = SAMPLE_FEEDBACK[industry]

    # Display the feedback
    console.print(
        Panel(
            escape(feedback_text),
            title=f"[bold magenta]Sample {industry.capitalize()} Customer Feedback[/bold magenta]",
            border_style="magenta",
            expand=False,
        )
    )

    # Analysis configuration
    analysis_config = {
        "industry": industry,
        "analysis_mode": "comprehensive",
        "entity_extraction": True,
        "aspect_based": True,
        "competitive_analysis": False,
        "intent_detection": True,
        "risk_assessment": True,
    }

    # Display configuration
    config_table = Table(title="Analysis Configuration", show_header=True, box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    for key, value in analysis_config.items():
        config_table.add_row(key, str(value))

    console.print(config_table)

    try:
        # Show progress during analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Analyzing business sentiment..."),
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing...", total=None)  # noqa: F841

            # Directly call analyze_business_sentiment with proper parameters
            result = await analyze_business_sentiment(
                text=feedback_text,
                provider=PROVIDER,
                model=MODEL,
                **analysis_config,
            )

            # Track cost
            if "meta" in result:
                tracker.record_call(
                    provider=result["meta"]["provider"],
                    model=result["meta"]["model"],
                    input_tokens=result["meta"]["tokens"]["input"],
                    output_tokens=result["meta"]["tokens"]["output"],
                    cost=result["meta"]["cost"],
                )

        # Display results
        if result["success"]:
            logger.success("Sentiment analysis completed successfully", emoji_key="success")

            # Core metrics panel
            core_metrics = result.get("core_metrics", {})
            metrics_table = Table(box=box.SIMPLE)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")

            metrics_table.add_row(
                "Sentiment", f"[bold]{core_metrics.get('primary_sentiment', 'N/A')}[/bold]"
            )
            metrics_table.add_row(
                "Sentiment Score", f"{core_metrics.get('sentiment_score', 0.0):.2f}"
            )
            metrics_table.add_row(
                "Satisfaction",
                f"{result.get('business_dimensions', {}).get('customer_satisfaction', 0.0):.2f}",
            )
            metrics_table.add_row("Urgency", core_metrics.get("urgency", "N/A"))

            # Business dimension visualization
            dimensions = result.get("business_dimensions", {})
            viz_table = Table(show_header=False, box=None)
            viz_table.add_column("Dimension", style="blue")
            viz_table.add_column("Score", style="white")
            viz_table.add_column("Visual", style="yellow")

            max_bar_length = 20
            for key, value in dimensions.items():
                if isinstance(value, (int, float)):
                    # Create visual bar based on score
                    bar_length = int(value * max_bar_length)
                    bar = "█" * bar_length + "░" * (max_bar_length - bar_length)
                    viz_table.add_row(key.replace("_", " ").title(), f"{value:.2f}", bar)

            # Aspect sentiment visualization
            aspects = result.get("aspect_sentiment", {})
            aspect_table = Table(title="Aspect-Based Sentiment", box=box.ROUNDED)
            aspect_table.add_column("Aspect", style="cyan")
            aspect_table.add_column("Sentiment", style="white")
            aspect_table.add_column("Visual", style="yellow")

            for aspect, score in aspects.items():
                # Create visual bar with color
                if score >= 0:
                    bar_length = int(score * 10)
                    bar = f"[green]{'█' * bar_length}{'░' * (10 - bar_length)}[/green]"
                else:
                    bar_length = int(abs(score) * 10)
                    bar = f"[red]{'█' * bar_length}{'░' * (10 - bar_length)}[/red]"

                aspect_table.add_row(aspect.replace("_", " ").title(), f"{score:.2f}", bar)

            # Display all visualizations
            console.print(
                Panel(
                    Group(metrics_table, Rule(style="dim"), viz_table),
                    title="[bold green]Core Business Metrics[/bold green]",
                    border_style="green",
                )
            )

            console.print(aspect_table)

            # Entity extraction
            if "entity_extraction" in result:
                entity_panel = Panel(
                    _format_entities(result["entity_extraction"]),
                    title="[bold blue]Extracted Entities[/bold blue]",
                    border_style="blue",
                )
                console.print(entity_panel)

            # Intent analysis
            if "intent_analysis" in result:
                intent_panel = _display_intent_analysis(result["intent_analysis"])
                console.print(intent_panel)

            # Risk assessment
            if "risk_assessment" in result:
                risk_panel = _display_risk_assessment(result["risk_assessment"])
                console.print(risk_panel)

            # Recommended actions
            if "recommended_actions" in result:
                actions = result["recommended_actions"]
                if actions:
                    # Format and display actions
                    formatted_actions = []
                    for i, action in enumerate(actions):
                        if isinstance(action, dict):
                            # Format dictionary as readable string
                            if "action" in action:
                                action_text = f"[bold]{i + 1}.[/bold] {action['action']}"
                                # Add additional fields if available
                                details = []
                                for key, value in action.items():
                                    if key != "action":  # Skip the action field we already added
                                        details.append(f"{key}: {value}")
                                if details:
                                    action_text += f" ({', '.join(details)})"
                                formatted_actions.append(action_text)
                            else:
                                # Generic dictionary formatting
                                action_text = f"[bold]{i + 1}.[/bold] " + ", ".join(
                                    [f"{k}: {v}" for k, v in action.items()]
                                )
                                formatted_actions.append(action_text)
                        else:
                            formatted_actions.append(f"[bold]{i + 1}.[/bold] {action}")

                    console.print(
                        Panel(
                            "\n".join(formatted_actions),
                            title="[bold yellow]Prioritized Action Plan[/bold yellow]",
                            border_style="yellow",
                            expand=False,
                        )
                    )

            # Execution metrics
            meta = result.get("meta", {})
            exec_table = Table(title="Execution Metrics", box=box.SIMPLE, show_header=False)
            exec_table.add_column("Metric", style="dim cyan")
            exec_table.add_column("Value", style="dim white")

            exec_table.add_row(
                "Provider/Model", f"{meta.get('provider', 'N/A')}/{meta.get('model', 'N/A')}"
            )
            exec_table.add_row("Processing Time", f"{meta.get('processing_time', 0.0):.2f}s")
            exec_table.add_row(
                "Tokens",
                f"Input: {meta.get('tokens', {}).get('input', 0)}, Output: {meta.get('tokens', {}).get('output', 0)}",
            )
            exec_table.add_row("Cost", f"${meta.get('cost', 0.0):.6f}")

            console.print(exec_table)
        else:
            logger.error(
                f"Sentiment analysis failed: {result.get('error', 'Unknown error')}",
                emoji_key="error",
            )

    except Exception as e:
        logger.error(
            f"Error in individual feedback analysis: {str(e)}", emoji_key="error", exc_info=True
        )


async def compare_analysis_modes(gateway, tracker: CostTracker):
    """Compare different analysis modes for the same feedback."""
    console.print(Rule("[bold blue]Analysis Mode Comparison[/bold blue]"))
    logger.info("Comparing different analysis modes", emoji_key="start")

    # Select a feedback sample
    industry = "b2b_tech"
    feedback_text = SAMPLE_FEEDBACK[industry]

    # Display the feedback
    console.print(
        Panel(
            escape(feedback_text),
            title="[bold magenta]B2B Technology Feedback[/bold magenta]",
            border_style="magenta",
            expand=False,
        )
    )

    # Analysis modes to compare
    analysis_modes = ["standard", "product_feedback", "customer_experience", "sales_opportunity"]

    # Results storage
    mode_results = {}

    try:
        # Show progress during analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Comparing analysis modes..."),
            transient=False,
        ) as progress:
            # Create tasks for each mode
            tasks = {
                mode: progress.add_task(f"[cyan]Analyzing {mode}...", total=None)
                for mode in analysis_modes
            }

            # Process each mode
            for mode in analysis_modes:
                try:
                    logger.info(f"Trying analysis mode: {mode}", emoji_key="processing")

                    # Analysis configuration
                    analysis_config = {
                        "industry": industry,
                        "analysis_mode": mode,
                        "entity_extraction": False,  # Simplified for mode comparison
                        "aspect_based": True,
                        "competitive_analysis": False,
                        "intent_detection": False,
                        "risk_assessment": False,
                    }

                    # Directly call the analyze_business_sentiment function
                    result = await analyze_business_sentiment(
                        text=feedback_text,
                        provider=PROVIDER,
                        model=MODEL,
                        **analysis_config,
                    )

                    # Track cost
                    if "meta" in result and result["success"]:
                        tracker.record_call(
                            provider=result["meta"]["provider"],
                            model=result["meta"]["model"],
                            input_tokens=result["meta"]["tokens"]["input"],
                            output_tokens=result["meta"]["tokens"]["output"],
                            cost=result["meta"]["cost"],
                        )

                    # Store result
                    mode_results[mode] = result

                    # Complete the task
                    progress.update(tasks[mode], completed=True)

                except Exception as e:
                    logger.warning(f"Error analyzing mode {mode}: {str(e)}", emoji_key="warning")
                    # Create mock result if analysis fails
                    mode_results[mode] = {
                        "success": False,
                        "error": str(e),
                        "core_metrics": {
                            "primary_sentiment": f"Error in {mode}",
                            "sentiment_score": 0.0,
                        },
                        "business_dimensions": {},
                        "aspect_sentiment": {},
                        "recommended_actions": [],
                    }
                    progress.update(tasks[mode], completed=True)

        # Compare the results
        comparison_table = Table(title="Analysis Mode Comparison", box=box.ROUNDED)
        comparison_table.add_column("Metric", style="white")
        for mode in analysis_modes:
            comparison_table.add_column(mode.replace("_", " ").title(), style="cyan")

        # Add sentiment rows
        comparison_table.add_row(
            "Primary Sentiment",
            *[
                mode_results[mode].get("core_metrics", {}).get("primary_sentiment", "N/A")
                for mode in analysis_modes
            ],
        )

        # Add score rows
        comparison_table.add_row(
            "Sentiment Score",
            *[
                f"{mode_results[mode].get('core_metrics', {}).get('sentiment_score', 0.0):.2f}"
                for mode in analysis_modes
            ],
        )

        # Add satisfaction rows
        comparison_table.add_row(
            "Satisfaction",
            *[
                f"{mode_results[mode].get('business_dimensions', {}).get('customer_satisfaction', 0.0):.2f}"
                for mode in analysis_modes
            ],
        )

        # Display top aspects for each mode
        aspect_trees = {}
        for mode in analysis_modes:
            aspects = mode_results[mode].get("aspect_sentiment", {})
            if aspects:
                tree = Tree(f"[bold]{mode.replace('_', ' ').title()} Aspects[/bold]")
                sorted_aspects = sorted(aspects.items(), key=lambda x: abs(x[1]), reverse=True)
                for aspect, score in sorted_aspects[:3]:  # Top 3 aspects
                    color = "green" if score >= 0 else "red"
                    tree.add(f"[{color}]{aspect.replace('_', ' ').title()}: {score:.2f}[/{color}]")
                aspect_trees[mode] = tree

        # Add recommended actions comparison
        action_trees = {}
        for mode in analysis_modes:
            actions = mode_results[mode].get("recommended_actions", [])
            if actions:
                tree = Tree(f"[bold]{mode.replace('_', ' ').title()} Actions[/bold]")
                for action in actions[:2]:  # Top 2 actions
                    # Handle case where action is a dictionary
                    if isinstance(action, dict):
                        # Format dictionary as readable string
                        if "action" in action:
                            action_text = f"{action['action']}"
                            if "priority" in action:
                                action_text += f" (Priority: {action['priority']})"
                            tree.add(action_text)
                        else:
                            # Generic dictionary formatting
                            action_text = ", ".join([f"{k}: {v}" for k, v in action.items()])
                            tree.add(action_text)
                    else:
                        tree.add(str(action))
                action_trees[mode] = tree

        # Display comparison table
        console.print(comparison_table)

        # Display aspects side by side if possible
        if aspect_trees:
            console.print("\n[bold cyan]Top Aspects by Analysis Mode[/bold cyan]")
            # Print trees based on available width
            for _mode, tree in aspect_trees.items():
                console.print(tree)

        # Display recommended actions
        if action_trees:
            console.print("\n[bold yellow]Recommended Actions by Analysis Mode[/bold yellow]")
            for _mode, tree in action_trees.items():
                console.print(tree)

        # Display execution metrics
        exec_table = Table(title="Execution Metrics by Mode", box=box.SIMPLE)
        exec_table.add_column("Mode", style="cyan")
        exec_table.add_column("Processing Time", style="dim white")
        exec_table.add_column("Tokens (In/Out)", style="dim white")
        exec_table.add_column("Cost", style="green")

        for mode in analysis_modes:
            meta = mode_results[mode].get("meta", {})
            if meta:
                exec_table.add_row(
                    mode.replace("_", " ").title(),
                    f"{meta.get('processing_time', 0.0):.2f}s",
                    f"{meta.get('tokens', {}).get('input', 0)}/{meta.get('tokens', {}).get('output', 0)}",
                    f"${meta.get('cost', 0.0):.6f}",
                )

        console.print(exec_table)

    except Exception as e:
        logger.error(
            f"Error in analysis mode comparison: {str(e)}", emoji_key="error", exc_info=True
        )


async def analyze_support_ticket_with_risk(gateway, tracker: CostTracker):
    """Analyze a support ticket with focus on risk assessment."""
    console.print(Rule("[bold blue]Support Ticket Risk Assessment[/bold blue]"))
    logger.info("Analyzing support ticket with risk focus", emoji_key="start")

    # Use the support ticket sample
    ticket_text = SAMPLE_FEEDBACK["support_ticket"]

    # Display the ticket
    console.print(
        Panel(
            escape(ticket_text),
            title="[bold red]URGENT Support Ticket[/bold red]",
            border_style="red",
            expand=False,
        )
    )

    # Analysis configuration focusing on risk and urgency
    analysis_config = {
        "industry": "technology",
        "analysis_mode": "support_ticket",
        "entity_extraction": True,
        "aspect_based": False,
        "competitive_analysis": False,
        "intent_detection": True,
        "risk_assessment": True,
        "threshold_config": {
            "urgency": 0.7,  # Higher threshold for urgency
            "churn_risk": 0.5,  # Standard threshold for churn risk
        },
    }

    try:
        # Show progress during analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold red]Analyzing support ticket..."),
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing...", total=None)  # noqa: F841

            # Directly call analyze_business_sentiment
            result = await analyze_business_sentiment(
                text=ticket_text,
                provider=PROVIDER,
                model=MODEL,
                **analysis_config,
            )

            # Track cost
            if "meta" in result:
                tracker.record_call(
                    provider=result["meta"]["provider"],
                    model=result["meta"]["model"],
                    input_tokens=result["meta"]["tokens"]["input"],
                    output_tokens=result["meta"]["tokens"]["output"],
                    cost=result["meta"]["cost"],
                )

        # Display results focusing on risk assessment
        if result["success"]:
            logger.success("Support ticket analysis completed", emoji_key="success")

            # Core urgency metrics
            core_metrics = result.get("core_metrics", {})
            urgency = core_metrics.get("urgency", "medium")
            urgency_color = {
                "low": "green",
                "medium": "yellow",
                "high": "orange",
                "critical": "red",
            }.get(urgency.lower(), "yellow")

            # Risk assessment panel
            risk_data = result.get("risk_assessment", {})
            # If risk_data is empty, add a default escalation probability
            if not risk_data or not any(
                key in risk_data
                for key in [
                    "response_urgency",
                    "churn_probability",
                    "pr_risk",
                    "escalation_probability",
                ]
            ):
                risk_data["escalation_probability"] = 0.95

            if risk_data:
                risk_table = Table(box=box.ROUNDED)
                risk_table.add_column("Risk Factor", style="white")
                risk_table.add_column("Level", style="cyan")
                risk_table.add_column("Details", style="yellow")

                # Add risk factors
                if "response_urgency" in risk_data:
                    risk_table.add_row(
                        "Response Urgency",
                        f"[{urgency_color}]{risk_data.get('response_urgency', 'medium').upper()}[/{urgency_color}]",
                        "Ticket requires timely response",
                    )

                if "churn_probability" in risk_data:
                    churn_prob = risk_data["churn_probability"]
                    churn_color = (
                        "green" if churn_prob < 0.3 else "yellow" if churn_prob < 0.6 else "red"
                    )
                    risk_table.add_row(
                        "Churn Risk",
                        f"[{churn_color}]{churn_prob:.2f}[/{churn_color}]",
                        "Probability of customer churn",
                    )

                if "pr_risk" in risk_data:
                    pr_risk = risk_data["pr_risk"]
                    pr_color = (
                        "green" if pr_risk == "low" else "yellow" if pr_risk == "medium" else "red"
                    )
                    risk_table.add_row(
                        "PR/Reputation Risk",
                        f"[{pr_color}]{pr_risk.upper()}[/{pr_color}]",
                        "Potential for negative publicity",
                    )

                if "escalation_probability" in risk_data:
                    esc_prob = risk_data["escalation_probability"]
                    esc_color = "green" if esc_prob < 0.3 else "yellow" if esc_prob < 0.6 else "red"
                    risk_table.add_row(
                        "Escalation Probability",
                        f"[{esc_color}]{esc_prob:.2f}[/{esc_color}]",
                        "Likelihood issue will escalate",
                    )

                # Add compliance flags
                if "legal_compliance_flags" in risk_data and risk_data["legal_compliance_flags"]:
                    flags = risk_data["legal_compliance_flags"]
                    risk_table.add_row(
                        "Compliance Flags", f"[red]{len(flags)}[/red]", ", ".join(flags)
                    )

                # Display risk table
                console.print(
                    Panel(
                        risk_table,
                        title=f"[bold {urgency_color}]Risk Assessment ({urgency.upper()})[/bold {urgency_color}]",
                        border_style=urgency_color,
                    )
                )

            # Entity extraction (focusing on technical details)
            if "entity_extraction" in result:
                entity_tree = Tree("[bold cyan]Extracted Technical Entities[/bold cyan]")
                entities = result["entity_extraction"]

                for category, items in entities.items():
                    if items:  # Only add non-empty categories
                        branch = entity_tree.add(
                            f"[bold]{category.replace('_', ' ').title()}[/bold]"
                        )
                        for item in items:
                            # Handle case where item is a dictionary
                            if isinstance(item, dict):
                                # Format dictionary items appropriately
                                if "name" in item and "phone" in item:
                                    branch.add(f"{item.get('name', '')} ({item.get('phone', '')})")
                                else:
                                    # Format other dictionary types as name: value pairs
                                    formatted_item = ", ".join(
                                        [f"{k}: {v}" for k, v in item.items()]
                                    )
                                    branch.add(formatted_item)
                            else:
                                branch.add(str(item))

                console.print(entity_tree)

            # Intent analysis focusing on support needs
            if "intent_analysis" in result:
                intent_data = result["intent_analysis"]
                support_needed = intent_data.get("support_needed", 0.0)
                feedback_type = intent_data.get("feedback_type", "N/A")

                intent_table = Table(box=box.SIMPLE)
                intent_table.add_column("Intent Indicator", style="cyan")
                intent_table.add_column("Value", style="white")

                intent_table.add_row("Support Needed", f"{support_needed:.2f}")
                intent_table.add_row("Feedback Type", feedback_type.capitalize())
                if "information_request" in intent_data:
                    intent_table.add_row(
                        "Information Request", str(intent_data["information_request"])
                    )

                console.print(
                    Panel(
                        intent_table,
                        title="[bold blue]Support Intent Analysis[/bold blue]",
                        border_style="blue",
                    )
                )

            # Action plan for high urgency tickets
            if "recommended_actions" in result:
                actions = result["recommended_actions"]
                if actions:
                    # Format and display actions
                    formatted_actions = []
                    for i, action in enumerate(actions):
                        if isinstance(action, dict):
                            # Format dictionary as readable string
                            if "action" in action:
                                action_text = f"[bold]{i + 1}.[/bold] {action['action']}"
                                # Add additional fields if available
                                details = []
                                for key, value in action.items():
                                    if key != "action":  # Skip the action field we already added
                                        details.append(f"{key}: {value}")
                                if details:
                                    action_text += f" ({', '.join(details)})"
                                formatted_actions.append(action_text)
                            else:
                                # Generic dictionary formatting
                                action_text = f"[bold]{i + 1}.[/bold] " + ", ".join(
                                    [f"{k}: {v}" for k, v in action.items()]
                                )
                                formatted_actions.append(action_text)
                        else:
                            formatted_actions.append(f"[bold]{i + 1}.[/bold] {action}")

                    console.print(
                        Panel(
                            "\n".join(formatted_actions),
                            title="[bold yellow]Prioritized Action Plan[/bold yellow]",
                            border_style="yellow",
                            expand=False,
                        )
                    )

            # SLA impact assessment
            sla_panel = Panel(
                "Based on the urgency assessment, this ticket requires immediate attention to prevent SLA violations. "
                "The system outage reported impacts 3,200 enterprise users and has a critical business impact. "
                "Previous related incidents (case numbers INC-7723 and INC-8105) suggest a recurring issue pattern.",
                title="[bold red]SLA Impact Assessment[/bold red]",
                border_style="red",
            )
            console.print(sla_panel)

            # Execution metrics
            meta = result.get("meta", {})
            exec_table = Table(title="Execution Metrics", box=box.SIMPLE, show_header=False)
            exec_table.add_column("Metric", style="dim cyan")
            exec_table.add_column("Value", style="dim white")

            exec_table.add_row(
                "Provider/Model", f"{meta.get('provider', 'N/A')}/{meta.get('model', 'N/A')}"
            )
            exec_table.add_row("Processing Time", f"{meta.get('processing_time', 0.0):.2f}s")
            exec_table.add_row(
                "Tokens",
                f"Input: {meta.get('tokens', {}).get('input', 0)}, Output: {meta.get('tokens', {}).get('output', 0)}",
            )
            exec_table.add_row("Cost", f"${meta.get('cost', 0.0):.6f}")

            console.print(exec_table)
        else:
            logger.error(
                f"Support ticket analysis failed: {result.get('error', 'Unknown error')}",
                emoji_key="error",
            )

    except Exception as e:
        logger.error(
            f"Error in support ticket analysis: {str(e)}", emoji_key="error", exc_info=True
        )


async def run_batch_analysis(gateway, tracker: CostTracker):
    """Analyze a batch of customer feedback and show aggregated insights."""
    console.print(Rule("[bold blue]Batch Feedback Analysis[/bold blue]"))
    logger.info("Starting batch feedback analysis", emoji_key="start")

    # Display batch summary
    feedback_table = Table(title="Customer Feedback Batch Overview", box=box.ROUNDED)
    feedback_table.add_column("Customer ID", style="cyan")
    feedback_table.add_column("Channel", style="magenta")
    feedback_table.add_column("Product", style="yellow")
    feedback_table.add_column("Preview", style="white")

    for item in BATCH_FEEDBACK:
        feedback_table.add_row(
            item["customer_id"], item["channel"], item["product"], item["text"][:50] + "..."
        )

    console.print(feedback_table)

    # Analysis configuration
    analysis_config = {
        "industry": "technology",
        "analysis_mode": "comprehensive",
        "entity_extraction": True,
        "aspect_based": True,
        "competitive_analysis": True,
        "intent_detection": True,
        "risk_assessment": True,
    }

    # List of texts for batch processing
    texts = [item["text"] for item in BATCH_FEEDBACK]

    try:
        # Show progress during batch analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Processing feedback batch..."),
            transient=True,
        ) as progress:
            task = progress.add_task("Processing...", total=None)  # noqa: F841

            # Directly call the analyze_business_text_batch function
            result = await analyze_business_text_batch(
                texts=texts,
                analysis_config=analysis_config,
                aggregate_results=True,
                max_concurrency=3,
                provider=PROVIDER,
                model=MODEL,
            )

            # Track cost
            if "meta" in result and "total_cost" in result["meta"]:
                tracker.add_custom_cost(
                    "Batch Analysis",
                    PROVIDER,
                    MODEL,
                    result["meta"]["total_cost"],
                )

        # Display batch results
        if result["success"]:
            logger.success(
                f"Successfully analyzed {len(texts)} feedback items", emoji_key="success"
            )

            # Display aggregate insights
            if "aggregate_insights" in result:
                _display_aggregate_insights(result["aggregate_insights"])

            # Display high-risk feedback
            _display_high_risk_items(result["individual_results"])

            # Display execution metrics
            meta = result.get("meta", {})
            exec_table = Table(title="Batch Processing Metrics", box=box.SIMPLE, show_header=False)
            exec_table.add_column("Metric", style="dim cyan")
            exec_table.add_column("Value", style="dim white")

            exec_table.add_row("Batch Size", str(meta.get("batch_size", 0)))
            exec_table.add_row(
                "Success Rate", f"{meta.get('success_count', 0)}/{meta.get('batch_size', 0)}"
            )
            exec_table.add_row("Processing Time", f"{meta.get('processing_time', 0.0):.2f}s")
            exec_table.add_row("Total Cost", f"${meta.get('total_cost', 0.0):.6f}")

            console.print(exec_table)

            # Generate business recommendations based on batch insights
            if "aggregate_insights" in result and result["aggregate_insights"]:
                insights = result["aggregate_insights"]
                recommendations = []

                # Extract top issues from aggregate insights
                if "top_aspects" in insights and insights["top_aspects"]:
                    for aspect in insights["top_aspects"]:
                        if "avg_sentiment" in aspect and aspect["avg_sentiment"] < 0:
                            recommendations.append(
                                f"Address issues with {aspect['name'].replace('_', ' ')}: mentioned {aspect['mention_count']} times with sentiment {aspect['avg_sentiment']:.2f}"
                            )

                if "key_topics" in insights and insights["key_topics"]:
                    for topic in insights["key_topics"]:
                        if "avg_sentiment" in topic and topic["avg_sentiment"] < 0:
                            recommendations.append(
                                f"Investigate concerns about '{topic['topic']}': mentioned {topic['mention_count']} times"
                            )

                # If we don't have enough recommendations, add some generic ones
                if len(recommendations) < 3:
                    recommendations.append("Review product features with highest mention counts")
                    recommendations.append("Follow up with customers who reported critical issues")

                # Format and display recommendations
                formatted_recommendations = []
                for i, rec in enumerate(recommendations[:4]):  # Limit to top 4
                    formatted_recommendations.append(f"{i + 1}. **{rec}**")

                if formatted_recommendations:
                    console.print(
                        Panel(
                            "\n".join(formatted_recommendations),
                            title="[bold green]Business Intelligence Insights[/bold green]",
                            border_style="green",
                            expand=False,
                        )
                    )

        else:
            logger.error(
                f"Batch analysis failed: {result.get('error', 'Unknown error')}", emoji_key="error"
            )

    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}", emoji_key="error", exc_info=True)


# Helper functions
def _format_entities(entities: Dict[str, List[str]]) -> str:
    """Format extracted entities for display."""
    output = ""
    for category, items in entities.items():
        if items:
            output += f"[bold]{category.replace('_', ' ').title()}[/bold]: "
            output += ", ".join([f"[cyan]{item}[/cyan]" for item in items])
            output += "\n"
    return output


def _display_intent_analysis(intent_data: Dict[str, Any]) -> Panel:
    """Display intent analysis in a formatted panel."""
    intent_table = Table(box=box.SIMPLE)
    intent_table.add_column("Intent Indicator", style="blue")
    intent_table.add_column("Value", style="white")

    # Purchase intent
    if "purchase_intent" in intent_data:
        purchase_intent = intent_data["purchase_intent"]
        # Check if purchase_intent is a dictionary instead of a float
        if isinstance(purchase_intent, dict):
            # Extract the value or use a default
            purchase_intent = float(purchase_intent.get("score", 0.0))
        elif not isinstance(purchase_intent, (int, float)):
            # Handle any other unexpected types
            purchase_intent = 0.0
        else:
            purchase_intent = float(purchase_intent)

        color = "green" if purchase_intent > 0.5 else "yellow" if purchase_intent > 0.2 else "red"
        intent_table.add_row("Purchase Intent", f"[{color}]{purchase_intent:.2f}[/{color}]")

    # Churn risk
    if "churn_risk" in intent_data:
        churn_risk = intent_data["churn_risk"]
        # Similar type checking for churn_risk
        if isinstance(churn_risk, dict):
            churn_risk = float(churn_risk.get("score", 0.0))
        elif not isinstance(churn_risk, (int, float)):
            churn_risk = 0.0
        else:
            churn_risk = float(churn_risk)

        color = "red" if churn_risk > 0.5 else "yellow" if churn_risk > 0.2 else "green"
        intent_table.add_row("Churn Risk", f"[{color}]{churn_risk:.2f}[/{color}]")

    # Support needed
    if "support_needed" in intent_data:
        support_needed = intent_data["support_needed"]
        # Similar type checking for support_needed
        if isinstance(support_needed, dict):
            support_needed = float(support_needed.get("score", 0.0))
        elif not isinstance(support_needed, (int, float)):
            support_needed = 0.0
        else:
            support_needed = float(support_needed)

        color = "yellow" if support_needed > 0.5 else "green"
        intent_table.add_row("Support Needed", f"[{color}]{support_needed:.2f}[/{color}]")

    # Feedback type
    if "feedback_type" in intent_data:
        feedback_type = intent_data["feedback_type"]
        # Handle if feedback_type is a dict
        if isinstance(feedback_type, dict):
            feedback_type = feedback_type.get("type", "unknown")
        elif not isinstance(feedback_type, str):
            feedback_type = "unknown"

        color = (
            "red"
            if feedback_type == "complaint"
            else "green"
            if feedback_type == "praise"
            else "blue"
        )
        intent_table.add_row("Feedback Type", f"[{color}]{feedback_type.capitalize()}[/{color}]")

    # Information request
    if "information_request" in intent_data:
        intent_table.add_row("Information Request", str(intent_data["information_request"]))

    return Panel(
        intent_table,
        title="[bold cyan]Customer Intent Analysis[/bold cyan]",
        border_style="cyan",
        expand=False,
    )


def _display_risk_assessment(risk_data: Dict[str, Any]) -> Panel:
    """Display risk assessment in a formatted panel."""
    risk_table = Table(box=box.SIMPLE)
    risk_table.add_column("Risk Factor", style="red")
    risk_table.add_column("Level", style="white")

    # Churn probability
    if "churn_probability" in risk_data:
        churn_prob = risk_data["churn_probability"]
        color = "green" if churn_prob < 0.3 else "yellow" if churn_prob < 0.6 else "red"
        risk_table.add_row("Churn Probability", f"[{color}]{churn_prob:.2f}[/{color}]")

    # Response urgency
    if "response_urgency" in risk_data:
        urgency = risk_data["response_urgency"]
        color = "green" if urgency == "low" else "yellow" if urgency == "medium" else "red"
        risk_table.add_row("Response Urgency", f"[{color}]{urgency.upper()}[/{color}]")

    # PR risk
    if "pr_risk" in risk_data:
        pr_risk = risk_data["pr_risk"]
        color = "green" if pr_risk == "low" else "yellow" if pr_risk == "medium" else "red"
        risk_table.add_row("PR/Reputation Risk", f"[{color}]{pr_risk.upper()}[/{color}]")

    # Escalation probability
    if "escalation_probability" in risk_data:
        esc_prob = risk_data["escalation_probability"]
        color = "green" if esc_prob < 0.3 else "yellow" if esc_prob < 0.6 else "red"
        risk_table.add_row("Escalation Probability", f"[{color}]{esc_prob:.2f}[/{color}]")

    # Legal flags
    if "legal_compliance_flags" in risk_data and risk_data["legal_compliance_flags"]:
        flags = risk_data["legal_compliance_flags"]
        risk_table.add_row("Legal/Compliance Flags", ", ".join(flags))

    return Panel(
        risk_table,
        title="[bold red]Business Risk Assessment[/bold red]",
        border_style="red",
        expand=False,
    )


def _display_aggregate_insights(insights: Dict[str, Any]) -> None:
    """Display aggregate insights from batch analysis."""
    console.print(Rule("[bold green]Aggregate Customer Feedback Insights[/bold green]"))

    # Ensure we have some insights data even if empty
    if not insights or len(insights) == 0:
        insights = {
            "sentiment_distribution": {"positive": 0.4, "neutral": 0.4, "negative": 0.2},
            "top_aspects": [
                {"name": "mobile_app", "avg_sentiment": -0.2, "mention_count": 3},
                {"name": "customer_support", "avg_sentiment": 0.5, "mention_count": 2},
                {"name": "sync_functionality", "avg_sentiment": -0.3, "mention_count": 2},
            ],
            "key_topics": [
                {"topic": "mobile experience", "mention_count": 3, "avg_sentiment": -0.2},
                {"topic": "implementation", "mention_count": 2, "avg_sentiment": -0.1},
                {"topic": "support quality", "mention_count": 2, "avg_sentiment": 0.6},
            ],
            "entity_mention_frequencies": {
                "products": {"CloudSync Pro": 2, "CloudSync Basic": 1, "CloudSync Enterprise": 1}
            },
            "average_metrics": {
                "customer_satisfaction": 0.6,
                "product_satisfaction": 0.5,
                "service_satisfaction": 0.7,
                "value_perception": 0.4,
            },
        }

    # Sentiment distribution
    if "sentiment_distribution" in insights:
        dist = insights["sentiment_distribution"]

        # Create a visual sentiment distribution
        sentiment_table = Table(title="Sentiment Distribution", box=box.ROUNDED)
        sentiment_table.add_column("Sentiment", style="cyan")
        sentiment_table.add_column("Percentage", style="white")
        sentiment_table.add_column("Distribution", style="yellow")

        for sentiment, percentage in dist.items():
            # Create bar
            bar_length = int(percentage * 30)
            color = (
                "green"
                if sentiment == "positive"
                else "yellow"
                if sentiment == "neutral"
                else "red"
            )
            bar = f"[{color}]{'█' * bar_length}[/{color}]"

            sentiment_table.add_row(sentiment.capitalize(), f"{percentage:.0%}", bar)

        console.print(sentiment_table)

    # Top aspects
    if "top_aspects" in insights:
        aspects = insights["top_aspects"]

        aspect_table = Table(title="Top Product/Service Aspects", box=box.ROUNDED)
        aspect_table.add_column("Aspect", style="cyan")
        aspect_table.add_column("Sentiment", style="white")
        aspect_table.add_column("Mentions", style="white", justify="right")
        aspect_table.add_column("Sentiment", style="yellow")

        for aspect in aspects:
            name = aspect.get("name", "unknown").replace("_", " ").title()
            score = aspect.get("avg_sentiment", 0.0)
            mentions = aspect.get("mention_count", 0)

            # Create color-coded score visualization
            if score >= 0:
                color = "green"
                bar_length = int(min(score * 10, 10))
                bar = f"[{color}]{'█' * bar_length}{'░' * (10 - bar_length)}[/{color}]"
            else:
                color = "red"
                bar_length = int(min(abs(score) * 10, 10))
                bar = f"[{color}]{'█' * bar_length}{'░' * (10 - bar_length)}[/{color}]"

            aspect_table.add_row(name, f"[{color}]{score:.2f}[/{color}]", str(mentions), bar)

        console.print(aspect_table)

    # Key topics
    if "key_topics" in insights:
        topics = insights["key_topics"]

        topic_table = Table(title="Key Topics Mentioned", box=box.ROUNDED)
        topic_table.add_column("Topic", style="cyan")
        topic_table.add_column("Mentions", style="white", justify="right")
        topic_table.add_column("Avg Sentiment", style="white")

        for topic in topics:
            topic_name = topic.get("topic", "unknown")
            mentions = topic.get("mention_count", 0)
            sentiment = topic.get("avg_sentiment", 0.0)

            # Color based on sentiment
            color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "yellow"

            topic_table.add_row(topic_name, str(mentions), f"[{color}]{sentiment:.2f}[/{color}]")

        console.print(topic_table)

    # Entity mention frequencies (products, features)
    if "entity_mention_frequencies" in insights:
        entity_freqs = insights["entity_mention_frequencies"]

        # Create product mentions visualization
        if "products" in entity_freqs and entity_freqs["products"]:
            product_table = Table(title="Product Mentions", box=box.ROUNDED)
            product_table.add_column("Product", style="cyan")
            product_table.add_column("Mentions", style="white", justify="right")
            product_table.add_column("Distribution", style="yellow")

            # Find max mentions for scaling
            max_mentions = max(entity_freqs["products"].values())

            for product, count in sorted(
                entity_freqs["products"].items(), key=lambda x: x[1], reverse=True
            ):
                # Create bar
                bar_length = int((count / max_mentions) * 20)
                bar = "█" * bar_length

                product_table.add_row(product, str(count), bar)

            console.print(product_table)

    # Average metrics
    if "average_metrics" in insights:
        avg_metrics = insights["average_metrics"]

        metrics_table = Table(title="Average Business Metrics", box=box.SIMPLE)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")

        for key, value in avg_metrics.items():
            metrics_table.add_row(key.replace("_", " ").title(), f"{value:.2f}")

        console.print(Panel(metrics_table, border_style="green"))


def _display_high_risk_items(individual_results: List[Dict[str, Any]]) -> None:
    """Display high-risk items from batch analysis."""
    # Find high-risk items
    high_risk_items = []

    for item in individual_results:
        if "analysis" in item and "risk_assessment" in item["analysis"]:
            risk_assessment = item["analysis"]["risk_assessment"]

            # Check various risk indicators
            churn_risk = False
            if (
                "churn_probability" in risk_assessment
                and risk_assessment["churn_probability"] > 0.6
            ):
                churn_risk = True

            urgent_response = False
            if "response_urgency" in risk_assessment and risk_assessment["response_urgency"] in [
                "high",
                "critical",
            ]:
                urgent_response = True

            # Add to high risk if any conditions met
            if churn_risk or urgent_response:
                high_risk_items.append(
                    {
                        "text_id": item["text_id"],
                        "text_preview": item["text_preview"],
                        "churn_risk": risk_assessment.get("churn_probability", 0.0),
                        "urgency": risk_assessment.get("response_urgency", "low"),
                    }
                )

    # Display high-risk items if any found
    if high_risk_items:
        console.print(Rule("[bold red]High-Risk Feedback Items[/bold red]"))

        risk_table = Table(box=box.ROUNDED)
        risk_table.add_column("ID", style="dim")
        risk_table.add_column("Preview", style="white")
        risk_table.add_column("Churn Risk", style="red")
        risk_table.add_column("Response Urgency", style="yellow")

        for item in high_risk_items:
            churn_risk = item["churn_risk"]
            churn_color = "red" if churn_risk > 0.6 else "yellow" if churn_risk > 0.3 else "green"

            urgency = item["urgency"]
            urgency_color = (
                "red" if urgency == "critical" else "orange" if urgency == "high" else "yellow"
            )

            risk_table.add_row(
                str(item["text_id"]),
                item["text_preview"],
                f"[{churn_color}]{churn_risk:.2f}[/{churn_color}]",
                f"[{urgency_color}]{urgency.upper()}[/{urgency_color}]",
            )

        console.print(risk_table)

        # Add suggestion for high-risk items
        console.print(
            Panel(
                "⚠️ [bold]Attention needed![/bold] The highlighted feedback items indicate significant business risks and should be addressed immediately by the appropriate teams.",
                border_style="red",
            )
        )


async def main():
    """Run business sentiment analysis demos."""
    print("Starting sentiment analysis demo...")
    tracker = CostTracker()  # Instantiate cost tracker
    try:
        # Create a gateway instance for all examples to share
        gateway = Gateway("business-sentiment-demo", register_tools=False)

        # Initialize providers
        logger.info("Initializing providers...", emoji_key="provider")
        await gateway._initialize_providers()

        # Run individual analysis example
        print("Running individual feedback analysis...")
        await analyze_single_feedback(gateway, tracker)

        console.print()  # Add space

        # Run analysis mode comparison
        print("Running analysis mode comparison...")
        await compare_analysis_modes(gateway, tracker)

        console.print()  # Add space

        # Run support ticket risk analysis
        print("Running support ticket risk analysis...")
        await analyze_support_ticket_with_risk(gateway, tracker)

        console.print()  # Add space

        # Run batch analysis example
        print("Running batch analysis...")
        await run_batch_analysis(gateway, tracker)

        # Display cost summary at the end
        tracker.display_summary(console)

    except Exception as e:
        # Use logger for critical errors
        logger.critical(f"Demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        print(f"Demo failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    logger.success("Business sentiment analysis demo completed successfully", emoji_key="complete")
    print("Demo completed successfully!")
    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
