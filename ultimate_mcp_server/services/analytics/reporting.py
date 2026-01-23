"""Reporting and visualization for Ultimate MCP Server analytics."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ultimate_mcp_server.services.analytics.metrics import get_metrics_tracker
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class AnalyticsReporting:
    """Provides comprehensive reporting and visualization capabilities for Ultimate MCP Server analytics.

    This class offers tools to generate detailed usage, cost, and provider-specific reports
    in various formats (JSON, HTML, Markdown) with optional data visualizations. It serves
    as the primary interface for extracting actionable insights from the server's operational
    metrics and presenting them in human-readable formats.

    Features:
    - Multiple report types: usage reports, provider-specific analysis, and cost breakdowns
    - Multiple output formats: JSON, HTML, and Markdown
    - Optional data visualizations using matplotlib (when available)
    - Customizable reporting periods
    - Persistent report storage

    The reporting system uses the metrics tracked by the MetricsTracker to generate these
    reports, providing insights into token usage, costs, request patterns, cache efficiency,
    and provider/model distribution.

    Usage:
        # Create a reporting instance
        reporter = AnalyticsReporting()

        # Generate a usage report for the last 7 days
        report_path = reporter.generate_usage_report(days=7, output_format="html")

        # Generate a cost analysis report for the last month
        cost_report = reporter.generate_cost_report(days=30, output_format="json")

        # Generate a provider-specific report
        provider_report = reporter.generate_provider_report(
            provider="anthropic",
            days=14,
            output_format="markdown"
        )
    """

    def __init__(self, reports_dir: Optional[Union[str, Path]] = None, include_plots: bool = True):
        """Initialize the analytics reporting.

        Args:
            reports_dir: Directory for reports storage
            include_plots: Whether to include plots in reports
        """
        # Set reports directory
        if reports_dir:
            self.reports_dir = Path(reports_dir)
        else:
            self.reports_dir = Path.home() / ".ultimate" / "reports"

        # Create reports directory if it doesn't exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Plotting settings
        self.include_plots = include_plots and PLOTTING_AVAILABLE

        # Get metrics tracker
        self.metrics = get_metrics_tracker()

        logger.info(
            f"Analytics reporting initialized (dir: {self.reports_dir}, plots: {self.include_plots})",
            emoji_key="analytics",
        )

    def generate_usage_report(
        self, days: int = 7, output_format: str = "json", include_plots: Optional[bool] = None
    ) -> Union[Dict[str, Any], str, Path]:
        """Generate a usage report.

        Args:
            days: Number of days to include in the report
            output_format: Output format (json, html, markdown)
            include_plots: Whether to include plots (overrides default setting)

        Returns:
            Report data or path to report file
        """
        # Get metrics
        metrics = self.metrics.get_stats()

        # Determine plotting
        do_plots = self.include_plots if include_plots is None else include_plots
        do_plots = do_plots and PLOTTING_AVAILABLE

        # Build report data
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "period": f"{days} days",
            "general": metrics["general"],
            "cache": metrics["cache"],
            "top_providers": metrics["top_providers"],
            "top_models": metrics["top_models"],
            "daily_usage": [
                day
                for day in metrics["daily_usage"]
                if (datetime.now() - datetime.strptime(day["date"], "%Y-%m-%d")).days < days
            ],
        }

        # Generate report based on format
        if output_format == "json":
            # JSON format
            report_path = (
                self.reports_dir / f"usage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Generated JSON usage report: {report_path}", emoji_key="analytics")

            return report_path

        elif output_format == "html":
            # HTML format (with optional plots)
            report_path = (
                self.reports_dir / f"usage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )

            # Generate plots if requested
            plot_paths = []
            if do_plots:
                plot_paths = self._generate_report_plots(report_data, days)

            # Generate HTML
            html = self._generate_html_report(report_data, plot_paths)

            with open(report_path, "w") as f:
                f.write(html)

            logger.info(f"Generated HTML usage report: {report_path}", emoji_key="analytics")

            return report_path

        elif output_format == "markdown":
            # Markdown format
            report_path = (
                self.reports_dir / f"usage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )

            # Generate plots if requested
            plot_paths = []
            if do_plots:
                plot_paths = self._generate_report_plots(report_data, days)

            # Generate Markdown
            markdown = self._generate_markdown_report(report_data, plot_paths)

            with open(report_path, "w") as f:
                f.write(markdown)

            logger.info(f"Generated Markdown usage report: {report_path}", emoji_key="analytics")

            return report_path

        else:
            # Default to raw data
            logger.warning(
                f"Unknown output format: {output_format}, returning raw data", emoji_key="warning"
            )
            return report_data

    def generate_provider_report(
        self,
        provider: str,
        days: int = 7,
        output_format: str = "json",
        include_plots: Optional[bool] = None,
    ) -> Union[Dict[str, Any], str, Path]:
        """Generate a provider-specific report.

        Args:
            provider: Provider name
            days: Number of days to include in the report
            output_format: Output format (json, html, markdown)
            include_plots: Whether to include plots (overrides default setting)

        Returns:
            Report data or path to report file
        """
        # Get metrics
        metrics = self.metrics.get_stats()

        # Check if provider exists
        if provider not in metrics["providers"]:
            logger.error(f"Unknown provider: {provider}", emoji_key="error")
            return {"error": f"Unknown provider: {provider}"}

        # Determine plotting
        do_plots = self.include_plots if include_plots is None else include_plots
        do_plots = do_plots and PLOTTING_AVAILABLE

        # Extract provider-specific data
        provider_data = metrics["providers"][provider]
        provider_models = {
            model: data
            for model, data in metrics["models"].items()
            if model.startswith(provider) or model.lower().startswith(provider.lower())
        }

        # Collect daily usage for this provider (approximate)
        provider_share = (
            provider_data["tokens"] / metrics["general"]["tokens_total"]
            if metrics["general"]["tokens_total"] > 0
            else 0
        )
        provider_daily = [
            {
                "date": day["date"],
                "tokens": int(day["tokens"] * provider_share),  # Approximate
                "cost": day["cost"] * provider_share,  # Approximate
            }
            for day in metrics["daily_usage"]
            if (datetime.now() - datetime.strptime(day["date"], "%Y-%m-%d")).days < days
        ]

        # Build report data
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "period": f"{days} days",
            "provider": provider,
            "stats": provider_data,
            "models": provider_models,
            "daily_usage": provider_daily,
            "percentage_of_total": provider_share * 100,
        }

        # Generate report based on format
        if output_format == "json":
            # JSON format
            report_path = (
                self.reports_dir
                / f"{provider}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Generated JSON provider report: {report_path}", emoji_key="analytics")

            return report_path

        elif output_format == "html":
            # HTML format (with optional plots)
            report_path = (
                self.reports_dir
                / f"{provider}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )

            # Generate plots if requested
            plot_paths = []
            if do_plots:
                plot_paths = self._generate_provider_plots(report_data, provider, days)

            # Generate HTML
            html = self._generate_html_provider_report(report_data, plot_paths)

            with open(report_path, "w") as f:
                f.write(html)

            logger.info(f"Generated HTML provider report: {report_path}", emoji_key="analytics")

            return report_path

        elif output_format == "markdown":
            # Markdown format
            report_path = (
                self.reports_dir
                / f"{provider}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )

            # Generate plots if requested
            plot_paths = []
            if do_plots:
                plot_paths = self._generate_provider_plots(report_data, provider, days)

            # Generate Markdown
            markdown = self._generate_markdown_provider_report(report_data, plot_paths)

            with open(report_path, "w") as f:
                f.write(markdown)

            logger.info(f"Generated Markdown provider report: {report_path}", emoji_key="analytics")

            return report_path

        else:
            # Default to raw data
            logger.warning(
                f"Unknown output format: {output_format}, returning raw data", emoji_key="warning"
            )
            return report_data

    def generate_cost_report(
        self, days: int = 30, output_format: str = "json", include_plots: Optional[bool] = None
    ) -> Union[Dict[str, Any], str, Path]:
        """Generate a cost analysis report.

        Args:
            days: Number of days to include in the report
            output_format: Output format (json, html, markdown)
            include_plots: Whether to include plots (overrides default setting)

        Returns:
            Report data or path to report file
        """
        # Get metrics
        metrics = self.metrics.get_stats()

        # Determine plotting
        do_plots = self.include_plots if include_plots is None else include_plots
        do_plots = do_plots and PLOTTING_AVAILABLE

        # Process daily cost data
        daily_costs = [
            {
                "date": day["date"],
                "cost": day["cost"],
            }
            for day in metrics["daily_usage"]
            if (datetime.now() - datetime.strptime(day["date"], "%Y-%m-%d")).days < days
        ]

        # Calculate cost by provider
        provider_costs = [
            {
                "provider": provider,
                "cost": data["cost"],
                "percentage": data["cost"] / metrics["general"]["cost_total"] * 100
                if metrics["general"]["cost_total"] > 0
                else 0,
            }
            for provider, data in metrics["providers"].items()
        ]
        provider_costs.sort(key=lambda x: x["cost"], reverse=True)

        # Calculate cost by model
        model_costs = [
            {
                "model": model,
                "cost": data["cost"],
                "percentage": data["cost"] / metrics["general"]["cost_total"] * 100
                if metrics["general"]["cost_total"] > 0
                else 0,
            }
            for model, data in metrics["models"].items()
        ]
        model_costs.sort(key=lambda x: x["cost"], reverse=True)

        # Calculate cost efficiency (tokens per dollar)
        cost_efficiency = [
            {
                "model": model,
                "tokens_per_dollar": data["tokens"] / data["cost"] if data["cost"] > 0 else 0,
                "tokens": data["tokens"],
                "cost": data["cost"],
            }
            for model, data in metrics["models"].items()
            if data["cost"] > 0
        ]
        cost_efficiency.sort(key=lambda x: x["tokens_per_dollar"], reverse=True)

        # Build report data
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "period": f"{days} days",
            "total_cost": metrics["general"]["cost_total"],
            "cache_savings": metrics["cache"]["saved_cost"],
            "daily_costs": daily_costs,
            "provider_costs": provider_costs,
            "model_costs": model_costs,
            "cost_efficiency": cost_efficiency,
        }

        # Generate report based on format
        if output_format == "json":
            # JSON format
            report_path = (
                self.reports_dir / f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Generated JSON cost report: {report_path}", emoji_key="analytics")

            return report_path

        elif output_format == "html":
            # HTML format (with optional plots)
            report_path = (
                self.reports_dir / f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )

            # Generate plots if requested
            plot_paths = []
            if do_plots:
                plot_paths = self._generate_cost_plots(report_data, days)

            # Generate HTML
            html = self._generate_html_cost_report(report_data, plot_paths)

            with open(report_path, "w") as f:
                f.write(html)

            logger.info(f"Generated HTML cost report: {report_path}", emoji_key="analytics")

            return report_path

        elif output_format == "markdown":
            # Markdown format
            report_path = (
                self.reports_dir / f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )

            # Generate plots if requested
            plot_paths = []
            if do_plots:
                plot_paths = self._generate_cost_plots(report_data, days)

            # Generate Markdown
            markdown = self._generate_markdown_cost_report(report_data, plot_paths)

            with open(report_path, "w") as f:
                f.write(markdown)

            logger.info(f"Generated Markdown cost report: {report_path}", emoji_key="analytics")

            return report_path

        else:
            # Default to raw data
            logger.warning(
                f"Unknown output format: {output_format}, returning raw data", emoji_key="warning"
            )
            return report_data

    def _generate_report_plots(self, report_data: Dict[str, Any], days: int) -> List[str]:
        """Generate plots for a usage report.

        Args:
            report_data: Report data
            days: Number of days to include

        Returns:
            List of plot file paths
        """
        if not PLOTTING_AVAILABLE:
            return []

        plot_paths = []

        # Create daily usage plot
        if report_data["daily_usage"]:
            try:
                # Prepare data
                df = pd.DataFrame(report_data["daily_usage"])
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

                # Create plot directory
                plot_dir = self.reports_dir / "plots"
                plot_dir.mkdir(exist_ok=True)

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(df["date"], df["tokens"], marker="o", linestyle="-", linewidth=2)
                plt.title(f"Daily Token Usage (Last {days} Days)")
                plt.xlabel("Date")
                plt.ylabel("Tokens")
                plt.grid(True, alpha=0.3)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 7)))
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Save plot
                plot_path = str(
                    plot_dir / f"daily_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path)
                plt.close()

                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(f"Failed to generate daily usage plot: {str(e)}", emoji_key="error")

        # Create provider distribution plot
        if report_data["top_providers"]:
            try:
                # Prepare data
                providers = [p["provider"] for p in report_data["top_providers"]]
                percentages = [p["percentage"] * 100 for p in report_data["top_providers"]]

                # Create plot
                plt.figure(figsize=(8, 8))
                plt.pie(
                    percentages, labels=providers, autopct="%1.1f%%", startangle=90, shadow=True
                )
                plt.axis("equal")
                plt.title("Token Usage by Provider")
                plt.tight_layout()

                # Save plot
                plot_path = str(
                    plot_dir
                    / f"provider_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path)
                plt.close()

                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(
                    f"Failed to generate provider distribution plot: {str(e)}", emoji_key="error"
                )

        # Create model distribution plot
        if report_data["top_models"]:
            try:
                # Prepare data
                models = [m["model"] for m in report_data["top_models"]]
                percentages = [m["percentage"] * 100 for m in report_data["top_models"]]

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.bar(models, percentages)
                plt.title("Token Usage by Model")
                plt.xlabel("Model")
                plt.ylabel("Percentage of Total Tokens")
                plt.xticks(rotation=45, ha="right")
                plt.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()

                # Save plot
                plot_path = str(
                    plot_dir / f"model_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path)
                plt.close()

                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(
                    f"Failed to generate model distribution plot: {str(e)}", emoji_key="error"
                )

        return plot_paths

    def _generate_provider_plots(
        self, report_data: Dict[str, Any], provider: str, days: int
    ) -> List[str]:
        """Generate plots for a provider report.

        Args:
            report_data: Report data
            provider: Provider name
            days: Number of days to include

        Returns:
            List of plot file paths
        """
        if not PLOTTING_AVAILABLE:
            return []

        plot_paths = []

        # Create plot directory
        plot_dir = self.reports_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Create daily usage plot
        if report_data["daily_usage"]:
            try:
                # Prepare data
                df = pd.DataFrame(report_data["daily_usage"])
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(df["date"], df["tokens"], marker="o", linestyle="-", linewidth=2)
                plt.title(f"{provider} Daily Token Usage (Last {days} Days)")
                plt.xlabel("Date")
                plt.ylabel("Tokens")
                plt.grid(True, alpha=0.3)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 7)))
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Save plot
                plot_path = str(
                    plot_dir
                    / f"{provider}_daily_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path)
                plt.close()

                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(
                    f"Failed to generate provider daily usage plot: {str(e)}", emoji_key="error"
                )

        # Create model distribution plot
        if report_data["models"]:
            try:
                # Prepare data
                models = list(report_data["models"].keys())
                tokens = [data["tokens"] for _, data in report_data["models"].items()]

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.bar(models, tokens)
                plt.title(f"{provider} Token Usage by Model")
                plt.xlabel("Model")
                plt.ylabel("Tokens")
                plt.xticks(rotation=45, ha="right")
                plt.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()

                # Save plot
                plot_path = str(
                    plot_dir
                    / f"{provider}_model_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path)
                plt.close()

                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(
                    f"Failed to generate provider model distribution plot: {str(e)}",
                    emoji_key="error",
                )

        return plot_paths

    def _generate_cost_plots(self, report_data: Dict[str, Any], days: int) -> List[str]:
        """Generate plots for a cost report.

        Args:
            report_data: Report data
            days: Number of days to include

        Returns:
            List of plot file paths
        """
        if not PLOTTING_AVAILABLE:
            return []

        plot_paths = []

        # Create plot directory
        plot_dir = self.reports_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Create daily cost plot
        if report_data["daily_costs"]:
            try:
                # Prepare data
                df = pd.DataFrame(report_data["daily_costs"])
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(df["date"], df["cost"], marker="o", linestyle="-", linewidth=2)
                plt.title(f"Daily Cost (Last {days} Days)")
                plt.xlabel("Date")
                plt.ylabel("Cost ($)")
                plt.grid(True, alpha=0.3)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 7)))
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Save plot
                plot_path = str(
                    plot_dir / f"daily_cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path)
                plt.close()

                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(f"Failed to generate daily cost plot: {str(e)}", emoji_key="error")

        # Create provider cost distribution plot
        if report_data["provider_costs"]:
            try:
                # Prepare data
                providers = [p["provider"] for p in report_data["provider_costs"]]
                costs = [p["cost"] for p in report_data["provider_costs"]]

                # Create plot
                plt.figure(figsize=(8, 8))
                plt.pie(costs, labels=providers, autopct="%1.1f%%", startangle=90, shadow=True)
                plt.axis("equal")
                plt.title("Cost by Provider")
                plt.tight_layout()

                # Save plot
                plot_path = str(
                    plot_dir / f"provider_cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path)
                plt.close()

                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(
                    f"Failed to generate provider cost distribution plot: {str(e)}",
                    emoji_key="error",
                )

        # Create cost efficiency plot
        if report_data["cost_efficiency"]:
            try:
                # Prepare data (limit to top 10 for readability)
                top_efficient = report_data["cost_efficiency"][:10]
                models = [m["model"] for m in top_efficient]
                efficiency = [m["tokens_per_dollar"] for m in top_efficient]

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.bar(models, efficiency)
                plt.title("Cost Efficiency (Tokens per Dollar)")
                plt.xlabel("Model")
                plt.ylabel("Tokens per Dollar")
                plt.xticks(rotation=45, ha="right")
                plt.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()

                # Save plot
                plot_path = str(
                    plot_dir / f"cost_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(plot_path)
                plt.close()

                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(
                    f"Failed to generate cost efficiency plot: {str(e)}", emoji_key="error"
                )

        return plot_paths

    def _generate_html_report(self, report_data: Dict[str, Any], plot_paths: List[str]) -> str:
        """Generate an HTML usage report.

        Args:
            report_data: Report data
            plot_paths: List of plot file paths

        Returns:
            HTML report content
        """
        # Basic HTML template
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ultimate MCP Server Usage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #f9f9f9; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .stat {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .plot {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Ultimate MCP Server Usage Report</h1>
        <p>Generated at: {report_data["generated_at"]}</p>
        <p>Period: {report_data["period"]}</p>
        
        <div class="card">
            <h2>General Statistics</h2>
            <table>
                <tr>
                    <td>Uptime</td>
                    <td class="stat">{report_data["general"]["uptime_human"]}</td>
                </tr>
                <tr>
                    <td>Total Requests</td>
                    <td class="stat">{report_data["general"]["requests_total"]:,}</td>
                </tr>
                <tr>
                    <td>Total Tokens</td>
                    <td class="stat">{report_data["general"]["tokens_total"]:,}</td>
                </tr>
                <tr>
                    <td>Total Cost</td>
                    <td class="stat">${report_data["general"]["cost_total"]:.2f}</td>
                </tr>
                <tr>
                    <td>Average Response Time</td>
                    <td class="stat">{report_data["general"]["avg_response_time"]:.3f}s</td>
                </tr>
                <tr>
                    <td>Total Errors</td>
                    <td class="stat">{report_data["general"]["errors_total"]}</td>
                </tr>
                <tr>
                    <td>Error Rate</td>
                    <td class="stat">{report_data["general"]["error_rate"] * 100:.2f}%</td>
                </tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Cache Statistics</h2>
            <table>
                <tr>
                    <td>Cache Hits</td>
                    <td class="stat">{report_data["cache"]["hits"]:,}</td>
                </tr>
                <tr>
                    <td>Cache Misses</td>
                    <td class="stat">{report_data["cache"]["misses"]:,}</td>
                </tr>
                <tr>
                    <td>Hit Ratio</td>
                    <td class="stat">{report_data["cache"]["hit_ratio"] * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Cost Savings</td>
                    <td class="stat">${report_data["cache"]["saved_cost"]:.2f}</td>
                </tr>
            </table>
        </div>
"""

        # Add plots if available
        if plot_paths:
            html += """
        <div class="card">
            <h2>Usage Visualizations</h2>
"""
            for plot_path in plot_paths:
                # Use relative path
                rel_path = os.path.relpath(plot_path, self.reports_dir)
                html += f"""
            <img class="plot" src="{rel_path}" alt="Usage Plot">
"""
            html += """
        </div>
"""

        # Add top providers
        if report_data["top_providers"]:
            html += """
        <div class="card">
            <h2>Top Providers</h2>
            <table>
                <tr>
                    <th>Provider</th>
                    <th>Tokens</th>
                    <th>Percentage</th>
                </tr>
"""
            for provider in report_data["top_providers"]:
                html += f"""
                <tr>
                    <td>{provider["provider"]}</td>
                    <td>{provider["tokens"]:,}</td>
                    <td>{provider["percentage"] * 100:.2f}%</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        # Add top models
        if report_data["top_models"]:
            html += """
        <div class="card">
            <h2>Top Models</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Tokens</th>
                    <th>Percentage</th>
                </tr>
"""
            for model in report_data["top_models"]:
                html += f"""
                <tr>
                    <td>{model["model"]}</td>
                    <td>{model["tokens"]:,}</td>
                    <td>{model["percentage"] * 100:.2f}%</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        # Add daily usage
        if report_data["daily_usage"]:
            html += """
        <div class="card">
            <h2>Daily Usage</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                </tr>
"""
            for day in sorted(report_data["daily_usage"], key=lambda x: x["date"], reverse=True):
                html += f"""
                <tr>
                    <td>{day["date"]}</td>
                    <td>{day["tokens"]:,}</td>
                    <td>${day["cost"]:.2f}</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        # Close HTML
        html += """
    </div>
</body>
</html>
"""

        return html

    def _generate_markdown_report(self, report_data: Dict[str, Any], plot_paths: List[str]) -> str:
        """Generate a Markdown usage report.

        Args:
            report_data: Report data
            plot_paths: List of plot file paths

        Returns:
            Markdown report content
        """
        # Basic Markdown template
        markdown = f"""# Ultimate MCP Server Usage Report

Generated at: {report_data["generated_at"]}  
Period: {report_data["period"]}

## General Statistics

- **Uptime:** {report_data["general"]["uptime_human"]}
- **Total Requests:** {report_data["general"]["requests_total"]:,}
- **Total Tokens:** {report_data["general"]["tokens_total"]:,}
- **Total Cost:** ${report_data["general"]["cost_total"]:.2f}
- **Average Response Time:** {report_data["general"]["avg_response_time"]:.3f}s
- **Total Errors:** {report_data["general"]["errors_total"]}
- **Error Rate:** {report_data["general"]["error_rate"] * 100:.2f}%

## Cache Statistics

- **Cache Hits:** {report_data["cache"]["hits"]:,}
- **Cache Misses:** {report_data["cache"]["misses"]:,}
- **Hit Ratio:** {report_data["cache"]["hit_ratio"] * 100:.2f}%
- **Cost Savings:** ${report_data["cache"]["saved_cost"]:.2f}

"""

        # Add plots if available
        if plot_paths:
            markdown += """## Usage Visualizations

"""
            for plot_path in plot_paths:
                # Use relative path
                rel_path = os.path.relpath(plot_path, self.reports_dir)
                markdown += f"""![Usage Plot]({rel_path})

"""

        # Add top providers
        if report_data["top_providers"]:
            markdown += """## Top Providers

| Provider | Tokens | Percentage |
|----------|--------|------------|
"""
            for provider in report_data["top_providers"]:
                markdown += f"""| {provider["provider"]} | {provider["tokens"]:,} | {provider["percentage"] * 100:.2f}% |
"""
            markdown += "\n"

        # Add top models
        if report_data["top_models"]:
            markdown += """## Top Models

| Model | Tokens | Percentage |
|-------|--------|------------|
"""
            for model in report_data["top_models"]:
                markdown += f"""| {model["model"]} | {model["tokens"]:,} | {model["percentage"] * 100:.2f}% |
"""
            markdown += "\n"

        # Add daily usage
        if report_data["daily_usage"]:
            markdown += """## Daily Usage

| Date | Tokens | Cost |
|------|--------|------|
"""
            for day in sorted(report_data["daily_usage"], key=lambda x: x["date"], reverse=True):
                markdown += f"""| {day["date"]} | {day["tokens"]:,} | ${day["cost"]:.2f} |
"""

        return markdown

    def _generate_html_provider_report(
        self, report_data: Dict[str, Any], plot_paths: List[str]
    ) -> str:
        """Generate an HTML provider report.

        Args:
            report_data: Report data
            plot_paths: List of plot file paths

        Returns:
            HTML report content
        """
        # Basic HTML template (similar to usage report but provider-specific)
        provider = report_data["provider"]
        provider_stats = report_data["stats"]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{provider} Provider Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #f9f9f9; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .stat {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .plot {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{provider} Provider Report</h1>
        <p>Generated at: {report_data["generated_at"]}</p>
        <p>Period: {report_data["period"]}</p>
        
        <div class="card">
            <h2>Provider Statistics</h2>
            <table>
                <tr>
                    <td>Total Requests</td>
                    <td class="stat">{provider_stats["requests"]:,}</td>
                </tr>
                <tr>
                    <td>Total Tokens</td>
                    <td class="stat">{provider_stats["tokens"]:,}</td>
                </tr>
                <tr>
                    <td>Total Cost</td>
                    <td class="stat">${provider_stats["cost"]:.2f}</td>
                </tr>
                <tr>
                    <td>Average Response Time</td>
                    <td class="stat">{provider_stats["avg_response_time"]:.3f}s</td>
                </tr>
                <tr>
                    <td>Total Errors</td>
                    <td class="stat">{provider_stats["errors"]}</td>
                </tr>
                <tr>
                    <td>Percentage of Total Usage</td>
                    <td class="stat">{report_data["percentage_of_total"]:.2f}%</td>
                </tr>
            </table>
        </div>
"""

        # Add plots if available
        if plot_paths:
            html += """
        <div class="card">
            <h2>Usage Visualizations</h2>
"""
            for plot_path in plot_paths:
                # Use relative path
                rel_path = os.path.relpath(plot_path, self.reports_dir)
                html += f"""
            <img class="plot" src="{rel_path}" alt="Provider Usage Plot">
"""
            html += """
        </div>
"""

        # Add models
        if report_data["models"]:
            html += """
        <div class="card">
            <h2>Models</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Requests</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                    <th>Avg Response Time</th>
                </tr>
"""
            for model, data in report_data["models"].items():
                html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{data["requests"]:,}</td>
                    <td>{data["tokens"]:,}</td>
                    <td>${data["cost"]:.2f}</td>
                    <td>{data["avg_response_time"]:.3f}s</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        # Add daily usage
        if report_data["daily_usage"]:
            html += """
        <div class="card">
            <h2>Daily Usage</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                </tr>
"""
            for day in sorted(report_data["daily_usage"], key=lambda x: x["date"], reverse=True):
                html += f"""
                <tr>
                    <td>{day["date"]}</td>
                    <td>{day["tokens"]:,}</td>
                    <td>${day["cost"]:.2f}</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        # Close HTML
        html += """
    </div>
</body>
</html>
"""

        return html

    def _generate_markdown_provider_report(
        self, report_data: Dict[str, Any], plot_paths: List[str]
    ) -> str:
        """Generate a Markdown provider report.

        Args:
            report_data: Report data
            plot_paths: List of plot file paths

        Returns:
            Markdown report content
        """
        # Basic Markdown template (similar to usage report but provider-specific)
        provider = report_data["provider"]
        provider_stats = report_data["stats"]

        markdown = f"""# {provider} Provider Report

Generated at: {report_data["generated_at"]}  
Period: {report_data["period"]}

## Provider Statistics

- **Total Requests:** {provider_stats["requests"]:,}
- **Total Tokens:** {provider_stats["tokens"]:,}
- **Total Cost:** ${provider_stats["cost"]:.2f}
- **Average Response Time:** {provider_stats["avg_response_time"]:.3f}s
- **Total Errors:** {provider_stats["errors"]}
- **Percentage of Total Usage:** {report_data["percentage_of_total"]:.2f}%

"""

        # Add plots if available
        if plot_paths:
            markdown += """## Usage Visualizations

"""
            for plot_path in plot_paths:
                # Use relative path
                rel_path = os.path.relpath(plot_path, self.reports_dir)
                markdown += f"""![Provider Usage Plot]({rel_path})

"""

        # Add models
        if report_data["models"]:
            markdown += """## Models

| Model | Requests | Tokens | Cost | Avg Response Time |
|-------|----------|--------|------|-------------------|
"""
            for model, data in report_data["models"].items():
                markdown += f"""| {model} | {data["requests"]:,} | {data["tokens"]:,} | ${data["cost"]:.2f} | {data["avg_response_time"]:.3f}s |
"""
            markdown += "\n"

        # Add daily usage
        if report_data["daily_usage"]:
            markdown += """## Daily Usage

| Date | Tokens | Cost |
|------|--------|------|
"""
            for day in sorted(report_data["daily_usage"], key=lambda x: x["date"], reverse=True):
                markdown += f"""| {day["date"]} | {day["tokens"]:,} | ${day["cost"]:.2f} |
"""

        return markdown

    def _generate_html_cost_report(self, report_data: Dict[str, Any], plot_paths: List[str]) -> str:
        """Generate an HTML cost report.

        Args:
            report_data: Report data
            plot_paths: List of plot file paths

        Returns:
            HTML report content
        """
        # Basic HTML template (cost-focused)
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ultimate MCP Server Cost Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #f9f9f9; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .stat {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .cost {{ font-size: 24px; font-weight: bold; color: #cc0000; }}
        .savings {{ font-size: 24px; font-weight: bold; color: #00cc00; }}
        .plot {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Ultimate MCP Server Cost Report</h1>
        <p>Generated at: {report_data["generated_at"]}</p>
        <p>Period: {report_data["period"]}</p>
        
        <div class="card">
            <h2>Cost Overview</h2>
            <table>
                <tr>
                    <td>Total Cost</td>
                    <td class="cost">${report_data["total_cost"]:.2f}</td>
                </tr>
                <tr>
                    <td>Cache Savings</td>
                    <td class="savings">${report_data["cache_savings"]:.2f}</td>
                </tr>
                <tr>
                    <td>Net Cost</td>
                    <td class="cost">${report_data["total_cost"] - report_data["cache_savings"]:.2f}</td>
                </tr>
            </table>
        </div>
"""

        # Add plots if available
        if plot_paths:
            html += """
        <div class="card">
            <h2>Cost Visualizations</h2>
"""
            for plot_path in plot_paths:
                # Use relative path
                rel_path = os.path.relpath(plot_path, self.reports_dir)
                html += f"""
            <img class="plot" src="{rel_path}" alt="Cost Plot">
"""
            html += """
        </div>
"""

        # Add provider costs
        if report_data["provider_costs"]:
            html += """
        <div class="card">
            <h2>Cost by Provider</h2>
            <table>
                <tr>
                    <th>Provider</th>
                    <th>Cost</th>
                    <th>Percentage</th>
                </tr>
"""
            for provider in report_data["provider_costs"]:
                html += f"""
                <tr>
                    <td>{provider["provider"]}</td>
                    <td>${provider["cost"]:.2f}</td>
                    <td>{provider["percentage"]:.2f}%</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        # Add model costs
        if report_data["model_costs"]:
            html += """
        <div class="card">
            <h2>Cost by Model</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Cost</th>
                    <th>Percentage</th>
                </tr>
"""
            for model in report_data["model_costs"]:
                html += f"""
                <tr>
                    <td>{model["model"]}</td>
                    <td>${model["cost"]:.2f}</td>
                    <td>{model["percentage"]:.2f}%</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        # Add cost efficiency
        if report_data["cost_efficiency"]:
            html += """
        <div class="card">
            <h2>Cost Efficiency (Tokens per Dollar)</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Tokens per Dollar</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                </tr>
"""
            for model in report_data["cost_efficiency"]:
                html += f"""
                <tr>
                    <td>{model["model"]}</td>
                    <td class="stat">{model["tokens_per_dollar"]:,.0f}</td>
                    <td>{model["tokens"]:,}</td>
                    <td>${model["cost"]:.2f}</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        # Add daily costs
        if report_data["daily_costs"]:
            html += """
        <div class="card">
            <h2>Daily Costs</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Cost</th>
                </tr>
"""
            for day in sorted(report_data["daily_costs"], key=lambda x: x["date"], reverse=True):
                html += f"""
                <tr>
                    <td>{day["date"]}</td>
                    <td>${day["cost"]:.2f}</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        # Close HTML
        html += """
    </div>
</body>
</html>
"""

        return html

    def _generate_markdown_cost_report(
        self, report_data: Dict[str, Any], plot_paths: List[str]
    ) -> str:
        """Generate a Markdown cost report.

        Args:
            report_data: Report data
            plot_paths: List of plot file paths

        Returns:
            Markdown report content
        """
        # Basic Markdown template (cost-focused)
        markdown = f"""# Ultimate MCP Server Cost Report

Generated at: {report_data["generated_at"]}  
Period: {report_data["period"]}

## Cost Overview

- **Total Cost:** ${report_data["total_cost"]:.2f}
- **Cache Savings:** ${report_data["cache_savings"]:.2f}
- **Net Cost:** ${report_data["total_cost"] - report_data["cache_savings"]:.2f}

"""

        # Add plots if available
        if plot_paths:
            markdown += """## Cost Visualizations

"""
            for plot_path in plot_paths:
                # Use relative path
                rel_path = os.path.relpath(plot_path, self.reports_dir)
                markdown += f"""![Cost Plot]({rel_path})

"""

        # Add provider costs
        if report_data["provider_costs"]:
            markdown += """## Cost by Provider

| Provider | Cost | Percentage |
|----------|------|------------|
"""
            for provider in report_data["provider_costs"]:
                markdown += f"""| {provider["provider"]} | ${provider["cost"]:.2f} | {provider["percentage"]:.2f}% |
"""
            markdown += "\n"

        # Add model costs
        if report_data["model_costs"]:
            markdown += """## Cost by Model

| Model | Cost | Percentage |
|-------|------|------------|
"""
            for model in report_data["model_costs"]:
                markdown += f"""| {model["model"]} | ${model["cost"]:.2f} | {model["percentage"]:.2f}% |
"""
            markdown += "\n"

        # Add cost efficiency
        if report_data["cost_efficiency"]:
            markdown += """## Cost Efficiency (Tokens per Dollar)

| Model | Tokens per Dollar | Tokens | Cost |
|-------|-------------------|--------|------|
"""
            for model in report_data["cost_efficiency"]:
                markdown += f"""| {model["model"]} | {model["tokens_per_dollar"]:,.0f} | {model["tokens"]:,} | ${model["cost"]:.2f} |
"""
            markdown += "\n"

        # Add daily costs
        if report_data["daily_costs"]:
            markdown += """## Daily Costs

| Date | Cost |
|------|------|
"""
            for day in sorted(report_data["daily_costs"], key=lambda x: x["date"], reverse=True):
                markdown += f"""| {day["date"]} | ${day["cost"]:.2f} |
"""

        return markdown
