"""Metrics collection and monitoring for Ultimate MCP Server."""

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)

try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsTracker:
    """Comprehensive metrics tracking and monitoring system for Ultimate MCP Server.

    The MetricsTracker is a singleton class that collects, processes, and persists
    operational metrics related to LLM usage, costs, performance, and errors. It provides
    the data foundation for analytics reporting and monitoring tools.

    Key features:
    - Singleton design pattern ensures consistent metrics across application
    - Persistent storage with automatic serialization to JSON
    - Tracks usage by provider, model, and time periods (hourly, daily)
    - Records request counts, token usage, costs, errors, and performance metrics
    - Cache efficiency monitoring (hits, misses, cost savings)
    - Optional Prometheus integration for external monitoring systems
    - Asynchronous persistence to minimize performance impact
    - Automatic data retention policies to prevent memory bloat

    The metrics are automatically persisted to disk and can be loaded on startup,
    providing continuity across server restarts. Time-series data is maintained
    for historical analysis and trend visualization.

    Usage:
        # Get the singleton instance
        metrics = get_metrics_tracker()

        # Record a request
        metrics.record_request(
            provider="anthropic",
            model="claude-3-opus",
            input_tokens=150,
            output_tokens=500,
            cost=0.0325,
            duration=2.5
        )

        # Record cache operations
        metrics.record_cache_hit(cost_saved=0.015)
        metrics.record_cache_miss()

        # Get current statistics
        stats = metrics.get_stats()

        # Manually trigger persistence (usually automatic)
        metrics.save_metrics()
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(MetricsTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        metrics_dir: Optional[Union[str, Path]] = None,
        enable_prometheus: bool = False,
        reset_on_start: bool = False,
    ):
        """Initialize the metrics tracker.

        Args:
            metrics_dir: Directory for metrics storage
            enable_prometheus: Whether to enable Prometheus metrics
            reset_on_start: Whether to reset metrics on startup
        """
        # Only initialize once for singleton
        if self._initialized:
            return

        # Set metrics directory
        if metrics_dir:
            self.metrics_dir = Path(metrics_dir)
        else:
            self.metrics_dir = Path.home() / ".ultimate" / "metrics"

        # Create metrics directory if it doesn't exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Prometheus settings
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE

        # Initialize metrics data
        if reset_on_start:
            self._reset_metrics()
        else:
            self._load_metrics()

        # Initialize Prometheus metrics if enabled
        if self.enable_prometheus:
            self._init_prometheus_metrics()

        self._initialized = True

        logger.info(
            f"Metrics tracker initialized (dir: {self.metrics_dir}, prometheus: {self.enable_prometheus})",
            emoji_key="analytics",
        )

    def _reset_metrics(self):
        """Reset all metrics data."""
        # General stats
        self.start_time = time.time()
        self.requests_total = 0
        self.tokens_total = 0
        self.cost_total = 0.0

        # Provider-specific stats
        self.provider_requests = defaultdict(int)
        self.provider_tokens = defaultdict(int)
        self.provider_costs = defaultdict(float)

        # Model-specific stats
        self.model_requests = defaultdict(int)
        self.model_tokens = defaultdict(int)
        self.model_costs = defaultdict(float)

        # Request timing stats
        self.request_times = []
        self.request_times_by_provider = defaultdict(list)
        self.request_times_by_model = defaultdict(list)

        # Error stats
        self.errors_total = 0
        self.errors_by_provider = defaultdict(int)
        self.errors_by_model = defaultdict(int)

        # Token usage by time period
        self.hourly_tokens = defaultdict(int)
        self.daily_tokens = defaultdict(int)

        # Request counts by time period
        self.hourly_requests = defaultdict(int)
        self.daily_requests = defaultdict(int)

        # Cost by time period
        self.hourly_costs = defaultdict(float)
        self.daily_costs = defaultdict(float)

        # Cache stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_saved_cost = 0.0

    def _load_metrics(self):
        """Load metrics from disk."""
        metrics_file = self.metrics_dir / "metrics.json"

        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)

                # Load general stats
                self.start_time = data.get("start_time", time.time())
                self.requests_total = data.get("requests_total", 0)
                self.tokens_total = data.get("tokens_total", 0)
                self.cost_total = data.get("cost_total", 0.0)

                # Load provider stats
                self.provider_requests = defaultdict(int, data.get("provider_requests", {}))
                self.provider_tokens = defaultdict(int, data.get("provider_tokens", {}))
                self.provider_costs = defaultdict(float, data.get("provider_costs", {}))

                # Load model stats
                self.model_requests = defaultdict(int, data.get("model_requests", {}))
                self.model_tokens = defaultdict(int, data.get("model_tokens", {}))
                self.model_costs = defaultdict(float, data.get("model_costs", {}))

                # Load timing stats (limited to last 1000 for memory)
                self.request_times = data.get("request_times", [])[-1000:]
                self.request_times_by_provider = defaultdict(list)
                for provider, times in data.get("request_times_by_provider", {}).items():
                    self.request_times_by_provider[provider] = times[-1000:]

                self.request_times_by_model = defaultdict(list)
                for model, times in data.get("request_times_by_model", {}).items():
                    self.request_times_by_model[model] = times[-1000:]

                # Load error stats
                self.errors_total = data.get("errors_total", 0)
                self.errors_by_provider = defaultdict(int, data.get("errors_by_provider", {}))
                self.errors_by_model = defaultdict(int, data.get("errors_by_model", {}))

                # Load time period stats
                self.hourly_tokens = defaultdict(int, data.get("hourly_tokens", {}))
                self.daily_tokens = defaultdict(int, data.get("daily_tokens", {}))
                self.hourly_costs = defaultdict(float, data.get("hourly_costs", {}))
                self.daily_costs = defaultdict(float, data.get("daily_costs", {}))
                self.hourly_requests = defaultdict(int, data.get("hourly_requests", {}))
                self.daily_requests = defaultdict(int, data.get("daily_requests", {}))

                # Load cache stats
                self.cache_hits = data.get("cache_hits", 0)
                self.cache_misses = data.get("cache_misses", 0)
                self.cache_saved_cost = data.get("cache_saved_cost", 0.0)

                logger.info(f"Loaded metrics from {metrics_file}", emoji_key="analytics")

            except Exception as e:
                logger.error(f"Failed to load metrics: {str(e)}", emoji_key="error")
                self._reset_metrics()
        else:
            self._reset_metrics()

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        # Request metrics
        self.prom_requests_total = Counter(
            "ultimate_requests_total", "Total number of requests", ["provider", "model"]
        )

        # Token metrics
        self.prom_tokens_total = Counter(
            "ultimate_tokens_total",
            "Total number of tokens",
            ["provider", "model", "type"],  # type: input or output
        )

        # Cost metrics
        self.prom_cost_total = Counter(
            "ultimate_cost_total", "Total cost in USD", ["provider", "model"]
        )

        # Timing metrics
        self.prom_request_duration = Histogram(
            "ultimate_request_duration_seconds",
            "Request duration in seconds",
            ["provider", "model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )

        # Error metrics
        self.prom_errors_total = Counter(
            "ultimate_errors_total", "Total number of errors", ["provider", "model"]
        )

        # Cache metrics
        self.prom_cache_hits = Counter("ultimate_cache_hits_total", "Total number of cache hits")
        self.prom_cache_misses = Counter(
            "ultimate_cache_misses_total", "Total number of cache misses"
        )
        self.prom_cache_saved_cost = Counter(
            "ultimate_cache_saved_cost_total", "Total cost saved by cache in USD"
        )

    async def _save_metrics_async(self):
        """Save metrics to disk asynchronously."""
        if not AIOFILES_AVAILABLE:
            return

        metrics_file = self.metrics_dir / "metrics.json"
        temp_file = metrics_file.with_suffix(".tmp")

        try:
            # Prepare data for storage
            data = {
                "start_time": self.start_time,
                "requests_total": self.requests_total,
                "tokens_total": self.tokens_total,
                "cost_total": self.cost_total,
                "provider_requests": dict(self.provider_requests),
                "provider_tokens": dict(self.provider_tokens),
                "provider_costs": dict(self.provider_costs),
                "model_requests": dict(self.model_requests),
                "model_tokens": dict(self.model_tokens),
                "model_costs": dict(self.model_costs),
                "request_times": self.request_times,
                "request_times_by_provider": {
                    k: v for k, v in self.request_times_by_provider.items()
                },
                "request_times_by_model": {k: v for k, v in self.request_times_by_model.items()},
                "errors_total": self.errors_total,
                "errors_by_provider": dict(self.errors_by_provider),
                "errors_by_model": dict(self.errors_by_model),
                "hourly_tokens": dict(self.hourly_tokens),
                "daily_tokens": dict(self.daily_tokens),
                "hourly_costs": dict(self.hourly_costs),
                "daily_costs": dict(self.daily_costs),
                "hourly_requests": dict(self.hourly_requests),
                "daily_requests": dict(self.daily_requests),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_saved_cost": self.cache_saved_cost,
                "last_updated": time.time(),
            }

            # Save to temp file
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(data, indent=2))

            # Rename temp file to actual file
            os.replace(temp_file, metrics_file)

        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}", emoji_key="error")

    def save_metrics(self):
        """Save metrics to disk synchronously."""
        metrics_file = self.metrics_dir / "metrics.json"
        temp_file = metrics_file.with_suffix(".tmp")

        try:
            # Prepare data for storage
            data = {
                "start_time": self.start_time,
                "requests_total": self.requests_total,
                "tokens_total": self.tokens_total,
                "cost_total": self.cost_total,
                "provider_requests": dict(self.provider_requests),
                "provider_tokens": dict(self.provider_tokens),
                "provider_costs": dict(self.provider_costs),
                "model_requests": dict(self.model_requests),
                "model_tokens": dict(self.model_tokens),
                "model_costs": dict(self.model_costs),
                "request_times": self.request_times,
                "request_times_by_provider": {
                    k: v for k, v in self.request_times_by_provider.items()
                },
                "request_times_by_model": {k: v for k, v in self.request_times_by_model.items()},
                "errors_total": self.errors_total,
                "errors_by_provider": dict(self.errors_by_provider),
                "errors_by_model": dict(self.errors_by_model),
                "hourly_tokens": dict(self.hourly_tokens),
                "daily_tokens": dict(self.daily_tokens),
                "hourly_costs": dict(self.hourly_costs),
                "daily_costs": dict(self.daily_costs),
                "hourly_requests": dict(self.hourly_requests),
                "daily_requests": dict(self.daily_requests),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_saved_cost": self.cache_saved_cost,
                "last_updated": time.time(),
            }

            # Save to temp file
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)

            # Rename temp file to actual file
            os.replace(temp_file, metrics_file)

        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}", emoji_key="error")

    def record_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        duration: float,
        success: bool = True,
    ):
        """Record metrics for a request.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost of the request
            duration: Duration of the request in seconds
            success: Whether the request was successful
        """
        # Update general stats
        self.requests_total += 1
        total_tokens = input_tokens + output_tokens
        self.tokens_total += total_tokens
        self.cost_total += cost

        # Update provider stats
        self.provider_requests[provider] += 1
        self.provider_tokens[provider] += total_tokens
        self.provider_costs[provider] += cost

        # Update model stats
        self.model_requests[model] += 1
        self.model_tokens[model] += total_tokens
        self.model_costs[model] += cost

        # Update timing stats
        self.request_times.append(duration)
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]

        self.request_times_by_provider[provider].append(duration)
        if len(self.request_times_by_provider[provider]) > 1000:
            self.request_times_by_provider[provider] = self.request_times_by_provider[provider][
                -1000:
            ]

        self.request_times_by_model[model].append(duration)
        if len(self.request_times_by_model[model]) > 1000:
            self.request_times_by_model[model] = self.request_times_by_model[model][-1000:]

        # Update error stats if request failed
        if not success:
            self.errors_total += 1
            self.errors_by_provider[provider] += 1
            self.errors_by_model[model] += 1

        # Update time period stats
        current_time = time.time()
        hour_key = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d-%H")
        day_key = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d")

        self.hourly_tokens[hour_key] += total_tokens
        self.daily_tokens[day_key] += total_tokens
        self.hourly_costs[hour_key] += cost
        self.daily_costs[day_key] += cost
        self.hourly_requests[hour_key] += 1
        self.daily_requests[day_key] += 1

        # Update Prometheus metrics if enabled
        if self.enable_prometheus:
            self.prom_requests_total.labels(provider=provider, model=model).inc()
            self.prom_tokens_total.labels(provider=provider, model=model, type="input").inc(
                input_tokens
            )
            self.prom_tokens_total.labels(provider=provider, model=model, type="output").inc(
                output_tokens
            )
            self.prom_cost_total.labels(provider=provider, model=model).inc(cost)
            self.prom_request_duration.labels(provider=provider, model=model).observe(duration)

            if not success:
                self.prom_errors_total.labels(provider=provider, model=model).inc()

        # Schedule metrics saving
        try:
            import asyncio

            asyncio.create_task(self._save_metrics_async())
        except (ImportError, RuntimeError):
            # Fall back to synchronous saving if asyncio not available
            self.save_metrics()

    def record_cache_hit(self, cost_saved: float = 0.0):
        """Record a cache hit.

        Args:
            cost_saved: Cost saved by the cache hit
        """
        self.cache_hits += 1
        self.cache_saved_cost += cost_saved

        # Update Prometheus metrics if enabled
        if self.enable_prometheus:
            self.prom_cache_hits.inc()
            self.prom_cache_saved_cost.inc(cost_saved)

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1

        # Update Prometheus metrics if enabled
        if self.enable_prometheus:
            self.prom_cache_misses.inc()

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary of metrics
        """
        # Calculate uptime
        uptime = time.time() - self.start_time

        # Calculate request rate (per minute)
        request_rate = self.requests_total / (uptime / 60) if uptime > 0 else 0

        # Calculate average response time
        avg_response_time = (
            sum(self.request_times) / len(self.request_times) if self.request_times else 0
        )

        # Calculate cache hit ratio
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_ratio = self.cache_hits / cache_total if cache_total > 0 else 0

        # Get top providers by usage
        top_providers = sorted(
            [(provider, tokens) for provider, tokens in self.provider_tokens.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Get top models by usage
        top_models = sorted(
            [(model, tokens) for model, tokens in self.model_tokens.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Get daily token usage for last 7 days
        today = datetime.now().strftime("%Y-%m-%d")
        daily_usage = []
        for i in range(7):
            day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_usage.append(
                (
                    day,
                    self.daily_tokens.get(day, 0),
                    self.daily_costs.get(day, 0.0),
                    self.daily_requests.get(day, 0),
                )
            )

        # Compile stats
        return {
            "general": {
                "uptime": uptime,
                "uptime_human": self._format_duration(uptime),
                "requests_total": self.requests_total,
                "tokens_total": self.tokens_total,
                "cost_total": self.cost_total,
                "request_rate": request_rate,
                "avg_response_time": avg_response_time,
                "errors_total": self.errors_total,
                "error_rate": self.errors_total / self.requests_total
                if self.requests_total > 0
                else 0,
            },
            "providers": {
                provider: {
                    "requests": count,
                    "tokens": self.provider_tokens.get(provider, 0),
                    "cost": self.provider_costs.get(provider, 0.0),
                    "avg_response_time": sum(self.request_times_by_provider.get(provider, []))
                    / len(self.request_times_by_provider.get(provider, []))
                    if self.request_times_by_provider.get(provider, [])
                    else 0,
                    "errors": self.errors_by_provider.get(provider, 0),
                }
                for provider, count in self.provider_requests.items()
            },
            "models": {
                model: {
                    "requests": count,
                    "tokens": self.model_tokens.get(model, 0),
                    "cost": self.model_costs.get(model, 0.0),
                    "avg_response_time": sum(self.request_times_by_model.get(model, []))
                    / len(self.request_times_by_model.get(model, []))
                    if self.request_times_by_model.get(model, [])
                    else 0,
                    "errors": self.errors_by_model.get(model, 0),
                }
                for model, count in self.model_requests.items()
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_ratio": cache_hit_ratio,
                "saved_cost": self.cache_saved_cost,
            },
            "top_providers": [
                {
                    "provider": provider,
                    "tokens": tokens,
                    "percentage": tokens / self.tokens_total if self.tokens_total > 0 else 0,
                }
                for provider, tokens in top_providers
            ],
            "top_models": [
                {
                    "model": model,
                    "tokens": tokens,
                    "percentage": tokens / self.tokens_total if self.tokens_total > 0 else 0,
                }
                for model, tokens in top_models
            ],
            "daily_usage": [
                {"date": date, "tokens": tokens, "cost": cost, "requests": requests}
                for date, tokens, cost, requests in daily_usage
            ],
            "today": {
                "tokens": self.daily_tokens.get(today, 0),
                "cost": self.daily_costs.get(today, 0.0),
                "requests": self.daily_requests.get(today, 0),
            },
        }

    def _format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration
        """
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
        else:
            days = seconds / 86400
            return f"{days:.1f} days"

    def reset(self):
        """Reset all metrics."""
        self._reset_metrics()
        logger.info("Metrics reset", emoji_key="analytics")


# Singleton instance getter
def get_metrics_tracker(
    metrics_dir: Optional[Union[str, Path]] = None,
    enable_prometheus: bool = False,
    reset_on_start: bool = False,
) -> MetricsTracker:
    """Get the metrics tracker singleton instance.

    Args:
        metrics_dir: Directory for metrics storage
        enable_prometheus: Whether to enable Prometheus metrics
        reset_on_start: Whether to reset metrics on startup

    Returns:
        MetricsTracker singleton instance
    """
    return MetricsTracker(metrics_dir, enable_prometheus, reset_on_start)
