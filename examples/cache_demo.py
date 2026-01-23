#!/usr/bin/env python
"""Cache demonstration for Ultimate MCP Server."""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.markup import escape
from rich.rule import Rule

from ultimate_mcp_server.services.cache import get_cache_service, run_completion_with_cache
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker, display_cache_stats

# --- Add Rich Imports ---
from ultimate_mcp_server.utils.logging.console import console

# ----------------------

# Initialize logger
logger = get_logger("example.cache_demo")


async def demonstrate_cache(tracker: CostTracker = None):
    """Demonstrate cache functionality using Rich."""
    console.print(Rule("[bold blue]Cache Demonstration[/bold blue]"))
    logger.info("Starting cache demonstration", emoji_key="start")

    cache_service = get_cache_service()

    if not cache_service.enabled:
        logger.warning(
            "Cache is disabled by default. Enabling for demonstration.", emoji_key="warning"
        )
        cache_service.enabled = True

    cache_service.clear()  # Start with a clean slate
    logger.info("Cache cleared for demonstration", emoji_key="cache")

    prompt = "Explain how caching works in distributed systems."
    console.print(f"[cyan]Using Prompt:[/cyan] {escape(prompt)}")
    console.print()

    results = {}
    times = {}
    stats_log = {}

    try:
        # Helper function to get current stats snapshot
        def get_current_stats_dict():
            return {
                "get_count": getattr(cache_service.metrics, "gets", 0),  # Use gets for Total Gets
                "hit_count": getattr(cache_service.metrics, "hits", 0),
                "miss_count": getattr(cache_service.metrics, "misses", 0),
                "set_count": getattr(
                    cache_service.metrics, "stores", 0
                ),  # Use stores for Total Sets
                # Add other stats if needed by display_cache_stats later
            }

        # 1. Cache Miss
        logger.info("1. Running first completion (expect cache MISS)...", emoji_key="processing")
        start_time = time.time()
        results[1] = await run_completion_with_cache(prompt, use_cache=True)
        times[1] = time.time() - start_time
        stats_log[1] = get_current_stats_dict()

        # Track cost - only for non-cache hits (actual API calls)
        if tracker:
            tracker.add_call(results[1])

        console.print(
            f"   [yellow]MISS:[/yellow] Took [bold]{times[1]:.3f}s[/bold] (Cost: ${results[1].cost:.6f}, Tokens: {results[1].total_tokens})"
        )

        # 2. Cache Hit
        logger.info("2. Running second completion (expect cache HIT)...", emoji_key="processing")
        start_time = time.time()
        results[2] = await run_completion_with_cache(prompt, use_cache=True)
        times[2] = time.time() - start_time
        stats_log[2] = get_current_stats_dict()
        speedup = times[1] / times[2] if times[2] > 0 else float("inf")
        console.print(
            f"   [green]HIT:[/green]  Took [bold]{times[2]:.3f}s[/bold] (Speed-up: {speedup:.1f}x vs Miss)"
        )

        # 3. Cache Bypass
        logger.info("3. Running third completion (BYPASS cache)...", emoji_key="processing")
        start_time = time.time()
        results[3] = await run_completion_with_cache(prompt, use_cache=False)
        times[3] = time.time() - start_time
        stats_log[3] = get_current_stats_dict()  # Stats shouldn't change much for bypass

        # Track cost - bypassing cache calls the API
        if tracker:
            tracker.add_call(results[3])

        console.print(
            f"   [cyan]BYPASS:[/cyan] Took [bold]{times[3]:.3f}s[/bold] (Cost: ${results[3].cost:.6f}, Tokens: {results[3].total_tokens})"
        )

        # 4. Another Cache Hit
        logger.info(
            "4. Running fourth completion (expect cache HIT again)...", emoji_key="processing"
        )
        start_time = time.time()
        results[4] = await run_completion_with_cache(prompt, use_cache=True)
        times[4] = time.time() - start_time
        stats_log[4] = get_current_stats_dict()
        speedup_vs_bypass = times[3] / times[4] if times[4] > 0 else float("inf")
        console.print(
            f"   [green]HIT:[/green]  Took [bold]{times[4]:.3f}s[/bold] (Speed-up: {speedup_vs_bypass:.1f}x vs Bypass)"
        )
        console.print()

    except Exception as e:
        logger.error(f"Error during cache demonstration run: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error during demo run:[/bold red] {escape(str(e))}")
        # Attempt to display stats even if error occurred mid-way
        final_stats_dict = get_current_stats_dict()  # Get stats even on error
    else:
        # Get final stats if all runs succeeded
        final_stats_dict = get_current_stats_dict()

    # Prepare the final stats dictionary for display_cache_stats
    # It expects top-level keys like 'enabled', 'persistence', and a 'stats' sub-dict
    display_stats = {
        "enabled": cache_service.enabled,
        "persistence": cache_service.enable_persistence,
        "stats": final_stats_dict,
        # Add savings if available/calculated (Example: Placeholder)
        # "savings": { "cost": getattr(cache_service.metrics, "saved_cost", 0.0) }
    }

    # Display Final Cache Statistics using our display function
    display_cache_stats(display_stats, stats_log, console)

    console.print()
    # Use the persistence setting directly from cache_service
    if cache_service.enable_persistence:
        logger.info("Cache persistence is enabled.", emoji_key="cache")
        if hasattr(cache_service, "cache_dir"):
            console.print(f"[dim]Cache Directory: {cache_service.cache_dir}[/dim]")
    else:
        logger.info("Cache persistence is disabled.", emoji_key="cache")
    console.print()


async def main():
    """Run cache demonstration."""
    tracker = CostTracker()  # Create cost tracker instance
    try:
        await demonstrate_cache(tracker)

        # Display cost summary at the end
        tracker.display_summary(console)

    except Exception as e:
        logger.critical(f"Cache demonstration failed: {str(e)}", emoji_key="critical")
        return 1

    return 0


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
