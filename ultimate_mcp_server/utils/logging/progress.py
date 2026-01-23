"""
Progress tracking and visualization for Gateway.

This module provides enhanced progress tracking capabilities with Rich,
supporting nested tasks, task groups, and dynamic progress updates.
"""

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, TypeVar

from rich.box import ROUNDED
from rich.console import Console, ConsoleRenderable, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    TaskID,  # Import TaskID type hint
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import Progress as RichProgress  # Renamed to avoid clash
from rich.table import Table

from .console import console as default_console  # Use the shared console instance

# Use relative imports

# TypeVar for generic progress tracking over iterables
T = TypeVar("T")


@dataclass
class TaskInfo:
    """Information about a single task being tracked."""

    description: str
    total: float
    completed: float = 0.0
    status: str = "running"  # running, success, error, skipped
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    parent_id: Optional[str] = None
    rich_task_id: Optional[TaskID] = None  # ID from Rich Progress
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        """Calculate elapsed time."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def is_complete(self) -> bool:
        """Check if the task is in a terminal state."""
        return self.status in ("success", "error", "skipped")


class GatewayProgress:
    """Manages multiple progress tasks with Rich integration and context.

    Allows for nested tasks and displays an overall summary.
    Uses a single Rich Progress instance managed internally.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        transient: bool = False,  # Keep visible after completion?
        auto_refresh: bool = True,
        expand: bool = True,  # Expand progress bars to full width?
        show_summary: bool = True,
        summary_refresh_rate: float = 1.0,  # How often to refresh summary
    ):
        """Initialize the progress manager.

        Args:
            console: Rich Console instance (defaults to shared console)
            transient: Hide progress bars upon completion
            auto_refresh: Automatically refresh the display
            expand: Expand bars to console width
            show_summary: Display the summary panel below progress bars
            summary_refresh_rate: Rate limit for summary updates (seconds)
        """
        self.console = console or default_console
        self._rich_progress = self._create_progress(transient, auto_refresh, expand)
        self._live: Optional[Live] = None
        self._tasks: Dict[str, TaskInfo] = {}
        self._task_stack: List[str] = []  # For context managers
        self.show_summary = show_summary
        self._summary_renderable = self._render_summary()  # Initial summary
        self._last_summary_update = 0.0
        self.summary_refresh_rate = summary_refresh_rate

    def _create_progress(self, transient: bool, auto_refresh: bool, expand: bool) -> RichProgress:
        """Create the underlying Rich Progress instance."""
        return RichProgress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None if expand else 40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=transient,
            auto_refresh=auto_refresh,
            expand=expand,
            # disable=True # Useful for debugging
        )

    def _render_summary(self) -> Group:
        """Render the overall progress summary table."""
        if not self.show_summary or not self._tasks:
            return Group()  # Empty group if no summary needed or no tasks yet

        completed_count = sum(1 for t in self._tasks.values() if t.is_complete)
        running_count = len(self._tasks) - completed_count
        success_count = sum(1 for t in self._tasks.values() if t.status == "success")
        error_count = sum(1 for t in self._tasks.values() if t.status == "error")
        skipped_count = sum(1 for t in self._tasks.values() if t.status == "skipped")

        total_elapsed = (
            time.time() - min(t.start_time for t in self._tasks.values()) if self._tasks else 0
        )

        # Calculate overall percentage (weighted average might be better?)
        overall_total = sum(t.total for t in self._tasks.values())
        overall_completed = sum(t.completed for t in self._tasks.values())
        overall_perc = (overall_completed / overall_total * 100) if overall_total > 0 else 100.0

        summary_table = Table(box=ROUNDED, show_header=False, padding=(0, 1), expand=True)
        summary_table.add_column("Metric", style="dim", width=15)
        summary_table.add_column("Value", style="bold")

        summary_table.add_row("Overall Prog.", f"{overall_perc:.1f}%")
        summary_table.add_row("Total Tasks", str(len(self._tasks)))
        summary_table.add_row("  Running", str(running_count))
        summary_table.add_row("  Completed", str(completed_count))
        if success_count > 0:
            summary_table.add_row("    Success", f"[success]{success_count}[/]")
        if error_count > 0:
            summary_table.add_row("    Errors", f"[error]{error_count}[/]")
        if skipped_count > 0:
            summary_table.add_row("    Skipped", f"[warning]{skipped_count}[/]")
        summary_table.add_row("Elapsed Time", f"{total_elapsed:.2f}s")

        return Group(summary_table)

    def _get_renderable(self) -> ConsoleRenderable:
        """Get the combined renderable for the Live display."""
        # Throttle summary updates
        now = time.time()
        if self.show_summary and (now - self._last_summary_update > self.summary_refresh_rate):
            self._summary_renderable = self._render_summary()
            self._last_summary_update = now

        if self.show_summary:
            return Group(self._rich_progress, self._summary_renderable)
        else:
            return self._rich_progress

    def add_task(
        self,
        description: str,
        name: Optional[str] = None,
        total: float = 100.0,
        parent: Optional[str] = None,  # Name of parent task
        visible: bool = True,
        start: bool = True,  # Start the Rich task immediately
        **meta: Any,  # Additional metadata
    ) -> str:
        """Add a new task to track.

        Args:
            description: Text description of the task.
            name: Unique name/ID for this task (auto-generated if None).
            total: Total steps/units for completion.
            parent: Name of the parent task for nesting (visual indent).
            visible: Whether the task is initially visible.
            start: Start the task in the Rich progress bar immediately.
            **meta: Arbitrary metadata associated with the task.

        Returns:
            The unique name/ID of the added task.
        """
        if name is None:
            name = str(uuid.uuid4())  # Generate unique ID if not provided

        if name in self._tasks:
            raise ValueError(f"Task with name '{name}' already exists.")

        parent_rich_id = None
        if parent:
            if parent not in self._tasks:
                raise ValueError(f"Parent task '{parent}' not found.")
            parent_task_info = self._tasks[parent]
            if parent_task_info.rich_task_id is not None:
                parent_rich_id = parent_task_info.rich_task_id
                # Quick hack for indentation - needs better Rich integration? Rich doesn't directly support tree view in Progress
                # description = f"  {description}"

        task_info = TaskInfo(
            description=description,
            total=total,
            parent_id=parent,
            meta=meta,
        )

        # Add to Rich Progress if active
        rich_task_id = None
        if self._live and self._rich_progress:
            rich_task_id = self._rich_progress.add_task(
                description,
                total=total,
                start=start,
                visible=visible,
                parent=parent_rich_id,  # Rich uses TaskID for parent
            )
            task_info.rich_task_id = rich_task_id

        self._tasks[name] = task_info
        return name

    def update_task(
        self,
        name: str,
        description: Optional[str] = None,
        advance: Optional[float] = None,
        completed: Optional[float] = None,
        total: Optional[float] = None,
        visible: Optional[bool] = None,
        status: Optional[str] = None,  # running, success, error, skipped
        **meta: Any,
    ) -> None:
        """Update an existing task.

        Args:
            name: The unique name/ID of the task to update.
            description: New description text.
            advance: Amount to advance the completion progress.
            completed: Set completion to a specific value.
            total: Set a new total value.
            visible: Change task visibility.
            status: Update the task status (affects summary).
            **meta: Update or add metadata.
        """
        if name not in self._tasks:
            # Optionally log a warning or error
            # default_console.print(f"[warning]Attempted to update non-existent task: {name}[/]")
            return

        task_info = self._tasks[name]
        update_kwargs = {}

        if description is not None:
            task_info.description = description
            update_kwargs["description"] = description

        if total is not None:
            task_info.total = float(total)
            update_kwargs["total"] = task_info.total

        # Update completed status
        if completed is not None:
            task_info.completed = max(0.0, min(float(completed), task_info.total))
            update_kwargs["completed"] = task_info.completed
        elif advance is not None:
            task_info.completed = max(
                0.0, min(task_info.completed + float(advance), task_info.total)
            )
            update_kwargs["completed"] = task_info.completed

        if visible is not None:
            update_kwargs["visible"] = visible

        if meta:
            task_info.meta.update(meta)

        # Update status (after completion update)
        if status is not None:
            task_info.status = status
            if task_info.is_complete and task_info.end_time is None:
                task_info.end_time = time.time()
                # Ensure Rich task is marked as complete
                if "completed" not in update_kwargs:
                    update_kwargs["completed"] = task_info.total

        # Update Rich progress bar if active
        if task_info.rich_task_id is not None and self._live and self._rich_progress:
            self._rich_progress.update(task_info.rich_task_id, **update_kwargs)

    def complete_task(self, name: str, status: str = "success") -> None:
        """Mark a task as complete with a final status.

        Args:
            name: The unique name/ID of the task.
            status: Final status ('success', 'error', 'skipped').
        """
        if name not in self._tasks:
            return  # Or raise error/log warning

        task_info = self._tasks[name]
        self.update_task(
            name,
            completed=task_info.total,  # Ensure it reaches 100%
            status=status,
        )

    def start(self) -> "GatewayProgress":
        """Start the Rich Live display."""
        if self._live is None:
            # Add any tasks that were created before start()
            for _name, task_info in self._tasks.items():
                if task_info.rich_task_id is None:
                    parent_rich_id = None
                    if task_info.parent_id and task_info.parent_id in self._tasks:
                        parent_rich_id = self._tasks[task_info.parent_id].rich_task_id

                    task_info.rich_task_id = self._rich_progress.add_task(
                        task_info.description,
                        total=task_info.total,
                        completed=task_info.completed,
                        start=True,  # Assume tasks added before start should be started
                        visible=True,  # Assume visible
                        parent=parent_rich_id,
                    )

            self._live = Live(
                self._get_renderable(),
                console=self.console,
                refresh_per_second=10,
                vertical_overflow="visible",
            )
            self._live.start(refresh=True)
        return self

    def stop(self) -> None:
        """Stop the Rich Live display."""
        if self._live is not None:
            # Ensure all running tasks in Rich are marked complete before stopping Live
            # to avoid them getting stuck visually
            if self._rich_progress:
                for task in self._rich_progress.tasks:
                    if not task.finished:
                        self._rich_progress.update(task.id, completed=task.total)

            self._live.stop()
            self._live = None
            # Optional: Clear the Rich Progress tasks?
            # self._rich_progress = self._create_progress(...) # Recreate if needed

    def update(self) -> None:
        """Force a refresh of the Live display (if active)."""
        if self._live:
            self._live.update(self._get_renderable(), refresh=True)

    def reset(self) -> None:
        """Reset the progress tracker, clearing all tasks."""
        self.stop()  # Stop live display
        self._tasks.clear()
        self._task_stack.clear()
        # Recreate Rich progress to clear its tasks
        self._rich_progress = self._create_progress(
            self._rich_progress.transient,
            self._rich_progress.auto_refresh,
            True,  # Assuming expand is derived from console width anyway
        )
        self._summary_renderable = self._render_summary()
        self._last_summary_update = 0.0

    @contextmanager
    def task(
        self,
        description: str,
        name: Optional[str] = None,
        total: float = 100.0,
        parent: Optional[str] = None,
        autostart: bool = True,  # Start Live display if not already started?
        **meta: Any,
    ) -> Generator["GatewayProgress", None, None]:  # Yields self for updates
        """Context manager for a single task.

        Args:
            description: Description of the task.
            name: Optional unique name/ID (auto-generated if None).
            total: Total steps/units for the task.
            parent: Optional parent task name.
            autostart: Start the overall progress display if not running.
            **meta: Additional metadata for the task.

        Yields:
            The GatewayProgress instance itself, allowing updates via `update_task`.
        """
        if autostart and self._live is None:
            self.start()

        task_name = self.add_task(description, name, total, parent, **meta)
        self._task_stack.append(task_name)

        try:
            yield self  # Yield self to allow calling update_task(task_name, ...)
        except Exception:
            # Mark task as errored on exception
            self.complete_task(task_name, status="error")
            raise  # Re-raise the exception
        else:
            # Mark task as successful if no exception
            # Check if it was already completed with a different status
            if task_name in self._tasks and not self._tasks[task_name].is_complete:
                self.complete_task(task_name, status="success")
        finally:
            # Pop task from stack
            if self._task_stack and self._task_stack[-1] == task_name:
                self._task_stack.pop()
            # No automatic stop here - allow multiple context managers
            # self.stop()

    def track(
        self,
        iterable: Iterable[T],
        description: str,
        name: Optional[str] = None,
        total: Optional[float] = None,
        parent: Optional[str] = None,
        autostart: bool = True,
        **meta: Any,
    ) -> Iterable[T]:
        """Track progress over an iterable.

        Args:
            iterable: The iterable to track progress over.
            description: Description of the task.
            name: Optional unique name/ID (auto-generated if None).
            total: Total number of items (estimated if None).
            parent: Optional parent task name.
            autostart: Start the overall progress display if not running.
            **meta: Additional metadata for the task.

        Returns:
            The iterable, yielding items while updating progress.
        """
        if total is None:
            try:
                total = float(len(iterable))  # type: ignore
            except (TypeError, AttributeError):
                total = 100.0  # Default if length cannot be determined

        if autostart and self._live is None:
            self.start()

        task_name = self.add_task(description, name, total, parent, **meta)

        try:
            for item in iterable:
                yield item
                self.update_task(task_name, advance=1)
        except Exception:
            self.complete_task(task_name, status="error")
            raise
        else:
            # Check if it was already completed with a different status
            if task_name in self._tasks and not self._tasks[task_name].is_complete:
                self.complete_task(task_name, status="success")
        # No automatic stop
        # finally:
        # self.stop()

    def __enter__(self) -> "GatewayProgress":
        """Enter context manager, starts the display."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, stops the display."""
        self.stop()


# --- Global Convenience Functions (using a default progress instance) ---
# Note: Managing a truly global progress instance can be tricky.
# It might be better to explicitly create and manage GatewayProgress instances.
_global_progress: Optional[GatewayProgress] = None


def get_global_progress() -> GatewayProgress:
    """Get or create the default global progress manager."""
    global _global_progress
    if _global_progress is None:
        _global_progress = GatewayProgress()
    return _global_progress


def track(
    iterable: Iterable[T],
    description: str,
    name: Optional[str] = None,
    total: Optional[float] = None,
    parent: Optional[str] = None,
) -> Iterable[T]:
    """Track progress over an iterable using the global progress manager."""
    prog = get_global_progress()
    # Ensure global progress is started if used this way
    if prog._live is None:
        prog.start()
    return prog.track(iterable, description, name, total, parent, autostart=False)


@contextmanager
def task(
    description: str,
    name: Optional[str] = None,
    total: float = 100.0,
    parent: Optional[str] = None,
) -> Generator["GatewayProgress", None, None]:
    """Context manager for a single task using the global progress manager."""
    prog = get_global_progress()
    # Ensure global progress is started if used this way
    if prog._live is None:
        prog.start()
    with prog.task(description, name, total, parent, autostart=False) as task_context:
        yield task_context  # Yields the progress manager itself
