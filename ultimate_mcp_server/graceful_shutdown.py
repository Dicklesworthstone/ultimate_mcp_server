"""
Graceful shutdown utilities for Ultimate MCP Server.

This module provides utilities to handle signals and gracefully terminate
the application with ZERO error outputs during shutdown using OS-level redirection.
"""

import asyncio
import logging
import os
import signal
import sys
import warnings
from contextlib import suppress
from typing import Callable, List, Optional

logger = logging.getLogger("ultimate_mcp_server.shutdown")

# Track registered shutdown handlers and state
_shutdown_handlers: List[Callable] = []
_shutdown_in_progress = False
_original_stderr_fd = None
_devnull_fd = None


def _redirect_stderr_to_devnull():
    """Redirect stderr to /dev/null at the OS level"""
    global _original_stderr_fd, _devnull_fd

    try:
        if _original_stderr_fd is None:
            # Save original stderr file descriptor
            _original_stderr_fd = os.dup(sys.stderr.fileno())

            # Open /dev/null
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)

            # Redirect stderr to /dev/null
            os.dup2(_devnull_fd, sys.stderr.fileno())

    except Exception:
        # If redirection fails, just continue
        pass


def _restore_stderr():
    """Restore original stderr"""
    global _original_stderr_fd, _devnull_fd

    try:
        if _original_stderr_fd is not None:
            os.dup2(_original_stderr_fd, sys.stderr.fileno())
            os.close(_original_stderr_fd)
            _original_stderr_fd = None

        if _devnull_fd is not None:
            os.close(_devnull_fd)
            _devnull_fd = None

    except Exception:
        pass


def register_shutdown_handler(handler: Callable) -> None:
    """Register a function to be called during graceful shutdown."""
    if handler not in _shutdown_handlers:
        _shutdown_handlers.append(handler)


def remove_shutdown_handler(handler: Callable) -> None:
    """Remove a previously registered shutdown handler."""
    if handler in _shutdown_handlers:
        _shutdown_handlers.remove(handler)


async def _execute_shutdown_handlers():
    """Execute all registered shutdown handlers with complete error suppression"""
    for handler in _shutdown_handlers:
        with suppress(Exception):  # Suppress ALL exceptions
            if asyncio.iscoroutinefunction(handler):
                with suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(handler(), timeout=3.0)
            else:
                handler()


def _handle_shutdown_signal(signum, frame):
    """Handle shutdown signals - IMMEDIATE TERMINATION"""
    global _shutdown_in_progress

    if _shutdown_in_progress:
        # Force immediate exit on second signal
        os._exit(1)
        return

    _shutdown_in_progress = True

    # Print final message to original stderr if possible
    try:
        if _original_stderr_fd:
            os.write(_original_stderr_fd, b"\n[Graceful Shutdown] Signal received. Exiting...\n")
        else:
            print("\n[Graceful Shutdown] Signal received. Exiting...", file=sys.__stderr__)
    except Exception:
        pass

    # Immediately redirect stderr to suppress any error output
    _redirect_stderr_to_devnull()

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Try to run shutdown handlers quickly, but don't wait long
    try:
        loop = asyncio.get_running_loop()
        # Create a task but don't wait for it - just exit
        asyncio.create_task(_execute_shutdown_handlers())
        # Give it a tiny bit of time then exit
        loop.call_later(0.5, lambda: os._exit(0))
    except RuntimeError:
        # No running loop - just exit immediately
        os._exit(0)


def setup_signal_handlers(loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """Set up signal handlers for immediate shutdown"""
    # Use traditional signal handlers for immediate termination
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    # Also try to set up async handlers if we have a loop
    if loop is not None:
        try:
            for sig in [signal.SIGINT, signal.SIGTERM]:
                try:
                    loop.add_signal_handler(sig, lambda s=sig: _handle_shutdown_signal(s, None))
                except (NotImplementedError, OSError):
                    # Platform doesn't support async signal handlers
                    pass
        except Exception:
            # Fallback is already set up with signal.signal above
            pass


def enable_quiet_shutdown():
    """Enable comprehensive quiet shutdown - immediate termination approach"""
    # Set up signal handlers immediately
    setup_signal_handlers()

    # Suppress asyncio debug mode
    try:
        asyncio.get_event_loop().set_debug(False)
    except RuntimeError:
        pass

    # Suppress warnings
    warnings.filterwarnings("ignore")


def force_silent_exit():
    """Force immediate silent exit with no output whatsoever"""
    global _shutdown_in_progress
    _shutdown_in_progress = True
    _redirect_stderr_to_devnull()
    os._exit(0)


class QuietUvicornServer:
    """Custom Uvicorn server that overrides signal handling for quiet shutdown"""

    def __init__(self, config):
        import uvicorn

        self.config = config
        self.server = uvicorn.Server(config)

    def install_signal_handlers(self):
        """Override uvicorn's signal handlers with our quiet ones"""
        # Set up our own signal handlers instead of uvicorn's
        setup_signal_handlers()

    def run(self):
        """Run the server with custom signal handling"""
        # Patch the server's install_signal_handlers method
        self.server.install_signal_handlers = self.install_signal_handlers

        # Set up our signal handlers immediately
        setup_signal_handlers()

        # Run the server
        self.server.run()


def create_quiet_server(config):
    """Create a uvicorn server with quiet shutdown handling"""
    return QuietUvicornServer(config)
