# ultimate_mcp_server/tools/python_sandbox.py

"""Pyodide-backed sandbox tool for Ultimate MCP Server.

Provides a secure environment for executing Python code within a headless browser,
with stdout/stderr capture, package management, security controls, and optional REPL functionality.

Includes integrated offline asset caching for Pyodide.
"""

###############################################################################
# Standard library & typing
###############################################################################
import argparse
import asyncio
import atexit
import base64
import collections
import gzip
import hashlib
import json
import logging  # Import logging for fallback
import mimetypes
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, OrderedDict

# ---------------------------------

###############################################################################
# Third‑party – runtime dependency only on Playwright
###############################################################################
try:
    import playwright.async_api as pw

    if TYPE_CHECKING:
        # Import SPECIFIC types for type hints inside TYPE_CHECKING
        from playwright.async_api import Browser, Page, Request, Route
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pw = None
    # Define placeholder types ONLY if playwright is unavailable,
    # and inside TYPE_CHECKING if you still want hints to reference *something*
    # Although the imports above should handle this for the type checker.
    if TYPE_CHECKING:
        Browser = Any
        Page = Any
        Route = Any
        Request = Any
    PLAYWRIGHT_AVAILABLE = False

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

###############################################################################
# Project specific imports
###############################################################################
# Assuming these are correctly located within your project structure
try:
    from ultimate_mcp_server.constants import TaskType
    from ultimate_mcp_server.exceptions import (
        ProviderError,
        ToolError,
        ToolInputError,
    )
    from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
    from ultimate_mcp_server.utils import get_logger
except ImportError as e:
    # Provide a fallback or clearer error if these imports fail
    print(f"WARNING: Failed to import Ultimate MCP Server components: {e}")
    print("Running in standalone mode or environment misconfiguration.")

    # Define dummy logger/decorators if running standalone for preloading
    def get_logger(name):
        _logger = logging.getLogger(name)
        if not _logger.handlers:  # Setup basic config only if no handlers exist
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        return _logger

    def with_tool_metrics(func):
        return func

    def with_error_handling(func):
        return func

    # Define dummy exceptions
    class ProviderError(Exception):
        pass

    class ToolError(Exception):
        pass

    class ToolInputError(Exception):
        pass

    class TaskType:
        CODE_EXECUTION = "code_execution"  # Dummy enum value


from ultimate_mcp_server.utils.logging.console import console

logger = get_logger("ultimate_mcp_server.tools.python_sandbox")

# Constant for posting messages back to the sandbox page context
JS_POST_MESSAGE = "(msg) => globalThis.postMessage(msg, '*')"

###############################################################################
# Constants & Caching Configuration
###############################################################################
COMMON_PACKAGES: list[str] = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "networkx",
]
# Define JSON string *after* COMMON_PACKAGES is defined
COMMON_PACKAGES_JSON = json.dumps(COMMON_PACKAGES)

MAX_SANDBOXES = 6  # Max number of concurrent browser tabs/sandboxes
GLOBAL_CONCURRENCY = 8  # Max number of simultaneous code executions across all sandboxes
MEM_LIMIT_MB = 512  # Memory limit for the heap watchdog in the browser tab

# --- Pyodide Version and CDN ---
_PYODIDE_VERSION = "0.27.5"  # <<< Ensure this matches the intended version
_CDN_BASE = f"https://cdn.jsdelivr.net/pyodide/v{_PYODIDE_VERSION}/full"
# Note: PYODIDE_CDN variable might not be strictly necessary if importing .mjs directly
PYODIDE_CDN = f"{_CDN_BASE}/pyodide.js"

# --- Define the packages to be loaded AT STARTUP ---
# These will be baked into the loadPyodide call via the template
CORE_PACKAGES_TO_LOAD_AT_STARTUP: list[str] = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "networkx",
    "micropip",  # Good to include if you often load wheels later
]
# Generate the JSON string to be injected into the HTML template
CORE_PACKAGES_JSON_FOR_TEMPLATE = json.dumps(CORE_PACKAGES_TO_LOAD_AT_STARTUP)

# --- Asset Caching Configuration ---
_CACHE_DIR = (
    pathlib.Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
    / "ultimate_mcp_server"
    / "pyodide"
    / _PYODIDE_VERSION  # Versioned cache directory
)
try:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using Pyodide asset cache directory: {_CACHE_DIR}")
except OSError as e:
    logger.error(
        f"Failed to create Pyodide asset cache directory {_CACHE_DIR}: {e}. Caching might fail."
    )

################################################################################
# Diagnostic logging helpers
################################################################################
# level 0 = quiet, 1 = basic req/resp, 2 = full body/hex dumps
_VERBOSE_SANDBOX_LOGGING = int(os.getenv("PYODIDE_SANDBOX_DEBUG", "0") or 0)


def _wire_page_logging(page: "Page", session_id: str) -> None:  # type: ignore
    """
    Mirrors everything interesting coming out of the browser tab back into our
    Python logger. When PYODIDE_SANDBOX_DEBUG=2 we also dump request/response
    headers and first 64 bytes of every body.
    """

    # ───────── console / JS errors ───────────────────────────────────────────
    def _log_console(msg):
        try:
            # Safely access properties, defaulting if necessary
            lvl = msg.type if not callable(getattr(msg, "type", None)) else msg.type()
            txt = msg.text if not callable(getattr(msg, "text", None)) else msg.text()
            loc = msg.location if not callable(getattr(msg, "location", None)) else msg.location()

            src = ""
            if isinstance(loc, dict):
                src = f"{loc.get('url', '')}:{loc.get('lineNumber', '?')}:{loc.get('columnNumber', '?')}"
            elif loc:
                src = str(loc)

            line = f"SB[{session_id}] {src} ▶ {txt}" if src else f"SB[{session_id}] ▶ {txt}"

            log_func = {
                "error": logger.error,
                "warning": logger.warning,
                "warn": logger.warning,
                "info": logger.info,
                "log": logger.info,
                "debug": logger.debug,
                "trace": logger.debug,
            }.get(str(lvl).lower(), logger.debug)

            log_func(line)
        except Exception as e:
            logger.error(f"SB[{session_id}] Error in console message processing: {e}")

    try:
        page.on("console", _log_console)
        page.on(
            "pageerror",
            lambda e: logger.error(f"SB[{session_id}] PageError ▶ {e.message}\n{e.stack}"),
        )
        page.on("crash", lambda: logger.critical(f"SB[{session_id}] **PAGE CRASHED**"))
    except Exception as e:
        logger.error(f"SB[{session_id}] Failed to attach basic page log listeners: {e}")

    # ───────── high-level net trace ─────────────────────────────────────────
    if _VERBOSE_SANDBOX_LOGGING > 0:
        try:
            page.on("request", lambda r: logger.debug(f"SB[{session_id}] → {r.method} {r.url}"))
            page.on(
                "requestfailed",
                lambda r: logger.warning(f"SB[{session_id}] ✗ {r.method} {r.url} ▶ {r.failure}"),
            )

            async def _resp_logger(resp: "pw.Response"):  # type: ignore # Use string literal hint
                try:
                    status = resp.status
                    url = resp.url
                    if status == 200 and url.startswith("data:") and _VERBOSE_SANDBOX_LOGGING < 2:
                        return

                    # Use resp.all_headers() which returns a dict directly
                    hdrs = await resp.all_headers()
                    ce = hdrs.get("content-encoding", "")
                    ct = hdrs.get("content-type", "")
                    log_line = (
                        f"SB[{session_id}] ← {status} {url} (type='{ct}', enc='{ce or 'none'}')"
                    )

                    if _VERBOSE_SANDBOX_LOGGING > 1 or status >= 400:  # Log body for errors too
                        try:
                            body = await resp.body()
                            sig = body[:64]
                            hexs = " ".join(f"{b:02x}" for b in sig)
                            log_line += f" (len={len(body)}, first-64: {hexs})"
                        except Exception as body_err:
                            # Handle cases where body might not be available (e.g., redirects)
                            log_line += f" (body unavailable: {body_err})"
                    logger.debug(log_line)
                except Exception as e:
                    logger.warning(f"SB[{session_id}] Error in response logger: {e}")

            page.on("response", lambda r: asyncio.create_task(_resp_logger(r)))
        except Exception as e:
            logger.error(f"SB[{session_id}] Failed to attach network trace log listeners: {e}")


###############################################################################
# Asset Caching Helper Functions (Integrated)
###############################################################################


def _local_path(remote_url: str) -> pathlib.Path:
    """Generates the local cache path for a given remote URL."""
    try:
        parsed_url = urllib.parse.urlparse(remote_url)
        path_part = parsed_url.path if parsed_url.path else "/"
        fname = pathlib.Path(path_part).name
        if not fname or fname == "/":
            fname = hashlib.md5(remote_url.encode()).hexdigest() + ".cache"
            logger.debug(
                f"No filename in path '{path_part}', using hash '{fname}' for {remote_url}"
            )
    except Exception as e:
        logger.warning(f"Error parsing URL '{remote_url}' for filename: {e}. Falling back to hash.")
        fname = hashlib.md5(remote_url.encode()).hexdigest() + ".cache"

    return _CACHE_DIR / fname


def _fetch_asset_sync(remote_url: str, max_age_s: int = 7 * 24 * 3600) -> bytes:
    """
    Synchronous version: Return requested asset from cache or download.
    Used by Playwright interceptor and preloader.
    """
    p = _local_path(remote_url)
    use_cache = False
    if p.exists():
        try:
            file_stat = p.stat()
            file_age = time.time() - file_stat.st_mtime
            if file_age < max_age_s:
                if file_stat.st_size > 0:
                    logger.debug(
                        f"[Cache] HIT for {remote_url} (age: {file_age:.0f}s < {max_age_s}s)"
                    )
                    use_cache = True
                else:
                    logger.warning(
                        f"[Cache] Hit for {remote_url}, but file is empty. Re-downloading."
                    )
            else:
                logger.info(
                    f"[Cache] STALE for {remote_url} (age: {file_age:.0f}s >= {max_age_s}s)"
                )
        except OSError as e:
            logger.warning(f"[Cache] Error accessing cache file {p}: {e}. Will attempt download.")

    if use_cache:
        try:
            return p.read_bytes()
        except OSError as e:
            logger.warning(f"[Cache] Error reading cached file {p}: {e}. Will attempt download.")

    logger.info(f"[Cache] MISS or STALE/Error for {remote_url}. Downloading...")
    downloaded_data = None
    try:
        req = urllib.request.Request(
            remote_url,
            headers={"User-Agent": "UltimateMCPServer-AssetCache/1.0", "Accept-Encoding": "gzip"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise urllib.error.HTTPError(
                    remote_url, resp.status, resp.reason, resp.headers, None
                )
            downloaded_data = resp.read()
            # Handle potential gzip encoding from server
            if resp.headers.get("Content-Encoding") == "gzip":
                try:
                    downloaded_data = gzip.decompress(downloaded_data)
                    logger.debug(f"[Cache] Decompressed gzip response for {remote_url}")
                except gzip.BadGzipFile:
                    logger.warning(
                        f"[Cache] Received gzip header but invalid gzip data for {remote_url}. Using raw."
                    )
                except Exception as gz_err:
                    logger.warning(
                        f"[Cache] Error decompressing gzip for {remote_url}: {gz_err}. Using raw."
                    )

            logger.info(
                f"[Cache] Downloaded {len(downloaded_data)} bytes from {remote_url} (status: {resp.status})"
            )

    except urllib.error.HTTPError as e:
        logger.warning(f"[Cache] HTTP error downloading {remote_url}: {e.code} {e.reason}")
        if p.exists():
            try:
                stale_stat = p.stat()
                if stale_stat.st_size > 0:
                    logger.warning(
                        f"[Cache] Using STALE cache file {p} as fallback due to HTTP {e.code}."
                    )
                    return p.read_bytes()
            except OSError as read_err:
                logger.error(
                    f"[Cache] Failed reading fallback cache {p} after download error: {read_err}"
                )
        raise RuntimeError(
            f"Cannot download {remote_url} (HTTP {e.code}) and no usable cache available"
        ) from e
    except urllib.error.URLError as e:
        logger.warning(f"[Cache] Network error downloading {remote_url}: {e.reason}")
        if p.exists():
            try:
                stale_stat = p.stat()
                if stale_stat.st_size > 0:
                    logger.warning(
                        f"[Cache] Using STALE cache file {p} as fallback due to network error."
                    )
                    return p.read_bytes()
            except OSError as read_err:
                logger.error(
                    f"[Cache] Failed reading fallback cache {p} after network error: {read_err}"
                )
        raise RuntimeError(
            f"Cannot download {remote_url} ({e.reason}) and no usable cache available"
        ) from e
    except Exception as e:
        logger.error(f"[Cache] Unexpected error downloading {remote_url}: {e}", exc_info=True)
        if p.exists():
            try:
                stale_stat = p.stat()
                if stale_stat.st_size > 0:
                    logger.warning(
                        f"[Cache] Using STALE cache file {p} as fallback due to unexpected error."
                    )
                    return p.read_bytes()
            except OSError as read_err:
                logger.error(
                    f"[Cache] Failed reading fallback cache {p} after unexpected error: {read_err}"
                )
        raise RuntimeError(
            f"Cannot download {remote_url} (unexpected error: {e}) and no usable cache available"
        ) from e

    if downloaded_data is not None:
        try:
            tmp_suffix = f".tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}"
            tmp_path = p.with_suffix(p.suffix + tmp_suffix)
            tmp_path.write_bytes(downloaded_data)
            tmp_path.replace(p)
            logger.info(f"[Cache] Saved {len(downloaded_data)} bytes for {remote_url} to {p}")
        except OSError as e:
            logger.error(f"[Cache] Failed write cache file {p}: {e}")
        return downloaded_data
    else:
        raise RuntimeError(f"Download completed for {remote_url} but data is None (internal error)")


###############################################################################
# Browser / bookkeeping singletons
###############################################################################
_BROWSER: Optional["Browser"] = None  # type: ignore # Use string literal hint
_PAGES: OrderedDict[str, "PyodideSandbox"] = collections.OrderedDict()
_GLOBAL_SEM: Optional[asyncio.Semaphore] = None


###############################################################################
# PyodideSandbox Class Definition
###############################################################################
@dataclass(slots=True)
class PyodideSandbox:
    """One Chromium tab with Pyodide runtime (optionally persistent)."""

    page: "Page"  # type: ignore # Use string literal hint
    allow_network: bool = False
    allow_fs: bool = False
    ready_evt: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    _init_timeout: int = 90
    _message_handlers: Dict[str, asyncio.Queue] = field(default_factory=dict)
    _init_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    async def init(self):
        """Load boot HTML, set up messaging (direct callback), and wait for ready signal."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright is not installed. Cannot initialize sandbox.")

        logger.info(f"Initializing PyodideSandbox instance (Page: {self.page.url})...")
        init_start_time = time.monotonic()

        # === 1. Network Interception Setup ===
        logger.debug("Setting up network interception...")
        try:
            # Define the interception logic inline or call an external helper
            cdn_base_lower = _CDN_BASE.lower()

            async def _block(route: "Route", request: "Request"):  # type: ignore
                url = request.url
                low = url.lower()
                is_cdn = low.startswith(cdn_base_lower)
                is_pypi = "pypi.org/simple" in low or "files.pythonhosted.org" in low

                if is_cdn:
                    try:
                        # Assuming _fetch_asset_sync is correctly defined elsewhere
                        body = _fetch_asset_sync(url)
                        ctype = mimetypes.guess_type(url)[0] or "application/octet-stream"
                        headers = {
                            "Content-Type": ctype,
                            "Access-Control-Allow-Origin": "*",
                            "Cache-Control": "public, max-age=31536000",
                        }
                        # Simple check for gzip magic bytes; don't decompress here, let browser handle it
                        if body.startswith(b"\x1f\x8b"):
                            headers["Content-Encoding"] = "gzip"

                        if _VERBOSE_SANDBOX_LOGGING > 1:
                            logger.debug(
                                f"[Intercept] FULFILL CDN {url} (type={ctype}, enc={headers.get('Content-Encoding', 'none')}, len={len(body)})"
                            )
                        await route.fulfill(status=200, body=body, headers=headers)
                        return
                    except Exception as exc:
                        logger.error(
                            f"[Intercept] FAILED serving CDN {url} from cache/download: {exc}",
                            exc_info=_VERBOSE_SANDBOX_LOGGING > 1,
                        )
                        await route.abort(error_code="failed")
                        return

                # Allow PyPI only if explicitly enabled
                if self.allow_network and is_pypi:
                    if _VERBOSE_SANDBOX_LOGGING > 0:
                        logger.debug(f"[Intercept] ALLOW PyPI {url}")
                    try:
                        await route.continue_()
                    except Exception as cont_err:
                        logger.warning(
                            f"[Intercept] Error continuing PyPI request {url}: {cont_err}"
                        )
                        try:
                            await route.abort(error_code="failed")
                        except Exception:
                            pass
                    return

                # Block other network requests by default
                # Log less aggressively for common browser noise
                if not any(low.endswith(ext) for ext in [".ico", ".png", ".woff", ".woff2"]):
                    if _VERBOSE_SANDBOX_LOGGING > 0:
                        logger.debug(f"[Intercept] BLOCK {url}")
                try:
                    await route.abort(error_code="blockedbyclient")
                except Exception:
                    pass  # Ignore errors aborting (e.g., already handled)

            await self.page.route("**/*", _block)
            logger.info("Network interception active.")
        except Exception as e:
            logger.error(f"Failed to set up network interception: {e}", exc_info=True)
            await self._try_close_page("Network Intercept Setup Error")
            raise ToolError(f"Failed to configure sandbox network rules: {e}") from e

        # === 2. Load Boot HTML ===
        logger.debug("Loading boot HTML template...")
        try:
            template_path = pathlib.Path(__file__).parent / "pyodide_boot_template.html"
            if not template_path.is_file():
                raise FileNotFoundError(f"Boot template not found at {template_path}")
            boot_html_template = template_path.read_text(encoding="utf-8")

            # Replace placeholders, including the CORE packages JSON
            processed_boot_html = (
                boot_html_template.replace("__CDN_BASE__", _CDN_BASE)
                .replace("__PYODIDE_VERSION__", _PYODIDE_VERSION)
                # *** Use the new constant for core packages ***
                .replace("__CORE_PACKAGES_JSON__", CORE_PACKAGES_JSON_FOR_TEMPLATE)
                .replace("__MEM_LIMIT_MB__", str(MEM_LIMIT_MB))  # Keep MEM_LIMIT if using watchdog
            )

            # Check essential placeholders
            essential_placeholders = ["__CDN_BASE__", "__PYODIDE_VERSION__"]
            # Check optional placeholders based on template features
            optional_placeholders = ["__CORE_PACKAGES_JSON__", "__MEM_LIMIT_MB__"]
            missing_essential = [p for p in essential_placeholders if p in processed_boot_html]
            missing_optional = [p for p in optional_placeholders if p in processed_boot_html]

            if missing_essential:
                logger.critical(
                    f"CRITICAL: Essential placeholders missing in boot HTML: {missing_essential}. Aborting."
                )
                raise ToolError(
                    f"Essential placeholders missing in boot template: {missing_essential}"
                )
            if missing_optional:
                logger.warning(
                    f"Optional placeholders missing in boot HTML: {missing_optional}. Check template if features are expected."
                )

            await self.page.set_content(
                processed_boot_html,
                wait_until="domcontentloaded",
                timeout=60000,  # Slightly longer timeout for package loading
            )
            logger.info("Boot HTML loaded into page.")
        except FileNotFoundError as e:
            logger.error(f"Failed to load boot HTML template: {e}", exc_info=True)
            await self._try_close_page("Boot HTML Template Not Found")
            raise ToolError(f"Could not find sandbox boot HTML template: {e}") from e
        except Exception as e:
            logger.error(f"Failed loading boot HTML content: {e}", exc_info=True)
            await self._try_close_page("Boot HTML Load Error")
            raise ToolError(f"Could not load sandbox boot HTML content: {e}") from e

        # === 3. Setup Communication Channels ===
        # This involves two parts:
        #   a) Exposing a Python function for JS to send *replies* directly.
        #   b) Exposing a Python function for JS to send the initial *ready/error* signal.

        # --- 3a. Setup for Execution Replies ---
        logger.debug("Setting up direct reply mechanism (JS->Python)...")
        try:
            # This Python function will be called by JavaScript's `window._deliverReplyToHost(reply)`
            async def _deliver_reply_to_host(payload: Any):
                msg_id = None  # Define outside try block
                try:
                    if not isinstance(payload, dict):
                        if _VERBOSE_SANDBOX_LOGGING > 1:
                            logger.debug(f"Host received non-dict reply payload: {type(payload)}")
                        return
                    data = payload
                    msg_id = data.get("id")
                    if not msg_id:
                        logger.warning(f"Host received reply payload without an ID: {data}")
                        return

                    # Log received reply
                    if _VERBOSE_SANDBOX_LOGGING > 0:
                        log_detail = (
                            f"ok={data.get('ok')}"
                            if _VERBOSE_SANDBOX_LOGGING == 1
                            else json.dumps(data, default=str)
                        )
                        logger.debug(
                            f"Host received reply via exposed function (id: {msg_id}): {log_detail}"
                        )

                    # Route reply to the waiting asyncio Queue in _message_handlers
                    if msg_id in self._message_handlers:
                        await self._message_handlers[msg_id].put(data)
                        if _VERBOSE_SANDBOX_LOGGING > 0:
                            logger.debug(f"Reply payload for ID {msg_id} routed.")
                    elif _VERBOSE_SANDBOX_LOGGING > 0:
                        logger.debug(
                            f"Host received reply for unknown/stale execution ID: {msg_id}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing execution reply payload (id: {msg_id or 'unknown'}) from sandbox: {e}",
                        exc_info=True,
                    )

            reply_handler_name = "_deliverReplyToHost"  # Must match the name called in JS template
            await self.page.expose_function(reply_handler_name, _deliver_reply_to_host)
            logger.info(f"Python function '{reply_handler_name}' exposed for JS execution replies.")

        except Exception as e:
            logger.error(f"Failed to expose reply handler function: {e}", exc_info=True)
            await self._try_close_page("Reply Handler Setup Error")
            raise ToolError(f"Could not expose reply handler to sandbox: {e}") from e

        # --- 3b. Setup for Initial Ready/Error Signal ---
        # The JS template sends the initial 'pyodide_ready' or 'pyodide_init_error' via postMessage.
        # We need a way to capture *only* that specific message and put it on _init_queue.
        logger.debug("Setting up listener for initial ready/error signal (JS->Python)...")
        try:
            # This Python function will be called by the JS listener below
            async def _handle_initial_message(payload: Any):
                try:
                    if not isinstance(payload, dict):
                        return  # Ignore non-dicts
                    msg_id = payload.get("id")
                    if msg_id == "pyodide_ready" or msg_id == "pyodide_init_error":
                        log_level = logger.info if payload.get("ready") else logger.error
                        log_level(
                            f"Received initial status message from sandbox via exposed function: {payload}"
                        )
                        await self._init_queue.put(payload)  # Put it on the init queue
                        # Optionally remove the listener after receiving the first signal? Might be risky.
                    # Ignore other messages potentially caught by this listener
                except Exception as e:
                    logger.error(
                        f"Error processing initial message from sandbox: {e}", exc_info=True
                    )
                    # Put an error onto the queue to unblock init waiter
                    await self._init_queue.put(
                        {
                            "id": "pyodide_init_error",
                            "ok": False,
                            "error": {
                                "type": "HostProcessingError",
                                "message": f"Error handling init message: {e}",
                            },
                        }
                    )

            init_handler_name = "_handleInitialMessage"
            await self.page.expose_function(init_handler_name, _handle_initial_message)

            # Evaluate JavaScript to add a *specific* listener that calls the exposed init handler
            await self.page.evaluate(f"""
                console.log('[PyodideBoot] Adding specific listener for initial ready/error messages...');
                // Ensure we don't add multiple listeners if init is somehow re-run
                if (!window._initialMessageListenerAdded) {{
                    window.addEventListener('message', (event) => {{
                        const data = event.data;
                        // Check if the exposed function exists and if it's the specific message we want
                        if (typeof window.{init_handler_name} === 'function' &&
                            typeof data === 'object' && data !== null &&
                            (data.id === 'pyodide_ready' || data.id === 'pyodide_init_error'))
                        {{
                            // Forward only specific initial messages to the exposed Python function
                            console.log('[PyodideBoot] Forwarding initial message to host:', data.id);
                            window.{init_handler_name}(data);
                        }}
                    }});
                    window._initialMessageListenerAdded = true; // Flag to prevent multiple adds
                    console.log('[PyodideBoot] Initial message listener added.');
                }} else {{
                    console.log('[PyodideBoot] Initial message listener already added.');
                }}
            """)
            logger.info(
                f"Python function '{init_handler_name}' exposed and JS listener added for initial signal."
            )

        except Exception as e:
            logger.error(f"Failed to set up initial signal listener: {e}", exc_info=True)
            await self._try_close_page("Initial Signal Listener Setup Error")
            raise ToolError(f"Could not set up initial signal listener: {e}") from e

        # === 4. Wait for Ready Signal ===
        logger.info(f"Waiting for sandbox ready signal (timeout: {self._init_timeout}s)...")
        try:
            # Wait for a message to appear on the _init_queue
            init_data = await asyncio.wait_for(self._init_queue.get(), timeout=self._init_timeout)

            # Check the content of the message
            if init_data.get("id") == "pyodide_init_error" or init_data.get("ok") is False:
                error_details = init_data.get(
                    "error", {"message": "Unknown initialization error reported by sandbox."}
                )
                error_msg = error_details.get("message", "Unknown Error")
                logger.error(f"Pyodide sandbox initialization failed inside browser: {error_msg}")
                await self._try_close_page("Initialization Error Reported by JS")
                raise ToolError(f"Pyodide sandbox initialization failed: {error_msg}")

            if not init_data.get("ready"):
                logger.error(f"Received unexpected init message without 'ready' flag: {init_data}")
                await self._try_close_page("Unexpected Init Message from JS")
                raise ToolError("Received unexpected initialization message from sandbox.")

            # If we received the correct ready message
            self.ready_evt.set()  # Set the event flag
            boot_ms_reported = init_data.get("boot_ms", "N/A")
            init_duration = time.monotonic() - init_start_time
            logger.info(
                f"Pyodide sandbox ready signal received (reported boot: {boot_ms_reported}ms, total init wait: {init_duration:.2f}s)"
            )

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout ({self._init_timeout}s) waiting for Pyodide ready signal.")
            await self._check_page_responsiveness(
                "Timeout Waiting for Ready"
            )  # Check if page is stuck
            await self._try_close_page("Timeout Waiting for Ready")
            raise ToolError(
                f"Sandbox failed to initialize within timeout ({self._init_timeout}s)."
            ) from e
        except Exception as e:
            # Catch other errors during the wait/processing phase
            logger.error(f"Error during sandbox initialization wait: {e}", exc_info=True)
            await self._try_close_page("Initialization Wait Error")
            if isinstance(e, ToolError):  # Don't wrap existing ToolErrors
                raise e
            raise ToolError(f"Unexpected error during sandbox initialization wait: {e}") from e

    async def _check_page_responsiveness(self, context: str) -> bool:  # Return boolean
        """Tries to evaluate a simple JS command to check if the page is alive."""
        if self.page and not self.page.is_closed():
            try:
                await asyncio.wait_for(self.page.evaluate("1+1"), timeout=5.0)
                logger.debug(f"Page responded after {context}.")
                return True
            except Exception as page_err:
                logger.error(f"Page seems unresponsive after {context}: {page_err}")
                return False
        else:
            logger.debug(
                f"Page already closed or non-existent during responsiveness check ({context})."
            )
            return False  # Not responsive if closed

    async def _try_close_page(self, reason: str):
        """Attempts to close the sandbox page, logging errors."""
        if self.page and not self.page.is_closed():
            logger.info(f"Attempting to close sandbox page due to: {reason}")
            try:
                await self.page.close()
                logger.info(f"Sandbox page closed successfully after {reason}.")
            except Exception as close_err:
                logger.warning(f"Error closing page after {reason}: {close_err}")
        else:
            logger.debug(
                f"Page already closed or non-existent when trying to close due to: {reason}"
            )

    async def execute(
        self,
        code: str,
        packages: list[str] | None,
        wheels: list[str] | None,
        timeout_ms: int,
        repl_mode: bool = False,
    ) -> Dict[str, Any]:
        """Sends code to the sandbox for execution and returns the result."""
        if not PLAYWRIGHT_AVAILABLE:
            # This condition should ideally be checked before creating/getting the sandbox
            # but is included here for robustness.
            raise ToolError("Playwright is not installed.")
        if not self.page or self.page.is_closed():
            raise ToolError("Cannot execute code: Sandbox page is closed.")
        if not self.ready_evt.is_set():
            # Wait briefly for the ready event if it's not set yet, in case of race conditions
            try:
                await asyncio.wait_for(self.ready_evt.wait(), timeout=1.0)
            except asyncio.TimeoutError as e:
                raise ToolError(
                    "Cannot execute code: Sandbox is not ready (or timed out becoming ready)."
                ) from e

        self.last_used = time.time()
        global _GLOBAL_SEM
        if _GLOBAL_SEM is None:
            # Initialize if it hasn't been already (should be done in _get_sandbox, but safety check)
            logger.warning("Global execution semaphore not initialized, initializing now.")
            _GLOBAL_SEM = asyncio.Semaphore(GLOBAL_CONCURRENCY)

        # Acquire the semaphore to limit concurrency across all sandboxes
        async with _GLOBAL_SEM:
            exec_id = f"exec-{uuid.uuid4().hex[:8]}"
            response_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
            self._message_handlers[exec_id] = response_queue

            try:
                # Encode the user's Python code to Base64
                code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
            except Exception as enc_err:
                # If encoding fails, it's an input error, no need to involve the sandbox
                self._message_handlers.pop(exec_id, None)  # Clean up handler
                raise ToolInputError(f"Failed to encode code to base64: {enc_err}") from enc_err

            # Prepare the message payload for the JavaScript side
            payload = {
                "type": "exec",
                "id": exec_id,
                "code_b64": code_b64,
                "packages": packages or [],
                "wheels": wheels or [],
                "repl_mode": repl_mode,
            }
            data: dict[str, Any] = {}  # Initialize response data dictionary

            try:
                logger.debug(
                    f"Sending execution request to sandbox (id: {exec_id}, repl={repl_mode})"
                )
                # Send the message to the sandbox page's window context
                await self.page.evaluate("window.postMessage", payload)

                logger.debug(
                    f"Waiting for execution result (id: {exec_id}, timeout: {timeout_ms}ms)..."
                )
                # Wait for the response message from the sandbox via the queue
                data = await asyncio.wait_for(response_queue.get(), timeout=timeout_ms / 1000.0)
                logger.debug(f"Received execution result (id: {exec_id}): ok={data.get('ok')}")

            except asyncio.TimeoutError:
                logger.warning(
                    f"Execution timed out waiting for response (id: {exec_id}, timeout: {timeout_ms}ms)"
                )
                # Check if the page is still responsive after the timeout
                await self._check_page_responsiveness(f"Timeout id={exec_id}")
                # Return a structured timeout error
                return {
                    "ok": False,
                    "error": {
                        "type": "TimeoutError",
                        "message": f"Execution timed out after {timeout_ms}ms waiting for sandbox response.",
                        "traceback": None,  # No Python traceback available in this case
                    },
                    "stdout": "",  # Default values on timeout
                    "stderr": "",
                    "result": None,
                    "elapsed": 0,  # No Python elapsed time available
                    "wall_ms": timeout_ms,  # Wall time is the timeout duration
                }
            except Exception as e:
                # Catch potential Playwright communication errors during evaluate/wait
                logger.error(
                    f"Error communicating during execution (id: {exec_id}): {e}", exc_info=True
                )
                # Return a structured communication error
                return {
                    "ok": False,
                    "error": {
                        "type": "CommunicationError",
                        "message": f"Error communicating with sandbox during execution: {e}",
                        "traceback": None,  # Or potentially include JS stack if available from e
                    },
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "elapsed": 0,
                    "wall_ms": 0,
                }
            finally:
                # Always remove the message handler for this execution ID
                self._message_handlers.pop(exec_id, None)

            # --- Validate the structure of the received response ---
            if not isinstance(data, dict) or "ok" not in data:
                logger.error(
                    f"Received malformed response from sandbox (id: {exec_id}, structure invalid): {str(data)[:500]}"
                )
                # Return a structured error indicating the malformed response
                return {
                    "ok": False,
                    "error": {
                        "type": "MalformedResponseError",
                        "message": "Received malformed or incomplete response from sandbox.",
                        "traceback": None,
                        # Safely include details for debugging, converting non-serializable types to string
                        "details": data
                        if isinstance(data, (dict, list, str, int, float, bool, type(None)))
                        else str(data),
                    },
                    # Provide default values for other fields
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "elapsed": 0,
                    # Try to get wall_ms if data is a dict, otherwise default to 0
                    "wall_ms": data.get("wall_ms", 0) if isinstance(data, dict) else 0,
                }

            # --- Ensure essential fields exist with default values before returning ---
            # This guarantees the caller receives a consistent structure even if the sandbox
            # somehow missed fields (though the JS side now also sets defaults).
            data.setdefault("stdout", "")
            data.setdefault("stderr", "")
            data.setdefault("result", None)
            data.setdefault("elapsed", 0)
            data.setdefault("wall_ms", 0)
            # Ensure 'error' field is present if 'ok' is false
            if not data.get("ok", False):
                data.setdefault(
                    "error",
                    {
                        "type": "UnknownSandboxError",
                        "message": "Sandbox reported failure with no specific details.",
                    },
                )
            else:
                # Ensure 'error' is None if 'ok' is true
                data["error"] = None

            # Return the validated and defaulted data dictionary
            return data

    async def reset_repl_state(self) -> Dict[str, Any]:
        """Sends a reset request to the REPL sandbox."""
        if not PLAYWRIGHT_AVAILABLE:
            return {
                "ok": False,
                "error": {"type": "SetupError", "message": "Playwright not installed."},
            }
        if not self.page or self.page.is_closed():
            return {"ok": False, "error": {"type": "StateError", "message": "REPL page is closed."}}
        if not self.ready_evt.is_set():
            return {
                "ok": False,
                "error": {"type": "StateError", "message": "REPL sandbox is not ready."},
            }

        reset_id = f"reset-{uuid.uuid4().hex[:8]}"
        response_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._message_handlers[reset_id] = response_queue
        try:
            message_payload = {"type": "reset", "id": reset_id}
            logger.debug(f"Sending REPL reset message (id: {reset_id})")
            await self.page.evaluate("window.postMessage", message_payload)
            logger.debug(f"Waiting for REPL reset confirmation (id: {reset_id}, timeout: 5s)...")
            data = await asyncio.wait_for(response_queue.get(), timeout=5.0)
            logger.debug(f"Received REPL reset confirmation: {data}")
            return data
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for REPL reset confirmation (id: {reset_id})")
            return {
                "ok": False,
                "error": {
                    "type": "TimeoutError",
                    "message": "Timeout waiting for reset confirmation.",
                },
            }
        except Exception as e:
            logger.error(f"Error during REPL reset call (id: {reset_id}): {e}", exc_info=True)
            return {
                "ok": False,
                "error": {
                    "type": "CommunicationError",
                    "message": f"Error during reset operation: {e}",
                },
            }
        finally:
            self._message_handlers.pop(reset_id, None)

    async def _inject_mcpfs_stub(self) -> None:
        """Creates a minimal stub module `mcpfs` inside the Pyodide interpreter."""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available, cannot inject mcpfs stub.")
            return
        # This stub code is executed within Pyodide, it doesn't need COMMON_PACKAGES_JSON from host Python
        stub_code = r"""
import sys
import types
import asyncio
import json
from js import globalThis

# Simple check if stub already exists
if "mcpfs" in sys.modules:
    print("mcpfs module stub already exists.")
else:
    print("Initializing mcpfs module stub...")
    _mcpfs_msg_id_counter = 0
    _mcpfs_pending_futures = {}

    async def _mcpfs_roundtrip(op: str, *args):
        '''Sends an operation to the host and waits for the response.'''
        nonlocal _mcpfs_msg_id_counter
        _mcpfs_msg_id_counter += 1
        current_id = f"mcpfs-{_mcpfs_msg_id_counter}"

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        _mcpfs_pending_futures[current_id] = fut

        payload = {"type": "mcpfs", "id": current_id, "op": op, "args": args}
        globalThis.postMessage(payload)

        try:
            response = await asyncio.wait_for(fut, timeout=15.0)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout waiting for host mcpfs op '{op}' (id: {current_id})")
        finally:
            _mcpfs_pending_futures.pop(current_id, None)

        if response is None: raise RuntimeError(f"Null response from host for mcpfs op '{op}' (id: {current_id})")
        if "error" in response:
            err_details = response.get('details', '')
            raise RuntimeError(f"Host error for mcpfs op '{op}': {response['error']} {err_details}")
        return response.get("result")

    def _mcpfs_message_callback(event):
        '''Callback attached to Pyodide's message listener to resolve futures.'''
        data = event.data
        if isinstance(data, dict) and data.get("type") == "mcpfs_response":
           msg_id = data.get("id")
           fut = _mcpfs_pending_futures.get(msg_id)
           if fut and not fut.done(): fut.set_result(data)

    globalThis.addEventListener("message", _mcpfs_message_callback)

    mcpfs_module = types.ModuleType("mcpfs")
    async def read_text_async(p): return await _mcpfs_roundtrip("read", p)
    async def write_text_async(p, t): return await _mcpfs_roundtrip("write", p, t)
    async def listdir_async(p): return await _mcpfs_roundtrip("list", p)
    mcpfs_module.read_text_async = read_text_async
    mcpfs_module.write_text_async = write_text_async
    mcpfs_module.listdir_async = listdir_async
    mcpfs_module.read_text = read_text_async
    mcpfs_module.write_text = write_text_async
    mcpfs_module.listdir = listdir_async
    sys.modules["mcpfs"] = mcpfs_module
    print("mcpfs Python module stub initialized successfully.")
# --- End of MCPFS Stub Logic ---
"""
        if not self.page or self.page.is_closed():
            logger.error("Cannot inject mcpfs stub: Sandbox page is closed.")
            return
        try:
            logger.debug("Injecting mcpfs stub into Pyodide environment...")
            await self.page.evaluate(
                f"""(async () => {{
                    try {{
                        if (typeof self.pyodide === 'undefined' || !self.pyodide.runPythonAsync) {{
                             console.error('Pyodide instance not ready for mcpfs stub injection.'); return;
                        }}
                        await self.pyodide.runPythonAsync(`{stub_code}`);
                        console.log("mcpfs Python stub injection script executed.");
                    }} catch (err) {{
                        console.error("Error running mcpfs stub injection Python code:", err);
                        globalThis.postMessage({{ type: 'error', id:'mcpfs_stub_inject_fail', error: {{ type: 'InjectionError', message: 'Failed to inject mcpfs stub: ' + err.toString() }} }}, "*");
                    }}
                }})();"""
            )
            logger.info("mcpfs stub injection command sent to sandbox.")
        except Exception as e:
            logger.error(f"Failed to evaluate mcpfs stub injection script: {e}", exc_info=True)
            # Don't raise ToolError here, log it. FS might not be critical.


# --- End of PyodideSandbox Class ---


###############################################################################
# Browser / sandbox lifecycle helpers – with LRU eviction
###############################################################################
async def _get_browser() -> "Browser":  # type: ignore # Use string literal hint
    """Initializes and returns the shared Playwright browser instance."""
    global _BROWSER
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright is not installed.")
    browser_connected = False
    if _BROWSER is not None:
        try:
            browser_connected = _BROWSER.is_connected()
        except Exception as check_err:
            logger.warning(
                f"Error checking browser connection status: {check_err}. Assuming disconnected."
            )
            browser_connected = False
            _BROWSER = None
    if _BROWSER is None or not browser_connected:
        logger.info("Launching headless Chromium for Pyodide sandbox...")
        try:
            playwright = await pw.async_playwright().start()
            launch_options = {
                "headless": True,
                "args": [
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--disable-features=Translate",
                    "--disable-extensions",
                    "--disable-component-extensions-with-background-pages",
                    "--disable-background-networking",
                    "--disable-sync",
                    "--metrics-recording-only",
                    "--disable-default-apps",
                    "--mute-audio",
                    "--no-first-run",
                    "--safebrowsing-disable-auto-update",
                    "--disable-popup-blocking",
                    "--disable-setuid-sandbox",
                    "--disable-web-security",
                    "--allow-file-access-from-files",
                    "--allow-universal-access-from-file-urls",
                    "--disable-permissions-api",
                ],
                "timeout": 90000,
            }
            _BROWSER = await playwright.chromium.launch(**launch_options)

            def _sync_cleanup():
                global _BROWSER
                if _BROWSER and _BROWSER.is_connected():
                    logger.info("Closing Playwright browser via atexit handler...")
                    try:
                        loop = asyncio.get_event_loop_policy().get_event_loop()
                        if loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(_BROWSER.close(), loop)
                            future.result(timeout=15)
                        else:
                            loop.run_until_complete(_BROWSER.close())
                        logger.info("Playwright browser closed successfully via atexit.")
                        _BROWSER = None
                    except Exception as e:
                        logger.error(
                            f"Error during atexit Playwright browser cleanup: {e}", exc_info=True
                        )

            atexit.register(_sync_cleanup)
            logger.info("Headless Chromium launched successfully and atexit cleanup registered.")
        except Exception as e:
            logger.error(f"Failed to launch Playwright browser: {e}", exc_info=True)
            _BROWSER = None
            raise ProviderError(f"Failed to launch browser for sandbox: {e}") from e
    if not _BROWSER:
        raise ProviderError("Browser instance is None after launch attempt.")
    return _BROWSER


async def _get_sandbox(session_id: str, **kwargs) -> PyodideSandbox:
    """Retrieves or creates a PyodideSandbox instance, managing LRU cache."""
    global _GLOBAL_SEM, _PAGES
    if not PLAYWRIGHT_AVAILABLE:
        # Check upfront if Playwright is available
        raise RuntimeError("Playwright is not installed. Cannot create sandboxes.")

    # Initialize the global semaphore if this is the first call
    if _GLOBAL_SEM is None:
        _GLOBAL_SEM = asyncio.Semaphore(GLOBAL_CONCURRENCY)
        logger.debug(
            f"Initialized global execution semaphore with concurrency {GLOBAL_CONCURRENCY}"
        )

    # Check if a sandbox for this session ID already exists in our cache
    sb = _PAGES.get(session_id)
    if sb is not None:
        page_valid = False
        if sb.page:
            # Verify the associated Playwright Page object is still open
            try:
                page_valid = not sb.page.is_closed()
            except Exception as page_check_err:
                # Handle potential errors during the check (e.g., context destroyed)
                logger.warning(
                    f"Error checking page status for {session_id}: {page_check_err}. Assuming invalid."
                )
                page_valid = False

        if page_valid:
            # If the page is valid, reuse the existing sandbox
            logger.debug(f"Reusing existing sandbox session: {session_id}")
            # Move the accessed sandbox to the end of the OrderedDict (marks it as recently used)
            _PAGES.move_to_end(session_id)
            sb.last_used = time.time()  # Update last used timestamp
            return sb
        else:
            # If the page is closed or invalid, remove the entry from the cache
            logger.warning(f"Removing closed/invalid sandbox session from cache: {session_id}")
            _PAGES.pop(session_id, None)
            # Attempt to gracefully close the page object if it exists
            if sb.page:
                await sb._try_close_page("Invalid page found in cache")

    # If no valid sandbox found, create a new one, potentially evicting the LRU
    while len(_PAGES) >= MAX_SANDBOXES:
        # Remove the least recently used sandbox (first item in OrderedDict)
        try:
            victim_id, victim_sb = _PAGES.popitem(last=False)
            logger.info(
                f"Sandbox cache full ({len(_PAGES) + 1}/{MAX_SANDBOXES}). Evicting LRU session: {victim_id} "
                f"(created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(victim_sb.created_at))}, "
                f"last used: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(victim_sb.last_used))})"
            )
            # Attempt to close the evicted sandbox's page
            await victim_sb._try_close_page(f"LRU eviction (victim: {victim_id})")
        except KeyError:
            # Should not happen if len(_PAGES) >= MAX_SANDBOXES, but handle defensively
            logger.warning("LRU eviction attempted but cache was empty.")
            break  # Avoid infinite loop if something is wrong

    logger.info(f"Creating new sandbox session: {session_id}")
    browser = await _get_browser()  # Get or initialize the shared browser instance
    page: Optional["Page"] = None  # type: ignore # Initialize page variable

    try:
        # Create a new browser page (tab)
        page = await browser.new_page()
        # Set up logging and error handlers for this specific page
        _wire_page_logging(page, session_id)
        logger.debug(f"New browser page created for session {session_id}")

        # Create the PyodideSandbox object instance
        sb = PyodideSandbox(
            page=page, **kwargs
        )  # Pass through any extra kwargs (like allow_network)

        # Initialize the sandbox (loads HTML, waits for ready signal)
        await sb.init()

        # Add the newly created and initialized sandbox to the cache
        _PAGES[session_id] = sb
        logger.info(f"New sandbox session {session_id} created and initialized successfully.")
        return sb

    except Exception as e:
        # Handle errors during page creation or sandbox initialization
        logger.error(f"Failed to create or initialize new sandbox {session_id}: {e}", exc_info=True)
        # If the page was created but initialization failed, try to close it
        if page and not page.is_closed():
            # Use a temporary sandbox object just to call the closing method
            await PyodideSandbox(page=page)._try_close_page(
                f"Failed sandbox creation/init ({session_id})"
            )
        # Re-raise the exception, preserving type if it's a known error type
        if isinstance(e, (ToolError, ProviderError)):
            raise e
        # Wrap unexpected errors in ProviderError for consistent error handling upstream
        raise ProviderError(f"Failed to create sandbox {session_id}: {e}") from e


async def _close_all_sandboxes():
    """Gracefully close all active sandbox pages and the browser."""
    global _BROWSER, _PAGES
    logger.info("Closing all active Pyodide sandboxes...")
    page_close_tasks = []
    sandboxes_to_close = list(_PAGES.values())
    _PAGES.clear()
    for sb in sandboxes_to_close:
        close_task = asyncio.create_task(sb._try_close_page("Global shutdown"))
        page_close_tasks.append(close_task)
    if page_close_tasks:
        gathered_results = await asyncio.gather(*page_close_tasks, return_exceptions=True)
        closed_count = sum(1 for result in gathered_results if not isinstance(result, Exception))
        errors = [result for result in gathered_results if isinstance(result, Exception)]
        logger.info(
            f"Attempted to close {len(page_close_tasks)} sandbox pages. Success: {closed_count}."
        )
        if errors:
            logger.warning(f"{len(errors)} errors during page close: {errors}")

    browser_needs_closing = False
    if _BROWSER:
        try:
            browser_needs_closing = _BROWSER.is_connected()
        except Exception as browser_check_err:
            logger.warning(
                f"Error checking browser connection during close: {browser_check_err}. Assuming needs closing."
            )
            browser_needs_closing = True
    if browser_needs_closing:
        logger.info("Closing Playwright browser instance...")
        try:
            await _BROWSER.close()
            logger.info("Playwright browser closed successfully.")
        except Exception as e:
            logger.error(f"Error closing Playwright browser: {e}")
    _BROWSER = None


def display_sandbox_result(
    title: str, result: Optional[Dict[str, Any]], code_str: Optional[str] = None
) -> None:
    """Display sandbox execution result with enhanced formatting."""
    console.print(Rule(f"[bold cyan]{escape(title)}[/bold cyan]"))

    if code_str:
        console.print(
            Panel(
                Syntax(
                    code_str.strip(), "python", theme="monokai", line_numbers=True, word_wrap=True
                ),
                title="Executed Code",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if result is None:
        console.print(
            Panel(
                "[bold yellow]No result object returned from tool call.[/]",
                title="Warning",
                border_style="yellow",
            )
        )
        console.print()
        return

    # Check for errors based on the result structure from safe_tool_call
    if not result.get("success", False) and "error" in result:
        error_msg = result.get("error", "Unknown error")
        error_type = result.get("error_type", "UnknownError")
        error_code = result.get("error_code", "UNKNOWN")
        details = result.get("details", {})

        error_renderable = f"[bold red]:x: Operation Failed ({escape(error_type)} / {escape(error_code)}):[/]\n{escape(error_msg)}"
        if details:
            try:
                details_str = escape(str(details))  # Basic string representation
                error_renderable += f"\n\n[bold]Details:[/]\n{details_str}"
            except Exception:
                error_renderable += "\n\n[bold]Details:[/]\n(Could not display details)"

        console.print(
            Panel(error_renderable, title="Error", border_style="red", padding=(1, 2), expand=False)
        )
        console.print()
        return

    # --- Display Success Case ---
    actual_result = result.get(
        "result", {}
    )  # Get the nested result dict from execute_python/repl_python

    # Create output panel for stdout/stderr
    output_parts = []
    if stdout := actual_result.get("stdout", ""):
        output_parts.append(f"[bold green]STDOUT:[/]\n{escape(stdout)}")

    if stderr := actual_result.get("stderr", ""):
        if output_parts:
            output_parts.append("\n" + ("-" * 20) + "\n")  # Separator
        output_parts.append(f"[bold red]STDERR:[/]\n{escape(stderr)}")

    if output_parts:
        console.print(
            Panel(
                "\n".join(output_parts),
                title="Output (stdout/stderr)",
                border_style="yellow",
                padding=(1, 2),
            )
        )
    else:
        console.print("[dim]No stdout or stderr captured.[/dim]")

    # Display result value if present and not None
    result_value = actual_result.get(
        "result"
    )  # This is the value assigned to 'result' in the executed code
    if result_value is not None:
        try:
            # Attempt to pretty-print common types
            if isinstance(result_value, (dict, list)):
                result_str = str(result_value)  # Keep it simple for now
            else:
                result_str = str(result_value)

            # Limit length for display
            max_len = 500
            display_str = result_str[:max_len] + ("..." if len(result_str) > max_len else "")

            console.print(
                Panel(
                    Syntax(
                        display_str, "python", theme="monokai", line_numbers=False, word_wrap=True
                    ),
                    title="Result Variable ('result')",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        except Exception as e:
            console.print(
                Panel(
                    f"[yellow]Could not format result value: {e}[/]",
                    title="Result Variable ('result')",
                    border_style="yellow",
                )
            )
            console.print(f"Raw Result Type: {type(result_value)}")
            try:
                console.print(f"Raw Result Repr: {escape(repr(result_value)[:500])}...")
            except Exception:
                pass

    # Display execution stats
    stats_table = Table(
        title="Execution Statistics",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 1),
        border_style="dim",
    )
    stats_table.add_column("Metric", style="cyan", justify="right")
    stats_table.add_column("Value", style="white")

    if "elapsed_py_ms" in actual_result:
        stats_table.add_row("Python Execution Time", f"{actual_result['elapsed_py_ms']:.2f} ms")
    if "elapsed_wall_ms" in actual_result:
        stats_table.add_row("Sandbox Wall Clock Time", f"{actual_result['elapsed_wall_ms']:.2f} ms")
    if "total_duration_ms" in result:  # From safe_tool_call wrapper
        stats_table.add_row("Total Tool Call Time", f"{result['total_duration_ms']:.2f} ms")
    if "session_id" in actual_result:
        stats_table.add_row("Session ID", actual_result["session_id"])
    if "handle" in actual_result:
        stats_table.add_row("REPL Handle", actual_result["handle"])

    if stats_table.row_count > 0:
        console.print(stats_table)

    console.print()  # Add spacing


###############################################################################
# mcpfs bridge – listens for postMessage & proxies to secure FS tool
###############################################################################
async def _listen_for_mcpfs_calls(page: "Page"):  # type: ignore # Use string literal hint
    """Sets up listener for 'mcpfs' messages from the sandbox page."""
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("Playwright not available, cannot listen for mcpfs calls.")
        return

    async def _handle_mcpfs_message(payload: Any):
        """Processes 'mcpfs' request from Pyodide and sends 'mcpfs_response' back."""
        data = payload
        is_mcpfs_message = isinstance(data, dict) and data.get("type") == "mcpfs"
        if not is_mcpfs_message:
            return

        call_id = data.get("id")
        op = data.get("op")
        args = data.get("args", [])
        if not call_id or not op:
            logger.warning(
                f"MCPFS Bridge: Received invalid mcpfs message (missing id or op): {data}"
            )
            return
        response_payload: dict[str, Any] = {"type": "mcpfs_response", "id": call_id}
        try:
            try:
                from ultimate_mcp_server.tools import filesystem as fs
            except ImportError as e:
                logger.error("MCPFS Bridge: Failed to import 'filesystem' tool.", exc_info=True)
                raise ToolError("Filesystem tool backend not available.") from e
            if _VERBOSE_SANDBOX_LOGGING > 1:
                logger.debug(f"MCPFS Bridge: Received op='{op}', args={args}, id={call_id}")

            if op == "read":
                if len(args) != 1 or not isinstance(args[0], str):
                    raise ValueError("read requires 1 string arg (path)")
                res = await fs.read_file(path=args[0])
                if res.get("success") and isinstance(res.get("content"), list) and res["content"]:
                    file_content = res["content"][0].get("text")
                    if file_content is None:
                        raise ToolError("Read succeeded but missing 'text' key.")
                    response_payload["result"] = file_content
                else:
                    raise ToolError(res.get("error", "Read failed"), details=res.get("details"))
            elif op == "write":
                if len(args) != 2 or not isinstance(args[0], str) or not isinstance(args[1], str):
                    raise ValueError("write requires 2 string args (path, content)")
                res = await fs.write_file(path=args[0], content=args[1])
                if res.get("success"):
                    response_payload["result"] = True
                else:
                    raise ToolError(res.get("error", "Write failed"), details=res.get("details"))
            elif op == "list":
                if len(args) != 1 or not isinstance(args[0], str):
                    raise ValueError("list requires 1 string arg (path)")
                res = await fs.list_directory(path=args[0])
                if res.get("success"):
                    response_payload["result"] = res.get("entries", [])
                else:
                    raise ToolError(res.get("error", "List failed"), details=res.get("details"))
            else:
                raise ValueError(f"Unsupported mcpfs operation: '{op}'")
        except (ToolError, ToolInputError, ProviderError, ValueError) as tool_exc:
            error_message = f"{type(tool_exc).__name__}: {tool_exc}"
            logger.warning(
                f"MCPFS Bridge Error processing op='{op}' (id={call_id}): {error_message}"
            )
            response_payload["error"] = error_message
            if hasattr(tool_exc, "details") and tool_exc.details:
                try:
                    response_payload["details"] = json.loads(
                        json.dumps(tool_exc.details, default=str)
                    )
                except Exception:
                    response_payload["details"] = {"error": "Serialization failed"}
        except Exception as exc:
            error_message = f"Unexpected Host Error: {exc}"
            logger.error(
                f"Unexpected MCPFS Bridge Error (op='{op}', id={call_id}): {error_message}",
                exc_info=True,
            )
            response_payload["error"] = error_message
        try:
            response_successful = "error" not in response_payload
            if _VERBOSE_SANDBOX_LOGGING > 1:
                logger.debug(
                    f"MCPFS Bridge: Sending response (op='{op}', id={call_id}, success={response_successful})"
                )
            await page.evaluate(JS_POST_MESSAGE, response_payload)
        except Exception as post_err:
            logger.warning(
                f"Failed to send mcpfs response back to sandbox (id: {call_id}, op: '{op}'): {post_err}"
            )

    handler_func_name = "_handleMcpFsMessageFromHost"
    try:
        await page.expose_function(handler_func_name, _handle_mcpfs_message)
        await page.evaluate(f"""
            if (!window._mcpfsListenerAttached) {{
                console.log('Setting up MCPFS message listener in browser context...');
                window.addEventListener('message', (event) => {{
                    if (event.data && event.data.type === 'mcpfs' && typeof window.{handler_func_name} === 'function') {{
                        window.{handler_func_name}(event.data);
                    }}
                }});
                window._mcpfsListenerAttached = true;
                console.log('MCPFS message listener attached.');
            }}
        """)
        logger.info("MCPFS listener bridge established successfully.")
    except Exception as e:
        logger.error(f"Failed to set up MCPFS listener bridge: {e}", exc_info=True)
        raise ToolError(f"Filesystem bridge listener setup failed: {e}") from e


def _format_sandbox_error(error_payload: Optional[Dict[str, Any]]) -> str:
    if not error_payload or not isinstance(error_payload, dict):
        return "Unknown sandbox execution error."
    err_type = error_payload.get("type", "UnknownError")
    err_msg = error_payload.get("message", "No details provided.")
    # Optionally include traceback snippet if needed, but keep main message clean
    tb = error_payload.get("traceback")
    if tb:
        err_msg += f"\nTraceback (see logs/details):\n{str(tb)[:200]}..."
    return f"{err_type} - {err_msg}"


###############################################################################
# Standalone Tool Functions (execute_python, repl_python)
###############################################################################
@with_tool_metrics
@with_error_handling
async def execute_python(
    code: str,
    packages: Optional[List[str]] = None,
    wheels: Optional[List[str]] = None,
    allow_network: bool = False,
    allow_fs: bool = False,
    session_id: Optional[str] = None,
    timeout_ms: int = 15_000,
    ctx: Optional[Dict[str, Any]] = None,  # Context often used by decorators
) -> Dict[str, Any]:
    """
    Runs Python code in a one-shot Pyodide sandbox.

    Args:
        code: The Python code string to execute.
        packages: A list of Pyodide packages to ensure are loaded. Do not include stdlib modules.
        wheels: A list of Python wheel URLs to install via micropip.
        allow_network: If True, allows network access (e.g., for micropip to PyPI).
        allow_fs: If True, enables the mcpfs filesystem bridge (requires host setup).
        session_id: Optional ID to reuse or create a specific sandbox session. If None, a new ID is generated.
        timeout_ms: Timeout for waiting for the sandbox execution result (in milliseconds).
        ctx: Optional context dictionary, often passed by framework/decorators.

    Returns:
        A dictionary containing execution results:
        {
            'success': bool,
            'stdout': str,
            'stderr': str,
            'result': Any, # Value of the 'result' variable in the Python code, if set
            'elapsed_py_ms': int, # Time spent executing Python code (reported by sandbox)
            'elapsed_wall_ms': int, # Total wall clock time from JS perspective (reported by sandbox)
            'session_id': str,
            'error_message': Optional[str], # Formatted error if success is False
            'error_details': Optional[Dict], # Original error dict from sandbox if success is False
        }

    Raises:
        ProviderError: If the sandbox environment (Playwright/browser) cannot be set up.
        ToolInputError: If input arguments are invalid.
        ToolError: If sandbox execution fails (contains formatted message and details).
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise ProviderError("Playwright dependency is missing for Python Sandbox.")
    if not isinstance(code, str) or not code:
        raise ToolInputError(
            "Input 'code' must be a non-empty string.", param="code", value=repr(code)
        )
    if not isinstance(timeout_ms, int) or timeout_ms <= 0:
        raise ToolInputError(
            "Input 'timeout_ms' must be a positive integer.", param="timeout_ms", value=timeout_ms
        )
    # Basic type checks for lists/bools - could add more specific validation
    if packages is not None and not isinstance(packages, list):
        raise ToolInputError(
            "Input 'packages' must be a list or None.", param="packages", value=packages
        )
    if wheels is not None and not isinstance(wheels, list):
        raise ToolInputError("Input 'wheels' must be a list or None.", param="wheels", value=wheels)
    if not isinstance(allow_network, bool):
        raise ToolInputError(
            "Input 'allow_network' must be a boolean.", param="allow_network", value=allow_network
        )
    if not isinstance(allow_fs, bool):
        raise ToolInputError(
            "Input 'allow_fs' must be a boolean.", param="allow_fs", value=allow_fs
        )
    if session_id is not None and not isinstance(session_id, str):
        raise ToolInputError(
            "Input 'session_id' must be a string or None.", param="session_id", value=session_id
        )

    # Normalize package/wheel lists
    # IMPORTANT: Filter out common stdlib modules that shouldn't be passed
    stdlib_modules_to_filter = {
        "math",
        "sys",
        "os",
        "json",
        "io",
        "contextlib",
        "time",
        "base64",
        "traceback",
        "collections",
        "re",
        "datetime",
    }
    packages_normalized = [pkg for pkg in (packages or []) if pkg not in stdlib_modules_to_filter]
    wheels_normalized = wheels or []

    # Generate a session ID if one wasn't provided
    current_session_id = session_id or f"exec-{uuid.uuid4().hex[:12]}"  # Add prefix for clarity

    # Get or create the sandbox instance
    try:
        # Assuming _get_sandbox is defined elsewhere and returns PyodideSandbox instance
        sb = await _get_sandbox(current_session_id, allow_network=allow_network, allow_fs=allow_fs)
    except Exception as e:
        # Catch potential errors during sandbox acquisition/initialization
        if isinstance(e, (ToolError, ProviderError)):
            raise e  # Re-raise known error types
        # Wrap unexpected errors
        raise ProviderError(
            f"Failed to get or initialize sandbox '{current_session_id}': {e}",
            tool_name="python_sandbox",
            cause=e,
        ) from e

    t0 = time.perf_counter()  # Start timer just before execute call
    data: Dict[str, Any] = {}  # Initialize data dict

    # Execute the code within the sandbox
    try:
        # Call the execute method on the sandbox object
        # Pass repl_mode=False for one-shot execution
        data = await sb.execute(
            code, packages_normalized, wheels_normalized, timeout_ms, repl_mode=False
        )
    except Exception as e:
        # Catch potential host-side errors during the .execute() call itself
        # (e.g., Playwright communication errors not caught internally by execute)
        wall_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            f"Unexpected host error calling sandbox execute for {current_session_id}: {e}",
            exc_info=True,
        )
        raise ToolError(
            f"Unexpected host error during sandbox execution call: {e}",
            error_code="HostExecutionError",
            details={"session_id": current_session_id, "elapsed_wall_ms": wall_ms, "cause": str(e)},
        ) from e

    # Process the results received from the sandbox
    wall_ms_host = int((time.perf_counter() - t0) * 1000)  # Wall time measured by host
    is_success = data.get("ok", False)
    error_info = data.get(
        "error"
    )  # This is the structured {type, message, traceback} dict from sandbox
    js_wall_ms = int(data.get("wall_ms", 0))  # Wall time reported by JS sandbox handler

    # Format error message IF execution failed inside the sandbox
    error_message_for_caller = None
    error_code_for_caller = "UnknownSandboxError"  # Default error code
    if not is_success:
        error_message_for_caller = _format_sandbox_error(
            error_info
        )  # Use helper to get "Type - Message" string
        if isinstance(error_info, dict):
            error_code_for_caller = error_info.get(
                "type", "UnknownSandboxError"
            )  # Get specific code

    # Prepare structured logging details
    log_details = {
        "session_id": current_session_id,
        "elapsed_wall_ms_host": wall_ms_host,
        "elapsed_wall_ms_js": js_wall_ms,  # Log both wall times for comparison
        "elapsed_py_ms": int(data.get("elapsed", 0)),
        "packages_requested": packages or [],  # Log original requested packages
        "packages_loaded": packages_normalized,  # Log packages actually sent to loadPackage
        "wheels_count": len(wheels_normalized),
        "stdout_len": len(data.get("stdout", "")),
        "stderr_len": len(data.get("stderr", "")),
        "result_type": type(data.get("result")).__name__,
        "success": is_success,
        "repl_mode": False,
    }

    # Log and return/raise based on success
    if is_success:
        logger.success(
            f"Python code executed successfully (session: {current_session_id})",
            TaskType.CODE_EXECUTION,  # Assumes TaskType is defined/imported
            **log_details,
        )
        # Return success dictionary matching specified structure
        return {
            "success": True,
            "stdout": data.get("stdout", ""),
            "stderr": data.get("stderr", ""),
            "result": data.get("result"),  # Can be None
            "elapsed_py_ms": int(data.get("elapsed", 0)),
            "elapsed_wall_ms": js_wall_ms or wall_ms_host,  # Prefer JS wall time
            "session_id": current_session_id,
            "error_message": None,  # Explicitly None on success
            "error_details": None,  # Explicitly None on success
        }
    else:
        # Log the failure with details
        logger.error(
            f"Python code execution failed (session: {current_session_id}): {error_message_for_caller}",
            TaskType.CODE_EXECUTION,  # Assumes TaskType is defined/imported
            **log_details,
            error_details=error_info,  # Log the original structured error details
        )
        # Raise a ToolError containing the formatted message and original details
        raise ToolError(
            f"Python execution failed: {error_message_for_caller}",  # User-friendly message
            error_code=error_code_for_caller,  # Specific error code from sandbox
            details=error_info,  # Original structured error from sandbox
        )


@with_tool_metrics
@with_error_handling
async def repl_python(
    code: str,
    packages: Optional[List[str]] = None,
    wheels: Optional[List[str]] = None,
    allow_network: bool = False,
    allow_fs: bool = False,
    handle: Optional[str] = None,  # Session handle for persistence
    timeout_ms: int = 15_000,
    reset: bool = False,  # Flag to reset the REPL state before execution
    ctx: Optional[Dict[str, Any]] = None,  # Context often used by decorators
) -> Dict[str, Any]:
    """
    Runs Python code in a persistent REPL-like sandbox environment.

    Args:
        code: The Python code string to execute in the session. Can be empty if only resetting.
        packages: Additional Pyodide packages to ensure are loaded for this specific call.
        wheels: Additional Python wheel URLs to install for this specific call.
        allow_network: If True, allows network access for the sandbox session.
        allow_fs: If True, enables the mcpfs filesystem bridge for the session.
        handle: A specific session ID to use. If None, a new session is created.
                Use the returned handle for subsequent calls to maintain state.
        timeout_ms: Timeout for waiting for this specific execution call.
        reset: If True, clears the REPL session's state (_MCP_REPL_NS) before executing code.
        ctx: Optional context dictionary.

    Returns:
        A dictionary containing execution results for *this call*:
        {
            'success': bool, # Success of *this specific code execution* (or reset)
            'stdout': str,
            'stderr': str,
            'result': Any, # Value of 'result' variable from this execution, if set
            'elapsed_py_ms': int,
            'elapsed_wall_ms': int,
            'handle': str, # The session handle (same as input or newly generated)
            'error_message': Optional[str], # Formatted error if success is False
            'error_details': Optional[Dict], # Original error dict if success is False
            'reset_status': Optional[Dict], # Included only if reset=True, contains reset ack
        }

    Raises:
        ProviderError: If the sandbox environment cannot be set up.
        ToolInputError: If input arguments are invalid.
        ToolError: If a non-recoverable error occurs during execution (contains details).
                 Note: Standard Python errors within the code are returned in the 'error' fields,
                 not typically raised as ToolError unless they prevent result processing.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise ProviderError("Playwright dependency is missing for Python Sandbox.")
    # Code can be empty if reset is True
    if not isinstance(code, str):
        raise ToolInputError("Input 'code' must be a string.", param="code", value=repr(code))
    if not code and not reset:
        raise ToolInputError(
            "Input 'code' cannot be empty unless 'reset' is True.", param="code", value=repr(code)
        )
    if not isinstance(timeout_ms, int) or timeout_ms <= 0:
        raise ToolInputError(
            "Input 'timeout_ms' must be a positive integer.", param="timeout_ms", value=timeout_ms
        )
    # Basic type checks - can be expanded
    if packages is not None and not isinstance(packages, list):
        raise ToolInputError(
            "Input 'packages' must be a list or None.", param="packages", value=packages
        )
    if wheels is not None and not isinstance(wheels, list):
        raise ToolInputError("Input 'wheels' must be a list or None.", param="wheels", value=wheels)
    if not isinstance(allow_network, bool):
        raise ToolInputError(
            "Input 'allow_network' must be a boolean.", param="allow_network", value=allow_network
        )
    if not isinstance(allow_fs, bool):
        raise ToolInputError(
            "Input 'allow_fs' must be a boolean.", param="allow_fs", value=allow_fs
        )
    if handle is not None and not isinstance(handle, str):
        raise ToolInputError(
            "Input 'handle' must be a string or None.", param="handle", value=handle
        )
    if not isinstance(reset, bool):
        raise ToolInputError("Input 'reset' must be a boolean.", param="reset", value=reset)

    # IMPORTANT: Filter out common stdlib modules that shouldn't be passed
    stdlib_modules_to_filter = {
        "math",
        "sys",
        "os",
        "json",
        "io",
        "contextlib",
        "time",
        "base64",
        "traceback",
        "collections",
        "re",
        "datetime",
    }
    packages_normalized = [pkg for pkg in (packages or []) if pkg not in stdlib_modules_to_filter]
    wheels_normalized = wheels or []

    # Use provided handle or generate a new persistent one
    session_id = handle or f"repl-{uuid.uuid4().hex[:12]}"

    # Get or create the sandbox instance (will reuse if handle exists and page is open)
    try:
        # Pass allow_network/allow_fs, they are session-level properties
        sb = await _get_sandbox(session_id, allow_network=allow_network, allow_fs=allow_fs)
    except Exception as e:
        if isinstance(e, (ToolError, ProviderError)):
            raise e
        raise ProviderError(
            f"Failed to get or initialize REPL sandbox '{session_id}': {e}",
            tool_name="python_sandbox",
            cause=e,
        ) from e

    t0 = time.perf_counter()  # Start timer before potential reset/execute
    reset_ack_data: Optional[Dict] = None  # To store the ack from the reset call

    # --- Handle Reset Request ---
    if reset:
        logger.info(f"Resetting REPL state for session: {session_id}")
        try:
            # Assuming PyodideSandbox has a method like this that uses the direct callback
            reset_ack_data = await sb.reset_repl_state()  # This should wait for the JS ack
            if not reset_ack_data or not reset_ack_data.get("ok"):
                # Log warning but don't necessarily fail the whole call yet
                error_msg = (
                    _format_sandbox_error(reset_ack_data.get("error"))
                    if reset_ack_data
                    else "No confirmation received"
                )
                logger.warning(
                    f"REPL state reset failed or unconfirmed for session {session_id}: {error_msg}"
                )
                # Optionally add this warning to the final result?
        except Exception as e:
            # Handle errors during the reset call itself
            logger.warning(
                f"Error during REPL reset call for session {session_id}: {e}", exc_info=True
            )
            # Store this error to potentially include in final result if no code is run
            reset_ack_data = {"ok": False, "error": {"type": "ResetHostError", "message": str(e)}}

        # If ONLY resetting (no code provided), return immediately after reset attempt
        if not code:
            host_wall_ms = int((time.perf_counter() - t0) * 1000)
            final_result = {
                "success": reset_ack_data.get("ok", False)
                if reset_ack_data
                else False,  # Reflect reset success
                "stdout": "",
                "stderr": "",
                "result": None,
                "elapsed_py_ms": 0,
                "elapsed_wall_ms": host_wall_ms,  # Only host time available
                "handle": session_id,
                "error_message": None
                if (reset_ack_data and reset_ack_data.get("ok"))
                else _format_sandbox_error(reset_ack_data.get("error") if reset_ack_data else None),
                "error_details": reset_ack_data.get("error")
                if (reset_ack_data and not reset_ack_data.get("ok"))
                else None,
                "reset_status": reset_ack_data,  # Always include reset ack if reset was true
            }
            return final_result

    # --- Execute Code (if provided) ---
    data: Dict[str, Any] = {}  # Initialize data dict for execution results
    execution_successful_this_call = True  # Assume success unless execution fails
    if code:
        try:
            # Call the execute method, ensuring repl_mode=True is passed
            data = await sb.execute(
                code, packages_normalized, wheels_normalized, timeout_ms, repl_mode=True
            )
            execution_successful_this_call = data.get(
                "ok", False
            )  # Get success status from execution result
        except Exception as e:
            # Catch host-side errors during the execute call
            execution_successful_this_call = False
            wall_ms_host_error = int((time.perf_counter() - t0) * 1000)
            logger.error(
                f"Unexpected host error calling REPL sandbox execute for {session_id}: {e}",
                exc_info=True,
            )
            # Create a failure structure similar to what execute returns
            data = {
                "ok": False,
                "error": {
                    "type": "HostExecutionError",
                    "message": f"Host error during REPL exec: {e}",
                },
                "wall_ms": wall_ms_host_error,  # Use host time
                "elapsed": 0,
                "stdout": "",
                "stderr": "",
                "result": None,
            }

    # --- Format results and potential errors ---
    wall_ms_host_final = int((time.perf_counter() - t0) * 1000)
    js_wall_ms = int(data.get("wall_ms", 0))  # Wall time reported by JS sandbox handler
    py_elapsed_ms = int(data.get("elapsed", 0))
    stdout_content = data.get("stdout", "")
    stderr_content = data.get("stderr", "")
    result_val = data.get("result")
    error_info = data.get("error")  # Original error dict from sandbox execution
    error_message_for_caller = None
    error_code_for_caller = "UnknownError"

    if not execution_successful_this_call:
        error_message_for_caller = _format_sandbox_error(error_info)
        if isinstance(error_info, dict):
            error_code_for_caller = error_info.get("type", "UnknownSandboxError")  # noqa: F841

    # --- Logging ---
    action_desc = "executed" if code else "accessed (no code run)"
    action_desc += " with reset" if reset else ""
    log_details = {
        "session_id": session_id,
        "action": action_desc,
        "reset_requested": reset,
        "reset_successful": reset_ack_data.get("ok") if reset_ack_data else None,
        "elapsed_wall_ms_host": wall_ms_host_final,
        "elapsed_wall_ms_js": js_wall_ms,
        "elapsed_py_ms": py_elapsed_ms,
        "packages_requested": packages or [],
        "packages_loaded": packages_normalized,
        "wheels_count": len(wheels_normalized),
        "stdout_len": len(stdout_content),
        "stderr_len": len(stderr_content),
        "result_type": type(result_val).__name__,
        "success_this_call": execution_successful_this_call,
        "repl_mode": True,
    }
    log_level = logger.success if execution_successful_this_call else logger.warning
    log_level(
        f"Python code {action_desc} in REPL sandbox (session: {session_id})",
        TaskType.CODE_EXECUTION,  # Assumes TaskType is defined/imported
        **log_details,
        error_details=error_info if not execution_successful_this_call else None,
    )

    # --- Construct final return dictionary ---
    final_result = {
        "success": execution_successful_this_call,  # Reflect success of *this* call
        "stdout": stdout_content,
        "stderr": stderr_content,
        "result": result_val,
        "elapsed_py_ms": py_elapsed_ms,
        "elapsed_wall_ms": js_wall_ms or wall_ms_host_final,  # Prefer JS wall time
        "handle": session_id,  # Always return the handle
        "error_message": error_message_for_caller,  # Formatted string or None
        "error_details": error_info
        if not execution_successful_this_call
        else None,  # Original dict or None
    }
    # Include reset status if reset was requested
    if reset:
        final_result["reset_status"] = reset_ack_data

    # Do NOT raise ToolError for standard Python errors caught inside sandbox,
    # return them in the dictionary structure instead.
    # Only raise ToolError for host-level/unrecoverable issues earlier.
    return final_result


###############################################################################
# Optional: Asset Preloader Function (Integrated)
###############################################################################


def _get_pyodide_asset_list_from_manifest(manifest_url: str) -> List[str]:
    """
    Generates a list of essential Pyodide assets to preload based on version.
    For v0.27.5, uses a hardcoded list as repodata.json isn't typically used
    for core file listing in the same way older versions might have.
    """
    global _PYODIDE_VERSION  # Access the global version
    logger.info(f"[Preload] Generating asset list for Pyodide v{_PYODIDE_VERSION}.")

    # Version-specific logic (can be expanded for other versions)
    if _PYODIDE_VERSION.startswith("0.27."):
        # Hardcoded list for v0.27.x - VERIFY these against the actual CDN structure for 0.27.5!
        # These are the most common core files needed for initialization.
        core_files = {
            # --- Core Runtime ---
            "pyodide.js",  # Main JS loader (UMD potentially)
            "pyodide.mjs",  # Main JS loader (ESM)
            "pyodide.asm.js",  # Wasm loader fallback/glue
            "pyodide.asm.wasm",  # The main WebAssembly module
            # --- Standard Library ---
            "python_stdlib.zip",  # Packed standard library
            # --- Metadata/Lock Files ---
            "pyodide-lock.json",  # Package lock file (crucial for loadPackage)
            # --- Potential Depencencies (Less common to preload, but check CDN) ---
            # "distutils.tar",
            # "pyodide_py.tar",
        }
        logger.info(f"[Preload] Using hardcoded core asset list for v{_PYODIDE_VERSION}.")
        # Log the URL that was passed but ignored for clarity
        if (
            manifest_url != f"{_CDN_BASE}/repodata.json"
        ):  # Only log if it differs from default assumption
            logger.warning(f"[Preload] Ignoring provided manifest_url: {manifest_url}")
    else:
        # Placeholder for potentially different logic for other versions
        # (e.g., actually fetching and parsing repodata.json if needed)
        logger.warning(
            f"[Preload] No specific asset list logic for Pyodide v{_PYODIDE_VERSION}. Using empty list."
        )
        core_files = set()
        # If you needed to fetch/parse repodata.json for other versions:
        # try:
        #     logger.info(f"[Preload] Fetching manifest from {manifest_url}")
        #     # ... (fetch manifest_url using _fetch_asset_sync or urllib) ...
        #     # ... (parse JSON) ...
        #     # ... (extract file names based on manifest structure) ...
        # except Exception as e:
        #     logger.error(f"[Preload] Failed to fetch or parse manifest {manifest_url}: {e}")
        #     core_files = set() # Fallback to empty on error

    if not core_files:
        logger.warning("[Preload] The generated core asset list is empty!")
    else:
        logger.debug(f"[Preload] Identified {len(core_files)} essential core files to fetch.")

    # Common packages are loaded on demand *within* the sandbox, not typically preloaded here.
    # Explicitly state this.
    logger.info(
        "[Preload] Common packages (like numpy, pandas) are NOT included in this core preload list. "
        "They will be fetched on demand by the sandbox if needed and cached separately "
        "when `loadPackage` is called."
    )

    # Return the sorted list of unique filenames
    return sorted(list(core_files))


def preload_pyodide_assets(force_download: bool = False):
    """Downloads Pyodide assets to the local cache directory."""
    print("-" * 60)
    print("Starting Pyodide Asset Preloader")
    print(f"Target Pyodide Version: {_PYODIDE_VERSION}")
    print(f"CDN Base URL: {_CDN_BASE}")
    print(f"Cache Directory: {_CACHE_DIR}")
    print(f"Force Re-download: {force_download}")
    print("-" * 60)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"ERROR: Failed to ensure cache directory exists at {_CACHE_DIR}: {e}\nPreloading cannot proceed."
        )
        return
    manifest_url = f"{_CDN_BASE}/repodata.json"  # Dummy URL for v0.27.5 preloader logic
    asset_files = _get_pyodide_asset_list_from_manifest(manifest_url)
    if not asset_files:
        print("ERROR: No asset files were identified.\nPreloading cannot proceed.")
        return
    print(f"\nAttempting to cache/verify {len(asset_files)} assets...")
    cached_count = 0
    verified_count = 0
    error_count = 0
    total_bytes_downloaded = 0
    total_bytes_verified = 0
    max_age = 0 if force_download else (10 * 365 * 24 * 3600)
    num_files = len(asset_files)
    width = len(str(num_files))
    for i, filename in enumerate(asset_files):
        if not filename:
            logger.warning(f"[Preload] Skipping empty filename at index {i}.")
            continue
        file_url = f"{_CDN_BASE}/{filename}"
        progress = f"[{i + 1:>{width}}/{num_files}]"
        local_file_path = _local_path(file_url)
        file_exists = local_file_path.exists()
        is_stale = False
        action = "Fetching"
        if file_exists:
            try:
                file_stat = local_file_path.stat()
                if file_stat.st_size == 0:
                    logger.warning(
                        f"[Preload] Cached file {local_file_path} is empty. Will re-fetch."
                    )
                    file_exists = False
                else:
                    file_age = time.time() - file_stat.st_mtime
                    if file_age >= max_age:
                        is_stale = True
                    else:
                        action = "Verifying" if not force_download else "Re-fetching (forced)"
            except OSError as stat_err:
                logger.warning(
                    f"[Preload] Error checking status of {local_file_path}: {stat_err}. Will re-fetch."
                )
                file_exists = False
                action = "Fetching (stat failed)"
        if file_exists and is_stale and not force_download:
            action = "Re-fetching (stale)"
        display_name = filename if len(filename) <= 60 else filename[:57] + "..."
        print(f"{progress} {action:<25} {display_name:<60} ... ", end="", flush=True)
        try:
            data = _fetch_asset_sync(file_url, max_age_s=max_age)
            file_size_kb = len(data) // 1024
            if action == "Verifying":
                verified_count += 1
                total_bytes_verified += len(data)
                print(f"OK (cached, {file_size_kb:>5} KB)")
            else:
                cached_count += 1
                total_bytes_downloaded += len(data)
                status = "OK" if action.startswith("Fetch") else "OK (updated)"
                print(f"{status} ({file_size_kb:>5} KB)")
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"[Preload] Failed to fetch/cache {file_url}: {e}", exc_info=False)
            error_count += 1
    print("\n" + "-" * 60)
    print("Preload Summary")
    print("-" * 60)
    print(f"Assets already cached & verified: {verified_count}")
    print(f"Assets newly downloaded/updated:  {cached_count}")
    print(f"Total assets processed:          {verified_count + cached_count}")
    print(f"Errors encountered:              {error_count}")
    print("-" * 60)
    print(f"Size of verified assets: {total_bytes_verified / (1024 * 1024):,.1f} MB")
    print(f"Size of downloaded assets: {total_bytes_downloaded / (1024 * 1024):,.1f} MB")
    print(
        f"Total cache size (approx): {(total_bytes_verified + total_bytes_downloaded) / (1024 * 1024):,.1f} MB"
    )
    print("-" * 60)
    if error_count == 0:
        print("Preloading completed successfully. Assets should be cached for offline use.")
    else:
        print(
            f"WARNING: {error_count} assets failed to download. Offline functionality may be incomplete."
        )
    print("-" * 60)


###############################################################################
# Main execution block for preloading (if script is run directly)
###############################################################################
if __name__ == "__main__":
    # Setup logging if run as main script
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Utility for the Python Sandbox module. Includes Pyodide asset preloader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" Examples:\n  Cache Pyodide assets (download if missing/stale):\n    python %(prog)s --preload\n\n  Force re-download of all assets, ignoring cache:\n    python %(prog)s --preload --force """,
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Run the Pyodide asset preloader to cache files required for offline operation.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download of all assets during preload, ignoring existing cache validity.",
    )
    args = parser.parse_args()
    if args.preload:
        preload_pyodide_assets(force_download=args.force)
    else:
        print(
            "This script contains the PythonSandbox tool implementation.\nUse the --preload argument to cache Pyodide assets for offline use.\nExample: python path/to/python_sandbox.py --preload"
        )
