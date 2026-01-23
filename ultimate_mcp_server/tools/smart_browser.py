# ultimate_mcp_server/tools/smart_browser.py
"""
Smart Browser - Standalone Playwright-powered web automation tools for Ultimate MCP Server.

Provides enterprise-grade web automation with comprehensive features for scraping,
testing, and browser automation tasks with built-in security, resilience, and ML capabilities.

Refactored into standalone functions for compatibility with the MCP tool registration system.
State and lifecycle are managed via global variables and explicit init/shutdown calls.
"""

# Python Standard Library Imports
import asyncio
import atexit
import base64
import concurrent.futures
import difflib
import functools
import hashlib
import json
import os
import random
import re
import signal
import sqlite3
import subprocess
import textwrap
import threading
import time
import unicodedata
import urllib.parse

# Python Standard Library Type Hinting and Collections Imports
from collections import deque
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlparse

# Third-Party Library Imports
import aiofiles
import httpx
from bs4 import BeautifulSoup
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from playwright._impl._errors import Error as PlaywrightException
from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import Browser, BrowserContext, Locator, Page, async_playwright

# First-Party Library Imports (MCP Specific)
from ultimate_mcp_server.config import SmartBrowserConfig, get_config

# Assuming these are available and work standalone
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import get_provider, parse_model_string
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError

# Import STANDALONE filesystem and completion tools
from ultimate_mcp_server.tools.completion import chat_completion
from ultimate_mcp_server.tools.filesystem import (
    create_directory,
    get_unique_filepath,
    read_file,
    write_file,
)
from ultimate_mcp_server.utils import get_logger

# For loop binding and forked process detection
_pid = os.getpid()

# --- Global Logger ---
logger = get_logger("ultimate_mcp_server.tools.smart_browser")

# --- Load External Tools Dynamically (Best Effort) ---
# This allows using tools defined later without circular imports at top level
# We'll look them up by name when needed in autopilot.
_filesystem_tools_module = None
_completion_tools_module = None


def _get_filesystem_tool(name):
    global _filesystem_tools_module
    if _filesystem_tools_module is None:
        import ultimate_mcp_server.tools.filesystem as fs

        _filesystem_tools_module = fs
    tool_func = getattr(_filesystem_tools_module, name, None)
    return tool_func


def _get_completion_tool(name):
    global _completion_tools_module
    if _completion_tools_module is None:
        import ultimate_mcp_server.tools.completion as cm

        _completion_tools_module = cm
    tool_func = getattr(_completion_tools_module, name, None)
    return tool_func


# --- Global Configuration Variables ---
# (These will be populated by _ensure_initialized)
_sb_state_key_b64_global: Optional[str] = None
_sb_max_tabs_global: int = 5
_sb_tab_timeout_global: int = 300
_sb_inactivity_timeout_global: int = 600
_headless_mode_global: bool = True
_vnc_enabled_global: bool = False
_vnc_password_global: Optional[str] = None
_proxy_pool_str_global: str = ""
_proxy_allowed_domains_str_global: str = "*"
_vault_allowed_paths_str_global: str = "secret/data/,kv/data/"
_max_widgets_global: int = 300
_max_section_chars_global: int = 5000
_dom_fp_limit_global: int = 20000
_llm_model_locator_global: str = "openai/gpt-4.1-mini"  # Updated default
_retry_after_fail_global: int = 1
_seq_cutoff_global: float = 0.72
_area_min_global: int = 400
_high_risk_domains_set_global: Set[str] = set()
_SB_INTERNAL_BASE_PATH_STR: Optional[str] = None
_STATE_FILE: Optional[Path] = None
_LOG_FILE: Optional[Path] = None
_CACHE_DB: Optional[Path] = None
_READ_JS_CACHE: Optional[Path] = None
_PROXY_CONFIG_DICT: Optional[Dict[str, Any]] = None
_PROXY_ALLOWED_DOMAINS_LIST: Optional[List[str]] = None
_ALLOWED_VAULT_PATHS: Set[str] = set()

# --- Global State Variables ---
_pw: Optional[async_playwright] = None
_browser: Optional[Browser] = None
_ctx: Optional[BrowserContext] = None  # Shared context
_vnc_proc: Optional[subprocess.Popen] = None
_last_hash: str | None = None
_js_lib_cached: Set[str] = set()
_db_connection: sqlite3.Connection | None = None
_locator_cache_cleanup_task_handle: Optional[asyncio.Task] = None
_inactivity_monitor_task_handle: Optional[asyncio.Task] = None  # New handle for monitor task
_last_activity: float = 0.0  # Global last activity timestamp

# --- Locks ---
_init_lock = asyncio.Lock()
_playwright_lock = asyncio.Lock()
_js_lib_lock = asyncio.Lock()
_audit_log_lock = asyncio.Lock()
_db_conn_pool_lock = threading.RLock()  # Keep RLock for sync DB access from async context
_shutdown_lock = asyncio.Lock()

# --- Flags ---
_is_initialized = False
_shutdown_initiated = False

# --- Thread Pool ---
_cpu_count = os.cpu_count() or 1
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, _cpu_count * 2 + 4), thread_name_prefix="sb_worker"
)

# --- Helper Functions ---


def _update_activity():
    """Updates the global activity timestamp. Should be called by user-facing tool functions."""
    global _last_activity
    now = time.monotonic()
    logger.debug(f"Updating last activity timestamp to {now}")
    _last_activity = now


def _get_pool():  # Keep as is
    global _thread_pool, _pid
    if _pid != os.getpid():
        _thread_pool.shutdown(wait=False)
        pool_max_workers = min(32, _sb_max_tabs_global * 2)
        _thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=pool_max_workers, thread_name_prefix="sb_worker"
        )
        _pid = os.getpid()
    return _thread_pool


# --- Encryption ---
CIPHER_VERSION = b"SB1"
AAD_TAG = b"smart-browser-state-v1"


def _key() -> bytes | None:  # Uses global _sb_state_key_b64_global
    """Get AES-GCM key from the globally set config value."""
    if not _sb_state_key_b64_global:
        return None
    try:
        decoded = base64.b64decode(_sb_state_key_b64_global)
        key_length = len(decoded)
        if key_length not in (16, 24, 32):
            logger.warning(f"Invalid SB State Key length: {key_length} bytes. Need 16, 24, or 32.")
            return None
        return decoded
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid base64 SB State Key: {e}")
        return None


def _enc(buf: bytes) -> bytes:  # Uses global _key
    """Encrypt data using AES-GCM with AAD if key is set."""
    k = _key()
    if not k:
        logger.debug("SB_STATE_KEY not set. Skipping encryption for state.")
        return buf
    try:
        nonce = os.urandom(12)
        cipher = AESGCM(k)
        encrypted_data = cipher.encrypt(nonce, buf, AAD_TAG)
        result = CIPHER_VERSION + nonce + encrypted_data
        return result
    except Exception as e:
        logger.error(f"Encryption failed: {e}", exc_info=True)
        raise RuntimeError(f"Encryption failed: {e}") from e


def _dec(buf: bytes) -> bytes | None:  # Uses global _key, _STATE_FILE
    """Decrypt data using AES-GCM with AAD if key is set and buffer looks encrypted."""
    k = _key()
    if not k:
        logger.debug("SB_STATE_KEY not set. Assuming state is unencrypted.")
        try:
            stripped_buf = buf.strip()
            if stripped_buf.startswith(b"{") or stripped_buf.startswith(b"["):
                return buf
            else:
                logger.warning("Unencrypted state file doesn't look like JSON. Ignoring.")
                return None
        except Exception:
            logger.warning("Error checking unencrypted state file format. Ignoring.")
            return None

    if not buf.startswith(CIPHER_VERSION):
        logger.warning(
            "State file exists but lacks expected encryption header. Treating as legacy/invalid."
        )
        if _STATE_FILE and _STATE_FILE.exists():
            try:
                _STATE_FILE.unlink()
            except Exception:
                pass
        return None

    hdr_len = len(CIPHER_VERSION)
    nonce_len = 12
    min_len = hdr_len + nonce_len + 1  # Header + Nonce + Tag(at least 1 byte)
    if len(buf) < min_len:
        logger.error("State file too short to be valid encrypted data")
        return None

    _hdr_start = 0
    _hdr_end = hdr_len
    _nonce_start = _hdr_end
    _nonce_end = _hdr_end + nonce_len
    _ciphertext_start = _nonce_end

    _HDR = buf[_hdr_start:_hdr_end]
    nonce = buf[_nonce_start:_nonce_end]
    ciphertext = buf[_ciphertext_start:]

    try:
        cipher = AESGCM(k)
        decrypted_data = cipher.decrypt(nonce, ciphertext, AAD_TAG)
        return decrypted_data
    except InvalidTag:
        logger.error("Decryption failed: Invalid tag (tampered/wrong key?)")
        if _STATE_FILE and _STATE_FILE.exists():
            try:
                _STATE_FILE.unlink()
            except Exception:
                pass
        raise RuntimeError("State-file authentication failed (InvalidTag)") from None
    except Exception as e:
        logger.error(f"Decryption failed: {e}.", exc_info=True)
        if _STATE_FILE and _STATE_FILE.exists():
            try:
                _STATE_FILE.unlink()
            except Exception:
                pass
        return None


# --- Locator Cache DB ---
def _get_db_connection() -> sqlite3.Connection:  # Uses global _db_connection, _CACHE_DB
    """Get or create the single shared SQLite connection."""
    global _db_connection
    with _db_conn_pool_lock:
        if _db_connection is None:
            if _CACHE_DB is None:
                raise RuntimeError("Database path (_CACHE_DB) not initialized before DB access.")
            try:
                conn = sqlite3.connect(
                    _CACHE_DB,
                    check_same_thread=False,
                    isolation_level=None,
                    timeout=10,
                )
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("PRAGMA busy_timeout = 10000")
                _db_connection = conn
                logger.info(f"Initialized SQLite DB connection to {_CACHE_DB}")
            except sqlite3.Error as e:
                logger.critical(
                    f"Failed to connect/init SQLite DB at {_CACHE_DB}: {e}", exc_info=True
                )
                raise RuntimeError(f"Failed to initialize database: {e}") from e
        return _db_connection


def _close_db_connection():  # Uses global _db_connection
    """Close the SQLite connection."""
    global _db_connection
    with _db_conn_pool_lock:
        if _db_connection is not None:
            conn_to_close = _db_connection
            _db_connection = None  # Set to None first
            try:
                conn_to_close.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.Error as e:
                logger.warning(f"Error during WAL checkpoint before closing DB: {e}")
            try:
                conn_to_close.close()
                logger.info("Closed SQLite DB connection.")
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite DB connection: {e}")


atexit.register(_close_db_connection)  # Keep atexit hook


def _init_locator_cache_db_sync():  # Uses global _CACHE_DB
    """Synchronous DB schema initialization for the locator cache."""
    conn = None
    if _CACHE_DB is None:
        logger.error("Cannot initialize locator DB: Path not set.")
        return  # Cannot proceed without path
    try:
        conn = _get_db_connection()
        with closing(conn.cursor()) as cursor:
            create_table_sql = """CREATE TABLE IF NOT EXISTS selector_cache(
                    key       TEXT,
                    selector  TEXT NOT NULL,
                    dom_fp    TEXT NOT NULL,
                    hits      INTEGER DEFAULT 1,
                    created_ts INTEGER DEFAULT (strftime('%s', 'now')),
                    last_hit  INTEGER DEFAULT (strftime('%s', 'now')),
                    PRIMARY KEY (key, dom_fp)
                );"""
            cursor.execute(create_table_sql)
            try:
                cursor.execute("SELECT last_hit FROM selector_cache LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("Adding last_hit column to selector_cache table...")
                alter_table_sql = "ALTER TABLE selector_cache ADD COLUMN last_hit INTEGER DEFAULT(strftime('%s','now'))"
                cursor.execute(alter_table_sql)
            logger.info(f"Enhanced Locator cache DB schema initialized/verified at {_CACHE_DB}")
    except sqlite3.Error as e:
        logger.critical(f"Failed to initialize locator cache DB schema: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize locator cache database: {e}") from e
    except RuntimeError as e:  # Catch error from _get_db_connection if path is missing
        logger.critical(f"Failed to get DB connection for schema init: {e}")
        raise


def _cache_put_sync(key: str, selector: str, dom_fp: str) -> None:  # Uses global _get_db_connection
    """Synchronous write/update to the locator cache."""
    try:
        conn = _get_db_connection()
        insert_sql = """INSERT INTO selector_cache(key, selector, dom_fp, created_ts, last_hit)
               VALUES (?, ?, ?, strftime('%s', 'now'), strftime('%s', 'now'))
               ON CONFLICT(key, dom_fp) DO UPDATE SET
                 hits = hits + 1,
                 last_hit = strftime('%s', 'now')
               WHERE key = excluded.key AND dom_fp = excluded.dom_fp;"""
        params = (key, selector, dom_fp)
        conn.execute(insert_sql, params)
    except sqlite3.Error as e:
        key_prefix = key[:8]
        logger.error(f"Failed to write to locator cache (key prefix={key_prefix}...): {e}")
    except RuntimeError as e:
        logger.error(f"Failed to get DB connection for cache put: {e}")


def _cache_delete_sync(key: str) -> None:  # Uses global _get_db_connection
    """Synchronously delete an entry from the locator cache by key."""
    key_prefix = key[:8]
    try:
        conn = _get_db_connection()
        logger.debug(f"Deleting stale cache entry with key prefix: {key_prefix}...")
        delete_sql = "DELETE FROM selector_cache WHERE key = ?"
        params = (key,)
        cursor = conn.execute(delete_sql, params)
        if cursor.rowcount > 0:
            logger.debug(f"Successfully deleted stale cache entry {key_prefix}...")
    except sqlite3.Error as e:
        logger.error(f"Failed to delete stale cache entry (key prefix={key_prefix}...): {e}")
    except RuntimeError as e:
        logger.error(f"Failed to get DB connection for cache delete: {e}")
    except Exception as e:
        logger.error(
            f"Unexpected error deleting cache entry (key prefix={key_prefix}...): {e}",
            exc_info=True,
        )


def _cache_get_sync(key: str, dom_fp: str) -> Optional[str]:  # Uses global _get_db_connection
    """Synchronous read from cache, checking fingerprint. Deletes stale entries."""
    row = None
    try:
        conn = _get_db_connection()
        with closing(conn.cursor()) as cursor:
            select_sql = "SELECT selector FROM selector_cache WHERE key=? AND dom_fp=?"
            params_select = (key, dom_fp)
            cursor.execute(select_sql, params_select)
            row = cursor.fetchone()
            if row:
                update_sql = "UPDATE selector_cache SET last_hit = strftime('%s', 'now') WHERE key=? AND dom_fp=?"
                params_update = (key, dom_fp)
                conn.execute(update_sql, params_update)
                selector = row[0]
                return selector
            # If row not found with matching key and fp, check if key exists at all
            check_key_sql = "SELECT 1 FROM selector_cache WHERE key=? LIMIT 1"
            params_check = (key,)
            cursor.execute(check_key_sql, params_check)
            key_exists = cursor.fetchone()
            if key_exists:
                key_prefix = key[:8]
                logger.debug(
                    f"Cache key '{key_prefix}...' found but DOM fingerprint mismatch. Deleting."
                )
                _cache_delete_sync(key)
    except sqlite3.Error as e:
        logger.error(f"Failed to read from locator cache (key={key}): {e}")
    except RuntimeError as e:
        logger.error(f"Failed to get DB connection for cache get: {e}")
    return None


# --- Locator Cache Cleanup ---
def _cleanup_locator_cache_db_sync(
    retention_days: int = 90,
) -> int:  # Uses global _get_db_connection
    """Synchronously removes old entries from the locator cache DB."""
    deleted_count = 0
    if retention_days <= 0:
        logger.info("Locator cache cleanup skipped (retention_days <= 0).")
        return 0
    try:
        conn = _get_db_connection()
        # Note: f-string for time modification is safe as retention_days is an int
        cutoff_time_sql = f"strftime('%s', 'now', '-{retention_days} days')"
        logger.info(
            f"Running locator cache cleanup: Removing entries older than {retention_days} days or with hits=0..."
        )
        with closing(conn.cursor()) as cursor:
            # Use placeholder for the time comparison to be safer if possible, but strftime makes it tricky
            # For this controlled use case, f-string is acceptable.
            delete_sql = (
                f"DELETE FROM selector_cache WHERE created_ts < ({cutoff_time_sql}) OR hits = 0"
            )
            cursor.execute(delete_sql)
            deleted_count = cursor.rowcount
            # Vacuum only if significant changes were made
            if deleted_count > 500:
                logger.info(f"Vacuuming locator cache DB after deleting {deleted_count} entries...")
                cursor.execute("VACUUM;")
        logger.info(f"Locator cache cleanup finished. Removed {deleted_count} old entries.")
        return deleted_count
    except sqlite3.Error as e:
        logger.error(f"Error during locator cache cleanup: {e}")
        return -1
    except RuntimeError as e:
        logger.error(f"Failed to get DB connection for cache cleanup: {e}")
        return -1
    except Exception as e:
        logger.error(f"Unexpected error during locator cache cleanup: {e}", exc_info=True)
        return -1


async def _locator_cache_cleanup_task(
    interval_seconds: int = 24 * 60 * 60,
):  # Uses global _get_pool
    """Background task to periodically run locator cache cleanup."""
    if interval_seconds <= 0:
        logger.info("Locator cache cleanup task disabled (interval <= 0).")
        return
    logger.info(f"Locator cache cleanup task started. Running every {interval_seconds} seconds.")
    # Initial delay before first run
    await asyncio.sleep(interval_seconds)
    while True:
        try:
            loop = asyncio.get_running_loop()
            pool = _get_pool()
            result_count = await loop.run_in_executor(pool, _cleanup_locator_cache_db_sync)
            if result_count < 0:
                logger.warning("Locator cache cleanup run encountered an error.")
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Locator cache cleanup task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in locator cache cleanup task loop: {e}", exc_info=True)
            # Wait longer after an error before retrying
            await asyncio.sleep(60 * 5)


# --- Audit Log ---
_salt = os.urandom(16)


def _sanitize_for_log(obj: Any) -> Any:  # Keep as is
    # ... (implementation largely unchanged, but split multi-line expressions) ...
    if isinstance(obj, str):
        try:
            # Remove control characters
            s = re.sub(r"[\x00-\x1f\x7f]", "", obj)
            # JSON encode to handle quotes, backslashes etc.
            encoded = json.dumps(s)
            # Remove the outer quotes added by json.dumps
            if len(encoded) >= 2:
                return encoded[1:-1]
            else:
                return ""
        except TypeError:
            return "???"  # Should not happen for str, but safety first
    elif isinstance(obj, dict):
        # Recursively sanitize dictionary values
        new_dict = {}
        for k, v in obj.items():
            sanitized_v = _sanitize_for_log(v)
            str_k = str(k)  # Ensure keys are strings
            new_dict[str_k] = sanitized_v
        return new_dict
    elif isinstance(obj, list):
        # Recursively sanitize list items
        new_list = []
        for item in obj:
            sanitized_item = _sanitize_for_log(item)
            new_list.append(sanitized_item)
        return new_list
    elif isinstance(obj, (int, float, bool, type(None))):
        # Allow simple types directly
        return obj
    else:
        # Attempt to stringify, sanitize, and encode other types
        try:
            s = str(obj)
            s = re.sub(r"[\x00-\x1f\x7f]", "", s)
            encoded = json.dumps(s)
            if len(encoded) >= 2:
                return encoded[1:-1]
            else:
                return ""
        except Exception:
            # Fallback for types that fail stringification/encoding
            return "???"


_EVENT_EMOJI_MAP = {  # Keep as is
    # ... (emoji map unchanged) ...
    "browser_start": "🚀",
    "browser_shutdown": "🛑",
    "browser_shutdown_complete": "🏁",
    "browser_context_create": "➕",
    "browser_incognito_context": "🕶️",
    "browser_context_close_shared": "➖",
    "browser_close": "🚪",
    "page_open": "📄",
    "page_close": "덮",
    "page_error": "🔥",
    "tab_timeout": "⏱️",
    "tab_cancelled": "🚫",
    "tab_error": "💥",
    "navigate": "➡️",
    "navigate_start": "➡️",
    "navigate_success": "✅",
    "navigate_fail_playwright": "❌",
    "navigate_fail_unexpected": "💣",
    "navigate_wait_selector_ok": "👌",
    "navigate_wait_selector_timeout": "⏳",
    "page_state_extracted": "ℹ️",
    "browse_fail_proxy_disallowed": "🛡️",
    "click": "🖱️",
    "click_success": "🖱️✅",
    "click_fail_notfound": "🖱️❓",
    "click_fail_playwright": "🖱️❌",
    "click_fail_unexpected": "🖱️💣",
    "type": "⌨️",
    "type_success": "⌨️✅",
    "type_fail_secret": "⌨️🔑",
    "type_fail_notfound": "⌨️❓",
    "type_fail_playwright": "⌨️❌",
    "type_fail_unexpected": "⌨️💣",
    "scroll": "↕️",
    "locator_cache_hit": "⚡",
    "locator_heuristic_match": "🧠",
    "locator_llm_pick": "🤖🎯",
    "locator_fail_all": "❓❓",
    "locator_text_fallback": "✍️",
    "locator_success": "🎯",
    "locator_fail": "❓",
    "download": "💾",
    "download_navigate": "🚚",
    "download_success": "💾✅",
    "download_fail_notfound": "💾❓",
    "download_fail_timeout": "💾⏱️",
    "download_fail_playwright": "💾❌",
    "download_fail_unexpected": "💾💣",
    "download_pdf_http": "📄💾",
    "download_direct_success": "✨💾",
    "download_pdf_error": "📄🔥",
    "download_site_pdfs_complete": "📚✅",
    "table_extract_success": "📊✅",
    "table_extract_error": "📊❌",
    "docs_collected_success": "📖✅",
    "docs_harvest": "📖",
    "search": "🔍",
    "search_start": "🔍➡️",
    "search_complete": "🔍✅",
    "search_captcha": "🤖",
    "search_no_results_selector": "🤷",
    "search_error_playwright": "🔍❌",
    "search_error_unexpected": "🔍💣",
    "macro_plan": "📝",
    "macro_plan_generated": "📝✅",
    "macro_plan_empty": "📝🤷",
    "macro_step_result": "▶️",
    "macro_complete": "🎉",
    "macro_finish_action": "🏁",
    "macro_error": "💥",
    "macro_exceeded_rounds": "🔄",
    "macro_fail_step": "❌",
    "macro_error_tool": "🛠️💥",
    "macro_error_unexpected": "💣💥",
    "macro_navigate": "🗺️➡️",
    "click_extract_navigate": "🖱️🗺️",
    "click_extract_success": "🖱️✅✨",
    "fill_form_navigate": "✍️🗺️",
    "fill_form_field": "✍️",
    "fill_form_submit": "✔️",
    "fill_form_success": "✍️✅",
    "autopilot_run": "🧑‍✈️",
    "autopilot_step_start": "▶️",
    "autopilot_step_success": "✅",
    "autopilot_step_fail": "❌",
    "autopilot_replan_success": "🧠🔄",
    "autopilot_replan_fail": "🧠❌",
    "autopilot_max_steps": "🚫🔄",
    "autopilot_plan_end": "🏁",
    "autopilot_critical_error": "💥🧑‍✈️",
    "parallel_navigate": "🚦➡️",
    "parallel_url_error": "🚦🔥",
    "parallel_process_complete": "🚦🏁",
    "retry": "⏳",
    "retry_fail": "⚠️",
    "retry_fail_unexpected": "💣⚠️",
    "retry_unexpected": "⏳💣",
    "llm_call_complete": "🤖💬",
}


async def _log(event: str, **details):  # Uses global _last_hash, _salt, _LOG_FILE
    """Append a hash-chained entry to the audit log asynchronously."""
    global _last_hash, _salt
    if _LOG_FILE is None:  # Need to check if path is set
        logger.warning(f"Audit log skipped for event '{event}': Log file path not initialized.")
        return
    now_utc = datetime.now(timezone.utc)
    ts_iso = now_utc.isoformat()
    sanitized_details = _sanitize_for_log(details)
    emoji_key = _EVENT_EMOJI_MAP.get(event, "❓")
    async with _audit_log_lock:
        current_last_hash = _last_hash
        entry = {
            "ts": ts_iso,
            "event": event,
            "details": sanitized_details,
            "prev": current_last_hash,
            "emoji": emoji_key,
        }
        entry_json = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        payload = _salt + entry_json.encode("utf-8")
        hasher = hashlib.sha256(payload)
        h = hasher.hexdigest()
        log_entry_data = {"hash": h, **entry}
        log_entry_line = json.dumps(log_entry_data, separators=(",", ":")) + "\n"
        try:
            async with aiofiles.open(_LOG_FILE, "a", encoding="utf-8") as f:
                await f.write(log_entry_line)
                await f.flush()
                # os.fsync is sync, run in executor if strict atomic persistence needed
                # loop = asyncio.get_running_loop()
                # await loop.run_in_executor(_get_pool(), os.fsync, f.fileno())
            _last_hash = h
        except IOError as e:
            logger.error(f"Failed to write to audit log {_LOG_FILE}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing audit log: {e}", exc_info=True)


def _init_last_hash():  # Uses global _LOG_FILE, _last_hash
    """Initializes the last hash from the audit log file."""
    global _last_hash
    if _LOG_FILE is None:
        logger.info("Audit log initialization skipped: _LOG_FILE path not set yet.")
        return
    if _LOG_FILE.exists():
        try:
            with open(_LOG_FILE, "rb") as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                if file_size == 0:  # Empty file
                    _last_hash = None
                    logger.info("Audit log file found but is empty.")
                    return

                # Read backwards efficiently (simplified version)
                buffer_size = 4096
                last_line = b""
                read_pos = max(0, file_size - buffer_size)

                while read_pos >= 0:
                    f.seek(read_pos)
                    buffer = f.read(buffer_size)
                    lines = buffer.splitlines()  # Split by \n, \r, or \r\n
                    if lines:
                        # Find the last *complete* line in the buffer
                        # A complete line will either be the last one if the buffer ends with newline,
                        # or the second to last one otherwise.
                        is_last_line_complete = buffer.endswith(b"\n") or buffer.endswith(b"\r")
                        if is_last_line_complete:
                            last_line_candidate = lines[-1]
                        elif len(lines) > 1:
                            last_line_candidate = lines[-2]  # Use second-to-last if last is partial
                        else:  # File smaller than buffer or only one partial line
                            last_line_candidate = b""  # Assume partial

                        # Ensure candidate is not empty and potentially valid JSON before breaking
                        if last_line_candidate.strip().startswith(b"{"):
                            last_line = last_line_candidate
                            break  # Found a likely valid, complete line

                    if read_pos == 0:
                        # Reached beginning, check if the first line itself is the only one
                        if len(lines) == 1 and lines[0].strip().startswith(b"{"):
                            last_line = lines[0]
                        break

                    # Move back, overlapping slightly to ensure line endings are caught
                    read_pos = max(0, read_pos - (buffer_size // 2))

            if last_line:
                try:
                    decoded_line = last_line.decode("utf-8")
                    last_entry = json.loads(decoded_line)
                    found_hash = last_entry.get("hash")
                    _last_hash = found_hash
                    if _last_hash:
                        hash_preview = _last_hash[:8]
                        logger.info(
                            f"Initialized audit log chain from last hash: {hash_preview}..."
                        )
                    else:
                        logger.warning(
                            "Last log entry parsed but missing 'hash'. Starting new chain."
                        )
                        _last_hash = None
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(f"Error decoding last line of audit log: {e}. Starting new chain.")
                    _last_hash = None
            else:
                logger.info("Could not read last complete line from audit log. Starting new chain.")
                _last_hash = None
        except Exception as e:
            logger.error(
                f"Failed to read last hash from audit log {_LOG_FILE}: {e}. Starting new chain.",
                exc_info=True,
            )
            _last_hash = None
    else:
        logger.info("No existing audit log found. Starting new chain.")
        _last_hash = None


# --- Resilient Decorator ---
def resilient(max_attempts: int = 3, backoff: float = 0.3):  # Uses global _log
    """Decorator for async functions; retries on common transient errors."""

    def wrap(fn):
        import functools  # Ensure functools is imported locally for the decorator

        @functools.wraps(fn)
        async def inner(*a, **kw):
            attempt = 0
            while True:
                try:
                    if attempt > 0:
                        # Calculate jittered delay before retrying
                        delay_factor = 2 ** (attempt - 1)
                        base_delay = backoff * delay_factor
                        jitter = random.uniform(0.8, 1.2)
                        jitter_delay = base_delay * jitter
                        await asyncio.sleep(jitter_delay)
                    result = await fn(*a, **kw)
                    return result
                except (PlaywrightTimeoutError, httpx.RequestError, asyncio.TimeoutError) as e:
                    attempt += 1
                    func_name = getattr(fn, "__name__", "unknown_func")
                    if attempt >= max_attempts:
                        await _log(
                            "retry_fail", func=func_name, attempts=max_attempts, error=str(e)
                        )
                        raise ToolError(
                            f"Operation '{func_name}' failed after {max_attempts} attempts: {e}"
                        ) from e
                    # Calculate delay for logging purposes (actual sleep is at loop start)
                    delay_factor_log = 2 ** (attempt - 1)
                    base_delay_log = backoff * delay_factor_log
                    jitter_log = random.uniform(
                        0.8, 1.2
                    )  # Recalculate for log consistency, might differ slightly from sleep
                    delay_log = base_delay_log * jitter_log
                    rounded_delay = round(delay_log, 2)
                    await _log(
                        "retry",
                        func=func_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        sleep=rounded_delay,
                        error=str(e),
                    )
                    # Sleep moved to start of the next iteration
                except (
                    ToolError,
                    ValueError,
                    TypeError,
                    KeyError,
                    KeyboardInterrupt,
                    sqlite3.Error,
                ):
                    # Non-retryable errors specific to the application or unrecoverable
                    raise  # Re-raise immediately
                except Exception as e:
                    # Catch other unexpected exceptions and retry them
                    attempt += 1
                    func_name = getattr(fn, "__name__", "unknown_func")
                    if attempt >= max_attempts:
                        await _log(
                            "retry_fail_unexpected",
                            func=func_name,
                            attempts=max_attempts,
                            error=str(e),
                        )
                        raise ToolError(
                            f"Operation '{func_name}' failed with unexpected error after {max_attempts} attempts: {e}"
                        ) from e
                    # Calculate delay for logging
                    delay_factor_log = 2 ** (attempt - 1)
                    base_delay_log = backoff * delay_factor_log
                    jitter_log = random.uniform(0.8, 1.2)
                    delay_log = base_delay_log * jitter_log
                    rounded_delay = round(delay_log, 2)
                    await _log(
                        "retry_unexpected",
                        func=func_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        sleep=rounded_delay,
                        error=str(e),
                    )
                    # Sleep moved to start of the next iteration

        return inner

    return wrap


# --- Secret Vault ---
def _update_vault_paths():  # Uses global _vault_allowed_paths_str_global, _ALLOWED_VAULT_PATHS
    """Parse the vault allowed paths string from global config into the global set."""
    global _ALLOWED_VAULT_PATHS
    new_set = set()
    path_list = _vault_allowed_paths_str_global.split(",")
    for path in path_list:
        stripped_path = path.strip()
        if stripped_path:
            # Ensure path ends with a slash for prefix matching
            formatted_path = stripped_path.rstrip("/") + "/"
            new_set.add(formatted_path)
    _ALLOWED_VAULT_PATHS = new_set


def get_secret(path_key: str) -> str:  # Uses global _ALLOWED_VAULT_PATHS
    """Retrieves secret from environment or HashiCorp Vault."""
    # ... (implementation largely unchanged, relies on _ALLOWED_VAULT_PATHS global, split multi-line expressions) ...
    if path_key.startswith("env:"):
        var = path_key[4:]
        val = os.getenv(var)
        if val is None:
            raise ToolInputError(f"Environment variable secret '{var}' not set.")
        return val
    if path_key.startswith("vault:"):
        try:
            import hvac
        except ImportError as e:
            raise RuntimeError("'hvac' library required for Vault access.") from e
        addr = os.getenv("VAULT_ADDR")
        token = os.getenv("VAULT_TOKEN")
        if not addr or not token:
            raise RuntimeError("VAULT_ADDR and VAULT_TOKEN environment variables must be set.")

        vault_uri_part = path_key[len("vault:") :]
        if "://" in vault_uri_part:
            raise ValueError("Vault path cannot contain '://'. Use format 'mount/path#key'.")
        if "#" not in vault_uri_part:
            raise ValueError("Vault path must include '#key'. Use format 'mount/path#key'.")

        path_part_raw, key_name = vault_uri_part.split("#", 1)
        path_part = path_part_raw.strip("/")

        if not _ALLOWED_VAULT_PATHS:
            _update_vault_paths()  # Ensure allowed paths are populated

        # Check if the requested path is allowed
        path_to_check = path_part + "/"  # Ensure trailing slash for prefix check
        found_prefix = False
        for prefix in _ALLOWED_VAULT_PATHS:
            if path_to_check.startswith(prefix):
                found_prefix = True
                break
        if not found_prefix:
            logger.warning(
                f"Access denied for Vault path '{path_part}'. Allowed prefixes: {_ALLOWED_VAULT_PATHS}"
            )
            raise ValueError(f"Access to Vault path '{path_part}' is not allowed.")

        client = hvac.Client(url=addr, token=token)
        if not client.is_authenticated():
            raise RuntimeError(f"Vault authentication failed for {addr}.")

        path_segments = path_part.split("/")
        if not path_segments:
            raise ValueError(f"Invalid Vault path format: '{path_part}'")

        mount_point = path_segments[0]
        rest_segments = path_segments[1:]
        secret_sub_path = "/".join(rest_segments)

        # Try KV v2 first
        try:
            resp_v2 = client.secrets.kv.v2.read_secret_version(
                mount_point=mount_point, path=secret_sub_path
            )
            data_v2 = resp_v2["data"]["data"]
            if key_name in data_v2:
                return data_v2[key_name]
            else:
                # Key not found in this v2 secret
                pass  # Will proceed to check v1 or raise later
        except hvac.exceptions.InvalidPath:
            # Path doesn't exist in KV v2 mount, try KV v1
            pass
        except (KeyError, TypeError):
            # Error accessing nested data['data'], indicates issue with response structure
            logger.warning(
                f"Unexpected response structure from Vault KV v2 for path '{path_part}'."
            )
            pass
        except Exception as e:
            logger.error(f"Error reading Vault KV v2 secret '{path_part}': {e}")
            # Don't raise immediately, allow fallback to v1 if configured
            pass

        # Try KV v1
        try:
            resp_v1 = client.secrets.kv.v1.read_secret(
                mount_point=mount_point, path=secret_sub_path
            )
            data_v1 = resp_v1["data"]
            if key_name in data_v1:
                return data_v1[key_name]
            else:
                # Key not found in v1 either
                raise KeyError(
                    f"Key '{key_name}' not found in Vault secret at '{path_part}' (tried KV v2 & v1)."
                )
        except hvac.exceptions.InvalidPath:
            # Path not found in v1 either (and wasn't found in v2 or errored)
            raise KeyError(
                f"Secret path '{path_part}' not found in Vault (tried KV v2 & v1)."
            ) from None
        except KeyError:
            # Re-raise the KeyError from the v1 check if key wasn't found there
            raise KeyError(f"Key '{key_name}' not found at '{path_part}' (KV v1).") from None
        except Exception as e:
            logger.error(f"Error reading Vault KV v1 secret '{path_part}': {e}")
            raise RuntimeError(f"Failed to read Vault secret (KV v1): {e}") from e

    # If scheme is not 'env:' or 'vault:'
    raise ValueError(f"Unknown secret scheme or invalid path format: {path_key}")


# --- Playwright Lifecycle ---
def _update_proxy_settings():  # Uses globals
    """Parse global proxy config strings into usable dict/list."""
    global _PROXY_CONFIG_DICT, _PROXY_ALLOWED_DOMAINS_LIST
    _PROXY_CONFIG_DICT = None  # Reset
    if _proxy_pool_str_global:
        # Split and filter empty strings
        proxies_raw = _proxy_pool_str_global.split(";")
        proxies = []
        for p in proxies_raw:
            stripped_p = p.strip()
            if stripped_p:
                proxies.append(stripped_p)

        if proxies:
            chosen_proxy = random.choice(proxies)
            try:
                parsed = urlparse(chosen_proxy)
                # Basic validation
                is_valid_scheme = parsed.scheme in ("http", "https", "socks5", "socks5h")
                has_netloc = bool(parsed.netloc)
                no_fragment = "#" not in chosen_proxy  # Fragments not allowed in proxy URL itself

                if is_valid_scheme and has_netloc and no_fragment:
                    # Construct base server URL without credentials
                    if parsed.port:
                        hostname_port = f"{parsed.hostname}:{parsed.port}"
                    else:
                        hostname_port = parsed.hostname
                    server_url = f"{parsed.scheme}://{hostname_port}"

                    proxy_dict: Dict[str, Any] = {"server": server_url}
                    if parsed.username:
                        unquoted_username = urllib.parse.unquote(parsed.username)
                        proxy_dict["username"] = unquoted_username
                    if parsed.password:
                        unquoted_password = urllib.parse.unquote(parsed.password)
                        proxy_dict["password"] = unquoted_password

                    _PROXY_CONFIG_DICT = proxy_dict
                    logger.info(f"Proxy settings parsed: Using {proxy_dict.get('server')}")
                else:
                    logger.warning(f"Invalid proxy URL format/scheme: '{chosen_proxy}'. Skipping.")
            except Exception as e:
                logger.warning(f"Error parsing proxy URL '{chosen_proxy}': {e}")

    # Parse allowed domains
    if not _proxy_allowed_domains_str_global or _proxy_allowed_domains_str_global == "*":
        _PROXY_ALLOWED_DOMAINS_LIST = None  # None means allow all
        logger.info("Proxy allowed domains: * (all allowed)")
    else:
        domains_raw = _proxy_allowed_domains_str_global.split(",")
        domains = []
        for d in domains_raw:
            stripped_d = d.strip()
            if stripped_d:
                lower_d = stripped_d.lower()
                domains.append(lower_d)

        # Ensure domains start with a dot for proper suffix matching
        new_domain_list = []
        for d in domains:
            if d.startswith("."):
                new_domain_list.append(d)
            else:
                new_domain_list.append("." + d)
        _PROXY_ALLOWED_DOMAINS_LIST = new_domain_list
        logger.info(f"Proxy allowed domains parsed: {_PROXY_ALLOWED_DOMAINS_LIST}")


def _get_proxy_config() -> Optional[Dict[str, Any]]:  # Uses global _PROXY_CONFIG_DICT
    """Returns the globally cached parsed proxy dictionary."""
    return _PROXY_CONFIG_DICT


def _is_domain_allowed_for_proxy(url: str) -> bool:  # Uses global _PROXY_ALLOWED_DOMAINS_LIST
    """Checks if the URL's domain is allowed based on globally cached list."""
    if _PROXY_ALLOWED_DOMAINS_LIST is None:
        return True  # Allow all if list is None (wildcard)
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        if not domain:
            return False  # Cannot determine domain

        # Check domain and its superdomains against the allowed list
        domain_parts = domain.split(".")
        for i in range(len(domain_parts)):
            sub_domain_check = "." + ".".join(domain_parts[i:])
            if sub_domain_check in _PROXY_ALLOWED_DOMAINS_LIST:
                return True
        # Check exact domain match as well (if domain doesn't start with .)
        # The logic above already covers this because we ensure allowed domains start with '.'
        # e.g. if "example.com" is requested and ".example.com" is allowed, it matches.
        return False  # No allowed suffix matched
    except Exception as e:
        logger.warning(f"Error parsing URL '{url}' for proxy domain check: {e}")
        return False  # Deny on error


def _run_sync(coro):  # Keep as is
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, run in a new one
        return asyncio.run(coro)
    else:
        # Loop exists, run in threadsafe way if called from sync context
        future = asyncio.run_coroutine_threadsafe(coro, loop)  # noqa: F841
        # If needing the result synchronously (careful with deadlocks):
        # return future.result()
        return None  # Or return future if caller handles it


async def _try_close_browser():  # Uses global _browser
    """Attempt to close the browser gracefully via atexit."""
    global _browser
    browser_to_close = _browser  # Capture current browser instance
    if browser_to_close and browser_to_close.is_connected():
        logger.info("Attempting to close browser via atexit handler...")
        try:
            await browser_to_close.close()
            logger.info("Browser closed successfully via atexit.")
        except Exception as e:
            logger.error(f"Error closing browser during atexit: {e}")
        finally:
            # Only reset global _browser if it hasn't changed in the meantime
            if _browser == browser_to_close:
                _browser = None


async def get_browser_context(
    use_incognito: bool = False,
    context_args: Optional[Dict[str, Any]] = None,
) -> tuple[BrowserContext, Browser]:  # Uses MANY globals
    """Get or create a browser context using global config values."""
    global _pw, _browser, _ctx
    async with _playwright_lock:
        # 1. Ensure Playwright is started
        if not _pw:
            try:
                playwright_manager = async_playwright()
                _pw = await playwright_manager.start()
                logger.info("Playwright started.")
            except Exception as e:
                raise RuntimeError(f"Failed to start Playwright: {e}") from e

        # 2. Handle Headless Mode and VNC
        is_headless = _headless_mode_global
        if not is_headless:
            _start_vnc()  # Starts VNC if enabled and not already running

        # 3. Ensure Browser is launched and connected
        if not _browser or not _browser.is_connected():
            if _browser:  # Close previous instance if disconnected
                try:
                    await _browser.close()
                except Exception as close_err:
                    logger.warning(
                        f"Error closing previous disconnected browser instance: {close_err}"
                    )
            try:
                browser_args = [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--window-size=1280,1024",
                ]
                launched_browser = await _pw.chromium.launch(
                    headless=is_headless,
                    args=browser_args,
                )
                _browser = launched_browser
                logger.info(f"Browser launched (Headless: {is_headless}).")
                # Register atexit handler *after* successful launch
                atexit.register(lambda: _run_sync(_try_close_browser()))
            except PlaywrightException as e:
                raise RuntimeError(f"Failed to launch browser: {e}") from e

        # 4. Prepare Context Arguments
        default_args = {
            "viewport": {"width": 1280, "height": 1024},
            "locale": "en-US",
            "timezone_id": "UTC",
            "accept_downloads": True,
        }
        if context_args:
            default_args.update(context_args)

        # 5. Handle Incognito Context Request
        if use_incognito:
            try:
                incog_ctx = await _browser.new_context(**default_args)
                await _log("browser_incognito_context", args=default_args)
                # Apply proxy routing rules if necessary for incognito context
                proxy_cfg = _get_proxy_config()
                if proxy_cfg:
                    await _add_proxy_routing_rule(incog_ctx, proxy_cfg)
                return incog_ctx, _browser
            except PlaywrightException as e:
                raise ToolError(f"Failed to create incognito context: {e}") from e

        # 6. Handle Shared Context Request
        if not _ctx or not _ctx.browser:  # Check if shared context needs creation/recreation
            if _ctx:  # Close previous invalid context if any
                try:
                    await _ctx.close()
                except Exception as close_err:
                    logger.warning(f"Error closing previous invalid shared context: {close_err}")

            try:
                # Load state before creating context
                loaded_state = await _load_state()
                proxy_cfg = _get_proxy_config()

                final_ctx_args = default_args.copy()
                if loaded_state:
                    final_ctx_args["storage_state"] = loaded_state
                if proxy_cfg:
                    # Note: Using context.route for proxy filtering now,
                    # but setting proxy here is still needed for Playwright to use it.
                    final_ctx_args["proxy"] = proxy_cfg

                # Create the new shared context
                new_shared_ctx = await _browser.new_context(**final_ctx_args)
                _ctx = new_shared_ctx

                # Log context creation details (excluding potentially large state)
                log_args = {}
                for k, v in final_ctx_args.items():
                    if k != "storage_state":
                        log_args[k] = v
                await _log(
                    "browser_context_create",
                    headless=is_headless,
                    proxy=bool(proxy_cfg),
                    args=log_args,
                )

                # Apply proxy routing rules if needed
                if proxy_cfg:
                    await _add_proxy_routing_rule(_ctx, proxy_cfg)

                # Start maintenance loop for the *new* shared context
                asyncio.create_task(_context_maintenance_loop(_ctx))

            except PlaywrightException as e:
                raise RuntimeError(f"Failed to create shared context: {e}") from e
            except Exception as e:  # Catch errors during state load/save too
                raise RuntimeError(f"Failed during shared context creation/state load: {e}") from e

        # 7. Return the valid shared context and browser
        return _ctx, _browser


async def _add_proxy_routing_rule(
    context: BrowserContext, proxy_config: Dict[str, Any]
):  # Uses global _PROXY_ALLOWED_DOMAINS_LIST
    """Adds routing rule to enforce proxy domain restrictions if enabled."""
    # Check if domain restrictions are active
    if _PROXY_ALLOWED_DOMAINS_LIST is None:
        logger.debug("No proxy domain restrictions configured. Skipping routing rule.")
        return

    async def handle_route(route):
        request_url = route.request.url
        if not _is_domain_allowed_for_proxy(request_url):
            logger.warning(f"Proxy blocked for disallowed domain: {request_url}. Aborting request.")
            try:
                await route.abort("accessdenied")
            except PlaywrightException as e:
                # Log error but don't crash the handler
                logger.error(f"Error aborting route for {request_url}: {e}")
        else:
            # Domain is allowed, let the request proceed (through the proxy set on the context)
            try:
                await route.continue_()
            except PlaywrightException as e:
                # Log error but don't crash the handler
                logger.error(f"Error continuing route for {request_url}: {e}")

    try:
        # Route all network requests ('**/*')
        await context.route("**/*", handle_route)
        logger.info("Proxy domain restriction routing rule added.")
    except PlaywrightException as e:
        logger.error(f"Failed to add proxy routing rule: {e}")


def _start_vnc():  # Uses globals
    """Starts X11VNC if VNC enabled and password set."""
    global _vnc_proc
    # Check if already running or not enabled
    if _vnc_proc or not _vnc_enabled_global:
        return

    vnc_pass = _vnc_password_global
    if not vnc_pass:
        logger.debug("VNC start skipped: Password not set.")
        return

    display = os.getenv("DISPLAY", ":0")
    try:
        # Check if x11vnc command exists
        which_cmd = ["which", "x11vnc"]
        result = subprocess.run(which_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.warning("x11vnc command not found in PATH. Cannot start VNC server.")
            return

        # Prepare command arguments
        cmd = [
            "x11vnc",
            "-display",
            display,
            "-passwd",
            vnc_pass,  # Use the password directly
            "-forever",  # Keep running until explicitly killed
            "-localhost",  # Only listen on localhost
            "-quiet",  # Reduce log output
            "-noxdamage",  # Compatibility option
        ]

        # Use setsid to run in a new session, allowing clean termination
        if hasattr(os, "setsid"):
            preexec_fn = os.setsid
        else:
            preexec_fn = None  # Not available on Windows

        # Start the process
        vnc_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,  # Redirect stdout
            stderr=subprocess.DEVNULL,  # Redirect stderr
            preexec_fn=preexec_fn,  # Run in new session if possible
        )
        _vnc_proc = vnc_process
        logger.info(
            f"Password-protected VNC server started on display {display} (localhost only). PID: {_vnc_proc.pid}"
        )

        # Register cleanup function to run on exit
        atexit.register(_cleanup_vnc)

    except FileNotFoundError:
        # This shouldn't happen if `which` check passed, but belts and suspenders
        logger.warning("x11vnc command found by 'which' but Popen failed (FileNotFoundError).")
    except Exception as e:
        logger.error(f"Failed to start VNC server: {e}", exc_info=True)
        _vnc_proc = None  # Ensure proc is None if start failed


def _cleanup_vnc():  # Uses global _vnc_proc
    """Terminates the VNC server process."""
    global _vnc_proc
    proc = _vnc_proc  # Capture current process instance
    if proc and proc.poll() is None:  # Check if process exists and is running
        logger.info(f"Terminating VNC server process (PID: {proc.pid})...")
        try:
            # Try to terminate the whole process group first (more reliable)
            if hasattr(os, "getpgid") and hasattr(os, "killpg"):
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    logger.debug(f"Sent SIGTERM to process group {pgid}.")
                except ProcessLookupError:
                    # Process group might already be gone
                    logger.debug("VNC process group not found, trying direct SIGTERM.")
                    proc.terminate()
                except Exception as pg_err:
                    logger.warning(
                        f"Error sending SIGTERM to process group, trying direct SIGTERM: {pg_err}"
                    )
                    proc.terminate()  # Fallback to terminating just the process
            else:
                # Fallback if killpg/getpgid not available
                proc.terminate()
                logger.debug("Sent SIGTERM directly to VNC process.")

            # Wait for termination with timeout
            proc.wait(timeout=5)
            logger.info("VNC server process terminated gracefully.")
        except subprocess.TimeoutExpired:
            logger.warning("VNC server did not terminate after SIGTERM. Sending SIGKILL.")
            # Force kill if SIGTERM failed
            if hasattr(os, "getpgid") and hasattr(os, "killpg"):
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                    logger.debug(f"Sent SIGKILL to process group {pgid}.")
                except ProcessLookupError:
                    logger.debug("VNC process group not found for SIGKILL, trying direct SIGKILL.")
                    proc.kill()  # Fallback to killing just the process
                except Exception as pg_kill_err:
                    logger.warning(
                        f"Error sending SIGKILL to process group, trying direct SIGKILL: {pg_kill_err}"
                    )
                    proc.kill()  # Fallback
            else:
                proc.kill()  # Fallback if killpg not available
                logger.debug("Sent SIGKILL directly to VNC process.")
            # Wait briefly after SIGKILL
            try:
                proc.wait(timeout=2)
            except Exception:
                # Ignore errors during wait after SIGKILL
                pass
        except ProcessLookupError:
            # Process was already gone before we could signal it
            logger.info("VNC process already terminated before cleanup.")
        except Exception as e:
            logger.error(f"Error during VNC cleanup: {e}")
        finally:
            # Ensure global state reflects VNC is stopped
            if _vnc_proc == proc:  # Avoid race condition if started again quickly
                _vnc_proc = None


async def _load_state() -> dict[str, Any] | None:  # Uses global _STATE_FILE, _get_pool, _dec
    """Loads browser state asynchronously. Decryption runs in executor if needed."""
    if _STATE_FILE is None or not _STATE_FILE.exists():
        logger.info("Browser state file path not set or file not found. No state loaded.")
        return None

    loop = asyncio.get_running_loop()
    pool = _get_pool()
    try:
        # Read the potentially encrypted file content
        async with aiofiles.open(_STATE_FILE, "rb") as f:
            file_data = await f.read()

        # Decrypt if necessary (runs sync _dec in thread pool)
        # _dec handles the check for whether encryption is active or not
        try:
            decrypted_data = await loop.run_in_executor(pool, _dec, file_data)
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e):
                logger.warning(
                    "Thread pool is shutdown. Creating a temporary pool for state loading."
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as temp_pool:
                    decrypted_data = await loop.run_in_executor(temp_pool, _dec, file_data)
            else:
                raise

        if decrypted_data is None:
            # _dec logs specific reasons (invalid format, decryption failure, etc.)
            logger.warning("Failed to load or decrypt state data. State file might be invalid.")
            # Optionally remove the invalid file here if desired
            # try: _STATE_FILE.unlink(); except Exception: pass
            return None

        # Parse the decrypted JSON data
        state_dict = json.loads(decrypted_data)
        logger.info(f"Browser state loaded successfully from {_STATE_FILE}.")
        return state_dict

    except FileNotFoundError:
        # This case should be caught by the initial check, but handle defensively
        logger.info(f"Browser state file {_STATE_FILE} not found during read.")
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse browser state JSON from {_STATE_FILE}: {e}. Removing corrupt file."
        )
        if _STATE_FILE:
            try:
                _STATE_FILE.unlink()
            except Exception as unlink_e:
                logger.error(f"Failed to remove corrupt state file {_STATE_FILE}: {unlink_e}")
        return None
    except RuntimeError as e:  # Catch auth errors from _dec (InvalidTag)
        logger.error(
            f"Failed to authenticate/load browser state from {_STATE_FILE}: {e}", exc_info=True
        )
        if _STATE_FILE:
            try:
                _STATE_FILE.unlink()
            except Exception as unlink_e:
                logger.error(
                    f"Failed to remove unauthenticated state file {_STATE_FILE}: {unlink_e}"
                )
        return None
    except Exception as e:
        logger.error(f"Failed to load browser state from {_STATE_FILE}: {e}", exc_info=True)
        # Optionally remove the problematic file
        if _STATE_FILE:
            try:
                _STATE_FILE.unlink()
            except Exception as unlink_e:
                logger.error(f"Failed to remove problematic state file {_STATE_FILE}: {unlink_e}")
        return None


async def _save_state(
    ctx: BrowserContext,
):  # Uses global _get_pool, _enc, _STATE_FILE, _key, _playwright_lock
    """Saves browser state asynchronously using FileSystemTool's write_file."""
    if _STATE_FILE is None:
        logger.warning("Skipping save state: State file path (_STATE_FILE) not initialized.")
        return

    # Acquire lock *before* checking context validity to prevent race with shutdown
    async with _playwright_lock:
        # Re-check context validity *after* acquiring the lock
        if not ctx or not ctx.browser or not ctx.browser.is_connected():
            logger.debug(
                "Skipping save state: Context or browser became invalid/disconnected before save."
            )
            return

        loop = asyncio.get_running_loop()
        pool = _get_pool()
        validated_fpath = str(_STATE_FILE)

        try:
            # 1. Get the current storage state from Playwright (NOW protected by lock)
            state = await ctx.storage_state()

            # 2. Serialize state to JSON bytes
            state_json = json.dumps(state)
            state_bytes = state_json.encode("utf-8")

            # 3. Encrypt the state bytes if key is configured (runs sync _enc in thread pool)
            try:
                data_to_write = await loop.run_in_executor(pool, _enc, state_bytes)
            except RuntimeError as e:
                if "cannot schedule new futures after shutdown" in str(e):
                    logger.warning(
                        "Thread pool is shutdown. Creating a temporary pool for state encryption."
                    )
                    # Fallback pool creation remains useful
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as temp_pool:
                        data_to_write = await loop.run_in_executor(temp_pool, _enc, state_bytes)
                else:
                    raise  # Re-raise other RuntimeErrors

            # 4. Write the (potentially encrypted) bytes using the standalone filesystem tool
            logger.debug(f"Attempting to save state to: {validated_fpath} using filesystem tool.")
            write_result = await write_file(
                path=validated_fpath, content=data_to_write
            )  # Pass bytes

            # 5. Check result from filesystem tool
            if not isinstance(write_result, dict) or not write_result.get("success"):
                error_detail = "Invalid response"
                if isinstance(write_result, dict):
                    error_detail = write_result.get("error", "Unknown")
                logger.error(
                    f"Failed to save browser state using filesystem tool. Reason: {error_detail}"
                )
                # Log but don't raise ToolError here directly, let the maintenance loop handle logging it
                return  # Exit if write failed

            # 6. Log success
            actual_path = write_result.get("path", validated_fpath)
            logger.debug(f"Successfully saved state to file: {actual_path}")  # Changed log level

        except PlaywrightException as e:
            # Catch errors specifically from ctx.storage_state() if the context closed unexpectedly
            # even with the lock (less likely now, but possible)
            logger.warning(f"Playwright error during save state (context likely closed): {e}")
            # Don't raise, let the loop continue/exit gracefully
        except ToolError as e:
            # Pass ToolError through (e.g., from write_file) - should be logged by caller
            logger.error(f"ToolError during save state: {e}")
            # Don't re-raise here, maintenance loop will log the error
        except Exception as e:
            logger.error(
                f"Unexpected error saving browser state (path: {validated_fpath}): {e}",
                exc_info=True,
            )
            # Don't raise ToolError here, let the maintenance loop log the failure


@asynccontextmanager
async def _tab_context(ctx: BrowserContext):  # Uses global _log
    """Async context manager for creating and cleaning up a Page."""
    page = None
    context_id = id(ctx)  # Get ID for logging before potential errors
    try:
        page = await ctx.new_page()
        await _log("page_open", context_id=context_id)
        yield page
    except PlaywrightException as e:
        # Log the error before raising
        await _log("page_error", context_id=context_id, action="create", error=str(e))
        raise ToolError(f"Failed to create browser page: {e}") from e
    finally:
        if page and not page.is_closed():
            try:
                await page.close()
                await _log("page_close", context_id=context_id)
            except PlaywrightException as e:
                # Log error during close, but don't prevent cleanup completion
                logger.warning(f"Error closing page for context {context_id}: {e}")
                await _log("page_error", context_id=context_id, action="close", error=str(e))


async def _context_maintenance_loop(ctx: BrowserContext):  # Uses global _save_state
    """Periodically saves state for the shared context."""
    save_interval_seconds = 15 * 60  # Save every 15 minutes
    context_id = id(ctx)
    logger.info(f"Starting context maintenance loop for shared context {context_id}.")

    while True:
        try:
            # Check if context is still valid *before* sleeping
            # Use is_connected() for a more robust check
            if not ctx or not ctx.browser or not ctx.browser.is_connected():
                logger.info(
                    f"Shared context {context_id} seems invalid or disconnected. Stopping maintenance loop."
                )
                break

            # Wait for the specified interval
            await asyncio.sleep(save_interval_seconds)

            # Re-check context validity *after* sleeping, before saving
            if not ctx or not ctx.browser or not ctx.browser.is_connected():
                logger.info(
                    f"Shared context {context_id} became invalid/disconnected during sleep. Stopping maintenance loop."
                )
                break

            # Save the state (which now handles its own locking and errors more gracefully)
            await _save_state(ctx)

        except asyncio.CancelledError:
            logger.info(f"Context maintenance loop for {context_id} cancelled.")
            break  # Exit loop cleanly on cancellation
        except Exception as e:
            # Log unexpected errors in the loop itself (e.g., during the sleep?)
            logger.error(
                f"Unexpected error in context maintenance loop for {context_id}: {e}", exc_info=True
            )
            # Wait a bit longer before retrying after an unexpected loop error
            await asyncio.sleep(60)


# --- Standalone Shutdown Function ---
# --- Replace the existing shutdown function in smart_browser.py ---
async def shutdown():  # Uses MANY globals
    """Gracefully shut down Playwright, browser, context, VNC, and thread pool."""
    global _pw, _browser, _ctx, _vnc_proc, _thread_pool, _locator_cache_cleanup_task_handle, _inactivity_monitor_task_handle, _is_initialized, _shutdown_initiated  # Added missing global reference

    # Use lock to prevent concurrent shutdown calls
    # Check _shutdown_initiated flag inside lock for atomicity
    async with _shutdown_lock:
        if _shutdown_initiated:
            logger.debug("Shutdown already initiated or in progress. Skipping.")
            return
        if not _is_initialized:
            logger.info("Shutdown called but Smart Browser was not initialized. Skipping.")
            return
        # Mark shutdown as initiated *inside* the lock
        _shutdown_initiated = True

    logger.info("Initiating graceful shutdown for Smart Browser...")

    # Set a global shutdown timeout to prevent hanging
    shutdown_timeout = 10.0  # 10 seconds to complete shutdown or we'll force through
    shutdown_start_time = time.monotonic()

    # Function to check if shutdown is taking too long
    def is_shutdown_timeout():
        return (time.monotonic() - shutdown_start_time) > shutdown_timeout

    # 1. Cancel Background Tasks First
    tasks_to_cancel = [
        (_locator_cache_cleanup_task_handle, "Locator Cache Cleanup Task"),
        (_inactivity_monitor_task_handle, "Inactivity Monitor Task"),
    ]
    for task_handle, task_name in tasks_to_cancel:
        if task_handle and not task_handle.done():
            logger.info(f"Cancelling {task_name}...")
            task_handle.cancel()
            try:
                # Wait briefly for cancellation to complete
                await asyncio.wait_for(task_handle, timeout=2.0)
                logger.info(f"{task_name} cancellation confirmed.")  # Changed log level
            except asyncio.CancelledError:
                logger.info(f"{task_name} cancellation confirmed.")  # Expected outcome
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for {task_name} cancellation.")
            except Exception as e:
                err_type = type(e).__name__
                logger.warning(f"Error waiting for {task_name} cancellation: {err_type}")

    # Reset task handles
    _locator_cache_cleanup_task_handle = None
    _inactivity_monitor_task_handle = None

    # 2. Cancel any active tab pool operations
    await tab_pool.cancel_all()  # Handles incognito contexts

    # 3. Close Playwright resources (under lock to prevent concurrent access)
    async with _playwright_lock:
        # Close Shared Context (save state first, if possible)
        ctx_to_close = _ctx
        _ctx = None  # Immediately unset global reference

        # Skip state saving if we're already at the timeout
        if is_shutdown_timeout():
            logger.warning("Skipping state saving due to shutdown timeout")
        # --- Robust Check and Save State ---
        elif ctx_to_close and ctx_to_close.browser and ctx_to_close.browser.is_connected():
            logger.info("Attempting to save state for shared browser context...")
            try:
                # Add timeout for state saving
                await asyncio.wait_for(_save_state(ctx_to_close), timeout=3.0)
                logger.info(
                    "State saving attempted for shared context."
                )  # Log attempt, success logged within _save_state
            except asyncio.TimeoutError:
                logger.warning("State saving timed out after 3 seconds")
            except Exception as e:
                # Catch any unexpected error from _save_state itself (should be rare now)
                logger.error(
                    f"Unexpected error during final state save attempt: {e}", exc_info=True
                )
        elif ctx_to_close:
            logger.warning(
                "Skipping final state save: Shared context or browser already closed/disconnected."
            )
        else:
            logger.debug("Skipping final state save: No shared context exists.")
        # --- End Robust Check and Save State ---

        # Close the context object itself
        if ctx_to_close and not is_shutdown_timeout():
            logger.info("Closing shared browser context object...")
            try:
                # Add timeout for context closing
                await asyncio.wait_for(ctx_to_close.close(), timeout=3.0)
                await _log("browser_context_close_shared")
                logger.info("Shared browser context closed.")
            except asyncio.TimeoutError:
                logger.warning("Browser context close timed out after 3 seconds")
            except Exception as e:
                # Log error but continue shutdown
                logger.error(
                    f"Error closing shared context object: {e}", exc_info=False
                )  # Keep log less verbose
        elif ctx_to_close:
            logger.warning("Skipping browser context close due to shutdown timeout")

        # Close Browser
        browser_to_close = _browser
        _browser = None  # Immediately unset global reference
        if browser_to_close and browser_to_close.is_connected() and not is_shutdown_timeout():
            logger.info("Closing browser instance...")
            try:
                # Add timeout for browser closing - shorter timeout to avoid hanging
                await asyncio.wait_for(browser_to_close.close(), timeout=3.0)
                await _log("browser_close")
                logger.info("Browser instance closed.")
            except asyncio.TimeoutError:
                logger.warning("Browser close timed out after 3 seconds")
            except Exception as e:
                logger.error(f"Error closing browser: {e}", exc_info=False)  # Keep log less verbose
        elif browser_to_close and browser_to_close.is_connected():
            logger.warning("Skipping browser close due to shutdown timeout")

        # Stop Playwright
        pw_to_stop = _pw
        _pw = None  # Immediately unset global reference
        if pw_to_stop and not is_shutdown_timeout():
            logger.info("Stopping Playwright...")
            try:
                # Add timeout for playwright stop - shorter timeout
                await asyncio.wait_for(pw_to_stop.stop(), timeout=2.0)
                logger.info("Playwright stopped.")
            except asyncio.TimeoutError:
                logger.warning("Playwright stop timed out after 2 seconds")
            except Exception as e:
                logger.error(
                    f"Error stopping Playwright: {e}", exc_info=False
                )  # Keep log less verbose
        elif pw_to_stop:
            logger.warning("Skipping Playwright stop due to shutdown timeout")

    # 4. Cleanup Synchronous Resources - always do this regardless of timeout
    _cleanup_vnc()
    _close_db_connection()

    # 5. Log completion and reset flags
    await _log("browser_shutdown_complete")
    if is_shutdown_timeout():
        logger.warning(
            "Smart Browser shutdown reached timeout limit - some resources may not be fully released"
        )
    else:
        logger.info("Smart Browser graceful shutdown complete.")
    _is_initialized = False

    # 6. Shutdown Thread Pool (MOVED TO THE VERY END)
    logger.info("Shutting down thread pool...")
    pool_to_shutdown = _get_pool()
    # Don't wait for tasks if we're already at timeout
    if is_shutdown_timeout():
        try:
            pool_to_shutdown.shutdown(wait=False)
            logger.info("Thread pool shutdown initiated without waiting")
        except Exception as e:
            logger.error(f"Error during thread pool non-waiting shutdown: {e}")
    else:
        # Give the pool a short timeout to avoid hanging
        try:
            time_left = max(0, shutdown_timeout - (time.monotonic() - shutdown_start_time))
            # Use the minimum of 3 seconds or remaining time
            wait_time = min(3.0, time_left)

            # Create a separate thread to shut down the pool with a timeout
            import threading

            shutdown_complete = threading.Event()

            def shutdown_pool_with_timeout():
                try:
                    pool_to_shutdown.shutdown(wait=True)
                    shutdown_complete.set()
                except Exception as e:
                    logger.error(f"Error in thread pool shutdown thread: {e}")

            # Start the shutdown in a separate thread
            thread = threading.Thread(target=shutdown_pool_with_timeout)
            thread.daemon = True
            thread.start()

            # Wait for completion or timeout
            if shutdown_complete.wait(wait_time):
                logger.info("Thread pool shut down successfully.")
            else:
                logger.warning(f"Thread pool shutdown timed out after {wait_time} seconds")
                # Try non-waiting shutdown as fallback
                try:
                    pool_to_shutdown.shutdown(wait=False)
                except Exception:
                    pass  # Already logged above
        except Exception as e:
            logger.error(f"Error setting up thread pool shutdown: {e}")
            # Fallback to non-waiting shutdown
            pool_to_shutdown.shutdown(wait=False)


async def _initiate_shutdown():  # Uses global _shutdown_initiated, _shutdown_lock
    """Ensures shutdown runs only once."""
    global _shutdown_initiated
    async with _shutdown_lock:
        if not _shutdown_initiated:
            _shutdown_initiated = True
            await shutdown()
        else:
            logger.debug("Shutdown already initiated. Ignoring duplicate request.")


# --- Signal Handling (Keep top-level) ---
def _signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    signal_name = signal.Signals(sig).name
    logger.info(f"Received signal {signal_name} ({sig}). Initiating Smart Browser shutdown...")
    try:
        # Try to get the running event loop
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # Schedule shutdown in the running loop, don't block signal handler
            asyncio.create_task(_initiate_shutdown())
        else:
            # No running loop, attempt synchronous run (best effort)
            logger.warning(
                "No running event loop found in signal handler. Attempting sync shutdown."
            )
            try:
                asyncio.run(_initiate_shutdown())
            except RuntimeError as e:
                logger.error(f"Could not run async shutdown synchronously from signal handler: {e}")
    except RuntimeError as e:
        # Error getting the loop itself
        logger.error(
            f"Error getting event loop in signal handler: {e}. Shutdown might be incomplete."
        )


# Register signal handlers in a try-except block
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)  # Handle Ctrl+C too
except ValueError:
    # This can happen if not running in the main thread
    logger.warning(
        "Could not register signal handlers (not running in main thread?). Graceful shutdown on SIGTERM/SIGINT might not work."
    )


# --- Tab Pool (Keep global instance) ---
class TabPool:  # Keep class definition
    """Runs async callables needing a Page in parallel, bounded by global config."""

    def __init__(self, max_tabs: int | None = None):
        if max_tabs is not None:
            self.max_tabs = max_tabs
        else:
            self.max_tabs = _sb_max_tabs_global

        if self.max_tabs <= 0:
            logger.warning(f"TabPool max_tabs configured to {self.max_tabs}. Setting to 1.")
            self.max_tabs = 1
        self.sem = asyncio.Semaphore(self.max_tabs)
        self._active_contexts: Set[BrowserContext] = set()  # Store contexts being used
        self._context_lock = asyncio.Lock()  # Protect access to _active_contexts
        logger.info(f"TabPool initialized with max_tabs={self.max_tabs}")

    async def _run(self, fn: Callable[[Page], Awaitable[Any]]) -> Any:
        """Internal method to run a single function within a managed tab."""
        timeout_seconds = _sb_tab_timeout_global
        incognito_ctx: Optional[BrowserContext] = None
        task = asyncio.current_task()
        task_id = id(task)
        func_name = getattr(fn, "__name__", "anon_tab_fn")

        try:
            # Acquire semaphore before creating context/page
            async with self.sem:
                # Create a new incognito context for isolation
                # Pass None for context_args to use defaults
                incognito_ctx, _ = await get_browser_context(use_incognito=True, context_args=None)

                # Add context to active set under lock
                async with self._context_lock:
                    self._active_contexts.add(incognito_ctx)

                # Use the async context manager for the page
                async with _tab_context(incognito_ctx) as page:
                    # Run the provided function with timeout
                    result = await asyncio.wait_for(fn(page), timeout=timeout_seconds)
                    return result  # Return the successful result

        except asyncio.TimeoutError:
            await _log("tab_timeout", function=func_name, timeout=timeout_seconds, task_id=task_id)
            # Return error structure on timeout
            return {
                "error": f"Tab operation '{func_name}' timed out after {timeout_seconds}s",
                "success": False,
            }
        except asyncio.CancelledError:
            # Log cancellation and re-raise
            await _log("tab_cancelled", function=func_name, task_id=task_id)
            raise  # Important to propagate cancellation
        except Exception as e:
            # Log any other exceptions during execution
            await _log(
                "tab_error", function=func_name, error=str(e), task_id=task_id, exc_info=True
            )
            # Return error structure
            return {"error": f"Tab operation '{func_name}' failed: {e}", "success": False}
        finally:
            # Cleanup: Remove context from active set and close it
            if incognito_ctx:
                incog_ctx_id = id(incognito_ctx)  # Get ID before potential close error
                async with self._context_lock:
                    self._active_contexts.discard(incognito_ctx)
                try:
                    await incognito_ctx.close()
                    logger.debug(f"Incognito context {incog_ctx_id} closed for task {task_id}.")
                except PlaywrightException as close_err:
                    # Log error but don't let it prevent other cleanup
                    logger.warning(
                        f"Error closing incognito context {incog_ctx_id} for task {task_id}: {close_err}"
                    )

    async def map(self, fns: Sequence[Callable[[Page], Awaitable[Any]]]) -> List[Any]:
        """Runs multiple functions concurrently using the tab pool."""
        if not fns:
            return []

        # Create tasks for each function using the internal _run method
        tasks = []
        for fn in fns:
            task = asyncio.create_task(self._run(fn))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling potential exceptions returned by gather
        processed_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                # Log the exception if a task failed unexpectedly
                func_name = getattr(fns[i], "__name__", f"fn_{i}")
                logger.error(f"Error in TabPool.map for '{func_name}': {res}", exc_info=res)
                # Append an error dictionary for failed tasks
                processed_results.append(
                    {"error": f"Task '{func_name}' failed with exception: {res}", "success": False}
                )
            else:
                # Append the result directly (which might be an error dict from _run)
                processed_results.append(res)
        return processed_results

    async def cancel_all(self):
        """Attempts to close all currently active incognito contexts managed by the pool."""
        contexts_to_close: List[BrowserContext] = []
        # Safely get the list of active contexts and clear the set under lock
        async with self._context_lock:
            contexts_to_close = list(self._active_contexts)
            self._active_contexts.clear()

        if not contexts_to_close:
            logger.debug("TabPool cancel_all: No active contexts to close.")
            return

        logger.info(
            f"TabPool cancel_all: Attempting to close {len(contexts_to_close)} active incognito contexts."
        )
        # Create closing tasks for each context
        close_tasks = []
        for ctx in contexts_to_close:
            task = asyncio.create_task(ctx.close())
            close_tasks.append(task)

        # Wait for all close tasks to complete, collecting results/exceptions
        results = await asyncio.gather(*close_tasks, return_exceptions=True)

        # Count and log errors during closure
        errors = 0
        for res in results:
            if isinstance(res, Exception):
                errors += 1
        if errors:
            logger.warning(
                f"TabPool cancel_all: Encountered {errors} errors while closing contexts."
            )
        else:
            logger.info(
                f"TabPool cancel_all: Successfully closed {len(contexts_to_close)} contexts."
            )


# Global instance of the TabPool
tab_pool = TabPool()


# --- Human Jitter ---
def _risk_factor(url: str) -> float:  # Uses global _high_risk_domains_set_global
    """Calculates risk factor based on URL's domain (higher for known tricky domains)."""
    if not url:
        return 1.0  # Default risk if URL is empty
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        # Remove common www prefix
        if domain.startswith("www."):
            domain = domain[4:]

        if not domain:
            return 1.0  # Default risk if domain cannot be parsed

        # Check if domain or its parent domains are in the high-risk set
        domain_parts = domain.split(".")
        for i in range(len(domain_parts)):
            # Construct subdomain like ".example.com", ".com"
            sub_domain_check = "." + ".".join(domain_parts[i:])
            if sub_domain_check in _high_risk_domains_set_global:
                return 2.0  # High risk factor

        # No match found in high-risk set
        return 1.0  # Standard risk factor
    except Exception as e:
        logger.warning(f"Error calculating risk factor for URL '{url}': {e}")
        return 1.0  # Default risk on error


async def _pause(
    page: Page, base_ms_range: tuple[int, int] = (150, 500)
):  # Uses global _risk_factor
    """Introduce a short, randomized pause, adjusted by URL risk factor and page complexity."""
    if not page or page.is_closed():
        return  # Do nothing if page is invalid

    risk = _risk_factor(page.url)
    min_ms, max_ms = base_ms_range
    base_delay_ms = random.uniform(min_ms, max_ms)
    adjusted_delay_ms = base_delay_ms * risk

    try:
        # Estimate page complexity based on number of interactive elements
        # Use a simpler selector for broad compatibility
        selector = "a, button, input, select, textarea, [role=button], [role=link], [onclick]"
        js_expr = f"() => document.querySelectorAll('{selector}').length"
        element_count = await page.evaluate(js_expr)

        # If count is 0, might be an error or very simple page, assume moderate complexity
        if element_count == 0:
            element_count = max(element_count, 100)  # Avoid division by zero/tiny factors

        # Skip pauses for low-risk, very simple pages
        is_low_risk = risk == 1.0
        is_simple_page = element_count < 50
        if is_low_risk and is_simple_page:
            return  # No pause needed

        # Increase delay slightly based on complexity, capping the factor
        complexity_factor_base = 1.0 + (element_count / 500.0)
        complexity_factor = min(complexity_factor_base, 1.5)  # Cap factor at 1.5
        adjusted_delay_ms *= complexity_factor

    except PlaywrightException as e:
        # Ignore errors during element count evaluation, proceed with risk-adjusted delay
        logger.debug(f"Could not evaluate element count for pause adjustment: {e}")
        pass

    # Cap the final delay to avoid excessive pauses
    final_delay_ms = min(adjusted_delay_ms, 3000)  # Max 3 seconds pause

    # Convert ms to seconds and sleep
    final_delay_sec = final_delay_ms / 1000.0
    await asyncio.sleep(final_delay_sec)


# --- Enhanced Locator Helpers (Depend on globals, use Filesystem tools) ---
_READ_JS_WRAPPER = textwrap.dedent("""
    (html) => {
        // Ensure Readability library is loaded in the window scope
        const R = window.__sbReadability;
        if (!R || !html) {
            console.warn('Readability object or HTML missing.');
            return ""; // Cannot proceed without library or content
        }
        try {
            // Create a DOM from the HTML string
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, "text/html");

            // Basic validation of the parsed document
            if (!doc || !doc.body || doc.body.innerHTML.trim() === '') {
                 console.warn('Parsed document is invalid or empty.');
                 return "";
            }

            // Use Readability to parse the article content
            const article = new R.Readability(doc).parse();

            // Return the text content if parsing was successful
            return article ? article.textContent : "";

        } catch (e) {
            // Log errors during parsing
            console.warn('Readability parsing failed:', e);
            return ""; // Return empty string on error
        }
    }
""")


async def _ensure_readability(page: Page) -> None:  # Uses global _READ_JS_CACHE
    """Ensures Readability.js is injected, using STANDALONE filesystem tools."""
    # Check if already injected
    is_injected_js = "() => window.__sbReadability !== undefined"
    already_injected = await page.evaluate(is_injected_js)
    if already_injected:
        logger.debug("Readability.js already injected.")
        return

    if _READ_JS_CACHE is None:
        logger.warning("Readability cache path (_READ_JS_CACHE) not set. Cannot cache script.")
        # Proceed to fetch, but won't cache
    else:
        cache_file_path = str(_READ_JS_CACHE)

    src: Optional[str] = None

    # Try reading from cache if path is set
    if _READ_JS_CACHE:
        try:
            logger.debug(f"Attempting to load Readability.js from cache: {cache_file_path}")
            read_result = await read_file(path=cache_file_path)
            if isinstance(read_result, dict) and not read_result.get("success"):
                error_msg = read_result.get("error", "Unknown read error")
                error_code = read_result.get("error_code", "")
                logger.warning(
                    f"Failed to read Readability.js cache {cache_file_path}: {error_msg} (Code: {error_code}). Full response: {read_result}. Will attempt fetch."  # Log full dict
                )
            if isinstance(read_result, dict) and read_result.get("success"):
                content_list = read_result.get("content", [])
                if isinstance(content_list, list) and content_list:
                    # Assuming single file content for this cache
                    file_content = content_list[0]
                    if isinstance(file_content, dict):
                        src = file_content.get("text")
                        if src:
                            logger.debug(
                                f"Readability.js loaded successfully from cache: {cache_file_path}"
                            )
                        else:
                            logger.warning(
                                f"Readability cache file {cache_file_path} content missing 'text'."
                            )
                    else:
                        logger.warning(
                            f"Readability cache file {cache_file_path} content format unexpected."
                        )
                else:
                    logger.info(
                        f"Readability cache file {cache_file_path} exists but is empty or has no content list."
                    )
            # Handle specific file not found error (or other read errors) from standalone tool
            elif isinstance(read_result, dict) and not read_result.get("success"):
                error_msg = read_result.get("error", "Unknown read error")
                error_code = read_result.get("error_code", "")
                if "does not exist" in error_msg.lower() or "PATH_NOT_FOUND" in error_code:
                    logger.info(
                        f"Readability.js cache file not found ({cache_file_path}). Will attempt fetch."
                    )
                else:
                    logger.warning(
                        f"Failed to read Readability.js cache {cache_file_path}: {error_msg}. Will attempt fetch."
                    )
            else:  # Unexpected response format
                logger.warning(
                    f"Unexpected response from read_file for {cache_file_path}. Will attempt fetch."
                )

        except ToolError as e:  # Catch explicit ToolError if raised by read_file internally
            if "does not exist" in str(e).lower() or "PATH_NOT_FOUND" in getattr(
                e, "error_code", ""
            ):
                logger.info(
                    f"Readability.js cache file not found ({cache_file_path}). Will attempt fetch."
                )
            else:
                logger.warning(
                    f"ToolError reading Readability.js cache {cache_file_path}: {e}. Will attempt fetch."
                )
        except Exception as e:
            # Catch any other unexpected errors during cache read
            logger.warning(
                f"Unexpected error reading Readability.js cache {cache_file_path}: {e}. Will attempt fetch.",
                exc_info=True,
            )

    # Fetch from CDN if not loaded from cache
    if src is None:
        logger.info("Fetching Readability.js from CDN...")
        try:
            async with httpx.AsyncClient() as client:
                # Use a reliable CDN link
                cdn_url = "https://cdnjs.cloudflare.com/ajax/libs/readability/0.5.0/Readability.js"
                response = await client.get(cdn_url, timeout=15.0)
                response.raise_for_status()  # Raise exception for bad status codes
                fetched_src = response.text
                fetched_size = len(fetched_src)
                await _log("readability_js_fetch", url=cdn_url, size=fetched_size)

            if fetched_src:
                # Try writing to cache if path is set
                if _READ_JS_CACHE:
                    try:
                        logger.debug(
                            f"Attempting to save fetched Readability.js to cache: {cache_file_path}"
                        )
                        # Use STANDALONE write_file tool
                        write_res = await write_file(
                            path=cache_file_path, content=fetched_src
                        )  # Pass string content

                        if isinstance(write_res, dict) and write_res.get("success"):
                            logger.info(f"Saved fetched Readability.js to cache: {cache_file_path}")
                        else:
                            error_msg = (
                                write_res.get("error", "Unknown write error")
                                if isinstance(write_res, dict)
                                else "Invalid write_file response"
                            )
                            logger.warning(
                                f"Failed to write Readability.js cache ({cache_file_path}): {error_msg}"
                            )
                    except Exception as write_err:
                        # Log error but proceed with injection using fetched source
                        logger.warning(
                            f"Error writing Readability.js cache ({cache_file_path}): {write_err}"
                        )

                # Use the fetched source for injection
                src = fetched_src
            else:
                logger.warning("Fetched empty content for Readability.js from CDN.")

        except httpx.HTTPStatusError as fetch_err:
            logger.error(
                f"HTTP error fetching Readability.js from {fetch_err.request.url}: {fetch_err.response.status_code}"
            )
        except httpx.RequestError as fetch_err:
            logger.error(f"Network error fetching Readability.js: {fetch_err}")
        except Exception as fetch_err:
            logger.error(f"Failed to fetch/cache Readability.js: {fetch_err}", exc_info=True)

    # Inject the script if source code was successfully obtained (from cache or fetch)
    if src:
        # Wrap the source code to assign the Readability class to a window property
        wrapped_src = f"window.__sbReadability = (() => {{ {src}; return Readability; }})();"
        try:
            await page.add_script_tag(content=wrapped_src)
            logger.debug("Readability.js injected successfully.")
        except PlaywrightException as e:
            # Handle potential injection errors (e.g., Content Security Policy)
            err_str = str(e)
            if "Content Security Policy" in err_str:
                page_url = page.url  # Get URL for context
                logger.warning(
                    f"Could not inject Readability.js due to Content Security Policy on {page_url}."
                )
            else:
                logger.error(f"Failed to inject Readability.js script tag: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error injecting Readability.js: {e}", exc_info=True)
    else:
        # Log if source couldn't be obtained
        logger.warning("Failed to load or fetch Readability.js source. Proceeding without it.")


async def _dom_fingerprint(page: Page) -> str:  # Uses global _dom_fp_limit_global
    """Calculates a fingerprint of the page's visible text content."""
    try:
        # Evaluate JS to get the initial part of the body's innerText
        js_expr = f"() => document.body.innerText.slice(0, {_dom_fp_limit_global})"
        txt_content = await page.main_frame.evaluate(js_expr)

        # Ensure text is not None and strip whitespace
        cleaned_txt = (txt_content or "").strip()

        # Encode the text to bytes (ignoring errors) and hash it
        txt_bytes = cleaned_txt.encode("utf-8", "ignore")
        hasher = hashlib.sha256(txt_bytes)
        fingerprint = hasher.hexdigest()
        return fingerprint

    except PlaywrightException as e:
        # Log error if evaluation fails, return hash of empty string
        logger.warning(f"Could not get text for DOM fingerprint: {e}")
        empty_hash = hashlib.sha256(b"").hexdigest()
        return empty_hash
    except Exception as e:
        # Catch unexpected errors
        logger.error(f"Unexpected error calculating DOM fingerprint: {e}", exc_info=True)
        empty_hash = hashlib.sha256(b"").hexdigest()
        return empty_hash


def _shadow_deep_js() -> str:  # Uses globals _max_widgets_global, _area_min_global
    """JS function string to find elements, traversing shadow DOM."""
    # This JS function is complex but self-contained. Keep as multi-line f-string.
    # Relies on _max_widgets_global and _area_min_global from Python scope.
    return f"""
    (prefix) => {{
        const MAX_ELEMENTS = {_max_widgets_global};
        const MIN_ELEMENT_AREA = {_area_min_global};

        // --- Helper Functions ---
        const isVisible = (el) => {{
            if (!el || typeof el.getBoundingClientRect !== 'function') {{ return false; }}
            try {{
                // Check CSS visibility properties
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0 || el.hidden) {{
                    return false;
                }}
                // Check if it has an offset parent (not detached or position:fixed parent hidden)
                if (!el.offsetParent && style.position !== 'fixed') {{
                     return false;
                }}

                // Check bounding box dimensions and position
                const rect = el.getBoundingClientRect();
                const hasPositiveSize = rect.width > 1 && rect.height > 1; // Needs some dimensions
                const hasSufficientArea = (rect.width * rect.height) >= MIN_ELEMENT_AREA;

                // Check if it's within the viewport bounds (partially is sufficient)
                const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
                const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
                const isInViewportVertically = rect.bottom > 0 && rect.top < viewportHeight;
                const isInViewportHorizontally = rect.right > 0 && rect.left < viewportWidth;
                const isOnscreen = isInViewportVertically && isInViewportHorizontally;

                // Combine checks: Must have size, be on screen, and either have min area or be a link/button.
                return hasPositiveSize && isOnscreen && (hasSufficientArea || el.tagName === 'A' || el.tagName === 'BUTTON');
            }} catch (e) {{
                // Errors during checks mean we can't be sure, assume not visible
                console.warn('Error in isVisible check:', e);
                return false;
            }}
        }};

        const isInteractiveOrSignificant = (el) => {{
            const tag = el.tagName.toLowerCase();
            const role = (el.getAttribute('role') || '').toLowerCase();

            // Common interactive HTML tags
            const interactiveTags = ['a', 'button', 'input', 'select', 'textarea', 'option', 'label', 'form', 'fieldset', 'details', 'summary', 'dialog', 'menu', 'menuitem'];
            // Common interactive ARIA roles
            const interactiveRoles = ['button', 'link', 'checkbox', 'radio', 'menuitem', 'tab', 'switch', 'option', 'searchbox', 'textbox', 'dialog', 'slider', 'spinbutton', 'combobox', 'listbox'];

            if (interactiveTags.includes(tag) || interactiveRoles.includes(role)) {{
                return true;
            }}

            // Check for explicit interaction handlers or attributes
            if (el.onclick || el.href || el.getAttribute('tabindex') !== null || el.getAttribute('contenteditable') === 'true') {{
                return true;
            }}

            // Consider non-interactive containers with text content if they have sufficient area
            if ((tag === 'div' || tag === 'section' || tag === 'article' || tag === 'main' || tag === 'span') && el.innerText && el.innerText.trim().length > 0) {{
                try {{ const rect = el.getBoundingClientRect(); if (rect.width * rect.height >= MIN_ELEMENT_AREA) return true; }} catch(e) {{}}
            }}

             // Consider images with alt text if they have sufficient area
            if (tag === 'img' && el.alt && el.alt.trim().length > 0) {{
                 try {{ const rect = el.getBoundingClientRect(); if (rect.width * rect.height >= MIN_ELEMENT_AREA) return true; }} catch(e) {{}}
            }}

            return false; // Default to not significant
        }};

        const getElementText = (el) => {{
            try {{
                // Handle specific input types
                if (el.tagName === 'INPUT') {{
                    const inputType = el.type.toLowerCase();
                    if (inputType === 'button' || inputType === 'submit' || inputType === 'reset') return el.value || '';
                    if (inputType === 'password') return 'Password input field'; // Don't expose value
                    // For other inputs, prioritize placeholder, then name, then type
                    return el.placeholder || el.name || el.getAttribute('aria-label') || inputType || '';
                }}
                if (el.tagName === 'TEXTAREA') {{
                     return el.placeholder || el.name || el.getAttribute('aria-label') || '';
                }}
                if (el.tagName === 'SELECT') {{
                     // Try associated label first
                     if (el.id) {{
                         const labels = document.querySelectorAll(`label[for="${{el.id}}"]`);
                         if (labels.length > 0 && labels[0].textContent) return labels[0].textContent.trim();
                     }}
                     return el.name || el.getAttribute('aria-label') || '';
                }}
                if (el.tagName === 'IMG') {{
                    return el.alt || ''; // Use alt text for images
                }}
                // Prefer aria-label if present
                const ariaLabel = el.getAttribute('aria-label');
                if (ariaLabel) return ariaLabel.trim();

                // Look for associated label via `for` attribute (if not already handled for select)
                if (el.id && el.tagName !== 'SELECT') {{
                    const labels = document.querySelectorAll(`label[for="${{el.id}}"]`);
                     if (labels.length > 0 && labels[0].textContent) return labels[0].textContent.trim();
                }}

                // Fallback to combined text content of direct children text nodes
                let textContent = '';
                for (const node of el.childNodes) {{
                    // Only include direct text node children
                    if (node.nodeType === Node.TEXT_NODE) {{
                        textContent += node.textContent;
                    }}
                }}
                textContent = textContent.trim();

                // If text node content is empty, fallback to innerText (which includes descendants)
                if (!textContent) {{
                    textContent = el.innerText ? el.innerText.trim() : '';
                }}

                // Limit text length? Maybe not here, handle later.
                return textContent;

            }} catch (e) {{
                console.warn('Error in getElementText:', e);
                return ''; // Return empty string on error
            }}
        }};

        // --- Traversal Logic ---
        const outputElements = [];
        const queue = [document.documentElement]; // Start traversal from root
        const visited = new Set(); // Keep track of visited nodes
        let elementIndex = 0; // Counter for unique element IDs

        while (queue.length > 0 && outputElements.length < MAX_ELEMENTS) {{
            const node = queue.shift(); // Get next node from queue

            if (!node || visited.has(node)) {{
                continue; // Skip if node is null or already visited
            }}
            visited.add(node);

            // Process the node if it's interactive/significant and visible
            if (isInteractiveOrSignificant(node) && isVisible(node)) {{
                try {{
                    const rect = node.getBoundingClientRect();
                    // Assign a unique ID for referencing later
                    const elementId = `${{prefix || ''}}el_${{elementIndex++}}`;
                    node.dataset.sbId = elementId; // Store ID on the element itself

                    // Collect element information
                    outputElements.push({{
                        id: elementId,
                        tag: node.tagName.toLowerCase(),
                        role: node.getAttribute("role") || "", // Get ARIA role
                        text: getElementText(node), // Get representative text
                        bbox: [ // Bounding box coordinates
                            Math.round(rect.x),
                            Math.round(rect.y),
                            Math.round(rect.width),
                            Math.round(rect.height)
                        ]
                    }});
                }} catch (e) {{
                    console.warn('Error processing element:', node, e);
                }}
            }}

            // --- Queue Children for Traversal ---
            // Check for Shadow DOM children first
            if (node.shadowRoot) {{
                const shadowChildren = node.shadowRoot.children;
                if (shadowChildren) {{
                    for (let i = 0; i < shadowChildren.length; i++) {{
                        if (!visited.has(shadowChildren[i])) {{
                            queue.push(shadowChildren[i]);
                        }}
                    }}
                }}
            }}
            // Check for regular children
            else if (node.children) {{
                 const children = node.children;
                 for (let i = 0; i < children.length; i++) {{
                     if (!visited.has(children[i])) {{
                         queue.push(children[i]);
                     }}
                 }}
            }}

            // Check for IFRAME content document
            if (node.tagName === 'IFRAME') {{
                 try {{
                    // Access contentDocument carefully due to potential cross-origin restrictions
                    if (node.contentDocument && node.contentDocument.documentElement) {{
                         if (!visited.has(node.contentDocument.documentElement)) {{
                             queue.push(node.contentDocument.documentElement);
                         }}
                    }}
                 }} catch (iframeError) {{
                     console.warn('Could not access iframe content:', node.src || '[no src]', iframeError.message);
                 }}
            }}
        }} // End while loop

        return outputElements; // Return the collected element data
    }}
    """


async def _build_page_map(
    page: Page,
) -> Tuple[
    Dict[str, Any], str
]:  # Uses globals _max_section_chars_global, _max_widgets_global, _log
    """Builds a structured representation (map) of the current page content and elements."""
    # Calculate fingerprint first to check cache
    fp = await _dom_fingerprint(page)

    # Check if cached map exists on the page object for the current fingerprint
    if hasattr(page, "_sb_page_map") and hasattr(page, "_sb_fp") and page._sb_fp == fp:
        logger.debug(f"Using cached page map for {page.url} (FP: {fp[:8]}...).")
        cached_map = page._sb_page_map
        return cached_map, fp

    logger.debug(f"Building new page map for {page.url} (FP: {fp[:8]}...).")
    # Initialize map components
    await _ensure_readability(page)  # Ensure Readability.js is available
    main_txt = ""
    elems: List[Dict[str, Any]] = []
    page_title = "[Error Getting Title]"

    try:
        # 1. Extract Main Text Content
        html_content = await page.content()
        if html_content:
            # Try Readability first
            extracted_text = await page.evaluate(_READ_JS_WRAPPER, html_content)
            main_txt = extracted_text or ""

            # Fallback if Readability yields short content
            if len(main_txt) < 200:
                logger.debug("Readability text short (<200 chars), trying basic text extraction.")

                # Define the synchronous extraction helper locally
                def extract_basic_text(html_str):
                    try:
                        # Limit HTML size processed by BeautifulSoup
                        max_html_size = 3 * 1024 * 1024
                        limited_html = html_str[:max_html_size]
                        soup = BeautifulSoup(limited_html, "lxml")
                        # Remove common non-content tags before text extraction
                        tags_to_remove = [
                            "script",
                            "style",
                            "nav",
                            "footer",
                            "header",
                            "aside",
                            "form",
                            "figure",
                        ]
                        found_tags = soup(tags_to_remove)
                        for tag in found_tags:
                            tag.decompose()
                        # Get text, join with spaces, strip extra whitespace
                        basic_text = soup.get_text(" ", strip=True)
                        return basic_text
                    except Exception as bs_err:
                        logger.warning(f"Basic text extraction with BeautifulSoup failed: {bs_err}")
                        return ""  # Return empty on error

                # Run the sync extraction in the thread pool
                loop = asyncio.get_running_loop()
                pool = _get_pool()
                fallback_text = await loop.run_in_executor(pool, extract_basic_text, html_content)
                main_txt = fallback_text  # Use fallback result

            # Limit the length of the extracted main text
            main_txt = main_txt[:_max_section_chars_global]
        else:
            logger.warning(f"Failed to get HTML content for page map on {page.url}.")

        # 2. Extract Interactive Elements (across all frames)
        js_func = _shadow_deep_js()  # Get the JS function string
        all_extracted_elems = []
        all_frames = page.frames
        for i, frame in enumerate(all_frames):
            if frame.is_detached():
                logger.debug(f"Skipping detached frame {i}.")
                continue
            frame_url_short = (frame.url or "unknown")[:80]
            try:
                # Evaluate element extraction JS in the frame with timeout
                frame_prefix = f"f{i}:"  # Prefix IDs with frame index
                frame_elems = await asyncio.wait_for(
                    frame.evaluate(js_func, frame_prefix), timeout=5.0
                )
                all_extracted_elems.extend(frame_elems)
                # Log extraction count per frame *only if* elements were found
                if frame_elems:
                    logger.debug(
                        f"Extracted {len(frame_elems)} elements from frame {i} ({frame_url_short})."
                    )
            except PlaywrightTimeoutError:
                logger.warning(f"Timeout evaluating elements in frame {i} ({frame_url_short})")
            except PlaywrightException as e:
                # Be more specific about error logging - avoid logging full exception in normal operation unless debug level
                logger.warning(
                    f"Playwright error evaluating elements in frame {i} ({frame_url_short}): {type(e).__name__}"
                )
                logger.debug(
                    f"Full PlaywrightException in frame {i}: {e}", exc_info=False
                )  # Log full exception only at debug
            except Exception as e:
                logger.error(
                    f"Unexpected error evaluating elements in frame {i} ({frame_url_short}): {e}",
                    exc_info=True,  # Log full traceback for unexpected errors
                )

        # Limit the total number of elements stored
        elems = all_extracted_elems[:_max_widgets_global]
        logger.debug(
            f"Total elements extracted: {len(all_extracted_elems)}, stored (limited): {len(elems)}"
        )  # Log total and limited count

        # 3. Get Page Title
        try:
            page_title_raw = await page.title()
            page_title = page_title_raw.strip() if page_title_raw else "[No Title]"
        except PlaywrightException as title_err:
            logger.warning(f"Could not get page title for {page.url}: {title_err}")
            # Keep default error title

    except PlaywrightException as e:
        logger.error(
            f"Could not build page map for {page.url}: Playwright error: {e}", exc_info=True
        )
    except Exception as e:
        logger.error(f"Unexpected error building page map for {page.url}: {e}", exc_info=True)

    # Removed the specific logging block that depended on URL_BOOKSTORE

    # Assemble the final page map dictionary
    page_map = {
        "url": page.url,
        "title": page_title,
        "main_text": main_txt,
        "elements": elems,  # Contains the limited list of elements
    }

    # Cache the newly built map and its fingerprint on the page object
    page._sb_page_map = page_map
    page._sb_fp = fp
    logger.debug(f"Page map built and cached for {page.url}.")

    return page_map, fp


_SM_GLOBAL = difflib.SequenceMatcher(autojunk=False)


def _ratio(a: str, b: str) -> float:  # Keep as is
    """Calculate similarity ratio between two strings using SequenceMatcher."""
    if not a or not b:
        return 0.0
    # Set sequences for the global matcher instance
    _SM_GLOBAL.set_seqs(a, b)
    # Calculate and return the ratio
    similarity_ratio = _SM_GLOBAL.ratio()
    return similarity_ratio


def _heuristic_pick(
    pm: Dict[str, Any], hint: str, role: Optional[str]
) -> Optional[str]:  # Uses global _seq_cutoff_global
    """Finds the best element ID based on text similarity and heuristics."""
    # Basic validation
    if not hint or not pm or not pm.get("elements"):
        return None

    # Normalize hint text (Unicode normalization and lowercase)
    h_norm = unicodedata.normalize("NFC", hint).lower()
    best_id: Optional[str] = None
    best_score: float = -1.0
    target_role_lower = role.lower() if role else None

    elements_list = pm.get("elements", [])
    for e in elements_list:
        if not e or not isinstance(e, dict):
            continue  # Skip invalid element entries

        el_id = e.get("id")
        el_text_raw = e.get("text", "")
        el_role_raw = e.get("role", "")
        el_tag_raw = e.get("tag", "")

        if not el_id:
            continue  # Skip elements without our assigned ID

        # Normalize element text
        el_text_norm = unicodedata.normalize("NFC", el_text_raw).lower()
        el_role_lower = el_role_raw.lower()
        el_tag_lower = el_tag_raw.lower()

        # Role filtering (if role specified)
        # Allow matching button tag if role is button
        is_role_match = target_role_lower == el_role_lower
        is_button_match = target_role_lower == "button" and el_tag_lower == "button"
        if target_role_lower and not is_role_match and not is_button_match:
            continue  # Skip if role doesn't match

        # --- Calculate Score ---
        # Base score: Text similarity
        score = _ratio(h_norm, el_text_norm)

        # Bonus: Exact role match
        if target_role_lower and is_role_match:
            score += 0.1

        # Bonus: Keyword matching (e.g., hint mentions "button" and element is button/role=button)
        hint_keywords = {
            "button",
            "submit",
            "link",
            "input",
            "download",
            "checkbox",
            "radio",
            "tab",
            "menu",
        }
        element_keywords = {el_role_lower, el_tag_lower}
        # Find hint keywords present in the hint text itself
        hint_words_in_hint = set()
        split_hint = h_norm.split()
        for w in split_hint:
            if w in hint_keywords:
                hint_words_in_hint.add(w)
        # Check for intersection between keywords in hint and element's keywords
        common_keywords = hint_words_in_hint.intersection(element_keywords)
        if common_keywords:
            score += 0.15

        # Bonus: Hint likely refers to label/placeholder and element seems related
        has_label_hints = "label for" in h_norm or "placeholder" in h_norm
        if has_label_hints and score > 0.6:  # Apply only if base similarity is decent
            score += 0.1

        # Penalty: Very short element text compared to a long hint
        is_short_text = len(el_text_raw) < 5
        is_long_hint = len(hint) > 10
        if is_short_text and is_long_hint:
            score -= 0.1

        # Penalty: Generic container tags without a specific role
        is_generic_container = el_tag_lower in ("div", "span")
        has_no_role = not el_role_lower
        if is_generic_container and has_no_role:
            score -= 0.05
        # --- End Score Calculation ---

        # Update best match if current score is higher
        if score > best_score:
            best_id = el_id
            best_score = score

    # Return the best ID found if the score meets the cutoff threshold
    if best_score >= _seq_cutoff_global:
        return best_id
    else:
        return None


async def _llm_pick(
    pm: Dict[str, Any], task_hint: str, attempt: int
) -> Optional[str]:  # Uses global _llm_model_locator_global
    """Asks the LLM to pick the best element ID for a given task hint."""
    if not pm or not task_hint:
        logger.warning("LLM pick skipped: Missing page map or task hint.")
        return None

    # Prepare summary of elements for the LLM prompt
    elements_summary = []
    elements_list = pm.get("elements", [])
    for el in elements_list:
        el_id = el.get("id")
        el_tag = el.get("tag")
        el_role = el.get("role", " ")  # Use space if empty for formatting
        el_text = el.get("text", " ")  # Use space if empty
        # Truncate long text for the prompt
        max_text_len = 80
        truncated_text = el_text[:max_text_len] + ("..." if len(el_text) > max_text_len else "")
        # Format summary string
        summary_str = f"id={el_id} tag={el_tag} role='{el_role}' text='{truncated_text}'"
        elements_summary.append(summary_str)

    # System prompt defining the task
    system_prompt = textwrap.dedent("""
        You are an expert web automation assistant. Your task is to identify the single best HTML element ID from the provided list that corresponds to the user's request.
        Analyze the user's task hint and the list of elements (with their ID, tag, role, and text).
        Choose the element ID (e.g., "el_12" or "f0:el_5") that is the most likely target for the user's action.
        Consider the element's text, role, tag, and the user's likely intent.
        If multiple elements seem possible, prioritize elements with clear interactive roles (button, link, input, etc.) or specific text matches.
        If no element is a clear match for the task hint, respond with `{"id": null}`.
        Respond ONLY with a JSON object containing the chosen element ID under the key "id". Example: `{"id": "el_42"}` or `{"id": "f1:el_10"}` or `{"id": null}`. Do NOT include explanations or markdown formatting.
    """).strip()

    # User prompt containing the context and request
    elements_str = "\n".join(elements_summary)
    user_prompt = textwrap.dedent(f"""
        Page Title: {pm.get("title", "[No Title]")}
        Page URL: {pm.get("url", "[No URL]")}

        Available Elements:
        {elements_str}

        User Task Hint: "{task_hint}"
        Attempt Number: {attempt}

        Based on the task hint and element list, which element ID should be targeted?
        Respond ONLY with a JSON object containing the 'id' (string or null).
    """).strip()

    # Prepare messages for the LLM call
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    # Call the LLM, expecting a JSON response
    res = await _call_llm(
        msgs,
        model=_llm_model_locator_global,  # Use configured model
        expect_json=True,
        temperature=0.0,  # Low temperature for deterministic selection
        max_tokens=100,  # Should be enough for {"id": "fX:el_YYY"}
    )

    # Process the LLM response
    if isinstance(res, dict):
        if "id" in res:
            el_id = res.get("id")
            # Validate the format of the returned ID (string starting with el_ or f*:el_, or null)
            is_valid_null = el_id is None
            is_valid_string_format = isinstance(el_id, str) and re.match(
                r"^(?:f\d+:)?el_\d+$", el_id
            )
            if is_valid_null or is_valid_string_format:
                if el_id:
                    logger.debug(
                        f"LLM picked ID: {el_id} for hint '{task_hint}' (Attempt {attempt})"
                    )
                else:
                    logger.debug(
                        f"LLM explicitly picked null ID for hint '{task_hint}' (Attempt {attempt})"
                    )
                return el_id
            else:
                # Log warning if ID format is invalid
                logger.warning(
                    f"LLM returned invalid ID format: {el_id} for hint '{task_hint}' (Attempt {attempt})"
                )
                return None  # Treat invalid format as no pick
        elif "error" in res:
            # Log error if LLM call failed
            error_msg = res["error"]
            logger.warning(
                f"LLM picker failed for hint '{task_hint}' (Attempt {attempt}): {error_msg}"
            )
            return None  # Treat LLM error as no pick
        else:
            # Log warning if response dictionary format is unexpected
            logger.warning(
                f"LLM picker returned unexpected dict format: {res.keys()} for hint '{task_hint}' (Attempt {attempt})"
            )
            return None  # Treat unexpected format as no pick
    else:
        # Log warning if the response is not a dictionary
        res_type = type(res).__name__
        logger.warning(
            f"LLM picker returned unexpected response type: {res_type} for hint '{task_hint}' (Attempt {attempt})"
        )
        return None  # Treat unexpected type as no pick


async def _loc_from_id(page: Page, el_id: str) -> Locator:  # Keep as is
    """Gets a Playwright Locator object from a data-sb-id attribute."""
    if not el_id:
        raise ValueError("Element ID cannot be empty when creating locator.")

    # Escape the ID for use in CSS selector (esp. if ID contains quotes or backslashes)
    # Double backslashes for Python string literal, then double again for CSS escaping
    escaped_id_inner = el_id.replace("\\", "\\\\").replace('"', '\\"')
    selector = f'[data-sb-id="{escaped_id_inner}"]'

    # Check if the ID indicates a specific frame (e.g., "f0:el_12")
    if ":" in el_id and el_id.startswith("f"):
        try:
            frame_prefix, element_part = el_id.split(":", 1)
            frame_index_str = frame_prefix[1:]  # Get the number part after 'f'
            frame_index = int(frame_index_str)
            all_frames = page.frames
            if 0 <= frame_index < len(all_frames):
                target_frame = all_frames[frame_index]
                # Return the locator within the specified frame
                locator_in_frame = target_frame.locator(selector).first
                return locator_in_frame
            else:
                # Log warning if frame index is out of bounds, fallback to main frame
                logger.warning(
                    f"Frame index {frame_index} from ID '{el_id}' is out of bounds (0-{len(all_frames) - 1}). Falling back to main frame search."
                )
        except (ValueError, IndexError) as e:
            # Log warning if parsing fails, fallback to main frame
            logger.warning(
                f"Could not parse frame index from ID '{el_id}'. Falling back to main frame search. Error: {e}"
            )

    # Default: return locator in the main frame
    locator_in_main = page.locator(selector).first
    return locator_in_main


# --- Enhanced Locator (as a helper class, not a tool itself) ---
class EnhancedLocator:  # Keep class, but it's used INTERNALLY by standalone functions
    """Unified locator using cache, heuristics, and LLM fallback."""

    def __init__(self, page: Page):
        self.page = page
        # Determine site identifier from URL for caching
        self.site = "unknown"
        try:
            page_url = page.url or ""  # Handle case where URL might be None/empty
            parsed = urlparse(page_url)
            netloc_raw = parsed.netloc.lower()
            # Remove www. prefix if present
            netloc_clean = netloc_raw.replace("www.", "")
            # Use cleaned netloc, fallback to 'unknown' if empty
            self.site = netloc_clean or "unknown"
        except Exception as e:
            logger.warning(f"Error parsing site from URL '{page.url}' for EnhancedLocator: {e}")
            # Keep self.site as "unknown"
            pass
        # Internal cache for page map and fingerprint for current instance lifecycle
        self._pm: Optional[Dict[str, Any]] = None
        self._pm_fp: Optional[str] = None
        # Timestamp for throttling network idle checks
        self._last_idle_check: float = 0.0

    async def _maybe_wait_for_idle(self, timeout: float = 1.5):  # Uses global _last_idle_check
        """Waits for network idle state, throttled to avoid excessive waits."""
        now = time.monotonic()
        time_since_last_check = now - self._last_idle_check
        # Only check if enough time has passed since the last check
        if time_since_last_check > 1.0:  # Check at most once per second
            try:
                # Wait for network to be idle for a short period
                timeout_ms = int(timeout * 1000)
                await self.page.wait_for_load_state("networkidle", timeout=timeout_ms)
                self._last_idle_check = time.monotonic()  # Update timestamp on success
            except PlaywrightException:
                # Ignore timeout or other errors, just update timestamp
                self._last_idle_check = time.monotonic()

    async def _get_page_map(self) -> Tuple[Dict[str, Any], str]:  # Calls global _build_page_map
        """Gets the current page map, potentially building it if needed."""
        # Short wait/idle check before building map to allow dynamic content to settle
        await self._maybe_wait_for_idle()
        sleep_duration = random.uniform(0.1, 0.25)  # Small random delay
        await asyncio.sleep(sleep_duration)

        # Build the page map (which includes fingerprint check internally)
        pm, fp = await _build_page_map(self.page)

        # Store locally for potential reuse within this instance lifecycle
        self._pm = pm
        self._pm_fp = fp
        return pm, fp

    async def _selector_cached(
        self, key: str, fp: str
    ) -> Optional[Locator]:  # Calls global _cache_get_sync, _log
        """Checks cache for a selector, validates it, and returns Locator if valid."""
        loop = asyncio.get_running_loop()
        pool = _get_pool()
        # Perform synchronous cache read in thread pool
        sel = await loop.run_in_executor(pool, _cache_get_sync, key, fp)

        if sel:
            logger.debug(f"Cache hit for key prefix {key[:8]}. Selector: '{sel}'")
            try:
                # Extract element ID from selector string like '[data-sb-id="f0:el_12"]'
                match = re.search(r'data-sb-id="([^"]+)"', sel)
                if not match:
                    logger.warning(
                        f"Cached selector '{sel}' has unexpected format. Ignoring cache."
                    )
                    return None
                loc_id = match.group(1)

                # Get the Playwright Locator object using the ID
                loc = await _loc_from_id(self.page, loc_id)

                # Quick check if the element is visible (short timeout)
                await loc.wait_for(state="visible", timeout=500)  # 500ms check

                # Log cache hit and return the valid locator
                log_key = key[:8]
                await _log("locator_cache_hit", selector=sel, key=log_key)
                return loc
            except (PlaywrightException, ValueError) as e:
                # Log if cached selector is no longer valid/visible or ID parsing fails
                logger.debug(
                    f"Cached selector '{sel}' failed visibility/location check. Error: {e}"
                )
                # Consider deleting the stale cache entry here?
                # await loop.run_in_executor(pool, _cache_delete_sync, key) # Potentially aggressive
        return None  # Cache miss or invalid cached selector

    async def locate(
        self, task_hint: str, *, role: Optional[str] = None, timeout: int = 5000
    ) -> (
        Locator
    ):  # Uses globals _retry_after_fail_global, _log, _get_pool, _cache_put_sync, _llm_pick
        """
        Finds the best Locator for a task hint using cache, heuristics, LLM, and smarter fallbacks.

        Args:
            task_hint: Natural language description of the element to locate.
            role: Optional specific ARIA role to filter potential matches.
            timeout: Maximum time in milliseconds to find the element.

        Returns:
            A Playwright Locator object pointing to the best match found.

        Raises:
            ValueError: If task_hint is empty.
            PlaywrightTimeoutError: If no suitable element is found within the timeout across all methods.
            ToolError: For internal errors during location.
        """
        if not task_hint or not task_hint.strip():
            raise ValueError("locate requires a non-empty 'task_hint'")

        start_time = time.monotonic()
        timeout_sec = timeout / 1000.0
        loop = asyncio.get_running_loop()
        pool = _get_pool()

        # --- 1. Generate Cache Key ---
        page_url = self.page.url or ""
        parsed_url = urlparse(page_url)
        path = parsed_url.path or "/"
        # Normalize hint and role for cache key stability
        normalized_hint = unicodedata.normalize("NFC", task_hint).lower().strip()
        normalized_role = role.lower().strip() if role else None
        key_data = {
            "site": self.site,
            "path": path,
            "hint": normalized_hint,
            "role": normalized_role,
        }
        key_src = json.dumps(key_data, sort_keys=True)
        key_src_bytes = key_src.encode("utf-8")
        cache_key = hashlib.sha256(key_src_bytes).hexdigest()
        key_preview = cache_key[:8]
        log_prefix = (
            f"EnhancedLocator(key={key_preview}, hint='{task_hint[:50]}...', role='{role}')"
        )
        logger.debug(f"{log_prefix}: Initiating locate.")

        # --- 2. Check Cache with Current DOM Fingerprint ---
        logger.debug(f"{log_prefix}: Checking cache...")
        current_dom_fp = await _dom_fingerprint(self.page)
        logger.debug(f"{log_prefix}: Current DOM FP: {current_dom_fp[:12]}...")
        try:
            cached_loc = await self._selector_cached(cache_key, current_dom_fp)
            if cached_loc:
                logger.info(f"{log_prefix}: Cache HIT.")
                await _log(
                    "locator_success", hint=task_hint, role=role, method="cache", key=key_preview
                )
                return cached_loc
            else:
                logger.debug(f"{log_prefix}: Cache MISS.")
        except Exception as cache_err:
            logger.warning(f"{log_prefix}: Error checking cache: {cache_err}")

        # --- 3. Cache Miss: Get Page Map and Try Heuristics ---
        logger.debug(f"{log_prefix}: Trying heuristics...")
        try:
            (
                pm,
                current_dom_fp,
            ) = await self._get_page_map()  # Get map (updates fingerprint if changed)
            map_keys = list(pm.keys()) if pm else []
            num_elements = len(pm.get("elements", [])) if pm else 0
            logger.debug(
                f"{log_prefix}: Page map obtained. FP={current_dom_fp[:8]}, Keys={map_keys}, Elements={num_elements}"
            )

            heuristic_id = _heuristic_pick(pm, task_hint, role)
            logger.debug(f"{log_prefix}: Heuristic pick result ID: '{heuristic_id}'")

            if heuristic_id:
                try:
                    logger.debug(f"{log_prefix}: Validating heuristic pick ID '{heuristic_id}'...")
                    loc = await _loc_from_id(self.page, heuristic_id)
                    await loc.scroll_into_view_if_needed(timeout=2000)
                    wait_timeout_heur = max(1000, timeout // 3)  # Use portion of timeout
                    logger.debug(
                        f"{log_prefix}: Waiting for heuristic element visibility ({wait_timeout_heur}ms)..."
                    )
                    await loc.wait_for(state="visible", timeout=wait_timeout_heur)
                    logger.info(f"{log_prefix}: Heuristic pick VALIDATED (ID: {heuristic_id}).")

                    # Cache the successful heuristic result
                    selector_str = f'[data-sb-id="{heuristic_id}"]'
                    await loop.run_in_executor(
                        pool, _cache_put_sync, cache_key, selector_str, current_dom_fp
                    )
                    await _log(
                        "locator_heuristic_match", selector=heuristic_id, hint=task_hint, role=role
                    )
                    await _log(
                        "locator_success",
                        hint=task_hint,
                        role=role,
                        method="heuristic",
                        selector=heuristic_id,
                    )
                    return loc
                except (PlaywrightException, ValueError) as e_heur_val:
                    logger.debug(
                        f"{log_prefix}: Heuristic pick '{heuristic_id}' validation FAILED. Error: {e_heur_val}"
                    )
                    # Continue to LLM fallback
        except Exception as map_heur_err:
            logger.warning(
                f"{log_prefix}: Error during page map or heuristic processing: {map_heur_err}"
            )
            # Ensure pm is defined for LLM step, even if empty
            pm = pm if "pm" in locals() else {}
            current_dom_fp = (
                current_dom_fp
                if "current_dom_fp" in locals()
                else await _dom_fingerprint(self.page)
            )

        # --- 4. Heuristic Failed: Try LLM Picker (with retries) ---
        logger.debug(f"{log_prefix}: Trying LLM picker...")
        num_llm_attempts = 1 + _retry_after_fail_global
        for att in range(1, num_llm_attempts + 1):
            elapsed_sec = time.monotonic() - start_time
            if elapsed_sec >= timeout_sec:
                logger.warning(f"{log_prefix}: Timeout reached before completing LLM attempts.")
                break  # Break loop, proceed to fallback

            logger.debug(f"{log_prefix}: LLM pick attempt {att}/{num_llm_attempts}...")
            # Ensure page map 'pm' is available from heuristic step or refreshed
            if not pm or (
                "error" in pm and att > 1
            ):  # Refresh if map invalid or after first attempt
                logger.debug(f"{log_prefix}: Refreshing page map before LLM attempt {att}...")
                try:
                    pm, current_dom_fp = await self._get_page_map()
                    logger.debug(f"{log_prefix}: Page map refreshed. FP={current_dom_fp[:8]}.")
                except Exception as map_refresh_err:
                    logger.warning(
                        f"{log_prefix}: Failed to refresh page map for LLM attempt {att}: {map_refresh_err}"
                    )
                    # Try proceeding without map refresh? Or break? Let's break to avoid confusing LLM.
                    break

            llm_id = await _llm_pick(pm, task_hint, att)
            logger.debug(f"{log_prefix}: LLM pick result (Attempt {att}): ID='{llm_id}'")

            if not llm_id:
                logger.debug(f"{log_prefix}: LLM pick attempt {att} returned no ID.")
                if att < num_llm_attempts:
                    continue  # Refresh happens at start of next loop iteration if needed
                else:
                    break  # Last LLM attempt failed, proceed to fallback

            # LLM returned an ID, try to validate it
            try:
                logger.debug(f"{log_prefix}: Validating LLM pick ID '{llm_id}' (Attempt {att})...")
                loc = await _loc_from_id(self.page, llm_id)
                try:  # Log outerHTML for debugging LLM picks
                    loc_llm_outer_html = await loc.evaluate(
                        "element => element.outerHTML", timeout=500
                    )
                    logger.debug(
                        f"{log_prefix}: LLM picked element outerHTML: {loc_llm_outer_html[:200]}..."
                    )
                except Exception as eval_err:
                    logger.debug(
                        f"{log_prefix}: Error getting outerHTML for LLM pick {llm_id}: {eval_err}"
                    )

                await loc.scroll_into_view_if_needed(timeout=2000)
                elapsed_now_sec = time.monotonic() - start_time
                remaining_timeout_ms = max(500, timeout - int(elapsed_now_sec * 1000))
                if remaining_timeout_ms <= 0:
                    raise PlaywrightTimeoutError("Timeout before LLM validation wait.")
                logger.debug(
                    f"{log_prefix}: Waiting for LLM element visibility ({remaining_timeout_ms}ms)..."
                )
                await loc.wait_for(state="visible", timeout=remaining_timeout_ms)
                logger.info(f"{log_prefix}: LLM pick VALIDATED (ID: {llm_id}, Attempt {att}).")

                # Cache the successful LLM result
                selector_str = f'[data-sb-id="{llm_id}"]'
                await loop.run_in_executor(
                    pool, _cache_put_sync, cache_key, selector_str, current_dom_fp
                )
                await _log(
                    "locator_llm_pick", selector=llm_id, attempt=att, hint=task_hint, role=role
                )
                await _log(
                    "locator_success",
                    hint=task_hint,
                    role=role,
                    method="llm",
                    selector=llm_id,
                    attempt=att,
                )
                return loc
            except (PlaywrightException, ValueError) as e_llm_val:
                logger.debug(
                    f"{log_prefix}: LLM pick '{llm_id}' (attempt {att}) validation FAILED. Error: {e_llm_val}"
                )
                # Continue to next LLM attempt loop iteration (map refresh handled at loop start)

        # --- 5. LLM Failed: Try Fallback Selectors ---
        logger.debug(f"{log_prefix}: Trying fallback selectors...")

        fallback_strategies = [
            (
                "placeholder",
                f'[placeholder*="{task_hint}" i]',
            ),  # Case-insensitive placeholder contains hint
            (
                "aria-label",
                f'[aria-label*="{task_hint}" i]',
            ),  # Case-insensitive aria-label contains hint
            ("exact_text", f'text="{task_hint}"'),  # Exact text match
            (
                "contains_text",
                f'text*="{task_hint}" i',
            ),  # Case-insensitive text contains hint (use cautiously)
        ]

        for name, selector in fallback_strategies:
            elapsed_sec_fb = time.monotonic() - start_time
            remaining_timeout_ms_fb = max(500, timeout - int(elapsed_sec_fb * 1000))
            if remaining_timeout_ms_fb <= 500 and elapsed_sec_fb >= timeout_sec:  # Check both
                logger.warning(
                    f"{log_prefix}: Timeout reached before trying fallback selector '{name}'."
                )
                break  # Stop trying fallbacks if time is up

            logger.debug(
                f"{log_prefix}: Trying fallback strategy '{name}' with selector: {selector}"
            )
            try:
                loc = self.page.locator(selector).first
                # Adjust scroll/wait timeout based on remaining time
                scroll_timeout_fb = max(500, remaining_timeout_ms_fb // 3)
                wait_timeout_fb = max(500, remaining_timeout_ms_fb // 2)

                await loc.scroll_into_view_if_needed(timeout=scroll_timeout_fb)
                logger.debug(
                    f"{log_prefix}: Waiting for fallback '{name}' visibility ({wait_timeout_fb}ms)..."
                )
                await loc.wait_for(state="visible", timeout=wait_timeout_fb)

                # Fallback succeeded
                logger.info(f"{log_prefix}: Locator found via fallback strategy '{name}'.")
                await _log(
                    "locator_text_fallback",
                    selector=selector,
                    hint=task_hint,
                    role=role,
                    strategy=name,
                )
                await _log(
                    "locator_success",
                    hint=task_hint,
                    role=role,
                    method="fallback",
                    strategy=name,
                    selector=selector,
                )
                return loc
            except PlaywrightTimeoutError:
                logger.debug(
                    f"{log_prefix}: Fallback strategy '{name}' (selector: {selector}) failed (Timeout)."
                )
            except PlaywrightException as text_fallback_err:
                logger.debug(
                    f"{log_prefix}: Fallback strategy '{name}' (selector: {selector}) failed (Playwright Error: {text_fallback_err})."
                )
            except Exception as fallback_unexpected:
                logger.warning(
                    f"{log_prefix}: Unexpected error during fallback strategy '{name}': {fallback_unexpected}"
                )

        # --- 6. All Methods Failed ---
        final_elapsed_sec = time.monotonic() - start_time
        log_hint = task_hint[:120]
        log_duration = round(final_elapsed_sec, 1)
        await _log("locator_fail_all", hint=log_hint, duration_s=log_duration, role=role)
        logger.error(
            f"{log_prefix}: FAILED to find element within {timeout_sec:.1f}s using all methods."
        )
        raise PlaywrightTimeoutError(
            f"EnhancedLocator failed to find element for hint: '{task_hint}' within {timeout_sec:.1f}s using all methods (cache, heuristic, LLM, fallbacks)."
        )


# --- Smart Actions (Helpers using EnhancedLocator) ---
async def _detect_web_obstacles(page: Page) -> Dict[str, Any]:
    """Detect common web obstacles that might interfere with automation."""
    obstacles = {
        "captcha_detected": False,
        "cookie_banner": False,
        "cloudflare_challenge": False,
        "login_required": False,
        "details": [],
    }

    try:
        # Comprehensive CAPTCHA detection
        captcha_js = """() => {
            const indicators = [];
            
            // Text-based detection
            if (document.body.innerText.toLowerCase().includes('captcha') ||
                document.body.innerText.toLowerCase().includes('recaptcha') ||
                document.body.innerText.toLowerCase().includes('i\'m not a robot')) {
                indicators.push('captcha_text_found');
            }
            
            // Element-based detection
            if (document.querySelector('iframe[title*="captcha" i]') ||
                document.querySelector('iframe[src*="captcha" i]') ||
                document.querySelector('[id*="captcha" i]') ||
                document.querySelector('[class*="captcha" i]') ||
                document.querySelector('div[class*="recaptcha" i]') ||
                document.querySelector('.g-recaptcha') ||
                document.querySelector('#recaptcha')) {
                indicators.push('captcha_element_found');
            }
            
            // Cookie banner detection
            if (document.querySelector('[class*="cookie" i]') ||
                document.querySelector('[id*="cookie" i]') ||
                document.body.innerText.toLowerCase().includes('accept cookies') ||
                document.body.innerText.toLowerCase().includes('cookie policy')) {
                indicators.push('cookie_banner_found');
            }
            
            // Cloudflare detection
            if (document.body.innerText.includes('Cloudflare') &&
                (document.body.innerText.includes('checking') || 
                 document.body.innerText.includes('security'))) {
                indicators.push('cloudflare_challenge');
            }
            
            // Login detection
            if (document.querySelector('input[type="password"]') &&
                (document.body.innerText.toLowerCase().includes('sign in') ||
                 document.body.innerText.toLowerCase().includes('log in') ||
                 document.body.innerText.toLowerCase().includes('login'))) {
                indicators.push('login_required');
            }
            
            return indicators;
        }"""

        detected_indicators = await page.evaluate(captcha_js)

        # Process results
        for indicator in detected_indicators:
            if "captcha" in indicator:
                obstacles["captcha_detected"] = True
                obstacles["details"].append(f"CAPTCHA detected: {indicator}")
            elif "cookie" in indicator:
                obstacles["cookie_banner"] = True
                obstacles["details"].append(f"Cookie banner detected: {indicator}")
            elif "cloudflare" in indicator:
                obstacles["cloudflare_challenge"] = True
                obstacles["details"].append(f"Cloudflare challenge detected: {indicator}")
            elif "login" in indicator:
                obstacles["login_required"] = True
                obstacles["details"].append(f"Login requirement detected: {indicator}")

        return obstacles

    except Exception as e:
        logger.warning(f"Error detecting web obstacles: {e}")
        return obstacles


@resilient(max_attempts=3, backoff=0.5)
async def smart_click(
    page: Page, task_hint: str, *, target_kwargs: Optional[Dict] = None, timeout_ms: int = 5000
) -> bool:  # Uses global _log, _get_pool, _cache_put_sync
    """Locates an element using a hint and clicks it."""
    # Validate or generate task_hint
    effective_task_hint = task_hint
    if not task_hint or not task_hint.strip():
        if target_kwargs:
            name = target_kwargs.get("name", "")
            role = target_kwargs.get("role", "")
            if name or role:
                role_part = role or "element"
                name_part = f" named '{name}'" if name else ""
                effective_task_hint = f"Click the {role_part}{name_part}"
                logger.warning(f"smart_click missing hint, generated: '{effective_task_hint}'")
            else:
                # Neither name nor role provided in target_kwargs
                raise ToolInputError(
                    "smart_click requires a non-empty 'task_hint' or a 'target' dictionary with 'name' or 'role'."
                )
        else:
            # No target_kwargs provided either
            raise ToolInputError("smart_click requires a non-empty 'task_hint'.")

    # First, detect web obstacles that might interfere with automation
    try:
        obstacles = await _detect_web_obstacles(page)

        # Handle CAPTCHA detection - fail early if trying to click CAPTCHA
        if obstacles["captcha_detected"] and (
            "captcha" in effective_task_hint.lower() or "recaptcha" in effective_task_hint.lower()
        ):
            logger.error(
                f"Cannot click CAPTCHA element: '{effective_task_hint}'. CAPTCHAs are designed to prevent automation."
            )
            raise ToolError(
                f"CAPTCHA interaction blocked for task: '{effective_task_hint}'. "
                "Manual intervention required. CAPTCHAs cannot be automatically solved."
            )

        # Log any obstacles detected for diagnostic purposes
        if any(
            [
                obstacles["captcha_detected"],
                obstacles["cookie_banner"],
                obstacles["cloudflare_challenge"],
                obstacles["login_required"],
            ]
        ):
            await _log(
                "smart_click_obstacles_detected", task_hint=effective_task_hint, obstacles=obstacles
            )
            logger.info(f"Web obstacles detected before click attempt: {obstacles['details']}")

            # Try to handle cookie banners automatically
            if obstacles["cookie_banner"]:
                logger.info("Attempting to dismiss cookie banner before main click action...")
                cookie_selectors = [
                    'button:has-text("Accept")',
                    'button:has-text("Accept All")',
                    'button:has-text("OK")',
                    'button:has-text("Allow")',
                    '[id*="accept" i]',
                    '[class*="accept" i]',
                ]
                for selector in cookie_selectors:
                    try:
                        cookie_btn = page.locator(selector).first
                        await cookie_btn.click(timeout=2000)
                        logger.info(f"Successfully dismissed cookie banner using: {selector}")
                        await asyncio.sleep(0.5)  # Brief pause after dismissal
                        break
                    except Exception:
                        continue

            # Give Cloudflare challenges a moment to complete
            if obstacles["cloudflare_challenge"]:
                logger.info("Cloudflare challenge detected, waiting briefly...")
                await asyncio.sleep(3)

    except Exception as obstacle_err:
        logger.warning(
            f"Error during obstacle detection: {obstacle_err}. Proceeding with click attempt."
        )

    loc_helper = EnhancedLocator(page)
    # Prepare log details, prioritizing target_kwargs if available
    log_target = {}
    if target_kwargs:
        log_target.update(target_kwargs)
    else:
        log_target["hint"] = effective_task_hint  # Log the hint used

    try:
        # Locate the element using the enhanced locator
        element = await loc_helper.locate(task_hint=effective_task_hint, timeout=timeout_ms)
        element_id_for_cache = await element.get_attribute("data-sb-id")

        # Prepare and execute the click
        await element.scroll_into_view_if_needed(timeout=3000)  # Scroll with timeout
        await _pause(page)  # Add jitter before click
        click_timeout = max(1000, timeout_ms // 2)  # Use portion of overall timeout
        await element.click(timeout=click_timeout)

        # Update cache if successful and ID was retrieved
        if element_id_for_cache:
            fp = await _dom_fingerprint(
                page
            )  # Get current fingerprint after click potentially changed DOM
            # Generate cache key again (could be helper function)
            page_url_after_click = page.url or ""
            parsed_url_after_click = urlparse(page_url_after_click)
            path_after_click = parsed_url_after_click.path or "/"
            key_data_after_click = {
                "site": loc_helper.site,
                "path": path_after_click,  # Use path *after* click
                "hint": effective_task_hint.lower(),
            }
            key_src_after_click = json.dumps(key_data_after_click, sort_keys=True)
            cache_key_after_click = hashlib.sha256(key_src_after_click.encode()).hexdigest()
            selector_str = f'[data-sb-id="{element_id_for_cache}"]'
            loop_after_click = asyncio.get_running_loop()
            pool_after_click = _get_pool()
            await loop_after_click.run_in_executor(
                pool_after_click, _cache_put_sync, cache_key_after_click, selector_str, fp
            )

        # Log success
        await _log("click_success", target=log_target)
        return True

    except PlaywrightTimeoutError as e:
        # Element not found or visible within timeout
        await _log("click_fail_notfound", target=log_target, error=str(e))
        raise ToolError(
            f"Click failed: Element not found/visible for hint '{effective_task_hint}'. {e}",
            details=log_target,
        ) from e
    except PlaywrightException as e:
        # Other Playwright errors during click/scroll/locate
        await _log("click_fail_playwright", target=log_target, error=str(e))
        raise ToolError(f"Click failed due to Playwright error: {e}", details=log_target) from e
    except Exception as e:
        # Unexpected errors
        await _log("click_fail_unexpected", target=log_target, error=str(e))
        raise ToolError(f"Unexpected error during click: {e}", details=log_target) from e


@resilient(max_attempts=3, backoff=0.5)
async def smart_type(
    page: Page,
    task_hint: str,
    text: str,
    *,
    press_enter: bool = False,
    clear_before: bool = True,
    target_kwargs: Optional[Dict] = None,
    timeout_ms: int = 5000,
) -> bool:  # Uses global _log, get_secret, _get_pool, _cache_put_sync
    """Locates an element using a hint and types text into it."""
    # Validate or generate task_hint
    effective_task_hint = task_hint
    if not task_hint or not task_hint.strip():
        if target_kwargs:
            name = target_kwargs.get("name", "")
            role = target_kwargs.get("role", "input")  # Default role to input for type
            if name or role:
                role_part = role or "element"
                name_part = f" named '{name}'" if name else ""
                effective_task_hint = f"Type into the {role_part}{name_part}"
                logger.warning(f"smart_type missing hint, generated: '{effective_task_hint}'")
            else:
                raise ToolInputError(
                    "smart_type requires a non-empty 'task_hint' or a 'target' dictionary with 'name' or 'role'."
                )
        else:
            raise ToolInputError("smart_type requires a non-empty 'task_hint'.")

    loc_helper = EnhancedLocator(page)
    # Prepare log details
    log_target = {}
    if target_kwargs:
        log_target.update(target_kwargs)
    else:
        log_target["hint"] = effective_task_hint

    resolved_text = text
    log_value = "***SECRET***"  # Default log value for secrets
    # Resolve secrets if needed
    if text.startswith("secret:"):
        secret_path = text[len("secret:") :]
        try:
            resolved_text = get_secret(secret_path)
            # Keep log_value as "***SECRET***"
        except (KeyError, ValueError, RuntimeError) as e:
            await _log("type_fail_secret", target=log_target, secret_ref=secret_path, error=str(e))
            raise ToolInputError(f"Failed to resolve secret '{secret_path}': {e}") from e
    else:
        # Create safe log value for non-secrets (truncate if long)
        if len(text) > 23:
            log_value = text[:20] + "..."
        else:
            log_value = text

    try:
        # Locate the element
        element = await loc_helper.locate(task_hint=effective_task_hint, timeout=timeout_ms)
        element_id_for_cache = await element.get_attribute("data-sb-id")

        # Prepare and perform the typing action
        await element.scroll_into_view_if_needed(timeout=3000)
        await _pause(page)  # Jitter before interaction

        if clear_before:
            await element.fill("")  # Clear the field first

        # Type the resolved text with human-like delay
        type_delay = random.uniform(30, 80)
        await element.type(resolved_text, delay=type_delay)

        # Optionally press Enter
        if press_enter:
            await _pause(page, (50, 150))  # Short pause before Enter
            try:
                # Try pressing Enter directly
                await element.press(
                    "Enter", timeout=1000, noWaitAfter=True
                )  # Don't wait for navigation here
            except PlaywrightException as e:
                # Fallback: If Enter press fails (e.g., on non-input), try clicking the element again
                # This might trigger submission if it's also a button or linked element.
                logger.warning(
                    f"Enter key press failed for hint '{effective_task_hint}', trying smart_click fallback: {e}"
                )
                try:
                    await smart_click(
                        page, task_hint=effective_task_hint, target_kwargs=target_kwargs
                    )
                except Exception as click_e:
                    logger.warning(
                        f"Fallback smart_click after failed Enter press also failed: {click_e}"
                    )
                    # Decide if this should re-raise or just log. Logging for now.

        # Update cache if successful
        if element_id_for_cache:
            fp = await _dom_fingerprint(page)
            page_url_after_type = page.url or ""
            parsed_url_after_type = urlparse(page_url_after_type)
            path_after_type = parsed_url_after_type.path or "/"
            key_data_after_type = {
                "site": loc_helper.site,
                "path": path_after_type,
                "hint": effective_task_hint.lower(),
            }
            key_src_after_type = json.dumps(key_data_after_type, sort_keys=True)
            cache_key_after_type = hashlib.sha256(key_src_after_type.encode()).hexdigest()
            selector_str = f'[data-sb-id="{element_id_for_cache}"]'
            loop_after_type = asyncio.get_running_loop()
            pool_after_type = _get_pool()
            await loop_after_type.run_in_executor(
                pool_after_type, _cache_put_sync, cache_key_after_type, selector_str, fp
            )

        # Log success
        await _log("type_success", target=log_target, value=log_value, entered=press_enter)
        return True

    except PlaywrightTimeoutError as e:
        # Element not found or visible
        await _log("type_fail_notfound", target=log_target, value=log_value, error=str(e))
        raise ToolError(
            f"Type failed: Element not found/visible for hint '{effective_task_hint}'. {e}",
            details=log_target,
        ) from e
    except PlaywrightException as e:
        # Other Playwright errors
        await _log("type_fail_playwright", target=log_target, value=log_value, error=str(e))
        raise ToolError(f"Type failed due to Playwright error: {e}", details=log_target) from e
    except Exception as e:
        # Unexpected errors
        await _log("type_fail_unexpected", target=log_target, value=log_value, error=str(e))
        raise ToolError(f"Unexpected error during type: {e}", details=log_target) from e


# --- LATE IMPORT TO BREAK CYCLE ---
# Import the decorators here, just before they are needed for the tool functions.
# This assumes the rest of the module has been initialized by the time Python reaches here.
try:
    from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
except ImportError as e:
    # This indicates the cycle might still exist or base failed to load for other reasons
    logger.critical(
        f"CRITICAL: Failed to late-import base decorators needed for Smart Browser tools: {e}"
    )
    raise


@with_tool_metrics
@with_error_handling
async def browse(
    url: str, wait_for_selector: Optional[str] = None, wait_for_navigation: bool = True
) -> Dict[str, Any]:
    """
    Navigates to a URL using a dedicated browser tab, waits for load state
    (and optionally a selector), then extracts and returns the page state.

    Args:
        url: The URL to navigate to (scheme will be added if missing).
        wait_for_selector: Optional CSS selector to wait for after navigation.
        wait_for_navigation: Whether to wait for 'networkidle' (True) or
                             'domcontentloaded' (False).

    Returns:
        A dictionary containing success status and the final page state.
    """
    await _ensure_initialized()
    _update_activity()

    # --- Input Validation ---
    if not isinstance(url, str) or not url.strip():
        raise ToolInputError("URL cannot be empty.")
    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
        logger.debug(f"Prepended 'https://' to URL: {url}")

    # --- Proxy Check ---
    proxy_cfg = _get_proxy_config()
    if proxy_cfg and _PROXY_ALLOWED_DOMAINS_LIST is not None:
        if not _is_domain_allowed_for_proxy(url):
            proxy_server = proxy_cfg.get("server", "Configured Proxy")
            error_msg = f"Navigation blocked by proxy domain rules for '{url}' via {proxy_server}."
            await _log("browse_fail_proxy_disallowed", url=url, proxy=proxy_server)
            raise ToolError(error_msg, error_code="proxy_domain_disallowed")

    # --- Execution ---
    ctx, _ = await get_browser_context()  # Get shared context
    async with _tab_context(ctx) as page:  # Use temp page from shared context
        await _log("navigate_start", url=url)
        try:
            # Determine wait state based on argument
            wait_until_state = "networkidle" if wait_for_navigation else "domcontentloaded"
            nav_timeout = 60000  # 60 seconds
            await page.goto(url, wait_until=wait_until_state, timeout=nav_timeout)

            # Optionally wait for a specific selector
            if wait_for_selector:
                selector_timeout = 15000  # 15 seconds
                try:
                    await page.wait_for_selector(
                        wait_for_selector, state="visible", timeout=selector_timeout
                    )
                    await _log("navigate_wait_selector_ok", url=url, selector=wait_for_selector)
                except PlaywrightTimeoutError:
                    # Log timeout but proceed, might still be usable
                    logger.warning(
                        f"Timeout waiting for selector '{wait_for_selector}' at {url} after navigation."
                    )
                    await _log(
                        "navigate_wait_selector_timeout", url=url, selector=wait_for_selector
                    )

            # Pause and get final state
            await _pause(page, (50, 200))
            state = await get_page_state(page)  # Use helper to get structured state
            await _log("navigate_success", url=url, title=state.get("title"))

            # Return success and page state
            return {"success": True, "page_state": state}

        except PlaywrightException as e:
            # Handle Playwright-specific navigation errors
            await _log("navigate_fail_playwright", url=url, error=str(e))
            # Decorator will wrap this in ToolError
            raise ToolError(f"Navigation failed for {url}: {e}") from e
        except Exception as e:
            # Handle unexpected errors during navigation/state extraction
            await _log("navigate_fail_unexpected", url=url, error=str(e))
            # Decorator will wrap this in ToolError
            raise ToolError(f"Unexpected error browsing {url}: {e}") from e


@with_tool_metrics
@with_error_handling
async def click(
    url: str,
    target: Optional[Dict[str, Any]] = None,
    task_hint: Optional[str] = None,
    wait_ms: int = 1000,
) -> Dict[str, Any]:
    """
    Navigates to a URL, clicks an element identified by task_hint or target,
    waits, and returns the resulting page state.

    Args:
        url: The URL to navigate to first.
        target: Optional dictionary (like Plan-Step target) used to generate hint if task_hint missing.
        task_hint: Natural language description of the element to click.
        wait_ms: Milliseconds to wait after the click action completes.

    Returns:
        A dictionary containing success status and the final page state after the click.
    """
    await _ensure_initialized()
    _update_activity()

    # --- Input Validation: Determine task_hint ---
    effective_task_hint = task_hint
    if not effective_task_hint:
        if target and (target.get("name") or target.get("role")):
            name = target.get("name", "")
            role = target.get("role", "")
            role_part = role or "element"
            name_part = f" named '{name}'" if name else ""
            effective_task_hint = f"Click the {role_part}{name_part}"
            logger.debug(f"click tool generated task_hint: '{effective_task_hint}'")
        else:
            raise ToolInputError(
                "click tool requires 'task_hint', or 'target' dict with 'name' or 'role'."
            )

    # --- Execution ---
    ctx, _ = await get_browser_context()
    async with _tab_context(ctx) as page:
        await _log("click_extract_navigate", url=url, hint=effective_task_hint)
        # Navigate to the page
        try:
            nav_timeout = 60000
            await page.goto(url, wait_until="networkidle", timeout=nav_timeout)
        except PlaywrightException as e:
            raise ToolError(f"Navigation to '{url}' failed before click attempt: {e}") from e

        # Perform the click using the smart helper
        # smart_click handles EnhancedLocator, interaction, logging, and errors
        await smart_click(
            page,
            task_hint=effective_task_hint,
            target_kwargs=target,  # Pass target for logging inside smart_click
            timeout_ms=10000,  # Timeout for locating the element
        )

        # Wait after click if specified
        if wait_ms > 0:
            await page.wait_for_timeout(wait_ms)

        # Wait for network to potentially settle after click (best effort)
        try:
            idle_timeout = 10000
            await page.wait_for_load_state("networkidle", timeout=idle_timeout)
        except PlaywrightTimeoutError:
            logger.debug("Network idle wait timeout after click action.")

        # Pause and get final state
        await _pause(page, (50, 200))
        final_state = await get_page_state(page)
        await _log("click_extract_success", url=page.url, hint=effective_task_hint)

        # Return success and the state after the click
        return {"success": True, "page_state": final_state}


@with_tool_metrics
@with_error_handling
async def type_text(
    url: str,
    fields: List[Dict[str, Any]],
    submit_hint: Optional[str] = None,
    submit_target: Optional[Dict[str, Any]] = None,
    wait_after_submit_ms: int = 2000,
) -> Dict[str, Any]:
    """
    Navigates to a URL, fills specified form fields using task hints,
    optionally clicks a submit element, waits, and returns the final page state.

    Args:
        url: The URL containing the form.
        fields: A list of dictionaries, each specifying a field to type into.
                Required keys per dict: 'task_hint' (or 'target') and 'text'.
                Optional keys: 'enter' (bool), 'clear_before' (bool).
        submit_hint: Optional natural language description of the submit element.
        submit_target: Optional target dictionary for the submit element.
        wait_after_submit_ms: Milliseconds to wait after submission.

    Returns:
        A dictionary containing success status and the final page state.
    """
    await _ensure_initialized()
    _update_activity()

    # --- Input Validation ---
    if not fields or not isinstance(fields, list):
        raise ToolInputError("'fields' must be a non-empty list of dictionaries.")
    if submit_hint and submit_target:
        logger.warning("Both submit_hint and submit_target provided; submit_hint will be used.")
    elif not submit_hint and not submit_target:
        logger.debug("No submit_hint or submit_target provided; form will not be submitted.")

    # --- Execution ---
    ctx, _ = await get_browser_context()
    async with _tab_context(ctx) as page:
        await _log("fill_form_navigate", url=url)
        # Navigate to the form page
        try:
            nav_timeout = 60000
            await page.goto(url, wait_until="networkidle", timeout=nav_timeout)
        except PlaywrightException as e:
            raise ToolError(f"Navigation to '{url}' failed before filling form: {e}") from e

        # Wait briefly for form elements to likely appear (best effort)
        try:
            form_wait_timeout = 5000
            await page.wait_for_selector(
                "form, input, textarea, select", state="visible", timeout=form_wait_timeout
            )
            logger.debug("Form elements found, proceeding with field filling.")
        except PlaywrightTimeoutError:
            logger.warning("Did not quickly find typical form elements. Proceeding anyway.")

        # Loop through fields and type text
        for i, field in enumerate(fields):
            if not isinstance(field, dict):
                raise ToolInputError(f"Item at index {i} in 'fields' is not a dictionary.")

            # Determine hint for the field
            field_hint = field.get("task_hint")
            field_target = field.get("target")
            if not field_hint:
                if field_target and (field_target.get("name") or field_target.get("role")):
                    name = field_target.get("name", "")
                    role = field_target.get("role", "input")
                    field_hint = (
                        f"{role or 'Input field'} '{name}'" if name else f"{role or 'Input field'}"
                    )
                else:
                    raise ToolInputError(
                        f"Field at index {i} requires 'task_hint' or 'target' with name/role."
                    )

            # Get text to type
            text_to_type = field.get("text")
            if text_to_type is None:  # Allow empty string, but not None
                raise ToolInputError(
                    f"Field at index {i} ('{field_hint}') missing required 'text'."
                )

            # Log the action for this field
            await _log("fill_form_field", index=i, hint=field_hint)

            # Use smart_type helper for the actual typing
            await smart_type(
                page,
                task_hint=field_hint,
                text=text_to_type,
                press_enter=field.get("enter", False),
                clear_before=field.get("clear_before", True),
                target_kwargs=field_target,  # Pass target for logging inside smart_type
                timeout_ms=5000,
            )
            await _pause(page, (50, 150))  # Short pause between fields

        # Handle optional submission
        final_submit_hint = submit_hint
        if not final_submit_hint and submit_target:  # Generate hint from target if needed
            if submit_target.get("name") or submit_target.get("role"):
                name = submit_target.get("name", "")
                role = submit_target.get("role", "button")
                final_submit_hint = f"Submit {role or 'button'}" + (f" '{name}'" if name else "")
            else:
                logger.warning(
                    "submit_target provided but lacks 'name' or 'role'; cannot generate hint. Skipping submit."
                )
                final_submit_hint = None  # Ensure submit doesn't happen

        if final_submit_hint:
            await _log("fill_form_submit", hint=final_submit_hint)
            # Use smart_click helper for submission
            await smart_click(
                page,
                task_hint=final_submit_hint,
                target_kwargs=submit_target,
                timeout_ms=10000,
            )
            # Wait after submission
            try:
                submit_idle_timeout = 15000
                await page.wait_for_load_state("networkidle", timeout=submit_idle_timeout)
            except PlaywrightTimeoutError:
                logger.debug("Network idle wait timeout after form submission.")
            if wait_after_submit_ms > 0:
                await page.wait_for_timeout(wait_after_submit_ms)

        # Get final page state
        await _pause(page, (100, 300))
        final_state = await get_page_state(page)
        await _log(
            "fill_form_success",
            url=page.url,
            num_fields=len(fields),
            submitted=bool(final_submit_hint),
        )

        return {"success": True, "page_state": final_state}


@with_tool_metrics
@with_error_handling
async def parallel(
    urls: List[str], action: str = "get_state", max_tabs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Processes multiple URLs in parallel using isolated browser tabs via TabPool.
    Currently only supports the 'get_state' action for each URL.

    Args:
        urls: A list of URLs to process.
        action: The action to perform on each URL (currently only 'get_state').
        max_tabs: Optional override for the maximum number of concurrent tabs.
                  If None, uses the globally configured limit.

    Returns:
        A dictionary containing success status, a list of results for each URL,
        and counts of processed and successful URLs.
    """
    await _ensure_initialized()
    _update_activity()

    # --- Input Validation ---
    if not urls or not isinstance(urls, list):
        raise ToolInputError("'urls' must be a non-empty list.")
    if not all(isinstance(u, str) and u.strip() for u in urls):
        raise ToolInputError("All items in 'urls' list must be non-empty strings.")
    if action != "get_state":
        raise ToolInputError(
            f"Unsupported action '{action}'. Currently only 'get_state' is allowed."
        )
    if (
        max_tabs is not None
        and not isinstance(max_tabs, int)
        or (isinstance(max_tabs, int) and max_tabs <= 0)
    ):
        raise ToolInputError("'max_tabs' override must be a positive integer if provided.")

    # --- Setup Tab Pool ---
    # Use global pool unless max_tabs override is provided
    pool_to_use = tab_pool
    if max_tabs is not None:
        logger.info(f"Using temporary TabPool with max_tabs override: {max_tabs}")
        pool_to_use = TabPool(max_tabs=max_tabs)

    # --- Define Per-URL Processing Function ---
    # This function runs inside the TabPool's managed page context
    async def process_url_action(page: Page, *, url_to_process: str) -> Dict[str, Any]:
        # Ensure URL has scheme
        full_url = (
            url_to_process
            if url_to_process.startswith(("http://", "https://"))
            else f"https://{url_to_process}"
        )
        result = {"url": url_to_process, "success": False}  # Default result structure

        try:
            await _log("parallel_navigate", url=full_url, action=action)
            # Navigate to the URL
            nav_timeout = 45000  # Shorter timeout for parallel tasks
            await page.goto(full_url, wait_until="networkidle", timeout=nav_timeout)

            # Perform the specified action
            if action == "get_state":
                page_state = await get_page_state(page)
                result["success"] = True
                result["page_state"] = page_state
            # Add other actions here if needed in the future
            # elif action == "some_other_action":
            #     # ... perform other action ...
            #     result["success"] = True
            #     result["details"] = ...

            return result

        except PlaywrightException as e:
            error_msg = f"Playwright error processing {full_url}: {e}"
            logger.warning(error_msg)
            await _log("parallel_url_error", url=full_url, action=action, error=str(e))
            result["error"] = error_msg
            return result
        except Exception as e:
            error_msg = f"Unexpected error processing {full_url}: {e}"
            logger.error(error_msg, exc_info=True)  # Log traceback for unexpected
            await _log("parallel_url_error", url=full_url, action=action, error=str(e))
            result["error"] = error_msg
            return result

    # --- Create Tasks for TabPool ---
    # Use functools.partial to pass the specific URL to each task instance
    tasks_to_run = []
    for u in urls:
        # Create a partial function that captures the url_to_process kwarg
        task_func = functools.partial(process_url_action, url_to_process=u)
        tasks_to_run.append(task_func)

    # --- Run Tasks Concurrently using TabPool ---
    logger.info(f"Starting parallel processing of {len(urls)} URLs with action '{action}'...")
    # pool.map handles concurrency, semaphore, context/page creation/cleanup
    results = await pool_to_use.map(tasks_to_run)
    logger.info("Parallel processing complete.")

    # --- Process Results ---
    successful_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    processed_count = len(results)
    await _log(
        "parallel_process_complete",
        total=len(urls),
        processed=processed_count,
        successful=successful_count,
        action=action,
    )

    # --- Return Final Summary ---
    return {
        "success": True,  # Indicates the overall parallel orchestration completed
        "results": results,  # List containing result dict for each URL
        "processed_count": processed_count,
        "successful_count": successful_count,
    }


# --- Download Helpers ---
async def _run_in_thread(func, *args):  # Keep as is
    """Runs a synchronous function in the thread pool."""
    loop = asyncio.get_running_loop()
    pool = _get_pool()
    try:
        result = await loop.run_in_executor(pool, func, *args)
        return result
    except RuntimeError as e:
        if "cannot schedule new futures after shutdown" in str(e):
            logger.warning("Thread pool is shutdown. Creating a temporary pool for operation.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as temp_pool:
                result = await loop.run_in_executor(temp_pool, func, *args)
                return result
        else:
            raise


async def _compute_hash_async(data: bytes) -> str:  # Keep as is
    """Computes SHA256 hash of bytes data asynchronously in a thread."""

    # Define the synchronous hashing function locally
    def sync_hash(d):
        hasher = hashlib.sha256()
        hasher.update(d)
        return hasher.hexdigest()

    # Run the sync function in the thread pool
    hex_digest = await _run_in_thread(sync_hash, data)
    return hex_digest


async def _read_file_async(path: Path) -> bytes:  # Keep as is
    """Reads file content asynchronously using aiofiles."""
    async with aiofiles.open(path, mode="rb") as f:
        content = await f.read()
        return content


async def _write_file_async(path: Path, data: bytes):  # Keep as is
    """Writes bytes data to a file asynchronously using aiofiles."""
    async with aiofiles.open(path, mode="wb") as f:
        await f.write(data)


def _extract_tables_sync(path: Path) -> List[Dict]:  # Keep as is
    """Synchronously extracts tables from PDF, Excel, or CSV files."""
    ext = path.suffix.lower()
    results: List[Dict] = []
    try:
        if ext == ".pdf":
            try:
                import tabula  # Optional dependency

                # Read all tables from all pages, keep data as strings
                dfs = tabula.read_pdf(
                    str(path),
                    pages="all",
                    multiple_tables=True,
                    pandas_options={"dtype": str},
                    silent=True,
                )
                if dfs:  # If tables were found
                    table_list = []
                    for i, df in enumerate(dfs):
                        # Convert DataFrame to list of dicts (rows)
                        rows_data = df.to_dict(orient="records")
                        table_entry = {"type": "pdf_table", "page": i + 1, "rows": rows_data}
                        table_list.append(table_entry)
                    results = table_list
            except ImportError:
                logger.debug("tabula-py library not installed. Skipping PDF table extraction.")
            except Exception as pdf_err:
                # Catch errors during Tabula processing
                logger.warning(f"Tabula PDF table extraction failed for {path.name}: {pdf_err}")

        elif ext in (".xls", ".xlsx"):
            try:
                import pandas as pd  # Optional dependency

                # Read all sheets, keep data as strings
                xl_dict = pd.read_excel(str(path), sheet_name=None, dtype=str)
                sheet_list = []
                for sheet_name, df in xl_dict.items():
                    rows_data = df.to_dict(orient="records")
                    sheet_entry = {
                        "type": "excel_sheet",
                        "sheet_name": sheet_name,
                        "rows": rows_data,
                    }
                    sheet_list.append(sheet_entry)
                results = sheet_list
            except ImportError:
                logger.debug(
                    "pandas/openpyxl/xlrd library not installed. Skipping Excel table extraction."
                )
            except Exception as excel_err:
                logger.warning(f"Pandas Excel table extraction failed for {path.name}: {excel_err}")

        elif ext == ".csv":
            try:
                import pandas as pd  # Optional dependency

                # Read CSV, keep data as strings
                df = pd.read_csv(str(path), dtype=str)
                rows_data = df.to_dict(orient="records")
                # Create a list containing the single table representation
                results = [{"type": "csv_table", "rows": rows_data}]
            except ImportError:
                logger.debug("pandas library not installed. Skipping CSV table extraction.")
            except Exception as csv_err:
                logger.warning(f"Pandas CSV table extraction failed for {path.name}: {csv_err}")

    except Exception as outer_err:
        # Catch errors during import or setup
        logger.error(f"Error during table extraction setup for {path.name}: {outer_err}")

    return results


async def _extract_tables_async(path: Path) -> list:  # Uses global _log
    """Asynchronously extracts tables by running sync helper in thread pool."""
    try:
        # Run the synchronous extraction function in the thread pool
        tables = await asyncio.to_thread(_extract_tables_sync, path)
        if tables:
            num_tables = len(tables)
            await _log("table_extract_success", file=str(path), num_tables=num_tables)
        # Return the list of tables (or empty list if none found/error)
        return tables
    except Exception as e:
        # Log error during async execution/threading
        await _log("table_extract_error", file=str(path), error=str(e))
        return []  # Return empty list on error


@resilient()  # Keep the retry decorator if desired
async def smart_download(
    page: Page,
    task_hint: str,
    dest_dir: Optional[Union[str, Path]] = None,
    target_kwargs: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Initiates download via click, saves via Playwright, reads file directly
    for analysis (hash, tables), managing paths via FileSystem Tools.
    """
    final_dl_dir_path_str = "Unknown"  # For logging context, default value
    out_path: Optional[Path] = None  # Define earlier for clarity, default None

    # --- Determine and Prepare Download Directory using FileSystemTool ---
    try:
        # Determine the target directory path string
        if dest_dir:
            download_dir_path_str = str(dest_dir)
        else:
            # Default: Use a path relative to the allowed 'storage' base directory
            default_dl_subdir = "smart_browser_downloads"
            download_dir_path_str = f"storage/{default_dl_subdir}"

        logger.info(
            f"Ensuring download directory exists: '{download_dir_path_str}' using filesystem tool."
        )
        # Use STANDALONE create_directory tool
        create_dir_result = await create_directory(path=download_dir_path_str)

        # Validate the result from the filesystem tool
        if not isinstance(create_dir_result, dict) or not create_dir_result.get("success"):
            error_detail = "Invalid response"
            if isinstance(create_dir_result, dict):
                error_detail = create_dir_result.get("error", "Unknown")
            raise ToolError(
                f"Failed to prepare download directory '{download_dir_path_str}'. Filesystem tool error: {error_detail}"
            )

        # Use the actual absolute path returned by the tool
        final_dl_dir_path_str = create_dir_result.get(
            "path", download_dir_path_str
        )  # Use path from result, fallback to input
        final_dl_dir_path = Path(final_dl_dir_path_str)  # Convert to Path object for local use
        logger.info(f"Download directory confirmed/created at: {final_dl_dir_path}")

    except ToolError as e:
        logger.error(
            f"ToolError preparing download directory '{download_dir_path_str}': {e}", exc_info=True
        )
        raise  # Re-raise ToolError
    except Exception as e:
        # Catch any other unexpected errors during directory prep
        logger.error(
            f"Unexpected error preparing download directory '{download_dir_path_str}': {e}",
            exc_info=True,
        )
        raise ToolError(
            f"An unexpected error occurred preparing download directory: {str(e)}"
        ) from e
    # --- End Directory Preparation ---

    # Prepare log details
    log_target = {}
    if target_kwargs:
        log_target.update(target_kwargs)
    else:
        log_target["hint"] = task_hint

    try:
        # --- Initiate Download ---
        # Wait for the download event to occur after the click
        download_timeout_ms = 60000  # 60 seconds for download to start
        async with page.expect_download(timeout=download_timeout_ms) as dl_info:
            # Use the smart_click helper function to trigger the download
            click_timeout_ms = 10000  # 10 seconds for the click itself
            await smart_click(
                page, task_hint=task_hint, target_kwargs=target_kwargs, timeout_ms=click_timeout_ms
            )
            logger.debug(
                f"Click initiated for download hint: '{task_hint}'. Waiting for download start..."
            )

        # Get the Download object
        dl = await dl_info.value
        logger.info(
            f"Download started. Suggested filename: '{dl.suggested_filename}', URL: {dl.url}"
        )

        # Sanitize filename provided by browser
        suggested_fname_raw = dl.suggested_filename
        default_fname = f"download_{int(time.time())}.dat"
        suggested_fname = suggested_fname_raw or default_fname

        # Remove potentially harmful characters
        safe_fname_chars = re.sub(r"[^\w.\- ]", "_", suggested_fname)
        # Replace whitespace with underscores
        safe_fname_spaces = re.sub(r"\s+", "_", safe_fname_chars)
        # Remove leading/trailing problematic characters
        safe_fname_strip = safe_fname_spaces.strip("._-")
        # Ensure filename is not empty after sanitization
        safe_fname = safe_fname_strip or default_fname

        # --- Construct initial desired path (within the verified directory) ---
        initial_desired_path = final_dl_dir_path / safe_fname

        # --- Get Unique Path using FileSystemTool ---
        logger.debug(f"Requesting unique path based on initial suggestion: {initial_desired_path}")
        try:
            # Use STANDALONE get_unique_filepath tool
            unique_path_result = await get_unique_filepath(path=str(initial_desired_path))
            if not isinstance(unique_path_result, dict) or not unique_path_result.get("success"):
                error_detail = "Invalid response"
                if isinstance(unique_path_result, dict):
                    error_detail = unique_path_result.get("error", "Unknown")
                raise ToolError(
                    f"Failed to get unique download path. Filesystem tool error: {error_detail}"
                )

            final_unique_path_str = unique_path_result.get("path")
            if not final_unique_path_str:
                raise ToolError(
                    "Filesystem tool get_unique_filepath succeeded but did not return a path."
                )

            out_path = Path(final_unique_path_str)  # Use the unique path for saving
            logger.info(f"Determined unique download save path: {out_path}")

        except ToolError as e:
            logger.error(
                f"Error determining unique download path based on '{initial_desired_path}': {e}",
                exc_info=True,
            )
            raise  # Re-raise ToolError
        except Exception as e:
            logger.error(
                f"Unexpected error getting unique download path for '{initial_desired_path}': {e}",
                exc_info=True,
            )
            raise ToolError(
                f"An unexpected error occurred finding a unique save path: {str(e)}"
            ) from e
        # --- End Getting Unique Path ---

        # --- Save Download using Playwright ---
        logger.info(f"Playwright saving download from '{dl.url}' to unique path: {out_path}")
        # Playwright handles the actual streaming and saving to the specified path
        await dl.save_as(out_path)
        logger.info(f"Playwright download save complete: {out_path}")

        # --- Read back file DIRECTLY for Analysis (using out_path) ---
        file_data: Optional[bytes] = None
        file_size = -1
        sha256_hash = None
        read_back_error = None

        try:
            # Read the file content using our async helper
            logger.debug(f"Reading back downloaded file directly from {out_path} for analysis...")
            file_data = await _read_file_async(out_path)
            file_size = len(file_data)
            logger.debug(f"Successfully read back {file_size} bytes from {out_path} directly.")

        # Handle potential errors during the direct read-back
        except FileNotFoundError:
            read_back_error = f"Downloaded file {out_path} disappeared before read-back."
            # Optionally try to delete the potentially incomplete entry if FS allows
            # try: await delete_path(str(out_path)) # Needs delete_path tool
            # except Exception as del_e: logger.warning(f"Failed to cleanup missing file {out_path}: {del_e}")
        except IOError as e:
            read_back_error = f"IO error reading back downloaded file {out_path}: {e}"
        except Exception as e:
            read_back_error = f"Unexpected error reading back downloaded file {out_path}: {e}"
            # Log full traceback for unexpected errors
            logger.error(f"Unexpected error reading back {out_path}: {e}", exc_info=True)

        # If read-back failed, log and raise ToolError indicating partial success/failure
        if read_back_error:
            logger.error(read_back_error)
            # Prepare info about the failed read-back
            partial_info = {
                "success": False,  # Mark overall operation as failed due to analysis failure
                "file_path": str(out_path),
                "file_name": out_path.name,
                "error": f"Download saved, but failed to read back for analysis: {read_back_error}",
                "url": dl.url,
            }
            await _log("download_success_readback_fail", target=log_target, **partial_info)
            # Raise ToolError to signal failure clearly to the caller
            raise ToolError(partial_info["error"], details=partial_info)

        # --- Hashing and Table Extraction (if read-back succeeded) ---
        # Compute hash from the bytes read directly
        if file_data is not None:  # Should always be true if read_back_error is None
            sha256_hash = await _compute_hash_async(file_data)
            logger.debug(f"Computed SHA256 hash for {out_path.name}: {sha256_hash[:8]}...")
        else:
            # This case should technically not be reachable if read_back_error is None
            logger.error(
                f"Internal state error: file_data is None after successful read back for {out_path}."
            )
            # Fallback hash or handle as error? For now, hash will be None.

        tables = []
        # Check file extension to decide if table extraction is applicable
        file_extension = out_path.suffix.lower()
        is_table_extractable = file_extension in (".pdf", ".xls", ".xlsx", ".csv")

        if is_table_extractable:
            logger.debug(f"Attempting table extraction for {out_path.name}...")
            try:
                # Use the async helper which runs sync extraction in a thread
                # _extract_tables_async reads the file itself from out_path
                table_extraction_task = asyncio.create_task(_extract_tables_async(out_path))
                # Wait for extraction with a timeout
                extraction_timeout = 120  # seconds
                tables = await asyncio.wait_for(table_extraction_task, timeout=extraction_timeout)
                if tables:
                    logger.info(
                        f"Successfully extracted {len(tables)} table(s) from {out_path.name}"
                    )
                else:
                    logger.debug(f"No tables found or extracted from {out_path.name}")

            except asyncio.TimeoutError:
                logger.warning(
                    f"Table extraction timed out after {extraction_timeout}s for {out_path.name}"
                )
                # Ensure the task is cancelled if it timed out
                if "table_extraction_task" in locals() and not table_extraction_task.done():
                    table_extraction_task.cancel()
                    try:
                        # Give cancellation a moment to propagate (best effort)
                        await asyncio.wait_for(table_extraction_task, timeout=1.0)
                    except asyncio.CancelledError:
                        pass  # Expected outcome of cancellation
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Timeout waiting for table extraction task cancellation after initial timeout."
                        )
                    except Exception as cancel_err:
                        logger.warning(
                            f"Error during table extraction task cancellation: {cancel_err}"
                        )
                # Continue, tables will remain empty list
            except Exception as extract_err:
                # Catch other errors during extraction process
                logger.error(
                    f"Table extraction failed unexpectedly for {out_path.name}: {extract_err}",
                    exc_info=True,
                )
                # Continue, tables will remain empty list

        # --- Success ---
        # Prepare the success result dictionary
        info = {
            "success": True,
            "file_path": str(out_path),  # Return the final unique absolute path
            "file_name": out_path.name,
            "sha256": sha256_hash,  # Use the hash computed from read-back data
            "size_bytes": file_size,  # Use the size from read-back data
            "url": dl.url,  # URL the download originated from
            "tables_extracted": bool(tables),  # Indicate if tables were extracted
            "tables": tables[:5],  # Include a preview of first 5 tables (if any)
        }
        # Log success event (exclude large tables data from log details)
        log_info_safe = info.copy()
        if "tables" in log_info_safe:
            del log_info_safe["tables"]  # Remove tables for cleaner log
        log_info_safe["num_tables"] = len(tables) if tables else 0
        await _log("download_success", target=log_target, **log_info_safe)
        return info

    # --- Error Handling (Catch errors from download initiation or Playwright saving) ---
    except (ToolInputError, ToolError) as e:
        # These errors are raised explicitly above (e.g., dir prep, unique path, read-back) or by smart_click
        # Log the specific error type and message
        error_path_context = str(out_path) if out_path else "N/A"
        await _log("download_fail_other", target=log_target, error=str(e), path=error_path_context)
        raise  # Re-raise the specific ToolError/InputError
    except PlaywrightTimeoutError as e:
        # Timeout occurred during page.expect_download or within smart_click
        error_path_context = str(out_path) if out_path else "N/A"
        await _log(
            "download_fail_timeout", target=log_target, error=str(e), path=error_path_context
        )
        raise ToolError(f"Download operation timed out: {e}") from e
    except PlaywrightException as e:
        # Other playwright errors during expect_download, save_as, or smart_click
        error_path_context = str(out_path) if out_path else "N/A"
        await _log(
            "download_fail_playwright", target=log_target, error=str(e), path=error_path_context
        )
        raise ToolError(f"Download failed due to Playwright error: {e}") from e
    except Exception as e:
        # Catch-all for unexpected errors during the download process
        error_path_context = str(out_path) if out_path else "N/A"
        await _log(
            "download_fail_unexpected", target=log_target, error=str(e), path=error_path_context
        )
        logger.error(
            f"Unexpected error during smart_download for hint '{task_hint}': {e}", exc_info=True
        )  # Log traceback
        raise ToolError(f"Unexpected error during download: {e}") from e


# --- PDF/Docs Crawler Helpers (Keep as is, minor splits) ---
_SLUG_RE = re.compile(r"[^a-z0-9\-_]+")


def _slugify(text: str, max_len: int = 60) -> str:
    """Converts text to a URL-friendly slug."""
    if not text:
        return "file"  # Default slug for empty input

    # Normalize Unicode characters (e.g., accents to base letters)
    normalized_text = unicodedata.normalize("NFKD", text)
    # Encode to ASCII, ignoring characters that cannot be represented
    ascii_bytes = normalized_text.encode("ascii", "ignore")
    # Decode back to string
    ascii_text = ascii_bytes.decode()
    # Convert to lowercase
    lower_text = ascii_text.lower()
    # Replace non-alphanumeric (excluding '-', '_') with hyphens
    slug_hyphens = _SLUG_RE.sub("-", lower_text)
    # Remove leading/trailing hyphens
    slug_trimmed = slug_hyphens.strip("-")
    # Replace multiple consecutive hyphens with a single hyphen
    slug_single_hyphens = re.sub(r"-{2,}", "-", slug_trimmed)
    # Truncate to maximum length
    slug_truncated = slug_single_hyphens[:max_len]
    # Trim hyphens again after potential truncation
    final_slug = slug_truncated.strip("-")

    # Ensure slug is not empty after all operations
    return final_slug or "file"  # Return default if empty


def _get_dir_slug(url: str) -> str:
    """Creates a slug based on the last path components or domain of a URL."""
    try:
        parsed_url = urlparse(url)
        # Split path into components, filtering out empty strings and root slash
        path_obj = Path(parsed_url.path)
        path_parts = []
        for part in path_obj.parts:
            if part and part != "/":
                path_parts.append(part)

        # Create slug based on path components
        num_parts = len(path_parts)
        if num_parts >= 2:
            # Use last two path parts if available
            part_minus_2_slug = _slugify(path_parts[-2], 20)
            part_minus_1_slug = _slugify(path_parts[-1], 20)
            dir_slug = f"{part_minus_2_slug}-{part_minus_1_slug}"
            return dir_slug
        elif num_parts == 1:
            # Use the single path part
            part_slug = _slugify(path_parts[-1], 40)
            return part_slug
        else:
            # Fallback to domain name if path is empty or just '/'
            domain_slug = _slugify(parsed_url.netloc, 40)
            return domain_slug or "domain"  # Use 'domain' if netloc is also empty

    except Exception as e:
        logger.warning(f"Error creating directory slug for URL '{url}': {e}")
        return "path"  # Fallback slug on error


async def _fetch_html(
    client: httpx.AsyncClient, url: str, rate_limiter: Optional["RateLimiter"] = None
) -> Optional[str]:
    """Fetches HTML content from a URL using httpx, respecting rate limits."""
    try:
        # Acquire rate limit permit if limiter is provided
        if rate_limiter:
            await rate_limiter.acquire()

        # Make GET request with streaming response
        request_timeout = 20.0
        async with client.stream(
            "GET", url, follow_redirects=True, timeout=request_timeout
        ) as response:
            # Check for non-success status codes
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx

            # Handle No Content response
            if response.status_code == 204:
                logger.debug(f"Received HTTP 204 No Content for {url}")
                return None

            # Check content type - must be HTML
            content_type_header = response.headers.get("content-type", "")
            content_type = content_type_header.lower()
            if "text/html" not in content_type:
                logger.debug(f"Skipping non-HTML content type '{content_type}' for {url}")
                return None

            # Check content length limit
            max_html_size = 5 * 1024 * 1024  # 5 MiB
            content_length_header = response.headers.get("content-length")
            if content_length_header:
                try:
                    content_length = int(content_length_header)
                    if content_length > max_html_size:
                        logger.debug(
                            f"Skipping large HTML content ({content_length} bytes) for {url}"
                        )
                        return None
                except ValueError:
                    logger.warning(
                        f"Invalid Content-Length header '{content_length_header}' for {url}"
                    )
                    # Proceed cautiously without length check

            # Read the response body bytes
            html_bytes = await response.aread()

            # Decode HTML bytes to string (try UTF-8, then fallback)
            decoded_html: Optional[str] = None
            try:
                decoded_html = html_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    # Fallback to Latin-1 if UTF-8 fails
                    decoded_html = html_bytes.decode("iso-8859-1")
                    logger.debug(f"Decoded HTML from {url} using iso-8859-1 fallback.")
                except UnicodeDecodeError:
                    # Log warning if both decodings fail
                    logger.warning(f"Could not decode HTML from {url} using utf-8 or iso-8859-1.")
                    return None  # Cannot process undecodable content

            return decoded_html

    except httpx.HTTPStatusError as e:
        # Log client/server errors (4xx/5xx)
        status_code = e.response.status_code
        logger.debug(f"HTTP error {status_code} fetching {url}: {e}")
        return None
    except httpx.RequestError as e:
        # Log network-related errors (DNS, connection, timeout etc.)
        logger.warning(f"Network error fetching {url}: {e}")
        return None
    except Exception as e:
        # Log other unexpected errors during fetch
        logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
        return None


def _extract_links(base_url: str, html: str) -> Tuple[List[str], List[str]]:
    """Extracts absolute PDF and internal HTML page links from HTML content."""
    pdfs: Set[str] = set()
    pages: Set[str] = set()
    try:
        soup = BeautifulSoup(html, "html.parser")  # Use default parser
        parsed_base_url = urlparse(base_url)
        base_netloc = parsed_base_url.netloc

        # Find all <a> tags with an href attribute
        anchor_tags = soup.find_all("a", href=True)

        for a in anchor_tags:
            href_raw = a["href"]
            # Skip empty, fragment, mailto, tel, or javascript links
            if not href_raw or href_raw.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue

            try:
                # Resolve relative URLs to absolute URLs
                abs_url = urllib.parse.urljoin(base_url, href_raw)
                parsed_url = urlparse(abs_url)

                # Clean URL by removing fragment identifier
                clean_url = parsed_url._replace(fragment="").geturl()
                path_lower = parsed_url.path.lower()

                # Check if it's a PDF link
                if path_lower.endswith(".pdf"):
                    pdfs.add(clean_url)
                # Check if it's an internal HTML page link
                elif parsed_url.netloc == base_netloc:
                    # Check if path seems like HTML or directory listing
                    is_html_like = path_lower.endswith((".html", ".htm", "/"))
                    # Or if it has no file extension in the last path segment
                    path_name = Path(parsed_url.path).name
                    has_no_ext = "." not in path_name
                    # Ensure it's not mistakenly identified as PDF again
                    not_pdf = not path_lower.endswith(".pdf")

                    if (is_html_like or has_no_ext) and not_pdf:
                        pages.add(clean_url)

            except ValueError:
                # Ignore errors resolving invalid URLs (e.g., bad characters)
                pass
            except Exception as link_err:
                # Log other errors during link processing
                logger.warning(f"Error processing link '{href_raw}' on page {base_url}: {link_err}")

    except Exception as soup_err:
        # Log errors during BeautifulSoup parsing
        logger.error(f"Error parsing HTML for links on {base_url}: {soup_err}", exc_info=True)

    # Return lists of unique PDF and page URLs found
    return list(pdfs), list(pages)


class RateLimiter:  # Keep class definition
    """Simple asynchronous rate limiter using asyncio.Lock."""

    def __init__(self, rate_limit: float = 1.0):
        if rate_limit <= 0:
            raise ValueError("Rate limit must be positive.")
        # Calculate the minimum interval between requests in seconds
        self.interval = 1.0 / rate_limit
        self.last_request_time: float = 0  # Time of the last request completion
        self.lock = asyncio.Lock()  # Lock to ensure atomic check/wait/update

    async def acquire(self):
        """Acquires a permit, sleeping if necessary to maintain the rate limit."""
        async with self.lock:
            now = time.monotonic()
            time_since_last = now - self.last_request_time
            # Calculate how long we need to wait
            time_to_wait = self.interval - time_since_last

            if time_to_wait > 0:
                # Sleep for the required duration
                await asyncio.sleep(time_to_wait)
                # Update 'now' after sleeping
                now = time.monotonic()

            # Update the last request time to the current time
            self.last_request_time = now


async def crawl_for_pdfs(
    start_url: str,
    include_regex: Optional[str] = None,
    max_depth: int = 2,
    max_pdfs: int = 100,
    max_pages_crawl: int = 500,
    rate_limit_rps: float = 2.0,
) -> List[str]:
    """Crawls a website to find PDF links."""
    # Compile include regex if provided
    inc_re: Optional[re.Pattern] = None
    if include_regex:
        try:
            inc_re = re.compile(include_regex, re.IGNORECASE)
        except re.error as e:
            raise ToolInputError(f"Invalid include_regex provided: {e}") from e

    # Initialize crawl state
    seen_urls: Set[str] = set()
    pdf_urls_found: Set[str] = set()
    # Queue stores tuples of (url, depth)
    queue: deque[tuple[str, int]] = deque()
    queue.append((start_url, 0))  # Start at depth 0
    seen_urls.add(start_url)
    visit_count = 0
    rate_limiter = RateLimiter(rate_limit_rps)
    base_netloc = urlparse(start_url).netloc
    # Basic user agent for politeness
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SmartBrowserBot/1.0; +http://example.com/bot)"
    }

    # Use httpx.AsyncClient for connection pooling
    client_timeout = 30.0
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=client_timeout, headers=headers
    ) as client:
        # Main crawl loop
        while queue:
            # Check stopping conditions
            if len(pdf_urls_found) >= max_pdfs:
                logger.info(f"PDF crawl stopped: Max PDFs ({max_pdfs}) reached.")
                break
            if visit_count >= max_pages_crawl:
                logger.warning(f"PDF crawl stopped: Max pages crawled ({max_pages_crawl}) reached.")
                break

            # Get next URL and depth from queue
            current_url, current_depth = queue.popleft()
            visit_count += 1
            logger.debug(f"Crawling [Depth {current_depth}, Visit {visit_count}]: {current_url}")

            # Fetch HTML content for the current page
            html = await _fetch_html(client, current_url, rate_limiter)
            if not html:
                continue  # Skip if fetch failed or not HTML

            # Extract links from the fetched HTML
            pdfs, pages = _extract_links(current_url, html)

            # Process found PDF links
            for pdf_url in pdfs:
                if pdf_url not in pdf_urls_found:
                    # Apply include regex if specified
                    if inc_re is None or inc_re.search(pdf_url):
                        pdf_urls_found.add(pdf_url)
                        logger.info(f"PDF found: {pdf_url} (Total: {len(pdf_urls_found)})")
                        # Check if max PDFs reached after adding
                        if len(pdf_urls_found) >= max_pdfs:
                            break  # Exit inner loop

            # Check max PDFs again after processing all PDFs on page
            if len(pdf_urls_found) >= max_pdfs:
                break  # Exit outer loop

            # Process found HTML page links for further crawling
            if current_depth < max_depth:
                for page_url in pages:
                    try:
                        parsed_page_url = urlparse(page_url)
                        # Only crawl pages on the same domain and not seen before
                        is_same_domain = parsed_page_url.netloc == base_netloc
                        is_not_seen = page_url not in seen_urls
                        if is_same_domain and is_not_seen:
                            seen_urls.add(page_url)
                            # Add to queue with incremented depth
                            queue.append((page_url, current_depth + 1))
                    except ValueError:
                        # Ignore errors parsing potential page URLs
                        pass

    # Log final counts after loop finishes
    logger.info(
        f"PDF crawl finished. Found {len(pdf_urls_found)} matching PDFs after visiting {visit_count} pages."
    )
    return list(pdf_urls_found)


async def _download_file_direct(
    url: str, dest_dir_str: str, seq: int = 1
) -> Dict:  # Uses Filesystem Tools
    """Downloads a file directly using httpx and saves using filesystem tools."""
    final_output_path_str: Optional[str] = None  # Path where file is ultimately saved
    downloaded_content: Optional[bytes] = None
    initial_filename = ""  # Keep track for error reporting

    try:
        # --- Determine Initial Filename ---
        parsed_url = urlparse(url)
        path_basename = os.path.basename(parsed_url.path) if parsed_url.path else ""

        # Create a filename if URL path is empty or root, or has no extension
        use_generated_name = not path_basename or path_basename == "/" or "." not in path_basename

        if use_generated_name:
            dir_slug = _get_dir_slug(url)  # Slug based on parent path or domain
            base_name = f"{seq:03d}_{dir_slug}_{_slugify(path_basename or 'download')}"
            # Add appropriate extension (default .dat)
            file_ext = ".pdf" if url.lower().endswith(".pdf") else ".dat"
            initial_filename = base_name + file_ext
        else:
            # Use and sanitize the filename from the URL path
            sanitized_basename = _slugify(path_basename)
            initial_filename = f"{seq:03d}_{sanitized_basename}"

        # Initial desired path within the destination directory
        initial_desired_path = os.path.join(dest_dir_str, initial_filename)
        refined_desired_path = initial_desired_path  # Start with initial path

        # --- Fetch File Content ---
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",  # Standard UA
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        download_timeout = 120.0  # Allow 2 minutes for download
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=download_timeout, headers=headers
        ) as client:
            async with client.stream("GET", url) as response:
                # Check for successful status code
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code} {response.reason_phrase}"
                    status_code = response.status_code
                    # Return error dictionary immediately
                    return {
                        "url": url,
                        "error": error_msg,
                        "status_code": status_code,
                        "success": False,
                        "path": initial_desired_path,  # Report intended path on error
                    }

                # --- Refine Filename based on Headers (Content-Disposition, Content-Type) ---
                # Check Content-Disposition header for filename suggestion
                content_disposition = response.headers.get("content-disposition")
                if content_disposition:
                    # Simple regex to find filename*= or filename=
                    match = re.search(r'filename\*?="?([^"]+)"?', content_disposition)
                    if match:
                        header_filename_raw = match.group(1)
                        # Try URL decoding potential encoding
                        try:
                            header_filename_decoded = urllib.parse.unquote(header_filename_raw)
                        except Exception:
                            header_filename_decoded = header_filename_raw  # Fallback
                        # Sanitize and prepend sequence number
                        refined_filename = f"{seq:03d}_{_slugify(header_filename_decoded)}"
                        refined_desired_path = os.path.join(dest_dir_str, refined_filename)
                        logger.debug(
                            f"Refined filename from Content-Disposition: {refined_filename}"
                        )

                # Check Content-Type header to potentially correct extension
                content_type_header = response.headers.get("content-type", "")
                content_type = content_type_header.split(";")[0].strip().lower()
                current_stem, current_ext = os.path.splitext(refined_desired_path)
                # Correct extension if Content-Type is PDF and current ext isn't
                if content_type == "application/pdf" and current_ext.lower() != ".pdf":
                    refined_desired_path = current_stem + ".pdf"
                    logger.debug("Corrected file extension to .pdf based on Content-Type.")

                # Read the downloaded content
                downloaded_content = await response.aread()
                bytes_read = len(downloaded_content)
                logger.debug(f"Downloaded {bytes_read} bytes for {url}.")

        # Ensure content was downloaded
        if downloaded_content is None:
            raise ToolError(
                "Downloaded content is unexpectedly None after successful HTTP request."
            )

        # --- Get Unique Save Path using Filesystem Tool ---
        try:
            unique_path_result = await get_unique_filepath(
                path=refined_desired_path
            )  # STANDALONE call
            if not isinstance(unique_path_result, dict) or not unique_path_result.get("success"):
                error_msg = (
                    unique_path_result.get("error", "Unknown")
                    if isinstance(unique_path_result, dict)
                    else "Invalid response"
                )
                raise ToolError(f"Failed to get unique download path. Error: {error_msg}")

            final_output_path_str = unique_path_result.get("path")
            if not final_output_path_str:
                raise ToolError(
                    "Filesystem tool get_unique_filepath succeeded but did not return path."
                )
            logger.info(f"Determined unique download save path: {final_output_path_str}")
        except Exception as e:
            # Wrap error getting unique path
            raise ToolError(
                f"Could not determine unique save path based on '{refined_desired_path}': {str(e)}"
            ) from e

        # --- Write File using Filesystem Tool ---
        try:
            write_result = await write_file(
                path=final_output_path_str, content=downloaded_content
            )  # STANDALONE call
            if not isinstance(write_result, dict) or not write_result.get("success"):
                error_msg = (
                    write_result.get("error", "Unknown")
                    if isinstance(write_result, dict)
                    else "Invalid response"
                )
                raise ToolError(
                    f"Filesystem tool failed to write downloaded file to '{final_output_path_str}'. Error: {error_msg}"
                )
            logger.info(f"Successfully saved file to: {final_output_path_str}")
        except Exception as e:
            # Wrap error during file write
            raise ToolError(
                f"Could not write downloaded file to '{final_output_path_str}': {str(e)}"
            ) from e

        # --- Calculate Hash ---
        hasher = hashlib.sha256()
        hasher.update(downloaded_content)
        file_hash = hasher.hexdigest()

        # --- Log and Return Success ---
        await _log(
            "download_direct_success",
            url=url,
            file=final_output_path_str,
            size=bytes_read,
            sha256=file_hash,
        )
        return {
            "url": url,
            "file": final_output_path_str,  # The actual saved path
            "size": bytes_read,
            "sha256": file_hash,
            "success": True,
        }

    except httpx.RequestError as e:
        # Handle network errors during download attempt
        logger.warning(f"Network error downloading {url}: {e}")
        return {
            "url": url,
            "error": f"Network error: {e}",
            "success": False,
            "path": final_output_path_str or initial_filename,
        }  # Report final path if available
    except (ToolError, ToolInputError) as e:
        # Handle errors raised explicitly during path/write operations
        logger.error(f"Tool error downloading {url} directly: {e}", exc_info=True)
        return {
            "url": url,
            "error": f"Download failed: {e}",
            "success": False,
            "path": final_output_path_str or initial_filename,
        }
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error downloading {url} directly: {e}", exc_info=True)
        return {
            "url": url,
            "error": f"Download failed unexpectedly: {e}",
            "success": False,
            "path": final_output_path_str or initial_filename,
        }


# --- OSS Documentation Crawler Helpers ---
_DOC_EXTS = (".html", ".htm", "/")  # Common extensions/endings for HTML pages
_DOC_STOP_PAT = re.compile(
    r"\.(png|jpg|jpeg|gif|svg|css|js|zip|tgz|gz|whl|exe|dmg|ico|woff|woff2|map|json|xml|txt|pdf|md)$",  # Added pdf, md
    re.IGNORECASE,
)  # File extensions to ignore during crawl


def _looks_like_docs_url(url: str) -> bool:
    """
    Heuristically checks if a URL looks like a documentation page.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL appears to be a documentation page, False otherwise.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        url_low = url.lower()
        parsed = urllib.parse.urlparse(url_low)

        # 1. Penalize URLs with query strings (often dynamic/non-doc pages)
        if parsed.query:
            return False

        # 2. Penalize common non-doc paths explicitly
        common_non_doc_paths = [
            # Common application paths
            "/api/",  # Sometimes docs, but often API endpoints themselves
            "/blog/",
            "/news/",
            "/community/",
            "/forum/",
            "/support/",
            "/contact/",
            "/about/",
            "/pricing/",
            "/login/",
            "/register/",
            "/signup/",
            "/signin/",
            "/account/",
            "/profile/",
            "/cart/",
            "/checkout/",
            # Common asset/download paths
            "/download/",
            "/install/",
            "/_static/",
            "/_images/",
            "/assets/",
            "/media/",
            "/static/",
            "/vendor/",
            "/node_modules/",
            # Specific framework/site paths unlikely to be main docs
            "/wp-content/",
            "/wp-admin/",
            "/sites/default/files/",
        ]
        # Use a generator expression for slightly better efficiency
        if any(non_doc_path in parsed.path for non_doc_path in common_non_doc_paths):
            return False

        # 3. Check for keywords indicating documentation in URL or path
        doc_keywords = [
            "docs",
            "doc",
            "documentation",
            "guide",
            "manual",
            "tutorial",
            "tuto",
            "reference",
            "ref",
            "api",
            "faq",
            "howto",
            "userguide",
            "develop",
            "example",
            "usage",
            "getting-started",
            "quickstart",
        ]
        # Check in netloc (e.g., docs.example.com) and path
        has_doc_keyword = any(
            keyword in parsed.netloc or keyword in parsed.path for keyword in doc_keywords
        )

        # 4. Check if URL ends with typical HTML extension or directory slash
        ends_with_doc_ext = url_low.endswith(_DOC_EXTS)

        # 5. Check if URL is hosted on a common documentation platform
        common_doc_hosts = [
            "readthedocs.io",
            "netlify.app",
            "vercel.app",
            "github.io",
            "gitlab.io",
            "pages.dev",  # Cloudflare Pages
            "gitbook.io",
            "docusaurus.io",  # Often custom domains, but sometimes subdomains
        ]
        is_common_host = any(host in parsed.netloc for host in common_doc_hosts)

        # 6. Check if URL path contains a file extension we want to stop at
        path_has_stop_ext = bool(_DOC_STOP_PAT.search(parsed.path))

        # Combine checks:
        # - MUST NOT have a stop extension
        # - MUST satisfy one of the positive indicators:
        #   - Contains a documentation keyword
        #   - Ends like an HTML page or directory
        #   - Is hosted on a common documentation platform
        is_likely_doc = not path_has_stop_ext and (
            has_doc_keyword or ends_with_doc_ext or is_common_host
        )

        # Log decision process if debugging needed
        # logger.debug(f"URL Check: {url_low} -> StopExt:{path_has_stop_ext}, Keyword:{has_doc_keyword}, DocExt:{ends_with_doc_ext}, CommonHost:{is_common_host} => LikelyDoc:{is_likely_doc}")

        return is_likely_doc

    except ValueError:  # Handle potential errors from urlparse
        logger.warning(f"Error parsing URL for documentation check: {url}", exc_info=True)
        return False
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"Unexpected error in _looks_like_docs_url for {url}: {e}", exc_info=True)
        return False


async def _pick_docs_root(pkg_name: str) -> Optional[str]:
    """
    Attempts to find the root documentation URL for a package using web search.

    Uses multiple search queries and engines, then applies heuristics (_looks_like_docs_url)
    to find the most likely documentation root URL.

    Args:
        pkg_name: The name of the package to find documentation for.

    Returns:
        The most likely documentation root URL as a string, or None if not found.

    Raises:
        ToolInputError: If the package name is invalid.
        ToolError: If the web search fails critically or no suitable URL is found.
    """
    if not pkg_name or not isinstance(pkg_name, str):
        raise ToolInputError("Package name must be a non-empty string.")

    try:
        logger.info(f"Searching for documentation root for package: '{pkg_name}'")

        # --- Prepare search queries and engines ---
        queries = [
            f'"{pkg_name}" official documentation website',  # More precise
            f"{pkg_name} documentation",
            f"{pkg_name} python library docs",  # Specific to python
            f"{pkg_name} user guide",
            f"how to use {pkg_name}",
        ]
        # Cycle engines to mitigate potential blocks/bias or differing results
        engines = ["duckduckgo", "bing"]
        all_search_hits: List[Dict[str, Any]] = []
        MAX_RESULTS_PER_QUERY = 3  # Get fewer results per query, but run more queries

        # --- Run searches ---
        for i, query in enumerate(queries):
            engine = engines[i % len(engines)]
            logger.debug(f"Trying search query [{i + 1}/{len(queries)}]: '{query}' on {engine}")
            try:
                await asyncio.sleep(0.2)  # Small delay between searches
                # Assuming search_web returns a list of dicts directly now
                search_res_list = await search_web(
                    query, engine=engine, max_results=MAX_RESULTS_PER_QUERY
                )
                if isinstance(search_res_list, list):
                    all_search_hits.extend(search_res_list)
                else:
                    # Log if search_web returns unexpected format (it shouldn't based on its definition)
                    logger.warning(
                        f"Search query '{query}' on {engine} returned unexpected format: {type(search_res_list)}. Expected list."
                    )

            except ToolError as search_err:
                # Log specific tool errors from search_web but continue trying other queries
                logger.warning(f"Web search query '{query}' failed on {engine}: {search_err}")
            except Exception as e:
                # Log unexpected errors during a specific search call but continue
                logger.error(
                    f"Unexpected error during web search for query '{query}': {e}", exc_info=True
                )

        # Check if any results were gathered at all
        if not all_search_hits:
            raise ToolError(
                f"Web search yielded no results for documentation queries related to '{pkg_name}'."
            )

        # --- Evaluate results ---
        logger.debug(
            f"Evaluating {len(all_search_hits)} potential documentation URLs for '{pkg_name}'."
        )
        best_candidate: Optional[str] = None
        candidate_urls_considered: Set[str] = set()

        for i, hit in enumerate(all_search_hits):  # Add index for logging
            url = hit.get("url")
            title = hit.get("title", "N/A")  # Get title for context
            logger.debug(
                f"  Hit [{i + 1}/{len(all_search_hits)}]: URL='{url}', Title='{title}'"
            )  # Log the hit being processed

            if not url:
                logger.debug("    -> Skipping hit (no URL)")
                continue

            # Basic URL cleaning: normalize scheme, netloc, path; remove fragment
            try:
                parsed_hit = urllib.parse.urlparse(url)
                # Remove www. prefix for easier comparison
                cleaned_netloc = parsed_hit.netloc.lower().replace("www.", "")
                # Reconstruct URL without fragment, using cleaned netloc
                cleaned_url = parsed_hit._replace(fragment="", netloc=cleaned_netloc).geturl()

                # Ensure URL is not already processed (avoids redundant checks)
                if cleaned_url in candidate_urls_considered:
                    logger.debug(f"    -> Skipping hit (already considered: {cleaned_url})")
                    continue
                candidate_urls_considered.add(cleaned_url)

            except ValueError:
                # Handle potential errors during URL parsing
                logger.warning(f"    -> Skipping hit (invalid URL): {url}")
                continue

            # Apply the heuristic check (_looks_like_docs_url assumes it's defined elsewhere)
            is_likely = _looks_like_docs_url(cleaned_url)
            logger.debug(
                f"    -> Heuristic check for '{cleaned_url}': {is_likely}"
            )  # Log heuristic result

            if is_likely:
                logger.info(
                    f"Found likely documentation page via search: {cleaned_url} (Original: {url})"
                )
                # Simple strategy: take the *first* likely candidate found.
                best_candidate = cleaned_url
                break  # Stop after finding the first likely candidate

        # --- Fallback if heuristic finds nothing ---
        if not best_candidate and all_search_hits:
            # Fallback: Take the first result URL, clean it, and hope for the best.
            first_url_original = all_search_hits[0].get("url")
            if first_url_original:
                try:
                    parsed_first = urllib.parse.urlparse(first_url_original)
                    # Perform the same cleaning as above for consistency
                    cleaned_first_netloc = parsed_first.netloc.lower().replace("www.", "")
                    cleaned_first_url = parsed_first._replace(
                        fragment="", netloc=cleaned_first_netloc
                    ).geturl()
                    logger.warning(
                        f"_looks_like_docs_url heuristic failed. Falling back to first search result: {cleaned_first_url}"
                    )
                    best_candidate = cleaned_first_url
                except ValueError:
                    logger.error(f"Could not parse fallback first URL: {first_url_original}")
                    # best_candidate remains None, error will be raised below

        # --- Final Check and Root Derivation ---
        if not best_candidate:
            logger.error(
                f"Could not find any suitable documentation URL for '{pkg_name}' after evaluating {len(candidate_urls_considered)} candidates."
            )
            # Optionally log considered URLs if helpful for debugging
            # logger.debug(f"Considered URLs: {candidate_urls_considered}")
            raise ToolError(
                f"Could not automatically find a likely documentation site for package '{pkg_name}'. Web search did not yield a suitable URL."
            )

        # Try to derive a more "root" URL from the best candidate found
        final_root_url: str
        try:
            parsed_candidate = urllib.parse.urlparse(best_candidate)
            path_segments = [seg for seg in parsed_candidate.path.split("/") if seg]

            # If the path has multiple segments, try going up one level
            # Only do this if the parent path still looks like documentation
            if len(path_segments) > 1:
                parent_path = "/".join(path_segments[:-1])
                # Ensure trailing slash for derived root URL, clear query/fragment
                root_derived = parsed_candidate._replace(
                    path=f"/{parent_path}/", query="", fragment=""
                ).geturl()

                # Check if the derived parent path still looks like docs
                if _looks_like_docs_url(root_derived):
                    logger.info(
                        f"Derived potential docs root by going up one level: {root_derived}"
                    )
                    final_root_url = root_derived
                else:
                    # Parent doesn't look like docs, stick with the cleaned candidate URL
                    final_root_url = parsed_candidate._replace(query="", fragment="").geturl()
                    logger.info(
                        f"Parent path '{parent_path}/' didn't seem like docs root. Using original candidate (cleaned): {final_root_url}"
                    )
            else:
                # Only one path segment or root path, use the cleaned candidate URL as is
                final_root_url = parsed_candidate._replace(query="", fragment="").geturl()
                logger.info(
                    f"Candidate URL is shallow or root. Using cleaned candidate as root: {final_root_url}"
                )

        except Exception as parse_err:
            # Handle errors during parsing or root derivation
            logger.warning(
                f"Error parsing/deriving root from best candidate URL {best_candidate}: {parse_err}. Using candidate as is (cleaned)."
            )
            # Fallback: Clean the best candidate URL (remove query/fragment) and return it
            try:
                parsed_fallback = urllib.parse.urlparse(best_candidate)
                final_root_url = parsed_fallback._replace(query="", fragment="").geturl()
            except ValueError:
                # Should not happen if best_candidate was parseable before, but handle defensively
                logger.error(
                    f"Failed to parse even the fallback candidate {best_candidate}. Returning original candidate."
                )
                final_root_url = best_candidate  # Last resort

        return final_root_url

    # Note: ToolError is raised explicitly above if no candidate found or web search fails.
    # This catch block handles unexpected errors during the process.
    except Exception as e:
        logger.error(
            f"Unexpected error finding documentation root for '{pkg_name}': {e}", exc_info=True
        )
        # Raise a generic ToolError indicating the failure cause
        raise ToolError(
            f"An unexpected error occurred while finding documentation for '{pkg_name}': {str(e)}"
        ) from e


# Import optional libraries for summarization, handle missing imports
try:
    import trafilatura
except ImportError:
    trafilatura = None
    logger.debug("trafilatura library not found, summarization quality may be reduced.")
try:
    from readability import Document  # Using python-readability (lxml based)
except ImportError:
    Document = None
    logger.debug("readability-lxml library not found, summarization quality may be reduced.")


def _summarize_html_sync(html: str, max_len: int = 10000) -> str:
    """Synchronously extracts main text content from HTML using multiple libraries."""
    if not html:
        return ""

    # Limit input HTML size to prevent excessive memory/CPU usage
    MAX_HTML_SIZE = 3 * 1024 * 1024  # 3 MiB
    if len(html) > MAX_HTML_SIZE:
        logger.warning(f"HTML content truncated to {MAX_HTML_SIZE} bytes for summarization.")
        html = html[:MAX_HTML_SIZE]

    text = ""

    # 1. Try Trafilatura (often good for articles/main content)
    if trafilatura is not None:
        try:
            # Favor precision over recall, exclude comments/tables
            extracted = trafilatura.extract(
                html, include_comments=False, include_tables=False, favor_precision=True
            )
            if (
                extracted and len(extracted) > 100
            ):  # Basic check if extraction yielded substantial text
                text = extracted
                logger.debug("Summarized HTML using Trafilatura.")
        except Exception as e:
            logger.warning(f"Trafilatura failed during HTML summarization: {e}")
            # Continue to next method if it fails

    # 2. Try Readability-lxml if Trafilatura failed or yielded short text
    if (not text or len(text) < 200) and Document is not None:
        try:
            doc = Document(html)
            # Get summary HTML (main content block)
            summary_html = doc.summary(html_partial=True)
            # Parse the summary HTML and extract text
            soup = BeautifulSoup(
                summary_html, "html.parser"
            )  # Use html.parser for potentially partial HTML
            extracted_text = soup.get_text(" ", strip=True)
            if extracted_text and len(extracted_text) > 50:  # Lower threshold for readability
                text = extracted_text
                logger.debug("Summarized HTML using Readability-lxml.")
        except Exception as e:
            logger.warning(f"Readability-lxml failed during HTML summarization: {e}")
            # Continue to fallback if it fails

    # 3. Fallback: BeautifulSoup basic text extraction (if others failed/short)
    if not text or len(text) < 100:
        logger.debug("Using BeautifulSoup fallback for HTML summarization.")
        try:
            soup = BeautifulSoup(html, "lxml")  # Use lxml for robustness
            # Remove common non-content tags before text extraction
            tags_to_remove = [
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "aside",
                "form",
                "figure",
                "figcaption",
                "noscript",
            ]
            found_tags = soup(tags_to_remove)
            for tag in found_tags:
                tag.decompose()
            # Get remaining text, join with spaces, strip extra whitespace
            extracted_text = soup.get_text(" ", strip=True)
            text = extracted_text  # Use BS result even if short
        except Exception as e:
            logger.warning(f"BeautifulSoup fallback failed during HTML summarization: {e}")
            # text might remain empty if BS also fails

    # Final cleanup: normalize whitespace and truncate
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    final_text = cleaned_text[:max_len]
    return final_text


async def _grab_readable(
    client: httpx.AsyncClient, url: str, rate_limiter: RateLimiter
) -> Optional[str]:
    """Fetches HTML and extracts readable text content asynchronously."""
    # Fetch HTML using the helper function
    html = await _fetch_html(client, url, rate_limiter)
    if html:
        # Run the synchronous summarization function in the thread pool
        readable_text = await _run_in_thread(_summarize_html_sync, html)
        return readable_text
    else:
        # Return None if HTML fetch failed
        return None


async def crawl_docs_site(
    root_url: str, max_pages: int = 40, rate_limit_rps: float = 3.0
) -> List[Tuple[str, str]]:
    """Crawls a documentation site starting from root_url and extracts readable text."""
    # Validate root URL and get starting domain
    try:
        parsed_start_url = urlparse(root_url)
        start_netloc = parsed_start_url.netloc
        if not start_netloc:
            raise ValueError("Root URL must have a valid domain name.")
    except (ValueError, AssertionError) as e:
        raise ToolInputError(
            f"Invalid root URL provided for documentation crawl: '{root_url}'. Error: {e}"
        ) from e

    # Initialize crawl state
    seen_urls: Set[str] = set()
    queue: deque[str] = deque()
    queue.append(root_url)  # Start with the root URL
    seen_urls.add(root_url)
    # List to store tuples of (url, extracted_text)
    output_pages: List[Tuple[str, str]] = []
    visit_count = 0
    # Set a max number of visits to prevent infinite loops on large/cyclic sites
    max_visits = max(max_pages * 5, 200)  # Visit more URLs than pages needed
    rate_limiter = RateLimiter(rate_limit_rps)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SmartBrowserDocBot/1.0)"}
    logger.info(
        f"Starting documentation crawl from: {root_url} (Max pages: {max_pages}, Max visits: {max_visits})"
    )

    # Use httpx.AsyncClient for connection pooling
    client_timeout = 30.0
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=client_timeout, headers=headers
    ) as client:
        # Main crawl loop
        while queue:
            # Check stopping conditions
            if len(output_pages) >= max_pages:
                logger.info(f"Doc crawl stopped: Reached max pages ({max_pages}).")
                break
            if visit_count >= max_visits:
                logger.warning(f"Doc crawl stopped: Reached max visits ({max_visits}).")
                break

            # Get next URL from queue
            current_url = queue.popleft()
            visit_count += 1
            logger.debug(
                f"Doc Crawl [Visit {visit_count}/{max_visits}, Found {len(output_pages)}/{max_pages}]: {current_url}"
            )

            # Grab readable text content from the URL
            readable_text = await _grab_readable(client, current_url, rate_limiter)

            # If readable text was extracted, add it to results
            if readable_text:
                output_pages.append((current_url, readable_text))
                logger.debug(
                    f"Collected readable content from: {current_url} (Length: {len(readable_text)})"
                )

                # Check if max pages reached after adding
                if len(output_pages) >= max_pages:
                    break  # Exit loop early

                # Fetch HTML again (or reuse if cached) to extract links for further crawling
                # Re-fetching ensures we get links even if _grab_readable modified/simplified HTML structure
                # (Could potentially optimize by passing HTML between functions if summarizer doesn't modify structure needed for links)
                html_for_links = await _fetch_html(client, current_url, rate_limiter)
                if html_for_links:
                    _, page_links = _extract_links(current_url, html_for_links)
                    # Process found page links
                    for link_url in page_links:
                        try:
                            parsed_link = urlparse(link_url)
                            # Check if link is on the same domain
                            is_same_domain = parsed_link.netloc == start_netloc
                            # Check if it looks like a doc page we haven't seen
                            is_doc_link = _looks_like_docs_url(link_url)
                            is_not_seen = link_url not in seen_urls

                            if is_same_domain and is_doc_link and is_not_seen:
                                seen_urls.add(link_url)
                                queue.append(link_url)  # Add to crawl queue
                        except ValueError:
                            # Ignore errors parsing potential link URLs
                            pass
            else:
                logger.debug(f"No readable content extracted from: {current_url}")

    # Log final results after loop finishes
    logger.info(
        f"Documentation crawl finished. Collected content from {len(output_pages)} pages after {visit_count} visits."
    )
    return output_pages


# --- Page State Extraction ---
async def get_page_state(
    page: Page, max_elements: Optional[int] = None
) -> dict[str, Any]:  # Uses global _log
    """Extracts the current state of the page using the page map functionality."""
    if max_elements is not None:
        # Note: _max_widgets_global now controls element count in _build_page_map
        logger.warning(
            "get_page_state 'max_elements' argument is deprecated and has no effect. Use global config 'max_widgets' instead."
        )

    # Check if page is valid
    if not page or page.is_closed():
        logger.warning("get_page_state called on closed or invalid page.")
        return {
            "error": "Page is closed or invalid",
            "url": getattr(page, "url", "unknown"),  # Try to get URL even if closed
            "title": "[Error: Page Closed]",
            "elements": [],
            "main_text": "",
        }

    start_time = time.monotonic()
    try:
        # Use the helper function to build (or retrieve cached) page map
        page_map, fingerprint = await _build_page_map(page)
        duration = time.monotonic() - start_time
        duration_ms = int(duration * 1000)
        num_elements = len(page_map.get("elements", []))
        page_url = page_map.get("url")
        page_title = page_map.get("title")

        # Log successful extraction
        await _log(
            "page_state_extracted",
            url=page_url,
            title=page_title,
            duration_ms=duration_ms,
            num_elements=num_elements,
            fp=fingerprint[:8],
        )

        # Return the constructed page map
        return page_map

    except Exception as e:
        # Catch any unexpected errors during state extraction
        duration = time.monotonic() - start_time
        duration_ms = int(duration * 1000)
        page_url = page.url or "unknown"  # Get URL directly from page on error
        logger.error(f"Error getting page state for {page_url}: {e}", exc_info=True)
        # Log error event
        await _log(
            "page_error", action="get_state", url=page_url, error=str(e), duration_ms=duration_ms
        )
        # Return error structure
        return {
            "error": f"Failed to get page state: {e}",
            "url": page_url,
            "title": "[Error Getting State]",
            "elements": [],
            "main_text": "",
        }


# --- LLM Bridge ---
def _extract_json_block(text: str) -> Optional[str]:  # Keep as is
    """Extracts the first JSON code block (markdown or bare) from text."""
    # Try finding markdown code block first ```json ... ```
    pattern_md = r"```json\s*(\{.*\}|\[.*\])\s*```"
    match_markdown = re.search(pattern_md, text, re.DOTALL)
    if match_markdown:
        json_str = match_markdown.group(1).strip()
        return json_str

    # Try finding bare JSON object or array { ... } or [ ... ]
    # This is less reliable, might match partial structures
    pattern_bare = r"(\{.*\}|\[.*\])"
    match_bare = re.search(pattern_bare, text, re.DOTALL)
    if match_bare:
        block = match_bare.group(0)
        # Basic sanity check for balanced braces/brackets
        has_balanced_braces = block.count("{") == block.count("}")
        has_balanced_brackets = block.count("[") == block.count("]")
        if has_balanced_braces and has_balanced_brackets:
            return block.strip()  # Return the matched bare block

    # No JSON block found
    return None


def _llm_resilient(max_attempts: int = 3, backoff: float = 1.0):  # Keep as is
    """Decorator for LLM calls, retrying on rate limits and transient errors."""

    def wrap(fn):
        @functools.wraps(fn)
        async def inner(*a, **kw):
            attempt = 0
            while True:
                try:
                    # Add delay before retrying (not on first attempt)
                    if attempt > 0:
                        delay_factor = 2 ** (attempt - 1)
                        base_delay = backoff * delay_factor
                        jitter = random.uniform(0.8, 1.2)
                        jitter_delay = base_delay * jitter
                        logger.debug(
                            f"LLM resilient retry {attempt}: Sleeping for {jitter_delay:.2f}s..."
                        )
                        await asyncio.sleep(jitter_delay)
                    # Call the wrapped LLM function
                    result = await fn(*a, **kw)
                    return result

                except ProviderError as e:
                    # Check if it's a rate limit error (common for 429 status)
                    err_str_lower = str(e).lower()
                    is_rate_limit = (
                        "429" in str(e)  # Check status code in error message
                        or "rate limit" in err_str_lower
                        or "too many requests" in err_str_lower
                        or "quota" in err_str_lower
                    )
                    if is_rate_limit:
                        attempt += 1
                        func_name = getattr(fn, "__name__", "?")
                        if attempt >= max_attempts:
                            logger.error(
                                f"LLM rate limit: '{func_name}' failed after {max_attempts} attempts: {e}"
                            )
                            raise ToolError(
                                f"LLM rate-limit exceeded after {max_attempts} attempts: {e}"
                            ) from e

                        # Check for Retry-After header suggestion in error
                        retry_after_seconds = None
                        retry_after_match = re.search(r"retry[- ]after[: ]+(\d+)", err_str_lower)
                        if retry_after_match:
                            try:
                                retry_after_seconds = int(retry_after_match.group(1))
                            except ValueError:
                                pass  # Ignore if number parsing fails

                        # Calculate delay: Use Retry-After if available, else exponential backoff
                        if retry_after_seconds:
                            delay = retry_after_seconds
                            logger.warning(
                                f"LLM rate limit for '{func_name}'. Retrying after suggested {delay:.2f}s (attempt {attempt}/{max_attempts})"
                            )
                        else:
                            delay_factor = 2 ** (
                                attempt - 1
                            )  # Use previous attempt for backoff calculation
                            base_delay = backoff * delay_factor
                            jitter = random.uniform(0.8, 1.2)
                            delay = base_delay * jitter
                            logger.warning(
                                f"LLM rate limit for '{func_name}'. Retrying after {delay:.2f}s (attempt {attempt}/{max_attempts})"
                            )

                        # Sleep before next attempt (actual sleep happens at loop start)
                        # await asyncio.sleep(delay) # Moved sleep logic to loop start
                        continue  # Go to next iteration to retry
                    else:
                        # Different ProviderError, re-raise
                        raise
                except (httpx.RequestError, asyncio.TimeoutError) as e:
                    # Handle transient network errors or timeouts
                    attempt += 1
                    func_name = getattr(fn, "__name__", "?")
                    if attempt >= max_attempts:
                        logger.error(
                            f"LLM call: '{func_name}' failed due to transient error after {max_attempts} attempts: {e}"
                        )
                        raise ToolError(
                            f"LLM call failed after {max_attempts} attempts: {e}"
                        ) from e
                    # Calculate delay using exponential backoff
                    # delay_factor = 2**(attempt - 1) # Using previous attempt number
                    # base_delay = backoff * delay_factor
                    # jitter = random.uniform(0.8, 1.2)
                    # delay = base_delay * jitter
                    # logger.warning(f"LLM transient error for '{func_name}'. Retrying after {delay:.2f}s (attempt {attempt}/{max_attempts})")
                    # await asyncio.sleep(delay) # Moved sleep logic to loop start
                    logger.warning(
                        f"LLM transient error for '{func_name}'. Retrying (attempt {attempt}/{max_attempts}). Error: {e}"
                    )
                    continue  # Go to next iteration
                except Exception:
                    # For any other unexpected errors, re-raise immediately
                    raise

        return inner

    return wrap


@_llm_resilient(max_attempts=3, backoff=1.0)
async def _call_llm(
    messages: Sequence[Dict[str, str]],
    model: str = _llm_model_locator_global,
    expect_json: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Union[Dict[str, Any], List[Any]]:  # Uses global _log
    """Makes a call to the LLM using the standalone chat_completion tool."""
    if not messages:
        logger.error("_call_llm received empty messages list.")
        return {"error": "No messages provided to LLM."}

    # Determine provider and model name
    llm_provider = Provider.OPENAI.value  # Default provider
    llm_model_name = model  # Default model name
    if model:
        try:
            extracted_provider, extracted_model = parse_model_string(model)
            if extracted_provider:
                llm_provider = extracted_provider
            if extracted_model:
                llm_model_name = extracted_model
        except Exception as parse_err:
            logger.warning(f"Could not parse model string '{model}': {parse_err}. Using defaults.")

    # Prepare arguments for chat_completion
    llm_args: Dict[str, Any] = {
        "provider": llm_provider,
        "model": llm_model_name,
        "messages": list(messages),  # Ensure it's a mutable list
        "temperature": temperature,
        "max_tokens": max_tokens,
        "additional_params": {},  # For provider-specific params like response_format
    }

    # Handle JSON mode expectation
    use_json_instruction = (
        False  # Flag to add manual instruction if native JSON mode fails/unsupported
    )
    if expect_json:
        try:
            # Check if the provider/model combination supports native JSON response format
            provider_instance = await get_provider(llm_provider)
            # Example check (adapt based on actual provider capabilities)
            supports_native_json = False
            if llm_provider == Provider.OPENAI.value and llm_model_name.startswith(
                ("gpt-4", "gpt-3.5-turbo")
            ):  # Check specific OpenAI models known to support it
                supports_native_json = True
            # Or use a generic check if provider interface defines it
            elif hasattr(provider_instance, "supports_json_response_format"):
                supports_native_json = await provider_instance.supports_json_response_format(
                    llm_model_name
                )

            if supports_native_json:
                logger.debug(
                    f"Provider '{llm_provider}' model '{llm_model_name}' supports native JSON mode."
                )
                # Add the provider-specific parameter for JSON mode
                # This varies by provider (e.g., OpenAI uses response_format)
                if llm_provider == Provider.OPENAI.value:
                    llm_args["additional_params"]["response_format"] = {"type": "json_object"}
                # Add other providers' JSON format params here if needed
                use_json_instruction = False  # Native mode used
            else:
                logger.debug(
                    f"Provider '{llm_provider}' model '{llm_model_name}' does not natively support JSON mode. Using manual instruction."
                )
                use_json_instruction = True  # Need manual instruction
        except Exception as e:
            logger.warning(
                f"Could not determine native JSON support for provider '{llm_provider}': {e}. Assuming manual instruction needed."
            )
            use_json_instruction = True

    # Add manual JSON instruction if needed
    if use_json_instruction:
        json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. Your entire response must start with `{` or `[` and end with `}` or `]`. Do not include ```json markers, comments, or any explanatory text before or after the JSON structure."
        modified_messages = list(llm_args["messages"])  # Work on a copy
        # Append instruction to the last user message, or add a new user message
        if modified_messages and modified_messages[-1]["role"] == "user":
            modified_messages[-1]["content"] += json_instruction
        else:
            # Add a new user message if last wasn't 'user' or list was empty
            modified_messages.append(
                {
                    "role": "user",
                    "content": "Provide the response based on the previous messages."
                    + json_instruction,
                }
            )
        llm_args["messages"] = modified_messages  # Update args with modified messages

    # Make the actual call to the standalone chat_completion tool
    try:
        start_time = time.monotonic()
        resp = await chat_completion(**llm_args)
        duration = time.monotonic() - start_time
        duration_ms = int(duration * 1000)
        model_returned = resp.get(
            "model", llm_model_name
        )  # Use model returned in response if available
        is_success = resp.get("success", False)
        is_cached = resp.get("cached_result", False)

        # Log the completion details
        await _log(
            "llm_call_complete",
            model=model_returned,
            duration_ms=duration_ms,
            success=is_success,
            cached=is_cached,
            provider=llm_provider,
        )

        # Process the response
        if not is_success:
            error_msg = resp.get("error", "LLM call failed with no specific error message.")
            # Try to get raw response details for debugging
            raw_resp_detail = None
            if isinstance(resp.get("details"), dict):
                raw_resp_detail = resp["details"].get("raw_response")
            if not raw_resp_detail:
                raw_resp_detail = resp.get("raw_response")  # Fallback check
            logger.warning(
                f"LLM call failed: {error_msg}. Raw response preview: {str(raw_resp_detail)[:200]}"
            )
            return {"error": f"LLM API Error: {error_msg}", "raw_response": raw_resp_detail}

        # Extract content from the successful response message
        assistant_message = resp.get("message", {})
        content = assistant_message.get("content")
        raw_text = content.strip() if isinstance(content, str) else ""

        if not raw_text:
            logger.warning("LLM returned empty response content.")
            return {"error": "LLM returned empty response content."}

        # Handle based on whether JSON was expected
        if not expect_json:
            # Return the raw text directly
            return {"text": raw_text}
        else:
            # Attempt to parse the response as JSON
            try:
                # Try direct JSON parsing first
                parsed_json = json.loads(raw_text)
                return parsed_json
            except json.JSONDecodeError:
                # If direct parsing fails, try extracting a JSON block
                logger.warning(
                    "LLM response was not valid JSON directly. Trying to extract JSON block..."
                )
                json_block = _extract_json_block(raw_text)
                if json_block:
                    try:
                        parsed_block = json.loads(json_block)
                        logger.warning(
                            "Successfully parsed JSON block extracted from LLM response."
                        )
                        return parsed_block
                    except json.JSONDecodeError as e:
                        # Error parsing the extracted block
                        block_preview = json_block[:500]
                        error_msg = f"Could not parse extracted JSON block: {e}. Block preview: {block_preview}..."
                        logger.error(error_msg)
                        return {
                            "error": error_msg,
                            "raw_response": raw_text[:1000],
                        }  # Return raw text for debugging
                else:
                    # No valid JSON block found within the text
                    error_msg = "Could not parse JSON from LLM response (no valid block found)."
                    logger.error(error_msg)
                    return {
                        "error": error_msg,
                        "raw_response": raw_text[:1000],
                    }  # Return raw text for debugging

    except ProviderError as e:
        # Catch errors raised by the chat_completion tool itself (e.g., auth, config)
        logger.error(f"LLM Provider error during chat_completion call: {e}")
        raw_resp_detail = None
        if hasattr(e, "details") and isinstance(getattr(e, "details", None), dict):
            raw_resp_detail = e.details.get("raw_response")
        return {"error": f"LLM Provider Error: {e}", "raw_response": raw_resp_detail}
    except Exception as e:
        # Catch any other unexpected errors during the call or processing
        logger.error(f"Unexpected error during LLM call: {e}", exc_info=True)
        return {"error": f"LLM call failed unexpectedly: {e}"}


# --- Macro/Autopilot Planners ---
ALLOWED_ACTIONS = {"click", "type", "wait", "download", "extract", "finish", "scroll"}


async def _plan_macro(
    page_state: Dict[str, Any], task: str, model: str = _llm_model_locator_global
) -> List[Dict[str, Any]]:  # Uses global _llm_model_locator_global
    """Generates a sequence of browser actions (macro steps) based on page state and a task."""
    # Detailed description of allowed actions for the LLM
    action_details = """
    Allowed Actions:
    - `click`: Clicks an element. Requires `task_hint` (description of the element to click).
    - `type`: Types text into an input field. Requires `task_hint` (description of the field) and `text` (the text to type). Optional: `enter: true` to press Enter after typing, `clear_before: false` to avoid clearing field first.
    - `wait`: Pauses execution. Requires `ms` (milliseconds to wait). Use sparingly for unavoidable dynamic content delays.
    - `download`: Clicks a link/button to initiate a download. Requires `task_hint` (description of download element). Optional: `dest` (destination directory path relative to storage).
    - `extract`: Extracts text from elements matching a CSS selector. Requires `selector`. Returns a list of strings.
    - `scroll`: Scrolls the page. Requires `direction` ('up', 'down', 'top', 'bottom'). Optional: `amount_px` (pixels for 'up'/'down', default 500).
    - `finish`: Indicates the task is complete. No arguments needed. Should be the last step if the task goal is achieved.
    """

    # Prepare summary of elements for the LLM prompt
    elements_summary = []
    elements_list = page_state.get("elements", [])
    for el in elements_list:
        el_id = el.get("id")
        el_tag = el.get("tag")
        el_role = el.get("role", " ")
        el_text = el.get("text", " ")
        max_text_len = 80
        truncated_text = el_text[:max_text_len] + ("..." if len(el_text) > max_text_len else "")
        summary_str = f"id={el_id} tag={el_tag} role='{el_role}' text='{truncated_text}'"
        elements_summary.append(summary_str)

    # System prompt for the macro planner LLM
    system_prompt = textwrap.dedent(f"""
        You are an expert web automation assistant. Your goal is to create a sequence of steps (a macro) to accomplish a user's task on the current web page.
        You will be given the current page state (URL, Title, main text content, and a list of interactive elements with their IDs, tags, roles, and text).
        You will also be given the user's task.
        Based on the page state and task, generate a JSON list of action steps.
        
        EACH step in the list MUST be a JSON object containing an "action" key specifying the action name (e.g., "click", "type").
        Other keys in the object should be the required arguments for that action (e.g., "task_hint", "text", "ms", "selector", "direction").

        {action_details}

        Generate ONLY the JSON list of steps following this structure: `[ {{"action": "action_name", "arg1": "value1", ...}}, ... ]`.

        DO NOT include explanations or markdown formatting!
        
        If the task seems impossible or cannot be mapped to the available actions/elements, return an empty list `[]`.
        
        If the task is already complete based on the current state (e.g., "find the price" and price is visible), you can return a `finish` step or an empty list.
    """).strip()

    # User prompt with page state and task
    elements_str = "\n".join(elements_summary)
    main_text_preview = page_state.get("main_text", "")[:500]  # Preview main text
    user_prompt = textwrap.dedent(f"""
        Current Page State:
        URL: {page_state.get("url", "[No URL]")}
        Title: {page_state.get("title", "[No Title]")}
        Main Text (Preview): {main_text_preview}...
        Elements:
        {elements_str}

        User Task: "{task}"

        Generate the JSON list of steps to accomplish this task. Respond ONLY with the JSON list.
    """).strip()

    # Prepare messages and call LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = await _call_llm(
        messages,
        model=model,
        expect_json=True,
        temperature=0.0,
        max_tokens=2048,  # Allow reasonable size for plan
    )

    # Process and validate the LLM response (Revised to handle single dict)
    plan_list: Optional[List[Dict[str, Any]]] = None
    if isinstance(result, list):
        plan_list = result
    elif isinstance(result, dict) and "error" in result:
        # Handle errors reported by the LLM call itself
        error_detail = result.get("raw_response", result["error"])
        raise ToolError(f"Macro planner LLM call failed: {result['error']}", details=error_detail)
    elif isinstance(result, dict):
        # --- Handling case where LLM returns a single step dict ---
        if "action" in result:  # Check if it looks like a valid step
            logger.warning(
                "LLM returned a single step dictionary instead of a list for macro plan. Wrapping it in a list."
            )
            plan_list = [result]
        elif "steps" in result and isinstance(result["steps"], list):
            # Handle cases where LLM wraps the list in a "steps" key (existing logic)
            logger.warning("LLM wrapped macro plan in 'steps' key. Extracting list.")
            plan_list = result["steps"]
        else:
            # It's a dict, but doesn't look like a step or contain 'steps'
            response_type = type(result).__name__
            response_preview = str(result)[:500]
            raise ToolError(
                f"Macro planner returned unexpected dictionary format: {response_type}. Preview: '{response_preview}...'",
                details={"raw_response": response_preview},
            )
    else:
        # Handle other unexpected response formats
        response_type = type(result).__name__
        response_preview = str(result)[:500]
        raise ToolError(
            f"Macro planner returned unexpected format: {response_type}. Expected list or dict. Preview: '{response_preview}...'",
            details={"raw_response": response_preview},
        )

    # Validate individual steps in the plan
    validated_plan = []
    if plan_list is not None:  # Check if we have a list to validate (could be empty list)
        for i, step in enumerate(plan_list):
            if not isinstance(step, dict) or "action" not in step:
                # Log raw response preview on validation error
                logger.warning(
                    f"Macro plan step {i + 1} invalid format (not dict or missing 'action'): {step}. RAW LLM RESPONSE PREVIEW: {str(result)[:500]}"
                )
                continue  # Skip invalid step format

            action = step.get("action")
            if action not in ALLOWED_ACTIONS:
                logger.warning(f"Macro plan step {i + 1} has invalid action '{action}': {step}")
                continue  # Skip step with unknown action

            # --- Basic argument checks ---
            error_flag = False
            if action in ("click", "download") and not step.get("task_hint"):
                logger.warning(f"Macro plan step {i + 1} '{action}' missing 'task_hint': {step}")
                error_flag = True
            if action == "type":
                if not step.get("task_hint"):
                    logger.warning(f"Macro plan step {i + 1} 'type' missing 'task_hint': {step}")
                    error_flag = True
                if step.get("text") is None:  # Allow empty string, but not None
                    logger.warning(f"Macro plan step {i + 1} 'type' missing 'text': {step}")
                    error_flag = True
            if action == "wait" and step.get("ms") is None:
                logger.warning(f"Macro plan step {i + 1} 'wait' missing 'ms': {step}")
                error_flag = True
            if action == "extract" and not step.get("selector"):
                logger.warning(f"Macro plan step {i + 1} 'extract' missing 'selector': {step}")
                error_flag = True
            if action == "scroll" and step.get("direction") not in ("up", "down", "top", "bottom"):
                logger.warning(
                    f"Macro plan step {i + 1} 'scroll' has invalid or missing 'direction': {step}"
                )
                error_flag = True
            # Add more specific checks as needed...

            if not error_flag:
                validated_plan.append(step)  # Add valid step to the final plan
            else:
                logger.warning(
                    f"Skipping invalid macro step {i + 1} due to missing/invalid arguments."
                )

    # --- Final check and logging/error based on validation outcome ---
    if not validated_plan:  # If plan is empty after validation
        response_preview = str(result)[:500] if result else "None"
        # Distinguish between LLM intentionally returning [] and validation failing all steps
        if plan_list is not None and len(plan_list) > 0:
            # LLM returned steps, but all were invalid
            raise ToolError(
                "Macro planner generated plan, but all steps were invalid.",
                details={"raw_response": response_preview, "original_plan_length": len(plan_list)},
            )
        elif plan_list is None:
            # This case should ideally be caught earlier by the type checking
            raise ToolError(
                "Macro planner failed to generate a valid list or dictionary of steps.",
                details={"raw_response": response_preview},
            )
        else:  # LLM returned [], which is valid
            logger.info(
                "Macro planner returned an empty list, indicating task completion or impossibility."
            )
            # Return the empty list in this case
            return []

    logger.debug(f"Validated macro plan has {len(validated_plan)} steps.")
    return validated_plan


_AVAILABLE_TOOLS = {  # Keep as is
    # Tool Name: (Standalone Function Name, {Arg Name: Arg Type Hint})
    "search_web": (
        "search",
        {
            "query": "str",
            "engine": "Optional[str: bing|duckduckgo|yandex]",
            "max_results": "Optional[int]",
        },
    ),
    "browse_page": (
        "browse",
        {
            "url": "str",
            "wait_for_selector": "Optional[str]",
            "wait_for_navigation": "Optional[bool]",
        },
    ),  # Updated browse args
    "click_element": (
        "click",
        {
            "url": "str",
            "task_hint": "Optional[str]",
            "target": "Optional[dict]",
            "wait_ms": "Optional[int]",
        },
    ),  # Updated click args
    "type_into_fields": (
        "type_text",
        {
            "url": "str",
            "fields": "List[dict{'task_hint':str,'text':str,'enter':bool?,'clear_before':bool?}]",
            "submit_hint": "Optional[str]",
            "submit_target": "Optional[dict]",
            "wait_after_submit_ms": "Optional[int]",
        },
    ),  # Updated type_text args
    "download_file_via_click": (
        "download",
        {
            "url": "str",
            "task_hint": "Optional[str]",
            "target": "Optional[dict]",
            "dest_dir": "Optional[str]",
        },
    ),  # Updated download args
    "run_page_macro": (
        "run_macro",
        {
            "url": "str",
            "task": "str",
            "model": "Optional[str]",
            "max_rounds": "Optional[int]",
            "timeout_seconds": "Optional[int]",
        },
    ),  # Updated run_macro args
    "download_all_pdfs_from_site": (
        "download_site_pdfs",
        {
            "start_url": "str",
            "dest_subfolder": "Optional[str]",
            "include_regex": "Optional[str]",
            "max_depth": "Optional[int]",
            "max_pdfs": "Optional[int]",
            "rate_limit_rps": "Optional[float]",
        },
    ),  # Updated download_site_pdfs args
    "collect_project_documentation": (
        "collect_documentation",
        {"package": "str", "max_pages": "Optional[int]", "rate_limit_rps": "Optional[float]"},
    ),  # Updated collect_documentation args
    "process_urls_in_parallel": (
        "parallel",
        {"urls": "List[str]", "action": "str('get_state')", "max_tabs": "Optional[int]"},
    ),  # Updated parallel args
    "get_filesystem_status": ("filesystem_status", {}),  # Example Filesystem tool
    "read_file": ("read_file", {"path": "str"}),  # Example Filesystem tool
    "write_file": (
        "write_file",
        {"path": "str", "content": "Union[str, bytes]", "append": "Optional[bool]"},
    ),  # Example Filesystem tool
}

_PLANNER_SYS = textwrap.dedent("""
    You are an AI assistant acting as the central planner for a web automation and information retrieval system.
    Your goal is to achieve the user's complex task by selecting the appropriate tool and providing the correct arguments for each step.
    You will be given the user's overall task and a summary of results from previous steps (if any).
    You have access to a set of tools, described below with their names and argument schemas (use JSON format for args).
    Select ONE tool to execute next that will make progress towards the user's goal.
    Carefully consider the user's task and the previous results to choose the best tool and arguments.
    If a previous step failed, analyze the error and decide whether to retry, try a different approach, or ask for clarification (if interaction allowed). For now, focus on selecting the next best tool.
    If the task requires information from the web, use `search_web` first unless a specific URL is provided or implied.
    If the task involves interacting with a specific webpage (clicking, typing, downloading), use the appropriate browser tool (`browse_page`, `click_element`, `type_into_fields`, `download_file_via_click`, `run_page_macro`). Use the URL from previous steps if available.
    For filesystem operations, use the filesystem tools like `read_file`, `write_file`.
    Use `run_page_macro` for multi-step interactions on a single page described in natural language.
    Use `collect_project_documentation` or `download_all_pdfs_from_site` for specialized crawling tasks.
    Use `process_urls_in_parallel` only when needing the *same* simple action (like getting state) on *multiple* distinct URLs.

    Respond ONLY with a JSON list containing a single step object. The object must have:
    - "tool": The name of the selected tool (string).
    - "args": A JSON object containing the arguments for the tool (matching the schema).

    Example Response:
    ```json
    [
      {
        "tool": "search_web",
        "args": {
          "query": "latest news on AI regulation",
          "engine": "duckduckgo"
        }
      }
    ]
    ```
    If you determine the task is complete based on the prior results, respond with an empty JSON list `[]`.
""").strip()


async def _plan_autopilot(
    task: str, prior_results: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:  # Uses global _AVAILABLE_TOOLS, _PLANNER_SYS, _call_llm
    """Generates the next step (tool call) for the Autopilot based on task and history."""
    # Describe available tools for the LLM prompt
    tools_desc = {}
    for name, data in _AVAILABLE_TOOLS.items():
        func_name, schema = data
        tools_desc[name] = schema

    # Summarize prior results concisely
    prior_summary = "None"
    if prior_results:
        summaries = []
        # Summarize last 3 steps for context, or fewer if less than 3 executed
        start_index = max(0, len(prior_results) - 3)
        for i, res in enumerate(prior_results[start_index:], start=start_index + 1):
            tool_used = res.get("tool", "?")
            was_success = res.get("success", False)
            outcome_marker = "[OK]" if was_success else "[FAIL]"
            # Get result summary or error message - prefer 'message' if present, else result/error
            result_data = res.get("message", res.get("result", res.get("error", "")))
            # Handle dict results slightly better
            if isinstance(result_data, dict):
                # Extract key info or just summarize keys
                dict_preview = str(list(result_data.keys()))
                details_str = f"Dict{dict_preview[:130]}" + (
                    "..." if len(dict_preview) > 130 else ""
                )
            else:
                details_str = str(result_data)[:150] + (
                    "..." if len(str(result_data)) > 150 else ""
                )  # Truncate long results/errors

            summary_line = f"Step {i}: Ran {tool_used} -> {outcome_marker} ({details_str})"
            summaries.append(summary_line)
        prior_summary = "\n".join(summaries)

    # Construct the user prompt
    tools_json_str = json.dumps(tools_desc, indent=2)
    # Use the same _PLANNER_SYS prompt, as it requests a list with one step
    user_prompt = (
        f"AVAILABLE TOOLS (Schema):\n{tools_json_str}\n\n"
        f"PRIOR RESULTS SUMMARY (Last {len(summaries) if prior_results else 0} steps):\n{prior_summary}\n\n"
        f"USER TASK:\n{task}\n\n"
        "Select the single best tool and arguments for the *next* step to achieve the user task. "
        "Respond ONLY with a JSON list containing exactly one step object (tool, args), or an empty list [] if the task is complete or cannot proceed."
    )

    # Prepare messages and call the LLM planner
    messages = [
        {"role": "system", "content": _PLANNER_SYS},  # Use the standardized system prompt
        {"role": "user", "content": user_prompt},
    ]
    response = await _call_llm(
        messages,
        expect_json=True,
        temperature=0.0,
        max_tokens=2048,
    )

    # --- Process and validate the LLM response (Revised) ---
    if isinstance(response, dict) and "error" in response:
        raise ToolError(f"Autopilot planner LLM call failed: {response['error']}")

    current_plan_list: List[Dict[str, Any]] = []  # Initialize as empty list

    if isinstance(response, list):
        current_plan_list = response  # LLM returned the expected list
    elif isinstance(response, dict):
        # --- Handling case where LLM returns a single step dict ---
        if "tool" in response and "args" in response:  # Check if it looks like a valid step
            logger.warning(
                "Autopilot planner returned a single step dictionary instead of a list. Wrapping it."
            )
            current_plan_list = [response]
        else:
            # It's a dict, but doesn't look like a valid step
            response_type = type(response).__name__
            raise ToolError(
                f"Autopilot planner returned unexpected dictionary format: {response_type}. Expected a JSON list or a valid step dict."
            )
    else:
        # Handle other unexpected response formats
        response_type = type(response).__name__
        raise ToolError(
            f"Autopilot planner returned unexpected format: {response_type}. Expected a JSON list."
        )

    # --- Validate the structure and content of the step(s) ---
    validated_plan: List[Dict[str, Any]] = []
    if len(current_plan_list) > 1:
        logger.warning(
            f"Autopilot planner returned multiple steps ({len(current_plan_list)}). Only using the first one."
        )
    elif len(current_plan_list) == 0:
        logger.info(
            "Autopilot planner returned an empty list, indicating task completion or inability to proceed."
        )
        return []  # Return empty list as intended

    # Process the first (and only expected) step
    if len(current_plan_list) >= 1:
        step = current_plan_list[0]
        if not isinstance(step, dict):
            logger.warning(f"Autopilot planner step is not a dictionary: {step}")
            return []  # Return empty plan if format is wrong

        tool_name = step.get("tool")
        tool_args = step.get("args")

        if not tool_name or not isinstance(tool_args, dict):
            logger.warning(
                f"Autopilot planner step missing 'tool' or 'args' (must be dict): {step}"
            )
            return []  # Return empty plan if structure is wrong

        if tool_name not in _AVAILABLE_TOOLS:
            logger.warning(f"Autopilot planner selected unknown tool '{tool_name}': {step}")
            return []  # Return empty plan if tool is unknown

        # Optional: Add deeper validation of args based on _AVAILABLE_TOOLS schema if needed

        # If validation passes, add the single step to the plan
        validated_plan.append(step)

    # Return the validated plan (containing 0 or 1 step)
    return validated_plan


# --- Step Runner (for Macro) ---
async def run_steps(
    page: Page, steps: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:  # Uses global smart_click, smart_type, smart_download
    """Executes a sequence of predefined macro steps on a given page."""
    results: List[Dict[str, Any]] = []  # Stores results of each step

    for i, step in enumerate(steps):
        action = step.get("action")
        step_result = step.copy()  # Start with original step data
        step_result["success"] = False  # Default to failure
        start_time = time.monotonic()
        step_num = i + 1
        should_break = False  # Initialize break flag for this step

        if not action:
            step_result["error"] = f"Step {step_num}: Missing 'action' key."
            logger.warning(step_result["error"])
            results.append(step_result)
            continue  # Skip to next step

        try:
            logger.debug(
                f"Executing Macro Step {step_num}: Action='{action}', Args={ {k: v for k, v in step.items() if k != 'action'} }"
            )
            # --- Execute Action ---
            if action == "click":
                hint = step.get("task_hint")
                target_fallback = step.get("target")  # Optional fallback args
                if not hint:
                    raise ToolInputError(
                        f"Step {step_num} ('click'): Missing required argument 'task_hint'."
                    )

                # Check for and handle common obstacles like reCAPTCHA
                if "recaptcha" in hint.lower() or "captcha" in hint.lower():
                    # Try to detect CAPTCHA presence first
                    captcha_js = """() => {
                        return document.body.innerText.toLowerCase().includes('captcha') || 
                               document.querySelector('iframe[title*=captcha]') !== null ||
                               document.querySelector('[id*=captcha]') !== null ||
                               document.querySelector('[class*=captcha]') !== null ||
                               document.querySelector('div[class*="recaptcha"]') !== null;
                    }"""
                    captcha_detected = await page.evaluate(captcha_js)
                    if captcha_detected:
                        logger.warning(
                            f"Step {step_num}: CAPTCHA detected but cannot be automatically solved. Marking as failed."
                        )
                        step_result["error"] = "CAPTCHA detected - requires manual intervention"
                        step_result["success"] = False
                        # Continue to finally block without raising exception
                    else:
                        # Use the smart_click helper
                        click_success = await smart_click(
                            page, task_hint=hint, target_kwargs=target_fallback
                        )
                        step_result["success"] = click_success
                else:
                    # Use the smart_click helper
                    click_success = await smart_click(
                        page, task_hint=hint, target_kwargs=target_fallback
                    )
                    step_result["success"] = click_success  # Should be True if no exception

            elif action == "type":
                hint = step.get("task_hint")
                target_fallback = step.get("target")
                text = step.get("text")
                if not hint:
                    raise ToolInputError(
                        f"Step {step_num} ('type'): Missing required argument 'task_hint'."
                    )
                if text is None:  # Allow empty string, but not None
                    raise ToolInputError(
                        f"Step {step_num} ('type'): Missing required argument 'text'."
                    )
                # Get optional arguments
                press_enter = step.get("enter", False)  # Default False
                clear_before = step.get("clear_before", True)  # Default True
                # Use the smart_type helper
                type_success = await smart_type(
                    page,
                    task_hint=hint,
                    text=text,
                    press_enter=press_enter,
                    clear_before=clear_before,
                    target_kwargs=target_fallback,
                    timeout_ms=5000,
                )
                step_result["success"] = type_success

            elif action == "wait":
                ms = step.get("ms")
                if ms is None:
                    raise ToolInputError(
                        f"Step {step_num} ('wait'): Missing required argument 'ms'."
                    )
                try:
                    wait_ms = int(ms)
                    if wait_ms < 0:
                        raise ValueError("Wait time must be non-negative")
                    await page.wait_for_timeout(wait_ms)
                    step_result["success"] = True
                except (ValueError, TypeError) as e:
                    raise ToolInputError(
                        f"Step {step_num} ('wait'): Invalid 'ms' value '{ms}'. {e}"
                    ) from e

            elif action == "download":
                hint = step.get("task_hint")
                target_fallback = step.get("target")
                if not hint:
                    raise ToolInputError(
                        f"Step {step_num} ('download'): Missing required argument 'task_hint'."
                    )
                dest_dir = step.get("dest")  # Optional destination directory
                # Use the smart_download helper
                download_outcome = await smart_download(
                    page, task_hint=hint, dest_dir=dest_dir, target_kwargs=target_fallback
                )
                step_result["result"] = download_outcome  # Store full download result
                # Success is determined by the helper's output
                step_result["success"] = download_outcome.get("success", False)

            elif action == "extract":
                selector = step.get("selector")
                if not selector:
                    raise ToolInputError(
                        f"Step {step_num} ('extract'): Missing required argument 'selector'."
                    )
                # Use Playwright's evaluate_all to get text from matching elements
                js_func = "(elements => elements.map(el => el.innerText || el.textContent || ''))"
                extracted_texts_raw = await page.locator(selector).evaluate_all(js_func)
                # Clean up results: filter empty strings and strip whitespace
                extracted_texts_clean = []
                for t in extracted_texts_raw:
                    stripped_t = t.strip()
                    if stripped_t:
                        extracted_texts_clean.append(stripped_t)
                step_result["result"] = extracted_texts_clean
                step_result["success"] = True  # Extraction itself succeeds if selector is valid

            elif action == "scroll":
                direction = step.get("direction")
                amount = step.get("amount_px")
                if not direction or direction not in ["up", "down", "top", "bottom"]:
                    error_msg = f"Step {step_num} ('scroll'): Invalid or missing scroll direction: '{direction}'. Must be 'up', 'down', 'top', or 'bottom'."
                    step_result["error"] = error_msg
                    step_result["success"] = False
                    logger.warning(error_msg)
                    # Continue to finally block without raising, as scroll failure might not be critical
                else:
                    if direction == "top":
                        js_scroll = "() => window.scrollTo(0, 0)"
                        await page.evaluate(js_scroll)
                    elif direction == "bottom":
                        js_scroll = "() => window.scrollTo(0, document.body.scrollHeight)"
                        await page.evaluate(js_scroll)
                    elif direction == "up":
                        scroll_amount = int(amount or 500)  # Default 500px
                        js_scroll = "(px) => window.scrollBy(0, -px)"
                        await page.evaluate(js_scroll, scroll_amount)
                    elif direction == "down":
                        scroll_amount = int(amount or 500)  # Default 500px
                        js_scroll = "(px) => window.scrollBy(0, px)"
                        await page.evaluate(js_scroll, scroll_amount)
                    step_result["success"] = True

            elif action == "finish":
                logger.info(f"Macro execution Step {step_num}: Reached 'finish' action.")
                step_result["success"] = True
                # No Playwright action needed

            else:
                # Should not happen if _plan_macro validates actions, but safety check
                raise ValueError(
                    f"Step {step_num}: Unknown action '{action}' encountered during execution."
                )

            # Record duration on success or handled failure (like scroll direction)
            duration_ms = int((time.monotonic() - start_time) * 1000)
            step_result["duration_ms"] = duration_ms

        except (
            PlaywrightTimeoutError,
            ToolError,
            ToolInputError,
            ValueError,
            AssertionError,
            Exception,
        ) as e:
            # Catch errors during action execution
            err_type = type(e).__name__
            error_msg = f"{err_type} during action '{action}': {e}"
            step_result["error"] = error_msg
            step_result["success"] = False  # Ensure success is false on error
            logger.warning(f"Macro Step {step_num} ('{action}') failed: {error_msg}")

            # Special handling for CAPTCHA-related errors
            if "captcha" in str(e).lower() or "recaptcha" in str(e).lower():
                logger.warning(
                    f"Step {step_num}: CAPTCHA-related error detected. Suggesting manual intervention."
                )
                step_result["error"] = (
                    f"CAPTCHA-related error: {error_msg}. Manual intervention may be required."
                )

            # Record duration even on failure
            duration_ms = int((time.monotonic() - start_time) * 1000)
            step_result["duration_ms"] = duration_ms

        finally:
            # Always log the step result and append to the list
            log_details = step_result.copy()  # Create copy for logging
            # Avoid logging potentially large results directly
            if "result" in log_details:
                log_details["result_summary"] = str(log_details["result"])[:200] + "..."
                del log_details["result"]
            await _log("macro_step_result", **log_details)
            results.append(step_result)

            # If a 'finish' action succeeded, stop executing further steps
            if action == "finish" and step_result.get("success", False):
                logger.info(
                    f"Stopping macro execution after successful 'finish' action at step {step_num}."
                )
                should_break = True

        # Check break flag - this now works because should_break is properly scoped
        if should_break:
            logger.info(f"Breaking execution loop after step {step_num}")
            break

    return results  # Return list of results for all executed steps


# --- Universal Search ---
_ua_rotation_count = 0
_user_agent_pools = {  # Keep as is, ensure actual UAs are filled in
    "bing": deque(
        [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.51",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        ]
    ),
    "duckduckgo": deque(
        [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        ]
    ),
    "yandex": deque(
        [  # Yandex might be more sensitive, use diverse UAs
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 YaBrowser/23.5.2.625 Yowser/2.5 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 YaBrowser/23.5.2.625 Yowser/2.5 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 YaBrowser/23.5.2.625 Yowser/2.5 Safari/537.36",
        ]
    ),
}


@resilient(max_attempts=2, backoff=1.0)
async def search_web(
    query: str, engine: str = "bing", max_results: int = 10
) -> List[Dict[str, str]]:  # Uses global _log, _ua_rotation_count, _user_agent_pools
    """Performs a web search using a specified engine via browser automation."""
    global _ua_rotation_count
    engine_lower = engine.lower()
    if engine_lower not in ("bing", "duckduckgo", "yandex"):
        raise ToolInputError(
            f"Invalid search engine specified: '{engine}'. Use 'bing', 'duckduckgo', or 'yandex'."
        )

    # Sanitize query (basic removal of non-alphanumeric/space/hyphen/dot)
    safe_query_chars = re.sub(r"[^\w\s\-\.]", "", query)
    safe_query = safe_query_chars.strip()
    if not safe_query:
        raise ToolInputError("Search query cannot be empty or contain only invalid characters.")

    # URL encode the safe query
    qs = urllib.parse.quote_plus(safe_query)
    nonce = random.randint(1000, 9999)  # Simple nonce/cache buster

    # Search URLs and CSS Selectors per engine
    search_details = {
        "bing": {
            "url": f"https://www.bing.com/search?q={qs}&form=QBLH&nc={nonce}",
            "selectors": {
                "item": "li.b_algo",  # Correct: Main result container
                "title": "h2 > a",  # Correct: Targets link within H2 for title
                "link": "h2 > a",  # Correct: Same element for link href
                "snippet": "div.b_caption p, .TextContainer.OrganicText",  # CORRECTED: Handles standard captions and organic text containers
                "snippet_alt": ".b_caption",  # CORRECTED: General caption container as fallback
            },
        },
        "duckduckgo": {
            "url": f"https://html.duckduckgo.com/html/?q={qs}&nc={nonce}",
            "selectors": {
                # Use the more specific class for the main result item
                "item": "div.web-result",
                # Add classes for specificity, although h2>a might have been okay
                "title": "h2.result__title > a.result__a",
                "link": "h2.result__title > a.result__a",
                # Snippet selector looks correct
                "snippet": "a.result__snippet",
                # Add the snippet_alt just in case structure varies slightly
                "snippet_alt": "div.result__snippet",  # Alternative if snippet isn't a link
            },
        },
        "yandex": {
            # Yandex search results structure
            "url": f"https://yandex.com/search/?text={qs}&nc={nonce}&lr=202",  # Added &lr=202 based on your example URL for consistency
            "selectors": {
                "item": "li.serp-item",  # Correct: Main result container
                "title": "a.OrganicTitle-Link",  # CORRECTED: Target the main link for title text
                "link": "a.OrganicTitle-Link",  # CORRECTED: Target the main link for href attribute
                "snippet": ".TextContainer.OrganicText",  # Correct: Specific snippet container
                "snippet_alt": ".Organic-ContentWrapper",  # Correct: Parent as fallback
            },
        },
    }

    engine_info = search_details[engine_lower]
    search_url = engine_info["url"]
    sel = engine_info["selectors"]

    # Rotate User Agent
    _ua_rotation_count += 1
    ua_pool = _user_agent_pools[engine_lower]
    if _ua_rotation_count % 20 == 0 and len(ua_pool) > 1:
        # Rotate deque periodically
        first_ua = ua_pool.popleft()
        ua_pool.append(first_ua)
    ua = ua_pool[0]  # Use the current first UA

    # Get incognito context with specific UA
    context_args = {"user_agent": ua, "locale": "en-US"}  # Set UA and locale
    ctx, _ = await get_browser_context(use_incognito=True, context_args=context_args)
    page = None  # Initialize page variable

    try:
        page = await ctx.new_page()
        await _log("search_start", engine=engine_lower, query=query, url=search_url, ua=ua)
        # Navigate to search URL
        nav_timeout = 30000  # 30 seconds
        await page.goto(search_url, wait_until="domcontentloaded", timeout=nav_timeout)

        # Handle DuckDuckGo HTML meta refresh if present
        if engine_lower == "duckduckgo":
            try:
                meta_refresh_selector = 'meta[http-equiv="refresh"]'
                meta_refresh = await page.query_selector(meta_refresh_selector)
                if meta_refresh:
                    content_attr = await meta_refresh.get_attribute("content")
                    if content_attr and "url=" in content_attr.lower():
                        # Extract redirect URL
                        match = re.search(r'url=([^"]+)', content_attr, re.IGNORECASE)
                        if match:
                            redirect_url_raw = match.group(1)
                            # Basic clean up of URL just in case
                            redirect_url = redirect_url_raw.strip("'\" ")
                            logger.info(
                                f"Following meta refresh redirect on DDG HTML: {redirect_url}"
                            )
                            await page.goto(
                                redirect_url, wait_until="domcontentloaded", timeout=20000
                            )
                            await asyncio.sleep(0.5)  # Brief pause after redirect
            except PlaywrightException as e:
                logger.warning(f"Error checking/following meta refresh on DDG HTML: {e}")

        # Wait for results container to be visible
        wait_selector_timeout = 10000  # 10 seconds
        try:
            await page.wait_for_selector(
                sel["item"], state="visible", timeout=wait_selector_timeout
            )
        except PlaywrightTimeoutError as e:
            # Check for CAPTCHA before assuming no results
            captcha_js = "() => document.body.innerText.toLowerCase().includes('captcha') || document.querySelector('iframe[title*=captcha]') || document.querySelector('[id*=captcha]')"
            captcha_found = await page.evaluate(captcha_js)
            if captcha_found:
                await _log("search_captcha", engine=engine_lower, query=query)
                raise ToolError(
                    f"CAPTCHA detected on {engine_lower} search.", error_code="captcha_detected"
                ) from e
            else:
                # No results selector found, and no obvious CAPTCHA
                await _log(
                    "search_no_results_selector",
                    engine=engine_lower,
                    query=query,
                    selector=sel["item"],
                )
                return []  # Return empty list for no results

        # Brief pause and try to accept consent cookies (best effort)
        await asyncio.sleep(random.uniform(0.5, 1.5))
        consent_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Agree")',
            'button[id*="consent"]',
            'button[class*="consent"]',
        ]
        for btn_sel in consent_selectors:
            try:
                consent_button = page.locator(btn_sel).first
                await consent_button.click(timeout=1000)  # Short timeout for consent click
                logger.debug(f"Clicked potential consent button: {btn_sel}")
                await asyncio.sleep(0.3)  # Pause after click
                break  # Stop after first successful click
            except PlaywrightException:
                pass  # Ignore if selector not found or click fails

        # Extract results using page.evaluate
        extract_js = """
        (args) => {
            const results = [];
            const items = document.querySelectorAll(args.sel.item);
            for (let i = 0; i < Math.min(items.length, args.max_results); i++) {
                const item = items[i];
                const titleEl = item.querySelector(args.sel.title);
                const linkEl = item.querySelector(args.sel.link);
                let snippetEl = item.querySelector(args.sel.snippet);
                // Use fallback snippet selector if primary not found
                if (!snippetEl && args.sel.snippet_alt) {
                     snippetEl = item.querySelector(args.sel.snippet_alt);
                }

                const title = titleEl ? titleEl.innerText.trim() : '';
                let link = linkEl ? linkEl.href : '';
                // Clean DDG HTML links
                if (link && link.includes('uddg=')) {
                    try {
                        const urlParams = new URLSearchParams(link.split('?')[1]);
                        link = urlParams.get('uddg') || link;
                    } catch (e) { /* ignore URL parsing errors */ }
                }
                const snippet = snippetEl ? snippetEl.innerText.trim() : '';

                // Only add if essential parts (link and title or snippet) are present
                if (link && (title || snippet)) {
                    results.push({ title, link, snippet });
                }
            }
            return results;
        }
        """
        eval_args = {"sel": sel, "max_results": max_results}
        results = await page.evaluate(extract_js, eval_args)

        # Log completion and return results
        num_results = len(results)
        await _log("search_complete", engine=engine_lower, query=query, num_results=num_results)
        return results

    except PlaywrightException as e:
        # Handle Playwright errors during navigation or interaction
        await _log("search_error_playwright", engine=engine_lower, query=query, error=str(e))
        raise ToolError(f"Playwright error during {engine_lower} search for '{query}': {e}") from e
    except Exception as e:
        # Handle unexpected errors
        await _log("search_error_unexpected", engine=engine_lower, query=query, error=str(e))
        raise ToolError(f"Unexpected error during {engine_lower} search for '{query}': {e}") from e
    finally:
        # Ensure page and context are closed
        if page and not page.is_closed():
            await page.close()
        if ctx:
            await ctx.close()


# --- Initialization Function ---
async def _ensure_initialized():  # Uses MANY globals
    """Main initialization sequence for standalone Smart Browser tools."""
    global \
        _is_initialized, \
        _thread_pool, \
        _locator_cache_cleanup_task_handle, \
        _inactivity_monitor_task_handle
    global _SB_INTERNAL_BASE_PATH_STR, _STATE_FILE, _LOG_FILE, _CACHE_DB, _READ_JS_CACHE
    # Globals for config values
    global _sb_state_key_b64_global, _sb_max_tabs_global, _sb_tab_timeout_global
    global _sb_inactivity_timeout_global, _headless_mode_global, _vnc_enabled_global
    global _vnc_password_global, _proxy_pool_str_global, _proxy_allowed_domains_str_global
    global _vault_allowed_paths_str_global, _max_widgets_global, _max_section_chars_global
    global _dom_fp_limit_global, _llm_model_locator_global, _retry_after_fail_global
    global _seq_cutoff_global, _area_min_global, _high_risk_domains_set_global
    global _cpu_count, _pw, _browser, _ctx
    global _pid, _last_activity

    # Ensure _last_activity has a valid monotonic time ASAP if it's still at its module-load default.
    # This is a defensive measure.
    if _last_activity == 0.0:
        _last_activity = time.monotonic()
        logger.debug(
            f"Defensively setting initial _last_activity in _ensure_initialized: {_last_activity}"
        )

    # Quick check if already initialized
    if _is_initialized:
        return

    # Use lock to prevent concurrent initialization
    async with _init_lock:
        # Double-check after acquiring lock
        if _is_initialized:
            return
        logger.info("Performing first-time async initialization of SmartBrowser tools...")

        # --- Step 1: Load Config into Globals ---
        try:
            config = get_config()
            sb_config: SmartBrowserConfig = config.smart_browser  # Access nested config

            # Assign config values to globals, using defaults if config value is None/missing
            _sb_state_key_b64_global = sb_config.sb_state_key_b64 or _sb_state_key_b64_global
            _sb_max_tabs_global = sb_config.sb_max_tabs or _sb_max_tabs_global
            _sb_tab_timeout_global = sb_config.sb_tab_timeout or _sb_tab_timeout_global
            _sb_inactivity_timeout_global = (
                sb_config.sb_inactivity_timeout or _sb_inactivity_timeout_global
            )
            # Handle booleans carefully (check for None, not just falsiness)
            if sb_config.headless_mode is not None:
                _headless_mode_global = sb_config.headless_mode
            if sb_config.vnc_enabled is not None:
                _vnc_enabled_global = sb_config.vnc_enabled
            _vnc_password_global = sb_config.vnc_password or _vnc_password_global
            _proxy_pool_str_global = sb_config.proxy_pool_str or _proxy_pool_str_global
            _proxy_allowed_domains_str_global = (
                sb_config.proxy_allowed_domains_str or _proxy_allowed_domains_str_global
            )
            _vault_allowed_paths_str_global = (
                sb_config.vault_allowed_paths_str or _vault_allowed_paths_str_global
            )
            _max_widgets_global = sb_config.max_widgets or _max_widgets_global
            _max_section_chars_global = sb_config.max_section_chars or _max_section_chars_global
            _dom_fp_limit_global = sb_config.dom_fp_limit or _dom_fp_limit_global
            _llm_model_locator_global = sb_config.llm_model_locator or _llm_model_locator_global
            if sb_config.retry_after_fail is not None:
                _retry_after_fail_global = sb_config.retry_after_fail
            if sb_config.seq_cutoff is not None:
                _seq_cutoff_global = sb_config.seq_cutoff
            _area_min_global = sb_config.area_min or _area_min_global
            # Handle set carefully (assign if present in config)
            if sb_config.high_risk_domains_set is not None:
                _high_risk_domains_set_global = sb_config.high_risk_domains_set

            logger.info("Smart Browser configuration loaded into global variables.")
            # Update derived settings from config strings
            _update_proxy_settings()
            _update_vault_paths()

            # --- Reconfigure thread pool based on loaded config ---
            # Get current max_workers (handle potential attribute absence)
            current_max_workers = getattr(
                _thread_pool, "_max_workers", min(32, (_cpu_count or 1) * 2 + 4)
            )
            # Calculate desired based on *loaded* max tabs config
            desired_max_workers = min(32, _sb_max_tabs_global * 2)
            # Recreate pool only if worker count needs to change
            if current_max_workers != desired_max_workers:
                logger.info(
                    f"Reconfiguring thread pool max_workers from {current_max_workers} to {desired_max_workers} based on config."
                )
                _thread_pool.shutdown(wait=True)  # Wait for existing tasks
                _thread_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=desired_max_workers, thread_name_prefix="sb_worker"
                )

        except Exception as e:
            logger.error(
                f"Error loading Smart Browser config: {e}. Using default global values.",
                exc_info=True,
            )
            # Ensure derived settings are updated even if config load fails
            _update_proxy_settings()
            _update_vault_paths()

        # --- Step 2: Prepare Internal Storage Directory ---
        try:
            # Define relative path for internal storage (within the main storage area)
            internal_storage_relative_path = "storage/smart_browser_internal"
            logger.info(
                f"Ensuring internal storage directory exists: '{internal_storage_relative_path}' using filesystem tool."
            )
            # Use STANDALONE create_directory tool
            create_dir_result = await create_directory(path=internal_storage_relative_path)
            # Validate result
            if not isinstance(create_dir_result, dict) or not create_dir_result.get("success"):
                error_msg = (
                    create_dir_result.get("error", "Unknown")
                    if isinstance(create_dir_result, dict)
                    else "Invalid response"
                )
                raise ToolError(
                    f"Filesystem tool failed to create internal directory '{internal_storage_relative_path}'. Error: {error_msg}"
                )

            resolved_base_path_str = create_dir_result.get("path")
            if not resolved_base_path_str:
                raise ToolError(
                    "Filesystem tool create_directory succeeded but did not return the absolute path."
                )

            # Set global path variables based on the resolved absolute path
            _SB_INTERNAL_BASE_PATH_STR = resolved_base_path_str
            internal_base_path = Path(_SB_INTERNAL_BASE_PATH_STR)
            _STATE_FILE = internal_base_path / "storage_state.enc"
            _LOG_FILE = internal_base_path / "audit.log"
            _CACHE_DB = internal_base_path / "locator_cache.db"  # Adjusted name from original
            _READ_JS_CACHE = internal_base_path / "readability.js"
            logger.info(
                f"Smart Browser internal file paths configured within: {internal_base_path}"
            )

            # Initialize components that depend on these paths
            _init_last_hash()  # Initialize audit log hash chain (sync)
            _init_locator_cache_db_sync()  # Initialize DB schema (sync)

        except Exception as e:
            # If storage setup fails, it's critical, stop initialization
            logger.critical(
                f"CRITICAL FAILURE: Could not initialize Smart Browser internal storage at '{internal_storage_relative_path}': {e}",
            )
            return  # Do not proceed

        # --- Step 3: Initialize Browser Context (triggers Playwright launch if needed) ---
        try:
            logger.info("Initializing Playwright browser and shared context...")
            await get_browser_context()  # Call helper to ensure PW, browser, shared context exist
            logger.info("Playwright browser and shared context initialized successfully.")
        except Exception as e:
            logger.critical(
                f"CRITICAL FAILURE: Failed to initialize Playwright components: {e}", exc_info=True
            )
            # Attempt cleanup? Maybe not here, shutdown handler should cover it.
            return  # Stop initialization

        # --- Step 4: Start Background Tasks ---
        # Start Inactivity Monitor
        timeout_sec = _sb_inactivity_timeout_global
        if timeout_sec > 0:
            if _inactivity_monitor_task_handle is None or _inactivity_monitor_task_handle.done():
                logger.info(
                    f"Starting browser inactivity monitor task (Timeout: {timeout_sec}s)..."
                )
                _inactivity_monitor_task_handle = asyncio.create_task(
                    _inactivity_monitor(timeout_sec)
                )
            else:
                logger.debug("Inactivity monitor task already running.")
        else:
            logger.info("Browser inactivity monitor disabled (timeout <= 0).")

        # Start Locator Cache Cleanup Task
        cleanup_interval_sec = 24 * 60 * 60  # Run daily
        if _locator_cache_cleanup_task_handle is None or _locator_cache_cleanup_task_handle.done():
            logger.info(
                f"Starting locator cache cleanup task (Interval: {cleanup_interval_sec}s)..."
            )
            _locator_cache_cleanup_task_handle = asyncio.create_task(
                _locator_cache_cleanup_task(interval_seconds=cleanup_interval_sec)
            )
        else:
            logger.debug("Locator cache cleanup task already running.")

        # --- Finalize ---
        _is_initialized = True
        _last_activity = time.monotonic()  # Set initial activity time after successful init
        logger.info("SmartBrowser tools async components initialized successfully.")


# --- Helper: Inactivity Monitor ---
async def _inactivity_monitor(timeout_seconds: int):  # Uses globals _browser, _last_activity
    """Monitors browser inactivity and triggers shutdown if idle for too long."""
    check_interval = 60  # Check every 60 seconds
    logger.info(
        f"Inactivity monitor started. Timeout: {timeout_seconds}s, Check Interval: {check_interval}s."
    )
    while True:
        await asyncio.sleep(check_interval)
        browser_active = False
        try:
            # Safely check browser status under lock
            async with _playwright_lock:
                if _browser is not None and _browser.is_connected():
                    browser_active = True
        except Exception as check_err:
            logger.warning(f"Error checking browser status in inactivity monitor: {check_err}")
            # Assume active or handle error? Assume active to avoid premature shutdown on check error.
            browser_active = True  # Or consider stopping monitor?

        if not browser_active:
            logger.info("Inactivity monitor: Browser is closed or disconnected. Stopping monitor.")
            break  # Exit monitor loop if browser is gone

        # Calculate idle time
        current_time = time.monotonic()
        idle_time = current_time - _last_activity

        logger.debug(
            f"Inactivity check: Idle time = {idle_time:.1f}s (Timeout = {timeout_seconds}s)"
        )

        # Check if idle time exceeds timeout
        if idle_time > timeout_seconds:
            logger.info(
                f"Browser inactive for {idle_time:.1f}s (exceeds {timeout_seconds}s timeout). Initiating automatic shutdown."
            )
            # First stop this monitor task to prevent further shutdown attempts
            logger.info("Inactivity monitor stopped.")
            try:
                # Initiate shutdown (ensures it runs only once)
                await _initiate_shutdown()
            except Exception as e:
                # Log error during shutdown attempt, but break anyway
                logger.error(
                    f"Error during automatic shutdown initiated by inactivity monitor: {e}",
                    exc_info=True,
                )
            # Exit monitor loop after attempting shutdown
            break


@with_tool_metrics
@with_error_handling
async def search(query: str, engine: str = "bing", max_results: int = 10) -> Dict[str, Any]:
    """Performs a web search using the helper function and returns results."""
    # Ensure SB is initialized
    await _ensure_initialized()
    # Update activity timestamp
    _update_activity()

    # --- Input Validation ---
    if max_results <= 0:
        logger.warning(f"max_results was {max_results}. Setting to default 10.")
        max_results = 10
    # Engine validation happens within search_web helper

    # --- Execute Search ---
    # Call the underlying search_web helper function
    results = await search_web(query, engine=engine, max_results=max_results)
    result_count = len(results)

    # --- Return Result ---
    return {
        "success": True,
        "query": query,
        "engine": engine.lower(),  # Return normalized engine name
        "results": results,
        "result_count": result_count,
    }


@with_tool_metrics
@with_error_handling
async def download(  # This is the exported tool function
    url: str,
    target: Optional[Dict[str, Any]] = None,
    task_hint: Optional[str] = None,
    dest_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Navigates, clicks (using hint/target) to download, saves file, returns info."""
    # Ensure SB is initialized
    await _ensure_initialized()
    # Update activity timestamp
    _update_activity()

    # --- Input Validation: Determine task_hint ---
    effective_task_hint = task_hint
    if not effective_task_hint:  # Generate hint if missing
        if target and (target.get("name") or target.get("role")):
            name = target.get("name", "")
            role = target.get("role", "")  # Default role empty if not specified
            hint_base = "Download link/button"
            target_desc = f"{name or role}".strip()  # Use name or role
            if target_desc:
                effective_task_hint = f"{hint_base} '{target_desc}'"
            else:
                effective_task_hint = hint_base  # Fallback if target has no name/role
            logger.debug(f"download tool generated task_hint: '{effective_task_hint}'")
        else:
            raise ToolInputError(
                "download tool requires 'task_hint', or 'target' dict containing 'name' or 'role'."
            )

    # --- Get Context and Execute ---
    ctx, _ = await get_browser_context()
    async with _tab_context(ctx) as page:
        # Navigate to the page containing the download link
        await _log("download_navigate", url=url, hint=effective_task_hint)
        try:
            nav_timeout = 60000
            await page.goto(url, wait_until="networkidle", timeout=nav_timeout)
        except PlaywrightException as e:
            # Use f-string for cleaner message concatenation
            raise ToolError(
                f"Navigation failed for URL '{url}' before download attempt: {e}"
            ) from e

        # Call the underlying smart_download helper function
        # This helper now handles the click, waiting for download, saving, and analysis
        download_info = await smart_download(
            page,
            task_hint=effective_task_hint,
            dest_dir=dest_dir,  # Pass optional destination directory
            target_kwargs=target,  # Pass optional target details
        )

        # smart_download raises ToolError on failure, so this check is mostly redundant
        # but kept as a safeguard. The result structure is also slightly different now.
        if not download_info.get("success"):
            error_msg = download_info.get("error", "Download failed with unknown error.")
            raise ToolError(f"Download failed: {error_msg}", details=download_info)

        # Return success structure containing the download details
        return {"success": True, "download": download_info}


@with_tool_metrics
@with_error_handling
async def download_site_pdfs(
    start_url: str,
    dest_subfolder: Optional[str] = None,
    include_regex: Optional[str] = None,
    max_depth: int = 2,
    max_pdfs: int = 100,
    max_pages_crawl: int = 500,
    rate_limit_rps: float = 1.0,
) -> Dict[str, Any]:
    """Crawls site, finds PDFs, downloads them directly using httpx and FileSystemTool."""
    # Ensure SB is initialized
    await _ensure_initialized()
    # Update activity timestamp
    _update_activity()

    # --- Validate Inputs ---
    if not start_url:
        raise ToolInputError("start_url cannot be empty.")
    if max_depth < 0:
        raise ToolInputError("max_depth cannot be negative.")
    if max_pdfs <= 0:
        raise ToolInputError("max_pdfs must be positive.")
    if max_pages_crawl <= 0:
        raise ToolInputError("max_pages_crawl must be positive.")
    if rate_limit_rps <= 0:
        raise ToolInputError("rate_limit_rps must be positive.")

    # --- Prepare Download Directory ---
    final_dest_dir_str: Optional[str] = None
    try:
        # Generate a safe subfolder name from input or domain
        if dest_subfolder:
            safe_subfolder = _slugify(dest_subfolder, 50)
        else:
            try:
                parsed_start = urlparse(start_url)
                domain_slug = _slugify(parsed_start.netloc, 50)
                safe_subfolder = domain_slug or "downloaded_pdfs"  # Fallback if domain is empty
            except Exception:
                safe_subfolder = "downloaded_pdfs"  # Fallback on URL parse error

        # Define relative path within the main storage area
        dest_dir_relative_path = f"storage/smart_browser_site_pdfs/{safe_subfolder}"
        logger.info(
            f"Ensuring download directory exists for PDF crawl: '{dest_dir_relative_path}' using filesystem tool."
        )
        # Use STANDALONE create_directory tool
        create_dir_result = await create_directory(path=dest_dir_relative_path)
        if not isinstance(create_dir_result, dict) or not create_dir_result.get("success"):
            error_msg = (
                create_dir_result.get("error", "Unknown")
                if isinstance(create_dir_result, dict)
                else "Invalid response"
            )
            raise ToolError(
                f"Filesystem tool failed to create directory '{dest_dir_relative_path}'. Error: {error_msg}"
            )

        # Get the absolute path returned by the tool
        final_dest_dir_str = create_dir_result.get("path")
        if not final_dest_dir_str:
            raise ToolError(
                f"Filesystem tool create_directory succeeded for '{dest_dir_relative_path}' but did not return an absolute path."
            )
        logger.info(f"PDF crawl download directory confirmed/created at: {final_dest_dir_str}")
    except Exception as e:
        # Wrap directory preparation errors
        raise ToolError(
            f"Could not prepare download directory '{dest_dir_relative_path}': {str(e)}"
        ) from e

    # --- Crawl for PDF URLs ---
    logger.info(
        f"Starting PDF crawl from: {start_url} (Max Depth: {max_depth}, Max PDFs: {max_pdfs}, Max Pages: {max_pages_crawl})"
    )
    try:
        # Use the helper function to find PDF URLs
        pdf_urls = await crawl_for_pdfs(
            start_url,
            include_regex,
            max_depth,
            max_pdfs,
            max_pages_crawl,
            rate_limit_rps=5.0,  # Use slightly higher rate for crawl itself
        )
    except Exception as crawl_err:
        raise ToolError(
            f"Error during PDF crawl phase from '{start_url}': {crawl_err}"
        ) from crawl_err

    if not pdf_urls:
        logger.info("No matching PDF URLs found during crawl.")
        return {
            "success": True,
            "pdf_count": 0,
            "failed_count": 0,
            "dest_dir": final_dest_dir_str,
            "files": [],  # Empty list as no files were downloaded
        }

    # --- Download Found PDFs ---
    num_found = len(pdf_urls)
    logger.info(
        f"Found {num_found} PDF URLs. Starting downloads to '{final_dest_dir_str}' (Rate Limit: {rate_limit_rps}/s)..."
    )
    # Use the specified rate limit for downloads
    limiter = RateLimiter(rate_limit_rps)

    # Define the async task for downloading a single file
    async def download_task(url, seq):
        await limiter.acquire()  # Wait for rate limit permit
        # Use the direct download helper
        result = await _download_file_direct(url, final_dest_dir_str, seq)
        return result

    # Create and run download tasks concurrently
    download_tasks = []
    for i, url in enumerate(pdf_urls):
        task = asyncio.create_task(download_task(url, i + 1))
        download_tasks.append(task)

    results = await asyncio.gather(*download_tasks)  # Wait for all downloads

    # Process results
    successful_downloads = []
    failed_downloads = []
    for r in results:
        if isinstance(r, dict) and r.get("success"):
            successful_downloads.append(r)
        else:
            failed_downloads.append(r)  # Includes non-dict results or dicts with success=False

    num_successful = len(successful_downloads)
    num_failed = len(failed_downloads)

    # Log summary
    log_details = {
        "start_url": start_url,
        "found": num_found,
        "successful": num_successful,
        "failed": num_failed,
        "dest_dir": final_dest_dir_str,
    }
    if failed_downloads:
        # Log preview of failed download errors
        errors_preview = []
        for res in failed_downloads[:3]:  # Log first 3 errors
            err_url = res.get("url", "N/A") if isinstance(res, dict) else "N/A"
            err_msg = res.get("error", "Unknown error") if isinstance(res, dict) else str(res)
            errors_preview.append(f"{err_url}: {err_msg}")
        log_details["errors_preview"] = errors_preview
    await _log("download_site_pdfs_complete", **log_details)

    # Return final result
    return {
        "success": True,  # Overall tool execution success
        "pdf_count": num_successful,
        "failed_count": num_failed,
        "dest_dir": final_dest_dir_str,
        "files": results,  # Return list of all result dicts (success and failure)
    }


@with_tool_metrics
@with_error_handling
async def collect_documentation(
    package: str, max_pages: int = 40, rate_limit_rps: float = 2.0
) -> Dict[str, Any]:
    """Finds docs site, crawls, extracts text, saves using FileSystemTool."""
    # Ensure SB is initialized
    await _ensure_initialized()
    # Update activity timestamp
    _update_activity()

    # --- Validate Inputs ---
    if not package:
        raise ToolInputError("Package name cannot be empty.")
    if max_pages <= 0:
        raise ToolInputError("max_pages must be positive.")
    if rate_limit_rps <= 0:
        raise ToolInputError("rate_limit_rps must be positive.")

    # --- Find Documentation Root URL ---
    try:
        docs_root = await _pick_docs_root(package)
        if not docs_root:
            raise ToolError(
                f"Could not automatically find a likely documentation site for package '{package}'."
            )
    except Exception as e:
        # Wrap errors during root finding
        raise ToolError(f"Error finding documentation root for '{package}': {str(e)}") from e

    # --- Crawl Documentation Site ---
    logger.info(f"Found potential docs root: {docs_root}. Starting documentation crawl...")
    try:
        # Use the helper function to crawl and extract content
        pages_content = await crawl_docs_site(
            docs_root, max_pages=max_pages, rate_limit_rps=rate_limit_rps
        )
    except Exception as e:
        # Wrap errors during crawling
        raise ToolError(
            f"Error crawling documentation site starting from {docs_root}: {str(e)}"
        ) from e

    # Check if content was collected
    num_pages_collected = len(pages_content)
    if num_pages_collected == 0:
        logger.info(f"No readable content collected from documentation site for '{package}'.")
        return {
            "success": True,  # Tool ran successfully, but found no content
            "package": package,
            "pages_collected": 0,
            "file_path": None,  # No file saved
            "root_url": docs_root,
            "message": "No readable content pages were collected from the documentation site.",
        }
    logger.info(f"Collected readable content from {num_pages_collected} pages for '{package}'.")

    # --- Prepare Output Directory ---
    output_dir_relative_path = "storage/smart_browser_docs_collected"
    created_dir_path: Optional[str] = None
    try:
        logger.info(
            f"Ensuring documentation output directory exists: '{output_dir_relative_path}' using filesystem tool."
        )
        create_result = await create_directory(path=output_dir_relative_path)  # STANDALONE call
        if not isinstance(create_result, dict) or not create_result.get("success"):
            error_msg = (
                create_result.get("error", "Unknown")
                if isinstance(create_result, dict)
                else "Invalid response"
            )
            raise ToolError(
                f"Filesystem tool failed to create directory '{output_dir_relative_path}'. Error: {error_msg}"
            )
        created_dir_path = create_result.get("path")  # Get absolute path
        if not created_dir_path:
            raise ToolError(
                f"Filesystem tool create_directory for '{output_dir_relative_path}' did not return an absolute path."
            )
        logger.info(f"Ensured output directory exists at: '{created_dir_path}'")
    except Exception as e:
        # Wrap directory preparation errors
        raise ToolError(
            f"Could not prepare output directory '{output_dir_relative_path}': {str(e)}"
        ) from e

    # --- Format Content and Determine Filename ---
    # Create a unique filename based on package and timestamp
    now_utc_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_pkg_name = _slugify(package, 40)
    filename = f"{safe_pkg_name}_docs_{now_utc_str}.txt"
    # Construct relative path for writing (FS tool handles base path resolution)
    fpath_relative = f"{output_dir_relative_path}/{filename}"

    # Combine collected content into a single string
    separator = "\n\n" + ("=" * 80) + "\n\n"  # Separator between pages
    header = f"# Documentation for: {package}\n# Crawl Root: {docs_root}\n{separator}"
    combined_content = header
    try:
        page_texts = []
        for i, (url, text) in enumerate(pages_content):
            page_header = f"## Page {i + 1}: {str(url)}\n\n"
            page_body = str(text).strip()  # Ensure text is string and stripped
            page_texts.append(page_header + page_body)
        # Join all page sections with the separator
        combined_content += separator.join(page_texts)
    except Exception as e:
        # Handle potential errors during string formatting/joining
        raise ToolError(f"Error formatting collected documentation content: {str(e)}") from e

    # --- Write Combined Content using Filesystem Tool ---
    final_absolute_fpath: Optional[str] = None
    try:
        logger.info(f"Writing combined documentation content to relative path: {fpath_relative}")
        write_result = await write_file(
            path=fpath_relative, content=combined_content
        )  # STANDALONE call
        if not isinstance(write_result, dict) or not write_result.get("success"):
            error_msg = (
                write_result.get("error", "Unknown")
                if isinstance(write_result, dict)
                else "Invalid response"
            )
            raise ToolError(
                f"Filesystem tool failed to write documentation file '{fpath_relative}'. Error: {error_msg}"
            )

        # Get the absolute path where the file was actually written
        final_absolute_fpath = write_result.get("path")
        if not final_absolute_fpath:
            logger.warning(
                f"Filesystem tool write_file for '{fpath_relative}' did not return an absolute path. Using relative path in result."
            )
            final_absolute_fpath = fpath_relative  # Fallback for logging/return value

        logger.info(f"Successfully wrote combined documentation to: {final_absolute_fpath}")
    except Exception as e:
        # Wrap errors during file write
        raise ToolError(f"Could not write documentation file '{fpath_relative}': {str(e)}") from e

    # --- Log Success and Return Result ---
    await _log(
        "docs_collected_success",
        package=package,
        root=docs_root,
        pages=num_pages_collected,
        file=str(final_absolute_fpath),
    )
    return {
        "success": True,
        "package": package,
        "pages_collected": num_pages_collected,
        "file_path": str(final_absolute_fpath),  # Return the absolute path
        "root_url": docs_root,
        "message": f"Collected and saved content from {num_pages_collected} pages for '{package}'.",
    }


@with_tool_metrics
@with_error_handling
async def run_macro(  # Renamed from execute_macro to avoid confusion
    url: str,
    task: str,
    model: str = _llm_model_locator_global,
    max_rounds: int = 7,
    timeout_seconds: int = 600,
) -> Dict[str, Any]:
    """Navigates to URL and executes a natural language task using LLM planner and step runner."""
    # Ensure SB is initialized
    await _ensure_initialized()
    # Update activity timestamp
    _update_activity()

    # --- Input Validation ---
    if not url:
        raise ToolInputError("URL cannot be empty.")
    if not task:
        raise ToolInputError("Task description cannot be empty.")
    if max_rounds <= 0:
        raise ToolInputError("max_rounds must be positive.")
    if timeout_seconds <= 0:
        raise ToolInputError("timeout_seconds must be positive.")

    # Define the inner function to run with timeout
    async def run_macro_inner():
        ctx, _ = await get_browser_context()
        async with _tab_context(ctx) as page:
            # Navigate to the starting URL
            await _log("macro_navigate", url=url, task=task)
            try:
                nav_timeout = 60000
                await page.goto(url, wait_until="networkidle", timeout=nav_timeout)
            except PlaywrightException as e:
                # Use f-string for cleaner message
                raise ToolError(f"Navigation to '{url}' failed before starting macro: {e}") from e

            # Call the helper function that contains the plan-act loop
            # This helper handles planning, running steps, and logging rounds/errors
            step_results = await _run_macro_execution_loop(page, task, max_rounds, model)

            # Get final page state after macro execution
            final_state = {}  # Initialize as empty dict
            try:
                final_state = await get_page_state(page)
            except Exception as e:
                logger.error(f"Failed to get final page state after macro execution: {e}")
                final_state = {"error": f"Failed to get final page state: {e}"}

            # Determine overall macro success
            # Success if:
            # 1. A 'finish' step was executed successfully OR
            # 2. All steps executed (excluding wait/finish/extract?) succeeded.
            finished_successfully = any(
                s.get("action") == "finish" and s.get("success") for s in step_results
            )
            # Check if all non-finish/wait/extract steps succeeded (if any exist)
            all_other_steps_succeeded = True
            non_terminal_steps_exist = False
            for s in step_results:
                action = s.get("action")
                # Consider steps other than these potentially "passive" ones for failure check
                if action not in ("finish", "wait", "extract", "scroll", "error"):
                    non_terminal_steps_exist = True  # noqa: F841
                    if not s.get("success", False):
                        all_other_steps_succeeded = False
                        break  # Found a failed critical step

            # Macro succeeds if finished explicitly or if all critical steps passed (and at least one step ran)
            macro_success = finished_successfully or (
                bool(step_results) and all_other_steps_succeeded
            )

            # Return final results
            return {
                "success": macro_success,
                "task": task,
                "steps": step_results,  # List of results for each step executed
                "final_page_state": final_state,
            }

    # Run the inner function with an overall timeout
    try:
        result = await asyncio.wait_for(run_macro_inner(), timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        # Handle overall macro timeout
        await _log("macro_timeout", url=url, task=task, timeout=timeout_seconds)
        return {
            "success": False,
            "task": task,
            "error": f"Macro execution timed out after {timeout_seconds}s.",
            "steps": [],  # No steps completed within timeout (or results lost)
            "final_page_state": {"error": "Macro timed out"},
        }


async def _run_macro_execution_loop(
    page: Page, task: str, max_rounds: int, model: str
) -> List[Dict[str, Any]]:
    """Internal helper containing the plan-and-execute loop for run_macro."""
    all_step_results: List[Dict[str, Any]] = []
    current_task_description = task  # Initial task

    for i in range(max_rounds):
        round_num = i + 1
        logger.info(f"--- Macro Round {round_num}/{max_rounds} ---")
        task_preview = current_task_description[:100] + (
            "..." if len(current_task_description) > 100 else ""
        )
        logger.info(f"Current Task: {task_preview}")

        try:
            # 1. Get Current Page State
            logger.debug(f"Macro Round {round_num}: Getting page state...")
            state = await get_page_state(page)
            if "error" in state:  # Handle error getting state
                error_msg = (
                    f"Failed to get page state before planning round {round_num}: {state['error']}"
                )
                logger.error(error_msg)
                # Add error step and stop
                all_step_results.append(
                    {"action": "error", "success": False, "error": error_msg, "round": round_num}
                )
                return all_step_results

            # 2. Plan Next Steps using LLM
            logger.debug(f"Macro Round {round_num}: Planning steps with LLM...")
            plan = await _plan_macro(state, current_task_description, model)
            await _log(
                "macro_plan_generated",
                round=round_num,
                task=current_task_description,
                plan_length=len(plan),
                plan_preview=plan[:2],
            )

            # Check if plan is empty (task complete or impossible)
            if not plan:
                logger.info(
                    f"Macro Round {round_num}: Planner returned empty plan. Assuming task complete or impossible."
                )
                await _log("macro_plan_empty", round=round_num, task=current_task_description)
                break  # Exit loop

            # 3. Execute Planned Steps
            logger.info(f"Macro Round {round_num}: Executing {len(plan)} planned steps...")
            step_results_this_round = await run_steps(page, plan)
            all_step_results.extend(step_results_this_round)  # Add results to overall list

            # 4. Check Round Outcome
            finished_this_round = any(
                s.get("action") == "finish" and s.get("success") for s in step_results_this_round
            )
            last_step_failed = False
            if step_results_this_round:
                last_step = step_results_this_round[-1]
                # Check if the *last* step failed and wasn't a passive action
                is_passive_action = last_step.get("action") in (
                    "wait",
                    "finish",
                    "extract",
                    "scroll",
                    "error",
                )
                if not last_step.get("success", False) and not is_passive_action:
                    last_step_failed = True
                    error_info = last_step.get("error", "?")
                    failed_action = last_step.get("action", "?")
                    await _log(
                        "macro_fail_step", round=round_num, action=failed_action, error=error_info
                    )
                    logger.warning(
                        f"Macro Round {round_num} stopped due to failed critical step: Action='{failed_action}', Error='{error_info}'"
                    )

            # Exit loop if 'finish' action succeeded or last critical step failed
            if finished_this_round:
                await _log("macro_finish_action", round=round_num)
                logger.info(
                    f"Macro finished successfully via 'finish' action in round {round_num}."
                )
                return all_step_results  # Return immediately after successful finish
            if last_step_failed:
                logger.info(f"Stopping macro execution after failed step in round {round_num}.")
                return all_step_results  # Return results up to the failure

            # If loop continues, update task description for next round?
            # (Currently, task description remains the same throughout)
            # current_task_description = "Refine based on results..." # Example modification point

        except ToolError as e:
            # Handle errors during planning or state retrieval specifically
            await _log(
                "macro_error_tool", round=round_num, task=current_task_description, error=str(e)
            )
            logger.error(f"Macro Round {round_num} failed with ToolError: {e}")
            all_step_results.append(
                {
                    "action": "error",
                    "success": False,
                    "error": f"ToolError in Round {round_num}: {e}",
                    "round": round_num,
                }
            )
            return all_step_results  # Stop execution on tool errors
        except Exception as e:
            # Handle unexpected errors during the round
            await _log(
                "macro_error_unexpected",
                round=round_num,
                task=current_task_description,
                error=str(e),
            )
            logger.error(f"Macro Round {round_num} failed unexpectedly: {e}", exc_info=True)
            all_step_results.append(
                {
                    "action": "error",
                    "success": False,
                    "error": f"Unexpected Error in Round {round_num}: {e}",
                    "round": round_num,
                }
            )
            return all_step_results  # Stop execution on unexpected errors

    # If loop finishes due to max_rounds
    await _log("macro_exceeded_rounds", max_rounds=max_rounds, task=task)
    logger.warning(f"Macro stopped after reaching maximum rounds ({max_rounds}) for task: {task}")
    return all_step_results  # Return all collected results


@with_tool_metrics
@with_error_handling
async def autopilot(
    task: str,
    scratch_subdir: str = "autopilot_runs",
    max_steps: int = 10,
    timeout_seconds: int = 1800,
) -> Dict[str, Any]:
    """Executes a complex multi-step task using LLM planning and available tools."""
    # Ensure SB is initialized
    await _ensure_initialized()

    # --- Validate Inputs ---
    if not task:
        raise ToolInputError("Task description cannot be empty.")
    if max_steps <= 0:
        raise ToolInputError("max_steps must be positive.")
    if timeout_seconds <= 0:
        raise ToolInputError("timeout_seconds must be positive.")

    # --- Prepare Scratch Directory and Logging ---
    final_scratch_dir_str: Optional[str] = None
    log_path: Optional[Path] = None
    try:
        # Define base path for scratch files
        scratch_base_relative = "storage/smart_browser_scratch"
        # Sanitize user-provided subdir name
        safe_subdir = _slugify(scratch_subdir, 50) or "autopilot_run"  # Fallback name
        scratch_dir_relative_path = f"{scratch_base_relative}/{safe_subdir}"

        logger.info(
            f"Ensuring autopilot scratch directory exists: '{scratch_dir_relative_path}' using filesystem tool."
        )
        # Use STANDALONE create_directory tool
        create_dir_result = await create_directory(path=scratch_dir_relative_path)
        if not isinstance(create_dir_result, dict) or not create_dir_result.get("success"):
            error_msg = (
                create_dir_result.get("error", "Unknown")
                if isinstance(create_dir_result, dict)
                else "Invalid response"
            )
            raise ToolError(
                f"Filesystem tool failed to create scratch directory '{scratch_dir_relative_path}'. Error: {error_msg}"
            )

        # Get the absolute path
        final_scratch_dir_str = create_dir_result.get("path")
        if not final_scratch_dir_str:
            raise ToolError(
                f"Filesystem tool create_directory for '{scratch_dir_relative_path}' did not return an absolute path."
            )
        final_scratch_dir_path = Path(final_scratch_dir_str)
        logger.info(f"Autopilot scratch directory confirmed/created at: {final_scratch_dir_path}")

        # Prepare log file path within the scratch directory
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_filename = f"autopilot_run_{run_id}.jsonl"
        log_path = final_scratch_dir_path / log_filename
        logger.info(f"Autopilot run '{run_id}' started. Execution log: {log_path}")

    except Exception as e:
        # Wrap directory preparation errors
        raise ToolError(
            f"Could not prepare scratch directory '{scratch_dir_relative_path}': {str(e)}"
        ) from e

    # Define the inner function to run with timeout
    async def autopilot_inner():
        all_results: List[Dict] = []  # Stores results of each step
        current_task_description = task  # Initial task

        try:
            # --- Initial Planning ---
            logger.info("Autopilot: Generating initial plan...")
            current_plan = await _plan_autopilot(
                current_task_description, None
            )  # Initial plan has no prior results
            step_num = 0

            # --- Execution Loop ---
            while step_num < max_steps and current_plan:
                step_num += 1
                step_to_execute = current_plan[0]  # Get the next step
                tool_name = step_to_execute.get("tool")
                args = step_to_execute.get("args", {})
                # Initialize log entry for this step
                step_log = {
                    "step": step_num,
                    "tool": tool_name,
                    "args": args,
                    "success": False,
                    "result": None,
                    "error": None,
                }
                logger.info(
                    f"--- Autopilot Step {step_num}/{max_steps}: Executing Tool '{tool_name}' ---"
                )
                logger.debug(f"Step {step_num} Args: {args}")

                # Validate tool exists
                if tool_name not in _AVAILABLE_TOOLS:
                    error_msg = f"Planner selected unknown tool '{tool_name}'."
                    step_log["error"] = error_msg
                    logger.error(error_msg)
                    current_plan = []  # Stop execution if tool is unknown
                else:
                    # --- Tool Lookup and Execution ---
                    method_name = _AVAILABLE_TOOLS[tool_name][0]  # Get function name string
                    # Look up the actual function object
                    tool_func = globals().get(method_name)  # Check current module globals first
                    if not tool_func or not callable(tool_func):
                        # Try external tool lookups if not found locally
                        tool_func = _get_filesystem_tool(method_name) or _get_completion_tool(
                            method_name
                        )

                    if not tool_func or not callable(tool_func):
                        # Tool function implementation not found
                        error_msg = f"Internal error: Could not find function implementation for tool '{tool_name}' (expected function: '{method_name}')."
                        step_log["error"] = error_msg
                        logger.error(error_msg)
                        current_plan = []  # Stop execution
                    else:
                        # --- Execute the Found Tool Function ---
                        try:
                            await _log(
                                "autopilot_step_start", step=step_num, tool=tool_name, args=args
                            )
                            _update_activity()  # Update activity before long tool call
                            # Call the standalone tool function with its arguments
                            outcome = await tool_func(**args)
                            _update_activity()  # Update activity after tool call returns

                            # Record outcome in step log
                            step_log["success"] = outcome.get("success", False)
                            step_log["result"] = outcome  # Store the full result dict

                            # --- Plan for Next Step (or Replan on Failure) ---
                            if step_log["success"]:
                                await _log(
                                    "autopilot_step_success",
                                    step=step_num,
                                    tool=tool_name,
                                    result_summary=str(outcome)[:200],
                                )
                                logger.info(
                                    f"Autopilot Step {step_num} ({tool_name}) completed successfully."
                                )
                                # Remove completed step and plan next based on success
                                current_plan.pop(0)  # Remove executed step
                                if current_plan:  # If plan wasn't just one step
                                    logger.debug("Plan has remaining steps, continuing...")
                                elif not current_plan:  # Plan is now empty after successful step
                                    logger.info(
                                        "Autopilot: Attempting to generate next plan step..."
                                    )
                                    try:
                                        current_plan = await _plan_autopilot(
                                            current_task_description, all_results + [step_log]
                                        )
                                        plan_count = len(current_plan)
                                        logger.info(f"Generated next plan ({plan_count} step(s)).")
                                        await _log(
                                            "autopilot_replan_success",
                                            reason="step_complete",
                                            new_steps=plan_count,
                                        )
                                    except Exception as replan_err:
                                        logger.error(
                                            f"Autopilot replanning after step success failed: {replan_err}"
                                        )
                                        await _log(
                                            "autopilot_replan_fail",
                                            reason="step_complete",
                                            error=str(replan_err),
                                        )
                                        current_plan = []  # Stop if replanning fails
                            else:
                                # Step failed
                                step_log["error"] = outcome.get(
                                    "error", f"Tool '{tool_name}' failed without specific error."
                                )
                                await _log(
                                    "autopilot_step_fail",
                                    step=step_num,
                                    tool=tool_name,
                                    error=step_log["error"],
                                )
                                logger.warning(
                                    f"Autopilot Step {step_num} ({tool_name}) failed: {step_log['error']}"
                                )
                                logger.info(f"Attempting replan after failed step {step_num}...")
                                try:
                                    # Replan based on the failure
                                    new_plan_tail = await _plan_autopilot(
                                        current_task_description, all_results + [step_log]
                                    )
                                    current_plan = new_plan_tail  # Replace old plan with new one
                                    plan_count = len(current_plan)
                                    logger.info(
                                        f"Replanning successful after failure. New plan has {plan_count} step(s)."
                                    )
                                    await _log(
                                        "autopilot_replan_success",
                                        reason="step_fail",
                                        new_steps=plan_count,
                                    )
                                except Exception as replan_err:
                                    logger.error(
                                        f"Autopilot replanning after step failure failed: {replan_err}"
                                    )
                                    await _log(
                                        "autopilot_replan_fail",
                                        reason="step_fail",
                                        error=str(replan_err),
                                    )
                                    current_plan = []  # Stop if replanning fails

                        except (
                            ToolInputError,
                            ToolError,
                            ValueError,
                            TypeError,
                            AssertionError,
                        ) as e:
                            # Catch errors *during* tool execution (e.g., bad args passed validation but failed in tool)
                            error_msg = f"{type(e).__name__} executing '{tool_name}': {e}"
                            step_log["error"] = error_msg
                            step_log["success"] = False
                            logger.error(
                                f"Autopilot Step {step_num} ({tool_name}) execution failed: {error_msg}",
                                exc_info=True,
                            )
                            current_plan = []  # Stop execution on tool error
                        except Exception as e:
                            # Catch unexpected errors during tool execution
                            error_msg = f"Unexpected error executing '{tool_name}': {e}"
                            step_log["error"] = error_msg
                            step_log["success"] = False
                            logger.critical(
                                f"Autopilot Step {step_num} ({tool_name}) failed unexpectedly: {error_msg}",
                                exc_info=True,
                            )
                            current_plan = []  # Stop execution

                # Append the result of this step to the overall results
                all_results.append(step_log)
                # --- Log Step Result to File ---
                if log_path:
                    try:
                        log_entry = (
                            json.dumps(step_log, default=str) + "\n"
                        )  # Use default=str for non-serializable types
                        async with aiofiles.open(log_path, "a", encoding="utf-8") as log_f:
                            await log_f.write(log_entry)
                    except IOError as log_e:
                        logger.error(f"Failed to write autopilot step log to {log_path}: {log_e}")
                    except Exception as json_e:
                        logger.error(f"Failed to serialize step log for writing: {json_e}")

            # --- Loop End Conditions ---
            if step_num >= max_steps:
                logger.warning(f"Autopilot stopped: Reached maximum step limit ({max_steps}).")
                await _log("autopilot_max_steps", task=task, steps=step_num)
            elif not current_plan and step_num > 0:
                # Plan became empty (either task finished or replan failed/returned empty)
                final_step_success = all_results[-1].get("success", False) if all_results else False
                if final_step_success:
                    logger.info(f"Autopilot plan complete after {step_num} steps.")
                    await _log("autopilot_plan_end", task=task, steps=step_num, status="completed")
                else:
                    logger.warning(
                        f"Autopilot stopped after {step_num} steps due to failure or inability to plan next step."
                    )
                    await _log(
                        "autopilot_plan_end", task=task, steps=step_num, status="failed_or_stuck"
                    )
            elif step_num == 0:
                # Initial plan was empty
                logger.warning("Autopilot: Initial plan was empty. No steps executed.")
                await _log("autopilot_plan_end", task=task, steps=0, status="no_plan")

            # Determine overall success based on the success of the *last* executed step
            overall_success = bool(all_results) and all_results[-1].get("success", False)
            # Return final summary
            return {
                "success": overall_success,
                "steps_executed": step_num,
                "run_log": str(log_path) if log_path else None,
                "final_results": all_results[-3:],  # Return summary of last few steps
            }
        except Exception as autopilot_err:
            # Catch critical errors during planning or loop setup
            logger.critical(
                f"Autopilot run failed critically before or during execution loop: {autopilot_err}",
                exc_info=True,
            )
            await _log("autopilot_critical_error", task=task, error=str(autopilot_err))
            # Log error to file if possible
            error_entry = {
                "step": 0,
                "success": False,
                "error": f"Autopilot critical failure: {autopilot_err}",
            }
            if log_path:
                try:
                    log_entry = json.dumps(error_entry, default=str) + "\n"
                    async with aiofiles.open(log_path, "a", encoding="utf-8") as log_f:
                        await log_f.write(log_entry)
                except Exception as final_log_e:
                    logger.error(
                        f"Failed to write final critical error log to {log_path}: {final_log_e}"
                    )
            # Raise ToolError to indicate autopilot failure
            raise ToolError(f"Autopilot failed critically: {autopilot_err}") from autopilot_err

    # --- Run with Timeout ---
    try:
        result = await asyncio.wait_for(autopilot_inner(), timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        error_msg = f"Autopilot execution timed out after {timeout_seconds}s."
        logger.error(error_msg)
        await _log("autopilot_timeout", task=task, timeout=timeout_seconds)
        # Log timeout to file if possible
        if log_path:
            try:
                timeout_entry = {"step": -1, "success": False, "error": error_msg}
                log_entry = json.dumps(timeout_entry, default=str) + "\n"
                async with aiofiles.open(log_path, "a", encoding="utf-8") as log_f:
                    await log_f.write(log_entry)
            except Exception as timeout_log_e:
                logger.error(f"Failed to write timeout log entry to {log_path}: {timeout_log_e}")
        # Return timeout failure
        return {
            "success": False,
            "error": error_msg,
            "steps_executed": -1,  # Indicate timeout before completion
            "run_log": str(log_path) if log_path else None,
            "final_results": [],  # No final results available on timeout
        }
