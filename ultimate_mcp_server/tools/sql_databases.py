# ultimate_mcp_server/tools/sql_databases.py
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# --- START: Expanded typing imports ---
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# --- END: Expanded typing imports ---
# --- Removed BaseTool import ---
# SQLAlchemy imports
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

# Local imports
from ultimate_mcp_server.exceptions import ToolError, ToolInputError

# --- START: Expanded base imports ---
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics

# --- END: Expanded base imports ---
from ultimate_mcp_server.tools.completion import generate_completion  # For NL→SQL
from ultimate_mcp_server.utils import get_logger

# Optional imports with graceful fallbacks
try:
    import boto3  # For AWS Secrets Manager
except ImportError:
    boto3 = None

try:
    import hvac  # For HashiCorp Vault
except ImportError:
    hvac = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pandera as pa
except ImportError:
    pa = None

try:
    import prometheus_client as prom
except ImportError:
    prom = None

logger = get_logger("ultimate_mcp_server.tools.sql_databases")

# =============================================================================
# Global State and Configuration (Replaces instance variables)
# =============================================================================


# --- Connection Management ---
class ConnectionManager:
    """Manages database connections with automatic cleanup after inactivity."""

    # (Keep ConnectionManager class as is - it's a helper utility)
    def __init__(self, cleanup_interval_seconds=600, check_interval_seconds=60):
        self.connections: Dict[str, Tuple[AsyncEngine, float]] = {}
        self.cleanup_interval = cleanup_interval_seconds
        self.check_interval = check_interval_seconds
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()  # Added lock for thread-safe modifications

    async def start_cleanup_task(self):
        async with self._lock:
            cleanup_task_is_none = self._cleanup_task is None
            cleanup_task_is_done = self._cleanup_task is not None and self._cleanup_task.done()
            if cleanup_task_is_none or cleanup_task_is_done:
                try:
                    loop = asyncio.get_running_loop()
                    task_coro = self._cleanup_loop()
                    self._cleanup_task = loop.create_task(task_coro)
                    logger.info("Started connection cleanup task.")
                except RuntimeError:
                    logger.warning("No running event loop found, cleanup task not started.")

    async def _cleanup_loop(self):
        log_msg = f"Cleanup loop started. Check interval: {self.check_interval}s, Inactivity threshold: {self.cleanup_interval}s"
        logger.debug(log_msg)
        while True:
            await asyncio.sleep(self.check_interval)
            try:
                await self.cleanup_inactive_connections()
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled.")
                break  # Exit loop cleanly on cancellation
            except Exception as e:
                logger.error(f"Error during connection cleanup: {e}", exc_info=True)

    async def cleanup_inactive_connections(self):
        current_time = time.time()
        conn_ids_to_close = []

        # Need lock here as we iterate over potentially changing dict
        async with self._lock:
            # Use items() for safe iteration while potentially modifying dict later
            # Create a copy to avoid issues if the dict is modified elsewhere concurrently (though unlikely with lock)
            current_connections = list(self.connections.items())

        for conn_id, (_engine, last_accessed) in current_connections:
            idle_time = current_time - last_accessed
            is_inactive = idle_time > self.cleanup_interval
            if is_inactive:
                log_msg = f"Connection {conn_id} exceeded inactivity timeout ({idle_time:.1f}s > {self.cleanup_interval}s)"
                logger.info(log_msg)
                conn_ids_to_close.append(conn_id)

        closed_count = 0
        for conn_id in conn_ids_to_close:
            # close_connection acquires its own lock
            closed = await self.close_connection(conn_id)
            if closed:
                logger.info(f"Auto-closed inactive connection: {conn_id}")
                closed_count += 1
        if closed_count > 0:
            logger.debug(f"Closed {closed_count} inactive connections.")
        elif conn_ids_to_close:
            num_attempted = len(conn_ids_to_close)
            logger.debug(
                f"Attempted to close {num_attempted} connections, but they might have been removed already."
            )

    async def get_connection(self, conn_id: str) -> AsyncEngine:
        async with self._lock:
            if conn_id not in self.connections:
                details = {"error_type": "CONNECTION_NOT_FOUND"}
                raise ToolInputError(
                    f"Unknown connection_id: {conn_id}", param_name="connection_id", details=details
                )

            engine, _ = self.connections[conn_id]
            # Update last accessed time
            current_time = time.time()
            self.connections[conn_id] = (engine, current_time)
            logger.debug(f"Accessed connection {conn_id}, updated last accessed time.")
            return engine

    async def add_connection(self, conn_id: str, engine: AsyncEngine):
        # close_connection handles locking internally
        has_existing = conn_id in self.connections
        if has_existing:
            logger.warning(f"Overwriting existing connection entry for {conn_id}.")
            await self.close_connection(conn_id)  # Close the old one first

        async with self._lock:
            current_time = time.time()
            self.connections[conn_id] = (engine, current_time)
        url_str = str(engine.url)
        url_prefix = url_str.split("@")[0]
        log_msg = (
            f"Added connection {conn_id} for URL: {url_prefix}..."  # Avoid logging credentials
        )
        logger.info(log_msg)
        await self.start_cleanup_task()  # Ensure cleanup is running

    async def close_connection(self, conn_id: str) -> bool:
        engine = None
        async with self._lock:
            connection_exists = conn_id in self.connections
            if connection_exists:
                engine, _ = self.connections.pop(conn_id)
            else:
                logger.warning(f"Attempted to close non-existent connection ID: {conn_id}")
                return False  # Not found

        if engine:
            logger.info(f"Closing connection {conn_id}...")
            try:
                await engine.dispose()
                logger.info(f"Connection {conn_id} disposed successfully.")
                return True
            except Exception as e:
                log_msg = f"Error disposing engine for connection {conn_id}: {e}"
                logger.error(log_msg, exc_info=True)
                # Removed from dict, but disposal failed
                return False
        return False  # Should not be reached if found

    async def shutdown(self):
        logger.info("Shutting down Connection Manager...")
        # Cancel cleanup task first
        cleanup_task = None
        async with self._lock:
            task_exists = self._cleanup_task is not None
            task_not_done = task_exists and not self._cleanup_task.done()
            if task_exists and task_not_done:
                cleanup_task = self._cleanup_task  # Get reference before clearing
                self._cleanup_task = None  # Prevent restarting

        if cleanup_task:
            cleanup_task.cancel()
            try:
                # Add timeout for task cancellation
                await asyncio.wait_for(cleanup_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Cleanup task cancellation timed out after 2 seconds")
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled.")
            except Exception as e:
                logger.error(f"Error stopping cleanup task: {e}", exc_info=True)

        # Close remaining connections
        async with self._lock:
            conn_ids = list(self.connections.keys())

        if conn_ids:
            num_conns = len(conn_ids)
            logger.info(f"Closing {num_conns} active connections...")
            # Call close_connection which handles locking and removal
            close_tasks = []
            for conn_id in conn_ids:
                # Create a task that times out for each connection
                async def close_with_timeout(conn_id):
                    try:
                        await asyncio.wait_for(self.close_connection(conn_id), timeout=2.0)
                        return True
                    except asyncio.TimeoutError:
                        logger.warning(f"Connection {conn_id} close timed out after 2 seconds")
                        return False

                close_tasks.append(close_with_timeout(conn_id))

            # Wait for all connections to close with an overall timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*close_tasks, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some connections did not close within the 5 second timeout")

        async with self._lock:
            # Final check
            remaining = len(self.connections)
            if remaining > 0:
                logger.warning(f"{remaining} connections still remain after shutdown attempt.")
            self.connections.clear()  # Clear the dictionary

        logger.info("Connection Manager shutdown complete.")


_connection_manager = ConnectionManager()

# --- Security and Validation ---
_PROHIBITED_SQL_PATTERN = r"""^\s*(DROP\s+(TABLE|DATABASE|INDEX|VIEW|FUNCTION|PROCEDURE|USER|ROLE)|
             TRUNCATE\s+TABLE|
             DELETE\s+FROM|
             ALTER\s+(TABLE|DATABASE)\s+\S+\s+DROP\s+|
             UPDATE\s+|INSERT\s+INTO(?!\s+OR\s+IGNORE)|
             GRANT\s+|REVOKE\s+|
             CREATE\s+USER|ALTER\s+USER|DROP\s+USER|
             CREATE\s+ROLE|ALTER\s+ROLE|DROP\s+ROLE|
             SHUTDOWN|REBOOT|RESTART)"""
_PROHIBITED_SQL_REGEX = re.compile(_PROHIBITED_SQL_PATTERN, re.I | re.X)

_TABLE_RX = re.compile(r"\b(?:FROM|JOIN|UPDATE|INSERT\s+INTO|DELETE\s+FROM)\s+([\w.\"$-]+)", re.I)


# --- Masking ---
@dataclass
class MaskRule:
    rx: re.Pattern
    repl: Union[str, callable]


# Helper lambda for credit card masking
def _mask_cc(v: str) -> str:
    return f"XXXX-...-{v[-4:]}"


# Helper lambda for email masking
def _mask_email(v: str) -> str:
    if "@" in v:
        parts = v.split("@")
        prefix = parts[0][:2] + "***"
        domain = parts[-1]
        return f"{prefix}@{domain}"
    else:
        return "***"


_MASKING_RULES = [
    MaskRule(re.compile(r"^\d{3}-\d{2}-\d{4}$"), "***-**-XXXX"),  # SSN
    MaskRule(re.compile(r"(\b\d{4}-?){3}\d{4}\b"), _mask_cc),  # CC basic mask
    MaskRule(re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"), _mask_email),  # Email
]

# --- ACLs ---
_RESTRICTED_TABLES: Set[str] = set()
_RESTRICTED_COLUMNS: Set[str] = set()

# --- Auditing ---
_AUDIT_LOG: List[Dict[str, Any]] = []
_AUDIT_ID_COUNTER: int = 0
_audit_lock = asyncio.Lock()  # Lock for modifying audit counter and log

# --- Schema Drift Detection ---
_LINEAGE: List[Dict[str, Any]] = []
_SCHEMA_VERSIONS: Dict[str, str] = {}  # connection_id -> schema_hash

# --- Prometheus Metrics ---
# Initialized as None, populated in initialize function if prom is available
_Q_CNT: Optional[Any] = None
_Q_LAT: Optional[Any] = None
_CONN_GAUGE: Optional[Any] = None


# =============================================================================
# Initialization and Shutdown Functions
# =============================================================================

# Flag to track if metrics have been initialized
_sql_metrics_initialized = False


async def initialize_sql_tools():
    """Initialize global state for SQL tools, like starting the cleanup task and metrics."""
    global _sql_metrics_initialized
    global _Q_CNT, _Q_LAT, _CONN_GAUGE  # Ensure globals are declared for assignment

    # Initialize metrics only once
    if not _sql_metrics_initialized:
        logger.info("Initializing SQL Tools module metrics...")
        if prom:
            try:
                # Define metrics
                _Q_CNT = prom.Counter(
                    "mcp_sqltool_calls", "SQL tool calls", ["tool", "action", "db"]
                )
                latency_buckets = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)
                _Q_LAT = prom.Histogram(
                    "mcp_sqltool_latency_seconds",
                    "SQL latency",
                    ["tool", "action", "db"],
                    buckets=latency_buckets,
                )
                _CONN_GAUGE = prom.Gauge(
                    "mcp_sqltool_active_connections", "Number of active SQL connections"
                )

                # Define the gauge function referencing the global manager
                # Wrap in try-except as accessing length during shutdown might be tricky
                def _get_active_connections():
                    try:
                        # Access length directly if manager state is simple enough
                        # If complex state, acquire lock if necessary (_connection_manager._lock)
                        # For just length, direct access is usually okay unless adding/removing heavily concurrent
                        return len(_connection_manager.connections)
                    except Exception:
                        logger.exception("Error getting active connection count for Prometheus.")
                        return 0  # Default to 0 if error accessing

                _CONN_GAUGE.set_function(_get_active_connections)
                logger.info("Prometheus metrics initialized for SQL tools.")
                _sql_metrics_initialized = True  # Set flag only after successful initialization

            except ValueError as e:
                # Catch the specific duplicate error and log nicely, but don't crash
                if "Duplicated timeseries" in str(e):
                    logger.warning(
                        f"Prometheus metrics already registered: {e}. Skipping re-initialization."
                    )
                    _sql_metrics_initialized = (
                        True  # Assume they are initialized if duplicate error occurs
                    )
                else:
                    # Re-raise other ValueErrors
                    logger.error(
                        f"ValueError during Prometheus metric initialization: {e}", exc_info=True
                    )
                    raise  # Re-raise unexpected ValueError
            except Exception as e:
                logger.error(
                    f"Failed to initialize Prometheus metrics for SQL tools: {e}", exc_info=True
                )
                # Continue without metrics if initialization fails? Or raise? Let's continue for now.

        else:
            logger.info("Prometheus client not available, metrics disabled for SQL tools.")
            _sql_metrics_initialized = (
                True  # Mark as "initialized" (i.e., done trying) even if prom not present
            )
    else:
        logger.debug("SQL tools metrics already initialized, skipping metric creation.")

    # Always try to start the cleanup task (it's internally idempotent)
    # Ensure this happens *after* logging initialization attempt
    logger.info("Ensuring SQL connection cleanup task is running...")
    await _connection_manager.start_cleanup_task()


async def shutdown_sql_tools():
    """Gracefully shut down SQL tool resources, like the connection manager."""
    logger.info("Shutting down SQL Tools module...")
    try:
        # Add timeout to connection manager shutdown
        await asyncio.wait_for(_connection_manager.shutdown(), timeout=8.0)
    except asyncio.TimeoutError:
        logger.warning("Connection Manager shutdown timed out after 8 seconds")
    # Clear other global state if necessary (e.g., save audit log)
    logger.info("SQL Tools module shutdown complete.")


# =============================================================================
# Helper Functions (Private module-level functions)
# =============================================================================


@lru_cache(maxsize=64)
def _pull_secret_from_sources(name: str) -> str:
    """Retrieve a secret from various sources."""
    # (Implementation remains the same as in the original class)
    if boto3:
        try:
            client = boto3.client("secretsmanager")
            # Consider region_name=os.getenv("AWS_REGION") or similar config
            secret_value_response = client.get_secret_value(SecretId=name)
            # Handle binary vs string secrets
            if "SecretString" in secret_value_response:
                secret = secret_value_response["SecretString"]
                return secret
            elif "SecretBinary" in secret_value_response:
                # Decode binary appropriately if needed, default to utf-8
                secret_bytes = secret_value_response["SecretBinary"]
                secret = secret_bytes.decode("utf-8")
                return secret
        except Exception as aws_err:
            logger.debug(f"Secret '{name}' not found or error in AWS Secrets Manager: {aws_err}")
            pass

    if hvac:
        try:
            vault_url = os.getenv("VAULT_ADDR")
            vault_token = os.getenv("VAULT_TOKEN")
            if vault_url and vault_token:
                vault_client = hvac.Client(
                    url=vault_url, token=vault_token, timeout=2
                )  # Short timeout
                is_auth = vault_client.is_authenticated()
                if is_auth:
                    mount_point = os.getenv("VAULT_KV_MOUNT_POINT", "secret")
                    secret_path = name
                    read_response = vault_client.secrets.kv.v2.read_secret_version(
                        path=secret_path, mount_point=mount_point
                    )
                    # Standard KV v2 structure: response['data']['data'] is the dict of secrets
                    has_outer_data = "data" in read_response
                    has_inner_data = has_outer_data and "data" in read_response["data"]
                    if has_inner_data:
                        # Try common key names 'value' or the secret name itself
                        secret_data = read_response["data"]["data"]
                        if "value" in secret_data:
                            value = secret_data["value"]
                            return value
                        elif name in secret_data:
                            value = secret_data[name]
                            return value
                        else:
                            log_msg = f"Secret keys 'value' or '{name}' not found at path '{secret_path}' in Vault."
                            logger.debug(log_msg)
                else:
                    logger.warning(f"Vault authentication failed for address: {vault_url}")

        except Exception as e:
            logger.debug(f"Error accessing Vault for secret '{name}': {e}")
            pass

    # Try environment variables
    env_val_direct = os.getenv(name)
    if env_val_direct:
        return env_val_direct

    mcp_secret_name = f"MCP_SECRET_{name.upper()}"
    env_val_prefixed = os.getenv(mcp_secret_name)
    if env_val_prefixed:
        logger.debug(f"Found secret '{name}' using prefixed env var '{mcp_secret_name}'.")
        return env_val_prefixed

    error_msg = (
        f"Secret '{name}' not found in any source (AWS, Vault, Env: {name}, Env: {mcp_secret_name})"
    )
    details = {"secret_name": name, "error_type": "SECRET_NOT_FOUND"}
    raise ToolError(error_msg, http_status_code=404, details=details)


async def _sql_get_engine(cid: str) -> AsyncEngine:
    """Get engine by connection ID using the global ConnectionManager."""
    engine = await _connection_manager.get_connection(cid)
    return engine


def _sql_get_next_audit_id() -> str:
    """Generate the next sequential audit ID (thread-safe)."""
    # Locking happens in _sql_audit where this is called
    global _AUDIT_ID_COUNTER
    _AUDIT_ID_COUNTER += 1
    audit_id_str = f"a{_AUDIT_ID_COUNTER:09d}"
    return audit_id_str


def _sql_now() -> str:
    """Get current UTC timestamp in ISO format."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    iso_str = now_utc.isoformat(timespec="seconds")
    return iso_str


async def _sql_audit(
    *,
    tool_name: str,
    action: str,
    connection_id: Optional[str],
    sql: Optional[str],
    tables: Optional[List[str]],
    row_count: Optional[int],
    success: bool,
    error: Optional[str],
    user_id: Optional[str],
    session_id: Optional[str],
    **extra_data: Any,
) -> None:
    """Record an audit trail entry (thread-safe)."""
    global _AUDIT_LOG
    async with _audit_lock:
        audit_id = _sql_get_next_audit_id()  # Get ID while locked
        timestamp = _sql_now()
        log_entry = {}
        log_entry["audit_id"] = audit_id
        log_entry["timestamp"] = timestamp
        log_entry["tool_name"] = tool_name
        log_entry["action"] = action
        log_entry["user_id"] = user_id
        log_entry["session_id"] = session_id
        log_entry["connection_id"] = connection_id
        log_entry["sql"] = sql
        log_entry["tables"] = tables
        log_entry["row_count"] = row_count
        log_entry["success"] = success
        log_entry["error"] = error
        log_entry.update(extra_data)  # Add extra data

        _AUDIT_LOG.append(log_entry)

    # Optional: Log to logger (outside lock)
    log_base = f"Audit[{audit_id}]: Tool={tool_name}, Action={action}, Conn={connection_id}, Success={success}"
    log_error = f", Error={error}" if error else ""
    logger.info(log_base + log_error)


def _sql_update_acl(
    *, tables: Optional[List[str]] = None, columns: Optional[List[str]] = None
) -> None:
    """Update the global ACL lists."""
    global _RESTRICTED_TABLES, _RESTRICTED_COLUMNS
    if tables is not None:
        lowered_tables = {t.lower() for t in tables}
        _RESTRICTED_TABLES = lowered_tables
        logger.info(f"Updated restricted tables ACL: {_RESTRICTED_TABLES}")
    if columns is not None:
        lowered_columns = {c.lower() for c in columns}
        _RESTRICTED_COLUMNS = lowered_columns
        logger.info(f"Updated restricted columns ACL: {_RESTRICTED_COLUMNS}")


def _sql_check_acl(sql: str) -> None:
    """Check if SQL contains any restricted tables or columns using global ACLs."""
    # (Implementation remains the same, uses global _RESTRICTED_TABLES/_COLUMNS)
    raw_toks = re.findall(r'[\w$"\'.]+', sql.lower())
    toks = set(raw_toks)
    normalized_toks = set()
    for tok in toks:
        tok_norm = tok.strip("\"`'[]")
        normalized_toks.add(tok_norm)
        has_dot = "." in tok_norm
        if has_dot:
            last_part = tok_norm.split(".")[-1]
            normalized_toks.add(last_part)

    restricted_tables_found_set = _RESTRICTED_TABLES.intersection(normalized_toks)
    restricted_tables_found = list(restricted_tables_found_set)
    if restricted_tables_found:
        tables_str = ", ".join(restricted_tables_found)
        logger.warning(
            f"ACL Violation: Restricted table(s) found in query: {restricted_tables_found}"
        )
        details = {
            "restricted_tables": restricted_tables_found,
            "error_type": "ACL_TABLE_VIOLATION",
        }
        raise ToolError(
            f"Access denied: Query involves restricted table(s): {tables_str}",
            http_status_code=403,
            details=details,
        )

    restricted_columns_found_set = _RESTRICTED_COLUMNS.intersection(normalized_toks)
    restricted_columns_found = list(restricted_columns_found_set)
    if restricted_columns_found:
        columns_str = ", ".join(restricted_columns_found)
        logger.warning(
            f"ACL Violation: Restricted column(s) found in query: {restricted_columns_found}"
        )
        details = {
            "restricted_columns": restricted_columns_found,
            "error_type": "ACL_COLUMN_VIOLATION",
        }
        raise ToolError(
            f"Access denied: Query involves restricted column(s): {columns_str}",
            http_status_code=403,
            details=details,
        )


def _sql_resolve_conn(raw: str) -> str:
    """Resolve connection string, handling secret references."""
    # (Implementation remains the same)
    is_secret_ref = raw.startswith("secrets://")
    if is_secret_ref:
        secret_name = raw[10:]
        logger.info(f"Resolving secret reference: '{secret_name}'")
        resolved_secret = _pull_secret_from_sources(secret_name)
        return resolved_secret
    return raw


def _sql_mask_val(v: Any) -> Any:
    """Apply masking rules to a single value using global rules."""
    # (Implementation remains the same, uses global _MASKING_RULES)
    is_string = isinstance(v, str)
    is_not_empty = bool(v)
    if not is_string or not is_not_empty:
        return v
    for rule in _MASKING_RULES:
        matches = rule.rx.fullmatch(v)
        if matches:
            is_callable = callable(rule.repl)
            if is_callable:
                try:
                    masked_value = rule.repl(v)
                    return masked_value
                except Exception as e:
                    log_msg = f"Error applying dynamic mask rule {rule.rx.pattern}: {e}"
                    logger.error(log_msg)
                    return "MASKING_ERROR"
            else:
                return rule.repl
    return v


def _sql_mask_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply masking rules to an entire row of data."""
    masked_dict = {}
    for k, v in row.items():
        masked_val = _sql_mask_val(v)
        masked_dict[k] = masked_val
    return masked_dict
    # return {k: _sql_mask_val(v) for k, v in row.items()} # Keep single-line comprehension


def _sql_driver_url(conn_str: str) -> Tuple[str, str]:
    """Convert generic connection string to dialect-specific async URL."""
    # Check if it looks like a path (no ://) and exists or is :memory:
    has_protocol = "://" in conn_str
    looks_like_path = not has_protocol
    path_obj = Path(conn_str)
    path_exists = path_obj.exists()
    is_memory = conn_str == ":memory:"
    is_file_path = looks_like_path and (path_exists or is_memory)

    if is_file_path:
        if is_memory:
            url_str = "sqlite+aiosqlite:///:memory:"
            logger.info("Using in-memory SQLite database.")
        else:
            sqlite_path = path_obj.expanduser().resolve()
            parent_dir = sqlite_path.parent
            parent_exists = parent_dir.exists()
            if not parent_exists:
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory for SQLite DB: {parent_dir}")
                except OSError as e:
                    details = {"path": str(parent_dir)}
                    raise ToolError(
                        f"Failed to create directory for SQLite DB '{parent_dir}': {e}",
                        http_status_code=500,
                        details=details,
                    ) from e
            url_str = f"sqlite+aiosqlite:///{sqlite_path}"
            logger.info(f"Using SQLite database file: {sqlite_path}")
        url = make_url(url_str)
        final_url_str = str(url)
        return final_url_str, "sqlite"
    else:
        url_str = conn_str
        try:
            url = make_url(url_str)
        except Exception as e:
            details = {"value": conn_str}
            raise ToolInputError(
                f"Invalid connection string format: {e}",
                param_name="connection_string",
                details=details,
            ) from e

    drv = url.drivername.lower()
    drivername = url.drivername  # Preserve original case for setting later if needed

    if drv.startswith("sqlite"):
        new_url = url.set(drivername="sqlite+aiosqlite")
        return str(new_url), "sqlite"
    if drv.startswith("postgresql") or drv == "postgres":
        new_url = url.set(drivername="postgresql+asyncpg")
        return str(new_url), "postgresql"
    if drv.startswith("mysql") or drv == "mariadb":
        query = dict(url.query)
        query.setdefault("charset", "utf8mb4")
        new_url = url.set(drivername="mysql+aiomysql", query=query)
        return str(new_url), "mysql"
    if drv.startswith("mssql") or drv == "sqlserver":
        odbc_driver = url.query.get("driver")
        if not odbc_driver:
            logger.warning(
                "MSSQL connection string lacks 'driver' parameter. Ensure a valid ODBC driver (e.g., 'ODBC Driver 17 for SQL Server') is installed and specified."
            )
        new_url = url.set(drivername="mssql+aioodbc")
        return str(new_url), "sqlserver"
    if drv.startswith("snowflake"):
        # Keep original snowflake driver
        new_url = url.set(drivername=drivername)
        return str(new_url), "snowflake"

    logger.error(f"Unsupported database dialect: {drv}")
    details = {"dialect": drv}
    raise ToolInputError(
        f"Unsupported database dialect: '{drv}'. Supported: sqlite, postgresql, mysql, mssql, snowflake",
        param_name="connection_string",
        details=details,
    )


def _sql_auto_pool(db_type: str) -> Dict[str, Any]:
    """Provide sensible default connection pool settings."""
    # (Implementation remains the same)
    # Single-line dict return is acceptable
    defaults = {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": 1800,
        "pool_pre_ping": True,
        "pool_timeout": 30,
    }
    if db_type == "sqlite":
        return {"pool_pre_ping": True}
    if db_type == "postgresql":
        return {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_recycle": 900,
            "pool_pre_ping": True,
            "pool_timeout": 30,
        }
    if db_type == "mysql":
        return {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_recycle": 900,
            "pool_pre_ping": True,
            "pool_timeout": 30,
        }
    if db_type == "sqlserver":
        return {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_recycle": 900,
            "pool_pre_ping": True,
            "pool_timeout": 30,
        }
    if db_type == "snowflake":
        return {"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True, "pool_timeout": 60}
    logger.warning(f"Using default pool settings for unknown db_type: {db_type}")
    return defaults


def _sql_extract_tables(sql: str) -> List[str]:
    """Extract table names referenced in a SQL query."""
    matches = _TABLE_RX.findall(sql)
    tables = set()
    for match in matches:
        # Chained strip is one expression
        table_stripped = match.strip()
        table = table_stripped.strip("\"`'[]")
        has_dot = "." in table
        if has_dot:
            # table.split('.')[-1].strip('"`\'[]') # Original combined
            parts = table.split(".")
            last_part = parts[-1]
            table = last_part.strip("\"`'[]")
        if table:
            tables.add(table)
    sorted_tables = sorted(list(tables))
    return sorted_tables


def _sql_check_safe(sql: str, read_only: bool = True) -> None:
    """Validate SQL for safety using global patterns and ACLs."""
    # Check ACLs first
    _sql_check_acl(sql)

    # Check prohibited statements
    normalized_sql = sql.lstrip().upper()
    check_sql_part = normalized_sql  # Default part to check

    starts_with_with = normalized_sql.startswith("WITH")
    if starts_with_with:
        try:
            # Regex remains single-line expression assignment
            search_regex = r"\)\s*(SELECT|INSERT|UPDATE|DELETE|MERGE)"
            search_flags = re.IGNORECASE | re.DOTALL
            main_statement_match = re.search(search_regex, normalized_sql, search_flags)
            if main_statement_match:
                # Chained calls okay on one line
                main_statement_group = main_statement_match.group(0)
                check_sql_part = main_statement_group.lstrip(") \t\n\r")
            # else: keep check_sql_part as normalized_sql
        except Exception:
            # Ignore regex errors, fallback to checking whole normalized_sql
            pass

    prohibited_match_obj = _PROHIBITED_SQL_REGEX.match(check_sql_part)
    if prohibited_match_obj:
        # Chained calls okay on one line
        prohibited_match = prohibited_match_obj.group(1)
        prohibited_statement = prohibited_match.strip()
        logger.warning(f"Security Violation: Prohibited statement detected: {prohibited_statement}")
        details = {"statement": prohibited_statement, "error_type": "PROHIBITED_STATEMENT"}
        raise ToolInputError(
            f"Prohibited statement type detected: '{prohibited_statement}'",
            param_name="query",
            details=details,
        )

    # Check read-only constraint
    if read_only:
        allowed_starts = ("SELECT", "SHOW", "EXPLAIN", "DESCRIBE", "PRAGMA")
        is_read_query = check_sql_part.startswith(allowed_starts)
        if not is_read_query:
            query_preview = sql[:100]
            logger.warning(
                f"Security Violation: Write operation attempted in read-only mode: {query_preview}..."
            )
            details = {"error_type": "READ_ONLY_VIOLATION"}
            raise ToolInputError(
                "Write operation attempted in read-only mode", param_name="query", details=details
            )


async def _sql_exec(
    eng: AsyncEngine,
    sql: str,
    params: Optional[Dict[str, Any]],
    *,
    limit: Optional[int],
    tool_name: str,
    action_name: str,
    timeout: float = 30.0,
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    """Core async SQL executor helper."""
    db_dialect = eng.dialect.name
    start_time = time.perf_counter()

    if _Q_CNT:
        # Chained call okay
        _Q_CNT.labels(tool=tool_name, action=action_name, db=db_dialect).inc()

    cols: List[str] = []
    rows_raw: List[Any] = []
    row_count: int = 0
    masked_rows: List[Dict[str, Any]] = []

    async def _run(conn: AsyncConnection):
        nonlocal cols, rows_raw, row_count, masked_rows
        statement = text(sql)
        query_params = params or {}
        try:
            res = await conn.execute(statement, query_params)
            has_cursor = res.cursor is not None
            has_description = has_cursor and res.cursor.description is not None
            if not has_cursor or not has_description:
                logger.debug(f"Query did not return rows or description. Action: {action_name}")
                # Ternary okay
                res_rowcount = res.rowcount if res.rowcount >= 0 else 0
                row_count = res_rowcount
                masked_rows = []  # Ensure it's an empty list
                empty_cols: List[str] = []
                empty_rows: List[Dict[str, Any]] = []
                return empty_cols, empty_rows, row_count  # Return empty lists for cols/rows

            cols = list(res.keys())
            try:
                # --- START: Restored SQLite Handling ---
                is_sqlite = db_dialect == "sqlite"
                if is_sqlite:
                    # aiosqlite fetchall/fetchmany might not work reliably with async iteration or limits in all cases
                    # Fetch all as mappings (dicts) directly
                    # Lambda okay if single line
                    def sync_lambda(sync_conn):
                        return list(sync_conn.execute(statement, query_params).mappings())

                    all_rows_mapped = await conn.run_sync(sync_lambda)
                    rows_raw = all_rows_mapped  # Keep the dict list format
                    needs_limit = limit is not None and limit >= 0
                    if needs_limit:
                        rows_raw = rows_raw[:limit]  # Apply limit in Python
                else:
                    # Standard async fetching for other dialects
                    needs_limit = limit is not None and limit >= 0
                    if needs_limit:
                        fetched_rows = await res.fetchmany(limit)  # Returns Row objects
                        rows_raw = fetched_rows
                    else:
                        fetched_rows = await res.fetchall()  # Returns Row objects
                        rows_raw = fetched_rows
                # --- END: Restored SQLite Handling ---

                row_count = len(rows_raw)  # Count based on fetched/limited rows

            except Exception as fetch_err:
                log_msg = f"Error fetching rows for {tool_name}/{action_name}: {fetch_err}"
                logger.error(log_msg, exc_info=True)
                query_preview = sql[:100] + "..."
                details = {"query": query_preview}
                raise ToolError(
                    f"Error fetching results: {fetch_err}", http_status_code=500, details=details
                ) from fetch_err

            # Apply masking using _sql_mask_row which uses global rules
            # Adjust masking based on fetched format
            if is_sqlite:
                # List comprehension okay
                masked_rows_list = [_sql_mask_row(r) for r in rows_raw]  # Already dicts
                masked_rows = masked_rows_list
            else:
                # List comprehension okay
                masked_rows_list = [
                    _sql_mask_row(r._mapping) for r in rows_raw
                ]  # Convert Row objects
                masked_rows = masked_rows_list

            return cols, masked_rows, row_count

        except (ProgrammingError, OperationalError) as db_err:
            err_type_name = type(db_err).__name__
            log_msg = f"Database execution error ({err_type_name}) for {tool_name}/{action_name} on {db_dialect}: {db_err}"
            logger.error(log_msg, exc_info=True)
            query_preview = sql[:100] + "..."
            details = {"db_error": str(db_err), "query": query_preview}
            raise ToolError(
                f"Database Error: {db_err}", http_status_code=400, details=details
            ) from db_err
        except SQLAlchemyError as sa_err:
            err_type_name = type(sa_err).__name__
            log_msg = f"SQLAlchemy error ({err_type_name}) for {tool_name}/{action_name} on {db_dialect}: {sa_err}"
            logger.error(log_msg, exc_info=True)
            query_preview = sql[:100] + "..."
            details = {"sqlalchemy_error": str(sa_err), "query": query_preview}
            raise ToolError(
                f"SQLAlchemy Error: {sa_err}", http_status_code=500, details=details
            ) from sa_err
        except Exception as e:  # Catch other potential errors within _run
            log_msg = f"Unexpected error within _run for {tool_name}/{action_name}: {e}"
            logger.error(log_msg, exc_info=True)
            raise ToolError(
                f"Unexpected error during query execution step: {e}", http_status_code=500
            ) from e

    try:
        async with eng.connect() as conn:
            # Run within timeout
            # Call okay
            run_coro = _run(conn)
            cols_res, masked_rows_res, cnt_res = await asyncio.wait_for(run_coro, timeout=timeout)
            cols = cols_res
            masked_rows = masked_rows_res
            cnt = cnt_res

            latency = time.perf_counter() - start_time
            if _Q_LAT:
                # Chained call okay
                _Q_LAT.labels(tool=tool_name, action=action_name, db=db_dialect).observe(latency)
            log_msg = f"Execution successful for {tool_name}/{action_name}. Latency: {latency:.3f}s, Rows fetched: {cnt}"
            logger.debug(log_msg)
            return cols, masked_rows, cnt

    except asyncio.TimeoutError:
        log_msg = (
            f"Query timeout ({timeout}s) exceeded for {tool_name}/{action_name} on {db_dialect}."
        )
        logger.warning(log_msg)
        query_preview = sql[:100] + "..."
        details = {"timeout": timeout, "query": query_preview}
        raise ToolError(
            f"Query timed out after {timeout} seconds", http_status_code=504, details=details
        ) from None
    except ToolError:
        # Re-raise known ToolErrors
        raise
    except Exception as e:
        log_msg = f"Unexpected error during _sql_exec for {tool_name}/{action_name}: {e}"
        logger.error(log_msg, exc_info=True)
        details = {"error_type": type(e).__name__}
        raise ToolError(
            f"An unexpected error occurred: {e}", http_status_code=500, details=details
        ) from e


def _sql_export_rows(
    cols: List[str],
    rows: List[Dict[str, Any]],
    export_format: str,
    export_path: Optional[str] = None,
) -> Tuple[Any | None, str | None]:
    """Export query results helper."""
    if not export_format:
        return None, None
    export_format_lower = export_format.lower()
    supported_formats = ["pandas", "excel", "csv"]
    if export_format_lower not in supported_formats:
        details = {"format": export_format}
        msg = f"Unsupported export format: '{export_format}'. Use 'pandas', 'excel', or 'csv'."
        raise ToolInputError(msg, param_name="export.format", details=details)
    if pd is None:
        details = {"library": "pandas"}
        msg = f"Pandas library is not installed. Cannot export to '{export_format_lower}'."
        raise ToolError(msg, http_status_code=501, details=details)

    try:
        # Ternary okay
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        logger.info(f"Created DataFrame with shape {df.shape} for export.")
    except Exception as e:
        logger.error(f"Error creating Pandas DataFrame: {e}", exc_info=True)
        raise ToolError(f"Failed to create DataFrame for export: {e}", http_status_code=500) from e

    if export_format_lower == "pandas":
        logger.debug("Returning raw Pandas DataFrame.")
        return df, None

    final_path: str
    temp_file_created = False
    if export_path:
        try:
            # Chained calls okay
            path_obj = Path(export_path)
            path_expanded = path_obj.expanduser()
            path_resolved = path_expanded.resolve()
            parent_dir = path_resolved.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            final_path = str(path_resolved)
            logger.info(f"Using specified export path: {final_path}")
        except OSError as e:
            details = {"path": export_path}
            raise ToolError(
                f"Cannot create directory for export path '{export_path}': {e}",
                http_status_code=500,
                details=details,
            ) from e
        except Exception as e:  # Catch other path errors
            details = {"path": export_path}
            msg = f"Invalid export path provided: {export_path}. Error: {e}"
            raise ToolInputError(msg, param_name="export.path", details=details) from e
    else:
        # Ternary okay
        suffix = ".xlsx" if export_format_lower == "excel" else ".csv"
        try:
            prefix = f"mcp_export_{export_format_lower}_"
            fd, final_path_temp = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            final_path = final_path_temp
            os.close(fd)
            temp_file_created = True
            logger.info(f"Created temporary file for export: {final_path}")
        except Exception as e:
            logger.error(f"Failed to create temporary file for export: {e}", exc_info=True)
            raise ToolError(f"Failed to create temporary file: {e}", http_status_code=500) from e

    try:
        if export_format_lower == "excel":
            df.to_excel(final_path, index=False, engine="xlsxwriter")
        elif export_format_lower == "csv":
            df.to_csv(final_path, index=False)
        log_msg = f"Exported data to {export_format_lower.upper()} file: {final_path}"
        logger.info(log_msg)
        return None, final_path
    except Exception as e:
        log_msg = f"Error exporting DataFrame to {export_format_lower} file '{final_path}': {e}"
        logger.error(log_msg, exc_info=True)
        path_exists = Path(final_path).exists()
        if temp_file_created and path_exists:
            try:
                Path(final_path).unlink()
            except OSError:
                logger.warning(f"Could not clean up temporary export file: {final_path}")
        raise ToolError(
            f"Failed to export data to {export_format_lower}: {e}", http_status_code=500
        ) from e


async def _sql_validate_df(df: Any, schema: Any | None) -> None:
    """Validate DataFrame against Pandera schema helper."""
    if schema is None:
        logger.debug("No Pandera schema provided for validation.")
        return
    if pa is None:
        logger.warning("Pandera library not installed, skipping validation.")
        return
    is_pandas_df = pd is not None and isinstance(df, pd.DataFrame)
    if not is_pandas_df:
        logger.warning("Pandas DataFrame not available for validation.")
        return

    logger.info(f"Validating DataFrame (shape {df.shape}) against provided Pandera schema.")
    try:
        schema.validate(df, lazy=True)
        logger.info("Pandera validation successful.")
    except pa.errors.SchemaErrors as se:
        # Ternary okay
        error_details_df = se.failure_cases
        can_dict = hasattr(error_details_df, "to_dict")
        error_details = (
            error_details_df.to_dict(orient="records") if can_dict else str(error_details_df)
        )
        # Ternary okay
        can_len = hasattr(error_details_df, "__len__")
        error_count = len(error_details_df) if can_len else "multiple"

        log_msg = f"Pandera validation failed with {error_count} errors. Details: {error_details}"
        logger.warning(log_msg)

        # Break down error message construction
        error_msg_base = f"Pandera validation failed ({error_count} errors):\n"
        error_msg_lines = []
        error_details_list = error_details if isinstance(error_details, list) else []
        errors_to_show = error_details_list[:5]

        for err in errors_to_show:
            col = err.get("column", "N/A")
            check = err.get("check", "N/A")
            index = err.get("index", "N/A")
            fail_case_raw = err.get("failure_case", "N/A")
            fail_case_str = str(fail_case_raw)[:50]
            line = f"- Column '{col}': {check} failed for index {index}. Data: {fail_case_str}..."
            error_msg_lines.append(line)

        error_msg = error_msg_base + "\n".join(error_msg_lines)

        num_errors = error_count if isinstance(error_count, int) else 0
        if num_errors > 5:
            more_errors_count = num_errors - 5
            error_msg += f"\n... and {more_errors_count} more errors."

        validation_errors = error_details  # Pass the original structure
        details = {"error_type": "VALIDATION_ERROR"}
        raise ToolError(
            error_msg, http_status_code=422, validation_errors=validation_errors, details=details
        ) from se
    except Exception as e:
        logger.error(f"Unexpected error during Pandera validation: {e}", exc_info=True)
        raise ToolError(
            f"An unexpected error occurred during schema validation: {e}", http_status_code=500
        ) from e


async def _sql_convert_nl_to_sql(
    connection_id: str,
    natural_language: str,
    confidence_threshold: float = 0.6,
    user_id: Optional[str] = None,  # Added for lineage
    session_id: Optional[str] = None,  # Added for lineage
) -> Dict[str, Any]:
    """Helper method to convert natural language to SQL."""
    # (Implementation largely the same, uses _sql_get_engine, _sql_check_safe, global state _SCHEMA_VERSIONS, _LINEAGE)
    nl_preview = natural_language[:100]
    logger.info(f"Converting NL to SQL for connection {connection_id}. Query: '{nl_preview}...'")
    eng = await _sql_get_engine(connection_id)

    def _get_schema_fingerprint_sync(sync_conn) -> str:
        # (Schema fingerprint sync helper implementation is the same)
        try:
            sync_inspector = sa_inspect(sync_conn)
            tbls = []
            schema_names = sync_inspector.get_schema_names()
            default_schema = sync_inspector.default_schema_name
            # List comprehension okay
            other_schemas = [s for s in schema_names if s != default_schema]
            schemas_to_inspect = [default_schema] + other_schemas

            for schema_name in schemas_to_inspect:
                # Ternary okay
                prefix = f"{schema_name}." if schema_name and schema_name != default_schema else ""
                table_names_in_schema = sync_inspector.get_table_names(schema=schema_name)
                for t in table_names_in_schema:
                    try:
                        cols = sync_inspector.get_columns(t, schema=schema_name)
                        # List comprehension okay
                        col_defs = [f"{c['name']}:{str(c['type']).split('(')[0]}" for c in cols]
                        col_defs_str = ",".join(col_defs)
                        tbl_def = f"{prefix}{t}({col_defs_str})"
                        tbls.append(tbl_def)
                    except Exception as col_err:
                        logger.warning(f"Could not get columns for table {prefix}{t}: {col_err}")
                        tbl_def_err = f"{prefix}{t}(...)"
                        tbls.append(tbl_def_err)
            # Call okay
            fp = "; ".join(sorted(tbls))
            if not fp:
                logger.warning("Schema fingerprint generation resulted in empty string.")
                return "Error: Could not retrieve schema."
            return fp
        except Exception as e:
            logger.error(f"Error in _get_schema_fingerprint_sync: {e}", exc_info=True)
            return "Error: Could not retrieve schema."

    async def _get_schema_fingerprint(conn: AsyncConnection) -> str:
        logger.debug("Generating schema fingerprint for NL->SQL...")
        try:
            # Lambda okay
            def sync_func(sync_conn):
                return _get_schema_fingerprint_sync(sync_conn)

            fingerprint = await conn.run_sync(sync_func)
            return fingerprint
        except Exception as e:
            logger.error(f"Error generating schema fingerprint: {e}", exc_info=True)
            return "Error: Could not retrieve schema."

    async with eng.connect() as conn:
        schema_fingerprint = await _get_schema_fingerprint(conn)

    # Multi-line string assignment okay
    prompt = (
        "You are a highly specialized AI assistant that translates natural language questions into SQL queries.\n"
        "You must adhere STRICTLY to the following rules:\n"
        "1. Generate only a SINGLE, executable SQL query for the given database schema and question.\n"
        "2. Use the exact table and column names provided in the schema fingerprint.\n"
        "3. Do NOT generate any explanatory text, comments, or markdown formatting.\n"
        "4. The output MUST be a valid JSON object containing two keys: 'sql' (the generated SQL query as a string) and 'confidence' (a float between 0.0 and 1.0 indicating your confidence in the generated SQL).\n"
        "5. If the question cannot be answered from the schema or is ambiguous, set confidence to 0.0 and provide a minimal, safe query like 'SELECT 1;' in the 'sql' field.\n"
        "6. Prioritize safety: Avoid generating queries that could modify data (UPDATE, INSERT, DELETE, DROP, etc.). Generate SELECT statements ONLY.\n\n"  # Stricter rule
        f"Database Schema Fingerprint:\n```\n{schema_fingerprint}\n```\n\n"
        f"Natural Language Question:\n```\n{natural_language}\n```\n\n"
        "JSON Output:"
    )

    try:
        logger.debug("Sending prompt to LLM for NL->SQL conversion.")
        # Call okay
        completion_result = await generate_completion(
            prompt=prompt, max_tokens=350, temperature=0.2
        )
        llm_response_dict = completion_result

        # Ternary okay
        is_dict_response = isinstance(llm_response_dict, dict)
        llm_response = llm_response_dict.get("text", "") if is_dict_response else ""

        llm_response_preview = llm_response[:300]
        logger.debug(f"LLM Response received: {llm_response_preview}...")
        if not llm_response:
            raise ToolError("LLM returned empty response for NL->SQL.", http_status_code=502)

    except Exception as llm_err:
        logger.error(f"LLM completion failed for NL->SQL: {llm_err}", exc_info=True)
        details = {"error_type": "LLM_ERROR"}
        raise ToolError(
            f"Failed to get response from LLM: {llm_err}", http_status_code=502, details=details
        ) from llm_err

    try:
        data = {}
        try:
            # Try parsing the whole response as JSON first
            data = json.loads(llm_response)
        except json.JSONDecodeError as e:
            # If that fails, look for a JSON block within the text
            # Call okay
            search_regex = r"\{.*\}"
            search_flags = re.DOTALL | re.MULTILINE
            json_match = re.search(search_regex, llm_response, search_flags)
            if not json_match:
                raise ValueError("No JSON object found in the LLM response.") from e
            json_str = json_match.group(0)
            data = json.loads(json_str)

        is_dict_data = isinstance(data, dict)
        has_sql = "sql" in data
        has_confidence = "confidence" in data
        if not is_dict_data or not has_sql or not has_confidence:
            raise ValueError("LLM response JSON is missing required keys ('sql', 'confidence').")

        sql = data["sql"]
        conf_raw = data["confidence"]
        conf = float(conf_raw)

        is_sql_str = isinstance(sql, str)
        is_conf_valid = 0.0 <= conf <= 1.0
        if not is_sql_str or not is_conf_valid:
            raise ValueError("LLM response has invalid types for 'sql' or 'confidence'.")

        sql_preview = sql[:150]
        logger.info(f"LLM generated SQL with confidence {conf:.2f}: {sql_preview}...")

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        response_preview = str(llm_response)[:200]
        error_detail = (
            f"LLM returned invalid or malformed JSON: {e}. Response: '{response_preview}...'"
        )
        logger.error(error_detail)
        details = {"error_type": "LLM_RESPONSE_INVALID"}
        raise ToolError(error_detail, http_status_code=500, details=details) from e

    is_below_threshold = conf < confidence_threshold
    if is_below_threshold:
        nl_query_preview = natural_language
        low_conf_msg = f"LLM confidence ({conf:.2f}) is below the required threshold ({confidence_threshold}). NL Query: '{nl_query_preview}'"
        logger.warning(low_conf_msg)
        details = {"error_type": "LOW_CONFIDENCE"}
        raise ToolError(
            low_conf_msg, http_status_code=400, generated_sql=sql, confidence=conf, details=details
        ) from None

    try:
        _sql_check_safe(sql, read_only=True)  # Enforce read-only for generated SQL
        # Call okay
        sql_upper = sql.upper()
        sql_stripped = sql_upper.lstrip()
        is_valid_start = sql_stripped.startswith(("SELECT", "WITH"))
        if not is_valid_start:
            details = {"error_type": "INVALID_GENERATED_SQL"}
            raise ToolError(
                "Generated query does not appear to be a valid SELECT statement.",
                http_status_code=400,
                details=details,
            )
        # Basic table check (optional, as before)
    except ToolInputError as safety_err:
        logger.error(f"Generated SQL failed safety check: {safety_err}. SQL: {sql}")
        details = {"error_type": "SAFETY_VIOLATION"}
        raise ToolError(
            f"Generated SQL failed validation: {safety_err}",
            http_status_code=400,
            generated_sql=sql,
            confidence=conf,
            details=details,
        ) from safety_err

    result_dict = {"sql": sql, "confidence": conf}
    return result_dict


# =============================================================================
# Public Tool Functions (Standalone replacements for SQLTool methods)
# =============================================================================


@with_tool_metrics
@with_error_handling
async def manage_database(
    action: str,
    connection_string: Optional[str] = None,
    connection_id: Optional[str] = None,
    echo: bool = False,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    ctx: Optional[Dict] = None,  # Added ctx for potential future use
    **options: Any,
) -> Dict[str, Any]:
    """
    Unified database connection management tool.

    Args:
        action: The action to perform: "connect", "disconnect", "test", or "status".
        connection_string: Database connection string or secrets:// reference. (Required for "connect").
        connection_id: An existing connection ID (Required for "disconnect", "test"). Can be provided for "connect" to suggest an ID.
        echo: Enable SQLAlchemy engine logging (For "connect" action, default: False).
        user_id: Optional user identifier for audit logging.
        session_id: Optional session identifier for audit logging.
        ctx: Optional context from MCP server (not used currently).
        **options: Additional options:
            - For "connect": Passed directly to SQLAlchemy's `create_async_engine`.
            - Can include custom audit context.

    Returns:
        Dict with action results and metadata. Varies based on action.
    """
    tool_name = "manage_database"
    db_dialect = "unknown"
    # Dict comprehension okay
    audit_extras_all = {k: v for k, v in options.items()}
    audit_extras = {k: v for k, v in audit_extras_all.items() if k not in ["echo"]}

    try:
        if action == "connect":
            if not connection_string:
                raise ToolInputError(
                    "connection_string is required for 'connect'", param_name="connection_string"
                )
            # Ternary okay
            cid = connection_id or str(uuid.uuid4())
            logger.info(f"Attempting to connect with connection_id: {cid}")
            resolved_conn_str = _sql_resolve_conn(connection_string)
            url, db_type = _sql_driver_url(resolved_conn_str)
            db_dialect = db_type  # Update dialect for potential error logging
            pool_opts = _sql_auto_pool(db_type)
            # Dict unpacking okay
            engine_opts = {**pool_opts, **options}
            # Dict comprehension okay
            log_opts = {k: v for k, v in engine_opts.items() if k != "password"}
            logger.debug(f"Creating engine for {db_type} with options: {log_opts}")
            connect_args = engine_opts.pop("connect_args", {})
            # Ternary okay
            exec_opts_base = {"async_execution": True} if db_type == "snowflake" else {}
            # Pass other engine options directly
            execution_options = {**exec_opts_base, **engine_opts.pop("execution_options", {})}

            # Separate create_async_engine call
            eng = create_async_engine(
                url,
                echo=echo,
                connect_args=connect_args,
                execution_options=execution_options,
                **engine_opts,  # Pass remaining options like pool settings
            )

            try:
                # Ternary okay
                test_sql = "SELECT CURRENT_TIMESTAMP" if db_type != "sqlite" else "SELECT 1"
                # Call okay
                await _sql_exec(
                    eng,
                    test_sql,
                    None,
                    limit=1,
                    tool_name=tool_name,
                    action_name="connect_test",
                    timeout=15,
                )
                logger.info(f"Connection test successful for {cid} ({db_type}).")
            except ToolError as test_err:
                logger.error(f"Connection test failed for {cid} ({db_type}): {test_err}")
                await eng.dispose()
                # Get details okay
                err_details = getattr(test_err, "details", None)
                raise ToolError(
                    f"Connection test failed: {test_err}", http_status_code=400, details=err_details
                ) from test_err
            except Exception as e:
                logger.error(
                    f"Unexpected error during connection test for {cid} ({db_type}): {e}",
                    exc_info=True,
                )
                await eng.dispose()
                raise ToolError(
                    f"Unexpected error during connection test: {e}", http_status_code=500
                ) from e

            await _connection_manager.add_connection(cid, eng)
            # Call okay
            await _sql_audit(
                tool_name=tool_name,
                action="connect",
                connection_id=cid,
                sql=None,
                tables=None,
                row_count=None,
                success=True,
                error=None,
                user_id=user_id,
                session_id=session_id,
                database_type=db_type,
                echo=echo,
                **audit_extras,
            )
            # Return dict okay
            return {
                "action": "connect",
                "connection_id": cid,
                "database_type": db_type,
                "success": True,
            }

        elif action == "disconnect":
            if not connection_id:
                raise ToolInputError(
                    "connection_id is required for 'disconnect'", param_name="connection_id"
                )
            logger.info(f"Attempting to disconnect connection_id: {connection_id}")
            db_dialect_for_audit = "unknown"  # Default if engine retrieval fails
            try:
                # Needs await before get_connection
                engine_to_close = await _connection_manager.get_connection(connection_id)
                db_dialect_for_audit = engine_to_close.dialect.name
            except ToolInputError:
                # This error means connection_id wasn't found by get_connection
                logger.warning(f"Disconnect requested for unknown connection_id: {connection_id}")
                # Call okay
                await _sql_audit(
                    tool_name=tool_name,
                    action="disconnect",
                    connection_id=connection_id,
                    sql=None,
                    tables=None,
                    row_count=None,
                    success=False,
                    error="Connection ID not found",
                    user_id=user_id,
                    session_id=session_id,
                    **audit_extras,
                )
                # Return dict okay
                return {
                    "action": "disconnect",
                    "connection_id": connection_id,
                    "success": False,
                    "message": "Connection ID not found",
                }
            except Exception as e:
                # Catch other errors during engine retrieval itself
                logger.error(f"Error retrieving engine for disconnect ({connection_id}): {e}")
                # Proceed to attempt close, but audit will likely show failure or non-existence

            # Attempt closing even if retrieval had issues (it might have been removed between check and close)
            success = await _connection_manager.close_connection(connection_id)
            # Ternary okay
            error_msg = None if success else "Failed to close or already closed/not found"
            # Call okay
            await _sql_audit(
                tool_name=tool_name,
                action="disconnect",
                connection_id=connection_id,
                sql=None,
                tables=None,
                row_count=None,
                success=success,
                error=error_msg,
                user_id=user_id,
                session_id=session_id,
                database_type=db_dialect_for_audit,
                **audit_extras,
            )
            # Return dict okay
            return {"action": "disconnect", "connection_id": connection_id, "success": success}

        elif action == "test":
            if not connection_id:
                raise ToolInputError(
                    "connection_id is required for 'test'", param_name="connection_id"
                )
            logger.info(f"Testing connection_id: {connection_id}")
            eng = await _sql_get_engine(connection_id)
            db_dialect = eng.dialect.name  # Now dialect is known for sure
            t0 = time.perf_counter()
            # Ternary conditions okay
            vsql = (
                "SELECT sqlite_version()"
                if db_dialect == "sqlite"
                else "SELECT CURRENT_VERSION()"
                if db_dialect == "snowflake"
                else "SELECT version()"
            )
            # Call okay
            cols, rows, _ = await _sql_exec(
                eng, vsql, None, limit=1, tool_name=tool_name, action_name="test", timeout=10
            )
            latency = time.perf_counter() - t0
            # Ternary okay
            has_rows_and_cols = rows and cols
            version_info = rows[0].get(cols[0], "N/A") if has_rows_and_cols else "N/A"
            log_msg = f"Connection test successful for {connection_id}. Version: {version_info}, Latency: {latency:.3f}s"
            logger.info(log_msg)
            # Return dict okay
            return {
                "action": "test",
                "connection_id": connection_id,
                "response_time_seconds": round(latency, 3),
                "version": version_info,
                "database_type": db_dialect,
                "success": True,
            }

        elif action == "status":
            logger.info("Retrieving connection status.")
            connections_info = {}
            current_time = time.time()
            # Access connections safely using async with lock if needed, or make copy
            conn_items = []
            async with _connection_manager._lock:  # Access lock directly for iteration safety
                # Call okay
                conn_items = list(_connection_manager.connections.items())

            for conn_id, (eng, last_access) in conn_items:
                try:
                    url_display_raw = str(eng.url)
                    parsed_url = make_url(url_display_raw)
                    url_display = url_display_raw  # Default
                    if parsed_url.password:
                        # Call okay
                        url_masked = parsed_url.set(password="***")
                        url_display = str(url_masked)
                    # Break down dict assignment
                    conn_info_dict = {}
                    conn_info_dict["url_summary"] = url_display
                    conn_info_dict["dialect"] = eng.dialect.name
                    last_access_dt = dt.datetime.fromtimestamp(last_access)
                    conn_info_dict["last_accessed"] = last_access_dt.isoformat()
                    idle_seconds = current_time - last_access
                    conn_info_dict["idle_time_seconds"] = round(idle_seconds, 1)
                    connections_info[conn_id] = conn_info_dict
                except Exception as status_err:
                    logger.error(f"Error retrieving status for connection {conn_id}: {status_err}")
                    connections_info[conn_id] = {"error": str(status_err)}
            # Return dict okay
            return {
                "action": "status",
                "active_connections_count": len(connections_info),
                "connections": connections_info,
                "cleanup_interval_seconds": _connection_manager.cleanup_interval,
                "success": True,
            }

        else:
            logger.error(f"Invalid action specified for manage_database: {action}")
            details = {"action": action}
            msg = f"Unknown action: '{action}'. Valid actions: connect, disconnect, test, status"
            raise ToolInputError(msg, param_name="action", details=details)

    except ToolInputError as tie:
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action,
            connection_id=connection_id,
            sql=None,
            tables=None,
            row_count=None,
            success=False,
            error=str(tie),
            user_id=user_id,
            session_id=session_id,
            database_type=db_dialect,
            **audit_extras,
        )
        raise tie
    except ToolError as te:
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action,
            connection_id=connection_id,
            sql=None,
            tables=None,
            row_count=None,
            success=False,
            error=str(te),
            user_id=user_id,
            session_id=session_id,
            database_type=db_dialect,
            **audit_extras,
        )
        raise te
    except Exception as e:
        log_msg = f"Unexpected error in manage_database (action: {action}): {e}"
        logger.error(log_msg, exc_info=True)
        error_str = f"Unexpected error: {e}"
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action,
            connection_id=connection_id,
            sql=None,
            tables=None,
            row_count=None,
            success=False,
            error=error_str,
            user_id=user_id,
            session_id=session_id,
            database_type=db_dialect,
            **audit_extras,
        )
        raise ToolError(
            f"An unexpected error occurred in manage_database: {e}", http_status_code=500
        ) from e


@with_tool_metrics
@with_error_handling
async def execute_sql(
    connection_id: str,
    query: Optional[str] = None,
    natural_language: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    pagination: Optional[Dict[str, int]] = None,
    read_only: bool = True,
    export: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    validate_schema: Optional[Any] = None,
    max_rows: Optional[int] = 1000,
    confidence_threshold: float = 0.6,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    ctx: Optional[Dict] = None,  # Added ctx
    **options: Any,
) -> Dict[str, Any]:
    """
    Unified SQL query execution tool.

    Handles direct SQL execution, NL-to-SQL conversion, pagination,
    result masking, safety checks, validation, and export.

    Args:
        connection_id: The ID of the database connection to use.
        query: The SQL query string to execute. (Use instead of natural_language).
        natural_language: A natural language question to convert to SQL. (Use instead of query).
        parameters: Dictionary of parameters for parameterized queries.
        pagination: Dict with "page" (>=1) and "page_size" (>=1) for paginated results.
                    Cannot be used with max_rows clipping if the dialect requires LIMIT/OFFSET.
        read_only: If True (default), enforces safety checks against write operations (UPDATE, DELETE, etc.). Set to False only if writes are explicitly intended and allowed.
        export: Dictionary with "format" ('pandas', 'excel', 'csv') and optional "path" (string) for exporting results.
        timeout: Maximum execution time in seconds (default: 60.0).
        validate_schema: A Pandera schema object to validate the results DataFrame against.
        max_rows: Maximum number of rows to return in the result (default: 1000). Set to None or -1 for unlimited (potentially dangerous).
        confidence_threshold: Minimum confidence score (0.0-1.0) required from the LLM for NL-to-SQL conversion (default: 0.6).
        user_id: Optional user identifier for audit logging.
        session_id: Optional session identifier for audit logging.
        ctx: Optional context from MCP server.
        **options: Additional options for audit logging or future extensions.

    Returns:
        A dictionary containing:
        - columns (List[str]): List of column names.
        - rows (List[Dict[str, Any]]): List of data rows (masked).
        - row_count (int): Number of rows returned in this batch/page.
        - truncated (bool): True if max_rows limited the results.
        - pagination (Optional[Dict]): Info about the current page if pagination was used.
        - generated_sql (Optional[str]): The SQL query generated from natural language, if applicable.
        - confidence (Optional[float]): The confidence score from the NL-to-SQL conversion, if applicable.
        - validation_status (Optional[str]): 'success', 'failed', 'skipped'.
        - validation_errors (Optional[Any]): Details if validation failed.
        - export_status (Optional[str]): Status message if export was attempted.
        - <format>_path (Optional[str]): Path to the exported file if export to file was successful.
        - dataframe (Optional[pd.DataFrame]): The raw Pandas DataFrame if export format was 'pandas'.
        - success (bool): Always True if no exception was raised.
    """
    tool_name = "execute_sql"
    action_name = "query"  # Default, may change
    original_query_input = query  # Keep track of original SQL input
    original_nl_input = natural_language  # Keep track of NL input
    generated_sql = None
    confidence = None
    final_query: str
    # Dict unpacking okay
    final_params = parameters or {}
    result: Dict[str, Any] = {}
    tables: List[str] = []
    # Dict unpacking okay
    audit_extras = {**options}

    try:
        # 1. Determine Query
        use_nl = natural_language and not query
        use_sql = query and not natural_language
        is_ambiguous = natural_language and query
        no_input = not natural_language and not query

        if is_ambiguous:
            msg = "Provide either 'query' or 'natural_language', not both."
            raise ToolInputError(msg, param_name="query/natural_language")
        if no_input:
            msg = "Either 'query' or 'natural_language' must be provided."
            raise ToolInputError(msg, param_name="query/natural_language")

        if use_nl:
            action_name = "nl_to_sql_exec"
            nl_preview = natural_language[:100]
            log_msg = (
                f"Received natural language query for connection {connection_id}: '{nl_preview}...'"
            )
            logger.info(log_msg)
            try:
                # Pass user_id/session_id to NL converter for lineage/audit trail consistency if needed
                # Call okay
                nl_result = await _sql_convert_nl_to_sql(
                    connection_id, natural_language, confidence_threshold, user_id, session_id
                )
                final_query = nl_result["sql"]
                generated_sql = final_query
                confidence = nl_result["confidence"]
                # original_query remains None, original_nl_input has the NL
                audit_extras["generated_sql"] = generated_sql
                audit_extras["confidence"] = confidence
                query_preview = final_query[:150]
                log_msg = f"Successfully converted NL to SQL (Confidence: {confidence:.2f}): {query_preview}..."
                logger.info(log_msg)
                read_only = True  # Ensure read-only for generated SQL
            except ToolError as nl_err:
                # Audit NL failure
                await _sql_audit(
                    tool_name=tool_name,
                    action="nl_to_sql_fail",
                    connection_id=connection_id,
                    sql=natural_language,  # Log the NL query that failed
                    tables=None,
                    row_count=None,
                    success=False,
                    error=str(nl_err),
                    user_id=user_id,
                    session_id=session_id,
                    **audit_extras,
                )
                raise nl_err  # Re-raise the error
        elif use_sql:
            # Action name remains 'query'
            final_query = query
            query_preview = final_query[:150]
            logger.info(f"Executing direct SQL query on {connection_id}: {query_preview}...")
            # original_query_input has the SQL, original_nl_input is None
        # else case already handled by initial checks

        # 2. Check Safety
        _sql_check_safe(final_query, read_only)
        tables = _sql_extract_tables(final_query)
        logger.debug(f"Query targets tables: {tables}")

        # 3. Get Engine
        eng = await _sql_get_engine(connection_id)

        # 4. Handle Pagination or Standard Execution
        if pagination:
            action_name = "query_paginated"
            page = pagination.get("page", 1)
            page_size = pagination.get("page_size", 100)
            is_page_valid = isinstance(page, int) and page >= 1
            is_page_size_valid = isinstance(page_size, int) and page_size >= 1
            if not is_page_valid:
                raise ToolInputError(
                    "Pagination 'page' must be an integer >= 1.", param_name="pagination.page"
                )
            if not is_page_size_valid:
                raise ToolInputError(
                    "Pagination 'page_size' must be an integer >= 1.",
                    param_name="pagination.page_size",
                )

            offset = (page - 1) * page_size
            db_dialect = eng.dialect.name
            paginated_query: str
            if db_dialect == "sqlserver":
                query_lower = final_query.lower()
                has_order_by = "order by" in query_lower
                if not has_order_by:
                    raise ToolInputError(
                        "SQL Server pagination requires an ORDER BY clause in the query.",
                        param_name="query",
                    )
                paginated_query = (
                    f"{final_query} OFFSET :_page_offset ROWS FETCH NEXT :_page_size ROWS ONLY"
                )
            elif db_dialect == "oracle":
                paginated_query = (
                    f"{final_query} OFFSET :_page_offset ROWS FETCH NEXT :_page_size ROWS ONLY"
                )
            else:  # Default LIMIT/OFFSET for others (MySQL, PostgreSQL, SQLite)
                paginated_query = f"{final_query} LIMIT :_page_size OFFSET :_page_offset"

            # Fetch one extra row to check for next page
            fetch_size = page_size + 1
            # Dict unpacking okay
            paginated_params = {**final_params, "_page_size": fetch_size, "_page_offset": offset}
            log_msg = (
                f"Executing paginated query (Page: {page}, Size: {page_size}): {paginated_query}"
            )
            logger.debug(log_msg)
            # Call okay
            cols, rows_with_extra, fetched_count_paged = await _sql_exec(
                eng,
                paginated_query,
                paginated_params,
                limit=None,  # Limit is applied in SQL for pagination
                tool_name=tool_name,
                action_name=action_name,
                timeout=timeout,
            )

            # Check if more rows exist than requested page size
            has_next_page = len(rows_with_extra) > page_size
            returned_rows = rows_with_extra[:page_size]
            returned_row_count = len(returned_rows)

            # Build result dict piece by piece
            pagination_info = {}
            pagination_info["page"] = page
            pagination_info["page_size"] = page_size
            pagination_info["has_next_page"] = has_next_page
            pagination_info["has_previous_page"] = page > 1

            result = {}
            result["columns"] = cols
            result["rows"] = returned_rows
            result["row_count"] = returned_row_count
            result["pagination"] = pagination_info
            result["truncated"] = False  # Not truncated by max_rows in pagination mode
            result["success"] = True

        else:  # Standard execution (no pagination dict)
            action_name = "query_standard"
            # Ternary okay
            needs_limit = max_rows is not None and max_rows >= 0
            fetch_limit = (max_rows + 1) if needs_limit else None

            query_preview = final_query[:150]
            log_msg = f"Executing standard query (Max rows: {max_rows}): {query_preview}..."
            logger.debug(log_msg)
            # Call okay
            cols, rows_maybe_extra, fetched_count = await _sql_exec(
                eng,
                final_query,
                final_params,
                limit=fetch_limit,  # Use fetch_limit (max_rows + 1 or None)
                tool_name=tool_name,
                action_name=action_name,
                timeout=timeout,
            )

            # Determine truncation based on fetch_limit
            truncated = fetch_limit is not None and fetched_count >= fetch_limit
            # Apply actual max_rows limit to returned data
            # Ternary okay
            returned_rows = rows_maybe_extra[:max_rows] if needs_limit else rows_maybe_extra
            returned_row_count = len(returned_rows)

            # Build result dict piece by piece
            result = {}
            result["columns"] = cols
            result["rows"] = returned_rows
            result["row_count"] = returned_row_count
            result["truncated"] = truncated
            result["success"] = True
            # No pagination key in standard mode

        # Add NL->SQL info if applicable
        if generated_sql:
            result["generated_sql"] = generated_sql
            result["confidence"] = confidence

        # 5. Handle Validation
        if validate_schema:
            temp_df = None
            validation_status = "skipped (unknown reason)"
            validation_errors = None
            if pd:
                try:
                    # Ternary okay
                    df_data = result["rows"]
                    df_cols = result["columns"]
                    temp_df = (
                        pd.DataFrame(df_data, columns=df_cols)
                        if df_data
                        else pd.DataFrame(columns=df_cols)
                    )
                    try:
                        # Call okay
                        await _sql_validate_df(temp_df, validate_schema)
                        validation_status = "success"
                        logger.info("Pandera validation passed.")
                    except ToolError as val_err:
                        logger.warning(f"Pandera validation failed: {val_err}")
                        validation_status = "failed"
                        # Get validation errors okay
                        validation_errors = getattr(val_err, "validation_errors", str(val_err))
                except Exception as df_err:
                    logger.error(f"Error creating DataFrame for validation: {df_err}")
                    validation_status = f"skipped (Failed to create DataFrame: {df_err})"
            else:
                logger.warning("Pandas not installed, skipping Pandera validation.")
                validation_status = "skipped (Pandas not installed)"

            result["validation_status"] = validation_status
            if validation_errors:
                result["validation_errors"] = validation_errors

        # 6. Handle Export
        export_requested = export and export.get("format")
        if export_requested:
            export_format = export["format"]  # Keep original case for path key
            export_format_lower = export_format.lower()
            req_path = export.get("path")
            log_msg = f"Export requested: Format={export_format}, Path={req_path or 'Temporary'}"
            logger.info(log_msg)
            export_status = "failed (unknown reason)"
            try:
                # Call okay
                dataframe, export_path = _sql_export_rows(
                    result["columns"], result["rows"], export_format_lower, req_path
                )
                export_status = "success"
                if dataframe is not None:  # Only if format was 'pandas'
                    result["dataframe"] = dataframe
                if export_path:  # If file was created
                    path_key = f"{export_format_lower}_path"  # Use lowercase format for key
                    result[path_key] = export_path
                log_msg = f"Export successful. Format: {export_format}, Path: {export_path or 'In-memory DataFrame'}"
                logger.info(log_msg)
                audit_extras["export_format"] = export_format
                audit_extras["export_path"] = export_path
            except (ToolError, ToolInputError) as export_err:
                logger.error(f"Export failed: {export_err}")
                export_status = f"Failed: {export_err}"
            result["export_status"] = export_status

        # 7. Audit Success
        # Determine which query to log based on input
        audit_sql = original_nl_input if use_nl else original_query_input
        audit_row_count = result.get("row_count", 0)
        audit_val_status = result.get("validation_status")
        audit_exp_status = result.get("export_status", "not requested")
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action_name,
            connection_id=connection_id,
            sql=audit_sql,
            tables=tables,
            row_count=audit_row_count,
            success=True,
            error=None,
            user_id=user_id,
            session_id=session_id,
            read_only=read_only,
            pagination_used=bool(pagination),
            validation_status=audit_val_status,
            export_status=audit_exp_status,
            **audit_extras,
        )
        return result

    except ToolInputError as tie:
        # Audit failure, use original inputs for logging context
        audit_sql = original_nl_input if original_nl_input else original_query_input
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action_name + "_fail",
            connection_id=connection_id,
            sql=audit_sql,
            tables=tables,
            row_count=0,
            success=False,
            error=str(tie),
            user_id=user_id,
            session_id=session_id,
            **audit_extras,
        )
        raise tie
    except ToolError as te:
        # Audit failure
        audit_sql = original_nl_input if original_nl_input else original_query_input
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action_name + "_fail",
            connection_id=connection_id,
            sql=audit_sql,
            tables=tables,
            row_count=0,
            success=False,
            error=str(te),
            user_id=user_id,
            session_id=session_id,
            **audit_extras,
        )
        raise te
    except Exception as e:
        log_msg = f"Unexpected error in execute_sql (action: {action_name}): {e}"
        logger.error(log_msg, exc_info=True)
        # Audit failure
        audit_sql = original_nl_input if original_nl_input else original_query_input
        error_str = f"Unexpected error: {e}"
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action_name + "_fail",
            connection_id=connection_id,
            sql=audit_sql,
            tables=tables,
            row_count=0,
            success=False,
            error=error_str,
            user_id=user_id,
            session_id=session_id,
            **audit_extras,
        )
        raise ToolError(
            f"An unexpected error occurred during SQL execution: {e}", http_status_code=500
        ) from e


@with_tool_metrics
@with_error_handling
async def explore_database(
    connection_id: str,
    action: str,
    table_name: Optional[str] = None,
    column_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    ctx: Optional[Dict] = None,  # Added ctx
    **options: Any,
) -> Dict[str, Any]:
    """
    Unified database schema exploration and documentation tool.

    Performs actions like listing schemas, tables, views, columns,
    getting table/column details, finding relationships, and generating documentation.

    Args:
        connection_id: The ID of the database connection to use.
        action: The exploration action:
            - "schema": Get full schema details (tables, views, columns, relationships).
            - "table": Get details for a specific table (columns, PK, FKs, indexes, optionally sample data/stats). Requires `table_name`.
            - "column": Get statistics for a specific column (nulls, distinct, optionally histogram). Requires `table_name` and `column_name`.
            - "relationships": Find related tables via foreign keys up to a certain depth. Requires `table_name`.
            - "documentation": Generate schema documentation (markdown or JSON).
        table_name: Name of the table for 'table', 'column', 'relationships' actions.
        column_name: Name of the column for 'column' action.
        schema_name: Specific schema to inspect (if supported by dialect and needed). Defaults to connection's default schema.
        user_id: Optional user identifier for audit logging.
        session_id: Optional session identifier for audit logging.
        ctx: Optional context from MCP server.
        **options: Additional options depending on the action:
            - schema: include_indexes (bool), include_foreign_keys (bool), detailed (bool)
            - table: include_sample_data (bool), sample_size (int), include_statistics (bool)
            - column: histogram (bool), num_buckets (int)
            - relationships: depth (int)
            - documentation: output_format ('markdown'|'json'), include_indexes(bool), include_foreign_keys(bool)

    Returns:
        A dictionary containing the results of the exploration action and a 'success' flag.
        Structure varies significantly based on the action.
    """
    tool_name = "explore_database"
    # Break down audit_extras creation
    audit_extras = {}
    audit_extras.update(options)
    audit_extras["table_name"] = table_name
    audit_extras["column_name"] = column_name
    audit_extras["schema_name"] = schema_name

    try:
        log_msg = f"Exploring database for connection {connection_id}. Action: {action}, Table: {table_name}, Column: {column_name}, Schema: {schema_name}"
        logger.info(log_msg)
        eng = await _sql_get_engine(connection_id)
        db_dialect = eng.dialect.name
        audit_extras["database_type"] = db_dialect

        # Define sync inspection helper (runs within connect block)
        def _run_sync_inspection(
            inspector_target: Union[AsyncConnection, AsyncEngine], func_to_run: callable
        ):
            # Call okay
            sync_inspector = sa_inspect(inspector_target)
            return func_to_run(sync_inspector)

        async with eng.connect() as conn:
            # --- Action: schema ---
            if action == "schema":
                include_indexes = options.get("include_indexes", True)
                include_foreign_keys = options.get("include_foreign_keys", True)
                detailed = options.get("detailed", False)
                filter_schema = schema_name  # Use provided schema or None for default

                def _get_full_schema(sync_conn) -> Dict[str, Any]:
                    # Separate inspector and target schema assignment
                    insp = sa_inspect(sync_conn)
                    target_schema = filter_schema or getattr(insp, "default_schema_name", None)

                    log_msg = f"Inspecting schema: {target_schema or 'Default'}. Detailed: {detailed}, Indexes: {include_indexes}, FKs: {include_foreign_keys}"
                    logger.info(log_msg)
                    tables_data: List[Dict[str, Any]] = []
                    views_data: List[Dict[str, Any]] = []
                    relationships: List[Dict[str, Any]] = []
                    try:
                        table_names = insp.get_table_names(schema=target_schema)
                        view_names = insp.get_view_names(schema=target_schema)
                    except Exception as inspect_err:
                        msg = f"Failed to list tables/views for schema '{target_schema}': {inspect_err}"
                        raise ToolError(msg, http_status_code=500) from inspect_err

                    for tbl_name in table_names:
                        try:
                            # Build t_info dict step-by-step
                            t_info: Dict[str, Any] = {}
                            t_info["name"] = tbl_name
                            t_info["columns"] = []
                            if target_schema:
                                t_info["schema"] = target_schema

                            columns_raw = insp.get_columns(tbl_name, schema=target_schema)
                            for c in columns_raw:
                                # Build col_info dict step-by-step
                                col_info = {}
                                col_info["name"] = c["name"]
                                col_info["type"] = str(c["type"])
                                col_info["nullable"] = c["nullable"]
                                col_info["primary_key"] = bool(c.get("primary_key"))
                                if detailed:
                                    col_info["default"] = c.get("default")
                                    col_info["comment"] = c.get("comment")
                                    col_info["autoincrement"] = c.get("autoincrement", "auto")
                                t_info["columns"].append(col_info)

                            if include_indexes:
                                try:
                                    idxs_raw = insp.get_indexes(tbl_name, schema=target_schema)
                                    # List comprehension okay
                                    t_info["indexes"] = [
                                        {
                                            "name": i["name"],
                                            "columns": i["column_names"],
                                            "unique": i.get("unique", False),
                                        }
                                        for i in idxs_raw
                                    ]
                                except Exception as idx_err:
                                    logger.warning(
                                        f"Could not retrieve indexes for table {tbl_name}: {idx_err}"
                                    )
                                    t_info["indexes"] = []
                            if include_foreign_keys:
                                try:
                                    fks_raw = insp.get_foreign_keys(tbl_name, schema=target_schema)
                                    if fks_raw:
                                        t_info["foreign_keys"] = []
                                        for fk in fks_raw:
                                            # Build fk_info dict step-by-step
                                            fk_info = {}
                                            fk_info["name"] = fk.get("name")
                                            fk_info["constrained_columns"] = fk[
                                                "constrained_columns"
                                            ]
                                            fk_info["referred_schema"] = fk.get("referred_schema")
                                            fk_info["referred_table"] = fk["referred_table"]
                                            fk_info["referred_columns"] = fk["referred_columns"]
                                            t_info["foreign_keys"].append(fk_info)

                                            # Build relationship dict step-by-step
                                            rel_info = {}
                                            rel_info["source_schema"] = target_schema
                                            rel_info["source_table"] = tbl_name
                                            rel_info["source_columns"] = fk["constrained_columns"]
                                            rel_info["target_schema"] = fk.get("referred_schema")
                                            rel_info["target_table"] = fk["referred_table"]
                                            rel_info["target_columns"] = fk["referred_columns"]
                                            relationships.append(rel_info)
                                except Exception as fk_err:
                                    logger.warning(
                                        f"Could not retrieve foreign keys for table {tbl_name}: {fk_err}"
                                    )
                            tables_data.append(t_info)
                        except Exception as tbl_err:
                            log_msg = f"Failed to inspect table '{tbl_name}' in schema '{target_schema}': {tbl_err}"
                            logger.error(log_msg, exc_info=True)
                            # Append error dict
                            error_entry = {
                                "name": tbl_name,
                                "schema": target_schema,
                                "error": f"Failed to inspect: {tbl_err}",
                            }
                            tables_data.append(error_entry)

                    for view_name in view_names:
                        try:
                            # Build view_info dict step-by-step
                            view_info: Dict[str, Any] = {}
                            view_info["name"] = view_name
                            if target_schema:
                                view_info["schema"] = target_schema
                            try:
                                view_def_raw = insp.get_view_definition(
                                    view_name, schema=target_schema
                                )
                                # Ternary okay
                                view_def = view_def_raw or ""
                                view_info["definition"] = view_def
                            except Exception as view_def_err:
                                log_msg = f"Could not retrieve definition for view {view_name}: {view_def_err}"
                                logger.warning(log_msg)
                                view_info["definition"] = "Error retrieving definition"
                            try:
                                view_cols_raw = insp.get_columns(view_name, schema=target_schema)
                                # List comprehension okay
                                view_info["columns"] = [
                                    {"name": vc["name"], "type": str(vc["type"])}
                                    for vc in view_cols_raw
                                ]
                            except Exception:
                                pass  # Ignore column errors for views if definition failed etc.
                            views_data.append(view_info)
                        except Exception as view_err:
                            log_msg = f"Failed to inspect view '{view_name}' in schema '{target_schema}': {view_err}"
                            logger.error(log_msg, exc_info=True)
                            # Append error dict
                            error_entry = {
                                "name": view_name,
                                "schema": target_schema,
                                "error": f"Failed to inspect: {view_err}",
                            }
                            views_data.append(error_entry)

                    # Build schema_result dict step-by-step
                    schema_result: Dict[str, Any] = {}
                    schema_result["action"] = "schema"
                    schema_result["database_type"] = db_dialect
                    schema_result["inspected_schema"] = target_schema or "Default"
                    schema_result["tables"] = tables_data
                    schema_result["views"] = views_data
                    schema_result["relationships"] = relationships
                    schema_result["success"] = True

                    # Schema Hashing and Lineage
                    try:
                        # Call okay
                        schema_json = json.dumps(schema_result, sort_keys=True, default=str)
                        schema_bytes = schema_json.encode()
                        # Call okay
                        schema_hash = hashlib.sha256(schema_bytes).hexdigest()

                        timestamp = _sql_now()
                        last_hash = _SCHEMA_VERSIONS.get(connection_id)
                        schema_changed = last_hash != schema_hash

                        if schema_changed:
                            _SCHEMA_VERSIONS[connection_id] = schema_hash
                            # Build lineage_entry dict step-by-step
                            lineage_entry = {}
                            lineage_entry["connection_id"] = connection_id
                            lineage_entry["timestamp"] = timestamp
                            lineage_entry["schema_hash"] = schema_hash
                            lineage_entry["previous_hash"] = last_hash
                            lineage_entry["user_id"] = user_id  # Include user from outer scope
                            lineage_entry["tables_count"] = len(tables_data)
                            lineage_entry["views_count"] = len(views_data)
                            lineage_entry["action_source"] = f"{tool_name}/{action}"
                            _LINEAGE.append(lineage_entry)

                            hash_preview = schema_hash[:8]
                            prev_hash_preview = last_hash[:8] if last_hash else "None"
                            log_msg = f"Schema change detected or initial capture for {connection_id}. New hash: {hash_preview}..., Previous: {prev_hash_preview}"
                            logger.info(log_msg)
                            schema_result["schema_hash"] = schema_hash
                            # Boolean conversion okay
                            schema_result["schema_change_detected"] = bool(last_hash)
                    except Exception as hash_err:
                        log_msg = f"Error generating schema hash or recording lineage: {hash_err}"
                        logger.error(log_msg, exc_info=True)

                    return schema_result

                # Call okay
                def sync_func(sync_conn_arg):
                    return _get_full_schema(sync_conn_arg)

                result = await conn.run_sync(sync_func)  # Pass sync connection

            # --- Action: table ---
            elif action == "table":
                if not table_name:
                    raise ToolInputError(
                        "`table_name` is required for 'table'", param_name="table_name"
                    )
                include_sample = options.get("include_sample_data", False)
                sample_size_raw = options.get("sample_size", 5)
                sample_size = int(sample_size_raw)
                include_stats = options.get("include_statistics", False)
                if sample_size < 0:
                    sample_size = 0

                def _get_basic_table_meta(sync_conn) -> Dict[str, Any]:
                    # Assign inspector and schema
                    insp = sa_inspect(sync_conn)
                    target_schema = schema_name or getattr(insp, "default_schema_name", None)
                    logger.info(f"Inspecting table details: {target_schema}.{table_name}")
                    try:
                        all_tables = insp.get_table_names(schema=target_schema)
                        if table_name not in all_tables:
                            msg = f"Table '{table_name}' not found in schema '{target_schema}'."
                            raise ToolInputError(msg, param_name="table_name")
                    except Exception as list_err:
                        msg = f"Could not verify if table '{table_name}' exists: {list_err}"
                        raise ToolError(msg, http_status_code=500) from list_err

                    # Initialize meta parts
                    cols = []
                    idx = []
                    fks = []
                    pk_constraint = {}
                    table_comment_text = None

                    cols = insp.get_columns(table_name, schema=target_schema)
                    try:
                        idx = insp.get_indexes(table_name, schema=target_schema)
                    except Exception as idx_err:
                        logger.warning(f"Could not get indexes for table {table_name}: {idx_err}")
                    try:
                        fks = insp.get_foreign_keys(table_name, schema=target_schema)
                    except Exception as fk_err:
                        logger.warning(
                            f"Could not get foreign keys for table {table_name}: {fk_err}"
                        )
                    try:
                        pk_info = insp.get_pk_constraint(table_name, schema=target_schema)
                        # Split pk_constraint assignment
                        if pk_info and pk_info.get("constrained_columns"):
                            pk_constraint = {
                                "name": pk_info.get("name"),
                                "columns": pk_info["constrained_columns"],
                            }
                        # else pk_constraint remains {}
                    except Exception as pk_err:
                        logger.warning(f"Could not get PK constraint for {table_name}: {pk_err}")
                    try:
                        table_comment_raw = insp.get_table_comment(table_name, schema=target_schema)
                        # Ternary okay
                        table_comment_text = (
                            table_comment_raw.get("text") if table_comment_raw else None
                        )
                    except Exception as cmt_err:
                        logger.warning(f"Could not get table comment for {table_name}: {cmt_err}")

                    # Build return dict step-by-step
                    meta_result = {}
                    meta_result["columns"] = cols
                    meta_result["indexes"] = idx
                    meta_result["foreign_keys"] = fks
                    meta_result["pk_constraint"] = pk_constraint
                    meta_result["table_comment"] = table_comment_text
                    meta_result["schema_name"] = target_schema  # Add schema name for reference
                    return meta_result

                # Call okay
                def sync_func_meta(sync_conn_arg):
                    return _get_basic_table_meta(sync_conn_arg)

                meta = await conn.run_sync(sync_func_meta)  # Pass sync connection

                # Build result dict step-by-step
                result = {}
                result["action"] = "table"
                result["table_name"] = table_name
                # Use schema name returned from meta function
                result["schema_name"] = meta.get("schema_name")
                result["comment"] = meta.get("table_comment")
                # List comprehension okay
                result["columns"] = [
                    {
                        "name": c["name"],
                        "type": str(c["type"]),
                        "nullable": c["nullable"],
                        "primary_key": bool(c.get("primary_key")),
                        "default": c.get("default"),
                        "comment": c.get("comment"),
                    }
                    for c in meta["columns"]
                ]
                result["primary_key"] = meta.get("pk_constraint")
                result["indexes"] = meta.get("indexes", [])
                result["foreign_keys"] = meta.get("foreign_keys", [])
                result["success"] = True

                # Quote identifiers
                id_prep = eng.dialect.identifier_preparer
                quoted_table_name = id_prep.quote(table_name)
                quoted_schema_name = id_prep.quote(schema_name) if schema_name else None
                # Ternary okay
                full_table_name = (
                    f"{quoted_schema_name}.{quoted_table_name}"
                    if quoted_schema_name
                    else quoted_table_name
                )

                # Row count
                try:
                    # Call okay
                    _, count_rows, _ = await _sql_exec(
                        eng,
                        f"SELECT COUNT(*) AS row_count FROM {full_table_name}",
                        None,
                        limit=1,
                        tool_name=tool_name,
                        action_name="table_count",
                        timeout=30,
                    )
                    # Ternary okay
                    result["row_count"] = count_rows[0]["row_count"] if count_rows else 0
                except Exception as count_err:
                    logger.warning(
                        f"Could not get row count for table {full_table_name}: {count_err}"
                    )
                    result["row_count"] = "Error"

                # Sample data
                if include_sample and sample_size > 0:
                    try:
                        # Call okay
                        sample_cols, sample_rows, _ = await _sql_exec(
                            eng,
                            f"SELECT * FROM {full_table_name} LIMIT :n",
                            {"n": sample_size},
                            limit=sample_size,
                            tool_name=tool_name,
                            action_name="table_sample",
                            timeout=30,
                        )
                        # Assign sample data dict okay
                        result["sample_data"] = {"columns": sample_cols, "rows": sample_rows}
                    except Exception as sample_err:
                        logger.warning(
                            f"Could not get sample data for table {full_table_name}: {sample_err}"
                        )
                        # Assign error dict okay
                        result["sample_data"] = {
                            "error": f"Failed to retrieve sample data: {sample_err}"
                        }

                # Statistics
                if include_stats:
                    stats = {}
                    logger.debug(f"Calculating basic statistics for columns in {full_table_name}")
                    columns_to_stat = result.get("columns", [])
                    for c in columns_to_stat:
                        col_name = c["name"]
                        quoted_col = id_prep.quote(col_name)
                        col_stat_data = {}
                        try:
                            # Null count
                            # Call okay
                            _, null_rows, _ = await _sql_exec(
                                eng,
                                f"SELECT COUNT(*) AS null_count FROM {full_table_name} WHERE {quoted_col} IS NULL",
                                None,
                                limit=1,
                                tool_name=tool_name,
                                action_name="col_stat_null",
                                timeout=20,
                            )
                            # Ternary okay
                            null_count = null_rows[0]["null_count"] if null_rows else "Error"

                            # Distinct count
                            # Call okay
                            _, distinct_rows, _ = await _sql_exec(
                                eng,
                                f"SELECT COUNT(DISTINCT {quoted_col}) AS distinct_count FROM {full_table_name}",
                                None,
                                limit=1,
                                tool_name=tool_name,
                                action_name="col_stat_distinct",
                                timeout=45,
                            )
                            # Ternary okay
                            distinct_count = (
                                distinct_rows[0]["distinct_count"] if distinct_rows else "Error"
                            )

                            # Assign stats dict okay
                            col_stat_data = {
                                "null_count": null_count,
                                "distinct_count": distinct_count,
                            }
                        except Exception as stat_err:
                            log_msg = f"Could not calculate statistics for column {col_name} in {full_table_name}: {stat_err}"
                            logger.warning(log_msg)
                            # Assign error dict okay
                            col_stat_data = {"error": f"Failed: {stat_err}"}
                        stats[col_name] = col_stat_data
                    result["statistics"] = stats

            # --- Action: column ---
            elif action == "column":
                if not table_name:
                    raise ToolInputError(
                        "`table_name` required for 'column'", param_name="table_name"
                    )
                if not column_name:
                    raise ToolInputError(
                        "`column_name` required for 'column'", param_name="column_name"
                    )

                generate_histogram = options.get("histogram", False)
                num_buckets_raw = options.get("num_buckets", 10)
                num_buckets = int(num_buckets_raw)
                num_buckets = max(1, num_buckets)  # Ensure at least one bucket

                # Quote identifiers
                id_prep = eng.dialect.identifier_preparer
                quoted_table = id_prep.quote(table_name)
                quoted_column = id_prep.quote(column_name)
                quoted_schema = id_prep.quote(schema_name) if schema_name else None
                # Ternary okay
                full_table_name = (
                    f"{quoted_schema}.{quoted_table}" if quoted_schema else quoted_table
                )
                logger.info(f"Analyzing column {full_table_name}.{quoted_column}")

                stats_data: Dict[str, Any] = {}
                try:
                    # Total Rows
                    # Call okay
                    _, total_rows_res, _ = await _sql_exec(
                        eng,
                        f"SELECT COUNT(*) as cnt FROM {full_table_name}",
                        None,
                        limit=1,
                        tool_name=tool_name,
                        action_name="col_stat_total",
                        timeout=30,
                    )
                    # Ternary okay
                    total_rows_count = total_rows_res[0]["cnt"] if total_rows_res else 0
                    stats_data["total_rows"] = total_rows_count

                    # Null Count
                    # Call okay
                    _, null_rows_res, _ = await _sql_exec(
                        eng,
                        f"SELECT COUNT(*) as cnt FROM {full_table_name} WHERE {quoted_column} IS NULL",
                        None,
                        limit=1,
                        tool_name=tool_name,
                        action_name="col_stat_null",
                        timeout=30,
                    )
                    # Ternary okay
                    null_count = null_rows_res[0]["cnt"] if null_rows_res else 0
                    stats_data["null_count"] = null_count
                    # Ternary okay
                    null_perc = (
                        round((null_count / total_rows_count) * 100, 2) if total_rows_count else 0
                    )
                    stats_data["null_percentage"] = null_perc

                    # Distinct Count
                    # Call okay
                    _, distinct_rows_res, _ = await _sql_exec(
                        eng,
                        f"SELECT COUNT(DISTINCT {quoted_column}) as cnt FROM {full_table_name}",
                        None,
                        limit=1,
                        tool_name=tool_name,
                        action_name="col_stat_distinct",
                        timeout=60,
                    )
                    # Ternary okay
                    distinct_count = distinct_rows_res[0]["cnt"] if distinct_rows_res else 0
                    stats_data["distinct_count"] = distinct_count
                    # Ternary okay
                    distinct_perc = (
                        round((distinct_count / total_rows_count) * 100, 2)
                        if total_rows_count
                        else 0
                    )
                    stats_data["distinct_percentage"] = distinct_perc
                except Exception as stat_err:
                    log_msg = f"Failed to get basic statistics for column {full_table_name}.{quoted_column}: {stat_err}"
                    logger.error(log_msg, exc_info=True)
                    stats_data["error"] = f"Failed to retrieve some statistics: {stat_err}"

                # Build result dict step-by-step
                result = {}
                result["action"] = "column"
                result["table_name"] = table_name
                result["column_name"] = column_name
                result["schema_name"] = schema_name
                result["statistics"] = stats_data
                result["success"] = True

                if generate_histogram:
                    logger.debug(f"Generating histogram for {full_table_name}.{quoted_column}")
                    histogram_data: Optional[Dict[str, Any]] = None
                    try:
                        hist_query = f"SELECT {quoted_column} FROM {full_table_name} WHERE {quoted_column} IS NOT NULL"
                        # Call okay
                        _, value_rows, _ = await _sql_exec(
                            eng,
                            hist_query,
                            None,
                            limit=None,  # Fetch all non-null values
                            tool_name=tool_name,
                            action_name="col_hist_fetch",
                            timeout=90,
                        )
                        # List comprehension okay
                        values = [r[column_name] for r in value_rows]

                        if not values:
                            histogram_data = {"type": "empty", "buckets": []}
                        else:
                            first_val = values[0]
                            # Check type okay
                            is_numeric = isinstance(first_val, (int, float))

                            if is_numeric:
                                try:
                                    min_val = min(values)
                                    max_val = max(values)
                                    buckets = []
                                    if min_val == max_val:
                                        # Single bucket dict okay
                                        bucket = {"range": f"{min_val}", "count": len(values)}
                                        buckets.append(bucket)
                                    else:
                                        # Calculate bin width okay
                                        val_range = max_val - min_val
                                        bin_width = val_range / num_buckets
                                        # List comprehension okay
                                        bucket_ranges_raw = [
                                            (min_val + i * bin_width, min_val + (i + 1) * bin_width)
                                            for i in range(num_buckets)
                                        ]
                                        # Adjust last bucket range okay
                                        last_bucket_idx = num_buckets - 1
                                        last_bucket_start = bucket_ranges_raw[last_bucket_idx][0]
                                        bucket_ranges_raw[last_bucket_idx] = (
                                            last_bucket_start,
                                            max_val,
                                        )
                                        bucket_ranges = bucket_ranges_raw

                                        # List comprehension for bucket init okay
                                        buckets = [
                                            {"range": f"{r[0]:.4g} - {r[1]:.4g}", "count": 0}
                                            for r in bucket_ranges
                                        ]
                                        for v in values:
                                            # Ternary okay
                                            idx_float = (
                                                (v - min_val) / bin_width if bin_width > 0 else 0
                                            )
                                            idx_int = int(idx_float)
                                            # Ensure index is within bounds
                                            idx = min(idx_int, num_buckets - 1)
                                            # Handle max value potentially falling into last bucket due to precision
                                            if v == max_val:
                                                idx = num_buckets - 1
                                            buckets[idx]["count"] += 1

                                    # Assign numeric histogram dict okay
                                    histogram_data = {
                                        "type": "numeric",
                                        "min": min_val,
                                        "max": max_val,
                                        "buckets": buckets,
                                    }
                                except Exception as num_hist_err:
                                    log_msg = f"Error generating numeric histogram: {num_hist_err}"
                                    logger.error(log_msg, exc_info=True)
                                    # Assign error dict okay
                                    histogram_data = {
                                        "error": f"Failed to generate numeric histogram: {num_hist_err}"
                                    }
                            else:  # Categorical / Frequency
                                try:
                                    # Import okay
                                    from collections import Counter

                                    # Call okay
                                    str_values = map(str, values)
                                    value_counts = Counter(str_values)
                                    # Call okay
                                    top_buckets_raw = value_counts.most_common(num_buckets)
                                    # List comprehension okay
                                    buckets_data = [
                                        {"value": str(k)[:100], "count": v}  # Limit value length
                                        for k, v in top_buckets_raw
                                    ]
                                    # Sum okay
                                    top_n_count = sum(b["count"] for b in buckets_data)
                                    other_count = len(values) - top_n_count

                                    # Assign frequency histogram dict okay
                                    histogram_data = {
                                        "type": "frequency",
                                        "top_n": num_buckets,
                                        "buckets": buckets_data,
                                    }
                                    if other_count > 0:
                                        histogram_data["other_values_count"] = other_count
                                except Exception as freq_hist_err:
                                    log_msg = (
                                        f"Error generating frequency histogram: {freq_hist_err}"
                                    )
                                    logger.error(log_msg, exc_info=True)
                                    # Assign error dict okay
                                    histogram_data = {
                                        "error": f"Failed to generate frequency histogram: {freq_hist_err}"
                                    }
                    except Exception as hist_err:
                        log_msg = f"Failed to generate histogram for column {full_table_name}.{quoted_column}: {hist_err}"
                        logger.error(log_msg, exc_info=True)
                        # Assign error dict okay
                        histogram_data = {"error": f"Histogram generation failed: {hist_err}"}
                    result["histogram"] = histogram_data

            # --- Action: relationships ---
            elif action == "relationships":
                if not table_name:
                    raise ToolInputError(
                        "`table_name` required for 'relationships'", param_name="table_name"
                    )
                depth_raw = options.get("depth", 1)
                depth_int = int(depth_raw)
                # Clamp depth
                depth = max(1, min(depth_int, 5))

                log_msg = f"Finding relationships for table '{table_name}' (depth: {depth}, schema: {schema_name})"
                logger.info(log_msg)
                # Call explore_database for schema info - this recursive call is okay
                schema_info = await explore_database(
                    connection_id=connection_id,
                    action="schema",
                    schema_name=schema_name,
                    include_indexes=False,  # Don't need indexes for relationships
                    include_foreign_keys=True,  # Need FKs
                    detailed=False,  # Don't need detailed column info
                )
                # Check success okay
                schema_success = schema_info.get("success", False)
                if not schema_success:
                    raise ToolError(
                        "Failed to retrieve schema information needed to find relationships."
                    )

                # Dict comprehension okay
                tables_list = schema_info.get("tables", [])
                tables_by_name: Dict[str, Dict] = {t["name"]: t for t in tables_list}

                if table_name not in tables_by_name:
                    msg = f"Starting table '{table_name}' not found in schema '{schema_name}'."
                    raise ToolInputError(msg, param_name="table_name")

                visited_nodes = set()  # Track visited nodes to prevent cycles

                # Define the recursive helper function *inside* this action block
                # so it has access to tables_by_name and visited_nodes
                def _build_relationship_graph_standalone(
                    current_table: str, current_depth: int
                ) -> Dict[str, Any]:
                    # Build node_id string okay
                    current_schema = schema_name or "default"
                    node_id = f"{current_schema}.{current_table}"

                    is_max_depth = current_depth >= depth
                    is_visited = node_id in visited_nodes
                    if is_max_depth or is_visited:
                        # Return dict okay
                        return {
                            "table": current_table,
                            "schema": schema_name,  # Use original schema_name context
                            "max_depth_reached": is_max_depth,
                            "cyclic_reference": is_visited,
                        }

                    visited_nodes.add(node_id)
                    node_info = tables_by_name.get(current_table)

                    if not node_info:
                        visited_nodes.remove(node_id)  # Backtrack
                        # Return dict okay
                        return {
                            "table": current_table,
                            "schema": schema_name,
                            "error": "Table info not found",
                        }

                    # Build graph_node dict step-by-step
                    graph_node: Dict[str, Any] = {}
                    graph_node["table"] = current_table
                    graph_node["schema"] = schema_name
                    graph_node["children"] = []
                    graph_node["parents"] = []

                    # Find Parents (current table's FKs point to parents)
                    foreign_keys_list = node_info.get("foreign_keys", [])
                    for fk in foreign_keys_list:
                        ref_table = fk["referred_table"]
                        ref_schema = fk.get(
                            "referred_schema", schema_name
                        )  # Assume same schema if not specified

                        if ref_table in tables_by_name:
                            # Recursive call okay
                            parent_node = _build_relationship_graph_standalone(
                                ref_table, current_depth + 1
                            )
                        else:
                            # Return dict okay for outside scope
                            parent_node = {
                                "table": ref_table,
                                "schema": ref_schema,
                                "outside_scope": True,
                            }

                        # Build relationship string okay
                        constrained_cols_str = ",".join(fk["constrained_columns"])
                        referred_cols_str = ",".join(fk["referred_columns"])
                        rel_str = f"{current_table}.({constrained_cols_str}) -> {ref_table}.({referred_cols_str})"
                        # Append parent relationship dict okay
                        graph_node["parents"].append(
                            {"relationship": rel_str, "target": parent_node}
                        )

                    # Find Children (other tables' FKs point to current table)
                    for other_table_name, other_table_info in tables_by_name.items():
                        if other_table_name == current_table:
                            continue  # Skip self-reference check here

                        other_fks = other_table_info.get("foreign_keys", [])
                        for fk in other_fks:
                            points_to_current = fk["referred_table"] == current_table
                            # Check schema match (use original schema_name context)
                            referred_schema_matches = (
                                fk.get("referred_schema", schema_name) == schema_name
                            )
                            if points_to_current and referred_schema_matches:
                                # Recursive call okay
                                child_node = _build_relationship_graph_standalone(
                                    other_table_name, current_depth + 1
                                )
                                # Build relationship string okay
                                constrained_cols_str = ",".join(fk["constrained_columns"])
                                referred_cols_str = ",".join(fk["referred_columns"])
                                rel_str = f"{other_table_name}.({constrained_cols_str}) -> {current_table}.({referred_cols_str})"
                                # Append child relationship dict okay
                                graph_node["children"].append(
                                    {"relationship": rel_str, "source": child_node}
                                )

                    visited_nodes.remove(node_id)  # Backtrack visited state
                    return graph_node

                # Initial call to the recursive function
                relationship_graph = _build_relationship_graph_standalone(table_name, 0)
                # Build result dict step-by-step
                result = {}
                result["action"] = "relationships"
                result["source_table"] = table_name
                result["schema_name"] = schema_name
                result["max_depth"] = depth
                result["relationship_graph"] = relationship_graph
                result["success"] = True

            # --- Action: documentation ---
            elif action == "documentation":
                output_format_raw = options.get("output_format", "markdown")
                output_format = output_format_raw.lower()
                valid_formats = ["markdown", "json"]
                if output_format not in valid_formats:
                    msg = "Invalid 'output_format'. Use 'markdown' or 'json'."
                    raise ToolInputError(msg, param_name="output_format")

                doc_include_indexes = options.get("include_indexes", True)
                doc_include_fks = options.get("include_foreign_keys", True)
                log_msg = f"Generating database documentation (Format: {output_format}, Schema: {schema_name})"
                logger.info(log_msg)

                # Call explore_database for schema info (recursive call okay)
                schema_data = await explore_database(
                    connection_id=connection_id,
                    action="schema",
                    schema_name=schema_name,
                    include_indexes=doc_include_indexes,
                    include_foreign_keys=doc_include_fks,
                    detailed=True,  # Need details for documentation
                )
                schema_success = schema_data.get("success", False)
                if not schema_success:
                    raise ToolError(
                        "Failed to retrieve schema information needed for documentation."
                    )

                if output_format == "json":
                    # Build result dict step-by-step
                    result = {}
                    result["action"] = "documentation"
                    result["format"] = "json"
                    result["documentation"] = schema_data  # Embed the schema result directly
                    result["success"] = True
                else:  # Markdown
                    # --- Markdown Generation ---
                    lines = []
                    lines.append(f"# Database Documentation ({db_dialect})")
                    db_schema_name = schema_data.get("inspected_schema", "Default Schema")
                    lines.append(f"Schema: **{db_schema_name}**")
                    now_str = _sql_now()
                    lines.append(f"Generated: {now_str}")
                    schema_hash_val = schema_data.get("schema_hash")
                    if schema_hash_val:
                        hash_preview = schema_hash_val[:12]
                        lines.append(f"Schema Version (Hash): `{hash_preview}`")
                    lines.append("")  # Blank line

                    lines.append("## Tables")
                    lines.append("")
                    # Sort okay
                    tables_list_raw = schema_data.get("tables", [])
                    tables = sorted(tables_list_raw, key=lambda x: x["name"])

                    if not tables:
                        lines.append("*No tables found in this schema.*")

                    for t in tables:
                        table_name_doc = t["name"]
                        if t.get("error"):
                            lines.append(f"### {table_name_doc} (Error)")
                            lines.append(f"```\n{t['error']}\n```")
                            lines.append("")
                            continue  # Skip rest for this table

                        lines.append(f"### {table_name_doc}")
                        lines.append("")
                        table_comment = t.get("comment")
                        if table_comment:
                            lines.append(f"> {table_comment}")
                            lines.append("")

                        # Column Header
                        lines.append("| Column | Type | Nullable | PK | Default | Comment |")
                        lines.append("|--------|------|----------|----|---------|---------|")
                        columns_list = t.get("columns", [])
                        for c in columns_list:
                            # Ternary okay
                            pk_flag = "✅" if c["primary_key"] else ""
                            null_flag = "✅" if c["nullable"] else ""
                            default_raw = c.get("default")
                            # Ternary okay
                            default_val_str = f"`{default_raw}`" if default_raw is not None else ""
                            comment_val = c.get("comment") or ""
                            col_name_str = f"`{c['name']}`"
                            col_type_str = f"`{c['type']}`"
                            # Build line okay
                            line = f"| {col_name_str} | {col_type_str} | {null_flag} | {pk_flag} | {default_val_str} | {comment_val} |"
                            lines.append(line)
                        lines.append("")  # Blank line after table

                        # Primary Key section
                        pk_info = t.get("primary_key")
                        pk_cols = pk_info.get("columns") if pk_info else None
                        if pk_info and pk_cols:
                            pk_name = pk_info.get("name", "PK")
                            # List comprehension okay
                            pk_cols_formatted = [f"`{c}`" for c in pk_cols]
                            pk_cols_str = ", ".join(pk_cols_formatted)
                            lines.append(f"**Primary Key:** `{pk_name}` ({pk_cols_str})")
                            lines.append("")

                        # Indexes section
                        indexes_list = t.get("indexes")
                        if doc_include_indexes and indexes_list:
                            lines.append("**Indexes:**")
                            lines.append("")
                            lines.append("| Name | Columns | Unique |")
                            lines.append("|------|---------|--------|")
                            for idx in indexes_list:
                                # Ternary okay
                                unique_flag = "✅" if idx["unique"] else ""
                                # List comprehension okay
                                idx_cols_formatted = [f"`{c}`" for c in idx["columns"]]
                                cols_str = ", ".join(idx_cols_formatted)
                                idx_name_str = f"`{idx['name']}`"
                                # Build line okay
                                line = f"| {idx_name_str} | {cols_str} | {unique_flag} |"
                                lines.append(line)
                            lines.append("")

                        # Foreign Keys section
                        fks_list = t.get("foreign_keys")
                        if doc_include_fks and fks_list:
                            lines.append("**Foreign Keys:**")
                            lines.append("")
                            lines.append("| Name | Column(s) | References |")
                            lines.append("|------|-----------|------------|")
                            for fk in fks_list:
                                # List comprehension okay
                                constrained_cols_fmt = [f"`{c}`" for c in fk["constrained_columns"]]
                                constrained_cols_str = ", ".join(constrained_cols_fmt)

                                ref_schema = fk.get("referred_schema", db_schema_name)
                                ref_table_name = fk["referred_table"]
                                ref_table_str = f"`{ref_schema}`.`{ref_table_name}`"

                                # List comprehension okay
                                ref_cols_fmt = [f"`{c}`" for c in fk["referred_columns"]]
                                ref_cols_str = ", ".join(ref_cols_fmt)

                                fk_name = fk.get("name", "FK")
                                fk_name_str = f"`{fk_name}`"
                                ref_full_str = f"{ref_table_str} ({ref_cols_str})"
                                # Build line okay
                                line = (
                                    f"| {fk_name_str} | {constrained_cols_str} | {ref_full_str} |"
                                )
                                lines.append(line)
                            lines.append("")

                    # Views Section
                    views_list_raw = schema_data.get("views", [])
                    views = sorted(views_list_raw, key=lambda x: x["name"])
                    if views:
                        lines.append("## Views")
                        lines.append("")
                        for v in views:
                            view_name_doc = v["name"]
                            if v.get("error"):
                                lines.append(f"### {view_name_doc} (Error)")
                                lines.append(f"```\n{v['error']}\n```")
                                lines.append("")
                                continue  # Skip rest for this view

                            lines.append(f"### {view_name_doc}")
                            lines.append("")
                            view_columns = v.get("columns")
                            if view_columns:
                                # List comprehension okay
                                view_cols_fmt = [
                                    f"`{vc['name']}` ({vc['type']})" for vc in view_columns
                                ]
                                view_cols_str = ", ".join(view_cols_fmt)
                                lines.append(f"**Columns:** {view_cols_str}")
                                lines.append("")

                            view_def = v.get("definition")
                            # Check for valid definition string
                            is_valid_def = (
                                view_def and view_def != "N/A (Not Implemented by Dialect)"
                            )
                            if is_valid_def:
                                lines.append("**Definition:**")
                                lines.append("```sql")
                                lines.append(view_def)
                                lines.append("```")
                                lines.append("")
                            else:
                                lines.append(
                                    "**Definition:** *Not available or not implemented by dialect.*"
                                )
                                lines.append("")
                    # --- End Markdown Generation ---

                    # Join lines okay
                    markdown_output = "\n".join(lines)
                    # Build result dict step-by-step
                    result = {}
                    result["action"] = "documentation"
                    result["format"] = "markdown"
                    result["documentation"] = markdown_output
                    result["success"] = True

            else:
                logger.error(f"Invalid action specified for explore_database: {action}")
                details = {"action": action}
                valid_actions = "schema, table, column, relationships, documentation"
                msg = f"Unknown action: '{action}'. Valid actions: {valid_actions}"
                raise ToolInputError(msg, param_name="action", details=details)

            # Audit success for all successful actions
            # Ternary okay
            audit_table = [table_name] if table_name else None
            # Call okay
            await _sql_audit(
                tool_name=tool_name,
                action=action,
                connection_id=connection_id,
                sql=None,
                tables=audit_table,
                row_count=None,
                success=True,
                error=None,
                user_id=user_id,
                session_id=session_id,
                **audit_extras,
            )
            return result  # Return the constructed result dict

    except ToolInputError as tie:
        # Audit failure
        # Ternary okay
        audit_table = [table_name] if table_name else None
        action_fail = action + "_fail"
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action_fail,
            connection_id=connection_id,
            sql=None,
            tables=audit_table,
            row_count=None,
            success=False,
            error=str(tie),
            user_id=user_id,
            session_id=session_id,
            **audit_extras,
        )
        raise tie
    except ToolError as te:
        # Audit failure
        # Ternary okay
        audit_table = [table_name] if table_name else None
        action_fail = action + "_fail"
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action_fail,
            connection_id=connection_id,
            sql=None,
            tables=audit_table,
            row_count=None,
            success=False,
            error=str(te),
            user_id=user_id,
            session_id=session_id,
            **audit_extras,
        )
        raise te
    except Exception as e:
        log_msg = f"Unexpected error in explore_database (action: {action}): {e}"
        logger.error(log_msg, exc_info=True)
        # Audit failure
        # Ternary okay
        audit_table = [table_name] if table_name else None
        action_fail = action + "_fail"
        error_str = f"Unexpected error: {e}"
        # Call okay
        await _sql_audit(
            tool_name=tool_name,
            action=action_fail,
            connection_id=connection_id,
            sql=None,
            tables=audit_table,
            row_count=None,
            success=False,
            error=error_str,
            user_id=user_id,
            session_id=session_id,
            **audit_extras,
        )
        raise ToolError(
            f"An unexpected error occurred during database exploration: {e}", http_status_code=500
        ) from e


@with_tool_metrics
@with_error_handling
async def access_audit_log(
    action: str = "view",
    export_format: Optional[str] = None,
    limit: Optional[int] = 100,
    user_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    ctx: Optional[Dict] = None,  # Added ctx
) -> Dict[str, Any]:
    """
    Access and export the in-memory SQL audit log.

    Allows viewing recent log entries or exporting them to a file.
    Note: The audit log is currently stored only in memory and will be lost on server restart.

    Args:
        action: "view" (default) or "export".
        export_format: Required if action is "export". Supports "json", "excel", "csv".
        limit: For "view", the maximum number of most recent records to return (default: 100). Use None or -1 for all.
        user_id: Filter log entries by this user ID.
        connection_id: Filter log entries by this connection ID.
        ctx: Optional context from MCP server.

    Returns:
        Dict containing results:
        - For "view": {action: "view", records: List[Dict], filtered_record_count: int, total_records_in_log: int, filters_applied: Dict, success: True}
        - For "export": {action: "export", path: str, format: str, record_count: int, success: True} or {action: "export", message: str, record_count: 0, success: True} if no records.
    """
    tool_name = "access_audit_log"  # noqa: F841

    # Apply filters using global _AUDIT_LOG
    async with _audit_lock:  # Need lock to safely read/copy log
        # Call okay
        full_log_copy = list(_AUDIT_LOG)
    total_records_in_log = len(full_log_copy)

    # Start with the full copy
    filtered_log = full_log_copy

    # Apply filters sequentially
    if user_id:
        # List comprehension okay
        filtered_log = [r for r in filtered_log if r.get("user_id") == user_id]
    if connection_id:
        # List comprehension okay
        filtered_log = [r for r in filtered_log if r.get("connection_id") == connection_id]
    filtered_record_count = len(filtered_log)

    if action == "view":
        # Ternary okay
        needs_limit = limit is not None and limit >= 0
        records_to_return = filtered_log[-limit:] if needs_limit else filtered_log
        num_returned = len(records_to_return)
        log_msg = f"View audit log requested. Returning {num_returned}/{filtered_record_count} filtered records (Total in log: {total_records_in_log})."
        logger.info(log_msg)

        # Build filters applied dict okay
        filters_applied = {"user_id": user_id, "connection_id": connection_id}
        # Build result dict step-by-step
        result = {}
        result["action"] = "view"
        result["records"] = records_to_return
        result["filtered_record_count"] = filtered_record_count
        result["total_records_in_log"] = total_records_in_log
        result["filters_applied"] = filters_applied
        result["success"] = True
        return result

    elif action == "export":
        if not export_format:
            raise ToolInputError(
                "`export_format` is required for 'export'", param_name="export_format"
            )
        export_format_lower = export_format.lower()
        log_msg = f"Export audit log requested. Format: {export_format_lower}. Records to export: {filtered_record_count}"
        logger.info(log_msg)

        if not filtered_log:
            logger.warning("Audit log is empty or filtered log is empty, nothing to export.")
            # Return dict okay
            return {
                "action": "export",
                "message": "No audit records found matching filters to export.",
                "record_count": 0,
                "success": True,
            }

        if export_format_lower == "json":
            path = ""  # Initialize path
            try:
                # Call okay
                fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="mcp_audit_export_")
                path = temp_path  # Assign path now we know mkstemp succeeded
                os.close(fd)
                # Use sync write for simplicity here
                with open(path, "w", encoding="utf-8") as f:
                    # Call okay
                    json.dump(filtered_log, f, indent=2, default=str)
                log_msg = (
                    f"Successfully exported {filtered_record_count} audit records to JSON: {path}"
                )
                logger.info(log_msg)
                # Return dict okay
                return {
                    "action": "export",
                    "path": path,
                    "format": "json",
                    "record_count": filtered_record_count,
                    "success": True,
                }
            except Exception as e:
                log_msg = f"Failed to export audit log to JSON: {e}"
                logger.error(log_msg, exc_info=True)
                # Clean up temp file if created
                if path and Path(path).exists():
                    try:
                        Path(path).unlink()
                    except OSError:
                        logger.warning(f"Could not clean up failed JSON export file: {path}")
                raise ToolError(
                    f"Failed to export audit log to JSON: {e}", http_status_code=500
                ) from e

        elif export_format_lower in ["excel", "csv"]:
            if pd is None:
                details = {"library": "pandas"}
                msg = f"Pandas library not installed, cannot export audit log to '{export_format_lower}'."
                raise ToolError(msg, http_status_code=501, details=details)
            path = ""  # Initialize path
            try:
                # Call okay
                df = pd.DataFrame(filtered_log)
                # Ternary okay for suffix/writer/engine
                is_excel = export_format_lower == "excel"
                suffix = ".xlsx" if is_excel else ".csv"
                writer_func = df.to_excel if is_excel else df.to_csv
                engine = "xlsxwriter" if is_excel else None

                # Call okay
                fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="mcp_audit_export_")
                path = temp_path  # Assign path
                os.close(fd)

                # Build export args dict okay
                export_kwargs: Dict[str, Any] = {"index": False}
                if engine:
                    export_kwargs["engine"] = engine

                # Call writer function
                writer_func(path, **export_kwargs)

                log_msg = f"Successfully exported {filtered_record_count} audit records to {export_format_lower.upper()}: {path}"
                logger.info(log_msg)
                # Return dict okay
                return {
                    "action": "export",
                    "path": path,
                    "format": export_format_lower,
                    "record_count": filtered_record_count,
                    "success": True,
                }
            except Exception as e:
                log_msg = f"Failed to export audit log to {export_format_lower}: {e}"
                logger.error(log_msg, exc_info=True)
                # Clean up temp file if created
                if path and Path(path).exists():
                    try:
                        Path(path).unlink()
                    except OSError:
                        logger.warning(f"Could not clean up temporary export file: {path}")
                msg = f"Failed to export audit log to {export_format_lower}: {e}"
                raise ToolError(msg, http_status_code=500) from e
        else:
            details = {"format": export_format}
            valid_formats = "'excel', 'csv', or 'json'"
            msg = f"Unsupported export format: '{export_format}'. Use {valid_formats}."
            raise ToolInputError(msg, param_name="export_format", details=details)
    else:
        details = {"action": action}
        msg = f"Unknown action: '{action}'. Use 'view' or 'export'."
        raise ToolInputError(msg, param_name="action", details=details)
