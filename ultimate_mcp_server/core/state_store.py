import asyncio
import os
import pickle
from typing import Any, Dict, Optional

import aiofiles


class StateStore:
    """
    Thread-safe, async-compatible state management system with optional persistence.

    The StateStore provides a robust solution for managing application state in asynchronous
    environments. It organizes data into namespaces, each containing key-value pairs, and
    provides thread-safe access through asyncio.Lock-based concurrency control.

    Key features:
    - Namespace-based organization to separate different types of state data
    - Thread-safe async methods for all operations (get, set, delete)
    - Optional persistence to disk with automatic load/save
    - Granular locking per namespace to maximize concurrency
    - Graceful handling of corrupted or missing persistent data

    Usage example:
    ```python
    # Initialize with persistence
    store = StateStore(persistence_dir="./state")

    # Store values
    await store.set("user_preferences", "theme", "dark")
    await store.set("session_data", "user_id", 12345)

    # Retrieve values (with default if missing)
    theme = await store.get("user_preferences", "theme", default="light")
    user_id = await store.get("session_data", "user_id", default=None)

    # Delete values
    await store.delete("session_data", "temp_token")
    ```

    The StateStore is used internally by the Ultimate MCP Server to maintain state
    across multiple tools and components, and is exposed to tools via the
    with_state_management decorator.
    """

    def __init__(self, persistence_dir: Optional[str] = None):
        """
        Initialize a new StateStore instance.

        The StateStore provides a thread-safe, async-compatible key-value store organized
        by namespaces. It supports both in-memory operation and optional persistence to disk.
        The store is designed for use in multi-threaded or async applications where state
        needs to be shared safely between components.

        Each namespace acts as a separate dictionary with its own concurrency protection.
        Operations within a namespace are serialized using asyncio.Lock, while operations
        across different namespaces can proceed concurrently.

        Args:
            persistence_dir: Optional directory path where state data will be persisted.
                            If provided, each namespace will be stored as a separate pickle
                            file in this directory. If None, the store operates in memory-only
                            mode and state is lost when the application stops.

        Notes:
            - The directory will be created if it doesn't exist
            - Each namespace is persisted as a separate file named "{namespace}.pickle"
            - Data is serialized using Python's pickle module, so stored values should be
              pickle-compatible
            - No automatic cleanup of old or unused namespaces is performed
        """
        self._in_memory_store: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._persistence_dir = persistence_dir
        if persistence_dir and not os.path.exists(persistence_dir):
            os.makedirs(persistence_dir)

    def _get_lock(self, namespace: str) -> asyncio.Lock:
        """
        Get or create an asyncio.Lock for a specific namespace.

        This private method manages the locks used for concurrency control. It maintains
        a dictionary of locks keyed by namespace name, creating new locks as needed.
        This ensures that operations on the same namespace are properly serialized to
        prevent race conditions, while allowing operations on different namespaces to
        proceed concurrently.

        Args:
            namespace: Name of the namespace for which to get or create a lock

        Returns:
            An asyncio.Lock instance specific to the requested namespace

        Notes:
            - Each namespace gets its own independent lock
            - Locks are created on-demand when a namespace is first accessed
            - Locks persist for the lifetime of the StateStore instance
            - This method is called by all public methods (get, set, delete) to
              ensure thread-safe access to namespaces
        """
        if namespace not in self._locks:
            self._locks[namespace] = asyncio.Lock()
        return self._locks[namespace]

    async def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the state store with thread-safe access control.

        This method provides a concurrency-safe way to retrieve state data from the specified
        namespace. If the namespace doesn't exist in memory, it attempts to load it from disk
        (if persistence is enabled) before returning the requested value or default.

        Retrieval behavior:
        - The method first acquires a lock for the specified namespace to ensure thread safety
        - If the namespace is not in memory, it attempts to load it from disk if persistence is enabled
        - If the namespace can't be loaded or doesn't exist, an empty namespace is created
        - Returns the value for the specified key, or the default value if the key is not found

        Args:
            namespace: Logical grouping for related state data (e.g., "tools", "user_settings")
            key: Unique identifier within the namespace for the data to retrieve
            default: Value to return if the key is not found in the namespace

        Returns:
            The stored value if found, otherwise the default value

        Notes:
            - Acquiring the namespace lock is an async operation and may block if another
              operation is currently accessing the same namespace
            - If persistence is enabled, this method may perform disk I/O when a namespace
              needs to be loaded from disk
        """
        async with self._get_lock(namespace):
            if namespace not in self._in_memory_store:
                # Try to load from disk if persistence is enabled
                if self._persistence_dir:
                    await self._load_namespace(namespace)
                else:
                    self._in_memory_store[namespace] = {}

            return self._in_memory_store[namespace].get(key, default)

    async def set(self, namespace: str, key: str, value: Any) -> None:
        """
        Store a value in the state store with thread-safe access control.

        This method provides a concurrency-safe way to store state data in the specified namespace.
        The implementation uses asyncio.Lock to ensure that concurrent access to the same namespace
        doesn't lead to race conditions or data corruption.

        Storage behavior:
        - Values are first stored in an in-memory dictionary
        - If persistence_dir is configured, values are also immediately persisted to disk
        - Each namespace is stored as a separate pickle file
        - Values can be any pickle-serializable Python object

        Args:
            namespace: Logical grouping for related state data (e.g., "tools", "user_settings")
            key: Unique identifier within the namespace for this piece of data
            value: Any pickle-serializable value to store

        Notes:
            - Acquiring the namespace lock is an async operation and may block if another
              operation is currently accessing the same namespace
            - If persistence is enabled, this method performs disk I/O which could take time
              depending on the value size and disk performance
        """
        async with self._get_lock(namespace):
            if namespace not in self._in_memory_store:
                self._in_memory_store[namespace] = {}

            self._in_memory_store[namespace][key] = value

            # Persist immediately if enabled
            if self._persistence_dir:
                await self._persist_namespace(namespace)

    async def delete(self, namespace: str, key: str) -> None:
        """
        Delete a value from the state store with thread-safe access control.

        This method safely removes a key-value pair from the specified namespace,
        and optionally persists the change to disk if persistence is enabled. The
        operation is concurrency-safe through the use of namespace-specific locks.

        Deletion behavior:
        - The method first acquires a lock for the specified namespace to ensure thread safety
        - If the namespace doesn't exist or the key is not found, the operation is a no-op
        - If persistence is enabled, the updated namespace state is written to disk
          immediately after deletion

        Args:
            namespace: Logical grouping for related state data (e.g., "tools", "user_settings")
            key: Unique identifier within the namespace for the data to delete

        Notes:
            - Acquiring the namespace lock is an async operation and may block if another
              operation is currently accessing the same namespace
            - If persistence is enabled, this method performs disk I/O when persisting
              the updated namespace after deletion
            - This method does not raise an exception if the key doesn't exist in the namespace
        """
        async with self._get_lock(namespace):
            if namespace in self._in_memory_store and key in self._in_memory_store[namespace]:
                del self._in_memory_store[namespace][key]

                # Persist the change if enabled
                if self._persistence_dir:
                    await self._persist_namespace(namespace)

    async def _persist_namespace(self, namespace: str) -> None:
        """
        Persist a namespace's data to disk as a pickle file.

        This private method handles the actual disk I/O for saving state data. It serializes
        the entire namespace dictionary to a pickle file named after the namespace in the
        configured persistence directory.

        Args:
            namespace: Name of the namespace whose data should be persisted

        Notes:
            - This method is a no-op if persistence_dir is not configured
            - Uses aiofiles for non-blocking async file I/O
            - The file is named "{namespace}.pickle" and stored in the persistence_dir
            - The entire namespace is serialized in a single operation, which may be
              inefficient for very large namespaces
            - This method is called internally by set() and delete() methods
              after modifying namespace data
        """
        if not self._persistence_dir:
            return

        file_path = os.path.join(self._persistence_dir, f"{namespace}.pickle")
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(pickle.dumps(self._in_memory_store[namespace]))

    async def _load_namespace(self, namespace: str) -> None:
        """
        Load a namespace's data from disk into memory.

        This private method handles loading serialized state data from disk into the in-memory store.
        It is called automatically by the get() method when a namespace is requested but not yet
        loaded in memory. The method implements the lazy-loading pattern, only reading from disk
        when necessary.

        The loading process follows these steps:
        1. Check if persistence is enabled; if not, initialize an empty namespace dictionary
        2. Locate the pickle file for the namespace (named "{namespace}.pickle")
        3. If the file doesn't exist, initialize an empty namespace dictionary
        4. If the file exists, read and deserialize it using pickle
        5. Handle potential serialization errors gracefully (corrupted files, version mismatches)

        Args:
            namespace: Name of the namespace whose data should be loaded. This corresponds
                      directly to a "{namespace}.pickle" file in the persistence directory.

        Returns:
            None: The method modifies the internal self._in_memory_store dictionary directly.

        Notes:
            - Uses aiofiles for non-blocking async file I/O
            - In case of corrupt data (pickle errors), the namespace is initialized as empty
              rather than raising exceptions to the caller
            - Example of file path: /path/to/persistence_dir/user_settings.pickle for the
              "user_settings" namespace
            - This method is idempotent - calling it multiple times for the same namespace
              has no additional effect after the first call

        Examples:
            ```python
            # This method is called internally by get(), not typically called directly
            store = StateStore(persistence_dir="./state")

            # When this executes, _load_namespace("user_settings") will be called internally
            # if the namespace is not already in memory
            value = await store.get("user_settings", "theme")
            ```
        """
        if not self._persistence_dir:
            self._in_memory_store[namespace] = {}
            return

        file_path = os.path.join(self._persistence_dir, f"{namespace}.pickle")
        if not os.path.exists(file_path):
            self._in_memory_store[namespace] = {}
            return

        try:
            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()
                self._in_memory_store[namespace] = pickle.loads(data)
        except (pickle.PickleError, EOFError):
            # Handle corrupt data
            self._in_memory_store[namespace] = {}
