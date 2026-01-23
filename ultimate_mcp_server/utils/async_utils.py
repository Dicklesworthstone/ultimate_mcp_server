"""Async utilities for Ultimate MCP Server."""

import asyncio
import functools
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)

# Type definitions
T = TypeVar("T")
AsyncCallable = Callable[..., Any]


class RateLimiter:
    """
    Rate limiter for controlling request rates to external services.

    This class implements a token bucket algorithm to enforce API rate limits,
    preventing too many requests in a short period of time. It's designed for use
    in asynchronous code and will automatically pause execution when limits are reached.

    The rate limiter tracks the timestamps of recent calls and blocks new calls
    if they would exceed the configured rate limit. When the limit is reached,
    the acquire() method blocks until enough time has passed to allow another call.

    This is useful for:
    - Respecting API rate limits of external services
    - Preventing service overload in high-concurrency applications
    - Implementing polite crawling/scraping behavior
    - Managing resource access in distributed systems

    Usage example:
        ```python
        # Create a rate limiter that allows 5 calls per second
        limiter = RateLimiter(max_calls=5, period=1.0)

        async def make_api_call():
            # This will automatically wait if we're over the limit
            await limiter.acquire()
            # Now make the actual API call...
            return await actual_api_call()
        ```
    """

    def __init__(self, max_calls: int, period: float):
        """
        Initialize the rate limiter.

        Args:
            max_calls: Maximum number of calls allowed within the specified period.
                       For example, 100 calls per period.
            period: Time period in seconds over which the max_calls limit applies.
                    For example, 60.0 for a per-minute rate limit.
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire permission to make a call, waiting if necessary.

        This method blocks until a call is allowed based on the rate limit. When the
        limit has been reached, it will sleep until enough time has passed to allow
        another call, respecting the configured max_calls within the period.

        The method ensures thread-safety through an asyncio lock, making it safe to use
        across multiple tasks. It also handles the case where waiting for rate limit
        permissions overlaps with multiple concurrent requests.

        Returns:
            None. When this method returns, the caller is allowed to proceed.

        Raises:
            asyncio.CancelledError: If the task is cancelled while waiting.
        """
        async with self.lock:
            now = time.time()

            # Remove expired timestamps
            self.calls = [t for t in self.calls if now - t < self.period]

            # Check if we're under the limit
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return

            # Calculate wait time
            wait_time = self.period - (now - self.calls[0])
            if wait_time > 0:
                # Release lock while waiting
                self.lock.release()
                try:
                    logger.debug(
                        f"Rate limit reached, waiting {wait_time:.2f}s", emoji_key="warning"
                    )
                    await asyncio.sleep(wait_time)
                finally:
                    # Reacquire lock
                    await self.lock.acquire()

                # Retry after waiting
                await self.acquire()
            else:
                # Oldest call just expired, record new call
                self.calls = self.calls[1:] + [now]


@asynccontextmanager
async def timed_context(name: str):
    """
    Async context manager for measuring and logging operation duration.

    This utility provides a simple way to time asynchronous operations and log their
    duration upon completion. It's useful for performance monitoring, debugging,
    and identifying bottlenecks in asynchronous code.

    The context manager:
    1. Records the start time when entering the context
    2. Allows the wrapped code to execute
    3. Calculates the elapsed time when the context exits
    4. Logs the operation name and duration with appropriate formatting

    This works with any async code, including API calls, database operations,
    file I/O, or computational tasks.

    Args:
        name: Descriptive name of the operation being timed. This name will appear
              in log messages for easy identification.

    Yields:
        None - This context manager doesn't provide any additional context variables.

    Example usage:
        ```python
        async def fetch_user_data(user_id):
            async with timed_context("Fetch user data"):
                return await database.get_user(user_id)

        async def process_document(doc_id):
            async with timed_context("Document processing"):
                # Multiple operations can be timed together
                doc = await fetch_document(doc_id)
                results = await analyze_document(doc)
                return results
        ```
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.debug(f"{name} completed in {duration:.3f}s", emoji_key="time", time=duration)


async def gather_with_concurrency(n: int, *tasks, return_exceptions: bool = False) -> List[Any]:
    """
    Run multiple async tasks with a controlled concurrency limit.

    This function provides a way to execute multiple asynchronous tasks in parallel
    while ensuring that no more than a specified number of tasks run simultaneously.
    It's similar to asyncio.gather() but with an added concurrency control mechanism.

    This is particularly valuable for:
    - Preventing resource exhaustion when processing many tasks
    - Respecting service capacity limitations
    - Managing memory usage by limiting parallel execution
    - Implementing "worker pool" patterns in async code

    The function uses a semaphore internally to control the number of concurrently
    executing tasks. Tasks beyond the concurrency limit will wait until a running
    task completes and releases the semaphore.

    Args:
        n: Maximum number of tasks to run concurrently. This controls resource usage
           and prevents overloading the system or external services.
        *tasks: Any number of awaitable coroutine objects to execute.
        return_exceptions: If True, exceptions are returned as results rather than being
                          raised. If False, the first raised exception will propagate.
                          This matches the behavior of asyncio.gather().

    Returns:
        List of task results in the same order as the tasks were provided, regardless
        of the order in which they completed.

    Example usage:
        ```python
        # Process a list of URLs with at most 5 concurrent requests
        urls = ["https://example.com/1", "https://example.com/2", ...]
        tasks = [fetch_url(url) for url in urls]
        results = await gather_with_concurrency(5, *tasks)

        # With exception handling
        try:
            results = await gather_with_concurrency(10, *tasks, return_exceptions=False)
        except Exception as e:
            # Handle first exception
            pass

        # Or capture exceptions in results
        results = await gather_with_concurrency(10, *tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                # Handle this exception
                pass
        ```
    """
    semaphore = asyncio.Semaphore(n)

    async def run_task_with_semaphore(task):
        async with semaphore:
            return await task

    return await asyncio.gather(
        *(run_task_with_semaphore(task) for task in tasks), return_exceptions=return_exceptions
    )


async def run_with_timeout(
    coro: Any, timeout: float, default: Optional[T] = None, log_timeout: bool = True
) -> Union[Any, T]:
    """
    Run an async coroutine with a timeout, returning a default value if time expires.

    This utility function executes an async operation with a strict time limit and
    provides graceful handling of timeouts. If the operation completes within the
    specified timeout, its result is returned normally. If the timeout is exceeded,
    the specified default value is returned instead, and the operation is cancelled.

    This functionality is particularly useful for:
    - Making external API calls that might hang or take too long
    - Implementing responsive UIs that can't wait indefinitely
    - Handling potentially slow operations in time-sensitive contexts
    - Creating fallback behavior for unreliable services

    The function uses asyncio.wait_for internally and properly handles the
    TimeoutError, converting it to a controlled return of the default value.

    Args:
        coro: The coroutine (awaitable) to execute with a timeout. This can be
              any async function call or awaitable object.
        timeout: Maximum execution time in seconds before timing out. Must be a
                positive number.
        default: The value to return if the operation times out. Defaults to None.
                This can be any type, and will be returned exactly as provided.
        log_timeout: Whether to log a warning message when a timeout occurs. Set
                    to False to suppress the warning. Default is True.

    Returns:
        The result of the coroutine if it completes within the timeout period,
        otherwise the specified default value.

    Raises:
        Exception: Any exception raised by the coroutine other than TimeoutError
                  will be propagated to the caller.

    Example usage:
        ```python
        # Basic usage with default fallback
        result = await run_with_timeout(
            fetch_data_from_api("https://example.com/data"),
            timeout=5.0,
            default={"status": "timeout", "data": None}
        )

        # Without logging timeouts
        result = await run_with_timeout(
            slow_operation(),
            timeout=10.0,
            default=None,
            log_timeout=False
        )

        # With type checking (using TypeVar)
        data: Optional[List[str]] = await run_with_timeout(
            get_string_list(),
            timeout=3.0,
            default=None
        )
        ```
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if log_timeout:
            logger.warning(f"Operation timed out after {timeout}s", emoji_key="time", time=timeout)
        return default


def async_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_exceptions: Optional[List[Type[Exception]]] = None,
    max_backoff: Optional[float] = None,
):
    """
    Decorator for automatically retrying async functions when they raise exceptions.

    This decorator implements a configurable exponential backoff retry strategy for
    asynchronous functions. When the decorated function raises a specified exception,
    the decorator will automatically wait and retry the operation, with an increasing
    delay between attempts.

    The retry behavior includes:
    - A configurable number of maximum retry attempts
    - Initial delay between retries
    - Exponential backoff (each retry waits longer than the previous one)
    - Optional filtering of which exception types trigger retries
    - Optional maximum backoff time to cap the exponential growth
    - Detailed logging of retry attempts and final failures

    This pattern is especially useful for:
    - Network operations that may fail temporarily
    - API calls subject to rate limiting or intermittent failures
    - Database operations that may encounter transient errors
    - Any resource access that may be temporarily unavailable

    Args:
        max_retries: Maximum number of retry attempts after the initial call
                    (default: 3). Total attempts will be max_retries + 1.
        retry_delay: Initial delay between retries in seconds (default: 1.0).
                    This is the wait time after the first failure.
        backoff_factor: Multiplier applied to delay between retries (default: 2.0).
                       Each retry will wait backoff_factor times longer than the previous.
        retry_exceptions: List of exception types that should trigger a retry.
                         If None (default), all exceptions trigger retries.
        max_backoff: Maximum delay between retries in seconds, regardless of the
                    backoff calculation. None (default) means no maximum.

    Returns:
        A decorator function that wraps the target async function with retry logic.

    Example usage:
        ```python
        # Basic usage - retry any exception up to 3 times
        @async_retry()
        async def fetch_data(url):
            return await make_request(url)

        # Custom configuration - retry specific exceptions with longer delays
        @async_retry(
            max_retries=5,
            retry_delay=2.0,
            backoff_factor=3.0,
            retry_exceptions=[ConnectionError, TimeoutError],
            max_backoff=30.0
        )
        async def send_to_service(data):
            return await service.process(data)
        ```

    Note:
        The decorated function's signature and return type are preserved, making this
        decorator compatible with static type checking.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            exceptions = []
            delay = retry_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if we should retry this exception
                    if retry_exceptions and not any(
                        isinstance(e, exc_type) for exc_type in retry_exceptions
                    ):
                        raise

                    exceptions.append(e)

                    # If this was the last attempt, reraise
                    if attempt >= max_retries:
                        if len(exceptions) > 1:
                            logger.error(
                                f"Function {func.__name__} failed after {max_retries + 1} attempts",
                                emoji_key="error",
                                attempts=max_retries + 1,
                            )
                        raise

                    # Log retry
                    logger.warning(
                        f"Retrying {func.__name__} after error: {str(e)} "
                        f"(attempt {attempt + 1}/{max_retries + 1})",
                        emoji_key="warning",
                        attempt=attempt + 1,
                        max_attempts=max_retries + 1,
                        error=str(e),
                    )

                    # Wait before retrying
                    await asyncio.sleep(delay)

                    # Increase delay for next retry
                    delay *= backoff_factor
                    if max_backoff:
                        delay = min(delay, max_backoff)

            # Shouldn't get here, but just in case
            raise exceptions[-1]

        return wrapper

    return decorator


async def map_async(
    func: Callable[[Any], Any],
    items: List[Any],
    concurrency: int = 10,
    chunk_size: Optional[int] = None,
) -> List[Any]:
    """Map a function over items with limited concurrency.

    This utility provides efficient parallel processing of a list of items while controlling the
    maximum number of concurrent operations. It applies the provided async function to each item
    in the list, respecting the concurrency limit set by the semaphore.

    The function supports two processing modes:
    1. Chunked processing: When chunk_size is provided, items are processed in batches to improve
       memory efficiency when dealing with large lists.
    2. Full parallel processing: When chunk_size is omitted, all items are processed in parallel
       but still limited by the concurrency parameter.

    Args:
        func: Async function to apply to each item. This function should accept a single item
              and return a result.
        items: List of items to process. Each item will be passed individually to the function.
        concurrency: Maximum number of concurrent tasks allowed. This controls the load on system
                     resources. Default is 10.
        chunk_size: Optional batch size for processing. If provided, items are processed in chunks
                    of this size to limit memory usage. If None, all items are processed at once
                    (but still constrained by concurrency).

    Returns:
        List of results from applying the function to each item, in the same order as the input items.

    Examples:
        ```python
        # Define an async function to process an item
        async def process_item(item):
            await asyncio.sleep(0.1)  # Simulate I/O or processing time
            return item * 2

        # Process a list of 100 items with max 5 concurrent tasks
        items = list(range(100))
        results = await map_async(process_item, items, concurrency=5)

        # Process a large list in chunks to manage memory usage
        large_list = list(range(10000))
        results = await map_async(process_item, large_list, concurrency=10, chunk_size=500)
        ```

    Notes:
        - If the items list is empty, an empty list is returned immediately.
        - The function preserves the original order of items in the result list.
        - For CPU-bound tasks, consider using ProcessPoolExecutor with asyncio.to_thread
          instead of this function, as this is optimized for I/O-bound tasks.
    """
    if not items:
        return []

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    # Define task function
    async def process_item(item):
        async with semaphore:
            return await func(item)

    # If using chunks, process in batches
    if chunk_size:
        results = []
        # Process in chunks for better memory management
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            chunk_results = await asyncio.gather(*(process_item(item) for item in chunk))
            results.extend(chunk_results)
        return results
    else:
        # Process all at once with concurrency limit
        return await asyncio.gather(*(process_item(item) for item in items))


class AsyncBatchProcessor:
    """
    Processor for efficiently batching async operations to optimize throughput.

    This class provides a framework for batch processing asynchronous operations,
    which is useful for optimizing I/O-bound tasks such as database writes, API calls,
    or other operations where batching improves efficiency. It automatically collects
    individual items and processes them in batches when:

    1. The batch reaches a specified size (controlled by batch_size)
    2. A specified time interval elapses (controlled by flush_interval)
    3. A manual flush is requested

    The processor also controls concurrency, allowing multiple batches to be processed
    simultaneously while limiting the maximum number of concurrent operations to prevent
    overwhelming system resources or external services.

    Common use cases include:
    - Batching database inserts or updates for better throughput
    - Aggregating API calls to services that support bulk operations
    - Optimizing data processing pipelines with chunked operations
    - Building high-performance ETL (Extract, Transform, Load) processes

    Usage involves extending this class to implement custom batch processing logic
    by overriding the _process_batch method with specific implementation details.

    This class implements the async context manager protocol, allowing for use in
    async with statements to ensure proper resource cleanup.
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_concurrency: int = 5,
        flush_interval: Optional[float] = None,
    ):
        """
        Initialize the batch processor with configuration settings.

        Args:
            batch_size: Maximum number of items to collect before processing a batch.
                       Higher values generally improve throughput at the cost of increased
                       latency and memory usage. Default is 100.
            max_concurrency: Maximum number of concurrent batch operations allowed.
                            This prevents overwhelming external services or system
                            resources. Default is 5 concurrent batch operations.
            flush_interval: Optional automatic flush interval in seconds. When specified,
                           any collected items will be flushed after this interval,
                           regardless of whether the batch_size has been reached.
                           Set to None (default) to disable automatic flushing.
        """
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.flush_interval = flush_interval

        self.items = []
        self.flush_task = None
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def add(self, item: Any):
        """
        Add an item to the current batch for processing.

        This method adds the provided item to the internal collection batch.
        The item will be processed when either:
        1. The batch reaches the configured batch_size
        2. The flush_interval elapses (if configured)
        3. A manual flush() is called

        The method automatically triggers a flush operation if the number of
        collected items reaches the configured batch_size.

        If a flush_interval is set and no auto-flush task is running, this method
        also initializes a background task to automatically flush items after
        the specified interval.

        Args:
            item: The item to add to the batch. Can be any type that your
                 _process_batch implementation can handle.

        Returns:
            None

        Example:
            ```python
            processor = MyBatchProcessor(batch_size=50)
            await processor.add({"id": 1, "value": "data"})
            ```
        """
        self.items.append(item)

        # Start flush task if needed
        if self.flush_interval and not self.flush_task:
            self.flush_task = asyncio.create_task(self._auto_flush())

        # Flush if batch is full
        if len(self.items) >= self.batch_size:
            await self.flush()

    async def flush(self) -> List[Any]:
        """
        Process all currently batched items immediately.

        This method forces processing of all currently collected items, regardless
        of whether the batch is full or the flush interval has elapsed. It's useful
        when you need to ensure all items are processed without waiting, such as:

        - When shutting down the application
        - Before a checkpoint or commit operation
        - When immediate processing is needed for time-sensitive data
        - At the end of a processing cycle

        The method handles empty batches gracefully, returning an empty list
        when there are no items to process.

        Returns:
            List of results from processing the batch. The exact content depends
            on what the _process_batch implementation returns. Returns an empty
            list if there were no items to process.

        Example:
            ```python
            # Process any pending items immediately
            results = await processor.flush()
            ```
        """
        if not self.items:
            return []

        # Get current items
        items = self.items
        self.items = []

        # Cancel flush task if running
        if self.flush_task:
            self.flush_task.cancel()
            self.flush_task = None

        # Process the batch
        return await self._process_batch(items)

    async def _auto_flush(self):
        """
        Internal background task for automatic periodic flushing.

        This method implements the auto-flush functionality that runs periodically
        when flush_interval is set. It sleeps for the configured interval and then
        triggers a flush operation if any items are pending.

        The task continues running until:
        1. It's cancelled (typically when a manual flush occurs)
        2. The processor is shut down (via __aexit__)

        This method is not intended to be called directly but is started automatically
        by the add() method when needed and a flush_interval is configured.

        Raises:
            asyncio.CancelledError: When the task is cancelled. This is caught
                                   internally and used to terminate the loop.
        """
        try:
            while True:
                await asyncio.sleep(self.flush_interval)
                if self.items:
                    await self.flush()
        except asyncio.CancelledError:
            # Task was cancelled, which is expected
            pass

    async def _process_batch(self, batch: List[Any]) -> List[Any]:
        """
        Process a batch of items (to be overridden by subclasses).

        This method should be overridden by subclasses to implement the actual
        batch processing logic. The base implementation simply returns the batch
        unchanged and logs a warning, as it's not meant to be used directly.

        When implementing this method in a subclass, typical patterns include:
        - Sending a bulk API request with all batch items
        - Executing a batch database operation
        - Processing items in parallel with controlled concurrency
        - Aggregating items for a combined operation

        Args:
            batch: List of items to process that were collected via add()

        Returns:
            List of processed results, where each result corresponds to an item
            in the input batch. The actual return type depends on the specific
            implementation in the subclass.

        Example implementation:
            ```python
            async def _process_batch(self, batch: List[dict]) -> List[dict]:
                # Add a batch_id to each item
                batch_id = str(uuid.uuid4())
                for item in batch:
                    item['batch_id'] = batch_id

                # Send to database in a single operation
                results = await self.db.insert_many(batch)
                return results
            ```
        """
        # This should be overridden by subclasses
        logger.warning(f"Default batch processing used for {len(batch)} items", emoji_key="warning")
        return batch

    async def __aenter__(self):
        """
        Enter the async context manager.

        Allows the batch processor to be used in an async with statement,
        which ensures proper cleanup when the context is exited.

        Returns:
            The batch processor instance, ready for use.

        Example:
            ```python
            async with MyBatchProcessor(batch_size=100) as processor:
                for item in items:
                    await processor.add(item)
            # All pending items are automatically flushed when the context exits
            ```
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the async context manager.

        This method is called when exiting an async with block. It ensures
        that any pending items are flushed before the context manager completes,
        preventing data loss when the processor goes out of scope.

        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            False, indicating that any exceptions should be propagated.
        """
        # Flush any remaining items
        if self.items:
            await self.flush()
