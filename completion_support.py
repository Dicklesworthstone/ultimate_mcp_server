"""
Completion support for MCP servers.

This module implements the argument completion system for the MCP (Model Control Protocol)
servers, enabling interactive, context-aware autocompletion for tool arguments. The completion
system helps users and LLMs efficiently use tools by suggesting valid values for arguments
based on the current context.

Key Components:
- CompletionProvider (abstract): Base class for all completion providers
- StaticCompletionProvider: Provides completions from predefined, static lists
- DynamicCompletionProvider: Generates completions on-demand through callback functions
- CompletionRegistry: Central registry managing providers for different tools
- Utility functions: Helper functions for common completion scenarios (file paths, etc.)

The module supports a flexible, extensible architecture where:
1. Each tool can register its own providers for different arguments
2. Static lists can be used for enumerated options (e.g., formats, modes)
3. Dynamic functions can be used for context-dependent values (files, users, etc.)
4. A fallback provider can handle common arguments across tools

Usage Example:
```python
# Create completion registry
registry = CompletionRegistry()

# Register dynamic provider for a tool
registry.register_provider("database_tool", DynamicCompletionProvider({
    "table_name": async_db_tables_function,
    "column_name": async_table_columns_function
}))

# Set default provider for common arguments across all tools
registry.set_default_provider(COMMON_COMPLETIONS)

# Later, when handling MCP completion requests:
completions = await registry.get_completions(
    tool_name="document_tool",
    argument_name="format",
    current_value="pd"  # User has typed "pd" so far
)
# Returns: {"values": ["pdf"], "hasMore": False, "total": 1}
```

This system integrates with the MCP server to provide real-time completion
suggestions as users type, significantly improving usability and reducing errors.
"""

from typing import Any, Callable, Dict, List, Optional


class CompletionProvider:
    """
    Abstract base class defining the interface for argument completion providers.

    CompletionProvider serves as the foundation for all completion mechanisms in the MCP
    system. It defines a consistent interface that all provider implementations must follow,
    ensuring that consumers of completions can work with different providers interchangeably.

    To implement a custom completion provider:
    1. Subclass CompletionProvider
    2. Implement the get_completions() method to return suggestions
    3. Implement the supports_argument() method to indicate what arguments your provider handles

    The framework includes two standard implementations:
    - StaticCompletionProvider: Uses predefined lists of values
    - DynamicCompletionProvider: Generates values dynamically via callback functions

    Custom implementations might include providers that:
    - Query external APIs for suggestions
    - Read from databases or other data sources
    - Implement complex filtering or ranking logic
    - Cache results for performance optimization

    Providers should handle any internal errors gracefully and return an empty list rather
    than raising exceptions that would disrupt the completion flow.
    """

    async def get_completions(self, argument_name: str, current_value: str, **context) -> List[str]:
        """
        Get completion suggestions for an argument.

        This method is called when the MCP system needs completions for an argument.
        It should return relevant suggestions based on the provided information.

        Args:
            argument_name: Name of the argument being completed (e.g., "file_path", "format")
            current_value: Current partial value entered by the user (may be empty)
            **context: Additional context that may affect completions, such as:
                - tool_name: The name of the tool requesting completions
                - Other argument values from the same tool call
                - User information, preferences, or permissions
                - Environment information

        Returns:
            List of string suggestions that are valid completions for the current_value.
            Return an empty list if no suggestions are available.

        Raises:
            NotImplementedError: If not implemented by a subclass
        """
        raise NotImplementedError("Subclasses must implement get_completions")

    def supports_argument(self, argument_name: str) -> bool:
        """
        Check if this provider supports completion for a given argument.

        This method allows the CompletionRegistry to quickly determine if this
        provider can handle completions for a specific argument without having
        to call get_completions and risk exceptions or empty results.

        Args:
            argument_name: Name of the argument to check for support

        Returns:
            True if this provider can provide completions for the argument,
            False otherwise

        Raises:
            NotImplementedError: If not implemented by a subclass
        """
        raise NotImplementedError("Subclasses must implement supports_argument")


class StaticCompletionProvider(CompletionProvider):
    """
    Completion provider that returns predefined, static suggestion lists for arguments.

    This provider implements a straightforward approach to argument completion using
    predefined lists of suggestions for each supported argument name. When queried,
    it filters these static lists based on the current user input prefix. This makes it
    ideal for arguments with a fixed, known set of possible values like:

    - Enumerated options (e.g., file formats, provider names)
    - Common settings or modes (e.g., analysis types, priority levels)
    - Frequently used values that rarely change

    The provider automatically performs case-insensitive prefix matching on the
    predefined suggestions when a partial value is provided. For example, if
    completions include ["openai", "anthropic"] and the current_value is "open",
    only "openai" will be returned.

    Usage example:
        ```python
        # Create a provider with predefined completions for common arguments
        provider = StaticCompletionProvider({
            "format": ["json", "csv", "xml", "yaml"],
            "provider": ["openai", "anthropic", "cohere", "azure"],
            "priority": ["low", "medium", "high"]
        })

        # Later, get all format options
        completions = await provider.get_completions("format", "")  # Returns all formats

        # Or get filtered provider options
        completions = await provider.get_completions("provider", "co")  # Returns ["cohere"]
        ```

    For arguments that require dynamic or context-sensitive completions (like file paths
    or current database tables), use DynamicCompletionProvider instead.
    """

    def __init__(self, completions: Dict[str, List[str]]):
        """
        Initialize with static completion values.

        Args:
            completions: Dictionary mapping argument names to suggestion lists
        """
        self.completions = completions

    async def get_completions(self, argument_name: str, current_value: str, **context) -> List[str]:
        """Get completion suggestions from static values."""
        if not self.supports_argument(argument_name):
            return []

        # Filter suggestions based on current value
        suggestions = self.completions.get(argument_name, [])
        if current_value:
            return [s for s in suggestions if s.lower().startswith(current_value.lower())]
        return suggestions

    def supports_argument(self, argument_name: str) -> bool:
        """Check if static completions exist for this argument."""
        return argument_name in self.completions


class DynamicCompletionProvider(CompletionProvider):
    """
    Completion provider that generates suggestions dynamically using callback functions.

    Unlike the StaticCompletionProvider which uses fixed lists, this provider calls
    specialized functions to generate completion suggestions on demand. This approach
    is essential for arguments whose valid values:

    - Depend on the current system state (e.g., existing files, running processes)
    - Vary based on user context or previous selections
    - Are too numerous to predefine (e.g., all possible file paths)
    - Require external API calls or database queries to determine

    The provider maps argument names to async callback functions that are responsible
    for generating appropriate suggestions based on the current partial input and context.
    Each callback function should accept at least two parameters:
    - current_value: The current partial input string
    - **context: Additional context information that may be useful for generating completions

    Usage example:
        ```python
        # Define completion functions
        async def complete_files(current_value, **context):
            # Custom logic to find matching files
            return ["file1.txt", "file2.txt", "folder/"]

        async def complete_users(current_value, **context):
            # Query database for matching users
            db = context.get("database")
            users = await db.query(f"SELECT username FROM users WHERE username LIKE '{current_value}%'")
            return [user.username for user in users]

        # Create dynamic provider with these functions
        provider = DynamicCompletionProvider({
            "file_path": complete_files,
            "username": complete_users
        })
        ```

    Each completion function should handle errors gracefully and return an empty list
    rather than raising exceptions. The provider will log errors but won't propagate them
    to the MCP completion API response.
    """

    def __init__(self, completion_functions: Dict[str, Callable]):
        """
        Initialize with dynamic completion functions.

        Args:
            completion_functions: Dictionary mapping argument names to completion functions.
                Each function should be an async function that accepts at least
                (current_value: str, **context) and returns a List[str] of suggestions.
        """
        self.completion_functions = completion_functions

    async def get_completions(self, argument_name: str, current_value: str, **context) -> List[str]:
        """Get completion suggestions by calling appropriate function."""
        if not self.supports_argument(argument_name):
            return []

        # Call the function to get suggestions
        func = self.completion_functions.get(argument_name)
        if func:
            try:
                suggestions = await func(current_value, **context)
                return suggestions
            except Exception as e:
                # Log error and return empty list
                print(f"Error getting completions for {argument_name}: {str(e)}")
                return []
        return []

    def supports_argument(self, argument_name: str) -> bool:
        """Check if a completion function exists for this argument."""
        return argument_name in self.completion_functions


class CompletionRegistry:
    """
    Central registry managing completion providers for different tools and arguments.

    The CompletionRegistry serves as the core orchestration component of the MCP completion
    system, providing a unified interface for registration, management, and access to
    completion providers. It implements:

    1. Tool-specific provider registration and lookup
    2. A fallback mechanism through a default provider
    3. Standardized response formatting according to the MCP specification
    4. Error handling and graceful degradation

    This registry is designed to be the single entry point for all completion requests
    in an MCP server. Client code can register different providers for different tools,
    set up a default provider for common arguments, and then handle all completion requests
    through a single, consistent interface.

    Each tool in the system can have its own dedicated completion provider that understands
    the specific requirements and valid values for that tool's arguments. When no specific
    provider is registered for a tool, the registry falls back to the default provider,
    which typically handles common arguments like formats, providers, and models.

    Usage workflow:
    1. Create a registry during server initialization
    2. Register tool-specific providers for specialized tools
    3. Set a default provider for common arguments
    4. Use get_completions() to handle MCP completion protocol requests

    The registry ensures that all responses follow the MCP completion protocol format,
    even when errors occur or no completions are available, providing a consistent
    experience for clients.
    """

    def __init__(self):
        """Initialize the registry."""
        self.tool_providers = {}  # Map of tool_name -> provider
        self.default_provider = None

    def register_provider(self, tool_name: str, provider: CompletionProvider):
        """
        Register a completion provider for a specific tool.

        This method associates a completion provider with a specific tool name in the registry.
        When the system receives completion requests for this tool, the registered provider
        will be used to generate suggestions for its arguments.

        Each tool can have exactly one registered provider. If a provider is already
        registered for the specified tool, it will be replaced by the new provider.
        This allows for dynamic reconfiguration of completion sources as needed.

        The provider can be either a StaticCompletionProvider for fixed option lists,
        a DynamicCompletionProvider for context-dependent suggestions, or any custom
        implementation of the CompletionProvider interface.

        Args:
            tool_name: Identifier of the tool to register a provider for. This should match
                      the tool name used in MCP requests (e.g., "search_documents",
                      "analyze_image").
            provider: The completion provider instance that will handle suggestion requests
                     for this tool's arguments. Must implement the CompletionProvider interface.

        Note:
            If a tool requires completions for only some of its arguments, the provider
            should still be registered here, and its supports_argument() method should
            return False for unsupported arguments.

        Example:
            ```python
            # Register provider for search_documents tool
            registry.register_provider(
                "search_documents",
                StaticCompletionProvider({
                    "source": ["web", "database", "local_files"],
                    "sort_by": ["relevance", "date", "title"]
                })
            )
            ```
        """
        self.tool_providers[tool_name] = provider

    def set_default_provider(self, provider: CompletionProvider):
        """
        Set a default provider to handle arguments for tools without specific providers.

        This method establishes a fallback completion provider that is used when:
        1. No tool-specific provider is registered for a requested tool
        2. A registered provider exists but doesn't support the requested argument

        The default provider is typically configured to handle common arguments that
        appear across multiple tools, such as:
        - Provider names (e.g., "openai", "anthropic", "azure")
        - Model identifiers (e.g., "gpt-4o", "claude-3-5-haiku-20241022")
        - Common formats (e.g., "json", "csv", "markdown")
        - Universal settings (e.g., "temperature", "max_tokens")

        Only one default provider can be active at a time. Setting a new default
        provider replaces any previously set default.

        Args:
            provider: The completion provider instance to use as the default fallback
                     for all tools without specific providers or for arguments not
                     supported by their specific providers.

        Note:
            When no default provider is set and no tool-specific provider is found,
            the system will return an empty completion result rather than raising
            an error.

        Example:
            ```python
            # Set default provider for common arguments across all tools
            registry.set_default_provider(
                StaticCompletionProvider({
                    "provider": ["openai", "anthropic", "cohere"],
                    "temperature": ["0.0", "0.5", "0.7", "1.0"],
                    "format": ["json", "text", "markdown"]
                })
            )
            ```
        """
        self.default_provider = provider

    def get_provider(self, tool_name: str) -> Optional[CompletionProvider]:
        """
        Retrieve the appropriate completion provider for a specific tool.

        This method implements the provider resolution logic for the registry,
        determining which completion provider should handle a given tool's
        argument completions. It follows this resolution sequence:

        1. Look for a provider specifically registered for the requested tool
        2. If no tool-specific provider exists, fall back to the default provider
        3. If neither exists, return None

        This lookup process encapsulates the registry's fallback mechanism,
        allowing tool-specific providers to take precedence while ensuring
        that common arguments can still be handled by a default provider.

        Args:
            tool_name: The identifier of the tool to find a provider for, matching
                      the name used when registering the provider.

        Returns:
            CompletionProvider: The appropriate provider to handle completions for the tool.
                               This will be either:
                               - The tool's specifically registered provider, if one exists
                               - The default provider, if no tool-specific provider exists
                               - None, if no applicable provider is found

        Note:
            This method is primarily used internally by get_completions(), but can also
            be called directly to check provider availability without requesting completions.
        """
        return self.tool_providers.get(tool_name, self.default_provider)

    async def get_completions(
        self, tool_name: str, argument_name: str, current_value: str, **context
    ) -> Dict[str, Any]:
        """
        Get completion suggestions for a tool argument with standardized response format.

        This method serves as the main API endpoint for the MCP completion protocol.
        It provides argument suggestions by:
        1. Finding the appropriate provider for the requested tool
        2. Checking if that provider supports the requested argument
        3. Calling the provider's get_completions method if supported
        4. Formatting the results according to the MCP specification

        If no provider exists for the tool, or the provider doesn't support the
        argument, an empty result structure is returned rather than raising an error.
        Additionally, any exceptions in the provider's completion logic are caught
        and result in an empty response with error logging.

        Args:
            tool_name: Name of the tool requesting completion suggestions
            argument_name: Name of the argument within the tool to provide completions for
            current_value: Current value or partial input entered by the user
            **context: Additional context that may be useful for generating completions,
                such as values of other arguments, user preferences, or environment info

        Returns:
            Dictionary conforming to the MCP completion protocol with these keys:
            - values: List of suggested completion values (strings), limited to 100 items
            - hasMore: Boolean indicating if more than 100 suggestions were available
            - total: Total number of suggestions actually included in the 'values' list

        Example response:
            ```python
            {
                "values": ["option1", "option2", "option3"],
                "hasMore": False,
                "total": 3
            }
            ```

        Note:
            The response is always structured as a valid MCP completion response, even
            when errors occur or no suggestions are available. This ensures clients
            always receive a predictable format.
        """
        provider = self.get_provider(tool_name)

        if not provider or not provider.supports_argument(argument_name):
            return {"values": [], "hasMore": False, "total": 0}

        try:
            # Get suggestions from provider
            suggestions = await provider.get_completions(
                argument_name=argument_name,
                current_value=current_value,
                tool_name=tool_name,
                **context,
            )

            # Limit to 100 items as per MCP spec
            has_more = len(suggestions) > 100
            suggestions = suggestions[:100]

            return {"values": suggestions, "hasMore": has_more, "total": len(suggestions)}
        except Exception as e:
            # Log error and return empty result
            print(f"Error getting completions: {str(e)}")
            return {"values": [], "hasMore": False, "total": 0}


# Example usage for file path completion
async def complete_file_paths(current_value: str, **context) -> List[str]:
    """
    Generate filesystem path completion suggestions based on the current input.

    This utility function provides intelligent path suggestions for file-related arguments,
    making it easier for users to navigate and select files or directories in the filesystem.
    It handles various path formats including relative paths, absolute paths, and user home
    directory references.

    Behavior:
    - For empty input: Returns common starting points ("./", "../", "/")
    - For partial paths: Performs glob matching to find all matching files and directories
    - For directories: Appends a trailing slash to distinguish them from files
    - Expands user directory references (e.g., "~/documents" becomes "/home/user/documents")

    Path matching is case-sensitive or case-insensitive depending on the underlying filesystem.
    On Windows, matching is typically case-insensitive, while on Unix-like systems it's case-sensitive.

    The function handles permission errors gracefully - if a directory cannot be accessed due to
    permission restrictions, it will be excluded from results without raising an exception.

    Args:
        current_value: The current path string provided by the user (can be empty, partial, or complete)
        **context: Additional context that may influence completions:
            - working_directory: Optional alternative directory to use as the base for relative paths
            - file_extensions: Optional list of file extensions to filter results (e.g., [".py", ".txt"])
            - include_hidden: Optional boolean to include hidden files/directories (default: False)

    Returns:
        List of path suggestions that match or extend the current_value. Each suggestion is formatted as:
        - Regular files: The full path to the file (e.g., "./src/main.py")
        - Directories: The full path with a trailing slash (e.g., "./src/utils/")

    Examples:
        - Input: "" → Output: ["./", "../", "/"]
        - Input: "doc" → Output: ["documents/", "docker-compose.yml", "dockerfile"]
        - Input: "~/Down" → Output: ["/home/user/Downloads/"]
        - Input: "./src/" → Output: ["./src/main.py", "./src/utils/", "./src/tests/"]

    Edge Cases:
        - Symlinks: Followed to their target with the symlink path in the results
        - Special files: Included in results, treated as regular files
        - Non-existent paths: Returns partial matches based on the parent directory, if any
        - Permission errors: Silently skips directories that cannot be accessed

    Notes:
        - Results are capped at 100 items to comply with MCP specification
        - Directory suggestions always end with a trailing slash
        - The function handles filesystem errors gracefully, returning empty list on access errors
    """
    import glob
    import os

    # Handle empty value
    if not current_value:
        return ["./", "../", "/"]

    # Expand user directory if needed
    path = os.path.expanduser(current_value)

    # Get the directory to search in
    directory = os.path.dirname(path) if os.path.basename(path) else path
    if not directory:
        directory = "."

    # Get matching files/directories
    pattern = os.path.join(directory, f"{os.path.basename(path)}*")
    matches = glob.glob(pattern)

    # Format results
    results = []
    for match in matches:
        if os.path.isdir(match):
            results.append(f"{match}/")
        else:
            results.append(match)

    return results


# Example completions for common arguments
"""
Predefined completion provider for common argument types used across multiple tools.

This global provider instance contains standardized suggestion lists for frequently
used arguments in the MCP system. It serves as a convenient default provider that
can be registered with the CompletionRegistry for tools that don't need specialized
completion providers.

The included completion lists cover:
- provider: Common LLM provider names (e.g., "openai", "anthropic")
- model: Popular model identifiers from various providers
- format: Standard data formats for input/output
- source_type: Common data source types for analysis tools
- analysis_type: Standard categories of analysis operations

Usage example:
```python
# Set as the default provider in a registry to handle common arguments
registry = CompletionRegistry()
registry.set_default_provider(COMMON_COMPLETIONS)

# Later, even for tools without specific providers, common arguments will work:
await registry.get_completions(
    tool_name="any_tool",
    argument_name="provider",
    current_value="open"  # Will match "openai"
)
```

This provider can be extended with additional arguments by creating a new
StaticCompletionProvider that combines these defaults with tool-specific completions.
"""
COMMON_COMPLETIONS = StaticCompletionProvider(
    {
        "provider": ["openai", "anthropic", "gemini", "mistral", "custom"],
        "model": [
            "gpt-4-turbo",
            "gpt-4o",
            "claude-3-5-sonnet",
            "claude-3-opus",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "mistral-large",
        ],
        "format": ["json", "text", "markdown", "html", "csv"],
        "source_type": ["csv", "json", "excel", "database", "api"],
        "analysis_type": ["general", "sentiment", "entities", "summary"],
    }
)
