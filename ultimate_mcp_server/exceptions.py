"""
Comprehensive exception system for the Ultimate MCP Server.

This module implements a hierarchical, structured exception system designed to provide
consistent error handling, reporting, and formatting across the MCP ecosystem. The exceptions
are designed to support both internal error handling and MCP protocol-compliant error responses
for client applications and LLMs.

Key design principles:
1. HIERARCHICAL: Exception types form a logical inheritance tree with ToolError at the root
2. CONTEXTUAL: Exceptions carry rich metadata including error codes, details, and context
3. FORMATTABLE: All exceptions can be converted to standard response dictionaries
4. TRACEABLE: Original causes and stack traces are preserved for debugging
5. ACTIONABLE: Error responses include specific information to aid recovery

Exception hierarchy:
- ToolError (base class)
  ├── ToolInputError (parameter validation issues)
  ├── ToolExecutionError (runtime execution failures)
  ├── ProviderError (LLM provider issues)
  │   ├── RateLimitError (provider throttling)
  │   └── AuthenticationError (auth failures)
  ├── ResourceError (resource access/manipulation)
  ├── ValidationError (general validation issues)
  ├── ConfigurationError (config problems)
  └── StorageError (storage operation failures)

The module also provides a format_error_response() function that standardizes any
exception (including non-ToolError exceptions) into a consistent error response
format compatible with the MCP protocol.

Example usage:
    ```python
    # Raising a specific exception with context
    if not os.path.exists(file_path):
        raise ResourceError(
            message="Cannot access required resource",
            resource_type="file",
            resource_id=file_path
        )

    # Catching and formatting an error
    try:
        result = process_data(data)
    except Exception as e:
        # Convert to standard response format for API
        error_response = format_error_response(e)
        return error_response
    ```
"""

import traceback
from typing import Any, Dict


class ToolError(Exception):
    """
    Base exception class for all tool-related errors in the Ultimate MCP Server.

    ToolError serves as the foundation of the MCP error handling system, providing a
    consistent interface for reporting, formatting, and categorizing errors that occur
    during tool execution. All specialized error types in the system inherit from this
    class, ensuring consistent error handling across the codebase.

    This exception class enhances Python's standard Exception with:

    - Error codes: Standardized identifiers for error categorization and programmatic handling
    - HTTP status mapping: Optional association with HTTP status codes for API responses
    - Detailed context: Support for rich error details and contextual information
    - Structured formatting: Conversion to standardized error response dictionaries

    The error hierarchy is designed to provide increasingly specific error types while
    maintaining a consistent structure that can be easily interpreted by error handlers,
    logging systems, and API responses.

    Error responses created from ToolError instances follow the MCP protocol format and
    include consistent fields for error type, message, details, and context.

    Usage example:
        ```python
        try:
            # Some operation that might fail
            result = process_data(data)
        except ToolInputError as e:
            # Handle input validation errors specifically
            log_validation_error(e)
        except ToolError as e:
            # Handle all other tool errors generically
            report_tool_error(e)
        ```
    """

    def __init__(
        self,
        message,
        error_code=None,
        details=None,
        context=None,
        http_status_code: int | None = None,
    ):
        """Initialize the tool error.

        Args:
            message: Error message
            error_code: Error code (for categorization)
            details: Additional error details dictionary
            context: Context dictionary (will be merged into details and stored)
            http_status_code: Optional HTTP status code associated with the error.
        """
        self.error_code = error_code or "TOOL_ERROR"
        self.http_status_code = http_status_code

        # Combine details and context, giving precedence to context if keys overlap
        combined_details = (
            details.copy() if details else {}
        )  # Start with a copy of details or empty dict
        if context and isinstance(context, dict):
            combined_details.update(context)  # Merge context into the combined dict

        self.details = combined_details  # Store the combined dictionary
        self.context = context or {}  # Also store original context separately for compatibility

        super().__init__(message)


class ToolInputError(ToolError):
    """Exception raised for errors in the tool input parameters."""

    def __init__(
        self, message, param_name=None, expected_type=None, provided_value=None, details=None
    ):
        """Initialize the tool input error.

        Args:
            message: Error message
            param_name: Name of the problematic parameter
            expected_type: Expected parameter type
            provided_value: Value that was provided
            details: Additional error details
        """
        error_details = details or {}
        if param_name:
            error_details["param_name"] = param_name
        if expected_type:
            error_details["expected_type"] = str(expected_type)
        if provided_value is not None:
            error_details["provided_value"] = str(provided_value)

        super().__init__(message, error_code="INVALID_PARAMETER", details=error_details)


class ToolExecutionError(ToolError):
    """Exception raised when a tool execution fails."""

    def __init__(self, message, cause=None, details=None):
        """Initialize the tool execution error.

        Args:
            message: Error message
            cause: Original exception that caused the error
            details: Additional error details
        """
        error_details = details or {}
        if cause:
            error_details["cause"] = str(cause)
            error_details["traceback"] = traceback.format_exc()

        super().__init__(message, error_code="EXECUTION_ERROR", details=error_details)


class ProviderError(ToolError):
    """Exception raised for provider-specific errors."""

    def __init__(self, message, provider=None, model=None, cause=None, details=None):
        """Initialize the provider error.

        Args:
            message: Error message
            provider: Name of the provider
            model: Model name
            cause: Original exception that caused the error
            details: Additional error details
        """
        error_details = details or {}
        if provider:
            error_details["provider"] = provider
        if model:
            error_details["model"] = model
        if cause:
            error_details["cause"] = str(cause)
            error_details["traceback"] = traceback.format_exc()

        super().__init__(message, error_code="PROVIDER_ERROR", details=error_details)


class ResourceError(ToolError):
    """Exception raised for resource-related errors."""

    def __init__(self, message, resource_type=None, resource_id=None, cause=None, details=None):
        """Initialize the resource error.

        Args:
            message: Error message
            resource_type: Type of resource (e.g., "document", "embedding")
            resource_id: Resource identifier
            cause: Original exception that caused the error
            details: Additional error details
        """
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if resource_id:
            error_details["resource_id"] = resource_id
        if cause:
            error_details["cause"] = str(cause)

        super().__init__(message, error_code="RESOURCE_ERROR", details=error_details)


class RateLimitError(ProviderError):
    """Exception raised when a provider's rate limit is reached."""

    def __init__(self, message, provider=None, retry_after=None, details=None):
        """Initialize the rate limit error.

        Args:
            message: Error message
            provider: Name of the provider
            retry_after: Seconds to wait before retrying
            details: Additional error details
        """
        error_details = details or {}
        if retry_after is not None:
            error_details["retry_after"] = retry_after

        super().__init__(
            message, provider=provider, error_code="RATE_LIMIT_ERROR", details=error_details
        )


class AuthenticationError(ProviderError):
    """Exception raised when authentication with a provider fails."""

    def __init__(self, message, provider=None, details=None):
        """Initialize the authentication error.

        Args:
            message: Error message
            provider: Name of the provider
            details: Additional error details
        """
        super().__init__(
            message, provider=provider, error_code="AUTHENTICATION_ERROR", details=details
        )


class ValidationError(ToolError):
    """Exception raised when validation of input/output fails."""

    def __init__(self, message, field_errors=None, details=None):
        """Initialize the validation error.

        Args:
            message: Error message
            field_errors: Dictionary of field-specific errors
            details: Additional error details
        """
        error_details = details or {}
        if field_errors:
            error_details["field_errors"] = field_errors

        super().__init__(message, error_code="VALIDATION_ERROR", details=error_details)


class ConfigurationError(ToolError):
    """Exception raised when there is an issue with configuration."""

    def __init__(self, message, config_key=None, details=None):
        """Initialize the configuration error.

        Args:
            message: Error message
            config_key: Key of the problematic configuration
            details: Additional error details
        """
        error_details = details or {}
        if config_key:
            error_details["config_key"] = config_key

        super().__init__(message, error_code="CONFIGURATION_ERROR", details=error_details)


class StorageError(ToolError):
    """Exception raised when there is an issue with storage operations."""

    def __init__(self, message, operation=None, location=None, details=None):
        """Initialize the storage error.

        Args:
            message: Error message
            operation: Storage operation that failed
            location: Location of the storage operation
            details: Additional error details
        """
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        if location:
            error_details["location"] = location

        super().__init__(message, error_code="STORAGE_ERROR", details=error_details)


def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    Format any exception into a standardized MCP-compliant error response dictionary.

    This utility function creates a structured error response that follows the MCP protocol
    format, ensuring consistency in error reporting across different components. It handles
    both ToolError instances (with their rich error metadata) and standard Python exceptions,
    automatically extracting relevant information to create detailed, actionable error responses.

    The function performs special processing for different error types:

    - For ToolError and subclasses: Extracts error code, details, and context from the exception
    - For ToolInputError with path validation: Enhances messages with more user-friendly text
    - For standard Python exceptions: Captures traceback and generates appropriate error codes

    The resulting dictionary always contains these standardized fields:
    - error: Human-readable error message (string)
    - error_code: Categorized error code (string)
    - error_type: Name of the exception class (string)
    - details: Dictionary with additional error information (object)
    - success: Always false for errors (boolean)
    - isError: Always true, used by MCP protocol handlers (boolean)

    Args:
        error: Any exception instance to format into a response

    Returns:
        Dictionary containing standardized error information following the MCP protocol

    Example:
        ```python
        try:
            result = perform_operation()
        except Exception as e:
            error_response = format_error_response(e)
            return error_response  # Ready for API response
        ```
    """
    if isinstance(error, ToolError):
        # For ToolError instances, extract structured information
        error_type = error.__class__.__name__
        error_message = str(error)
        error_details = error.details or {}

        # Include context in the message for better clarity in user-facing errors
        context = getattr(error, "context", None)
        if context and isinstance(context, dict):
            # Create a more specific error message based on error type
            if isinstance(error, ToolInputError):
                # For path validation errors, add more helpful information
                if "path" in context and error_message.endswith("does not exist."):
                    error_message = f"File not found: {context.get('path')}"
                elif "path" in context and "is not a regular file" in error_message:
                    if "directory" in error_message.lower():
                        error_message = f"Cannot read directory as file: {context.get('path')}. Use list_directory instead."
                    else:
                        error_message = f"Path exists but is not a file: {context.get('path')}"

            # Add context to details for more information
            error_details["context"] = context

        # Look for error_type in details if available
        if "error_type" in error_details:
            error_type_from_details = error_details["error_type"]
            # Use this in the response directly
            response_error_type = error_type_from_details
        else:
            response_error_type = error_type

        # Create a standard error response that the demo can easily process
        return {
            "error": error_message,
            "error_code": error.error_code,
            "error_type": response_error_type,
            "details": error_details,
            "success": False,
            "isError": True,
        }
    else:
        # For unknown errors, use the actual error message instead of a generic message
        error_message = str(error)
        if not error_message or error_message.strip() == "":
            error_message = f"Unknown error of type {type(error).__name__}"

        # Match the same response structure for consistency
        return {
            "error": error_message,
            "error_code": "UNKNOWN_ERROR",
            "error_type": type(error).__name__,
            "details": {
                "type": type(error).__name__,
                "message": error_message,
                "traceback": traceback.format_exc(),
            },
            "success": False,
            "isError": True,
        }
