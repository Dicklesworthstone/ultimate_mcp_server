"""
Comprehensive error handling framework for Model Control Protocol (MCP) systems.

This module implements a consistent, standardized approach to error handling for MCP tools
and services. It provides decorators, formatters, and utilities that transform Python
exceptions into structured, protocol-compliant error responses that LLMs and client
applications can reliably interpret and respond to.

The framework is designed around several key principles:

1. CONSISTENCY: All errors follow the same structured format regardless of their source
2. RECOVERABILITY: Errors include explicit information on whether operations can be retried
3. ACTIONABILITY: Error responses provide specific suggestions for resolving issues
4. DEBUGGABILITY: Rich error details are preserved for troubleshooting
5. CATEGORIZATION: Errors are mapped to standardized types for consistent handling

Key components:
- ErrorType enum: Categorization system for different error conditions
- format_error_response(): Creates standardized error response dictionaries
- with_error_handling: Decorator that catches exceptions and formats responses
- validate_inputs: Decorator for declarative parameter validation
- Validator functions: Reusable validation logic for common parameter types

Usage example:
    ```python
    @with_error_handling
    @validate_inputs(
        prompt=non_empty_string,
        temperature=in_range(0.0, 1.0)
    )
    async def generate_text(prompt, temperature=0.7):
        # Implementation...
        # Any exceptions thrown here will be caught and formatted
        # Input validation happens before execution
        if external_service_down:
            raise Exception("External service unavailable")
        return result
    ```

The error handling pattern is designed to work seamlessly with async functions and
integrates with the MCP protocol's expected error response structure.
"""

import functools
import inspect
import time
import traceback
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union


class ErrorType(str, Enum):
    """Types of errors that can occur in MCP tools."""

    VALIDATION_ERROR = "validation_error"  # Input validation failed
    EXECUTION_ERROR = "execution_error"  # Error during execution
    PERMISSION_ERROR = "permission_error"  # Insufficient permissions
    NOT_FOUND_ERROR = "not_found_error"  # Resource not found
    TIMEOUT_ERROR = "timeout_error"  # Operation timed out
    RATE_LIMIT_ERROR = "rate_limit_error"  # Rate limit exceeded
    EXTERNAL_ERROR = "external_error"  # Error in external service
    UNKNOWN_ERROR = "unknown_error"  # Unknown error


def format_error_response(
    error_type: Union[ErrorType, str],
    message: str,
    details: Optional[Dict[str, Any]] = None,
    retriable: bool = False,
    suggestions: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Format a standardized error response.

    Args:
        error_type: Type of error
        message: Human-readable error message
        details: Additional error details
        retriable: Whether the operation can be retried
        suggestions: List of suggestions for resolving the error

    Returns:
        Formatted error response
    """
    return {
        "success": False,
        "isError": True,  # MCP protocol flag
        "error": {
            "type": error_type if isinstance(error_type, str) else error_type.value,
            "message": message,
            "details": details or {},
            "retriable": retriable,
            "suggestions": suggestions or [],
            "timestamp": time.time(),
        },
    }


def with_error_handling(func: Callable) -> Callable:
    """
    Decorator that provides standardized exception handling for MCP tool functions.

    This decorator intercepts any exceptions raised by the wrapped function and transforms
    them into a structured error response format that follows the MCP protocol. The response
    includes consistent error categorization, helpful suggestions for recovery, and details
    to aid debugging.

    Key features:
    - Automatically categorizes exceptions into appropriate ErrorType values
    - Preserves the original exception message and stack trace
    - Adds relevant suggestions based on the error type
    - Indicates whether the operation can be retried
    - Adds a timestamp for error logging/tracking

    The error response structure always includes:
    - success: False
    - isError: True (MCP protocol flag)
    - error: A dictionary with type, message, details, retriable flag, and suggestions

    Exception mapping:
    - ValueError, TypeError, KeyError, AttributeError → VALIDATION_ERROR (retriable)
    - FileNotFoundError, KeyError, IndexError → NOT_FOUND_ERROR (not retriable)
    - PermissionError, AccessError → PERMISSION_ERROR (not retriable)
    - TimeoutError → TIMEOUT_ERROR (retriable)
    - Exceptions with "rate limit" in message → RATE_LIMIT_ERROR (retriable)
    - All other exceptions → UNKNOWN_ERROR (not retriable)

    Args:
        func: The async function to wrap with error handling

    Returns:
        Decorated async function that catches exceptions and returns structured error responses

    Example:
        ```python
        @with_error_handling
        async def my_tool_function(param1, param2):
            # Function implementation that might raise exceptions
            # If an exception occurs, it will be transformed into a structured response
        ```
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Call the original function
            return await func(*args, **kwargs)
        except Exception as e:
            # Get exception details
            exc_type = type(e).__name__
            exc_message = str(e)
            exc_traceback = traceback.format_exc()

            # Determine error type
            error_type = ErrorType.UNKNOWN_ERROR
            retriable = False

            # Map common exceptions to error types
            if exc_type in ("ValueError", "TypeError", "KeyError", "AttributeError"):
                error_type = ErrorType.VALIDATION_ERROR
                retriable = True
            elif exc_type in ("FileNotFoundError", "KeyError", "IndexError"):
                error_type = ErrorType.NOT_FOUND_ERROR
                retriable = False
            elif exc_type in ("PermissionError", "AccessError"):
                error_type = ErrorType.PERMISSION_ERROR
                retriable = False
            elif exc_type in ("TimeoutError"):
                error_type = ErrorType.TIMEOUT_ERROR
                retriable = True
            elif "rate limit" in exc_message.lower():
                error_type = ErrorType.RATE_LIMIT_ERROR
                retriable = True

            # Generate suggestions based on error type
            suggestions = []
            if error_type == ErrorType.VALIDATION_ERROR:
                suggestions = [
                    "Check that all required parameters are provided",
                    "Verify parameter types and formats",
                    "Ensure parameter values are within allowed ranges",
                ]
            elif error_type == ErrorType.NOT_FOUND_ERROR:
                suggestions = [
                    "Verify the resource ID or path exists",
                    "Check for typos in identifiers",
                    "Ensure the resource hasn't been deleted",
                ]
            elif error_type == ErrorType.RATE_LIMIT_ERROR:
                suggestions = [
                    "Wait before retrying the request",
                    "Reduce the frequency of requests",
                    "Implement backoff strategy for retries",
                ]

            # Format and return error response
            return format_error_response(
                error_type=error_type,
                message=exc_message,
                details={"exception_type": exc_type, "traceback": exc_traceback},
                retriable=retriable,
                suggestions=suggestions,
            )

    return wrapper


def validate_inputs(**validators):
    """
    Decorator for validating tool input parameters against custom validation rules.

    This decorator enables declarative input validation for async tool functions by applying
    validator functions to specified parameters before the decorated function is called.
    If any validation fails, the function returns a standardized error response instead
    of executing, preventing errors from propagating and providing clear feedback on the issue.

    The validation approach supports:
    - Applying different validation rules to different parameters
    - Detailed error messages explaining which parameter failed and why
    - Custom validation logic via any callable that raises ValueError on failure
    - Zero validation overhead for parameters not explicitly validated

    Validator functions should:
    1. Take a single parameter (the value to validate)
    2. Raise a ValueError with a descriptive message if validation fails
    3. Return None or any value (which is ignored) if validation passes
    4. Include a docstring that describes the constraint (used in error messages)

    Args:
        **validators: A mapping of parameter names to validator functions.
            Each key should match a parameter name in the decorated function.
            Each value should be a callable that validates the corresponding parameter.

    Returns:
        Decorator function that wraps an async function with input validation

    Example:
        ```
        # Define validators (or use the provided ones like non_empty_string)
        def validate_temperature(value):
            '''Temperature must be between 0.0 and 1.0.'''
            if not isinstance(value, float) or value < 0.0 or value > 1.0:
                raise ValueError("Temperature must be between 0.0 and 1.0")

        # Apply validation to specific parameters
        @validate_inputs(
            prompt=non_empty_string,
            temperature=validate_temperature,
            max_tokens=positive_number
        )
        async def generate_text(prompt, temperature=0.7, max_tokens=None):
            # This function will only be called if all validations pass
            # Otherwise a standardized error response is returned
            ...

        # The response structure when validation fails:
        # {
        #   "success": False,
        #   "isError": True,
        #   "error": {
        #     "type": "validation_error",
        #     "message": "Invalid value for parameter 'prompt': Value must be a non-empty string",
        #     "details": { ... },
        #     "retriable": true,
        #     "suggestions": [ ... ]
        #   }
        # }
        ```

    Note:
        This decorator should typically be applied before other decorators like
        with_error_handling so that validation errors are correctly formatted.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)

            # Build mapping of parameter names to values
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate inputs
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        # Run validation
                        validator(value)
                    except Exception as e:
                        # Return validation error
                        return format_error_response(
                            error_type=ErrorType.VALIDATION_ERROR,
                            message=f"Invalid value for parameter '{param_name}': {str(e)}",
                            details={
                                "parameter": param_name,
                                "value": str(value),
                                "constraint": str(validator.__doc__ or ""),
                            },
                            retriable=True,
                            suggestions=[
                                f"Provide a valid value for '{param_name}'",
                                "Check the parameter constraints in the tool description",
                            ],
                        )

            # Call the original function if validation passes
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Example validators
def non_empty_string(value):
    """
    Validates that a value is a non-empty string.

    This validator checks that the input is a string type and contains at least
    one non-whitespace character. Empty strings or strings containing only
    whitespace characters are rejected. This is useful for validating required
    text inputs where blank values should not be allowed.

    Args:
        value: The value to validate

    Raises:
        ValueError: If the value is not a string or is empty/whitespace-only
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Value must be a non-empty string")


def positive_number(value):
    """
    Validates that a value is a positive number (greater than zero).

    This validator ensures that the input is either an integer or float
    and has a value greater than zero. Zero or negative values are rejected.
    This is useful for validating inputs like quantities, counts, or rates
    that must be positive.

    Args:
        value: The value to validate

    Raises:
        ValueError: If the value is not a number or is not positive
    """
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("Value must be a positive number")


def in_range(min_val, max_val):
    """
    Creates a validator function for checking if a number falls within a specified range.

    This is a validator factory that returns a custom validator function
    configured with the given minimum and maximum bounds. The returned function
    checks that a value is a number and falls within the inclusive range
    [min_val, max_val]. This is useful for validating inputs that must fall
    within specific limits, such as probabilities, temperatures, or indexes.

    Args:
        min_val: The minimum allowed value (inclusive)
        max_val: The maximum allowed value (inclusive)

    Returns:
        A validator function that checks if values are within the specified range

    Example:
        ```python
        # Create a validator for temperature (0.0 to 1.0)
        validate_temperature = in_range(0.0, 1.0)

        # Use in validation decorator
        @validate_inputs(temperature=validate_temperature)
        async def generate_text(prompt, temperature=0.7):
            # Function body
            ...
        ```
    """

    def validator(value):
        """Value must be between {min_val} and {max_val}."""
        if not isinstance(value, (int, float)) or value < min_val or value > max_val:
            raise ValueError(f"Value must be between {min_val} and {max_val}")

    return validator
