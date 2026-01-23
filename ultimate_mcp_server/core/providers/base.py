"""Base LLM provider interface."""

import abc
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.constants import COST_PER_MILLION_TOKENS, Provider
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


class ModelResponse:
    """
    Standard response format for all LLM provider completions in the Ultimate MCP Server.

    This class provides a unified representation of responses from different LLM providers,
    normalizing their various formats into a consistent structure. It handles common
    operations like:

    - Storing and accessing the generated text content
    - Tracking token usage statistics (input, output, total)
    - Computing cost estimates based on provider pricing
    - Preserving metadata and raw responses for debugging
    - Formatting response data for serialization

    The ModelResponse serves as a bridge between provider-specific response formats
    and the standardized interface presented by the Ultimate MCP Server. This allows
    client code to work with responses in a consistent way regardless of which
    provider generated them.

    All provider implementations in the system should return responses wrapped in
    this class to ensure consistent behavior across the application.
    """

    def __init__(
        self,
        text: str,
        model: str,
        provider: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        processing_time: float = 0.0,
        raw_response: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a standardized model response object.

        This constructor creates a unified response object that normalizes the various
        response formats from different LLM providers into a consistent structure.
        It automatically calculates derived values like total token count and cost
        estimates based on the provided parameters.

        Args:
            text: The generated text content from the LLM. This is the primary output
                 that would be presented to users.
            model: The specific model name that generated this response (e.g., "gpt-4o",
                  "claude-3-5-haiku-20241022").
            provider: The provider name that served this response (e.g., "openai",
                     "anthropic").
            input_tokens: Number of input/prompt tokens consumed in this request.
                         Used for usage tracking and cost calculation.
            output_tokens: Number of output/completion tokens generated in this response.
                          Used for usage tracking and cost calculation.
            total_tokens: Total token count for the request. If not explicitly provided,
                         calculated as input_tokens + output_tokens.
            processing_time: Time taken to generate the response in seconds, measured
                            from request initiation to response completion.
            raw_response: The original, unmodified response object from the provider's API.
                         Useful for debugging and accessing provider-specific data.
            metadata: Additional response metadata as a dictionary. Can contain provider-specific
                     information like finish_reason, logprobs, etc.
        """
        self.text = text
        self.model = model
        self.provider = provider
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens or (input_tokens + output_tokens)
        self.processing_time = processing_time
        self.raw_response = raw_response
        self.metadata = metadata or {}

        # Calculate cost based on token usage
        self.cost = self._calculate_cost()

    def _calculate_cost(self) -> float:
        """
        Calculate the estimated cost of the request based on token usage and current pricing.

        This internal method computes a cost estimate by:
        1. Looking up the per-million-token costs for the specific model used
        2. Applying different rates for input (prompt) and output (completion) tokens
        3. Computing the final cost based on actual token counts

        If pricing data isn't available for the specific model, the method falls back
        to reasonable default estimations and logs a warning.

        Returns:
            Estimated cost in USD as a floating-point value. Returns 0.0 if token counts
            are not available or if the model name is not recognized.

        Note:
            This is an estimation and may not precisely match actual billing from providers,
            especially as pricing changes over time or for custom deployment configurations.
        """
        if not self.model or not self.input_tokens or not self.output_tokens:
            return 0.0

        # Extract model name without provider prefix (e.g., strip "openai/" from "openai/gpt-4o")
        model_name = self.model
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]

        # Get cost per token for this model
        model_costs = COST_PER_MILLION_TOKENS.get(model_name, None)
        if not model_costs:
            # If model not found, use a default estimation
            model_costs = {"input": 0.5, "output": 1.5}
            logger.warning(
                f"Cost data not found for model {self.model}. Using estimates.", emoji_key="cost"
            )

        # Calculate cost
        input_cost = (self.input_tokens / 1_000_000) * model_costs["input"]
        output_cost = (self.output_tokens / 1_000_000) * model_costs["output"]

        return input_cost + output_cost

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response object to a dictionary suitable for serialization.

        This method creates a structured dictionary representation of the response
        that can be easily serialized to JSON or other formats. The dictionary
        preserves all important fields while organizing them into a clean,
        hierarchical structure.

        The token usage statistics are grouped under a 'usage' key, making it
        easier to access and analyze metrics separately from the content.

        Returns:
            A dictionary containing all relevant response data with the following structure:
            {
                "text": str,              # The generated text content
                "model": str,             # Model name used
                "provider": str,          # Provider name
                "usage": {                # Token usage statistics
                    "input_tokens": int,
                    "output_tokens": int,
                    "total_tokens": int
                },
                "processing_time": float, # Time taken in seconds
                "cost": float,            # Estimated cost in USD
                "metadata": dict          # Additional response metadata
            }

        Example:
            ```python
            response = await provider.generate_completion(prompt="Hello")
            response_dict = response.to_dict()
            json_response = json.dumps(response_dict)
            ```
        """
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
            "processing_time": self.processing_time,
            "cost": self.cost,
            "metadata": self.metadata,
        }


class BaseProvider(abc.ABC):
    """
    Abstract base class that defines the interface for all LLM providers in Ultimate MCP Server.

    This class establishes the common API contract that all provider implementations must follow,
    ensuring consistent behavior regardless of the underlying LLM service (OpenAI, Anthropic, etc.).
    It standardizes key operations like:

    - Provider initialization and API key management
    - Text completion generation (both synchronous and streaming)
    - Model listing and default model selection
    - API key validation
    - Request timing and performance tracking

    By implementing this interface, each provider ensures compatibility with the broader
    Ultimate MCP Server framework. This abstraction layer allows the system to work with multiple
    LLM providers interchangeably, while hiding the provider-specific implementation details
    from the rest of the application.

    Provider implementations should inherit from this class and override all abstract methods.
    They may also extend the interface with provider-specific functionality as needed,
    though core components of the Ultimate MCP Server should rely only on the methods defined
    in this base class to ensure provider-agnostic operation.

    Usage example:
        ```python
        class OpenAIProvider(BaseProvider):
            provider_name = "openai"

            async def initialize(self) -> bool:
                # OpenAI-specific initialization...

            async def generate_completion(self, prompt: str, **kwargs) -> ModelResponse:
                # OpenAI-specific completion implementation...

            # Other required method implementations...
        ```
    """

    provider_name: str = "base"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the provider.

        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific options
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = None  # No longer try to get from env, will be provided by config system

        self.api_key = api_key
        self.options = kwargs
        self.client = None
        self.logger = get_logger(f"provider.{self.provider_name}")

    @abc.abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the provider client and verify API connectivity.

        This abstract method defines the standard interface for initializing a provider
        connection. All provider implementations must override this method with their
        provider-specific initialization logic while maintaining this signature.

        The initialization process typically includes:
        1. Creating the provider-specific client with the API key and configuration
        2. Setting up any required HTTP headers, authentication, or session management
        3. Verifying API connectivity with a lightweight request when possible
        4. Setting up provider-specific rate limiting or retry mechanisms
        5. Loading any required provider-specific resources or configurations

        This method is called:
        - When a provider is first instantiated via the get_provider factory
        - When a provider connection needs to be refreshed or re-established
        - Before any operations that require an active client connection

        Returns:
            bool: True if initialization was successful and the provider is ready for use,
                 False if initialization failed for any reason. A False return will
                 typically prevent the provider from being used by the calling code.

        Raises:
            No exceptions should be raised directly. All errors should be handled
            internally, logged appropriately, and reflected in the return value.
            If initialization fails, detailed error information should be logged
            to help diagnose the issue.

        Implementation guidelines:
            - Handle API keys securely, avoiding logging them even in error messages
            - Implement retries with exponential backoff for transient errors
            - Set reasonable timeouts on API connection attempts
            - Log detailed diagnostics on initialization failures
            - Cache expensive resources to improve subsequent initialization times
        """
        pass

    @abc.abstractmethod
    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> ModelResponse:
        """
        Generate a text completion from the provider (non-streaming).

        This abstract method defines the standard interface for generating text completions
        from any LLM provider. All provider implementations must override this method with
        their provider-specific implementation while maintaining this signature.

        The method handles sending a prompt to the LLM, processing the response, and
        converting it to the standardized ModelResponse format. It is responsible for
        handling provider-specific API calls, error handling, and token counting.

        Args:
            prompt: The text prompt to send to the model. This is the primary input that
                   the model will generate a completion for.
            model: The specific model identifier to use (e.g., "gpt-4o", "claude-3-opus").
                  If None, the provider's default model will be used.
            max_tokens: Maximum number of tokens to generate in the response. If None,
                       the provider's default or maximum limit will be used.
            temperature: Controls randomness in the output. Lower values (e.g., 0.1) make
                        the output more deterministic, while higher values (e.g., 1.0)
                        make it more random and creative. Range is typically 0.0-1.0
                        but may vary by provider.
            **kwargs: Additional provider-specific parameters such as:
                     - top_p: Nucleus sampling parameter (alternative to temperature)
                     - stop_sequences: Tokens/strings that will stop generation when encountered
                     - frequency_penalty: Penalty for using frequent tokens
                     - presence_penalty: Penalty for repeated tokens
                     - system_prompt: System instructions for providers that support it
                     - response_format: Structured format request (e.g., JSON)

        Returns:
            ModelResponse: A standardized response object containing:
                         - The generated text
                         - Token usage statistics (input, output, total)
                         - Cost estimation
                         - Processing time
                         - Provider and model information
                         - Any provider-specific metadata

        Raises:
            ValueError: For invalid parameter combinations or values
            ConnectionError: For network or API connectivity issues
            AuthenticationError: For API key or authentication problems
            RateLimitError: When provider rate limits are exceeded
            ProviderError: For other provider-specific errors

        Implementation guidelines:
            - Use the provider's official client library when available
            - Handle error conditions gracefully with meaningful error messages
            - Track token usage precisely for accurate cost estimation
            - Measure processing time with the process_with_timer utility
            - Include relevant provider-specific metadata in the response
        """
        pass

    @abc.abstractmethod
    async def generate_completion_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """
        Generate a streaming text completion with real-time token delivery.

        This abstract method defines the standard interface for streaming text completions
        from any LLM provider. All provider implementations must override this method with
        their provider-specific streaming implementation while maintaining this signature.

        Unlike the non-streaming generate_completion method, this method:
        - Returns content incrementally as it's generated
        - Uses an async generator that yields content chunks
        - Provides metadata with each chunk to track generation progress
        - Enables real-time display and processing of partial responses

        Args:
            prompt: The text prompt to send to the model. This is the primary input that
                   the model will generate a completion for.
            model: The specific model identifier to use (e.g., "gpt-4o", "claude-3-opus").
                  If None, the provider's default model will be used.
            max_tokens: Maximum number of tokens to generate in the response. If None,
                       the provider's default or maximum limit will be used.
            temperature: Controls randomness in the output. Lower values (e.g., 0.1) make
                        the output more deterministic, while higher values (e.g., 1.0)
                        make it more random and creative. Range is typically 0.0-1.0
                        but may vary by provider.
            **kwargs: Additional provider-specific parameters, identical to those
                     supported by generate_completion.

        Yields:
            Tuple[str, Dict[str, Any]]: Each yield returns:
                - str: The next chunk of generated text
                - Dict: Metadata about the generation process, including at minimum:
                  - done: Boolean indicating if this is the final chunk
                  - chunk_index: Integer index of the current chunk (0-based)

                  The metadata may also include provider-specific information such as:
                  - finish_reason: Why the generation stopped (e.g., "stop", "length")
                  - token_count: Running count of tokens generated
                  - model: Model information if it changed during generation

        Raises:
            ValueError: For invalid parameter combinations or values
            ConnectionError: For network or API connectivity issues
            AuthenticationError: For API key or authentication problems
            RateLimitError: When provider rate limits are exceeded
            ProviderError: For other provider-specific errors

        Implementation guidelines:
            - Use the provider's official streaming endpoints when available
            - Ensure chunks represent logical breaks where possible (e.g., words, not partial UTF-8)
            - Handle connection interruptions gracefully
            - Set the 'done' flag to True only in the final yielded chunk
            - Provide consistent metadata structure across all yielded chunks
        """
        pass

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from this provider with their capabilities and metadata.

        This method retrieves information about all available models from the provider,
        including their identifiers, capabilities, and contextual metadata. Providers
        typically override this method to query their API's model list endpoint and
        normalize the responses into a consistent format.

        The base implementation returns a minimal default model entry, but provider-specific
        implementations should:
        1. Query the provider's models API or endpoint
        2. Transform provider-specific model data into the standard format
        3. Enrich the models with useful metadata like token limits and capabilities
        4. Filter models based on access permissions if applicable

        Returns:
            A list of dictionaries, each representing a model with at least these keys:
            - id (str): The model identifier (e.g., "gpt-4o", "claude-3-opus")
            - provider (str): The provider name (e.g., "openai", "anthropic")
            - description (str): A human-readable description of the model

            Models may also include additional metadata such as:
            - max_tokens (int): Maximum combined tokens (prompt + completion)
            - created (str): Creation/version date of the model
            - pricing (dict): Cost information for input/output tokens
            - capabilities (list): Features the model supports (e.g., "vision", "function_calling")
            - deprecated (bool): Whether the model is deprecated or scheduled for retirement

        Raises:
            ConnectionError: If the provider API cannot be reached
            AuthenticationError: If authentication fails during the request
            ProviderError: For other provider-specific errors

        Note:
            Model data may be cached internally to reduce API calls. Providers should
            implement appropriate caching strategies to balance freshness with performance.
        """
        # Default implementation - override in provider-specific classes
        return [
            {
                "id": "default-model",
                "provider": self.provider_name,
                "description": "Default model",
            }
        ]

    def get_default_model(self) -> str:
        """
        Get the default model identifier for this provider.

        This method returns the standard or recommended model identifier to use when
        no specific model is requested. Each provider implementation must override this
        method to specify its default model.

        The default model should be:
        - Generally available to all users of the provider
        - Well-balanced between capabilities and cost
        - Appropriate for general-purpose text generation tasks
        - Stable and reliable for production use

        The implementation should consider:
        1. Provider-specific configuration settings
        2. Environment variables or system settings
        3. User access level and permissions
        4. Regional availability of models
        5. Current model deprecation status

        Returns:
            str: The model identifier string (e.g., "gpt-4o", "claude-3-haiku")
                that will be used when no model is explicitly specified.
                This identifier should be valid and usable without additional prefixing.

        Raises:
            NotImplementedError: In the base class implementation, signaling that
                               subclasses must override this method.

        Note:
            Provider implementations should periodically review and update their default
            model selections as newer, more capable models become available or as pricing
            structures change.
        """
        raise NotImplementedError("Provider must implement get_default_model")

    async def check_api_key(self) -> bool:
        """
        Check if the API key for this provider is valid and functional.

        This method verifies that the configured API key is valid and can be used
        to authenticate with the provider's API. The default implementation simply
        checks if an API key is present, but provider-specific implementations
        should override this to perform actual validation against the provider's API.

        A proper implementation should:
        1. Make a lightweight API call to an endpoint that requires authentication
        2. Handle authentication errors specifically to differentiate from other failures
        3. Cache results when appropriate to avoid excessive validation calls
        4. Respect rate limits during validation

        This method is typically called during:
        - Server startup to verify all configured providers
        - Before first use of a provider to ensure it's properly configured
        - Periodically as a health check to detect expired or revoked keys
        - After configuration changes that might affect authentication

        Returns:
            bool: True if the API key is valid and usable, False otherwise.
                 The default implementation returns True if an API key is present,
                 which does not guarantee the key is actually valid or functional.

        Note:
            Provider implementations should log descriptive error messages when
            validation fails to help with troubleshooting, but should avoid logging
            the actual API key or other sensitive credentials.
        """
        # Default implementation just checks if key exists
        return bool(self.api_key)

    async def process_with_timer(self, func: callable, *args, **kwargs) -> Tuple[Any, float]:
        """
        Process an async function call with precise timing measurement.

        This utility method provides a standardized way to execute any async function
        while measuring its execution time with high precision. It's particularly useful for:

        - Tracking LLM API call latency for performance monitoring
        - Measuring request-response round trip times
        - Providing accurate timing data for usage reports and optimizations
        - Including processing time in log messages and response metadata

        The method handles the time measurement boilerplate, ensuring consistent
        timing practices across all provider implementations. The measured processing
        time is returned alongside the function's result, allowing both to be used
        in subsequent operations.

        Args:
            func: The async function (callable) to execute and time. This should be
                 an awaitable function that performs the actual operation.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Tuple containing:
              - The result returned by the executed function (any type)
              - The processing time in seconds as a float, measured with
                high precision from just before the function call to just after
                it completes.

        Example usage:
            ```python
            # Timing an API call
            response, duration = await self.process_with_timer(
                self.client.create,
                model="gpt-4o",
                prompt="Hello, world!"
            )

            # Using the measured time in a response
            return ModelResponse(
                text=response.choices[0].text,
                model="gpt-4o",
                provider=self.provider_name,
                processing_time=duration
            )
            ```
        """
        start_time = time.time()
        result = await func(*args, **kwargs)
        processing_time = time.time() - start_time

        return result, processing_time


def parse_model_string(model_string: str) -> Tuple[str, str]:
    """Parse a model string that might include a provider prefix.

    This function parses a model identifier string that may include a provider prefix
    (e.g., 'openai/gpt-4o' or 'anthropic:claude-3-sonnet'). It supports two separator
    formats: forward slash ('/') and colon (':'). If a valid provider prefix is found,
    the function returns the provider name and model name as separate strings.

    Provider validation is performed against the Provider enum values to ensure the
    prefix represents a supported provider. If no valid provider prefix is found, the
    provider component will be None, indicating the model should use the default provider.

    This function is particularly useful in contexts where users can specify models with optional
    provider prefixes, allowing the system to route requests to the appropriate provider
    even when the provider isn't explicitly specified elsewhere.

    Args:
        model_string: A model string, possibly including a provider prefix.
                     Examples: "openai/gpt-4.1-mini", "anthropic/claude-3-opus",
                               "gemini:gemini-pro", "gpt-4o" (no provider)

    Returns:
        Tuple of (provider_name, model_name):
        - provider_name (str or None): Lowercase provider name if a valid prefix was found,
          or None if no valid provider prefix was detected.
        - model_name (str): The model identifier without the provider prefix.

    Examples:
        >>> parse_model_string("openai/gpt-4o")
        ('openai', 'gpt-4o')

        >>> parse_model_string("anthropic:claude-3-opus")
        ('anthropic', 'claude-3-opus')

        >>> parse_model_string("gpt-4o")  # No provider prefix
        (None, 'gpt-4o')

        >>> parse_model_string("unknown/model-name")  # Invalid provider
        (None, 'unknown/model-name')
    """
    separator = None
    if "/" in model_string:
        separator = "/"
    elif ":" in model_string:
        separator = ":"

    if separator:
        # Try to extract provider prefix
        parts = model_string.split(separator, 1)
        if len(parts) == 2:
            provider_prefix, model_name = parts

            # Check if the prefix is a valid provider name
            # Use list comprehension for cleaner check against Provider enum values
            valid_providers = [p.value.lower() for p in Provider]
            if provider_prefix.lower() in valid_providers:
                return provider_prefix.lower(), model_name

    # No valid provider prefix found or no separator
    return None, model_string


async def get_provider(provider_name: str, **kwargs) -> BaseProvider:
    """
    Factory function to dynamically create and initialize a provider instance by name.

    This function serves as the central provider instantiation mechanism in the Ultimate MCP Server,
    dynamically creating and initializing the appropriate provider implementation based on
    the requested provider name. It handles:

    1. Provider name validation and normalization
    2. Provider class selection based on the standardized Provider enum
    3. Model string parsing to extract provider information from model identifiers
    4. Configuration retrieval from the Ultimate MCP Server config system
    5. Provider instance creation with appropriate parameters
    6. Provider initialization and validation

    The function supports specifying provider names directly or extracting them from
    model identifiers that include provider prefixes (e.g., "openai/gpt-4o"). This flexibility
    allows for more intuitive access to providers when working with specific models.

    Args:
        provider_name: Provider identifier to instantiate. This should match one of the
                      values in the Provider enum (case-insensitive). Examples include
                      "openai", "anthropic", "gemini", etc.
        **kwargs: Additional provider-specific configuration options to pass to the
                 provider's constructor. Common options include:
                 - api_key: Override the API key from configuration
                 - model: Model name to use (may include provider prefix)
                 - base_url: Alternative API endpoint URL
                 - organization: Organization ID for providers that support it

    Returns:
        An initialized provider instance ready for use. The specific return type will
        be a subclass of BaseProvider corresponding to the requested provider.

    Raises:
        ValueError: If the provider name is invalid or initialization fails. This ensures
                   that only fully functional provider instances are returned.

    Example usage:
        ```python
        # Basic usage with direct provider name
        openai_provider = await get_provider("openai")

        # Using a model string with provider prefix
        provider = await get_provider("openai", model="anthropic/claude-3-opus")
        # The above actually returns an AnthropicProvider because the model string
        # overrides the provider_name parameter

        # With additional configuration
        custom_provider = await get_provider(
            "openai",
            api_key="custom-key",
            base_url="https://custom-endpoint.example.com/v1",
            model="gpt-4o"
        )
        ```
    """
    cfg = get_config()
    provider_name = provider_name.lower().strip()

    # If a model was provided, check if it has a provider prefix
    # This helps with models like "openai/gpt-4.1-mini" to ensure they go to the right provider
    if "model" in kwargs and isinstance(kwargs["model"], str):
        extracted_provider, extracted_model = parse_model_string(kwargs["model"])
        if extracted_provider:
            # If we have a provider prefix in the model string, use that provider
            # and update the model name to remove the prefix
            provider_name = extracted_provider
            kwargs["model"] = extracted_model
            logger.debug(
                f"Extracted provider '{provider_name}' and model '{extracted_model}' from model string"
            )

    from ultimate_mcp_server.core.providers.anthropic import AnthropicProvider
    from ultimate_mcp_server.core.providers.deepseek import DeepSeekProvider
    from ultimate_mcp_server.core.providers.gemini import GeminiProvider
    from ultimate_mcp_server.core.providers.grok import GrokProvider
    from ultimate_mcp_server.core.providers.ollama import OllamaProvider
    from ultimate_mcp_server.core.providers.openai import OpenAIProvider
    from ultimate_mcp_server.core.providers.openrouter import OpenRouterProvider

    providers = {
        Provider.OPENAI: OpenAIProvider,
        Provider.ANTHROPIC: AnthropicProvider,
        Provider.DEEPSEEK: DeepSeekProvider,
        Provider.GEMINI: GeminiProvider,
        Provider.OPENROUTER: OpenRouterProvider,
        Provider.GROK: GrokProvider,
        Provider.OLLAMA: OllamaProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"Invalid provider name: {provider_name}")

    # Get the top-level 'providers' config object, default to None if it doesn't exist
    providers_config = getattr(cfg, "providers", None)

    # Get the specific provider config (e.g., providers_config.openai) from the providers_config object
    # Default to None if providers_config is None or the specific provider attr doesn't exist
    provider_cfg = getattr(providers_config, provider_name, None) if providers_config else None

    # Now use provider_cfg to get the api_key if needed
    if (
        "api_key" not in kwargs
        and provider_cfg
        and hasattr(provider_cfg, "api_key")
        and provider_cfg.api_key
    ):
        kwargs["api_key"] = provider_cfg.api_key

    provider_class = providers[provider_name]
    instance = provider_class(**kwargs)

    # Initialize the provider immediately
    initialized = await instance.initialize()
    if not initialized:
        # Raise an error if initialization fails to prevent returning an unusable instance
        raise ValueError(f"Failed to initialize provider: {provider_name}")

    return instance
