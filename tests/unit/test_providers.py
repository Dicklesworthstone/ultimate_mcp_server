"""Tests for the provider implementations."""

from typing import Any, Dict

import pytest
from pytest import MonkeyPatch

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.anthropic import AnthropicProvider
from ultimate_mcp_server.core.providers.base import (
    BaseProvider,
    ModelResponse,
    get_provider,
)
from ultimate_mcp_server.core.providers.deepseek import DeepSeekProvider
from ultimate_mcp_server.core.providers.gemini import GeminiProvider
from ultimate_mcp_server.core.providers.openai import OpenAIProvider
from ultimate_mcp_server.utils import get_logger

logger = get_logger("test.providers")

# Set the loop scope for all tests - function scope is recommended for isolated test execution
pytestmark = pytest.mark.asyncio(loop_scope="function")


class TestBaseProvider:
    """Tests for the base provider class."""

    def test_init(self):
        """Test provider initialization."""
        logger.info("Testing base provider initialization", emoji_key="test")

        class TestProvider(BaseProvider):
            provider_name = "test"

            async def initialize(self):
                return True

            async def generate_completion(self, prompt, **kwargs):
                return ModelResponse(
                    text="Test response", model="test-model", provider=self.provider_name
                )

            async def generate_completion_stream(self, prompt, **kwargs):
                yield "Test", {}

            def get_default_model(self):
                return "test-model"

        provider = TestProvider(api_key="test-key", test_option="value")

        assert provider.api_key == "test-key"
        assert provider.options == {"test_option": "value"}
        assert provider.provider_name == "test"

    async def test_process_with_timer(self, mock_provider: BaseProvider):
        """Test the process_with_timer utility method."""
        logger.info("Testing process_with_timer", emoji_key="test")

        # Create a mock async function that returns a value
        async def mock_func(arg1, arg2=None):
            return {"result": arg1 + str(arg2 or "")}

        # Process with timer
        result, time_taken = await mock_provider.process_with_timer(mock_func, "test", arg2="arg")

        assert result == {"result": "testarg"}
        assert isinstance(time_taken, float)
        assert time_taken >= 0  # Time should be non-negative

    def test_model_response(self):
        """Test the ModelResponse class."""
        logger.info("Testing ModelResponse", emoji_key="test")

        # Create a response with minimal info
        response = ModelResponse(text="Test response", model="test-model", provider="test")

        assert response.text == "Test response"
        assert response.model == "test-model"
        assert response.provider == "test"
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.total_tokens == 0
        assert response.cost == 0.0  # No tokens, no cost

        # Create a response with token usage
        response = ModelResponse(
            text="Test response with tokens",
            model="gpt-4o",  # A model with known cost
            provider="openai",
            input_tokens=100,
            output_tokens=50,
        )

        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.total_tokens == 150
        assert response.cost > 0.0  # Should have calculated a cost

        # Test dictionary conversion
        response_dict = response.to_dict()
        assert response_dict["text"] == "Test response with tokens"
        assert response_dict["model"] == "gpt-4o"
        assert response_dict["provider"] == "openai"
        assert response_dict["usage"]["input_tokens"] == 100
        assert response_dict["usage"]["output_tokens"] == 50
        assert response_dict["usage"]["total_tokens"] == 150
        assert "cost" in response_dict

    def test_get_provider_factory(self, mock_env_vars):
        """Test the get_provider factory function."""
        logger.info("Testing get_provider factory", emoji_key="test")

        # Test getting a provider by name
        openai_provider = get_provider(Provider.OPENAI.value)
        assert isinstance(openai_provider, OpenAIProvider)
        assert openai_provider.provider_name == Provider.OPENAI.value

        # Test with different provider
        anthropic_provider = get_provider(Provider.ANTHROPIC.value)
        assert isinstance(anthropic_provider, AnthropicProvider)
        assert anthropic_provider.provider_name == Provider.ANTHROPIC.value

        # Test with invalid provider
        with pytest.raises(ValueError):
            get_provider("invalid-provider")

        # Test with custom API key
        custom_key_provider = get_provider(Provider.OPENAI.value, api_key="custom-key")
        assert custom_key_provider.api_key == "custom-key"


class TestOpenAIProvider:
    """Tests for the OpenAI provider."""

    @pytest.fixture
    def mock_openai_responses(self) -> Dict[str, Any]:
        """Mock responses for OpenAI API."""

        # Create proper class-based mocks with attributes instead of dictionaries
        class MockCompletion:
            def __init__(self):
                self.id = "mock-completion-id"
                self.choices = [MockChoice()]
                self.usage = MockUsage()

        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
                self.finish_reason = "stop"

        class MockMessage:
            def __init__(self):
                self.content = "Mock OpenAI response"

        class MockUsage:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 5
                self.total_tokens = 15

        class MockModelsResponse:
            def __init__(self):
                self.data = [
                    type("MockModel", (), {"id": "gpt-4o", "owned_by": "openai"}),
                    type("MockModel", (), {"id": "gpt-4.1-mini", "owned_by": "openai"}),
                    type("MockModel", (), {"id": "gpt-4.1-mini", "owned_by": "openai"}),
                ]

        return {"completion": MockCompletion(), "models": MockModelsResponse()}

    @pytest.fixture
    def mock_openai_provider(
        self, monkeypatch: MonkeyPatch, mock_openai_responses: Dict[str, Any]
    ) -> OpenAIProvider:
        """Get a mock OpenAI provider with patched methods."""
        # Create the provider
        provider = OpenAIProvider(api_key="mock-openai-key")

        # Mock the AsyncOpenAI client methods
        class MockAsyncOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.chat = MockChat()
                self.models = MockModels()

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class MockCompletions:
            async def create(self, **kwargs):
                return mock_openai_responses["completion"]

        class MockModels:
            async def list(self):
                return mock_openai_responses["models"]

        # Patch the AsyncOpenAI client
        monkeypatch.setattr("openai.AsyncOpenAI", MockAsyncOpenAI)

        # Initialize the provider with the mock client
        provider.client = MockAsyncOpenAI(api_key="mock-openai-key")

        return provider

    async def test_initialization(self, mock_openai_provider: OpenAIProvider):
        """Test OpenAI provider initialization."""
        logger.info("Testing OpenAI provider initialization", emoji_key="test")

        # Initialize
        success = await mock_openai_provider.initialize()
        assert success
        assert mock_openai_provider.client is not None

    async def test_completion(self, mock_openai_provider: OpenAIProvider):
        """Test OpenAI completion generation."""
        logger.info("Testing OpenAI completion", emoji_key="test")

        # Generate completion
        result = await mock_openai_provider.generate_completion(
            prompt="Test prompt", model="gpt-4o", temperature=0.7
        )

        # Check result
        assert isinstance(result, ModelResponse)
        assert result.text == "Mock OpenAI response"
        assert result.model == "gpt-4o"
        assert result.provider == Provider.OPENAI.value
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.total_tokens == 15

    async def test_list_models(self, mock_openai_provider: OpenAIProvider):
        """Test listing OpenAI models."""
        logger.info("Testing OpenAI list_models", emoji_key="test")

        # Initialize first
        await mock_openai_provider.initialize()

        # List models
        models = await mock_openai_provider.list_models()

        # Should return filtered list of models (chat-capable)
        assert isinstance(models, list)
        assert len(models) > 0

        # Check model format
        for model in models:
            assert "id" in model
            assert "provider" in model
            assert model["provider"] == Provider.OPENAI.value

    def test_default_model(self, mock_openai_provider: OpenAIProvider):
        """Test getting default model."""
        logger.info("Testing OpenAI default_model", emoji_key="test")

        # Should return a default model
        model = mock_openai_provider.get_default_model()
        assert model is not None
        assert isinstance(model, str)


class TestAnthropicProvider:
    """Tests for the Anthropic provider."""

    @pytest.fixture
    def mock_anthropic_responses(self) -> Dict[str, Any]:
        """Mock responses for Anthropic API."""

        class MockMessage:
            def __init__(self):
                # Content should be an array of objects with text property
                self.content = [type("ContentBlock", (), {"text": "Mock Claude response"})]
                self.usage = type("Usage", (), {"input_tokens": 20, "output_tokens": 10})

        return {"message": MockMessage()}

    @pytest.fixture
    def mock_anthropic_provider(
        self, monkeypatch: MonkeyPatch, mock_anthropic_responses: Dict[str, Any]
    ) -> AnthropicProvider:
        """Get a mock Anthropic provider with patched methods."""
        # Create the provider
        provider = AnthropicProvider(api_key="mock-anthropic-key")

        # Mock the AsyncAnthropic client methods
        class MockAsyncAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.messages = MockMessages()

        class MockMessages:
            async def create(self, **kwargs):
                return mock_anthropic_responses["message"]

            async def stream(self, **kwargs):
                class MockStream:
                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, exc_type, exc_val, exc_tb):
                        pass

                    async def __aiter__(self):
                        yield type(
                            "MockChunk",
                            (),
                            {
                                "type": "content_block_delta",
                                "delta": type("MockDelta", (), {"text": "Mock streaming content"}),
                            },
                        )

                    async def get_final_message(self):
                        return mock_anthropic_responses["message"]

                return MockStream()

        # Patch the AsyncAnthropic client
        monkeypatch.setattr("anthropic.AsyncAnthropic", MockAsyncAnthropic)

        # Initialize the provider with the mock client
        provider.client = MockAsyncAnthropic(api_key="mock-anthropic-key")

        return provider

    async def test_initialization(self, mock_anthropic_provider: AnthropicProvider):
        """Test Anthropic provider initialization."""
        logger.info("Testing Anthropic provider initialization", emoji_key="test")

        # Initialize
        success = await mock_anthropic_provider.initialize()
        assert success
        assert mock_anthropic_provider.client is not None

    async def test_completion(self, mock_anthropic_provider: AnthropicProvider):
        """Test Anthropic completion generation."""
        logger.info("Testing Anthropic completion", emoji_key="test")

        # Generate completion
        result = await mock_anthropic_provider.generate_completion(
            prompt="Test prompt", model="claude-3-sonnet-20240229", temperature=0.7
        )

        # Check result
        assert isinstance(result, ModelResponse)
        assert result.text == "Mock Claude response"
        assert result.model == "claude-3-sonnet-20240229"
        assert result.provider == Provider.ANTHROPIC.value
        assert result.input_tokens == 20
        assert result.output_tokens == 10
        assert result.total_tokens == 30

    async def test_list_models(self, mock_anthropic_provider: AnthropicProvider):
        """Test listing Anthropic models."""
        logger.info("Testing Anthropic list_models", emoji_key="test")

        # Initialize first
        await mock_anthropic_provider.initialize()

        # List models
        models = await mock_anthropic_provider.list_models()

        # Should return a list of models
        assert isinstance(models, list)
        assert len(models) > 0

        # Check model format
        for model in models:
            assert "id" in model
            assert "provider" in model
            assert model["provider"] == Provider.ANTHROPIC.value

    def test_default_model(self, mock_anthropic_provider: AnthropicProvider):
        """Test getting default model."""
        logger.info("Testing Anthropic default_model", emoji_key="test")

        # Should return a default model
        model = mock_anthropic_provider.get_default_model()
        assert model is not None
        assert isinstance(model, str)


# Brief tests for the other providers to save space
class TestOtherProviders:
    """Brief tests for other providers."""

    async def test_deepseek_provider(self, monkeypatch: MonkeyPatch):
        """Test DeepSeek provider."""
        logger.info("Testing DeepSeek provider", emoji_key="test")

        # Mock the API client
        monkeypatch.setattr(
            "openai.AsyncOpenAI",
            lambda **kwargs: type(
                "MockClient",
                (),
                {
                    "chat": type(
                        "MockChat",
                        (),
                        {
                            "completions": type(
                                "MockCompletions",
                                (),
                                {
                                    "create": lambda **kwargs: type(
                                        "MockResponse",
                                        (),
                                        {
                                            "choices": [
                                                type(
                                                    "MockChoice",
                                                    (),
                                                    {
                                                        "message": type(
                                                            "MockMessage",
                                                            (),
                                                            {"content": "Mock DeepSeek response"},
                                                        ),
                                                        "finish_reason": "stop",
                                                    },
                                                )
                                            ],
                                            "usage": type(
                                                "MockUsage",
                                                (),
                                                {
                                                    "prompt_tokens": 15,
                                                    "completion_tokens": 8,
                                                    "total_tokens": 23,
                                                },
                                            ),
                                        },
                                    )
                                },
                            )
                        },
                    )
                },
            ),
        )

        provider = DeepSeekProvider(api_key="mock-deepseek-key")
        assert provider.provider_name == Provider.DEEPSEEK.value

        # Should return a default model
        model = provider.get_default_model()
        assert model is not None
        assert isinstance(model, str)

    async def test_gemini_provider(self, monkeypatch: MonkeyPatch):
        """Test Gemini provider."""
        logger.info("Testing Gemini provider", emoji_key="test")

        # Create mock response
        mock_response = type(
            "MockResponse",
            (),
            {
                "text": "Mock Gemini response",
                "candidates": [
                    type(
                        "MockCandidate",
                        (),
                        {"content": {"parts": [{"text": "Mock Gemini response"}]}},
                    )
                ],
            },
        )

        # Mock the Google Generative AI Client
        class MockClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.models = MockModels()

        class MockModels:
            def generate_content(self, **kwargs):
                return mock_response

            def list(self):
                return [{"name": "gemini-2.0-flash-lite"}, {"name": "gemini-2.0-pro"}]

        # Patch the genai Client
        monkeypatch.setattr("google.genai.Client", MockClient)

        # Create and test the provider
        provider = GeminiProvider(api_key="mock-gemini-key")
        assert provider.provider_name == Provider.GEMINI.value

        # Initialize with the mock client
        await provider.initialize()

        # Should return a default model
        model = provider.get_default_model()
        assert model is not None
        assert isinstance(model, str)

        # Test completion
        result = await provider.generate_completion(prompt="Test prompt", model="gemini-2.0-pro")

        # Check result
        assert result.text is not None
        assert "Gemini" in result.text  # Should contain "Mock Gemini response"
