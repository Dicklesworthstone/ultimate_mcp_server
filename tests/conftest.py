"""Pytest fixtures for Ultimate MCP Server tests."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pytest import MonkeyPatch

from ultimate_mcp_server.config import Config
from ultimate_mcp_server.core.providers.base import BaseProvider, ModelResponse
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.utils import get_logger

logger = get_logger("tests")


class MockResponse:
    """Mock response for testing."""

    def __init__(self, status_code: int = 200, json_data: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.json_data = json_data or {}

    async def json(self):
        return self.json_data

    async def text(self):
        return json.dumps(self.json_data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockClient:
    """Mock HTTP client for testing."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.responses = responses or {}
        self.requests = []

    async def post(self, url: str, json: Dict[str, Any], headers: Optional[Dict[str, str]] = None):
        self.requests.append({"url": url, "json": json, "headers": headers})
        return MockResponse(
            json_data=self.responses.get(
                url, {"choices": [{"message": {"content": "Mock response"}}]}
            )
        )

    async def get(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.requests.append({"url": url, "headers": headers})
        return MockResponse(json_data=self.responses.get(url, {"data": [{"id": "mock-model"}]}))


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    provider_name = "mock"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.responses = kwargs.pop("responses", {})
        self.initialized = False
        self.calls = []

    async def initialize(self) -> bool:
        self.initialized = True
        self.logger.success("Mock provider initialized successfully", emoji_key="provider")
        return True

    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> ModelResponse:
        self.calls.append(
            {
                "type": "completion",
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )

        model = model or self.get_default_model()

        return ModelResponse(
            text=self.responses.get("text", "Mock completion response"),
            model=model,
            provider=self.provider_name,
            input_tokens=100,
            output_tokens=50,
            processing_time=0.1,
            raw_response={"id": "mock-response-id"},
        )

    async def generate_completion_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        self.calls.append(
            {
                "type": "stream",
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )

        model = model or self.get_default_model()

        chunks = self.responses.get("chunks", ["Mock ", "streaming ", "response"])

        for i, chunk in enumerate(chunks):
            yield (
                chunk,
                {
                    "model": model,
                    "provider": self.provider_name,
                    "chunk_index": i + 1,
                    "finish_reason": "stop" if i == len(chunks) - 1 else None,
                },
            )

    async def list_models(self) -> List[Dict[str, Any]]:
        return self.responses.get(
            "models",
            [
                {
                    "id": "mock-model-1",
                    "provider": self.provider_name,
                    "description": "Mock model 1",
                },
                {
                    "id": "mock-model-2",
                    "provider": self.provider_name,
                    "description": "Mock model 2",
                },
            ],
        )

    def get_default_model(self) -> str:
        return "mock-model-1"

    async def check_api_key(self) -> bool:
        return True


@pytest.fixture
def test_dir() -> Path:
    """Get the tests directory path."""
    return Path(__file__).parent


@pytest.fixture
def sample_data_dir(test_dir: Path) -> Path:
    """Get the sample data directory path."""
    data_dir = test_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def mock_env_vars(monkeypatch: MonkeyPatch) -> None:
    """Set mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock-deepseek-key")
    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def test_config() -> Config:
    """Get a test configuration."""
    # Create a test configuration
    test_config = Config()

    # Override settings for testing
    test_config.cache.enabled = True
    test_config.cache.ttl = 60  # Short TTL for testing
    test_config.cache.max_entries = 100
    test_config.server.port = 8888  # Different port for testing

    # Set test API keys
    test_config.providers.openai.api_key = "test-openai-key"
    test_config.providers.anthropic.api_key = "test-anthropic-key"
    test_config.providers.gemini.api_key = "test-gemini-key"
    test_config.providers.deepseek.api_key = "test-deepseek-key"

    return test_config


@pytest.fixture
def mock_provider() -> MockProvider:
    """Get a mock provider."""
    return MockProvider(api_key="mock-api-key")


@pytest.fixture
def mock_gateway(mock_provider: MockProvider) -> Gateway:
    """Get a mock gateway with the mock provider."""
    gateway = Gateway(name="test-gateway")

    # Add mock provider
    gateway.providers["mock"] = mock_provider
    gateway.provider_status["mock"] = {
        "enabled": True,
        "available": True,
        "api_key_configured": True,
        "models": [
            {"id": "mock-model-1", "provider": "mock", "description": "Mock model 1"},
            {"id": "mock-model-2", "provider": "mock", "description": "Mock model 2"},
        ],
    }

    return gateway


@pytest.fixture
def mock_http_client(monkeypatch: MonkeyPatch) -> MockClient:
    """Mock HTTP client to avoid real API calls."""
    mock_client = MockClient()

    # We'll need to patch any HTTP clients used by the providers
    # This will be implemented as needed in specific tests

    return mock_client


@pytest.fixture
def sample_document() -> str:
    """Get a sample document for testing."""
    return """
    # Sample Document
    
    This is a sample document for testing purposes.
    
    ## Section 1
    
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Nullam auctor, nisl eget ultricies aliquam, est libero tincidunt nisi,
    eu aliquet nunc nisl eu nisl.
    
    ## Section 2
    
    Praesent euismod, nisl eget ultricies aliquam, est libero tincidunt nisi,
    eu aliquet nunc nisl eu nisl.
    
    ### Subsection 2.1
    
    - Item 1
    - Item 2
    - Item 3
    
    ### Subsection 2.2
    
    1. First item
    2. Second item
    3. Third item
    """


@pytest.fixture
def sample_json_data() -> Dict[str, Any]:
    """Get sample JSON data for testing."""
    return {
        "name": "Test User",
        "age": 30,
        "email": "test@example.com",
        "address": {"street": "123 Test St", "city": "Test City", "state": "TS", "zip": "12345"},
        "tags": ["test", "sample", "json"],
    }


@pytest.fixture(scope="session")
def event_loop_policy():
    """Return an event loop policy for the test session."""
    return asyncio.DefaultEventLoopPolicy()
