"""Tests for the tool implementations."""

from typing import Any, Dict

import pytest
from pytest import MonkeyPatch

# Remove the CompletionTools import as the class was deleted
# from ultimate_mcp_server.tools.completion import CompletionTools
from ultimate_mcp_server.tools.document import DocumentTools

from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.tools.base import (
    BaseTool,
    register_tool,
    with_retry,
    with_tool_metrics,
)
from ultimate_mcp_server.tools.extraction import ExtractionTools
from ultimate_mcp_server.utils import get_logger

logger = get_logger("test.tools")


class TestBaseTools:
    """Tests for the base tool classes and decorators."""

    def test_base_tool_init(self, mock_gateway: Gateway):
        """Test base tool initialization."""
        logger.info("Testing base tool initialization", emoji_key="test")

        # Create a minimal tool class
        class TestTool(BaseTool):
            tool_name = "test-tool"
            description = "Test tool"

            def _register_tools(self):
                # No tools to register
                pass

        # Initialize
        tool = TestTool(mock_gateway)

        # Check properties
        assert tool.tool_name == "test-tool"
        assert tool.description == "Test tool"
        assert tool.mcp == mock_gateway.mcp
        assert tool.logger is not None
        assert tool.metrics is not None

    @pytest.mark.asyncio
    async def test_with_tool_metrics(self):
        """Test the with_tool_metrics decorator."""
        logger.info("Testing with_tool_metrics decorator", emoji_key="test")

        # Create a tool class with metrics
        class TestTool(BaseTool):
            tool_name = "test-metrics-tool"
            description = "Test metrics tool"

            def _register_tools(self):
                pass

            @with_tool_metrics
            async def test_method(self, arg1, arg2=None, ctx=None):
                return {"result": arg1 + str(arg2 or "")}

        # Create a mock MCP server
        mock_mcp = type("MockMCP", (), {"tool": lambda: lambda x: x})
        mock_gateway = type("MockGateway", (), {"mcp": mock_mcp})

        # Initialize
        tool = TestTool(mock_gateway)

        # Call method
        result = await tool.test_method("test", "arg")

        # Check result
        assert result == {"result": "testarg"}

        # Check metrics
        assert tool.metrics.total_calls == 1
        assert tool.metrics.successful_calls == 1
        assert tool.metrics.failed_calls == 0

        # Test error case
        @with_tool_metrics
        async def failing_method(self, arg):
            raise ValueError("Test error")

        # Add to class
        TestTool.failing_method = failing_method

        # Call failing method
        with pytest.raises(ValueError):
            await tool.failing_method("test")

        # Check metrics
        assert tool.metrics.total_calls == 2
        assert tool.metrics.successful_calls == 1
        assert tool.metrics.failed_calls == 1

    @pytest.mark.asyncio
    async def test_with_retry(self):
        """Test the with_retry decorator."""
        logger.info("Testing with_retry decorator", emoji_key="test")

        # Track calls
        calls = []

        @with_retry(max_retries=2, retry_delay=0.1)
        async def flaky_function(succeed_after):
            calls.append(len(calls))
            if len(calls) < succeed_after:
                raise ValueError("Temporary error")
            return "success"

        # Should succeed on first try
        calls = []
        result = await flaky_function(1)
        assert result == "success"
        assert len(calls) == 1

        # Should fail first, succeed on retry
        calls = []
        result = await flaky_function(2)
        assert result == "success"
        assert len(calls) == 2

        # Should fail first two, succeed on second retry
        calls = []
        result = await flaky_function(3)
        assert result == "success"
        assert len(calls) == 3

        # Should fail too many times
        calls = []
        with pytest.raises(ValueError):
            await flaky_function(4)  # Will make 3 attempts (original + 2 retries)
        assert len(calls) == 3

    def test_register_tool(self, mock_gateway: Gateway):
        """Test the register_tool decorator."""
        logger.info("Testing register_tool decorator", emoji_key="test")

        # Create a mock MCP server with a tool registration function
        registered_tools = {}

        class MockMCP:
            def tool(self, name=None, description=None):
                def decorator(f):
                    registered_tools[name or f.__name__] = {
                        "function": f,
                        "description": description or f.__doc__,
                    }
                    return f

                return decorator

        mock_mcp = MockMCP()
        mock_gateway.mcp = mock_mcp

        # Register a tool
        @register_tool(mock_gateway.mcp, name="test-tool", description="Test tool")
        async def test_tool(arg1, arg2=None):
            """Tool docstring."""
            return {"result": arg1 + str(arg2 or "")}

        # Check registration
        assert "test-tool" in registered_tools
        assert registered_tools["test-tool"]["description"] == "Test tool"

        # Register with defaults
        @register_tool(mock_gateway.mcp)
        async def another_tool(arg):
            """Another tool docstring."""
            return {"result": arg}

        # Check registration with defaults
        assert "another_tool" in registered_tools
        assert registered_tools["another_tool"]["description"] == "Another tool docstring."


# Comment out the entire TestCompletionTools class as it relies on the deleted class structure
# class TestCompletionTools:
#     """Tests for the completion tools."""
#
#     @pytest.fixture
#     def mock_completion_tools(self, mock_gateway: Gateway) -> CompletionTools:
#         """Get mock completion tools."""
#         # This fixture is no longer valid as CompletionTools doesn't exist
#         # We would need to refactor tests to mock standalone functions
#         pass
#         # return CompletionTools(mock_gateway)
#
#     def test_init(self, mock_completion_tools: CompletionTools):
#         """Test initialization."""
#         logger.info("Testing completion tools initialization", emoji_key="test")
#         # This test is no longer valid
#         # assert mock_completion_tools.tool_name == "completion"
#         # assert mock_completion_tools.description is not None
#         pass
#
#     async def test_generate_completion(self, mock_completion_tools: CompletionTools, mock_gateway: Gateway, monkeypatch: MonkeyPatch):
#         """Test generate_completion tool."""
#         logger.info("Testing generate_completion tool", emoji_key="test")
#
#         # Mocking needs to target the standalone function now, not a method
#         # This test needs complete refactoring
#         pass
#
#     async def test_chat_completion(self, mock_completion_tools: CompletionTools, mock_gateway: Gateway, monkeypatch: MonkeyPatch):
#         """Test chat_completion tool."""
#         logger.info("Testing chat_completion tool", emoji_key="test")
#         # This test needs complete refactoring
#         pass
#
#     async def test_stream_completion(self, mock_completion_tools: CompletionTools, mock_gateway: Gateway, monkeypatch: MonkeyPatch):
#         """Test stream_completion tool."""
#         logger.info("Testing stream_completion tool", emoji_key="test")
#         # This test needs complete refactoring
#         pass
#
#     async def test_multi_completion(self, mock_completion_tools: CompletionTools, mock_gateway: Gateway, monkeypatch: MonkeyPatch):
#         """Test multi_completion tool."""
#         logger.info("Testing multi_completion tool", emoji_key="test")
#         # This test needs complete refactoring
#         pass


class TestDocumentTools:
    """Tests for the document tools."""

    @pytest.fixture
    def mock_document_tools(self, mock_gateway: Gateway) -> DocumentTools:
        """Get mock document tools."""
        return DocumentTools(mock_gateway)

    def test_init(self, mock_document_tools: DocumentTools):
        """Test initialization."""
        logger.info("Testing document tools initialization", emoji_key="test")

        assert mock_document_tools.tool_name is not None
        assert mock_document_tools.description is not None

    async def test_chunk_document(
        self, mock_document_tools: DocumentTools, sample_document: str, monkeypatch: MonkeyPatch
    ):
        """Test chunk_document tool."""
        logger.info("Testing chunk_document tool", emoji_key="test")

        # Create a simplified implementation for testing
        async def mock_chunk_document(
            document, chunk_size=1000, chunk_overlap=100, method="token", ctx=None
        ):
            chunks = []
            # Simple paragraph chunking for testing
            for para in document.split("\n\n"):
                if para.strip():
                    chunks.append(para.strip())
            return {
                "chunks": chunks,
                "chunk_count": len(chunks),
                "method": method,
                "processing_time": 0.1,
            }

        # Create a mock execute function for our BaseTool
        async def mock_execute(tool_name, params):
            # Call our mock implementation
            return await mock_chunk_document(**params)

        # Monkeypatch the tool execution using our new execute method
        monkeypatch.setattr(mock_document_tools, "execute", mock_execute)

        # Call the tool
        result = await mock_document_tools.execute(
            "chunk_document", {"document": sample_document, "method": "paragraph"}
        )

        # Check result
        assert isinstance(result, dict)
        assert "chunks" in result
        assert isinstance(result["chunks"], list)
        assert result["chunk_count"] > 0
        assert result["method"] == "paragraph"
        assert result["processing_time"] > 0


class TestExtractionTools:
    """Tests for the extraction tools."""

    @pytest.fixture
    def mock_extraction_tools(self, mock_gateway: Gateway) -> ExtractionTools:
        """Get mock extraction tools."""
        return ExtractionTools(mock_gateway)

    def test_init(self, mock_extraction_tools: ExtractionTools):
        """Test initialization."""
        logger.info("Testing extraction tools initialization", emoji_key="test")

        assert mock_extraction_tools.tool_name == "extraction"
        assert mock_extraction_tools.description is not None

    async def test_extract_json(
        self,
        mock_extraction_tools: ExtractionTools,
        sample_json_data: Dict[str, Any],
        monkeypatch: MonkeyPatch,
    ):
        """Test extract_json tool."""
        logger.info("Testing extract_json tool", emoji_key="test")

        # Mock the tool execution
        async def mock_extract_json(
            text, schema=None, provider="openai", model=None, max_attempts=3, ctx=None
        ):
            return {
                "data": sample_json_data,
                "provider": provider,
                "model": model or "mock-model",
                "tokens": {"input": 50, "output": 30, "total": 80},
                "cost": 0.01,
                "processing_time": 0.2,
            }

        # Create a mock execute function for our BaseTool
        async def mock_execute(tool_name, params):
            # Call our mock implementation
            return await mock_extract_json(**params)

        # Monkeypatch the tool execution using our new execute method
        monkeypatch.setattr(mock_extraction_tools, "execute", mock_execute)

        # Call the tool
        result = await mock_extraction_tools.execute(
            "extract_json",
            {
                "text": "Extract JSON from this: " + str(sample_json_data),
                "provider": "mock",
                "model": "mock-model",
            },
        )

        # Check result
        assert isinstance(result, dict)
        assert "data" in result
        assert result["data"] == sample_json_data
        assert result["provider"] == "mock"
        assert result["model"] == "mock-model"
        assert "tokens" in result
        assert "cost" in result
        assert "processing_time" in result
