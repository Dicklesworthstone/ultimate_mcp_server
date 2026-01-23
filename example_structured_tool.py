"""
Example of a well-structured MCP tool with best practices.

This module demonstrates how to create a comprehensive MCP tool
that implements all the best practices for LLM usability:
- Tool annotations for better decision-making
- Standardized error handling
- Input validation
- Detailed documentation with examples
- Structured outputs with consistent formats
"""

import time
import uuid
from typing import Any, Dict, Optional

# ---------------------------------
from error_handling import non_empty_string, validate_inputs, with_error_handling
from tool_annotations import ToolAnnotations

# --- Import RAG tools/services ---
# Assuming direct function import for simplicity in example
# In a real structured app, might use dependency injection or service locators
from ultimate_mcp_server.tools.rag import (
    add_documents,
    create_knowledge_base,
    delete_knowledge_base,
    retrieve_context,
)

# --- Define KB Name for Demo ---
DEMO_KB_NAME = f"example_tool_kb_{uuid.uuid4().hex[:8]}"
# ------------------------------

# --- Sample Data (moved to top) ---
# This data will now be *added* to the KB during setup
SAMPLE_DOCUMENTS = [
    {
        "id": "kb-001",
        "title": "Introduction to Climate Change",
        "text": "An overview of climate change causes and effects.",
        "type": "article",
        "level": "beginner",
        "date": "2023-01-15",
        "score_for_ranking": 0.95,  # Keep score for potential sorting demonstration?
    },
    {
        "id": "kb-002",
        "title": "Machine Learning Fundamentals",
        "text": "Learn the basics of machine learning algorithms.",
        "type": "tutorial",
        "level": "beginner",
        "date": "2023-02-20",
        "score_for_ranking": 0.92,
    },
    {
        "id": "kb-003",
        "title": "Advanced Neural Networks",
        "text": "Deep dive into neural network architectures.",
        "type": "tutorial",
        "level": "advanced",
        "date": "2023-03-10",
        "score_for_ranking": 0.88,
    },
    {
        "id": "kb-004",
        "title": "Climate Policy FAQ",
        "text": "Frequently asked questions about climate policies.",
        "type": "faq",
        "level": "intermediate",
        "date": "2023-04-05",
        "score_for_ranking": 0.82,
    },
    {
        "id": "kb-005",
        "title": "Python Reference for Data Science",
        "text": "Reference guide for Python in data science applications.",
        "type": "reference",
        "level": "intermediate",
        "date": "2023-05-12",
        "score_for_ranking": 0.78,
    },
]
# -------------------------------------


class ExampleTool:
    """
    Example implementation of a well-structured MCP tool with best practices.

    The ExampleTool class serves as a reference implementation that demonstrates how to properly
    design and implement tools for the Model Control Protocol (MCP) ecosystem. It showcases
    a real-world RAG (Retrieval-Augmented Generation) tool that interacts with a knowledge base.

    Key design features:
    - Proper tool registration with the MCP server
    - Comprehensive schema definitions for inputs and outputs
    - Clear tool descriptions with usage guidance for LLMs
    - Tool annotations that provide semantic hints about tool behavior
    - Consistent error handling and input validation
    - Well-structured implementation with clean separation of concerns

    The class implements a search_knowledge_base tool that allows querying a vector store
    containing sample documents. The implementation demonstrates how to:
    - Process input parameters and apply validation
    - Interact with external services (the knowledge base)
    - Format response data in a consistent structure
    - Handle errors gracefully with meaningful error messages
    - Add appropriate metadata to help LLMs use the tool effectively

    This implementation is intended as an educational reference for developers creating
    their own MCP tools, showing patterns and practices that lead to tools that are
    easily discoverable, usable, and maintainable.

    Usage:
        ```python
        # Initialize the MCP server
        server = MCPServer()

        # Create an instance (automatically registers all tools)
        tool = ExampleTool(server)

        # The tool is now available for use through the server
        # After ensuring the knowledge base is set up
        await setup_demo_kb()
        ```
    """

    def __init__(self, mcp_server):
        """
        Initialize an ExampleTool instance and register its tools with the MCP server.

        This constructor creates a new instance of the ExampleTool class and automatically
        registers all tools implemented by this class with the provided MCP server. It
        serves as the entry point for integrating the example tools into an MCP server.

        The initialization process:
        1. Stores a reference to the provided MCP server instance
        2. Calls the _register_tools method to define and register all tools
        3. Establishes all necessary connections to the underlying knowledge base

        After initialization, the tools become available for use through the MCP server's
        tool invocation interface. No further setup is required for the tools themselves,
        although the underlying knowledge base (see setup_demo_kb) must be initialized
        before the tools can be used effectively.

        Args:
            mcp_server: An instance of the MCP server to register tools with. This must be
                       a fully initialized server object with a functional tool registration
                       system available through its 'tool' decorator.

        Returns:
            None

        Notes:
            - Tool registration happens immediately during initialization
            - Tool usage requires the demo knowledge base to be set up separately
            - The server instance is stored but not modified beyond tool registration
        """
        self.mcp = mcp_server
        self._register_tools()

    def _register_tools(self):
        """
        Register all tools provided by this class with the MCP server.

        This private method is called during initialization and handles the registration
        of all tools implemented by the ExampleTool class. It defines and registers
        individual tools with appropriate metadata, schemas, and implementations.

        For each tool, the method:
        1. Creates tool annotations with appropriate behavioral hints
        2. Defines the tool's description, input schema, and output schema
        3. Implements the tool function with error handling and input validation
        4. Registers the complete tool with the MCP server

        The primary tool defined here is:
        - search_knowledge_base: Searches the demo knowledge base for relevant documents
          based on user queries and optional filters

        Each tool is decorated with:
        - @self.mcp.tool: Registers the function as an MCP tool
        - @with_error_handling: Provides standardized exception handling
        - @validate_inputs: Validates required parameters before execution

        The detailed tool definitions include human-readable descriptions, parameter
        schemas with comprehensive type information, and examples demonstrating proper
        tool usage for LLMs.

        Returns:
            None - Tools are registered as a side effect
        """

        # Create tool annotations with appropriate hints
        search_annotations = ToolAnnotations(
            read_only_hint=True,  # This tool doesn't modify anything
            destructive_hint=False,  # No destructive operations
            idempotent_hint=True,  # Can be called repeatedly with same results
            open_world_hint=True,  # Interacts with external data sources
            audience=["assistant"],  # Intended for the LLM to use
            priority=0.8,  # High priority tool
            title="Search Knowledge Base",  # Human-readable title
            examples=[
                {
                    "name": "Basic search",
                    "description": "Search for information about a topic",
                    "input": {"query": "climate change", "filters": {"type": "article"}},
                    "output": {
                        "results": [
                            {"title": "Climate Change Basics", "score": 0.92},
                            {"title": "Effects of Global Warming", "score": 0.87},
                        ],
                        "total_matches": 2,
                        "search_time_ms": 105,
                    },
                },
                {
                    "name": "Advanced search",
                    "description": "Search with multiple filters and limits",
                    "input": {
                        "query": "machine learning",
                        "filters": {"type": "tutorial", "level": "beginner"},
                        "limit": 1,
                    },
                    "output": {
                        "results": [{"title": "Introduction to Machine Learning", "score": 0.95}],
                        "total_matches": 1,
                        "search_time_ms": 87,
                    },
                },
            ],
        )

        @self.mcp.tool(
            name="search_knowledge_base",
            description=(
                "Search for information in the knowledge base using keywords and filters.\n\n"
                "This tool is ideal for finding relevant information on specific topics. "
                "It supports filtering by content type, date ranges, and other metadata. "
                "The tool returns a list of matching results sorted by relevance score.\n\n"
                "WHEN TO USE:\n"
                "- When you need to find specific information on a topic\n"
                "- When you want to discover relevant articles or documentation\n"
                "- Before generating content to ensure accuracy\n\n"
                "WHEN NOT TO USE:\n"
                "- When you need to modify or create content (use content_* tools instead)\n"
                "- When you need very recent information that might not be in the knowledge base\n"
                "- When you need exact answers to questions (use qa_* tools instead)"
            ),
            annotations=search_annotations.to_dict(),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (required)"},
                    "filters": {
                        "type": "object",
                        "description": "Optional filters to narrow results",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["article", "tutorial", "reference", "faq"],
                                "description": "Content type filter",
                            },
                            "level": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                                "description": "Difficulty level filter",
                            },
                            "date_after": {
                                "type": "string",
                                "format": "date",
                                "description": "Only include content after this date (YYYY-MM-DD)",
                            },
                        },
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Maximum number of results to return (1-20, default 5)",
                    },
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "type": {"type": "string"},
                                "date": {"type": "string", "format": "date"},
                                "score": {"type": "number"},
                            },
                        },
                    },
                    "total_matches": {"type": "integer"},
                    "search_time_ms": {"type": "integer"},
                },
            },
        )
        @with_error_handling
        @validate_inputs(query=non_empty_string)
        async def search_knowledge_base(
            query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 5, ctx=None
        ) -> Dict[str, Any]:
            """
            Search for information in the knowledge base using keywords and filters.

            This tool is ideal for finding relevant information on specific topics.
            It supports filtering by content type, date ranges, and other metadata.

            Args:
                query: Search query string (required)
                filters: Optional filters to narrow results
                    - type: Content type filter (article, tutorial, reference, faq)
                    - level: Difficulty level filter (beginner, intermediate, advanced)
                    - date_after: Only include content after this date (YYYY-MM-DD)
                limit: Maximum number of results to return (1-20, default 5)
                ctx: Context object passed by the MCP server

            Returns:
                Dictionary containing:
                - results: List of retrieved document chunks with metadata and scores.
                - count: Number of results returned (respecting limit).
                - retrieval_time: Time taken for retrieval in seconds.

            Examples:
                Basic search:
                  search_knowledge_base(query="climate change")

                Filtered search:
                  search_knowledge_base(
                    query="machine learning",
                    filters={"type": "tutorial", "level": "beginner"},
                    limit=3
                  )
            """
            # Start timing
            start_time = time.time()

            # Convert simple filters to ChromaDB compatible format if needed
            # The retrieve_context tool might already handle this, depending on its implementation.
            # For simplicity, we pass the filters dict directly.
            metadata_filter = filters  # Pass filters directly

            # Ensure limit is positive
            limit = max(1, limit)

            try:
                # Call the actual retrieve_context tool
                # Ensure DEMO_KB_NAME is defined appropriately
                retrieval_result = await retrieve_context(
                    knowledge_base_name=DEMO_KB_NAME,
                    query=query,
                    top_k=limit,
                    metadata_filter=metadata_filter,
                    # Add other relevant params like min_score if needed
                )

                # Return formatted results
                # The retrieve_context tool already returns a dict with 'success', 'results', etc.
                # We can return it directly or reformat if needed.
                if retrieval_result.get("success"):
                    return {
                        "results": retrieval_result.get("results", []),
                        "count": len(retrieval_result.get("results", [])),
                        "retrieval_time": retrieval_result.get(
                            "retrieval_time", time.time() - start_time
                        ),
                    }
                else:
                    # Propagate the error from retrieve_context
                    return {
                        "error": retrieval_result.get("message", "Retrieval failed"),
                        "results": [],
                        "count": 0,
                        "retrieval_time": time.time() - start_time,
                    }

            except Exception as e:
                # Log the error (in a real implementation)
                print(f"Search error: {str(e)}")

                # Return error response
                return {"error": f"Search failed: {str(e)}"}


# --- Added Setup/Teardown for Demo KB ---
async def setup_demo_kb():
    """
    Creates and populates the demo knowledge base with sample documents.

    This function handles the initialization of the demo knowledge base used by
    the example tools. It performs the following operations in sequence:
    1. Creates a new knowledge base with the name defined in DEMO_KB_NAME
    2. Extracts documents, metadata, and IDs from the SAMPLE_DOCUMENTS constant
    3. Adds the extracted information to the newly created knowledge base

    The knowledge base is created with overwrite=True, which means any existing
    knowledge base with the same name will be deleted and recreated. This ensures
    a clean starting state for the demo.

    Each document in the sample data is structured with:
    - id: Unique identifier for the document
    - title: Document title
    - text: The actual document content to be vectorized
    - type: Document category (article, tutorial, reference, faq)
    - level: Difficulty level (beginner, intermediate, advanced)
    - date: Publication date in YYYY-MM-DD format
    - score_for_ranking: A number between 0-1 used for demonstration purposes

    The function logs its progress to stdout and raises any exceptions it encounters,
    allowing the caller to handle failures appropriately.

    Returns:
        None

    Raises:
        Exception: If any step in the setup process fails. The original exception is
                 preserved and propagated with context information.

    Usage:
        await setup_demo_kb()  # Must be called in an async context
    """
    print(f"Setting up demo knowledge base: {DEMO_KB_NAME}...")
    try:
        await create_knowledge_base(name=DEMO_KB_NAME, overwrite=True)
        texts_to_add = [doc["text"] for doc in SAMPLE_DOCUMENTS]
        metadatas_to_add = [
            {k: v for k, v in doc.items() if k != "text"} for doc in SAMPLE_DOCUMENTS
        ]
        ids_to_add = [doc["id"] for doc in SAMPLE_DOCUMENTS]
        await add_documents(
            knowledge_base_name=DEMO_KB_NAME,
            documents=texts_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add,
        )
        print("Demo knowledge base setup complete.")
    except Exception as e:
        print(f"Error setting up demo KB: {e}")
        raise


async def teardown_demo_kb():
    """
    Deletes the demo knowledge base and cleans up associated resources.

    This function is responsible for properly disposing of the demo knowledge base
    after the examples have been run. It ensures that temporary resources created
    for demonstration purposes don't persist unnecessarily. Specifically, it:

    1. Attempts to delete the knowledge base identified by DEMO_KB_NAME
    2. Logs the success or failure of the operation to stdout
    3. Suppresses any exceptions to prevent cleanup errors from propagating

    Unlike setup_demo_kb(), this function does not raise exceptions for failures,
    as cleanup errors should not prevent the application from continuing or shutting
    down normally. Instead, errors are logged but suppressed.

    The function can be safely called multiple times or even if the knowledge base
    doesn't exist (the underlying delete_knowledge_base function should handle such cases).

    This function should be called during application shutdown or after example
    tools are no longer needed, typically in one of these contexts:
    - Server shutdown hooks/lifecycle events
    - After example demonstration is complete
    - During application cleanup phases

    Returns:
        None

    Usage:
        await teardown_demo_kb()  # Must be called in an async context

    Note:
        In production systems, more robust cleanup might involve tracking created
        resources and ensuring proper disposal even after unexpected termination.
    """
    print(f"Cleaning up demo knowledge base: {DEMO_KB_NAME}...")
    try:
        await delete_knowledge_base(name=DEMO_KB_NAME)
        print("Demo knowledge base cleaned up.")
    except Exception as e:
        print(f"Error cleaning up demo KB: {e}")


# -----------------------------------------


def register_example_tools(mcp_server):
    """
    Register all example tools with the MCP server and set up required resources.

    This function serves as the main entry point for integrating the example tools
    into an MCP server instance. It instantiates the ExampleTool class, which registers
    all individual tools with the provided server. Additionally, it handles concerns
    related to the setup and teardown of resources required by the example tools.

    Key responsibilities:
    1. Creates an instance of ExampleTool, which registers all example tools with the server
    2. Manages the initialization of required resources (demo knowledge base)
    3. Documents integration concerns and known limitations

    Integration notes:
    - The demo knowledge base (DEMO_KB_NAME) must be set up before tools are used
    - In a production environment, the async setup should be handled as part of the
      server lifecycle (e.g., using lifespan or startup events) rather than directly here
    - Current implementation leaves knowledge base setup as a separate step due to
      challenges with mixing sync/async code in the registration process

    Args:
        mcp_server: An instance of the MCP server to register tools with. This should be
                   a fully initialized server object with a working tool registration system.

    Returns:
        None

    Usage:
        ```python
        # During server initialization:
        server = MCPServer()
        register_example_tools(server)

        # Remember to set up the knowledge base separately (due to async requirements):
        await setup_demo_kb()  # Before using the tools

        # And clean up when done:
        await teardown_demo_kb()  # After tools are no longer needed
        ```

    Known limitations:
    - Cannot perform async setup directly in this function due to sync/async boundary issues
    - Knowledge base setup must be handled separately as an async operation
    - Resource cleanup must also be manually triggered as an async operation
    """
    # Perform setup when tools are registered
    # Note: In a real server, setup/teardown might be handled differently (e.g., lifespan)
    # Running async setup directly here might block if called synchronously.
    # A better approach might be to trigger setup after server start.
    # For this example modification, we assume it can be awaited here or handled externally.
    # asyncio.run(setup_demo_kb()) # This would block if register_example_tools is sync
    # TODO: Need a way to run async setup/teardown non-blockingly or during server lifespan.
    # Skipping async setup call here due to potential blocking issues.
    # KB needs to be set up *before* the tool is called in a demo.

    ExampleTool(mcp_server)
