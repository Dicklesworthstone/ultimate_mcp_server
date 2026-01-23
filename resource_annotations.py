"""
Resource annotations for Model Control Protocol (MCP) systems.

This module implements the resource annotation system specified in the MCP protocol,
which enables AI systems to make intelligent decisions about how to process, prioritize,
and present different types of resources in multi-modal and multi-resource contexts.

Resource annotations serve multiple critical functions in AI/LLM systems:

1. PRIORITIZATION: Help AI systems allocate attention optimally among multiple resources
   when token constraints prevent processing everything (e.g., which document to focus on)

2. VISIBILITY CONTROL: Determine which resources should be visible to different actors
   in the system (e.g., assistant-only resources vs. user-facing resources)

3. FORMAT PRESERVATION: Indicate when resources have structured formats that should be
   maintained (e.g., code, tables, JSON) rather than freely interpreted

4. CHUNKING GUIDANCE: Provide hints about how to divide large resources efficiently
   for processing within context window constraints

The module provides:
- The ResourceAnnotations class for creating annotation metadata
- Pre-defined annotation templates for common resource types
- Utilities for working with annotated resources (e.g., chunking)

Usage example:
    ```python
    # Create custom annotations for a research paper
    paper_annotations = ResourceAnnotations(
        priority=0.8,
        audience=["assistant"],
        chunking_recommended=True,
        description="Research paper on quantum computing effects"
    )

    # Annotate and chunk a large document
    paper_content = open("quantum_paper.txt").read()
    chunks = format_chunked_content(paper_content, chunk_size=3000)

    # Use a predefined annotation template for code
    code_resource = {
        "content": "def calculate_entropy(data):\\n    ...",
        "annotations": CODE_RESOURCE.to_dict()
    }
    ```

These annotations integrate with the MCP protocol to help LLMs process resources
more intelligently and efficiently in complex, multi-resource scenarios.
"""

from typing import List, Optional


class ResourceAnnotations:
    """
    Annotations that guide LLMs in handling and prioritizing resources within the MCP protocol.

    ResourceAnnotations provide crucial metadata that helps LLMs make intelligent decisions about:
    - IMPORTANCE: How critical a resource is to the current task (via priority)
    - AUDIENCE: Who should see or interact with the resource
    - FORMATTING: How the resource should be rendered or processed
    - CHUNKING: Whether and how to divide large resources into manageable pieces

    These annotations serve multiple purposes in the MCP ecosystem:
    1. Help LLMs prioritize which resources to analyze first when multiple are available
    2. Control visibility of resources between assistants and users
    3. Preserve structural integrity of formatted content (code, tables, etc.)
    4. Provide chunking guidance for efficient processing of large resources

    When resources are annotated appropriately, LLMs can make better decisions about:
    - Which resources deserve the most attention in token-constrained contexts
    - When to preserve formatting vs. when content structure is less important
    - How to efficiently process large documents while maintaining context
    - Whether certain resources are meant for the assistant's understanding only

    Usage example:
        ```python
        # For a source code file that should preserve formatting
        code_annotations = ResourceAnnotations(
            priority=0.8,              # High importance
            audience=["assistant"],    # Only the assistant needs to see this
            structured_format=True,    # Preserve code formatting
            chunking_recommended=True, # Chunk if large
            max_recommended_chunk_size=2000,
            description="Python source code implementing the core algorithm"
        )

        # Apply annotations to a resource
        resource = {
            "id": "algorithm.py",
            "content": "def calculate(x, y):\n    return x + y",
            "annotations": code_annotations.to_dict()
        }
        ```
    """

    def __init__(
        self,
        priority: float = 0.5,
        audience: List[str] = None,
        structured_format: bool = False,
        chunking_recommended: bool = False,
        max_recommended_chunk_size: Optional[int] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize resource annotations.

        Args:
            priority: How important this resource is (0.0-1.0, higher is more important).
                0.0 = entirely optional, 1.0 = effectively required.
                Affects how much attention an LLM should give this resource when multiple
                resources are available but context limits prevent using all of them.
                Default: 0.5 (medium importance)

            audience: Who should see this resource, as a list of roles:
                - "assistant": The AI assistant should process this resource
                - "user": The human user should see this resource
                Both can be specified for resources relevant to both parties.
                Default: ["assistant"] (assistant-only)

            structured_format: Whether this resource has a structured format that
                should be preserved (e.g., code, JSON, tables). When True, the LLM should
                maintain the exact formatting, indentation, and structure of the content.
                Default: False

            chunking_recommended: Whether this resource should be chunked if large.
                Setting this to True signals that the content is suitable for being
                divided into smaller pieces for processing (e.g., long documents).
                Default: False

            max_recommended_chunk_size: Maximum recommended chunk size in characters.
                Provides guidance on how large each chunk should be if chunking is applied.
                Default: None (no specific recommendation)

            description: Optional description of the resource that provides context
                about its purpose, content, or importance.
        """
        self.priority = max(0.0, min(1.0, priority))  # Clamp between 0 and 1
        self.audience = audience or ["assistant"]
        self.structured_format = structured_format
        self.chunking_recommended = chunking_recommended
        self.max_recommended_chunk_size = max_recommended_chunk_size
        self.description = description

    def to_dict(self) -> dict:
        """Convert annotations to dictionary for MCP protocol."""
        result = {"priority": self.priority, "audience": self.audience}

        # Add extended properties
        if self.description:
            result["description"] = self.description

        # Add chunking metadata if recommended
        if self.chunking_recommended:
            result["chunking"] = {"recommended": True}
            if self.max_recommended_chunk_size:
                result["chunking"]["maxSize"] = self.max_recommended_chunk_size

        # Add format information
        if self.structured_format:
            result["format"] = {"structured": True}

        return result


# Pre-defined annotation templates for common resource types

# For critical resources that need immediate attention
# Use for resources essential to the current task's success
# Examples: Primary task instructions, critical context documents
HIGH_PRIORITY_RESOURCE = ResourceAnnotations(
    priority=0.9,
    audience=["assistant", "user"],
    description="Critical resource that should be prioritized",
)

# For source code and programming-related content
# Preserves indentation, formatting, and structure
# Recommends chunking for large codebases
# Examples: Source files, configuration files, scripts
CODE_RESOURCE = ResourceAnnotations(
    priority=0.8,
    audience=["assistant"],
    structured_format=True,
    chunking_recommended=True,
    max_recommended_chunk_size=2000,
    description="Source code that should preserve formatting",
)

# For lengthy text resources that should be divided into smaller parts
# Good for processing long documents without overwhelming context windows
# Examples: Articles, documentation, books, long explanations
LARGE_TEXT_RESOURCE = ResourceAnnotations(
    priority=0.6,
    audience=["assistant"],
    chunking_recommended=True,
    max_recommended_chunk_size=4000,
    description="Large text that should be chunked for processing",
)

# For data formats where structure is important
# Preserves formatting but doesn't automatically suggest chunking
# Examples: JSON data, database records, tabular data, XML
STRUCTURED_DATA_RESOURCE = ResourceAnnotations(
    priority=0.7,
    audience=["assistant"],
    structured_format=True,
    description="Structured data like JSON or tables",
)

# For supplementary information that provides additional context
# Low priority indicates it can be skipped if context is limited
# Examples: Background information, history, tangential details
OPTIONAL_RESOURCE = ResourceAnnotations(
    priority=0.2,
    audience=["assistant"],
    description="Supplementary information that isn't critical",
)

# For content meant to be shown to the user directly
# Not intended for assistant's processing (assistant not in audience)
# Examples: Final results, generated content, presentations
USER_FACING_RESOURCE = ResourceAnnotations(
    priority=0.7, audience=["user"], description="Resource meant for user consumption"
)


def format_chunked_content(content: str, chunk_size: int = 4000, overlap: int = 200) -> List[dict]:
    """
    Format content into overlapping chunks with rich metadata for efficient LLM processing.

    This utility function implements a sliding window approach to divide large content
    into manageable, context-aware chunks. Each chunk is annotated with detailed positioning
    metadata, allowing LLMs to understand the chunk's relationship to the overall content
    and maintain coherence across chunk boundaries.

    Key features:
    - Consistent overlap between chunks preserves context and prevents information loss
    - Automatic metadata generation provides LLMs with crucial positioning information
    - Standard annotation format compatible with the MCP resource protocol
    - Configurable chunk size to adapt to different model context window limitations

    The overlap between chunks is particularly important as it helps LLMs maintain
    coherence when processing information that spans chunk boundaries. Without overlap,
    context might be lost at chunk transitions, leading to degraded performance on tasks
    that require understanding the full content.

    Args:
        content: The source text content to be chunked. This can be any string content
            like a document, article, code file, or other text-based resource.
        chunk_size: Maximum size of each chunk in characters (default: 4000).
            This should be set based on the target LLM's context window limitations,
            typically 25-50% less than the model's maximum to allow room for prompts.
        overlap: Number of characters to overlap between consecutive chunks (default: 200).
            Larger overlap values provide more context continuity between chunks but
            increase redundancy and total token usage.

    Returns:
        List of dictionaries, each representing a content chunk with metadata:
        - "text": The actual chunk content (substring of the original content)
        - "annotations": Metadata dictionary containing:
          - priority: Importance hint for the LLM (default: 0.7)
          - audience: Who should see this chunk (default: ["assistant"])
          - chunk_info: Detailed positioning metadata including:
            - index: Zero-based index of this chunk in the sequence
            - total_chunks: Total number of chunks in the complete content
            - start_position: Character offset where this chunk begins in the original content
            - end_position: Character offset where this chunk ends in the original content
            - has_more: Boolean indicating if more chunks follow this one

    Usage examples:
        # Basic usage with default parameters
        chunks = format_chunked_content("Long document text...")

        # Using smaller chunks for models with limited context windows
        small_chunks = format_chunked_content(
            content="Large article text...",
            chunk_size=1000,
            overlap=100
        )

        # Process chunks sequentially while maintaining context
        for chunk in chunks:
            response = await generate_completion(
                prompt=f"Analyze this text: {chunk['text']}",
                # Include chunk metadata so the LLM understands context
                additional_context=f"This is chunk {chunk['annotations']['chunk_info']['index']+1} "
                                  f"of {chunk['annotations']['chunk_info']['total_chunks']}"
            )
    """
    chunks = []

    # Create chunks with overlap
    for i in range(0, len(content), chunk_size - overlap):
        chunk_text = content[i : i + chunk_size]
        if chunk_text:
            # Create chunk with annotations
            chunk = {
                "text": chunk_text,
                "annotations": {
                    "priority": 0.7,
                    "audience": ["assistant"],
                    "chunk_info": {
                        "index": len(chunks),
                        "total_chunks": (len(content) + chunk_size - 1) // (chunk_size - overlap),
                        "start_position": i,
                        "end_position": min(i + chunk_size, len(content)),
                        "has_more": i + chunk_size < len(content),
                    },
                },
            }
            chunks.append(chunk)

    return chunks
