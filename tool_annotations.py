"""
Tool annotations for MCP servers.

This module provides a standardized way to annotate tools with hints that help LLMs
understand when and how to use them effectively.
"""

from typing import List, Optional


class ToolAnnotations:
    """
    Tool annotations providing hints to LLMs about tool behavior and usage patterns.

    ToolAnnotations supply metadata that helps LLMs make informed decisions about:
    - WHEN to use a particular tool (appropriate contexts and priority)
    - HOW to use the tool correctly (through examples and behavior hints)
    - WHAT the potential consequences of using the tool might be (read-only vs. destructive)
    - WHO should use the tool (assistant, user, or both via audience hints)

    These annotations serve as a bridge between tool developers and LLMs, providing
    crucial context beyond just function signatures and descriptions. For example, the
    annotations can indicate that a file deletion tool is destructive and should be used
    with caution, or that a search tool is safe to retry multiple times.

    The system supports four key behavioral hints:
    - read_only_hint: Tool doesn't modify state (safe for exploratory use)
    - destructive_hint: Tool may perform irreversible changes (use with caution)
    - idempotent_hint: Repeated calls with same arguments produce same results
    - open_world_hint: Tool interacts with external systems beyond the LLM's knowledge

    Additional metadata includes:
    - audience: Who can/should use this tool
    - priority: How important/commonly used this tool is
    - title: Human-readable name for the tool
    - examples: Sample inputs and expected outputs

    Usage example:
        ```python
        # For a document deletion tool
        delete_doc_annotations = ToolAnnotations(
            read_only_hint=False,       # Modifies state
            destructive_hint=True,      # Deletion is destructive
            idempotent_hint=True,       # Deleting twice has same effect as once
            open_world_hint=True,       # Changes external file system
            audience=["assistant"],     # Only assistant should use it
            priority=0.3,               # Lower priority (use cautiously)
            title="Delete Document",
            examples=[{
                "input": {"document_id": "doc-123"},
                "output": {"success": True, "message": "Document deleted"}
            }]
        )
        ```

    Note: All hints are advisory only - they don't enforce behavior but help LLMs
    make better decisions about tool usage.
    """

    def __init__(
        self,
        read_only_hint: bool = False,
        destructive_hint: bool = True,
        idempotent_hint: bool = False,
        open_world_hint: bool = True,
        audience: List[str] = None,
        priority: float = 0.5,
        title: Optional[str] = None,
        examples: List[dict] = None,
    ):
        """
        Initialize tool annotations.

        Args:
            read_only_hint: If True, indicates this tool does not modify its environment.
                Tools with read_only_hint=True are safe to call for exploration without
                side effects. Examples: search tools, data retrieval, information queries.
                Default: False

            destructive_hint: If True, the tool may perform destructive updates that
                can't easily be reversed or undone. Only meaningful when read_only_hint
                is False. Examples: deletion operations, irreversible state changes, payments.
                Default: True

            idempotent_hint: If True, calling the tool repeatedly with the same arguments
                will have no additional effect beyond the first call. Useful for retry logic.
                Only meaningful when read_only_hint is False. Examples: setting a value,
                deleting an item (calling it twice doesn't delete it twice).
                Default: False

            open_world_hint: If True, this tool may interact with systems or information
                outside the LLM's knowledge context (external APIs, file systems, etc.).
                If False, the tool operates in a closed domain the LLM can fully model.
                Default: True

            audience: Who is the intended user of this tool, as a list of roles:
                - "assistant": The AI assistant can use this tool
                - "user": The human user can use this tool
                Default: ["assistant"]

            priority: How important this tool is, from 0.0 (lowest) to 1.0 (highest).
                Higher priority tools should be considered first when multiple tools
                might accomplish a similar task. Default: 0.5 (medium priority)

            title: Human-readable title for the tool. If not provided, the tool's
                function name is typically used instead.

            examples: List of usage examples, each containing 'input' and 'output' keys.
                These help the LLM understand expected patterns of use and responses.
        """
        self.read_only_hint = read_only_hint
        self.destructive_hint = destructive_hint
        self.idempotent_hint = idempotent_hint
        self.open_world_hint = open_world_hint
        self.audience = audience or ["assistant"]
        self.priority = max(0.0, min(1.0, priority))  # Clamp between 0 and 1
        self.title = title
        self.examples = examples or []

    def to_dict(self) -> dict:
        """Convert annotations to dictionary for MCP protocol."""
        return {
            "readOnlyHint": self.read_only_hint,
            "destructiveHint": self.destructive_hint,
            "idempotentHint": self.idempotent_hint,
            "openWorldHint": self.open_world_hint,
            "title": self.title,
            "audience": self.audience,
            "priority": self.priority,
            "examples": self.examples,
        }


# Pre-defined annotation templates for common tool types

# A tool that only reads/queries data without modifying any state
READONLY_TOOL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
    priority=0.8,
    title="Read-Only Tool",
)

# A tool that queries external systems or APIs for information
QUERY_TOOL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
    priority=0.7,
    title="Query Tool",
)

# A tool that performs potentially irreversible changes to state
# The LLM should use these with caution, especially without confirmation
DESTRUCTIVE_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
    priority=0.3,
    title="Destructive Tool",
)

# A tool that modifies state but can be safely called multiple times
# with the same arguments (e.g., setting a value, creating if not exists)
IDEMPOTENT_UPDATE_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
    priority=0.5,
    title="Idempotent Update Tool",
)
