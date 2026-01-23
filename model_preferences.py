"""
Model preferences for MCP servers.

This module implements the ModelPreferences capability from the MCP protocol,
allowing servers to express preferences for model selection during sampling.
"""

from typing import List, Optional


class ModelHint:
    """
    Hint for model selection.

    Model hints allow the server to suggest specific models or model families
    that would be appropriate for a given task.
    """

    def __init__(self, name: str):
        """
        Initialize a model hint.

        Args:
            name: A hint for a model name (e.g., 'claude-3-5-sonnet', 'sonnet', 'claude').
                 This should be treated as a substring matching.
        """
        self.name = name

    def to_dict(self) -> dict:
        """Convert model hint to dictionary."""
        return {"name": self.name}


class ModelPreferences:
    """
    Preferences for model selection to guide LLM client decisions.

    The ModelPreferences class provides a standardized way for servers to express
    prioritization along three key dimensions (intelligence, speed, cost) that can
    help clients make more informed decisions when selecting LLM models for specific tasks.

    These preferences serve as advisory hints that help optimize the tradeoffs between:
    - Intelligence/capability: Higher quality, more capable models (but often slower/costlier)
    - Speed: Faster response time and lower latency (but potentially less capable)
    - Cost: Lower token or API costs (but potentially less capable or slower)

    The class also supports model-specific hints that can recommend particular models
    or model families that are well-suited for specific tasks (e.g., suggesting Claude
    models for creative writing or GPT-4V for image analysis).

    All preferences are expressed with normalized values between 0.0 (lowest priority)
    and 1.0 (highest priority) to allow for consistent interpretation across different
    implementations.

    Note: These preferences are always advisory. Clients may use them as guidance but
    are not obligated to follow them, particularly if there are overriding user preferences
    or system constraints.

    Usage example:
        ```python
        # For a coding task requiring high intelligence but where cost is a major concern
        preferences = ModelPreferences(
            intelligence_priority=0.8,  # High priority on capability
            speed_priority=0.4,         # Moderate priority on speed
            cost_priority=0.7,          # High priority on cost
            hints=[ModelHint("gpt-4-turbo")]  # Specific model recommendation
        )
        ```
    """

    def __init__(
        self,
        intelligence_priority: float = 0.5,
        speed_priority: float = 0.5,
        cost_priority: float = 0.5,
        hints: Optional[List[ModelHint]] = None,
    ):
        """
        Initialize model preferences.

        Args:
            intelligence_priority: How much to prioritize intelligence/capabilities (0.0-1.0).
                Higher values favor more capable, sophisticated models that may produce
                higher quality outputs, handle complex tasks, or follow instructions better.
                Default: 0.5 (balanced)
            speed_priority: How much to prioritize sampling speed/latency (0.0-1.0).
                Higher values favor faster models with lower latency, which is important
                for real-time applications, interactive experiences, or time-sensitive tasks.
                Default: 0.5 (balanced)
            cost_priority: How much to prioritize cost efficiency (0.0-1.0).
                Higher values favor more economical models with lower token or API costs,
                which is important for budget-constrained applications or high-volume usage.
                Default: 0.5 (balanced)
            hints: Optional model hints in preference order. These can suggest specific
                models or model families that would be appropriate for the task.
                The list should be ordered by preference (most preferred first).
        """
        # Clamp values between 0 and 1
        self.intelligence_priority = max(0.0, min(1.0, intelligence_priority))
        self.speed_priority = max(0.0, min(1.0, speed_priority))
        self.cost_priority = max(0.0, min(1.0, cost_priority))
        self.hints = hints or []

    def to_dict(self) -> dict:
        """Convert model preferences to dictionary."""
        return {
            "intelligencePriority": self.intelligence_priority,
            "speedPriority": self.speed_priority,
            "costPriority": self.cost_priority,
            "hints": [hint.to_dict() for hint in self.hints],
        }


# Pre-defined preference templates for common use cases

# Default balanced preference profile - no strong bias in any direction
# Use when there's no clear priority between intelligence, speed, and cost
# Good for general-purpose applications where trade-offs are acceptable
BALANCED_PREFERENCES = ModelPreferences(
    intelligence_priority=0.5, speed_priority=0.5, cost_priority=0.5
)

# Prioritizes high-quality, sophisticated model responses
# Use for complex reasoning, creative tasks, or critical applications
# where accuracy and capability matter more than speed or cost
INTELLIGENCE_FOCUSED = ModelPreferences(
    intelligence_priority=0.9,
    speed_priority=0.3,
    cost_priority=0.3,
    hints=[ModelHint("claude-3-5-opus")],
)

# Prioritizes response speed and low latency
# Use for real-time applications, interactive experiences,
# chatbots, or any use case where user wait time is critical
SPEED_FOCUSED = ModelPreferences(
    intelligence_priority=0.3,
    speed_priority=0.9,
    cost_priority=0.5,
    hints=[ModelHint("claude-3-haiku"), ModelHint("gemini-flash")],
)

# Prioritizes cost efficiency and token economy
# Use for high-volume applications, background processing,
# or when operating under strict budget constraints
COST_FOCUSED = ModelPreferences(
    intelligence_priority=0.3,
    speed_priority=0.5,
    cost_priority=0.9,
    hints=[ModelHint("mistral"), ModelHint("gemini-flash")],
)
