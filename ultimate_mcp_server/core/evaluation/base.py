# --- core/evaluation/base.py (NEW) ---
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Type  # Added Type

from pydantic import BaseModel, Field  # Added Field

# Assuming ModelResponseData is correctly importable from its location
# This path might need adjustment based on your actual project structure.
# If core.models.tournament is in the same parent directory as core.evaluation,
# then this relative import should work if Python's import system can find 'ultimate_mcp_server'.
# A common way is to have 'ultimate_mcp_server' as a top-level package.
try:
    from ultimate_mcp_server.core.models.tournament import ModelResponseData
except ImportError:
    # Fallback for different structures or if running script directly for testing
    # This is a common pattern but ensure your PYTHONPATH or project structure handles it in production
    # For instance, if 'ultimate_mcp_server' is the root of your installable package.
    from ..models.tournament import ModelResponseData


class EvaluationScore(BaseModel):
    score: float  # Primary numerical score
    details: Optional[str] = None  # Textual explanation or breakdown
    metrics: Dict[str, Any] = Field(
        default_factory=dict
    )  # Additional quantitative metrics from this evaluator


class Evaluator(ABC):
    """Abstract base class for evaluators."""

    evaluator_type: str  # Must be overridden by subclasses, e.g., "llm_grader"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator with its specific configuration.
        `config` comes from `EvaluatorConfig.params`.
        """
        self.config = config

    @abstractmethod
    async def score(
        self,
        response_data: ModelResponseData,
        original_prompt: str,
        tournament_type: Literal["code", "text"],  # "code" and "text" were undefined before
    ) -> EvaluationScore:
        """
        Scores a model's response.

        Args:
            response_data: The ModelResponseData object containing the response text, code, etc.
            original_prompt: The initial prompt for the tournament.
            tournament_type: The type of the tournament ('code' or 'text').

        Returns:
            An EvaluationScore object.
        """
        pass

    @classmethod
    def get_config_schema(cls) -> Optional[Dict[str, Any]]:
        """
        Optional: Returns a JSON schema for the evaluator's specific `params`.
        This can be used for validation or UI generation.
        """
        return None


# Example: A registry for evaluators (could be more sophisticated with entry points)
EVALUATOR_REGISTRY: Dict[str, Type[Evaluator]] = {}


def register_evaluator(cls: Type[Evaluator]):
    if not hasattr(cls, "evaluator_type") or not cls.evaluator_type:
        raise ValueError(
            f"Evaluator class {cls.__name__} must define a 'evaluator_type' attribute."
        )
    if cls.evaluator_type in EVALUATOR_REGISTRY:
        raise ValueError(f"Evaluator type '{cls.evaluator_type}' already registered.")
    EVALUATOR_REGISTRY[cls.evaluator_type] = cls
    return cls
