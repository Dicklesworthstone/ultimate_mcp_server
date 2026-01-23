"""Tournament and synthesis models for Ultimate MCP Server."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TournamentStatus(str, Enum):
    """Status states for tournaments and rounds."""

    PENDING = "PENDING"
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ModelConfig(BaseModel):
    """Configuration for a model participating in a tournament."""

    model_id: str = Field(..., description="Model identifier (e.g., 'openai/gpt-4o')")
    diversity_count: int = Field(
        default=1, ge=1, description="Number of variants to run for this model"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Temperature for generation"
    )
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for the model")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class EvaluatorConfig(BaseModel):
    """Configuration for an evaluator in a tournament."""

    evaluator_id: str = Field(..., description="Unique identifier for this evaluator instance")
    type: str = Field(..., description="Type of evaluator (e.g., 'llm_grader', 'code_execution')")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Evaluator-specific parameters"
    )
    weight: float = Field(
        default=1.0, ge=0.0, description="Weight of this evaluator's score in overall ranking"
    )


class TournamentConfig(BaseModel):
    """Configuration for a tournament."""

    name: str = Field(..., description="Human-readable name for the tournament")
    prompt: str = Field(..., description="The task prompt for all models")
    models: List[ModelConfig] = Field(..., min_length=1, description="List of model configurations")
    rounds: int = Field(default=3, ge=1, description="Number of tournament rounds")
    tournament_type: Literal["code", "text"] = Field(
        default="code", description="Type of tournament"
    )
    extraction_model_id: Optional[str] = Field(
        default="anthropic/claude-3-5-haiku-20241022",
        description="Model to use for code extraction",
    )
    evaluators: List[EvaluatorConfig] = Field(
        default_factory=list, description="List of evaluator configurations"
    )
    max_retries_per_model_call: int = Field(
        default=3, ge=0, description="Max retries per model call"
    )
    retry_backoff_base_seconds: float = Field(
        default=1.0, ge=0.1, description="Base seconds for retry backoff"
    )
    max_concurrent_model_calls: int = Field(
        default=5, ge=1, description="Max concurrent model calls"
    )


class ModelResponseData(BaseModel):
    """Data for a single model's response in a tournament round."""

    model_id_original: str = Field(..., description="Original model ID from config")
    model_id_variant: str = Field(..., description="Unique variant ID (e.g., 'openai/gpt-4o/v0')")
    round_num: int = Field(..., ge=0, description="Round number this response belongs to")
    response_text: Optional[str] = Field(
        default=None, description="Raw response text from the model"
    )
    thinking_process: Optional[str] = Field(
        default=None, description="Extracted thinking/reasoning block"
    )
    extracted_code: Optional[str] = Field(default=None, description="Extracted code from response")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    timestamp: Optional[datetime] = Field(
        default=None, description="When the response was generated"
    )
    response_file_path: Optional[str] = Field(
        default=None, description="Path to saved response file"
    )
    extracted_code_file_path: Optional[str] = Field(
        default=None, description="Path to saved code file"
    )
    error: Optional[str] = Field(default=None, description="Error message if generation failed")
    scores: Dict[str, Any] = Field(default_factory=dict, description="Scores from evaluators")
    overall_score: Optional[float] = Field(default=None, description="Weighted overall score")


class TournamentRoundResult(BaseModel):
    """Results for a single round of a tournament."""

    round_num: int = Field(..., ge=0, description="Round number")
    status: TournamentStatus = Field(default=TournamentStatus.PENDING, description="Round status")
    start_time: Optional[datetime] = Field(default=None, description="When the round started")
    end_time: Optional[datetime] = Field(default=None, description="When the round ended")
    error_message: Optional[str] = Field(
        default=None, alias="error", description="Error message if round failed"
    )
    responses: Dict[str, ModelResponseData] = Field(
        default_factory=dict, description="Map of variant_id to response data"
    )
    comparison_file_path: Optional[str] = Field(
        default=None, description="Path to comparison report"
    )
    leaderboard_file_path: Optional[str] = Field(
        default=None, description="Path to leaderboard file"
    )

    class Config:
        populate_by_name = True


class TournamentData(BaseModel):
    """Full tournament data including configuration, status, and results."""

    tournament_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique tournament ID"
    )
    name: str = Field(..., description="Tournament name")
    config: TournamentConfig = Field(..., description="Tournament configuration")
    status: TournamentStatus = Field(
        default=TournamentStatus.CREATED, description="Current tournament status"
    )
    current_round: int = Field(default=-1, description="Current round number (-1 if not started)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp"
    )
    start_time: Optional[datetime] = Field(default=None, description="When execution started")
    end_time: Optional[datetime] = Field(default=None, description="When execution ended")
    error_message: Optional[str] = Field(
        default=None, description="Error message if tournament failed"
    )
    storage_path: Optional[str] = Field(
        default=None, description="Path to tournament storage directory"
    )
    rounds_results: List[TournamentRoundResult] = Field(
        default_factory=list, description="Results for each round"
    )
    overall_best_response: Optional[ModelResponseData] = Field(
        default=None, description="Best response across all rounds"
    )


# --- Input/Output models for tool functions ---


class CreateTournamentInput(BaseModel):
    """Input for creating a tournament."""

    name: str = Field(..., description="Tournament name")
    prompt: str = Field(..., description="Task prompt")
    model_configs: List[ModelConfig] = Field(
        ..., alias="models", min_length=1, description="Model configurations"
    )
    rounds: int = Field(default=3, ge=1, description="Number of rounds")
    tournament_type: Literal["code", "text"] = Field(default="code", description="Tournament type")
    extraction_model_id: Optional[str] = Field(default="anthropic/claude-3-5-haiku-20241022")
    evaluators: List[EvaluatorConfig] = Field(default_factory=list)
    max_retries_per_model_call: int = Field(default=3, ge=0)
    retry_backoff_base_seconds: float = Field(default=1.0, ge=0.1)
    max_concurrent_model_calls: int = Field(default=5, ge=1)

    class Config:
        populate_by_name = True


class CreateTournamentOutput(BaseModel):
    """Output after creating a tournament."""

    tournament_id: str
    status: TournamentStatus
    storage_path: Optional[str] = None
    message: Optional[str] = None


class GetTournamentStatusInput(BaseModel):
    """Input for getting tournament status."""

    tournament_id: str


class GetTournamentStatusOutput(BaseModel):
    """Output for tournament status."""

    tournament_id: str
    name: str
    tournament_type: str
    status: TournamentStatus
    current_round: int
    total_rounds: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None


class GetTournamentResultsInput(BaseModel):
    """Input for getting tournament results."""

    tournament_id: str


class CancelTournamentInput(BaseModel):
    """Input for cancelling a tournament."""

    tournament_id: str


class CancelTournamentOutput(BaseModel):
    """Output after attempting to cancel a tournament."""

    tournament_id: str
    status: TournamentStatus
    message: str


class TournamentBasicInfo(BaseModel):
    """Basic tournament information for listing."""

    tournament_id: str
    name: str
    tournament_type: str
    status: TournamentStatus
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# --- Single-shot synthesis models ---


class SingleShotGeneratorModelConfig(BaseModel):
    """Configuration for a model in single-shot synthesis."""

    model_id: str = Field(..., description="Model identifier")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    system_prompt: Optional[str] = Field(default=None)


class SingleShotIndividualResponse(BaseModel):
    """Response from a single expert model in synthesis."""

    model_id: str
    response_text: Optional[str] = None
    thinking_process: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class SingleShotSynthesisInput(BaseModel):
    """Input for single-shot synthesis."""

    name: str = Field(..., description="Name for this synthesis task")
    prompt: str = Field(..., description="The prompt for all expert models")
    expert_model_configs: List[SingleShotGeneratorModelConfig] = Field(
        ..., alias="expert_models", min_length=1, description="Expert model configurations"
    )
    synthesizer_model_config: SingleShotGeneratorModelConfig = Field(
        ..., alias="synthesizer_model", description="Synthesizer model configuration"
    )
    tournament_type: Literal["code", "text"] = Field(default="text")
    synthesis_instructions: Optional[str] = Field(default=None)

    class Config:
        populate_by_name = True


class SingleShotSynthesisOutput(BaseModel):
    """Output from single-shot synthesis."""

    request_id: str
    name: str
    status: str  # SUCCESS, PARTIAL_SUCCESS, FAILED
    expert_responses: List[SingleShotIndividualResponse] = Field(default_factory=list)
    synthesized_response_text: Optional[str] = None
    synthesizer_thinking_process: Optional[str] = None
    synthesized_extracted_code: Optional[str] = None
    synthesizer_metrics: Dict[str, Any] = Field(default_factory=dict)
    total_metrics: Dict[str, Any] = Field(default_factory=dict)
    storage_path: Optional[str] = None
    error_message: Optional[str] = None
