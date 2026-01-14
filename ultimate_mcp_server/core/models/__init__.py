# Core models module for ultimate_mcp_server
# This module contains all Pydantic models used for tournaments, requests, and synthesis

from ultimate_mcp_server.core.models.tournament import (
    TournamentStatus,
    ModelConfig,
    EvaluatorConfig,
    TournamentConfig,
    TournamentRoundResult,
    ModelResponseData,
    TournamentData,
    CreateTournamentInput,
    CreateTournamentOutput,
    GetTournamentStatusInput,
    GetTournamentStatusOutput,
    GetTournamentResultsInput,
    CancelTournamentInput,
    CancelTournamentOutput,
    TournamentBasicInfo,
    SingleShotGeneratorModelConfig,
    SingleShotIndividualResponse,
    SingleShotSynthesisInput,
    SingleShotSynthesisOutput,
)

from ultimate_mcp_server.core.models.requests import (
    CompletionRequest,
)

__all__ = [
    # Tournament models
    "TournamentStatus",
    "ModelConfig",
    "EvaluatorConfig",
    "TournamentConfig",
    "TournamentRoundResult",
    "ModelResponseData",
    "TournamentData",
    "CreateTournamentInput",
    "CreateTournamentOutput",
    "GetTournamentStatusInput",
    "GetTournamentStatusOutput",
    "GetTournamentResultsInput",
    "CancelTournamentInput",
    "CancelTournamentOutput",
    "TournamentBasicInfo",
    # Single-shot synthesis models
    "SingleShotGeneratorModelConfig",
    "SingleShotIndividualResponse",
    "SingleShotSynthesisInput",
    "SingleShotSynthesisOutput",
    # Request models
    "CompletionRequest",
]
