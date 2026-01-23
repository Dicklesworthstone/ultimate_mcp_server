"""Request models for Ultimate MCP Server."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """Request model for LLM completion calls."""

    prompt: str = Field(..., description="The prompt to send to the model")
    model: str = Field(..., description="Model identifier (e.g., 'openai/gpt-4o')")
    provider: Optional[str] = Field(
        default=None, description="Provider name (inferred from model if not specified)"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    messages: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Message history for chat models"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None, description="Sequences that stop generation"
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling parameter")
    presence_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description="Presence penalty"
    )
    frequency_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    additional_params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )
