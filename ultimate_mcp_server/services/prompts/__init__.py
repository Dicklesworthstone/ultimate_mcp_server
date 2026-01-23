"""Prompt service for Ultimate MCP Server."""

from ultimate_mcp_server.services.prompts.repository import (
    PromptRepository,
    get_prompt_repository,
)
from ultimate_mcp_server.services.prompts.templates import (
    PromptTemplate,
    PromptTemplateRenderer,
    render_prompt,
    render_prompt_template,
)


class PromptService:
    """Service for managing prompts."""

    def __init__(self):
        """Initialize prompt service."""
        self.repository = get_prompt_repository()
        self.renderer = PromptTemplateRenderer()

    def get_prompt(self, prompt_id: str) -> PromptTemplate:
        """Get prompt by ID.

        Args:
            prompt_id: Prompt ID

        Returns:
            Prompt template
        """
        return self.repository.get_prompt(prompt_id)

    def render_prompt(self, prompt_id: str, variables: dict = None) -> str:
        """Render prompt with variables.

        Args:
            prompt_id: Prompt ID
            variables: Variables to use in rendering

        Returns:
            Rendered prompt text
        """
        prompt = self.get_prompt(prompt_id)
        return self.renderer.render(prompt, variables or {})


def get_prompt_service() -> PromptService:
    """Get prompt service.

    Returns:
        Prompt service instance
    """
    return PromptService()


__all__ = [
    "PromptRepository",
    "get_prompt_repository",
    "PromptTemplate",
    "PromptTemplateRenderer",
    "render_prompt",
    "render_prompt_template",
    "PromptService",
    "get_prompt_service",
]
