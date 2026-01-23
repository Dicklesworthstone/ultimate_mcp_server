"""Prompt template management and rendering for Ultimate MCP Server."""

import json
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.services.prompts.repository import get_prompt_repository
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


class TemplateFormat(str, Enum):
    """Template format options."""

    JINJA = "jinja"
    SIMPLE = "simple"
    MARKDOWN = "markdown"
    JSON = "json"


class TemplateType(str, Enum):
    """Template type options."""

    COMPLETION = "completion"
    CHAT = "chat"
    SYSTEM = "system"
    USER = "user"
    FUNCTION = "function"
    EXTRACTION = "extraction"


class PromptTemplate:
    """Template for generating prompts for LLM providers."""

    def __init__(
        self,
        template: str,
        template_id: str,
        format: Union[str, TemplateFormat] = TemplateFormat.JINJA,
        type: Union[str, TemplateType] = TemplateType.COMPLETION,
        metadata: Optional[Dict[str, Any]] = None,
        provider_defaults: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        required_vars: Optional[List[str]] = None,
        example_vars: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a prompt template.

        Args:
            template: Template string
            template_id: Unique identifier for this template
            format: Template format (jinja, simple, markdown, or json)
            type: Template type (completion, chat, system, user, function, extraction)
            metadata: Optional metadata for the template
            provider_defaults: Optional provider-specific defaults
            description: Optional description of the template
            required_vars: Optional list of required variables
            example_vars: Optional example variables for testing
        """
        self.template = template
        self.template_id = template_id

        # Normalize format and type to enum values
        self.format = TemplateFormat(format) if isinstance(format, str) else format
        self.type = TemplateType(type) if isinstance(type, str) else type

        # Store additional attributes
        self.metadata = metadata or {}
        self.provider_defaults = provider_defaults or {}
        self.description = description
        self.example_vars = example_vars or {}

        # Extract required variables based on format
        self.required_vars = required_vars or self._extract_required_vars()

        # Compile template if using Jinja format
        self._compiled_template: Optional[Template] = None
        if self.format == TemplateFormat.JINJA:
            self._compiled_template = self._compile_template()

    def _extract_required_vars(self) -> List[str]:
        """Extract required variables from template based on format.

        Returns:
            List of required variable names
        """
        if self.format == TemplateFormat.JINJA:
            # Extract variables using regex for basic Jinja pattern
            matches = re.findall(r"{{(.*?)}}", self.template)
            vars_set: Set[str] = set()

            for match in matches:
                # Extract variable name (removing filters and whitespace)
                var_name = match.split("|")[0].strip()
                if var_name and not var_name.startswith("_"):
                    vars_set.add(var_name)

            return sorted(list(vars_set))

        elif self.format == TemplateFormat.SIMPLE:
            # Extract variables from {variable} format
            matches = re.findall(r"{([^{}]*)}", self.template)
            return sorted(list(set(matches)))

        elif self.format == TemplateFormat.JSON:
            # Try to find JSON template variables
            try:
                # Parse as JSON to find potential variables
                template_dict = json.loads(self.template)
                return self._extract_json_vars(template_dict)
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse JSON template: {self.template_id}", emoji_key="warning"
                )
                return []

        # Default: no variables detected
        return []

    def _extract_json_vars(self, obj: Any, prefix: str = "") -> List[str]:
        """Recursively extract variables from a JSON object.

        Args:
            obj: JSON object to extract variables from
            prefix: Prefix for nested variables

        Returns:
            List of variable names
        """
        vars_list = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    # This is a variable placeholder
                    var_name = value[2:-1]  # Remove ${ and }
                    vars_list.append(f"{prefix}{var_name}")
                elif isinstance(value, (dict, list)):
                    # Recursively extract from nested structures
                    nested_prefix = f"{prefix}{key}." if prefix else f"{key}."
                    vars_list.extend(self._extract_json_vars(value, nested_prefix))
        elif isinstance(obj, list):
            for _i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    vars_list.extend(self._extract_json_vars(item, prefix))
                elif isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                    var_name = item[2:-1]
                    vars_list.append(f"{prefix}{var_name}")

        return sorted(list(set(vars_list)))

    def _compile_template(self) -> Template:
        """Compile the Jinja template.

        Returns:
            Compiled Jinja template

        Raises:
            ValueError: If template compilation fails
        """
        try:
            env = Environment(autoescape=select_autoescape(["html", "xml"]))
            return env.from_string(self.template)
        except Exception as e:
            logger.error(
                f"Failed to compile template {self.template_id}: {str(e)}", emoji_key="error"
            )
            raise ValueError(f"Invalid template format: {str(e)}") from e

    def render(self, variables: Dict[str, Any]) -> str:
        """Render the template with the provided variables.

        Args:
            variables: Dictionary of variables to render with

        Returns:
            Rendered template string

        Raises:
            ValueError: If required variables are missing
        """
        # Check for required variables
        missing_vars = [var for var in self.required_vars if var not in variables]
        if missing_vars:
            raise ValueError(
                f"Missing required variables for template {self.template_id}: {', '.join(missing_vars)}"
            )

        # Render based on format
        if self.format == TemplateFormat.JINJA:
            if not self._compiled_template:
                self._compiled_template = self._compile_template()
            return self._compiled_template.render(**variables)

        elif self.format == TemplateFormat.SIMPLE:
            # Simple variable substitution with {var} syntax
            result = self.template
            for var_name, var_value in variables.items():
                result = result.replace(f"{{{var_name}}}", str(var_value))
            return result

        elif self.format == TemplateFormat.JSON:
            try:
                # Parse template as JSON
                template_dict = json.loads(self.template)

                # Replace variables in the JSON structure
                rendered_dict = self._render_json_vars(template_dict, variables)

                # Convert back to JSON string
                return json.dumps(rendered_dict)

            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse JSON template: {self.template_id}", emoji_key="error"
                )
                # Fall back to simple replacement
                return self.template

        elif self.format == TemplateFormat.MARKDOWN:
            # Process markdown with simple variable substitution
            result = self.template
            for var_name, var_value in variables.items():
                result = result.replace(f"{{{var_name}}}", str(var_value))
            return result

        # Default: return template as is
        return self.template

    def _render_json_vars(self, obj: Any, variables: Dict[str, Any]) -> Any:
        """Recursively render variables in a JSON object.

        Args:
            obj: JSON object to render variables in
            variables: Dictionary of variables to render with

        Returns:
            Rendered JSON object
        """
        if isinstance(obj, dict):
            return {key: self._render_json_vars(value, variables) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._render_json_vars(item, variables) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            # This is a variable placeholder
            var_name = obj[2:-1]  # Remove ${ and }
            # Get the variable value, or keep placeholder if not found
            return variables.get(var_name, obj)
        else:
            return obj

    def validate_variables(self, variables: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that all required variables are provided.

        Args:
            variables: Dictionary of variables to validate

        Returns:
            Tuple of (is_valid, missing_variables)
        """
        missing_vars = [var for var in self.required_vars if var not in variables]
        return len(missing_vars) == 0, missing_vars

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary representation.

        Returns:
            Dictionary representation of template
        """
        return {
            "template_id": self.template_id,
            "template": self.template,
            "format": self.format.value,
            "type": self.type.value,
            "metadata": self.metadata,
            "provider_defaults": self.provider_defaults,
            "description": self.description,
            "required_vars": self.required_vars,
            "example_vars": self.example_vars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create a template from dictionary representation.

        Args:
            data: Dictionary representation of template

        Returns:
            PromptTemplate instance
        """
        return cls(
            template=data["template"],
            template_id=data["template_id"],
            format=data.get("format", TemplateFormat.JINJA),
            type=data.get("type", TemplateType.COMPLETION),
            metadata=data.get("metadata"),
            provider_defaults=data.get("provider_defaults"),
            description=data.get("description"),
            required_vars=data.get("required_vars"),
            example_vars=data.get("example_vars"),
        )

    def get_provider_defaults(self, provider: str) -> Dict[str, Any]:
        """Get provider-specific default parameters.

        Args:
            provider: Provider name

        Returns:
            Dictionary of default parameters
        """
        return self.provider_defaults.get(provider, {})


class PromptTemplateRenderer:
    """Service for rendering prompt templates."""

    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        """Initialize the prompt template renderer.

        Args:
            template_dir: Optional directory containing template files
        """
        # Set template directory
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Default to project directory / templates
            self.template_dir = Path.home() / ".ultimate" / "templates"

        # Create directory if it doesn't exist
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Set up Jinja environment for file-based templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Get prompt repository for template storage
        self.repository = get_prompt_repository()

        # Template cache
        self._template_cache: Dict[str, PromptTemplate] = {}

    async def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID.

        Args:
            template_id: Template identifier

        Returns:
            PromptTemplate instance or None if not found
        """
        # Check cache first
        if template_id in self._template_cache:
            return self._template_cache[template_id]

        # Look up in repository
        template_data = await self.repository.get_prompt(template_id)
        if template_data:
            template = PromptTemplate.from_dict(template_data)
            # Cache for future use
            self._template_cache[template_id] = template
            return template

        # Try to load from file if not in repository
        template_path = self.template_dir / f"{template_id}.j2"
        if template_path.exists():
            # Load template from file
            with open(template_path, "r", encoding="utf-8") as f:
                template_content = f.read()

            # Create template instance
            template = PromptTemplate(
                template=template_content,
                template_id=template_id,
                format=TemplateFormat.JINJA,
            )

            # Cache for future use
            self._template_cache[template_id] = template
            return template

        return None

    async def render_template(
        self, template_id: str, variables: Dict[str, Any], provider: Optional[str] = None
    ) -> str:
        """Render a template with variables.

        Args:
            template_id: Template identifier
            variables: Variables to render the template with
            provider: Optional provider name for provider-specific adjustments

        Returns:
            Rendered template string

        Raises:
            ValueError: If template not found or rendering fails
        """
        # Get the template
        template = await self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        # Check if all required variables are provided
        is_valid, missing_vars = template.validate_variables(variables)
        if not is_valid:
            raise ValueError(
                f"Missing required variables for template {template_id}: {', '.join(missing_vars)}"
            )

        # Render the template
        rendered = template.render(variables)

        # Apply provider-specific adjustments if provided
        if provider:
            rendered = self._apply_provider_adjustments(rendered, provider, template)

        return rendered

    def _apply_provider_adjustments(
        self, rendered: str, provider: str, template: PromptTemplate
    ) -> str:
        """Apply provider-specific adjustments to rendered template.

        Args:
            rendered: Rendered template string
            provider: Provider name
            template: Template being rendered

        Returns:
            Adjusted template string
        """
        # Apply provider-specific transformations
        if provider == Provider.ANTHROPIC.value:
            # Anthropic-specific adjustments
            if template.type == TemplateType.SYSTEM:
                # Ensure no trailing newlines for system prompts
                rendered = rendered.rstrip()
        elif provider == Provider.OPENAI.value:
            # OpenAI-specific adjustments
            pass
        elif provider == Provider.GEMINI.value:
            # Gemini-specific adjustments
            pass

        return rendered

    async def save_template(self, template: PromptTemplate) -> bool:
        """Save a template to the repository.

        Args:
            template: Template to save

        Returns:
            True if successful
        """
        # Update cache
        self._template_cache[template.template_id] = template

        # Save to repository
        return await self.repository.save_prompt(
            prompt_id=template.template_id, prompt_data=template.to_dict()
        )

    async def delete_template(self, template_id: str) -> bool:
        """Delete a template from the repository.

        Args:
            template_id: Template identifier

        Returns:
            True if successful
        """
        # Remove from cache
        if template_id in self._template_cache:
            del self._template_cache[template_id]

        # Delete from repository
        return await self.repository.delete_prompt(template_id)

    async def list_templates(self) -> List[str]:
        """List available templates.

        Returns:
            List of template IDs
        """
        # Get templates from repository
        return await self.repository.list_prompts()

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()


# Global template renderer instance
_template_renderer: Optional[PromptTemplateRenderer] = None


def get_template_renderer() -> PromptTemplateRenderer:
    """Get the global template renderer instance.

    Returns:
        PromptTemplateRenderer instance
    """
    global _template_renderer
    if _template_renderer is None:
        _template_renderer = PromptTemplateRenderer()
    return _template_renderer


@lru_cache(maxsize=32)
def get_template_path(template_id: str) -> Optional[Path]:
    """Get the path to a template file.

    Args:
        template_id: Template identifier

    Returns:
        Path to template file or None if not found
    """
    # Try standard locations
    template_dirs = [
        # First check the user's template directory
        Path.home() / ".ultimate" / "templates",
        # Then check the package's template directory
        Path(__file__).parent.parent.parent.parent / "templates",
    ]

    for template_dir in template_dirs:
        # Check for .j2 extension first
        template_path = template_dir / f"{template_id}.j2"
        if template_path.exists():
            return template_path

        # Check for .tpl extension
        template_path = template_dir / f"{template_id}.tpl"
        if template_path.exists():
            return template_path

        # Check for .md extension
        template_path = template_dir / f"{template_id}.md"
        if template_path.exists():
            return template_path

        # Check for .json extension
        template_path = template_dir / f"{template_id}.json"
        if template_path.exists():
            return template_path

    return None


async def render_prompt_template(
    template_id: str, variables: Dict[str, Any], provider: Optional[str] = None
) -> str:
    """Render a prompt template.

    Args:
        template_id: Template identifier
        variables: Variables to render the template with
        provider: Optional provider name for provider-specific adjustments

    Returns:
        Rendered template string

    Raises:
        ValueError: If template not found or rendering fails
    """
    renderer = get_template_renderer()
    return await renderer.render_template(template_id, variables, provider)


async def render_prompt(
    template_content: str,
    variables: Dict[str, Any],
    format: Union[str, TemplateFormat] = TemplateFormat.JINJA,
) -> str:
    """Render a prompt from template content.

    Args:
        template_content: Template content string
        variables: Variables to render the template with
        format: Template format

    Returns:
        Rendered template string

    Raises:
        ValueError: If rendering fails
    """
    # Create a temporary template
    template = PromptTemplate(
        template=template_content,
        template_id="_temp_template",
        format=format,
    )

    # Render and return
    return template.render(variables)


async def get_template_defaults(template_id: str, provider: str) -> Dict[str, Any]:
    """Get provider-specific default parameters for a template.

    Args:
        template_id: Template identifier
        provider: Provider name

    Returns:
        Dictionary of default parameters

    Raises:
        ValueError: If template not found
    """
    renderer = get_template_renderer()
    template = await renderer.get_template(template_id)
    if not template:
        raise ValueError(f"Template not found: {template_id}")

    return template.get_provider_defaults(provider)
