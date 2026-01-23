"""Prompt template service for managing and rendering prompt templates."""

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.exceptions import PromptTemplateError
from ultimate_mcp_server.utils.logging import get_logger

logger = get_logger(__name__)

# Singleton instance
_prompt_service = None


def get_prompt_service():
    """Get the global prompt service instance."""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService()
    return _prompt_service


class PromptService:
    """
    Service for managing, storing, rendering, and versioning prompt templates.

    The PromptService provides a centralized system for handling prompt templates
    used throughout the MCP application. It manages the entire lifecycle of prompt
    templates, from loading them from disk to rendering them with variables for use
    with language models. The service provides both persistent storage and runtime
    management of templates.

    Key Features:
    - File-based template storage using both .txt and .json formats
    - Runtime template registration and modification
    - Variable substitution for dynamic prompt generation
    - Categorization of templates for organizational purposes
    - Asynchronous persistence to avoid blocking operations
    - Error handling and logging for template issues

    Template Organization:
    Templates are organized using a naming convention where the prefix before the
    first underscore represents the category (e.g., "rag_query" belongs to the "rag"
    category). This categorization is used when saving templates to disk, with each
    category stored in its own JSON file.

    File Formats:
    - Individual .txt files: One template per file, filename is the template name
    - JSON files: Multiple templates in a single file, typically grouped by category

    This service employs a singleton pattern, ensuring only one instance exists
    across the application. Always use the get_prompt_service() or get_prompt_manager()
    functions to access it, rather than instantiating directly.

    Usage Example:
    ```python
    # Get the service
    prompt_service = get_prompt_service()

    # Retrieve a template
    template = prompt_service.get_template("rag_query")

    # Register a new template
    prompt_service.register_template(
        "greeting",
        "Hello {name}, welcome to {service_name}!"
    )

    # Render a template with variables
    greeting = prompt_service.render_template(
        "greeting",
        {"name": "Alice", "service_name": "Ultimate MCP"}
    )
    ```

    Note:
        All file operations are handled with proper error handling to ensure
        the service continues functioning even if individual template files
        are corrupted or missing.
    """

    def __init__(self):
        """Initialize the prompt service.

        Args:
            templates_dir: Directory containing template files
        """
        self.templates: Dict[str, str] = {}
        try:
            config = get_config()
            self.templates_dir = config.prompt_templates_directory
            logger.info(
                f"Initializing PromptService. Looking for templates in: {self.templates_dir}"
            )
            self._load_templates()
        except Exception as e:
            logger.error(f"Failed to initialize PromptService: {e}", exc_info=True)
            # Allow service to exist even if loading fails, get_template will raise errors

        # Create templates directory if it doesn't exist
        os.makedirs(self.templates_dir, exist_ok=True)

        # Read templates from files
        self._read_templates()
        logger.info(f"Prompt service initialized with {len(self.templates)} templates")

    def _load_templates(self):
        """
        Load prompt templates from individual .txt files in the templates directory.

        This method scans the configured templates directory for .txt files and loads
        each file as a separate template. It uses the filename (without extension)
        as the template name and the file content as the template text. This provides
        a simple way to manage templates as individual files, which can be useful for
        version control and template organization.

        The loading process:
        1. Verifies the templates directory exists and is accessible
        2. Scans for all .txt files using glob pattern matching
        3. For each file:
           - Extracts the template name from the filename
           - Reads the file content as the template text
           - Adds the template to the in-memory template dictionary
           - Logs the successful load
        4. Handles exceptions for each file individually to prevent a single corrupted
           file from blocking all template loading
        5. Logs summary information about the loading process

        This approach allows templates to be:
        - Managed individually in separate files
        - Edited directly using text editors
        - Organized in a flat structure for simplicity
        - Added or removed without changing code

        The method is called during service initialization but can be called again
        to refresh templates from disk if needed.

        Note:
            This method only processes .txt files. JSON format templates are handled
            by the separate _read_templates method. Both methods work together to
            provide a complete template loading solution.
        """
        if not Path(self.templates_dir).is_dir():
            logger.warning(
                f"Prompt templates directory not found or not a directory: {self.templates_dir}"
            )
            return

        loaded_count = 0
        for filepath in Path(self.templates_dir).glob("*.txt"):
            try:
                template_name = filepath.stem  # Use filename without extension as name
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                self.templates[template_name] = content
                logger.debug(f"Loaded prompt template: {template_name}")
                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load prompt template {filepath.name}: {e}")

        if loaded_count > 0:
            logger.info(f"Successfully loaded {loaded_count} prompt templates.")
        else:
            logger.info("No prompt templates found or loaded.")

    def _read_templates(self) -> None:
        """
        Load prompt templates from JSON files in the templates directory.

        This method complements _load_templates by handling template collections
        stored in JSON format. It scans for .json files in the templates directory
        and processes each file to extract multiple templates. Each JSON file can
        contain multiple templates, organized as a dictionary with template names
        as keys and template content as values.

        The loading process:
        1. Scans the templates directory for all .json files
        2. For each JSON file:
           - Parses the JSON content
           - Extracts each key-value pair as a template name and content
           - Handles both simple string templates and structured template objects
           - Adds valid templates to the in-memory template collection
        3. Logs detailed information about successful and failed loads

        Template JSON Format Support:
        - Simple format: `{"template_name": "Template content with {variables}"}`
        - Structured format: `{"template_name": {"text": "Template content", ...}}`
          where the template text is extracted from the "text" field

        The JSON format is particularly useful for:
        - Storing multiple related templates in a single file
        - Organizing templates by category or function
        - Including metadata or configuration alongside templates
        - Efficiently managing large collections of templates

        Error Handling:
        - Each JSON file is processed independently, so errors in one file won't
          prevent loading from other files
        - Invalid template formats trigger warnings but don't halt processing
        - JSON parse errors are logged with file information for easier debugging

        Note:
            This method works in conjunction with _load_templates to provide a
            comprehensive template loading system supporting both individual
            .txt files and collections in .json files.
        """
        try:
            template_files = list(Path(self.templates_dir).glob("*.json"))
            logger.info(f"Found {len(template_files)} template files")

            for template_file in template_files:
                try:
                    with open(template_file, "r", encoding="utf-8") as f:
                        templates_data = json.load(f)

                    # Add templates from file
                    for template_name, template_content in templates_data.items():
                        if isinstance(template_content, str):
                            self.templates[template_name] = template_content
                        elif isinstance(template_content, dict) and "text" in template_content:
                            self.templates[template_name] = template_content["text"]
                        else:
                            logger.warning(f"Invalid template format for {template_name}")

                    logger.info(f"Loaded templates from {template_file.name}")
                except Exception as e:
                    logger.error(f"Error loading template file {template_file.name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading templates: {str(e)}")

    def _save_templates(self) -> None:
        """
        Persist all in-memory templates to disk in categorized JSON files.

        This method implements the template persistence strategy, organizing templates
        by category and saving them to appropriate JSON files on disk. It ensures that
        any runtime changes to templates (additions, modifications, or deletions) are
        preserved across application restarts.

        The saving process:
        1. Groups templates into categories based on naming conventions
           - Extracts the category from the template name (prefix before first underscore)
           - Templates without underscores go to the "general" category
        2. For each category:
           - Creates or updates a JSON file named "{category}_templates.json"
           - Writes all templates in that category as a formatted JSON object
           - Uses proper indentation for human readability
        3. Logs detailed information about the save operation

        Template Categorization:
        The method uses a convention-based approach to categorize templates:
        - "rag_query" → Category: "rag", saved to "rag_templates.json"
        - "chat_system" → Category: "chat", saved to "chat_templates.json"
        - "greeting" → Category: "general", saved to "general_templates.json"

        This categorization approach:
        - Keeps related templates together for easier management
        - Avoids a single monolithic file for all templates
        - Makes it easier to locate templates by purpose
        - Reduces the chance of merge conflicts in version control

        Error Handling:
        - The entire save operation is wrapped in exception handling to prevent
          crashes due to file system issues
        - Detailed error information is logged for debugging
        - Even if saving fails, the in-memory templates remain intact

        Note:
            This method is called both directly and asynchronously through
            _async_save_templates to provide both immediate and non-blocking
            persistence options.
        """
        try:
            # Group templates by category
            categorized_templates: Dict[str, Dict[str, Any]] = {}

            for template_name, template_text in self.templates.items():
                # Extract category from template name (before first _)
                parts = template_name.split("_", 1)
                category = parts[0] if len(parts) > 1 else "general"

                if category not in categorized_templates:
                    categorized_templates[category] = {}

                categorized_templates[category][template_name] = template_text

            # Save each category to its own file
            for category, templates in categorized_templates.items():
                file_path = Path(self.templates_dir) / f"{category}_templates.json"

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(templates, f, indent=2)

                logger.info(f"Saved {len(templates)} templates to {file_path.name}")

        except Exception as e:
            logger.error(f"Error saving templates: {str(e)}")

    def get_template(self, template_name: str) -> Optional[str]:
        """
        Retrieve a specific prompt template by its name.

        This method provides access to individual templates stored in the service.
        It performs a simple dictionary lookup, returning the template text if found
        or None if the requested template doesn't exist in the collection.

        The lookup is exact and case-sensitive, with no fuzzy matching or fallback
        behavior. This design ensures predictable template resolution, which is
        important for maintaining consistent prompt behavior in production systems.

        Args:
            template_name: The exact name of the template to retrieve

        Returns:
            The template text as a string if found, None if not found

        Usage Example:
        ```python
        template = prompt_service.get_template("rag_query")
        if template:
            # Template found, use it
            prompt = template.format(query="What is machine learning?")
        else:
            # Template not found, handle the error
            logger.error(f"Template 'rag_query' not found")
            prompt = "Default fallback prompt: {query}"
        ```

        Note:
            Consider checking the return value for None before using the template,
            or use a default template as a fallback to handle missing templates
            gracefully.
        """
        return self.templates.get(template_name)

    def get_all_templates(self) -> Dict[str, str]:
        """
        Retrieve a copy of all available prompt templates.

        This method returns a dictionary containing all templates currently loaded
        in the service, with template names as keys and template texts as values.
        The returned dictionary is a shallow copy of the internal templates collection,
        ensuring that modifications to the returned dictionary won't affect the
        service's template storage.

        Use cases for this method include:
        - Listing available templates in an admin interface
        - Analyzing or processing multiple templates at once
        - Creating a template catalog or documentation
        - Debugging template availability issues

        Returns:
            A dictionary mapping template names to their content

        Usage Example:
        ```python
        all_templates = prompt_service.get_all_templates()

        # Display available templates
        print(f"Available templates ({len(all_templates)}): ")
        for name in sorted(all_templates.keys()):
            print(f" - {name}")

        # Find templates by pattern
        rag_templates = {
            name: content
            for name, content in all_templates.items()
            if name.startswith("rag_")
        }
        ```

        Note:
            While the dictionary is a copy, the template strings themselves
            are not deep-copied. This is generally not an issue since strings
            are immutable in Python.
        """
        return self.templates.copy()

    def register_template(self, template_name: str, template_text: str) -> bool:
        """
        Register a new template or update an existing one in the template collection.

        This method adds a new template to the in-memory template collection or updates
        an existing template if the name already exists. After adding or updating the
        template, it initiates an asynchronous save operation to persist the changes
        to disk, ensuring durability without blocking the calling code.

        The template registration process:
        1. Adds or updates the template in the in-memory dictionary
        2. Schedules an asynchronous task to save all templates to disk
        3. Returns a success indicator

        This method is the primary way to programmatically add or modify templates
        at runtime, enabling dynamic template management without requiring file
        system access or application restarts.

        Template Naming Conventions:
        While not enforced, it's recommended to follow these naming conventions:
        - Use lowercase names with underscores for readability
        - Prefix with category name for organizational purposes (e.g., "rag_query")
        - Use descriptive names that indicate the template's purpose

        Args:
            template_name: Name for the template (used for later retrieval)
            template_text: Content of the template with variable placeholders

        Returns:
            True if the template was successfully registered, False if an error occurred

        Usage Example:
        ```python
        # Register a simple greeting template
        success = prompt_service.register_template(
            "greeting_formal",
            "Dear {title} {last_name},\n\nI hope this message finds you well."
        )

        # Register a more complex template with formatting options
        success = prompt_service.register_template(
            "invoice_summary",
            "Invoice #{invoice_id}\nDate: {date}\nTotal: ${amount:.2f}\n\n{items}"
        )
        ```

        Note:
            This method handles the persistence automatically through an asynchronous
            save operation. The changes are immediately available in memory but may
            take a moment to be written to disk.
        """
        try:
            self.templates[template_name] = template_text

            # Schedule template save
            asyncio.create_task(self._async_save_templates())

            return True
        except Exception as e:
            logger.error(f"Error registering template {template_name}: {str(e)}")
            return False

    async def _async_save_templates(self) -> None:
        """
        Asynchronously persist templates to disk without blocking the main execution flow.

        This method provides a non-blocking way to save templates by delegating to
        the synchronous _save_templates method. It's designed to be called from
        contexts where immediate persistence is desired but blocking operations
        would be problematic, such as during API request handling.

        When called:
        - The method executes _save_templates directly rather than creating a task
        - Despite being async, it doesn't actually perform any async operations
        - This approach simplifies the interface while maintaining consistent
          method signatures

        Usage Context:
        This method is typically called after template modifications to ensure
        changes are persisted, such as after:
        - Registering new templates
        - Updating existing templates
        - Removing templates

        Since saving templates is an I/O-bound operation that involves disk writes,
        this async wrapper helps to:
        - Prevent UI freezing in interactive contexts
        - Avoid blocking the event loop in server contexts
        - Return control quickly to the calling code
        - Ensure template persistence happens reliably in the background

        Note:
            While designed for asynchronous usage, this implementation currently
            performs blocking I/O. In a future optimization, this could be changed
            to use true async file I/O using libraries like aiofiles.
        """
        self._save_templates()

    def remove_template(self, template_name: str) -> bool:
        """
        Remove a template from the collection if it exists.

        This method deletes a template from the in-memory template collection
        and initiates an asynchronous save operation to persist the deletion to disk.
        If the specified template doesn't exist, the method returns False but
        doesn't raise an exception, following a fail-soft approach for easier
        error handling.

        The template removal process:
        1. Checks if the template exists in the collection
        2. If found, removes it from the in-memory dictionary
        3. Schedules an asynchronous task to save the updated template collection
        4. Returns a boolean indicating success or failure

        This method enables runtime management of templates, allowing obsolete
        or incorrect templates to be removed without requiring file system access
        or application restarts.

        Args:
            template_name: Name of the template to remove

        Returns:
            True if the template was found and removed, False if it wasn't found

        Usage Example:
        ```python
        # Check if removal was successful
        if prompt_service.remove_template("outdated_template"):
            logger.info("Template successfully removed")
        else:
            logger.warning("Template not found, nothing to remove")

        # Unconditional removal attempt (ignoring result)
        prompt_service.remove_template("temporary_template")
        ```

        Note:
            The template is immediately removed from memory but the disk
            persistence happens asynchronously. If the application crashes
            immediately after this call, the template might still exist in
            the persisted files when the application restarts.
        """
        if template_name in self.templates:
            del self.templates[template_name]

            # Schedule template save
            asyncio.create_task(self._async_save_templates())

            return True
        return False

    def render_template(self, template_name: str, variables: Dict[str, Any]) -> Optional[str]:
        """
        Render a prompt template by substituting variables into the template text.

        This method performs dynamic template rendering using Python's string formatting
        system. It takes a template by name and a dictionary of variables, substitutes
        the variables into the template placeholders, and returns the fully rendered text
        ready for use with language models or other downstream components.

        The rendering process:
        1. Retrieves the template by name from the template repository
        2. Validates that the template exists
        3. Performs variable substitution using Python's str.format() method
        4. Handles any errors that occur during substitution

        Template Format:
        Templates use Python's string formatting syntax with curly braces:
        - Simple variables: "Hello, {name}!"
        - Nested attributes: "Author: {book.author}"
        - Formatting options: "Score: {score:.2f}"

        Error Handling:
        The method has comprehensive error handling for common issues:
        - Missing templates: Returns None with a warning log
        - Missing variables: Logs the specific missing variable and returns None
        - Format errors: Logs the formatting error and returns None

        Variable handling:
        - All variables must be provided in the variables dictionary
        - Variable types should be compatible with string formatting
        - Complex objects can be used if they have string representations

        Args:
            template_name: Name of the template to render
            variables: Dictionary mapping variable names to their values

        Returns:
            Rendered template text with variables substituted, or None if rendering fails

        Example:
            ```python
            # Define a template
            service.register_template(
                "user_profile",
                "Name: {name}\nAge: {age}\nRole: {role}"
            )

            # Render with variables
            profile = service.render_template(
                "user_profile",
                {"name": "Alice", "age": 30, "role": "Administrator"}
            )
            # Result: "Name: Alice\nAge: 30\nRole: Administrator"
            ```
        """
        template = self.get_template(template_name)
        if not template:
            logger.warning(f"Template {template_name} not found")
            return None

        try:
            return template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in template {template_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {str(e)}")
            return None


# Global instance
_prompt_manager_instance = None
_prompt_manager_lock = threading.Lock()


def get_prompt_manager() -> PromptService:
    """
    Get or create the global thread-safe singleton PromptService instance.

    This function implements a thread-safe singleton pattern for the PromptService,
    ensuring that only one instance is created and shared across the entire application,
    regardless of which thread accesses it. It uses a mutex lock to prevent race conditions
    when multiple threads attempt to create the instance simultaneously.

    The singleton pattern ensures all components throughout the application use the same
    prompt template repository and caching system, providing consistent behavior across
    different threads and request contexts.

    Returns:
        PromptService: The global singleton PromptService instance.

    Example:
        ```python
        # This will always return the same instance, even from different threads
        prompt_manager = get_prompt_manager()
        template = prompt_manager.render_template("greeting", {"name": "User"})
        ```
    """
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        with _prompt_manager_lock:
            if _prompt_manager_instance is None:
                _prompt_manager_instance = PromptService()
    return _prompt_manager_instance


# Example Usage
if __name__ == "__main__":
    from ultimate_mcp_server.utils.logging import setup_logging

    setup_logging(log_level="DEBUG")

    # Create dummy templates dir and file for example
    EXAMPLE_TEMPLATES_DIR = Path("./temp_prompt_templates_example")
    EXAMPLE_TEMPLATES_DIR.mkdir(exist_ok=True)
    (EXAMPLE_TEMPLATES_DIR / "greeting.txt").write_text("Hello, {{name}}! How are you today?")
    (EXAMPLE_TEMPLATES_DIR / "summary.txt").write_text("Summarize the following text:\n\n{{text}}")

    # Set env var to use this temp dir
    os.environ["GATEWAY_PROMPT_TEMPLATES_DIR"] = str(EXAMPLE_TEMPLATES_DIR.resolve())
    os.environ["GATEWAY_FORCE_CONFIG_RELOAD"] = "true"  # Force reload

    try:
        manager = get_prompt_manager()
        print(f"Templates directory: {manager.templates_dir}")
        print(f"Available templates: {manager.list_templates()}")

        greeting_template = manager.get_template("greeting")
        print(f"Greeting Template: {greeting_template}")

        try:
            manager.get_template("non_existent")
        except PromptTemplateError as e:
            print(f"Caught expected error: {e}")

    finally:
        # Clean up
        import shutil

        shutil.rmtree(EXAMPLE_TEMPLATES_DIR)
        print(f"Cleaned up {EXAMPLE_TEMPLATES_DIR}")
        if "GATEWAY_PROMPT_TEMPLATES_DIR" in os.environ:
            del os.environ["GATEWAY_PROMPT_TEMPLATES_DIR"]
        if "GATEWAY_FORCE_CONFIG_RELOAD" in os.environ:
            del os.environ["GATEWAY_FORCE_CONFIG_RELOAD"]
