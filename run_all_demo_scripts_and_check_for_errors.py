#!/usr/bin/env python3
"""
Runs all demo scripts in the 'examples' folder sequentially and checks for errors.

Uses rich for progress tracking and a summary report.
Incorporates specific knowledge about expected outcomes for individual scripts.
"""

import asyncio
import re  # Import regex
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table

# --- Configuration ---
EXAMPLES_DIR = Path(__file__).parent / "examples"
PYTHON_EXECUTABLE = sys.executable  # Use the same Python interpreter that runs this script
OUTPUT_LOG_FILE = Path(__file__).parent / "all_demo_script_console_output_log.txt"

# Scripts to skip (not actual demo scripts or special cases)
SCRIPTS_TO_SKIP = ["sse_client_demo.py", "web_automation_instruction_packs.py", "__init__.py"]

# Strings indicating a critical error in the output (used if no specific allowed patterns)
DEFAULT_ERROR_INDICATORS = ["Traceback (most recent call last):", "CRITICAL"]

# --- Individual Demo Expectations ---
# Define expected outcomes for specific scripts.
# - expected_exit_code: The code the script should exit with (default: 0)
# - allowed_stderr_patterns: List of regex patterns for stderr messages that are OK for this script.
#                            If this list exists, DEFAULT_ERROR_INDICATORS are ignored for stderr.
# - allowed_stdout_patterns: List of regex patterns for stdout messages that are OK (less common).
#                            If this list exists, DEFAULT_ERROR_INDICATORS are ignored for stdout.
DEMO_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    # --- Scripts with specific known patterns ---
    "text_redline_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues - expected when API keys aren't configured
            r"Provider '(openai|anthropic|google)' not available or initialized",
            r"Failed to get provider: No valid OpenAI key found",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "filesystem_operations_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC intentional demo patterns - these test protection features
            r"Protection Triggered! Deletion of \d+ files blocked",  # Specific deletion protection test
            r"Could not set utime for.*?: \[Errno \d+\]",  # Specific file timestamp issue with exact error format
            # Configuration verification messages - specific to demo setup
            r"WARNING: No allowed directories loaded in filesystem configuration",  # Specific verification message
            r"WARNING: Temporary directory .* not found in loaded allowed dirs:",  # Specific verification message
            # OS-specific limitations - with specific reasons
            r"WARNING: Symlink creation might not be supported or permitted on this system",  # Windows-specific limitation
            r"WARNING: Could not create symlink \(.*\): \[Errno \d+\]",  # OS-specific permission error with exact format
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            r"Forcing configuration reload due to GATEWAY_FORCE_CONFIG_RELOAD=true\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
        "allowed_stdout_patterns": [
            # Specific allowed stdout patterns that aren't errors
            r"WARNING: .*",  # Warning messages in stdout
            r"ERROR: .*",  # Error messages in stdout (these are demo outputs, not actual errors)
        ],
    },
    "sql_database_interactions_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC column statistics computation issues - known data type limitation
            r"Could not compute statistics for column customers\.signup_date: 'str' object has no attribute 'isoformat'",  # Specific data type issue
            r"Could not compute statistics for column orders\.order_date: 'str' object has no attribute 'isoformat'",  # Specific data type issue
            # Demo-specific database connection scenarios - intentional examples
            r"Connection failed: \(sqlite3\.OperationalError\) unable to open database file",  # Specific SQLite error format
            r"Failed to connect to database \(sqlite:///.*\): \(sqlite3\.OperationalError\) unable to open database file",  # Specific connection error format
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
        ],
    },
    "rag_example.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC cleanup messages with reasons - intentional error handling
            r"Could not delete collection 'demo_.*?': Collection '.*?' does not exist",  # Non-existent collection during cleanup
            r"Error deleting knowledge base 'demo-kb': Knowledge base 'demo-kb' not found",  # Non-existent KB during cleanup
            r"Error directly deleting vector collection 'demo_.*?': Collection '.*?' does not exist",  # Non-existent collection
            # SPECIFIC provider availability issues - expected when API keys aren't configured
            r"Provider '(openai|anthropic|google)' not available or initialized",  # Missing specific provider
            r"No suitable provider found for embedding generation",  # No embedding provider available
            r"OpenAIError: No API key provided.",  # Specific API key error
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            r"Initializing Gateway: Loading configuration\.\.\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "marqo_fused_search_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC setup/config issues - expected on systems without Marqo
            r"Marqo config file not found at path: .*config/marqo\.json",  # Specific config file path
            r"Error decoding Marqo config file: No JSON object could be decoded",  # Specific JSON parsing error
            r"Exiting demo as Marqo config could not be loaded\.",  # Specific exit message
            # SPECIFIC connection issues - expected on systems without Marqo service
            r"Connection refused: \[Errno 111\] Connection refused",  # Specific connection error with errno
            # SPECIFIC skipping behavior - expected for incomplete setup
            r"Skipping Example \d+: No suitable .* field found in dataset",  # Specific reason for skipping
        ],
    },
    "advanced_vector_search_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues
            r"Provider '(openai|anthropic|google)' not available or initialized",
            r"No suitable provider found for embedding generation",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "vector_search_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC cleanup messages with reasons - intentional cleanup operations
            r"Could not delete collection 'demo_.*?': Collection '.*?' does not exist",  # Non-existent collection during cleanup
            # SPECIFIC provider availability issues - expected when API keys aren't configured
            r"Failed to initialize provider '(openai|anthropic|google)': .*API key.*",  # Specific provider with API key issue
            r"No suitable provider found for embedding generation",  # Specific embedding provider error
            # SPECIFIC demo workflow messages - expected for educational examples
            r"Skipping RAG demo - embedding provider not available",  # Specific reason for skipping demo
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "prompt_templates_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC intentional template demo cases - expected behavior demonstrations
            r"Template 'non_existent_template\.txt' not found in .*/templates/",  # Specific non-existent template
            r"Could not render with missing variables: \['variable_name'\]",  # Specific missing variable demonstration
            # SPECIFIC provider availability messages - expected when API keys aren't configured
            r"No providers available for completion with template",  # Specific provider availability message
            # Standard setup messages - not errors
            r"Initializing Gateway: Loading configuration\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            r"Ultimate MCP Server .* initialized",
            r"Initializing LLM providers",
            r"Configuration not yet loaded\. Loading now\.\.\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "tournament_code_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Intentional demo cases (clean slate testing)
            r"Error reading state file directly",
            r"State file not found at",  # First run of tournament
            r"No functions found in the code",  # Test for empty code
            # Known state handling messages
            r"Cleanup error:",  # Non-critical cleanup issues
            # Provider availability (expected if not configured)
            r"Failed to initialize providers",  # Expected when API keys not configured
            # Initialization logging (not errors)
            r"Gateway initialized",
            r"Initializing Gateway.*",
            # Common setup/config messages (not errors)
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # Formatting patterns (not errors)
            r"─+.*─+",  # Section dividers
            r"INFO.*",  # INFO level log messages
        ],
    },
    "tournament_text_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Intentional demo cases (clean slate testing)
            r"Error reading state file directly",
            r"State file not found at",  # First run of tournament
            # Provider availability (expected if not configured)
            r"Provider .* not available for evaluation",  # Expected when API keys missing
            r"Failed to initialize providers",  # Expected when API keys missing
            # Timeout handling (acceptable on slow CI)
            r"Evaluation with .* timed out",  # Long-running ops may timeout
            # Common setup/config messages (not errors)
            r"Gateway initialized",
            r"Initializing Gateway.*",
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # Formatting patterns (not errors)
            r"─+.*─+",  # Section dividers
            r"INFO.*",  # INFO level log messages
        ],
    },
    "test_code_extraction.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Intentional demo cases (clean slate testing)
            r"Error loading tournament state: .*No such file or directory",  # First run
            r"Failed to load tournament state",  # First run
            r"No round results found",  # Expected for empty state
            # Provider availability (expected if not configured)
            r"Failed to initialize providers",  # Expected if API keys not present
            # Common setup/config messages (not errors)
            r"Initializing Gateway",
            r"Configuration loaded",
            r"Ultimate MCP Server .* initialized",
            r"Initializing LLM providers",
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # Formatting patterns (not errors)
            r"─+.*─+",  # Section dividers
            r"INFO.*",  # INFO level log messages
            r"WARNING.*",  # WARNING level log messages
        ],
    },
    "advanced_extraction_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability (expected if not configured)
            r"Failed to get OpenAI provider",  # Expected if API key not present
            r"Failed to initialize OpenAI provider",  # Expected if API key not present
            # Common setup/config messages (not errors)
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # Formatting patterns (not errors)
            r"─+.*─+",  # Section dividers
        ],
        # Allow the skip message in stdout
        "allowed_stdout_patterns": [
            r"Skipping .* demo - no provider available",
            r"Raw Model Output \(JSON parsing failed\)",
        ],
    },
    "analytics_reporting_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC provider availability issues - expected when API keys aren't configured
            r"Failed to get/initialize provider '(openai|anthropic|google)': .*",  # Specific provider with reason
            r"No providers could be initialized for this demonstration",  # Specific provider initialization message
            r"No default model found for provider '(openai|anthropic|google)'",  # Specific model availability issue
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
            # Logging patterns - not errors
            r"INFO \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamped INFO logs
            # Initialization messages - not errors
            r"Simulating usage with \d+ providers\.",  # Specific simulation statement
        ],
    },
    "basic_completion_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC provider availability issues - expected when API keys aren't configured
            r"Provider '(openai|anthropic|google)' not available or initialized",  # Specific missing provider
            r"All providers failed: No providers available for completion",  # Specific provider failure
            # SPECIFIC demo features - expected component testing
            r"Error with cached completion demo: Cache is disabled",  # Specific cache demo error
            # Standard setup and logging messages - not errors
            r"Initializing Gateway: Loading configuration\.\.\.",
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            r"INFO \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamped INFO logs
            r"Ultimate MCP Server 'basic-completion-demo' initialized",  # Specific initialization message
        ],
    },
    "browser_automation_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Browser automation issues - expected during demos
            r"Could not find search input element with selectors: .*",  # Element not found error
            r"playwright\._impl\._api_types\.TimeoutError: Timeout \d+ms exceeded",  # Timeout error
            r"net::ERR_CONNECTION_REFUSED at .*",  # Connection error
            r"Navigation failed: net::ERR_CONNECTION_REFUSED at .*",  # Navigation error
            r"Execution error in.*: .*",  # General execution errors
            r"Traceback \(most recent call last\):.*",  # Tracebacks from browser automation
            # Provider availability issues
            r"Provider '(openai|anthropic|google)' not available or initialized",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "claude_integration_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC provider availability issues - expected when API keys aren't configured
            r"Provider 'anthropic' not available or initialized",  # Specific Claude provider missing
            r"No suitable Claude model found in available models: \[\]",  # Specific Claude model selection issue
            r"Selected models not found: \['claude-3-opus-20240229', 'claude-3-sonnet-20240229'\]",  # Specific model availability issue
            r"Model 'claude-3-opus-20240229' not available, falling back to default\.",  # Specific fallback behavior
            # Standard setup messages - not errors
            r"Initializing Gateway: Loading configuration\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            r"Ultimate MCP Server 'claude-demo' initialized",
            r"Initializing LLM providers",
            r"Configuration not yet loaded\. Loading now\.\.\.",
            # Logging patterns - not errors
            r"INFO \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamped INFO logs
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "compare_synthesize_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC provider availability issues - expected when API keys aren't configured
            r"Failed to initialize providers: No providers available",  # Specific provider initialization message
            # SPECIFIC tool registration messages - expected behavior for specialized tools
            r"compare_and_synthesize tool FAILED to register: Tool 'compare_and_synthesize' requires 2\+ providers",  # Specific registration failure reason
            # Standard setup messages - not errors
            r"Initializing Gateway: Loading configuration\.\.\.",
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            r"INFO \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamped INFO logs
            r"Ultimate MCP Server 'compare-synthesize-demo-v2' initialized",  # Specific initialization message
        ],
    },
    "cost_optimization.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC provider availability issues - expected when API keys aren't configured
            r"API key for provider '(openai|anthropic|google)' not found",  # Specific API key missing message
            r"Could not determine provider for model '.*?'",  # Specific model-provider mapping issue
            r"No models met criteria: max_cost=\$\d+\.\d+, .*",  # Specific criteria filtering result
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
            # Logging patterns - not errors
            r"INFO \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamped INFO logs
        ],
    },
    "document_processing.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC initialization messages - expected setup steps
            r"Clearing cache before demonstration\.\.\.",  # Specific cache operation
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
            # Logging patterns - not errors
            r"INFO \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamped INFO logs
        ],
    },
    "multi_provider_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC provider availability issues - expected when API keys aren't configured
            r"Provider '(openai|anthropic|google)' not available or initialized",  # Specific provider not available
            r"All providers failed: \['(openai|anthropic|google)'.*?\]",  # Specific list of failed providers
            # Standard setup messages - not errors
            r"Initializing Gateway: Loading configuration\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            r"Ultimate MCP Server 'multi-provider-demo' initialized",
            r"Initializing LLM providers",
            r"Configuration not yet loaded\. Loading now\.\.\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
            # Logging patterns - not errors
            r"INFO \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamped INFO logs
        ],
    },
    "simple_completion_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC provider availability issues - expected when API keys aren't configured
            r"Provider '(openai|anthropic|google)' not available",  # Specific provider not available
            # Standard setup messages - not errors
            r"Initializing Gateway: Loading configuration\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            r"Ultimate MCP Server 'simple-demo' initialized",
            r"Initializing LLM providers",
            r"Configuration not yet loaded\. Loading now\.\.\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
            # Logging patterns - not errors
            r"INFO \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamped INFO logs
        ],
    },
    "workflow_delegation_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC provider availability messages - expected initialization info
            r"Some API keys missing: \['(openai|anthropic|google)'.*?\]",  # Specific API keys warning
            r"Provider '(openai|anthropic|google)' not available",  # Specific provider not available
            r"Failed to initialize provider: Invalid API key or provider configuration",  # Specific initialization error
            # SPECIFIC initialization messages - expected setup steps
            r"Initializing required providers for delegation demo",  # Specific initialization message
            r"All required API keys seem to be present",  # Specific configuration check
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
            # Logging patterns - not errors
            r"INFO \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamped INFO logs
        ],
    },
    "cache_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # SPECIFIC operational messages - expected configuration info
            r"Cache is disabled \(GATEWAY__CACHE__ENABLED=false\)",  # Specific cache configuration message
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "audio_transcription_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues
            r"Provider '(openai|anthropic|google)' not available or initialized",
            r"Failed to initialize OpenAI provider: Invalid API key",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "entity_relation_graph_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues
            r"Provider '(openai|anthropic|google)' not available or initialized",
            r"Skipping provider initialization as no API keys are available",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "grok_integration_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues
            r"Provider 'grok' not available or initialized",
            r"No API key found for Grok",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "html_to_markdown_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues
            r"Provider '(openai|anthropic|google)' not available or initialized",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "measure_model_speeds.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues
            r"Provider '(openai|anthropic|google|grok|meta)' not available or initialized",
            r"No providers could be initialized",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "meta_api_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues
            r"Provider 'meta' not available or initialized",
            r"No API key found for Meta",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "research_workflow_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues
            r"Provider '(openai|anthropic|google)' not available or initialized",
            # Search and web access related messages
            r"Failed to perform web search: .*",
            r"Web search failed: .*",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
    "text_classification_demo.py": {
        "expected_exit_code": 0,
        "allowed_stderr_patterns": [
            # Provider availability issues
            r"Provider '(openai|anthropic|google)' not available or initialized",
            # Standard setup messages - not errors
            r"Configuration not yet loaded\. Loading now\.\.\.",
            r"Configuration loaded and environment variables applied via decouple\.",
            # UI formatting patterns - not errors
            r"─+.*─+",  # Section dividers
        ],
    },
}

console = Console()


def find_demo_scripts() -> List[Path]:
    """Find all Python demo scripts in the examples directory."""
    if not EXAMPLES_DIR.is_dir():
        console.print(
            f"[bold red]Error:[/bold red] Examples directory not found at '{EXAMPLES_DIR}'"
        )
        return []

    scripts = sorted(
        [p for p in EXAMPLES_DIR.glob("*.py") if p.is_file() and p.name not in SCRIPTS_TO_SKIP]
    )
    return scripts


async def run_script(script_path: Path) -> Tuple[int, str, str]:
    """
    Run a single Python script as a subprocess and capture its output.

    This async function executes a Python script in a separate process using the same
    Python interpreter that's running this script. It captures both standard output
    and standard error streams, as well as the exit code of the process.

    The function uses asyncio.create_subprocess_exec for non-blocking execution,
    allowing multiple scripts to be run concurrently if needed, although the current
    implementation runs them sequentially.

    Args:
        script_path (Path): The path to the Python script to be executed.
            This should be a fully resolved path object pointing to a valid Python file.

    Returns:
        Tuple[int, str, str]: A tuple containing:
            - exit_code (int): The return code of the process (0 typically means success)
            - stdout (str): The captured standard output as a string, with encoding errors ignored
            - stderr (str): The captured standard error as a string, with encoding errors ignored

    Note:
        - The function waits for the script to complete before returning
        - Any encoding errors in stdout/stderr are ignored during decoding
        - The script is executed with the same Python interpreter as the parent process
        - No environment variables or arguments are passed to the script
    """
    command = [PYTHON_EXECUTABLE, str(script_path)]

    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    exit_code = process.returncode

    return exit_code, stdout.decode(errors="ignore"), stderr.decode(errors="ignore")


def check_for_errors(
    script_name: str, exit_code: int, stdout: str, stderr: str
) -> Tuple[bool, str]:
    """
    Check script output against predefined expectations to determine success or failure.

    This function analyzes the execution results of a demo script and determines if it
    succeeded or failed based on:
    1. Comparing the actual exit code against the expected exit code for the script
    2. Checking for unexpected error messages in stdout and stderr
    3. Applying script-specific patterns for allowed errors and warnings

    The function uses the DEMO_EXPECTATIONS dictionary to get script-specific expectations
    including allowed error patterns. For scripts without specific expectations defined,
    it applies default success criteria (exit code 0 and no critical error indicators).

    The function handles two types of patterns for allowed output:
    - allowed_stderr_patterns: Regex patterns for permitted messages in stderr
    - allowed_stdout_patterns: Regex patterns for permitted messages in stdout

    Args:
        script_name (str): Name of the script being checked (used to lookup expectations)
        exit_code (int): The actual exit code returned by the script
        stdout (str): The captured standard output from the script
        stderr (str): The captured standard error from the script

    Returns:
        Tuple[bool, str]: A tuple containing:
            - success (bool): True if the script execution meets all success criteria
            - reason (str): A descriptive message explaining the result
                            "Success" for successful executions
                            Error details for failed executions

    Note:
        - Log messages at INFO, DEBUG, and WARNING levels are generally ignored
          unless they match critical error patterns
        - Script-specific allowed patterns take precedence over default error indicators
        - If no script-specific expectations exist, only the DEFAULT_ERROR_INDICATORS
          are used to check for problems
    """

    expectations = DEMO_EXPECTATIONS.get(script_name, {})
    expected_exit_code = expectations.get("expected_exit_code", 0)
    allowed_stderr_patterns = expectations.get("allowed_stderr_patterns", [])
    allowed_stdout_patterns = expectations.get("allowed_stdout_patterns", [])

    # 1. Check Exit Code
    if exit_code != expected_exit_code:
        return False, f"Exited with code {exit_code} (expected {expected_exit_code})"

    # --- Refined Error Log Checking ---

    def find_unexpected_lines(
        output: str, allowed_patterns: List[str], default_indicators: List[str]
    ) -> List[str]:
        """
        Find lines in script output that indicate errors or unexpected behavior.

        This function analyzes the output of a script (either stdout or stderr) and
        identifies lines that may indicate an error or unexpected behavior. It handles
        two different checking modes:

        1. With allowed_patterns: All lines that don't match at least one of the allowed
           patterns are considered unexpected.
        2. Without allowed_patterns: Only lines containing any of the default_indicators
           are considered unexpected.

        The first mode is more restrictive (whitelist approach) while the second is
        more permissive (blacklist approach). The function chooses the appropriate mode
        based on whether allowed_patterns is provided.

        Args:
            output (str): The script output to analyze (either stdout or stderr)
            allowed_patterns (List[str]): List of regex patterns for allowed output lines.
                If provided, any line not matching at least one pattern is unexpected.
            default_indicators (List[str]): List of string indicators of critical errors.
                Only used when allowed_patterns is empty, to identify error lines.

        Returns:
            List[str]: A list of lines from the output that are considered unexpected or
                      indicative of errors. Empty list means no unexpected lines found.

        Note:
            - Empty lines are always ignored
            - When allowed_patterns is provided, the function uses a whitelist approach
            - When allowed_patterns is empty, the function uses a blacklist approach
            - Regex matching is used for allowed_patterns, simple substring matching for default_indicators
        """
        lines = output.strip().splitlines()
        unexpected_lines = []
        for line in lines:
            line_content = line.strip()
            if not line_content:  # Skip blank lines
                continue

            is_allowed = False
            # Check against specific allowed patterns for this script
            if allowed_patterns:
                for pattern in allowed_patterns:
                    if re.search(pattern, line_content):
                        is_allowed = True
                        break

            # If specific patterns were defined and line wasn't allowed, it's unexpected
            if allowed_patterns and not is_allowed:
                unexpected_lines.append(line)
            # If no specific patterns were defined, check against default critical indicators only
            elif not allowed_patterns:
                for indicator in default_indicators:
                    if indicator in line_content:  # Use 'in' for default indicators for simplicity
                        unexpected_lines.append(line)
                        break  # Found a default indicator, no need to check others for this line

        return unexpected_lines

    unexpected_stderr = find_unexpected_lines(
        stderr, allowed_stderr_patterns, DEFAULT_ERROR_INDICATORS
    )
    unexpected_stdout = find_unexpected_lines(
        stdout, allowed_stdout_patterns, DEFAULT_ERROR_INDICATORS
    )

    # Filter out lines that are just INFO/DEBUG/WARNING level logs unless they are explicitly disallowed
    # (This assumes default log format: YYYY-MM-DD HH:MM:SS] LEVEL ...) or rich format
    def is_ignorable_log(line: str) -> bool:
        """
        Determine if a log line can be safely ignored for error detection.

        This function identifies standard INFO, DEBUG, and WARNING level log messages
        that should typically be ignored when checking for errors, unless they are
        explicitly flagged as problematic by other patterns.

        The function recognizes common log line formats:
        - Standard timestamp-prefixed format: [YYYY-MM-DD HH:MM:SS] LEVEL message
        - Simple level-prefixed format: LEVEL message

        Args:
            line (str): The log line to analyze

        Returns:
            bool: True if the line appears to be a standard INFO, DEBUG, or WARNING
                 log message that can be safely ignored. False otherwise.

        Note:
            This function only identifies the format of standard log lines;
            it doesn't analyze the content of the messages themselves.
        """
        line_lower = line.lower()  # noqa: F841
        return re.match(
            r"^\[\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\]\s+(INFO|DEBUG|WARNING)\s+", line.strip()
        ) or re.match(r"^\s*(INFO|DEBUG|WARNING)\s+", line.strip())

    actual_stderr_errors = [line for line in unexpected_stderr if not is_ignorable_log(line)]
    actual_stdout_errors = [line for line in unexpected_stdout if not is_ignorable_log(line)]

    if actual_stderr_errors:
        return False, f"Unexpected errors found in stderr: ...{escape(actual_stderr_errors[0])}..."

    if actual_stdout_errors:
        return False, f"Unexpected errors found in stdout: ...{escape(actual_stdout_errors[0])}..."
    # --- End Refined Error Log Checking ---

    # If exit code matches and no unexpected critical errors found
    return True, "Success"


def write_script_output_to_log(
    script_name: str, exit_code: int, stdout: str, stderr: str, is_success: bool
):
    """
    Write the complete output of a script run to the consolidated log file.

    This function appends the execution results of a single script to a consolidated
    log file for record-keeping and debugging purposes. The log includes:
    - A header with the script name, exit code, and success/failure status
    - The complete stdout output captured during execution
    - The complete stderr output captured during execution

    Each script's log entry is clearly separated with delimiters for easy navigation
    and searching within the log file.

    Args:
        script_name (str): Name of the script that was executed
        exit_code (int): The exit code returned by the script
        stdout (str): The complete standard output captured during execution
        stderr (str): The complete standard error captured during execution
        is_success (bool): Whether the script execution was considered successful
                         according to the check_for_errors criteria

    Returns:
        None: The function writes to the log file specified by OUTPUT_LOG_FILE
             but doesn't return any value

    Note:
        - The function appends to the log file, preserving previous entries
        - If stdout or stderr is empty, a placeholder message is logged
        - No limit is placed on the size of the logged output
    """
    with open(OUTPUT_LOG_FILE, "a", encoding="utf-8") as log_file:
        # Write script header with result
        log_file.write(f"\n{'=' * 80}\n")
        status = "SUCCESS" if is_success else "FAILURE"
        log_file.write(f"SCRIPT: {script_name} - EXIT CODE: {exit_code} - STATUS: {status}\n")
        log_file.write(f"{'-' * 80}\n\n")

        # Write stdout
        log_file.write("STDOUT:\n")
        log_file.write(stdout if stdout.strip() else "(No stdout)\n")
        log_file.write("\n")

        # Write stderr
        log_file.write("STDERR:\n")
        log_file.write(stderr if stderr.strip() else "(No stderr)\n")
        log_file.write("\n")


async def main():
    """
    Main function to run all demo scripts and generate a comprehensive report.

    This async function coordinates the entire process of:
    1. Finding all demo scripts in the examples directory
    2. Running each script sequentially and capturing its output
    3. Checking each script's result against expected behavior
    4. Logging detailed output to a consolidated log file
    5. Generating a rich, interactive summary report in the console

    The function implements a progress bar display using rich.progress to provide
    real-time feedback during execution. After all scripts have run, it displays
    a detailed table summarizing the results of each script, including status,
    exit code, and relevant output snippets.

    The function follows these specific steps:
    - Locate Python scripts in the examples directory (skipping certain files)
    - Initialize/clear the consolidated log file
    - Run each script in sequence, updating the progress bar
    - Check each script's output against expectations
    - Write detailed output for each script to the log file
    - Generate and display a summary table with success/failure indicators
    - Display final counts of succeeded and failed scripts

    Returns:
        int: Exit code for the parent process:
            - 0 if all scripts succeed
            - 1 if any script fails or if no scripts are found

    Note:
        - Scripts listed in SCRIPTS_TO_SKIP are excluded from execution
        - The function creates a new consolidated log file each time it runs
        - Progress information is displayed using a rich progress bar
        - The summary table highlights both successful and failed scripts
    """
    console.print(Rule("[bold blue]Running All Example Scripts[/bold blue]"))

    scripts = find_demo_scripts()
    if not scripts:
        console.print("[yellow]No demo scripts found to run.[/yellow]")
        return 1

    console.print(f"Found {len(scripts)} demo scripts in '{EXAMPLES_DIR}'.")

    # Initialize/clear the output log file
    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as log_file:
        log_file.write("DEMO SCRIPT CONSOLE OUTPUT LOG\n")
        log_file.write(f"Generated by {Path(__file__).name}\n")
        log_file.write(f"{'=' * 80}\n\n")

    results = []
    success_count = 0
    fail_count = 0

    # --- Progress Bar Setup ---
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
        transient=False,  # Keep progress bar visible after completion
    )

    task_id = progress.add_task("[cyan]Running scripts...", total=len(scripts))

    with Live(progress, console=console, vertical_overflow="visible"):
        for script in scripts:
            script_name = script.name
            progress.update(task_id, description=f"[cyan]Running {script_name}...")

            exit_code, stdout, stderr = await run_script(script)
            is_success, reason = check_for_errors(script_name, exit_code, stdout, stderr)

            # Log all output to the consolidated log file
            write_script_output_to_log(script_name, exit_code, stdout, stderr, is_success)

            results.append(
                {
                    "script": script_name,
                    "success": is_success,
                    "reason": reason,
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )

            if is_success:
                success_count += 1
            else:
                fail_count += 1

            progress.update(task_id, advance=1)

        progress.update(task_id, description="[bold green]All scripts finished![/bold green]")
        await asyncio.sleep(0.5)  # Allow final update to render

    # --- Summary Report ---
    console.print(Rule("[bold blue]Demo Run Summary[/bold blue]"))

    summary_table = Table(
        title="Script Execution Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Script Name", style="cyan", no_wrap=True)
    summary_table.add_column("Status", style="white")
    summary_table.add_column("Exit Code", style="yellow", justify="right")
    summary_table.add_column("Reason / Output Snippet", style="white")

    for result in results:
        status_icon = (
            "[green]✅ SUCCESS[/green]" if result["success"] else "[bold red]❌ FAILURE[/bold red]"
        )
        reason_or_output = result["reason"]

        # --- Enhanced Snippet Logic ---
        # Prioritize showing snippet related to the failure reason
        if not result["success"]:
            output_to_search = result["stderr"] + result["stdout"]  # Combined output
            snippet = ""

            # If failure is due to unexpected error message
            if "Unexpected errors found" in reason_or_output:
                # Extract the specific error shown in the reason
                match = re.search(
                    r"Unexpected errors found in (stdout|stderr): \.\.\.(.*)\.\.\.\"?",
                    reason_or_output,
                )
                if match:
                    error_snippet_text = match.group(2).strip()
                    # Try to find this snippet in the actual output
                    start_idx = output_to_search.find(error_snippet_text)
                    if start_idx != -1:
                        # Find the start of the line containing the snippet
                        line_start_idx = output_to_search.rfind("\n", 0, start_idx) + 1
                        lines_around_error = output_to_search[line_start_idx:].splitlines()
                        snippet = "\n".join(lines_around_error[:5])  # Show 5 lines from error
                        if len(lines_around_error) > 5:
                            snippet += "\n..."

            # If failure is due to exit code, show end of stderr/stdout
            elif "Exited with code" in reason_or_output:
                if result["stderr"].strip():
                    lines = result["stderr"].strip().splitlines()
                    snippet = "\n".join(lines[-5:])  # Last 5 lines of stderr
                elif result["stdout"].strip():
                    lines = result["stdout"].strip().splitlines()
                    snippet = "\n".join(lines[-5:])  # Last 5 lines of stdout

            # Fallback if no specific snippet found yet for failure
            if not snippet:
                lines = output_to_search.strip().splitlines()
                snippet = "\n".join(lines[-5:])  # Last 5 lines overall

            if snippet:
                reason_or_output += f"\n---\n[dim]{escape(snippet)}[/dim]"

        elif result["success"]:
            # Show last few lines of stdout for successful runs
            lines = result["stdout"].strip().splitlines()
            if lines:
                snippet = "\n".join(lines[-3:])  # Show last 3 lines
                reason_or_output += f"\n---\n[dim]{escape(snippet)}[/dim]"
            else:  # Handle case with no stdout
                reason_or_output += "\n---\n[dim](No stdout produced)[/dim]"
        # --- End Enhanced Snippet Logic ---

        summary_table.add_row(
            result["script"], status_icon, str(result["exit_code"]), reason_or_output
        )

    console.print(summary_table)

    # --- Final Count ---
    console.print(Rule())
    total_scripts = len(scripts)
    final_message = f"[bold green]{success_count}[/bold green] succeeded, [bold red]{fail_count}[/bold red] failed out of {total_scripts} scripts."
    final_color = "green" if fail_count == 0 else "red"
    console.print(Panel(final_message, border_style=final_color, expand=False))

    console.print(f"\nComplete output log saved to: [cyan]{OUTPUT_LOG_FILE}[/cyan]")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
