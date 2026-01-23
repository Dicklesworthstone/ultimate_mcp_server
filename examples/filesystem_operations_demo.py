#!/usr/bin/env python
"""Filesystem operations demo for Ultimate MCP Server Tools.

This example demonstrates the secure asynchronous filesystem operations tools,
covering file/directory manipulation, searching, metadata retrieval, and
security features like allowed directory restrictions and deletion protection.
"""

import argparse
import asyncio
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

# --- Configuration --- (Standard libs only here)
# Add project root to path for imports when running as script
# Adjust this path if your script location relative to the project root differs
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if not (PROJECT_ROOT / "ultimate").is_dir():
        # Fallback if running from a different structure
        PROJECT_ROOT = Path(__file__).resolve().parent
        if not (PROJECT_ROOT / "ultimate").is_dir():
            print(
                "Error: Could not reliably determine project root. Make sure ultimate is importable.",
                file=sys.stderr,
            )
            sys.exit(1)
    sys.path.insert(0, str(PROJECT_ROOT))

    # --- Important: Set Environment Variables FIRST ---
    DEMO_TEMP_DIR = tempfile.mkdtemp(prefix="ultimate_fs_demo_")
    os.environ["FILESYSTEM__ALLOWED_DIRECTORIES"] = json.dumps([DEMO_TEMP_DIR])
    os.environ["GATEWAY_FILESYSTEM_ALLOWED_DIRECTORIES"] = json.dumps([DEMO_TEMP_DIR])
    os.environ["GATEWAY_FORCE_CONFIG_RELOAD"] = "true"

    print(f"INFO: Temporarily allowing access to: {DEMO_TEMP_DIR}")
    print("DEBUG: Environment variables set:")
    print(f"  FILESYSTEM__ALLOWED_DIRECTORIES = {os.environ['FILESYSTEM__ALLOWED_DIRECTORIES']}")
    print(
        f"  GATEWAY_FILESYSTEM_ALLOWED_DIRECTORIES = {os.environ['GATEWAY_FILESYSTEM_ALLOWED_DIRECTORIES']}"
    )
except Exception as e:
    print(f"Error during initial setup: {e}", file=sys.stderr)
    sys.exit(1)

# --- Defer ALL ultimate imports until AFTER env vars are set ---
# Import Rich components (can happen earlier, but keep grouped for clarity)
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule

from ultimate_mcp_server.config import get_config

# Import necessary exceptions
# Filesystem Tools
from ultimate_mcp_server.tools.filesystem import (
    create_directory,
    delete_path,
    directory_tree,
    edit_file,
    get_file_info,
    list_allowed_directories,
    list_directory,
    move_file,
    read_file,
    read_multiple_files,
    search_files,
    write_file,
)
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import generate_rich_directory_tree, safe_tool_call

# Shared console and display utils
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger AFTER all relevant imports
logger = get_logger("example.filesystem")


def parse_arguments():
    """Parse command line arguments for the demo."""
    parser = argparse.ArgumentParser(
        description="Filesystem Operations Demo for Ultimate MCP Server Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Available demos:
  all           - Run all demos (default)
  read          - File reading operations
  write         - File writing and editing operations
  directory     - Directory operations (create, list, tree)
  move_delete   - Move, delete, search & info operations
  security      - Security features demo
""",
    )

    parser.add_argument(
        "demo",
        nargs="?",
        default="all",
        choices=["all", "read", "write", "directory", "move_delete", "security"],
        help="Specific demo to run (default: all)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")

    parser.add_argument(
        "--rich-tree",
        action="store_true",
        help="Use enhanced rich tree visualization for directory trees",
    )

    return parser.parse_args()


# --- Verify Configuration Loading ---
def verify_config():
    """Verify that the filesystem configuration has loaded correctly."""
    try:
        # Get config only ONCE
        config = get_config()
        fs_config = config.filesystem
        allowed_dirs = fs_config.allowed_directories

        print("Configuration verification:")
        print(f"  Allowed directories: {allowed_dirs}")

        if not allowed_dirs:
            print("WARNING: No allowed directories loaded in filesystem configuration!")
            print("Check these environment variables:")
            for key in os.environ:
                if "ALLOWED_DIRECTORIES" in key:
                    print(f"  {key} = {os.environ[key]}")
            print(f"DEMO_TEMP_DIR set to: {DEMO_TEMP_DIR}")
            # Do NOT attempt to force update - rely on initial load
            print(
                "ERROR: Configuration failed to load allowed_directories from environment variables."
            )
            return False  # Fail verification if dirs are missing

        # If allowed_dirs were loaded, check if our temp dir is in it
        if DEMO_TEMP_DIR in allowed_dirs:
            print(f"SUCCESS: Temporary directory {DEMO_TEMP_DIR} properly loaded in configuration!")
            return True
        else:
            print(
                f"WARNING: Temporary directory {DEMO_TEMP_DIR} not found in loaded allowed dirs: {allowed_dirs}"
            )
            return False  # Fail verification if temp dir is missing

    except Exception as e:
        print(f"ERROR during config verification: {e}")
        import traceback

        traceback.print_exc()
        return False


# --- Demo Setup ---
# DEMO_ROOT is the base *within* the allowed temporary directory
DEMO_ROOT = Path(DEMO_TEMP_DIR) / "demo_project"
BULK_FILES_COUNT = 110  # Number of files to create for deletion protection demo (>100)


async def setup_demo_environment():
    """Create a temporary directory structure for the demo."""
    logger.info("Setting up demo environment...", emoji_key="setup")
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    project_dirs = [
        DEMO_ROOT / "docs",
        DEMO_ROOT / "src" / "utils",
        DEMO_ROOT / "data",
        DEMO_ROOT / "config",
        DEMO_ROOT / "tests",
        DEMO_ROOT / ".hidden_dir",
        DEMO_ROOT / "bulk_files",  # For deletion protection demo
    ]
    for directory in project_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    # Create some sample files
    sample_files = {
        DEMO_ROOT / "README.md": """# Project Demo

This is a demonstration project for testing the secure filesystem operations.

## Features

- File reading and writing
- Directory manipulation
- File searching capabilities
- Metadata retrieval

## Security

All operations are restricted to allowed directories for safety.""",
        DEMO_ROOT / "src" / "main.py": """#!/usr/bin/env python
'''Main entry point for the demo application.'''
import sys
from pathlib import Path
# Import logger for edit demo
# Assume logger is configured elsewhere
# import logging
# logger = logging.getLogger(__name__)

# A line with different whitespace for editing demo
def main():
	'''Main function to run the application.'''
	print("Hello from the demo application!")

	# Get configuration
	config = get_config_local() # Renamed to avoid conflict
	print(f"Running with debug mode: {config['debug']}")

	return 0

def get_config_local(): # Renamed
    '''Get application configuration.'''
    return {
        "debug": True,
        "log_level": "INFO",
        "max_connections": 10
    }

if __name__ == "__main__":
    sys.exit(main())
""",
        DEMO_ROOT / "src" / "utils" / "helpers.py": """'''Helper utilities for the application.'''

def format_message(message, level="info"):
    '''Format a message with level prefix.'''
    return f"[{level.upper()}] {message}"

class DataProcessor:
    '''Process application data.'''

    def __init__(self, data_source):
        self.data_source = data_source

    def process(self):
        '''Process the data.'''
        # TODO: Implement actual processing
        return f"Processed {self.data_source}"
""",
        DEMO_ROOT / "docs" / "api.md": """# API Documentation

## Endpoints

### GET /api/v1/status

Returns the current system status.

### POST /api/v1/data

Submit data for processing.

## Authentication

All API calls require an authorization token.
""",
        DEMO_ROOT / "config" / "settings.json": """{
    "appName": "Demo Application",
    "version": "1.0.0",
    "debug": false,
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "demo_db"
    },
    "logging": {
        "level": "info",
        "file": "app.log"
    }
}""",
        DEMO_ROOT / "data" / "sample.csv": "ID,Value,Category\n1,10.5,A\n2,15.2,B\n3,9.8,A",
        DEMO_ROOT / "tests" / "test_helpers.py": """import pytest
# Adjust import path if needed relative to test execution
from src.utils.helpers import format_message

def test_format_message():
    assert format_message("Test", "debug") == "[DEBUG] Test"
""",
        DEMO_ROOT / ".gitignore": "*.log\n*.tmp\n.hidden_dir/\n",
        DEMO_ROOT / "temp.log": "Log file content - should be excluded by search patterns.",
        # Add a file with potentially non-UTF8 data (simulated)
        DEMO_ROOT
        / "data"
        / "binary_data.bin": b"\x80\x02\x95\n\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x04data\x94\x8c\x06binary\x94s.",
    }

    for file_path, content in sample_files.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            file_path.write_text(content, encoding="utf-8")
        else:
            file_path.write_bytes(content)

    # Create bulk files for deletion protection test
    bulk_dir = DEMO_ROOT / "bulk_files"
    bulk_dir.mkdir(exist_ok=True)  # Ensure bulk dir exists

    # Create files with deliberately varied timestamps to trigger protection
    current_time = time.time()
    file_types = [".txt", ".log", ".dat", ".csv", ".tmp", ".bak", ".json"]

    for i in range(BULK_FILES_COUNT):
        # Use a wider variety of extensions
        ext = file_types[i % len(file_types)]
        fpath = bulk_dir / f"file_{i:03d}{ext}"
        fpath.write_text(f"Content for file {i}")

        # Create highly varied timestamps spanning days/weeks, not just minutes
        # Some files very old, some very new, to ensure high standard deviation
        if i < BULK_FILES_COUNT // 3:
            # First third: older files (30-60 days old)
            age = 60 * 60 * 24 * (30 + (i % 30))  # 30-60 days in seconds
        elif i < 2 * (BULK_FILES_COUNT // 3):
            # Middle third: medium age (1-10 days old)
            age = 60 * 60 * 24 * (1 + (i % 10))  # 1-10 days in seconds
        else:
            # Final third: very recent (0-12 hours old)
            age = 60 * 60 * (i % 12)  # 0-12 hours in seconds

        # Set both access and modification times to the calculated age
        try:
            timestamp = current_time - age
            os.utime(fpath, (timestamp, timestamp))
        except OSError as e:
            logger.warning(f"Could not set utime for {fpath}: {e}", emoji_key="warning")

    # Add a message about the setup
    logger.info(
        f"Created {BULK_FILES_COUNT} files in 'bulk_files/' with highly varied timestamps and {len(file_types)} different extensions",
        emoji_key="setup",
    )

    # Create a symlink (if supported)
    SYMLINK_PATH = DEMO_ROOT / "link_to_src"
    TARGET_PATH = DEMO_ROOT / "src"  # Link to src directory
    try:
        # Check if symlinks are supported (e.g., Windows needs admin rights or dev mode)
        can_symlink = hasattr(os, "symlink")
        test_link_path = DEMO_ROOT / "test_link_nul_delete"
        if platform.system() == "Windows":
            # Basic check, might not be perfect
            try:
                # Use a file target for test link on Windows if dir links need special perms
                test_target = DEMO_ROOT / "README.md"
                os.symlink(test_target, test_link_path, target_is_directory=False)
                test_link_path.unlink()  # Clean up test link
            except (OSError, AttributeError, NotImplementedError):
                can_symlink = False
                logger.warning(
                    "Symlink creation might not be supported or permitted on this system. Skipping symlink tests.",
                    emoji_key="warning",
                )

        if can_symlink:
            # Ensure target exists before creating link
            if TARGET_PATH.is_dir():
                # Use await aiofiles.os.symlink for consistency? No, os.symlink is sync only.
                os.symlink(TARGET_PATH, SYMLINK_PATH, target_is_directory=True)
                logger.info(f"Created symlink: {SYMLINK_PATH} -> {TARGET_PATH}", emoji_key="link")
            else:
                logger.warning(
                    f"Symlink target {TARGET_PATH} does not exist or is not a directory. Skipping symlink creation.",
                    emoji_key="warning",
                )
                SYMLINK_PATH = None
        else:
            SYMLINK_PATH = None  # Indicate symlink wasn't created
    except OSError as e:
        # Handle errors like EEXIST if link already exists, or permission errors
        if e.errno == 17:  # EEXIST
            logger.warning(
                f"Symlink {SYMLINK_PATH} already exists. Assuming correct setup.",
                emoji_key="warning",
            )
        else:
            logger.warning(
                f"Could not create symlink ({SYMLINK_PATH} -> {TARGET_PATH}): {e}. Skipping symlink tests.",
                emoji_key="warning",
            )
            SYMLINK_PATH = None  # Indicate symlink wasn't created
    except Exception as e:
        logger.error(f"Unexpected error creating symlink: {e}", emoji_key="error", exc_info=True)
        SYMLINK_PATH = None

    logger.success(f"Demo environment set up at: {DEMO_ROOT}", emoji_key="success")
    console.print(
        Panel(
            f"Created demo project within [cyan]{DEMO_ROOT.parent}[/cyan] at [cyan]{DEMO_ROOT.name}[/cyan]\n"
            f"Created [bold]{len(project_dirs)}[/bold] directories and [bold]{len(sample_files)}[/bold] sample files.\n"
            f"Created [bold]{BULK_FILES_COUNT}[/bold] files in 'bulk_files/' for deletion test.\n"
            f"Symlink created: {'Yes' if SYMLINK_PATH else 'No'}",
            title="Demo Environment Ready",
            border_style="green",
            expand=False,
        )
    )
    return SYMLINK_PATH


async def cleanup_demo_environment():
    """Remove the temporary directory structure using standard shutil."""
    global DEMO_TEMP_DIR
    if DEMO_TEMP_DIR and Path(DEMO_TEMP_DIR).exists():
        try:
            # Use synchronous shutil for cleanup simplicity
            shutil.rmtree(DEMO_TEMP_DIR)
            logger.info(f"Cleaned up demo directory: {DEMO_TEMP_DIR}", emoji_key="cleanup")
            console.print(f"Cleaned up demo directory: [dim]{DEMO_TEMP_DIR}[/dim]")
        except Exception as e:
            logger.error(f"Error during cleanup of {DEMO_TEMP_DIR}: {e}", emoji_key="error")
            console.print(
                f"[bold red]Error cleaning up demo directory {DEMO_TEMP_DIR}: {e}[/bold red]"
            )
    DEMO_TEMP_DIR = None


async def demonstrate_file_reading(symlink_path):
    """Demonstrate file reading operations."""
    console.print(Rule("[bold cyan]1. File Reading Operations[/bold cyan]", style="cyan"))
    logger.info("Demonstrating file reading operations...", emoji_key="file")

    # --- Read Single File (Text) ---
    readme_path = str(DEMO_ROOT / "README.md")
    await safe_tool_call(
        read_file, {"path": readme_path}, description="Reading a text file (README.md)"
    )

    # --- Read Single File (JSON) ---
    settings_path = str(DEMO_ROOT / "config" / "settings.json")
    await safe_tool_call(
        read_file, {"path": settings_path}, description="Reading a JSON file (settings.json)"
    )

    # --- Read Single File (Simulated Binary) ---
    binary_path = str(DEMO_ROOT / "data" / "binary_data.bin")
    await safe_tool_call(
        read_file,
        {"path": binary_path},
        description="Reading a binary file (expecting hex preview)",
    )

    # --- Read Non-Existent File ---
    non_existent_path = str(DEMO_ROOT / "non_existent.txt")
    await safe_tool_call(
        read_file,
        {"path": non_existent_path},
        description="Attempting to read a non-existent file (should fail)",
    )

    # --- Read a Directory (should fail) ---
    dir_path = str(DEMO_ROOT / "src")
    await safe_tool_call(
        read_file,
        {"path": dir_path},
        description="Attempting to read a directory as a file (should fail)",
    )

    # --- Read Multiple Files (Success and Failure Mix) ---
    paths_to_read = [
        str(DEMO_ROOT / "README.md"),
        str(DEMO_ROOT / "src" / "main.py"),
        str(DEMO_ROOT / "non_existent.txt"),  # This one will fail
        str(DEMO_ROOT / "config" / "settings.json"),
        str(DEMO_ROOT / "src"),  # Reading a directory will also fail here
    ]
    await safe_tool_call(
        read_multiple_files,
        {"paths": paths_to_read},
        description="Reading multiple files (including some that will fail)",
    )

    # --- Read file via Symlink (if created) ---
    if symlink_path:
        # Reading a file within the linked directory
        linked_file_path = str(symlink_path / "main.py")
        await safe_tool_call(
            read_file,
            {"path": linked_file_path},
            description=f"Reading a file via symlink ({os.path.basename(symlink_path)}/main.py)",
        )


async def demonstrate_file_writing_editing():
    """Demonstrate file writing and editing operations."""
    console.print(Rule("[bold cyan]2. File Writing & Editing Operations[/bold cyan]", style="cyan"))
    logger.info("Demonstrating file writing and editing operations...", emoji_key="file")

    # --- Write New File ---
    new_file_path = str(DEMO_ROOT / "data" / "report.md")
    file_content = """# Analysis Report

## Summary
This report contains the analysis of project performance metrics.

## Key Findings
1. Response time improved by 15%
2. Error rate decreased to 0.5%
3. User satisfaction score: 4.8/5.0

## Recommendations
- Continue monitoring performance
- Implement suggested optimizations
- Schedule follow-up review next quarter
"""
    await safe_tool_call(
        write_file,
        {"path": new_file_path, "content": file_content},
        description="Writing a new file (report.md)",
    )

    # --- Overwrite Existing File ---
    overwrite_content = "# Analysis Report (V2)\n\nReport updated."
    await safe_tool_call(
        write_file,
        {"path": new_file_path, "content": overwrite_content},
        description="Overwriting the existing file (report.md)",
    )
    # Verify overwrite
    await safe_tool_call(
        read_file, {"path": new_file_path}, description="Reading the overwritten file to verify"
    )

    # --- Attempt to Write to a Directory (should fail) ---
    await safe_tool_call(
        write_file,
        {"path": str(DEMO_ROOT / "src"), "content": "test"},
        description="Attempting to write over a directory (should fail)",
    )

    # --- Edit File (main.py) ---
    target_edit_file = str(DEMO_ROOT / "src" / "main.py")

    # Edits including one requiring whitespace-insensitive fallback
    edits = [
        {
            "oldText": 'print("Hello from the demo application!")',  # Exact match
            "newText": 'print("Hello from the UPDATED demo application!")\n    logger.info("App started")',
        },
        {
            # This uses different leading whitespace than the original file
            "oldText": "def main():\n    '''Main function to run the application.'''",
            # Expected fallback behavior: find based on stripped lines, replace using original indentation
            "newText": "def main():\n    '''The primary execution function.''' # Docstring updated",
        },
        {
            "oldText": '    return {\n        "debug": True,\n        "log_level": "INFO",\n        "max_connections": 10\n    }',
            "newText": '    return {\n        "debug": False, # Changed to False\n        "log_level": "WARNING",\n        "max_connections": 25 # Increased limit\n    }',
        },
    ]

    await safe_tool_call(
        read_file, {"path": target_edit_file}, description="Reading main.py before editing"
    )

    # Edit with Dry Run
    await safe_tool_call(
        edit_file,
        {"path": target_edit_file, "edits": edits, "dry_run": True},
        description="Editing main.py (Dry Run - showing diff)",
    )

    # Apply Edits for Real
    await safe_tool_call(
        edit_file,
        {"path": target_edit_file, "edits": edits, "dry_run": False},
        description="Applying edits to main.py",
    )

    # Verify Edits
    await safe_tool_call(
        read_file, {"path": target_edit_file}, description="Reading main.py after editing"
    )

    # --- Edit with Non-Existent Old Text (should fail) ---
    failed_edit = [{"oldText": "This text does not exist in the file", "newText": "Replacement"}]
    await safe_tool_call(
        edit_file,
        {"path": target_edit_file, "edits": failed_edit},
        description="Attempting edit with non-existent 'oldText' (should fail)",
    )


async def demonstrate_directory_operations(symlink_path, use_rich_tree=False):
    """Demonstrate directory creation, listing, and tree view."""
    console.print(Rule("[bold cyan]3. Directory Operations[/bold cyan]", style="cyan"))
    logger.info("Demonstrating directory operations...", emoji_key="directory")

    # --- Create Directory ---
    # First ensure parent directory exists
    logs_dir_path = str(DEMO_ROOT / "logs")
    await safe_tool_call(
        create_directory, {"path": logs_dir_path}, description="Creating parent directory (logs)"
    )

    # Now create nested directory
    new_dir_path = str(DEMO_ROOT / "logs" / "debug")
    await safe_tool_call(
        create_directory,
        {"path": new_dir_path},
        description="Creating a new nested directory (logs/debug)",
    )

    # --- Create Directory (already exists) ---
    await safe_tool_call(
        create_directory,
        {"path": new_dir_path},
        description="Attempting to create the same directory again (idempotent)",
    )

    # --- Attempt to Create Directory over a File (should fail) ---
    file_path_for_dir = str(DEMO_ROOT / "README.md")
    await safe_tool_call(
        create_directory,
        {"path": file_path_for_dir},
        description="Attempting to create directory over an existing file (README.md - should fail)",
    )

    # --- List Directory (Root) ---
    await safe_tool_call(
        list_directory,
        {"path": str(DEMO_ROOT)},
        description=f"Listing contents of demo root ({DEMO_ROOT.name})",
    )

    # --- List Directory (Subdir) ---
    await safe_tool_call(
        list_directory,
        {"path": str(DEMO_ROOT / "src")},
        description="Listing contents of subdirectory (src)",
    )

    # --- List Directory (via Symlink, if created) ---
    if symlink_path:
        await safe_tool_call(
            list_directory,
            {"path": str(symlink_path)},
            description=f"Listing contents via symlink ({os.path.basename(symlink_path)})",
        )

    # --- List Non-Existent Directory (should fail) ---
    await safe_tool_call(
        list_directory,
        {"path": str(DEMO_ROOT / "no_such_dir")},
        description="Attempting to list non-existent directory (should fail)",
    )

    # --- Enhanced visualization for directory tree if requested ---
    if use_rich_tree:
        # Restore direct call to async tree generator utility
        console.print(
            "\n[bold cyan]Enhanced Directory Tree Visualization (Async Tool Based)[/bold cyan]"
        )

        try:
            # Generate the tree using the async utility function from display.py
            rich_tree = await generate_rich_directory_tree(str(DEMO_ROOT), max_depth=3)
            console.print(rich_tree)
        except Exception as e:
            logger.error(f"Error generating async directory tree: {e}", exc_info=True)
            console.print(f"[bold red]Error generating directory tree: {escape(str(e))}[/bold red]")

        console.print()  # Add newline

    # --- Directory Tree (Default Depth) --- # This uses the directory_tree TOOL
    # The safe_tool_call will now use its built-in tree renderer for this standard call
    # Note: The tool 'directory_tree' produces a similar but potentially slightly different
    # structure/detail level than the custom async generator above.
    await safe_tool_call(
        directory_tree,
        {"path": str(DEMO_ROOT)},
        description="Generating directory tree for demo root (default depth - using tool)",
    )

    # --- Directory Tree (Specific Depth) ---
    await safe_tool_call(
        directory_tree,
        {"path": str(DEMO_ROOT), "max_depth": 1},
        description="Generating directory tree (max_depth=1)",
    )

    # --- Directory Tree (Include Size) ---
    await safe_tool_call(
        directory_tree,
        {"path": str(DEMO_ROOT), "max_depth": 2, "include_size": True},
        description="Generating directory tree (max_depth=2, include_size=True)",
    )

    # --- Directory Tree (via Symlink, if created) ---
    if symlink_path:
        await safe_tool_call(
            directory_tree,
            {"path": str(symlink_path), "max_depth": 1},
            description=f"Generating directory tree via symlink ({os.path.basename(symlink_path)}, max_depth=1)",
        )


async def demonstrate_move_delete_search(symlink_path):
    """Demonstrate file/directory moving, deletion, searching, and info retrieval."""
    console.print(
        Rule("[bold cyan]4. Move, Delete, Search & Info Operations[/bold cyan]", style="cyan")
    )
    logger.info("Demonstrating move, delete, search, info operations...", emoji_key="file")

    # --- Get File Info (File) ---
    settings_json_path = str(DEMO_ROOT / "config" / "settings.json")
    await safe_tool_call(
        get_file_info,
        {"path": settings_json_path},
        description="Getting file info for settings.json",
    )

    # --- Get File Info (Directory) ---
    src_dir_path = str(DEMO_ROOT / "src")
    await safe_tool_call(
        get_file_info, {"path": src_dir_path}, description="Getting file info for src directory"
    )

    # --- Get File Info (Symlink, if created) ---
    if symlink_path:
        await safe_tool_call(
            get_file_info,
            {"path": str(symlink_path)},
            description=f"Getting file info for symlink ({os.path.basename(symlink_path)}) - uses lstat",
        )

    # --- Search Files (Name Match, Case Insensitive) ---
    await safe_tool_call(
        search_files,
        {"path": str(DEMO_ROOT), "pattern": "readme"},
        description="Searching for 'readme' (case insensitive)",
    )

    # --- Search Files (Name Match, Case Sensitive) ---
    await safe_tool_call(
        search_files,
        {"path": str(DEMO_ROOT), "pattern": "README", "case_sensitive": True},
        description="Searching for 'README' (case sensitive)",
    )

    # --- Search Files (With Exclusions) ---
    await safe_tool_call(
        search_files,
        {
            "path": str(DEMO_ROOT),
            "pattern": ".py",
            "exclude_patterns": ["*/test*", ".hidden_dir/*"],
        },
        description="Searching for '*.py', excluding tests and hidden dir",
    )

    # --- Search Files (Content Search) ---
    await safe_tool_call(
        search_files,
        {"path": str(DEMO_ROOT), "pattern": "localhost", "search_content": True},
        description="Searching for content 'localhost' inside files",
    )

    # --- Search Files (Content Search, Case Sensitive) ---
    await safe_tool_call(
        search_files,
        {
            "path": str(DEMO_ROOT),
            "pattern": "DataProcessor",
            "search_content": True,
            "case_sensitive": True,
        },
        description="Searching for content 'DataProcessor' (case sensitive)",
    )

    # --- Search Files (No Matches) ---
    await safe_tool_call(
        search_files,
        {"path": str(DEMO_ROOT), "pattern": "xyz_no_match_xyz"},
        description="Searching for pattern guaranteed not to match",
    )

    # --- Move File ---
    source_move_path = str(DEMO_ROOT / "data" / "sample.csv")
    dest_move_path = str(DEMO_ROOT / "data" / "renamed_sample.csv")
    await safe_tool_call(
        move_file,
        {"source": source_move_path, "destination": dest_move_path},
        description="Moving (renaming) sample.csv",
    )
    # Verify move by trying to get info on new path
    await safe_tool_call(
        get_file_info,
        {"path": dest_move_path},
        description="Verifying move by getting info on new path",
    )

    # --- Move File (Overwrite) ---
    # First create a file to be overwritten
    overwrite_target_path = str(DEMO_ROOT / "data" / "overwrite_me.txt")
    await safe_tool_call(
        write_file,
        {"path": overwrite_target_path, "content": "Original content"},
        description="Creating file to be overwritten",
    )
    # Now move onto it with overwrite=True
    await safe_tool_call(
        move_file,
        {"source": dest_move_path, "destination": overwrite_target_path, "overwrite": True},
        description="Moving renamed_sample.csv onto overwrite_me.txt (overwrite=True)",
    )
    # Verify overwrite
    await safe_tool_call(
        get_file_info,
        {"path": overwrite_target_path},
        description="Verifying overwrite by getting info",
    )

    # --- Move Directory ---
    source_dir_move = str(DEMO_ROOT / "tests")
    dest_dir_move = str(DEMO_ROOT / "tests_moved")
    await safe_tool_call(
        move_file,
        {"source": source_dir_move, "destination": dest_dir_move},
        description="Moving the 'tests' directory",
    )
    # Verify move
    await safe_tool_call(
        list_directory,
        {"path": dest_dir_move},
        description="Verifying directory move by listing new path",
    )

    # --- Attempt Move (Destination Exists, No Overwrite - should fail) ---
    await safe_tool_call(
        move_file,
        {
            "source": str(DEMO_ROOT / "README.md"),
            "destination": str(DEMO_ROOT / "config" / "settings.json"),
        },
        description="Attempting to move README.md onto settings.json (no overwrite - should fail)",
    )

    # --- Delete File ---
    file_to_delete = str(DEMO_ROOT / "temp.log")
    await safe_tool_call(
        get_file_info,
        {"path": file_to_delete},
        description="Checking temp.log exists before deleting",
    )
    await safe_tool_call(
        delete_path, {"path": file_to_delete}, description="Deleting single file (temp.log)"
    )
    await safe_tool_call(
        get_file_info,
        {"path": file_to_delete},
        description="Verifying temp.log deletion (should fail)",
    )

    # --- Delete Symlink (if created) ---
    if symlink_path:
        # Get the exact path string to the symlink without resolving it
        symlink_str = str(symlink_path)
        await safe_tool_call(
            get_file_info,
            {"path": symlink_str},
            description=f"Checking symlink {os.path.basename(symlink_path)} exists before deleting",
        )

        # Explicitly tell the user what we're doing
        console.print(
            f"[cyan]Note:[/cyan] Deleting the symlink itself (not its target) at path: {symlink_str}"
        )

        await safe_tool_call(
            delete_path,
            {"path": symlink_str},
            description=f"Deleting symlink ({os.path.basename(symlink_path)})",
        )
        await safe_tool_call(
            get_file_info,
            {"path": symlink_str},
            description="Verifying symlink deletion (should fail)",
        )

    # --- Delete Empty Directory ---
    empty_dir_to_delete = str(DEMO_ROOT / "logs" / "debug")  # Created earlier, should be empty
    await safe_tool_call(
        get_file_info,
        {"path": empty_dir_to_delete},
        description="Checking logs/debug exists before deleting",
    )
    await safe_tool_call(
        delete_path,
        {"path": empty_dir_to_delete},
        description="Deleting empty directory (logs/debug)",
    )
    await safe_tool_call(
        get_file_info,
        {"path": empty_dir_to_delete},
        description="Verifying empty directory deletion (should fail)",
    )

    # --- Delete Directory with Content (Testing Deletion Protection) ---
    bulk_dir_path = str(DEMO_ROOT / "bulk_files")
    console.print(
        Panel(
            f"Attempting to delete directory '{os.path.basename(bulk_dir_path)}' which contains {BULK_FILES_COUNT} files.\n"
            "This will trigger the deletion protection check (heuristics based on file count, timestamps, types).\n"
            "Whether it blocks depends on the config thresholds and calculated variances.",
            title="🛡️ Testing Deletion Protection 🛡️",
            border_style="yellow",
        )
    )
    # This call might raise ProtectionTriggeredError, which safe_tool_call will catch and display
    await safe_tool_call(
        delete_path,
        {"path": bulk_dir_path},
        description=f"Deleting directory with {BULK_FILES_COUNT} files (bulk_files)",
    )
    # Check if it was actually deleted or blocked by protection
    await safe_tool_call(
        get_file_info,
        {"path": bulk_dir_path},
        description="Checking if bulk_files directory still exists after delete attempt",
    )


async def demonstrate_security_features():
    """Demonstrate security features like allowed directories."""
    console.print(Rule("[bold cyan]5. Security Features[/bold cyan]", style="cyan"))
    logger.info("Demonstrating security features...", emoji_key="security")

    # --- List Allowed Directories ---
    # This reads from the config (which we set via env var for the demo)
    await safe_tool_call(
        list_allowed_directories, {}, description="Listing configured allowed directories"
    )
    console.print(
        f"[dim]Note: For this demo, only the temporary directory [cyan]{DEMO_TEMP_DIR}[/cyan] was allowed via environment variable.[/dim]"
    )

    # --- Try to Access Standard System Root (should fail) ---
    # Choose a path guaranteed outside the temp allowed dir
    outside_path_root = "/" if platform.system() != "Windows" else "C:\\"
    console.print(
        f"\nAttempting operation outside allowed directory: [red]Listing '{outside_path_root}'[/red]"
    )
    await safe_tool_call(
        list_directory,
        {"path": outside_path_root},
        description=f"Attempting to list root directory '{outside_path_root}' (should fail)",
    )

    # --- Try to Access Specific Sensitive File (should fail) ---
    outside_path_file = (
        "/etc/passwd"
        if platform.system() != "Windows"
        else "C:\\Windows\\System32\\drivers\\etc\\hosts"
    )
    console.print(
        f"\nAttempting operation outside allowed directory: [red]Reading '{outside_path_file}'[/red]"
    )
    await safe_tool_call(
        read_file,
        {"path": outside_path_file},
        description=f"Attempting to read sensitive file '{outside_path_file}' (should fail)",
    )

    # --- Try to use '..' to escape (should fail due to normalization) ---
    escape_path = str(DEMO_ROOT / ".." / "..")  # Attempt to go above the allowed temp dir
    # Note: validate_path normalizes this, so it might resolve to something unexpected but still potentially outside
    # Or, more likely, the normalized path check against allowed dirs will fail.
    console.print(
        f"\nAttempting operation using '..' to potentially escape: [red]Listing '{escape_path}'[/red]"
    )
    await safe_tool_call(
        list_directory,
        {"path": escape_path},
        description=f"Attempting to list path using '..' ('{escape_path}')",
    )

    console.print(
        Panel(
            "Security checks demonstrated:\n"
            "1. Operations are confined to the `allowed_directories`.\n"
            "2. Accessing paths outside these directories is denied.\n"
            "3. Path normalization prevents trivial directory traversal escapes (`..`).\n"
            "4. Symlink targets are also validated against `allowed_directories` (implicitly tested via symlink operations).\n"
            "5. Deletion protection provides a safety net against accidental bulk deletions (demonstrated earlier).",
            title="Security Summary",
            border_style="green",
            expand=False,
        )
    )


async def main():
    """Run the filesystem operations demonstration."""
    global DEMO_TEMP_DIR  # Make sure main knows about this path
    symlink_path = None
    exit_code = 0

    # Parse command line arguments
    args = parse_arguments()

    try:
        console.print(
            Rule("[bold blue]Secure Filesystem Operations Demo[/bold blue]", style="white")
        )
        logger.info("Starting filesystem operations demonstration", emoji_key="start")

        # --- Verify Config Loading ---
        print(Rule("Verifying Configuration", style="dim"))
        config_valid = verify_config()
        if not config_valid:
            # Abort with a clear message if config verification fails
            console.print(
                "[bold red]Error:[/bold red] Configuration verification failed. Aborting demonstration.",
                style="red",
            )
            return 1  # Exit early if config is wrong

        # --- Verify Config Loading ---
        try:
            current_config = get_config()
            fs_config = current_config.filesystem
            loaded_allowed_dirs = fs_config.allowed_directories
            console.print(f"[dim]Config Check: Loaded allowed dirs: {loaded_allowed_dirs}[/dim]")
            if not loaded_allowed_dirs or DEMO_TEMP_DIR not in loaded_allowed_dirs:
                console.print(
                    "[bold red]Error:[/bold red] Demo temporary directory not found in loaded allowed directories. Configuration failed.",
                    style="red",
                )
                console.print(f"[dim]Expected: {DEMO_TEMP_DIR}[/dim]")
                console.print(
                    f"[dim]Loaded Config: {current_config.model_dump()}"
                )  # Dump entire config
                return 1  # Exit early if config is wrong
        except Exception as config_err:
            console.print(
                f"[bold red]Error checking loaded configuration:[/bold red] {config_err}",
                style="red",
            )
            console.print_exception(show_locals=False)
            return 1
        # --- End Verify Config Loading ---

        # Display available options if running all demos
        if args.demo == "all":
            console.print(
                Panel(
                    "This demo includes multiple sections showcasing different filesystem operations.\n"
                    "You can run individual sections using the following commands:\n\n"
                    "[yellow]python examples/filesystem_operations_demo.py read[/yellow] - File reading operations\n"
                    "[yellow]python examples/filesystem_operations_demo.py write[/yellow] - File writing and editing operations\n"
                    "[yellow]python examples/filesystem_operations_demo.py directory[/yellow] - Directory operations\n"
                    "[yellow]python examples/filesystem_operations_demo.py move_delete[/yellow] - Move, delete, search & info operations\n"
                    "[yellow]python examples/filesystem_operations_demo.py security[/yellow] - Security features demo\n\n"
                    "Add [yellow]--rich-tree[/yellow] for enhanced directory visualization!",
                    title="Demo Options",
                    border_style="cyan",
                    expand=False,
                )
            )

        # Display info message
        console.print(
            Panel(
                "This demo showcases the secure asynchronous filesystem tools.\n"
                f"A temporary directory ([cyan]{DEMO_TEMP_DIR}[/cyan]) has been created and configured as the ONLY allowed directory for this demo's operations via environment variables.",
                title="About This Demo",
                border_style="cyan",
            )
        )

        # Set up the demo environment *inside* the allowed temp dir
        symlink_path = await setup_demo_environment()

        # Run the selected demonstration(s)
        if args.demo == "all" or args.demo == "read":
            await demonstrate_file_reading(symlink_path)
            console.print()  # Add newline

        if args.demo == "all" or args.demo == "write":
            await demonstrate_file_writing_editing()
            console.print()  # Add newline

        if args.demo == "all" or args.demo == "directory":
            await demonstrate_directory_operations(symlink_path, use_rich_tree=args.rich_tree)
            console.print()  # Add newline

        if args.demo == "all" or args.demo == "move_delete":
            await demonstrate_move_delete_search(symlink_path)
            console.print()  # Add newline

        if args.demo == "all" or args.demo == "security":
            await demonstrate_security_features()

        logger.success(
            f"Filesystem Operations Demo(s) completed: {args.demo}", emoji_key="complete"
        )
        console.print(Rule("[bold green]Demo Complete[/bold green]", style="green"))

    except Exception as e:
        logger.critical(f"Demo crashed unexpectedly: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"\n[bold red]CRITICAL ERROR:[/bold red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        exit_code = 1

    finally:
        # Clean up the demo environment
        console.print(Rule("Cleanup", style="dim"))
        await cleanup_demo_environment()

    return exit_code


def get_config_local():  # Renamed
    """Get application configuration."""
    return {"debug": True, "log_level": "INFO", "max_connections": 10}


if __name__ == "__main__":
    # Basic check for asyncio policy on Windows if needed
    # if sys.platform == "win32" and sys.version_info >= (3, 8):
    #     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run the demo
    final_exit_code = asyncio.run(main())
    sys.exit(final_exit_code)
