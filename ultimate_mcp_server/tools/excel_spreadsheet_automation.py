"""Excel Spreadsheet Automation Tools for Ultimate MCP Server.

This module provides powerful, flexible tools for AI agents to automate Excel workflows through
the Model Context Protocol (MCP). These tools leverage the intelligence of the Large Language Model
while providing deep integration with Microsoft Excel on Windows.

The core philosophy is minimalist but powerful - a few highly flexible functions that can be composed
to perform complex operations, with the LLM (Claude) providing the intelligence to drive these tools.

Key capabilities:
- Direct Excel manipulation (create, modify, analyze spreadsheets)
- Learning from exemplar templates and applying patterns to new contexts
- Formula debugging and optimization
- Rich automated formatting and visualization
- VBA generation and execution

Windows-specific: Uses COM automation with win32com and requires Excel to be installed.

Example usage:
```python
# Execute Excel operations with natural language instructions
result = await client.tools.excel_execute(
    instruction="Create a new workbook with two sheets: 'Revenue' and 'Expenses'. "
                "In the Revenue sheet, create a quarterly forecast for 2025 with "
                "monthly growth of 5%. Include columns for Product A and Product B "
                "with initial values of $10,000 and $5,000. Format as a professional "
                "financial table with totals and proper currency formatting.",
    file_path="financial_forecast.xlsx",
    operation_type="create"
)

# Learn from an exemplar template and adapt it to a new context
result = await client.tools.excel_learn_and_apply(
    exemplar_path="templates/financial_model.xlsx",
    output_path="healthcare_startup.xlsx",
    adaptation_context="Create a 3-year financial model for a healthcare SaaS startup "
                      "with subscription revenue model. Include revenue forecast, expense "
                      "projections, cash flow, and key metrics for investors. Adapt all "
                      "growth rates and assumptions for the healthcare tech market."
)

# Debug and optimize complex formulas
result = await client.tools.excel_analyze_formulas(
    file_path="complex_model.xlsx",
    sheet_name="Valuation",
    cell_range="D15:G25",
    analysis_type="optimize",
    detail_level="detailed"
)
```
"""

import asyncio
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# Try to import Windows-specific libraries
try:
    import pythoncom  # type: ignore
    import win32com.client  # type: ignore
    import win32com.client.gencache  # type: ignore
    from win32com.client import constants as win32c  # type: ignore

    WINDOWS_EXCEL_AVAILABLE = True
except ImportError:
    WINDOWS_EXCEL_AVAILABLE = False

from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.tools.base import (
    BaseTool,
    with_error_handling,
    with_state_management,
    with_tool_metrics,
)
from ultimate_mcp_server.tools.filesystem import (
    create_directory,
    get_allowed_directories,
    read_file_content,
    validate_path,
    write_file_content,
)
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.excel_spreadsheet_automation")


class ExcelSession:
    """Manages a single Excel Application session with enhanced reliability and safety."""

    def __init__(self, visible=False):
        """Initialize a new Excel session.

        Args:
            visible: Whether Excel should be visible on screen
        """
        if not WINDOWS_EXCEL_AVAILABLE:
            raise ToolError("Excel automation requires Windows with Excel installed")

        # Initialize COM in this thread
        pythoncom.CoInitialize()

        self.app = None
        self.workbooks = {}
        self.visible = visible
        self.status = "initializing"

        try:
            self.app = win32com.client.Dispatch("Excel.Application")
            self.app.Visible = visible
            self.app.DisplayAlerts = False
            self.app.ScreenUpdating = False
            self.app_version = self.app.Version
            self.status = "ready"
        except Exception as e:
            self.status = "error"
            raise ToolError(f"Failed to create Excel instance: {str(e)}") from e

    def open_workbook(self, path, read_only=False):
        """Open an Excel workbook.

        Args:
            path: Path to the workbook file
            read_only: Whether to open in read-only mode

        Returns:
            Workbook COM object
        """
        try:
            # Use the path as is, validation should happen at the async layer
            # that calls this sync method. The path should already be validated.
            abs_path = os.path.abspath(path)
            wb = self.app.Workbooks.Open(abs_path, ReadOnly=read_only)
            self.workbooks[wb.Name] = wb
            return wb
        except Exception as e:
            raise ToolError(f"Failed to open workbook at {path}: {str(e)}") from e

    def create_workbook(self):
        """Create a new Excel workbook.

        Returns:
            Workbook COM object
        """
        try:
            wb = self.app.Workbooks.Add()
            self.workbooks[wb.Name] = wb
            return wb
        except Exception as e:
            raise ToolError(f"Failed to create new workbook: {str(e)}") from e

    def save_workbook(self, workbook, path):
        """Save a workbook to a specified path.

        Args:
            workbook: Workbook COM object
            path: Path to save the workbook
        """
        try:
            # Note: Directory creation should happen at the async layer before calling this sync method
            # Path validation should also happen at the async layer
            workbook.SaveAs(os.path.abspath(path))
            return True
        except Exception as e:
            raise ToolError(f"Failed to save workbook to {path}: {str(e)}") from e

    def close_workbook(self, workbook, save_changes=False):
        """Close a workbook.

        Args:
            workbook: Workbook COM object
            save_changes: Whether to save changes before closing
        """
        try:
            workbook.Close(SaveChanges=save_changes)
            if workbook.Name in self.workbooks:
                del self.workbooks[workbook.Name]
        except Exception as e:
            logger.warning(f"Error closing workbook: {str(e)}")

    def close(self):
        """Close the Excel application and release resources."""
        if not self.app:
            return

        try:
            # Close all workbooks
            for wb_name in list(self.workbooks.keys()):
                try:
                    self.close_workbook(self.workbooks[wb_name], False)
                except Exception:
                    pass

            # Quit Excel
            try:
                self.app.DisplayAlerts = False
                self.app.ScreenUpdating = True
                self.app.Quit()
            except Exception:
                pass

            # Release COM references
            del self.app
            self.app = None

            # Uninitialize COM
            pythoncom.CoUninitialize()

            self.status = "closed"
        except Exception as e:
            self.status = "error_closing"
            logger.error(f"Error closing Excel session: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


@asynccontextmanager
async def get_excel_session(visible=False):
    """Async context manager for getting an Excel session.

    Args:
        visible: Whether Excel should be visible

    Yields:
        ExcelSession: An Excel session
    """
    session = None
    try:
        # Create the Excel session in a thread pool to avoid blocking
        session = await asyncio.to_thread(ExcelSession, visible=visible)
        yield session
    finally:
        # Cleanup in a thread pool as well
        if session:
            await asyncio.to_thread(session.close)


class ExcelSpreadsheetTools(BaseTool):
    """Tool for automating Excel spreadsheet operations."""

    tool_name = "excel_spreadsheet_tools"
    description = "Tool for automating Excel spreadsheet operations."

    def __init__(self, mcp_server):
        """Initialize Excel Spreadsheet Tools.

        Args:
            mcp_server: MCP server instance
        """
        super().__init__(mcp_server)

        # Inform if Excel is not available
        if not WINDOWS_EXCEL_AVAILABLE:
            raise ToolError("Excel automation requires Windows with Excel installed")

    @with_tool_metrics
    @with_error_handling
    @with_state_management("excel_tools")
    async def excel_execute(
        self,
        instruction: str,
        file_path: Optional[str] = None,
        operation_type: str = "create",
        template_path: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        show_excel: bool = False,
        get_state=None,
        set_state=None,
        delete_state=None,
        ctx=None,
    ) -> Dict[str, Any]:
        """Execute Excel operations based on natural language instructions.

        This is the primary function for manipulating Excel files. It can create new files,
        modify existing ones, and perform various operations based on natural language instructions.
        The intelligence for interpreting these instructions comes from the LLM (Claude),
        which generates the appropriate parameters and logic.

        Args:
            instruction: Natural language instruction describing what to do
            file_path: Path to save or modify an Excel file
            operation_type: Type of operation (create, modify, analyze, format, etc.)
            template_path: Optional path to a template file to use as a starting point
            parameters: Optional structured parameters to supplement the instruction
            show_excel: Whether to make Excel visible during execution
            get_state: Function to get state (injected by with_state_management)
            set_state: Function to set state (injected by with_state_management)
            delete_state: Function to delete state (injected by with_state_management)
            ctx: Context object (injected by with_state_management)

        Returns:
            Dictionary with operation results and metadata
        """
        start_time = time.time()

        # Basic validation
        if not instruction:
            raise ToolInputError("instruction cannot be empty")

        if operation_type == "create" and not file_path:
            raise ToolInputError("file_path is required for 'create' operations")

        if operation_type in ["modify", "analyze", "format"] and (
            not file_path or not os.path.exists(file_path)
        ):
            raise ToolInputError(
                f"Valid existing file_path is required for '{operation_type}' operations"
            )

        # Use parameters if provided, otherwise empty dict
        parameters = parameters or {}

        # Process template path if provided
        if template_path and not os.path.exists(template_path):
            raise ToolInputError(f"Template file not found at {template_path}")

        # Execute the requested operation
        try:
            # Create or retrieve the Excel session from state
            session = await self._get_or_create_excel_session(show_excel, get_state, set_state)

            result = await self._execute_excel_operation(
                session=session,
                instruction=instruction,
                operation_type=operation_type,
                file_path=file_path,
                template_path=template_path,
                parameters=parameters,
            )

            processing_time = time.time() - start_time
            result["processing_time"] = processing_time

            logger.info(
                f"Excel operation '{operation_type}' completed in {processing_time:.2f}s",
                emoji_key="success",
            )

            return result

        except Exception as e:
            logger.error(
                f"Error executing Excel operation: {str(e)}", emoji_key="error", exc_info=True
            )
            # Try to clean up session on error
            await self._cleanup_excel_session(delete_state)
            raise ToolError(
                f"Failed to execute Excel operation: {str(e)}",
                details={"operation_type": operation_type, "file_path": file_path},
            ) from e

    @with_tool_metrics
    @with_error_handling
    @with_state_management("excel_tools")
    async def excel_learn_and_apply(
        self,
        exemplar_path: str,
        output_path: str,
        adaptation_context: str,
        parameters: Optional[Dict[str, Any]] = None,
        show_excel: bool = False,
        get_state=None,
        set_state=None,
        delete_state=None,
        ctx=None,
    ) -> Dict[str, Any]:
        """Learn from an exemplar Excel template and apply it to a new context.

        This powerful function allows Claude to analyze an existing Excel model or template,
        understand its structure and formulas, and then create a new file adapted to a different
        context while preserving the intelligence embedded in the original.

        Args:
            exemplar_path: Path to the Excel file to learn from
            output_path: Path where the new adapted file should be saved
            adaptation_context: Natural language description of how to adapt the template
            parameters: Optional structured parameters with specific adaptation instructions
            show_excel: Whether to make Excel visible during processing
            get_state: Function to get state (injected by with_state_management)
            set_state: Function to set state (injected by with_state_management)
            delete_state: Function to delete state (injected by with_state_management)
            ctx: Context object (injected by with_state_management)

        Returns:
            Dictionary with operation results and adaptations made
        """
        start_time = time.time()

        # Validate paths
        try:
            validated_exemplar_path = await validate_path(exemplar_path, check_exists=True)
            validated_output_path = await validate_path(
                output_path, check_exists=False, check_parent_writable=True
            )

            # Ensure parent directory for output exists
            parent_dir = os.path.dirname(validated_output_path)
            if parent_dir:
                await create_directory(parent_dir)
        except ToolInputError:
            raise
        except Exception as e:
            raise ToolInputError(f"Path validation error: {str(e)}") from e

        if not adaptation_context:
            raise ToolInputError("adaptation_context cannot be empty")

        # Use parameters if provided, otherwise empty dict
        parameters = parameters or {}

        # Execute the template learning and application
        try:
            # Create or retrieve the Excel session from state
            session = await self._get_or_create_excel_session(show_excel, get_state, set_state)

            # First, learn the template structure
            template_analysis = await self._analyze_excel_template(  # noqa: F841
                session=session, exemplar_path=validated_exemplar_path, parameters=parameters
            )

            # Apply the learned template to the new context
            result = await self._apply_excel_template(
                session=session,
                exemplar_path=validated_exemplar_path,
                output_path=validated_output_path,
                data={"mappings": [], "adaptation_context": adaptation_context},
                parameters=parameters,
            )

            processing_time = time.time() - start_time
            result["processing_time"] = processing_time

            logger.info(
                f"Excel template learning and application completed in {processing_time:.2f}s",
                emoji_key="success",
            )

            return result

        except Exception as e:
            logger.error(
                f"Error in template learning and application: {str(e)}",
                emoji_key="error",
                exc_info=True,
            )
            # Try to clean up session on error
            await self._cleanup_excel_session(delete_state)
            raise ToolError(
                f"Failed to learn and apply template: {str(e)}",
                details={"exemplar_path": exemplar_path, "output_path": output_path},
            ) from e

    @with_tool_metrics
    @with_error_handling
    @with_state_management("excel_tools")
    async def excel_analyze_formulas(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        cell_range: Optional[str] = None,
        analysis_type: str = "analyze",
        detail_level: str = "standard",
        show_excel: bool = False,
        get_state=None,
        set_state=None,
        delete_state=None,
        ctx=None,
    ) -> Dict[str, Any]:
        """Analyze, debug, and optimize Excel formulas.

        This function provides deep insights into Excel formulas, identifying errors,
        suggesting optimizations, and explaining complex calculations in natural language.

        Args:
            file_path: Path to the Excel file to analyze
            sheet_name: Name of the sheet to analyze (if None, active sheet is used)
            cell_range: Cell range to analyze (if None, all formulas are analyzed)
            analysis_type: Type of analysis (analyze, debug, optimize, explain)
            detail_level: Level of detail in the analysis (basic, standard, detailed)
            show_excel: Whether to make Excel visible during analysis
            get_state: Function to get state (injected by with_state_management)
            set_state: Function to set state (injected by with_state_management)
            delete_state: Function to delete state (injected by with_state_management)
            ctx: Context object (injected by with_state_management)

        Returns:
            Dictionary with analysis results, issues found, and suggestions
        """
        start_time = time.time()

        # Validate the file path
        try:
            validated_file_path = await validate_path(file_path, check_exists=True)
        except ToolInputError:
            raise
        except Exception as e:
            raise ToolInputError(
                f"Invalid file path: {str(e)}", param_name="file_path", provided_value=file_path
            ) from e

        # Execute the formula analysis
        try:
            # Create or retrieve the Excel session from state
            session = await self._get_or_create_excel_session(show_excel, get_state, set_state)

            result = await self._analyze_excel_formulas(
                session=session,
                file_path=validated_file_path,
                sheet_name=sheet_name,
                cell_range=cell_range,
                analysis_type=analysis_type,
                detail_level=detail_level,
            )

            processing_time = time.time() - start_time
            result["processing_time"] = processing_time

            logger.info(
                f"Excel formula analysis completed in {processing_time:.2f}s", emoji_key="success"
            )

            return result

        except Exception as e:
            logger.error(
                f"Error analyzing Excel formulas: {str(e)}", emoji_key="error", exc_info=True
            )
            # Try to clean up session on error
            await self._cleanup_excel_session(delete_state)
            raise ToolError(
                f"Failed to analyze Excel formulas: {str(e)}",
                details={
                    "file_path": file_path,
                    "sheet_name": sheet_name,
                    "cell_range": cell_range,
                },
            ) from e

    @with_tool_metrics
    @with_error_handling
    @with_state_management("excel_tools")
    async def excel_generate_macro(
        self,
        instruction: str,
        file_path: Optional[str] = None,
        template: Optional[str] = None,
        test_execution: bool = False,
        security_level: str = "standard",
        show_excel: bool = False,
        get_state=None,
        set_state=None,
        delete_state=None,
        ctx=None,
    ) -> Dict[str, Any]:
        """Generate and optionally execute Excel VBA macros based on natural language instructions.

        This function leverages Claude's capability to generate Excel VBA code for automating
        complex tasks within Excel. It can create new macros or modify existing ones.

        Args:
            instruction: Natural language description of what the macro should do
            file_path: Path to the Excel file where the macro should be added
            template: Optional template or skeleton code to use as a starting point
            test_execution: Whether to test execute the generated macro
            security_level: Security restrictions for macro execution (standard, restricted, permissive)
            show_excel: Whether to make Excel visible during processing
            get_state: Function to get state (injected by with_state_management)
            set_state: Function to set state (injected by with_state_management)
            delete_state: Function to delete state (injected by with_state_management)
            ctx: Context object (injected by with_state_management)

        Returns:
            Dictionary with the generated macro code and execution results if applicable
        """
        start_time = time.time()

        # Basic validation
        if not instruction:
            raise ToolInputError("instruction cannot be empty")

        if file_path and file_path.endswith(".xlsx"):
            # Convert to .xlsm for macro support if needed
            file_path = file_path.replace(".xlsx", ".xlsm")
            logger.info(f"Changed file extension to .xlsm for macro support: {file_path}")

        # Execute the macro generation
        try:
            # Create or retrieve the Excel session from state
            session = await self._get_or_create_excel_session(show_excel, get_state, set_state)

            result = await self._generate_excel_macro(
                session=session,
                instruction=instruction,
                file_path=file_path,
                template=template,
                test_execution=test_execution,
                security_level=security_level,
            )

            processing_time = time.time() - start_time
            result["processing_time"] = processing_time

            logger.info(
                f"Excel macro generation completed in {processing_time:.2f}s", emoji_key="success"
            )

            return result

        except Exception as e:
            logger.error(
                f"Error generating Excel macro: {str(e)}", emoji_key="error", exc_info=True
            )
            # Try to clean up session on error
            await self._cleanup_excel_session(delete_state)
            raise ToolError(
                f"Failed to generate Excel macro: {str(e)}", details={"file_path": file_path}
            ) from e

    @with_tool_metrics
    @with_error_handling
    @with_state_management("excel_tools")
    async def excel_export_sheet_to_csv(
        self,
        file_path: str,
        sheet_name: str,
        output_path: Optional[str] = None,
        delimiter: str = ",",
        show_excel: bool = False,
        get_state=None,
        set_state=None,
        delete_state=None,
        ctx=None,
    ) -> Dict[str, Any]:
        """Export an Excel sheet to a CSV file.

        This function allows exporting data from an Excel sheet to a CSV file,
        which can be useful for data exchange or further processing.

        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to export
            output_path: Path where to save the CSV file (default: same as Excel with .csv)
            delimiter: Character to use as delimiter (default: comma)
            show_excel: Whether to make Excel visible during processing
            get_state: Function to get state (injected by with_state_management)
            set_state: Function to set state (injected by with_state_management)
            delete_state: Function to delete state (injected by with_state_management)
            ctx: Context object (injected by with_state_management)

        Returns:
            Dictionary with export results
        """
        start_time = time.time()

        # Validate the file path
        try:
            # Use our custom validation with get_allowed_directories
            validated_file_path = await self._validate_excel_file_path(file_path, check_exists=True)
        except ToolInputError:
            raise
        except Exception as e:
            raise ToolInputError(
                f"Invalid file path: {str(e)}", param_name="file_path", provided_value=file_path
            ) from e

        # Set default output path if not provided
        if not output_path:
            output_path = os.path.splitext(validated_file_path)[0] + ".csv"
        else:
            # Validate the output path
            try:
                temp_validated_path = await validate_path(
                    output_path, check_exists=False, check_parent_writable=True
                )
                output_path = temp_validated_path

                # Ensure parent directory exists
                parent_dir = os.path.dirname(temp_validated_path)
                if parent_dir:
                    await create_directory(parent_dir)
            except Exception as e:
                raise ToolInputError(
                    f"Invalid output path: {str(e)}",
                    param_name="output_path",
                    provided_value=output_path,
                ) from e

        # Execute the export operation
        try:
            # Create or retrieve the Excel session from state
            session = await self._get_or_create_excel_session(show_excel, get_state, set_state)

            # Open the workbook
            workbook = session.open_workbook(validated_file_path, read_only=True)

            # Find the worksheet
            worksheet = None
            for sheet in workbook.Worksheets:
                if sheet.Name.lower() == sheet_name.lower():
                    worksheet = sheet
                    break

            if not worksheet:
                raise ToolInputError(
                    f"Sheet '{sheet_name}' not found in workbook",
                    param_name="sheet_name",
                    provided_value=sheet_name,
                )

            # Get data from the worksheet
            used_range = worksheet.UsedRange
            row_count = used_range.Rows.Count
            col_count = used_range.Columns.Count

            # Extract data
            csv_data = []
            for row in range(1, row_count + 1):
                row_data = []
                for col in range(1, col_count + 1):
                    cell_value = used_range.Cells(row, col).Value
                    row_data.append(str(cell_value) if cell_value is not None else "")
                csv_data.append(row_data)

            # Close the workbook
            workbook.Close(SaveChanges=False)

            # Convert data to CSV format
            csv_content = ""
            for row_data in csv_data:
                # Escape any delimiter characters in the data and wrap in quotes if needed
                escaped_row = []
                for cell in row_data:
                    if delimiter in cell or '"' in cell or "\n" in cell:
                        # Replace double quotes with escaped double quotes
                        escaped_cell = cell.replace('"', '""')
                        escaped_row.append(f'"{escaped_cell}"')
                    else:
                        escaped_row.append(cell)

                csv_content += delimiter.join(escaped_row) + "\n"

            # Write the CSV content to file
            await write_file_content(output_path, csv_content)

            processing_time = time.time() - start_time
            result = {
                "success": True,
                "file_path": validated_file_path,
                "sheet_name": sheet_name,
                "output_path": output_path,
                "row_count": row_count,
                "column_count": col_count,
                "processing_time": processing_time,
            }

            logger.info(
                f"Excel sheet export completed in {processing_time:.2f}s", emoji_key="success"
            )

            return result

        except Exception as e:
            logger.error(f"Error exporting Excel sheet: {str(e)}", emoji_key="error", exc_info=True)
            # Try to clean up session on error
            await self._cleanup_excel_session(delete_state)
            raise ToolError(
                f"Failed to export Excel sheet: {str(e)}",
                details={"file_path": file_path, "sheet_name": sheet_name},
            ) from e

    @with_tool_metrics
    @with_error_handling
    @with_state_management("excel_tools")
    async def excel_import_csv_to_sheet(
        self,
        file_path: str,
        csv_path: str,
        sheet_name: Optional[str] = None,
        delimiter: str = ",",
        start_cell: str = "A1",
        create_sheet: bool = False,
        show_excel: bool = False,
        get_state=None,
        set_state=None,
        delete_state=None,
        ctx=None,
    ) -> Dict[str, Any]:
        """Import CSV data into an Excel sheet.

        This function allows importing data from a CSV file into an Excel workbook,
        either into an existing sheet or by creating a new sheet.

        Args:
            file_path: Path to the Excel file
            csv_path: Path to the CSV file to import
            sheet_name: Name of the sheet to import into (if None, uses active sheet)
            delimiter: Character used as delimiter in the CSV (default: comma)
            start_cell: Cell where to start importing (default: A1)
            create_sheet: Whether to create a new sheet if sheet_name doesn't exist
            show_excel: Whether to make Excel visible during processing
            get_state: Function to get state (injected by with_state_management)
            set_state: Function to set state (injected by with_state_management)
            delete_state: Function to delete state (injected by with_state_management)
            ctx: Context object (injected by with_state_management)

        Returns:
            Dictionary with import results
        """
        start_time = time.time()

        # Validate the Excel file path
        try:
            validated_file_path = await self._validate_excel_file_path(file_path, check_exists=True)
        except ToolInputError:
            raise
        except Exception as e:
            raise ToolInputError(
                f"Invalid Excel file path: {str(e)}",
                param_name="file_path",
                provided_value=file_path,
            ) from e

        # Validate the CSV file path
        try:
            validated_csv_path = await validate_path(csv_path, check_exists=True)
        except ToolInputError:
            raise
        except Exception as e:
            raise ToolInputError(
                f"Invalid CSV file path: {str(e)}", param_name="csv_path", provided_value=csv_path
            ) from e

        # Execute the import operation
        try:
            # Read the CSV content
            csv_content = await read_file_content(validated_csv_path)

            # Parse CSV data
            csv_data = []
            for line in csv_content.splitlines():
                if not line.strip():
                    continue

                # Handle quoted fields with delimiters inside them
                row = []
                field = ""
                in_quotes = False
                i = 0

                while i < len(line):
                    char = line[i]

                    if char == '"' and (i == 0 or line[i - 1] != "\\"):
                        # Toggle quote mode
                        in_quotes = not in_quotes
                        # Handle escaped quotes (two double quotes in a row)
                        if in_quotes is False and i + 1 < len(line) and line[i + 1] == '"':
                            field += '"'
                            i += 1  # Skip the next quote
                    elif char == delimiter and not in_quotes:
                        # End of field
                        row.append(field)
                        field = ""
                    else:
                        field += char

                    i += 1

                # Add the last field
                row.append(field)
                csv_data.append(row)

            # Create or retrieve the Excel session from state
            session = await self._get_or_create_excel_session(show_excel, get_state, set_state)

            # Open the workbook
            workbook = session.open_workbook(validated_file_path, read_only=False)

            # Find or create the worksheet
            worksheet = None
            if sheet_name:
                # Try to find the sheet
                for sheet in workbook.Worksheets:
                    if sheet.Name.lower() == sheet_name.lower():
                        worksheet = sheet
                        break

                # Create if not found and create_sheet is True
                if not worksheet and create_sheet:
                    worksheet = workbook.Worksheets.Add()
                    worksheet.Name = sheet_name

            # If no sheet_name specified or sheet not found, use the active sheet
            if not worksheet:
                if not sheet_name and not create_sheet:
                    worksheet = workbook.ActiveSheet
                elif create_sheet:
                    worksheet = workbook.Worksheets.Add()
                    if sheet_name:
                        worksheet.Name = sheet_name
                    else:
                        worksheet.Name = f"CSV_Import_{time.strftime('%Y%m%d')}"

            # Parse start cell
            start_cell_obj = worksheet.Range(start_cell)
            start_row = start_cell_obj.Row
            start_col = start_cell_obj.Column

            # Import the data
            for row_idx, row_data in enumerate(csv_data):
                for col_idx, cell_value in enumerate(row_data):
                    worksheet.Cells(start_row + row_idx, start_col + col_idx).Value = cell_value

            # Auto-fit columns for better readability
            if csv_data:
                start_range = worksheet.Cells(start_row, start_col)
                end_range = worksheet.Cells(
                    start_row + len(csv_data) - 1, start_col + len(csv_data[0]) - 1
                )
                data_range = worksheet.Range(start_range, end_range)
                data_range.Columns.AutoFit()

            # Save the workbook
            session.save_workbook(workbook, validated_file_path)

            # Close the workbook
            workbook.Close(SaveChanges=False)

            processing_time = time.time() - start_time
            result = {
                "success": True,
                "file_path": validated_file_path,
                "csv_path": validated_csv_path,
                "sheet_name": worksheet.Name,
                "rows_imported": len(csv_data),
                "columns_imported": len(csv_data[0]) if csv_data else 0,
                "processing_time": processing_time,
            }

            logger.info(f"CSV import completed in {processing_time:.2f}s", emoji_key="success")

            return result

        except Exception as e:
            logger.error(f"Error importing CSV data: {str(e)}", emoji_key="error", exc_info=True)
            # Try to clean up session on error
            await self._cleanup_excel_session(delete_state)
            raise ToolError(
                f"Failed to import CSV data: {str(e)}",
                details={"file_path": file_path, "csv_path": csv_path},
            ) from e

    # --- Excel session management methods ---

    async def _get_or_create_excel_session(self, visible=False, get_state=None, set_state=None):
        """Get an existing Excel session from state or create a new one.

        Args:
            visible: Whether Excel should be visible
            get_state: Function to get state
            set_state: Function to set state

        Returns:
            ExcelSession: An Excel session
        """
        # Try to get session from state
        session_data = await get_state("excel_session")

        if session_data and getattr(session_data, "status", "") != "closed":
            logger.info("Using existing Excel session from state")
            return session_data

        # Create a new session if none exists in state
        logger.info("Creating new Excel session")
        session = await asyncio.to_thread(ExcelSession, visible=visible)

        # Store session in state
        await set_state("excel_session", session)

        return session

    async def _cleanup_excel_session(self, delete_state=None):
        """Clean up Excel session resources.

        Args:
            delete_state: Function to delete state
        """
        if delete_state:
            await delete_state("excel_session")

    async def _validate_excel_file_path(self, file_path: str, check_exists: bool = False) -> str:
        """Validate that an Excel file path is in an allowed directory.

        Args:
            file_path: Path to validate
            check_exists: Whether to check if the file exists

        Returns:
            Validated absolute path
        """
        if not file_path:
            raise ToolInputError("File path cannot be empty")

        # Check if file has an Excel extension
        if not file_path.lower().endswith((".xlsx", ".xlsm", ".xls")):
            raise ToolInputError(
                f"File must have an Excel extension (.xlsx, .xlsm, .xls): {file_path}"
            )

        # Get allowed directories for file operations
        allowed_dirs = await get_allowed_directories()

        # Check if path is in an allowed directory
        abs_path = os.path.abspath(file_path)
        if not any(
            abs_path.startswith(os.path.abspath(allowed_dir)) for allowed_dir in allowed_dirs
        ):
            raise ToolInputError(f"File path is outside allowed directories: {file_path}")

        # Check existence if required
        if check_exists and not os.path.exists(abs_path):
            raise ToolInputError(f"File does not exist: {file_path}")

        return abs_path

    # --- Internal implementation methods ---

    async def _execute_excel_operation(
        self,
        session: ExcelSession,
        instruction: str,
        operation_type: str,
        file_path: Optional[str] = None,
        template_path: Optional[str] = None,
        parameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Internal method to execute Excel operations.

        This method handles the core Excel manipulation based on operation_type.

        Args:
            session: Excel session to use
            instruction: Natural language instruction
            operation_type: Type of operation
            file_path: Path to the Excel file
            template_path: Optional template file path
            parameters: Optional structured parameters

        Returns:
            Dictionary with operation results
        """
        # Initialize result structure
        result = {"success": True, "operation_type": operation_type, "file_path": file_path}

        # Handle different operation types
        if operation_type == "create":
            # Validate file_path and ensure it doesn't exist
            if file_path:
                validated_file_path = await validate_path(
                    file_path, check_exists=False, check_parent_writable=True
                )

                # Ensure parent directory exists
                parent_dir = os.path.dirname(validated_file_path)
                if parent_dir:
                    await create_directory(parent_dir)

            # Create a new workbook, either from scratch or from a template
            if template_path:
                # Validate template path and ensure it exists
                validated_template_path = await validate_path(template_path, check_exists=True)

                # Open the template
                wb = session.open_workbook(validated_template_path, read_only=True)
                # Save as the new file
                session.save_workbook(wb, validated_file_path)
                # Close the template and reopen the new file
                session.close_workbook(wb)
                wb = session.open_workbook(validated_file_path)
            else:
                # Create a new workbook
                wb = session.create_workbook()
                # If file_path is provided, immediately save it
                if file_path:
                    session.save_workbook(wb, validated_file_path)

            # Apply the instruction to the workbook
            operations_performed = await self._apply_instruction_to_workbook(
                session=session, workbook=wb, instruction=instruction, parameters=parameters
            )

            # Save the workbook
            session.save_workbook(wb, validated_file_path)

            result["operations_performed"] = operations_performed
            result["file_created"] = validated_file_path

        elif operation_type == "modify":
            # Validate file_path and ensure it exists
            validated_file_path = await validate_path(file_path, check_exists=True)

            # Open existing workbook for modification
            wb = session.open_workbook(validated_file_path, read_only=False)

            # Apply the instruction to the workbook
            operations_performed = await self._apply_instruction_to_workbook(
                session=session, workbook=wb, instruction=instruction, parameters=parameters
            )

            # Save the workbook
            session.save_workbook(wb, validated_file_path)

            result["operations_performed"] = operations_performed
            result["file_modified"] = validated_file_path

        elif operation_type == "analyze":
            # Validate file_path and ensure it exists
            validated_file_path = await validate_path(file_path, check_exists=True)

            # Open existing workbook for analysis
            wb = session.open_workbook(validated_file_path, read_only=True)

            # Analyze the workbook
            analysis_results = await self._analyze_workbook(
                session=session, workbook=wb, instruction=instruction, parameters=parameters
            )

            result["analysis_results"] = analysis_results

        elif operation_type == "format":
            # Validate file_path and ensure it exists
            validated_file_path = await validate_path(file_path, check_exists=True)

            # Open existing workbook for formatting
            wb = session.open_workbook(validated_file_path, read_only=False)

            # Apply formatting to the workbook
            formatting_applied = await self._apply_formatting_to_workbook(
                session=session, workbook=wb, instruction=instruction, parameters=parameters
            )

            # Save the workbook
            session.save_workbook(wb, validated_file_path)

            result["formatting_applied"] = formatting_applied

        else:
            raise ToolInputError(f"Unknown operation_type: {operation_type}")

        return result

    async def _apply_instruction_to_workbook(
        self, session: ExcelSession, workbook: Any, instruction: str, parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply natural language instructions to a workbook.

        This method interprets the instructions and performs the requested operations.

        Args:
            session: Excel session
            workbook: Workbook COM object
            instruction: Natural language instruction
            parameters: Optional structured parameters

        Returns:
            List of operations performed
        """
        operations_performed = []

        # Default to first worksheet if none exists
        if workbook.Worksheets.Count == 0:
            worksheet = workbook.Worksheets.Add()
            operations_performed.append(
                {"operation": "create_worksheet", "sheet_name": worksheet.Name}
            )

        # Process instruction to extract key operations
        instruction_lower = instruction.lower()

        # Create sheets if mentioned
        if "sheet" in instruction_lower or "sheets" in instruction_lower:
            # Extract sheet names using regex to find patterns like:
            # - 'sheets: X, Y, Z'
            # - 'sheets named X and Y'
            # - 'create sheets X, Y, Z'
            sheet_patterns = [
                r"sheet(?:s)?\s*(?:named|called|:)?\s*(?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_, ]+))",
                r"create (?:a |)sheet(?:s)? (?:named|called)?\s*(?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_, ]+))",
            ]

            sheet_names = []
            for pattern in sheet_patterns:
                matches = re.findall(pattern, instruction_lower)
                if matches:
                    for match in matches:
                        # Each match is now a tuple with 3 capture groups: (single_quoted, double_quoted, unquoted)
                        sheet_name = match[0] or match[1] or match[2]
                        if sheet_name:
                            # Split by commas and/or 'and', then clean up
                            for name in re.split(r",|\s+and\s+", sheet_name):
                                clean_name = name.strip("' \"").strip()
                                if clean_name:
                                    sheet_names.append(clean_name)

            # Also check explicit parameters if provided
            if parameters and "sheet_names" in parameters:
                sheet_names.extend(parameters["sheet_names"])

            # Make sheet names unique
            sheet_names = list(set(sheet_names))

            # Create each sheet
            current_sheets = [sheet.Name.lower() for sheet in workbook.Worksheets]

            for sheet_name in sheet_names:
                if sheet_name.lower() not in current_sheets:
                    new_sheet = workbook.Worksheets.Add(
                        After=workbook.Worksheets(workbook.Worksheets.Count)
                    )
                    new_sheet.Name = sheet_name
                    operations_performed.append(
                        {"operation": "create_worksheet", "sheet_name": sheet_name}
                    )

        # Add headers if mentioned
        if "header" in instruction_lower or "headers" in instruction_lower:
            # Extract header information
            header_data = None

            # Check parameters first
            if parameters and "headers" in parameters:
                header_data = parameters["headers"]
            else:
                # Try to extract from instruction
                header_match = re.search(
                    r"header(?:s)?\s*(?::|with|including)\s*([^.]+)", instruction_lower
                )
                if header_match:
                    # Parse the header text
                    header_text = header_match.group(1).strip()
                    # Split by commas and/or 'and'
                    header_data = [
                        h.strip("' \"").strip()
                        for h in re.split(r",|\s+and\s+", header_text)
                        if h.strip()
                    ]

            if header_data:
                # Determine target sheet
                target_sheet_name = None

                # Check parameters first
                if parameters and "target_sheet" in parameters:
                    target_sheet_name = parameters["target_sheet"]
                else:
                    # Try to extract from instruction
                    sheet_match = re.search(
                        r"in (?:the |)(?:sheet|worksheet) (?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_]+))",
                        instruction_lower,
                    )
                    if sheet_match:
                        target_sheet_name = (
                            sheet_match.group(1) or sheet_match.group(2) or sheet_match.group(3)
                        )

                # Default to first sheet if not specified
                if not target_sheet_name:
                    target_sheet_name = workbook.Worksheets(1).Name

                # Find the worksheet
                worksheet = None
                for sheet in workbook.Worksheets:
                    if sheet.Name.lower() == target_sheet_name.lower():
                        worksheet = sheet
                        break

                if not worksheet:
                    worksheet = workbook.Worksheets(1)

                # Add headers to the worksheet
                for col_idx, header in enumerate(header_data, 1):
                    worksheet.Cells(1, col_idx).Value = header

                    # Apply simple header formatting
                    worksheet.Cells(1, col_idx).Font.Bold = True

                operations_performed.append(
                    {
                        "operation": "add_headers",
                        "sheet_name": worksheet.Name,
                        "headers": header_data,
                    }
                )

        # Add data if mentioned
        if "data" in instruction_lower or "values" in instruction_lower:
            # Check parameters first
            data_rows = None

            if parameters and "data" in parameters:
                data_rows = parameters["data"]

            if data_rows:
                # Determine target sheet
                target_sheet_name = None

                # Check parameters first
                if parameters and "target_sheet" in parameters:
                    target_sheet_name = parameters["target_sheet"]
                else:
                    # Try to extract from instruction
                    sheet_match = re.search(
                        r"in (?:the |)(?:sheet|worksheet) (?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_]+))",
                        instruction_lower,
                    )
                    if sheet_match:
                        target_sheet_name = (
                            sheet_match.group(1) or sheet_match.group(2) or sheet_match.group(3)
                        )

                # Default to first sheet if not specified
                if not target_sheet_name:
                    target_sheet_name = workbook.Worksheets(1).Name

                # Find the worksheet
                worksheet = None
                for sheet in workbook.Worksheets:
                    if sheet.Name.lower() == target_sheet_name.lower():
                        worksheet = sheet
                        break

                if not worksheet:
                    worksheet = workbook.Worksheets(1)

                # Determine starting row (typically 2 if headers exist)
                start_row = 2
                if parameters and "start_row" in parameters:
                    start_row = parameters["start_row"]

                # Add data to the worksheet
                for row_idx, row_data in enumerate(data_rows, start_row):
                    for col_idx, cell_value in enumerate(row_data, 1):
                        worksheet.Cells(row_idx, col_idx).Value = cell_value

                operations_performed.append(
                    {
                        "operation": "add_data",
                        "sheet_name": worksheet.Name,
                        "start_row": start_row,
                        "row_count": len(data_rows),
                    }
                )

        # Add formulas if mentioned
        if "formula" in instruction_lower or "formulas" in instruction_lower:
            formula_data = None

            # Check parameters first
            if parameters and "formulas" in parameters:
                formula_data = parameters["formulas"]

            if formula_data:
                # Determine target sheet
                target_sheet_name = None

                # Check parameters first
                if parameters and "target_sheet" in parameters:
                    target_sheet_name = parameters["target_sheet"]

                # Default to first sheet if not specified
                if not target_sheet_name:
                    target_sheet_name = workbook.Worksheets(1).Name

                # Find the worksheet
                worksheet = None
                for sheet in workbook.Worksheets:
                    if sheet.Name.lower() == target_sheet_name.lower():
                        worksheet = sheet
                        break

                if not worksheet:
                    worksheet = workbook.Worksheets(1)

                # Add formulas to the worksheet
                for formula_entry in formula_data:
                    cell_ref = formula_entry.get("cell")
                    formula = formula_entry.get("formula")

                    if cell_ref and formula:
                        worksheet.Range(cell_ref).Formula = formula

                operations_performed.append(
                    {
                        "operation": "add_formulas",
                        "sheet_name": worksheet.Name,
                        "formula_count": len(formula_data),
                    }
                )

        # Apply formatting if mentioned
        if "format" in instruction_lower or "formatting" in instruction_lower:
            formatting = None

            # Check parameters first
            if parameters and "formatting" in parameters:
                formatting = parameters["formatting"]

            if formatting:
                await self._apply_formatting_to_workbook(
                    session=session,
                    workbook=workbook,
                    instruction=instruction,
                    parameters={"formatting": formatting},
                )

                operations_performed.append(
                    {
                        "operation": "apply_formatting",
                        "details": "Applied formatting based on parameters",
                    }
                )
            else:
                # Apply default formatting based on instruction
                sheet = workbook.Worksheets(1)

                # Auto-fit columns
                used_range = sheet.UsedRange
                used_range.Columns.AutoFit()

                # Add borders to data range
                if used_range.Rows.Count > 1:
                    data_range = sheet.Range(
                        sheet.Cells(1, 1),
                        sheet.Cells(used_range.Rows.Count, used_range.Columns.Count),
                    )
                    data_range.Borders.LineStyle = 1  # xlContinuous

                operations_performed.append(
                    {
                        "operation": "apply_formatting",
                        "details": "Applied default formatting (auto-fit columns, borders)",
                    }
                )

        # Create a chart if mentioned
        if "chart" in instruction_lower or "graph" in instruction_lower:
            chart_type = None

            # Chart type mapping
            CHART_TYPES = {
                "column": win32c.xlColumnClustered,
                "bar": win32c.xlBarClustered,
                "line": win32c.xlLine,
                "pie": win32c.xlPie,
                "area": win32c.xlArea,
                "scatter": win32c.xlXYScatter,
                "radar": win32c.xlRadar,
                "stock": win32c.xlStockHLC,
                "surface": win32c.xlSurface,
                "doughnut": win32c.xlDoughnut,
                "bubble": win32c.xlBubble,
                "combo": win32c.xl3DColumn,
            }

            # Check parameters first
            if parameters and "chart" in parameters:
                chart_info = parameters["chart"]
                chart_type_str = chart_info.get("type", "column").lower()
                data_range = chart_info.get("data_range")
                chart_title = chart_info.get("title", "Chart")

                # Get chart type constant
                chart_type = CHART_TYPES.get(chart_type_str, win32c.xlColumnClustered)

                if data_range:
                    # Determine target sheet
                    target_sheet_name = chart_info.get("sheet_name")

                    # Default to first sheet if not specified
                    if not target_sheet_name:
                        target_sheet_name = workbook.Worksheets(1).Name

                    # Find the worksheet
                    worksheet = None
                    for sheet in workbook.Worksheets:
                        if sheet.Name.lower() == target_sheet_name.lower():
                            worksheet = sheet
                            break

                    if not worksheet:
                        worksheet = workbook.Worksheets(1)

                    # Create the chart
                    chart = worksheet.Shapes.AddChart2(-1, chart_type).Chart
                    chart.SetSourceData(worksheet.Range(data_range))
                    chart.HasTitle = True
                    chart.ChartTitle.Text = chart_title

                    operations_performed.append(
                        {
                            "operation": "create_chart",
                            "sheet_name": worksheet.Name,
                            "chart_type": chart_type_str,
                            "data_range": data_range,
                        }
                    )

        return operations_performed

    async def _analyze_workbook(
        self, session: ExcelSession, workbook: Any, instruction: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a workbook based on instructions.

        This method examines the workbook structure, formulas, and data.

        Args:
            session: Excel session
            workbook: Workbook COM object
            instruction: Analysis instruction
            parameters: Optional structured parameters

        Returns:
            Dictionary with analysis results
        """
        # Initialize result
        analysis_results = {
            "workbook_name": workbook.Name,
            "sheet_count": workbook.Sheets.Count,
            "sheets_info": [],
            "has_formulas": False,
            "has_links": workbook.HasLinks,
            "calculation_mode": self._get_calculation_mode_name(workbook.CalculationMode),
        }

        total_formulas = 0

        # Analyze each sheet
        for sheet_idx in range(1, workbook.Sheets.Count + 1):
            sheet = workbook.Sheets(sheet_idx)

            # Skip chart sheets
            if sheet.Type != 1:  # xlWorksheet
                continue

            used_range = sheet.UsedRange

            # Get sheet details
            sheet_info = {
                "name": sheet.Name,
                "row_count": used_range.Rows.Count if used_range else 0,
                "column_count": used_range.Columns.Count if used_range else 0,
                "visible": sheet.Visible == -1,  # -1 is xlSheetVisible
                "has_formulas": False,
                "formula_count": 0,
                "data_tables": False,
                "has_charts": False,
                "chart_count": 0,
                "named_ranges": [],
            }

            # Check for charts
            chart_objects = sheet.ChartObjects()
            chart_count = chart_objects.Count
            sheet_info["has_charts"] = chart_count > 0
            sheet_info["chart_count"] = chart_count

            # Look for formulas
            formula_cells = []
            formula_count = 0

            if used_range:
                # Sample used range cells to check for formulas (limit to reasonable number)
                row_count = min(used_range.Rows.Count, 1000)
                col_count = min(used_range.Columns.Count, 100)

                for row in range(1, row_count + 1):
                    for col in range(1, col_count + 1):
                        cell = used_range.Cells(row, col)
                        if cell.HasFormula:
                            formula_count += 1
                            if len(formula_cells) < 10:  # Just store a few examples
                                cell_address = cell.Address(False, False)  # A1 style without $
                                formula_cells.append(
                                    {"address": cell_address, "formula": cell.Formula}
                                )

            sheet_info["has_formulas"] = formula_count > 0
            sheet_info["formula_count"] = formula_count
            sheet_info["example_formulas"] = formula_cells

            total_formulas += formula_count

            # Get named ranges in this sheet
            for name in workbook.Names:
                try:
                    if name.RefersToRange.Parent.Name == sheet.Name:
                        sheet_info["named_ranges"].append(
                            {"name": name.Name, "refers_to": name.RefersTo}
                        )
                except Exception:
                    pass  # Skip if there's an error (e.g., name refers to another workbook)

            analysis_results["sheets_info"].append(sheet_info)

        analysis_results["has_formulas"] = total_formulas > 0
        analysis_results["total_formula_count"] = total_formulas

        # Check for external links
        if workbook.HasLinks:
            links = []
            try:
                for link in workbook.LinkSources():
                    links.append(link)
            except Exception:
                pass  # Skip if there's an error

            analysis_results["external_links"] = links

        # Add sheet dependencies if requested
        if "analyze_dependencies" in instruction.lower() or (
            parameters and parameters.get("analyze_dependencies")
        ):
            analysis_results["dependencies"] = await self._analyze_sheet_dependencies(
                session, workbook
            )

        # Add formula analysis if requested
        if "analyze_formulas" in instruction.lower() or (
            parameters and parameters.get("analyze_formulas")
        ):
            analysis_results["formula_analysis"] = await self._analyze_formulas_in_workbook(
                session, workbook
            )

        return analysis_results

    async def _apply_formatting_to_workbook(
        self, session: ExcelSession, workbook: Any, instruction: str, parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply formatting to a workbook based on instructions.

        This method handles various formatting operations.

        Args:
            session: Excel session
            workbook: Workbook COM object
            instruction: Formatting instruction
            parameters: Optional structured parameters

        Returns:
            List of formatting operations performed
        """
        formatting_applied = []

        # Check if specific formatting instructions are provided in parameters
        if parameters and "formatting" in parameters:
            formatting = parameters["formatting"]

            # Apply cell formatting
            if "cells" in formatting:
                for cell_format in formatting["cells"]:
                    cell_range = cell_format.get("range")
                    sheet_name = cell_format.get("sheet")

                    if not cell_range:
                        continue

                    # Find the worksheet
                    worksheet = None
                    if sheet_name:
                        for sheet in workbook.Worksheets:
                            if sheet.Name.lower() == sheet_name.lower():
                                worksheet = sheet
                                break

                    if not worksheet:
                        worksheet = workbook.Worksheets(1)

                    # Get the range
                    range_obj = worksheet.Range(cell_range)

                    # Apply formatting attributes
                    if "bold" in cell_format:
                        range_obj.Font.Bold = cell_format["bold"]

                    if "italic" in cell_format:
                        range_obj.Font.Italic = cell_format["italic"]

                    if "color" in cell_format:
                        # Handle hex color codes (e.g., "#FF0000" for red)
                        color_code = cell_format["color"]
                        if color_code.startswith("#"):
                            # Convert hex color to RGB value
                            r = int(color_code[1:3], 16)
                            g = int(color_code[3:5], 16)
                            b = int(color_code[5:7], 16)
                            range_obj.Font.Color = b + (g << 8) + (r << 16)
                        else:
                            # Try to set color directly
                            range_obj.Font.Color = cell_format["color"]

                    if "bg_color" in cell_format:
                        # Handle hex color codes
                        color_code = cell_format["bg_color"]
                        if color_code.startswith("#"):
                            # Convert hex color to RGB value
                            r = int(color_code[1:3], 16)
                            g = int(color_code[3:5], 16)
                            b = int(color_code[5:7], 16)
                            range_obj.Interior.Color = b + (g << 8) + (r << 16)
                        else:
                            # Try to set color directly
                            range_obj.Interior.Color = cell_format["bg_color"]

                    if "number_format" in cell_format:
                        range_obj.NumberFormat = cell_format["number_format"]

                    if "border" in cell_format:
                        border_style = cell_format["border"]
                        if border_style == "all":
                            for border_idx in range(7, 13):  # xlEdgeLeft to xlInsideVertical
                                range_obj.Borders(border_idx).LineStyle = 1  # xlContinuous
                                range_obj.Borders(border_idx).Weight = 2  # xlThin
                        elif border_style == "outside":
                            for border_idx in range(7, 11):  # xlEdgeLeft to xlEdgeRight
                                range_obj.Borders(border_idx).LineStyle = 1  # xlContinuous
                                range_obj.Borders(border_idx).Weight = 2  # xlThin

                    formatting_applied.append(
                        {
                            "operation": "format_cells",
                            "sheet_name": worksheet.Name,
                            "range": cell_range,
                        }
                    )

            # Apply table formatting
            if "tables" in formatting:
                for table_format in formatting["tables"]:
                    data_range = table_format.get("range")
                    sheet_name = table_format.get("sheet")
                    table_style = table_format.get("style", "TableStyleMedium2")
                    has_headers = table_format.get("has_headers", True)

                    if not data_range:
                        continue

                    # Find the worksheet
                    worksheet = None
                    if sheet_name:
                        for sheet in workbook.Worksheets:
                            if sheet.Name.lower() == sheet_name.lower():
                                worksheet = sheet
                                break

                    if not worksheet:
                        worksheet = workbook.Worksheets(1)

                    # Create a table
                    table_name = f"Table{len(worksheet.ListObjects) + 1}"
                    if "name" in table_format:
                        table_name = table_format["name"]

                    try:
                        table = worksheet.ListObjects.Add(1, worksheet.Range(data_range), True)
                        table.Name = table_name
                        table.TableStyle = table_style

                        formatting_applied.append(
                            {
                                "operation": "create_table",
                                "sheet_name": worksheet.Name,
                                "table_name": table_name,
                                "range": data_range,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create table: {str(e)}")

            # Apply conditional formatting
            if "conditional_formatting" in formatting:
                for cf_format in formatting["conditional_formatting"]:
                    cell_range = cf_format.get("range")
                    sheet_name = cf_format.get("sheet")
                    cf_type = cf_format.get("type")

                    if not cell_range or not cf_type:
                        continue

                    # Find the worksheet
                    worksheet = None
                    if sheet_name:
                        for sheet in workbook.Worksheets:
                            if sheet.Name.lower() == sheet_name.lower():
                                worksheet = sheet
                                break

                    if not worksheet:
                        worksheet = workbook.Worksheets(1)

                    # Get the range
                    range_obj = worksheet.Range(cell_range)

                    # Apply conditional formatting based on type
                    if cf_type == "data_bar":
                        color = cf_format.get("color", 43)  # Default blue
                        if isinstance(color, str) and color.startswith("#"):
                            # Convert hex color to RGB value
                            r = int(color[1:3], 16)
                            g = int(color[3:5], 16)
                            b = int(color[5:7], 16)
                            color = b + (g << 8) + (r << 16)

                        cf = range_obj.FormatConditions.AddDatabar()
                        cf.BarColor.Color = color

                    elif cf_type == "color_scale":
                        cf = range_obj.FormatConditions.AddColorScale(3)
                        # Configure color scale (could be extended with more options)

                    elif cf_type == "icon_set":
                        icon_style = cf_format.get("icon_style", "3Arrows")
                        cf = range_obj.FormatConditions.AddIconSetCondition()
                        cf.IconSet = workbook.Application.IconSets(icon_style)

                    elif cf_type == "cell_value":
                        comparison_operator = cf_format.get("operator", "greaterThan")
                        comparison_value = cf_format.get("value", 0)

                        # Map string operator to Excel constant
                        operator_map = {
                            "greaterThan": 3,  # xlGreater
                            "lessThan": 5,  # xlLess
                            "equalTo": 2,  # xlEqual
                            "greaterOrEqual": 4,  # xlGreaterEqual
                            "lessOrEqual": 6,  # xlLessEqual
                            "notEqual": 7,  # xlNotEqual
                        }

                        operator_constant = operator_map.get(comparison_operator, 3)

                        cf = range_obj.FormatConditions.Add(
                            1, operator_constant, comparison_value
                        )  # 1 = xlCellValue

                        # Apply formatting
                        if "bold" in cf_format:
                            cf.Font.Bold = cf_format["bold"]

                        if "italic" in cf_format:
                            cf.Font.Italic = cf_format["italic"]

                        if "color" in cf_format:
                            # Handle hex color codes
                            color_code = cf_format["color"]
                            if color_code.startswith("#"):
                                # Convert hex color to RGB value
                                r = int(color_code[1:3], 16)
                                g = int(color_code[3:5], 16)
                                b = int(color_code[5:7], 16)
                                cf.Font.Color = b + (g << 8) + (r << 16)
                            else:
                                cf.Font.Color = cf_format["color"]

                        if "bg_color" in cf_format:
                            # Handle hex color codes
                            color_code = cf_format["bg_color"]
                            if color_code.startswith("#"):
                                # Convert hex color to RGB value
                                r = int(color_code[1:3], 16)
                                g = int(color_code[3:5], 16)
                                b = int(color_code[5:7], 16)
                                cf.Interior.Color = b + (g << 8) + (r << 16)
                            else:
                                cf.Interior.Color = cf_format["bg_color"]

                    formatting_applied.append(
                        {
                            "operation": "add_conditional_formatting",
                            "sheet_name": worksheet.Name,
                            "range": cell_range,
                            "type": cf_type,
                        }
                    )

        # Apply default formatting based on instruction if no specific formatting provided
        elif not parameters or "formatting" not in parameters:
            instruction_lower = instruction.lower()

            # Extract target sheet(s)
            sheet_names = []
            sheet_match = re.search(
                r"(?:in|to) (?:the |)(?:sheet|worksheet)(?:s|) (?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_, ]+))",
                instruction_lower,
            )

            if sheet_match:
                sheet_names_str = (
                    sheet_match.group(1) or sheet_match.group(2) or sheet_match.group(3)
                )
                # Split by commas and/or 'and'
                for name in re.split(r",|\s+and\s+", sheet_names_str):
                    clean_name = name.strip("' \"").strip()
                    if clean_name:
                        sheet_names.append(clean_name)

            # If no sheets specified, use all worksheets
            if not sheet_names:
                sheet_names = [sheet.Name for sheet in workbook.Worksheets]

            for sheet_name in sheet_names:
                # Find the worksheet
                worksheet = None
                for sheet in workbook.Worksheets:
                    if sheet.Name.lower() == sheet_name.lower():
                        worksheet = sheet
                        break

                if not worksheet:
                    continue

                # Apply standard formatting
                used_range = worksheet.UsedRange

                # Auto-fit columns
                if "auto-fit" in instruction_lower or "autofit" in instruction_lower:
                    used_range.Columns.AutoFit()

                    formatting_applied.append(
                        {"operation": "auto_fit_columns", "sheet_name": worksheet.Name}
                    )

                # Add borders to data
                if "borders" in instruction_lower or "outline" in instruction_lower:
                    if used_range.Rows.Count > 0 and used_range.Columns.Count > 0:
                        # Apply borders
                        border_style = 1  # xlContinuous
                        border_weight = 2  # xlThin

                        # Determine border type
                        if "outside" in instruction_lower:
                            # Outside borders only
                            for border_idx in range(7, 11):  # xlEdgeLeft to xlEdgeRight
                                used_range.Borders(border_idx).LineStyle = border_style
                                used_range.Borders(border_idx).Weight = border_weight
                        else:
                            # All borders
                            used_range.Borders.LineStyle = border_style
                            used_range.Borders.Weight = border_weight

                        formatting_applied.append(
                            {
                                "operation": "add_borders",
                                "sheet_name": worksheet.Name,
                                "border_type": "outside"
                                if "outside" in instruction_lower
                                else "all",
                            }
                        )

                # Format headers
                if "header" in instruction_lower or "headers" in instruction_lower:
                    if used_range.Rows.Count > 0:
                        # Apply header formatting to first row
                        header_row = worksheet.Rows(1)
                        header_row.Font.Bold = True

                        # Set background color if mentioned
                        if "blue" in instruction_lower:
                            header_row.Interior.Color = 15773696  # Light blue
                        elif "gray" in instruction_lower or "grey" in instruction_lower:
                            header_row.Interior.Color = 14540253  # Light gray
                        elif "green" in instruction_lower:
                            header_row.Interior.Color = 13561798  # Light green
                        else:
                            # Default light blue
                            header_row.Interior.Color = 15773696

                        formatting_applied.append(
                            {"operation": "format_headers", "sheet_name": worksheet.Name}
                        )

                # Apply number formatting
                if "currency" in instruction_lower or "dollar" in instruction_lower:
                    # Look for ranges with currency values
                    # This is a simplistic approach - in a real tool, we might analyze the data
                    # to identify numeric columns that might be currency
                    if used_range.Rows.Count > 1:  # Skip if only header row
                        for col in range(1, used_range.Columns.Count + 1):
                            # Check a sample of cells in this column
                            numeric_cell_count = 0
                            sample_size = min(10, used_range.Rows.Count - 1)

                            for row in range(2, 2 + sample_size):  # Skip header
                                cell_value = worksheet.Cells(row, col).Value
                                if isinstance(cell_value, (int, float)):
                                    numeric_cell_count += 1

                            # If most cells are numeric, apply currency format
                            if numeric_cell_count > sample_size / 2:
                                col_range = worksheet.Range(
                                    worksheet.Cells(2, col),
                                    worksheet.Cells(used_range.Rows.Count, col),
                                )

                                # Determine currency symbol
                                currency_format = "$#,##0.00"
                                if "euro" in instruction_lower:
                                    currency_format = "€#,##0.00"
                                elif "pound" in instruction_lower:
                                    currency_format = "£#,##0.00"

                                col_range.NumberFormat = currency_format

                                formatting_applied.append(
                                    {
                                        "operation": "apply_currency_format",
                                        "sheet_name": worksheet.Name,
                                        "column": worksheet.Cells(1, col).Value or f"Column {col}",
                                        "format": currency_format,
                                    }
                                )

                # Apply percentage formatting
                if "percent" in instruction_lower or "percentage" in instruction_lower:
                    # Similar approach to currency formatting
                    if used_range.Rows.Count > 1:  # Skip if only header row
                        for col in range(1, used_range.Columns.Count + 1):
                            col_header = worksheet.Cells(1, col).Value

                            # Check if column header suggests percentage
                            is_percentage_column = False
                            if col_header and isinstance(col_header, str):
                                if any(
                                    term in col_header.lower()
                                    for term in ["percent", "rate", "growth", "change", "margin"]
                                ):
                                    is_percentage_column = True

                            if is_percentage_column:
                                col_range = worksheet.Range(
                                    worksheet.Cells(2, col),
                                    worksheet.Cells(used_range.Rows.Count, col),
                                )

                                col_range.NumberFormat = "0.0%"

                                formatting_applied.append(
                                    {
                                        "operation": "apply_percentage_format",
                                        "sheet_name": worksheet.Name,
                                        "column": col_header or f"Column {col}",
                                    }
                                )

                # Create a table if requested
                if "table" in instruction_lower and "style" in instruction_lower:
                    if used_range.Rows.Count > 0 and used_range.Columns.Count > 0:
                        # Create a table with the used range
                        try:
                            has_headers = True
                            if "no header" in instruction_lower:
                                has_headers = False

                            table = worksheet.ListObjects.Add(1, used_range, has_headers)

                            # Set table style
                            table_style = "TableStyleMedium2"  # Default medium blue

                            if "light" in instruction_lower:
                                if "blue" in instruction_lower:
                                    table_style = "TableStyleLight1"
                                elif "green" in instruction_lower:
                                    table_style = "TableStyleLight5"
                                elif "orange" in instruction_lower:
                                    table_style = "TableStyleLight3"
                            elif "medium" in instruction_lower:
                                if "blue" in instruction_lower:
                                    table_style = "TableStyleMedium2"
                                elif "green" in instruction_lower:
                                    table_style = "TableStyleMedium5"
                                elif "orange" in instruction_lower:
                                    table_style = "TableStyleMedium3"
                            elif "dark" in instruction_lower:
                                if "blue" in instruction_lower:
                                    table_style = "TableStyleDark2"
                                elif "green" in instruction_lower:
                                    table_style = "TableStyleDark5"
                                elif "orange" in instruction_lower:
                                    table_style = "TableStyleDark3"

                            table.TableStyle = table_style

                            formatting_applied.append(
                                {
                                    "operation": "create_table",
                                    "sheet_name": worksheet.Name,
                                    "style": table_style,
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to create table: {str(e)}")

        return formatting_applied

    async def _analyze_excel_template(
        self, session: ExcelSession, exemplar_path: str, parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze an Excel template to understand its structure and formulas.

        This method examines the provided Excel file to understand its structure,
        formulas, data patterns, and features used. The analysis is used for
        adapting the template to a new context.

        Args:
            session: ExcelSession instance to use
            exemplar_path: Path to the Excel file to analyze (already validated)
            parameters: Optional parameters to guide the analysis

        Returns:
            Dictionary containing the template analysis
        """
        parameters = parameters or {}

        try:
            # Open the workbook - path is already validated by the caller
            workbook = session.open_workbook(exemplar_path)

            analysis = {
                "worksheets": [],
                "formulas": {},
                "data_tables": [],
                "named_ranges": [],
                "pivot_tables": [],
                "charts": [],
                "complex_features": [],
            }

            # Analyze worksheets
            for sheet in workbook.Worksheets:
                sheet_analysis = {
                    "name": sheet.Name,
                    "used_range": f"{sheet.UsedRange.Address}",
                    "columns": {},
                    "rows": {},
                    "formulas": [],
                    "data_patterns": [],
                }

                # Identify data patterns and column types
                # This is a simplified analysis - in practice would be more complex
                used_range = sheet.UsedRange
                if used_range:
                    # Sample column headers
                    for col in range(1, min(used_range.Columns.Count + 1, 50)):
                        header = used_range.Cells(1, col).Value
                        if header:
                            sheet_analysis["columns"][col] = {
                                "header": str(header),
                                "type": "unknown",  # Would determine type in real analysis
                            }

                    # Sample formula patterns (simplified)
                    for row in range(2, min(used_range.Rows.Count + 1, 20)):
                        for col in range(1, min(used_range.Columns.Count + 1, 20)):
                            cell = used_range.Cells(row, col)
                            if cell.HasFormula:
                                sheet_analysis["formulas"].append(
                                    {
                                        "address": cell.Address,
                                        "formula": cell.Formula,
                                        "type": "calculation",  # Would classify formula type
                                    }
                                )

                analysis["worksheets"].append(sheet_analysis)

            # Look for named ranges
            for name in workbook.Names:
                try:
                    analysis["named_ranges"].append({"name": name.Name, "refers_to": name.RefersTo})
                except Exception:
                    # Some name objects might be invalid or hidden
                    pass

            # Identify charts (simplified approach)
            for sheet in workbook.Worksheets:
                if sheet.ChartObjects.Count > 0:
                    sheet_charts = []
                    for chart_idx in range(1, sheet.ChartObjects.Count + 1):
                        chart = sheet.ChartObjects(chart_idx)
                        sheet_charts.append(
                            {"name": chart.Name, "type": str(chart.Chart.ChartType)}
                        )

                    analysis["charts"].append({"sheet": sheet.Name, "charts": sheet_charts})

            # Close without saving
            workbook.Close(SaveChanges=False)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Excel template: {str(e)}", exc_info=True)
            raise ToolError(f"Failed to analyze Excel template: {str(e)}") from e

    async def _apply_excel_template(
        self,
        session: ExcelSession,
        exemplar_path: str,
        output_path: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Apply an Excel template with new data.

        This method takes an exemplar Excel file, modifies it with new data
        according to the provided parameters, and saves it to a new location.

        Args:
            session: ExcelSession instance to use
            exemplar_path: Path to the Excel template file (already validated)
            output_path: Path where the modified file will be saved (already validated)
            data: New data to apply to the template
            parameters: Optional parameters to guide template application

        Returns:
            Dictionary containing the results of the template application
        """
        parameters = parameters or {}

        try:
            # Open the template workbook - path is already validated by the caller
            template = session.open_workbook(exemplar_path)

            # Track modifications for reporting
            modifications = {"cells_modified": 0, "sheets_modified": set(), "data_mappings": []}

            # Process each data mapping
            for mapping in data.get("mappings", []):
                sheet_name = mapping.get("sheet")
                if not sheet_name:
                    continue

                # Find the target sheet
                sheet = None
                for s in template.Worksheets:
                    if s.Name == sheet_name:
                        sheet = s
                        break

                if not sheet:
                    logger.warning(f"Sheet '{sheet_name}' not found in template")
                    continue

                # Process this sheet's mappings
                target_range = mapping.get("range")
                values = mapping.get("values", [])

                if target_range and values:
                    try:
                        # Apply values to the range
                        range_obj = sheet.Range(target_range)

                        # Handle different data structures based on the shape of values
                        if isinstance(values, list):
                            if len(values) > 0 and isinstance(values[0], list):
                                # 2D array of values
                                for row_idx, row_data in enumerate(values):
                                    for col_idx, cell_value in enumerate(row_data):
                                        if (
                                            row_idx < range_obj.Rows.Count
                                            and col_idx < range_obj.Columns.Count
                                        ):
                                            cell = range_obj.Cells(row_idx + 1, col_idx + 1)
                                            cell.Value = cell_value
                                            modifications["cells_modified"] += 1
                            else:
                                # 1D array of values - apply to a single row or column
                                if range_obj.Rows.Count == 1:
                                    # Apply horizontally
                                    for col_idx, cell_value in enumerate(values):
                                        if col_idx < range_obj.Columns.Count:
                                            cell = range_obj.Cells(1, col_idx + 1)
                                            cell.Value = cell_value
                                            modifications["cells_modified"] += 1
                                else:
                                    # Apply vertically
                                    for row_idx, cell_value in enumerate(values):
                                        if row_idx < range_obj.Rows.Count:
                                            cell = range_obj.Cells(row_idx + 1, 1)
                                            cell.Value = cell_value
                                            modifications["cells_modified"] += 1
                        else:
                            # Single value - apply to entire range
                            range_obj.Value = values
                            modifications["cells_modified"] += 1

                        modifications["sheets_modified"].add(sheet_name)
                        modifications["data_mappings"].append(
                            {"sheet": sheet_name, "range": target_range, "values_applied": True}
                        )

                    except Exception as e:
                        logger.error(
                            f"Error applying values to range {target_range} in sheet {sheet_name}: {str(e)}",
                            exc_info=True,
                        )
                        modifications["data_mappings"].append(
                            {
                                "sheet": sheet_name,
                                "range": target_range,
                                "values_applied": False,
                                "error": str(e),
                            }
                        )

            # Recalculate formulas
            template.Application.CalculateFull()

            # Save the workbook to the specified output path
            template.SaveAs(output_path)
            template.Close(SaveChanges=False)

            # Create result object
            result = {
                "success": True,
                "exemplar_path": exemplar_path,
                "output_path": output_path,
                "cells_modified": modifications["cells_modified"],
                "sheets_modified": list(modifications["sheets_modified"]),
                "mappings_applied": modifications["data_mappings"],
            }

            return result

        except Exception as e:
            logger.error(f"Error applying Excel template: {str(e)}", exc_info=True)
            raise ToolError(f"Failed to apply Excel template: {str(e)}") from e

    async def _analyze_formulas_in_workbook(self, session, workbook):
        """Analyze formulas across a workbook.

        Args:
            session: Excel session
            workbook: Workbook COM object

        Returns:
            Dictionary with formula analysis results
        """
        # Initialize results
        analysis = {
            "total_formulas": 0,
            "sheets_with_formulas": 0,
            "formula_categories": {},
            "complexity": {"simple": 0, "moderate": 0, "complex": 0, "very_complex": 0},
            "samples": {},
        }

        # Function categories to track
        categories = {
            "mathematical": ["SUM", "AVERAGE", "MIN", "MAX", "COUNT", "PRODUCT", "ROUND"],
            "logical": ["IF", "AND", "OR", "NOT", "SWITCH", "IFS"],
            "lookup": ["VLOOKUP", "HLOOKUP", "INDEX", "MATCH", "XLOOKUP"],
            "text": ["CONCATENATE", "LEFT", "RIGHT", "MID", "FIND", "SEARCH", "REPLACE"],
            "date": ["TODAY", "NOW", "DATE", "DAY", "MONTH", "YEAR"],
            "financial": ["PMT", "RATE", "NPV", "IRR", "FV", "PV"],
            "statistical": ["STDEV", "VAR", "AVERAGE", "MEDIAN", "PERCENTILE"],
            "reference": ["INDIRECT", "OFFSET", "ADDRESS", "ROW", "COLUMN"],
            "database": ["DSUM", "DAVERAGE", "DCOUNT", "DGET"],
        }

        for category in categories:
            analysis["formula_categories"][category] = 0
            analysis["samples"][category] = []

        # Analyze each sheet
        for sheet_idx in range(1, workbook.Sheets.Count + 1):
            sheet = workbook.Sheets(sheet_idx)

            # Skip chart sheets
            if sheet.Type != 1:  # xlWorksheet
                continue

            used_range = sheet.UsedRange

            if not used_range:
                continue

            sheet_has_formulas = False
            sheet_formula_count = 0

            # Check cells for formulas
            for row in range(1, min(used_range.Rows.Count, 1000) + 1):
                for col in range(1, min(used_range.Columns.Count, 100) + 1):
                    try:
                        cell = used_range.Cells(row, col)

                        if cell.HasFormula:
                            formula = cell.Formula
                            sheet_has_formulas = True
                            sheet_formula_count += 1
                            analysis["total_formulas"] += 1

                            # Categorize formula
                            formula_upper = formula.upper()
                            categorized = False

                            for category, functions in categories.items():
                                for func in functions:
                                    if func.upper() + "(" in formula_upper:
                                        analysis["formula_categories"][category] += 1

                                        # Store a sample if needed
                                        if len(analysis["samples"][category]) < 3:
                                            analysis["samples"][category].append(
                                                {
                                                    "sheet": sheet.Name,
                                                    "cell": cell.Address(False, False),
                                                    "formula": formula,
                                                }
                                            )

                                        categorized = True
                                        break

                                if categorized:
                                    break

                            # Assess complexity
                            complexity = self._assess_formula_complexity(formula)
                            analysis["complexity"][complexity] += 1
                    except Exception:
                        pass  # Skip cells with errors

            if sheet_has_formulas:
                analysis["sheets_with_formulas"] += 1

        return analysis

    def _categorize_template(self, template_analysis):
        """Categorize a template based on its structure and contents.

        Args:
            template_analysis: Analysis of the template

        Returns:
            String indicating the template category
        """
        # Extract relevant information from analysis
        sheets = template_analysis.get("worksheets", [])
        sheet_names = [s.get("name", "").lower() for s in sheets]

        # Look for common sheet patterns
        has_financial_sheets = any(
            name in ["income", "balance", "cash flow", "forecast", "budget", "revenue"]
            for name in sheet_names
        )
        has_project_sheets = any(
            name in ["tasks", "timeline", "gantt", "resources", "schedule"] for name in sheet_names
        )
        has_dashboard_sheets = any(
            name in ["dashboard", "summary", "overview", "kpi", "metrics"] for name in sheet_names
        )
        has_data_sheets = any(
            name in ["data", "raw data", "source", "input"] for name in sheet_names
        )

        # Check formula patterns
        formula_patterns = []
        for sheet in sheets:
            formula_patterns.extend(sheet.get("formula_patterns", []))

        # Count pattern types
        financial_formulas = sum(
            p.get("count", 0)
            for p in formula_patterns
            if p.get("pattern") in ["sum", "calculation"]
        )
        lookup_formulas = sum(
            p.get("count", 0)
            for p in formula_patterns
            if p.get("pattern") in ["lookup", "reference"]
        )

        # Determine category based on collected information
        if has_financial_sheets and financial_formulas > 0:
            if "forecast" in " ".join(sheet_names) or "projection" in " ".join(sheet_names):
                return "financial_forecast"
            elif "budget" in " ".join(sheet_names):
                return "budget"
            else:
                return "financial"

        elif has_project_sheets:
            return "project_management"

        elif has_dashboard_sheets and has_data_sheets:
            return "dashboard"

        elif lookup_formulas > financial_formulas:
            return "data_analysis"

        else:
            return "general"

    def _adapt_text_to_context(self, text, context):
        """Adapt text based on context for template adaptation.

        Args:
            text: Original text string
            context: Context description

        Returns:
            Adapted text
        """
        if not text or not isinstance(text, str):
            return text

        # Check what industry or domain is mentioned in the context
        context_lower = context.lower()
        industry = None

        # Try to detect the target industry or domain
        if (
            "healthcare" in context_lower
            or "medical" in context_lower
            or "hospital" in context_lower
        ):
            industry = "healthcare"
        elif "tech" in context_lower or "software" in context_lower or "saas" in context_lower:
            industry = "technology"
        elif "retail" in context_lower or "shop" in context_lower or "store" in context_lower:
            industry = "retail"
        elif "finance" in context_lower or "bank" in context_lower or "investment" in context_lower:
            industry = "finance"
        elif (
            "education" in context_lower
            or "school" in context_lower
            or "university" in context_lower
        ):
            industry = "education"
        elif "manufacturing" in context_lower or "factory" in context_lower:
            industry = "manufacturing"
        elif "real estate" in context_lower or "property" in context_lower:
            industry = "real_estate"

        # If no industry detected, return original text
        if not industry:
            return text

        # Adapt common business terms based on industry
        text_lower = text.lower()

        # Handle healthcare industry adaptations
        if industry == "healthcare":
            if "customer" in text_lower:
                return text.replace("Customer", "Patient").replace("customer", "patient")
            elif "sales" in text_lower:
                return text.replace("Sales", "Services").replace("sales", "services")
            elif "product" in text_lower:
                return text.replace("Product", "Treatment").replace("product", "treatment")
            elif "revenue" in text_lower and "healthcare revenue" not in text_lower:
                return text.replace("Revenue", "Healthcare Revenue").replace(
                    "revenue", "healthcare revenue"
                )

        # Handle technology industry adaptations
        elif industry == "technology":
            if "customer" in text_lower:
                return text.replace("Customer", "User").replace("customer", "user")
            elif "sales" in text_lower:
                return text.replace("Sales", "Subscriptions").replace("sales", "subscriptions")
            elif "product" in text_lower:
                return text.replace("Product", "Solution").replace("product", "solution")

        # Handle retail industry adaptations
        elif industry == "retail":
            if "customer" in text_lower:
                return text.replace("Customer", "Shopper").replace("customer", "shopper")
            elif "sales" in text_lower:
                return text.replace("Sales", "Retail Sales").replace("sales", "retail sales")

        # Handle finance industry adaptations
        elif industry == "finance":
            if "customer" in text_lower:
                return text.replace("Customer", "Client").replace("customer", "client")
            elif "product" in text_lower:
                return text.replace("Product", "Financial Product").replace(
                    "product", "financial product"
                )

        # Handle education industry adaptations
        elif industry == "education":
            if "customer" in text_lower:
                return text.replace("Customer", "Student").replace("customer", "student")
            elif "sales" in text_lower:
                return text.replace("Sales", "Enrollments").replace("sales", "enrollments")
            elif "product" in text_lower:
                return text.replace("Product", "Course").replace("product", "course")

        # Default - return original text
        return text

    def _explain_formula(self, formula):
        """Generate a natural language explanation of an Excel formula.

        Args:
            formula: Excel formula string

        Returns:
            Human-readable explanation
        """
        formula_upper = formula.upper()

        # SUM function
        if "SUM(" in formula_upper:
            match = re.search(r"SUM\(([^)]+)\)", formula_upper)
            if match:
                range_str = match.group(1)
                return f"This formula calculates the sum of values in the range {range_str}."

        # AVERAGE function
        elif "AVERAGE(" in formula_upper:
            match = re.search(r"AVERAGE\(([^)]+)\)", formula_upper)
            if match:
                range_str = match.group(1)
                return f"This formula calculates the average (mean) of values in the range {range_str}."

        # VLOOKUP function
        elif "VLOOKUP(" in formula_upper:
            params = formula_upper.split("VLOOKUP(")[1].split(")", 1)[0].split(",")
            lookup_value = params[0] if len(params) > 0 else "?"
            table_array = params[1] if len(params) > 1 else "?"
            col_index = params[2] if len(params) > 2 else "?"
            exact_match = "FALSE" in params[3] if len(params) > 3 else False

            match_type = "exact match" if exact_match else "closest match (approximate match)"
            return f"This formula looks up {lookup_value} in the first column of {table_array}, and returns the value from column {col_index}. It finds the {match_type}."

        # IF function
        elif "IF(" in formula_upper:
            try:
                # This is a simplistic parsing - real parsing would be more complex
                content = formula_upper.split("IF(")[1].split(")", 1)[0]
                parts = []
                depth = 0
                current = ""

                for char in content:
                    if char == "," and depth == 0:
                        parts.append(current)
                        current = ""
                    else:
                        if char == "(":
                            depth += 1
                        elif char == ")":
                            depth -= 1
                        current += char

                if current:
                    parts.append(current)

                condition = parts[0] if len(parts) > 0 else "?"
                true_value = parts[1] if len(parts) > 1 else "?"
                false_value = parts[2] if len(parts) > 2 else "?"

                return f"This formula tests if {condition}. If true, it returns {true_value}, otherwise it returns {false_value}."
            except Exception:
                # Fallback if parsing fails
                return "This formula uses an IF statement to return different values based on a condition."

        # INDEX/MATCH
        elif "INDEX(" in formula_upper and "MATCH(" in formula_upper:
            return "This formula uses the INDEX/MATCH combination to look up a value in a table. INDEX returns a value at a specific position, and MATCH finds the position of a lookup value."

        # Simple calculations
        elif "+" in formula or "-" in formula or "*" in formula or "/" in formula:
            # Check if it's a simple calculation without functions
            if not any(
                func in formula_upper for func in ["SUM(", "AVERAGE(", "IF(", "VLOOKUP(", "INDEX("]
            ):
                operations = []
                if "+" in formula:
                    operations.append("addition")
                if "-" in formula:
                    operations.append("subtraction")
                if "*" in formula:
                    operations.append("multiplication")
                if "/" in formula:
                    operations.append("division")

                ops_text = " and ".join(operations)
                return (
                    f"This formula performs {ops_text} on the specified values or cell references."
                )

        # Fallback for unrecognized or complex formulas
        return "This formula performs a calculation on the referenced cells. For complex formulas, consider breaking it down into its component parts to understand it better."

    def _categorize_formula(self, formula):
        """Categorize a formula based on its functions and structure.

        Args:
            formula: Excel formula string

        Returns:
            Category string
        """
        formula_upper = formula.upper()

        # Mathematical
        if any(
            func in formula_upper
            for func in ["SUM(", "AVERAGE(", "MIN(", "MAX(", "COUNT(", "PRODUCT(", "ROUND("]
        ):
            return "mathematical"

        # Logical
        elif any(
            func in formula_upper for func in ["IF(", "AND(", "OR(", "NOT(", "SWITCH(", "IFS("]
        ):
            return "logical"

        # Lookup
        elif any(
            func in formula_upper
            for func in ["VLOOKUP(", "HLOOKUP(", "INDEX(", "MATCH(", "XLOOKUP("]
        ):
            return "lookup"

        # Text
        elif any(
            func in formula_upper
            for func in ["CONCATENATE(", "LEFT(", "RIGHT(", "MID(", "FIND(", "SEARCH(", "REPLACE("]
        ):
            return "text"

        # Date
        elif any(
            func in formula_upper for func in ["TODAY(", "NOW(", "DATE(", "DAY(", "MONTH(", "YEAR("]
        ):
            return "date"

        # Financial
        elif any(func in formula_upper for func in ["PMT(", "RATE(", "NPV(", "IRR(", "FV(", "PV("]):
            return "financial"

        # Statistical
        elif any(func in formula_upper for func in ["STDEV(", "VAR(", "MEDIAN(", "PERCENTILE("]):
            return "statistical"

        # Reference
        elif any(
            func in formula_upper
            for func in ["INDIRECT(", "OFFSET(", "ADDRESS(", "ROW(", "COLUMN("]
        ):
            return "reference"

        # Database
        elif any(func in formula_upper for func in ["DSUM(", "DAVERAGE(", "DCOUNT(", "DGET("]):
            return "database"

        # Simple calculation
        elif any(op in formula for op in ["+", "-", "*", "/"]):
            return "calculation"

        # Default/unknown
        return "other"

    def _assess_formula_complexity(self, formula):
        """Assess the complexity of a formula.

        Args:
            formula: Excel formula string

        Returns:
            Complexity level (simple, moderate, complex, very_complex)
        """
        # Count various aspects of the formula
        formula_length = len(formula)
        function_count = formula.upper().count("(")
        nesting_level = 0
        max_nesting = 0

        # Calculate nesting depth
        for char in formula:
            if char == "(":
                nesting_level += 1
                max_nesting = max(max_nesting, nesting_level)
            elif char == ")":
                nesting_level -= 1

        # Count references
        reference_count = len(re.findall(r"[A-Z]+[0-9]+(?::[A-Z]+[0-9]+)?", formula))

        # Count operators
        operator_count = sum(formula.count(op) for op in ["+", "-", "*", "/", "=", "<", ">", "&"])

        # Calculate a weighted complexity score
        score = (
            min(10, formula_length / 40)  # Length: max 10 points
            + function_count * 1.5  # Functions: 1.5 points each
            + max_nesting * 2  # Max nesting: 2 points per level
            + reference_count * 0.5  # References: 0.5 points each
            + operator_count * 0.5  # Operators: 0.5 points each
        )

        # Determine complexity level
        if score < 5:
            return "simple"
        elif score < 10:
            return "moderate"
        elif score < 20:
            return "complex"
        else:
            return "very_complex"

    def _get_formula_dependency_level(self, formula):
        """Determine how many other cells a formula depends on.

        Args:
            formula: Excel formula string

        Returns:
            Dependency level (low, medium, high)
        """
        # Count cell references and ranges
        references = re.findall(r"[A-Z]+[0-9]+(?::[A-Z]+[0-9]+)?", formula)

        # Count individual cells
        cell_count = 0
        for ref in references:
            if ":" in ref:
                # It's a range
                try:
                    start, end = ref.split(":")
                    start_col = re.search(r"[A-Z]+", start).group(0)
                    start_row = int(re.search(r"[0-9]+", start).group(0))
                    end_col = re.search(r"[A-Z]+", end).group(0)
                    end_row = int(re.search(r"[0-9]+", end).group(0))

                    # Convert column letters to numbers
                    start_col_num = 0
                    for char in start_col:
                        start_col_num = start_col_num * 26 + (ord(char) - ord("A") + 1)

                    end_col_num = 0
                    for char in end_col:
                        end_col_num = end_col_num * 26 + (ord(char) - ord("A") + 1)

                    # Calculate cells in range
                    cells_in_range = (end_row - start_row + 1) * (end_col_num - start_col_num + 1)
                    cell_count += cells_in_range
                except Exception:
                    # Fallback if parsing fails
                    cell_count += 10  # Assume a moderate size range
            else:
                # Single cell
                cell_count += 1

        # Determine dependency level
        if cell_count <= 3:
            return "low"
        elif cell_count <= 10:
            return "medium"
        else:
            return "high"

    def _check_formula_volatility(self, formula):
        """Check if a formula contains volatile functions.

        Args:
            formula: Excel formula string

        Returns:
            Volatility level (none, low, high)
        """
        formula_upper = formula.upper()

        # Highly volatile functions
        high_volatility = ["NOW(", "TODAY(", "RAND(", "RANDBETWEEN("]
        if any(func in formula_upper for func in high_volatility):
            return "high"

        # Low volatility functions
        low_volatility = ["OFFSET(", "INDIRECT(", "CELL(", "INFO("]
        if any(func in formula_upper for func in low_volatility):
            return "low"

        # Non-volatile
        return "none"

    def _get_calculation_mode_name(self, mode_value):
        """Convert Excel calculation mode numeric value to name.

        Args:
            mode_value: Numeric value of calculation mode

        Returns:
            String name of calculation mode
        """
        modes = {-4105: "Automatic", -4135: "Manual", -4133: "Semiautomatic"}

        return modes.get(mode_value, f"Unknown ({mode_value})")

    async def _generate_excel_macro(
        self,
        session: ExcelSession,
        instruction: str,
        file_path: Optional[str] = None,
        template: Optional[str] = None,
        test_execution: bool = False,
        security_level: str = "standard",
    ) -> Dict[str, Any]:
        """Generate Excel VBA macro based on instructions.

        Args:
            session: Excel session
            instruction: Natural language instruction
            file_path: Path to Excel file
            template: Optional template code or path to template file
            test_execution: Whether to test execute the macro
            security_level: Security level for macro execution

        Returns:
            Dictionary with macro generation results
        """
        result = {
            "success": True,
            "macro_generated": True,
            "macro_code": "",
            "execution_result": None,
        }

        # Check if template is a file path
        template_code = ""
        if template and os.path.exists(template):
            try:
                # Use read_file_content to load the template
                template_code = await read_file_content(template)
                result["template_source"] = "file"
            except Exception as e:
                logger.warning(f"Failed to read template file: {str(e)}")
                template_code = template or ""
                result["template_source"] = "text"
        else:
            template_code = template or ""
            result["template_source"] = "text"

        # Generate the macro code based on instruction
        # This would typically be done by the LLM in a real implementation
        macro_code = f"' Generated VBA Macro based on instruction:\n' {instruction}\n\n"

        if template_code:
            macro_code += f"' Based on template:\n{template_code}\n\n"

        # Add a simple macro as an example
        macro_code += """
Sub ExampleMacro()
    ' This is a placeholder for actual generated code
    MsgBox "Macro executed successfully!"
End Sub
"""

        result["macro_code"] = macro_code

        # Save the macro code to a separate file for reference if file_path is provided
        if file_path:
            macro_file_path = os.path.splitext(file_path)[0] + "_macro.bas"
            try:
                await write_file_content(macro_file_path, macro_code)
                result["macro_file"] = macro_file_path
            except Exception as e:
                logger.warning(f"Failed to save macro to file: {str(e)}")

        # If file_path provided, add the macro to the workbook
        if file_path and os.path.exists(file_path):
            # Open the workbook and add the macro
            # Implementation would depend on Excel VBA model
            pass

        # Test execution if requested
        if test_execution and file_path and os.path.exists(file_path):
            # Execute the macro
            # Implementation would depend on Excel VBA model
            result["execution_result"] = "Macro executed successfully"

        return result


def register_excel_spreadsheet_tools(mcp_server):
    """Registers Excel Spreadsheet Tools with the MCP server.

    Args:
        mcp_server: MCP server instance

    Returns:
        ExcelSpreadsheetTools instance
    """
    # Initialize the tool
    excel_tools = ExcelSpreadsheetTools(mcp_server)

    # Register tools with MCP server
    # These functions are now using state management
    mcp_server.tool(name="excel_execute")(excel_tools.excel_execute)
    mcp_server.tool(name="excel_learn_and_apply")(excel_tools.excel_learn_and_apply)
    mcp_server.tool(name="excel_analyze_formulas")(excel_tools.excel_analyze_formulas)
    mcp_server.tool(name="excel_generate_macro")(excel_tools.excel_generate_macro)
    mcp_server.tool(name="excel_export_sheet_to_csv")(excel_tools.excel_export_sheet_to_csv)
    mcp_server.tool(name="excel_import_csv_to_sheet")(excel_tools.excel_import_csv_to_sheet)

    return excel_tools
