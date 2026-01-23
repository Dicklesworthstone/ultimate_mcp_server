"""
Tool composition patterns for MCP servers.

This module demonstrates how to design tools that work together effectively
in sequences and patterns, making it easier for LLMs to understand how to
compose tools for multi-step operations.
"""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from error_handling import non_empty_string, validate_inputs, with_error_handling
from tool_annotations import QUERY_TOOL, READONLY_TOOL
from ultimate_mcp_server.exceptions import ToolExecutionError, ToolInputError
from ultimate_mcp_server.tools.document_conversion_and_processing import (
    summarize_document_standalone,
)
from ultimate_mcp_server.tools.filesystem import delete_file, read_file, write_file
from ultimate_mcp_server.tools.local_text_tools import run_sed
from ultimate_mcp_server.utils import get_logger

logger = get_logger("tool_composition_examples")


class DocumentProcessingExample:
    """
    Example of tool composition for document processing.

    This class demonstrates a pattern where multiple tools work together
    to process a document through multiple stages:
    1. Chunking - Break large document into manageable pieces
    2. Analysis - Process each chunk individually
    3. Aggregation - Combine results into a final output

    This pattern is ideal for working with large documents that exceed
    context windows.
    """

    def __init__(self, mcp_server):
        """Initialize with an MCP server instance."""
        self.mcp = mcp_server
        self._register_tools()

    def _register_tools(self):
        """Register document processing tools with the MCP server."""

        @self.mcp.tool(
            description=(
                "Split a document into manageable chunks for processing. "
                "This is the FIRST step in processing large documents that exceed context windows. "
                "After chunking, process each chunk separately with analyze_chunk()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Chunk research paper",
                    "description": "Split a research paper into chunks",
                    "input": {"document": "This is a long research paper...", "chunk_size": 1000},
                    "output": {
                        "chunks": ["Chunk 1...", "Chunk 2..."],
                        "chunk_count": 2,
                        "chunk_ids": ["doc123_chunk_1", "doc123_chunk_2"],
                    },
                }
            ],
        )
        @with_error_handling
        @validate_inputs(document=non_empty_string)
        async def chunk_document(
            document: str, chunk_size: int = 1000, overlap: int = 100, ctx=None
        ) -> Dict[str, Any]:
            """
            Split a document into manageable chunks for processing.

            This tool is the first step in a multi-step document processing workflow:
            1. First, chunk the document with chunk_document() (this tool)
            2. Then, process each chunk with analyze_chunk()
            3. Finally, combine results with aggregate_chunks()

            Args:
                document: The document text to split
                chunk_size: Maximum size of each chunk in characters
                overlap: Number of characters to overlap between chunks
                ctx: Context object passed by the MCP server

            Returns:
                Dictionary containing the chunks and their metadata
            """
            # Simple chunking strategy - split by character count with overlap
            chunks = []
            chunk_ids = []
            doc_id = f"doc_{hash(document) % 10000}"

            # Create chunks with overlap
            for i in range(0, len(document), chunk_size - overlap):
                chunk_text = document[i : i + chunk_size]
                if chunk_text:
                    chunk_id = f"{doc_id}_chunk_{len(chunks) + 1}"
                    chunks.append(chunk_text)
                    chunk_ids.append(chunk_id)

            return {
                "chunks": chunks,
                "chunk_count": len(chunks),
                "chunk_ids": chunk_ids,
                "document_id": doc_id,
                "next_step": "analyze_chunk",  # Hint for the next tool to use
            }

        @self.mcp.tool(
            description=(
                "Analyze a single document chunk by summarizing it. "
                "This is the SECOND step in the document processing workflow. "
                "Use after chunk_document() and before aggregate_chunks()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Analyze document chunk",
                    "description": "Analyze a single chunk from a research paper",
                    "input": {
                        "chunk": "This chunk discusses methodology...",
                        "chunk_id": "doc123_chunk_1",
                    },
                    "output": {
                        "analysis": {"key_topics": ["methodology", "experiment design"]},
                        "chunk_id": "doc123_chunk_1",
                    },
                }
            ],
        )
        @with_error_handling
        @validate_inputs(chunk=non_empty_string)
        async def analyze_chunk(
            chunk: str, chunk_id: str, analysis_type: str = "general", ctx=None
        ) -> Dict[str, Any]:
            """
            Analyze a single document chunk by summarizing it.

            This tool is the second step in a multi-step document processing workflow:
            1. First, chunk the document with chunk_document()
            2. Then, process each chunk with analyze_chunk() (this tool)
            3. Finally, combine results with aggregate_chunks()

            Args:
                chunk: The text chunk to analyze
                chunk_id: The ID of the chunk (from chunk_document)
                analysis_type: Type of analysis to perform
                ctx: Context object passed by the MCP server

            Returns:
                Dictionary containing the analysis results
            """
            # --- Call the actual summarize_document tool ---
            logger.info(f"Analyzing chunk {chunk_id} with summarize_document...")
            try:
                # Use a concise summary for chunk analysis
                summary_result = await summarize_document_standalone(
                    document=chunk,
                    summary_format="key_points",  # Use key points for chunk analysis
                    max_length=100,  # Keep chunk summaries relatively short
                    # We might need to specify provider/model if defaults aren't suitable
                )

                if summary_result.get("success"):
                    analysis = {
                        "summary": summary_result.get("summary", "[Summary Unavailable]"),
                        "analysis_type": "summary",  # Indicate the type of analysis performed
                        "metrics": {  # Include metrics from the summary call
                            "cost": summary_result.get("cost", 0.0),
                            "tokens": summary_result.get("tokens", {}),
                            "processing_time": summary_result.get("processing_time", 0.0),
                        },
                    }
                    logger.success(f"Chunk {chunk_id} analyzed successfully.")
                else:
                    logger.warning(
                        f"Summarize tool failed for chunk {chunk_id}: {summary_result.get('error')}"
                    )
                    analysis = {"error": f"Analysis failed: {summary_result.get('error')}"}
            except Exception as e:
                logger.error(
                    f"Error calling summarize_document for chunk {chunk_id}: {e}", exc_info=True
                )
                analysis = {"error": f"Analysis error: {str(e)}"}
            # -------------------------------------------------

            return {
                "analysis": analysis,
                "chunk_id": chunk_id,
                "next_step": "aggregate_chunks",  # Hint for the next tool to use
            }

        @self.mcp.tool(
            description=(
                "Aggregate analysis results from multiple document chunks. "
                "This is the FINAL step in the document processing workflow. "
                "Use after analyzing individual chunks with analyze_chunk()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Aggregate analysis results",
                    "description": "Combine analysis results from multiple chunks",
                    "input": {
                        "analysis_results": [
                            {
                                "analysis": {"key_topics": ["methodology"]},
                                "chunk_id": "doc123_chunk_1",
                            },
                            {"analysis": {"key_topics": ["results"]}, "chunk_id": "doc123_chunk_2"},
                        ]
                    },
                    "output": {
                        "document_summary": "This document covers methodology and results...",
                        "overall_statistics": {"total_chunks": 2, "word_count": 2500},
                    },
                }
            ],
        )
        @with_error_handling
        async def aggregate_chunks(
            analysis_results: List[Dict[str, Any]], aggregation_type: str = "summary", ctx=None
        ) -> Dict[str, Any]:
            """
            Aggregate analysis results from multiple document chunks.

            This tool is the final step in a multi-step document processing workflow:
            1. First, chunk the document with chunk_document()
            2. Then, process each chunk with analyze_chunk()
            3. Finally, combine results with aggregate_chunks() (this tool)

            Args:
                analysis_results: List of analysis results from analyze_chunk
                aggregation_type: Type of aggregation to perform
                ctx: Context object passed by the MCP server

            Returns:
                Dictionary containing the aggregated results
            """
            # Validate input
            if not analysis_results or not isinstance(analysis_results, list):
                return {
                    "error": "Invalid analysis_results. Must provide a non-empty list of analysis results."
                }

            # Extract all analyses
            all_analyses = [
                result.get("analysis", {}) for result in analysis_results if "analysis" in result
            ]
            total_chunks = len(all_analyses)

            # Calculate overall statistics
            total_word_count = sum(analysis.get("word_count", 0) for analysis in all_analyses)
            all_key_sentences = [
                sentence
                for analysis in all_analyses
                for sentence in analysis.get("key_sentences", [])
            ]

            # Generate summary based on aggregation type
            if aggregation_type == "summary":
                summary = (
                    f"Document contains {total_chunks} chunks with {total_word_count} words total."
                )
                if all_key_sentences:
                    summary += f" Key points include: {' '.join(all_key_sentences[:3])}..."
            elif aggregation_type == "sentiment":
                # Aggregate sentiment scores if available
                sentiment_scores = [
                    analysis.get("sentiment_score", 0.5)
                    for analysis in all_analyses
                    if "sentiment_score" in analysis
                ]
                avg_sentiment = (
                    sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
                )
                sentiment_label = (
                    "positive"
                    if avg_sentiment > 0.6
                    else "neutral"
                    if avg_sentiment > 0.4
                    else "negative"
                )
                summary = f"Document has an overall {sentiment_label} sentiment (score: {avg_sentiment:.2f})."
            else:
                summary = f"Aggregated {total_chunks} chunks with {total_word_count} total words."

            return {
                "document_summary": summary,
                "overall_statistics": {
                    "total_chunks": total_chunks,
                    "word_count": total_word_count,
                    "key_sentences_count": len(all_key_sentences),
                },
                "workflow_complete": True,  # Indicate this is the end of the workflow
            }


# --- Helper: Get a temporary path within allowed storage ---
# Assume storage directory exists and is allowed for this demo context
STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
TEMP_DATA_DIR = STORAGE_DIR / "temp_pipeline_data"


async def _setup_temp_data_files():
    """Create temporary data files for the pipeline demo."""
    TEMP_DATA_DIR.mkdir(exist_ok=True)
    # Sample CSV Data
    csv_data = io.StringIO()
    writer = csv.writer(csv_data)
    writer.writerow(["date", "amount", "category"])
    writer.writerow(["2023-01-01", "1,200", "electronics"])  # Note: Amount as string with comma
    writer.writerow(["2023-01-02", "950", "clothing"])
    writer.writerow(["2023-01-03", "1500", "electronics"])
    writer.writerow(["2023-01-04", "800", "food"])
    csv_content = csv_data.getvalue()
    csv_path = TEMP_DATA_DIR / "temp_sales.csv"
    await write_file(path=str(csv_path), content=csv_content)  # Use write_file tool implicitly

    # Sample JSON Data
    json_data = [
        {"user_id": 101, "name": "Alice", "active": True, "last_login": "2023-01-10"},
        {"user_id": 102, "name": "Bob", "active": False, "last_login": "2022-12-15"},
        {"user_id": 103, "name": "Charlie", "active": True, "last_login": "2023-01-05"},
    ]
    json_content = json.dumps(json_data, indent=2)
    json_path = TEMP_DATA_DIR / "temp_users.json"
    await write_file(path=str(json_path), content=json_content)  # Use write_file tool implicitly

    return {"csv": str(csv_path), "json": str(json_path)}


async def _cleanup_temp_data_files(temp_files: Dict[str, str]):
    """Remove temporary data files."""
    for file_path in temp_files.values():
        try:
            await delete_file(path=file_path)  # Use delete_file tool implicitly
            logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {file_path}: {e}")
    try:
        # Attempt to remove the directory if empty
        if TEMP_DATA_DIR.exists() and not any(TEMP_DATA_DIR.iterdir()):
            TEMP_DATA_DIR.rmdir()
            logger.debug(f"Cleaned up temp directory: {TEMP_DATA_DIR}")
    except Exception as e:
        logger.warning(f"Failed to remove temp directory {TEMP_DATA_DIR}: {e}")


# --- End Helper ---


class DataPipelineExample:
    """
    Example of tool composition for data processing pipelines.

    This class demonstrates a pattern where tools form a processing
    pipeline to transform, filter, and analyze data:
    1. Fetch - Get data from a source
    2. Transform - Clean and process the data
    3. Filter - Select relevant data
    4. Analyze - Perform analysis on filtered data

    This pattern is ideal for working with structured data that
    needs multiple processing steps.
    """

    def __init__(self, mcp_server):
        """Initialize with an MCP server instance."""
        self.mcp = mcp_server
        self._register_tools()

    def _register_tools(self):
        """Register data pipeline tools with the MCP server."""

        @self.mcp.tool(
            description=(
                "Fetch data from a temporary source file based on type. "
                "This is the FIRST step in the data pipeline. "
                "Continue with transform_data() to clean the fetched data."
            ),
            annotations=QUERY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Fetch CSV data",
                    "description": "Fetch data from a CSV source",
                    "input": {"source_type": "csv"},
                    "output": {
                        "data": [
                            {"date": "2023-01-01", "amount": 1200},
                            {"date": "2023-01-02", "amount": 950},
                        ],
                        "record_count": 2,
                        "schema": {"date": "string", "amount": "number"},
                    },
                }
            ],
        )
        @with_error_handling
        async def fetch_data(
            source_type: str, limit: Optional[int] = None, ctx=None
        ) -> Dict[str, Any]:
            """
            Fetch data from a temporary source file based on type.

            This tool is the first step in a data processing pipeline:
            1. First, fetch data with fetch_data() (this tool) - Creates temp files if needed.
            2. Then, clean the data with transform_data()
            3. Then, filter the data with filter_data()
            4. Finally, analyze the data with analyze_data()

            Args:
                source_type: Type of data source (csv or json for this demo).
                limit: Maximum number of records to fetch/read (applied after reading).
                ctx: Context object passed by the MCP server.

            Returns:
                Dictionary containing the fetched data and metadata
            """
            # Ensure temp files exist
            temp_files = await _setup_temp_data_files()
            source_path = temp_files.get(source_type.lower())

            if not source_path:
                raise ToolInputError(
                    f"Unsupported source_type for demo: {source_type}. Use 'csv' or 'json'."
                )

            logger.info(f"Fetching data from temporary file: {source_path}")

            # Use read_file tool implicitly to get content
            read_result = await read_file(path=source_path)
            if not read_result.get("success"):
                raise ToolExecutionError(
                    f"Failed to read temporary file {source_path}: {read_result.get('error')}"
                )

            # Assuming read_file returns content in a predictable way (e.g., result['content'][0]['text'])
            # Adjust parsing based on actual read_file output structure
            content = read_result.get("content", [])
            if not content or not isinstance(content, list) or "text" not in content[0]:
                raise ToolExecutionError(
                    f"Unexpected content structure from read_file for {source_path}"
                )

            file_content = content[0]["text"]
            data = []
            schema = {}

            try:
                if source_type.lower() == "csv":
                    # Parse CSV data
                    csv_reader = csv.reader(io.StringIO(file_content))
                    headers = next(csv_reader)
                    for row in csv_reader:
                        if row:  # Skip empty rows
                            data.append(dict(zip(headers, row, strict=False)))
                elif source_type.lower() == "json":
                    # Parse JSON data
                    data = json.loads(file_content)
                else:
                    # Default dummy data if somehow type is wrong despite check
                    data = [{"id": i, "value": f"Sample {i}"} for i in range(1, 6)]
            except Exception as parse_error:
                raise ToolExecutionError(
                    f"Failed to parse content from {source_path}: {parse_error}"
                ) from parse_error

            # Apply limit if specified AFTER reading/parsing
            if limit and limit > 0 and len(data) > limit:
                data = data[:limit]

            # Infer schema from first record
            if data:
                first_record = data[0]
                for key, value in first_record.items():
                    value_type = "string"
                    if isinstance(value, (int, float)):
                        value_type = "number"
                    elif isinstance(value, bool):
                        value_type = "boolean"
                    schema[key] = value_type

            return {
                "data": data,
                "record_count": len(data),
                "schema": schema,
                "source_info": {"type": source_type, "path": source_path},
                "next_step": "transform_data",  # Hint for the next tool to use
            }

        @self.mcp.tool(
            description=(
                "Transform and clean data using basic text processing tools (sed). "
                "This is the SECOND step in the data pipeline. "
                "Use after fetch_data() and before filter_data()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Transform sales data",
                    "description": "Clean and transform sales data",
                    "input": {
                        "data": [
                            {"date": "2023-01-01", "amount": "1,200"},
                            {"date": "2023-01-02", "amount": "950"},
                        ],
                        "transformations": ["convert_dates", "normalize_numbers"],
                    },
                    "output": {
                        "transformed_data": [
                            {"date": "2023-01-01", "amount": 1200},
                            {"date": "2023-01-02", "amount": 950},
                        ],
                        "transformation_log": ["Converted 2 dates", "Normalized 2 numbers"],
                    },
                }
            ],
        )
        @with_error_handling
        async def transform_data(
            data: List[Dict[str, Any]],
            transformations: List[str] = None,
            custom_transformations: Dict[str, str] = None,
            ctx=None,
        ) -> Dict[str, Any]:
            """
            Transform and clean data using basic text processing tools (sed).

            This tool is the second step in a data processing pipeline:
            1. First, fetch data with fetch_data()
            2. Then, clean the data with transform_data() (this tool)
            3. Then, filter the data with filter_data()
            4. Finally, analyze the data with analyze_data()

            Args:
                data: List of data records to transform
                transformations: List of built-in transformations to apply
                custom_transformations: Dictionary of field->transform_expression
                ctx: Context object passed by the MCP server

            Returns:
                Dictionary containing the transformed data and transformation log
            """
            # Validate input
            if not data or not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
                return {
                    "error": "Invalid data. Must provide a non-empty list of records (dictionaries)."
                }

            transformation_log = []
            # Convert input data (list of dicts) to a string format suitable for sed (e.g., JSON lines)
            try:
                input_text = "\n".join(json.dumps(record) for record in data)
            except Exception as e:
                return {"error": f"Could not serialize input data for transformation: {e}"}

            current_text = input_text
            sed_scripts = []  # Accumulate sed commands

            # Apply standard transformations if specified
            transformations = transformations or []
            for transform in transformations:
                if transform == "convert_dates":
                    # Use sed to replace '/' with '-' in date-like fields (heuristic)
                    # This is complex with JSON structure, better done after parsing.
                    # For demo, we apply a simple global substitution (less robust)
                    sed_scripts.append("s|/|-|g")
                    transformation_log.append("Applied date conversion (sed: s|/|-|g)")

                elif transform == "normalize_numbers":
                    # Use sed to remove commas from numbers (heuristic)
                    # Example: "amount": "1,200" -> "amount": "1200"
                    sed_scripts.append(
                        's/"([a-zA-Z_]+)":"([0-9,]+)"/"\1":"\2"/g; s/,//g'
                    )  # More complex sed needed
                    transformation_log.append("Applied number normalization (sed: remove commas)")

            # --- Execute accumulated sed scripts ---
            if sed_scripts:
                # Combine scripts with -e for each
                combined_script = " ".join([f"-e '{s}'" for s in sed_scripts])
                logger.info(f"Running sed transformation with script: {combined_script}")
                try:
                    sed_result = await run_sed(
                        args_str=combined_script,  # Pass combined script
                        input_data=current_text,
                    )

                    if sed_result.get("success"):
                        current_text = sed_result["stdout"]
                        logger.success("Sed transformation completed successfully.")
                    else:
                        error_msg = sed_result.get("error", "Sed command failed")
                        logger.error(f"Sed transformation failed: {error_msg}")
                        return {"error": f"Transformation failed: {error_msg}"}
                except Exception as e:
                    logger.error(f"Error running sed transformation: {e}", exc_info=True)
                    return {"error": f"Transformation execution error: {e}"}

            # --- Attempt to parse back to list of dicts ---
            try:
                transformed_data = []
                for line in current_text.strip().split("\n"):
                    if line:
                        record = json.loads(line)
                        # Post-processing for number normalization (sed only removes commas)
                        if "normalize_numbers" in transformations:
                            for key, value in record.items():
                                if isinstance(value, str) and value.replace(".", "", 1).isdigit():
                                    try:
                                        record[key] = float(value) if "." in value else int(value)
                                    except ValueError:
                                        pass  # Keep as string if conversion fails
                        transformed_data.append(record)
                logger.success("Successfully parsed transformed data back to JSON objects.")
            except Exception as e:
                logger.error(f"Could not parse transformed data back to JSON: {e}", exc_info=True)
                # Return raw text if parsing fails
                return {
                    "transformed_data_raw": current_text,
                    "transformation_log": transformation_log,
                    "warning": "Could not parse final data back to JSON records",
                }
            # ---------------------------------------------------

            return {
                "transformed_data": transformed_data,
                "transformation_log": transformation_log,
                "record_count": len(transformed_data),
                "next_step": "filter_data",  # Hint for the next tool to use
            }

        @self.mcp.tool(
            description=(
                "Filter data based on criteria. "
                "This is the THIRD step in the data pipeline. "
                "Use after transform_data() and before analyze_data()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Filter data",
                    "description": "Filter data based on criteria",
                    "input": {
                        "data": [
                            {"date": "2023-01-01", "amount": 1200},
                            {"date": "2023-01-02", "amount": 950},
                        ],
                        "filter_criteria": {"amount": {"$gt": 1000}},
                    },
                    "output": {
                        "filtered_data": [{"date": "2023-01-01", "amount": 1200}],
                        "filter_criteria": {"amount": {"$gt": 1000}},
                    },
                }
            ],
        )
        @with_error_handling
        async def filter_data(
            data: List[Dict[str, Any]], filter_criteria: Dict[str, Any], ctx=None
        ) -> Dict[str, Any]:
            """
            Filter data based on criteria.

            This tool is the third step in a data processing pipeline:
            1. First, fetch data with fetch_data()
            2. Then, clean the data with transform_data()
            3. Then, filter the data with filter_data() (this tool)
            4. Finally, analyze the data with analyze_data()

            Args:
                data: List of data records to filter
                filter_criteria: Criteria to filter data
                ctx: Context object passed by the MCP server

            Returns:
                Dictionary containing the filtered data and filter criteria
            """
            # Filter data based on criteria
            filtered_data = [
                record
                for record in data
                if all(record.get(key) == value for key, value in filter_criteria.items())
            ]

            return {
                "filtered_data": filtered_data,
                "filter_criteria": filter_criteria,
                "record_count": len(filtered_data),
            }

        @self.mcp.tool(
            description=(
                "Analyze data. "
                "This is the FINAL step in the data pipeline. "
                "Use after filtering data with filter_data()."
            ),
            annotations=READONLY_TOOL.to_dict(),
            examples=[
                {
                    "name": "Analyze data",
                    "description": "Analyze filtered data",
                    "input": {
                        "data": [
                            {"date": "2023-01-01", "amount": 1200},
                            {"date": "2023-01-02", "amount": 950},
                        ],
                        "analysis_type": "summary",
                    },
                    "output": {
                        "analysis_results": [
                            {
                                "analysis": {"key_topics": ["methodology"]},
                                "chunk_id": "doc123_chunk_1",
                            },
                            {"analysis": {"key_topics": ["results"]}, "chunk_id": "doc123_chunk_2"},
                        ],
                        "analysis_type": "summary",
                    },
                }
            ],
        )
        @with_error_handling
        async def analyze_data(
            data: List[Dict[str, Any]], analysis_type: str = "summary", ctx=None
        ) -> Dict[str, Any]:
            """
            Analyze data.

            This tool is the final step in a data processing pipeline:
            1. First, fetch data with fetch_data()
            2. Then, clean the data with transform_data()
            3. Then, filter the data with filter_data()
            4. Finally, analyze the data with analyze_data() (this tool)

            Args:
                data: List of data records to analyze
                analysis_type: Type of analysis to perform
                ctx: Context object passed by the MCP server

            Returns:
                Dictionary containing the analysis results
            """
            # Simulate analysis based on analysis_type
            if analysis_type == "summary":
                # Aggregate analysis results
                analysis_results = [
                    {"analysis": {"key_topics": ["methodology"]}, "chunk_id": "doc123_chunk_1"},
                    {"analysis": {"key_topics": ["results"]}, "chunk_id": "doc123_chunk_2"},
                ]
            else:
                # Placeholder for other analysis types
                analysis_results = []

            return {"analysis_results": analysis_results, "analysis_type": analysis_type}

    async def cleanup_pipeline_data(self):
        """Cleans up temporary data files created by fetch_data."""
        await _cleanup_temp_data_files(
            {
                "csv": str(TEMP_DATA_DIR / "temp_sales.csv"),
                "json": str(TEMP_DATA_DIR / "temp_users.json"),
            }
        )


# Example usage (if this file were run directly or imported)
# async def run_pipeline_example():
#     # ... initialize MCP server ...
#     pipeline = DataPipelineExample(mcp_server)
#     try:
#         # ... run pipeline steps ...
#         fetch_result = await pipeline.fetch_data(source_type="csv")
#         transform_result = await pipeline.transform_data(data=fetch_result['data'])
#         # ... etc ...
#     finally:
#         await pipeline.cleanup_pipeline_data()

# asyncio.run(run_pipeline_example())
