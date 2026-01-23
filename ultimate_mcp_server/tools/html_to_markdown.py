"""HTML to Markdown conversion tools for Ultimate MCP Server."""

import re
import time
from typing import Any, Dict, List

import html2text
import readability
import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from ultimate_mcp_server.exceptions import ToolInputError
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.html_to_markdown")

# --- Helper Functions ---


def _is_html_fragment(text: str) -> bool:
    """Detect if text is likely an HTML fragment.

    Args:
        text: Input text to check

    Returns:
        bool: True if the text appears to be HTML, False otherwise
    """
    # Simple heuristics to check if the text contains HTML
    html_patterns = [
        r"<\s*[a-zA-Z]+[^>]*>",  # Basic HTML tag pattern
        r"<\s*/\s*[a-zA-Z]+\s*>",  # Closing HTML tag
        r"&[a-zA-Z]+;",  # HTML entities
        r"<!\s*DOCTYPE",  # DOCTYPE declaration
        r"<!\s*--",  # HTML comment
        r"style\s*=\s*['\"]",  # style attribute
        r"class\s*=\s*['\"]",  # class attribute
        r"id\s*=\s*['\"]",  # id attribute
        r"href\s*=\s*['\"]",  # href attribute
        r"src\s*=\s*['\"]",  # src attribute
    ]

    # Check if the text matches any of the patterns
    for pattern in html_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def _clean_html_with_beautifulsoup(html: str) -> str:
    """Clean HTML using BeautifulSoup.

    Args:
        html: HTML content to clean

    Returns:
        Cleaned HTML string with unwanted elements removed
    """
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "svg", "iframe", "canvas", "noscript"]):
            element.decompose()

        # Remove base64 data attributes and other potentially problematic attributes
        for tag in soup.find_all(True):
            for attr in list(tag.attrs):
                # Clean data URLs
                if (
                    attr == "src"
                    and isinstance(tag.attrs[attr], str)
                    and "data:" in tag.attrs[attr]
                ):
                    del tag.attrs[attr]
                # Remove other problematic attributes
                elif attr.startswith("on") or attr == "style" or attr.startswith("data-"):
                    del tag.attrs[attr]

        return str(soup)
    except Exception as e:
        logger.warning(f"Error cleaning HTML with BeautifulSoup: {str(e)}")
        # If BeautifulSoup fails, return the original HTML
        return html


def _html_to_markdown_with_html2text(html: str) -> str:
    """Convert HTML to Markdown using html2text.

    Args:
        html: HTML content to convert

    Returns:
        Markdown formatted text
    """
    try:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_tables = False
        h.unicode_snob = True  # Use Unicode instead of ASCII
        h.body_width = 0  # No wrapping

        return h.handle(html)
    except Exception as e:
        logger.warning(f"Error converting HTML to Markdown with html2text: {str(e)}")
        # If html2text fails, try a simpler approach
        return html


def _html_to_markdown_with_markdownify(html: str) -> str:
    """Convert HTML to Markdown using markdownify.

    Args:
        html: HTML content to convert

    Returns:
        Markdown formatted text
    """
    try:
        return md(html, heading_style="ATX")
    except Exception as e:
        logger.warning(f"Error converting HTML to Markdown with markdownify: {str(e)}")
        # If markdownify fails, try a simpler approach
        return html


def _extract_content_with_readability(html: str) -> str:
    """Extract main content from HTML using readability.

    Args:
        html: HTML content to process

    Returns:
        HTML string containing only the main content
    """
    try:
        doc = readability.Document(html)
        content = doc.summary()
        return content
    except Exception as e:
        logger.warning(f"Error extracting content with readability: {str(e)}")
        # If readability fails, return the original HTML
        return html


def _extract_content_with_trafilatura(html: str) -> str:
    """Extract main content from HTML using trafilatura.

    Args:
        html: HTML content to process

    Returns:
        Extracted text content
    """
    try:
        extracted_text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if extracted_text:
            return extracted_text
        # Fall back to HTML extraction if text extraction fails
        extracted_html = trafilatura.extract(
            html, output_format="html", include_comments=False, include_tables=True
        )
        return extracted_html or html
    except Exception as e:
        logger.warning(f"Error extracting content with trafilatura: {str(e)}")
        # If trafilatura fails, return the original HTML
        return html


def _sanitize_markdown(markdown: str) -> str:
    """Clean up and format the markdown to be more readable.

    Args:
        markdown: Markdown text to sanitize

    Returns:
        Cleaned markdown text
    """
    # Fix excessive newlines (more than 2 consecutive)
    sanitized = re.sub(r"\n{3,}", "\n\n", markdown)

    # Fix list item spacing
    sanitized = re.sub(r"(\n[*-].*\n)(?!\n)", r"\1\n", sanitized)

    # Remove trailing whitespace from lines
    sanitized = re.sub(r" +$", "", sanitized, flags=re.MULTILINE)

    # Fix markdown heading formatting (ensure space after #)
    sanitized = re.sub(r"(^|\n)(#{1,6})([^#\s])", r"\1\2 \3", sanitized)

    # Fix code block formatting
    sanitized = re.sub(r"```\s*\n", "```\n", sanitized)
    sanitized = re.sub(r"\n\s*```", "\n```", sanitized)

    # Ensure proper code block syntax (start with language or leave empty)
    sanitized = re.sub(r"```([^a-zA-Z\s\n][^`\n]*)$", "```\n\\1", sanitized, flags=re.MULTILINE)

    # Normalize list indicators (consistent use of - or * for unordered lists)
    sanitized = re.sub(r"^[*+] ", "- ", sanitized, flags=re.MULTILINE)

    return sanitized


def _improve_markdown_formatting(markdown: str) -> str:
    """Improve the formatting of the markdown to make it more readable.

    Args:
        markdown: Markdown text to improve

    Returns:
        Improved markdown text
    """
    # Ensure proper spacing for headings
    improved = re.sub(r"(\n#{1,6}[^\n]+)(\n[^\n#])", r"\1\n\2", markdown)

    # Ensure paragraphs have proper spacing
    improved = re.sub(r"(\n[^\s#>*-][^\n]+)(\n[^\s#>*-])", r"\1\n\2", improved)

    # Fix blockquote formatting
    improved = re.sub(r"(\n>[ ][^\n]+)(\n[^>\s])", r"\1\n\2", improved)

    # Fix nested list formatting
    improved = re.sub(r"(\n[ ]{2,}[*-][ ][^\n]+)(\n[^\s*-])", r"\1\n\2", improved)

    # Add horizontal rules for clear section breaks (if large content gaps exist)
    improved = re.sub(r"\n\n\n\n+", "\n\n---\n\n", improved)

    return improved


def _convert_html_tables_to_markdown(html: str) -> str:
    """Specifically handle HTML tables and convert them to markdown tables.

    Args:
        html: HTML content with tables to convert

    Returns:
        Markdown text with properly formatted tables
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")

        # If no tables, return original HTML
        if not tables:
            return html

        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue

            markdown_table = []

            # Process header row
            header_cells = rows[0].find_all(["th", "td"])
            if header_cells:
                header_row = (
                    "| " + " | ".join([cell.get_text().strip() for cell in header_cells]) + " |"
                )
                markdown_table.append(header_row)

                # Add separator row
                separator_row = "| " + " | ".join(["---" for _ in header_cells]) + " |"
                markdown_table.append(separator_row)

            # Process data rows
            for row in rows[1:]:
                cells = row.find_all("td")
                if cells:
                    data_row = "| " + " | ".join([cell.get_text().strip() for cell in cells]) + " |"
                    markdown_table.append(data_row)

            # Replace the table with its markdown equivalent
            table_html = str(table)
            table_markdown = "\n".join(markdown_table)
            html = html.replace(table_html, table_markdown)

        return html

    except Exception as e:
        logger.warning(f"Error converting HTML tables to Markdown: {str(e)}")
        # If conversion fails, return the original HTML
        return html


# --- Main Tool Function ---


@with_tool_metrics
@with_error_handling
async def clean_and_format_text_as_markdown(
    text: str,
    force_markdown_conversion: bool = False,
    extraction_method: str = "auto",
    preserve_tables: bool = True,
    preserve_links: bool = True,
    preserve_images: bool = False,
    max_line_length: int = 0,  # 0 means no wrapping
) -> Dict[str, Any]:
    """Converts plain text or HTML to clean, well-formatted markdown.

    Automatically detects if input is HTML, then cleans and converts it.
    For non-HTML text, it applies minimal formatting to create valid markdown.

    Args:
        text: The input text to clean and format (plain text or HTML).
        force_markdown_conversion: Whether to force markdown conversion even if the text doesn't
                                  look like HTML. Default is False.
        extraction_method: Method to extract content from HTML. Options:
                          - "auto": Automatically choose the best method
                          - "readability": Use Mozilla's Readability algorithm
                          - "trafilatura": Use trafilatura library
                          - "raw": Don't extract main content, convert the whole document
                          Default is "auto".
        preserve_tables: Whether to preserve and convert HTML tables to markdown tables.
                        Default is True.
        preserve_links: Whether to preserve and convert HTML links to markdown links.
                       Default is True.
        preserve_images: Whether to preserve and convert HTML images to markdown image syntax.
                        Default is False.
        max_line_length: Maximum line length for text wrapping. 0 means no wrapping.
                        Default is 0.

    Returns:
        Dictionary containing:
        {
            "markdown_text": "Cleaned and formatted markdown text",
            "was_html": true,  # Whether the input was detected as HTML
            "extraction_method_used": "readability",  # Which extraction method was used
            "processing_time": 0.35,  # Time taken in seconds
            "success": true
        }

    Raises:
        ToolInputError: If the input text is empty or not a string.
    """
    start_time = time.time()

    # Input validation
    if not text:
        raise ToolInputError("Input text cannot be empty")
    if not isinstance(text, str):
        raise ToolInputError("Input text must be a string")

    # Determine if input is HTML
    is_html = _is_html_fragment(text) or force_markdown_conversion

    # Process based on content type
    if is_html:
        logger.info("Input detected as HTML, processing for conversion to markdown")

        # Convert HTML tables to markdown before main processing
        if preserve_tables:
            text = _convert_html_tables_to_markdown(text)

        # Extract main content based on specified method
        extraction_method_used = extraction_method
        if extraction_method == "auto":
            # If the text is a small fragment, use raw conversion
            if len(text) < 1000:
                extraction_method_used = "raw"
            else:
                # Try trafilatura first, fallback to readability
                try:
                    extracted = _extract_content_with_trafilatura(text)
                    if extracted and len(extracted) > 0.2 * len(
                        text
                    ):  # Ensure we got meaningful extraction
                        text = extracted
                        extraction_method_used = "trafilatura"
                    else:
                        text = _extract_content_with_readability(text)
                        extraction_method_used = "readability"
                except Exception:
                    text = _extract_content_with_readability(text)
                    extraction_method_used = "readability"
        elif extraction_method == "readability":
            text = _extract_content_with_readability(text)
        elif extraction_method == "trafilatura":
            text = _extract_content_with_trafilatura(text)
        # For "raw", we use the text as is

        # Clean HTML before conversion
        text = _clean_html_with_beautifulsoup(text)

        # Set up conversion options based on parameters
        h = html2text.HTML2Text()
        h.ignore_links = not preserve_links
        h.ignore_images = not preserve_images
        h.ignore_tables = not preserve_tables
        h.body_width = max_line_length
        h.unicode_snob = True

        # Try multiple conversion methods and use the best result
        try:
            markdown_text = h.handle(text)

            # Fallback to markdownify if html2text result looks problematic
            if (
                "&lt;" in markdown_text
                or "&gt;" in markdown_text
                or len(markdown_text.strip()) < 100
                and len(text) > 500
            ):
                try:
                    alternative = _html_to_markdown_with_markdownify(text)
                    if len(alternative.strip()) > len(markdown_text.strip()):
                        markdown_text = alternative
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Primary markdown conversion failed: {str(e)}")
            try:
                markdown_text = _html_to_markdown_with_markdownify(text)
            except Exception:
                # Last resort: strip tags and return plain text
                markdown_text = re.sub(r"<[^>]*>", "", text)
    else:
        logger.info("Input detected as plain text, applying minimal markdown formatting")
        # For plain text, just clean it up a bit
        markdown_text = text
        extraction_method_used = "none"

    # Final cleanup and formatting of the markdown
    markdown_text = _sanitize_markdown(markdown_text)
    markdown_text = _improve_markdown_formatting(markdown_text)

    processing_time = time.time() - start_time
    logger.info(f"Text cleaned and formatted as markdown in {processing_time:.2f}s")

    return {
        "markdown_text": markdown_text,
        "was_html": is_html,
        "extraction_method_used": extraction_method_used,
        "processing_time": processing_time,
        "success": True,
    }


# --- Additional Tool Functions ---


@with_tool_metrics
@with_error_handling
async def detect_content_type(text: str) -> Dict[str, Any]:
    """Analyzes text to detect its type: HTML, markdown, code, or plain text.

    Applies multiple heuristics to determine the most likely content type
    of the provided text string.

    Args:
        text: The input text to analyze

    Returns:
        Dictionary containing:
        {
            "content_type": "html",  # One of: "html", "markdown", "code", "plain_text"
            "confidence": 0.85,  # Confidence score (0.0-1.0)
            "details": {
                "html_markers": 12,  # Count of HTML markers found
                "markdown_markers": 3,  # Count of markdown markers found
                "code_markers": 1,  # Count of code markers found
                "detected_language": "javascript"  # If code is detected
            },
            "success": true
        }

    Raises:
        ToolInputError: If the input text is empty or not a string.
    """
    if not text:
        raise ToolInputError("Input text cannot be empty")
    if not isinstance(text, str):
        raise ToolInputError("Input text must be a string")

    # Initialize counters for markers
    html_markers = 0
    markdown_markers = 0
    code_markers = 0
    detected_language = None

    # Check for HTML markers
    html_patterns = [
        (r"<\s*[a-zA-Z]+[^>]*>", 1),  # HTML tag
        (r"<\s*/\s*[a-zA-Z]+\s*>", 1),  # Closing HTML tag
        (r"&[a-zA-Z]+;", 0.5),  # HTML entity
        (r"<!\s*DOCTYPE", 2),  # DOCTYPE
        (r"<!\s*--", 1),  # HTML comment
        (r"<!--.*?-->", 1),  # Complete HTML comment
        (r"<(div|span|p|a|img|table|ul|ol|li|h[1-6])\b", 1.5),  # Common HTML tags
        (r"</(div|span|p|a|img|table|ul|ol|li|h[1-6])>", 1.5),  # Common closing tags
        (r"<(html|head|body|meta|link|script|style)\b", 2),  # Structure tags
        (r"</(html|head|body|script|style)>", 2),  # Structure closing tags
        (r"style\s*=\s*['\"]", 1),  # style attribute
        (r"class\s*=\s*['\"]", 1),  # class attribute
        (r"id\s*=\s*['\"]", 1),  # id attribute
        (r"href\s*=\s*['\"]", 1),  # href attribute
        (r"src\s*=\s*['\"]", 1),  # src attribute
    ]

    for pattern, weight in html_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        html_markers += len(matches) * weight

    # Check for Markdown markers
    markdown_patterns = [
        (r"^#\s+.+$", 2),  # Heading level 1
        (r"^#{2,6}\s+.+$", 1.5),  # Headings levels 2-6
        (r"^\s*[*-]\s+.+$", 1),  # Unordered list
        (r"^\s*\d+\.\s+.+$", 1),  # Ordered list
        (r"^\s*>\s+.+$", 1.5),  # Blockquote
        (r"\[.+?\]\(.+?\)", 2),  # Link
        (r"!\[.+?\]\(.+?\)", 2),  # Image
        (r"`[^`\n]+`", 1),  # Inline code
        (r"^```\s*\w*$", 2),  # Code block start
        (r"^```$", 2),  # Code block end
        (r"\*\*.+?\*\*", 1),  # Bold
        (r"\*.+?\*", 0.5),  # Italic
        (r"__(.+?)__", 1),  # Bold with underscore
        (r"_(.+?)_", 0.5),  # Italic with underscore
        (r"~~.+?~~", 1),  # Strikethrough
        (r"^\s*[-*_]{3,}\s*$", 1.5),  # Horizontal rule
        (r"^\s*\|(.+\|)+\s*$", 2),  # Table row
        (r"^\s*\|([-:]+\|)+\s*$", 3),  # Table header/divider
    ]

    for pattern, weight in markdown_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        markdown_markers += len(matches) * weight

    # Check for code markers
    code_patterns = [
        (r"function\s+\w+\s*\(.*?\)\s*\{", 2),  # Function declaration
        (r"(var|let|const)\s+\w+\s*=", 1.5),  # Variable declaration JS
        (r"if\s*\(.*?\)\s*\{", 1),  # If statement
        (r"for\s*\(.*?;.*?;.*?\)\s*\{", 2),  # For loop
        (r"while\s*\(.*?\)\s*\{", 2),  # While loop
        (r"class\s+\w+(\s+extends\s+\w+)?\s*\{", 2),  # Class declaration
        (r"import\s+.*?from\s+['\"].*?['\"]", 2),  # ES6 Import
        (r"def\s+\w+\s*\(.*?\):", 2),  # Python function
        (r"class\s+\w+(\(\w+\))?:", 2),  # Python class
        (r"import\s+\w+(\s+as\s+\w+)?", 1.5),  # Python import
        (r"from\s+\w+(\.\w+)*\s+import", 1.5),  # Python from import
        (r"public\s+(static\s+)?(void|int|String)\s+\w+\s*\(", 2),  # Java method
        (r"#include\s*<.*?>", 2),  # C/C++ include
        (r"^\s*package\s+[\w\.]+;", 2),  # Java/Kotlin package
        (r"^\s*using\s+[\w\.]+;", 2),  # C# using
        (r"^\s*(public|private|protected)\s+class", 2),  # Access modifier
    ]

    for pattern, weight in code_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        code_markers += len(matches) * weight

    # Detect programming language if it looks like code
    if code_markers > 5:
        # Very basic language detection based on unique syntax
        language_patterns = [
            (r"function\s+\w+|var\s+\w+|let\s+\w+|const\s+\w+|document\.|\$\(", "javascript"),
            (r"<\?php|\$[a-zA-Z_]", "php"),
            (r"def\s+\w+\s*\(.*?\):|import\s+\w+|from\s+\w+\s+import", "python"),
            (r"public\s+class\s+\w+|public\s+static\s+void\s+main", "java"),
            (r"#include\s*<.*?>|int\s+main\s*\(", "c/c++"),
            (r"^\s*using\s+System;|namespace\s+\w+|public\s+class\s+\w+\s*:", "c#"),
            (r"module\s+\w+|fn\s+\w+|let\s+\w+|impl", "rust"),
            (r"^\s*import\s+\w+\s+from\s+['\"]|export\s+(default\s+)?", "typescript"),
            (r"^package\s+main|func\s+\w+\(|import\s+\([^)]*\)", "go"),
        ]

        for pattern, lang in language_patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                detected_language = lang
                break

    # Calculate final scores and confidence
    html_score = html_markers / max(len(text) / 100, 1)
    markdown_score = markdown_markers / max(len(text.split("\n")), 1)
    code_score = code_markers / max(len(text.split("\n")), 1)

    # Plain text has no specific markers, so it's the default fallback
    plain_text_score = 1.0 - max(
        min(html_score / 10, 1), min(markdown_score / 5, 1), min(code_score / 5, 1)
    )

    # Determine the content type
    scores = {
        "html": html_score,
        "markdown": markdown_score,
        "code": code_score,
        "plain_text": plain_text_score,
    }

    content_type = max(scores, key=scores.get)
    max_score = scores[content_type]

    # Calculate confidence based on how dominant the max score is
    total_score = sum(scores.values())
    if total_score > 0:
        confidence = max_score / total_score
    else:
        confidence = 0.25  # Equal probability for all types

    # Adjust confidence if very few markers were found
    if content_type != "plain_text" and (html_markers + markdown_markers + code_markers) < 3:
        confidence *= 0.7

    return {
        "content_type": content_type,
        "confidence": min(confidence, 1.0),
        "details": {
            "html_markers": html_markers,
            "markdown_markers": markdown_markers,
            "code_markers": code_markers,
            "detected_language": detected_language if content_type == "code" else None,
        },
        "success": True,
    }


@with_tool_metrics
@with_error_handling
async def batch_format_texts(
    texts: List[str],
    force_markdown_conversion: bool = False,
    extraction_method: str = "auto",
    max_concurrency: int = 5,
    preserve_tables: bool = True,
) -> Dict[str, Any]:
    """Processes multiple text inputs in parallel, converting each to markdown.

    Efficiently handles a batch of text inputs by processing them concurrently
    up to a specified concurrency limit.

    Args:
        texts: List of text strings to clean and format.
        force_markdown_conversion: Whether to force markdown conversion for all inputs.
                                  Default is False.
        extraction_method: Method to extract content from HTML. Options:
                          - "auto": Automatically choose the best method
                          - "readability": Use Mozilla's Readability algorithm
                          - "trafilatura": Use trafilatura library
                          - "raw": Don't extract main content, convert the whole document
                          Default is "auto".
        max_concurrency: Maximum number of texts to process simultaneously.
                        Default is 5.
        preserve_tables: Whether to preserve and convert HTML tables to markdown tables.
                        Default is True.

    Returns:
        Dictionary containing:
        {
            "results": [
                {
                    "markdown_text": "Cleaned and formatted markdown text",
                    "was_html": true,
                    "extraction_method_used": "readability"
                },
                ...
            ],
            "total_processing_time": 2.45,  # Total time in seconds
            "success_count": 5,  # Number of successfully processed texts
            "failure_count": 0,  # Number of failed texts
            "success": true
        }

    Raises:
        ToolInputError: If the input list is empty or not a list of strings.
    """
    import asyncio

    start_time = time.time()

    # Input validation
    if not texts:
        raise ToolInputError("Input texts list cannot be empty")
    if not isinstance(texts, list):
        raise ToolInputError("Input must be a list of text strings")

    # Set up concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_text(text, index):
        """Process a single text with semaphore control."""
        async with semaphore:
            try:
                result = await clean_and_format_text_as_markdown(
                    text=text,
                    force_markdown_conversion=force_markdown_conversion,
                    extraction_method=extraction_method,
                    preserve_tables=preserve_tables,
                )
                result["index"] = index  # Add original index for ordering
                return result
            except Exception as e:
                logger.error(f"Error processing text at index {index}: {str(e)}")
                return {"index": index, "error": str(e), "success": False}

    # Process all texts concurrently
    tasks = [process_text(text, i) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks)

    # Sort results by original index
    sorted_results = sorted(results, key=lambda x: x.get("index", 0))

    # Remove index from results
    for result in sorted_results:
        if "index" in result:
            del result["index"]

    # Calculate statistics
    success_count = sum(1 for result in sorted_results if result.get("success", False))
    failure_count = len(sorted_results) - success_count
    total_time = time.time() - start_time

    return {
        "results": sorted_results,
        "total_processing_time": total_time,
        "success_count": success_count,
        "failure_count": failure_count,
        "success": True,
    }


@with_tool_metrics
@with_error_handling
async def optimize_markdown_formatting(
    markdown: str,
    normalize_headings: bool = False,
    fix_lists: bool = True,
    fix_links: bool = True,
    add_line_breaks: bool = True,
    compact_mode: bool = False,
    max_line_length: int = 0,
) -> Dict[str, Any]:
    """Optimizes and improves the formatting of existing markdown text.

    Takes markdown text and enhances its formatting by fixing common issues
    and applying stylistic improvements.

    Args:
        markdown: The markdown text to optimize.
        normalize_headings: If True, ensures heading levels start at h1 and are sequential.
                           Default is False.
        fix_lists: If True, fixes common issues with list formatting.
                  Default is True.
        fix_links: If True, fixes common issues with link formatting.
                  Default is True.
        add_line_breaks: If True, ensures proper paragraph breaks.
                        Default is True.
        compact_mode: If True, reduces whitespace for a more compact presentation.
                     Default is False.
        max_line_length: Maximum line length for wrapping. 0 means no wrapping.
                        Default is 0.

    Returns:
        Dictionary containing:
        {
            "optimized_markdown": "Cleaned and formatted markdown text",
            "changes_made": {
                "headings_normalized": true,
                "lists_fixed": true,
                "links_fixed": true,
                "line_breaks_added": true
            },
            "processing_time": 0.15,  # Time taken in seconds
            "success": true
        }

    Raises:
        ToolInputError: If the input markdown is empty or not a string.
    """
    import re

    start_time = time.time()

    # Input validation
    if not markdown:
        raise ToolInputError("Input markdown cannot be empty")
    if not isinstance(markdown, str):
        raise ToolInputError("Input markdown must be a string")

    # Track changes made
    changes_made = {
        "headings_normalized": False,
        "lists_fixed": False,
        "links_fixed": False,
        "line_breaks_added": False,
        "whitespace_adjusted": False,
    }

    optimized = markdown

    # Fix markdown heading formatting (ensure space after #)
    if "#" in optimized:
        original = optimized
        optimized = re.sub(r"(^|\n)(#{1,6})([^#\s])", r"\1\2 \3", optimized)
        changes_made["headings_normalized"] = original != optimized

    # Normalize heading levels if requested
    if normalize_headings and "#" in optimized:
        original = optimized

        # Find all headings and their levels
        heading_pattern = r"(^|\n)(#{1,6})\s+(.*?)(\n|$)"
        headings = [
            (m.group(2), m.group(3), m.start(), m.end())
            for m in re.finditer(heading_pattern, optimized)
        ]

        if headings:
            # Find the minimum heading level used
            min_level = min(len(h[0]) for h in headings)

            # Adjust heading levels if the minimum isn't h1
            if min_level > 1:
                # Process headings in reverse order to avoid messing up positions
                for level, text, start, end in reversed(headings):
                    new_level = "#" * (len(level) - min_level + 1)
                    replacement = f"{optimized[start : start + 1]}{new_level} {text}{optimized[end - 1 : end]}"
                    optimized = optimized[:start] + replacement + optimized[end:]

                changes_made["headings_normalized"] = True

    # Fix list formatting
    if fix_lists and any(c in optimized for c in ["-", "*", "+"]):
        original = optimized

        # Ensure consistent list markers
        optimized = re.sub(r"^([*+]) ", r"- ", optimized, flags=re.MULTILINE)

        # Fix list item spacing
        optimized = re.sub(r"(\n- .+)(\n[^-\s])", r"\1\n\2", optimized)

        # Fix indentation in nested lists
        optimized = re.sub(r"(\n- .+\n)(\s{1,3}- )", r"\1  \2", optimized)

        changes_made["lists_fixed"] = original != optimized

    # Fix link formatting
    if fix_links and "[" in optimized:
        original = optimized

        # Fix reference-style links (ensure consistent spacing)
        optimized = re.sub(r"\]\[", r"] [", optimized)

        # Fix malformed links with space between []()
        optimized = re.sub(r"\] \(", r"](", optimized)

        # Ensure proper spacing around links in sentences
        optimized = re.sub(r"([^\s])\[", r"\1 [", optimized)
        optimized = re.sub(r"\]([^\(\s])", r"] \1", optimized)

        changes_made["links_fixed"] = original != optimized

    # Add proper line breaks for readability
    if add_line_breaks:
        original = optimized

        # Ensure headings have a blank line before (except at start of document)
        optimized = re.sub(r"(?<!\n\n)(^|\n)#", r"\1\n#", optimized)

        # Ensure paragraphs have blank lines between them
        optimized = re.sub(r"(\n[^\s#>*-][^\n]+)(\n[^\s#>*-])", r"\1\n\2", optimized)

        # Clean up any excessive blank lines created
        optimized = re.sub(r"\n{3,}", r"\n\n", optimized)

        changes_made["line_breaks_added"] = original != optimized

    # Adjust whitespace based on compact_mode
    original = optimized
    if compact_mode:
        # Reduce blank lines to single blank lines
        optimized = re.sub(r"\n\s*\n", r"\n\n", optimized)

        # Remove trailing whitespace
        optimized = re.sub(r" +$", "", optimized, flags=re.MULTILINE)
    else:
        # Ensure consistent double line breaks for section transitions
        optimized = re.sub(r"(\n#{1,6}[^\n]+\n)(?!\n)", r"\1\n", optimized)

    changes_made["whitespace_adjusted"] = original != optimized

    # Apply line wrapping if specified
    if max_line_length > 0:
        import textwrap

        # Split into paragraphs, wrap each, then rejoin
        paragraphs = re.split(r"\n\s*\n", optimized)
        wrapped_paragraphs = []

        for p in paragraphs:
            # Skip wrapping for code blocks, lists, and headings
            if (
                p.strip().startswith("```")
                or re.match(r"^\s*[*\-+]", p, re.MULTILINE)
                or re.match(r"^#{1,6}\s", p.strip())
            ):
                wrapped_paragraphs.append(p)
            else:
                # Wrap regular paragraphs
                lines = p.split("\n")
                wrapped_lines = []
                for line in lines:
                    if not line.strip().startswith((">", "#", "-", "*", "+")):
                        wrapped = textwrap.fill(line, width=max_line_length)
                        wrapped_lines.append(wrapped)
                    else:
                        wrapped_lines.append(line)
                wrapped_paragraphs.append("\n".join(wrapped_lines))

        optimized = "\n\n".join(wrapped_paragraphs)

    processing_time = time.time() - start_time

    return {
        "optimized_markdown": optimized,
        "changes_made": changes_made,
        "processing_time": processing_time,
        "success": True,
    }
