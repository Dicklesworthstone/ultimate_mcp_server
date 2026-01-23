"""OCR Tools for Ultimate MCP Server.

This module provides tools for OCR (Optical Character Recognition) processing,
leveraging LLMs to improve the quality of extracted text from PDFs and images.

Features:
- PDF to image conversion with optimized preprocessing
- Multiple extraction methods (OCR, direct text extraction, hybrid approach)
- Intelligent text segmentation and processing for large documents
- LLM-based error correction and formatting
- Table detection and formatting
- Multi-language support
- Quality assessment with detailed metrics
- PDF structure analysis
- Batch processing with concurrency control
- Sophisticated caching for improved performance

Example usage:
```python
# Extract text from a PDF file with LLM correction
result = await client.tools.extract_text_from_pdf(
    file_path="document.pdf",
    extraction_method="hybrid",  # Try direct text extraction first, fall back to OCR if needed
    max_pages=5,
    skip_pages=0,
    reformat_as_markdown=True,
    suppress_headers=True
)

# Process an image file with custom preprocessing
result = await client.tools.process_image_ocr(
    image_path="scan.jpg",
    preprocessing_options={
        "denoise": True,
        "threshold": "adaptive",
        "deskew": True
    },
    ocr_language="eng+fra",  # Multi-language support
    assess_quality=True
)

# Enhance existing OCR text with LLM
result = await client.tools.enhance_ocr_text(
    ocr_text="Text with OCK errors and broken lin- es",
    reformat_as_markdown=True,
    remove_headers=True
)

# Analyze PDF structure without full extraction
info = await client.tools.analyze_pdf_structure(
    file_path="document.pdf",
    extract_metadata=True,
    extract_outline=True,
    extract_fonts=True
)

# Batch process multiple PDFs
result = await client.tools.batch_process_documents(
    folder_path="/path/to/documents",
    file_pattern="*.pdf",
    output_folder="/path/to/output",
    max_concurrency=3
)
```
"""

import asyncio
import base64
import functools
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Try importing required libraries with fallbacks
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image, ImageEnhance, ImageFilter

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pytesseract

    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

try:
    from pdf2image import convert_from_bytes, convert_from_path

    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import pymupdf  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Import tools and helpers from ultimate
from ultimate_mcp_server.constants import Provider, TaskType
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError
from ultimate_mcp_server.tools.base import (
    with_cache,
    with_error_handling,
    with_retry,
    with_tool_metrics,
)
from ultimate_mcp_server.tools.completion import generate_completion
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.ocr")

# Cache for storing preprocessed images and extracted text
OCR_CACHE = {}


# Check if required dependencies are available
def _check_ocr_dependencies():
    """Checks if OCR dependencies are available and returns a dictionary of requirements."""
    requirements = {
        "numpy": HAS_NUMPY,
        "PIL": HAS_PIL,
        "cv2": HAS_CV2,
        "pytesseract": HAS_PYTESSERACT,
        "pdf2image": HAS_PDF2IMAGE,
        "pdfplumber": HAS_PDFPLUMBER,
        "pymupdf": HAS_PYMUPDF,
    }

    missing = [lib for lib, available in requirements.items() if not available]

    if missing:
        logger.warning(f"Some OCR dependencies are missing: {', '.join(missing)}")
        logger.warning("OCR functionality may be limited. Install required packages with:")
        packages = {
            "numpy": "numpy",
            "PIL": "pillow",
            "cv2": "opencv-python-headless",
            "pytesseract": "pytesseract",
            "pdf2image": "pdf2image",
            "pdfplumber": "pdfplumber",
            "pymupdf": "pymupdf",
        }

        pip_command = f"pip install {' '.join(packages[lib] for lib in missing)}"
        logger.warning(f"  {pip_command}")

    return requirements, missing


# Check dependencies early
OCR_REQUIREMENTS, MISSING_REQUIREMENTS = _check_ocr_dependencies()

# --- Helper functions for OCR processing ---


def _validate_file_path(file_path: str, expected_extension: Optional[str] = None) -> None:
    """
    Validates a file path exists and optionally has the expected extension.

    Args:
        file_path: Path to the file to validate
        expected_extension: Optional file extension to check (e.g., '.pdf')

    Raises:
        ToolInputError: If validation fails
    """
    if not file_path:
        raise ToolInputError("File path cannot be empty")

    file_path = os.path.expanduser(os.path.normpath(file_path))

    if not os.path.exists(file_path):
        raise ToolInputError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ToolInputError(f"Path is not a file: {file_path}")

    if expected_extension and not file_path.lower().endswith(expected_extension.lower()):
        raise ToolInputError(
            f"File does not have the expected extension ({expected_extension}): {file_path}"
        )


def _get_task_type_for_ocr(extraction_method: str = "hybrid") -> str:
    """
    Returns the appropriate TaskType for OCR operations based on extraction method.

    Args:
        extraction_method: The extraction method being used

    Returns:
        The TaskType value as a string
    """
    if extraction_method == "direct":
        return TaskType.TEXT_EXTRACTION.value
    elif extraction_method == "ocr":
        return TaskType.OCR.value
    else:  # hybrid
        return TaskType.OCR.value


def _handle_provider_error(e: Exception, operation: str) -> ToolError:
    """
    Handles provider-specific errors and converts them to tool errors.

    Args:
        e: The exception that was raised
        operation: Description of the operation that failed

    Returns:
        A ToolError with appropriate message
    """
    if isinstance(e, ProviderError):
        # Handle specific provider errors
        return ToolError(f"Provider error during {operation}: {str(e)}")
    else:
        # Handle generic errors
        return ToolError(f"Error during {operation}: {str(e)}")


def _preprocess_image(
    image: Image.Image, preprocessing_options: Optional[Dict[str, Any]] = None
) -> Image.Image:
    """
    Preprocesses an image for better OCR results.

    Args:
        image: PIL Image object
        preprocessing_options: Dictionary of preprocessing options
            - denoise: Whether to apply denoising (default: True)
            - threshold: Thresholding method ('otsu', 'adaptive', 'none') (default: 'otsu')
            - deskew: Whether to deskew the image (default: True)
            - enhance_contrast: Whether to enhance contrast (default: True)
            - enhance_brightness: Whether to enhance brightness (default: False)
            - enhance_sharpness: Whether to enhance sharpness (default: False)
            - apply_filters: List of filters to apply (default: [])
            - resize_factor: Factor to resize the image by (default: 1.0)

    Returns:
        Preprocessed PIL Image object
    """
    if not HAS_CV2 or not HAS_NUMPY or not HAS_PIL:
        logger.warning(
            "Image preprocessing requires opencv-python, numpy, and pillow. Using original image."
        )
        return image

    # Default preprocessing options
    if preprocessing_options is None:
        preprocessing_options = {
            "denoise": True,
            "threshold": "otsu",
            "deskew": True,
            "enhance_contrast": True,
            "enhance_brightness": False,
            "enhance_sharpness": False,
            "apply_filters": [],
            "resize_factor": 1.0,
        }

    # Apply PIL enhancements before OpenCV processing if enabled
    if HAS_PIL:
        # Enhance brightness if requested
        if preprocessing_options.get("enhance_brightness", False):
            enhancer = ImageEnhance.Brightness(image)
            # Increase brightness by 30%
            image = enhancer.enhance(1.3)

        # Enhance contrast if requested using PIL (in addition to OpenCV method)
        if preprocessing_options.get("enhance_contrast", True):
            enhancer = ImageEnhance.Contrast(image)
            # Increase contrast by 40%
            image = enhancer.enhance(1.4)

        # Enhance sharpness if requested
        if preprocessing_options.get("enhance_sharpness", False):
            enhancer = ImageEnhance.Sharpness(image)
            # Increase sharpness by 50%
            image = enhancer.enhance(1.5)

        # Apply filters if specified
        filters = preprocessing_options.get("apply_filters", [])
        for filter_name in filters:
            if filter_name == "unsharp_mask":
                image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
            elif filter_name == "detail":
                image = image.filter(ImageFilter.DETAIL)
            elif filter_name == "edge_enhance":
                image = image.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_name == "smooth":
                image = image.filter(ImageFilter.SMOOTH)

    # Convert PIL Image to OpenCV format
    img = np.array(image)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Calculate optimal scaling based on image size and content
    original_height, original_width = gray.shape[:2]
    resize_factor = preprocessing_options.get("resize_factor", 1.0)

    # Adaptive scaling based on image dimensions for optimal OCR
    # For very small images, increase size; for very large images, reduce
    if resize_factor == 1.0:  # Only auto-adjust if user didn't specify
        # Calculate the ideal size range for OCR (1500-3500 pixels on longest edge)
        longest_edge = max(original_width, original_height)
        if longest_edge < 1500:
            # For small images, scale up to improve OCR
            resize_factor = math.ceil(1500 / longest_edge * 10) / 10  # Round to nearest 0.1
        elif longest_edge > 3500:
            # For large images, scale down to improve performance
            resize_factor = math.floor(3500 / longest_edge * 10) / 10  # Round to nearest 0.1

    # Enhance contrast
    if preprocessing_options.get("enhance_contrast", True):
        gray = cv2.equalizeHist(gray)

    # Apply thresholding
    threshold_method = preprocessing_options.get("threshold", "otsu")
    if threshold_method == "otsu":
        _, img_thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif threshold_method == "adaptive":
        # Calculate optimal block size based on image dimensions (odd number)
        block_size = math.floor(min(gray.shape) / 30)
        block_size = max(3, block_size)
        if block_size % 2 == 0:
            block_size += 1
        img_thresholded = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2
        )
    else:
        img_thresholded = gray

    # Denoise
    if preprocessing_options.get("denoise", True):
        # Calculate optimal denoising parameters based on image size
        h_param = math.ceil(10 * math.log10(min(original_width, original_height)))
        img_denoised = cv2.fastNlMeansDenoising(img_thresholded, None, h_param, 7, 21)
    else:
        img_denoised = img_thresholded

    # Deskew
    if preprocessing_options.get("deskew", True) and HAS_NUMPY:
        try:
            coords = np.column_stack(np.where(img_denoised > 0))
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Rotate to correct skew if significant skew detected
            if abs(angle) > 0.5:
                (h, w) = img_denoised.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img_deskewed = cv2.warpAffine(
                    img_denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )
            else:
                img_deskewed = img_denoised
        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}. Using non-deskewed image.")
            img_deskewed = img_denoised
    else:
        img_deskewed = img_denoised

    # Resize if needed
    if resize_factor != 1.0:
        # Use ceiling to ensure we don't lose pixels in important small details
        new_w = math.ceil(original_width * resize_factor)
        new_h = math.ceil(original_height * resize_factor)
        img_resized = cv2.resize(img_deskewed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        img_resized = img_deskewed

    # Convert back to PIL Image
    return Image.fromarray(img_resized)


def _extract_text_with_ocr(
    image: Image.Image, ocr_language: str = "eng", ocr_config: str = ""
) -> str:
    """
    Extracts text from an image using OCR.

    Args:
        image: PIL Image object
        ocr_language: Language(s) for OCR (default: "eng")
        ocr_config: Additional configuration for Tesseract

    Returns:
        Extracted text
    """
    if not HAS_PYTESSERACT:
        raise ToolError("pytesseract is required for OCR text extraction")

    try:
        custom_config = f"-l {ocr_language} {ocr_config}"
        return pytesseract.image_to_string(image, config=custom_config)
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        raise ToolError(f"OCR extraction failed: {str(e)}") from e


def _extract_text_from_pdf_direct(
    file_path: str, start_page: int = 0, max_pages: int = 0
) -> Tuple[List[str], bool]:
    """
    Extracts text directly from a PDF file without OCR.

    Args:
        file_path: Path to the PDF file
        start_page: First page to extract (0-indexed)
        max_pages: Maximum number of pages to extract (0 = all)

    Returns:
        Tuple of (extracted_text_list, has_text)
    """
    texts = []
    has_text = False

    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                end_page = (
                    total_pages if max_pages == 0 else min(start_page + max_pages, total_pages)
                )

                for i in range(start_page, end_page):
                    try:
                        page = pdf.pages[i]
                        text = page.extract_text(x_tolerance=3, y_tolerance=3)
                        if text and text.strip():
                            has_text = True
                        texts.append(text or "")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {i + 1}: {str(e)}")
                        texts.append("")
        except Exception as e:
            logger.error(f"Error extracting text directly from PDF: {str(e)}")
            raise ToolError(f"Failed to extract text directly from PDF: {str(e)}") from e

    elif HAS_PYMUPDF:
        try:
            with pymupdf.open(file_path) as doc:
                total_pages = len(doc)
                end_page = (
                    total_pages if max_pages == 0 else min(start_page + max_pages, total_pages)
                )

                for i in range(start_page, end_page):
                    try:
                        page = doc[i]
                        text = page.get_text()
                        if text and text.strip():
                            has_text = True
                        texts.append(text or "")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {i + 1}: {str(e)}")
                        texts.append("")
        except Exception as e:
            logger.error(f"Error extracting text directly from PDF: {str(e)}")
            raise ToolError(f"Failed to extract text directly from PDF: {str(e)}") from e

    else:
        logger.warning("No PDF text extraction library available (pdfplumber or PyMuPDF)")
        raise ToolError("No PDF text extraction library available. Install pdfplumber or PyMuPDF.")

    return texts, has_text


def _convert_pdf_to_images(file_path, start_page=0, max_pages=0, dpi=300):
    """
    Converts pages of a PDF file to PIL Image objects.

    Args:
        file_path: Path to the PDF file
        start_page: First page to convert (0-indexed)
        max_pages: Maximum number of pages to convert (0 = all)
        dpi: DPI for rendering (default: 300)

    Returns:
        List of PIL Image objects
    """
    if not HAS_PDF2IMAGE:
        raise ToolError("pdf2image is required for PDF to image conversion")

    try:
        # Create a temporary directory to store intermediate images
        # This helps with memory management for large PDFs
        with tempfile.TemporaryDirectory() as temp_dir:
            # pdf2image uses 1-based indexing
            first_page = start_page + 1
            last_page = None if max_pages == 0 else first_page + max_pages - 1

            # Use the temp directory for output_folder
            images = convert_from_path(
                file_path,
                dpi=dpi,
                first_page=first_page,
                last_page=last_page,
                output_folder=temp_dir,
            )

            return images
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {str(e)}")
        raise ToolError(f"Failed to convert PDF to images: {str(e)}") from e


def _convert_pdf_bytes_to_images(pdf_bytes, start_page=0, max_pages=0, dpi=300):
    """
    Converts pages of a PDF from bytes to PIL Image objects.

    Args:
        pdf_bytes: PDF content as bytes
        start_page: First page to convert (0-indexed)
        max_pages: Maximum number of pages to convert (0 = all)
        dpi: DPI for rendering (default: 300)

    Returns:
        List of PIL Image objects
    """
    if not HAS_PDF2IMAGE:
        raise ToolError("pdf2image is required for PDF to image conversion")

    try:
        # Create a temporary directory to store intermediate images
        # This helps with memory management for large PDFs
        with tempfile.TemporaryDirectory() as temp_dir:
            # pdf2image uses 1-based indexing
            first_page = start_page + 1
            last_page = None if max_pages == 0 else first_page + max_pages - 1

            # Use the temp directory for output_folder
            images = convert_from_bytes(
                pdf_bytes,
                dpi=dpi,
                first_page=first_page,
                last_page=last_page,
                output_folder=temp_dir,
            )

            return images
    except Exception as e:
        logger.error(f"PDF bytes to image conversion failed: {str(e)}")
        raise ToolError(f"Failed to convert PDF bytes to images: {str(e)}") from e


def _generate_cache_key(data, prefix="ocr"):
    """Generate a cache key for the given data."""
    if isinstance(data, str) and os.path.exists(data):
        # For file paths, use mtime and size
        stat = os.stat(data)
        key_data = f"{data}:{stat.st_mtime}:{stat.st_size}"
    elif isinstance(data, Image.Image):
        # For PIL images, convert to bytes and hash
        img_bytes = io.BytesIO()
        data.save(img_bytes, format=data.format or "PNG")
        key_data = img_bytes.getvalue()
    elif isinstance(data, dict):
        # For dictionaries, convert to JSON
        key_data = json.dumps(data, sort_keys=True)
    else:
        # For other data, use string representation
        key_data = str(data)

    # Generate hash
    h = hashlib.md5(key_data.encode() if isinstance(key_data, str) else key_data)

    # Add a UUID component for uniqueness across process restarts
    unique_id = str(uuid.uuid4())[:8]

    return f"{prefix}_{h.hexdigest()}_{unique_id}"


def _split_text_into_chunks(text, max_chunk_size=8000, overlap=200):
    """
    Splits text into chunks of specified maximum size with overlap.

    Args:
        text: Text to split
        max_chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # Ensure reasonable values
    max_chunk_size = max(1000, min(max_chunk_size, 15000))
    overlap = max(50, min(overlap, max_chunk_size // 4))

    # Split by paragraphs first
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        para_length = len(paragraph)

        if current_length + para_length <= max_chunk_size:
            # Paragraph fits in current chunk
            current_chunk.append(paragraph)
            current_length += para_length + 2  # +2 for the newlines
        else:
            # Paragraph doesn't fit
            if current_chunk:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk))

            if para_length <= max_chunk_size:
                # Start new chunk with this paragraph
                current_chunk = [paragraph]
                current_length = para_length + 2
            else:
                # Paragraph too large, split into sentences
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                current_chunk = []
                current_length = 0

                for sentence in sentences:
                    sentence_length = len(sentence)

                    if current_length + sentence_length <= max_chunk_size:
                        # Sentence fits in current chunk
                        current_chunk.append(sentence)
                        current_length += sentence_length + 1  # +1 for the space
                    else:
                        # Sentence doesn't fit
                        if current_chunk:
                            # Save current chunk
                            chunks.append(" ".join(current_chunk))

                        if sentence_length <= max_chunk_size:
                            # Start new chunk with this sentence
                            current_chunk = [sentence]
                            current_length = sentence_length + 1
                        else:
                            # Sentence too large, split by words
                            words = sentence.split()
                            current_chunk = []
                            current_length = 0
                            current_part = []
                            part_length = 0

                            for word in words:
                                word_length = len(word)

                                if part_length + word_length + 1 <= max_chunk_size:
                                    current_part.append(word)
                                    part_length += word_length + 1  # +1 for the space
                                else:
                                    if current_part:
                                        chunks.append(" ".join(current_part))
                                    current_part = [word]
                                    part_length = word_length + 1

                            if current_part:
                                current_chunk = current_part
                                current_length = part_length

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append("\n\n".join(current_chunk) if len(current_chunk) > 1 else current_chunk[0])

    # Add overlap between chunks
    result = []
    prev_end = ""

    for i, chunk in enumerate(chunks):
        if i > 0 and prev_end:
            # Find a good overlap point (try to break at paragraph or sentence)
            overlap_text = prev_end
            if "\n\n" in overlap_text:
                parts = overlap_text.split("\n\n")
                if len(parts) > 1:
                    overlap_text = parts[-1]

            # Prepend overlap to current chunk
            chunk = overlap_text + " " + chunk

        # Save end of current chunk for next iteration
        prev_end = chunk[-overlap:] if len(chunk) > overlap else chunk

        result.append(chunk)

    return result


def _detect_tables(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Detects potential tables in an image.

    Args:
        image: PIL Image object

    Returns:
        List of detected table regions as (x, y, width, height) tuples
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []

    # Convert PIL Image to OpenCV format
    img = np.array(image)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Apply thresholding and morphological operations
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Create a kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=5)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find potential tables
    table_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Tables usually have a certain aspect ratio and size
        aspect_ratio = w / h
        area = w * h
        img_area = img.shape[0] * img.shape[1]

        if 0.5 <= aspect_ratio <= 3.0 and area > img_area * 0.05:
            table_regions.append((x, y, w, h))

    return table_regions


def _crop_image(image: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crops an image to the specified region.

    Args:
        image: PIL Image object
        region: Tuple of (x, y, width, height)

    Returns:
        Cropped PIL Image object
    """
    x, y, width, height = region
    return image.crop((x, y, x + width, y + height))


def _is_text_mostly_noise(text, noise_threshold=0.3):
    """Determine if extracted text is mostly noise based on character distribution."""
    if not text or len(text) < 10:
        return False

    # Calculate the ratio of non-alphanumeric and non-punctuation characters
    total_chars = len(text)
    valid_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,;:!?\"-'()[]{}")

    noise_ratio = 1 - (valid_chars / total_chars)
    return noise_ratio > noise_threshold


def _is_likely_header_or_footer(text, line_length_threshold=50):
    """Determine if a text line is likely a header or footer."""
    text = text.strip()
    if len(text) == 0:
        return False

    # Short lines with page numbers
    if len(text) < line_length_threshold and re.search(r"\b\d+\b", text):
        return True

    # Common header/footer patterns
    patterns = [
        r"^\d+$",  # Just a page number
        r"^Page\s+\d+(\s+of\s+\d+)?$",  # Page X of Y
        r"^[\w\s]+\s+\|\s+\d+$",  # Title | Page
        r"^\w+\s+\d{1,2},?\s+\d{4}$",  # Date format
        r"^Copyright",  # Copyright notices
        r"^\w+\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}$",  # Date with ordinal
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",  # Date in MM/DD/YY format
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def _remove_headers_and_footers(text, max_line_length=70):
    """
    Removes headers and footers from text.

    Args:
        text: Text to process
        max_line_length: Maximum length for a line to be considered a header/footer

    Returns:
        Text with headers and footers removed
    """
    if not text:
        return text

    # Split text into lines
    lines = text.splitlines()
    result = []

    for _i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            result.append(line)
            continue

        # Check if line is likely a header or footer
        if len(line.strip()) <= max_line_length and _is_likely_header_or_footer(line):
            # Replace with empty line to maintain spacing
            result.append("")
            continue

        result.append(line)

    # Join lines back together
    return "\n".join(result)


async def _process_text_chunk(
    chunk: str, reformat_as_markdown: bool = False, remove_headers: bool = False
) -> str:
    """
    Processes a chunk of OCR text with LLM enhancement.

    Args:
        chunk: Text chunk to process
        reformat_as_markdown: Whether to format as markdown
        remove_headers: Whether to remove headers and footers

    Returns:
        Enhanced text chunk
    """
    if not chunk.strip():
        return ""

    # First apply simple rule-based fixes
    cleaned_text = chunk

    # Fix hyphenated words at line breaks
    cleaned_text = re.sub(
        r"(\w+)-\s*\n\s*(\w+)", lambda m: f"{m.group(1)}{m.group(2)}", cleaned_text
    )

    # Remove obvious noise
    if _is_text_mostly_noise(cleaned_text):
        logger.warning("Text chunk appears to be mostly noise, applying aggressive cleaning")
        # Replace unusual characters with spaces
        cleaned_text = re.sub(r'[^\w\s.,;:!?"\'\(\)\[\]\{\}-]', " ", cleaned_text)
        # Normalize spaces
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    # Remove headers and footers if requested
    if remove_headers:
        cleaned_text = _remove_headers_and_footers(cleaned_text)

    # Prepare LLM enhancement prompt
    if reformat_as_markdown:
        prompt = f"""Correct OCR errors in this text and format it as markdown. Follow these instructions:

1. Fix OCR-induced errors:
   - Correct words split across line breaks (e.g., "cor- rect" → "correct")
   - Fix typos like 'rn' misread as 'm', '0' misread as 'O', etc.
   - Merge split paragraphs but preserve intentional paragraph breaks
   - Use context and common sense to correct errors

2. Format as markdown:
   - Convert headings to markdown headings (# for main title, ## for subtitles, etc.)
   - Format lists as proper markdown lists
   - Use emphasis (*italic*) and strong (**bold**) where appropriate
   - Create tables using markdown syntax if tabular data is detected
   - For code or equations, use appropriate markdown formatting

3. Clean up formatting:
   - Remove unnecessary line breaks within paragraphs
   - Preserve paragraph structure
   - Remove duplicated text
   - {"Remove headers, footers, and page numbers" if remove_headers else "Preserve all content including headers/footers"}

4. Preserve the original content's meaning and information.

Here is the text to correct and format:

```
{cleaned_text}
```

Provide ONLY the corrected markdown text with no explanations or comments.
"""
    else:
        prompt = f"""Correct OCR errors in this text. Follow these instructions:

1. Fix OCR-induced errors:
   - Correct words split across line breaks (e.g., "cor- rect" → "correct")
   - Fix typos like 'rn' misread as 'm', '0' misread as 'O', etc.
   - Merge split paragraphs but preserve intentional paragraph breaks
   - Use context and common sense to correct errors

2. Clean up formatting:
   - Remove unnecessary line breaks within paragraphs
   - Preserve paragraph structure
   - Remove duplicated text
   - {"Remove headers, footers, and page numbers" if remove_headers else "Preserve all content including headers/footers"}

3. Preserve the original content's meaning and information.

Here is the text to correct:

```
{cleaned_text}
```

Provide ONLY the corrected text with no explanations or comments.
"""

    try:
        # Use generate_completion to process the text
        task_type = TaskType.TEXT_ENHANCEMENT.value

        result = await generate_completion(
            prompt=prompt,
            provider=Provider.ANTHROPIC.value,  # Default to Anthropic for high-quality text processing
            temperature=0.2,  # Low temperature for consistent results
            max_tokens=len(cleaned_text) + 1000,  # Allow some expansion for formatting
            task_type=task_type,
        )

        if not result or not result.get("text"):
            logger.warning("LLM text enhancement returned empty result")
            return cleaned_text

        enhanced_text = result["text"]

        # Remove any "Here is the corrected..." prefixes that LLMs sometimes add
        enhanced_text = re.sub(
            r"^(Here is|The corrected|Here\'s)[^:]*:?\s*", "", enhanced_text, flags=re.IGNORECASE
        )

        return enhanced_text
    except ProviderError as e:
        logger.error(f"Provider error during text enhancement: {str(e)}")
        # Fall back to the cleaned text
        return cleaned_text
    except Exception as e:
        logger.error(f"Error during LLM text enhancement: {str(e)}")
        # Fall back to the cleaned text
        return cleaned_text


# --- Main OCR tool functions ---


@with_cache(ttl=24 * 60 * 60)  # Cache for 24 hours
@with_tool_metrics
@with_retry(max_retries=3, retry_delay=1)
@with_error_handling
async def extract_text_from_pdf(
    file_path: str,
    extraction_method: str = "hybrid",
    max_pages: int = 0,
    skip_pages: int = 0,
    preprocessing_options: Optional[Dict[str, Any]] = None,
    ocr_language: str = "eng",
    reformat_as_markdown: bool = False,
    suppress_headers: bool = False,
    assess_quality: bool = False,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Extracts and enhances text from a PDF document.

    This tool can use multiple extraction methods: direct text extraction from the PDF,
    OCR-based extraction, or a hybrid approach that uses direct extraction when possible
    and falls back to OCR when necessary. The extracted text is then enhanced using an
    LLM to correct OCR errors and optionally format the output as markdown.

    Args:
        file_path: Path to the PDF file
        extraction_method: Method to use for text extraction:
            - "direct": Extract text directly from the PDF (fastest, but may fail for scanned PDFs)
            - "ocr": Always use OCR (slower but works for scanned PDFs)
            - "hybrid": Try direct extraction first, fall back to OCR if needed (default)
        max_pages: Maximum number of pages to process (0 = all pages)
        skip_pages: Number of pages to skip from the beginning (0-indexed)
        preprocessing_options: Dictionary of options for image preprocessing:
            - denoise: Whether to apply denoising (default: True)
            - threshold: Thresholding method ('otsu', 'adaptive', 'none') (default: 'otsu')
            - deskew: Whether to deskew the image (default: True)
            - enhance_contrast: Whether to enhance contrast (default: True)
            - resize_factor: Factor to resize the image (default: 1.0)
        ocr_language: Language(s) for OCR, e.g., "eng" or "eng+fra" (default: "eng")
        reformat_as_markdown: Whether to format the output as markdown (default: False)
        suppress_headers: Whether to remove headers, footers, and page numbers (default: False)
        assess_quality: Whether to assess the quality of the OCR improvement (default: False)
        dpi: DPI for PDF rendering when using OCR (default: 300)

    Returns:
        A dictionary containing:
        {
            "success": true,
            "text": "The extracted and enhanced text...",
            "raw_text": "The original OCR text before enhancement...",
            "pages_processed": 5,
            "extraction_method_used": "hybrid",
            "file_path": "/path/to/document.pdf",
            "quality_metrics": {  # Only if assess_quality=True
                "score": 85,
                "explanation": "Explanation of quality score..."
            },
            "processing_time": 12.34  # Seconds
        }

    Raises:
        ToolInputError: If the file path is invalid or the file is not a PDF
        ToolError: If text extraction fails
    """
    start_time = time.time()

    # Validate file path
    _validate_file_path(file_path, expected_extension=".pdf")

    # Check extraction method
    valid_methods = ["direct", "ocr", "hybrid"]
    if extraction_method not in valid_methods:
        raise ToolInputError(
            f"Invalid extraction method: '{extraction_method}'. Must be one of: {', '.join(valid_methods)}"
        )

    # Check dependencies based on extraction method
    if extraction_method in ["ocr", "hybrid"]:
        if not HAS_PDF2IMAGE or not HAS_PYTESSERACT:
            logger.warning(
                f"OCR extraction requires pdf2image and pytesseract. {extraction_method} may fail."
            )

    if extraction_method in ["direct", "hybrid"]:
        if not HAS_PDFPLUMBER and not HAS_PYMUPDF:
            logger.warning("Direct extraction requires pdfplumber or PyMuPDF.")

    # Initialize result
    result = {
        "success": False,
        "file_path": file_path,
        "pages_processed": 0,
        "extraction_method_used": extraction_method,
    }

    method_used = extraction_method
    raw_text_list = []
    extracted_text_list = []
    has_direct_text = False

    try:
        # Step 1: Extract text
        if extraction_method in ["direct", "hybrid"]:
            try:
                logger.info(f"Attempting direct text extraction from PDF: {file_path}")
                direct_text_list, has_direct_text = _extract_text_from_pdf_direct(
                    file_path, start_page=skip_pages, max_pages=max_pages
                )

                raw_text_list = direct_text_list
                logger.info(
                    f"Direct text extraction {'succeeded' if has_direct_text else 'failed'}"
                )

                if has_direct_text and extraction_method == "direct":
                    # If direct extraction found text and that's the requested method, we're done
                    method_used = "direct"
                    extracted_text_list = direct_text_list
                    logger.info(
                        f"Using direct extraction result with {len(extracted_text_list)} pages"
                    )

                elif has_direct_text and extraction_method == "hybrid":
                    # If hybrid mode and direct extraction worked, use it
                    method_used = "direct"
                    extracted_text_list = direct_text_list
                    logger.info(
                        f"Using direct extraction result with {len(extracted_text_list)} pages (hybrid mode)"
                    )

                elif extraction_method == "direct" and not has_direct_text:
                    # If direct mode but no text found, we fail
                    raise ToolError("Direct text extraction failed to find text in the PDF")

                # If hybrid mode and no text found, fall back to OCR
                if extraction_method == "hybrid" and not has_direct_text:
                    logger.info(
                        "No text found via direct extraction, falling back to OCR (hybrid mode)"
                    )
                    method_used = "ocr"
                    # Continue to OCR extraction below

            except Exception as e:
                logger.error(f"Direct text extraction failed: {str(e)}")
                if extraction_method == "direct":
                    raise ToolError(f"Direct text extraction failed: {str(e)}") from e

                logger.info("Falling back to OCR extraction")
                method_used = "ocr"

        # Step 2: OCR extraction if needed
        if method_used == "ocr" or extraction_method == "ocr":
            method_used = "ocr"
            logger.info(f"Performing OCR-based text extraction on PDF: {file_path}")

            # Convert PDF to images
            images = _convert_pdf_to_images(
                file_path, start_page=skip_pages, max_pages=max_pages, dpi=dpi
            )

            # Extract text using OCR
            raw_text_list = []
            with ThreadPoolExecutor() as executor:
                # Preprocess images in parallel
                preprocessed_images = list(
                    executor.map(lambda img: _preprocess_image(img, preprocessing_options), images)
                )

                # Extract text in parallel
                ocr_config = ""
                ocr_results = list(
                    executor.map(
                        lambda img: _extract_text_with_ocr(img, ocr_language, ocr_config),
                        preprocessed_images,
                    )
                )

            extracted_text_list = ocr_results
            raw_text_list = ocr_results
            logger.info(f"OCR extraction completed for {len(extracted_text_list)} pages")

        # Step 3: Process extracted text
        logger.info("Processing extracted text with LLM enhancement")

        # Combine text from pages
        full_raw_text = "\n\n".join(raw_text_list)

        # Split into chunks for LLM processing
        chunks = _split_text_into_chunks(full_raw_text)
        logger.info(f"Text split into {len(chunks)} chunks for LLM processing")

        # Process chunks in parallel
        enhanced_chunks = await asyncio.gather(
            *[
                _process_text_chunk(chunk, reformat_as_markdown, suppress_headers)
                for chunk in chunks
            ]
        )

        # Combine chunks
        enhanced_text = "\n\n".join(enhanced_chunks)

        # Step 4: Assess quality if requested
        quality_metrics = None
        if assess_quality:
            logger.info("Assessing quality of text enhancement")
            quality_metrics = await _assess_text_quality(full_raw_text, enhanced_text)

        # Prepare final result
        processing_time = time.time() - start_time
        result.update(
            {
                "success": True,
                "text": enhanced_text,
                "raw_text": full_raw_text,
                "pages_processed": len(raw_text_list),
                "extraction_method_used": method_used,
                "processing_time": processing_time,
            }
        )

        if quality_metrics:
            result["quality_metrics"] = quality_metrics

        logger.info(
            f"Text extraction and enhancement completed successfully in {processing_time:.2f}s"
        )
        return result

    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf: {str(e)}")
        logger.error(traceback.format_exc())
        raise ToolError(f"Failed to extract and enhance text from PDF: {str(e)}") from e


@with_cache(ttl=24 * 60 * 60)  # Cache for 24 hours
@with_tool_metrics
@with_retry(max_retries=3, retry_delay=1)
@with_error_handling
async def extract_text_from_pdf_bytes(
    pdf_bytes: bytes,
    extraction_method: str = "hybrid",
    max_pages: int = 0,
    skip_pages: int = 0,
    preprocessing_options: Optional[Dict[str, Any]] = None,
    ocr_language: str = "eng",
    reformat_as_markdown: bool = False,
    suppress_headers: bool = False,
    assess_quality: bool = False,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Extracts and enhances text from PDF bytes data.

    This tool works like extract_text_from_pdf but accepts PDF data as bytes instead of a file path.
    It can use multiple extraction methods and enhance the extracted text using an LLM.

    Args:
        pdf_bytes: PDF content as bytes
        extraction_method: Method to use for text extraction:
            - "direct": Extract text directly from the PDF (fastest, but may fail for scanned PDFs)
            - "ocr": Always use OCR (slower but works for scanned PDFs)
            - "hybrid": Try direct extraction first, fall back to OCR if needed (default)
        max_pages: Maximum number of pages to process (0 = all pages)
        skip_pages: Number of pages to skip from the beginning (0-indexed)
        preprocessing_options: Dictionary of options for image preprocessing
        ocr_language: Language(s) for OCR, e.g., "eng" or "eng+fra" (default: "eng")
        reformat_as_markdown: Whether to format the output as markdown (default: False)
        suppress_headers: Whether to remove headers, footers, and page numbers (default: False)
        assess_quality: Whether to assess the quality of the OCR improvement (default: False)
        dpi: DPI for PDF rendering when using OCR (default: 300)

    Returns:
        A dictionary with the extracted and enhanced text, same format as extract_text_from_pdf

    Raises:
        ToolInputError: If the PDF bytes are invalid
        ToolError: If text extraction fails
    """
    start_time = time.time()

    # Validate input
    if not pdf_bytes:
        raise ToolInputError("PDF bytes cannot be empty")

    # Check extraction method
    valid_methods = ["direct", "ocr", "hybrid"]
    if extraction_method not in valid_methods:
        raise ToolInputError(
            f"Invalid extraction method: '{extraction_method}'. Must be one of: {', '.join(valid_methods)}"
        )

    # Check dependencies based on extraction method
    if extraction_method in ["ocr", "hybrid"]:
        if not HAS_PDF2IMAGE or not HAS_PYTESSERACT:
            logger.warning(
                f"OCR extraction requires pdf2image and pytesseract. {extraction_method} may fail."
            )

    if extraction_method in ["direct", "hybrid"]:
        if not HAS_PDFPLUMBER and not HAS_PYMUPDF:
            logger.warning("Direct extraction requires pdfplumber or PyMuPDF.")

    # Initialize result
    result = {"success": False, "pages_processed": 0, "extraction_method_used": extraction_method}

    method_used = extraction_method
    raw_text_list = []
    extracted_text_list = []
    has_direct_text = False

    try:
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_path = temp_pdf.name
            temp_pdf.write(pdf_bytes)
            temp_pdf.flush()

        try:
            # Step 1: Extract text
            if extraction_method in ["direct", "hybrid"]:
                try:
                    logger.info("Attempting direct text extraction from PDF bytes")
                    direct_text_list, has_direct_text = _extract_text_from_pdf_direct(
                        temp_path, start_page=skip_pages, max_pages=max_pages
                    )

                    raw_text_list = direct_text_list
                    logger.info(
                        f"Direct text extraction {'succeeded' if has_direct_text else 'failed'}"
                    )

                    if has_direct_text and extraction_method == "direct":
                        method_used = "direct"
                        extracted_text_list = direct_text_list
                        logger.info(
                            f"Using direct extraction result with {len(extracted_text_list)} pages"
                        )

                    elif has_direct_text and extraction_method == "hybrid":
                        method_used = "direct"
                        extracted_text_list = direct_text_list
                        logger.info(
                            f"Using direct extraction result with {len(extracted_text_list)} pages (hybrid mode)"
                        )

                    elif extraction_method == "direct" and not has_direct_text:
                        raise ToolError("Direct text extraction failed to find text in the PDF")

                    if extraction_method == "hybrid" and not has_direct_text:
                        logger.info(
                            "No text found via direct extraction, falling back to OCR (hybrid mode)"
                        )
                        method_used = "ocr"

                except Exception as e:
                    logger.error(f"Direct text extraction failed: {str(e)}")
                    if extraction_method == "direct":
                        raise ToolError(f"Direct text extraction failed: {str(e)}") from e

                    logger.info("Falling back to OCR extraction")
                    method_used = "ocr"

            # Step 2: OCR extraction if needed
            if method_used == "ocr" or extraction_method == "ocr":
                method_used = "ocr"
                logger.info("Performing OCR-based text extraction on PDF bytes")

                # Convert PDF bytes to images
                images = _convert_pdf_bytes_to_images(
                    pdf_bytes, start_page=skip_pages, max_pages=max_pages, dpi=dpi
                )

                # Extract text using OCR
                raw_text_list = []
                with ThreadPoolExecutor() as executor:
                    # Preprocess images in parallel
                    preprocessed_images = list(
                        executor.map(
                            lambda img: _preprocess_image(img, preprocessing_options), images
                        )
                    )

                    # Extract text in parallel
                    ocr_config = ""
                    ocr_results = list(
                        executor.map(
                            lambda img: _extract_text_with_ocr(img, ocr_language, ocr_config),
                            preprocessed_images,
                        )
                    )

                extracted_text_list = ocr_results
                raw_text_list = ocr_results
                logger.info(f"OCR extraction completed for {len(extracted_text_list)} pages")

            # Step 3: Process extracted text
            logger.info("Processing extracted text with LLM enhancement")

            # Combine text from pages
            full_raw_text = "\n\n".join(raw_text_list)

            # Split into chunks for LLM processing
            chunks = _split_text_into_chunks(full_raw_text)
            logger.info(f"Text split into {len(chunks)} chunks for LLM processing")

            # Process chunks in parallel
            enhanced_chunks = await asyncio.gather(
                *[
                    _process_text_chunk(chunk, reformat_as_markdown, suppress_headers)
                    for chunk in chunks
                ]
            )

            # Combine chunks
            enhanced_text = "\n\n".join(enhanced_chunks)

            # Step 4: Assess quality if requested
            quality_metrics = None
            if assess_quality:
                logger.info("Assessing quality of text enhancement")
                quality_metrics = await _assess_text_quality(full_raw_text, enhanced_text)

            # Prepare final result
            processing_time = time.time() - start_time
            result.update(
                {
                    "success": True,
                    "text": enhanced_text,
                    "raw_text": full_raw_text,
                    "pages_processed": len(raw_text_list),
                    "extraction_method_used": method_used,
                    "processing_time": processing_time,
                }
            )

            if quality_metrics:
                result["quality_metrics"] = quality_metrics

            logger.info(
                f"Text extraction and enhancement completed successfully in {processing_time:.2f}s"
            )
            return result

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")

    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf_bytes: {str(e)}")
        logger.error(traceback.format_exc())
        raise ToolError(f"Failed to extract and enhance text from PDF bytes: {str(e)}") from e


@with_cache(ttl=24 * 60 * 60)  # Cache for 24 hours
@with_tool_metrics
@with_retry(max_retries=2, retry_delay=1)
@with_error_handling
async def process_image_ocr(
    image_path: Optional[str] = None,
    image_data: Optional[str] = None,
    preprocessing_options: Optional[Dict[str, Any]] = None,
    ocr_language: str = "eng",
    reformat_as_markdown: bool = False,
    assess_quality: bool = False,
) -> Dict[str, Any]:
    """
    Processes an image with OCR and enhances the extracted text.

    This tool accepts either a path to an image file or base64-encoded image data,
    performs OCR on the image, and then enhances the extracted text using an LLM.

    Args:
        image_path: Path to the image file (mutually exclusive with image_data)
        image_data: Base64-encoded image data (mutually exclusive with image_path)
        preprocessing_options: Dictionary of options for image preprocessing:
            - denoise: Whether to apply denoising (default: True)
            - threshold: Thresholding method ('otsu', 'adaptive', 'none') (default: 'otsu')
            - deskew: Whether to deskew the image (default: True)
            - enhance_contrast: Whether to enhance contrast (default: True)
            - resize_factor: Factor to resize the image (default: 1.0)
        ocr_language: Language(s) for OCR, e.g., "eng" or "eng+fra" (default: "eng")
        reformat_as_markdown: Whether to format the output as markdown (default: False)
        assess_quality: Whether to assess the quality of the OCR improvement (default: False)

    Returns:
        A dictionary containing:
        {
            "success": true,
            "text": "The extracted and enhanced text...",
            "raw_text": "The original OCR text before enhancement...",
            "table_detected": false,  # Whether a table was detected in the image
            "quality_metrics": {  # Only if assess_quality=True
                "score": 85,
                "explanation": "Explanation of quality score..."
            },
            "processing_time": 3.45  # Seconds
        }

    Raises:
        ToolInputError: If input is invalid
        ToolError: If processing fails
    """
    start_time = time.time()

    # Check dependencies
    if not HAS_PIL or not HAS_PYTESSERACT:
        missing = []
        if not HAS_PIL:
            missing.append("pillow")
        if not HAS_PYTESSERACT:
            missing.append("pytesseract")
        raise ToolError(f"Required dependencies missing: {', '.join(missing)}")

    # Validate input
    if not image_path and not image_data:
        raise ToolInputError("Either image_path or image_data must be provided")

    if image_path and image_data:
        raise ToolInputError("Only one of image_path or image_data should be provided")

    try:
        # Load image
        if image_path:
            _validate_file_path(image_path)
            image = Image.open(image_path)
        else:
            # Decode base64 image data
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                raise ToolInputError(f"Invalid base64 image data: {str(e)}") from e

        # Preprocess image
        logger.info("Preprocessing image for OCR")
        preprocessed_image = _preprocess_image(image, preprocessing_options)

        # Detect tables
        table_regions = _detect_tables(preprocessed_image)
        table_detected = len(table_regions) > 0
        logger.info(f"Table detection: {len(table_regions)} potential tables found")

        # Extract text with OCR
        logger.info(f"Performing OCR with language(s): {ocr_language}")
        raw_text = _extract_text_with_ocr(preprocessed_image, ocr_language)

        # Process tables separately if detected
        table_texts = []
        if table_detected and HAS_CV2:
            logger.info("Processing detected tables separately")
            for i, region in enumerate(table_regions):
                try:
                    table_image = _crop_image(preprocessed_image, region)
                    # Use a different preprocessing for tables (less aggressive)
                    table_options = {"denoise": True, "threshold": "adaptive", "deskew": False}
                    processed_table_image = _preprocess_image(table_image, table_options)
                    table_text = _extract_text_with_ocr(processed_table_image, ocr_language)
                    if table_text.strip():
                        table_texts.append(f"\n\nTable {i + 1}:\n{table_text}\n")
                except Exception as e:
                    logger.warning(f"Error processing table {i + 1}: {str(e)}")

        # Include table texts with the main text
        if table_texts:
            raw_text += "\n\n" + "\n".join(table_texts)

        # Process with LLM
        logger.info("Processing extracted text with LLM enhancement")
        enhanced_text = await _process_text_chunk(
            raw_text, reformat_as_markdown, suppress_headers=False
        )

        # Assess quality if requested
        quality_metrics = None
        if assess_quality:
            logger.info("Assessing quality of text enhancement")
            quality_metrics = await _assess_text_quality(raw_text, enhanced_text)

        # Prepare result
        processing_time = time.time() - start_time
        result = {
            "success": True,
            "text": enhanced_text,
            "raw_text": raw_text,
            "table_detected": table_detected,
            "processing_time": processing_time,
        }

        if quality_metrics:
            result["quality_metrics"] = quality_metrics

        logger.info(f"Image OCR processing completed in {processing_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error in process_image_ocr: {str(e)}")
        logger.error(traceback.format_exc())
        raise ToolError(f"Failed to process image with OCR: {str(e)}") from e


@with_cache(ttl=24 * 60 * 60)  # Cache for 24 hours
@with_tool_metrics
@with_retry(max_retries=2, retry_delay=1)
@with_error_handling
async def enhance_ocr_text(
    ocr_text: str,
    reformat_as_markdown: bool = False,
    remove_headers: bool = False,
    detect_tables: bool = True,
    assess_quality: bool = False,
) -> Dict[str, Any]:
    """
    Enhances existing OCR text using an LLM to correct errors and improve formatting.

    This tool takes OCR text (e.g., from a different OCR engine) and uses an LLM to
    correct errors, improve formatting, and optionally convert to markdown.

    Args:
        ocr_text: The OCR text to enhance
        reformat_as_markdown: Whether to format the output as markdown (default: False)
        remove_headers: Whether to remove headers, footers, and page numbers (default: False)
        detect_tables: Whether to attempt to detect and format tables (default: True)
        assess_quality: Whether to assess the quality of the OCR improvement (default: False)

    Returns:
        A dictionary containing:
        {
            "success": true,
            "text": "The enhanced text...",
            "raw_text": "The original OCR text...",
            "quality_metrics": {  # Only if assess_quality=True
                "score": 85,
                "explanation": "Explanation of quality score..."
            },
            "processing_time": 2.34  # Seconds
        }

    Raises:
        ToolInputError: If the OCR text is empty
        ToolError: If enhancement fails
    """
    start_time = time.time()

    # Validate input
    if not ocr_text or not isinstance(ocr_text, str):
        raise ToolInputError("OCR text must be a non-empty string")

    try:
        # Split into chunks if large
        if len(ocr_text) > 10000:
            logger.info(f"Splitting large OCR text ({len(ocr_text)} chars) into chunks")
            chunks = _split_text_into_chunks(ocr_text)

            # Process chunks in parallel
            enhanced_chunks = await asyncio.gather(
                *[
                    _process_text_chunk(chunk, reformat_as_markdown, remove_headers)
                    for chunk in chunks
                ]
            )

            # Combine chunks
            enhanced_text = "\n\n".join(enhanced_chunks)
            logger.info(f"Processed {len(chunks)} text chunks")
        else:
            # Process directly if small enough
            enhanced_text = await _process_text_chunk(
                ocr_text, reformat_as_markdown, remove_headers
            )

        # Detect and format tables if requested
        if detect_tables and reformat_as_markdown:
            logger.info("Attempting table detection and formatting")
            enhanced_text = await _format_tables_in_text(enhanced_text)

        # Assess quality if requested
        quality_metrics = None
        if assess_quality:
            logger.info("Assessing quality of text enhancement")
            quality_metrics = await _assess_text_quality(ocr_text, enhanced_text)

        # Prepare result
        processing_time = time.time() - start_time
        result = {
            "success": True,
            "text": enhanced_text,
            "raw_text": ocr_text,
            "processing_time": processing_time,
        }

        if quality_metrics:
            result["quality_metrics"] = quality_metrics

        logger.info(f"OCR text enhancement completed in {processing_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error in enhance_ocr_text: {str(e)}")
        logger.error(traceback.format_exc())
        raise ToolError(f"Failed to enhance OCR text: {str(e)}") from e


@with_tool_metrics
@with_retry(max_retries=2, retry_delay=1.0)
@with_error_handling
async def analyze_pdf_structure(
    file_path: str,
    extract_metadata: bool = True,
    extract_outline: bool = True,
    extract_fonts: bool = False,
    extract_images: bool = False,
    estimate_ocr_needs: bool = True,
) -> Dict[str, Any]:
    """
    Analyzes the structure of a PDF file without performing full text extraction.

    This tool examines a PDF file and provides information about its structure,
    including metadata, outline (table of contents), fonts, embedded images,
    and an assessment of whether OCR would be beneficial.

    Args:
        file_path: Path to the PDF file
        extract_metadata: Whether to extract document metadata (default: True)
        extract_outline: Whether to extract the document outline/TOC (default: True)
        extract_fonts: Whether to extract font information (default: False)
        extract_images: Whether to extract information about embedded images (default: False)
        estimate_ocr_needs: Whether to estimate if OCR would benefit this PDF (default: True)

    Returns:
        A dictionary containing:
        {
            "success": true,
            "file_path": "/path/to/document.pdf",
            "page_count": 42,
            "metadata": {  # Only if extract_metadata=True
                "title": "Document Title",
                "author": "Author Name",
                "subject": "Document Subject",
                "keywords": "keyword1, keyword2",
                "creator": "Creator Application",
                "producer": "Producer Application",
                "creation_date": "2023-01-01T12:00:00",
                "modification_date": "2023-02-01T13:00:00"
            },
            "outline": [  # Only if extract_outline=True
                {
                    "title": "Chapter 1",
                    "page": 5,
                    "children": [
                        {"title": "Section 1.1", "page": 6, "children": []}
                    ]
                },
                {"title": "Chapter 2", "page": 15, "children": []}
            ],
            "font_info": {  # Only if extract_fonts=True
                "total_fonts": 3,
                "embedded_fonts": 2,
                "font_names": ["Arial", "Times New Roman", "Courier"]
            },
            "image_info": {  # Only if extract_images=True
                "total_images": 12,
                "image_types": {"jpeg": 8, "png": 4},
                "average_size": "120kb"
            },
            "ocr_assessment": {  # Only if estimate_ocr_needs=True
                "needs_ocr": false,
                "confidence": "high",
                "reason": "PDF contains extractable text throughout"
            },
            "processing_time": 1.23  # Seconds
        }

    Raises:
        ToolInputError: If the file path is invalid or the file is not a PDF
        ToolError: If analysis fails
    """
    start_time = time.time()

    # Validate file path
    _validate_file_path(file_path, expected_extension=".pdf")

    # Check for required libraries
    pdf_lib_available = False
    if HAS_PYMUPDF:
        pdf_lib = "pymupdf"
        pdf_lib_available = True
    elif HAS_PDFPLUMBER:
        pdf_lib = "pdfplumber"
        pdf_lib_available = True

    if not pdf_lib_available:
        raise ToolError("PDF analysis requires PyMuPDF or pdfplumber")

    try:
        result = {"success": False, "file_path": file_path, "processing_time": 0}

        if pdf_lib == "pymupdf":
            # Use PyMuPDF for analysis
            with pymupdf.open(file_path) as doc:
                # Basic information
                result["page_count"] = len(doc)

                # Extract metadata if requested
                if extract_metadata:
                    metadata = doc.metadata
                    if metadata:
                        result["metadata"] = {
                            "title": metadata.get("title", ""),
                            "author": metadata.get("author", ""),
                            "subject": metadata.get("subject", ""),
                            "keywords": metadata.get("keywords", ""),
                            "creator": metadata.get("creator", ""),
                            "producer": metadata.get("producer", ""),
                            "creation_date": metadata.get("creationDate", ""),
                            "modification_date": metadata.get("modDate", ""),
                        }

                # Extract outline if requested
                if extract_outline:
                    toc = doc.get_toc()
                    if toc:
                        # Process TOC into a nested structure
                        result["outline"] = _process_toc(toc)

                # Extract font information if requested
                if extract_fonts:
                    fonts: Set[str] = set()
                    embedded_fonts: Set[str] = set()

                    for page_num in range(min(10, len(doc))):  # Analyze first 10 pages
                        page = doc[page_num]
                        page_fonts = page.get_fonts()

                        for font in page_fonts:
                            fonts.add(font[3])  # Font name
                            if font[2]:  # Embedded flag
                                embedded_fonts.add(font[3])

                    result["font_info"] = {
                        "total_fonts": len(fonts),
                        "embedded_fonts": len(embedded_fonts),
                        "font_names": list(fonts),
                    }

                # Extract image information if requested
                if extract_images:
                    image_count = 0
                    image_types: Dict[str, int] = {}
                    total_size = 0

                    for page_num in range(min(5, len(doc))):  # Analyze first 5 pages
                        page = doc[page_num]
                        images = page.get_images(full=True)

                        for img in images:
                            image_count += 1
                            xref = img[0]
                            img_info = doc.extract_image(xref)

                            if img_info:
                                img_type = img_info["ext"]
                                img_size = len(img_info["image"])

                                image_types[img_type] = image_types.get(img_type, 0) + 1
                                total_size += img_size

                    # Extrapolate total images based on sample
                    estimated_total = int(image_count * (len(doc) / max(1, min(5, len(doc)))))
                    avg_size = (
                        f"{int(total_size / max(1, image_count) / 1024)}kb"
                        if image_count > 0
                        else "0kb"
                    )

                    result["image_info"] = {
                        "total_images": image_count,
                        "estimated_total": estimated_total,
                        "image_types": image_types,
                        "average_size": avg_size,
                    }

                # Estimate OCR needs if requested
                if estimate_ocr_needs:
                    text_pages = 0
                    total_pages = len(doc)
                    sample_size = min(10, total_pages)

                    for page_num in range(sample_size):
                        page = doc[page_num]
                        text = page.get_text()
                        if text and len(text.strip()) > 50:  # Page has meaningful text
                            text_pages += 1

                    text_ratio = text_pages / sample_size

                    if text_ratio > 0.9:
                        needs_ocr = False
                        confidence = "high"
                        reason = "PDF contains extractable text throughout"
                    elif text_ratio > 0.5:
                        needs_ocr = True
                        confidence = "medium"
                        reason = "PDF has some extractable text but may benefit from OCR for certain pages"
                    else:
                        needs_ocr = True
                        confidence = "high"
                        reason = "PDF appears to be scanned or has minimal extractable text"

                    result["ocr_assessment"] = {
                        "needs_ocr": needs_ocr,
                        "confidence": confidence,
                        "reason": reason,
                        "text_coverage_ratio": text_ratio,
                    }

        elif pdf_lib == "pdfplumber":
            # Use pdfplumber for analysis
            with pdfplumber.open(file_path) as pdf:
                # Basic information
                result["page_count"] = len(pdf.pages)

                # Extract metadata if requested
                if extract_metadata:
                    metadata = pdf.metadata
                    if metadata:
                        result["metadata"] = {
                            "title": metadata.get("Title", ""),
                            "author": metadata.get("Author", ""),
                            "subject": metadata.get("Subject", ""),
                            "keywords": metadata.get("Keywords", ""),
                            "creator": metadata.get("Creator", ""),
                            "producer": metadata.get("Producer", ""),
                            "creation_date": metadata.get("CreationDate", ""),
                            "modification_date": metadata.get("ModDate", ""),
                        }

                # Outline not supported in pdfplumber
                if extract_outline:
                    result["outline"] = []

                # Font and image info not supported in pdfplumber
                if extract_fonts:
                    result["font_info"] = {"total_fonts": 0, "embedded_fonts": 0, "font_names": []}

                if extract_images:
                    result["image_info"] = {
                        "total_images": 0,
                        "image_types": {},
                        "average_size": "0kb",
                    }

                # Estimate OCR needs if requested
                if estimate_ocr_needs:
                    text_pages = 0
                    total_pages = len(pdf.pages)
                    sample_size = min(10, total_pages)

                    for page_num in range(sample_size):
                        page = pdf.pages[page_num]
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:  # Page has meaningful text
                            text_pages += 1

                    text_ratio = text_pages / sample_size

                    if text_ratio > 0.9:
                        needs_ocr = False
                        confidence = "high"
                        reason = "PDF contains extractable text throughout"
                    elif text_ratio > 0.5:
                        needs_ocr = True
                        confidence = "medium"
                        reason = "PDF has some extractable text but may benefit from OCR for certain pages"
                    else:
                        needs_ocr = True
                        confidence = "high"
                        reason = "PDF appears to be scanned or has minimal extractable text"

                    result["ocr_assessment"] = {
                        "needs_ocr": needs_ocr,
                        "confidence": confidence,
                        "reason": reason,
                        "text_coverage_ratio": text_ratio,
                    }

        # Update result
        processing_time = time.time() - start_time
        result["success"] = True
        result["processing_time"] = processing_time

        logger.info(f"PDF structure analysis completed in {processing_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error in analyze_pdf_structure: {str(e)}")
        logger.error(traceback.format_exc())
        raise ToolError(f"Failed to analyze PDF structure: {str(e)}") from e


@with_tool_metrics
@with_retry(max_retries=2, retry_delay=1.0)
@with_error_handling
async def batch_process_documents(
    folder_path: str,
    file_pattern: str = "*.pdf",
    output_folder: Optional[str] = None,
    extraction_method: str = "hybrid",
    max_pages_per_file: int = 0,
    reformat_as_markdown: bool = True,
    suppress_headers: bool = True,
    max_concurrency: int = 3,
    skip_on_error: bool = True,
    bytes_data: Optional[Dict[str, Union[bytes, str]]] = None,
) -> Dict[str, Any]:
    """
    Processes multiple document files in a folder with OCR and LLM enhancement.

    This tool handles batch processing of documents (PDFs and images) in a folder,
    extracting text, correcting OCR errors, and saving the results to an output folder.
    It can also process documents provided as bytes data.

    Args:
        folder_path: Path to the folder containing files to process
        file_pattern: Pattern to match files (default: "*.pdf", can be "*.jpg", "*.png", etc.)
        output_folder: Path to save the output files (default: create 'processed' subfolder)
        extraction_method: Method for PDF text extraction ("direct", "ocr", "hybrid")
        max_pages_per_file: Maximum pages to process per PDF (0 = all pages)
        reformat_as_markdown: Whether to format the output as markdown (default: True)
        suppress_headers: Whether to remove headers and footers (default: True)
        max_concurrency: Maximum number of files to process in parallel (default: 3)
        skip_on_error: Whether to continue processing other files if one fails (default: True)
        bytes_data: Optional dictionary of filename to bytes data for processing data directly

    Returns:
        A dictionary containing:
        {
            "success": true,
            "processed_files": [
                {
                    "file": "/path/to/document1.pdf",
                    "output_file": "/path/to/output/document1.md",
                    "pages_processed": 5,
                    "extraction_method": "hybrid",
                    "processing_time": 12.34,
                    "quality_score": 85  # if quality assessment is performed
                },
                {
                    "file": "/path/to/document2.pdf",
                    "error": "Error message",  # if processing failed
                    "status": "failed"
                }
            ],
            "total_files": 5,
            "successful_files": 4,
            "failed_files": 1,
            "output_folder": "/path/to/output",
            "total_processing_time": 45.67  # Seconds
        }

    Raises:
        ToolInputError: If the folder path is invalid
        ToolError: If batch processing fails
    """
    start_time = time.time()

    # Validate input if processing files from a folder
    all_files = []

    if not bytes_data:
        # Standard file processing from a folder
        if not folder_path or not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise ToolInputError(f"Invalid folder path: {folder_path}")

        # Set output folder if not provided
        if not output_folder:
            output_folder = os.path.join(folder_path, "processed")

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Find files matching the pattern
        matching_files: List[Path] = sorted(list(Path(folder_path).glob(file_pattern)))

        if not matching_files:
            raise ToolInputError(f"No files found in {folder_path} matching pattern {file_pattern}")

        all_files = [(str(f), None) for f in matching_files]  # (path, bytes_data)
    else:
        # Processing from bytes data
        if not output_folder:
            # Create a temporary output folder if not specified
            output_folder = tempfile.mkdtemp(prefix="ocr_batch_")
        else:
            os.makedirs(output_folder, exist_ok=True)

        # Convert bytes_data to our format
        for filename, data in bytes_data.items():
            if isinstance(data, str) and data.startswith("data:"):
                # Handle base64 data URLs
                try:
                    mime_type, b64data = data.split(";base64,", 1)
                    file_bytes = base64.b64decode(b64data)
                    all_files.append((filename, file_bytes))
                except Exception as e:
                    logger.error(f"Error decoding base64 data for {filename}: {str(e)}")
                    if not skip_on_error:
                        raise ToolError(f"Failed to decode base64 data: {str(e)}") from e
            elif isinstance(data, bytes):
                # Already in bytes format
                all_files.append((filename, data))
            else:
                logger.error(f"Unsupported data format for {filename}")
                if not skip_on_error:
                    raise ToolInputError(f"Unsupported data format for {filename}")

    if not all_files:
        raise ToolInputError("No files to process")

    # Get task type for batch processing
    task_type = _get_task_type_for_ocr(extraction_method)
    logger.info(f"Batch processing documents with task type: {task_type}")

    # Initialize result
    result = {
        "success": False,
        "processed_files": [],
        "total_files": len(all_files),
        "successful_files": 0,
        "failed_files": 0,
        "output_folder": output_folder,
        "total_processing_time": 0,
        "task_type": task_type,
    }

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)

    # Create partially-applied functions for better reuse and readability
    # This allows us to pre-configure the processing functions with common parameters
    extract_pdf_with_config = functools.partial(
        extract_text_from_pdf,
        extraction_method=extraction_method,
        max_pages=max_pages_per_file,
        skip_pages=0,
        reformat_as_markdown=reformat_as_markdown,
        suppress_headers=suppress_headers,
        assess_quality=True,
    )

    extract_pdf_bytes_with_config = functools.partial(
        extract_text_from_pdf_bytes,
        extraction_method=extraction_method,
        max_pages=max_pages_per_file,
        skip_pages=0,
        reformat_as_markdown=reformat_as_markdown,
        suppress_headers=suppress_headers,
        assess_quality=True,
    )

    process_image_with_config = functools.partial(
        process_image_ocr, reformat_as_markdown=reformat_as_markdown, assess_quality=True
    )

    # Define worker function for processing each file
    async def process_file(file_info: Tuple[str, Optional[bytes]]) -> Dict[str, Any]:
        file_path, file_bytes = file_info
        async with semaphore:
            logger.info(f"Processing file: {file_path}")
            file_start_time = time.time()

            try:
                # Determine file type based on extension
                is_pdf = file_path.lower().endswith(".pdf")

                # Process according to file type
                if is_pdf:
                    # Extract base name
                    base_name = os.path.splitext(os.path.basename(file_path))[0]

                    # Determine output file extension
                    output_extension = ".md" if reformat_as_markdown else ".txt"

                    # Define output file path
                    output_file = os.path.join(output_folder, f"{base_name}{output_extension}")

                    # Extract text based on whether we have bytes or file path
                    if file_bytes is not None:
                        # Process PDF from bytes
                        extraction_result = await extract_pdf_bytes_with_config(
                            pdf_bytes=file_bytes
                        )
                    else:
                        # Process PDF from file path
                        extraction_result = await extract_pdf_with_config(file_path=file_path)

                    # Save the enhanced text
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(extraction_result["text"])

                    # Save the raw text for reference
                    raw_output_file = os.path.join(output_folder, f"{base_name}_raw.txt")
                    with open(raw_output_file, "w", encoding="utf-8") as f:
                        f.write(extraction_result["raw_text"])

                    # Create file result
                    file_processing_time = time.time() - file_start_time
                    file_result = {
                        "file": file_path,
                        "output_file": output_file,
                        "raw_output_file": raw_output_file,
                        "pages_processed": extraction_result["pages_processed"],
                        "extraction_method_used": extraction_result["extraction_method_used"],
                        "processing_time": file_processing_time,
                        "status": "success",
                    }

                    # Add quality metrics if available
                    if "quality_metrics" in extraction_result:
                        quality_metrics = extraction_result["quality_metrics"]
                        file_result["quality_score"] = quality_metrics.get("score")

                    logger.info(f"Successfully processed PDF: {file_path}")

                else:
                    # Handle image file
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_extension = ".md" if reformat_as_markdown else ".txt"
                    output_file = os.path.join(output_folder, f"{base_name}{output_extension}")

                    # Process image with OCR based on whether we have bytes or file path
                    if file_bytes is not None:
                        # Process image from bytes
                        ocr_result = await process_image_with_config(
                            image_data=base64.b64encode(file_bytes).decode("utf-8")
                        )
                    else:
                        # Process image from file path
                        ocr_result = await process_image_with_config(image_path=file_path)

                    # Save the enhanced text
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(ocr_result["text"])

                    # Save the raw text for reference
                    raw_output_file = os.path.join(output_folder, f"{base_name}_raw.txt")
                    with open(raw_output_file, "w", encoding="utf-8") as f:
                        f.write(ocr_result["raw_text"])

                    # Create file result
                    file_processing_time = time.time() - file_start_time
                    file_result = {
                        "file": file_path,
                        "output_file": output_file,
                        "raw_output_file": raw_output_file,
                        "table_detected": ocr_result.get("table_detected", False),
                        "processing_time": file_processing_time,
                        "status": "success",
                    }

                    # Add quality metrics if available
                    if "quality_metrics" in ocr_result:
                        quality_metrics = ocr_result["quality_metrics"]
                        file_result["quality_score"] = quality_metrics.get("score")

                    logger.info(f"Successfully processed image: {file_path}")

                return file_result
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                return {"file": file_path, "error": str(e), "status": "failed"}

    try:
        # Process files in parallel
        tasks = [process_file(file_info) for file_info in all_files]
        processed_results = await asyncio.gather(*tasks)

        # Update result
        result["processed_files"] = processed_results
        result["successful_files"] = sum(
            1 for r in processed_results if r.get("status") == "success"
        )
        result["failed_files"] = sum(1 for r in processed_results if r.get("status") == "failed")
        result["success"] = True

        # Calculate total processing time
        total_processing_time = time.time() - start_time
        result["total_processing_time"] = total_processing_time

        logger.info(
            f"Batch processing completed: {result['successful_files']} successful, {result['failed_files']} failed"
        )
        return result

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise ToolError(f"Failed to batch process documents: {str(e)}") from e


# --- Additional helper functions ---


def _process_toc(toc: List) -> List[Dict[str, Any]]:
    """
    Processes a PDF table of contents into a nested structure.

    Args:
        toc: Table of contents from PyMuPDF

    Returns:
        Nested outline structure
    """
    if not toc:
        return []

    # Convert flat list with indentation levels to nested structure
    result = []
    stack = [(-1, result)]  # (level, children_list)

    for item in toc:
        level, title, page = item

        # Find parent in stack
        while stack[-1][0] >= level:
            stack.pop()

        # Create new entry
        entry = {"title": title, "page": page, "children": []}
        stack[-1][1].append(entry)

        # Add to stack
        stack.append((level, entry["children"]))

    return result


async def _format_tables_in_text(text: str) -> str:
    """
    Detects and formats potential tables in text using markdown.

    Args:
        text: Text to process

    Returns:
        Text with tables formatted in markdown
    """
    # Simple pattern to detect table-like content
    table_patterns = [
        # Multiple lines with similar column separator patterns
        r"(\n|^)(((\s*\S+\s*\|\s*\S+\s*)+\|?(\s*\n)){2,})",
        # Multiple lines with similar tab/space alignment
        r"(\n|^)((\s*\S+\s+\S+\s+\S+\s+\S+\s*\n){3,})",
    ]

    table_sections: List[Tuple[int, int, str]] = []
    for pattern in table_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            table_sections.append((match.start(), match.end(), match.group(2)))

    # Sort by start position
    table_sections.sort(key=lambda x: x[0])

    # No tables found
    if not table_sections:
        return text

    # Process each potential table
    result_parts = []
    last_end = 0

    for start, end, table_text in table_sections:
        # Add text before table
        if start > last_end:
            result_parts.append(text[last_end:start])

        # Process table
        try:
            formatted_table = await _enhance_table_formatting(table_text)
            result_parts.append(formatted_table)
        except Exception as e:
            logger.warning(f"Error formatting table: {str(e)}")
            result_parts.append(table_text)

        last_end = end

    # Add remaining text
    if last_end < len(text):
        result_parts.append(text[last_end:])

    return "".join(result_parts)


async def _enhance_table_formatting(table_text):
    """
    Enhances table formatting using LLM.

    Args:
        table_text: Potential table text

    Returns:
        Formatted table in markdown
    """
    prompt = f"""Format the following text as a markdown table. The text appears to contain tabular data but may not be properly formatted.

1. Detect column headers and content
2. Create a proper markdown table with headers, separator row, and content rows
3. Preserve all information but improve readability
4. If the input is not actually tabular data, return it unchanged with a comment indicating it's not a table

Here is the text to format:

```
{table_text}
```

Provide ONLY the formatted markdown table with no explanations or comments.
"""

    try:
        result = await generate_completion(
            prompt=prompt,
            provider=Provider.ANTHROPIC.value,
            temperature=0.2,
            max_tokens=len(table_text) + 500,
        )

        if not result or not result.get("text"):
            return table_text

        formatted_table = result["text"]

        # Check if it's actually formatted as a markdown table
        if "|" in formatted_table and "-|-" in formatted_table:
            return "\n" + formatted_table + "\n"
        else:
            return table_text
    except Exception as e:
        logger.warning(f"Error enhancing table format: {str(e)}")
        return table_text


async def _assess_text_quality(original_text: str, enhanced_text: str) -> Dict[str, Any]:
    """
    Assesses the quality of OCR enhancement using LLM.

    Args:
        original_text: Original OCR text
        enhanced_text: LLM-enhanced text

    Returns:
        Dictionary with quality assessment
    """
    # Truncate texts to reasonable lengths for assessment
    max_sample = 5000
    original_sample = original_text[:max_sample]
    enhanced_sample = enhanced_text[:max_sample]

    prompt = f"""Assess the quality improvement between the original OCR text and the enhanced version. Consider:

1. Error correction (typos, OCR artifacts, broken words)
2. Formatting improvements (paragraph structure, headings, lists)
3. Readability enhancement
4. Preservation of original content and meaning
5. Removal of unnecessary elements (headers, footers, artifacts)

Original OCR text:
```
{original_sample}
```

Enhanced text:
```
{enhanced_sample}
```

Provide:
1. A quality score from 0-100 where 100 is perfect enhancement
2. A brief explanation of improvements and any issues
3. Specific examples of corrections (max 3 examples)

Format your response as follows:
SCORE: [score]
EXPLANATION: [explanation]
EXAMPLES:
- [example 1]
- [example 2]
- [example 3]
"""

    try:
        result = await generate_completion(
            prompt=prompt, provider=Provider.ANTHROPIC.value, temperature=0.3, max_tokens=1000
        )

        if not result or not result.get("text"):
            return {"score": None, "explanation": "Failed to assess quality"}

        assessment_text = result["text"]

        # Parse the assessment
        score_match = re.search(r"SCORE:\s*(\d+)", assessment_text)
        explanation_match = re.search(
            r"EXPLANATION:\s*(.*?)(?:\n\s*EXAMPLES|\Z)", assessment_text, re.DOTALL
        )
        examples_match = re.search(r"EXAMPLES:\s*(.*?)(?:\Z)", assessment_text, re.DOTALL)

        score = int(score_match.group(1)) if score_match else None
        explanation = (
            explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        )

        examples = []
        if examples_match:
            examples_text = examples_match.group(1)
            examples = [ex.strip().lstrip("- ") for ex in examples_text.split("\n") if ex.strip()]

        return {"score": score, "explanation": explanation, "examples": examples}
    except Exception as e:
        logger.warning(f"Error assessing text quality: {str(e)}")
        return {"score": None, "explanation": f"Failed to assess quality: {str(e)}"}
