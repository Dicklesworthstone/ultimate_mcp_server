"""Utility functions for the knowledge base services."""

from typing import Any, Dict, List, Optional


def build_metadata_filter(
    filters: Optional[Dict[str, Any]] = None, operator: str = "$and"
) -> Optional[Dict[str, Any]]:
    """Build a ChromaDB metadata filter.

    Args:
        filters: Dictionary of metadata filters (field->value or field->{op: value})
        operator: Logical operator to combine filters ($and or $or)

    Returns:
        ChromaDB-compatible filter or None
    """
    if not filters:
        return None

    # Handle direct equality case with single filter
    if len(filters) == 1 and not any(isinstance(v, dict) for v in filters.values()):
        field, value = next(iter(filters.items()))
        return {field: value}  # ChromaDB handles direct equality

    # Process complex filters
    filter_conditions = []

    for field, condition in filters.items():
        if isinstance(condition, dict):
            # Already has operators
            if any(k.startswith("$") for k in condition.keys()):
                filter_conditions.append({field: condition})
            else:
                # Convert to $eq
                filter_conditions.append({field: {"$eq": condition}})
        else:
            # Simple equality
            filter_conditions.append({field: {"$eq": condition}})

    # If only one condition, no need for logical operator
    if len(filter_conditions) == 1:
        return filter_conditions[0]

    # Combine with logical operator
    return {operator: filter_conditions}


def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """Extract important keywords from text.

    Args:
        text: Input text
        min_length: Minimum length of keywords
        max_keywords: Maximum number of keywords to extract

    Returns:
        List of keywords
    """
    # Simple keyword extraction (could be improved with NLP)
    words = text.lower().split()

    # Filter out short words and common stop words
    stop_words = {
        "the",
        "and",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "of",
        "that",
    }

    keywords = [
        word.strip(".,?!\"'()[]{}:;")
        for word in words
        if len(word) >= min_length and word.lower() not in stop_words
    ]

    # Count occurrences
    keyword_counts = {}
    for word in keywords:
        if word in keyword_counts:
            keyword_counts[word] += 1
        else:
            keyword_counts[word] = 1

    # Sort by frequency
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

    # Return top keywords
    return [k for k, _ in sorted_keywords[:max_keywords]]


def generate_token_estimate(text: str) -> int:
    """Generate a rough estimate of token count.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Rough estimate based on whitespace tokenization and a multiplier
    # This is a very crude approximation
    words = len(text.split())

    # Adjust for non-English or technical content
    if any(ord(c) > 127 for c in text):  # Has non-ASCII chars
        return int(words * 1.5)  # Non-English texts need more tokens

    # Standard English approximation
    return int(words * 1.3)  # Account for tokenization differences


def create_document_metadata(
    document: str, source: Optional[str] = None, document_type: Optional[str] = None
) -> Dict[str, Any]:
    """Create metadata for a document.

    Args:
        document: Document text
        source: Optional source of the document
        document_type: Optional document type

    Returns:
        Document metadata
    """
    # Basic metadata
    metadata = {
        "length": len(document),
        "token_estimate": generate_token_estimate(document),
        "created_at": int(1000 * import_time()),
    }

    # Add source if provided
    if source:
        metadata["source"] = source

    # Add document type if provided
    if document_type:
        metadata["type"] = document_type

    # Extract potential title from first line
    lines = document.strip().split("\n")
    if lines and len(lines[0]) < 100:  # Potential title
        metadata["potential_title"] = lines[0]

    return metadata


# Import at the end to avoid circular imports
import time as import_time  # noqa: E402
