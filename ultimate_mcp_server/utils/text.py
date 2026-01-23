"""Text processing utilities for Ultimate MCP Server."""

import re
import string
from typing import Any, Dict, List, Optional

from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """Intelligently truncate text to a maximum length at natural boundaries.

    This function shortens text to fit within a specified maximum length while
    attempting to preserve semantic coherence by cutting at natural text boundaries
    like the end of sentences or paragraphs. This produces more readable truncated
    text compared to simple character-based truncation.

    The truncation algorithm:
    1. If text is already shorter than max_length, return it unchanged
    2. Otherwise, truncate at max_length character position
    3. Look backwards for a natural boundary (., ?, !, or paragraph break)
    4. If a good boundary is found beyond 80% of max_length, truncate there
    5. Optionally add an ellipsis to indicate truncation has occurred

    This approach is useful for:
    - Creating text previews or snippets
    - Fitting text into UI components with size constraints
    - Preparing content for displays with character limits
    - Generating summaries while maintaining readability

    Args:
        text: Text to truncate. Can be any length, including empty.
        max_length: Maximum character length of the returned text (not including ellipsis)
        add_ellipsis: Whether to append "..." to truncated text (default: True)

    Returns:
        Truncated text, optionally with ellipsis. If the input text is shorter than
        max_length, it's returned unchanged. If the input is None or empty, it's returned as is.

    Examples:
        >>> # Basic truncation
        >>> truncate_text("This is a long sentence that needs truncation.", 20)
        'This is a long...'

        >>> # Truncation at sentence boundary
        >>> truncate_text("Short sentence. Another sentence. Yet another one.", 20)
        'Short sentence...'

        >>> # Without ellipsis
        >>> truncate_text("A very long text to truncate.", 10, add_ellipsis=False)
        'A very lon'

        >>> # No truncation needed
        >>> truncate_text("Short text", 20)
        'Short text'
    """
    if not text or len(text) <= max_length:
        return text

    # Try to truncate at sentence boundary
    truncated = text[:max_length]

    # Find the last sentence boundary in the truncated text
    last_boundary = max(
        truncated.rfind(". "), truncated.rfind("? "), truncated.rfind("! "), truncated.rfind("\n\n")
    )

    if last_boundary > max_length * 0.8:  # Only truncate at boundary if it's not too short
        truncated = truncated[: last_boundary + 1]

    # Add ellipsis if requested and text was truncated
    if add_ellipsis and len(text) > len(truncated):
        truncated = truncated.rstrip() + "..."

    return truncated


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate the number of tokens in text for LLM processing.

    This function calculates how many tokens would be consumed when sending text
    to a language model. It uses model-specific tokenizers when possible (via tiktoken)
    for accurate counts, or falls back to a character-based heuristic estimation when
    tokenizers aren't available.

    Token count is important for:
    - Estimating LLM API costs (which are typically billed per token)
    - Ensuring text fits within model context windows
    - Optimizing content to minimize token usage
    - Debugging token-related issues in model interactions

    The function selects the appropriate tokenizer based on the model parameter:
    - For GPT-4o models: Uses the "gpt-4o" tokenizer
    - For Claude models: Uses "cl100k_base" tokenizer as an approximation
    - For other models or when model is not specified: Uses "cl100k_base" (works well for most recent models)

    If the tiktoken library isn't available, the function falls back to character-based
    estimation, which applies heuristics based on character types to approximate token count.

    Args:
        text: Text string to count tokens for. Can be any length, including empty.
        model: Optional model name to select the appropriate tokenizer. Common values include
               "gpt-4o", "gpt-4", "claude-3-5-haiku-20241022", "claude-3-sonnet", etc.

    Returns:
        Estimated number of tokens in the text. Returns 0 for empty input.

    Examples:
        >>> count_tokens("Hello, world!")  # Using default tokenizer
        3

        >>> count_tokens("GPT-4 is a multimodal model.", model="gpt-4o")
        7

        >>> # Using fallback estimation if tiktoken is not available
        >>> # (actual result may vary based on the estimation algorithm)
        >>> count_tokens("This is a fallback example")
        6

    Dependencies:
        - Requires the "tiktoken" library for accurate counting
        - Falls back to character-based estimation if tiktoken is not available

    Note:
        The character-based fallback estimation is approximate and may differ
        from actual tokenization, especially for non-English text, code, or
        text with many special characters or numbers.
    """
    if not text:
        return 0

    # Try to use tiktoken if available (accurate for OpenAI models)
    try:
        import tiktoken

        # Select encoding based on model
        if model and model.startswith("gpt-4o"):
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif model and "claude" in model.lower():
            # For Claude, use cl100k_base as approximation
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Default to cl100k_base (used by most recent models)
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    except ImportError:
        # Fallback to character-based estimation if tiktoken is not available
        return _estimate_tokens_by_chars(text)


def _estimate_tokens_by_chars(text: str) -> int:
    """Estimate token count using character-based heuristics when tokenizers aren't available.

    This internal fallback function provides a rough approximation of token count based on
    character analysis when the preferred tokenizer-based method (via tiktoken) is not available.
    It applies various heuristics based on observed tokenization patterns across common models.

    The estimation algorithm works as follows:
    1. Use a base ratio of 4.0 characters per token (average for English text)
    2. Count the total number of characters in the text
    3. Apply adjustments based on character types:
       - Whitespace: Count separately and add with reduced weight (0.5)
         since whitespace is often combined with other characters in tokens
       - Digits: Count separately and subtract weight (0.5)
         since numbers are often encoded more efficiently
    4. Calculate the final token estimate based on adjusted character count

    While not as accurate as model-specific tokenizers, this approach provides a reasonable
    approximation that works across different languages and text types. The approximation
    tends to be more accurate for:
    - Plain English text with standard punctuation
    - Text with a typical mix of words and whitespace
    - Content with a moderate number of special characters

    The estimation may be less accurate for:
    - Text with many numbers or special characters
    - Code snippets or markup languages
    - Non-Latin script languages
    - Very short texts (under 10 characters)

    Args:
        text: Text string to estimate token count for

    Returns:
        Estimated number of tokens (always at least 1 for non-empty text)

    Note:
        This function is intended for internal use by count_tokens() as a fallback when
        tiktoken is not available. It always returns at least 1 token for any non-empty text.
    """
    # Character-based estimation (rough approximation)
    avg_chars_per_token = 4.0

    # Count characters
    char_count = len(text)

    # Account for whitespace more efficiently representing tokens
    whitespace_count = sum(1 for c in text if c.isspace())

    # Count numbers (numbers are often encoded efficiently)
    digit_count = sum(1 for c in text if c.isdigit())

    # Adjust total count based on character types
    adjusted_count = char_count + (whitespace_count * 0.5) - (digit_count * 0.5)

    # Estimate tokens
    return max(1, int(adjusted_count / avg_chars_per_token))


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    remove_whitespace: bool = False,
    remove_urls: bool = False,
    remove_numbers: bool = False,
) -> str:
    """Normalize text with configurable cleaning options for text processing.

    This function standardizes text by applying various normalization procedures
    based on the specified parameters. It's useful for preparing text for natural
    language processing tasks, text comparison, search operations, and other scenarios
    where consistent formatting is important.

    The function applies normalizations in a specific order to ensure consistent results:
    1. Lowercase conversion (if enabled)
    2. URL removal (if enabled)
    3. Number removal (if enabled)
    4. Punctuation removal (if enabled)
    5. Whitespace normalization (if enabled)

    Each normalization step is optional and controlled by a separate parameter,
    allowing precise control over the transformations applied.

    Args:
        text: The input text to normalize. Can be any length, including empty.
        lowercase: Whether to convert text to lowercase (default: True)
        remove_punctuation: Whether to remove all punctuation marks (default: False)
        remove_whitespace: Whether to replace all whitespace sequences (spaces, tabs,
                          newlines) with a single space and trim leading/trailing
                          whitespace (default: False)
        remove_urls: Whether to remove web URLs (http, https, www) (default: False)
        remove_numbers: Whether to remove all numeric digits (default: False)

    Returns:
        Normalized text with requested transformations applied. Empty input
        text is returned unchanged.

    Examples:
        >>> # Default behavior (only lowercase)
        >>> normalize_text("Hello World! Visit https://example.com")
        'hello world! visit https://example.com'

        >>> # Multiple normalizations
        >>> normalize_text("Hello, World! 123",
        ...                lowercase=True,
        ...                remove_punctuation=True,
        ...                remove_numbers=True)
        'hello world'

        >>> # URL and whitespace normalization
        >>> normalize_text("Check   https://example.com   for more info!",
        ...                remove_urls=True,
        ...                remove_whitespace=True)
        'Check for more info!'

    Notes:
        - Removing punctuation eliminates all symbols in string.punctuation
        - URL removal uses a regex pattern that matches common URL formats
        - When remove_whitespace is True, all sequences of whitespace are collapsed
          to a single space and leading/trailing whitespace is removed
    """
    if not text:
        return text

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove URLs
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove numbers
    if remove_numbers:
        text = re.sub(r"\d+", "", text)

    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    # Normalize whitespace
    if remove_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_key_phrases(text: str, max_phrases: int = 5, min_word_length: int = 3) -> List[str]:
    """Extract key phrases from text using statistical methods.

    This function identifies significant phrases from the input text using a frequency-based
    approach. It works by normalizing the text, splitting it into sentences, extracting
    candidate noun phrases through regex pattern matching, and then ranking them by frequency.
    The approach is language-agnostic and works well for medium to large text passages.

    The extraction process follows these steps:
    1. Normalize the input text (lowercase, preserve punctuation, remove URLs)
    2. Split the text into individual sentences
    3. Within each sentence, identify potential noun phrases using regex patterns
       that match 1-3 word sequences where at least one word meets the minimum length
    4. Count phrase frequency across the entire text
    5. Sort phrases by frequency (most frequent first)
    6. Return the top N phrases based on the max_phrases parameter

    Args:
        text: Source text from which to extract key phrases
        max_phrases: Maximum number of phrases to return, default is 5
        min_word_length: Minimum length (in characters) for a word to be considered
                        in phrase extraction, default is 3

    Returns:
        List of key phrases sorted by frequency (most frequent first).
        Returns an empty list if input text is empty or no phrases are found.

    Examples:
        >>> extract_key_phrases("The quick brown fox jumps over the lazy dog. The dog was too lazy to react.")
        ['the dog', 'lazy', 'quick brown fox']

        >>> extract_key_phrases("Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.", max_phrases=3)
        ['machine learning', 'field of study', 'computers']

    Notes:
        - The function works best on longer text passages with repeated key concepts
        - The approach prioritizes frequency over linguistic sophistication, so it's
          more effective for factual text than creative writing
        - For better results on short texts, consider decreasing min_word_length
        List of key phrases
    """
    if not text:
        return []

    # Normalize text
    normalized = normalize_text(
        text,
        lowercase=True,
        remove_punctuation=False,
        remove_whitespace=True,
        remove_urls=True,
    )

    # Split into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", normalized)

    # Extract phrases (simple noun phrases)
    phrases = []
    for sentence in sentences:
        # Find potential noun phrases
        np_matches = re.finditer(
            r"\b((?:(?:[A-Za-z]+\s+){0,2}[A-Za-z]{%d,})|(?:[A-Za-z]{%d,}))\b"
            % (min_word_length, min_word_length),
            sentence,
        )
        for match in np_matches:
            phrase = match.group(0).strip()
            if len(phrase.split()) <= 3:  # Limit to 3-word phrases
                phrases.append(phrase)

    # Count phrase frequency
    phrase_counts = {}
    for phrase in phrases:
        if phrase in phrase_counts:
            phrase_counts[phrase] += 1
        else:
            phrase_counts[phrase] = 1

    # Sort by frequency
    sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)

    # Return top phrases
    return [phrase for phrase, _ in sorted_phrases[:max_phrases]]


def split_text_by_similarity(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks of similar size at natural boundaries.

    This function divides a long text into smaller, semantically meaningful chunks while
    respecting natural text boundaries. It intelligently searches for boundaries like
    paragraph breaks, sentence endings, commas, or whitespace to ensure chunks don't break
    in the middle of a sentence or idea, which is important for text processing tasks
    like semantic analysis, embedding generation, or LLM processing.

    The chunking algorithm works as follows:
    1. If the text is shorter than the chunk_size, return it as a single chunk
    2. Otherwise, iteratively:
       a. Determine the target end position (start + chunk_size)
       b. Search for natural boundaries near the target end, prioritizing:
          - Paragraph breaks (\n\n)
          - Sentence boundaries (. followed by space and capital letter)
          - Commas
          - Any whitespace
       c. Split at the best available boundary
       d. Move the start position for the next chunk, including overlap
       e. Repeat until the entire text is processed

    Args:
        text: The text to split into chunks
        chunk_size: Target size of each chunk in characters (default: 1000)
        overlap: Number of characters to overlap between chunks (default: 100)
                This helps maintain context between chunks for tasks like semantic search

    Returns:
        List of text chunks. If the input text is empty or less than chunk_size,
        returns a list containing only the original text.

    Examples:
        >>> text = "Paragraph one.\\n\\nParagraph two. This is a sentence. And another.\\n\\nParagraph three."
        >>> chunks = split_text_by_similarity(text, chunk_size=30, overlap=5)
        >>> chunks
        ['Paragraph one.', 'Paragraph two. This is a sentence.', ' This is a sentence. And another.', 'Paragraph three.']

    Notes:
        - The function prioritizes finding natural boundaries over strictly adhering to chunk_size
        - With small chunk_size values and complex texts, some chunks may exceed chunk_size
          if no suitable boundary is found within a reasonable range
        - The overlap parameter helps maintain context between chunks, which is important
          for tasks like semantic search or text analysis
    """
    if not text or len(text) <= chunk_size:
        return [text]

    # Define boundary patterns in order of preference
    boundaries = [
        r"\n\s*\n",  # Double newline (paragraph)
        r"\.\s+[A-Z]",  # End of sentence
        r",\s+",  # Comma with space
        r"\s+",  # Any whitespace
    ]

    chunks = []
    start = 0

    while start < len(text):
        # Determine end position for this chunk
        end = min(start + chunk_size, len(text))

        # If we're not at the end of the text, find a good boundary
        if end < len(text):
            # Try each boundary pattern in order
            for pattern in boundaries:
                # Search for the boundary pattern before the end position
                search_area = text[max(start, end - chunk_size // 4) : end]
                matches = list(re.finditer(pattern, search_area))

                if matches:
                    # Found a good boundary, adjust end position
                    match_end = matches[-1].end()
                    end = max(start, end - chunk_size // 4) + match_end
                    break

        # Extract the chunk
        chunk = text[start:end]
        chunks.append(chunk)

        # Move to the next chunk with overlap
        start = end - overlap

    return chunks


def sanitize_text(text: str, allowed_tags: Optional[List[str]] = None) -> str:
    """Sanitize text by removing potentially harmful elements.

    This function cleans input text by removing potentially dangerous HTML/XML content
    that could lead to XSS (Cross-Site Scripting) or other injection attacks. It strips
    out script tags, style tags, HTML comments, and by default removes all HTML markup
    unless specific tags are explicitly allowed via the allowed_tags parameter.

    The sanitization process follows these steps:
    1. Remove all <script> tags and their contents (highest security priority)
    2. Remove all <style> tags and their contents (to prevent CSS-based attacks)
    3. Remove all HTML comments (which might contain sensitive information)
    4. Process HTML tags based on the allowed_tags parameter:
       - If allowed_tags is None: remove ALL HTML tags
       - If allowed_tags is provided: keep only those specific tags, remove all others
    5. Convert HTML entities like &amp;, &lt;, etc. to their character equivalents

    Args:
        text: The text to sanitize, potentially containing unsafe HTML/XML content
        allowed_tags: Optional list of HTML tags to preserve (e.g., ["p", "br", "strong"]).
                     If None (default), all HTML tags will be removed.

    Returns:
        Sanitized text with dangerous elements removed and HTML entities decoded.
        The original string is returned if the input is empty.

    Examples:
        >>> sanitize_text("<p>Hello <script>alert('XSS')</script> World</p>")
        'Hello  World'

        >>> sanitize_text("<p>Hello <b>Bold</b> World</p>", allowed_tags=["b"])
        'Hello <b>Bold</b> World'

        >>> sanitize_text("Safe &amp; sound")
        'Safe & sound'

    Note:
        While this function provides basic sanitization, it is not a complete defense
        against all possible injection attacks. For highly sensitive applications,
        consider using specialized HTML sanitization libraries like bleach or html-sanitizer.
    """
    if not text:
        return text

    # Remove script tags and content
    text = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", text)

    # Remove style tags and content
    text = re.sub(r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", "", text)

    # Remove comments
    text = re.sub(r"<!--.*?-->", "", text)

    # Handle HTML tags based on allowed_tags
    if allowed_tags:
        # Allow specified tags but remove all others
        allowed_pattern = "|".join(allowed_tags)  # noqa: F841

        # Function to process tag matches
        def tag_replacer(match):
            tag = match.group(1).lower()
            if tag in allowed_tags:
                return match.group(0)
            else:
                return ""

        # Replace tags not in allowed_tags
        text = re.sub(r"<(\w+)(?:\s[^>]*)?(?:\/?>|>.*?<\/\1>)", tag_replacer, text)
    else:
        # Remove all HTML tags
        text = re.sub(r"<[^>]*>", "", text)

    # Convert HTML entities
    text = _convert_html_entities(text)

    return text


def _convert_html_entities(text: str) -> str:
    """Convert common HTML entities to their corresponding characters.

    This internal utility function translates HTML entity references (both named and numeric)
    into their equivalent Unicode characters. It handles common named entities like &amp;,
    &lt;, &gt;, as well as decimal (&#123;) and hexadecimal (&#x7B;) numeric entity references.

    The conversion process:
    1. Replace common named entities with their character equivalents using a lookup table
    2. Convert decimal numeric entities (&#nnn;) to characters using int() and chr()
    3. Convert hexadecimal numeric entities (&#xhh;) to characters using int(hex, 16) and chr()

    This function is primarily used internally by sanitize_text() to ensure that entity-encoded
    content is properly decoded after HTML tag processing.

    Args:
        text: String containing HTML entities to convert

    Returns:
        String with HTML entities replaced by their corresponding Unicode characters.
        If the input is empty or contains no entities, the original string is returned.

    Examples:
        >>> _convert_html_entities("&lt;div&gt;")
        '<div>'

        >>> _convert_html_entities("Copyright &copy; 2023")
        'Copyright © 2023'

        >>> _convert_html_entities("&#65;&#66;&#67;")
        'ABC'

        >>> _convert_html_entities("&#x41;&#x42;&#x43;")
        'ABC'

    Limitations:
        - Only handles a subset of common named entities (amp, lt, gt, quot, apos, nbsp)
        - Entity references must be properly formed (e.g., &amp; not &amp )
        - Doesn't validate that numeric references point to valid Unicode code points
    """
    # Define common HTML entities
    entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&apos;": "'",
        "&nbsp;": " ",
    }

    # Replace each entity
    for entity, char in entities.items():
        text = text.replace(entity, char)

    # Handle numeric entities
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = re.sub(r"&#x([0-9a-f]+);", lambda m: chr(int(m.group(1), 16)), text)

    return text


def extract_structured_data(text: str, patterns: Dict[str, str]) -> Dict[str, Any]:
    """Extract structured key-value data from text using regex patterns.

    This function applies a set of regular expression patterns to extract specific
    information from unstructured text, converting it into a structured dictionary.
    It's useful for parsing semi-structured text like logs, emails, reports, or any
    content that follows consistent patterns.

    Each key in the patterns dictionary represents a field name in the output,
    while the corresponding value is a regex pattern used to extract that field's
    value from the input text. If the pattern contains capturing groups, the first
    group's match is used as the value; otherwise, the entire match is used.

    Features:
    - Multi-field extraction in a single pass
    - Case-insensitive matching by default
    - Support for multi-line patterns with DOTALL mode
    - Capturing group extraction for fine-grained control
    - Automatic whitespace trimming of extracted values

    Args:
        text: Source text to extract data from
        patterns: Dictionary mapping field names to regex patterns.
                 Example: {"email": r"Email:\\s*([^\\s@]+@[^\\s@]+\\.[^\\s@]+)",
                          "phone": r"Phone:\\s*(\\d{3}[-\\.\\s]??\\d{3}[-\\.\\s]??\\d{4})"}

    Returns:
        Dictionary with field names as keys and extracted values as strings.
        Only fields with successful matches are included in the result.
        Returns an empty dictionary if the input text is empty or no patterns match.

    Examples:
        >>> text = "Name: John Doe\\nEmail: john@example.com\\nAge: 30"
        >>> patterns = {
        ...     "name": r"Name:\\s*(.*?)(?:\\n|$)",
        ...     "email": r"Email:\\s*([^\\s@]+@[^\\s@]+\\.[^\\s@]+)",
        ...     "age": r"Age:\\s*(\\d+)"
        ... }
        >>> extract_structured_data(text, patterns)
        {'name': 'John Doe', 'email': 'john@example.com', 'age': '30'}

        >>> # Using a pattern without capturing groups
        >>> extract_structured_data("Status: Active", {"status": r"Status: \\w+"})
        {'status': 'Status: Active'}

        >>> # No matches
        >>> extract_structured_data("Empty content", {"field": r"NotFound"})
        {}

    Tips:
        - Use captured groups (parentheses) to extract specific parts of matches
        - Make patterns as specific as possible to avoid false positives
        - For complex multi-line extractions, use (?s) flag in your regex or rely on
          the built-in DOTALL mode this function applies
        - Remember to escape special regex characters when matching literals
    """
    if not text:
        return {}

    result = {}

    # Apply each pattern
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # If the pattern has groups, use the first group
            if match.groups():
                result[field] = match.group(1).strip()
            else:
                result[field] = match.group(0).strip()

    return result


def find_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using character n-grams and Jaccard similarity.

    This function measures the similarity between two text strings using character-level
    trigrams (3-character sequences) and the Jaccard similarity coefficient. This approach
    provides a language-agnostic way to detect similarity that works well for:

    - Fuzzy matching of text fragments
    - Detecting near-duplicate content
    - Finding related sentences or paragraphs
    - Language-independent text comparisons

    The algorithm works as follows:
    1. Normalize both texts (lowercase, remove excess whitespace)
    2. Generate character trigrams (sets of all 3-character sequences) for each text
    3. Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
       (size of intersection divided by size of union)

    This approach emphasizes shared character patterns rather than exact word matches,
    making it robust to minor spelling variations, word order changes, and formatting
    differences.

    Args:
        text1: First text string to compare
        text2: Second text string to compare

    Returns:
        Similarity score as a float between 0.0 (completely different) and
        1.0 (identical after normalization). Returns 0.0 if either input is empty.
        For very short texts (<3 chars), returns 1.0 if both are identical after
        normalization, otherwise 0.0.

    Examples:
        >>> find_text_similarity("hello world", "hello world")
        1.0

        >>> find_text_similarity("hello world", "world hello")
        0.6153846153846154  # High similarity despite word order change

        >>> find_text_similarity("color", "colour")
        0.5714285714285714  # Handles spelling variations

        >>> find_text_similarity("completely different", "nothing alike")
        0.0

    Performance note:
        The function builds complete sets of trigrams for both texts, which can
        consume significant memory for very large inputs. Consider chunking or
        sampling when processing large documents.
    """
    if not text1 or not text2:
        return 0.0

    # Normalize texts
    text1 = normalize_text(text1, lowercase=True, remove_whitespace=True)
    text2 = normalize_text(text2, lowercase=True, remove_whitespace=True)

    # Generate character trigrams
    def get_trigrams(s):
        return set(s[i : i + 3] for i in range(len(s) - 2))

    trigrams1 = get_trigrams(text1)
    trigrams2 = get_trigrams(text2)

    # Find common trigrams
    common = trigrams1.intersection(trigrams2)

    # Calculate Jaccard similarity
    if not trigrams1 and not trigrams2:
        return 1.0  # Both strings are too short for trigrams

    return len(common) / max(1, len(trigrams1.union(trigrams2)))


def get_text_stats(text: str) -> Dict[str, Any]:
    """Analyze text to compute various linguistic and structural statistics.

    This function calculates a comprehensive set of metrics that characterize the
    input text, including volume metrics (character/word counts), structural features
    (sentence/paragraph counts), and readability indicators (average word/sentence length).
    It also estimates the number of tokens that would be consumed by LLM processing.

    These statistics are useful for:
    - Assessing text complexity and readability
    - Estimating processing costs for LLM operations
    - Content analysis and comparison
    - Enforcing length constraints in applications
    - Debugging text processing pipelines

    The function uses regex-based analyses to identify linguistic boundaries
    (words, sentences, paragraphs) and delegates token estimation to the count_tokens
    function, which uses model-specific tokenizers when available.

    The metrics provided in the output dictionary:
    - char_count: Total number of characters in the text, including whitespace.
    - word_count: Total number of words, using word boundary regex (\\b\\w+\\b).
      This counts sequences of alphanumeric characters as words.
    - sentence_count: Number of sentences, detected by looking for periods,
      question marks, or exclamation points followed by spaces, with special
      handling for common abbreviations to reduce false positives.
    - paragraph_count: Number of paragraphs, determined by double newline
      sequences (\\n\\n) which typically indicate paragraph breaks.
    - avg_word_length: Average length of words in characters, rounded to
      one decimal place. Provides a simple readability indicator.
    - avg_sentence_length: Average number of words per sentence, rounded to
      one decimal place. Higher values typically indicate more complex text.
    - estimated_tokens: Estimated number of tokens for LLM processing using
      the count_tokens function, which uses model-specific tokenizers when
      available or falls back to character-based estimation.

    Args:
        text: The text to analyze. Can be any length, including empty text.

    Returns:
        A dictionary containing the linguistic and structural statistics described above.
        For empty input, returns a dictionary with all values set to 0.

    Examples:
        >>> stats = get_text_stats("Hello world. This is a sample text with two sentences.")
        >>> stats['word_count']
        10
        >>> stats['sentence_count']
        2
        >>> stats['avg_word_length']
        4.2

        >>> # Multiple paragraphs
        >>> text = "First paragraph with multiple sentences. Second sentence here.\\n\\n"
        >>> text += "Second paragraph. This has shorter sentences."
        >>> stats = get_text_stats(text)
        >>> stats['paragraph_count']
        2
        >>> stats['sentence_count']
        4

        >>> # Empty input
        >>> get_text_stats("")
        {'char_count': 0, 'word_count': 0, 'sentence_count': 0, 'paragraph_count': 0,
         'avg_word_length': 0, 'avg_sentence_length': 0, 'estimated_tokens': 0}
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "estimated_tokens": 0,
        }

    # Character count
    char_count = len(text)

    # Word count
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    # Sentence count
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Paragraph count
    paragraphs = re.split(r"\n\s*\n", text)
    paragraph_count = len([p for p in paragraphs if p.strip()])

    # Average word length
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)

    # Average sentence length (in words)
    avg_sentence_length = word_count / max(1, sentence_count)

    # Estimated tokens
    estimated_tokens = count_tokens(text)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_word_length": round(avg_word_length, 1),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "estimated_tokens": estimated_tokens,
    }


def preprocess_text(text: str) -> str:
    """Standardize and clean text for machine learning and NLP tasks.

    This function applies a series of transformations to normalize input text
    into a standardized format suitable for classification, embedding generation,
    semantic analysis, and other natural language processing tasks. It focuses on
    removing noise and irregularities that could interfere with ML/NLP model performance
    while preserving the semantic content of the text.

    Transformations applied:
    1. Whitespace normalization: Collapses multiple spaces, tabs, newlines into single spaces
    2. Control character removal: Strips non-printable ASCII control characters
    3. Punctuation normalization: Reduces excessive repeated punctuation (e.g., "!!!!!!" → "!!!")
    4. Length truncation: For extremely long texts, preserves beginning and end with a
       truncation marker in the middle to stay under token limits

    This preprocessing is particularly useful for:
    - Text classification tasks where consistent input format is important
    - Before vectorization or embedding generation
    - Preparing text for input to language models
    - Reducing noise in text analytics

    Args:
        text: The input text to preprocess. Can be any length, including empty.

    Returns:
        Preprocessed text with standardized formatting. The original text is returned
        if it's empty. For extremely long inputs (>100,000 chars), returns a truncated
        version preserving the beginning and end portions.

    Examples:
        >>> preprocess_text("Hello   world!!!\nHow are\t\tyou?")
        'Hello world!!! How are you?'

        >>> preprocess_text("Too much punctuation!!!!!!!!")
        'Too much punctuation!!!'

        >>> preprocess_text("")
        ''

    Note:
        This function preserves case, punctuation, and special characters (beyond control chars),
        as these may be semantically relevant for many NLP tasks. For more aggressive normalization,
        consider using the normalize_text() function with appropriate parameters.
    """
    if not text:
        return text

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove control characters
    text = re.sub(r"[\x00-\x09\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Remove excessive punctuation repetition
    text = re.sub(r"([.!?]){3,}", r"\1\1\1", text)

    # Truncate if extremely long (preserve beginning and end)
    max_chars = 100000  # Reasonable limit to prevent token explosion
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + " [...text truncated...] " + text[-half:]

    return text
