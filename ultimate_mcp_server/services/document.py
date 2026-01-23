"""Document processing service for chunking and analyzing text documents."""

import re
from typing import List

from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Service for intelligent text document processing, chunking, and preparation.

    The DocumentProcessor provides sophisticated document handling capabilities
    focused on breaking down long documents into meaningful, properly-sized chunks
    optimized for various downstream NLP tasks such as embedding generation,
    semantic search, and RAG (Retrieval Augmented Generation).

    Key Features:
    - Multiple chunking strategies optimized for different content types
    - Configurable chunk size and overlap parameters
    - Semantic-aware chunking that preserves context and meaning
    - Sentence boundary detection for natural text segmentation
    - Token-based chunking for precise size control
    - Singleton implementation for efficient resource usage

    Chunking Methods:
    1. Semantic Chunking: Preserves paragraph structure and semantic meaning,
       preventing splits that would break logical content boundaries. Best for
       maintaining context in well-structured documents.

    2. Sentence Chunking: Splits documents at sentence boundaries, ensuring
       no sentence is broken across chunks. Ideal for natural language text
       where sentence integrity is important.

    3. Token Chunking: Divides text based on approximate token counts without
       special consideration for semantic boundaries. Provides the most precise
       control over chunk size for token-limited systems.

    Each method implements configurable overlap between chunks to maintain
    context across chunk boundaries, ensuring information isn't lost when a
    concept spans multiple chunks.

    Usage Example:
    ```python
    processor = get_document_processor()

    # Chunk a document with default settings (token-based)
    chunks = await processor.chunk_document(
        document=long_text,
        chunk_size=1000,
        chunk_overlap=200
    )

    # Use semantic chunking for a well-structured document
    semantic_chunks = await processor.chunk_document(
        document=article_text,
        chunk_size=1500,
        chunk_overlap=150,
        method="semantic"
    )

    # Process chunks for embedding or RAG
    for chunk in chunks:
        # Process each chunk...
    ```

    Note:
        This service implements the singleton pattern, ensuring only one instance
        exists throughout the application. Always use the get_document_processor()
        function to obtain the shared instance rather than creating instances directly.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(DocumentProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the document processor."""
        # Only initialize once for singleton
        if getattr(self, "_initialized", False):
            return

        logger.info("Document processor initialized", extra={"emoji_key": "success"})
        self._initialized = True

    async def chunk_document(
        self, document: str, chunk_size: int = 1000, chunk_overlap: int = 200, method: str = "token"
    ) -> List[str]:
        """
        Split a document into optimally sized, potentially overlapping chunks.

        This method intelligently divides a document into smaller segments using
        one of several chunking strategies, balancing chunk size requirements with
        preserving semantic coherence. The chunking process is critical for preparing
        documents for embedding, retrieval, and other NLP operations that have
        input size limitations or benefit from focused context.

        Chunking Methods:
        - "token": (Default) Splits text based on approximate token count.
          Simple and precise for size control, but may break semantic units.
        - "sentence": Preserves sentence boundaries, ensuring no sentence is broken
          across chunks. Better for maintaining local context and readability.
        - "semantic": Most sophisticated approach that attempts to preserve paragraph
          structure and semantic coherence. Best for maintaining document meaning
          but may result in more size variation between chunks.

        The chunk_size parameter is approximate for all methods, as they prioritize
        maintaining semantic boundaries where appropriate. The actual size of returned
        chunks may vary, especially when using sentence or semantic methods.

        Chunk overlap creates a sliding window effect, where the end of one chunk
        overlaps with the beginning of the next. This helps maintain context across
        chunk boundaries and improves retrieval quality by ensuring concepts that
        span multiple chunks can still be found.

        Selecting Parameters:
        - For embedding models with strict token limits: Use "token" with chunk_size
          set safely below the model's limit
        - For maximizing context preservation: Use "semantic" with larger overlap
        - For balancing size precision and sentence integrity: Use "sentence"
        - Larger overlap (25-50% of chunk_size) improves retrieval quality but
          increases storage and processing requirements

        Args:
            document: Text content to be chunked
            chunk_size: Target size of each chunk in approximate tokens (default: 1000)
            chunk_overlap: Number of tokens to overlap between chunks (default: 200)
            method: Chunking strategy to use ("token", "sentence", or "semantic")

        Returns:
            List of text chunks derived from the original document

        Note:
            Returns an empty list if the input document is empty or None.
            The token estimation is approximate and based on whitespace splitting,
            not a true tokenizer, so actual token counts may differ when processed
            by specific models.
        """
        if not document:
            return []

        logger.debug(
            f"Chunking document using method '{method}' (size: {chunk_size}, overlap: {chunk_overlap})",
            extra={"emoji_key": "processing"},
        )

        if method == "semantic":
            return await self._chunk_semantic(document, chunk_size, chunk_overlap)
        elif method == "sentence":
            return await self._chunk_by_sentence(document, chunk_size, chunk_overlap)
        else:
            # Default to token-based chunking
            return await self._chunk_by_tokens(document, chunk_size, chunk_overlap)

    async def _chunk_by_tokens(
        self, document: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """
        Split document into chunks by approximate token count without preserving semantic structures.

        This is the most straightforward chunking method, dividing text based solely
        on approximate token counts without special consideration for sentence or
        paragraph boundaries. It provides the most predictable and precise control
        over chunk sizes at the cost of potentially breaking semantic units like
        sentences or paragraphs.

        Algorithm implementation:
        1. Approximates tokens by splitting text on whitespace (creating "words")
        2. Divides the document into chunks of specified token length
        3. Implements sliding window overlaps between consecutive chunks
        4. Handles edge cases like empty documents and final chunks

        The token approximation used is simple whitespace splitting, which provides
        a reasonable estimation for most Western languages and common tokenization
        schemes. While not as accurate as model-specific tokenizers, it offers a
        good balance between performance and approximation quality for general use.

        Chunk overlap is implemented by including tokens from the end of one chunk
        at the beginning of the next, creating a sliding window effect that helps
        maintain context across chunk boundaries.

        This method is ideal for:
        - Working with strict token limits in downstream models
        - Processing text where exact chunk sizes are more important than
          preserving semantic structures
        - High-volume processing where simplicity and performance are priorities
        - Text with unusual or inconsistent formatting where sentence/paragraph
          detection might fail

        Args:
            document: Text content to split by tokens
            chunk_size: Number of tokens (words) per chunk
            chunk_overlap: Number of tokens to overlap between chunks

        Returns:
            List of text chunks of approximately equal token counts

        Note:
            True token counts in NLP models may differ from this approximation,
            especially for models with subword tokenization. For applications
            requiring exact token counts, consider using the model's specific
            tokenizer for more accurate size estimates.
        """
        # Simple token estimation (split by whitespace)
        words = document.split()

        # No words, return empty list
        if not words:
            return []

        # Simple chunking
        chunks = []
        start = 0

        while start < len(words):
            # Calculate end position with potential overlap
            end = min(start + chunk_size, len(words))

            # Create chunk
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - chunk_overlap

            # Avoid getting stuck at the end
            if start >= len(words) - chunk_overlap:
                break

        logger.debug(
            f"Split document into {len(chunks)} chunks by token", extra={"emoji_key": "processing"}
        )

        return chunks

    async def _chunk_by_sentence(
        self, document: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """
        Split document into chunks by preserving complete sentences.

        This chunking method respects sentence boundaries when dividing documents,
        ensuring that no sentence is fragmented across multiple chunks. It balances
        chunk size requirements with maintaining the integrity of natural language
        structures, producing more readable and semantically coherent chunks than
        simple token-based approaches.

        Algorithm details:
        1. Detects sentence boundaries using regular expressions that handle:
           - Standard end punctuation (.!?)
           - Common abbreviations (Mr., Dr., etc.)
           - Edge cases like decimal numbers or acronyms
        2. Builds chunks by adding complete sentences until the target chunk size
           is approached
        3. Creates overlap between chunks by including ending sentences from the
           previous chunk at the beginning of the next chunk
        4. Maintains approximate token count targets while prioritizing sentence
           integrity

        The sentence detection uses a regex pattern that aims to balance accuracy
        with simplicity and efficiency. It identifies likely sentence boundaries by:
        - Looking for punctuation marks followed by whitespace
        - Excluding common patterns that are not sentence boundaries (e.g., "Mr.")
        - Handling basic cases like quotes and parentheses

        This method is ideal for:
        - Natural language text where sentence flow is important
        - Content where breaking mid-sentence would harm readability or context
        - General purpose document processing where semantic units matter
        - Documents that don't have clear paragraph structure

        Args:
            document: Text content to split by sentences
            chunk_size: Target approximate size per chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks

        Returns:
            List of document chunks with complete sentences

        Note:
            The sentence detection uses regex patterns that work well for most
            standard English text but may not handle all edge cases perfectly.
            For specialized text with unusual punctuation patterns, additional
            customization may be needed.
        """
        # Simple sentence splitting (not perfect but works for most cases)
        sentence_delimiters = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        sentences = re.split(sentence_delimiters, document)
        sentences = [s.strip() for s in sentences if s.strip()]

        # No sentences, return empty list
        if not sentences:
            return []

        # Chunk by sentences, trying to reach target size
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            # Estimate size in tokens (approximate)
            sentence_size = len(sentence.split())

            # If adding this sentence exceeds the chunk size and we have content,
            # finalize the current chunk
            if current_chunk and current_size + sentence_size > chunk_size:
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_size = 0
                overlap_chunk = []

                # Add sentences from the end of previous chunk for overlap
                for s in reversed(current_chunk):
                    s_size = len(s.split())
                    if overlap_size + s_size <= chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += s_size
                    else:
                        break

                current_chunk = overlap_chunk
                current_size = overlap_size

            # Add current sentence
            current_chunk.append(sentence)
            current_size += sentence_size

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.debug(
            f"Split document into {len(chunks)} chunks by sentence",
            extra={"emoji_key": "processing"},
        )

        return chunks

    async def _chunk_semantic(
        self, document: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """
        Split document into chunks by semantic meaning, preserving paragraph structure.

        This advanced chunking method attempts to maintain the semantic coherence and
        natural structure of the document by respecting paragraph boundaries whenever
        possible. It implements a hierarchical approach that:

        1. First divides the document by paragraph breaks (blank lines)
        2. Evaluates each paragraph for length
        3. Keeps short and medium paragraphs intact to preserve their meaning
        4. Further splits overly long paragraphs using sentence boundary detection
        5. Assembles chunks with appropriate overlap for context continuity

        The algorithm prioritizes three key aspects of document structure:
        - Paragraph integrity: Treats paragraphs as coherent units of thought
        - Logical flow: Maintains document organization when possible
        - Size constraints: Respects chunk size limitations for downstream processing

        Implementation details:
        - Double newlines (\n\n) are treated as paragraph boundaries
        - If a document lacks clear paragraph structure (e.g., single paragraph),
          it falls back to sentence-based chunking
        - For paragraphs exceeding the chunk size, sentence-based chunking is applied
        - Context preservation is achieved by ensuring the last paragraph of a chunk
          becomes the first paragraph of the next chunk (when appropriate)

        This method is ideal for:
        - Well-structured documents like articles, papers, or reports
        - Content where paragraph organization conveys meaning
        - Documents where natural breaks exist between conceptual sections
        - Cases where preserving document structure improves retrieval quality

        Args:
            document: Text content to split semantically
            chunk_size: Maximum approximate size per chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks

        Returns:
            List of semantic chunks with paragraph structure preserved

        Note:
            Chunk sizes may vary more with semantic chunking than with other methods,
            as maintaining coherent paragraph groups takes precedence over exact
            size enforcement. For strict size control, use token-based chunking.
        """
        # For simplicity, this implementation is similar to sentence chunking
        # but with paragraph awareness

        # Split by paragraphs first
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]

        # Fallback to sentence chunking if no clear paragraphs
        if len(paragraphs) <= 1:
            return await self._chunk_by_sentence(document, chunk_size, chunk_overlap)

        # Process each paragraph and create semantic chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            # Estimate size in tokens
            paragraph_size = len(paragraph.split())

            # If paragraph is very large, chunk it further
            if paragraph_size > chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Chunk large paragraph by sentences
                paragraph_chunks = await self._chunk_by_sentence(
                    paragraph, chunk_size, chunk_overlap
                )
                chunks.extend(paragraph_chunks)
                continue

            # If adding this paragraph exceeds the chunk size and we have content,
            # finalize the current chunk
            if current_chunk and current_size + paragraph_size > chunk_size:
                chunks.append("\n\n".join(current_chunk))

                # Start new chunk with last paragraph for better context
                if current_chunk[-1] != paragraph and len(current_chunk) > 0:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[-1].split())
                else:
                    current_chunk = []
                    current_size = 0

            # Add current paragraph
            current_chunk.append(paragraph)
            current_size += paragraph_size

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        logger.debug(
            f"Split document into {len(chunks)} chunks semantically",
            extra={"emoji_key": "processing"},
        )

        return chunks


# Singleton instance
_document_processor = None


def get_document_processor() -> DocumentProcessor:
    """
    Get or create the singleton DocumentProcessor instance.

    This function implements the singleton pattern for the DocumentProcessor class,
    ensuring that only one instance is created and shared throughout the application.
    It provides a consistent, centralized access point for document processing
    capabilities while conserving system resources.

    Using a singleton for the DocumentProcessor offers several benefits:
    - Resource efficiency: Prevents multiple instantiations of the processor
    - Consistency: Ensures all components use the same processing configuration
    - Centralized access: Provides a clean API for obtaining the processor
    - Lazy initialization: Creates the instance only when first needed

    This function should be used instead of directly instantiating the
    DocumentProcessor class to maintain the singleton pattern and ensure
    proper initialization.

    Returns:
        The shared DocumentProcessor instance

    Usage Example:
    ```python
    # Get the document processor from anywhere in the codebase
    processor = get_document_processor()

    # Use the processor's methods
    chunks = await processor.chunk_document(document_text)
    ```

    Note:
        Even though the DocumentProcessor class itself implements singleton logic in
        its __new__ method, this function is the preferred access point as it handles
        the global instance management and follows the established pattern used
        throughout the MCP server codebase.
    """
    global _document_processor

    if _document_processor is None:
        _document_processor = DocumentProcessor()

    return _document_processor
