"""MCP tools for Retrieval-Augmented Generation (RAG).

Provides functions to create, manage, and query knowledge bases (vector stores)
and generate text responses augmented with retrieved context.
"""

import re
from typing import Any, Dict, List, Optional

# Import specific exceptions for better error handling hints
from ultimate_mcp_server.exceptions import ProviderError, ResourceError, ToolInputError
from ultimate_mcp_server.services import get_rag_engine

# Moved imports for services to the top level
from ultimate_mcp_server.services.knowledge_base import (
    get_knowledge_base_manager,
    get_knowledge_base_retriever,
)
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)

# --- Service Lazy Initialization ---

_kb_manager = None
_kb_retriever = None
_rag_engine = None


def _get_kb_manager():
    """Lazily initializes and returns the Knowledge Base Manager."""
    global _kb_manager
    if _kb_manager is None:
        logger.debug("Initializing KnowledgeBaseManager...")
        _kb_manager = get_knowledge_base_manager()
        logger.info("KnowledgeBaseManager initialized.")
    return _kb_manager


def _get_kb_retriever():
    """Lazily initializes and returns the Knowledge Base Retriever."""
    global _kb_retriever
    if _kb_retriever is None:
        logger.debug("Initializing KnowledgeBaseRetriever...")
        _kb_retriever = get_knowledge_base_retriever()
        logger.info("KnowledgeBaseRetriever initialized.")
    return _kb_retriever


def _get_rag_engine():
    """Lazily initializes and returns the RAG Engine."""
    global _rag_engine
    if _rag_engine is None:
        logger.debug("Initializing RAGEngine...")
        _rag_engine = get_rag_engine()
        logger.info("RAGEngine initialized.")
    return _rag_engine


# --- Standalone Tool Functions ---


@with_tool_metrics
@with_error_handling
async def create_knowledge_base(
    name: str,
    description: Optional[str] = None,
    embedding_model: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Creates a new, empty knowledge base (vector store) to hold documents.

    This is the first step before adding documents.

    Args:
        name: A unique name for the knowledge base (e.g., "project_docs_v1").
              Must be a valid identifier (letters, numbers, underscores).
        description: (Optional) A brief description of the knowledge base's content or purpose.
        embedding_model: (Optional) The specific embedding model ID to use for this knowledge base
                         (e.g., "openai/text-embedding-3-small"). If None, uses the system default.
                         Consistency is important; use the same model when adding documents later.
        overwrite: (Optional) If True, deletes and recreates the knowledge base if one with the
                   same name already exists. Defaults to False (raises an error if exists).

    Returns:
        A dictionary confirming the creation:
        {
            "success": true,
            "name": "project_docs_v1",
            "message": "Knowledge base 'project_docs_v1' created successfully."
        }
        or an error dictionary if creation failed:
        {
            "success": false,
            "name": "project_docs_v1",
            "error": "Knowledge base 'project_docs_v1' already exists."
        }

    Raises:
        ResourceError: If the knowledge base already exists (and overwrite=False) or
                        if there's an issue during creation (e.g., invalid name).
        ToolInputError: If the provided name is invalid.
    """
    # Input validation (basic example)
    if not name or not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise ToolInputError(
            f"Invalid knowledge base name: '{name}'. Use only letters, numbers, underscores."
        )

    kb_manager = _get_kb_manager()  # Use lazy getter
    try:
        result = await kb_manager.create_knowledge_base(
            name=name, description=description, embedding_model=embedding_model, overwrite=overwrite
        )
        return result
    except Exception as e:
        logger.error(f"Failed to create knowledge base '{name}': {e}", exc_info=True)
        # Re-raise specific error if possible, otherwise wrap
        if isinstance(e, (ResourceError, ToolInputError)):
            raise
        raise ResourceError(
            f"Failed to create knowledge base '{name}': {str(e)}",
            resource_type="knowledge_base",
            resource_id=name,
            cause=e,
        ) from e


@with_tool_metrics
@with_error_handling
async def list_knowledge_bases() -> Dict[str, Any]:
    """Lists all available knowledge bases and their metadata.

    Returns:
        A dictionary containing a list of knowledge base details:
        {
            "success": true,
            "knowledge_bases": [
                {
                    "name": "project_docs_v1",
                    "description": "Documentation for Project X",
                    "embedding_model": "openai/text-embedding-3-small",
                    "document_count": 150,
                    "created_at": "2023-10-27T10:00:00Z"
                },
                { ... } # Other knowledge bases
            ]
        }
        or an error dictionary:
        {
            "success": false,
            "error": "Failed to retrieve knowledge base list."
        }
    Raises:
        ResourceError: If there's an issue retrieving the list from the backend.
    """
    kb_manager = _get_kb_manager()
    try:
        result = await kb_manager.list_knowledge_bases()
        return result
    except Exception as e:
        logger.error(f"Failed to list knowledge bases: {e}", exc_info=True)
        raise ResourceError(
            f"Failed to list knowledge bases: {str(e)}", resource_type="knowledge_base", cause=e
        ) from e


@with_tool_metrics
@with_error_handling
async def delete_knowledge_base(name: str) -> Dict[str, Any]:
    """Deletes an existing knowledge base and all its documents.

    Warning: This action is irreversible.

    Args:
        name: The exact name of the knowledge base to delete.

    Returns:
        A dictionary confirming the deletion:
        {
            "success": true,
            "name": "project_docs_v1",
            "message": "Knowledge base 'project_docs_v1' deleted successfully."
        }
        or an error dictionary:
        {
            "success": false,
            "name": "project_docs_v1",
            "error": "Knowledge base 'project_docs_v1' not found."
        }

    Raises:
        ResourceError: If the knowledge base doesn't exist or if deletion fails.
        ToolInputError: If the provided name is invalid.
    """
    if not name:
        raise ToolInputError("Knowledge base name cannot be empty.")

    kb_manager = _get_kb_manager()
    try:
        result = await kb_manager.delete_knowledge_base(name)
        return result
    except Exception as e:
        logger.error(f"Failed to delete knowledge base '{name}': {e}", exc_info=True)
        if isinstance(e, (ResourceError, ToolInputError)):
            raise
        raise ResourceError(
            f"Failed to delete knowledge base '{name}': {str(e)}",
            resource_type="knowledge_base",
            resource_id=name,
            cause=e,
        ) from e


@with_tool_metrics
@with_error_handling
async def add_documents(
    knowledge_base_name: str,
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunk_method: str = "semantic",
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Adds one or more documents to a specified knowledge base.

    The documents are split into chunks, embedded, and stored for later retrieval.

    Args:
        knowledge_base_name: The name of the existing knowledge base to add documents to.
        documents: A list of strings, where each string is the full text content of a document.
        metadatas: (Optional) A list of dictionaries, one for each document in the `documents` list.
                   Each dictionary should contain metadata relevant to the corresponding document
                   (e.g., {"source": "filename.pdf", "page": 1, "author": "Alice"}).
                   This metadata is stored alongside the document chunks and can be used for filtering during retrieval.
                   If provided, `len(metadatas)` MUST equal `len(documents)`.
        chunk_size: (Optional) The target size for document chunks. Interpretation depends on `chunk_method`
                    (e.g., tokens for "token" method, characters for "character", approximate size for "semantic").
                    Defaults to 1000.
        chunk_overlap: (Optional) The number of units (tokens, characters) to overlap between consecutive chunks.
                       Helps maintain context across chunk boundaries. Defaults to 200.
        chunk_method: (Optional) The method used for splitting documents into chunks.
                      Options: "semantic" (attempts to split at meaningful semantic boundaries, recommended),
                      "token" (splits by token count using tiktoken), "sentence" (splits by sentence).
                      Defaults to "semantic".
        embedding_model: (Optional) The specific embedding model ID to use. If None, uses the model
                         associated with the knowledge base (or the system default if none was specified
                         at creation). It's best practice to ensure this matches the KB's model.

    Returns:
        A dictionary summarizing the addition process:
        {
            "success": true,
            "knowledge_base_name": "project_docs_v1",
            "documents_added": 5,
            "chunks_created": 75,
            "message": "Successfully added 5 documents (75 chunks) to 'project_docs_v1'."
        }
        or an error dictionary:
        {
            "success": false,
            "knowledge_base_name": "project_docs_v1",
            "error": "Knowledge base 'project_docs_v1' not found."
        }

    Raises:
        ResourceError: If the knowledge base doesn't exist or if there's an error during processing/storage.
        ToolInputError: If inputs are invalid (e.g., documents/metadatas length mismatch, invalid chunk_method).
        ProviderError: If the LLM provider fails during generation.
    """
    if not knowledge_base_name:
        raise ToolInputError("Knowledge base name cannot be empty.")
    if (
        not documents
        or not isinstance(documents, list)
        or not all(isinstance(d, str) for d in documents)
    ):
        raise ToolInputError("'documents' must be a non-empty list of strings.")
    if metadatas and (not isinstance(metadatas, list) or len(metadatas) != len(documents)):
        raise ToolInputError(
            "'metadatas', if provided, must be a list with the same length as 'documents'."
        )
    if chunk_method not in [
        "semantic",
        "token",
        "sentence",
        "character",
        "paragraph",
    ]:  # Added more methods
        raise ToolInputError(
            f"Invalid chunk_method: '{chunk_method}'. Must be one of: semantic, token, sentence, character, paragraph."
        )

    kb_manager = _get_kb_manager()
    try:
        result = await kb_manager.add_documents(
            knowledge_base_name=knowledge_base_name,
            documents=documents,
            metadatas=metadatas,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_method=chunk_method,
            embedding_model=embedding_model,
        )
        return result
    except Exception as e:
        logger.error(
            f"Failed to add documents to knowledge base '{knowledge_base_name}': {e}", exc_info=True
        )
        if isinstance(e, (ResourceError, ToolInputError, ProviderError)):
            raise
        raise ResourceError(
            f"Failed to add documents to knowledge base '{knowledge_base_name}': {str(e)}",
            resource_type="knowledge_base",
            resource_id=knowledge_base_name,
            cause=e,
        ) from e


@with_tool_metrics
@with_error_handling
async def retrieve_context(
    knowledge_base_name: str,
    query: str,
    top_k: int = 5,
    retrieval_method: str = "vector",
    min_score: Optional[float] = None,  # Changed default to None
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieves relevant document chunks (context) from a knowledge base based on a query.

    Searches the specified knowledge base for chunks semantically similar to the query.

    Args:
        knowledge_base_name: The name of the knowledge base to query.
        query: The text query to search for relevant context.
        top_k: (Optional) The maximum number of relevant chunks to retrieve. Defaults to 5.
        retrieval_method: (Optional) The method used for retrieval.
                          Options: "vector" (semantic similarity search), "hybrid" (combines vector search
                          with keyword-based search, may require specific backend support).
                          Defaults to "vector".
        min_score: (Optional) The minimum similarity score (typically between 0 and 1) for a chunk
                   to be included in the results. Higher values mean stricter relevance.
                   If None (default), the backend decides or no filtering is applied.
        metadata_filter: (Optional) A dictionary used to filter results based on metadata associated
                         with the chunks during `add_documents`. Filters use exact matches.
                         Example: {"source": "filename.pdf", "page": 5}
                         Example: {"author": "Alice"}
                         Defaults to None (no metadata filtering).

    Returns:
        A dictionary containing the retrieved context:
        {
            "success": true,
            "query": "What are the project goals?",
            "knowledge_base_name": "project_docs_v1",
            "retrieved_chunks": [
                {
                    "content": "The main goal of Project X is to improve user engagement...",
                    "score": 0.85,
                    "metadata": {"source": "project_plan.docx", "page": 1}
                },
                { ... } # Other relevant chunks
            ]
        }
        or an error dictionary:
        {
            "success": false,
            "knowledge_base_name": "project_docs_v1",
            "error": "Knowledge base 'project_docs_v1' not found."
        }

    Raises:
        ResourceError: If the knowledge base doesn't exist or retrieval fails.
        ToolInputError: If inputs are invalid (e.g., invalid retrieval_method).
    """
    if not knowledge_base_name:
        raise ToolInputError("Knowledge base name cannot be empty.")
    if not query or not isinstance(query, str):
        raise ToolInputError("Query must be a non-empty string.")
    if retrieval_method not in ["vector", "hybrid"]:  # Add more methods if supported by backend
        raise ToolInputError(
            f"Invalid retrieval_method: '{retrieval_method}'. Must be one of: vector, hybrid."
        )

    kb_retriever = _get_kb_retriever()
    try:
        # Note: The actual implementation might vary based on the retriever service
        # Here we assume the service handles different methods via parameters or distinct functions
        # Keeping the previous logic structure for now.
        if retrieval_method == "hybrid":
            # Assuming a specific hybrid method exists or the main retrieve handles it
            # This might need adjustment based on the actual service implementation
            logger.debug(f"Attempting hybrid retrieval for '{knowledge_base_name}'")
            result = await kb_retriever.retrieve_hybrid(  # Or potentially kb_retriever.retrieve with a method flag
                knowledge_base_name=knowledge_base_name,
                query=query,
                top_k=top_k,
                min_score=min_score,
                metadata_filter=metadata_filter,
            )
        else:  # Default to vector
            logger.debug(f"Attempting vector retrieval for '{knowledge_base_name}'")
            result = await kb_retriever.retrieve(
                knowledge_base_name=knowledge_base_name,
                query=query,
                top_k=top_k,
                rerank=True,  # Assuming rerank is often desired for vector
                min_score=min_score,
                metadata_filter=metadata_filter,
            )
        return result
    except Exception as e:
        logger.error(
            f"Failed to retrieve context from knowledge base '{knowledge_base_name}' for query '{query}': {e}",
            exc_info=True,
        )
        if isinstance(e, (ResourceError, ToolInputError)):
            raise
        raise ResourceError(
            f"Failed to retrieve context from knowledge base '{knowledge_base_name}': {str(e)}",
            resource_type="knowledge_base",
            resource_id=knowledge_base_name,
            cause=e,
        ) from e


@with_tool_metrics
@with_error_handling
async def generate_with_rag(
    knowledge_base_name: str,
    query: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    template: str = "rag_default",
    max_tokens: int = 1000,
    temperature: float = 0.3,
    top_k: int = 5,
    retrieval_method: str = "vector",
    min_score: Optional[float] = None,  # Changed default to None
    include_sources: bool = True,
) -> Dict[str, Any]:
    """Generates a response to a query using context retrieved from a knowledge base (RAG).

    This function first retrieves relevant document chunks using `retrieve_context` parameters,
    then feeds the query and the retrieved context into an LLM using a specified prompt template
    to generate a final, context-aware answer.

    Args:
        knowledge_base_name: The name of the knowledge base to retrieve context from.
        query: The user's query or question to be answered.
        provider: (Optional) The LLM provider for the generation step (e.g., "openai", "anthropic").
                  If None, the RAG engine selects a default provider.
        model: (Optional) The specific LLM model ID for generation (e.g., "openai/gpt-4.1-mini").
               If None, the RAG engine selects a default model.
        template: (Optional) The name of the prompt template to use for combining the query and context.
                  Available templates might include: "rag_default" (standard Q&A), "rag_with_sources"
                  (default, includes source attribution), "rag_summarize" (summarizes retrieved context
                  based on query), "rag_analysis" (performs analysis based on context).
                  Defaults to "rag_default" (or potentially "rag_with_sources" depending on engine default).
        max_tokens: (Optional) Maximum number of tokens for the generated LLM response. Defaults to 1000.
        temperature: (Optional) Sampling temperature for the LLM generation (0.0 to 1.0). Lower values
                     are more deterministic, higher values more creative. Defaults to 0.3.
        top_k: (Optional) Maximum number of context chunks to retrieve (passed to retrieval). Defaults to 5.
        retrieval_method: (Optional) Method for retrieving context ("vector", "hybrid"). Defaults to "vector".
        min_score: (Optional) Minimum similarity score for retrieved chunks. Defaults to None.
        include_sources: (Optional) Whether the final response object should explicitly include details
                         of the source chunks used for generation. Defaults to True.

    Returns:
        A dictionary containing the generated response and related information:
        {
            "success": true,
            "query": "What are the project goals?",
            "knowledge_base_name": "project_docs_v1",
            "generated_response": "The main goal of Project X is to improve user engagement by implementing features A, B, and C.",
            "sources": [ # Included if include_sources=True
                {
                    "content": "The main goal of Project X is to improve user engagement...",
                    "score": 0.85,
                    "metadata": {"source": "project_plan.docx", "page": 1}
                },
                { ... } # Other source chunks used
            ],
            "model": "openai/gpt-4.1-mini", # Actual model used
            "provider": "openai",
            "tokens": { "input": ..., "output": ..., "total": ... }, # Generation tokens
            "cost": 0.000120,
            "processing_time": 5.2,
            "retrieval_time": 0.8, # Time spent only on retrieval
            "generation_time": 4.4 # Time spent only on generation
        }
        or an error dictionary:
        {
            "success": false,
            "knowledge_base_name": "project_docs_v1",
            "error": "RAG generation failed: Knowledge base 'project_docs_v1' not found."
        }

    Raises:
        ResourceError: If the knowledge base doesn't exist or retrieval fails.
        ProviderError: If the LLM provider fails during generation.
        ToolInputError: If inputs are invalid.
    """
    if not knowledge_base_name:
        raise ToolInputError("Knowledge base name cannot be empty.")
    if not query or not isinstance(query, str):
        raise ToolInputError("Query must be a non-empty string.")

    rag_engine = _get_rag_engine()
    try:
        result = await rag_engine.generate_with_rag(
            knowledge_base_name=knowledge_base_name,
            query=query,
            provider=provider,
            model=model,
            template=template,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            retrieval_method=retrieval_method,
            min_score=min_score,
            include_sources=include_sources,
        )
        return result
    except Exception as e:
        logger.error(
            f"RAG generation failed for query on '{knowledge_base_name}': {e}", exc_info=True
        )
        if isinstance(e, (ResourceError, ProviderError, ToolInputError)):
            raise
        # Wrap generic errors
        raise ResourceError(
            f"RAG generation failed: {str(e)}",
            resource_type="knowledge_base",
            resource_id=knowledge_base_name,
            cause=e,
        ) from e
