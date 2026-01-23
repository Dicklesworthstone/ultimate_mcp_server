"""High-level client for RAG (Retrieval-Augmented Generation) operations."""

from typing import Any, Dict, List, Optional

from ultimate_mcp_server.services.knowledge_base import (
    get_knowledge_base_manager,
    get_knowledge_base_retriever,
    get_rag_service,
)
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.clients.rag")


class RAGClient:
    """
    High-level client for Retrieval-Augmented Generation (RAG) operations.

    The RAGClient provides a simplified, unified interface for building and using
    RAG systems within the MCP ecosystem. It encapsulates all the key operations in
    the RAG workflow, from knowledge base creation and document ingestion to context
    retrieval and LLM-augmented generation.

    RAG is a technique that enhances LLM capabilities by retrieving relevant information
    from external knowledge bases before generation, allowing models to access information
    beyond their training data and produce more accurate, up-to-date responses.

    Key capabilities:
    - Knowledge base creation and management
    - Document ingestion with automatic chunking
    - Semantic retrieval of relevant context
    - LLM generation with retrieved context
    - Various retrieval methods (vector, hybrid, keyword)

    Architecture:
    The client follows a modular architecture with three main components:
    1. Knowledge Base Manager: Handles creation, deletion, and document ingestion
    2. Knowledge Base Retriever: Responsible for context retrieval using various methods
    3. RAG Service: Combines retrieval with LLM generation for complete RAG workflow

    Performance Considerations:
    - Document chunking size affects both retrieval quality and storage requirements
    - Retrieval method selection impacts accuracy vs. speed tradeoffs:
      * Vector search: Fast with good semantic understanding but may miss keyword matches
      * Keyword search: Good for exact matches but misses semantic similarities
      * Hybrid search: Most comprehensive but computationally more expensive
    - Top-k parameter balances between providing sufficient context and relevance dilution
    - Different LLM models may require different prompt templates for optimal performance

    This client abstracts away the complexity of the underlying vector stores,
    embeddings, and retrieval mechanisms, providing a simple API for RAG operations.

    Example usage:
    ```python
    # Create a RAG client
    client = RAGClient()

    # Create a knowledge base
    await client.create_knowledge_base(
        "company_docs",
        "Company documentation and policies"
    )

    # Add documents to the knowledge base with metadata
    await client.add_documents(
        "company_docs",
        documents=[
            "Our return policy allows returns within 30 days of purchase with receipt.",
            "Product warranties cover manufacturing defects for a period of one year."
        ],
        metadatas=[
            {"source": "policies/returns.pdf", "page": 1, "department": "customer_service"},
            {"source": "policies/warranty.pdf", "page": 3, "department": "legal"}
        ],
        chunk_size=500,
        chunk_method="semantic"
    )

    # Retrieve context without generation (for inspection or custom handling)
    context = await client.retrieve(
        "company_docs",
        query="What is our return policy?",
        top_k=3,
        retrieval_method="hybrid"
    )

    # Print retrieved context and sources
    for i, (doc, meta) in enumerate(zip(context["documents"], context["metadatas"])):
        print(f"Source: {meta.get('source', 'unknown')} | Score: {context['distances'][i]:.3f}")
        print(f"Content: {doc[:100]}...\n")

    # Generate a response using RAG with specific provider and template
    result = await client.generate_with_rag(
        "company_docs",
        query="Explain our warranty coverage for electronics",
        provider="openai",
        model="gpt-4",
        template="customer_service_response",
        temperature=0.3,
        retrieval_method="hybrid"
    )

    print(result["response"])
    print("\nSources:")
    for source in result.get("sources", []):
        print(f"- {source['metadata'].get('source')}")
    ```
    """

    def __init__(self):
        """Initialize the RAG client."""
        self.kb_manager = get_knowledge_base_manager()
        self.kb_retriever = get_knowledge_base_retriever()
        self.rag_service = get_rag_service()

    async def create_knowledge_base(
        self, name: str, description: Optional[str] = None, overwrite: bool = False
    ) -> Dict[str, Any]:
        """Create a knowledge base.

        Args:
            name: The name of the knowledge base
            description: Optional description
            overwrite: Whether to overwrite an existing KB with the same name

        Returns:
            Result of the operation
        """
        logger.info(f"Creating knowledge base: {name}", emoji_key="processing")

        try:
            result = await self.kb_manager.create_knowledge_base(
                name=name, description=description, overwrite=overwrite
            )

            logger.success(f"Knowledge base created: {name}", emoji_key="success")
            return result
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {str(e)}", emoji_key="error")
            raise

    async def add_documents(
        self,
        knowledge_base_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_method: str = "semantic",
    ) -> Dict[str, Any]:
        """Add documents to the knowledge base.

        Args:
            knowledge_base_name: Name of the knowledge base to add to
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            chunk_size: Size of chunks to split documents into
            chunk_method: Method to use for chunking ('simple', 'semantic', etc.)

        Returns:
            Result of the operation
        """
        logger.info(
            f"Adding documents to knowledge base: {knowledge_base_name}", emoji_key="processing"
        )

        try:
            result = await self.kb_manager.add_documents(
                knowledge_base_name=knowledge_base_name,
                documents=documents,
                metadatas=metadatas,
                chunk_size=chunk_size,
                chunk_method=chunk_method,
            )

            added_count = result.get("added_count", 0)
            logger.success(f"Added {added_count} documents to knowledge base", emoji_key="success")
            return result
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}", emoji_key="error")
            raise

    async def list_knowledge_bases(self) -> List[Any]:
        """List all knowledge bases.

        Returns:
            List of knowledge base information
        """
        logger.info("Retrieving list of knowledge bases", emoji_key="processing")

        try:
            knowledge_bases = await self.kb_manager.list_knowledge_bases()
            return knowledge_bases
        except Exception as e:
            logger.error(f"Failed to list knowledge bases: {str(e)}", emoji_key="error")
            raise

    async def retrieve(
        self, knowledge_base_name: str, query: str, top_k: int = 3, retrieval_method: str = "vector"
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from a knowledge base for a given query.

        This method performs the retrieval stage of RAG, finding the most relevant
        documents in the knowledge base based on the query. The method is useful
        for standalone retrieval operations or when you want to examine retrieved
        context before generating a response.

        The retrieval process works by:
        1. Converting the query into a vector representation (embedding)
        2. Finding documents with similar embeddings and/or matching keywords
        3. Ranking and returning the most relevant documents

        Available retrieval methods:
        - "vector": Embedding-based similarity search using cosine distance
          Best for conceptual/semantic queries where exact wording may differ
        - "keyword": Traditional text search using BM25 or similar algorithms
          Best for queries with specific terms that must be matched
        - "hybrid": Combines vector and keyword approaches with a weighted blend
          Good general-purpose approach that balances semantic and keyword matching
        - "rerank": Two-stage retrieval that first gets candidates, then reranks them
          More computationally intensive but often more accurate

        Optimization strategies:
        - For factual queries with specific terminology, use "keyword" or "hybrid"
        - For conceptual or paraphrased queries, use "vector"
        - For highest accuracy at cost of performance, use "rerank"
        - Adjust top_k based on document length; shorter documents may need higher top_k
        - Pre-filter by metadata before retrieval when targeting specific sections

        Understanding retrieval metrics:
        - "distances" represent similarity scores where lower values indicate higher similarity
          for vector search, and higher values indicate better matches for keyword search
        - Scores are normalized differently between retrieval methods, so direct
          comparison between methods is not meaningful
        - Score thresholds for "good matches" vary based on embedding model and content domain

        Args:
            knowledge_base_name: Name of the knowledge base to search
            query: The search query (question or keywords)
            top_k: Maximum number of documents to retrieve (default: 3)
            retrieval_method: Method to use for retrieval ("vector", "keyword", "hybrid", "rerank")

        Returns:
            Dictionary containing:
            - "documents": List of retrieved documents (text chunks)
            - "metadatas": List of metadata for each document (source info, etc.)
            - "distances": List of similarity scores or relevance metrics
              (interpretation depends on retrieval_method)
            - "query": The original query
            - "retrieval_method": The method used for retrieval
            - "processing_time_ms": Time taken for retrieval in milliseconds

        Raises:
            ValueError: If knowledge_base_name doesn't exist or query is invalid
            Exception: If the retrieval process fails

        Example:
            ```python
            # Retrieve context about product returns
            results = await rag_client.retrieve(
                knowledge_base_name="support_docs",
                query="How do I return a product?",
                top_k=5,
                retrieval_method="hybrid"
            )

            # Check if we got high-quality matches
            if results["documents"] and min(results["distances"]) < 0.3:  # Good match threshold
                print("Found relevant information!")
            else:
                print("No strong matches found, consider reformulating the query")

            # Display retrieved documents and their sources with scores
            for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
                score = results["distances"][i]
                score_indicator = "🟢" if score < 0.3 else "🟡" if score < 0.6 else "🔴"
                print(f"{score_indicator} Result {i+1} (score: {score:.3f}):")
                print(f"Source: {meta.get('source', 'unknown')}")
                print(doc[:100] + "...\n")
            ```
        """
        logger.info(f"Retrieving context for query: '{query}'", emoji_key="processing")

        try:
            results = await self.kb_retriever.retrieve(
                knowledge_base_name=knowledge_base_name,
                query=query,
                top_k=top_k,
                retrieval_method=retrieval_method,
            )

            return results
        except Exception as e:
            logger.error(f"Failed to retrieve context: {str(e)}", emoji_key="error")
            raise

    async def generate_with_rag(
        self,
        knowledge_base_name: str,
        query: str,
        provider: str = "gemini",
        model: Optional[str] = None,
        template: str = "rag_default",
        temperature: float = 0.3,
        top_k: int = 3,
        retrieval_method: str = "hybrid",
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a response using Retrieval-Augmented Generation (RAG).

        This method performs the complete RAG process:
        1. Retrieves relevant documents from the specified knowledge base based on the query
        2. Constructs a prompt that includes both the query and retrieved context
        3. Sends the augmented prompt to the LLM to generate a response
        4. Optionally includes source information for transparency and citation

        The retrieval process can use different methods:
        - "vector": Pure semantic/embedding-based similarity search (good for conceptual queries)
        - "keyword": Traditional keyword-based search (good for specific terms or phrases)
        - "hybrid": Combines vector and keyword approaches (good general-purpose approach)
        - "rerank": Uses a two-stage approach with retrieval and reranking

        Args:
            knowledge_base_name: Name of the knowledge base to query for relevant context
            query: The user's question or request
            provider: LLM provider to use for generation (e.g., "openai", "anthropic", "gemini")
            model: Specific model to use (if None, uses provider's default)
            template: Prompt template name that defines how to format the RAG prompt
                      Different templates can be optimized for different use cases
            temperature: Sampling temperature for controlling randomness (0.0-1.0)
                         Lower values recommended for factual RAG responses
            top_k: Number of relevant documents to retrieve and include in the context
                   Higher values provide more context but may dilute relevance
            retrieval_method: The method to use for retrieving documents ("vector", "keyword", "hybrid")
            include_sources: Whether to include source information in the output for citations

        Returns:
            Dictionary containing:
            - "response": The generated text response from the LLM
            - "sources": List of source documents and their metadata (if include_sources=True)
            - "context": The retrieved context that was used for generation
            - "prompt": The full prompt that was sent to the LLM (useful for debugging)
            - "tokens": Token usage information (input, output, total)
            - "processing_time_ms": Total processing time in milliseconds

        Raises:
            ValueError: If knowledge_base_name or query is invalid
            Exception: If retrieval or generation fails

        Example:
            ```python
            # Generate a response using RAG with custom parameters
            result = await rag_client.generate_with_rag(
                knowledge_base_name="financial_docs",
                query="What were our Q3 earnings?",
                provider="openai",
                model="gpt-4",
                temperature=0.1,
                top_k=5,
                retrieval_method="hybrid"
            )

            # Access the response and sources
            print(result["response"])
            for src in result["sources"]:
                print(f"- {src['metadata']['source']} (relevance: {src['score']})")
            ```
        """
        logger.info(f"Generating RAG response for: '{query}'", emoji_key="processing")

        try:
            result = await self.rag_service.generate_with_rag(
                knowledge_base_name=knowledge_base_name,
                query=query,
                provider=provider,
                model=model,
                template=template,
                temperature=temperature,
                top_k=top_k,
                retrieval_method=retrieval_method,
                include_sources=include_sources,
            )

            return result
        except Exception as e:
            logger.error(f"Failed to call RAG service: {str(e)}", emoji_key="error")
            raise

    async def delete_knowledge_base(self, name: str) -> Dict[str, Any]:
        """Delete a knowledge base.

        Args:
            name: Name of the knowledge base to delete

        Returns:
            Result of the operation
        """
        logger.info(f"Deleting knowledge base: {name}", emoji_key="processing")

        try:
            result = await self.kb_manager.delete_knowledge_base(name=name)
            logger.success(f"Knowledge base {name} deleted successfully", emoji_key="success")
            return result
        except Exception as e:
            logger.error(f"Failed to delete knowledge base: {str(e)}", emoji_key="error")
            raise

    async def reset_knowledge_base(self, knowledge_base_name: str) -> None:
        """
        Reset (delete and recreate) a knowledge base.

        This method completely removes an existing knowledge base and creates a new
        empty one with the same name. This is useful when you need to:
        - Remove all documents from a knowledge base efficiently
        - Fix a corrupted knowledge base
        - Change the underlying embedding model without renaming the knowledge base
        - Update document chunking strategy for an entire collection
        - Clear outdated information before a complete refresh

        Performance Considerations:
        - Resetting is significantly faster than removing documents individually
        - For large knowledge bases, resetting and bulk re-adding documents can be
          orders of magnitude more efficient than incremental updates
        - New documents added after reset will use any updated embedding models or
          chunking strategies configured in the system

        Data Integrity:
        - This operation preserves knowledge base configuration but removes all content
        - The knowledge base name and any associated permissions remain intact
        - Custom configuration settings on the knowledge base will be preserved
          if the knowledge base service supports configuration persistence

        WARNING: This operation is irreversible. All documents and their embeddings
        will be permanently deleted. Consider backing up important data before resetting.

        Disaster Recovery:
        Before resetting a production knowledge base, consider these strategies:
        1. Create a backup by exporting documents and metadata if the feature is available
        2. Maintain source documents in original form outside the knowledge base
        3. Document the ingestion pipeline to reproduce the knowledge base if needed
        4. Consider creating a temporary duplicate before resetting critical knowledge bases

        The reset process:
        1. Deletes the entire knowledge base collection/index
        2. Creates a new empty knowledge base with the same name
        3. Re-initializes any associated metadata or settings

        Args:
            knowledge_base_name: Name of the knowledge base to reset

        Returns:
            None

        Raises:
            ValueError: If the knowledge base doesn't exist
            Exception: If deletion or recreation fails

        Example:
            ```python
            # Backup critical metadata before reset (if needed)
            kb_info = await rag_client.list_knowledge_bases()
            kb_config = next((kb for kb in kb_info if kb['name'] == 'product_documentation'), None)

            # Reset a knowledge base that may have outdated or corrupted data
            await rag_client.reset_knowledge_base("product_documentation")

            # After resetting, re-add documents with potentially improved chunking strategy
            await rag_client.add_documents(
                knowledge_base_name="product_documentation",
                documents=updated_docs,
                metadatas=updated_metadatas,
                chunk_size=800,  # Updated chunk size better suited for the content
                chunk_method="semantic"  # Using semantic chunking for better results
            )
            ```
        """
        logger.info(f"Deleting knowledge base: {knowledge_base_name}", emoji_key="processing")

        try:
            result = await self.kb_manager.delete_knowledge_base(name=knowledge_base_name)
            logger.success(
                f"Knowledge base {knowledge_base_name} deleted successfully", emoji_key="success"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to delete knowledge base: {str(e)}", emoji_key="error")
            raise
