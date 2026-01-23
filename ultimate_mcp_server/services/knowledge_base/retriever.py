"""Knowledge base retriever for RAG functionality."""

import time
from typing import Any, Dict, List, Optional

from ultimate_mcp_server.services.knowledge_base.feedback import get_rag_feedback_service
from ultimate_mcp_server.services.knowledge_base.utils import build_metadata_filter
from ultimate_mcp_server.services.vector import VectorDatabaseService
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


class KnowledgeBaseRetriever:
    """
    Advanced retrieval engine for knowledge base collections in RAG applications.

    The KnowledgeBaseRetriever provides sophisticated search capabilities for finding
    the most relevant documents within knowledge bases. It offers multiple retrieval
    strategies optimized for different search scenarios, from pure semantic vector
    search to hybrid approaches combining vector and keyword matching.

    Key Features:
    - Multiple retrieval methods (vector, hybrid, keyword)
    - Metadata filtering for targeted searches
    - Content-based filtering for keyword matching
    - Configurable similarity thresholds and relevance scoring
    - Feedback mechanisms for continuous retrieval improvement
    - Performance monitoring and diagnostics
    - Advanced parameter tuning for specialized search needs

    Retrieval Methods:
    1. Vector Search: Uses embeddings for semantic similarity matching
       - Best for finding conceptually related content
       - Handles paraphrasing and semantic equivalence
       - Computationally efficient for large collections

    2. Hybrid Search: Combines vector and keyword matching with weighted scoring
       - Balances semantic understanding with exact term matching
       - Addresses vocabulary mismatch problems
       - Provides more robust retrieval across diverse query types

    3. Keyword Filtering: Limits results to those containing specific text
       - Used for explicit term presence requirements
       - Can be combined with other search methods

    Architecture:
    The retriever operates as a higher-level service above the vector database,
    working in concert with:
    - Embedding services for query vectorization
    - Vector database services for efficient similarity search
    - Feedback services for result quality improvement
    - Metadata filters for context-aware retrieval

    Usage in RAG Applications:
    This retriever is a critical component in RAG pipelines, responsible for
    the quality and relevance of context provided to LLMs. Tuning retrieval
    parameters significantly impacts the quality of generated responses.

    Example Usage:
    ```python
    # Get retriever instance
    retriever = get_knowledge_base_retriever()

    # Simple vector search
    results = await retriever.retrieve(
        knowledge_base_name="company_policies",
        query="What is our remote work policy?",
        top_k=3,
        min_score=0.7
    )

    # Hybrid search with metadata filtering
    dept_results = await retriever.retrieve_hybrid(
        knowledge_base_name="company_policies",
        query="security requirements for customer data",
        top_k=5,
        vector_weight=0.6,
        keyword_weight=0.4,
        metadata_filter={"department": "security", "status": "active"}
    )

    # Process and use the retrieved documents
    for item in results["results"]:
        print(f"Document (score: {item['score']:.2f}): {item['document'][:100]}...")
        print(f"Source: {item['metadata'].get('source', 'unknown')}")

    # Record which documents were actually useful
    await retriever.record_feedback(
        knowledge_base_name="company_policies",
        query="What is our remote work policy?",
        retrieved_documents=results["results"],
        used_document_ids=["doc123", "doc456"]
    )
    ```
    """

    def __init__(self, vector_service: VectorDatabaseService):
        """Initialize the knowledge base retriever.

        Args:
            vector_service: Vector database service for retrieving embeddings
        """
        self.vector_service = vector_service
        self.feedback_service = get_rag_feedback_service()

        # Get embedding service for generating query embeddings
        from ultimate_mcp_server.services.vector.embeddings import get_embedding_service

        self.embedding_service = get_embedding_service()

        logger.info("Knowledge base retriever initialized", extra={"emoji_key": "success"})

    async def _validate_knowledge_base(self, name: str) -> Dict[str, Any]:
        """Validate that a knowledge base exists.

        Args:
            name: Knowledge base name

        Returns:
            Validation result
        """
        # Check if knowledge base exists
        collections = await self.vector_service.list_collections()

        if name not in collections:
            logger.warning(f"Knowledge base '{name}' not found", extra={"emoji_key": "warning"})
            return {"status": "not_found", "name": name}

        # Get metadata
        metadata = await self.vector_service.get_collection_metadata(name)

        if metadata.get("type") != "knowledge_base":
            logger.warning(
                f"Collection '{name}' is not a knowledge base", extra={"emoji_key": "warning"}
            )
            return {"status": "not_knowledge_base", "name": name}

        return {"status": "valid", "name": name, "metadata": metadata}

    async def retrieve(
        self,
        knowledge_base_name: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.6,
        metadata_filter: Optional[Dict[str, Any]] = None,
        content_filter: Optional[str] = None,
        embedding_model: Optional[str] = None,
        apply_feedback: bool = True,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve documents from a knowledge base using vector search.

        Args:
            knowledge_base_name: Knowledge base name
            query: Query text
            top_k: Number of results to return
            min_score: Minimum similarity score
            metadata_filter: Optional metadata filter (field->value or field->{op:value})
            content_filter: Text to search for in documents
            embedding_model: Optional embedding model name
            apply_feedback: Whether to apply feedback adjustments
            search_params: Optional ChromaDB search parameters

        Returns:
            Retrieved documents with metadata
        """
        start_time = time.time()

        # Validate knowledge base
        kb_info = await self._validate_knowledge_base(knowledge_base_name)

        if kb_info["status"] != "valid":
            logger.warning(
                f"Knowledge base '{knowledge_base_name}' not found or invalid",
                extra={"emoji_key": "warning"},
            )
            return {
                "status": "error",
                "message": f"Knowledge base '{knowledge_base_name}' not found or invalid",
            }

        logger.debug(f"DEBUG: Knowledge base validated - metadata: {kb_info['metadata']}")

        # Use the same embedding model that was used to create the knowledge base
        if not embedding_model and kb_info["metadata"].get("embedding_model"):
            embedding_model = kb_info["metadata"]["embedding_model"]
            logger.debug(f"Using embedding model from knowledge base metadata: {embedding_model}")

        # If embedding model is specified, ensure it's saved in the metadata for future use
        if embedding_model and not kb_info["metadata"].get("embedding_model"):
            try:
                await self.vector_service.update_collection_metadata(
                    name=knowledge_base_name,
                    metadata={**kb_info["metadata"], "embedding_model": embedding_model},
                )
                logger.debug(
                    f"Updated knowledge base metadata with embedding model: {embedding_model}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to update knowledge base metadata with embedding model: {str(e)}"
                )

        # Get or create ChromaDB collection
        collection = await self.vector_service.get_collection(knowledge_base_name)
        logger.debug(f"DEBUG: Retrieved collection type: {type(collection)}")

        # Set search parameters if provided
        if search_params:
            await self.vector_service.update_collection_metadata(
                collection_name=knowledge_base_name,
                metadata={
                    **kb_info["metadata"],
                    **{f"hnsw:{k}": v for k, v in search_params.items()},
                },
            )

        # Create includes parameter
        includes = ["documents", "metadatas", "distances"]

        # Create where_document parameter for content filtering
        where_document = {"$contains": content_filter} if content_filter else None

        # Convert metadata filter format if provided
        chroma_filter = build_metadata_filter(metadata_filter) if metadata_filter else None

        logger.debug(
            f"DEBUG: Search parameters - top_k: {top_k}, min_score: {min_score}, filter: {chroma_filter}, where_document: {where_document}"
        )

        try:
            # Generate embedding directly with our embedding service
            # Call create_embeddings with a list and get the first result
            query_embeddings = await self.embedding_service.create_embeddings(
                texts=[query],
                # model=embedding_model # Model is set during service init
            )
            if not query_embeddings:
                logger.error(f"Failed to generate embedding for query: {query}")
                return {"status": "error", "message": "Failed to generate query embedding"}
            query_embedding = query_embeddings[0]

            logger.debug(
                f"Generated query embedding with model: {self.embedding_service.model_name}, dimension: {len(query_embedding)}"
            )

            # Use correct query method based on collection type
            if hasattr(collection, "query") and not hasattr(collection, "search_by_text"):
                # ChromaDB collection
                logger.debug("Using ChromaDB direct query with embeddings")
                try:
                    search_results = collection.query(
                        query_embeddings=[query_embedding],  # Use our embedding directly
                        n_results=top_k * 2,
                        where=chroma_filter,
                        where_document=where_document,
                        include=includes,
                    )
                except Exception as e:
                    logger.error(f"ChromaDB query error: {str(e)}")
                    raise
            else:
                # Our custom VectorCollection
                logger.debug("Using VectorCollection search method")
                search_results = await collection.query(
                    query_texts=[query],
                    n_results=top_k * 2,
                    where=chroma_filter,
                    where_document=where_document,
                    include=includes,
                    embedding_model=embedding_model,
                )

            # Debug raw results
            logger.debug(f"DEBUG: Raw search results - keys: {search_results.keys()}")
            logger.debug(f"DEBUG: Documents count: {len(search_results.get('documents', [[]])[0])}")
            logger.debug(f"DEBUG: IDs: {search_results.get('ids', [[]])[0]}")
            logger.debug(f"DEBUG: Distances: {search_results.get('distances', [[]])[0]}")

            # Process results
            results = []
            for i, doc in enumerate(search_results["documents"][0]):
                # Convert distance to similarity score (1 = exact match, 0 = completely different)
                # Most distance metrics return 0 for exact match, so we use 1 - distance
                # This works for cosine, l2, etc.
                similarity = 1.0 - float(search_results["distances"][0][i])

                # Debug each document
                logger.debug(f"DEBUG: Document {i} - ID: {search_results['ids'][0][i]}")
                logger.debug(f"DEBUG: Similarity: {similarity} (min required: {min_score})")
                logger.debug(
                    f"DEBUG: Document content (first 100 chars): {doc[:100] if doc else 'Empty'}"
                )

                if search_results["metadatas"] and i < len(search_results["metadatas"][0]):
                    metadata = search_results["metadatas"][0][i]
                    logger.debug(f"DEBUG: Metadata: {metadata}")

                # Skip results below minimum score
                if similarity < min_score:
                    logger.debug(
                        f"DEBUG: Skipping document {i} due to low similarity: {similarity} < {min_score}"
                    )
                    continue

                results.append(
                    {
                        "id": search_results["ids"][0][i],
                        "document": doc,
                        "metadata": search_results["metadatas"][0][i]
                        if search_results["metadatas"]
                        else {},
                        "score": similarity,
                    }
                )

            logger.debug(f"DEBUG: After filtering, {len(results)} documents remain.")

            # Apply feedback adjustments if requested
            if apply_feedback:
                results = await self.feedback_service.apply_feedback_adjustments(
                    knowledge_base_name=knowledge_base_name, results=results, query=query
                )

            # Limit to top_k
            results = results[:top_k]

            # Track retrieval time
            retrieval_time = time.time() - start_time

            logger.info(
                f"Retrieved {len(results)} documents from '{knowledge_base_name}' in {retrieval_time:.2f}s",
                extra={"emoji_key": "success"},
            )

            return {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
                "retrieval_time": retrieval_time,
            }

        except Exception as e:
            logger.error(
                f"Error retrieving from knowledge base '{knowledge_base_name}': {str(e)}",
                extra={"emoji_key": "error"},
            )

            return {"status": "error", "message": str(e)}

    async def retrieve_hybrid(
        self,
        knowledge_base_name: str,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        min_score: float = 0.6,
        metadata_filter: Optional[Dict[str, Any]] = None,
        additional_keywords: Optional[List[str]] = None,
        apply_feedback: bool = True,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve documents using hybrid search.

        Args:
            knowledge_base_name: Knowledge base name
            query: Query text
            top_k: Number of documents to retrieve
            vector_weight: Weight for vector search component
            keyword_weight: Weight for keyword search component
            min_score: Minimum similarity score
            metadata_filter: Optional metadata filter
            additional_keywords: Additional keywords to include in search
            apply_feedback: Whether to apply feedback adjustments
            search_params: Optional ChromaDB search parameters

        Returns:
            Retrieved documents with metadata
        """
        start_time = time.time()

        # Validate knowledge base
        kb_info = await self._validate_knowledge_base(knowledge_base_name)

        if kb_info["status"] != "valid":
            logger.warning(
                f"Knowledge base '{knowledge_base_name}' not found or invalid",
                extra={"emoji_key": "warning"},
            )
            return {
                "status": "error",
                "message": f"Knowledge base '{knowledge_base_name}' not found or invalid",
            }

        # Get or create ChromaDB collection
        collection = await self.vector_service.get_collection(knowledge_base_name)

        # Set search parameters if provided
        if search_params:
            await self.vector_service.update_collection_metadata(
                collection_name=knowledge_base_name,
                metadata={
                    **kb_info["metadata"],
                    **{f"hnsw:{k}": v for k, v in search_params.items()},
                },
            )

        # Convert metadata filter format if provided
        chroma_filter = build_metadata_filter(metadata_filter) if metadata_filter else None

        # Create content filter based on query and additional keywords
        content_text = query
        if additional_keywords:
            content_text = f"{query} {' '.join(additional_keywords)}"

        # Use ChromaDB's hybrid search by providing both query text and content filter
        try:
            # Vector search results with content filter
            search_results = await collection.query(
                query_texts=[query],
                n_results=top_k * 3,  # Get more results for combining
                where=chroma_filter,
                where_document={"$contains": content_text} if content_text else None,
                include=["documents", "metadatas", "distances"],
                embedding_model=None,  # Use default embedding model
            )

            # Process results
            combined_results = {}

            # Process vector search results
            for i, doc in enumerate(search_results["documents"][0]):
                doc_id = search_results["ids"][0][i]
                vector_score = 1.0 - float(search_results["distances"][0][i])

                combined_results[doc_id] = {
                    "id": doc_id,
                    "document": doc,
                    "metadata": search_results["metadatas"][0][i]
                    if search_results["metadatas"]
                    else {},
                    "vector_score": vector_score,
                    "keyword_score": 0.0,
                    "score": vector_score * vector_weight,
                }

            # Now do a keyword-only search if we have keywords component
            if keyword_weight > 0:
                keyword_results = await collection.query(
                    query_texts=None,  # No vector query
                    n_results=top_k * 3,
                    where=chroma_filter,
                    where_document={"$contains": content_text},
                    include=["documents", "metadatas"],
                    embedding_model=None,  # No embedding model needed for keyword-only search
                )

                # Process keyword results
                for i, doc in enumerate(keyword_results["documents"][0]):
                    doc_id = keyword_results["ids"][0][i]
                    # Approximate keyword score based on position (best = 1.0)
                    keyword_score = 1.0 - (i / len(keyword_results["documents"][0]))

                    if doc_id in combined_results:
                        # Update existing result
                        combined_results[doc_id]["keyword_score"] = keyword_score
                        combined_results[doc_id]["score"] += keyword_score * keyword_weight
                    else:
                        # Add new result
                        combined_results[doc_id] = {
                            "id": doc_id,
                            "document": doc,
                            "metadata": keyword_results["metadatas"][0][i]
                            if keyword_results["metadatas"]
                            else {},
                            "vector_score": 0.0,
                            "keyword_score": keyword_score,
                            "score": keyword_score * keyword_weight,
                        }

            # Convert to list and filter by min_score
            results = [r for r in combined_results.values() if r["score"] >= min_score]

            # Apply feedback adjustments if requested
            if apply_feedback:
                results = await self.feedback_service.apply_feedback_adjustments(
                    knowledge_base_name=knowledge_base_name, results=results, query=query
                )

            # Sort by score and limit to top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

            # Track retrieval time
            retrieval_time = time.time() - start_time

            logger.info(
                f"Hybrid search retrieved {len(results)} documents from '{knowledge_base_name}' in {retrieval_time:.2f}s",
                extra={"emoji_key": "success"},
            )

            return {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
                "retrieval_time": retrieval_time,
            }

        except Exception as e:
            logger.error(
                f"Error performing hybrid search on '{knowledge_base_name}': {str(e)}",
                extra={"emoji_key": "error"},
            )

            return {"status": "error", "message": str(e)}

    async def record_feedback(
        self,
        knowledge_base_name: str,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        used_document_ids: Optional[List[str]] = None,
        explicit_feedback: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Record feedback for retrieval results.

        Args:
            knowledge_base_name: Knowledge base name
            query: Query text
            retrieved_documents: List of retrieved documents
            used_document_ids: List of document IDs used in the response
            explicit_feedback: Explicit feedback for documents

        Returns:
            Feedback recording result
        """
        # Convert list to set if provided
        used_ids_set = set(used_document_ids) if used_document_ids else None

        # Record feedback
        result = await self.feedback_service.record_retrieval_feedback(
            knowledge_base_name=knowledge_base_name,
            query=query,
            retrieved_documents=retrieved_documents,
            used_document_ids=used_ids_set,
            explicit_feedback=explicit_feedback,
        )

        return result
