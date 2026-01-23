"""Knowledge base manager for RAG functionality."""

import time
from typing import Any, Dict, List, Optional

from ultimate_mcp_server.services.vector import VectorDatabaseService
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


class KnowledgeBaseManager:
    """
    Manager for creating and maintaining knowledge bases for RAG applications.

    The KnowledgeBaseManager provides a high-level interface for working with vector
    databases as knowledge bases for Retrieval-Augmented Generation (RAG) systems.
    It abstracts the complexities of vector database operations, focusing on the
    domain-specific needs of knowledge management for AI applications.

    Key Features:
    - Knowledge base lifecycle management (create, delete, list, get)
    - Document ingestion with metadata support
    - Vector embedding management for semantic search
    - Document chunking and processing
    - Persistence and durability guarantees
    - Metadata tracking for knowledge base statistics

    Architecture:
    The manager sits between RAG applications and the underlying vector database,
    providing domain-specific operations while delegating storage and embedding
    to specialized services. It primarily interacts with:
    1. Vector Database Service - for persistent storage of embeddings and documents
    2. Embedding Service - for converting text to vector representations
    3. Text Chunking Service - for breaking documents into optimal retrieval units

    Technical Characteristics:
    - Asynchronous API for high throughput in server environments
    - Thread-safe operations for concurrent access
    - Consistent error handling and logging
    - Idempotent operations where possible
    - Transactional guarantees for critical operations

    This service is typically accessed through the singleton get_knowledge_base_manager()
    function, which ensures a single instance is shared across the application.

    Example Usage:
    ```python
    # Get the manager
    kb_manager = get_knowledge_base_manager()

    # Create a new knowledge base
    await kb_manager.create_knowledge_base(
        name="company_policies",
        description="Corporate policy documents and guidelines"
    )

    # Add documents with metadata
    await kb_manager.add_documents(
        knowledge_base_name="company_policies",
        documents=[
            "All employees must complete annual security training.",
            "Remote work is available for eligible positions with manager approval."
        ],
        metadatas=[
            {"source": "security_policy.pdf", "category": "security", "page": 12},
            {"source": "hr_handbook.pdf", "category": "remote_work", "page": 45}
        ],
        chunk_size=500,
        chunk_method="semantic"
    )

    # List available knowledge bases
    kb_list = await kb_manager.list_knowledge_bases()
    print(f"Found {kb_list['count']} knowledge bases")

    # Get details about a specific knowledge base
    kb_info = await kb_manager.get_knowledge_base("company_policies")
    doc_count = kb_info.get("metadata", {}).get("doc_count", 0)
    print(f"Knowledge base contains {doc_count} document chunks")
    ```
    """

    def __init__(self, vector_service: VectorDatabaseService):
        """Initialize the knowledge base manager.

        Args:
            vector_service: Vector database service for storing embeddings
        """
        self.vector_service = vector_service
        logger.info("Knowledge base manager initialized", extra={"emoji_key": "success"})

    async def create_knowledge_base(
        self,
        name: str,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Create a new knowledge base.

        Args:
            name: Knowledge base name
            description: Optional description
            embedding_model: Optional embedding model name
            overwrite: Whether to overwrite existing knowledge base

        Returns:
            Knowledge base metadata
        """
        # Check if knowledge base already exists
        collections = await self.vector_service.list_collections()

        if name in collections and not overwrite:
            logger.warning(
                f"Knowledge base '{name}' already exists", extra={"emoji_key": "warning"}
            )
            return {"status": "exists", "name": name}

        # Create new collection for knowledge base
        metadata = {
            "type": "knowledge_base",
            "description": description or "",
            "created_at": time.time(),
            "doc_count": 0,
        }

        # Only add embedding_model if not None (to avoid ChromaDB errors)
        if embedding_model is not None:
            metadata["embedding_model"] = embedding_model

        logger.debug(f"Creating knowledge base with metadata: {metadata}")

        # Ensure any existing collection is deleted first
        if overwrite:
            try:
                # Force delete any existing collection
                await self.vector_service.delete_collection(name)
                logger.debug(f"Force deleted existing collection '{name}' for clean creation")
                # Add a small delay to ensure deletion completes
                import asyncio

                await asyncio.sleep(0.2)
            except Exception as e:
                logger.debug(f"Error during force deletion: {str(e)}")

        try:
            await self.vector_service.create_collection(name, metadata=metadata)

            logger.info(f"Created knowledge base '{name}'", extra={"emoji_key": "success"})

            return {"status": "created", "name": name, "metadata": metadata}
        except Exception as e:
            logger.error(
                f"Failed to create knowledge base '{name}': {str(e)}", extra={"emoji_key": "error"}
            )
            raise ValueError(f"Failed to create knowledge base: {str(e)}") from e

    async def delete_knowledge_base(self, name: str) -> Dict[str, Any]:
        """Delete a knowledge base.

        Args:
            name: Knowledge base name

        Returns:
            Deletion status
        """
        # Check if knowledge base exists
        collections = await self.vector_service.list_collections()

        if name not in collections:
            logger.warning(f"Knowledge base '{name}' not found", extra={"emoji_key": "warning"})
            return {"status": "not_found", "name": name}

        # Delete collection
        await self.vector_service.delete_collection(name)

        logger.info(f"Deleted knowledge base '{name}'", extra={"emoji_key": "success"})

        return {"status": "deleted", "name": name}

    async def list_knowledge_bases(self):
        """List all knowledge bases.

        Returns:
            List of knowledge bases with metadata
        """
        collection_names = await self.vector_service.list_collections()
        kb_list = []

        for name in collection_names:
            try:
                metadata = await self.vector_service.get_collection_metadata(name)
                # Only include collections that are knowledge bases
                if metadata and metadata.get("type") == "knowledge_base":
                    # Create a simple dict with name and metadata
                    kb = {"name": name, "metadata": metadata}
                    kb_list.append(kb)
            except Exception as e:
                logger.error(
                    f"Error getting metadata for collection '{name}': {str(e)}",
                    extra={"emoji_key": "error"},
                )

        return {"count": len(kb_list), "knowledge_bases": kb_list}

    async def get_knowledge_base(self, name: str) -> Dict[str, Any]:
        """Get knowledge base metadata.

        Args:
            name: Knowledge base name

        Returns:
            Knowledge base metadata
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

        return {"status": "found", "name": name, "metadata": metadata}

    async def add_documents(
        self,
        knowledge_base_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_method: str = "semantic",
    ) -> Dict[str, Any]:
        """Add documents to a knowledge base.

        Args:
            knowledge_base_name: Knowledge base name
            documents: List of document texts
            metadatas: Optional list of document metadata
            ids: Optional list of document IDs
            embedding_model: Optional embedding model name
            chunk_size: Chunk size for document processing
            chunk_overlap: Chunk overlap for document processing
            chunk_method: Chunking method (token, semantic, etc.)

        Returns:
            Document addition status
        """
        logger.debug(f"DEBUG: Adding documents to knowledge base '{knowledge_base_name}'")
        logger.debug(f"DEBUG: Document count: {len(documents)}")
        logger.debug(f"DEBUG: First document sample: {documents[0][:100]}...")
        logger.debug(f"DEBUG: Metadatas: {metadatas[:2] if metadatas else None}")
        logger.debug(
            f"DEBUG: Chunk settings - size: {chunk_size}, overlap: {chunk_overlap}, method: {chunk_method}"
        )

        # Check if knowledge base exists
        kb_info = await self.get_knowledge_base(knowledge_base_name)

        if kb_info["status"] != "found":
            logger.warning(
                f"Knowledge base '{knowledge_base_name}' not found", extra={"emoji_key": "warning"}
            )
            return {"status": "not_found", "name": knowledge_base_name}

        try:
            # Add documents to vector store
            doc_ids = await self.vector_service.add_texts(
                collection_name=knowledge_base_name,
                texts=documents,
                metadatas=metadatas,
                ids=ids,
                embedding_model=embedding_model,
            )

            # Update document count in metadata
            current_metadata = await self.vector_service.get_collection_metadata(
                knowledge_base_name
            )
            doc_count = current_metadata.get("doc_count", 0) + len(documents)

            # Prepare metadata updates
            metadata_updates = {"doc_count": doc_count}

            # Store embedding model in metadata if provided (for consistent retrieval)
            if embedding_model:
                metadata_updates["embedding_model"] = embedding_model

            # Update metadata
            await self.vector_service.update_collection_metadata(
                name=knowledge_base_name, metadata=metadata_updates
            )

            logger.info(
                f"Added {len(documents)} documents to knowledge base '{knowledge_base_name}'",
                extra={"emoji_key": "success"},
            )

            return {
                "status": "success",
                "name": knowledge_base_name,
                "added_count": len(documents),
                "ids": doc_ids,
            }
        except Exception as e:
            logger.error(
                f"Error adding documents to knowledge base '{knowledge_base_name}': {str(e)}",
                extra={"emoji_key": "error"},
            )
            raise
