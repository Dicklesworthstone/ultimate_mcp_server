"""Knowledge base services for RAG functionality."""

from .manager import KnowledgeBaseManager
from .rag_engine import RAGEngine
from .retriever import KnowledgeBaseRetriever

__all__ = [
    "KnowledgeBaseManager",
    "KnowledgeBaseRetriever",
    "RAGEngine",
    "get_knowledge_base_manager",
    "get_knowledge_base_retriever",
    "get_rag_service",
]

# Singleton instances
_knowledge_base_manager = None
_knowledge_base_retriever = None
_rag_service = None


def get_knowledge_base_manager() -> KnowledgeBaseManager:
    """Get or create a knowledge base manager instance.

    Returns:
        KnowledgeBaseManager: Knowledge base manager instance
    """
    global _knowledge_base_manager

    if _knowledge_base_manager is None:
        from ultimate_mcp_server.services.vector import get_vector_database_service

        vector_service = get_vector_database_service()
        _knowledge_base_manager = KnowledgeBaseManager(vector_service)

    return _knowledge_base_manager


def get_knowledge_base_retriever() -> KnowledgeBaseRetriever:
    """Get or create a knowledge base retriever instance.

    Returns:
        KnowledgeBaseRetriever: Knowledge base retriever instance
    """
    global _knowledge_base_retriever

    if _knowledge_base_retriever is None:
        from ultimate_mcp_server.services.vector import get_vector_database_service

        vector_service = get_vector_database_service()
        _knowledge_base_retriever = KnowledgeBaseRetriever(vector_service)

    return _knowledge_base_retriever


def get_rag_service() -> RAGEngine:
    """Get or create a RAG engine instance.

    Returns:
        RAGEngine: RAG engine instance
    """
    global _rag_service

    if _rag_service is None:
        from ultimate_mcp_server.core import get_provider_manager
        from ultimate_mcp_server.services.analytics import get_analytics_service
        from ultimate_mcp_server.tools.optimization import get_optimization_service

        retriever = get_knowledge_base_retriever()
        provider_manager = get_provider_manager()
        optimization_service = get_optimization_service()
        analytics_service = get_analytics_service()

        _rag_service = RAGEngine(
            retriever=retriever,
            provider_manager=provider_manager,
            optimization_service=optimization_service,
            analytics_service=analytics_service,
        )

    return _rag_service
