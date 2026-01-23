"""Services for Ultimate MCP Server."""

# Example: Only keep get_analytics_service if get_rag_engine needs it directly
from ultimate_mcp_server.services.analytics import get_analytics_service

# __all__ should only export symbols defined *in this file* or truly essential high-level interfaces
# Avoid re-exporting everything from submodules.
__all__ = [
    "get_analytics_service",  # Only keep if it's fundamental/used here
    "get_rag_engine",  # get_rag_engine is defined below
]

_rag_engine = None


def get_rag_engine():
    """Get or create a RAG engine instance.

    Returns:
        RAGEngine: RAG engine instance
    """
    global _rag_engine

    if _rag_engine is None:
        # Import dependencies *inside* the function to avoid top-level cycles
        from ultimate_mcp_server.core import (
            get_provider_manager,  # Assuming this doesn't import services
        )
        from ultimate_mcp_server.services.knowledge_base import (
            get_knowledge_base_retriever,  # Import KB retriever here
        )
        from ultimate_mcp_server.services.knowledge_base.rag_engine import RAGEngine

        # Assuming OptimizationTools doesn't create cycles with services
        # This might need further investigation if OptimizationTools imports services
        from ultimate_mcp_server.tools.optimization import get_optimization_service

        # analytics_service is already imported at top-level
        # retriever = get_knowledge_base_retriever()
        # provider_manager = get_provider_manager()
        # optimization_service = get_optimization_service()
        # analytics_service = get_analytics_service()

        _rag_engine = RAGEngine(
            retriever=get_knowledge_base_retriever(),
            provider_manager=get_provider_manager(),
            optimization_service=get_optimization_service(),
            analytics_service=get_analytics_service(),  # Imported at top
        )

    return _rag_engine
