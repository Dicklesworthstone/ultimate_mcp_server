"""RAG engine for retrieval-augmented generation."""

import time
from typing import Any, Dict, List, Optional, Set

from ultimate_mcp_server.core.models.requests import CompletionRequest
from ultimate_mcp_server.services.cache import get_cache_service
from ultimate_mcp_server.services.knowledge_base.feedback import get_rag_feedback_service
from ultimate_mcp_server.services.knowledge_base.retriever import KnowledgeBaseRetriever
from ultimate_mcp_server.services.knowledge_base.utils import (
    extract_keywords,
    generate_token_estimate,
)
from ultimate_mcp_server.services.prompts import get_prompt_service
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


# Default RAG prompt templates
DEFAULT_RAG_TEMPLATES = {
    "rag_default": """Answer the question based only on the following context:

{context}

Question: {query}

Answer:""",
    "rag_with_sources": """Answer the question based only on the following context:

{context}

Question: {query}

Provide your answer along with the source document IDs in [brackets] for each piece of information:""",
    "rag_summarize": """Summarize the following context information:

{context}

Summary:""",
    "rag_analysis": """Analyze the following information and provide key insights:

{context}

Query: {query}

Analysis:""",
}


class RAGEngine:
    """Engine for retrieval-augmented generation."""

    def __init__(
        self,
        retriever: KnowledgeBaseRetriever,
        provider_manager,
        optimization_service=None,
        analytics_service=None,
    ):
        """Initialize the RAG engine.

        Args:
            retriever: Knowledge base retriever
            provider_manager: Provider manager for LLM access
            optimization_service: Optional optimization service for model selection
            analytics_service: Optional analytics service for tracking
        """
        self.retriever = retriever
        self.provider_manager = provider_manager
        self.optimization_service = optimization_service
        self.analytics_service = analytics_service

        # Initialize prompt service
        self.prompt_service = get_prompt_service()

        # Initialize feedback service
        self.feedback_service = get_rag_feedback_service()

        # Initialize cache service
        self.cache_service = get_cache_service()

        # Register RAG templates
        for template_name, template_text in DEFAULT_RAG_TEMPLATES.items():
            self.prompt_service.register_template(template_name, template_text)

        logger.info("RAG engine initialized", extra={"emoji_key": "success"})

    async def _select_optimal_model(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal model for a RAG task.

        Args:
            task_info: Task information

        Returns:
            Model selection
        """
        if self.optimization_service:
            try:
                return await self.optimization_service.get_optimal_model(task_info)
            except Exception as e:
                logger.error(
                    f"Error selecting optimal model: {str(e)}", extra={"emoji_key": "error"}
                )

        # Fallback to default models for RAG
        return {"provider": "openai", "model": "gpt-4.1-mini"}

    async def _track_rag_metrics(
        self, knowledge_base: str, query: str, provider: str, model: str, metrics: Dict[str, Any]
    ) -> None:
        """Track RAG operation metrics.

        Args:
            knowledge_base: Knowledge base name
            query: Query text
            provider: Provider name
            model: Model name
            metrics: Operation metrics
        """
        if not self.analytics_service:
            return

        try:
            await self.analytics_service.track_operation(
                operation_type="rag",
                provider=provider,
                model=model,
                input_tokens=metrics.get("input_tokens", 0),
                output_tokens=metrics.get("output_tokens", 0),
                total_tokens=metrics.get("total_tokens", 0),
                cost=metrics.get("cost", 0.0),
                duration=metrics.get("total_time", 0.0),
                metadata={
                    "knowledge_base": knowledge_base,
                    "query": query,
                    "retrieval_count": metrics.get("retrieval_count", 0),
                    "retrieval_time": metrics.get("retrieval_time", 0.0),
                    "generation_time": metrics.get("generation_time", 0.0),
                },
            )
        except Exception as e:
            logger.error(f"Error tracking RAG metrics: {str(e)}", extra={"emoji_key": "error"})

    def _format_context(self, results: List[Dict[str, Any]], include_metadata: bool = True) -> str:
        """Format retrieval results into context.

        Args:
            results: List of retrieval results
            include_metadata: Whether to include metadata

        Returns:
            Formatted context
        """
        context_parts = []

        for i, result in enumerate(results):
            # Format metadata if included
            metadata_str = ""
            if include_metadata and result.get("metadata"):
                # Extract relevant metadata fields
                metadata_fields = []
                for key in ["title", "source", "author", "date", "source_id", "potential_title"]:
                    if key in result["metadata"]:
                        metadata_fields.append(f"{key}: {result['metadata'][key]}")

                if metadata_fields:
                    metadata_str = " | ".join(metadata_fields)
                    metadata_str = f"[{metadata_str}]\n"

            # Add document with index
            context_parts.append(
                f"Document {i + 1} [ID: {result['id']}]:\n{metadata_str}{result['document']}"
            )

        return "\n\n".join(context_parts)

    async def _adjust_retrieval_params(
        self, query: str, knowledge_base_name: str
    ) -> Dict[str, Any]:
        """Dynamically adjust retrieval parameters based on query complexity.

        Args:
            query: Query text
            knowledge_base_name: Knowledge base name

        Returns:
            Adjusted parameters
        """
        # Analyze query complexity
        query_length = len(query.split())
        query_keywords = extract_keywords(query)

        # Base parameters
        params = {
            "top_k": 5,
            "retrieval_method": "vector",
            "min_score": 0.6,
            "search_params": {"search_ef": 100},
        }

        # Adjust based on query length
        if query_length > 30:  # Complex query
            params["top_k"] = 8
            params["search_params"]["search_ef"] = 200
            params["retrieval_method"] = "hybrid"
        elif query_length < 5:  # Very short query
            params["top_k"] = 10  # Get more results for short queries
            params["min_score"] = 0.5  # Lower threshold

        # Check if similar queries exist
        similar_queries = await self.feedback_service.get_similar_queries(
            knowledge_base_name=knowledge_base_name, query=query, top_k=1, threshold=0.85
        )

        # If we have similar past queries, use their parameters
        if similar_queries:
            params["retrieval_method"] = "hybrid"  # Hybrid works well for repeat queries

        # Add keywords
        params["additional_keywords"] = query_keywords

        return params

    async def _analyze_used_documents(self, answer: str, results: List[Dict[str, Any]]) -> Set[str]:
        """Analyze which documents were used in the answer.

        Args:
            answer: Generated answer
            results: List of retrieval results

        Returns:
            Set of document IDs used in the answer
        """
        used_ids = set()

        # Check for explicit mentions of document IDs
        for result in results:
            doc_id = result["id"]
            if f"[ID: {doc_id}]" in answer or f"[{doc_id}]" in answer:
                used_ids.add(doc_id)

        # Check content overlap (crude approximation)
        for result in results:
            if result["id"] in used_ids:
                continue

            # Check for significant phrases from document in answer
            doc_keywords = extract_keywords(result["document"], max_keywords=5)
            matched_keywords = sum(1 for kw in doc_keywords if kw in answer.lower())

            # If multiple keywords match, consider document used
            if matched_keywords >= 2:
                used_ids.add(result["id"])

        return used_ids

    async def _check_cached_response(
        self, knowledge_base_name: str, query: str
    ) -> Optional[Dict[str, Any]]:
        """Check for cached RAG response.

        Args:
            knowledge_base_name: Knowledge base name
            query: Query text

        Returns:
            Cached response or None
        """
        if not self.cache_service:
            return None

        cache_key = f"rag_{knowledge_base_name}_{query}"

        try:
            cached = await self.cache_service.get(cache_key)
            if cached:
                logger.info(
                    f"Using cached RAG response for query in '{knowledge_base_name}'",
                    extra={"emoji_key": "cache"},
                )
                return cached
        except Exception as e:
            logger.error(f"Error checking cache: {str(e)}", extra={"emoji_key": "error"})

        return None

    async def _cache_response(
        self, knowledge_base_name: str, query: str, response: Dict[str, Any]
    ) -> None:
        """Cache RAG response.

        Args:
            knowledge_base_name: Knowledge base name
            query: Query text
            response: Response to cache
        """
        if not self.cache_service:
            return

        cache_key = f"rag_{knowledge_base_name}_{query}"

        try:
            # Cache for 1 day
            await self.cache_service.set(cache_key, response, ttl=86400)
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}", extra={"emoji_key": "error"})

    async def generate_with_rag(
        self,
        knowledge_base_name: str,
        query: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        template: str = "rag_default",
        max_tokens: int = 1000,
        temperature: float = 0.3,
        top_k: Optional[int] = None,
        retrieval_method: Optional[str] = None,
        min_score: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_sources: bool = True,
        use_cache: bool = True,
        apply_feedback: bool = True,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a response using RAG.

        Args:
            knowledge_base_name: Knowledge base name
            query: Query text
            provider: Provider name (auto-selected if None)
            model: Model name (auto-selected if None)
            template: RAG prompt template name
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            top_k: Number of documents to retrieve (auto-adjusted if None)
            retrieval_method: Retrieval method (vector, hybrid)
            min_score: Minimum similarity score
            metadata_filter: Optional metadata filter
            include_metadata: Whether to include metadata in context
            include_sources: Whether to include sources in response
            use_cache: Whether to use cached responses
            apply_feedback: Whether to apply feedback adjustments
            search_params: Optional ChromaDB search parameters

        Returns:
            Generated response with sources and metrics
        """
        start_time = time.time()
        operation_metrics = {}

        # Check cache first if enabled
        if use_cache:
            cached_response = await self._check_cached_response(knowledge_base_name, query)
            if cached_response:
                return cached_response

        # Auto-select model if not specified
        if not provider or not model:
            # Determine task complexity based on query
            task_complexity = "medium"
            if len(query) > 100:
                task_complexity = "high"
            elif len(query) < 30:
                task_complexity = "low"

            # Get optimal model
            model_selection = await self._select_optimal_model(
                {
                    "task_type": "rag_completion",
                    "complexity": task_complexity,
                    "query_length": len(query),
                }
            )

            provider = provider or model_selection["provider"]
            model = model or model_selection["model"]

        # Dynamically adjust retrieval parameters if not specified
        if top_k is None or retrieval_method is None or min_score is None:
            adjusted_params = await self._adjust_retrieval_params(query, knowledge_base_name)

            # Use specified parameters or adjusted ones
            top_k = top_k or adjusted_params["top_k"]
            retrieval_method = retrieval_method or adjusted_params["retrieval_method"]
            min_score = min_score or adjusted_params["min_score"]
            search_params = search_params or adjusted_params.get("search_params")
            additional_keywords = adjusted_params.get("additional_keywords")
        else:
            additional_keywords = None

        # Retrieve context
        retrieval_start = time.time()

        if retrieval_method == "hybrid":
            # Use hybrid search
            retrieval_result = await self.retriever.retrieve_hybrid(
                knowledge_base_name=knowledge_base_name,
                query=query,
                top_k=top_k,
                min_score=min_score,
                metadata_filter=metadata_filter,
                additional_keywords=additional_keywords,
                apply_feedback=apply_feedback,
                search_params=search_params,
            )
        else:
            # Use standard vector search
            retrieval_result = await self.retriever.retrieve(
                knowledge_base_name=knowledge_base_name,
                query=query,
                top_k=top_k,
                min_score=min_score,
                metadata_filter=metadata_filter,
                content_filter=None,  # No content filter for vector-only search
                apply_feedback=apply_feedback,
                search_params=search_params,
            )

        retrieval_time = time.time() - retrieval_start
        operation_metrics["retrieval_time"] = retrieval_time

        # Check if retrieval was successful
        if retrieval_result.get("status") != "success" or not retrieval_result.get("results"):
            logger.warning(
                f"No relevant documents found for query in knowledge base '{knowledge_base_name}'",
                extra={"emoji_key": "warning"},
            )

            # Return error response
            error_response = {
                "status": "no_results",
                "message": "No relevant documents found for query",
                "query": query,
                "retrieval_time": retrieval_time,
                "total_time": time.time() - start_time,
            }

            # Cache error response if enabled
            if use_cache:
                await self._cache_response(knowledge_base_name, query, error_response)

            return error_response

        # Format context from retrieval results
        context = self._format_context(
            retrieval_result["results"], include_metadata=include_metadata
        )

        # Get prompt template
        template_text = self.prompt_service.get_template(template)
        if not template_text:
            # Fallback to default template
            template_text = DEFAULT_RAG_TEMPLATES["rag_default"]

        # Format prompt with template
        rag_prompt = template_text.format(context=context, query=query)

        # Calculate token estimates
        input_tokens = generate_token_estimate(rag_prompt)
        operation_metrics["context_tokens"] = generate_token_estimate(context)
        operation_metrics["input_tokens"] = input_tokens
        operation_metrics["retrieval_count"] = len(retrieval_result["results"])

        # Generate completion
        generation_start = time.time()

        provider_service = self.provider_manager.get_provider(provider)
        completion_request = CompletionRequest(
            prompt=rag_prompt, model=model, max_tokens=max_tokens, temperature=temperature
        )

        completion_result = await provider_service.generate_completion(request=completion_request)

        generation_time = time.time() - generation_start
        operation_metrics["generation_time"] = generation_time

        # Extract completion and metrics
        completion = completion_result.get("completion", "")
        operation_metrics["output_tokens"] = completion_result.get("output_tokens", 0)
        operation_metrics["total_tokens"] = completion_result.get("total_tokens", 0)
        operation_metrics["cost"] = completion_result.get("cost", 0.0)
        operation_metrics["total_time"] = time.time() - start_time

        # Prepare sources if requested
        sources = []
        if include_sources:
            for result in retrieval_result["results"]:
                # Include limited context for each source
                doc_preview = result["document"]
                if len(doc_preview) > 100:
                    doc_preview = doc_preview[:100] + "..."

                sources.append(
                    {
                        "id": result["id"],
                        "document": doc_preview,
                        "score": result["score"],
                        "metadata": result.get("metadata", {}),
                    }
                )

        # Analyze which documents were used in the answer
        used_doc_ids = await self._analyze_used_documents(completion, retrieval_result["results"])

        # Record feedback
        if apply_feedback:
            await self.retriever.record_feedback(
                knowledge_base_name=knowledge_base_name,
                query=query,
                retrieved_documents=retrieval_result["results"],
                used_document_ids=list(used_doc_ids),
            )

        # Track metrics
        await self._track_rag_metrics(
            knowledge_base=knowledge_base_name,
            query=query,
            provider=provider,
            model=model,
            metrics=operation_metrics,
        )

        logger.info(
            f"Generated RAG response using {provider}/{model} in {operation_metrics['total_time']:.2f}s",
            extra={"emoji_key": "success"},
        )

        # Create response
        response = {
            "status": "success",
            "query": query,
            "answer": completion,
            "sources": sources,
            "knowledge_base": knowledge_base_name,
            "provider": provider,
            "model": model,
            "used_document_ids": list(used_doc_ids),
            "metrics": operation_metrics,
        }

        # Cache response if enabled
        if use_cache:
            await self._cache_response(knowledge_base_name, query, response)

        return response
