"""Feedback and adaptive learning service for RAG."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ultimate_mcp_server.services.vector import get_embedding_service
from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


class RAGFeedbackService:
    """Service for collecting and utilizing feedback for RAG."""

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the feedback service.

        Args:
            storage_dir: Directory to store feedback data
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path("storage") / "rag_feedback"

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Feedback data structure
        self.document_feedback = {}  # Knowledge base -> document_id -> feedback
        self.query_feedback = {}  # Knowledge base -> query -> feedback
        self.retrieval_stats = {}  # Knowledge base -> document_id -> usage stats

        # Load existing feedback data
        self._load_feedback_data()

        # Get embedding service for similarity calculations
        self.embedding_service = get_embedding_service()

        # Improvement factors (weights for feedback)
        self.feedback_weights = {
            "thumbs_up": 0.1,  # Positive explicit feedback
            "thumbs_down": -0.15,  # Negative explicit feedback
            "used_in_answer": 0.05,  # Document was used in the answer
            "not_used": -0.02,  # Document was retrieved but not used
            "time_decay": 0.001,  # Decay factor for time
        }

        logger.info("RAG feedback service initialized", extra={"emoji_key": "success"})

    def _get_feedback_file(self, kb_name: str) -> Path:
        """Get path to feedback file for a knowledge base.

        Args:
            kb_name: Knowledge base name

        Returns:
            Path to feedback file
        """
        return self.storage_dir / f"{kb_name}_feedback.json"

    def _load_feedback_data(self):
        """Load feedback data from storage."""
        try:
            # Load all feedback files
            for file_path in self.storage_dir.glob("*_feedback.json"):
                try:
                    kb_name = file_path.stem.replace("_feedback", "")

                    with open(file_path, "r") as f:
                        data = json.load(f)

                    self.document_feedback[kb_name] = data.get("document_feedback", {})
                    self.query_feedback[kb_name] = data.get("query_feedback", {})
                    self.retrieval_stats[kb_name] = data.get("retrieval_stats", {})

                    logger.debug(
                        f"Loaded feedback data for knowledge base '{kb_name}'",
                        extra={"emoji_key": "cache"},
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading feedback data from {file_path}: {str(e)}",
                        extra={"emoji_key": "error"},
                    )
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}", extra={"emoji_key": "error"})

    def _save_feedback_data(self, kb_name: str):
        """Save feedback data to storage.

        Args:
            kb_name: Knowledge base name
        """
        try:
            file_path = self._get_feedback_file(kb_name)

            # Prepare data
            data = {
                "document_feedback": self.document_feedback.get(kb_name, {}),
                "query_feedback": self.query_feedback.get(kb_name, {}),
                "retrieval_stats": self.retrieval_stats.get(kb_name, {}),
                "last_updated": time.time(),
            }

            # Save to file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(
                f"Saved feedback data for knowledge base '{kb_name}'", extra={"emoji_key": "cache"}
            )
        except Exception as e:
            logger.error(
                f"Error saving feedback data for knowledge base '{kb_name}': {str(e)}",
                extra={"emoji_key": "error"},
            )

    async def record_retrieval_feedback(
        self,
        knowledge_base_name: str,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        used_document_ids: Optional[Set[str]] = None,
        explicit_feedback: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Record feedback about retrieval results.

        Args:
            knowledge_base_name: Knowledge base name
            query: Query text
            retrieved_documents: List of retrieved documents with IDs and scores
            used_document_ids: Set of document IDs that were used in the answer
            explicit_feedback: Optional explicit feedback (document_id -> feedback)

        Returns:
            Feedback recording result
        """
        # Initialize structures if needed
        if knowledge_base_name not in self.document_feedback:
            self.document_feedback[knowledge_base_name] = {}

        if knowledge_base_name not in self.query_feedback:
            self.query_feedback[knowledge_base_name] = {}

        if knowledge_base_name not in self.retrieval_stats:
            self.retrieval_stats[knowledge_base_name] = {}

        # Set default for used_document_ids
        if used_document_ids is None:
            used_document_ids = set()

        # Set default for explicit_feedback
        if explicit_feedback is None:
            explicit_feedback = {}

        # Record query
        query_hash = query[:100]  # Use prefix as key

        if query_hash not in self.query_feedback[knowledge_base_name]:
            self.query_feedback[knowledge_base_name][query_hash] = {
                "query": query,
                "count": 0,
                "last_used": time.time(),
                "retrieved_docs": [],
            }

        # Update query stats
        self.query_feedback[knowledge_base_name][query_hash]["count"] += 1
        self.query_feedback[knowledge_base_name][query_hash]["last_used"] = time.time()

        # Process each retrieved document
        for doc in retrieved_documents:
            doc_id = doc["id"]

            # Initialize document feedback if not exists
            if doc_id not in self.document_feedback[knowledge_base_name]:
                self.document_feedback[knowledge_base_name][doc_id] = {
                    "relevance_adjustment": 0.0,
                    "positive_feedback_count": 0,
                    "negative_feedback_count": 0,
                    "used_count": 0,
                    "retrieved_count": 0,
                    "last_used": time.time(),
                }

            # Update document stats
            doc_feedback = self.document_feedback[knowledge_base_name][doc_id]
            doc_feedback["retrieved_count"] += 1
            doc_feedback["last_used"] = time.time()

            # Record if document was used in the answer
            if doc_id in used_document_ids:
                doc_feedback["used_count"] += 1
                doc_feedback["relevance_adjustment"] += self.feedback_weights["used_in_answer"]
            else:
                doc_feedback["relevance_adjustment"] += self.feedback_weights["not_used"]

            # Apply explicit feedback if provided
            if doc_id in explicit_feedback:
                feedback_type = explicit_feedback[doc_id]

                if feedback_type == "thumbs_up":
                    doc_feedback["positive_feedback_count"] += 1
                    doc_feedback["relevance_adjustment"] += self.feedback_weights["thumbs_up"]
                elif feedback_type == "thumbs_down":
                    doc_feedback["negative_feedback_count"] += 1
                    doc_feedback["relevance_adjustment"] += self.feedback_weights["thumbs_down"]

            # Keep adjustment within bounds
            doc_feedback["relevance_adjustment"] = max(
                -0.5, min(0.5, doc_feedback["relevance_adjustment"])
            )

            # Record document in query feedback
            if doc_id not in self.query_feedback[knowledge_base_name][query_hash]["retrieved_docs"]:
                self.query_feedback[knowledge_base_name][query_hash]["retrieved_docs"].append(
                    doc_id
                )

        # Save feedback data
        self._save_feedback_data(knowledge_base_name)

        logger.info(
            f"Recorded feedback for {len(retrieved_documents)} documents in knowledge base '{knowledge_base_name}'",
            extra={"emoji_key": "success"},
        )

        return {
            "status": "success",
            "knowledge_base": knowledge_base_name,
            "query": query,
            "documents_count": len(retrieved_documents),
            "used_documents_count": len(used_document_ids),
        }

    async def get_document_boost(self, knowledge_base_name: str, document_id: str) -> float:
        """Get relevance boost for a document based on feedback.

        Args:
            knowledge_base_name: Knowledge base name
            document_id: Document ID

        Returns:
            Relevance boost factor
        """
        if (
            knowledge_base_name not in self.document_feedback
            or document_id not in self.document_feedback[knowledge_base_name]
        ):
            return 0.0

        # Get document feedback
        doc_feedback = self.document_feedback[knowledge_base_name][document_id]

        # Calculate time decay
        time_since_last_use = time.time() - doc_feedback.get("last_used", 0)
        time_decay = min(1.0, time_since_last_use / (86400 * 30))  # 30 days max decay

        # Apply decay to adjustment
        adjustment = doc_feedback["relevance_adjustment"] * (
            1.0 - time_decay * self.feedback_weights["time_decay"]
        )

        return adjustment

    async def get_similar_queries(
        self, knowledge_base_name: str, query: str, top_k: int = 3, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find similar previous queries.

        Args:
            knowledge_base_name: Knowledge base name
            query: Query text
            top_k: Number of similar queries to return
            threshold: Similarity threshold

        Returns:
            List of similar queries with metadata
        """
        if knowledge_base_name not in self.query_feedback:
            return []

        query_feedback = self.query_feedback[knowledge_base_name]

        if not query_feedback:
            return []

        # Get embedding for the query
        query_embedding = await self.embedding_service.get_embedding(query)

        # Calculate similarity with all previous queries
        similarities = []

        for _query_hash, data in query_feedback.items():
            try:
                prev_query = data["query"]
                prev_embedding = await self.embedding_service.get_embedding(prev_query)

                # Calculate cosine similarity
                similarity = np.dot(query_embedding, prev_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(prev_embedding)
                )

                if similarity >= threshold:
                    similarities.append(
                        {
                            "query": prev_query,
                            "similarity": float(similarity),
                            "count": data["count"],
                            "last_used": data["last_used"],
                            "retrieved_docs": data["retrieved_docs"],
                        }
                    )
            except Exception as e:
                logger.error(
                    f"Error calculating similarity for query: {str(e)}",
                    extra={"emoji_key": "error"},
                )

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    async def apply_feedback_adjustments(
        self, knowledge_base_name: str, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Apply feedback-based adjustments to retrieval results.

        Args:
            knowledge_base_name: Knowledge base name
            results: List of retrieval results
            query: Query text

        Returns:
            Adjusted retrieval results
        """
        # Check if we have feedback data
        if knowledge_base_name not in self.document_feedback:
            return results

        # Get similar queries
        similar_queries = await self.get_similar_queries(
            knowledge_base_name=knowledge_base_name, query=query, top_k=3, threshold=0.8
        )

        # Collect document IDs from similar queries
        similar_doc_ids = set()
        for sq in similar_queries:
            similar_doc_ids.update(sq["retrieved_docs"])

        # Apply boosts to results
        adjusted_results = []

        for result in results:
            doc_id = result["id"]
            score = result["score"]

            # Apply document-specific boost
            doc_boost = await self.get_document_boost(knowledge_base_name, doc_id)

            # Apply boost for documents from similar queries
            similar_query_boost = 0.05 if doc_id in similar_doc_ids else 0.0

            # Calculate final score with boosts
            adjusted_score = min(1.0, score + doc_boost + similar_query_boost)

            # Update result
            adjusted_result = result.copy()
            adjusted_result["original_score"] = score
            adjusted_result["feedback_boost"] = doc_boost
            adjusted_result["similar_query_boost"] = similar_query_boost
            adjusted_result["score"] = adjusted_score

            adjusted_results.append(adjusted_result)

        # Sort by adjusted score
        adjusted_results.sort(key=lambda x: x["score"], reverse=True)

        return adjusted_results


# Singleton instance
_rag_feedback_service = None


def get_rag_feedback_service() -> RAGFeedbackService:
    """Get or create a RAG feedback service instance.

    Returns:
        RAGFeedbackService: RAG feedback service instance
    """
    global _rag_feedback_service

    if _rag_feedback_service is None:
        _rag_feedback_service = RAGFeedbackService()

    return _rag_feedback_service
