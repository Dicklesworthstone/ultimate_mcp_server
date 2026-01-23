"""Cache strategy implementations."""

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ultimate_mcp_server.utils import get_logger

logger = get_logger(__name__)


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""

    @abstractmethod
    def generate_key(self, request: Dict[str, Any]) -> str:
        """Generate a cache key for the request.

        Args:
            request: Request parameters

        Returns:
            Cache key
        """
        pass

    @abstractmethod
    def should_cache(self, request: Dict[str, Any], response: Any) -> bool:
        """Determine if a response should be cached.

        Args:
            request: Request parameters
            response: Response data

        Returns:
            True if the response should be cached
        """
        pass

    def get_ttl(self, request: Dict[str, Any], response: Any) -> Optional[int]:
        """Get the TTL (time-to-live) for a cached response.

        Args:
            request: Request parameters
            response: Response data

        Returns:
            TTL in seconds or None to use default
        """
        return None


class ExactMatchStrategy(CacheStrategy):
    """Strategy for exact matching of requests."""

    def generate_key(self, request: Dict[str, Any]) -> str:
        """Generate an exact match cache key.

        Args:
            request: Request parameters

        Returns:
            Cache key based on normalized parameters
        """
        # Remove non-deterministic fields
        clean_request = self._clean_request(request)

        # Serialize and hash
        json_str = json.dumps(clean_request, sort_keys=True)
        return f"exact:{hashlib.sha256(json_str.encode('utf-8')).hexdigest()}"

    def should_cache(self, request: Dict[str, Any], response: Any) -> bool:
        """Determine if a response should be cached based on request type.

        Args:
            request: Request parameters
            response: Response data

        Returns:
            True if the response should be cached
        """
        # Don't cache if explicitly disabled
        if request.get("cache", True) is False:
            return False

        # Don't cache streaming responses
        if request.get("stream", False):
            return False

        # Don't cache high temperature responses (too random)
        if request.get("temperature", 0.7) > 0.9:
            return False

        return True

    def get_ttl(self, request: Dict[str, Any], response: Any) -> Optional[int]:
        """Get TTL based on request types.

        Args:
            request: Request parameters
            response: Response data

        Returns:
            TTL in seconds or None to use default
        """
        # Use custom TTL if specified
        if "cache_ttl" in request:
            return request["cache_ttl"]

        # Base TTL on content length - longer content gets longer TTL
        if hasattr(response, "text") and isinstance(response.text, str):
            content_length = len(response.text)

            # Simplified TTL scaling
            if content_length > 10000:
                return 7 * 24 * 60 * 60  # 1 week for long responses
            elif content_length > 1000:
                return 3 * 24 * 60 * 60  # 3 days for medium responses

        return None  # Use default TTL

    def _clean_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-deterministic fields from request.

        Args:
            request: Original request

        Returns:
            Cleaned request for caching
        """
        # Make a copy to avoid modifying the original
        clean = request.copy()

        # Remove non-deterministic fields
        for field in [
            "request_id",
            "timestamp",
            "session_id",
            "trace_id",
            "user_id",
            "cache",
            "cache_ttl",
        ]:
            clean.pop(field, None)

        return clean


class SemanticMatchStrategy(CacheStrategy):
    """Strategy for semantic matching of requests."""

    def __init__(self, similarity_threshold: float = 0.95):
        """Initialize semantic matching strategy.

        Args:
            similarity_threshold: Threshold for semantic similarity (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.exact_strategy = ExactMatchStrategy()

    def generate_key(self, request: Dict[str, Any]) -> str:
        """Generate both exact and semantic keys.

        Args:
            request: Request parameters

        Returns:
            Primary cache key (always the exact match key)
        """
        # Always use the exact match key as the primary key
        return self.exact_strategy.generate_key(request)

    def generate_semantic_key(self, request: Dict[str, Any]) -> Optional[str]:
        """Generate a semantic fingerprint for the request.

        Args:
            request: Request parameters

        Returns:
            Semantic key or None if request doesn't support semantic matching
        """
        # Extract the prompt or relevant text
        text = self._extract_text(request)
        if not text:
            return None

        # Normalize text
        text = self._normalize_text(text)

        # Generate fingerprint based on significant words and structure
        significant_words = self._extract_significant_words(text)

        # Create a fuzzy key
        if significant_words:
            words_key = " ".join(sorted(significant_words))
            return f"semantic:{hashlib.md5(words_key.encode('utf-8')).hexdigest()}"

        return None

    def should_cache(self, request: Dict[str, Any], response: Any) -> bool:
        """Determine if a response should be cached.

        Args:
            request: Request parameters
            response: Response data

        Returns:
            True if the response should be cached
        """
        # Use the same logic as exact matching
        return self.exact_strategy.should_cache(request, response)

    def get_ttl(self, request: Dict[str, Any], response: Any) -> Optional[int]:
        """Get the TTL for semantic matches.

        Args:
            request: Request parameters
            response: Response data

        Returns:
            TTL in seconds (shorter for semantic matches)
        """
        # Get base TTL from exact strategy
        base_ttl = self.exact_strategy.get_ttl(request, response)

        # For semantic matching, use shorter TTL
        if base_ttl is not None:
            return int(base_ttl * 0.5)  # 50% of exact match TTL

        return None

    def _extract_text(self, request: Dict[str, Any]) -> Optional[str]:
        """Extract the relevant text for semantic matching.

        Args:
            request: Request parameters

        Returns:
            Extracted text or None
        """
        # Try to extract prompt text
        if "prompt" in request:
            return request["prompt"]

        # Try to extract from messages
        if "messages" in request and isinstance(request["messages"], list):
            # Extract text from the last user message
            for message in reversed(request["messages"]):
                if message.get("role") == "user" and "content" in message:
                    if isinstance(message["content"], str):
                        return message["content"]
                    elif isinstance(message["content"], list):
                        # Handle content list (multimodal messages)
                        text_parts = []
                        for part in message["content"]:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        return " ".join(text_parts)

        # No suitable text found
        return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for semantic matching.

        Args:
            text: Original text

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        return text

    def _extract_significant_words(self, text: str, max_words: int = 15) -> List[str]:
        """Extract significant words from text.

        Args:
            text: Normalized text
            max_words: Maximum number of words to include

        Returns:
            List of significant words
        """
        # Split into words
        words = text.split()

        # Filter out short words and common stop words
        stop_words = {
            "the",
            "and",
            "a",
            "an",
            "in",
            "to",
            "for",
            "of",
            "with",
            "on",
            "is",
            "are",
            "am",
            "was",
            "were",
            "be",
            "been",
            "being",
            "this",
            "that",
            "these",
            "those",
            "it",
            "they",
            "them",
            "their",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "about",
            "above",
            "after",
            "again",
            "against",
            "all",
            "any",
            "because",
            "before",
            "below",
            "between",
            "both",
            "but",
            "by",
            "down",
            "during",
            "each",
            "few",
            "from",
            "further",
            "here",
            "how",
            "into",
            "more",
            "most",
            "no",
            "nor",
            "not",
            "only",
            "or",
            "other",
            "out",
            "over",
            "own",
            "same",
            "so",
            "than",
            "then",
            "there",
            "through",
            "under",
            "until",
            "up",
            "very",
            "what",
            "when",
            "where",
            "which",
            "while",
            "who",
            "whom",
            "why",
            "you",
            "your",
            "yours",
            "yourself",
            "ourselves",
            "i",
            "me",
            "my",
            "mine",
            "myself",
            "we",
            "us",
            "our",
            "ours",
        }

        # Initial filtering of short words and stopwords
        filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]

        # Calculate word frequencies for TF-IDF like weighing
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Score words based on a combination of:
        # 1. Length (longer words tend to be more significant)
        # 2. Frequency (less common words are often more significant)
        # 3. Position (words at the beginning are often more significant)
        word_scores = {}

        # Normalize position weight based on document length
        position_weight_factor = 100 / max(1, len(words))

        for i, word in enumerate(filtered_words):
            if word in word_scores:
                continue

            # Length score: favor longer words (0.1 to 1.0)
            length_score = min(1.0, 0.1 + (len(word) / 20))

            # Rarity score: favor words that appear less frequently (0.2 to 1.0)
            freq = word_freq[word]
            rarity_score = 1.0 / (0.5 + (freq / 5))
            rarity_score = max(0.2, min(1.0, rarity_score))

            # Position score: favor words that appear earlier (0.2 to 1.0)
            earliest_pos = min([i for i, w in enumerate(filtered_words) if w == word])
            position_score = 1.0 - min(0.8, (earliest_pos * position_weight_factor) / 100)

            # Calculate final score
            final_score = (length_score * 0.3) + (rarity_score * 0.5) + (position_score * 0.2)
            word_scores[word] = final_score

        # Sort words by score and take top max_words
        significant_words = sorted(word_scores.keys(), key=lambda w: word_scores[w], reverse=True)

        # Include at least a few of the most frequent words for context
        top_by_freq = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)[:5]

        # Ensure these frequent words are included in the result
        result = significant_words[:max_words]
        for word in top_by_freq:
            if word not in result and len(result) < max_words:
                result.append(word)

        return result


class TaskBasedStrategy(CacheStrategy):
    """Strategy based on task type."""

    def __init__(self):
        """Initialize task-based strategy."""
        self.exact_strategy = ExactMatchStrategy()
        self.semantic_strategy = SemanticMatchStrategy()

    def generate_key(self, request: Dict[str, Any]) -> str:
        """Generate key based on task type.

        Args:
            request: Request parameters

        Returns:
            Cache key
        """
        task_type = self._detect_task_type(request)

        # Use exact matching for most tasks
        key = self.exact_strategy.generate_key(request)

        # Add task type to the key
        return f"{task_type}:{key}"

    def should_cache(self, request: Dict[str, Any], response: Any) -> bool:
        """Determine if a response should be cached based on task type.

        Args:
            request: Request parameters
            response: Response data

        Returns:
            True if the response should be cached
        """
        task_type = self._detect_task_type(request)

        # Always cache these task types
        always_cache_tasks = {
            "summarization",
            "information_extraction",
            "classification",
            "translation",
            "rewriting",
            "question_answering",
        }

        if task_type in always_cache_tasks:
            return True

        # Use base strategy for other tasks
        return self.exact_strategy.should_cache(request, response)

    def get_ttl(self, request: Dict[str, Any], response: Any) -> Optional[int]:
        """Get TTL based on task type.

        Args:
            request: Request parameters
            response: Response data

        Returns:
            TTL in seconds
        """
        task_type = self._detect_task_type(request)

        # Task-specific TTLs
        ttl_map = {
            "summarization": 30 * 24 * 60 * 60,  # 30 days
            "information_extraction": 14 * 24 * 60 * 60,  # 14 days
            "extraction": 14 * 24 * 60 * 60,  # 14 days - Add explicit mapping for extraction
            "classification": 30 * 24 * 60 * 60,  # 30 days
            "translation": 60 * 24 * 60 * 60,  # 60 days
            "creative_writing": 1 * 24 * 60 * 60,  # 1 day
            "chat": 1 * 24 * 60 * 60,  # 1 day
        }

        if task_type in ttl_map:
            return ttl_map[task_type]

        # Default to base strategy
        return self.exact_strategy.get_ttl(request, response)

    def _detect_task_type(self, request: Dict[str, Any]) -> str:
        """Detect the task type from the request using multiple techniques.

        This function uses a combination of:
        1. Explicit tags in the request
        2. Request structure analysis
        3. NLP-based content analysis
        4. Model and parameter hints

        Args:
            request: Request parameters

        Returns:
            Task type identifier
        """
        # 1. Check for explicit task type
        if "task_type" in request:
            return request["task_type"]

        # 2. Check for task-specific parameters
        if "format" in request and request["format"] in ["json", "structured", "extraction"]:
            return "information_extraction"

        if "max_tokens" in request and request.get("max_tokens", 0) < 100:
            return "classification"  # Short responses often indicate classification

        # 3. Check system prompt for clues
        system_prompt = None
        if "system" in request:
            system_prompt = request["system"]
        elif "messages" in request:
            for msg in request.get("messages", []):
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                    break

        if system_prompt:
            system_lower = system_prompt.lower()
            # Check system prompt for task indicators
            if any(
                x in system_lower
                for x in ["summarize", "summary", "summarization", "summarize the following"]
            ):
                return "summarization"

            if any(x in system_lower for x in ["extract", "extraction", "identify all", "parse"]):
                return "information_extraction"

            if any(x in system_lower for x in ["classify", "categorize", "determine the type"]):
                return "classification"

            if any(x in system_lower for x in ["translate", "translation", "convert to"]):
                return "translation"

            if any(
                x in system_lower
                for x in ["creative", "write a story", "compose", "generate a poem"]
            ):
                return "creative_writing"

            if any(x in system_lower for x in ["reasoning", "solve", "think step by step"]):
                return "reasoning"

            if any(x in system_lower for x in ["chat", "conversation", "assistant", "helpful"]):
                return "chat"

        # 4. Extract text for content analysis
        text = self.semantic_strategy._extract_text(request)
        if not text:
            return "unknown"

        # 5. Sophisticated content analysis
        import re

        # Normalize text
        text_lower = text.lower()

        # Task-specific pattern matching
        task_patterns = {
            "summarization": [
                r"\bsummarize\b",
                r"\bsummary\b",
                r"\btldr\b",
                r"\bcondense\b",
                r"(provide|give|create).{1,20}(summary|overview)",
                r"(summarize|summarise).{1,30}(text|document|paragraph|content|article)",
                r"(key|main|important).{1,20}(points|ideas|concepts)",
            ],
            "information_extraction": [
                r"\bextract\b",
                r"\bidentify\b",
                r"\bfind all\b",
                r"\blist the\b",
                r"(extract|pull out|identify).{1,30}(information|data|details)",
                r"(list|enumerate).{1,20}(all|the)",
                r"(find|extract).{1,30}(names|entities|locations|dates)",
            ],
            "classification": [
                r"\bclassify\b",
                r"\bcategorize\b",
                r"\bgroup\b",
                r"\blabel\b",
                r"what (type|kind|category|class)",
                r"(determine|identify).{1,20}(type|class|category)",
                r"(which|what).{1,20}(category|group|type|class)",
            ],
            "translation": [
                r"\btranslate\b",
                r"\btranslation\b",
                r"(translate|convert).{1,30}(into|to|from).{1,20}(language|english|spanish|french)",
                r"(in|into).{1,10}(spanish|french|german|italian|japanese|chinese|korean)",
            ],
            "creative_writing": [
                r"\bwrite\b",
                r"\bcreate\b",
                r"\bgenerate\b",
                r"\bcompose\b",
                r"(write|create|generate|compose).{1,30}(story|poem|essay|article|blog post)",
                r"(creative|fiction|imaginative).{1,20}(writing|text|content)",
                r"(story|narrative|tale|fiction)",
            ],
            "question_answering": [
                r"(why|how|what|who|where|when).{1,30}\?",
                r"(explain|describe|define).{1,40}",
                r"(question|answer|respond)",
                r"(can you|could you|please).{1,30}(tell me|explain|describe)",
            ],
            "reasoning": [
                r"(solve|calculate|compute|reason|deduce)",
                r"(step by step|detailed|reasoning|rationale)",
                r"(problem|puzzle|challenge|riddle|question)",
                r"(math|mathematical|logic|logical)",
            ],
            "coding": [
                r"(code|function|program|script|algorithm)",
                r"(write|create|generate|implement).{1,30}(code|function|class|method)",
                r"(python|javascript|java|c\+\+|ruby|go|rust|typescript)",
            ],
            "chat": [
                r"(chat|conversation|discuss|talk)",
                r"(assist|help).{1,20}(me|with|in)",
                r"(you are|as a|you're).{1,20}(assistant|helper)",
            ],
        }

        # Score each task type
        task_scores = {}

        for task, patterns in task_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 2  # Each match adds 2 points

                # Award bonus point for match in the first 50 chars (likely the main request)
                if re.search(pattern, text_lower[:50]):
                    score += 3

            # Check for indicators in the first 100 characters (usually the intent)
            first_100 = text_lower[:100]
            if any(re.search(pattern, first_100) for pattern in patterns):
                score += 5

            task_scores[task] = score

        # 6. Check for additional structural clues

        # If JSON output requested, likely extraction
        if "json" in text_lower or "structured" in text_lower:
            task_scores["information_extraction"] += 5

        # If it contains code blocks or technical terms, likely coding
        if "```" in text or any(
            lang in text_lower for lang in ["python", "javascript", "java", "html", "css"]
        ):
            task_scores["coding"] += 5

        # Check for question mark presence and density
        question_marks = text.count("?")
        if question_marks > 0:
            # Multiple questions indicate question answering
            task_scores["question_answering"] += min(question_marks * 2, 10)

        # 7. Check model hints
        model = request.get("model", "")

        # Some models are specialized for specific tasks
        if "instruct" in model.lower():
            task_scores["question_answering"] += 2

        if "chat" in model.lower():
            task_scores["chat"] += 2

        if "code" in model.lower() or "davinci-code" in model.lower():
            task_scores["coding"] += 5

        # 8. Determine highest scoring task
        if not task_scores:
            return "general"

        # Get task with highest score
        best_task = max(task_scores.items(), key=lambda x: x[1])

        # If score is too low, default to general
        if best_task[1] < 3:
            return "general"

        return best_task[0]


# Factory function
def get_strategy(strategy_name: str) -> CacheStrategy:
    """Get a cache strategy by name.

    Args:
        strategy_name: Strategy name

    Returns:
        CacheStrategy instance

    Raises:
        ValueError: If strategy name is invalid
    """
    strategies = {
        "exact": ExactMatchStrategy(),
        "semantic": SemanticMatchStrategy(),
        "task": TaskBasedStrategy(),
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Invalid cache strategy: {strategy_name}. "
            + f"Valid options: {', '.join(strategies.keys())}"
        )

    return strategies[strategy_name]
