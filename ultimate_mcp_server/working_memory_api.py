"""
Working Memory Dashboard API
Provides real-time working memory management and optimization endpoints for the UMS Explorer.
"""

import asyncio
import difflib
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel


@dataclass
class WorkingMemoryItem:
    """Enhanced memory item with working memory specific metadata."""

    memory_id: str
    content: str
    memory_type: str
    memory_level: str
    importance: int
    confidence: float
    created_at: float
    last_accessed_at: Optional[float]
    access_count: int
    workflow_id: Optional[str]

    # Working memory specific fields
    temperature: float = 0.0  # Activity level (0-100)
    priority: str = "medium"  # critical, high, medium, low
    access_frequency: float = 0.0  # Normalized access frequency
    retention_score: float = 0.0  # How likely to remain in working memory
    added_at: float = 0.0  # When added to working memory


@dataclass
class QualityIssue:
    """Represents a memory quality issue."""

    issue_id: str
    issue_type: str  # duplicate, orphaned, low_quality, stale, corrupted
    severity: str  # critical, high, medium, low
    memory_ids: List[str]
    title: str
    description: str
    recommendation: str
    impact_score: float
    auto_fixable: bool
    estimated_savings: Dict[str, float]  # storage, performance, clarity
    metadata: Dict


@dataclass
class QualityAnalysisResult:
    """Result of memory quality analysis."""

    total_memories: int
    issues_found: int
    duplicates: int
    orphaned: int
    low_quality: int
    stale_memories: int
    corrupted: int
    overall_score: float  # 0-100
    issues: List[QualityIssue]
    recommendations: List[str]
    analysis_time: float


@dataclass
class DuplicateCluster:
    """Group of duplicate or similar memories."""

    cluster_id: str
    memory_ids: List[str]
    similarity_score: float
    primary_memory_id: str  # Best quality memory in cluster
    duplicate_count: int
    content_preview: str
    metadata: Dict


@dataclass
class BulkOperation:
    """Represents a bulk operation on memories."""

    operation_id: str
    operation_type: str  # delete, merge, update, archive
    memory_ids: List[str]
    preview_changes: List[Dict]
    estimated_impact: Dict[str, float]
    reversible: bool
    confirmation_required: bool


@dataclass
class WorkingMemoryStats:
    """Working memory statistics and metrics."""

    active_count: int
    capacity: int
    pressure: float  # 0-100%
    temperature: float  # Average activity level
    focus_score: float  # 0-100%
    efficiency: float  # 0-100%
    avg_retention_time: float
    total_accesses: int
    last_updated: float


@dataclass
class OptimizationSuggestion:
    """Memory optimization suggestion."""

    id: str
    title: str
    description: str
    priority: str  # high, medium, low
    impact: str  # High, Medium, Low
    icon: str
    action: str
    confidence: float = 0.0
    estimated_improvement: Dict[str, float] = None


class WorkingMemoryRequest(BaseModel):
    memory_id: str


class OptimizationRequest(BaseModel):
    suggestion_id: str


class FocusModeRequest(BaseModel):
    mode: str  # normal, deep, creative, analytical, maintenance
    retention_time: Optional[int] = None
    max_working_memory: Optional[int] = None


class QualityAnalysisRequest(BaseModel):
    analysis_type: str = "comprehensive"  # comprehensive, duplicates, orphaned, low_quality
    include_stale: bool = True
    include_low_importance: bool = True
    similarity_threshold: float = 0.85
    stale_threshold_days: int = 30


class BulkOperationRequest(BaseModel):
    operation_type: str  # delete, merge, archive, update
    memory_ids: List[str]
    merge_strategy: Optional[str] = "preserve_highest_importance"  # For merge operations
    target_memory_id: Optional[str] = None  # For merge operations
    update_data: Optional[Dict] = None  # For update operations


class MemoryQualityInspector:
    """Core memory quality analysis and management logic."""

    def __init__(self, db_path: str = "storage/unified_agent_memory.db"):
        self.db_path = db_path

    def get_db_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def calculate_content_hash(self, content: str) -> str:
        """Calculate content hash for duplicate detection."""
        normalized = content.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using difflib."""
        normalized1 = content1.strip().lower()
        normalized2 = content2.strip().lower()

        # Use sequence matcher for similarity
        similarity = difflib.SequenceMatcher(None, normalized1, normalized2).ratio()
        return similarity

    def detect_duplicates(
        self, memories: List[Dict], threshold: float = 0.85
    ) -> List[DuplicateCluster]:
        """Detect duplicate memories using content similarity."""
        clusters = []
        processed_ids = set()

        for i, memory1 in enumerate(memories):
            if memory1["memory_id"] in processed_ids:
                continue

            cluster_memories = [memory1]
            cluster_ids = {memory1["memory_id"]}

            for _j, memory2 in enumerate(memories[i + 1 :], i + 1):
                if memory2["memory_id"] in processed_ids:
                    continue

                similarity = self.calculate_similarity(memory1["content"], memory2["content"])

                if similarity >= threshold:
                    cluster_memories.append(memory2)
                    cluster_ids.add(memory2["memory_id"])

            if len(cluster_memories) > 1:
                # Find the best quality memory (highest importance * confidence)
                primary = max(
                    cluster_memories,
                    key=lambda m: (m.get("importance", 1) * m.get("confidence", 0.5)),
                )

                cluster = DuplicateCluster(
                    cluster_id=f"dup_{memory1['memory_id'][:8]}",
                    memory_ids=list(cluster_ids),
                    similarity_score=max(
                        self.calculate_similarity(memory1["content"], m["content"])
                        for m in cluster_memories[1:]
                    ),
                    primary_memory_id=primary["memory_id"],
                    duplicate_count=len(cluster_memories) - 1,
                    content_preview=memory1["content"][:100] + "..."
                    if len(memory1["content"]) > 100
                    else memory1["content"],
                    metadata={
                        "avg_importance": sum(m.get("importance", 1) for m in cluster_memories)
                        / len(cluster_memories),
                        "avg_confidence": sum(m.get("confidence", 0.5) for m in cluster_memories)
                        / len(cluster_memories),
                        "total_size": sum(len(m["content"]) for m in cluster_memories),
                    },
                )
                clusters.append(cluster)
                processed_ids.update(cluster_ids)

        return clusters

    def detect_orphaned_memories(self, memories: List[Dict]) -> List[Dict]:
        """Detect orphaned memories not connected to any workflow or relationship."""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()

            orphaned = []
            for memory in memories:
                memory_id = memory["memory_id"]

                # Check if memory has workflow association
                has_workflow = memory.get("workflow_id") is not None

                # Check if memory is linked to other memories
                cursor.execute(
                    """
                    SELECT COUNT(*) as count FROM memory_links 
                    WHERE source_memory_id = ? OR target_memory_id = ?
                """,
                    (memory_id, memory_id),
                )

                link_count = cursor.fetchone()["count"]

                # Check if memory is referenced in goals or actions
                cursor.execute(
                    """
                    SELECT COUNT(*) as action_count FROM actions 
                    WHERE memory_id = ? OR input_data LIKE ? OR output_data LIKE ?
                """,
                    (memory_id, f"%{memory_id}%", f"%{memory_id}%"),
                )

                action_refs = cursor.fetchone()["action_count"]

                cursor.execute(
                    """
                    SELECT COUNT(*) as goal_count FROM goals 
                    WHERE memory_id = ? OR description LIKE ?
                """,
                    (memory_id, f"%{memory_id}%"),
                )

                goal_refs = cursor.fetchone()["goal_count"]

                # Memory is orphaned if it has no workflow, no links, and no references
                if not has_workflow and link_count == 0 and action_refs == 0 and goal_refs == 0:
                    orphaned.append(
                        {
                            **memory,
                            "orphan_score": self.calculate_orphan_score(memory),
                            "isolation_level": "complete",
                        }
                    )
                elif link_count == 0 and (action_refs == 0 or goal_refs == 0):
                    orphaned.append(
                        {
                            **memory,
                            "orphan_score": self.calculate_orphan_score(memory),
                            "isolation_level": "partial",
                        }
                    )

            return orphaned

        finally:
            conn.close()

    def calculate_orphan_score(self, memory: Dict) -> float:
        """Calculate how orphaned a memory is (0-100, higher = more orphaned)."""
        score = 50  # Base score

        # Adjust based on importance (lower importance = more likely orphan)
        importance = memory.get("importance", 1)
        score += (5 - importance) * 10

        # Adjust based on confidence (lower confidence = more likely orphan)
        confidence = memory.get("confidence", 0.5)
        score += (0.5 - confidence) * 50

        # Adjust based on age (older = more likely to be orphaned)
        created_at = memory.get("created_at", time.time())
        age_days = (time.time() - created_at) / 86400
        if age_days > 30:
            score += min(20, age_days / 10)

        # Adjust based on access patterns
        access_count = memory.get("access_count", 0)
        if access_count == 0:
            score += 15
        elif access_count < 3:
            score += 10

        return min(100, max(0, score))

    def analyze_memory_quality(self, memory: Dict) -> Dict:
        """Analyze individual memory quality."""
        quality_score = 50  # Base score
        issues = []

        content = memory.get("content", "")
        importance = memory.get("importance", 1)
        confidence = memory.get("confidence", 0.5)

        # Content quality checks
        if len(content) < 10:
            issues.append("Content too short")
            quality_score -= 20
        elif len(content) > 10000:
            issues.append("Content extremely long")
            quality_score -= 10

        # Check for common quality issues
        if content.count("\n") / max(1, len(content)) > 0.1:  # Too many line breaks
            issues.append("Excessive line breaks")
            quality_score -= 5

        if (
            len(set(content.split())) / max(1, len(content.split())) < 0.3
        ):  # Low vocabulary diversity
            issues.append("Low vocabulary diversity")
            quality_score -= 10

        # Importance and confidence checks
        if importance < 3:
            issues.append("Low importance rating")
            quality_score -= 10

        if confidence < 0.3:
            issues.append("Low confidence rating")
            quality_score -= 15

        # Memory type consistency
        memory_type = memory.get("memory_type", "")
        memory_level = memory.get("memory_level", "")

        if not memory_type:
            issues.append("Missing memory type")
            quality_score -= 15

        if not memory_level:
            issues.append("Missing memory level")
            quality_score -= 15

        # Check for encoding issues or corruption
        try:
            content.encode("utf-8").decode("utf-8")
        except UnicodeError:
            issues.append("Encoding corruption detected")
            quality_score -= 25

        # Age and staleness
        created_at = memory.get("created_at", time.time())
        age_days = (time.time() - created_at) / 86400

        if age_days > 90 and memory.get("access_count", 0) == 0:
            issues.append("Stale memory (old and unaccessed)")
            quality_score -= 20

        return {
            "quality_score": max(0, min(100, quality_score)),
            "issues": issues,
            "recommendations": self.generate_quality_recommendations(memory, issues),
        }

    def generate_quality_recommendations(self, memory: Dict, issues: List[str]) -> List[str]:
        """Generate recommendations for improving memory quality."""
        recommendations = []

        if "Content too short" in issues:
            recommendations.append("Consider expanding content with more context or details")

        if "Content extremely long" in issues:
            recommendations.append("Consider breaking into smaller, focused memories")

        if "Low importance rating" in issues:
            recommendations.append("Review and adjust importance rating if memory is valuable")

        if "Low confidence rating" in issues:
            recommendations.append("Verify information accuracy and update confidence")

        if "Missing memory type" in issues:
            recommendations.append("Assign appropriate memory type classification")

        if "Stale memory (old and unaccessed)" in issues:
            recommendations.append("Archive or delete if no longer relevant")

        if "Encoding corruption detected" in issues:
            recommendations.append("Critical: Clean up encoding issues immediately")

        return recommendations

    async def perform_quality_analysis(
        self, request: QualityAnalysisRequest
    ) -> QualityAnalysisResult:
        """Perform comprehensive memory quality analysis."""
        start_time = time.time()

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()

            # Get all memories
            cursor.execute("SELECT * FROM memories ORDER BY created_at DESC")
            memories = [dict(row) for row in cursor.fetchall()]

            total_memories = len(memories)
            issues = []

            # Detect duplicates
            duplicates = []
            if request.analysis_type in ["comprehensive", "duplicates"]:
                duplicate_clusters = self.detect_duplicates(memories, request.similarity_threshold)
                for cluster in duplicate_clusters:
                    issue = QualityIssue(
                        issue_id=f"dup_{cluster.cluster_id}",
                        issue_type="duplicate",
                        severity="medium" if cluster.duplicate_count <= 2 else "high",
                        memory_ids=cluster.memory_ids,
                        title=f"Duplicate memories ({cluster.duplicate_count} duplicates)",
                        description=f"Found {cluster.duplicate_count} duplicate memories with {cluster.similarity_score:.2%} similarity",
                        recommendation=f"Merge duplicates into primary memory {cluster.primary_memory_id}",
                        impact_score=cluster.duplicate_count * 10,
                        auto_fixable=True,
                        estimated_savings={
                            "storage": len(cluster.content_preview) * cluster.duplicate_count * 0.8,
                            "performance": cluster.duplicate_count * 5,
                            "clarity": cluster.duplicate_count * 15,
                        },
                        metadata=cluster.metadata,
                    )
                    issues.append(issue)
                    duplicates.extend(cluster.memory_ids[1:])  # Exclude primary

            # Detect orphaned memories
            orphaned = []
            if request.analysis_type in ["comprehensive", "orphaned"]:
                orphaned_memories = self.detect_orphaned_memories(memories)
                for orphan in orphaned_memories:
                    issue = QualityIssue(
                        issue_id=f"orphan_{orphan['memory_id'][:8]}",
                        issue_type="orphaned",
                        severity="low" if orphan["orphan_score"] < 70 else "medium",
                        memory_ids=[orphan["memory_id"]],
                        title=f"Orphaned memory (isolation: {orphan['isolation_level']})",
                        description="Memory has no connections to workflows, goals, or other memories",
                        recommendation="Connect to relevant workflow or consider archiving",
                        impact_score=orphan["orphan_score"],
                        auto_fixable=orphan["isolation_level"] == "complete"
                        and orphan["orphan_score"] > 80,
                        estimated_savings={
                            "clarity": orphan["orphan_score"] * 0.5,
                            "organization": 20,
                        },
                        metadata={
                            "orphan_score": orphan["orphan_score"],
                            "isolation_level": orphan["isolation_level"],
                        },
                    )
                    issues.append(issue)
                    orphaned.append(orphan["memory_id"])

            # Analyze individual memory quality
            low_quality = []
            corrupted = []
            if request.analysis_type in ["comprehensive", "low_quality"]:
                for memory in memories:
                    quality_analysis = self.analyze_memory_quality(memory)

                    if quality_analysis["quality_score"] < 30:
                        issue = QualityIssue(
                            issue_id=f"quality_{memory['memory_id'][:8]}",
                            issue_type="low_quality",
                            severity="high" if quality_analysis["quality_score"] < 20 else "medium",
                            memory_ids=[memory["memory_id"]],
                            title=f"Low quality memory (score: {quality_analysis['quality_score']})",
                            description=f"Quality issues: {', '.join(quality_analysis['issues'])}",
                            recommendation="; ".join(quality_analysis["recommendations"]),
                            impact_score=50 - quality_analysis["quality_score"],
                            auto_fixable=False,
                            estimated_savings={"quality": 50 - quality_analysis["quality_score"]},
                            metadata={"quality_analysis": quality_analysis},
                        )
                        issues.append(issue)
                        low_quality.append(memory["memory_id"])

                    # Check for corruption
                    if "Encoding corruption detected" in quality_analysis["issues"]:
                        corrupted.append(memory["memory_id"])

            # Detect stale memories
            stale_memories = []
            if request.include_stale:
                stale_cutoff = time.time() - (request.stale_threshold_days * 86400)
                for memory in memories:
                    if (
                        memory.get("created_at", time.time()) < stale_cutoff
                        and memory.get("access_count", 0) == 0
                        and memory.get("importance", 1) < 5
                    ):
                        issue = QualityIssue(
                            issue_id=f"stale_{memory['memory_id'][:8]}",
                            issue_type="stale",
                            severity="low",
                            memory_ids=[memory["memory_id"]],
                            title=f"Stale memory ({(time.time() - memory.get('created_at', time.time())) / 86400:.0f} days old)",
                            description="Old memory with no recent access and low importance",
                            recommendation="Archive or delete if no longer relevant",
                            impact_score=min(
                                30,
                                (time.time() - memory.get("created_at", time.time())) / 86400 * 0.5,
                            ),
                            auto_fixable=True,
                            estimated_savings={"storage": len(memory.get("content", ""))},
                            metadata={
                                "age_days": (time.time() - memory.get("created_at", time.time()))
                                / 86400
                            },
                        )
                        issues.append(issue)
                        stale_memories.append(memory["memory_id"])

            # Calculate overall quality score
            issues_count = len(issues)
            overall_score = max(
                0, 100 - (issues_count * 5) - (len(duplicates) * 2) - (len(orphaned) * 1)
            )

            # Generate high-level recommendations
            recommendations = []
            if len(duplicates) > 10:
                recommendations.append(
                    "High number of duplicates detected. Run bulk duplicate cleanup."
                )
            if len(orphaned) > total_memories * 0.2:
                recommendations.append("Many orphaned memories. Review workflow organization.")
            if len(low_quality) > total_memories * 0.1:
                recommendations.append("Quality issues detected. Review content standards.")
            if len(stale_memories) > 50:
                recommendations.append("Archive old, unused memories to improve performance.")

            analysis_time = time.time() - start_time

            return QualityAnalysisResult(
                total_memories=total_memories,
                issues_found=issues_count,
                duplicates=len(duplicates),
                orphaned=len(orphaned),
                low_quality=len(low_quality),
                stale_memories=len(stale_memories),
                corrupted=len(corrupted),
                overall_score=overall_score,
                issues=issues,
                recommendations=recommendations,
                analysis_time=analysis_time,
            )

        finally:
            conn.close()

    async def preview_bulk_operation(self, request: BulkOperationRequest) -> BulkOperation:
        """Preview bulk operation changes before execution."""
        operation_id = f"bulk_{int(time.time())}"

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()

            # Get affected memories
            placeholders = ",".join("?" * len(request.memory_ids))
            cursor.execute(
                f"SELECT * FROM memories WHERE memory_id IN ({placeholders})", request.memory_ids
            )
            memories = [dict(row) for row in cursor.fetchall()]

            preview_changes = []
            estimated_impact = {"memories_affected": len(memories)}

            if request.operation_type == "delete":
                for memory in memories:
                    preview_changes.append(
                        {
                            "action": "delete",
                            "memory_id": memory["memory_id"],
                            "content_preview": memory["content"][:100] + "..."
                            if len(memory["content"]) > 100
                            else memory["content"],
                            "impact": "Memory will be permanently deleted",
                        }
                    )
                estimated_impact["storage_freed"] = sum(len(m["content"]) for m in memories)

            elif request.operation_type == "merge":
                if request.target_memory_id:
                    target = next(
                        (m for m in memories if m["memory_id"] == request.target_memory_id), None
                    )
                    if target:
                        others = [m for m in memories if m["memory_id"] != request.target_memory_id]
                        preview_changes.append(
                            {
                                "action": "merge_target",
                                "memory_id": target["memory_id"],
                                "impact": f"Will be kept as primary memory, enhanced with content from {len(others)} others",
                            }
                        )
                        for other in others:
                            preview_changes.append(
                                {
                                    "action": "merge_source",
                                    "memory_id": other["memory_id"],
                                    "impact": "Content will be merged into target, then deleted",
                                }
                            )

            elif request.operation_type == "archive":
                for memory in memories:
                    preview_changes.append(
                        {
                            "action": "archive",
                            "memory_id": memory["memory_id"],
                            "impact": "Memory will be marked as archived (soft delete)",
                        }
                    )

            return BulkOperation(
                operation_id=operation_id,
                operation_type=request.operation_type,
                memory_ids=request.memory_ids,
                preview_changes=preview_changes,
                estimated_impact=estimated_impact,
                reversible=request.operation_type in ["archive"],
                confirmation_required=request.operation_type in ["delete", "merge"],
            )

        finally:
            conn.close()

    async def execute_bulk_operation(self, operation: BulkOperation) -> Dict:
        """Execute bulk operation with safety checks."""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            results = {"success": 0, "failed": 0, "errors": []}

            if operation.operation_type == "delete":
                for memory_id in operation.memory_ids:
                    try:
                        # Delete related links first
                        cursor.execute(
                            "DELETE FROM memory_links WHERE source_memory_id = ? OR target_memory_id = ?",
                            (memory_id, memory_id),
                        )
                        # Delete memory
                        cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                        results["success"] += 1
                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to delete {memory_id}: {str(e)}")

            elif operation.operation_type == "archive":
                for memory_id in operation.memory_ids:
                    try:
                        cursor.execute(
                            "UPDATE memories SET archived = 1 WHERE memory_id = ?", (memory_id,)
                        )
                        results["success"] += 1
                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to archive {memory_id}: {str(e)}")

            conn.commit()
            return results

        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Bulk operation failed: {str(e)}") from e
        finally:
            conn.close()


class WorkingMemoryManager:
    """Core working memory management and optimization logic."""

    def __init__(self, db_path: str = "storage/unified_agent_memory.db"):
        self.db_path = db_path
        self.active_memories: Dict[str, WorkingMemoryItem] = {}
        self.capacity = 7  # Miller's rule: 7±2
        self.focus_mode = "normal"
        self.retention_time = 30  # minutes
        self.connected_clients: List[WebSocket] = []

    def get_db_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def calculate_memory_temperature(self, memory: Dict) -> float:
        """Calculate memory temperature based on access patterns."""
        now = time.time()
        last_access = memory.get("last_accessed_at", memory.get("created_at", now))
        access_count = memory.get("access_count", 0)

        # Recency component (decreases over time)
        time_since_access = now - last_access
        recency_score = max(0, 100 - (time_since_access / 3600) * 10)  # Decreases over hours

        # Frequency component
        frequency_score = min(100, access_count * 10)

        # Weighted combination
        temperature = recency_score * 0.7 + frequency_score * 0.3
        return round(temperature)

    def calculate_memory_priority(self, memory: Dict) -> str:
        """Calculate memory priority level."""
        importance = memory.get("importance", 1)
        if importance >= 9:
            return "critical"
        elif importance >= 7:
            return "high"
        elif importance >= 5:
            return "medium"
        else:
            return "low"

    def calculate_access_frequency(self, memory: Dict) -> float:
        """Calculate normalized access frequency."""
        access_count = memory.get("access_count", 0)
        return min(10, access_count / 5)  # Normalized to 0-10 scale

    def calculate_retention_score(self, memory: Dict) -> float:
        """Calculate how likely memory should remain in working memory."""
        importance = memory.get("importance", 1)
        confidence = memory.get("confidence", 0.5)
        access_count = memory.get("access_count", 0)

        score = (importance * 0.4 + confidence * 100 * 0.3 + min(access_count * 10, 100) * 0.3) / 10
        return round(score, 2)

    def enhance_memory_for_working_memory(self, memory: Dict) -> WorkingMemoryItem:
        """Convert database memory to enhanced working memory item."""
        return WorkingMemoryItem(
            memory_id=memory["memory_id"],
            content=memory["content"],
            memory_type=memory["memory_type"],
            memory_level=memory["memory_level"],
            importance=memory["importance"],
            confidence=memory.get("confidence", 0.5),
            created_at=memory["created_at"],
            last_accessed_at=memory.get("last_accessed_at"),
            access_count=memory.get("access_count", 0),
            workflow_id=memory.get("workflow_id"),
            temperature=self.calculate_memory_temperature(memory),
            priority=self.calculate_memory_priority(memory),
            access_frequency=self.calculate_access_frequency(memory),
            retention_score=self.calculate_retention_score(memory),
            added_at=time.time(),
        )

    def calculate_focus_score(self) -> float:
        """Calculate current focus score based on working memory coherence."""
        if not self.active_memories:
            return 100.0

        memories = list(self.active_memories.values())

        # Calculate average importance
        avg_importance = sum(m.importance for m in memories) / len(memories)

        # Calculate diversity penalty
        type_variety = len(set(m.memory_type for m in memories))
        level_variety = len(set(m.memory_level for m in memories))

        # Lower variety = higher focus
        variety_penalty = (type_variety + level_variety) * 5
        importance_bonus = avg_importance * 10

        focus_score = max(0, min(100, importance_bonus - variety_penalty + 20))
        return round(focus_score, 1)

    def calculate_efficiency(self) -> float:
        """Calculate working memory efficiency."""
        if not self.active_memories:
            return 100.0

        memories = list(self.active_memories.values())

        # Average temperature (activity level)
        avg_temperature = sum(m.temperature for m in memories) / len(memories)

        # Utilization rate
        utilization = (len(memories) / self.capacity) * 100

        # Optimal utilization is around 70%
        optimal_utilization = 100 - abs(utilization - 70) if abs(utilization - 70) < 30 else 70

        efficiency = avg_temperature * 0.6 + optimal_utilization * 0.4
        return round(efficiency)

    def get_working_memory_stats(self) -> WorkingMemoryStats:
        """Get current working memory statistics."""
        memories = list(self.active_memories.values())

        return WorkingMemoryStats(
            active_count=len(memories),
            capacity=self.capacity,
            pressure=round((len(memories) / self.capacity) * 100),
            temperature=round(sum(m.temperature for m in memories) / len(memories))
            if memories
            else 0,
            focus_score=self.calculate_focus_score(),
            efficiency=self.calculate_efficiency(),
            avg_retention_time=round(sum(m.retention_score for m in memories) / len(memories))
            if memories
            else 0,
            total_accesses=sum(m.access_count for m in memories),
            last_updated=time.time(),
        )

    def generate_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on current state."""
        suggestions = []
        stats = self.get_working_memory_stats()
        memories = list(self.active_memories.values())

        # High pressure suggestion
        if stats.pressure > 80:
            suggestions.append(
                OptimizationSuggestion(
                    id="reduce-pressure",
                    title="Reduce Memory Pressure",
                    description="Working memory is near capacity. Consider removing lower priority items.",
                    priority="high",
                    impact="High",
                    icon="alert-triangle",
                    action="Auto-Remove",
                    confidence=0.9,
                    estimated_improvement={"pressure": -20, "efficiency": 15},
                )
            )

        # Cold memories suggestion
        cold_memories = [m for m in memories if m.temperature < 30]
        if cold_memories:
            suggestions.append(
                OptimizationSuggestion(
                    id="remove-cold",
                    title="Remove Stale Memories",
                    description=f"{len(cold_memories)} memories haven't been accessed recently.",
                    priority="medium",
                    impact="Medium",
                    icon="snowflake",
                    action="Clear Stale",
                    confidence=0.8,
                    estimated_improvement={"temperature": 15, "efficiency": 10},
                )
            )

        # Low focus suggestion
        if stats.focus_score < 50:
            suggestions.append(
                OptimizationSuggestion(
                    id="improve-focus",
                    title="Improve Focus",
                    description="Working memory contains diverse, unrelated items. Consider focusing on a single task.",
                    priority="medium",
                    impact="High",
                    icon="target",
                    action="Focus Mode",
                    confidence=0.7,
                    estimated_improvement={"focus_score": 30, "efficiency": 20},
                )
            )

        # Underutilization suggestion
        if stats.active_count < self.capacity / 2:
            suggestions.append(
                OptimizationSuggestion(
                    id="add-related",
                    title="Add Related Memories",
                    description="Working memory has capacity for more relevant items.",
                    priority="low",
                    impact="Medium",
                    icon="plus-circle",
                    action="Add Related",
                    confidence=0.6,
                    estimated_improvement={"efficiency": 10, "focus_score": 5},
                )
            )

        return suggestions

    async def load_initial_working_memory(self) -> List[WorkingMemoryItem]:
        """Load initial working memory with high-importance memories."""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()

            # Get high-importance or working-level memories
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE memory_level = 'working' OR importance >= 8
                ORDER BY created_at DESC, importance DESC
                LIMIT ?
            """,
                (self.capacity,),
            )

            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                enhanced_memory = self.enhance_memory_for_working_memory(memory_dict)
                memories.append(enhanced_memory)
                self.active_memories[enhanced_memory.memory_id] = enhanced_memory

            return memories

        finally:
            conn.close()

    async def add_to_working_memory(self, memory_id: str) -> bool:
        """Add a memory to working memory."""
        if len(self.active_memories) >= self.capacity:
            return False

        if memory_id in self.active_memories:
            return False

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
            row = cursor.fetchone()

            if not row:
                return False

            memory_dict = dict(row)
            enhanced_memory = self.enhance_memory_for_working_memory(memory_dict)
            self.active_memories[memory_id] = enhanced_memory

            # Broadcast update to connected clients
            await self.broadcast_update()

            return True

        finally:
            conn.close()

    async def remove_from_working_memory(self, memory_id: str) -> bool:
        """Remove a memory from working memory."""
        if memory_id not in self.active_memories:
            return False

        del self.active_memories[memory_id]

        # Broadcast update to connected clients
        await self.broadcast_update()

        return True

    async def clear_working_memory(self):
        """Clear all working memory."""
        self.active_memories.clear()
        await self.broadcast_update()

    async def apply_focus_mode(
        self, mode: str, retention_time: Optional[int] = None, max_memory: Optional[int] = None
    ):
        """Apply focus mode settings."""
        mode_settings = {
            "deep": {"capacity": 5, "retention": 60},
            "creative": {"capacity": 9, "retention": 45},
            "analytical": {"capacity": 6, "retention": 90},
            "maintenance": {"capacity": 3, "retention": 20},
            "normal": {"capacity": 7, "retention": 30},
        }

        settings = mode_settings.get(mode, mode_settings["normal"])

        self.focus_mode = mode
        self.capacity = max_memory or settings["capacity"]
        self.retention_time = retention_time or settings["retention"]

        # If we're over capacity, remove lowest priority memories
        if len(self.active_memories) > self.capacity:
            memories_by_priority = sorted(
                self.active_memories.values(),
                key=lambda m: (m.importance, m.retention_score),
                reverse=True,
            )

            # Keep only the top memories
            to_keep = memories_by_priority[: self.capacity]
            self.active_memories = {m.memory_id: m for m in to_keep}

        await self.broadcast_update()

    async def auto_optimize(self) -> List[str]:
        """Apply automatic optimizations."""
        applied_optimizations = []
        suggestions = self.generate_optimization_suggestions()

        for suggestion in suggestions:
            if suggestion.priority in ["medium", "low"] and suggestion.confidence > 0.7:
                success = await self.apply_optimization(suggestion.id)
                if success:
                    applied_optimizations.append(suggestion.title)

        return applied_optimizations

    async def apply_optimization(self, suggestion_id: str) -> bool:
        """Apply a specific optimization."""
        memories = list(self.active_memories.values())

        if suggestion_id == "reduce-pressure":
            # Remove lowest priority memories
            low_priority = [m for m in memories if m.priority == "low"]
            for memory in low_priority[:2]:
                await self.remove_from_working_memory(memory.memory_id)
            return True

        elif suggestion_id == "remove-cold":
            # Remove cold memories
            cold_memories = [m for m in memories if m.temperature < 30]
            for memory in cold_memories[:3]:
                await self.remove_from_working_memory(memory.memory_id)
            return True

        elif suggestion_id == "improve-focus":
            # Switch to deep focus mode
            await self.apply_focus_mode("deep")
            return True

        elif suggestion_id == "add-related":
            # Add related memories
            await self.add_related_memories()
            return True

        return False

    async def add_related_memories(self):
        """Add memories related to current working memory."""
        if not self.active_memories or len(self.active_memories) >= self.capacity:
            return

        current_types = set(m.memory_type for m in self.active_memories.values())
        current_workflows = set(
            m.workflow_id for m in self.active_memories.values() if m.workflow_id
        )

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()

            # Find related memories
            placeholders = ",".join("?" * len(current_types)) if current_types else "''"
            workflow_placeholders = (
                ",".join("?" * len(current_workflows)) if current_workflows else "''"
            )

            query = f"""
                SELECT * FROM memories 
                WHERE memory_id NOT IN ({",".join("?" * len(self.active_memories))})
                AND (memory_type IN ({placeholders}) OR workflow_id IN ({workflow_placeholders}))
                AND importance >= 6
                ORDER BY importance DESC
                LIMIT ?
            """

            params = (
                list(self.active_memories.keys())
                + list(current_types)
                + list(current_workflows)
                + [self.capacity - len(self.active_memories)]
            )

            cursor.execute(query, params)

            for row in cursor.fetchall():
                memory_dict = dict(row)
                enhanced_memory = self.enhance_memory_for_working_memory(memory_dict)
                self.active_memories[enhanced_memory.memory_id] = enhanced_memory

                if len(self.active_memories) >= self.capacity:
                    break

        finally:
            conn.close()

        await self.broadcast_update()

    def get_memory_pool(
        self, search: str = "", filter_type: str = "", limit: int = 50
    ) -> List[Dict]:
        """Get available memory pool for working memory."""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()

            # Build query
            where_conditions = [
                "memory_id NOT IN ({})".format(",".join("?" * len(self.active_memories)))
            ]
            params = list(self.active_memories.keys())

            if search:
                where_conditions.append("(content LIKE ? OR memory_type LIKE ?)")
                params.extend([f"%{search}%", f"%{search}%"])

            if filter_type == "high":
                where_conditions.append("importance >= 8")
            elif filter_type == "recent":
                day_ago = time.time() - 86400
                where_conditions.append("created_at > ?")
                params.append(day_ago)
            elif filter_type == "related" and self.active_memories:
                current_types = set(m.memory_type for m in self.active_memories.values())
                current_workflows = set(
                    m.workflow_id for m in self.active_memories.values() if m.workflow_id
                )

                if current_types or current_workflows:
                    type_placeholders = (
                        ",".join("?" * len(current_types)) if current_types else "''"
                    )
                    workflow_placeholders = (
                        ",".join("?" * len(current_workflows)) if current_workflows else "''"
                    )
                    where_conditions.append(
                        f"(memory_type IN ({type_placeholders}) OR workflow_id IN ({workflow_placeholders}))"
                    )
                    params.extend(list(current_types) + list(current_workflows))

            query = f"""
                SELECT * FROM memories 
                WHERE {" AND ".join(where_conditions)}
                ORDER BY importance DESC
                LIMIT ?
            """
            params.append(limit)

            cursor.execute(query, params)

            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                memory_dict["access_frequency"] = self.calculate_access_frequency(memory_dict)
                memories.append(memory_dict)

            return memories

        finally:
            conn.close()

    def generate_heatmap_data(self, timeframe: str = "24h") -> List[Dict]:
        """Generate memory activity heatmap data."""
        now = time.time()
        intervals = []

        # Configure timeframe
        timeframe_config = {
            "1h": {"seconds": 300, "count": 12},  # 5 minute intervals
            "6h": {"seconds": 1800, "count": 12},  # 30 minute intervals
            "24h": {"seconds": 3600, "count": 24},  # 1 hour intervals
            "7d": {"seconds": 86400, "count": 7},  # 1 day intervals
        }

        config = timeframe_config.get(timeframe, timeframe_config["24h"])
        interval_seconds = config["seconds"]
        interval_count = config["count"]

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()

            for i in range(interval_count):
                interval_start = now - (interval_count - i) * interval_seconds
                interval_end = interval_start + interval_seconds

                # Count activities in this interval
                cursor.execute(
                    """
                    SELECT COUNT(*) as activity_count 
                    FROM memories 
                    WHERE created_at >= ? AND created_at <= ?
                """,
                    (interval_start, interval_end),
                )

                activity_count = cursor.fetchone()[0]

                intervals.append(
                    {
                        "time": interval_start,
                        "activity": activity_count,
                        "intensity": min(1.0, activity_count / 10),  # Normalize to 0-1
                    }
                )

            return intervals

        finally:
            conn.close()

    async def register_client(self, websocket: WebSocket):
        """Register a WebSocket client for real-time updates."""
        self.connected_clients.append(websocket)

    async def unregister_client(self, websocket: WebSocket):
        """Unregister a WebSocket client."""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)

    async def broadcast_update(self):
        """Broadcast working memory update to all connected clients."""
        if not self.connected_clients:
            return

        update_data = {
            "type": "working_memory_update",
            "stats": asdict(self.get_working_memory_stats()),
            "active_memories": [asdict(m) for m in self.active_memories.values()],
            "suggestions": [asdict(s) for s in self.generate_optimization_suggestions()],
            "timestamp": time.time(),
        }

        # Send to all connected clients
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(update_data))
            except Exception:
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            await self.unregister_client(client)


# Global working memory manager instance
working_memory_manager = WorkingMemoryManager()

# Global memory quality inspector instance
memory_quality_inspector = MemoryQualityInspector()


def setup_working_memory_routes(app: FastAPI):
    """Setup working memory API routes."""

    @app.get("/api/working-memory/status")
    async def get_working_memory_status():
        """Get current working memory status and statistics."""
        try:
            stats = working_memory_manager.get_working_memory_stats()
            active_memories = [asdict(m) for m in working_memory_manager.active_memories.values()]
            suggestions = [
                asdict(s) for s in working_memory_manager.generate_optimization_suggestions()
            ]

            return {
                "status": "connected",
                "stats": asdict(stats),
                "active_memories": active_memories,
                "suggestions": suggestions,
                "focus_mode": working_memory_manager.focus_mode,
                "capacity": working_memory_manager.capacity,
                "retention_time": working_memory_manager.retention_time,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/working-memory/initialize")
    async def initialize_working_memory():
        """Initialize working memory with default high-importance memories."""
        try:
            memories = await working_memory_manager.load_initial_working_memory()
            stats = working_memory_manager.get_working_memory_stats()

            return {
                "success": True,
                "message": f"Initialized with {len(memories)} memories",
                "stats": asdict(stats),
                "active_memories": [asdict(m) for m in memories],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/working-memory/add")
    async def add_memory_to_working_memory(request: WorkingMemoryRequest):
        """Add a memory to working memory."""
        try:
            success = await working_memory_manager.add_to_working_memory(request.memory_id)

            if success:
                stats = working_memory_manager.get_working_memory_stats()
                return {
                    "success": True,
                    "message": "Memory added to working memory",
                    "stats": asdict(stats),
                }
            else:
                return {
                    "success": False,
                    "message": "Could not add memory (capacity reached or already exists)",
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/working-memory/remove")
    async def remove_memory_from_working_memory(request: WorkingMemoryRequest):
        """Remove a memory from working memory."""
        try:
            success = await working_memory_manager.remove_from_working_memory(request.memory_id)

            if success:
                stats = working_memory_manager.get_working_memory_stats()
                return {
                    "success": True,
                    "message": "Memory removed from working memory",
                    "stats": asdict(stats),
                }
            else:
                return {"success": False, "message": "Memory not found in working memory"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/working-memory/clear")
    async def clear_working_memory():
        """Clear all working memory."""
        try:
            await working_memory_manager.clear_working_memory()
            stats = working_memory_manager.get_working_memory_stats()

            return {"success": True, "message": "Working memory cleared", "stats": asdict(stats)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/working-memory/focus-mode")
    async def set_focus_mode(request: FocusModeRequest):
        """Set focus mode and apply related optimizations."""
        try:
            await working_memory_manager.apply_focus_mode(
                request.mode, request.retention_time, request.max_working_memory
            )

            stats = working_memory_manager.get_working_memory_stats()

            return {
                "success": True,
                "message": f"Applied {request.mode} focus mode",
                "focus_mode": working_memory_manager.focus_mode,
                "capacity": working_memory_manager.capacity,
                "retention_time": working_memory_manager.retention_time,
                "stats": asdict(stats),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/working-memory/optimize")
    async def optimize_working_memory():
        """Apply automatic working memory optimizations."""
        try:
            applied = await working_memory_manager.auto_optimize()
            stats = working_memory_manager.get_working_memory_stats()

            return {
                "success": True,
                "message": f"Applied {len(applied)} optimizations",
                "optimizations_applied": applied,
                "stats": asdict(stats),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/working-memory/apply-suggestion")
    async def apply_optimization_suggestion(request: OptimizationRequest):
        """Apply a specific optimization suggestion."""
        try:
            success = await working_memory_manager.apply_optimization(request.suggestion_id)

            if success:
                stats = working_memory_manager.get_working_memory_stats()
                suggestions = [
                    asdict(s) for s in working_memory_manager.generate_optimization_suggestions()
                ]

                return {
                    "success": True,
                    "message": "Optimization applied successfully",
                    "stats": asdict(stats),
                    "suggestions": suggestions,
                }
            else:
                return {"success": False, "message": "Could not apply optimization"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/working-memory/pool")
    async def get_memory_pool(
        search: str = "",
        filter_type: str = "",  # "", "high", "recent", "related"
        limit: int = 50,
    ):
        """Get available memory pool for working memory."""
        try:
            memories = working_memory_manager.get_memory_pool(search, filter_type, limit)

            return {"success": True, "memories": memories, "count": len(memories)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/working-memory/heatmap")
    async def get_memory_heatmap(timeframe: str = "24h"):
        """Get memory activity heatmap data."""
        try:
            heatmap_data = working_memory_manager.generate_heatmap_data(timeframe)

            return {"success": True, "timeframe": timeframe, "data": heatmap_data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.websocket("/ws/working-memory")
    async def working_memory_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time working memory updates."""
        await websocket.accept()
        await working_memory_manager.register_client(websocket)

        try:
            # Send initial data
            initial_data = {
                "type": "initial_data",
                "stats": asdict(working_memory_manager.get_working_memory_stats()),
                "active_memories": [
                    asdict(m) for m in working_memory_manager.active_memories.values()
                ],
                "suggestions": [
                    asdict(s) for s in working_memory_manager.generate_optimization_suggestions()
                ],
                "focus_mode": working_memory_manager.focus_mode,
                "capacity": working_memory_manager.capacity,
            }
            await websocket.send_text(json.dumps(initial_data))

            # Keep connection alive and handle messages
            while True:
                try:
                    # Wait for messages from client
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    break

        finally:
            await working_memory_manager.unregister_client(websocket)

    # Memory Quality Inspector API Endpoints
    @app.post("/api/memory-quality/analyze")
    async def analyze_memory_quality(request: QualityAnalysisRequest):
        """Perform comprehensive memory quality analysis."""
        try:
            result = await memory_quality_inspector.perform_quality_analysis(request)

            return {"success": True, "analysis": asdict(result)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Quality analysis failed: {str(e)}") from e

    @app.get("/api/memory-quality/quick-scan")
    async def quick_quality_scan():
        """Perform quick quality scan with basic metrics."""
        try:
            request = QualityAnalysisRequest(
                analysis_type="comprehensive",
                include_stale=False,
                include_low_importance=False,
                similarity_threshold=0.90,
                stale_threshold_days=7,
            )
            result = await memory_quality_inspector.perform_quality_analysis(request)

            # Return simplified metrics for quick overview
            return {
                "success": True,
                "quick_metrics": {
                    "total_memories": result.total_memories,
                    "overall_score": result.overall_score,
                    "critical_issues": len([i for i in result.issues if i.severity == "critical"]),
                    "duplicates": result.duplicates,
                    "orphaned": result.orphaned,
                    "low_quality": result.low_quality,
                    "top_recommendations": result.recommendations[:3],
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Quick scan failed: {str(e)}") from e

    @app.post("/api/memory-quality/bulk-preview")
    async def preview_bulk_operation(request: BulkOperationRequest):
        """Preview bulk operation changes before execution."""
        try:
            operation = await memory_quality_inspector.preview_bulk_operation(request)

            return {"success": True, "operation": asdict(operation)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Bulk preview failed: {str(e)}") from e

    @app.post("/api/memory-quality/bulk-execute")
    async def execute_bulk_operation(operation_request: BulkOperationRequest):
        """Execute bulk operation with safety checks."""
        try:
            # First preview the operation
            operation = await memory_quality_inspector.preview_bulk_operation(operation_request)

            # Execute the operation
            results = await memory_quality_inspector.execute_bulk_operation(operation)

            return {
                "success": True,
                "operation_id": operation.operation_id,
                "results": results,
                "message": f"Bulk operation completed: {results['success']} successful, {results['failed']} failed",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Bulk operation failed: {str(e)}") from e

    @app.get("/api/memory-quality/duplicates")
    async def get_duplicates():
        """Get all duplicate memory clusters."""
        try:
            conn = memory_quality_inspector.get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memories ORDER BY created_at DESC")
                memories = [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

            clusters = memory_quality_inspector.detect_duplicates(memories, threshold=0.85)

            return {
                "success": True,
                "clusters": [asdict(cluster) for cluster in clusters],
                "total_clusters": len(clusters),
                "total_duplicates": sum(cluster.duplicate_count for cluster in clusters),
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Duplicate detection failed: {str(e)}"
            ) from e

    @app.get("/api/memory-quality/orphaned")
    async def get_orphaned_memories():
        """Get all orphaned memories."""
        try:
            conn = memory_quality_inspector.get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memories ORDER BY created_at DESC")
                memories = [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

            orphaned = memory_quality_inspector.detect_orphaned_memories(memories)

            return {
                "success": True,
                "orphaned_memories": orphaned,
                "total_orphaned": len(orphaned),
                "completely_isolated": len(
                    [m for m in orphaned if m["isolation_level"] == "complete"]
                ),
                "partially_isolated": len(
                    [m for m in orphaned if m["isolation_level"] == "partial"]
                ),
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Orphaned memory detection failed: {str(e)}"
            ) from e

    @app.get("/api/memory-quality/stats")
    async def get_quality_stats():
        """Get overall memory quality statistics."""
        try:
            conn = memory_quality_inspector.get_db_connection()
            try:
                cursor = conn.cursor()

                # Basic stats
                cursor.execute("SELECT COUNT(*) as total FROM memories")
                total_memories = cursor.fetchone()["total"]

                cursor.execute(
                    "SELECT AVG(importance) as avg_importance, AVG(confidence) as avg_confidence FROM memories"
                )
                quality_metrics = cursor.fetchone()

                cursor.execute(
                    "SELECT COUNT(*) as with_workflow FROM memories WHERE workflow_id IS NOT NULL"
                )
                with_workflow = cursor.fetchone()["with_workflow"]

                cursor.execute(
                    "SELECT COUNT(*) as recent FROM memories WHERE created_at > ?",
                    (time.time() - 86400 * 7,),
                )
                recent_memories = cursor.fetchone()["recent"]

                # Quality distribution
                cursor.execute("""
                    SELECT 
                        SUM(CASE WHEN importance >= 8 THEN 1 ELSE 0 END) as high_importance,
                        SUM(CASE WHEN importance >= 5 THEN 1 ELSE 0 END) as medium_importance,
                        SUM(CASE WHEN confidence >= 0.8 THEN 1 ELSE 0 END) as high_confidence,
                        SUM(CASE WHEN confidence >= 0.5 THEN 1 ELSE 0 END) as medium_confidence
                    FROM memories
                """)
                quality_dist = cursor.fetchone()

            finally:
                conn.close()

            return {
                "success": True,
                "stats": {
                    "total_memories": total_memories,
                    "avg_importance": round(quality_metrics["avg_importance"], 2),
                    "avg_confidence": round(quality_metrics["avg_confidence"], 2),
                    "workflow_coverage": round(with_workflow / max(1, total_memories) * 100, 1),
                    "recent_activity": recent_memories,
                    "quality_distribution": {
                        "high_importance": quality_dist["high_importance"],
                        "medium_importance": quality_dist["medium_importance"],
                        "high_confidence": quality_dist["high_confidence"],
                        "medium_confidence": quality_dist["medium_confidence"],
                    },
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stats collection failed: {str(e)}") from e


# Background task to periodically update working memory
async def working_memory_background_task():
    """Background task for periodic working memory updates."""
    while True:
        try:
            # Update temperatures and stats periodically
            for memory in working_memory_manager.active_memories.values():
                # Recalculate temperature based on current time
                memory.temperature = working_memory_manager.calculate_memory_temperature(
                    asdict(memory)
                )

            # Broadcast updates if there are connected clients
            if working_memory_manager.connected_clients:
                await working_memory_manager.broadcast_update()

            # Wait 30 seconds before next update
            await asyncio.sleep(30)

        except Exception as e:
            print(f"Background task error: {e}")
            await asyncio.sleep(60)  # Wait longer if there's an error


def start_background_tasks(app: FastAPI):
    """Start background tasks for working memory management."""

    @app.on_event("startup")
    async def startup_event():
        # Start background task
        asyncio.create_task(working_memory_background_task())

        # Initialize working memory with default data
        try:
            await working_memory_manager.load_initial_working_memory()
            print("✅ Working memory initialized successfully")
        except Exception as e:
            print(f"⚠️ Could not initialize working memory: {e}")
