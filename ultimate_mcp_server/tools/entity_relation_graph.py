# -*- coding: utf-8 -*-
"""Entity relationship graph tools for Ultimate MCP Server."""

import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# --- Optional Imports ---
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from pyvis.network import Network

    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

HAS_VISUALIZATION_LIBS = HAS_NETWORKX and HAS_PYVIS and HAS_MATPLOTLIB
# --- End Optional Imports ---

from ultimate_mcp_server.constants import Provider  # noqa: E402
from ultimate_mcp_server.core.providers.base import BaseProvider, get_provider  # noqa: E402
from ultimate_mcp_server.exceptions import ProviderError, ToolError, ToolInputError  # noqa: E402
from ultimate_mcp_server.tools.base import (  # noqa: E402
    with_cache,
    with_error_handling,
    with_retry,
    with_tool_metrics,
)
from ultimate_mcp_server.tools.completion import generate_completion  # noqa: E402
from ultimate_mcp_server.tools.document_conversion_and_processing import (  # noqa: E402
    chunk_document_standalone,
)
from ultimate_mcp_server.utils import get_logger  # noqa: E402

logger = get_logger("ultimate_mcp_server.tools.entity_graph")


# --- Enums ---
class GraphStrategy(Enum):
    """Strategies for entity graph extraction."""

    STANDARD = "standard"  # Basic prompt-based extraction
    MULTISTAGE = "multistage"  # Process in stages: entities first, then relationships
    CHUNKED = "chunked"  # Process large texts in chunks and merge results
    INCREMENTAL = "incremental"  # Build graph incrementally from existing graph
    STRUCTURED = "structured"  # Use structured examples for consistent extraction
    STRICT_SCHEMA = "strict_schema"  # Use a predefined schema of entities and relationships


class OutputFormat(Enum):
    """Output formats for entity graphs."""

    JSON = "json"  # Standard JSON (default)
    NETWORKX = "networkx"  # NetworkX graph object
    RDF = "rdf"  # Resource Description Framework (triples)
    CYTOSCAPE = "cytoscape"  # Cytoscape.js format
    D3 = "d3"  # D3.js force graph format (nodes/links)
    NEO4J = "neo4j"  # Neo4j Cypher queries


class VisualizationFormat(Enum):
    """Visualization formats for entity graphs."""

    NONE = "none"  # No visualization
    HTML = "html"  # Interactive HTML (Pyvis)
    SVG = "svg"  # Static SVG (NetworkX/Matplotlib)
    PNG = "png"  # Static PNG (NetworkX/Matplotlib) - Requires Matplotlib
    DOT = "dot"  # GraphViz DOT format


# --- Global Schemas & Examples ---
# (COMMON_SCHEMAS, SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES remain the same as in the original prompt)
# --- Global schemas for common domains ---
COMMON_SCHEMAS = {
    "business": {
        "entities": [
            {"type": "Person", "attributes": ["name", "title", "role"]},
            {"type": "Organization", "attributes": ["name", "industry", "location"]},
            {"type": "Product", "attributes": ["name", "category", "price"]},
            {"type": "Location", "attributes": ["name", "address", "type"]},
            {"type": "Event", "attributes": ["name", "date", "location"]},
        ],
        "relationships": [
            {"type": "WORKS_FOR", "source_types": ["Person"], "target_types": ["Organization"]},
            {"type": "PRODUCES", "source_types": ["Organization"], "target_types": ["Product"]},
            {
                "type": "COMPETES_WITH",
                "source_types": ["Organization", "Product"],
                "target_types": ["Organization", "Product"],
            },
            {
                "type": "LOCATED_IN",
                "source_types": ["Organization", "Person"],
                "target_types": ["Location"],
            },
            {"type": "FOUNDED", "source_types": ["Person"], "target_types": ["Organization"]},
            {
                "type": "ACQUIRED",
                "source_types": ["Organization"],
                "target_types": ["Organization"],
            },
            {
                "type": "SUPPLIES",
                "source_types": ["Organization"],
                "target_types": ["Organization"],
            },
            {
                "type": "PARTNERS_WITH",
                "source_types": ["Organization"],
                "target_types": ["Organization"],
            },
            {
                "type": "INVESTS_IN",
                "source_types": ["Organization", "Person"],
                "target_types": ["Organization"],
            },
            {"type": "ATTENDS", "source_types": ["Person"], "target_types": ["Event"]},
            {"type": "HOSTS", "source_types": ["Organization"], "target_types": ["Event"]},
        ],
    },
    "academic": {
        "entities": [
            {"type": "Researcher", "attributes": ["name", "affiliation", "field"]},
            {"type": "Institution", "attributes": ["name", "type", "location"]},
            {"type": "Publication", "attributes": ["title", "date", "journal", "impact_factor"]},
            {"type": "Concept", "attributes": ["name", "field", "definition"]},
            {"type": "Dataset", "attributes": ["name", "size", "source"]},
            {"type": "Research_Project", "attributes": ["name", "duration", "funding"]},
        ],
        "relationships": [
            {
                "type": "AFFILIATED_WITH",
                "source_types": ["Researcher"],
                "target_types": ["Institution"],
            },
            {"type": "AUTHORED", "source_types": ["Researcher"], "target_types": ["Publication"]},
            {"type": "CITES", "source_types": ["Publication"], "target_types": ["Publication"]},
            {"type": "INTRODUCES", "source_types": ["Publication"], "target_types": ["Concept"]},
            {
                "type": "COLLABORATES_WITH",
                "source_types": ["Researcher"],
                "target_types": ["Researcher"],
            },
            {
                "type": "USES",
                "source_types": ["Publication", "Researcher"],
                "target_types": ["Dataset", "Concept"],
            },
            {
                "type": "BUILDS_ON",
                "source_types": ["Concept", "Publication"],
                "target_types": ["Concept"],
            },
            {
                "type": "FUNDS",
                "source_types": ["Institution"],
                "target_types": ["Research_Project"],
            },
            {
                "type": "WORKS_ON",
                "source_types": ["Researcher"],
                "target_types": ["Research_Project"],
            },
        ],
    },
    "medical": {
        "entities": [
            {"type": "Patient", "attributes": ["id", "age", "gender"]},
            {"type": "Physician", "attributes": ["name", "specialty", "affiliation"]},
            {"type": "Condition", "attributes": ["name", "icd_code", "severity"]},
            {"type": "Medication", "attributes": ["name", "dosage", "manufacturer"]},
            {"type": "Procedure", "attributes": ["name", "code", "duration"]},
            {"type": "Healthcare_Facility", "attributes": ["name", "type", "location"]},
        ],
        "relationships": [
            {"type": "DIAGNOSED_WITH", "source_types": ["Patient"], "target_types": ["Condition"]},
            {"type": "TREATED_BY", "source_types": ["Patient"], "target_types": ["Physician"]},
            {"type": "PRESCRIBED", "source_types": ["Physician"], "target_types": ["Medication"]},
            {"type": "TAKES", "source_types": ["Patient"], "target_types": ["Medication"]},
            {
                "type": "TREATS",
                "source_types": ["Medication", "Procedure"],
                "target_types": ["Condition"],
            },
            {"type": "PERFORMED", "source_types": ["Physician"], "target_types": ["Procedure"]},
            {"type": "UNDERWENT", "source_types": ["Patient"], "target_types": ["Procedure"]},
            {
                "type": "WORKS_AT",
                "source_types": ["Physician"],
                "target_types": ["Healthcare_Facility"],
            },
            {
                "type": "ADMITTED_TO",
                "source_types": ["Patient"],
                "target_types": ["Healthcare_Facility"],
            },
            {
                "type": "INTERACTS_WITH",
                "source_types": ["Medication"],
                "target_types": ["Medication"],
            },
            {
                "type": "CONTRAINDICATES",
                "source_types": ["Condition"],
                "target_types": ["Medication"],
            },
        ],
    },
    "legal": {
        "entities": [
            {"type": "Person", "attributes": ["name", "role", "jurisdiction"]},
            {"type": "Legal_Entity", "attributes": ["name", "type", "jurisdiction"]},
            {"type": "Document", "attributes": ["name", "type", "date", "status"]},
            {"type": "Obligation", "attributes": ["description", "deadline", "status"]},
            {"type": "Claim", "attributes": ["description", "value", "status"]},
            {"type": "Asset", "attributes": ["description", "value", "type"]},
            {"type": "Court", "attributes": ["name", "jurisdiction", "type"]},
            {"type": "Law", "attributes": ["name", "jurisdiction", "date"]},
        ],
        "relationships": [
            {
                "type": "PARTY_TO",
                "source_types": ["Person", "Legal_Entity"],
                "target_types": ["Document"],
            },
            {
                "type": "HAS_OBLIGATION",
                "source_types": ["Person", "Legal_Entity"],
                "target_types": ["Obligation"],
            },
            {"type": "OWNS", "source_types": ["Person", "Legal_Entity"], "target_types": ["Asset"]},
            {
                "type": "CLAIMS",
                "source_types": ["Person", "Legal_Entity"],
                "target_types": ["Claim"],
            },
            {
                "type": "REPRESENTED_BY",
                "source_types": ["Person", "Legal_Entity"],
                "target_types": ["Person"],
            },
            {
                "type": "REFERENCED_IN",
                "source_types": ["Law", "Document"],
                "target_types": ["Document"],
            },
            {
                "type": "ADJUDICATED_BY",
                "source_types": ["Claim", "Document"],
                "target_types": ["Court"],
            },
            {
                "type": "REGULATES",
                "source_types": ["Law"],
                "target_types": ["Legal_Entity", "Obligation"],
            },
            {"type": "TRANSFERS", "source_types": ["Document"], "target_types": ["Asset"]},
            {
                "type": "AUTHORIZES",
                "source_types": ["Document"],
                "target_types": ["Person", "Legal_Entity"],
            },
        ],
    },
}

# --- Entity relationship detection prompts ---
SYSTEM_PROMPTS = {
    "entity_detection": """You are an expert entity extraction system. Your task is to identify and extract named entities from the input text with high precision. Follow these guidelines:

1. Focus on identifying complete entity mentions.
2. Classify entities into the specified types accurately.
3. Generate unique IDs for each distinct entity.
4. Include attributes and position information as requested.
5. Only extract entities actually mentioned in the text.
6. Do not hallucinate entities that aren't clearly present.

Output should be in valid JSON format containing a list of entities, adhering strictly to the requested structure.
""",
    "relationship_detection": """You are an expert relationship extraction system. Your task is to identify meaningful connections between the provided entities based on the input text. Follow these guidelines:

1. Only identify relationships between the provided entities that are explicitly stated or strongly implied in the text.
2. Capture the semantic relationship type accurately based on the specified types.
3. Identify the direction of the relationship (source -> target) using the provided entity IDs.
4. Include supporting evidence from the text and temporal information if requested.
5. Assign a confidence score (0.0-1.0) based on how explicitly the relationship is stated.
6. Do not invent relationships not supported by the text.

Output should be in valid JSON format containing a list of relationships, adhering strictly to the requested structure and referencing the provided entity IDs.
""",
    "standard_extraction": """You are an expert entity and relationship extraction system. Your task is to identify named entities and the relationships between them from the input text. Follow these guidelines:

1. Identify named entities and classify them into the specified types.
2. Generate unique IDs for each distinct entity.
3. Identify relationships between the extracted entities based on the specified types.
4. Ensure relationship source and target IDs correspond to the extracted entities.
5. Include attributes, position information, evidence, and temporal context as requested.
6. Assign a confidence score (0.0-1.0) for relationships.
7. Only extract entities and relationships explicitly mentioned or strongly implied.
8. Do not hallucinate information.

Output should be in valid JSON format containing lists of entities and relationships, adhering strictly to the requested structure.
""",
    "multilingual": """You are an expert entity and relationship extraction system with multilingual capabilities. Extract entities and relationships from text in the specified language (or identify the language if not specified). Apply language-specific extraction patterns to identify:

1. Named entities (people, organizations, locations, etc.) matching the specified types.
2. The relationships between these entities matching the specified types.
3. Evidence for each relationship from the text.
4. Generate unique IDs and include requested details (attributes, positions, temporal info).

Be attentive to language-specific naming patterns, grammatical structures, and relationship indicators. Output valid JSON strictly following the requested format.
""",
    "temporal": """You are a specialized entity and relationship extraction system focusing on temporal information. Extract entities and relationships, paying close attention to time references. Your goal is to capture:

1. Entities with their temporal attributes (founding dates, birth dates, etc.).
2. Relationships between entities with temporal context (start/end dates, duration).
3. Changes in relationships over time.
4. Sequence of events involving entities.

For each entity and relationship, capture explicit or implicit time information as precisely as possible in the 'temporal' field. Output valid JSON strictly following the requested format.
""",
}

# --- Few-Shot Examples ---
FEW_SHOT_EXAMPLES = {
    "business": {
        "text": """Apple Inc., founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976, announced its latest iPhone model yesterday at its headquarters in Cupertino, California. CEO Tim Cook showcased the device, which competes with Samsung's Galaxy series.

The company has partnered with TSMC to manufacture the A15 Bionic chip that powers the new iPhone. Meanwhile, Google, led by Sundar Pichai, continues to dominate the search engine market with products that compete with Apple's offerings.""",
        "entities": [
            {
                "id": "ent1",
                "name": "Apple Inc.",
                "type": "Organization",
                "mentions": [
                    {"text": "Apple Inc.", "pos": [0, 10]},
                    {"text": "Apple", "pos": [226, 231]},
                ],
                "attributes": {"location": "Cupertino, California"},
            },
            {
                "id": "ent2",
                "name": "Steve Jobs",
                "type": "Person",
                "mentions": [{"text": "Steve Jobs", "pos": [22, 32]}],
            },
            {
                "id": "ent3",
                "name": "Steve Wozniak",
                "type": "Person",
                "mentions": [{"text": "Steve Wozniak", "pos": [34, 47]}],
            },
            {
                "id": "ent4",
                "name": "Ronald Wayne",
                "type": "Person",
                "mentions": [{"text": "Ronald Wayne", "pos": [53, 65]}],
            },
            {
                "id": "ent5",
                "name": "iPhone",
                "type": "Product",
                "mentions": [
                    {"text": "iPhone", "pos": [95, 101]},
                    {"text": "iPhone", "pos": [324, 330]},
                ],
            },
            {
                "id": "ent6",
                "name": "Cupertino, California",
                "type": "Location",
                "mentions": [{"text": "Cupertino, California", "pos": [129, 149]}],
            },
            {
                "id": "ent7",
                "name": "Tim Cook",
                "type": "Person",
                "mentions": [{"text": "Tim Cook", "pos": [156, 164]}],
                "attributes": {"role": "CEO"},
            },
            {
                "id": "ent8",
                "name": "Samsung",
                "type": "Organization",
                "mentions": [{"text": "Samsung", "pos": [201, 208]}],
            },
            {
                "id": "ent9",
                "name": "Galaxy series",
                "type": "Product",
                "mentions": [{"text": "Galaxy series", "pos": [210, 223]}],
            },
            {
                "id": "ent10",
                "name": "TSMC",
                "type": "Organization",
                "mentions": [{"text": "TSMC", "pos": [261, 265]}],
            },
            {
                "id": "ent11",
                "name": "A15 Bionic chip",
                "type": "Product",
                "mentions": [{"text": "A15 Bionic chip", "pos": [281, 295]}],
            },
            {
                "id": "ent12",
                "name": "Google",
                "type": "Organization",
                "mentions": [{"text": "Google", "pos": [348, 354]}],
            },
            {
                "id": "ent13",
                "name": "Sundar Pichai",
                "type": "Person",
                "mentions": [{"text": "Sundar Pichai", "pos": [365, 378]}],
            },
        ],
        "relationships": [
            {
                "id": "rel1",
                "source": "ent2",
                "target": "ent1",
                "type": "FOUNDED",
                "confidence": 0.95,
                "evidence": "Apple Inc., founded by Steve Jobs... in 1976",
                "temporal": {"year": 1976},
            },
            {
                "id": "rel2",
                "source": "ent3",
                "target": "ent1",
                "type": "FOUNDED",
                "confidence": 0.95,
                "evidence": "Apple Inc., founded by... Steve Wozniak... in 1976",
                "temporal": {"year": 1976},
            },
            {
                "id": "rel3",
                "source": "ent4",
                "target": "ent1",
                "type": "FOUNDED",
                "confidence": 0.95,
                "evidence": "Apple Inc., founded by... Ronald Wayne in 1976",
                "temporal": {"year": 1976},
            },
            {
                "id": "rel4",
                "source": "ent7",
                "target": "ent1",
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "evidence": "CEO Tim Cook",
            },
            {
                "id": "rel5",
                "source": "ent1",
                "target": "ent5",
                "type": "PRODUCES",
                "confidence": 0.9,
                "evidence": "Apple Inc.... announced its latest iPhone model",
            },
            {
                "id": "rel6",
                "source": "ent1",
                "target": "ent6",
                "type": "LOCATED_IN",
                "confidence": 0.8,
                "evidence": "its headquarters in Cupertino, California",
            },
            {
                "id": "rel7",
                "source": "ent5",
                "target": "ent9",
                "type": "COMPETES_WITH",
                "confidence": 0.8,
                "evidence": "which competes with Samsung's Galaxy series",
            },
            {
                "id": "rel8",
                "source": "ent1",
                "target": "ent10",
                "type": "PARTNERS_WITH",
                "confidence": 0.9,
                "evidence": "The company has partnered with TSMC",
            },
            {
                "id": "rel9",
                "source": "ent10",
                "target": "ent11",
                "type": "PRODUCES",
                "confidence": 0.9,
                "evidence": "TSMC to manufacture the A15 Bionic chip",
            },  # Changed from MANUFACTURES to align with schema
            {
                "id": "rel10",
                "source": "ent11",
                "target": "ent5",
                "type": "COMPONENT_OF",
                "confidence": 0.9,
                "evidence": "A15 Bionic chip that powers the new iPhone",
            },
            {
                "id": "rel11",
                "source": "ent13",
                "target": "ent12",
                "type": "WORKS_FOR",
                "confidence": 0.85,
                "evidence": "Google, led by Sundar Pichai",
            },  # Changed from LEADS to align with schema
            {
                "id": "rel12",
                "source": "ent12",
                "target": "ent1",
                "type": "COMPETES_WITH",
                "confidence": 0.7,
                "evidence": "with products that compete with Apple's offerings",
            },
        ],
    },
    "academic": {
        "text": """Dr. Jennifer Chen from Stanford University published a groundbreaking paper in Nature on quantum computing applications in drug discovery. Her research, funded by the National Science Foundation, built upon earlier work by Dr. Richard Feynman.

Chen collaborated with Dr. Michael Layton at MIT, who provided the dataset used in their experiments. Their publication has been cited by researchers at IBM's Quantum Computing division led by Dr. Sarah Johnson.""",
        "entities": [
            {
                "id": "ent1",
                "name": "Jennifer Chen",
                "type": "Researcher",
                "attributes": {"affiliation": "Stanford University"},
                "mentions": [{"text": "Dr. Jennifer Chen", "pos": [0, 16]}],
            },
            {
                "id": "ent2",
                "name": "Stanford University",
                "type": "Institution",
                "mentions": [{"text": "Stanford University", "pos": [22, 41]}],
            },
            {
                "id": "ent3",
                "name": "Nature",
                "type": "Publication",
                "attributes": {"type": "Journal"},
                "mentions": [{"text": "Nature", "pos": [78, 84]}],
            },  # Added type attribute
            {
                "id": "ent4",
                "name": "Groundbreaking paper on quantum computing",
                "type": "Publication",
                "attributes": {
                    "title": "Groundbreaking paper on quantum computing applications in drug discovery",
                    "date": None,
                },
                "mentions": [{"text": "paper", "pos": [71, 76]}],
            },  # Modified name/title
            {
                "id": "ent5",
                "name": "National Science Foundation",
                "type": "Institution",
                "attributes": {"type": "Funding Organization"},
                "mentions": [{"text": "National Science Foundation", "pos": [124, 152]}],
            },  # Added type attribute
            {
                "id": "ent6",
                "name": "Richard Feynman",
                "type": "Researcher",
                "mentions": [{"text": "Dr. Richard Feynman", "pos": [178, 196]}],
            },
            {
                "id": "ent7",
                "name": "Michael Layton",
                "type": "Researcher",
                "attributes": {"affiliation": "MIT"},
                "mentions": [{"text": "Dr. Michael Layton", "pos": [223, 241]}],
            },
            {
                "id": "ent8",
                "name": "MIT",
                "type": "Institution",
                "mentions": [{"text": "MIT", "pos": [245, 248]}],
            },
            {
                "id": "ent9",
                "name": "Drug discovery dataset",
                "type": "Dataset",
                "mentions": [{"text": "dataset", "pos": [264, 271]}],
            },
            {
                "id": "ent10",
                "name": "IBM",
                "type": "Institution",
                "attributes": {"type": "Company"},
                "mentions": [{"text": "IBM", "pos": [334, 337]}],
            },  # Added type attribute
            {
                "id": "ent11",
                "name": "IBM Quantum Computing division",
                "type": "Institution",
                "attributes": {"type": "Research Division", "parent_org": "IBM"},
                "mentions": [{"text": "IBM's Quantum Computing division", "pos": [334, 365]}],
            },  # Added type/parent attributes
            {
                "id": "ent12",
                "name": "Sarah Johnson",
                "type": "Researcher",
                "mentions": [{"text": "Dr. Sarah Johnson", "pos": [375, 392]}],
            },
        ],
        "relationships": [
            {
                "id": "rel1",
                "source": "ent1",
                "target": "ent2",
                "type": "AFFILIATED_WITH",
                "confidence": 0.95,
                "evidence": "Dr. Jennifer Chen from Stanford University",
            },
            {
                "id": "rel2",
                "source": "ent1",
                "target": "ent4",
                "type": "AUTHORED",
                "confidence": 0.95,
                "evidence": "Dr. Jennifer Chen... published a groundbreaking paper",
            },
            {
                "id": "rel3",
                "source": "ent4",
                "target": "ent3",
                "type": "PUBLISHED_IN",
                "confidence": 0.9,
                "evidence": "published a groundbreaking paper in Nature",
            },
            {
                "id": "rel4",
                "source": "ent5",
                "target": "ent4",
                "type": "FUNDS",
                "confidence": 0.85,
                "evidence": "Her research, funded by the National Science Foundation",
            },  # Changed target to Publication ent4
            {
                "id": "rel5",
                "source": "ent4",
                "target": "ent6",
                "type": "BUILDS_ON",
                "confidence": 0.8,
                "evidence": "built upon earlier work by Dr. Richard Feynman",
            },
            {
                "id": "rel6",
                "source": "ent1",
                "target": "ent7",
                "type": "COLLABORATES_WITH",
                "confidence": 0.9,
                "evidence": "Chen collaborated with Dr. Michael Layton at MIT",
            },
            {
                "id": "rel7",
                "source": "ent7",
                "target": "ent8",
                "type": "AFFILIATED_WITH",
                "confidence": 0.9,
                "evidence": "Dr. Michael Layton at MIT",
            },
            {
                "id": "rel8",
                "source": "ent7",
                "target": "ent9",
                "type": "PROVIDED",
                "confidence": 0.85,
                "evidence": "who provided the dataset used in their experiments",
            },
            {
                "id": "rel9",
                "source": "ent4",
                "target": "ent9",
                "type": "USES",
                "confidence": 0.8,
                "evidence": "dataset used in their experiments",
            },
            {
                "id": "rel10",
                "source": "ent11",
                "target": "ent4",
                "type": "CITES",
                "confidence": 0.85,
                "evidence": "Their publication has been cited by researchers at IBM's Quantum Computing division",
            },
            {
                "id": "rel11",
                "source": "ent12",
                "target": "ent11",
                "type": "WORKS_ON",
                "confidence": 0.9,
                "evidence": "IBM's Quantum Computing division led by Dr. Sarah Johnson",
            },  # Changed from LEADS to align with schema
            {
                "id": "rel12",
                "source": "ent11",
                "target": "ent10",
                "type": "PART_OF",
                "confidence": 0.9,
                "evidence": "IBM's Quantum Computing division",
            },  # Added relationship
        ],
    },
}


# --- Main Tool Function ---
@with_cache(ttl=24 * 60 * 60)  # Cache results for 24 hours
@with_tool_metrics
@with_retry(max_retries=2, retry_delay=1.5)
@with_error_handling
async def extract_entity_graph(
    text: str,
    entity_types: Optional[List[str]] = None,
    relation_types: Optional[List[str]] = None,
    provider: str = Provider.OPENAI.value,
    model: Optional[str] = None,
    include_evidence: bool = True,
    include_attributes: bool = True,
    include_positions: bool = False,  # Default False as it can increase token usage significantly
    include_temporal_info: bool = True,
    max_entities: int = 100,
    max_relations: int = 200,
    min_confidence: float = 0.6,
    domain: Optional[str] = None,  # e.g., "business", "academic", "medical", "legal"
    output_format: Union[str, OutputFormat] = OutputFormat.JSON,
    visualization_format: Union[
        str, VisualizationFormat
    ] = VisualizationFormat.NONE,  # Default NONE
    strategy: Union[str, GraphStrategy] = GraphStrategy.STANDARD,
    example_entities: Optional[List[Dict[str, Any]]] = None,
    example_relationships: Optional[List[Dict[str, Any]]] = None,
    custom_entity_schema: Optional[List[Dict[str, Any]]] = None,  # Schema is a list of dicts now
    custom_relationship_schema: Optional[
        List[Dict[str, Any]]
    ] = None,  # Schema is a list of dicts now
    existing_graph: Optional[Dict[str, Any]] = None,
    context_window: Optional[int] = None,
    language: Optional[str] = None,
    automatic_coreference: bool = True,  # Applied in Multistage/Chunked merging
    chunk_size: Optional[int] = None,  # In tokens
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    normalize_entities: bool = True,  # Default True
    sort_by: str = "confidence",  # Options: "confidence", "centrality", "mentions"
    max_tokens_per_request: Optional[int] = 4000,  # Default max tokens for LLM response
    enable_reasoning: bool = False,  # Add reasoning steps in prompt
    additional_params: Optional[Dict[str, Any]] = None,  # e.g., temperature
) -> Dict[str, Any]:
    """
    Extracts entities and their relationships from text, building a knowledge graph.

    Analyzes unstructured text to identify entities and semantic relationships, creating a
    structured knowledge graph. Supports multiple strategies, formats, schemas, and visualization.

    Args:
        text: The input text to analyze.
        entity_types: Optional list of entity types to focus on (e.g., ["Person", "Organization"]).
                      If None, extracts all inferred types or types from the schema.
        relation_types: Optional list of relationship types to extract (e.g., ["WORKS_FOR", "LOCATED_IN"]).
                       If None, extracts all inferred types or types from the schema.
        provider: The LLM provider (e.g., "openai", "anthropic", "gemini"). Defaults to "openai".
        model: The specific model ID. If None, the provider's default model is used.
        include_evidence: Whether to include text snippets supporting each relationship. Default True.
        include_attributes: Whether to extract and include entity attributes. Default True.
        include_positions: Whether to include position information for entity mentions. Default False.
        include_temporal_info: Whether to extract temporal context for relationships. Default True.
        max_entities: Maximum number of entities to return after merging/filtering. Default 100.
        max_relations: Maximum number of relations to return after merging/filtering. Default 200.
        min_confidence: Minimum confidence score (0.0-1.0) for relationships. Default 0.6.
        domain: Optional domain for predefined schemas ("business", "academic", "medical", "legal").
        output_format: Desired output format ("json", "networkx", etc.). Default "json".
        visualization_format: Format for visualization ("none", "html", "svg", etc.). Default "none".
        strategy: Extraction strategy ("standard", "multistage", "chunked", etc.). Default "standard".
        example_entities: Optional list of example entities to guide extraction format (few-shot).
        example_relationships: Optional list of example relationships to guide extraction format (few-shot).
        custom_entity_schema: Optional custom schema for entity types and attributes (list of dicts).
        custom_relationship_schema: Optional custom schema for relationship types (list of dicts).
        existing_graph: Optional existing graph data {'entities': [], 'relationships': []} for incremental strategy.
        context_window: Optional max context window size (tokens) for chunking. Estimated if None.
        language: Optional language hint for multilingual extraction (e.g., "Spanish").
        automatic_coreference: Whether to attempt coreference resolution during merging (Multistage/Chunked). Default True.
        chunk_size: Optional custom chunk size (tokens) for chunked strategy. Overrides estimation.
        custom_prompt: Optional custom prompt template. Use placeholders {text}, {instructions}, {schema_info}, {examples}.
        system_prompt: Optional custom system prompt override.
        normalize_entities: Whether to normalize entity names and merge duplicates. Default True.
        sort_by: How to sort final entities/relationships ("confidence", "centrality", "mentions"). Default "confidence".
        max_tokens_per_request: Max tokens for LLM response generation. Default 4000.
        enable_reasoning: Include reasoning steps in the prompt for the LLM. Default False.
        additional_params: Additional provider-specific parameters (e.g., temperature, top_p).

    Returns:
        A dictionary containing the entity graph data and metadata, including:
        - 'entities': List of extracted entity dictionaries.
        - 'relationships': List of extracted relationship dictionaries.
        - 'metadata': Information about the extraction process, counts, types, strategy, etc.
        - 'visualization': Visualization data (e.g., HTML content, file URL) if requested.
        - 'query_interface': Helper functions for graph querying (if networkx available).
        - 'provider', 'model', 'tokens', 'cost', 'processing_time', 'success'.
        - Depending on output_format, may include 'graph' (NetworkX object), 'rdf_triples', etc.

    Raises:
        ToolInputError: If input text is empty or parameters are invalid.
        ProviderError: If the LLM provider fails during extraction.
        ToolError: For parsing errors, chunking failures, or other processing issues.
    """
    start_time = time.time()
    logger.info(f"Starting entity graph extraction with strategy: {strategy}")

    # --- Input Validation ---
    if not text or not isinstance(text, str):
        raise ToolInputError("Input 'text' must be a non-empty string.")
    if entity_types is not None and not isinstance(entity_types, list):
        raise ToolInputError("'entity_types' must be a list of strings or None.")
    if relation_types is not None and not isinstance(relation_types, list):
        raise ToolInputError("'relation_types' must be a list of strings or None.")
    if not (0.0 <= min_confidence <= 1.0):
        raise ToolInputError("'min_confidence' must be between 0.0 and 1.0.")
    if max_entities <= 0 or max_relations <= 0:
        raise ToolInputError("'max_entities' and 'max_relations' must be positive integers.")
    if custom_entity_schema is not None and not isinstance(custom_entity_schema, list):
        raise ToolInputError("'custom_entity_schema' must be a list of dictionaries or None.")
    if custom_relationship_schema is not None and not isinstance(custom_relationship_schema, list):
        raise ToolInputError("'custom_relationship_schema' must be a list of dictionaries or None.")
    if existing_graph is not None and not (
        isinstance(existing_graph, dict)
        and "entities" in existing_graph
        and "relationships" in existing_graph
    ):
        raise ToolInputError(
            "'existing_graph' must be a dictionary with 'entities' and 'relationships' lists."
        )

    # Validate and convert enums
    try:
        output_format = (
            OutputFormat(str(output_format).lower())
            if isinstance(output_format, str)
            else output_format
        )
        visualization_format = (
            VisualizationFormat(str(visualization_format).lower())
            if isinstance(visualization_format, str)
            else visualization_format
        )
        strategy = GraphStrategy(str(strategy).lower()) if isinstance(strategy, str) else strategy
        assert isinstance(output_format, OutputFormat)
        assert isinstance(visualization_format, VisualizationFormat)
        assert isinstance(strategy, GraphStrategy)
    except (ValueError, AssertionError) as e:
        raise ToolInputError(f"Invalid enum value provided: {e}") from e

    # Validate domain
    if domain and domain not in COMMON_SCHEMAS:
        valid_domains = list(COMMON_SCHEMAS.keys())
        raise ToolInputError(f"Invalid domain: '{domain}'. Valid options: {valid_domains}")

    # Check dependencies for strategy/formats
    if strategy == GraphStrategy.INCREMENTAL and existing_graph is None:
        raise ToolInputError("The 'incremental' strategy requires the 'existing_graph' parameter.")
    if strategy == GraphStrategy.STRICT_SCHEMA and not (
        domain or (custom_entity_schema and custom_relationship_schema)
    ):
        raise ToolInputError("The 'strict_schema' strategy requires a 'domain' or custom schemas.")
    if visualization_format != VisualizationFormat.NONE and not HAS_VISUALIZATION_LIBS:
        logger.warning(
            f"Visualization format '{visualization_format.value}' requested, but required libraries "
            f"(networkx, pyvis, matplotlib) are not installed. Falling back to 'none'."
        )
        visualization_format = VisualizationFormat.NONE
    if output_format == OutputFormat.NETWORKX and not HAS_NETWORKX:
        logger.warning(
            "Output format 'networkx' requested, but networkx library is not installed. Falling back to 'json'."
        )
        output_format = OutputFormat.JSON

    # --- Initialize Configuration ---
    try:
        provider_instance: BaseProvider = await get_provider(provider)
    except Exception as e:
        raise ProviderError(
            f"Failed to initialize provider '{provider}': {e}", provider=provider, cause=e
        ) from e

    additional_params = additional_params or {}
    model_name = model or provider_instance.default_model

    # Estimate context window if needed for chunking
    if strategy == GraphStrategy.CHUNKED and not context_window:
        # Basic estimation logic (replace with more sophisticated provider-specific checks if possible)
        model_context_estimates = {
            "gpt-4": 8000,
            "gpt-4-32k": 32000,
            "gpt-3.5-turbo": 4000,
            "gpt-3.5-turbo-16k": 16000,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "claude-2": 100000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "claude-3-5-sonnet": 200000,
            "gemini-pro": 32000,
            "gemini-1.5-pro": 1000000,
            "gemini-1.5-flash": 1000000,
        }
        context_window = 16000  # Default fallback
        if model_name:
            for key, window in model_context_estimates.items():
                if key in model_name.lower():
                    context_window = window
                    break
        logger.info(f"Estimated context window for model '{model_name}': {context_window} tokens.")

    # Determine effective schema
    schema = None
    if domain and domain in COMMON_SCHEMAS:
        schema = COMMON_SCHEMAS[domain]
        logger.info(f"Using predefined schema for domain: {domain}")
    elif custom_entity_schema and custom_relationship_schema:
        schema = {"entities": custom_entity_schema, "relationships": custom_relationship_schema}
        logger.info("Using custom entity and relationship schemas.")
    elif custom_entity_schema:
        schema = {"entities": custom_entity_schema, "relationships": []}  # Allow only entity schema
        logger.info("Using custom entity schema only.")
    elif custom_relationship_schema:
        schema = {
            "entities": [],
            "relationships": custom_relationship_schema,
        }  # Allow only rel schema
        logger.info("Using custom relationship schema only.")

    # --- Prepare Common Arguments for Strategy Functions ---
    common_args = {
        "text": text,
        "provider_instance": provider_instance,
        "model": model_name,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "include_evidence": include_evidence,
        "include_attributes": include_attributes,
        "include_positions": include_positions,
        "include_temporal_info": include_temporal_info,
        "min_confidence": min_confidence,
        "max_entities": max_entities,
        "max_relations": max_relations,
        "schema": schema,
        "custom_prompt": custom_prompt,
        "system_prompt": system_prompt,
        "language": language,
        "example_entities": example_entities,
        "example_relationships": example_relationships,
        "enable_reasoning": enable_reasoning,
        "max_tokens_per_request": max_tokens_per_request,
        "additional_params": additional_params,
    }

    # --- Execute Strategy ---
    extraction_result: Dict[str, Any] = {}
    used_model = model_name  # Default model
    total_tokens = {"input": 0, "output": 0, "total": 0}
    total_cost = 0.0

    try:
        if strategy == GraphStrategy.STANDARD:
            extraction_result = await _perform_standard_extraction(**common_args)
        elif strategy == GraphStrategy.MULTISTAGE:
            extraction_result = await _perform_multistage_extraction(
                automatic_coreference=automatic_coreference, **common_args
            )
        elif strategy == GraphStrategy.CHUNKED:
            extraction_result = await _perform_chunked_extraction(
                context_window=context_window,  # type: ignore # context_window guaranteed if strategy is CHUNKED
                chunk_size=chunk_size,
                automatic_coreference=automatic_coreference,
                **common_args,
            )
        elif strategy == GraphStrategy.INCREMENTAL:
            extraction_result = await _perform_incremental_extraction(
                existing_graph=existing_graph,  # type: ignore # existing_graph guaranteed if strategy is INCREMENTAL
                # Incremental doesn't use examples directly in the same way
                example_entities=None,
                example_relationships=None,
                **common_args,  # Pass other args, but they might be less relevant
            )
        elif strategy == GraphStrategy.STRUCTURED:
            # Ensure examples are provided or generated if possible
            if not example_entities and domain and domain in FEW_SHOT_EXAMPLES:
                common_args["example_entities"] = FEW_SHOT_EXAMPLES[domain]["entities"]
                common_args["example_relationships"] = FEW_SHOT_EXAMPLES[domain]["relationships"]
                logger.info(f"Using few-shot examples for domain: {domain}")
            elif not example_entities:
                logger.warning(
                    "Structured strategy selected but no examples provided or found for domain. Extraction quality may suffer."
                )
            extraction_result = await _perform_structured_extraction(**common_args)

        elif strategy == GraphStrategy.STRICT_SCHEMA:
            if not schema:  # Should have been caught earlier, but double-check
                raise ToolInputError("Strict schema strategy requires a schema (domain or custom).")
            # Remove schema from common_args to avoid passing it twice
            schema_copy = schema.copy()
            schema_args = common_args.copy()
            if "schema" in schema_args:
                del schema_args["schema"]
            extraction_result = await _perform_schema_guided_extraction(
                schema=schema_copy,
                # Strict schema doesn't use generic examples
                example_entities=None,
                example_relationships=None,
                **schema_args,
            )

        # Get usage stats from the result
        used_model = extraction_result.get("model", used_model)
        total_tokens = extraction_result.get("tokens", total_tokens)
        total_cost = extraction_result.get("cost", total_cost)

    except (ProviderError, ToolError) as e:
        logger.error(f"Entity graph extraction failed: {e}", exc_info=True)
        raise  # Re-raise ProviderError or ToolError
    except Exception as e:
        logger.error(f"An unexpected error occurred during extraction: {e}", exc_info=True)
        raise ToolError(f"Extraction failed due to an unexpected error: {e}") from e

    # --- Post-processing ---
    logger.info("Performing post-processing steps...")
    # Validate and clean final data (important after merging/incremental)
    extraction_result = _validate_graph_data(
        extraction_result,
        min_confidence=min_confidence,
        max_entities=max_entities,
        max_relations=max_relations,
    )

    # Normalize entity names if requested
    if normalize_entities:
        logger.info("Normalizing entities...")
        extraction_result = _normalize_entities(extraction_result)
        # Re-validate after normalization as IDs might change
        extraction_result = _validate_graph_data(
            extraction_result,
            min_confidence=min_confidence,
            max_entities=max_entities,
            max_relations=max_relations,
        )

    # Add computed graph metrics and sort
    logger.info(f"Calculating graph metrics and sorting by '{sort_by}'...")
    extraction_result = _add_graph_metrics(extraction_result, sort_by)

    # Generate visualization if requested
    visualization_data = None
    if visualization_format != VisualizationFormat.NONE:
        logger.info(f"Generating visualization in format: {visualization_format.value}")
        visualization_data = _generate_visualization(extraction_result, visualization_format)
        if visualization_data and "error" in visualization_data:
            logger.warning(f"Visualization generation failed: {visualization_data['error']}")

    # Create query interface
    query_interface = None
    if HAS_NETWORKX:
        logger.info("Creating query interface...")
        query_interface = _create_query_interface(extraction_result)

    processing_time = time.time() - start_time

    # Format output according to requested format
    logger.info(f"Formatting output as: {output_format.value}")
    formatted_result = _format_output(extraction_result, output_format)

    # --- Prepare Final Result ---
    final_result = {
        "entities": formatted_result.get("entities", []),
        "relationships": formatted_result.get("relationships", []),
        "metadata": {
            "entity_count": len(formatted_result.get("entities", [])),
            "relationship_count": len(formatted_result.get("relationships", [])),
            "entity_types": sorted(
                list(set(e.get("type", "Unknown") for e in formatted_result.get("entities", [])))
            ),
            "relation_types": sorted(
                list(
                    set(r.get("type", "Unknown") for r in formatted_result.get("relationships", []))
                )
            ),
            "processing_strategy": strategy.value,
            "extraction_date": datetime.now(timezone.utc).isoformat() + "Z",
            "schema_used": "domain: " + domain if domain else ("custom" if schema else "none"),
            "metrics": formatted_result.get(
                "metrics", extraction_result.get("metrics")
            ),  # Include metrics if calculated
            "incremental_stats": extraction_result.get(
                "incremental_stats"
            ),  # Include if incremental
        },
        "provider": provider,
        "model": used_model,
        "tokens": total_tokens,
        "cost": total_cost,
        "processing_time": round(processing_time, 2),
        "success": True,
    }

    # Add format-specific data
    if output_format == OutputFormat.NETWORKX and "graph" in formatted_result:
        final_result["graph"] = formatted_result["graph"]
    elif output_format == OutputFormat.RDF and "rdf_triples" in formatted_result:
        final_result["rdf_triples"] = formatted_result["rdf_triples"]
    elif output_format == OutputFormat.CYTOSCAPE and "cytoscape" in formatted_result:
        final_result["cytoscape"] = formatted_result["cytoscape"]
    elif output_format == OutputFormat.D3 and "d3" in formatted_result:
        final_result["d3"] = formatted_result["d3"]
    elif output_format == OutputFormat.NEO4J and "neo4j_queries" in formatted_result:
        final_result["neo4j_queries"] = formatted_result["neo4j_queries"]

    # Add visualization if generated
    if visualization_data:
        final_result["visualization"] = visualization_data

    # Add query interface if generated
    if query_interface:
        final_result["query_interface"] = query_interface

    # Log success
    logger.success(
        f"Entity graph extraction completed successfully ({final_result['metadata']['entity_count']} entities, "
        f"{final_result['metadata']['relationship_count']} relationships)",
        strategy=strategy.value,
        model=used_model,
        cost=f"{total_cost:.6f}",
        time=f"{processing_time:.2f}s",
    )

    return final_result


# --- Helper: Robust JSON Parsing ---
def _parse_json_from_response(response_text: str) -> Dict[str, Any]:
    """Attempts to parse JSON from the LLM response text, handling various formats."""
    try:
        # 1. Try direct parsing (if response is pure JSON)
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from response: {e}", exc_info=True)
        # 2. Try finding JSON within markdown code blocks (```json ... ```)
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL | re.IGNORECASE
        )
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass  # Fall through to next method

        # 3. Try finding the largest JSON object/array within the text
        # Be careful with greedy matching, find first '{' and last '}'
        start_index = response_text.find("{")
        end_index = response_text.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            potential_json = response_text[start_index : end_index + 1]
            try:
                # Validate structure minimally before parsing
                if potential_json.count("{") == potential_json.count("}"):
                    return json.loads(potential_json)
            except json.JSONDecodeError:
                pass  # Fall through

        # 4. If nothing worked, raise an error
        raise ValueError("No valid JSON object found in the response text.") from e


# --- Helper: Build Common Prompt Sections ---
def _build_common_prompt_instructions(
    entity_types: Optional[List[str]],
    relation_types: Optional[List[str]],
    schema: Optional[Dict[str, Any]],
    language: Optional[str],
    enable_reasoning: bool,
    format_structure: str,
    include_positions: bool,
    include_attributes: bool,
    include_evidence: bool,
    include_temporal_info: bool,
    min_confidence: float,
    max_entities: int,
    max_relations: int,
    examples: Optional[str] = None,
    task_description: str = "Extract entities and their relationships from the text below.",
) -> str:
    """Builds the common instruction part of the LLM prompt."""

    # Type constraints
    entity_types_str = (
        "ENTITY TYPES to extract (if specified):\n- " + "\n- ".join(entity_types)
        if entity_types
        else "Extract relevant entity types."
    )
    relation_types_str = (
        "RELATIONSHIP TYPES to extract (if specified):\n- " + "\n- ".join(relation_types)
        if relation_types
        else "Extract relevant relationship types."
    )

    # Schema guidance
    schema_guidance = ""
    if schema:
        schema_guidance = "SCHEMA:\n"
        ent_schema = schema.get("entities")
        rel_schema = schema.get("relationships")
        if ent_schema:
            schema_guidance += "Entity Types:\n"
            for et in ent_schema:
                attrs = (
                    f" (Attributes: {', '.join(et.get('attributes', []))})"
                    if et.get("attributes")
                    else ""
                )
                schema_guidance += f"- {et.get('type', 'Unknown')}{attrs}\n"
        if rel_schema:
            schema_guidance += "Relationship Types:\n"
            for rt in rel_schema:
                src = ", ".join(rt.get("source_types", ["Any"]))
                tgt = ", ".join(rt.get("target_types", ["Any"]))
                schema_guidance += f"- {rt.get('type', 'Unknown')} (From: {src}, To: {tgt})\n"
        schema_guidance += "\n"

    # Language
    language_instruction = (
        f"The text is in {language}. Adapt extraction accordingly.\n" if language else ""
    )

    # Reasoning steps
    reasoning_instruction = ""
    if enable_reasoning:
        reasoning_instruction = """
REASONING STEPS (Think step-by-step before generating the final JSON):
1. Identify potential entity mentions.
2. Group mentions referring to the same real-world entity.
3. Assign a unique ID and the most appropriate type (from schema/list if provided) to each entity.
4. Extract specified attributes for each entity.
5. Identify explicit and strongly implied relationships between extracted entities.
6. Assign the most appropriate type (from schema/list if provided) to each relationship.
7. Determine directionality (source -> target).
8. Extract evidence and temporal information if requested.
9. Estimate confidence (0.0-1.0) for each relationship.
10. Format the final output strictly as requested JSON.\n
"""
    # Output format definition
    format_instructions = f"""
OUTPUT FORMAT:
Respond with a valid JSON object containing two keys: "entities" and "relationships".
Adhere STRICTLY to this structure:
```json
{{
  "entities": [
    {{
      "id": "ent<unique_number>",
      "name": "entity_name",
      "type": "entity_type"
      {', "mentions": [{"text": "mention_text", "pos": [start_pos, end_pos]}]' if include_positions else ""}
      {', "attributes": {{ "attr_name": "value", ... }}' if include_attributes else ""}
    }}
    // ... more entities
  ],
  "relationships": [
    {{
      "id": "rel<unique_number>",
      "source": "source_entity_id", // Must be an ID from the 'entities' list
      "target": "target_entity_id", // Must be an ID from the 'entities' list
      "type": "relationship_type"
      {', "confidence": 0.xx'} // Confidence score 0.0-1.0
      {', "evidence": "text snippet supporting the relationship"' if include_evidence else ""}
      {', "temporal": {{ "start": "...", "end": "...", "point": "..." }}' if include_temporal_info else ""}
    }}
    // ... more relationships
  ]
}}
```

CONSTRAINTS:
- Generate unique IDs for all entities and relationships (e.g., "ent1", "rel1").
- Only include relationships with confidence >= {min_confidence:.2f}.
- Limit results to approximately {max_entities} most important entities and {max_relations} most significant relationships.
- Ensure all relationship `source` and `target` IDs exist in the `entities` list.
- If using a schema, strictly adhere to the defined types and attributes.
{examples if examples else ""}
"""

    # Combine parts
    full_instructions = f"""
{task_description}

{entity_types_str}
{relation_types_str}

{schema_guidance}
{language_instruction}
{reasoning_instruction}
{format_instructions}
"""
    return full_instructions.strip()


# --- Helper: Call LLM via generate_completion ---
async def _call_llm_for_extraction(
    prompt: str,
    system_prompt: Optional[str],
    provider_instance: BaseProvider,
    model: str,
    max_tokens_per_request: Optional[int],
    additional_params: Optional[Dict[str, Any]],
    task_name: str,  # For logging/error reporting
) -> Dict[str, Any]:
    """Helper to call the LLM and handle basic response validation."""
    logger.debug(f"Calling LLM for {task_name}. Prompt length: {len(prompt)} chars.")
    # Set low temperature for deterministic extraction by default
    temp = additional_params.pop("temperature", 0.1) if additional_params else 0.1

    # If system_prompt is provided, prepend it to the prompt
    if system_prompt:
        # Combine system prompt and user prompt
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
    else:
        full_prompt = prompt

    try:
        # Call generate_completion without system_prompt parameter
        completion_result = await generate_completion(
            prompt=full_prompt,
            model=model,
            provider=provider_instance.provider_name,  # Use provider name string
            temperature=temp,
            max_tokens=max_tokens_per_request,
            additional_params=additional_params,
        )

        if not completion_result.get("success", False):
            error_message = completion_result.get("error", f"Unknown error during {task_name}")
            logger.error(f"{task_name} failed via generate_completion: {error_message}")
            raise ProviderError(
                f"{task_name} failed: {error_message}",
                provider=provider_instance.provider_name,
                model=model,
                details=completion_result.get("details"),
            )

        # Parse JSON robustly
        try:
            graph_data = _parse_json_from_response(completion_result["text"])
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from {task_name} response: {e}", exc_info=True)
            raise ToolError(
                f"Failed to parse JSON from {task_name}: {e}",
                error_code="PARSING_ERROR",
                details={
                    "response_text": completion_result["text"][:1000]
                },  # Log truncated response
            ) from e

        # Add metadata to the result
        graph_data["model"] = completion_result.get("model", model)
        graph_data["tokens"] = completion_result.get(
            "tokens", {"input": 0, "output": 0, "total": 0}
        )
        graph_data["cost"] = completion_result.get("cost", 0.0)

        return graph_data

    except ProviderError as e:
        # Catch provider errors from generate_completion or raised above
        raise e  # Reraise ProviderError
    except ToolError as e:
        # Catch parsing errors raised above
        raise e  # Reraise ToolError
    except Exception as e:
        # Catch any other unexpected errors during the call or parsing
        logger.error(f"Unexpected error during {task_name} LLM call: {e}", exc_info=True)
        raise ProviderError(
            f"Unexpected error during {task_name} for model '{model}': {e}",
            provider=provider_instance.provider_name,
            model=model,
        ) from e


# --- Strategy Implementation Functions ---


def _validate_graph_data(
    graph_data: Dict[str, Any], min_confidence: float, max_entities: int, max_relations: int
) -> Dict[str, Any]:
    """Validates, cleans, and standardizes extracted graph data."""
    logger.debug("Validating and cleaning extracted graph data...")
    entities = graph_data.get("entities", [])
    relationships = graph_data.get("relationships", [])

    if not isinstance(entities, list):
        logger.warning(
            f"Entities data is not a list, received {type(entities)}. Resetting to empty list."
        )
        entities = []
    if not isinstance(relationships, list):
        logger.warning(
            f"Relationships data is not a list, received {type(relationships)}. Resetting to empty list."
        )
        relationships = []

    valid_entities_map: Dict[str, Dict[str, Any]] = {}
    entity_id_counter = 1
    processed_ids = set()

    # Validate and process entities
    for entity in entities:
        if not isinstance(entity, dict) or "name" not in entity or "type" not in entity:
            logger.warning(f"Skipping invalid entity format: {entity}")
            continue

        entity_id = entity.get("id")
        # Ensure ID exists and is unique, generate if needed
        if not entity_id or not isinstance(entity_id, str) or entity_id in processed_ids:
            entity_id = f"ent{entity_id_counter}"
            while entity_id in processed_ids:
                entity_id_counter += 1
                entity_id = f"ent{entity_id_counter}"
        entity_id_counter = max(
            entity_id_counter, int(re.sub(r"\D", "", entity_id) or "0") + 1
        )  # Keep counter high

        processed_ids.add(entity_id)
        entity["id"] = entity_id

        # Validate mentions
        if "mentions" in entity:
            if not isinstance(entity["mentions"], list):
                entity["mentions"] = []
            else:
                valid_mentions = []
                for mention in entity["mentions"]:
                    if isinstance(mention, dict) and "text" in mention:
                        if "pos" in mention:
                            pos = mention["pos"]
                            if not (
                                isinstance(pos, list)
                                and len(pos) == 2
                                and all(isinstance(p, (int, float)) for p in pos)
                            ):
                                logger.debug(
                                    f"Invalid position format for mention '{mention.get('text')}' in entity {entity_id}. Removing position."
                                )
                                mention.pop("pos", None)  # Remove invalid pos
                        valid_mentions.append(mention)
                entity["mentions"] = valid_mentions

        # Validate attributes
        if "attributes" in entity:
            if not isinstance(entity["attributes"], dict):
                entity["attributes"] = {}
            else:
                # Clean attribute values
                cleaned_attrs = {}
                for key, value in entity["attributes"].items():
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_attrs[key] = value
                    elif value is not None:
                        try:
                            cleaned_attrs[key] = str(value)  # Convert other types to string
                        except Exception:
                            logger.debug(
                                f"Could not convert attribute value for key '{key}' in entity {entity_id}. Skipping attribute."
                            )
                entity["attributes"] = cleaned_attrs

        valid_entities_map[entity_id] = entity

    # Limit entities if needed (do this before relationship validation)
    final_entities = list(valid_entities_map.values())
    if len(final_entities) > max_entities:
        logger.info(f"Reducing entities from {len(final_entities)} to {max_entities} (limit).")
        # Basic limiting, could be smarter (e.g., based on centrality if available later)
        final_entities = final_entities[:max_entities]
        valid_entity_ids = {e["id"] for e in final_entities}
    else:
        valid_entity_ids = set(valid_entities_map.keys())

    # Validate and process relationships
    valid_relationships = []
    rel_id_counter = 1
    processed_rel_ids = set()
    relationship_signatures = set()  # (source_id, target_id, type)

    for rel in relationships:
        if (
            not isinstance(rel, dict)
            or "source" not in rel
            or "target" not in rel
            or "type" not in rel
        ):
            logger.warning(f"Skipping invalid relationship format: {rel}")
            continue

        source_id = rel["source"]
        target_id = rel["target"]
        rel_type = rel["type"]

        # Check if source and target entities are valid *after* entity limiting
        if source_id not in valid_entity_ids or target_id not in valid_entity_ids:
            logger.debug(f"Skipping relationship referencing invalid/removed entity: {rel}")
            continue

        # Check confidence
        confidence = rel.get("confidence")
        try:
            confidence = (
                float(confidence) if confidence is not None else 1.0
            )  # Default to 1.0 if missing
            if not (0.0 <= confidence <= 1.0):
                confidence = max(0.0, min(1.0, confidence))  # Clamp to range
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid confidence '{confidence}' for relationship {rel.get('id')}. Setting to 0.5."
            )
            confidence = 0.5
        rel["confidence"] = confidence

        if confidence < min_confidence:
            logger.debug(
                f"Skipping relationship below confidence threshold ({confidence:.2f} < {min_confidence:.2f}): {rel}"
            )
            continue

        # Check for duplicates based on (source, target, type)
        signature = (source_id, target_id, rel_type)
        if signature in relationship_signatures:
            logger.debug(f"Skipping duplicate relationship: {signature}")
            continue
        relationship_signatures.add(signature)

        # Ensure ID exists and is unique, generate if needed
        rel_id = rel.get("id")
        if not rel_id or not isinstance(rel_id, str) or rel_id in processed_rel_ids:
            rel_id = f"rel{rel_id_counter}"
            while rel_id in processed_rel_ids:
                rel_id_counter += 1
                rel_id = f"rel{rel_id_counter}"
        rel_id_counter = max(rel_id_counter, int(re.sub(r"\D", "", rel_id) or "0") + 1)

        processed_rel_ids.add(rel_id)
        rel["id"] = rel_id

        # Validate evidence
        if "evidence" in rel and not isinstance(rel["evidence"], str):
            try:
                rel["evidence"] = str(rel["evidence"])
            except Exception:
                rel.pop("evidence")  # Remove invalid evidence

        # Validate temporal info
        if "temporal" in rel:
            if not isinstance(rel["temporal"], dict):
                rel.pop("temporal")
            else:
                # Ensure values are simple types
                cleaned_temporal = {}
                for key, value in rel["temporal"].items():
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_temporal[key] = value
                rel["temporal"] = cleaned_temporal if cleaned_temporal else None
                if rel["temporal"] is None:
                    rel.pop("temporal")

        valid_relationships.append(rel)

    # Limit relationships if needed
    if len(valid_relationships) > max_relations:
        logger.info(
            f"Reducing relationships from {len(valid_relationships)} to {max_relations} (limit)."
        )
        # Sort by confidence before limiting
        valid_relationships.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
        valid_relationships = valid_relationships[:max_relations]

    # Return validated data, ensuring the keys exist even if empty
    validated_data = {"entities": final_entities, "relationships": valid_relationships}
    # Carry over other keys like model, tokens, cost if they exist
    for key in graph_data:
        if key not in validated_data:
            validated_data[key] = graph_data[key]

    logger.debug(
        f"Validation complete. Entities: {len(validated_data['entities'])}, Relationships: {len(validated_data['relationships'])}"
    )
    return validated_data


async def _perform_standard_extraction(**kwargs) -> Dict[str, Any]:
    """Performs extraction in a single step."""
    logger.info("Performing standard extraction...")
    # Build prompt using common helper
    instructions = _build_common_prompt_instructions(
        task_description="Extract entities and relationships from the text below.",
        format_structure="STANDARD",  # Placeholder, actual structure is defined inside helper
        **{
            k: kwargs[k]
            for k in [
                "entity_types",
                "relation_types",
                "schema",
                "language",
                "enable_reasoning",
                "include_positions",
                "include_attributes",
                "include_evidence",
                "include_temporal_info",
                "min_confidence",
                "max_entities",
                "max_relations",
            ]
        },
    )

    # Handle custom prompt template
    if kwargs.get("custom_prompt"):
        prompt = kwargs["custom_prompt"].format(
            text=kwargs["text"],
            instructions=instructions,
            schema_info=kwargs.get("schema", ""),  # Provide schema if needed by template
            examples=kwargs.get("example_entities", ""),  # Provide examples if needed
        )
    else:
        prompt = f"{instructions}\n\nTEXT TO ANALYZE:\n{kwargs['text']}"

    # Determine system prompt
    sys_prompt = kwargs.get("system_prompt") or SYSTEM_PROMPTS.get("standard_extraction")

    # Call LLM
    graph_data = await _call_llm_for_extraction(
        prompt=prompt,
        system_prompt=sys_prompt,
        provider_instance=kwargs["provider_instance"],
        model=kwargs["model"],
        max_tokens_per_request=kwargs["max_tokens_per_request"],
        additional_params=kwargs["additional_params"],
        task_name="Standard Extraction",
    )

    return graph_data  # Validation happens in the main function


async def _perform_multistage_extraction(automatic_coreference: bool, **kwargs) -> Dict[str, Any]:
    """Performs extraction in two stages: entities first, then relationships."""
    logger.info("Performing multi-stage extraction...")
    total_tokens = {"input": 0, "output": 0, "total": 0}
    total_cost = 0.0
    model_used = kwargs["model"]  # Assume same model for both stages

    # --- Stage 1: Entity Extraction ---
    logger.info("Multi-stage: Starting entity extraction...")
    entity_instructions = _build_common_prompt_instructions(
        task_description="Extract entities from the text below.",
        relation_types=None,  # No relationships in this stage
        include_evidence=False,
        include_temporal_info=False,  # Not relevant for entities only
        max_relations=0,  # No relationships
        format_structure="ENTITIES_ONLY",  # Needs adaptation in helper or specific format here
        **{
            k: kwargs[k]
            for k in [
                "entity_types",
                "schema",
                "language",
                "enable_reasoning",
                "include_positions",
                "include_attributes",
                "min_confidence",
                "max_entities",
                # Pass min_confidence and max_entities to filter/limit early if desired
            ]
        },
    )
    # Adjust format instruction specifically for entities
    entity_instructions = entity_instructions.replace(
        'Respond with a valid JSON object containing two keys: "entities" and "relationships".',
        'Respond with a valid JSON object containing ONLY the key "entities".',
    )
    entity_instructions = re.sub(
        r'"relationships":\s*\[.*?\]\s*\}', "}", entity_instructions, flags=re.DOTALL
    )

    # Handle custom prompt template (assuming it can handle stages)
    if kwargs.get("custom_prompt"):
        entity_prompt = kwargs["custom_prompt"].format(
            text=kwargs["text"], instructions=entity_instructions, stage="entities"
        )
    else:
        entity_prompt = f"{entity_instructions}\n\nTEXT TO ANALYZE:\n{kwargs['text']}"

    entity_sys_prompt = kwargs.get("system_prompt") or SYSTEM_PROMPTS.get("entity_detection")

    entity_result = await _call_llm_for_extraction(
        prompt=entity_prompt,
        system_prompt=entity_sys_prompt,
        provider_instance=kwargs["provider_instance"],
        model=kwargs["model"],
        max_tokens_per_request=kwargs["max_tokens_per_request"],
        additional_params=kwargs["additional_params"],
        task_name="Entity Extraction (Multi-stage)",
    )

    entities = entity_result.get("entities", [])
    if not isinstance(entities, list):
        entities = []

    total_tokens["input"] += entity_result.get("tokens", {}).get("input", 0)
    total_tokens["output"] += entity_result.get("tokens", {}).get("output", 0)
    total_cost += entity_result.get("cost", 0.0)
    model_used = entity_result.get("model", model_used)

    if not entities:
        logger.warning("Multi-stage: No entities found in the first stage.")
        return {
            "entities": [],
            "relationships": [],
            "model": model_used,
            "tokens": total_tokens,
            "cost": total_cost,
        }

    # Preliminary validation/cleaning of entities
    temp_validated = _validate_graph_data(
        {"entities": entities}, 0.0, kwargs["max_entities"] * 2, 0
    )  # Higher limit before rel stage
    entities = temp_validated["entities"]
    if not entities:
        logger.warning("Multi-stage: No valid entities after preliminary validation.")
        return {
            "entities": [],
            "relationships": [],
            "model": model_used,
            "tokens": total_tokens,
            "cost": total_cost,
        }
    logger.info(f"Multi-stage: Found {len(entities)} potential entities.")

    # --- Stage 2: Relationship Extraction ---
    logger.info("Multi-stage: Starting relationship extraction...")
    # Format entity list for the relationship prompt
    entity_list_str = "EXTRACTED ENTITIES:\n"
    entity_id_map = {}
    for i, entity in enumerate(entities):
        entity_id = entity.get("id", f"temp_ent{i + 1}")
        entity["id"] = entity_id  # Ensure ID exists
        entity_id_map[entity_id] = entity
        entity_list_str += f"- ID: {entity_id}, Name: {entity.get('name', 'N/A')}, Type: {entity.get('type', 'N/A')}\n"

    relationship_instructions = _build_common_prompt_instructions(
        task_description="Identify relationships BETWEEN the provided entities based on the text.",
        entity_types=None,  # Entities are provided, not extracted here
        include_positions=False,
        include_attributes=False,  # Not relevant for relationships only
        max_entities=0,  # No new entities
        format_structure="RELATIONSHIPS_ONLY",  # Needs adaptation
        **{
            k: kwargs[k]
            for k in [
                "relation_types",
                "schema",
                "language",
                "enable_reasoning",
                "include_evidence",
                "include_temporal_info",
                "min_confidence",
                "max_relations",
            ]
        },
    )
    # Adjust format instruction specifically for relationships
    relationship_instructions = relationship_instructions.replace(
        'Respond with a valid JSON object containing two keys: "entities" and "relationships".',
        'Respond with a valid JSON object containing ONLY the key "relationships". Reference the provided entity IDs.',
    )
    relationship_instructions = re.sub(
        r'"entities":\s*\[.*?\]\s*,?\s*', "", relationship_instructions, flags=re.DOTALL
    )

    # Handle custom prompt template
    if kwargs.get("custom_prompt"):
        relationship_prompt = kwargs["custom_prompt"].format(
            text=kwargs["text"],
            instructions=relationship_instructions,
            entities=entity_list_str,  # Provide entities if template uses it
            stage="relationships",
        )
    else:
        relationship_prompt = (
            f"{entity_list_str}\n{relationship_instructions}\n\nTEXT TO ANALYZE:\n{kwargs['text']}"
        )

    relationship_sys_prompt = kwargs.get("system_prompt") or SYSTEM_PROMPTS.get(
        "relationship_detection"
    )

    relationship_result = await _call_llm_for_extraction(
        prompt=relationship_prompt,
        system_prompt=relationship_sys_prompt,
        provider_instance=kwargs["provider_instance"],
        model=kwargs["model"],
        max_tokens_per_request=kwargs["max_tokens_per_request"],
        additional_params=kwargs["additional_params"],
        task_name="Relationship Extraction (Multi-stage)",
    )

    relationships = relationship_result.get("relationships", [])
    if not isinstance(relationships, list):
        relationships = []

    total_tokens["input"] += relationship_result.get("tokens", {}).get("input", 0)
    total_tokens["output"] += relationship_result.get("tokens", {}).get("output", 0)
    total_tokens["total"] = total_tokens["input"] + total_tokens["output"]
    total_cost += relationship_result.get("cost", 0.0)
    model_used = relationship_result.get("model", model_used)  # Update model if different

    logger.info(f"Multi-stage: Found {len(relationships)} potential relationships.")

    # Combine results (Validation happens in the main function)
    # Note: Coreference resolution would happen here or during merging if chunked
    combined_result = {
        "entities": entities,
        "relationships": relationships,
        "model": model_used,
        "tokens": total_tokens,
        "cost": total_cost,
    }

    return combined_result


async def _perform_chunked_extraction(
    context_window: int, chunk_size: Optional[int], automatic_coreference: bool, **kwargs
) -> Dict[str, Any]:
    """Chunks large text, processes chunks, and merges results."""
    logger.info("Performing chunked extraction...")
    text = kwargs["text"]
    provider_instance = kwargs["provider_instance"]  # noqa: F841
    model = kwargs["model"]

    # --- Chunking ---
    # Estimate chunk size if not provided
    effective_chunk_size = chunk_size
    if not effective_chunk_size:
        # Estimate required tokens for prompt boilerplate + response
        # This is very approximate
        prompt_overhead = 1000  # Tokens for instructions, schema, etc.
        response_allowance = 1500  # Tokens expected in response
        available_for_text = context_window - prompt_overhead - response_allowance
        if available_for_text <= 100:  # Need at least some text
            raise ToolInputError(
                f"Estimated context window ({context_window}) too small for chunking with overhead."
            )
        # Use 80% of available space for text chunk to be safe
        effective_chunk_size = int(available_for_text * 0.8)
        logger.info(
            f"Calculated chunk size: {effective_chunk_size} tokens (based on context window {context_window})"
        )
    effective_chunk_size = max(500, min(effective_chunk_size, 16000))  # Apply reasonable bounds

    try:
        # Use semantic chunking if possible, fall back to simple
        chunks = await chunk_document_standalone(
            document=text,
            chunk_size=effective_chunk_size,
            chunk_overlap=int(effective_chunk_size * 0.1),  # 10% overlap
            chunk_method="semantic",  # Or "token" / "recursive" as fallback
        )
        if not chunks:
            raise ToolError(
                "Document chunking resulted in zero chunks.", error_code="CHUNKING_ERROR"
            )
        logger.info(
            f"Chunked document into {len(chunks)} chunks (size ~{effective_chunk_size} tokens, overlap ~{int(effective_chunk_size * 0.1)} tokens)."
        )
    except Exception as e:
        logger.error(
            f"Chunking failed: {e}. Attempting standard extraction on the whole text.",
            exc_info=True,
        )
        # Fallback to standard extraction on the entire text
        return await _perform_standard_extraction(**kwargs)

    # --- Process Chunks ---
    chunk_results = []
    total_tokens = {"input": 0, "output": 0, "total": 0}
    total_cost = 0.0
    model_used = model  # Assume same model

    for i, chunk_text in enumerate(chunks):
        logger.info(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_kwargs = {
            **kwargs,
            "text": chunk_text,
            # Use slightly higher limits per chunk, merge will apply final limits
            "max_entities": kwargs["max_entities"] * 2,
            "max_relations": kwargs["max_relations"] * 2,
        }
        try:
            # Use standard extraction for each chunk
            # Could potentially use multistage here if desired for better accuracy per chunk
            result = await _perform_standard_extraction(**chunk_kwargs)

            # Add chunk index for potential merging logic later
            result["_chunk_index"] = i
            chunk_results.append(result)

            # Accumulate usage stats
            total_tokens["input"] += result.get("tokens", {}).get("input", 0)
            total_tokens["output"] += result.get("tokens", {}).get("output", 0)
            total_cost += result.get("cost", 0.0)
            model_used = result.get("model", model_used)  # Update model if it changed

        except (ProviderError, ToolError) as e:
            logger.warning(f"Failed to process chunk {i + 1}: {e}. Skipping chunk.")
        except Exception as e:
            logger.error(
                f"Unexpected error processing chunk {i + 1}: {e}. Skipping chunk.", exc_info=True
            )

    total_tokens["total"] = total_tokens["input"] + total_tokens["output"]

    if not chunk_results:
        raise ToolError(
            "All document chunks failed to process.", error_code="CHUNK_PROCESSING_FAILURE"
        )

    # --- Merge Results ---
    logger.info(f"Merging results from {len(chunk_results)} processed chunks...")
    merged_result = _merge_chunk_results(
        chunk_results,
        max_entities=kwargs["max_entities"],
        max_relations=kwargs["max_relations"],
        min_confidence=kwargs["min_confidence"],
        perform_coreference=automatic_coreference,  # Pass flag
    )

    # Add aggregated usage stats back
    merged_result["model"] = model_used
    merged_result["tokens"] = total_tokens
    merged_result["cost"] = total_cost

    return merged_result  # Validation happens in the main function


def _merge_chunk_results(
    chunk_results: List[Dict[str, Any]],
    max_entities: int,
    max_relations: int,
    min_confidence: float,
    perform_coreference: bool,  # Flag for simple coref
) -> Dict[str, Any]:
    """Merges graph data from multiple chunks, handling duplicates and overlaps."""
    merged_entities: Dict[str, Dict[str, Any]] = {}  # Map normalized name -> entity dict
    merged_relationships: List[Dict[str, Any]] = []
    entity_id_map: Dict[Tuple[int, str], str] = {}  # Map (chunk_idx, old_id) -> new_merged_id
    rel_signatures: Set[Tuple[str, str, str]] = set()  # (source_id, target_id, type)
    entity_counter = 1
    rel_counter = 1

    # Pass 1: Merge Entities and build ID map
    logger.debug(f"Merging entities from {len(chunk_results)} chunks...")
    for i, result in enumerate(chunk_results):
        entities = result.get("entities", [])
        if not isinstance(entities, list):
            continue

        for entity in entities:
            if (
                not isinstance(entity, dict)
                or "name" not in entity
                or "type" not in entity
                or "id" not in entity
            ):
                continue

            original_id = entity["id"]
            original_chunk = i
            entity_name = entity.get("name", "").strip()
            entity_type = entity.get("type")
            normalized_name = entity_name.lower()

            # Key for merging: normalized name + type (can be refined)
            merge_key = (normalized_name, entity_type)

            # Basic Coreference: If perform_coreference is True, try merging based on name only
            # This is very simplistic and might over-merge. More advanced coref needed for better results.
            if perform_coreference:
                merge_key = normalized_name  # Merge based on name only if coref is enabled

            existing_entity = merged_entities.get(merge_key)

            if existing_entity:
                # Merge into existing entity
                merged_id = existing_entity["id"]
                entity_id_map[(original_chunk, original_id)] = merged_id

                # Merge mentions (simple union by text)
                existing_mentions = {
                    m["text"] for m in existing_entity.get("mentions", []) if "text" in m
                }
                for mention in entity.get("mentions", []):
                    if (
                        isinstance(mention, dict)
                        and "text" in mention
                        and mention["text"] not in existing_mentions
                    ):
                        existing_entity.setdefault("mentions", []).append(mention)
                        existing_mentions.add(mention["text"])

                # Merge attributes (simple update, new values overwrite old)
                if "attributes" in entity and isinstance(entity["attributes"], dict):
                    existing_entity.setdefault("attributes", {}).update(entity["attributes"])
            else:
                # Add as new entity
                new_id = f"ent{entity_counter}"
                entity_counter += 1
                entity["id"] = new_id  # Assign new final ID
                merged_entities[merge_key] = entity
                entity_id_map[(original_chunk, original_id)] = new_id

    final_merged_entities = list(merged_entities.values())
    logger.debug(f"Merged into {len(final_merged_entities)} unique entities.")

    # Pass 2: Merge Relationships using the new entity IDs
    logger.debug(f"Merging relationships from {len(chunk_results)} chunks...")
    for i, result in enumerate(chunk_results):
        relationships = result.get("relationships", [])
        if not isinstance(relationships, list):
            continue

        for rel in relationships:
            if (
                not isinstance(rel, dict)
                or "source" not in rel
                or "target" not in rel
                or "type" not in rel
            ):
                continue

            original_source = rel["source"]
            original_target = rel["target"]
            rel_type = rel["type"]

            # Map old chunk-local IDs to new merged IDs
            new_source_id = entity_id_map.get((i, original_source))
            new_target_id = entity_id_map.get((i, original_target))

            if not new_source_id or not new_target_id:
                logger.debug(f"Skipping relationship due to unmapped entity ID(s): {rel}")
                continue  # Skip if source/target entity wasn't merged

            # Check signature for duplicates
            signature = (new_source_id, new_target_id, rel_type)
            if signature in rel_signatures:
                logger.debug(f"Skipping duplicate relationship signature: {signature}")
                continue

            # Check confidence
            confidence = rel.get("confidence", 1.0)  # Assume 1.0 if missing
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = 0.5  # Default on error
            if confidence < min_confidence:
                logger.debug(f"Skipping relationship below confidence threshold: {rel}")
                continue

            # Add relationship with new IDs and new unique ID
            rel["source"] = new_source_id
            rel["target"] = new_target_id
            rel["id"] = f"rel{rel_counter}"
            rel_counter += 1
            rel["confidence"] = confidence  # Ensure confidence is stored

            merged_relationships.append(rel)
            rel_signatures.add(signature)

    logger.debug(f"Merged into {len(merged_relationships)} unique relationships.")

    # Final limiting is done by _validate_graph_data called after merge

    return {"entities": final_merged_entities, "relationships": merged_relationships}


async def _perform_incremental_extraction(
    existing_graph: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    """Extracts new info and merges it with an existing graph."""
    logger.info("Performing incremental extraction...")
    # Extract context from existing graph
    existing_entities = existing_graph.get("entities", [])
    existing_relationships = existing_graph.get(
        "relationships", []
    )  # Needed for context? Maybe not.
    existing_entity_ids = {e["id"] for e in existing_entities}
    existing_entity_names = {e["name"].lower().strip(): e["id"] for e in existing_entities}

    # Format existing entities for the prompt (limit context size)
    context_limit = 50
    entity_context = "EXISTING ENTITIES (Sample):\n"
    for entity in existing_entities[:context_limit]:
        entity_context += f"- ID: {entity.get('id', 'N/A')}, Name: {entity.get('name', 'N/A')}, Type: {entity.get('type', 'N/A')}\n"
    if len(existing_entities) > context_limit:
        entity_context += f"... and {len(existing_entities) - context_limit} more entities.\n"

    # Build prompt
    incremental_instructions = _build_common_prompt_instructions(
        task_description="Extract NEW entities and relationships from the text below, linking them to the EXISTING entities provided.",
        format_structure="INCREMENTAL",  # Needs adaptation
        **{
            k: kwargs[k]
            for k in [
                "entity_types",
                "relation_types",
                "schema",
                "language",
                "enable_reasoning",
                "include_positions",
                "include_attributes",
                "include_evidence",
                "include_temporal_info",
                "min_confidence",
                "max_entities",
                "max_relations",
            ]
        },
    )
    # Adjust format instructions for incremental output
    incremental_instructions = incremental_instructions.replace(
        'Respond with a valid JSON object containing two keys: "entities" and "relationships".',
        'Respond with a valid JSON object containing two keys: "new_entities" and "new_relationships".',
    )
    incremental_instructions = incremental_instructions.replace(
        '"entities": [', '"new_entities": ['
    )
    incremental_instructions = incremental_instructions.replace(
        '"relationships": [', '"new_relationships": ['
    )
    incremental_instructions += "\nINSTRUCTIONS:\n- Identify entities NOT ALREADY in the 'EXISTING ENTITIES' list.\n- Assign NEW unique IDs to these new entities.\n- Identify relationships involving at least one NEW entity.\n- When referencing an EXISTING entity in a relationship, use its provided ID.\n- When referencing a NEW entity, use the NEW ID you assigned."

    # Handle custom prompt
    if kwargs.get("custom_prompt"):
        prompt = kwargs["custom_prompt"].format(
            text=kwargs["text"],
            instructions=incremental_instructions,
            existing_entities=entity_context,
            schema_info=kwargs.get("schema", ""),
            examples="",  # Examples less useful here
            stage="incremental",
        )
    else:
        prompt = (
            f"{entity_context}\n{incremental_instructions}\n\nTEXT TO ANALYZE:\n{kwargs['text']}"
        )

    sys_prompt = kwargs.get("system_prompt") or SYSTEM_PROMPTS.get(
        "standard_extraction"
    )  # Use standard as base

    # Call LLM
    incremental_data = await _call_llm_for_extraction(
        prompt=prompt,
        system_prompt=sys_prompt,
        provider_instance=kwargs["provider_instance"],
        model=kwargs["model"],
        max_tokens_per_request=kwargs["max_tokens_per_request"],
        additional_params=kwargs["additional_params"],
        task_name="Incremental Extraction",
    )

    # --- Merge with Existing Graph ---
    new_entities = incremental_data.get("new_entities", [])
    new_relationships = incremental_data.get("new_relationships", [])
    if not isinstance(new_entities, list):
        new_entities = []
    if not isinstance(new_relationships, list):
        new_relationships = []

    logger.info(
        f"Incremental: Found {len(new_entities)} potential new entities and {len(new_relationships)} new relationships."
    )

    validated_new_entities, validated_new_relationships = _validate_incremental_data(
        new_entities, new_relationships, existing_entity_ids, existing_entity_names
    )

    logger.info(
        f"Incremental: Validated {len(validated_new_entities)} new entities and {len(validated_new_relationships)} new relationships after checks."
    )

    # Combine
    combined_entities = existing_entities + validated_new_entities
    combined_relationships = existing_relationships + validated_new_relationships

    # Return combined data (validation/limiting happens in main func)
    result = {
        "entities": combined_entities,
        "relationships": combined_relationships,
        "incremental_stats": {
            "existing_entities": len(existing_entities),
            "existing_relationships": len(existing_relationships),
            "new_entities_found": len(validated_new_entities),
            "new_relationships_found": len(validated_new_relationships),
        },
        "model": incremental_data.get("model"),
        "tokens": incremental_data.get("tokens"),
        "cost": incremental_data.get("cost"),
    }
    return result


def _validate_incremental_data(
    new_entities: List[Dict[str, Any]],
    new_relationships: List[Dict[str, Any]],
    existing_entity_ids: Set[str],
    existing_entity_names: Dict[str, str],  # lowercase name -> ID
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validates entities/relationships from incremental extraction."""
    validated_entities: List[Dict[str, Any]] = []
    validated_relationships: List[Dict[str, Any]] = []
    id_map: Dict[str, str] = {}  # Map temp ID from LLM response -> final ID (existing or new)
    assigned_new_ids: Set[str] = set()
    new_entity_counter = 1

    # Process new entities
    for entity in new_entities:
        if not isinstance(entity, dict) or "name" not in entity or "type" not in entity:
            continue

        temp_id = entity.get("id", f"temp_{uuid.uuid4()}")  # Assign temp if missing
        name = entity["name"].strip()
        normalized_name = name.lower()

        # Check if it's actually an existing entity
        if normalized_name in existing_entity_names:
            existing_id = existing_entity_names[normalized_name]
            id_map[temp_id] = existing_id
            logger.debug(
                f"Incremental validation: Entity '{name}' ({temp_id}) matched existing entity {existing_id}."
            )
            continue  # Don't add, just map the ID

        # If it's genuinely new, assign a proper new ID
        new_id = f"ent_new_{new_entity_counter}"
        while new_id in existing_entity_ids or new_id in assigned_new_ids:
            new_entity_counter += 1
            new_id = f"ent_new_{new_entity_counter}"

        entity["id"] = new_id
        validated_entities.append(entity)
        assigned_new_ids.add(new_id)
        id_map[temp_id] = new_id
        logger.debug(
            f"Incremental validation: Added new entity '{name}' with ID {new_id} (original temp ID: {temp_id})."
        )

    all_valid_ids = existing_entity_ids.union(assigned_new_ids)

    # Process new relationships
    rel_counter = 1
    for rel in new_relationships:
        if (
            not isinstance(rel, dict)
            or "source" not in rel
            or "target" not in rel
            or "type" not in rel
        ):
            continue

        temp_source = rel["source"]
        temp_target = rel["target"]

        # Map source and target IDs
        final_source_id = id_map.get(
            temp_source, temp_source
        )  # Use temp_source if not in map (means it should be existing)
        final_target_id = id_map.get(temp_target, temp_target)  # Use temp_target if not in map

        # Check if final IDs are valid (either existing or newly assigned)
        if final_source_id not in all_valid_ids or final_target_id not in all_valid_ids:
            logger.warning(
                f"Incremental validation: Skipping relationship '{rel.get('id')}' - invalid source ('{final_source_id}') or target ('{final_target_id}') ID after mapping."
            )
            continue

        # Ensure at least one entity involved is new (optional rule, depends on desired behavior)
        # if final_source_id in existing_entity_ids and final_target_id in existing_entity_ids:
        #    logger.debug(f"Incremental validation: Skipping relationship between two existing entities: {rel}")
        #    continue

        rel["source"] = final_source_id
        rel["target"] = final_target_id

        # Assign new relationship ID
        rel["id"] = f"rel_new_{rel_counter}"
        rel_counter += 1

        validated_relationships.append(rel)

    return validated_entities, validated_relationships


async def _perform_structured_extraction(**kwargs) -> Dict[str, Any]:
    """Uses structured few-shot examples for consistent extraction."""
    logger.info("Performing structured (few-shot) extraction...")
    examples_str = ""
    example_entities = kwargs.get("example_entities")
    example_relationships = kwargs.get("example_relationships")

    if example_entities and example_relationships:
        # Format provided examples
        examples_str = "\nEXAMPLES:\n\nExample Input Text:\n<Example text corresponding to the output below>\n\nExample Output JSON:\n"
        examples_str += "```json\n"
        examples_str += json.dumps(
            {"entities": example_entities, "relationships": example_relationships}, indent=2
        )
        examples_str += "\n```\n"
        logger.info(
            f"Using {len(example_entities)} example entities and {len(example_relationships)} relationships."
        )
    else:
        logger.warning(
            "Structured strategy chosen, but no examples were provided. Quality may be impacted."
        )

    # Build prompt using common helper, adding examples
    instructions = _build_common_prompt_instructions(
        task_description="Extract entities and relationships from the text below, following the provided examples EXACTLY.",
        examples=examples_str,
        format_structure="STRUCTURED",  # Placeholder
        **{
            k: kwargs[k]
            for k in [
                "entity_types",
                "relation_types",
                "schema",
                "language",
                "enable_reasoning",
                "include_positions",
                "include_attributes",
                "include_evidence",
                "include_temporal_info",
                "min_confidence",
                "max_entities",
                "max_relations",
            ]
        },
    )
    instructions += "\nIMPORTANT: Your output JSON structure MUST strictly match the structure in the 'Example Output JSON' above."

    # Handle custom prompt template
    if kwargs.get("custom_prompt"):
        prompt = kwargs["custom_prompt"].format(
            text=kwargs["text"], instructions=instructions, examples=examples_str
        )
    else:
        prompt = f"{instructions}\n\nTEXT TO ANALYZE:\n{kwargs['text']}"

    sys_prompt = kwargs.get("system_prompt") or SYSTEM_PROMPTS.get(
        "standard_extraction"
    )  # Base prompt

    # Call LLM
    graph_data = await _call_llm_for_extraction(
        prompt=prompt,
        system_prompt=sys_prompt,
        provider_instance=kwargs["provider_instance"],
        model=kwargs["model"],
        max_tokens_per_request=kwargs["max_tokens_per_request"],
        additional_params=kwargs["additional_params"],
        task_name="Structured Extraction",
    )

    return graph_data  # Validation happens in the main function


async def _perform_schema_guided_extraction(schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Performs extraction strictly adhering to the provided schema."""
    logger.info("Performing strict schema-guided extraction...")

    # Build prompt emphasizing schema adherence
    instructions = _build_common_prompt_instructions(
        task_description="Extract entities and relationships from the text below STRICTLY according to the provided schema.",
        schema=schema,  # Ensure schema is passed to helper for inclusion
        format_structure="SCHEMA_GUIDED",  # Placeholder
        **{
            k: kwargs[k]
            for k in [
                "entity_types",
                "relation_types",
                "language",
                "enable_reasoning",  # Pass entity/relation types as potential *subset* of schema
                "include_positions",
                "include_attributes",
                "include_evidence",
                "include_temporal_info",
                "min_confidence",
                "max_entities",
                "max_relations",
            ]
        },
    )
    instructions += "\nRULES:\n- ONLY extract entities and relationships matching the types defined in the SCHEMA.\n- ONLY include attributes specified in the SCHEMA for each entity type.\n- Ensure relationships connect entities with types allowed by the SCHEMA definition (Source/Target types).\n- DO NOT extract anything that does not conform to the SCHEMA."

    # Handle custom prompt
    if kwargs.get("custom_prompt"):
        prompt = kwargs["custom_prompt"].format(
            text=kwargs["text"],
            instructions=instructions,
            schema_info=json.dumps(schema, indent=2),  # Provide full schema if template needs it
            examples="",  # No generic examples
            stage="schema_guided",
        )
    else:
        prompt = f"{instructions}\n\nTEXT TO ANALYZE:\n{kwargs['text']}"

    sys_prompt = kwargs.get("system_prompt") or SYSTEM_PROMPTS.get(
        "standard_extraction"
    )  # Base prompt

    # Call LLM - potentially lower temperature for schema adherence
    strict_params = {**kwargs.get("additional_params", {})}
    strict_params["temperature"] = strict_params.get(
        "temperature", 0.05
    )  # Override default temp to be stricter

    graph_data = await _call_llm_for_extraction(
        prompt=prompt,
        system_prompt=sys_prompt,
        provider_instance=kwargs["provider_instance"],
        model=kwargs["model"],
        max_tokens_per_request=kwargs["max_tokens_per_request"],
        additional_params=strict_params,
        task_name="Schema-Guided Extraction",
    )

    # Optional: Add a validation step here specifically against the schema?
    # Currently relies on the LLM following instructions + standard validation.

    return graph_data  # Validation happens in the main function


# --- Post-processing Helper Functions ---


def _normalize_entities(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes entity names (e.g., capitalization, whitespace) and merges duplicates."""
    logger.debug("Normalizing entity names and merging duplicates...")
    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])
    if not isinstance(entities, list) or not entities:
        return extraction_result

    normalized_entities: Dict[
        Tuple[str, str], Dict[str, Any]
    ] = {}  # (normalized_name, type) -> entity
    id_map: Dict[str, str] = {}  # Map old ID -> new ID (of the kept entity)
    entity_counter = 1

    for entity in entities:
        if (
            not isinstance(entity, dict)
            or "name" not in entity
            or "type" not in entity
            or "id" not in entity
        ):
            continue

        original_id = entity["id"]
        name = entity["name"].strip()
        type = entity.get("type", "Unknown")

        # Basic normalization (can be customized)
        normalized_name = " ".join(name.split())  # Remove extra whitespace
        # Simple title case for common proper noun types
        if type in ["Person", "Organization", "Location", "Product", "Event", "Brand"]:
            # Avoid changing acronyms or mixed case like 'iPhone'
            if not any(c.islower() for c in normalized_name) or not any(
                c.isupper() for c in normalized_name
            ):
                normalized_name = normalized_name.title()  # Title case only if all upper/lower
            # More sophisticated logic could go here

        merge_key = (normalized_name.lower(), type)  # Merge key based on lower name and type

        existing_entity = normalized_entities.get(merge_key)

        if existing_entity:
            # Map old ID to the ID of the entity we are merging into
            id_map[original_id] = existing_entity["id"]
            logger.debug(
                f"Normalizing: Merging '{name}' ({original_id}) into '{existing_entity['name']}' ({existing_entity['id']})"
            )

            # Merge mentions
            existing_mentions = {
                m["text"] for m in existing_entity.get("mentions", []) if "text" in m
            }
            for mention in entity.get("mentions", []):
                if (
                    isinstance(mention, dict)
                    and "text" in mention
                    and mention["text"] not in existing_mentions
                ):
                    existing_entity.setdefault("mentions", []).append(mention)
                    existing_mentions.add(mention["text"])

            # Merge attributes (simple update)
            if "attributes" in entity and isinstance(entity["attributes"], dict):
                existing_entity.setdefault("attributes", {}).update(entity["attributes"])

        else:
            # This is the first time we see this normalized entity, keep it
            # Assign a stable new ID if needed (might be redundant if IDs are already good)
            new_id = f"ent{entity_counter}"
            entity_counter += 1
            entity["id"] = new_id
            entity["name"] = normalized_name  # Store normalized name

            normalized_entities[merge_key] = entity
            id_map[original_id] = new_id  # Map original ID to its new/kept ID

    final_entities = list(normalized_entities.values())

    # Update relationships with new IDs
    final_relationships = []
    if isinstance(relationships, list):
        for rel in relationships:
            if not isinstance(rel, dict) or "source" not in rel or "target" not in rel:
                continue

            original_source = rel["source"]
            original_target = rel["target"]

            new_source_id = id_map.get(original_source)
            new_target_id = id_map.get(original_target)

            if new_source_id and new_target_id:
                rel["source"] = new_source_id
                rel["target"] = new_target_id
                final_relationships.append(rel)
            else:
                logger.debug(
                    f"Normalizing: Dropping relationship '{rel.get('id')}' due to missing entity mapping for source '{original_source}' or target '{original_target}'."
                )

    extraction_result["entities"] = final_entities
    extraction_result["relationships"] = final_relationships
    logger.debug(
        f"Normalization complete. Entities: {len(final_entities)}, Relationships: {len(final_relationships)}"
    )
    return extraction_result


def _add_graph_metrics(
    extraction_result: Dict[str, Any], sort_by: str = "confidence"
) -> Dict[str, Any]:
    """Adds computed graph metrics (if networkx available) and sorts results."""
    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])
    if not isinstance(entities, list) or not isinstance(relationships, list):
        return extraction_result  # Cannot process

    # Calculate metrics using NetworkX if available
    metrics = {}
    node_centrality = {}
    node_degree = {}
    if HAS_NETWORKX:
        logger.debug("Calculating graph metrics using NetworkX...")
        G = nx.DiGraph()
        entity_ids = set()
        for entity in entities:
            if isinstance(entity, dict) and "id" in entity:
                G.add_node(entity["id"], **{k: v for k, v in entity.items() if k != "id"})
                entity_ids.add(entity["id"])

        for rel in relationships:
            if isinstance(rel, dict) and "source" in rel and "target" in rel:
                # Only add edges between nodes that actually exist
                if rel["source"] in entity_ids and rel["target"] in entity_ids:
                    G.add_edge(
                        rel["source"],
                        rel["target"],
                        **{k: v for k, v in rel.items() if k not in ["id", "source", "target"]},
                    )

        if G.number_of_nodes() > 0:
            try:
                metrics["node_count"] = G.number_of_nodes()
                metrics["edge_count"] = G.number_of_edges()
                metrics["density"] = nx.density(G) if G.number_of_nodes() > 1 else 0.0
                metrics["avg_degree"] = sum(d for n, d in G.degree()) / G.number_of_nodes()

                # Components (use weakly connected for directed graph)
                metrics["components"] = nx.number_weakly_connected_components(G)

                # Centrality (can be slow on large graphs)
                if G.number_of_nodes() < 1000:  # Limit calculation for performance
                    try:
                        node_centrality = nx.betweenness_centrality(G, normalized=True)
                        node_degree = nx.degree_centrality(G)
                        metrics["avg_betweenness_centrality"] = (
                            sum(node_centrality.values()) / G.number_of_nodes()
                        )
                        logger.debug("Calculated betweenness centrality.")
                    except Exception as cent_err:
                        logger.warning(f"Could not calculate centrality metrics: {cent_err}")
                else:
                    logger.warning("Skipping centrality calculation for large graph (>1000 nodes).")

            except Exception as metric_err:
                logger.warning(f"Failed to calculate some graph metrics: {metric_err}")
            extraction_result["metrics"] = metrics

    # Add centrality/degree to entities
    for entity in entities:
        if isinstance(entity, dict) and "id" in entity:
            entity_id = entity["id"]
            entity["centrality"] = node_centrality.get(entity_id, 0.0)
            entity["degree"] = node_degree.get(entity_id, 0.0)  # Use degree centrality as 'degree'

    # Sort entities and relationships
    logger.debug(f"Sorting results by '{sort_by}'...")
    # Sort entities
    if sort_by == "centrality" and HAS_NETWORKX and node_centrality:
        entities.sort(key=lambda x: x.get("centrality", 0.0), reverse=True)
    elif sort_by == "mentions":
        entities.sort(key=lambda x: len(x.get("mentions", [])), reverse=True)
    elif sort_by == "degree" and HAS_NETWORKX and node_degree:
        entities.sort(key=lambda x: x.get("degree", 0.0), reverse=True)
    else:  # Default or fallback: sort by name
        entities.sort(key=lambda x: x.get("name", "").lower())

    # Sort relationships (typically by confidence)
    relationships.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    extraction_result["entities"] = entities
    extraction_result["relationships"] = relationships

    return extraction_result


def _generate_visualization(
    extraction_result: Dict[str, Any], format: VisualizationFormat
) -> Optional[Dict[str, Any]]:
    """Generates a visualization of the entity graph if libraries are available."""
    if format == VisualizationFormat.NONE:
        return None
    if not HAS_VISUALIZATION_LIBS:
        logger.warning("Cannot generate visualization - required libraries not installed.")
        return {"error": "Visualization libraries (networkx, pyvis, matplotlib) not installed."}

    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])
    if not isinstance(entities, list) or not isinstance(relationships, list) or not entities:
        logger.warning("Cannot generate visualization - no valid entities or relationships found.")
        return {"error": "No entities/relationships to visualize."}

    viz_data = {}
    output_dir = tempfile.gettempdir()
    file_uuid = uuid.uuid4()
    base_filename = f"graph_{file_uuid}"

    try:
        if format == VisualizationFormat.HTML:
            if not HAS_PYVIS:
                return {"error": "Pyvis library not installed for HTML visualization."}
            logger.debug("Generating HTML visualization using Pyvis...")
            net = Network(notebook=False, height="800px", width="100%", directed=True)
            # Basic options - can be customized further
            net.set_options("""
            {
              "nodes": { "font": { "size": 10 }, "size": 15 },
              "edges": { "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } }, "smooth": { "type": "continuous" }, "font": { "size": 9, "align": "middle" } },
              "physics": { "solver": "forceAtlas2Based", "forceAtlas2Based": { "springLength": 100, "avoidOverlap": 0.5 } }
            }
            """)

            entity_ids = set()
            type_colors = {}
            unique_types = sorted(list(set(e.get("type", "Unknown") for e in entities)))
            # Simple color mapping
            palette = [
                "#e6194B",
                "#3cb44b",
                "#ffe119",
                "#4363d8",
                "#f58231",
                "#911eb4",
                "#42d4f4",
                "#f032e6",
                "#bfef45",
                "#fabed4",
                "#469990",
                "#dcbeff",
                "#9A6324",
                "#fffac8",
                "#800000",
                "#aaffc3",
                "#808000",
                "#ffd8b1",
                "#000075",
                "#a9a9a9",
            ]
            for i, etype in enumerate(unique_types):
                type_colors[etype] = palette[i % len(palette)]

            for entity in entities:
                eid = entity["id"]
                entity_ids.add(eid)
                label = entity.get("name", eid)
                etype = entity.get("type", "Unknown")
                title = f"ID: {eid}\nType: {etype}"
                if entity.get("attributes"):
                    title += "\n" + "\n".join(f"{k}: {v}" for k, v in entity["attributes"].items())
                size = 15 + int(entity.get("degree", 0.0) * 30)  # Scale size by degree
                net.add_node(
                    eid,
                    label=label,
                    title=title,
                    color=type_colors.get(etype, "#d3d3d3"),
                    size=size,
                )

            for rel in relationships:
                src = rel["source"]
                tgt = rel["target"]
                if src in entity_ids and tgt in entity_ids:
                    label = rel.get("type", "")
                    conf = rel.get("confidence", 1.0)
                    title = f"{label} (Conf: {conf:.2f})"
                    if rel.get("evidence"):
                        title += f"\nEvidence: {rel['evidence'][:100]}..."
                    width = 0.5 + (conf * 2.0)
                    net.add_edge(src, tgt, title=title, label=label, width=width)

            html_path = os.path.join(output_dir, f"{base_filename}.html")
            net.save_graph(html_path)
            with open(html_path, "r", encoding="utf-8") as f:
                viz_data["html"] = f.read()
            viz_data["url"] = f"file://{html_path}"
            logger.info(f"HTML visualization saved to {html_path}")

        elif format in [VisualizationFormat.SVG, VisualizationFormat.PNG]:
            if not HAS_NETWORKX or not HAS_MATPLOTLIB:
                return {"error": "NetworkX and Matplotlib required for SVG/PNG visualization."}
            logger.debug(
                f"Generating {format.value.upper()} visualization using NetworkX/Matplotlib..."
            )

            G = nx.DiGraph()
            entity_ids = set()
            type_colors = {}
            unique_types = sorted(list(set(e.get("type", "Unknown") for e in entities)))
            palette = plt.cm.get_cmap("tab20", len(unique_types))  # Use a colormap
            node_colors_map = {}
            node_labels = {}
            node_sizes = []

            for i, etype in enumerate(unique_types):
                type_colors[etype] = palette(i)

            for entity in entities:
                eid = entity["id"]
                entity_ids.add(eid)
                node_labels[eid] = entity.get("name", eid)[:20]  # Truncate long labels
                etype = entity.get("type", "Unknown")
                node_colors_map[eid] = type_colors.get(etype, "#d3d3d3")
                # Size based on degree, with min size
                node_sizes.append(300 + int(entity.get("degree", 0.0) * 1000))
                G.add_node(eid)  # Add node without attributes for plotting

            for rel in relationships:
                src = rel["source"]
                tgt = rel["target"]
                if src in entity_ids and tgt in entity_ids:
                    G.add_edge(src, tgt, label=rel.get("type", ""))

            if G.number_of_nodes() == 0:
                raise ValueError("Graph has no nodes after filtering.")

            plt.figure(figsize=(16, 12))
            # Use a layout algorithm
            pos = nx.spring_layout(
                G,
                k=0.5 / (G.number_of_nodes() ** 0.5) if G.number_of_nodes() > 1 else 1,
                iterations=50,
            )

            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=node_sizes,
                node_color=[node_colors_map[n] for n in G.nodes()],
                alpha=0.8,
            )
            nx.draw_networkx_edges(
                G, pos, alpha=0.5, arrows=True, arrowsize=10, node_size=node_sizes
            )
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

            # Add edge labels (can get cluttered)
            if G.number_of_edges() < 100:  # Only draw edge labels for smaller graphs
                edge_labels = nx.get_edge_attributes(G, "label")
                nx.draw_networkx_edge_labels(
                    G, pos, edge_labels=edge_labels, font_size=7, alpha=0.7
                )

            # Create legend
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=etype,
                    markersize=10,
                    markerfacecolor=color,
                )
                for etype, color in type_colors.items()
            ]
            plt.legend(handles=legend_handles, title="Entity Types", loc="upper right", fontsize=8)

            plt.title("Entity Relationship Graph", fontsize=14)
            plt.axis("off")

            file_extension = format.value
            output_path = os.path.join(output_dir, f"{base_filename}.{file_extension}")
            plt.savefig(output_path, format=file_extension, bbox_inches="tight", dpi=150)
            plt.close()  # Close the plot figure

            # Read SVG content if requested
            if format == VisualizationFormat.SVG:
                with open(output_path, "r", encoding="utf-8") as f:
                    viz_data["svg"] = f.read()

            viz_data["url"] = f"file://{output_path}"
            logger.info(f"{format.value.upper()} visualization saved to {output_path}")

        elif format == VisualizationFormat.DOT:
            if not HAS_NETWORKX:
                return {"error": "NetworkX required for DOT format generation."}
            logger.debug("Generating DOT visualization format...")
            G = nx.DiGraph()  # Rebuild graph specifically for DOT attributes if needed
            # ... (similar node/edge adding as above) ...
            dot_string = nx.nx_pydot.to_pydot(G).to_string()  # Requires pydot
            dot_path = os.path.join(output_dir, f"{base_filename}.dot")
            with open(dot_path, "w", encoding="utf-8") as f:
                f.write(dot_string)
            viz_data["dot"] = dot_string
            viz_data["url"] = f"file://{dot_path}"
            logger.info(f"DOT visualization saved to {dot_path}")

    except ImportError as e:
        logger.error(f"Visualization failed due to missing library: {e}", exc_info=True)
        viz_data["error"] = f"Missing library for {format.value} format: {e}"
    except Exception as e:
        logger.error(f"Failed to generate {format.value} visualization: {e}", exc_info=True)
        viz_data["error"] = f"Failed to generate {format.value} visualization: {e}"

    return viz_data if viz_data else None


def _format_output(extraction_result: Dict[str, Any], format: OutputFormat) -> Dict[str, Any]:
    """Formats the extraction result into the requested output structure."""
    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])

    if format == OutputFormat.JSON:
        return extraction_result  # Already in desired format

    if not isinstance(entities, list) or not isinstance(relationships, list):
        return {"error": "Invalid input data for formatting.", **extraction_result}

    logger.debug(f"Formatting output as {format.value}...")
    output = {
        "entities": entities,
        "relationships": relationships,
        **extraction_result,
    }  # Start with base data

    try:
        if format == OutputFormat.NETWORKX:
            if not HAS_NETWORKX:
                raise ImportError("networkx library not installed.")
            G = nx.DiGraph()
            entity_ids = set()
            for entity in entities:
                eid = entity["id"]
                entity_ids.add(eid)
                G.add_node(eid, **{k: v for k, v in entity.items() if k != "id"})
            for rel in relationships:
                if rel["source"] in entity_ids and rel["target"] in entity_ids:
                    G.add_edge(
                        rel["source"],
                        rel["target"],
                        **{k: v for k, v in rel.items() if k not in ["id", "source", "target"]},
                    )
            output["graph"] = G

        elif format == OutputFormat.RDF:
            triples = []
            ns = "urn:graph:"  # Simple namespace prefix
            for entity in entities:
                subj = f"{ns}entity#{entity['id']}"
                triples.append((subj, f"{ns}type", f"{ns}type#{entity.get('type', 'Unknown')}"))
                triples.append((subj, f"{ns}name", entity.get("name", "")))
                for attr, val in entity.get("attributes", {}).items():
                    triples.append((subj, f"{ns}attr#{attr}", str(val)))  # Simple string literal
            for rel in relationships:
                subj = f"{ns}entity#{rel['source']}"
                pred = f"{ns}rel#{rel.get('type', 'relatedTo')}"
                obj = f"{ns}entity#{rel['target']}"
                triples.append((subj, pred, obj))
                # Could add relationship attributes as reified triples if needed
            output["rdf_triples"] = triples

        elif format == OutputFormat.CYTOSCAPE:
            nodes = [
                {"data": {"id": e["id"], **e}} for e in entities
            ]  # Put all entity data under 'data'
            edges = [
                {
                    "data": {
                        "id": r.get("id", f"rel_{r['source']}_{r['target']}"),
                        "source": r["source"],
                        "target": r["target"],
                        **r,
                    }
                }
                for r in relationships
            ]
            output["cytoscape"] = {"nodes": nodes, "edges": edges}

        elif format == OutputFormat.D3:
            nodes = [
                {"id": e["id"], "name": e.get("name"), "group": e.get("type"), **e}
                for e in entities
            ]
            links = [
                {
                    "source": r["source"],
                    "target": r["target"],
                    "type": r.get("type"),
                    "value": r.get("confidence", 0.5) * 10,
                    **r,
                }
                for r in relationships
            ]
            output["d3"] = {"nodes": nodes, "links": links}

        elif format == OutputFormat.NEO4J:
            queries = []
            # Create constraints for uniqueness if desired (optional)
            # queries.append("CREATE CONSTRAINT ON (e:Entity) ASSERT e.id IS UNIQUE;")
            # queries.append("CREATE CONSTRAINT ON (p:Person) ASSERT p.id IS UNIQUE;") # etc. for specific types

            entity_types = set(e.get("type", "Entity") for e in entities)
            for etype in entity_types:
                queries.append(f"CREATE INDEX IF NOT EXISTS FOR (n:{etype}) ON (n.id);")

            # Create nodes
            for entity in entities:
                labels = ":".join(
                    label for label in ["Entity", entity.get("type", "Unknown")] if label
                )  # Base :Entity label + specific type
                props = {
                    k: v for k, v in entity.items() if k not in ["mentions"]
                }  # Exclude mentions from props
                props_str = json.dumps(props)[1:-1]  # Get inner K:V pairs
                queries.append(
                    f"MERGE (n:{labels} {{id: {json.dumps(entity['id'])}}}) SET n += {{{props_str}}}"
                )

            # Create relationships
            for rel in relationships:
                rel_type = re.sub(
                    r"\W+", "_", rel.get("type", "RELATED_TO")
                ).upper()  # Sanitize type for Neo4j
                props = {
                    k: v for k, v in rel.items() if k not in ["id", "source", "target", "type"]
                }
                props_str = json.dumps(props)
                queries.append(
                    f"MATCH (a {{id: {json.dumps(rel['source'])}}}), (b {{id: {json.dumps(rel['target'])}}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    f"SET r += {props_str}"
                )
            output["neo4j_queries"] = queries

    except ImportError as e:
        output["error"] = f"Missing library for {format.value} formatting: {e}"
        logger.warning(output["error"])
        if format == OutputFormat.NETWORKX:
            output = _format_output(extraction_result, OutputFormat.JSON)  # Fallback
    except Exception as e:
        output["error"] = f"Failed to format output as {format.value}: {e}"
        logger.error(output["error"], exc_info=True)

    return output


def _create_query_interface(extraction_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Creates helper functions for querying the graph (requires networkx)."""
    if not HAS_NETWORKX:
        return None

    entities = extraction_result.get("entities", [])
    relationships = extraction_result.get("relationships", [])
    if not isinstance(entities, list) or not isinstance(relationships, list):
        return None

    # Build graph once for querying
    G = nx.DiGraph()
    entity_map = {}
    entity_ids = set()
    for entity in entities:
        if isinstance(entity, dict) and "id" in entity:
            eid = entity["id"]
            entity_ids.add(eid)
            entity_map[eid] = entity
            G.add_node(eid, **{k: v for k, v in entity.items() if k != "id"})

    for rel in relationships:
        if isinstance(rel, dict) and "source" in rel and "target" in rel:
            if rel["source"] in entity_ids and rel["target"] in entity_ids:
                G.add_edge(
                    rel["source"],
                    rel["target"],
                    **{k: v for k, v in rel.items() if k not in ["id", "source", "target"]},
                )

    # --- Query Functions ---
    def find_entity(
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        attribute_key: Optional[str] = None,
        attribute_value: Optional[Any] = None,
    ) -> List[Dict]:
        """Finds entities matching specified criteria."""
        results = []
        name_lower = name.lower() if name else None
        for _eid, entity_data in entity_map.items():
            match = True
            if name_lower and name_lower not in entity_data.get("name", "").lower():
                match = False
            if entity_type and entity_data.get("type") != entity_type:
                match = False
            if attribute_key:
                attrs = entity_data.get("attributes", {})
                if attribute_key not in attrs or (
                    attribute_value is not None and attrs[attribute_key] != attribute_value
                ):
                    match = False
            if match:
                results.append(entity_data)
        return results

    def find_path(source_id: str, target_id: str, cutoff: int = 5) -> List[List[Dict]]:
        """Finds simple paths (lists of entity dicts) between two entity IDs."""
        if source_id not in G or target_id not in G:
            return []
        try:
            paths_nodes = list(
                nx.all_simple_paths(G, source=source_id, target=target_id, cutoff=cutoff)
            )
            return [[entity_map[nid] for nid in node_path] for node_path in paths_nodes]
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.warning(f"Path finding failed: {e}")
            return []

    def get_neighbors(entity_id: str, direction: str = "both") -> Dict[str, List[Dict]]:
        """Gets direct neighbors (entities) of a given entity ID."""
        if entity_id not in G:
            return {"incoming": [], "outgoing": []}
        neighbors = {"incoming": [], "outgoing": []}
        if direction in ["incoming", "both"]:
            neighbors["incoming"] = [
                entity_map[pred] for pred in G.predecessors(entity_id) if pred in entity_map
            ]
        if direction in ["outgoing", "both"]:
            neighbors["outgoing"] = [
                entity_map[succ] for succ in G.successors(entity_id) if succ in entity_map
            ]
        return neighbors

    # --- Interface Definition ---
    query_interface = {
        "find_entity": find_entity,
        "find_path": find_path,
        "get_neighbors": get_neighbors,
        "get_entity_by_id": lambda eid: entity_map.get(eid),
        "get_graph_object": lambda: G,  # Return the NetworkX graph
        "descriptions": {
            "find_entity": find_entity.__doc__,
            "find_path": find_path.__doc__,
            "get_neighbors": get_neighbors.__doc__,
            "get_entity_by_id": "Retrieves a single entity dictionary by its unique ID.",
            "get_graph_object": "Returns the underlying NetworkX DiGraph object.",
        },
    }
    return query_interface
