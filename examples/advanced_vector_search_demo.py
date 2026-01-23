#!/usr/bin/env python
"""Demo of advanced vector search capabilities using real Ultimate MCP Server tools."""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.markup import escape
from rich.rule import Rule

from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.services.vector import get_vector_db_service
from ultimate_mcp_server.services.vector.embeddings import cosine_similarity, get_embedding_service

# --- Add Marqo Tool Import ---
from ultimate_mcp_server.tools.marqo_fused_search import marqo_fused_search
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import (
    display_embedding_generation_results,
    display_text_content_result,
    display_vector_similarity_results,
    parse_and_display_result,
)

# ---------------------------
# --- Add Rich Imports ---
from ultimate_mcp_server.utils.logging.console import console

# ----------------------

# Initialize logger
logger = get_logger("example.advanced_vector_search")

# Initialize global gateway
gateway = None
vector_service = None
embedding_service = None


async def setup_services():
    """Set up the gateway and vector service for demonstration."""
    global gateway, vector_service, embedding_service

    logger.info("Initializing gateway and services...", emoji_key="start")
    gateway = Gateway("vector-demo", register_tools=False)
    await gateway._initialize_providers()

    embedding_service = (
        get_embedding_service()
    )  # Gateway will provide API keys through provider system
    vector_service = get_vector_db_service()

    logger.success("Services initialized.", emoji_key="success")


async def embedding_generation_demo():
    """Demonstrate embedding generation with real providers using Rich."""
    console.print(Rule("[bold blue]Embedding Generation Demo[/bold blue]"))
    logger.info("Starting embedding generation demo", emoji_key="start")

    text_samples = [
        "Quantum computing leverages quantum mechanics to perform computations",
        "Artificial intelligence systems can learn from data and improve over time",
        "Cloud infrastructure enables scalable and flexible computing resources",
    ]
    console.print("Input Text Samples:")
    for i, sample in enumerate(text_samples):
        console.print(f"  {i + 1}. {escape(sample)}")

    # Define models to test (ensure they are supported by your embedding_service config)
    models_to_test = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]

    # Collect results for display
    results_data = {"models": []}

    for model_name in models_to_test:
        try:
            logger.info(f"Generating embeddings with {model_name}...", emoji_key="processing")
            start_time = time.time()
            embeddings = await embedding_service.create_embeddings(texts=text_samples)
            processing_time = time.time() - start_time

            model_result = {
                "name": model_name,
                "success": embeddings and len(embeddings) > 0,
                "time": processing_time,
                "cost": embedding_service.last_request_cost
                if hasattr(embedding_service, "last_request_cost")
                else 0.0,
            }

            if embeddings and len(embeddings) > 0:
                dims = len(embeddings[0])
                model_result["dimensions"] = dims
                model_result["embedding_sample"] = embeddings[0][:3]
                logger.success(
                    f"Generated {len(embeddings)} embeddings ({dims} dims) for {model_name}",
                    emoji_key="success",
                )
            else:
                logger.warning(f"No embeddings returned for {model_name}", emoji_key="warning")

            results_data["models"].append(model_result)

        except Exception as e:
            logger.error(
                f"Error generating embeddings with {model_name}: {e}",
                emoji_key="error",
                exc_info=True,
            )
            results_data["models"].append({"name": model_name, "success": False, "error": str(e)})

    # Use the shared display utility to show results
    display_embedding_generation_results(results_data)


async def vector_search_demo():
    """Demonstrate vector search capabilities using Rich."""
    console.print(Rule("[bold blue]Vector Search Demo[/bold blue]"))
    logger.info("Starting vector search demo", emoji_key="start")

    documents = [
        "Quantum computing uses quantum bits or qubits to perform calculations.",
        "Machine learning algorithms learn patterns from data without explicit programming.",
        "Blockchain technology creates a distributed and immutable ledger of transactions.",
        "Cloud computing delivers computing services over the internet on demand.",
        "Natural language processing helps computers understand and interpret human language.",
        "Artificial intelligence systems can simulate human intelligence in machines.",
        "Edge computing processes data closer to where it is generated rather than in a centralized location.",
        "Cybersecurity involves protecting systems from digital attacks and unauthorized access.",
        "Internet of Things (IoT) connects everyday devices to the internet for data sharing.",
        "Virtual reality creates an immersive computer-generated environment.",
    ]
    document_metadata = [
        {"id": "doc1", "category": "quantum", "level": "advanced"},
        {"id": "doc2", "category": "ai", "level": "intermediate"},
        {"id": "doc3", "category": "blockchain", "level": "beginner"},
        {"id": "doc4", "category": "cloud", "level": "intermediate"},
        {"id": "doc5", "category": "ai", "level": "advanced"},
        {"id": "doc6", "category": "ai", "level": "beginner"},
        {"id": "doc7", "category": "cloud", "level": "advanced"},
        {"id": "doc8", "category": "security", "level": "intermediate"},
        {"id": "doc9", "category": "iot", "level": "beginner"},
        {"id": "doc10", "category": "vr", "level": "intermediate"},
    ]

    collection_name = "demo_vector_store_rich"
    embedding_dimension = 1536  # Default for text-embedding-ada-002 / 3-small

    try:
        logger.info(f"Creating/resetting collection: {collection_name}", emoji_key="db")
        await vector_service.create_collection(
            name=collection_name,
            dimension=embedding_dimension,
            overwrite=True,
            metadata={"description": "Demo collection for Rich vector search"},
        )

        logger.info("Adding documents to vector store...", emoji_key="processing")
        ids = await vector_service.add_texts(
            collection_name=collection_name,
            texts=documents,
            metadatas=document_metadata,
            batch_size=5,
        )
        logger.success(f"Added {len(ids)} documents.", emoji_key="success")

        # --- Perform Searches ---
        search_queries = [
            "How does quantum computing work?",
            "Machine learning for image recognition",
            "Secure blockchain implementation",
        ]

        console.print(Rule("[green]Vector Search Results[/green]"))
        for query in search_queries:
            logger.info(f'Searching for: "{escape(query)}"...', emoji_key="search")
            search_start_time = time.time()
            results = await vector_service.search_by_text(
                collection_name=collection_name,
                query_text=query,
                top_k=3,
                include_vectors=False,
                # Example filter: metadata_filter={"category": "ai"}
            )
            search_time = time.time() - search_start_time

            # Format the results for the display utility
            search_result = {"processing_time": search_time, "results": results, "query": query}

            # Use the shared display utility
            parse_and_display_result(
                title=f"Search: {query}", input_data={"query": query}, result=search_result
            )

    except Exception as e:
        logger.error(f"Error during vector search demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
    finally:
        # Clean up the collection
        try:
            logger.info(f"Deleting collection: {collection_name}", emoji_key="db")
            await vector_service.delete_collection(collection_name)
        except Exception as delete_err:
            logger.warning(
                f"Could not delete collection {collection_name}: {delete_err}", emoji_key="warning"
            )
    console.print()


async def hybrid_search_demo():
    """Demonstrate hybrid search using the marqo_fused_search tool."""
    console.print(Rule("[bold blue]Hybrid Search Demo (using Marqo Fused Search Tool)[/bold blue]"))
    logger.info("Starting hybrid search demo (conceptual)", emoji_key="start")

    # This demo uses the marqo_fused_search tool, which performs hybrid search.
    # It requires a running Marqo instance and a configured index
    # as defined in marqo_index_config.json.

    # Note: For this demo to work correctly, the configured Marqo index
    # should contain documents related to the query, potentially including
    # metadata fields like 'tags' if filtering is intended.
    # The setup below is removed as the data needs to be pre-indexed in Marqo.
    # collection_name = "demo_hybrid_store_rich"
    # try:
    #    logger.info(f"Creating/resetting collection: {collection_name}", emoji_key="db")
    #    # ... [Code to create collection and add documents would go here if using local DB] ...
    # except Exception as setup_e:
    #     logger.error(f"Failed to setup data for hybrid demo: {setup_e}", emoji_key="error")
    #     console.print(f"[bold red]Error setting up demo data: {escape(str(setup_e))}[/bold red]")
    #     return

    try:
        # --- Perform Hybrid Search (Simulated) ---
        query = "cloud semantic search techniques"
        # keywords = ["cloud", "semantic"] # Keywords can be included in query or filters
        semantic_weight_param = 0.6  # Weight for semantic search (alpha)

        logger.info(
            f'Hybrid search for: "{escape(query)}" with semantic weight {semantic_weight_param}',
            emoji_key="search",
        )

        # Call the marqo_fused_search tool directly
        hybrid_result = await marqo_fused_search(
            query=query,
            limit=3,  # Request top 3 results
            semantic_weight=semantic_weight_param,
            # Add filters={}, date_range=None etc. if needed based on schema
        )

        display_text_content_result(
            f"Hybrid Search Results (Weight={semantic_weight_param})",
            hybrid_result,  # Pass the result dict directly
        )

    except Exception as e:
        logger.error(f"Error during hybrid search demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
    # Removed cleanup as we assume Marqo index exists independently
    console.print()


async def semantic_similarity_demo():
    """Demonstrate calculating semantic similarity using Rich."""
    console.print(Rule("[bold blue]Semantic Similarity Demo[/bold blue]"))
    logger.info("Starting semantic similarity demo", emoji_key="start")

    text_pairs = [
        ("The cat sat on the mat", "A feline was resting upon the rug"),
        ("AI is transforming industries", "Artificial intelligence drives innovation"),
        ("Cloud computing offers scalability", "The weather today is sunny"),
    ]

    model_name = "text-embedding-ada-002"  # Use a consistent model
    logger.info(f"Calculating similarity using model: {model_name}", emoji_key="model")

    # Prepare data structure for the shared display utility
    similarity_data = {"pairs": [], "model": model_name}

    try:
        all_texts = [text for pair in text_pairs for text in pair]
        embeddings = await embedding_service.create_embeddings(texts=all_texts)

        if len(embeddings) == len(all_texts):
            for i, pair in enumerate(text_pairs):
                idx1 = i * 2
                idx2 = i * 2 + 1
                score = cosine_similarity(embeddings[idx1], embeddings[idx2])

                similarity_data["pairs"].append(
                    {"text1": pair[0], "text2": pair[1], "score": score}
                )

            # Use the specialized display function for similarity results
            display_vector_similarity_results(similarity_data)
        else:
            logger.error(
                "Mismatch between number of texts and embeddings received.", emoji_key="error"
            )
            console.print("[red]Error calculating similarities: Embedding count mismatch.[/red]")

    except Exception as e:
        logger.error(
            f"Error calculating semantic similarity: {e}", emoji_key="error", exc_info=True
        )
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")

    console.print()


async def main():
    """Run all advanced vector search demonstrations."""
    await setup_services()
    console.print(Rule("[bold magenta]Advanced Vector Search Demos Starting[/bold magenta]"))

    try:
        await embedding_generation_demo()
        await vector_search_demo()
        await hybrid_search_demo()
        await semantic_similarity_demo()

    except Exception as e:
        logger.critical(f"Vector search demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        return 1

    logger.success("Advanced Vector Search Demos Finished Successfully!", emoji_key="complete")
    console.print(Rule("[bold magenta]Advanced Vector Search Demos Complete[/bold magenta]"))
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
