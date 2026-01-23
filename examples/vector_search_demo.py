#!/usr/bin/env python
"""Vector database and semantic search demonstration for Ultimate MCP Server."""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import get_provider
from ultimate_mcp_server.services.vector import get_embedding_service, get_vector_db_service
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker

# --- Add Rich Imports ---
from ultimate_mcp_server.utils.logging.console import console

# ----------------------

# Initialize logger
logger = get_logger("example.vector_search")


async def demonstrate_vector_operations():
    """Demonstrate basic vector database operations using Rich."""
    console.print(Rule("[bold blue]Vector Database Operations Demo[/bold blue]"))
    logger.info("Starting vector database demonstration", emoji_key="start")

    embedding_service = get_embedding_service()
    vector_db = get_vector_db_service()

    if not embedding_service or not hasattr(embedding_service, "client"):
        logger.critical(
            "Failed to initialize embedding service. Is OPENAI_API_KEY configured correctly?",
            emoji_key="critical",
        )
        console.print(
            "[bold red]Error:[/bold red] Embedding service (likely OpenAI) failed to initialize. Check API key."
        )
        return False

    console.print(f"[dim]Vector DB Storage Path: {vector_db.base_dir}[/dim]")

    collection_name = "semantic_search_demo_rich"
    embedding_dimension = 1536  # Default for text-embedding-ada-002 / 3-small

    # --- Setup Collection ---
    try:
        logger.info(f"Creating/resetting collection: {collection_name}", emoji_key="db")
        await vector_db.create_collection(
            name=collection_name,
            dimension=embedding_dimension,
            overwrite=True,
            metadata={"description": "Rich Demo collection"},
        )

        documents = [
            "Machine learning is a field of study in artificial intelligence concerned with the development of algorithms that can learn from data.",
            "Natural language processing (NLP) is a subfield of linguistics and AI focused on interactions between computers and human language.",
            "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            "Transformer models have revolutionized natural language processing with their self-attention mechanism.",
            "Vector databases store and retrieve high-dimensional vectors for tasks like semantic search and recommendation systems.",
            "Embeddings are numerical representations that capture semantic meanings and relationships between objects.",
            "Clustering algorithms group data points into clusters based on similarity metrics.",
            "Reinforcement learning is about how software agents should take actions to maximize cumulative reward.",
            "Knowledge graphs represent knowledge in graph form with entities as nodes and relationships as edges.",
        ]
        document_ids = [f"doc_{i}" for i in range(len(documents))]
        document_metadata = [
            {"domain": "machine_learning", "type": "concept", "id": document_ids[i]}
            for i, doc in enumerate(documents)
        ]

        logger.info(f"Adding {len(documents)} documents...", emoji_key="processing")
        add_start_time = time.time()
        ids = await vector_db.add_texts(
            collection_name=collection_name,
            texts=documents,
            metadatas=document_metadata,
            ids=document_ids,
        )
        add_time = time.time() - add_start_time
        logger.success(f"Added {len(ids)} documents in {add_time:.2f}s", emoji_key="success")

        # --- Basic Search ---
        console.print(Rule("[green]Semantic Search[/green]"))
        query = "How do neural networks work?"
        logger.info(f"Searching for: '{escape(query)}'...", emoji_key="search")
        search_start_time = time.time()
        results = await vector_db.search_by_text(
            collection_name=collection_name, query_text=query, top_k=3, include_vectors=False
        )
        search_time = time.time() - search_start_time
        logger.success(f"Search completed in {search_time:.3f}s", emoji_key="success")

        results_table = Table(title=f'Search Results for: "{escape(query)}"', box=box.ROUNDED)
        results_table.add_column("#", style="dim", justify="right")
        results_table.add_column("Score", style="green", justify="right")
        results_table.add_column("Domain", style="cyan")
        results_table.add_column("Text Snippet", style="white")

        if results:
            for i, res in enumerate(results):
                metadata = res.get("metadata", {})
                text_snippet = escape(
                    res.get("text", "")[:120] + ("..." if len(res.get("text", "")) > 120 else "")
                )
                results_table.add_row(
                    str(i + 1),
                    f"{res.get('similarity', 0.0):.4f}",
                    escape(metadata.get("domain", "N/A")),
                    text_snippet,
                )
        else:
            results_table.add_row("-", "-", "-", "[dim]No results found.[/dim]")
        console.print(results_table)
        console.print()

        # --- Filtered Search ---
        console.print(Rule("[green]Filtered Semantic Search[/green]"))
        filter_query = "embeddings"
        domain_filter = {"domain": "machine_learning"}  # Example filter
        logger.info(
            f"Searching for '{escape(filter_query)}' with filter {escape(str(domain_filter))}...",
            emoji_key="filter",
        )

        f_search_start_time = time.time()
        filtered_results = await vector_db.search_by_text(
            collection_name=collection_name, query_text=filter_query, top_k=3, filter=domain_filter
        )
        f_search_time = time.time() - f_search_start_time
        logger.success(f"Filtered search completed in {f_search_time:.3f}s", emoji_key="success")

        f_results_table = Table(
            title=f'Filtered Results (domain=machine_learning) for: "{escape(filter_query)}"',
            box=box.ROUNDED,
        )
        f_results_table.add_column("#", style="dim", justify="right")
        f_results_table.add_column("Score", style="green", justify="right")
        f_results_table.add_column("Domain", style="cyan")
        f_results_table.add_column("Text Snippet", style="white")

        if filtered_results:
            for i, res in enumerate(filtered_results):
                metadata = res.get("metadata", {})
                text_snippet = escape(
                    res.get("text", "")[:120] + ("..." if len(res.get("text", "")) > 120 else "")
                )
                f_results_table.add_row(
                    str(i + 1),
                    f"{res.get('similarity', 0.0):.4f}",
                    escape(metadata.get("domain", "N/A")),
                    text_snippet,
                )
        else:
            f_results_table.add_row("-", "-", "-", "[dim]No results found.[/dim]")
        console.print(f_results_table)
        console.print()

        # --- Direct Embedding ---
        console.print(Rule("[green]Direct Embedding Generation[/green]"))
        logger.info("Demonstrating direct embedding generation", emoji_key="vector")
        sample_text = "Semantic search helps find conceptually similar content."
        console.print(f"[cyan]Input Text:[/cyan] {escape(sample_text)}")

        emb_start_time = time.time()
        embeddings_list = await embedding_service.create_embeddings([sample_text])
        embedding = embeddings_list[0]
        emb_time = time.time() - emb_start_time
        logger.success(f"Generated embedding in {emb_time:.3f}s", emoji_key="success")

        # Use embedding display utility
        from ultimate_mcp_server.utils.display import _display_embeddings_info

        _display_embeddings_info([embedding], "text-embedding-3-small", console)

        # Also show sample values in a simple format for demo clarity
        console.print(
            f"[cyan]Sample Values (first 5):[/cyan] [dim]{escape(str(embedding[:5]))}...[/dim]"
        )
        console.print()

        return True

    except Exception as e:
        logger.error(f"Error in vector operations: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
        return False
    finally:
        # Clean up collection
        try:
            logger.info(f"Deleting collection: {collection_name}", emoji_key="db")
            await vector_db.delete_collection(collection_name)
        except Exception as del_e:
            logger.warning(
                f"Could not delete collection {collection_name}: {del_e}", emoji_key="warning"
            )


async def demonstrate_llm_with_vector_retrieval(tracker: CostTracker):
    """Demonstrate RAG using vector search and LLM with Rich display."""
    console.print(Rule("[bold blue]Retrieval-Augmented Generation (RAG) Demo[/bold blue]"))
    logger.info("Starting RAG demo", emoji_key="start")

    vector_db = get_vector_db_service()
    # Let get_provider handle key loading internally AND await it
    provider = await get_provider(Provider.OPENAI.value)

    if not provider:
        logger.critical(
            "OpenAI provider failed to initialize for RAG demo. Is OPENAI_API_KEY configured?",
            emoji_key="critical",
        )
        console.print(
            "[bold red]Error:[/bold red] OpenAI provider failed to initialize. Check API key."
        )
        return False

    # Re-create collection and add docs for this demo part
    collection_name = "rag_demo_collection_rich"
    embedding_dimension = 1536
    try:
        logger.info(f"Setting up collection: {collection_name}", emoji_key="db")
        await vector_db.create_collection(
            name=collection_name, dimension=embedding_dimension, overwrite=True
        )
        documents = [
            "Deep learning uses artificial neural networks with many layers (deep architectures).",
            "Neural networks are inspired by biological brains and consist of interconnected nodes or neurons.",
            "While deep learning is a type of machine learning that uses neural networks, not all neural networks qualify as deep learning (e.g., shallow networks).",
            "Key difference: Deep learning implies significant depth (many layers) allowing hierarchical feature learning.",
        ]
        doc_ids = [f"rag_doc_{i}" for i in range(len(documents))]
        doc_metadatas = [
            {"topic": "machine_learning", "source": "demo_document", "id": doc_ids[i]}
            for i in range(len(documents))
        ]
        await vector_db.add_texts(
            collection_name=collection_name, texts=documents, metadatas=doc_metadatas, ids=doc_ids
        )
        logger.success(f"Added {len(documents)} documents for RAG.", emoji_key="success")

        question = "What is the difference between deep learning and neural networks?"
        console.print(f"[cyan]User Question:[/cyan] {escape(question)}")

        # Step 1: Retrieve Context
        logger.info("Retrieving relevant context...", emoji_key="search")
        search_start_time = time.time()
        search_results = await vector_db.search_by_text(
            collection_name=collection_name, query_text=question, top_k=3
        )
        search_time = time.time() - search_start_time

        logger.success(
            f"Retrieved {len(search_results)} context snippets in {search_time:.3f}s.",
            emoji_key="success",
        )

        # Use vector results display utility
        from ultimate_mcp_server.utils.display import _display_vector_results

        _display_vector_results(search_results, console)

        # Join context for LLM
        context_texts = [result["text"] for result in search_results]
        context = "\n\n".join(context_texts)
        console.print(
            Panel(
                escape(context),
                title="[yellow]Retrieved Context[/yellow]",
                border_style="dim yellow",
                expand=False,
            )
        )

        # Step 2: Generate Answer with Context
        prompt = f"""Answer the following question based *only* on the provided context:

Context:
{context}

Question: {question}

Answer:"""

        logger.info("Generating answer using retrieved context...", emoji_key="processing")
        gen_start_time = time.time()
        result = await provider.generate_completion(
            prompt=prompt,
            model="gpt-4.1-mini",  # Use a capable model
            temperature=0.2,  # Lower temperature for factual answer
            max_tokens=200,
        )
        gen_time = time.time() - gen_start_time
        logger.success("Answer generated.", emoji_key="success")

        # Track cost for the generation step
        tracker.add_call(result)

        # --- Display RAG Result ---
        console.print(
            Panel(
                escape(result.text.strip()),
                title="[bold green]Generated Answer (RAG)[/bold green]",
                border_style="green",
                expand=False,
            )
        )

        stats_table = Table(title="RAG Stats", box=box.MINIMAL, show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Search Time", f"{search_time:.3f}s")
        stats_table.add_row("Generation Time", f"{gen_time:.3f}s")
        stats_table.add_row("Input Tokens", str(result.input_tokens))
        stats_table.add_row("Output Tokens", str(result.output_tokens))
        stats_table.add_row("Total Cost", f"${result.cost:.6f}")
        console.print(stats_table)
        console.print()

        return True

    except Exception as e:
        logger.error(f"Error in RAG demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
        return False
    finally:
        # Clean up collection
        try:
            logger.info(f"Deleting collection: {collection_name}", emoji_key="db")
            await vector_db.delete_collection(collection_name)
        except Exception as del_e:
            logger.warning(
                f"Could not delete collection {collection_name}: {del_e}", emoji_key="warning"
            )


async def main():
    """Run all vector search demonstrations."""
    console.print(Rule("[bold magenta]Vector Search & RAG Demos Starting[/bold magenta]"))
    success = False
    tracker = CostTracker()
    try:
        operations_ok = await demonstrate_vector_operations()
        if operations_ok:
            rag_ok = await demonstrate_llm_with_vector_retrieval(tracker)
            success = rag_ok
        else:
            logger.warning("Skipping RAG demo due to vector operation errors.", emoji_key="skip")

    except Exception as e:
        logger.critical(f"Vector search demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        return 1

    if success:
        logger.success("Vector Search & RAG Demos Finished Successfully!", emoji_key="complete")
        console.print(Rule("[bold magenta]Vector Search & RAG Demos Complete[/bold magenta]"))
        tracker.display_summary(console)
        return 0
    else:
        logger.error("One or more vector search demos failed.", emoji_key="error")
        console.print(Rule("[bold red]Vector Search & RAG Demos Finished with Errors[/bold red]"))
        tracker.display_summary(console)
        return 1


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
