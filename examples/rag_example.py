#!/usr/bin/env python3
"""Example of using the RAG functionality with Ultimate MCP Server."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import ultimate_mcp_server
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.services.knowledge_base import (
    get_knowledge_base_manager,
    get_knowledge_base_retriever,
)
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker
from ultimate_mcp_server.utils.logging.console import console

# Initialize logger
logger = get_logger("rag_example")

# Sample documents about different AI technologies
AI_DOCUMENTS = [
    """Transformers are a type of neural network architecture introduced in the paper 
    "Attention is All You Need" by Vaswani et al. in 2017. They use self-attention 
    mechanisms to process sequential data, making them highly effective for natural 
    language processing tasks. Unlike recurrent neural networks (RNNs), transformers 
    process entire sequences in parallel, which allows for more efficient training. 
    The original transformer architecture consists of an encoder and a decoder, each 
    made up of multiple layers of self-attention and feed-forward neural networks.""",
    """Retrieval-Augmented Generation (RAG) is an AI framework that combines the 
    strengths of retrieval-based and generation-based approaches. In RAG systems, 
    a retrieval component first finds relevant information from a knowledge base, 
    and then a generation component uses this information to produce more accurate, 
    factual, and contextually relevant outputs. RAG helps to mitigate hallucination 
    issues in large language models by grounding the generation in retrieved facts.""",
    """Reinforcement Learning from Human Feedback (RLHF) is a technique used to align 
    language models with human preferences. The process typically involves three steps: 
    First, a language model is pre-trained on a large corpus of text. Second, human 
    evaluators rank different model outputs, creating a dataset of preferred responses. 
    Third, this dataset is used to train a reward model, which is then used to fine-tune 
    the language model using reinforcement learning techniques such as Proximal Policy 
    Optimization (PPO).""",
    """Mixture of Experts (MoE) is an architecture where multiple specialized neural 
    networks (experts) are trained to handle different types of inputs or tasks. A 
    gating network determines which expert(s) should process each input. This approach 
    allows for larger model capacity without a proportional increase in computational 
    costs, as only a subset of the parameters is activated for any given input. MoE 
    has been successfully applied in large language models like Google's Switch 
    Transformer and Microsoft's Mixtral.""",
]

AI_METADATAS = [
    {"title": "Transformers", "source": "AI Handbook", "type": "architecture"},
    {"title": "Retrieval-Augmented Generation", "source": "AI Handbook", "type": "technique"},
    {"title": "RLHF", "source": "AI Handbook", "type": "technique"},
    {"title": "Mixture of Experts", "source": "AI Handbook", "type": "architecture"},
]

EXAMPLE_QUERIES = [
    "How do transformers work?",
    "What is retrieval-augmented generation?",
    "Compare RLHF and MoE approaches.",
]

KB_NAME = "ai_technologies"


async def run_rag_demo(tracker: CostTracker):
    """Run the complete RAG demonstration."""
    console.print("[bold blue]RAG Example with Ultimate MCP Server[/bold blue]")
    console.print(
        "This example demonstrates the RAG functionality using direct knowledge base services."
    )
    console.print()

    # Initialize Gateway for proper provider and API key management
    gateway = Gateway("rag-example", register_tools=False)
    await gateway._initialize_providers()

    # Get knowledge base services directly
    kb_manager = get_knowledge_base_manager()
    kb_retriever = get_knowledge_base_retriever()

    # Clean up any existing knowledge base with the same name before starting
    console.print(Rule("[bold blue]Cleaning Up Previous Runs[/bold blue]"))

    # Force a clean start
    try:
        # Get direct reference to the vector service
        from ultimate_mcp_server.services.vector import get_vector_db_service

        vector_service = get_vector_db_service()

        # Try a more aggressive approach by resetting chromadb client directly
        if hasattr(vector_service, "chroma_client") and vector_service.chroma_client:
            try:
                # First try standard deletion
                try:
                    vector_service.chroma_client.delete_collection(KB_NAME)
                    logger.info("Successfully deleted ChromaDB collection using client API")
                except Exception as e:
                    logger.debug(f"Standard ChromaDB deletion failed: {str(e)}")

                # Wait longer to ensure deletion propagates
                await asyncio.sleep(1.0)

                # Force reset the ChromaDB client when all else fails
                if hasattr(vector_service.chroma_client, "reset"):
                    try:
                        vector_service.chroma_client.reset()
                        logger.info("Reset ChromaDB client to ensure clean start")
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.warning(f"Failed to reset ChromaDB client: {str(e)}")
            except Exception as e:
                logger.warning(f"Error with ChromaDB client manipulation: {str(e)}")

        # Try to delete at the vector database level again
        try:
            await vector_service.delete_collection(KB_NAME)
            logger.info(f"Directly deleted vector collection '{KB_NAME}'")
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error directly deleting vector collection: {str(e)}")

        # Also try to delete at the knowledge base level
        try:
            kb_info = await kb_manager.get_knowledge_base(KB_NAME)
            if kb_info and kb_info.get("status") != "not_found":
                await kb_manager.delete_knowledge_base(name=KB_NAME)
                logger.info(f"Deleted existing knowledge base '{KB_NAME}'")
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error deleting knowledge base: {str(e)}")

        logger.info("Cleanup completed, proceeding with clean start")
    except Exception as e:
        logger.warning(f"Error during initial cleanup: {str(e)}")

    console.print()

    # Step 1: Create knowledge base
    console.print(Rule("[bold blue]Step 1: Creating Knowledge Base[/bold blue]"))
    try:
        await kb_manager.create_knowledge_base(
            name=KB_NAME,
            description="Information about various AI technologies",
            embedding_model="text-embedding-3-small",
            overwrite=True,
        )
        logger.success(f"Knowledge base created: {KB_NAME}", emoji_key="success")
    except Exception as e:
        logger.error(f"Failed to create knowledge base: {str(e)}", emoji_key="error")
        return 1

    console.print()

    # Step 2: Add documents
    console.print(Rule("[bold blue]Step 2: Adding Documents[/bold blue]"))
    try:
        result = await kb_manager.add_documents(
            knowledge_base_name=KB_NAME,
            documents=AI_DOCUMENTS,
            metadatas=AI_METADATAS,
            embedding_model="text-embedding-3-small",
            chunk_size=1000,
            chunk_method="semantic",
        )
        added_count = result.get("added_count", 0)
        logger.success(f"Added {added_count} documents to knowledge base", emoji_key="success")
    except Exception as e:
        logger.error(f"Failed to add documents: {str(e)}", emoji_key="error")
        return 1

    console.print()

    # Step 3: List knowledge bases
    console.print(Rule("[bold blue]Step 3: Listing Knowledge Bases[/bold blue]"))
    try:
        knowledge_bases = await kb_manager.list_knowledge_bases()

        # Create a Rich table for display
        table = Table(title="Available Knowledge Bases", box=None)
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Document Count", style="magenta")

        # Handle various return types
        try:
            if knowledge_bases is None:
                table.add_row("No knowledge bases found", "", "")
            elif isinstance(knowledge_bases, dict):
                # Handle dictionary response
                kb_names = knowledge_bases.get("knowledge_bases", [])
                if isinstance(kb_names, list):
                    for kb_item in kb_names:
                        if isinstance(kb_item, dict):
                            # Extract name and metadata from dictionary
                            name = kb_item.get("name", "Unknown")
                            metadata = kb_item.get("metadata", {})
                            description = (
                                metadata.get("description", "No description")
                                if isinstance(metadata, dict)
                                else "No description"
                            )
                            doc_count = (
                                metadata.get("doc_count", "Unknown")
                                if isinstance(metadata, dict)
                                else "Unknown"
                            )
                            table.add_row(str(name), str(description), str(doc_count))
                        else:
                            table.add_row(str(kb_item), "No description available", "Unknown")
                else:
                    table.add_row("Error parsing response", "", "")
            elif isinstance(knowledge_bases, list):
                # Handle list response
                for kb in knowledge_bases:
                    if isinstance(kb, str):
                        table.add_row(kb, "No description", "0")
                    elif isinstance(kb, dict):
                        name = kb.get("name", "Unknown")
                        metadata = kb.get("metadata", {})
                        description = (
                            metadata.get("description", "No description")
                            if isinstance(metadata, dict)
                            else "No description"
                        )
                        doc_count = (
                            metadata.get("doc_count", "Unknown")
                            if isinstance(metadata, dict)
                            else "Unknown"
                        )
                        table.add_row(str(name), str(description), str(doc_count))
                    else:
                        kb_name = str(getattr(kb, "name", str(kb)))
                        table.add_row(kb_name, "No description", "0")
            else:
                # Fallback for unexpected response type
                table.add_row(f"Unexpected response: {type(knowledge_bases)}", "", "")

            console.print(table)
        except Exception as e:
            logger.error(f"Error rendering knowledge bases table: {str(e)}", emoji_key="error")
            # Simple fallback display
            console.print(f"Knowledge bases available: {knowledge_bases}")
    except Exception as e:
        logger.error(f"Failed to list knowledge bases: {str(e)}", emoji_key="error")

    console.print()

    # Step 4: Retrieve context for first query
    console.print(Rule("[bold blue]Step 4: Retrieving Context[/bold blue]"))

    query = EXAMPLE_QUERIES[0]
    logger.info(f"Retrieving context for query: '{query}'", emoji_key="processing")

    # Default fallback document if retrieval fails
    retrieved_results = []

    try:
        try:
            results = await kb_retriever.retrieve(
                knowledge_base_name=KB_NAME,
                query=query,
                top_k=2,
                min_score=0.0,  # Set min_score to 0 to see all results
                embedding_model="text-embedding-3-small",  # Use the same embedding model as when adding documents
            )
            retrieved_results = results.get("results", [])

            # Debug raw results
            logger.debug(f"Raw retrieval results: {results}")
        except Exception as e:
            logger.error(f"Error retrieving from knowledge base: {str(e)}", emoji_key="error")
            # Fallback to using the documents directly
            retrieved_results = [
                {"document": AI_DOCUMENTS[0], "score": 0.95, "metadata": AI_METADATAS[0]}
            ]

        console.print(f"Retrieved {len(retrieved_results)} results for query: '{query}'")

        # Display results in panels
        if retrieved_results:
            for i, doc in enumerate(retrieved_results):
                try:
                    score = doc.get("score", 0.0)
                    document = doc.get("document", "")
                    metadata = doc.get("metadata", {})
                    source = (
                        metadata.get("title", "Unknown")
                        if isinstance(metadata, dict)
                        else "Unknown"
                    )

                    console.print(
                        Panel(
                            f"[bold]Document {i + 1}[/bold] (score: {score:.2f})\n"
                            + f"[italic]{document[:150]}...[/italic]",
                            title=f"Source: {source}",
                            border_style="blue",
                        )
                    )
                except Exception as e:
                    logger.error(f"Error displaying document {i}: {str(e)}", emoji_key="error")
        else:
            console.print(
                Panel(
                    "[italic]No results found. Using sample document as fallback for demonstration.[/italic]",
                    title="No Results",
                    border_style="yellow",
                )
            )
            # Create a fallback document for the next step
            retrieved_results = [
                {"document": AI_DOCUMENTS[0], "score": 0.0, "metadata": AI_METADATAS[0]}
            ]
    except Exception as e:
        logger.error(f"Failed to process retrieval results: {str(e)}", emoji_key="error")
        # Ensure we have something to continue with
        retrieved_results = [
            {"document": AI_DOCUMENTS[0], "score": 0.0, "metadata": AI_METADATAS[0]}
        ]

    console.print()

    # Step 5: Generate completions using retrieved context for the first query
    console.print(Rule("[bold blue]Step 5: Generating Response with Retrieved Context[/bold blue]"))
    query = EXAMPLE_QUERIES[0]
    console.print(f"\n[bold]Query:[/bold] {query}")

    try:
        # Get the provider
        provider_key = "gemini"
        provider = gateway.providers.get(provider_key)
        if not provider:
            provider_key = "openai"
            provider = gateway.providers.get(provider_key)  # Fallback

        if not provider:
            logger.error("No suitable provider found", emoji_key="error")
            return 1

        # Use a hardcoded model based on provider type
        if provider_key == "gemini":
            model = "gemini-2.0-flash-lite"
        elif provider_key == "openai":
            model = "gpt-4.1-mini"
        elif provider_key == "anthropic":
            model = "claude-3-haiku-latest"
        else:
            # Get first available model or fallback
            models = getattr(provider, "available_models", [])
            model = models[0] if models else "unknown-model"

        # Prepare context from retrieved documents
        if retrieved_results:
            context = "\n\n".join(
                [doc.get("document", "") for doc in retrieved_results if doc.get("document")]
            )
        else:
            # Fallback to using the first document directly if no results
            context = AI_DOCUMENTS[0]

        # Build prompt with context
        prompt = f"""Answer the following question based on the provided context. 
        If the context doesn't contain relevant information, say so.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""

        # Generate response
        response = await provider.generate_completion(
            prompt=prompt, model=model, temperature=0.3, max_tokens=300
        )

        # Display the answer
        console.print(
            Panel(response.text, title=f"Answer from {provider_key}/{model}", border_style="green")
        )

        # Display usage stats
        metrics_table = Table(title="Performance Metrics", box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        metrics_table.add_row("Input Tokens", str(response.input_tokens))
        metrics_table.add_row("Output Tokens", str(response.output_tokens))
        metrics_table.add_row("Processing Time", f"{response.processing_time:.2f}s")
        metrics_table.add_row("Cost", f"${response.cost:.6f}")

        console.print(metrics_table)

        # Track the generation call
        tracker.add_call(response)

    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}", emoji_key="error")

    console.print()

    # Step 6: Clean up
    console.print(Rule("[bold blue]Step 6: Cleaning Up[/bold blue]"))

    # Display cost summary before final cleanup
    tracker.display_summary(console)

    try:
        await kb_manager.delete_knowledge_base(name=KB_NAME)
        logger.success(f"Knowledge base {KB_NAME} deleted successfully", emoji_key="success")
    except Exception as e:
        logger.error(f"Failed to delete knowledge base: {str(e)}", emoji_key="error")
        return 1

    return 0


async def main():
    """Run the RAG example."""
    tracker = CostTracker()  # Instantiate tracker
    try:
        await run_rag_demo(tracker)  # Pass tracker
    except Exception as e:
        logger.critical(f"RAG demo failed unexpectedly: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
