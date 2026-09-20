"""
src/tools/rag_tool.py

Wraps the Advanced Hybrid RAG pipeline as a LangChain Tool so the LangGraph agent can
invoke it alongside financial, Wikipedia, weather, and web search tools.
"""

from typing import List, Optional

from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS

from src.knowledge_indexer import search_knowledge_base, get_rag_pipeline


def create_rag_tool(
    vector_store: FAISS,
    domain_description: str = "internal company documents",
    top_k: int = 3,
) -> Tool:
    """
    Build a LangChain Tool that searches internal documents via Hybrid Retrieval
    (Semantic FAISS + Keyword BM25 + Cross-Encoder Reranking).

    Args:
        vector_store:       Loaded FAISS index returned by knowledge_indexer.
        domain_description: Contextual description of the knowledge base.
        top_k:              Number of top reranked chunks to retrieve.

    Returns:
        Configured Tool instance.
    """

    def _search(query: str) -> str:
        """
        Executes hybrid retrieval and cross-encoder reranking.
        """
        pipeline = get_rag_pipeline()
        if pipeline:
            retrieved = pipeline.retrieve(query, top_k=top_k, use_hybrid=True, use_reranker=True)
            return pipeline.assemble_context(retrieved)

        chunks: List[str] = search_knowledge_base(query, vector_store, k=top_k)
        if not chunks:
            return "No relevant information found in knowledge base."
        return "\n".join(chunks)

    description = (
        f"Search {domain_description} for relevant information. "
        "Use this for questions about internal policies, product documentation, "
        "strategy, technical architecture, or company knowledge. "
        "Input: a search query string."
    )

    rag_tool = Tool(
        name="search_knowledge_base",
        func=_search,
        description=description,
    )

    return rag_tool
