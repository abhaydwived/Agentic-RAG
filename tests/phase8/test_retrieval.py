"""
tests/phase8/test_retrieval.py

Phase 8 — Vector, BM25, Hybrid Retrieval and Reranking Tests
Covers:
  - FAISS semantic similarity search
  - BM25 keyword search
  - Reciprocal Rank Fusion (Hybrid) search
  - Reranking with fallback when CrossEncoder unavailable
  - retrieve() end-to-end flow
  - assemble_context() source attribution formatting
  - knowledge_indexer.search_knowledge_base() integration
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag_pipeline import HybridRAGPipeline
from src.knowledge_indexer import search_knowledge_base, _GLOBAL_PIPELINE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(docs):
    """Build a HybridRAGPipeline from a list of plain text strings (no FAISS)."""
    chunks = [
        Document(
            page_content=text,
            metadata={"document_name": f"doc{i}.txt", "page": 1,
                      "chunk_id": f"doc{i}.txt#p1_c1"},
        )
        for i, text in enumerate(docs)
    ]
    return HybridRAGPipeline(vector_store=None, chunks=chunks, enable_reranker=False), chunks


# ---------------------------------------------------------------------------
# BM25 Search
# ---------------------------------------------------------------------------

class TestBM25Search:

    def test_returns_top_match_for_exact_keyword(self):
        pipeline, _ = _make_pipeline([
            "deep neural networks use backpropagation",
            "quantum computing leverages superposition",
            "photosynthesis converts solar energy",
        ])
        results = pipeline.bm25_search("quantum superposition", k=1)
        assert len(results) == 1
        assert "quantum" in results[0][0].page_content.lower()

    def test_empty_query_returns_empty(self):
        pipeline, _ = _make_pipeline(["some content here"])
        results = pipeline.bm25_search("", k=3)
        assert results == []

    def test_no_match_returns_empty(self):
        pipeline, _ = _make_pipeline(["apple pie recipe"])
        results = pipeline.bm25_search("zzzyyyxxx", k=3)
        assert results == []

    def test_scores_are_non_negative(self):
        pipeline, _ = _make_pipeline([
            "machine learning transformers",
            "reinforcement learning agents",
        ])
        results = pipeline.bm25_search("machine learning", k=2)
        assert all(score >= 0 for _, score in results)

    def test_top_result_has_highest_score(self):
        pipeline, _ = _make_pipeline([
            "quantum quantum quantum",
            "photosynthesis biology plant",
        ])
        results = pipeline.bm25_search("quantum", k=2)
        if len(results) > 1:
            assert results[0][1] >= results[1][1]


# ---------------------------------------------------------------------------
# Hybrid Search (RRF)
# ---------------------------------------------------------------------------

class TestHybridSearch:

    def test_returns_results_without_vector_store(self):
        """Hybrid falls back gracefully when FAISS vector_store is None."""
        pipeline, _ = _make_pipeline([
            "python programming language",
            "machine learning neural networks",
        ])
        results = pipeline.hybrid_search("python programming", k=2)
        assert isinstance(results, list)

    def test_bm25_only_result_included(self):
        pipeline, _ = _make_pipeline([
            "photosynthesis in plants absorbs sunlight",
            "machine learning algorithms optimize weights",
            "quantum computing leverages qubit superposition",
            "reinforcement learning agents take actions",
        ])
        results = pipeline.hybrid_search("photosynthesis sunlight", k=2)
        texts = [r.page_content for r in results]
        assert any("photosynthesis" in t.lower() for t in texts)

    def test_result_count_bounded_by_k(self):
        pipeline, _ = _make_pipeline([
            "doc one content",
            "doc two content",
            "doc three content",
        ])
        results = pipeline.hybrid_search("content", k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

class TestReranking:

    def test_rerank_without_crossencoder_uses_fallback(self):
        """When enable_reranker=False, rerank() returns top_k with score 1.0."""
        pipeline, chunks = _make_pipeline([
            "neural network training",
            "quantum entanglement",
        ])
        results = pipeline.rerank("machine learning", chunks, top_k=1)
        assert len(results) == 1
        assert results[0][1] == 1.0  # fallback score

    def test_rerank_empty_docs_returns_empty(self):
        pipeline, _ = _make_pipeline(["some content"])
        results = pipeline.rerank("query", [], top_k=3)
        assert results == []

    def test_rerank_respects_top_k(self):
        pipeline, chunks = _make_pipeline([
            "doc a",
            "doc b",
            "doc c",
            "doc d",
        ])
        results = pipeline.rerank("query", chunks, top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# retrieve() end-to-end (BM25 only, no FAISS)
# ---------------------------------------------------------------------------

class TestRetrieve:

    def test_retrieve_returns_top_k(self):
        pipeline, _ = _make_pipeline([
            "deep learning backpropagation weights",
            "quantum computing superposition qubits",
            "photosynthesis chlorophyll sunlight",
        ])
        results = pipeline.retrieve("deep learning", top_k=1, use_hybrid=True, use_reranker=False)
        assert len(results) == 1

    def test_retrieve_returns_tuples(self):
        pipeline, _ = _make_pipeline(["content one", "content two"])
        results = pipeline.retrieve("content", top_k=2, use_hybrid=True, use_reranker=False)
        for item in results:
            assert isinstance(item, tuple)
            assert isinstance(item[0], Document)
            assert isinstance(item[1], float)

    def test_retrieve_empty_pipeline_returns_empty(self):
        pipeline = HybridRAGPipeline(vector_store=None, chunks=[], enable_reranker=False)
        results = pipeline.retrieve("anything", top_k=3)
        assert results == []


# ---------------------------------------------------------------------------
# assemble_context() — Source Attribution
# ---------------------------------------------------------------------------

class TestContextAssembly:

    def test_basic_format(self):
        pipeline = HybridRAGPipeline(enable_reranker=False)
        chunk = Document(
            page_content="The budget is $50M.",
            metadata={"document_name": "finance.pdf", "page": 4, "chunk_id": "finance.pdf#p4_c1"},
        )
        context = pipeline.assemble_context([(chunk, 0.91)])
        assert "[1] Source: finance.pdf" in context
        assert "Page 4" in context
        assert "Score: 0.910" in context
        assert "The budget is $50M." in context

    def test_multiple_results_numbered(self):
        pipeline = HybridRAGPipeline(enable_reranker=False)
        chunks = [
            (Document(page_content="First passage.", metadata={"document_name": "a.txt", "page": 1, "chunk_id": "a1"}), 0.9),
            (Document(page_content="Second passage.", metadata={"document_name": "b.txt", "page": 2, "chunk_id": "b1"}), 0.8),
        ]
        context = pipeline.assemble_context(chunks)
        assert "[1] Source:" in context
        assert "[2] Source:" in context

    def test_empty_retrieved_returns_no_documents_message(self):
        pipeline = HybridRAGPipeline(enable_reranker=False)
        context = pipeline.assemble_context([])
        assert "No relevant" in context


# ---------------------------------------------------------------------------
# knowledge_indexer.search_knowledge_base integration
# ---------------------------------------------------------------------------

class TestSearchKnowledgeBase:

    def test_returns_fallback_when_no_pipeline(self):
        """Without a loaded pipeline, returns a helpful message."""
        import src.knowledge_indexer as ki
        original = ki._GLOBAL_PIPELINE
        ki._GLOBAL_PIPELINE = None
        try:
            result = search_knowledge_base("any query", vector_store=None)
            assert isinstance(result, list)
            assert "No knowledge base available" in result[0]
        finally:
            ki._GLOBAL_PIPELINE = original

    def test_uses_global_pipeline_when_available(self):
        """search_knowledge_base uses _GLOBAL_PIPELINE, not the passed vector_store."""
        import src.knowledge_indexer as ki
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = [
            (Document(page_content="Relevant content.", metadata={"document_name": "x.txt", "page": 1, "chunk_id": "x1"}), 1.0)
        ]
        mock_pipeline.assemble_context.return_value = "[1] Source: x.txt\nRelevant content."

        original = ki._GLOBAL_PIPELINE
        ki._GLOBAL_PIPELINE = mock_pipeline
        try:
            result = search_knowledge_base("test query")
            assert "[1] Source: x.txt" in result[0]
            mock_pipeline.retrieve.assert_called_once()
        finally:
            ki._GLOBAL_PIPELINE = original

