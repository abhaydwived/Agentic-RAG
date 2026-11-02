"""
tests/phase8/test_chunking_embedding.py

Phase 8 — Chunking and Embedding Tests
Covers:
  - Recursive text splitting with configurable chunk_size / overlap
  - chunk_id, chunk_index, page, document_name metadata preservation
  - Chunk size bounds
  - Embedding dimension and type (CPU, no GPU required)
  - Embedding provider interface
"""
import pytest
from langchain_core.documents import Document
from unittest.mock import patch, MagicMock

from src.rag_pipeline import (
    split_documents_with_metadata,
    HuggingFaceEmbeddingProvider,
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:

    def _make_doc(self, text, name="test.txt", page=1):
        return Document(
            page_content=text,
            metadata={"document_name": name, "page": page, "source": name},
        )

    def test_single_short_doc_produces_one_chunk(self):
        doc = self._make_doc("Short text.", "short.txt", 1)
        chunks = split_documents_with_metadata([doc], chunk_size=500)
        assert len(chunks) == 1

    def test_long_doc_splits_into_multiple_chunks(self):
        doc = self._make_doc("word " * 300, "long.txt", 1)
        chunks = split_documents_with_metadata([doc], chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunk_id_format(self):
        doc = self._make_doc("word " * 300, "report.pdf", 5)
        chunks = split_documents_with_metadata([doc], chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert chunk.metadata["chunk_id"].startswith("report.pdf#p5_c")

    def test_document_name_preserved_in_chunks(self):
        doc = self._make_doc("word " * 300, "myfile.txt", 2)
        chunks = split_documents_with_metadata([doc], chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata["document_name"] == "myfile.txt"

    def test_page_number_preserved_in_chunks(self):
        doc = self._make_doc("word " * 300, "paged.pdf", 7)
        chunks = split_documents_with_metadata([doc], chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata["page"] == 7

    def test_chunk_index_increments_per_document(self):
        doc = self._make_doc("word " * 300, "seq.txt", 1)
        chunks = split_documents_with_metadata([doc], chunk_size=100, chunk_overlap=10)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(1, len(chunks) + 1))

    def test_chunk_size_respected(self):
        doc = self._make_doc("a " * 500, "big.txt", 1)
        chunks = split_documents_with_metadata([doc], chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk.page_content) <= 300  # allow a small margin for separators

    def test_multiple_docs_independent_chunk_ids(self):
        doc1 = self._make_doc("word " * 200, "d1.txt", 1)
        doc2 = self._make_doc("word " * 200, "d2.txt", 1)
        chunks = split_documents_with_metadata([doc1, doc2], chunk_size=100, chunk_overlap=10)
        ids = [c.metadata["chunk_id"] for c in chunks]
        d1_ids = [i for i in ids if i.startswith("d1.txt")]
        d2_ids = [i for i in ids if i.startswith("d2.txt")]
        assert len(d1_ids) > 0
        assert len(d2_ids) > 0
        assert set(d1_ids).isdisjoint(set(d2_ids))

    def test_empty_doc_list_returns_empty(self):
        assert split_documents_with_metadata([]) == []


# ---------------------------------------------------------------------------
# Embedding Provider
# ---------------------------------------------------------------------------

class TestEmbeddingProvider:

    def test_provider_returns_embeddings_object(self):
        """Embedding model loads from HuggingFace local cache (no GPU, no network)."""
        provider = HuggingFaceEmbeddingProvider()
        embeddings = provider.get_embeddings()
        assert embeddings is not None

    def test_embeddings_are_cached(self):
        """Calling get_embeddings() twice returns the same object (no double-load)."""
        provider = HuggingFaceEmbeddingProvider()
        e1 = provider.get_embeddings()
        e2 = provider.get_embeddings()
        assert e1 is e2

    def test_embed_query_returns_list_of_floats(self):
        """Embedding a single string produces a numeric vector on CPU."""
        provider = HuggingFaceEmbeddingProvider()
        embeddings = provider.get_embeddings()
        vec = embeddings.embed_query("What is machine learning?")
        assert isinstance(vec, list)
        assert len(vec) == 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
        assert all(isinstance(v, float) for v in vec)

    def test_embed_documents_batch(self):
        """Batch embedding produces one vector per document."""
        provider = HuggingFaceEmbeddingProvider()
        embeddings = provider.get_embeddings()
        texts = ["Hello world.", "Quantum computing.", "Deep learning."]
        vecs = embeddings.embed_documents(texts)
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_custom_model_name_stored(self):
        provider = HuggingFaceEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"

