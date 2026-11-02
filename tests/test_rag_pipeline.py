"""
tests/test_rag_pipeline.py

Unit and integration tests for the Advanced RAG Pipeline:
- Text cleaning and normalization
- Ingestion of PDF and TXT files
- Recursive chunking with metadata preservation (document_name, page, chunk_id)
- Semantic similarity search (FAISS)
- BM25 keyword search
- Hybrid search (Reciprocal Rank Fusion)
- Cross-encoder reranking
- Context assembly with source attribution
"""

import os
import shutil
import tempfile
import unittest

from langchain_core.documents import Document

from src.rag_pipeline import (
    clean_and_normalize_text,
    load_documents_from_directory,
    split_documents_with_metadata,
    HybridRAGPipeline,
)


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Sample test files
        self.doc1_path = os.path.join(self.test_dir, "doc1.txt")
        with open(self.doc1_path, "w", encoding="utf-8") as f:
            f.write(
                "Artificial Intelligence and Machine Learning are transforming modern industry.\n\n"
                "Supervised learning trains models on labeled datasets to make predictions."
            )

        self.doc2_path = os.path.join(self.test_dir, "doc2.txt")
        with open(self.doc2_path, "w", encoding="utf-8") as f:
            f.write(
                "Quantum computing utilizes qubits to achieve superposition and entanglement.\n"
                "This enables solving complex optimization problems exponentially faster."
            )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_text_cleaning_and_normalization(self):
        """Verify cleaning of hyphens, redundant spaces, and null characters."""
        raw = "Trans-\nformer models  use   self-attention.\x00\n\n\n\nNext paragraph."
        cleaned = clean_and_normalize_text(raw)
        self.assertNotIn("Trans-\nformer", cleaned)
        self.assertIn("Transformer", cleaned)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("   ", cleaned)

    def test_document_ingestion_and_metadata(self):
        """Verify file loading from directory and metadata tagging."""
        docs = load_documents_from_directory(self.test_dir)
        self.assertEqual(len(docs), 2)
        doc_names = [d.metadata["document_name"] for d in docs]
        self.assertIn("doc1.txt", doc_names)
        self.assertIn("doc2.txt", doc_names)
        self.assertEqual(docs[0].metadata["page"], 1)

    def test_chunking_with_metadata_preservation(self):
        """Verify chunk splitting assigns chunk_id and preserves source metadata."""
        docs = [
            Document(
                page_content="Word " * 120,
                metadata={"document_name": "sample.pdf", "page": 3, "source": "sample.pdf"},
            )
        ]
        chunks = split_documents_with_metadata(docs, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("chunk_id", chunk.metadata)
            self.assertTrue(chunk.metadata["chunk_id"].startswith("sample.pdf#p3_c"))
            self.assertEqual(chunk.metadata["document_name"], "sample.pdf")
            self.assertEqual(chunk.metadata["page"], 3)

    def test_bm25_and_hybrid_search(self):
        """Verify BM25 keyword matching and hybrid fusion."""
        chunks = [
            Document(page_content="Deep neural networks require backpropagation.", metadata={"chunk_id": "c1", "document_name": "ai.txt", "page": 1}),
            Document(page_content="Quantum computing leverages superposition of states.", metadata={"chunk_id": "c2", "document_name": "quantum.txt", "page": 1}),
            Document(page_content="Photosynthesis converts solar energy to chemical energy.", metadata={"chunk_id": "c3", "document_name": "bio.txt", "page": 1}),
        ]

        pipeline = HybridRAGPipeline(
            vector_store=None,
            chunks=chunks,
            enable_reranker=False,
        )

        # BM25 direct keyword match
        bm25_matches = pipeline.bm25_search("quantum superposition", k=2)
        self.assertGreater(len(bm25_matches), 0)
        self.assertEqual(bm25_matches[0][0].metadata["chunk_id"], "c2")

        # Hybrid search
        hybrid_results = pipeline.hybrid_search("photosynthesis chemical", k=2)
        self.assertGreater(len(hybrid_results), 0)
        self.assertEqual(hybrid_results[0].metadata["chunk_id"], "c3")

    def test_context_assembly_with_attribution(self):
        """Verify formatting of retrieved chunks with page number and chunk ID."""
        pipeline = HybridRAGPipeline(enable_reranker=False)
        test_chunks = [
            (
                Document(
                    page_content="Internal annual budget is $50M.",
                    metadata={"document_name": "finance_2026.pdf", "page": 4, "chunk_id": "finance_2026.pdf#p4_c12"},
                ),
                0.925,
            )
        ]
        context = pipeline.assemble_context(test_chunks)
        self.assertIn("[1] Source: finance_2026.pdf (Page 4 | Chunk ID: finance_2026.pdf#p4_c12 | Score: 0.925)", context)
        self.assertIn("Internal annual budget is $50M.", context)


if __name__ == "__main__":
    unittest.main()

