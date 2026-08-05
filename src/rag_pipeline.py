"""
src/rag_pipeline.py

Modular, advanced RAG pipeline providing:
1. Document ingestion for PDF and TXT files.
2. Text cleaning and normalization.
3. Configurable recursive text splitting.
4. Metadata preservation (document_name, page_number, chunk_id, file_path).
5. Modular Embedding generation interface.
6. FAISS vector indexing & semantic similarity search.
7. BM25 keyword-based retrieval.
8. Hybrid retrieval combining semantic & BM25 with Reciprocal Rank Fusion (RRF).
9. Cross-encoder reranking.
10. Configurable Top-K retrieval and context assembly with source attribution.
"""

import os
import re
import pickle
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)



# 1. Text Cleaning & Normalization


def clean_and_normalize_text(text: str) -> str:
    """
    Cleans and normalizes extracted document text:
    - Strips null bytes and non-printable control characters.
    - Normalizes Unicode spaces and non-breaking spaces.
    - Resolves hyphenated line breaks (e.g. 'retriev-\nal' -> 'retrieval').
    - Collapses multiple whitespace while preserving paragraph breaks.
    """
    if not text:
        return ""

    # Replace zero-width chars and tabs/newlines normalization
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    # Resolve words broken across lines: e.g. "connec-\ntion" -> "connection"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Replace non-breaking and unusual spaces
    text = re.sub(r"[\xa0\u2000-\u200b]", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse consecutive spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()



# 2. Document Ingestion


def load_documents_from_directory(kb_dir: str) -> List[Document]:
    """
    Scans kb_dir recursively for .pdf and .txt files, applies text normalization,
    and extracts metadata (document_name, source, page, file_type).
    """
    documents: List[Document] = []
    if not os.path.exists(kb_dir):
        logger.warning("Knowledge base directory '%s' does not exist.", kb_dir)
        return documents

    for root, _, files in os.walk(kb_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        raw_text = f.read()
                    cleaned = clean_and_normalize_text(raw_text)
                    if cleaned:
                        doc = Document(
                            page_content=cleaned,
                            metadata={
                                "source": file_path,
                                "document_name": file,
                                "page": 1,
                                "file_type": "txt",
                            },
                        )
                        documents.append(doc)
                except Exception as exc:
                    logger.error("Failed to load text file %s: %s", file_path, exc)

            elif ext == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    for page_idx, page in enumerate(reader.pages, start=1):
                        extracted = page.extract_text() or ""
                        cleaned = clean_and_normalize_text(extracted)
                        if cleaned:
                            doc = Document(
                                page_content=cleaned,
                                metadata={
                                    "source": file_path,
                                    "document_name": file,
                                    "page": page_idx,
                                    "file_type": "pdf",
                                },
                            )
                            documents.append(doc)
                except Exception as exc:
                    logger.error("Failed to load PDF file %s: %s", file_path, exc)

    return documents



# 3. Recursive Text Splitting & Metadata Preservation


def split_documents_with_metadata(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Splits documents into overlapping chunks using RecursiveCharacterTextSplitter,
    preserving and enriching metadata with unique chunk_id and sequence index.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []
    chunk_counter: Dict[str, int] = {}

    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        doc_name = doc.metadata.get("document_name", "unknown")

        for idx, chunk in enumerate(doc_chunks):
            current_count = chunk_counter.get(doc_name, 0) + 1
            chunk_counter[doc_name] = current_count

            # Create clean chunk_id: e.g. "annual_report.pdf#p1_c1"
            page_num = doc.metadata.get("page", 1)
            chunk_id = f"{doc_name}#p{page_num}_c{current_count}"

            # Merge and preserve metadata
            chunk.metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": current_count,
                "page": page_num,
                "document_name": doc_name,
                "source": doc.metadata.get("source", doc_name),
            })
            all_chunks.append(chunk)

    return all_chunks



# 4. Modular Embedding Provider Interface


class BaseEmbeddingProvider(ABC):
    """Abstract base class for pluggable embedding models."""

    @abstractmethod
    def get_embeddings(self) -> Any:
        """Returns the underlying LangChain-compatible embeddings instance."""
        pass


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Local HuggingFace embedding provider (default: all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings = None

    def get_embeddings(self) -> Any:
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._embeddings



# 5. Hybrid Retrieval & Reranking Engine


class HybridRAGPipeline:
    """
    Complete Advanced RAG Engine:
    - Semantic Search (FAISS)
    - Keyword Search (BM25)
    - Reciprocal Rank Fusion (RRF) Hybrid Combination
    - Cross-Encoder Neural Reranking
    - Context Assembly with Source Attribution
    """

    def __init__(
        self,
        vector_store: Optional[FAISS] = None,
        chunks: Optional[List[Document]] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranker: bool = False,
    ):
        self.vector_store = vector_store
        self.chunks: List[Document] = chunks or []
        self.embedding_provider = embedding_provider or HuggingFaceEmbeddingProvider()
        self.cross_encoder_model = cross_encoder_model
        self.enable_reranker = enable_reranker
        self._reranker = None
        self._bm25 = None
        self._tokenized_corpus = []

        if self.chunks:
            self._init_bm25()

    def _init_bm25(self) -> None:
        """Initializes the BM25 index over chunk text."""
        try:
            from rank_bm25 import BM25Okapi
            self._tokenized_corpus = [
                re.findall(r"\w+", doc.page_content.lower()) for doc in self.chunks
            ]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except Exception as exc:
            logger.warning("BM25 initialization failed: %s", exc)
            self._bm25 = None

    def _get_reranker(self):
        """Lazy-loads the cross-encoder model."""
        if self._reranker is None and self.enable_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.cross_encoder_model)
            except Exception as exc:
                logger.warning("Failed to load CrossEncoder (%s); proceeding without reranking.", exc)
                self.enable_reranker = False
        return self._reranker

    def semantic_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Performs vector similarity search via FAISS with scores."""
        if not self.vector_store:
            return []
        try:
            # similarity_search_with_score returns list of (Document, score)
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as exc:
            logger.error("Semantic search failed: %s", exc)
            return []

    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Performs BM25 keyword search over chunks."""
        if not self._bm25 or not self.chunks:
            return []
        tokenized_query = re.findall(r"\w+", query.lower())
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        # Top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Combines Semantic Vector Search and BM25 using Reciprocal Rank Fusion (RRF):
        RRF_score(d) = sum( weight / (rrf_k + rank) )
        """
        fetch_k = max(k * 2, 20)
        semantic_results = self.semantic_search(query, k=fetch_k)
        bm25_results = self.bm25_search(query, k=fetch_k)

        # Map document chunk_id -> {doc, score}
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, (doc, _) in enumerate(semantic_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[chunk_id] = doc
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (semantic_weight / (rrf_k + rank))

        for rank, (doc, _) in enumerate(bm25_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[chunk_id] = doc
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (bm25_weight / (rrf_k + rank))

        # Sort by fused score descending
        sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        return [doc_map[cid] for cid in sorted_ids[:fetch_k]]

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Reranks candidate chunks with CrossEncoder (query, passage) scoring.
        """
        if not documents:
            return []

        reranker = self._get_reranker()
        if not reranker:
            # Fallback if reranker is not available: return top_k with default score 1.0
            return [(doc, 1.0) for doc in documents[:top_k]]

        sentence_pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores = reranker.predict(sentence_pairs)
            scored_docs = list(zip(documents, [float(s) for s in scores]))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:top_k]
        except Exception as exc:
            logger.warning("Reranking failed (%s); returning unranked candidates.", exc)
            return [(doc, 1.0) for doc in documents[:top_k]]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        use_hybrid: bool = True,
        use_reranker: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Full end-to-end retrieval flow:
        Candidate Retrieval (Hybrid or Semantic) -> Cross-Encoder Reranking -> Top-K Results
        """
        if use_hybrid and self._bm25:
            candidates = self.hybrid_search(query, k=top_k * 3)
        else:
            candidates = [doc for doc, _ in self.semantic_search(query, k=top_k * 3)]

        if not candidates:
            return []

        if use_reranker and self.enable_reranker:
            return self.rerank(query, candidates, top_k=top_k)

        return [(doc, 1.0) for doc in candidates[:top_k]]

    def assemble_context(self, retrieved: List[Tuple[Document, float]]) -> str:
        """
        Formats retrieved passages with full source attribution:
        [1] Source: document_name.pdf (Page 2 | Chunk ID: doc.pdf#p2_c3)
        Passage text...
        """
        if not retrieved:
            return "No relevant internal documents found."

        sections = []
        for i, (doc, score) in enumerate(retrieved, start=1):
            doc_name = doc.metadata.get("document_name", "Document")
            page = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            clean_text = " ".join(doc.page_content.split())
            header = f"[{i}] Source: {doc_name} (Page {page} | Chunk ID: {chunk_id} | Score: {score:.3f})"
            sections.append(f"{header}\n{clean_text}")

        return "\n\n".join(sections)

    def save(self, directory: str) -> None:
        """Persists the FAISS vector store and document chunks to disk."""
        os.makedirs(directory, exist_ok=True)
        if self.vector_store:
            self.vector_store.save_local(directory)
        chunks_file = os.path.join(directory, "chunks.pkl")
        with open(chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(
        cls,
        directory: str,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        enable_reranker: bool = False,
    ) -> "HybridRAGPipeline":
        """Loads a persisted FAISS vector store and chunk store from disk."""
        provider = embedding_provider or HuggingFaceEmbeddingProvider()
        embeddings = provider.get_embeddings()

        vector_store = FAISS.load_local(
            directory,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        chunks_file = os.path.join(directory, "chunks.pkl")
        chunks = []
        if os.path.exists(chunks_file):
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)

        return cls(
            vector_store=vector_store,
            chunks=chunks,
            embedding_provider=provider,
            enable_reranker=enable_reranker,
        )

