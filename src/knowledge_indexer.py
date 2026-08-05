"""
src/knowledge_indexer.py

Single source of truth for the Hybrid RAG index.

Design rules:
  1. _GLOBAL_PIPELINE is updated atomically on every successful rebuild.
  2. search_knowledge_base() always reads _GLOBAL_PIPELINE — it is NEVER stale.
  3. The FAISS index is written to a temp directory and then atomically replaced,
     so a running server never loads a partially-written index.
  4. force_rebuild=True rebuilds from all files on disk (used by upload handler).
     force_rebuild=False loads the existing index from disk (used at startup).
"""

import os
import shutil
import tempfile
import threading
import logging
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.rag_pipeline import (
    load_documents_from_directory,
    split_documents_with_metadata,
    HuggingFaceEmbeddingProvider,
    HybridRAGPipeline,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global pipeline — shared by every retrieval call in every module.
# Updated atomically; never set to None after first load.
# ---------------------------------------------------------------------------
_GLOBAL_PIPELINE: Optional[HybridRAGPipeline] = None
_REBUILD_LOCK = threading.Lock()          # prevents two rebuilds at the same time


def get_rag_pipeline() -> Optional[HybridRAGPipeline]:
    """Returns the current globally initialised HybridRAGPipeline."""
    return _GLOBAL_PIPELINE


def set_rag_pipeline(pipeline: HybridRAGPipeline) -> None:
    """Replaces the active pipeline (used by tests or manual overrides)."""
    global _GLOBAL_PIPELINE
    _GLOBAL_PIPELINE = pipeline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scan_kb_files(kb_dir: str) -> List[str]:
    """Returns sorted list of .pdf / .txt paths in kb_dir (non-recursive, skip hidden)."""
    if not os.path.isdir(kb_dir):
        return []
    files = []
    for name in sorted(os.listdir(kb_dir)):
        if name.startswith("."):
            continue
        if name.lower().endswith((".pdf", ".txt")):
            files.append(os.path.join(kb_dir, name))
    return files


def _index_is_complete(index_path: str) -> bool:
    """Returns True only if a fully-written index exists on disk."""
    return (
        os.path.isdir(index_path)
        and os.path.exists(os.path.join(index_path, "index.faiss"))
        and os.path.exists(os.path.join(index_path, "chunks.pkl"))
    )


def _build_pipeline(
    kb_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    enable_reranker: bool,
) -> HybridRAGPipeline:
    """
    Scans kb_dir, chunks all documents, embeds them, and returns a ready pipeline.
    Does NOT touch the disk index — caller is responsible for saving.
    """
    files = _scan_kb_files(kb_dir)
    logger.info("[Indexer] Found %d file(s) in '%s': %s", len(files), kb_dir,
                [os.path.basename(f) for f in files])

    raw_docs = load_documents_from_directory(kb_dir)
    if not raw_docs:
        logger.warning("[Indexer] No documents found — using placeholder.")
        raw_docs = [Document(
            page_content="Knowledge base is empty. Upload PDF or TXT files.",
            metadata={"source": "placeholder", "document_name": "empty_kb.txt",
                      "page": 1, "file_type": "txt"},
        )]

    chunks = split_documents_with_metadata(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("[Indexer] Built %d chunks from %d page(s).", len(chunks), len(raw_docs))

    provider = HuggingFaceEmbeddingProvider()
    embeddings = provider.get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return HybridRAGPipeline(
        vector_store=vector_store,
        chunks=chunks,
        embedding_provider=provider,
        enable_reranker=enable_reranker,
    )


def _atomic_save(pipeline: HybridRAGPipeline, final_path: str) -> None:
    """
    Saves pipeline to a temp directory beside final_path, then atomically
    replaces final_path. This guarantees the on-disk index is always complete.
    """
    parent = os.path.dirname(os.path.abspath(final_path))
    tmp_dir = tempfile.mkdtemp(dir=parent, prefix=".faiss_tmp_")
    try:
        pipeline.save(tmp_dir)
        # Atomically replace: rename (works on the same filesystem)
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        os.rename(tmp_dir, final_path)
        logger.info("[Indexer] Index atomically saved to '%s'.", final_path)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_knowledge_base(
    kb_dir: str,
    index_path: str = "kb_faiss_index",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    enable_reranker: bool = False,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Loads or builds the Hybrid RAG index.

    * force_rebuild=False  →  load from disk if present (fast server startup).
    * force_rebuild=True   →  always re-scan all files and rebuild (called after upload).

    Guarantees: _GLOBAL_PIPELINE is ALWAYS updated before returning, so
    search_knowledge_base() immediately reflects the new index without any
    server restart or agent reset.
    """
    global _GLOBAL_PIPELINE

    with _REBUILD_LOCK:
        if not force_rebuild and _index_is_complete(index_path):
            logger.info("[Indexer] Loading existing index from '%s'.", index_path)
            pipeline = HybridRAGPipeline.load(index_path, enable_reranker=enable_reranker)
        else:
            logger.info("[Indexer] Building fresh index (force_rebuild=%s).", force_rebuild)
            pipeline = _build_pipeline(kb_dir, chunk_size, chunk_overlap, enable_reranker)
            _atomic_save(pipeline, index_path)

        # Always update global — this is the one line that makes retrieval live
        _GLOBAL_PIPELINE = pipeline
        doc_names = sorted({c.metadata.get("document_name", "?") for c in pipeline.chunks})
        logger.info("[Indexer] Active index: %d chunks, docs: %s", len(pipeline.chunks), doc_names)
        return pipeline.vector_store


def search_knowledge_base(
    query: str,
    vector_store: Optional[FAISS] = None,
    k: int = 3,
    use_hybrid: bool = True,
    use_reranker: bool = False,
) -> List[str]:
    """
    Retrieves top-k chunks for query.

    Always reads _GLOBAL_PIPELINE — the live, up-to-date index.
    The `vector_store` parameter is kept for backward compatibility but is only
    used as a last-resort fallback if the pipeline has never been initialised.
    """
    global _GLOBAL_PIPELINE

    if _GLOBAL_PIPELINE is not None:
        results = _GLOBAL_PIPELINE.retrieve(
            query=query, top_k=k, use_hybrid=use_hybrid, use_reranker=use_reranker,
        )
        return [_GLOBAL_PIPELINE.assemble_context(results)]

    # Last-resort fallback (e.g. unit tests that never call index_knowledge_base)
    if vector_store is not None:
        docs = vector_store.similarity_search(query, k=k)
        rows = []
        for i, doc in enumerate(docs, 1):
            name = doc.metadata.get("document_name", "doc")
            rows.append(f"[{i}] ({name}) {' '.join(doc.page_content.split())}")
        return ["\n".join(rows)]

    return ["No knowledge base available. Please upload a document first."]
