"""
src/server.py

FastAPI Application for Agentic RAG:
Endpoints:
- POST /chat: Process user queries through LangGraph workflow with session memory.
- POST /upload: Upload PDF/TXT files, ingest into knowledge base, and trigger re-indexing.
- GET /health: Health check, tool availability, and active model status.
- GET /sessions: List active conversation sessions.
- DELETE /sessions/{session_id}: Reset/delete a conversation session.
"""

import os
import shutil
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from src.knowledge_indexer import index_knowledge_base, get_rag_pipeline
from src.tool_registry import build_tool_registry
from src.agent import create_agent, run_agent_query
from src.session_manager import SessionManager

logger = logging.getLogger("agentic_rag_server")

app = FastAPI(
    title="Agentic RAG Real-Time API",
    description="High-performance Agentic RAG backend powered by LangGraph, Google Gemini, and Hybrid Retrieval.",
    version="2.0.0",
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global State & Lazy Initializer
# ---------------------------------------------------------------------------

session_manager = SessionManager()
KB_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base")
os.makedirs(KB_DIR, exist_ok=True)

_config = {
    "provider": os.getenv("LLM_PROVIDER", "gemini").lower(),
    "gemini_api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
    "openweathermap_api_key": os.getenv("OPENWEATHERMAP_API_KEY", ""),
    "domain_description": "internal company documents and knowledge base",
}

_vector_store = None
_tools = None
_agent = None


def _build_llm():
    """Construct the LLM instance from config."""
    if _config["provider"] == "gemini" and _config["gemini_api_key"]:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=_config["gemini_model"],
            google_api_key=_config["gemini_api_key"],
            temperature=0,
        )
    elif _config["openai_api_key"]:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=_config["openai_model"],
            openai_api_key=_config["openai_api_key"],
            temperature=0,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=_config["gemini_model"],
            google_api_key="unconfigured",
            temperature=0,
        )


def get_agent_instance():
    """
    Lazily initializes vector store, tools, and compiled agent on demand.
    Returns the cached agent if already built; rebuilds only when _agent is None.
    """
    global _vector_store, _tools, _agent
    if _agent is not None:
        return _agent

    index_path = os.path.join(KB_DIR, ".faiss_index")
    _vector_store = index_knowledge_base(kb_dir=KB_DIR, index_path=index_path)
    _tools = build_tool_registry(_vector_store, _config)
    llm = _build_llm()

    _agent = create_agent(
        tools=_tools,
        llm=llm,
        vector_store=_vector_store,
        memory=True,
        checkpointer=session_manager.checkpointer,
        verbose=True,
    )
    return _agent


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str = Field(description="The natural language question or instruction.")
    session_id: Optional[str] = Field(default=None, description="Optional session thread identifier.")


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    tools_used: List[str]
    retrieved_sources: Optional[str] = None
    live_external_data: Optional[Dict[str, str]] = None
    iteration_count: int
    is_valid: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing tool availability and model configuration."""
    pipeline = get_rag_pipeline()
    doc_names = []
    chunk_count = 0
    if pipeline and hasattr(pipeline, "chunks"):
        chunk_count = len(pipeline.chunks)
        doc_names = list({c.metadata.get("document_name") for c in pipeline.chunks if c.metadata.get("document_name")})

    return {
        "status": "healthy",
        "provider": _config["provider"],
        "model": _config["gemini_model"] if _config["provider"] == "gemini" else _config["openai_model"],
        "tools_registered": ["search_knowledge_base", "get_stock_data", "search_wikipedia", "web_search", "get_weather"],
        "knowledge_base_dir": KB_DIR,
        "indexed_documents": doc_names,
        "indexed_chunks": chunk_count,
        "has_gemini_key": bool(_config["gemini_api_key"] and _config["gemini_api_key"] != "your_gemini_api_key_here"),
        "has_tavily_key": bool(_config["tavily_api_key"] and _config["tavily_api_key"] != "your_tavily_api_key_here"),
        "has_owm_key": bool(_config["openweathermap_api_key"] and _config["openweathermap_api_key"] != "your_openweathermap_key_here"),
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Processes user queries through the LangGraph Agentic RAG workflow."""
    sid = request.session_id or session_manager.create_session()
    agent_inst = get_agent_instance()
    raw_result = run_agent_query(request.query, agent_inst, session_id=sid)

    answer = raw_result.get("output") or raw_result.get("generation", "No answer generated.")
    tools_used = list(set(raw_result.get("tools_used", [])))
    all_results = raw_result.get("tool_results", {})

    retrieved_sources = all_results.get("search_knowledge_base")
    live_external_data = {k: v for k, v in all_results.items() if k != "search_knowledge_base"}
    validation = raw_result.get("validation", {})

    return ChatResponse(
        session_id=sid,
        query=request.query,
        answer=answer,
        tools_used=tools_used,
        retrieved_sources=retrieved_sources,
        live_external_data=live_external_data if live_external_data else None,
        iteration_count=raw_result.get("iteration_count", 1),
        is_valid=validation.get("is_valid", True),
    )


@app.post("/upload")
def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Uploads a PDF or TXT file to the knowledge base and immediately re-indexes
    ALL documents so queries return results from the new file right away.

    Steps:
      1. Save file to disk.
      2. Fully rebuild the FAISS+BM25 index (force_rebuild=True) — scans every
         file in knowledge_base/ including the newly uploaded one.
         This updates _GLOBAL_PIPELINE inside knowledge_indexer.py so that
         search_knowledge_base() immediately uses the new index (no restart).
      3. Reset _agent so the next /chat call re-initialises with the new vector store.
    """
    allowed_extensions = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .pdf and .txt files are accepted.",
        )

    file_path = os.path.join(KB_DIR, file.filename)
    try:
        # Step 1: Save the uploaded file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved uploaded file: %s", file_path)

        # Step 2: Fully rebuild the index from all documents in KB_DIR.
        #   force_rebuild=True deletes the old stale index and re-scans every file.
        #   _GLOBAL_PIPELINE is updated in-place inside index_knowledge_base(),
        #   so search_knowledge_base() picks up the new documents immediately.
        global _vector_store, _agent
        index_path = os.path.join(KB_DIR, ".faiss_index")
        _vector_store = index_knowledge_base(
            kb_dir=KB_DIR,
            index_path=index_path,
            force_rebuild=True,
        )

        # Step 3: Discard the old agent so the next /chat builds a fresh one
        # with the updated vector store.
        _agent = None
        logger.info("Re-index complete. Agent reset. New uploads will be retrieved immediately.")

        # Report back which documents are now in the index
        pipeline = get_rag_pipeline()
        doc_names = []
        chunk_count = 0
        if pipeline and hasattr(pipeline, "chunks"):
            chunk_count = len(pipeline.chunks)
            doc_names = list({
                c.metadata.get("document_name")
                for c in pipeline.chunks
                if c.metadata.get("document_name")
            })

        return {
            "status": "success",
            "filename": file.filename,
            "message": (
                f"'{file.filename}' uploaded and indexed. "
                f"Knowledge base now has {chunk_count} chunks from {len(doc_names)} document(s)."
            ),
            "documents_in_kb": sorted(doc_names),
        }

    except Exception as exc:
        logger.exception("File upload failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {exc}")


@app.get("/sessions")
def list_sessions() -> Dict[str, Any]:
    """Lists all active session identifiers."""
    sessions = list(session_manager._active_sessions.keys())
    return {
        "active_sessions": sessions,
        "count": len(sessions),
    }


@app.delete("/sessions/{session_id}")
def reset_session_endpoint(session_id: str) -> Dict[str, Any]:
    """Resets/deletes a session thread."""
    new_sid = session_manager.reset_session(session_id)
    return {
        "status": "cleared",
        "old_session_id": session_id,
        "new_session_id": new_sid,
        "message": f"Session '{session_id}' cleared. Fresh session '{new_sid}' initiated.",
    }
