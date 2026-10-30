"""
src/app_ui.py

Streamlit Web Interface for Agentic RAG:
Features:
- Interactive multi-turn chat interface
- Session persistence and 'New Chat' button
- Drag-and-drop PDF and TXT document upload to knowledge base
- Clear visual distinction between:
  1. Final Agent Response
  2. Retrieved Internal Knowledge Base Sources (with Page & Chunk IDs)
  3. Live External Information (Stocks, Weather, Wikipedia, Web Search)
  4. Tools invoked and reasoning iterations
"""

import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Agentic RAG Real-Time",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar: Knowledge Base & Session Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚡ Control Panel")
    st.markdown(f"**Session:** `{st.session_state.session_id[:8]}...`")

    if st.button("🔄 New Conversation", use_container_width=True):
        try:
            requests.delete(f"{BACKEND_URL}/sessions/{st.session_state.session_id}", timeout=5)
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Backend health check
    st.subheader("System Health")
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success(f"Backend: Connected ({health_data.get('provider', 'gemini').upper()})")
            with st.expander("Active Tools"):
                for t in health_data.get("tools_registered", []):
                    st.write(f"• `{t}`")
        else:
            st.warning("Backend returned non-200 status.")
    except Exception:
        st.error(f"Cannot connect to backend at `{BACKEND_URL}`. Ensure FastAPI server is running.")

    st.divider()

    # Document Uploader
    st.subheader("📁 Ingest Documents")
    uploaded_file = st.file_uploader("Upload .pdf or .txt to Knowledge Base", type=["pdf", "txt"])
    if uploaded_file is not None:
        if st.button("Process & Index", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_file.name}' into Hybrid RAG (chunking & embedding)..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
                    if resp.status_code == 200:
                        st.success(resp.json().get("message", "File uploaded!"))
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {resp.text}")
                except Exception as exc:
                    st.error(f"Upload error: {exc}")


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

st.title("⚡ Agentic RAG — Real-Time Multi-Tool Intelligence")
st.caption("Powered by LangGraph, Google Gemini, Hybrid Vector Retrieval, and Live External Tools")

# Render prior chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display structured evidence if present
        if "retrieved_sources" in msg and msg["retrieved_sources"]:
            with st.expander("📚 Retrieved Knowledge Base Sources"):
                st.info(msg["retrieved_sources"])

        if "live_external_data" in msg and msg["live_external_data"]:
            with st.expander("🌐 Live External Tool Data"):
                for tool_name, data in msg["live_external_data"].items():
                    st.markdown(f"**Tool:** `{tool_name}`")
                    st.code(data)

        if "tools_used" in msg and msg["tools_used"]:
            st.caption(f"🔧 Tools Used: {', '.join(msg['tools_used'])}")


# Chat Input
if prompt := st.chat_input("Ask about internal documents, live stocks, weather, Wikipedia, or web search..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Agent reasoning and querying tools..."):
            try:
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id,
                }
                resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")
                    tools_used = data.get("tools_used", [])
                    retrieved_sources = data.get("retrieved_sources")
                    live_external_data = data.get("live_external_data")

                    st.markdown(answer)

                    # Show sources & external data separation
                    if retrieved_sources:
                        with st.expander("📚 Retrieved Knowledge Base Sources"):
                            st.info(retrieved_sources)

                    if live_external_data:
                        with st.expander("🌐 Live External Tool Data"):
                            for t_name, t_val in live_external_data.items():
                                st.markdown(f"**Tool:** `{t_name}`")
                                st.code(t_val)

                    if tools_used:
                        st.caption(f"🔧 Tools Used: {', '.join(tools_used)}")

                    # Record assistant message in state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "tools_used": tools_used,
                        "retrieved_sources": retrieved_sources,
                        "live_external_data": live_external_data,
                    })

                else:
                    err_msg = f"Error from backend: {resp.text}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as exc:
                err_msg = f"Failed to communicate with backend: {exc}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

