"""
main.py — Agentic RAG with Real-Time Tools, LangGraph, and Session Memory

Supports Google Gemini (default, free tier available at https://aistudio.google.com)
as well as OpenAI fallback. Provides multi-session memory and reset capabilities.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Load .env before importing project modules
load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_api_keys(config: dict) -> None:
    """
    Print a startup banner showing which LLM and tools are ready / missing API keys.
    """
    provider = config.get("provider", "gemini")
    gemini_key = config.get("gemini_api_key")
    openai_key = config.get("openai_api_key")
    tavily_key = config.get("tavily_api_key")
    owm_key = config.get("openweathermap_api_key")

    rag_status    = "✅ Hybrid RAG ready (FAISS + BM25 + Cross-Encoder)"
    finance_status = "✅ Finance Tool ready (yfinance — no key needed)"
    wiki_status   = "✅ Wikipedia Tool ready (no key needed)"
    web_status    = "✅ Web Search ready" if tavily_key else "❌ Web Search (no TAVILY_API_KEY)"
    weather_status = "✅ Weather Tool ready" if owm_key else "⚠️  Weather Tool (mock mode — no OPENWEATHERMAP_API_KEY)"

    has_gemini = bool(gemini_key and gemini_key != "your_gemini_api_key_here")
    has_openai = bool(openai_key and openai_key != "your_openai_api_key_here")

    if provider == "gemini":
        llm_status = f"✅ Google Gemini ({config['gemini_model']})" if has_gemini else "❌ Google Gemini (no GEMINI_API_KEY — free at https://aistudio.google.com)"
    else:
        llm_status = f"✅ OpenAI ({config['openai_model']})" if has_openai else "❌ OpenAI (no OPENAI_API_KEY)"

    print("\n" + "=" * 60)
    print("  Agentic RAG — LangGraph Architecture with Memory")
    print("=" * 60)
    for status in [llm_status, rag_status, finance_status, wiki_status, web_status, weather_status]:
        print(f"  {status}")
    print("=" * 60)

    if provider == "gemini" and not has_gemini:
        if has_openai:
            print("\n[NOTE] GEMINI_API_KEY not found, but OPENAI_API_KEY detected. Switching to OpenAI.")
            config["provider"] = "openai"
        else:
            print("\n[ERROR] GEMINI_API_KEY is required. Get a free key at https://aistudio.google.com/app/apikey and add it to your .env file:")
            print("  GEMINI_API_KEY=your_key_here")
            sys.exit(1)
    elif provider == "openai" and not has_openai:
        if has_gemini:
            print("\n[NOTE] OPENAI_API_KEY not found, but GEMINI_API_KEY detected. Switching to Gemini.")
            config["provider"] = "gemini"
        else:
            print("\n[ERROR] OPENAI_API_KEY is required. Add it to your .env file or use Gemini (GEMINI_API_KEY).")
            sys.exit(1)


def _print_example_queries() -> None:
    """Print suggested example queries so new users know what to try."""
    print("\nExample queries:")
    print('  • "What is the current price of AAPL?"')
    print('  • "What\'s the weather in London today?"')
    print('  • "Search Wikipedia for transformer neural networks"')
    print('  • "What does our internal strategy document say about AI adoption?"')
    print('  • "What is AAPL price and how does it compare to our internal valuation?"')
    print('  • "What are latest AI news stories relevant to our strategy?"')
    print("\nCommands in interactive mode:")
    print("  • 'new' or 'reset'  -> Start a fresh isolated conversation session")
    print("  • 'quit' or 'exit'  -> End the session")
    print()


def _build_config() -> dict:
    """Read all configuration from environment variables and return as a dict."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    default_provider = "gemini" if (gemini_key or not openai_key or openai_key == "your_openai_api_key_here") else "openai"

    return {
        "provider":                os.getenv("LLM_PROVIDER", default_provider).lower(),
        "gemini_api_key":          gemini_key,
        "gemini_model":            os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "openai_api_key":          openai_key,
        "openai_model":            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "tavily_api_key":          os.getenv("TAVILY_API_KEY", ""),
        "openweathermap_api_key":  os.getenv("OPENWEATHERMAP_API_KEY", ""),
        "domain_description":      "internal company documents and knowledge base",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG powered by LangGraph with real-time tools, memory checkpoints, and Google Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kb-dir",
        default=os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base"),
        help="Directory containing .pdf and .txt files to index (default: data/knowledge_base)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default=None,
        help="LLM provider: 'gemini' (default) or 'openai'",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use (default: gemini-2.5-flash for Gemini, gpt-4o-mini for OpenAI)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Custom session identifier for conversation persistence (default: auto-generated)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a single query and exit.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive multi-turn Q&A session.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable conversation memory (each query is independent).",
    )
    verbose_group = parser.add_mutually_exclusive_group()
    verbose_group.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=True,
        help="Show the agent's reasoning trace (default: on).",
    )
    verbose_group.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Hide the agent's reasoning trace.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    config = _build_config()

    if args.provider:
        config["provider"] = args.provider.lower()

    if args.model:
        if config["provider"] == "gemini":
            config["gemini_model"] = args.model
        else:
            config["openai_model"] = args.model

    _check_api_keys(config)

    # --- Step 1: Index / load the knowledge base ---
    print(f"\n[Setup] Indexing knowledge base from '{args.kb_dir}' …")
    from src.knowledge_indexer import index_knowledge_base
    vector_store = index_knowledge_base(
        kb_dir=args.kb_dir,
        index_path=os.path.join(args.kb_dir, ".faiss_index"),
    )

    # --- Step 2: Build tool registry ---
    print("[Setup] Building tool registry …")
    from src.tool_registry import build_tool_registry, get_tool_descriptions
    tools = build_tool_registry(vector_store, config)
    print(get_tool_descriptions(tools))

    # --- Step 3: Instantiate the LLM (Gemini or OpenAI) ---
    if config["provider"] == "gemini":
        print(f"\n[Setup] Connecting to Google Gemini model '{config['gemini_model']}' …")
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=config["gemini_model"],
            google_api_key=config["gemini_api_key"],
            temperature=0,
        )
    else:
        print(f"\n[Setup] Connecting to OpenAI model '{config['openai_model']}' …")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config["openai_model"],
            openai_api_key=config["openai_api_key"],
            temperature=0,
        )

    # --- Step 4: Create LangGraph agent with Session Memory ---
    use_memory = not args.no_memory
    print(f"[Setup] Creating LangGraph agent with checkpoint memory={'on' if use_memory else 'off'} …")
    from src.agent import create_agent, run_agent_query
    from src.session_manager import SessionManager

    session_manager = SessionManager()
    current_session_id = args.session_id or session_manager.create_session()

    agent = create_agent(
        tools=tools,
        llm=llm,
        vector_store=vector_store,
        memory=use_memory,
        checkpointer=session_manager.checkpointer if use_memory else None,
        verbose=args.verbose,
    )

    # --- Step 5: Run query/interactive loop ---
    from src.response_formatter import format_response

    def _run_and_display(query: str, session_id: str) -> None:
        """Run a single query through the LangGraph agent and print the formatted response."""
        print(f"\n[Session: {session_id[:8]}...] Query: {query}\n")
        result = run_agent_query(query, agent, session_id=session_id)

        answer = result.get("output", result.get("generation", "No answer generated."))
        tools_used = list(set(result.get("tools_used", [])))
        print("\n" + format_response(answer, tools_used))

    if args.query:
        _run_and_display(args.query, current_session_id)

    elif args.interactive:
        _print_example_queries()
        print(f"Active Session ID: {current_session_id}")

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if user_input.lower() in {"new", "reset", "clear"}:
                current_session_id = session_manager.reset_session(current_session_id)
                print(f"\n🔄 Started a new isolated session! New Session ID: {current_session_id}")
                continue

            _run_and_display(user_input, current_session_id)

    else:
        print("\nNo query mode selected. Use --query or --interactive.")
        _print_example_queries()
        print("Run with --help for all options.")


if __name__ == "__main__":
    main()
