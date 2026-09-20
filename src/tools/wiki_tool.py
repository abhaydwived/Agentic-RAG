"""
src/tools/wiki_tool.py

Provides Wikipedia lookups via direct, fast REST queries to the Wikipedia API.
Avoids slow/hanging third-party wrapper lookups and handles rate limits cleanly.
"""

import urllib.parse
import requests
from langchain_core.tools import Tool


def create_wiki_tool() -> Tool:
    """
    Build a LangChain Tool that queries Wikipedia directly via its official MediaWiki API.
    """

    def search_wikipedia(query: str) -> str:
        """
        Search Wikipedia for the query and return a concise, factual summary.
        """
        query = query.strip()
        if not query:
            return "Please provide a search topic for Wikipedia."

        headers = {
            "User-Agent": "AgenticRAG/2.0 (https://github.com/techwithprateek/agenticai-projects; contact@example.com)"
        }

        try:
            # 1. Search for closest matching page title
            search_url = (
                "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
                f"{urllib.parse.quote(query)}&utf8=&format=json&srlimit=1"
            )
            resp = requests.get(search_url, headers=headers, timeout=5)
            if resp.status_code != 200:
                return f"Wikipedia search unavailable (status {resp.status_code})."

            data = resp.json()
            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return f"No Wikipedia article found for '{query}'."

            title = search_results[0].get("title", query)

            # 2. Fetch page extract/summary
            extract_url = (
                "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=&explaintext=&titles="
                f"{urllib.parse.quote(title)}&format=json"
            )
            extract_resp = requests.get(extract_url, headers=headers, timeout=5)
            if extract_resp.status_code != 200:
                return f"Could not retrieve Wikipedia content for '{title}'."

            pages = extract_resp.json().get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                extract = page_data.get("extract", "").strip()
                if extract:
                    # Truncate to first 600 characters for token efficiency
                    if len(extract) > 600:
                        extract = extract[:600] + "…"
                    return f"Wikipedia summary for '{title}':\n{extract}"

            return f"No summary text available on Wikipedia for '{title}'."

        except requests.exceptions.Timeout:
            return f"Wikipedia lookup timed out for '{query}'. Please try again."
        except Exception as exc:
            return f"Wikipedia lookup error: {exc}"

    return Tool(
        name="search_wikipedia",
        func=search_wikipedia,
        description=(
            "Search Wikipedia for factual, encyclopaedic information, history, sports, or public biographies. "
            "Input: a topic or search term string."
        ),
    )
