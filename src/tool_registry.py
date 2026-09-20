"""
src/tool_registry.py

Centralized, extensible registry for all agent tools:
- Dynamic registration of custom tools.
- Modular tool factory functions.
- Provides metadata inspection and formatted system descriptions.
- Supports RAG, Finance, Weather, Wikipedia, and Web Search out of the box.
"""

from typing import Callable, Dict, List, Optional
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS

from src.tools.rag_tool import create_rag_tool
from src.tools.web_search_tool import create_web_search_tool, create_mock_web_search_tool
from src.tools.finance_tool import create_finance_tool
from src.tools.weather_tool import create_weather_tool, create_mock_weather_tool
from src.tools.wiki_tool import create_wiki_tool


class ToolRegistry:
    """
    Extensible Tool Registry that manages available tools, dynamic registration,
    and lookup.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """Registers a new LangChain Tool into the registry."""
        self._tools[tool.name] = tool

    def register_custom_function(
        self,
        name: str,
        func: Callable[[str], str],
        description: str,
    ) -> Tool:
        """Quickly wraps a plain Python function and registers it as a Tool."""
        tool = Tool(name=name, func=func, description=description)
        self.register_tool(tool)
        return tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Retrieves a registered tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """Returns all registered tools as a list."""
        return list(self._tools.values())

    def get_tools_map(self) -> Dict[str, Tool]:
        """Returns the dictionary mapping tool names to Tool objects."""
        return dict(self._tools)

    def describe_tools(self) -> str:
        """Returns a formatted markdown summary of all registered tools."""
        lines = ["\nAvailable Agent Tools:"]
        for tool in self._tools.values():
            lines.append(f"  • {tool.name}: {tool.description.strip()}")
        return "\n".join(lines)


def build_tool_registry(
    vector_store: FAISS,
    config: dict,
) -> List[Tool]:
    """
    Instantiates and registers the default set of tools into a ToolRegistry:
    1. search_knowledge_base (Hybrid FAISS + BM25 + Cross-Encoder)
    2. get_stock_data (yfinance)
    3. search_wikipedia (Wikipedia API with User-Agent)
    4. web_search (Tavily search or mock fallback)
    5. get_weather (OpenWeatherMap or mock fallback)

    Returns:
        List of Tool objects.
    """
    registry = ToolRegistry()

    # 1. Knowledge Base RAG Tool
    domain = config.get("domain_description", "internal company documents and knowledge base")
    rag = create_rag_tool(vector_store, domain_description=domain)
    registry.register_tool(rag)

    # 2. Finance Tool (yfinance)
    finance = create_finance_tool()
    registry.register_tool(finance)

    # 3. Wikipedia Tool
    wiki = create_wiki_tool()
    registry.register_tool(wiki)

    # 4. Web Search Tool (Tavily)
    tavily_key = config.get("tavily_api_key")
    if tavily_key:
        web = create_web_search_tool(tavily_key)
    else:
        web = create_mock_web_search_tool()
    registry.register_tool(web)

    # 5. Weather Tool (OpenWeatherMap)
    owm_key = config.get("openweathermap_api_key")
    if owm_key:
        weather = create_weather_tool(owm_key)
    else:
        weather = create_mock_weather_tool()
    registry.register_tool(weather)

    return registry.get_all_tools()


def get_tool_descriptions(tools: List[Tool]) -> str:
    """Formats descriptions of given tools for logging / startup banner."""
    lines = ["\nRegistered Tools:"]
    for t in tools:
        clean_desc = t.description.strip().replace("\n", " ")
        lines.append(f"  • {t.name}: {clean_desc}")
    return "\n".join(lines)
