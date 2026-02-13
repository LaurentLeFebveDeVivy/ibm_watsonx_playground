"""ReAct RAG agent built with LangGraph."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from app.core.llm import get_chat_model
from app.prompts.templates import RAG_SYSTEM_PROMPT
from app.tools.search import search_knowledge_base


def build_agent(db_tools: list[BaseTool] | None = None):
    """Build and return a compiled LangGraph ReAct agent.

    The agent is wired with:
    - An IBM watsonx chat model
    - A knowledge base search tool
    - Optional database tools (from watsonx.data SQL toolkit)
    - A system prompt guiding RAG behaviour

    Parameters
    ----------
    db_tools:
        Additional tools (e.g. from ``get_database_tools()``) to give
        the agent access to structured data via watsonx.data.
    """
    llm = get_chat_model()
    tools: list[BaseTool] = [search_knowledge_base]
    if db_tools:
        tools.extend(db_tools)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=RAG_SYSTEM_PROMPT,
    )
    return agent
