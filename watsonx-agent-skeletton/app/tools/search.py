"""Knowledge base search tool for the LangGraph agent."""

from __future__ import annotations

from langchain_core.tools import tool

from app.prompts.templates import format_retrieved_context
from app.rag.retrieval import retrieve_documents


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information relevant to the query.

    Use this tool when you need to find information from indexed documents
    to answer the user's question.

    Parameters
    ----------
    query:
        The search query describing what information is needed.
    """
    docs = retrieve_documents(query)
    return format_retrieved_context(docs)
