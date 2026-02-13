"""Prompt templates and context formatting for the RAG agent."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_SYSTEM_PROMPT = """\
You are a helpful AI assistant powered by IBM watsonx.
Use the provided tools to search the knowledge base and answer questions accurately.
If the retrieved context does not contain the answer, say so — do not fabricate information.
"""

AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def format_retrieved_context(documents: list) -> str:
    """Format a list of retrieved LangChain Documents into a context string.

    This is the "augmenter" step — it takes raw retrieval results and
    produces a single string suitable for injection into a prompt.

    Parameters
    ----------
    documents:
        A list of ``langchain_core.documents.Document`` objects.
    """
    if not documents:
        return "No relevant documents found."

    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)
