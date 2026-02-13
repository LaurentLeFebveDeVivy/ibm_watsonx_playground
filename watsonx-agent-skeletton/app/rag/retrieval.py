"""Query-time vector search with optional WatsonxRerank."""

from __future__ import annotations

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_ibm import WatsonxRerank

from app.core.client import get_watsonx_client
from app.rag.vectorstore import get_vectorstore


def _build_reranker(
    model_id: str = "cross-encoder/ms-marco-minilm-l-12-v2",
) -> WatsonxRerank:
    """Create a WatsonxRerank compressor."""
    return WatsonxRerank(
        model_id=model_id,
        watsonx_client=get_watsonx_client(),
    )


def retrieve_documents(
    query: str,
    collection_name: str | None = None,
    k: int = 4,
    rerank: bool = True,
    initial_k: int = 20,
) -> list[Document]:
    """Embed *query* and return the top-k most relevant documents.

    When *rerank* is True the pipeline first retrieves *initial_k*
    candidates from Milvus, then reranks them with WatsonxRerank and
    returns the top *k*.

    Parameters
    ----------
    query:
        The user's search query.
    collection_name:
        Milvus collection to search.
    k:
        Number of documents to return.
    rerank:
        Whether to apply WatsonxRerank post-retrieval.
    initial_k:
        Number of candidates to fetch before reranking.
    """
    vectorstore = get_vectorstore(collection_name)

    if not rerank:
        return vectorstore.similarity_search(query, k=k)

    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": initial_k},
    )
    compressor = _build_reranker()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    return compression_retriever.invoke(query)[:k]
