"""Milvus vector store setup (via watsonx.data)."""

from __future__ import annotations

from langchain_milvus import Milvus

from app.core.embeddings import get_embedding_model
from config.config import get_milvus_credentials


def get_vectorstore(collection_name: str | None = None) -> Milvus:
    """Get a LangChain Milvus vectorstore backed by watsonx embeddings.

    Parameters
    ----------
    collection_name:
        Override the default collection from config.
    """
    creds = get_milvus_credentials()
    name = collection_name or creds.collection_name
    return Milvus(
        collection_name=name,
        embedding_function=get_embedding_model(),
        connection_args={
            "host": creds.host,
            "port": creds.port,
            "token": creds.token,
            "secure": True,
        },
    )
