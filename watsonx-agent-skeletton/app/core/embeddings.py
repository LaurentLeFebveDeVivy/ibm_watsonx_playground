"""Embedding model factories."""

from __future__ import annotations

from ibm_watsonx_ai import APIClient
from langchain_ibm import WatsonxEmbeddings

from app.core.client import get_watsonx_client


def get_embedding_model(
    model_id: str = "ibm/slate-125m-english-rtrvr-v2",
    *,
    watsonx_client: APIClient | None = None,
) -> WatsonxEmbeddings:
    """Create a WatsonxEmbeddings instance for vector encoding.

    Parameters
    ----------
    model_id:
        The watsonx embedding model identifier.
    watsonx_client:
        Pre-built APIClient; defaults to the shared singleton.
    """
    client = watsonx_client or get_watsonx_client()
    return WatsonxEmbeddings(
        model_id=model_id,
        watsonx_client=client,
    )
