"""LLM client factories for IBM watsonx."""

from __future__ import annotations

from ibm_watsonx_ai import APIClient
from langchain_ibm import ChatWatsonx, WatsonxLLM

from app.core.client import get_watsonx_client


def get_chat_model(
    model_id: str = "ibm/granite-3-8b-instruct",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    *,
    watsonx_client: APIClient | None = None,
) -> ChatWatsonx:
    """Create a ChatWatsonx instance for conversational use.

    Parameters
    ----------
    model_id:
        The watsonx model identifier.
    max_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature.
    watsonx_client:
        Pre-built APIClient; defaults to the shared singleton.
    """
    client = watsonx_client or get_watsonx_client()
    return ChatWatsonx(
        model_id=model_id,
        watsonx_client=client,
        params={
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
    )

def get_llm(
    model_id: str = "ibm/granite-3-8b-instruct",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    *,
    watsonx_client: APIClient | None = None,
) -> WatsonxLLM:
    """Create a WatsonxLLM instance for plain text completion.

    Parameters
    ----------
    model_id:
        The watsonx model identifier.
    max_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature.
    watsonx_client:
        Pre-built APIClient; defaults to the shared singleton.
    """
    client = watsonx_client or get_watsonx_client()
    return WatsonxLLM(
        model_id=model_id,
        watsonx_client=client,
        params={
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
    )
