"""Cached singleton APIClient for IBM watsonx.ai."""

from __future__ import annotations

import functools

from ibm_watsonx_ai import APIClient, Credentials

from config.config import get_watsonx_credentials


@functools.lru_cache(maxsize=1)
def get_watsonx_client() -> APIClient:
    """Return a shared APIClient, creating it on first call.

    All modules that need an authenticated IBM watsonx connection
    should import this function rather than building their own client.
    """
    creds = get_watsonx_credentials()
    credentials = Credentials(url=creds.url, api_key=creds.api_key)
    return APIClient(credentials=credentials, project_id=creds.project_id)
