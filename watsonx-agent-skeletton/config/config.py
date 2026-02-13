"""Configuration: credential dataclasses and env-var loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

ROOT_PATH = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """Return the value of *name* from the environment, or raise."""
    val = os.getenv(name)
    if val is None or not val.strip():
        raise RuntimeError(f"Required environment variable '{name}' is not set")
    return val.strip()


def _optional_env(name: str, default: str) -> str:
    """Return the value of *name* from the environment, or *default*."""
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    return val.strip()


# ---------------------------------------------------------------------------
# watsonx.ai
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WatsonxCredentials:
    url: str
    api_key: str
    project_id: str


def get_watsonx_credentials() -> WatsonxCredentials:
    return WatsonxCredentials(
        url=_require_env("WATSONX_URL"),
        api_key=_require_env("WATSONX_APIKEY"),
        project_id=_require_env("WATSONX_PROJECT_ID"),
    )


# ---------------------------------------------------------------------------
# IBM Cloud Object Storage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class COSCredentials:
    endpoint: str
    api_key: str
    instance_id: str
    bucket_name: str


def get_cos_credentials() -> COSCredentials:
    return COSCredentials(
        endpoint=_require_env("COS_ENDPOINT"),
        api_key=_require_env("COS_API_KEY"),
        instance_id=_require_env("COS_INSTANCE_ID"),
        bucket_name=_require_env("COS_BUCKET_NAME"),
    )


# ---------------------------------------------------------------------------
# Milvus (watsonx.data vector DB)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MilvusCredentials:
    host: str
    port: int
    token: str
    collection_name: str


def get_milvus_credentials() -> MilvusCredentials:
    return MilvusCredentials(
        host=_require_env("MILVUS_HOST"),
        port=int(_require_env("MILVUS_PORT")),
        token=_require_env("MILVUS_TOKEN"),
        collection_name=_optional_env("MILVUS_COLLECTION", "default"),
    )


# ---------------------------------------------------------------------------
# Database (PostgreSQL via IBM Cloud Databases, or local SQLite fallback)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseCredentials:
    url: str
    # TODO


def get_database_credentials() -> DatabaseCredentials:
    return DatabaseCredentials(
        url=_optional_env("DATABASE_URL", "sqlite:///watsonx_agent.db"),
    )
