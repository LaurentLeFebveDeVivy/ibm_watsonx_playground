"""Pydantic schemas for the /ingest endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    cos_prefix: str = Field(
        ..., description="Object key prefix in the COS bucket to ingest."
    )
    collection_name: str = Field(
        default="default", description="Milvus collection to index into."
    )


class IngestResponse(BaseModel):
    documents_indexed: int = Field(
        ..., description="Number of document chunks indexed."
    )
    collection_name: str = Field(..., description="The target collection name.")
