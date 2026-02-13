"""Ingest endpoint — index documents from COS into the vector store."""

from __future__ import annotations

from fastapi import APIRouter

from app.rag.indexing import index_documents
from app.schemas.ingest import IngestRequest, IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Download documents from COS and index them into Milvus."""
    count = index_documents(
        cos_prefix=request.cos_prefix,
        collection_name=request.collection_name,
    )
    return IngestResponse(
        documents_indexed=count,
        collection_name=request.collection_name,
    )
