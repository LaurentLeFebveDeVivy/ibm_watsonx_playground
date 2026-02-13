"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import chat, ingest
from app.models.base import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle events."""
    init_db()
    yield


def create_app() -> FastAPI:
    """Build and return the FastAPI application instance."""
    app = FastAPI(
        title="watsonx RAG Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(chat.router)
    app.include_router(ingest.router)

    return app
