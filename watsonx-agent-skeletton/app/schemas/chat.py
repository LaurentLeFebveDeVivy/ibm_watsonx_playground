"""Pydantic schemas for the /chat endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message.")
    session_id: str | None = Field(
        default=None, description="Optional session ID for conversation continuity."
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="The agent's response.")
    session_id: str = Field(..., description="Session ID for this conversation.")
