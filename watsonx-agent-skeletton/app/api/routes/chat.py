"""Chat endpoint — invoke the RAG agent."""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from langchain_core.messages import HumanMessage

from app.agents.rag_agent import build_agent
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the RAG agent and get a response."""
    session_id = request.session_id or str(uuid.uuid4())
    agent = build_agent()

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config={"configurable": {"thread_id": session_id}},
    )

    ai_message = result["messages"][-1]
    return ChatResponse(reply=ai_message.content, session_id=session_id)
