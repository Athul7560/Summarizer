from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    student_id: str = "default"
    topic: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    used_context: list[str] = Field(default_factory=list)
    model: str
    fallback: bool


class SummarizeRequest(BaseModel):
    text: str = ""
    context_chunks: list[str] = Field(default_factory=list)


class SummarizeResponse(BaseModel):
    summary: str
    model: str
    fallback: bool


class IngestResponse(BaseModel):
    chunks_indexed: int
    documents_processed: int


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    student_id: str = "default"
    top_k: int = 4


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[str]


class FeatureRequest(BaseModel):
    student_id: str = "default"
    topic: str
    query: str
    top_k: int = 4


class FeatureResponse(BaseModel):
    feature: str
    output: str
    chunks: list[str]


class QuizAttemptCreate(BaseModel):
    student_id: str
    topic: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuizAttemptRead(QuizAttemptCreate):
    id: int
    attempted_at: datetime


class AgentFeedbackCreate(BaseModel):
    student_id: str
    topic: str
    suggestion_type: str
    accepted: bool


class DashboardResponse(BaseModel):
    topic_scores: list[dict[str, Any]]
    trend: list[dict[str, Any]]
    recent_attempts: list[dict[str, Any]]
    weak_topics: list[str]
