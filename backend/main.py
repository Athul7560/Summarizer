from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.agent import study_coach_agent
from backend.config import settings
from backend.database import student_repo
from backend.model_service import llm_service
from backend.rag_service import rag_service
from backend.schemas import (
    AgentFeedbackCreate,
    ChatRequest,
    ChatResponse,
    DashboardResponse,
    FeatureRequest,
    FeatureResponse,
    IngestResponse,
    QuizAttemptCreate,
    RetrieveRequest,
    RetrieveResponse,
    SummarizeRequest,
    SummarizeResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting backend with model=%s embed_model=%s", settings.model_id, settings.embed_model_id)
    settings.ensure_dirs()
    _ = llm_service.fallback_mode
    yield
    logger.info("Shutting down backend")


app = FastAPI(title="AI Study Buddy Backend", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model": llm_service.model_name,
        "fallback": llm_service.fallback_mode,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    context = request.retrieved_context
    if not context and request.topic:
        context = rag_service.retrieve(query=request.topic, student_id=request.student_id, top_k=settings.retrieve_top_k)

    history_text = "\n".join([f"{item.role}: {item.content}" for item in request.history])
    prompt = f"Conversation so far:\n{history_text}\n\nUser: {request.message}" if history_text else request.message
    response = llm_service.generate(task="chat", user_input=prompt, context=context)
    return ChatResponse(response=response, used_context=context, model=llm_service.model_name, fallback=llm_service.fallback_mode)


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    text = request.text.strip()
    context = request.context_chunks
    if not text and not context:
        raise HTTPException(status_code=400, detail="Provide text or context_chunks")

    merged = text if text else "\n\n".join(context)
    # If raw text is provided we keep retrieved chunks as explicit context.
    # If only chunks are provided, they are already merged as the primary input.
    summary_context = context if text else []
    summary = llm_service.generate(task="summarize", user_input=merged, context=summary_context, max_new_tokens=220)
    return SummarizeResponse(summary=summary, model=llm_service.model_name, fallback=llm_service.fallback_mode)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(student_id: str = "default", files: list[UploadFile] = File(...)) -> IngestResponse:
    docs: list[tuple[str, bytes]] = []
    for item in files:
        payload = await item.read()
        docs.append((item.filename or "upload.txt", payload))

    chunks, docs_processed = rag_service.index_documents(student_id=student_id, documents=docs)
    return IngestResponse(chunks_indexed=chunks, documents_processed=docs_processed)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    chunks = rag_service.retrieve(query=request.query, student_id=request.student_id, top_k=request.top_k)
    return RetrieveResponse(query=request.query, chunks=chunks)


@app.post("/feature/{feature_name}", response_model=FeatureResponse)
def generate_feature(feature_name: str, request: FeatureRequest) -> FeatureResponse:
    templates = {
        "summary": "Create a concise study summary",
        "flashcards": "Generate 10 flashcards in Q/A format",
        "quiz": "Generate 5 MCQs with options and answer key",
        "mindmap": "Generate an indented markdown mind map",
    }
    if feature_name not in templates:
        raise HTTPException(status_code=404, detail="Unknown feature")

    chunks = rag_service.retrieve(query=request.query, student_id=request.student_id, top_k=request.top_k)
    output = llm_service.generate(task=templates[feature_name], user_input=request.topic, context=chunks)
    return FeatureResponse(feature=feature_name, output=output, chunks=chunks)


@app.post("/quiz/attempt")
def record_quiz_attempt(payload: QuizAttemptCreate) -> dict[str, int]:
    if payload.score < 0 or payload.score > 100:
        raise HTTPException(status_code=400, detail="score must be between 0 and 100")
    attempt_id = student_repo.record_quiz_attempt(
        student_id=payload.student_id,
        topic=payload.topic,
        score=payload.score,
        metadata=json.dumps(payload.metadata),
    )
    return {"attempt_id": attempt_id}


@app.get("/dashboard/{student_id}", response_model=DashboardResponse)
def dashboard(student_id: str) -> DashboardResponse:
    topic_scores = student_repo.get_topic_scores(student_id)
    trend = student_repo.get_trend(student_id)
    recent_attempts = student_repo.get_recent_attempts(student_id, limit=20)
    weak_topics = study_coach_agent.get_weak_topics(student_id)
    return DashboardResponse(
        topic_scores=topic_scores,
        trend=trend,
        recent_attempts=recent_attempts,
        weak_topics=weak_topics,
    )


@app.get("/coach/recommend/{student_id}")
def coach_recommend(student_id: str) -> dict[str, str]:
    suggestion = study_coach_agent.recommend(student_id)
    return {
        "topic": suggestion.topic,
        "suggestion_type": suggestion.suggestion_type,
        "recommendation": suggestion.recommendation,
    }


@app.post("/coach/feedback")
def coach_feedback(payload: AgentFeedbackCreate) -> dict[str, int]:
    feedback_id = student_repo.record_feedback(
        student_id=payload.student_id,
        topic=payload.topic,
        suggestion_type=payload.suggestion_type,
        accepted=payload.accepted,
    )
    return {"feedback_id": feedback_id}


@app.post("/coach/execute/{student_id}")
def coach_execute(student_id: str) -> dict[str, str]:
    suggestion = study_coach_agent.recommend(student_id)
    output = study_coach_agent.run_tool(suggestion)
    return {
        "topic": suggestion.topic,
        "suggestion_type": suggestion.suggestion_type,
        "output": output,
    }
