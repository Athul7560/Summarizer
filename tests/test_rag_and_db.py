from __future__ import annotations

from pathlib import Path

from backend.database import StudentRepository
from backend.rag_service import RAGService


def test_chunk_text_overlap() -> None:
    text = "A" * 1000
    chunks = RAGService.chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_db_attempt_and_topic_aggregation(tmp_path: Path) -> None:
    repo = StudentRepository(str(tmp_path / "test.db"))
    repo.record_quiz_attempt(student_id="s1", topic="math", score=50, metadata="{}")
    repo.record_quiz_attempt(student_id="s1", topic="math", score=90, metadata="{}")
    scores = repo.get_topic_scores("s1")
    assert scores[0]["topic"] == "math"
    assert float(scores[0]["avg_score"]) == 70.0


def test_fallback_retrieval_roundtrip() -> None:
    service = RAGService()
    chunks, docs = service.index_documents(student_id="s1", documents=[("note.txt", b"Thermodynamics law and energy transfer")])
    assert chunks >= 1 and docs == 1
    found = service.retrieve(query="energy", student_id="s1", top_k=1)
    assert found
