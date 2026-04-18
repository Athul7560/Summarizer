from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_schema() -> None:
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {'status', 'model', 'fallback'}


def test_summarize_validation_error() -> None:
    response = client.post('/summarize', json={'text': '', 'context_chunks': []})
    assert response.status_code == 400


def test_chat_schema() -> None:
    response = client.post('/chat', json={'message': 'Hello', 'student_id': 's-api'})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {'response', 'used_context', 'model', 'fallback'}
