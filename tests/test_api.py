from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)
HEALTH_ENDPOINT = "/health"
SUMMARIZE_ENDPOINT = "/summarize"
CHAT_ENDPOINT = "/chat"


def test_health_schema() -> None:
    response = client.get(HEALTH_ENDPOINT)
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {'status', 'model', 'fallback'}


def test_summarize_validation_error() -> None:
    response = client.post(SUMMARIZE_ENDPOINT, json={'text': '', 'context_chunks': []})
    assert response.status_code == 400


def test_summarize_success_schema() -> None:
    response = client.post(SUMMARIZE_ENDPOINT, json={'text': 'Cell respiration converts glucose to ATP.'})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {'summary', 'model', 'fallback'}


def test_chat_schema() -> None:
    response = client.post(CHAT_ENDPOINT, json={'message': 'Hello', 'student_id': 's-api'})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {'response', 'used_context', 'model', 'fallback'}
