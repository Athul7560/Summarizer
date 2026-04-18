# 🎓 AI Study Buddy (Streamlit + FastAPI)

A complete Streamlit-first student learning assistant with a FastAPI backend, Qwen model serving, RAG over uploaded notes/PDFs, student performance tracking, and an agentic recommendation loop.

## Implemented roadmap

- **Phase 1 — Model serving**
  - `POST /chat`
  - `POST /summarize`
  - Reusable model runtime with `BitsAndBytesConfig` 4-bit defaults and graceful fallback
- **Phase 2 — RAG + study features**
  - Upload notes/PDFs, extract text, chunking, embeddings, Chroma persistence
  - Retrieval injection for generation
  - Feature templates: summary, flashcards, quiz, mind map
- **Phase 3 — Student data + dashboard**
  - SQLite schema: `students`, `quiz_attempts`, `agent_feedback`
  - Topic scores, trend, recent attempts, weak topics dashboard panels
- **Phase 4 — Agentic loop**
  - Tool-style methods: `get_weak_topics`, `trigger_summary`, `generate_quiz`, `create_flashcards`
  - Recommendation + accept/reject feedback logging with decline-aware behavior

## Project structure

```text
.
├── app.py                      # Streamlit entry shim
├── backend/
│   ├── main.py                 # FastAPI app and endpoints
│   ├── model_service.py        # Qwen loading + fallback generation
│   ├── rag_service.py          # Chunking, embedding, Chroma retrieval
│   ├── database.py             # SQLite repository and aggregations
│   ├── agent.py                # Agentic recommendation orchestration
│   ├── schemas.py              # Request/response schemas
│   └── config.py               # Env configuration
├── streamlit_app/
│   └── app.py                  # Streamlit UI tabs
├── tests/
│   └── test_rag_and_db.py      # Lightweight focused tests
├── data/
│   ├── chroma/
│   └── sqlite/
├── requirements.txt
└── .env.example
```

## Configuration

Copy `.env.example` to `.env` and adjust if needed:

- `MODEL_ID`
- `EMBED_MODEL_ID`
- `CHROMA_PERSIST_DIR`
- `SQLITE_PATH`
- `HF_TOKEN` (optional)
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVE_TOP_K`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run backend (FastAPI)

```bash
uvicorn backend.main:app --reload
```

Open docs at `http://127.0.0.1:8000/docs`.

## Run frontend (Streamlit)

```bash
streamlit run streamlit_app/app.py
```

## Streamlit UX tabs

- **Chat**
- **Upload & Index**
- **Study Tools** (Summary / Flashcards / Quiz / Mind Map)
- **Dashboard**
- **AI Coach**

## API endpoints

- `GET /health`
- `POST /chat`
- `POST /summarize`
- `POST /ingest`
- `POST /retrieve`
- `POST /feature/{feature_name}` where feature is `summary|flashcards|quiz|mindmap`
- `POST /quiz/attempt`
- `GET /dashboard/{student_id}`
- `GET /coach/recommend/{student_id}`
- `POST /coach/feedback`
- `POST /coach/execute/{student_id}`

## Troubleshooting (4-bit / model runtime)

- If 4-bit quantized load fails (missing GPU/CUDA/bitsandbytes constraints), backend logs the error and attempts CPU load.
- If model loading still fails, app enters deterministic **fallback mode** so endpoints/UI remain functional for local development.
- `/health` reports `fallback: true/false`.

## Tests

Run lightweight focused tests:

```bash
python -m pytest -q tests/test_rag_and_db.py
```

## UI screenshot

- User-provided screenshot URL: https://github.com/user-attachments/assets/5d83ff25-60e8-48d0-8119-580c050b0599
