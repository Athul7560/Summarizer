from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


class RAGService:
    def __init__(self) -> None:
        self._embedder = None
        self._chroma_collection = None
        self._fallback_records: list[ChunkRecord] = []
        self._ready = False

    def _init_clients(self) -> None:
        if self._ready:
            return
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
            self._chroma_collection = client.get_or_create_collection("study_materials")
            self._embedder = SentenceTransformer(settings.embed_model_id)
            logger.info("RAG initialized with ChromaDB + SentenceTransformer")
        except Exception as exc:
            logger.warning("Using fallback in-memory retrieval: %s", exc)
            self._chroma_collection = None
            self._embedder = None
        self._ready = True

    @staticmethod
    def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
        size = chunk_size or settings.chunk_size
        ov = overlap or settings.chunk_overlap
        if size <= 0:
            raise ValueError("chunk_size must be > 0")
        if ov < 0:
            raise ValueError("overlap must be >= 0")
        if ov >= size:
            raise ValueError("overlap must be smaller than chunk_size")

        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + size)
            chunks.append(cleaned[start:end])
            if end == len(cleaned):
                break
            start = end - ov
        return chunks

    @staticmethod
    def _extract_pdf_text(file_bytes: bytes) -> str:
        from io import BytesIO

        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    def extract_text(self, filename: str, file_bytes: bytes) -> str:
        if filename.lower().endswith(".pdf"):
            return self._extract_pdf_text(file_bytes)
        return file_bytes.decode("utf-8", errors="ignore")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._init_clients()
        if self._embedder is None:
            # Deterministic fallback vectors for environments without embedding models.
            # These hash-derived vectors keep behavior reproducible but are non-semantic.
            # Use this mode for development/testing only; semantic retrieval needs real embeddings.
            vectors: list[list[float]] = []
            for text in texts:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                vectors.append([float(b) / 255.0 for b in digest[:32]])
            return vectors
        return self._embedder.encode(texts).tolist()

    def index_documents(self, *, student_id: str, documents: list[tuple[str, bytes]]) -> tuple[int, int]:
        self._init_clients()
        total_chunks = 0
        docs_processed = 0

        for filename, payload in documents:
            extracted = self.extract_text(filename, payload)
            chunks = self.chunk_text(extracted)
            if not chunks:
                continue

            embeddings = self._embed(chunks)
            metadatas = [{"student_id": student_id, "source": filename} for _ in chunks]
            ids = [str(uuid.uuid4()) for _ in chunks]

            if self._chroma_collection is not None:
                self._chroma_collection.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            else:
                for chunk, metadata, embedding in zip(chunks, metadatas, embeddings):
                    self._fallback_records.append(ChunkRecord(text=chunk, metadata=metadata, embedding=embedding))

            total_chunks += len(chunks)
            docs_processed += 1

        return total_chunks, docs_processed

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, *, query: str, student_id: str, top_k: int | None = None) -> list[str]:
        self._init_clients()
        k = top_k or settings.retrieve_top_k

        if self._chroma_collection is not None and self._embedder is not None:
            q_embedding = self._embed([query])[0]
            result = self._chroma_collection.query(
                query_embeddings=[q_embedding],
                n_results=k,
                where={"student_id": student_id},
            )
            docs = result.get("documents", [[]])
            return docs[0] if docs else []

        query_vec = self._embed([query])[0]
        scored: list[tuple[float, str]] = []
        for record in self._fallback_records:
            if record.metadata.get("student_id") != student_id:
                continue
            score = self._cosine(query_vec, record.embedding)
            scored.append((score, record.text))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:k]]


rag_service = RAGService()
