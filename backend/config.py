from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    embed_model_id: str = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    sqlite_path: str = os.getenv("SQLITE_PATH", "data/sqlite/student_data.db")
    hf_token: str | None = os.getenv("HF_TOKEN")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retrieve_top_k: int = int(os.getenv("RETRIEVE_TOP_K", "4"))

    def ensure_dirs(self) -> None:
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Storage directories ensured")


settings = Settings()
settings.ensure_dirs()
