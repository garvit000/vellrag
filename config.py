"""
Configuration Module for Voice-Enabled Low-Latency RAG Pipeline
(HH Goa 2026 Task 2)
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load env variables from .env into os.environ
load_dotenv()


class SystemSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Keys & Endpoints
    GROQ_API_KEY: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", "mock_groq_key"))
    GROQ_MODEL: str = Field(default="llama-3.1-8b-instant")
    SARVAM_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("SARVAM_API_KEY"))
    ELEVENLABS_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY"))

    # Vector Store & Embedding Settings
    QDRANT_HOST: str = Field(default=":memory:")
    QDRANT_COLLECTION: str = Field(default="msmarco_xl_chunks")
    EMBEDDING_MODEL_NAME: str = Field(default="BAAI/bge-small-en-v1.5")
    EMBEDDING_DIMENSION: int = Field(default=384)

    # Multi-Strategy Chunking Settings
    CHILD_CHUNK_SIZE: int = Field(default=128)  # tokens for vector retrieval
    PARENT_CHUNK_SIZE: int = Field(default=512)  # tokens for LLM context injection
    SLIDING_WINDOW_SIZE: int = Field(default=256)
    SLIDING_WINDOW_OVERLAP: int = Field(default=64)
    SEMANTIC_SIMILARITY_THRESHOLD: float = Field(default=0.65)

    # RAG & Latency Constraints
    GROUNDING_SIMILARITY_THRESHOLD: float = Field(default=0.60)
    TARGET_PIPELINE_LATENCY_MS: float = Field(default=200.0)
    MAX_RETRIES: int = Field(default=3)
    BACKOFF_FACTOR: float = Field(default=0.2)

    # FastAPI Server
    HOST: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = Field(default_factory=lambda: int(os.getenv("PORT", 8000)))


settings = SystemSettings()
