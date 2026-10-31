"""
config.py — Centralised, environment-driven configuration.

All tuneable parameters live here so that nothing is hard-coded in the
application modules.  Values are read from a .env file (or real environment
variables) via pydantic-settings.

Supported LLM providers
───────────────────────
  openai  — OpenAI cloud API (GPT-4o-mini default)
  groq    — Groq cloud API  (llama-3.1-8b-instant default, free tier available)
  ollama  — Local Ollama server (GPU execution, no API key needed)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from the .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM provider ──────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "groq", "ollama"] = Field(
        default="groq",
        description="Which LLM backend to use: 'openai', 'groq', or 'ollama'.",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI chat model.")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model.",
    )

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key (console.groq.com).")
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        description=(
            "Groq chat model. Fast options: llama-3.1-8b-instant, "
            "llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it."
        ),
    )
    # Groq has no native embedding API — embeddings are always handled by a
    # local sentence-transformers model when groq is the chat provider.
    groq_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "sentence-transformers model used for embeddings when LLM_PROVIDER=groq. "
            "Runs fully locally — no API key required."
        ),
    )

    # ── Ollama (local GPU) ────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL.",
    )
    ollama_model: str = Field(default="llama3.1", description="Ollama chat model.")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model.",
    )

    # ── Qdrant ────────────────────────────────────────────────────────────────
    # Local Docker fallback configuration. Cloud settings below are active.
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_url: str = Field(default="")
    qdrant_api_key: str = Field(default="")
    qdrant_timeout: int = Field(default=120)
    qdrant_collection: str = Field(default="roast_resume")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    top_k: int = Field(default=5, description="Number of chunks to retrieve per query.")

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # ── Data paths ────────────────────────────────────────────────────────────
    data_dir: str = Field(default="data", description="Folder containing source PDFs.")

    # Document filenames (relative to data_dir)
    ats_pdf: str = Field(default="ApplicantTrackingSystem-Guide_2.pdf")
    leland_pdf: str = Field(
        default="Leland\u2014The-Ultimate-Career-Resume-Guide+Examples_removed_2.pdf"
    )
    mit_pdf: str = Field(default="Resume-Book_removed.pdf")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings object (cached after first call)."""
    return Settings()


def get_qdrant_client(
    settings: Settings | None = None,
    *,
    timeout: float | None = None,
):
    """Create the configured Qdrant Cloud client."""
    from qdrant_client import QdrantClient

    cfg = settings or get_settings()
    if not cfg.qdrant_url:
        raise ValueError(
            "QDRANT_URL is not set in .env. "
            "Configure the Qdrant Cloud URL before starting the application."
        )
    if not cfg.qdrant_api_key:
        raise ValueError(
            "QDRANT_API_KEY is not set in .env. "
            "Configure the Qdrant Cloud API key before starting the application."
        )

    client_options = {
        "url": cfg.qdrant_url,
        "api_key": cfg.qdrant_api_key,
    }
    client_options["timeout"] = timeout if timeout is not None else cfg.qdrant_timeout
    return QdrantClient(**client_options)
