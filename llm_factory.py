"""
llm_factory.py — Provider-agnostic LLM & Embedding factory.

Supported providers (set LLM_PROVIDER in .env)
───────────────────────────────────────────────
  openai  — ChatOpenAI + OpenAIEmbeddings
            Requires: OPENAI_API_KEY
            Models  : gpt-4o-mini (default), gpt-4o, gpt-3.5-turbo, …

  groq    — ChatGroq + local SentenceTransformers embeddings
            Requires: GROQ_API_KEY  (free tier at console.groq.com)
            Chat    : llama-3.1-8b-instant (default — very fast & free)
                      llama-3.3-70b-versatile, mixtral-8x7b-32768,
                      gemma2-9b-it, llama-3.1-70b-versatile
            Embeds  : sentence-transformers/all-MiniLM-L6-v2 (local, no key)
            Note    : Groq does not provide an embedding API, so embeddings
                      are computed locally via sentence-transformers. This is
                      100 % offline and requires no GPU for inference-only use.

  ollama  — ChatOllama + OllamaEmbeddings
            Requires: Ollama server running at OLLAMA_BASE_URL
            Models  : llama3.1 (default), mistral, phi3, …
            Embeds  : nomic-embed-text (default)

No other file needs to change when switching providers.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from loguru import logger

from config import Settings, get_settings


# ─────────────────────────────────────────────────────────────────────────────
# Chat LLM
# ─────────────────────────────────────────────────────────────────────────────


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """
    Return the appropriate LangChain chat model for the configured provider.

    Parameters
    ----------
    settings : Optional Settings override (useful for testing).

    Returns
    -------
    BaseChatModel  ready to call .invoke() on.
    """
    cfg = settings or get_settings()
    provider = cfg.llm_provider

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env. "
                "Add it or switch LLM_PROVIDER to 'groq' or 'ollama'."
            )
        logger.info(f"[LLM] OpenAI → {cfg.openai_model}")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.openai_model,
            api_key=cfg.openai_api_key,
            temperature=0.3,
        )

    # ── Groq ──────────────────────────────────────────────────────────────────
    if provider == "groq":
        if not cfg.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in .env. "
                "Get a free key at https://console.groq.com/keys"
            )
        logger.info(f"[LLM] Groq → {cfg.groq_model}")
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=cfg.groq_model,
                api_key=cfg.groq_api_key,
                temperature=0.3,
                # Groq enforces its own token limits — keep reasonable
                max_tokens=4096,
            )
        except ImportError as exc:
            raise ImportError(
                "langchain-groq is required for Groq support. "
                "Run: pip install langchain-groq"
            ) from exc

    # ── Ollama (local GPU) ────────────────────────────────────────────────────
    if provider == "ollama":
        logger.info(f"[LLM] Ollama → {cfg.ollama_model}  @ {cfg.ollama_base_url}")
        try:
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                model=cfg.ollama_model,
                base_url=cfg.ollama_base_url,
                temperature=0.3,
            )
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required for Ollama support. "
                "Run: pip install langchain-community"
            ) from exc

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. "
        "Valid options: 'openai', 'groq', 'ollama'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """
    Return the appropriate LangChain Embeddings object.

    Provider → Embedding backend mapping
    ─────────────────────────────────────
    openai  → OpenAIEmbeddings        (cloud, requires API key)
    groq    → SentenceTransformerEmbeddings  (local, no key needed)
    ollama  → OllamaEmbeddings        (local Ollama server)

    Important: whatever backend is used at *ingest* time must also be used
    at *query* time, because vectors must live in the same embedding space.
    """
    cfg = settings or get_settings()
    provider = cfg.llm_provider

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if provider == "openai":
        logger.info(f"[Embed] OpenAI → {cfg.openai_embedding_model}")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=cfg.openai_embedding_model,
            api_key=cfg.openai_api_key,
        )

    # ── Groq → local SentenceTransformers ─────────────────────────────────────
    if provider == "groq":
        logger.info(
            f"[Embed] Groq provider — using local SentenceTransformers "
            f"({cfg.groq_embedding_model}) for embeddings."
        )
        return _get_sentence_transformer_embeddings(cfg.groq_embedding_model)

    # ── Ollama ────────────────────────────────────────────────────────────────
    if provider == "ollama":
        logger.info(f"[Embed] Ollama → {cfg.ollama_embedding_model}")
        try:
            from langchain_community.embeddings import OllamaEmbeddings

            return OllamaEmbeddings(
                model=cfg.ollama_embedding_model,
                base_url=cfg.ollama_base_url,
            )
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required for Ollama embedding support. "
                "Run: pip install langchain-community"
            ) from exc

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. "
        "Valid options: 'openai', 'groq', 'ollama'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_sentence_transformer_embeddings(model_name: str) -> Embeddings:
    """
    Wrap a sentence-transformers model as a LangChain Embeddings object.
    Supports both langchain_huggingface and langchain_community.
    """
    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for local embeddings (used with Groq provider). "
            "Run: pip install sentence-transformers"
        ) from exc


def provider_info(settings: Settings | None = None) -> dict[str, str]:
    """
    Return a human-readable summary of the active provider configuration.
    Used by the Streamlit sidebar status panel.
    """
    cfg = settings or get_settings()
    provider = cfg.llm_provider

    if provider == "openai":
        return {
            "provider": "OpenAI",
            "chat_model": cfg.openai_model,
            "embed_model": cfg.openai_embedding_model,
            "embed_backend": "OpenAI API",
            "key_set": bool(cfg.openai_api_key),
        }
    if provider == "groq":
        return {
            "provider": "Groq",
            "chat_model": cfg.groq_model,
            "embed_model": cfg.groq_embedding_model,
            "embed_backend": "Local SentenceTransformers",
            "key_set": bool(cfg.groq_api_key),
        }
    if provider == "ollama":
        return {
            "provider": "Ollama (Local)",
            "chat_model": cfg.ollama_model,
            "embed_model": cfg.ollama_embedding_model,
            "embed_backend": "Ollama API",
            "key_set": True,  # no key needed
        }
    return {"provider": provider, "chat_model": "?", "embed_model": "?", "key_set": False}
