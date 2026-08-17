"""
Central configuration module using pydantic-settings.
All settings are loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings loaded from .env"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    openai_api_key: str = Field(default="", description="OpenAI API key (moderation)")

    # ── Storage Paths ─────────────────────────────────────────────────────
    chroma_db_path: str = Field(default="./chroma_db", description="ChromaDB persistence path")
    data_cache_path: str = Field(default="./data/cache", description="Dataset cache path")
    results_path: str = Field(default="./results", description="Results output path")

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k_retrieval: int = Field(default=5, description="Number of chunks to retrieve")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_model: str = Field(default="openai/gpt-oss-20b", description="Groq model name")
    llm_max_tokens: int = Field(default=512, description="Max tokens for LLM response")
    llm_temperature: float = Field(default=0.1, description="LLM temperature (low = factual)")

    # ── Guardrails ────────────────────────────────────────────────────────
    off_topic_threshold: float = Field(
        default=0.25,
        description="Min cosine similarity to domain centroid for on-topic detection",
    )
    faithfulness_threshold: float = Field(
        default=0.5,
        description="Min faithfulness score for answer grounding check",
    )

    # ── Chunking ──────────────────────────────────────────────────────────
    default_chunk_size: int = Field(default=512, description="Default chunk size in characters")
    default_chunk_overlap: int = Field(default=64, description="Default chunk overlap in characters")


# Singleton settings instance
settings = Settings()
