"""
app/config.py
=============
Centralised settings management using Pydantic BaseSettings.
Reads from environment variables and .env file.

GCP Mapping: Equivalent to Secret Manager + Cloud Run environment
             variable injection in production.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings for the Vertex AI Agent Platform + AlloyDB stack.

    All values can be overridden via environment variables or a .env file.
    In GCP production, these are injected via Cloud Run environment
    variables or fetched from Secret Manager at startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Google Cloud / Vertex AI
    # ------------------------------------------------------------------
    google_cloud_project: str = Field(
        default="local-dev-project",
        description="GCP project ID. Set to your real project for Vertex AI calls.",
    )
    vertex_ai_location: str = Field(
        default="us-central1",
        description="Vertex AI / AlloyDB region.",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for the Agent Platform. Options: gemini-2.5-flash, gemini-1.5-flash",
    )
    embedding_model: str = Field(
        default="text-embedding-004",
        description="Vertex AI embedding model. Output: 768-dimension vectors.",
    )

    # ------------------------------------------------------------------
    # AlloyDB / PostgreSQL connection
    # ------------------------------------------------------------------
    alloydb_host: str = Field(default="localhost", description="AlloyDB instance hostname / IP.")
    alloydb_port: int = Field(default=5432, description="AlloyDB PostgreSQL port.")
    alloydb_user: str = Field(default="alloydb_admin", description="AlloyDB database user.")
    alloydb_password: str = Field(default="alloydb_local_secret", description="AlloyDB password.")
    alloydb_database: str = Field(default="vertex_rag_db", description="AlloyDB target database.")
    alloydb_pool_size: int = Field(default=10, description="Connection pool size.")
    alloydb_max_overflow: int = Field(default=20, description="Max pool overflow.")
    alloydb_pool_timeout: int = Field(default=30, description="Pool checkout timeout (seconds).")

    # ------------------------------------------------------------------
    # FastAPI / Agent Platform API
    # ------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    api_reload: bool = Field(default=True)
    log_level: Literal["debug", "info", "warning", "error"] = Field(default="info")

    # ------------------------------------------------------------------
    # RAG / Vector Search
    # ------------------------------------------------------------------
    vector_search_top_k: int = Field(
        default=5,
        description="Number of nearest neighbours to return from AlloyDB vector search.",
    )
    vector_similarity_threshold: float = Field(
        default=0.7,
        description="Minimum cosine similarity score to include in RAG results.",
    )

    @field_validator("gemini_model")
    @classmethod
    def validate_gemini_model(cls, v: str) -> str:
        allowed = {"gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash-exp"}
        if v not in allowed:
            raise ValueError(f"gemini_model must be one of {allowed}, got: {v!r}")
        return v

    @property
    def alloydb_dsn(self) -> str:
        """PostgreSQL DSN for psycopg2 (AlloyDB wire-protocol compatible)."""
        return (
            f"host={self.alloydb_host} "
            f"port={self.alloydb_port} "
            f"dbname={self.alloydb_database} "
            f"user={self.alloydb_user} "
            f"password={self.alloydb_password}"
        )

    @property
    def alloydb_url(self) -> str:
        """SQLAlchemy connection URL for AlloyDB."""
        return (
            f"postgresql+psycopg2://{self.alloydb_user}:{self.alloydb_password}"
            f"@{self.alloydb_host}:{self.alloydb_port}/{self.alloydb_database}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.
    Cached so GCP Secret Manager lookups only happen once per process.
    """
    return Settings()
