from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///./app.db"
    upload_dir: Path = Path("uploads")
    max_file_size: int = 2 * 1024 * 1024

    # Vector backend. When ``qdrant_url`` is set the service connects to a
    # remote Qdrant (shared deployment); otherwise it uses an on-disk store at
    # ``qdrant_path`` (local default).
    qdrant_path: Path = Path("./qdrant_data")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "document_chunks"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Shared-service controls.
    # ``scope_policy_mode`` selects the retrieval-core scope policy: "namespace"
    # enforces allowed services/collections (fail-closed); "passthrough" only
    # validates structure.
    scope_policy_mode: Literal["passthrough", "namespace"] = "passthrough"
    # Optional JSON file of API keys -> scope grants. When unset, auth falls
    # back to ``LOCAL_RAG_API_KEYS`` env JSON, or an empty registry.
    api_keys_file: Path | None = None
    # Optional JSON file seeding per-service config profiles.
    profiles_file: Path | None = None
    # Expose a Prometheus ``/metrics`` endpoint and per-call metric logs.
    metrics_enabled: bool = True

    generation_base_url: str = "http://127.0.0.1:8080/v1"
    generation_endpoint: str = "/chat/completions"
    generation_timeout: float = 600.0
    generation_temperature: float = 0.2
    generation_max_output_tokens: int = 300
    generation_max_context_chars: int = 6000
    generation_max_chars_per_chunk: int = 1800

    dense_retrieval_limit: int = 15
    lexical_retrieval_limit: int = 15
    fusion_rrf_k: int = 60


settings = Settings()
