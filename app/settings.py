from pathlib import Path

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

    qdrant_path: Path = Path("./qdrant_data")
    qdrant_collection_name: str = "document_chunks"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

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
