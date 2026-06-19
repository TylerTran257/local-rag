from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.settings import settings


class EmbeddingService:
    """Multi-model embedding service.

    Loads sentence-transformer models lazily and caches them by name so a
    shared deployment can serve multiple per-service embedding models from a
    single instance. When no model name is given, the configured default
    (``settings.embedding_model_name``) is used, preserving single-model
    behavior for existing callers.
    """

    def __init__(self, default_model_name: str | None = None) -> None:
        self.default_model_name = default_model_name or settings.embedding_model_name
        self._models: dict[str, SentenceTransformer] = {}
        # Eagerly load the default model so first-request latency and any
        # load-time failures surface at startup, matching prior behavior.
        self._get_model(self.default_model_name)

    def _get_model(self, model_name: str | None) -> SentenceTransformer:
        name = model_name or self.default_model_name
        model = self._models.get(name)
        if model is None:
            model = SentenceTransformer(name)
            self._models[name] = model
        return model

    @property
    def model(self) -> SentenceTransformer:
        """The default model (kept for backward compatibility)."""
        return self._get_model(self.default_model_name)

    def embed_text(self, text: str, model_name: str | None = None) -> list[float]:
        vector = self._get_model(model_name).encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_texts(
        self, texts: list[str], model_name: str | None = None
    ) -> list[list[float]]:
        vectors = self._get_model(model_name).encode(texts, normalize_embeddings=True)
        return vectors.tolist()
