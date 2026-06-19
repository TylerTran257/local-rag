"""Per-service configuration profiles.

A ``ServiceProfile`` is the ingestion + retrieval configuration a consuming
service controls for its own ``service_name``: chunking, embedding model, and
retrieval defaults. When a service has no registered profile, a default profile
mirroring the platform defaults is used, so behavior is unchanged for callers
that never configure anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retrieval.types import RetrievalMode
from app.settings import settings

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class ServiceProfile:
    """Ingestion + retrieval configuration for a single service."""

    service_name: str
    embedding_model: str = settings.embedding_model_name
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    dense_limit: int = settings.dense_retrieval_limit
    lexical_limit: int = settings.lexical_retrieval_limit
    fusion_rrf_k: int = settings.fusion_rrf_k
    default_mode: RetrievalMode = RetrievalMode.HYBRID
    generation_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "dense_limit": self.dense_limit,
            "lexical_limit": self.lexical_limit,
            "fusion_rrf_k": self.fusion_rrf_k,
            "default_mode": self.default_mode.value,
            "generation_overrides": dict(self.generation_overrides),
        }


def default_profile(service_name: str) -> ServiceProfile:
    """Return the platform-default profile for a service.

    Reads live ``settings`` so environment overrides remain the source of truth
    when a service has not registered an explicit profile.
    """
    return ServiceProfile(
        service_name=service_name,
        embedding_model=settings.embedding_model_name,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        dense_limit=settings.dense_retrieval_limit,
        lexical_limit=settings.lexical_retrieval_limit,
        fusion_rrf_k=settings.fusion_rrf_k,
        default_mode=RetrievalMode.HYBRID,
        generation_overrides={},
    )
