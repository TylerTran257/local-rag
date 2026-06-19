"""Per-service configuration profiles."""
from app.profiles.model_registry import collection_for
from app.profiles.models import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ServiceProfile,
    default_profile,
)
from app.profiles.store import ProfileEmbeddingModelImmutableError, ProfileStore

__all__ = [
    "ServiceProfile",
    "default_profile",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "ProfileStore",
    "ProfileEmbeddingModelImmutableError",
    "collection_for",
]
