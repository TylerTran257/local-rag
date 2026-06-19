"""Maps embedding model names to their Qdrant collection.

Different embedding models produce different vector dimensions, which cannot
coexist in one Qdrant collection. Each model therefore gets its own collection.
The platform-default model maps to the existing default collection so that data
ingested before profiles existed (and the default-path tests) keep working.
"""
from __future__ import annotations

import re

from app.settings import settings

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(model_name: str) -> str:
    return _SLUG_RE.sub("_", model_name.lower()).strip("_")


def collection_for(model_name: str) -> str:
    """Return the Qdrant collection that holds vectors for ``model_name``."""
    if model_name == settings.embedding_model_name:
        return settings.qdrant_collection_name
    return f"{settings.qdrant_collection_name}__{_slug(model_name)}"
