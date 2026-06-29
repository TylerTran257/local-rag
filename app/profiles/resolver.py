"""Resolve a service_name to its profile and Qdrant collection.

This is the single place that turns a ``service_name`` into the
``(ServiceProfile, collection)`` pair used by ingest, retrieve, and delete.
Resolving in one place guarantees the cross-path invariant: a document is read
back (and deleted) with the same embedding model + collection it was written
with.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.profiles.model_registry import collection_for
from app.profiles.models import ServiceProfile, default_profile
from app.profiles.store import ProfileStore


@dataclass(frozen=True)
class ResolvedProfile:
    """A service's resolved profile together with its Qdrant collection."""

    profile: ServiceProfile
    collection: str


class ProfileResolver:
    """Resolves a service_name to its ``ResolvedProfile``.

    Owns the store-or-default fallback: when no profile store is wired, the
    platform-default profile is used, preserving default-path behavior.
    """

    def __init__(self, profile_store: ProfileStore | None = None) -> None:
        self._profile_store = profile_store

    def resolve(self, service_name: str) -> ResolvedProfile:
        if self._profile_store is None:
            profile = default_profile(service_name)
        else:
            profile = self._profile_store.get(service_name)
        return ResolvedProfile(
            profile=profile, collection=collection_for(profile.embedding_model)
        )
