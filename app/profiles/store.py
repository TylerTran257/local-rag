"""Persistence and resolution for per-service config profiles."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.db.database import SessionLocal
from app.db.models import ServiceProfileRecord
from app.profiles.models import ServiceProfile, default_profile
from app.retrieval.types import RetrievalMode

logger = logging.getLogger(__name__)


class ProfileEmbeddingModelImmutableError(ValueError):
    """Raised when an update tries to change a profile's embedding model.

    The embedding model determines the vector dimension and collection a
    service's documents are stored in; changing it would orphan already
    ingested vectors. It is therefore fixed once a profile is created.
    """

    def __init__(self, service_name: str, current: str, requested: str):
        self.service_name = service_name
        self.current = current
        self.requested = requested
        super().__init__(
            f"Embedding model for service '{service_name}' is immutable "
            f"(current='{current}', requested='{requested}'). Re-ingest under a "
            "new service to use a different embedding model."
        )


class ProfileStore:
    """Stores and resolves ``ServiceProfile`` records.

    ``get`` always returns a profile -- the platform default when none has been
    registered -- so call sites never special-case "no profile".
    """

    def __init__(self, session_factory=SessionLocal) -> None:
        self._session_factory = session_factory

    def seed_from_file(self, path: Path | None) -> None:
        """Seed profiles from a JSON file (list of profile dicts), if present."""
        if path is None or not Path(path).exists():
            return
        data = json.loads(Path(path).read_text())
        for entry in data:
            profile = _profile_from_dict(entry)
            try:
                self.upsert(profile)
            except ProfileEmbeddingModelImmutableError:
                # Seeding is idempotent; an existing identical profile is fine,
                # a conflicting embedding model is a config error worth surfacing.
                logger.warning(
                    "event=profile_seed_conflict service_name=%s", profile.service_name
                )

    def exists(self, service_name: str) -> bool:
        with self._session_factory() as session:
            return session.get(ServiceProfileRecord, service_name) is not None

    def get(self, service_name: str) -> ServiceProfile:
        with self._session_factory() as session:
            record = session.get(ServiceProfileRecord, service_name)
            if record is None:
                return default_profile(service_name)
            return _record_to_profile(record)

    def upsert(self, profile: ServiceProfile) -> ServiceProfile:
        now = datetime.now(timezone.utc)
        with self._session_factory() as session:
            record = session.get(ServiceProfileRecord, profile.service_name)
            if record is None:
                record = ServiceProfileRecord(
                    service_name=profile.service_name,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
            else:
                if record.embedding_model != profile.embedding_model:
                    raise ProfileEmbeddingModelImmutableError(
                        service_name=profile.service_name,
                        current=record.embedding_model,
                        requested=profile.embedding_model,
                    )
                record.updated_at = now

            record.embedding_model = profile.embedding_model
            record.chunk_size = profile.chunk_size
            record.chunk_overlap = profile.chunk_overlap
            record.dense_limit = profile.dense_limit
            record.lexical_limit = profile.lexical_limit
            record.fusion_rrf_k = profile.fusion_rrf_k
            record.default_mode = profile.default_mode.value
            record.generation_overrides = json.dumps(profile.generation_overrides)

            session.commit()
            session.refresh(record)
            logger.info(
                "event=profile_upserted service_name=%s embedding_model=%s",
                profile.service_name,
                profile.embedding_model,
            )
            return _record_to_profile(record)


def _record_to_profile(record: ServiceProfileRecord) -> ServiceProfile:
    return ServiceProfile(
        service_name=record.service_name,
        embedding_model=record.embedding_model,
        chunk_size=record.chunk_size,
        chunk_overlap=record.chunk_overlap,
        dense_limit=record.dense_limit,
        lexical_limit=record.lexical_limit,
        fusion_rrf_k=record.fusion_rrf_k,
        default_mode=RetrievalMode(record.default_mode),
        generation_overrides=json.loads(record.generation_overrides or "{}"),
    )


def _profile_from_dict(data: dict[str, Any]) -> ServiceProfile:
    base = default_profile(data["service_name"])
    mode = data.get("default_mode")
    return ServiceProfile(
        service_name=data["service_name"],
        embedding_model=data.get("embedding_model", base.embedding_model),
        chunk_size=int(data.get("chunk_size", base.chunk_size)),
        chunk_overlap=int(data.get("chunk_overlap", base.chunk_overlap)),
        dense_limit=int(data.get("dense_limit", base.dense_limit)),
        lexical_limit=int(data.get("lexical_limit", base.lexical_limit)),
        fusion_rrf_k=int(data.get("fusion_rrf_k", base.fusion_rrf_k)),
        default_mode=RetrievalMode(mode) if mode else base.default_mode,
        generation_overrides=data.get("generation_overrides", {}),
    )
