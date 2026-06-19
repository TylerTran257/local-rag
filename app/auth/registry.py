"""API key registry mapping keys to scope grants."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from app.auth.principal import Principal

logger = logging.getLogger(__name__)

ENV_VAR = "LOCAL_RAG_API_KEYS"


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _principal_from_entry(entry: dict[str, Any]) -> Principal:
    return Principal(
        key_id=entry.get("key_id", "unknown"),
        allowed_services=frozenset(entry.get("services", [])),
        allowed_tenants=frozenset(entry.get("tenants", [])),
        allowed_collections=frozenset(entry.get("collections", [])),
        is_admin=bool(entry.get("admin", False)),
    )


class ApiKeyRegistry:
    """Resolves raw API keys to :class:`Principal` grants.

    Keys are stored hashed; the raw secret is never retained. An empty registry
    rejects every request (fail closed), so deployments must provision keys.
    """

    def __init__(self, principals_by_hash: dict[str, Principal] | None = None) -> None:
        self._by_hash: dict[str, Principal] = principals_by_hash or {}

    @classmethod
    def from_config(cls, *, api_keys_file: Path | None) -> "ApiKeyRegistry":
        """Build a registry from a JSON file, falling back to the env var."""
        raw = None
        if api_keys_file is not None and Path(api_keys_file).exists():
            raw = Path(api_keys_file).read_text()
        elif os.environ.get(ENV_VAR):
            raw = os.environ[ENV_VAR]

        if not raw:
            logger.warning("event=api_key_registry_empty source=none")
            return cls({})

        data = json.loads(raw)
        entries = data.get("keys", data) if isinstance(data, dict) else data
        by_hash: dict[str, Principal] = {}
        for entry in entries:
            by_hash[_hash_key(entry["key"])] = _principal_from_entry(entry)
        logger.info("event=api_key_registry_loaded key_count=%s", len(by_hash))
        return cls(by_hash)

    @classmethod
    def from_entries(cls, entries: list[dict[str, Any]]) -> "ApiKeyRegistry":
        """Build a registry from in-memory entries (tests / programmatic use)."""
        return cls({_hash_key(e["key"]): _principal_from_entry(e) for e in entries})

    def resolve(self, raw_key: str | None) -> Principal | None:
        if not raw_key:
            return None
        return self._by_hash.get(_hash_key(raw_key))

    def __len__(self) -> int:
        return len(self._by_hash)
