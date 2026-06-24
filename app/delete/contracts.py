from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeleteRequest:
    service_name: str
    tenant_id: str
    collections: list[str]
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeleteResult:
    deleted_count: int
