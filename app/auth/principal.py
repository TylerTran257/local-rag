"""Authenticated caller identity and its scope grant."""
from __future__ import annotations

from dataclasses import dataclass, field

# Sentinel meaning "any value is allowed for this dimension".
WILDCARD = "*"


@dataclass(frozen=True)
class Principal:
    """An authenticated caller and the scope it is allowed to act within.

    A grant dimension containing :data:`WILDCARD` allows any value for that
    dimension. ``is_admin`` callers bypass all scope checks (used for
    provisioning and profile management across services).
    """

    key_id: str
    allowed_services: frozenset[str] = field(default_factory=frozenset)
    allowed_tenants: frozenset[str] = field(default_factory=frozenset)
    allowed_collections: frozenset[str] = field(default_factory=frozenset)
    is_admin: bool = False

    def _allows(self, allowed: frozenset[str], value: str) -> bool:
        return self.is_admin or WILDCARD in allowed or value in allowed

    def allows_service(self, service_name: str) -> bool:
        return self._allows(self.allowed_services, service_name)

    def allows_tenant(self, tenant_id: str) -> bool:
        return self._allows(self.allowed_tenants, tenant_id)

    def allows_collection(self, collection: str) -> bool:
        return self._allows(self.allowed_collections, collection)
