"""FastAPI auth dependencies and scope enforcement helpers."""
from __future__ import annotations

from collections.abc import Iterable

from fastapi import Request

from app.auth.errors import AuthenticationError, AuthorizationError
from app.auth.principal import Principal
from app.auth.registry import ApiKeyRegistry

API_KEY_HEADER = "X-API-Key"
BEARER_PREFIX = "Bearer "


def _extract_key(request: Request) -> str | None:
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        return api_key
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith(BEARER_PREFIX):
        return authorization[len(BEARER_PREFIX):].strip()
    return None


def require_principal(request: Request) -> Principal:
    """Resolve the caller's :class:`Principal` or raise ``AuthenticationError``.

    Used as a FastAPI dependency. The registry lives on ``app.state`` so it is
    shared and swappable in tests.
    """
    registry: ApiKeyRegistry | None = getattr(
        request.app.state, "api_key_registry", None
    )
    if registry is None:
        raise AuthenticationError("API key registry is not configured")

    principal = registry.resolve(_extract_key(request))
    if principal is None:
        raise AuthenticationError("Missing or invalid API key")
    return principal


def enforce_scope(
    principal: Principal,
    *,
    service_name: str,
    tenant_id: str,
    collections: Iterable[str] = (),
) -> None:
    """Raise ``AuthorizationError`` if the principal's grant does not cover the scope."""
    if not principal.allows_service(service_name):
        raise AuthorizationError(
            f"Key '{principal.key_id}' may not access service '{service_name}'",
            details={"service_name": service_name},
        )
    if not principal.allows_tenant(tenant_id):
        raise AuthorizationError(
            f"Key '{principal.key_id}' may not access tenant '{tenant_id}'",
            details={"tenant_id": tenant_id},
        )
    denied = [c for c in collections if not principal.allows_collection(c)]
    if denied:
        raise AuthorizationError(
            f"Key '{principal.key_id}' may not access collections {denied}",
            details={"denied_collections": denied},
        )
