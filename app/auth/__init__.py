"""API-key authentication and scope authorization."""
from app.auth.dependencies import enforce_scope, require_principal
from app.auth.errors import AuthenticationError, AuthError, AuthorizationError
from app.auth.principal import WILDCARD, Principal
from app.auth.registry import ApiKeyRegistry

__all__ = [
    "Principal",
    "WILDCARD",
    "ApiKeyRegistry",
    "AuthError",
    "AuthenticationError",
    "AuthorizationError",
    "require_principal",
    "enforce_scope",
]
