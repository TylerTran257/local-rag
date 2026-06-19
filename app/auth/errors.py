"""Authentication and authorization errors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AuthError(Exception):
    """Base class for auth errors carrying a stable code and details."""

    code: str
    message: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class AuthenticationError(AuthError):
    """Raised when a request presents no key or an unknown key (HTTP 401)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="AUTHENTICATION_FAILED", message=message, details=details or {}
        )


class AuthorizationError(AuthError):
    """Raised when an authenticated key requests an out-of-scope action (HTTP 403)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="AUTHORIZATION_FAILED", message=message, details=details or {}
        )
