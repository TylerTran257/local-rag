"""Protocol definitions for the Retrieval Core."""
from datetime import datetime
from typing import Protocol

from app.retrieval.types import (
    EffectiveRetrieveRequest,
    RetrievalGatewayResult,
    RetrievalScope,
    ScopeDecision,
    RetrievalTrace,
)


class RetrievalGateway(Protocol):
    """Component that executes retrieval against a backend."""

    def retrieve(self, request: EffectiveRetrieveRequest) -> RetrievalGatewayResult:
        """Execute retrieval for the given effective request."""
        ...


class ScopePolicy(Protocol):
    """Component that validates a retrieval scope and produces a scope decision."""

    def evaluate(self, scope: RetrievalScope) -> ScopeDecision:
        """Evaluate scope and return a scope decision."""
        ...


class RetrievalTraceSink(Protocol):
    """Component that consumes retrieval traces."""

    def emit(self, trace: RetrievalTrace) -> None:
        """Emit a retrieval trace."""
        ...


class Clock(Protocol):
    """Component that provides current time."""

    def now(self) -> datetime:
        """Return current datetime."""
        ...


class TraceIdGenerator(Protocol):
    """Component that generates trace IDs."""

    def generate(self) -> str:
        """Generate a new trace ID."""
        ...
