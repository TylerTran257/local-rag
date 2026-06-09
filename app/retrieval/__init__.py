"""Retrieval Core - framework-independent retrieval architecture."""

# Enums
from app.retrieval.types import (
    RetrievalMode,
    WarningCode,
    WarningSeverity,
    TraceStatus,
    FailureStage,
)

# Value objects
from app.retrieval.types import (
    RetrievalScope,
    RetrieveRequest,
    EffectiveRetrieveRequest,
    RetrievedChunk,
    RetrievalGatewayResult,
    ScopeDecision,
    RetrievalWarning,
    RetrievalTrace,
)

# Errors
from app.retrieval.errors import (
    RetrievalError,
    InvalidRetrievalRequestError,
    UnsupportedRetrievalModeError,
    NoIndexedCorpusError,
    RetrievalExecutionError,
    RetrievedChunkValidationError,
)

# Protocols
from app.retrieval.contracts import (
    RetrievalGateway,
    ScopePolicy,
    RetrievalTraceSink,
    Clock,
    TraceIdGenerator,
)

# Implementations
from app.retrieval.trace import (
    InMemoryTraceSink,
    NoOpTraceSink,
    StructuredLoggingTraceSink,
    SystemClock,
    UuidTraceIdGenerator,
)

from app.retrieval.policy import NamespacePolicy, PassthroughScopePolicy


__all__ = [
    # Enums
    "RetrievalMode",
    "WarningCode",
    "WarningSeverity",
    "TraceStatus",
    "FailureStage",
    # Value objects
    "RetrievalScope",
    "RetrieveRequest",
    "EffectiveRetrieveRequest",
    "RetrievedChunk",
    "RetrievalGatewayResult",
    "ScopeDecision",
    "RetrievalWarning",
    "RetrievalTrace",
    # Errors
    "RetrievalError",
    "InvalidRetrievalRequestError",
    "UnsupportedRetrievalModeError",
    "NoIndexedCorpusError",
    "RetrievalExecutionError",
    "RetrievedChunkValidationError",
    # Protocols
    "RetrievalGateway",
    "ScopePolicy",
    "RetrievalTraceSink",
    "Clock",
    "TraceIdGenerator",
    # Implementations
    "InMemoryTraceSink",
    "NoOpTraceSink",
    "StructuredLoggingTraceSink",
    "SystemClock",
    "UuidTraceIdGenerator",
    "NamespacePolicy",
    "PassthroughScopePolicy",
]
