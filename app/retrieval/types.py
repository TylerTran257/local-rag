"""Core value objects and enums for the Retrieval Core."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RetrievalMode(Enum):
    """Supported retrieval modes."""
    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class WarningCode(Enum):
    """Warning codes for retrieval warnings."""
    LEGACY_METADATA_DEFAULTED = "LEGACY_METADATA_DEFAULTED"
    EMPTY_RETRIEVAL_RESULT = "EMPTY_RETRIEVAL_RESULT"


class WarningSeverity(Enum):
    """Severity levels for retrieval warnings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TraceStatus(Enum):
    """Status of a retrieval trace."""
    SUCCESS = "success"
    FAILED = "failed"


class FailureStage(Enum):
    """Stage at which retrieval failed."""
    REQUEST_VALIDATION = "request_validation"
    SCOPE_POLICY = "scope_policy"
    GATEWAY_EXECUTION = "gateway_execution"
    POST_VALIDATION = "post_validation"


@dataclass
class RetrievalScope:
    """The boundary within which a retrieval operation is permitted to execute."""
    service_name: str
    tenant_id: str
    collections: list[str]
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieveRequest:
    """External request to retrieve content."""
    query: str
    retrieval_mode: RetrievalMode
    limit: int
    scope: RetrievalScope
    correlation_id: str | None = None


@dataclass
class RetrievalWarning:
    """Non-fatal observation emitted during retrieval."""
    code: WarningCode
    severity: WarningSeverity
    source: str
    message: str
    details: dict[str, Any] | None = None


@dataclass
class ScopeDecision:
    """Output of scope policy evaluation."""
    validated_scope: RetrievalScope
    policy_name: str
    warnings: list[RetrievalWarning] = field(default_factory=list)


@dataclass
class EffectiveRetrieveRequest:
    """Internal request created after validation and scope policy enforcement."""
    normalized_query: str
    original_query: str
    retrieval_mode: RetrievalMode
    limit: int
    validated_scope: ScopeDecision
    correlation_id: str | None = None


@dataclass
class RetrievedChunk:
    """A single piece of content returned from retrieval with normalized metadata."""
    content: str
    score: float
    metadata: dict[str, Any]


@dataclass
class RetrievalGatewayResult:
    """Result returned from a retrieval gateway."""
    chunks: list[RetrievedChunk]
    warnings: list[RetrievalWarning]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    """Structured diagnostic data generated for every retrieval attempt."""
    trace_id: str
    correlation_id: str | None
    status: TraceStatus
    failure_stage: FailureStage | None
    request_summary: dict[str, Any]
    timing: dict[str, Any]
    result_count: int
    warnings: list[RetrievalWarning] = field(default_factory=list)
