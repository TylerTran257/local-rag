"""Domain errors for the Retrieval Core."""
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalError(Exception):
    """Base error for all retrieval domain errors."""
    code: str
    trace_id: str
    internal_message: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.code}: {self.internal_message}"


@dataclass
class InvalidRetrievalRequestError(RetrievalError):
    """Raised when a retrieve request fails validation."""
    def __init__(self, trace_id: str, internal_message: str, details: dict[str, Any]):
        super().__init__(
            code="INVALID_RETRIEVAL_REQUEST",
            trace_id=trace_id,
            internal_message=internal_message,
            details=details
        )


@dataclass
class UnsupportedRetrievalModeError(RetrievalError):
    """Raised when an unsupported retrieval mode is requested."""
    def __init__(self, trace_id: str, internal_message: str, details: dict[str, Any]):
        super().__init__(
            code="UNSUPPORTED_RETRIEVAL_MODE",
            trace_id=trace_id,
            internal_message=internal_message,
            details=details
        )


@dataclass
class NoIndexedCorpusError(RetrievalError):
    """Raised when no indexed corpus is available for retrieval."""
    def __init__(self, trace_id: str, internal_message: str, details: dict[str, Any]):
        super().__init__(
            code="NO_INDEXED_CORPUS",
            trace_id=trace_id,
            internal_message=internal_message,
            details=details
        )


@dataclass
class RetrievalExecutionError(RetrievalError):
    """Raised when retrieval execution fails unexpectedly."""
    def __init__(self, trace_id: str, internal_message: str, details: dict[str, Any]):
        super().__init__(
            code="RETRIEVAL_EXECUTION_ERROR",
            trace_id=trace_id,
            internal_message=internal_message,
            details=details
        )


@dataclass
class RetrievedChunkValidationError(RetrievalError):
    """Raised when post-validation of retrieved chunks fails."""
    def __init__(self, trace_id: str, internal_message: str, details: dict[str, Any]):
        super().__init__(
            code="RETRIEVED_CHUNK_VALIDATION_ERROR",
            trace_id=trace_id,
            internal_message=internal_message,
            details=details
        )
