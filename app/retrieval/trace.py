"""Trace sink implementations for the Retrieval Core."""
import logging
from datetime import datetime
from uuid import uuid4

from app.retrieval.types import RetrievalTrace


logger = logging.getLogger(__name__)


class InMemoryTraceSink:
    """Stores traces in memory for test assertions."""

    def __init__(self) -> None:
        self.traces: list[RetrievalTrace] = []

    def emit(self, trace: RetrievalTrace) -> None:
        """Store trace in memory."""
        self.traces.append(trace)


class NoOpTraceSink:
    """Discards all traces."""

    def emit(self, trace: RetrievalTrace) -> None:
        """Discard trace."""
        pass


class StructuredLoggingTraceSink:
    """Logs traces via Python structured logging in key=value format."""

    def emit(self, trace: RetrievalTrace) -> None:
        """Log trace in structured format."""
        logger.info(
            "event=retrieval_trace trace_id=%s correlation_id=%s status=%s failure_stage=%s result_count=%s warning_count=%s",
            trace.trace_id,
            trace.correlation_id or "none",
            trace.status.value,
            trace.failure_stage.value if trace.failure_stage else "none",
            trace.result_count,
            len(trace.warnings),
        )


class SystemClock:
    """Provides current system time."""

    def now(self) -> datetime:
        """Return current datetime."""
        return datetime.now()


class UuidTraceIdGenerator:
    """Generates UUID-based trace IDs."""

    def generate(self) -> str:
        """Generate a new UUID trace ID."""
        return str(uuid4())
