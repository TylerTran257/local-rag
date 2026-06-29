from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from app.retrieval.types import RetrievalMode, RetrievalScope, RetrievedChunk


@dataclass(frozen=True)
class AnswerRequest:
    """External request to retrieve in-scope chunks and generate an answer."""

    query: str
    retrieval_mode: RetrievalMode
    limit: int
    scope: RetrievalScope


@dataclass(frozen=True)
class AnswerResult:
    """A complete (synchronous) grounded answer with its sources."""

    answer: str
    sources: list[RetrievedChunk]
    trace_id: str | None = None


@dataclass(frozen=True)
class AnswerStream:
    """A streaming grounded answer: sources known up-front, tokens lazy.

    ``tokens`` yields plain answer text; the transport frames it (e.g. SSE).
    """

    sources: list[RetrievedChunk]
    tokens: AsyncIterator[str]
    trace_id: str | None = None
