from __future__ import annotations

from app.answer.contracts import AnswerRequest, AnswerResult, AnswerStream
from app.retrieval.types import RetrievedChunk, RetrieveRequest
from app.retrieval.use_case import RetrieveResult, RetrieveUseCase
from app.services.generation_service import GenerationService

# The single definition of the service's response when no in-scope chunks are
# found. Returned by both execute() and stream(); generation is not called.
NO_GROUNDED_ANSWER = (
    "I couldn't find any relevant information to answer your question."
)


async def _single_token(text: str):
    """Yield one token — used to stream the no-grounded-answer fallback."""
    yield text


class AnswerUseCase:
    """Retrieve in-scope chunks and generate a grounded answer.

    Transport-agnostic: callers get either a complete answer (execute) or a
    token stream (stream); errors propagate to the caller for framing.
    """

    def __init__(
        self,
        retrieve_use_case: RetrieveUseCase,
        generation_service: GenerationService,
    ) -> None:
        self.retrieve_use_case = retrieve_use_case
        self.generation_service = generation_service

    def execute(self, request: AnswerRequest) -> AnswerResult:
        result = self._retrieve(request)
        if not result.chunks:
            return AnswerResult(
                answer=NO_GROUNDED_ANSWER, sources=[], trace_id=result.trace_id
            )
        answer = self.generation_service.answer_question(
            request.query, self._to_sources(result.chunks)
        )
        return AnswerResult(
            answer=answer, sources=result.chunks, trace_id=result.trace_id
        )

    async def stream(self, request: AnswerRequest) -> AnswerStream:
        # Retrieve eagerly so RetrievalError surfaces before streaming starts;
        # only token generation is lazy.
        result = self._retrieve(request)
        if not result.chunks:
            return AnswerStream(
                sources=[],
                tokens=_single_token(NO_GROUNDED_ANSWER),
                trace_id=result.trace_id,
            )
        tokens = self.generation_service.stream_answer_question(
            request.query, self._to_sources(result.chunks)
        )
        return AnswerStream(
            sources=result.chunks, tokens=tokens, trace_id=result.trace_id
        )

    def _retrieve(self, request: AnswerRequest) -> RetrieveResult:
        return self.retrieve_use_case.execute(
            RetrieveRequest(
                query=request.query,
                retrieval_mode=request.retrieval_mode,
                limit=request.limit,
                scope=request.scope,
            )
        )

    def _to_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        """Map retrieved chunks to the source dicts GenerationService expects."""
        return [
            {
                "document_id": chunk.metadata.get("document_id", "unknown"),
                "original_filename": chunk.metadata.get("source_label", "unknown"),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "score": chunk.score,
                "text": chunk.content,
            }
            for chunk in chunks
        ]
