import asyncio
from unittest.mock import Mock

import pytest

from app.answer.contracts import AnswerRequest
from app.answer.use_case import NO_GROUNDED_ANSWER, AnswerUseCase
from app.retrieval.errors import RetrievalError, RetrievalExecutionError
from app.retrieval.types import RetrievalMode, RetrievalScope, RetrievedChunk
from app.retrieval.use_case import RetrieveResult
from app.services.generation_service import GenerationServiceError


def _chunk(content: str = "context text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        content=content,
        score=0.9,
        rank=0,
        retrieval_mode=RetrievalMode.HYBRID,
        metadata={
            "service_name": "svc",
            "tenant_id": "ten",
            "collection": "docs",
            "source_label": "file.pdf",
            "chunk_index": 0,
            "document_id": "doc-1",
        },
    )


def _request() -> AnswerRequest:
    return AnswerRequest(
        query="what is x?",
        retrieval_mode=RetrievalMode.HYBRID,
        limit=5,
        scope=RetrievalScope(service_name="svc", tenant_id="ten", collections=["docs"]),
    )


@pytest.fixture
def retrieve_use_case():
    use_case = Mock()
    use_case.execute.return_value = RetrieveResult(
        chunks=[_chunk()], warnings=[], trace_id="trace-1"
    )
    return use_case


@pytest.fixture
def generation_service():
    service = Mock()
    service.answer_question.return_value = "generated answer"
    return service


class TestExecute:
    def test_grounded_returns_generated_answer_and_sources(
        self, retrieve_use_case, generation_service
    ):
        use_case = AnswerUseCase(retrieve_use_case, generation_service)

        result = use_case.execute(_request())

        assert result.answer == "generated answer"
        assert result.trace_id == "trace-1"
        assert [c.chunk_id for c in result.sources] == ["chunk-1"]

        # generation is called with the query and mapped source dicts
        generation_service.answer_question.assert_called_once()
        call_query, call_sources = generation_service.answer_question.call_args[0]
        assert call_query == "what is x?"
        assert call_sources[0]["text"] == "context text"
        assert call_sources[0]["document_id"] == "doc-1"
        assert call_sources[0]["original_filename"] == "file.pdf"

    def test_empty_returns_fallback_without_calling_generation(
        self, retrieve_use_case, generation_service
    ):
        retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[], warnings=[], trace_id="trace-empty"
        )
        use_case = AnswerUseCase(retrieve_use_case, generation_service)

        result = use_case.execute(_request())

        assert result.answer == NO_GROUNDED_ANSWER
        assert result.sources == []
        assert result.trace_id == "trace-empty"
        generation_service.answer_question.assert_not_called()

    def test_generation_failure_propagates(
        self, retrieve_use_case, generation_service
    ):
        generation_service.answer_question.side_effect = GenerationServiceError(
            "boom"
        )
        use_case = AnswerUseCase(retrieve_use_case, generation_service)

        with pytest.raises(GenerationServiceError):
            use_case.execute(_request())


def _drain(stream):
    async def _collect():
        return [t async for t in stream.tokens]

    return asyncio.run(_collect())


class TestStream:
    def test_grounded_exposes_sources_and_streams_tokens(
        self, retrieve_use_case, generation_service
    ):
        async def fake_stream(query, sources):
            for token in ["gen ", "tokens"]:
                yield token

        generation_service.stream_answer_question = fake_stream
        use_case = AnswerUseCase(retrieve_use_case, generation_service)

        stream = asyncio.run(use_case.stream(_request()))

        assert [c.chunk_id for c in stream.sources] == ["chunk-1"]
        assert stream.trace_id == "trace-1"
        assert _drain(stream) == ["gen ", "tokens"]

    def test_empty_streams_fallback_as_single_token(
        self, retrieve_use_case, generation_service
    ):
        retrieve_use_case.execute.return_value = RetrieveResult(
            chunks=[], warnings=[], trace_id="trace-empty"
        )
        generation_service.stream_answer_question = Mock(
            side_effect=AssertionError("generation must not be called")
        )
        use_case = AnswerUseCase(retrieve_use_case, generation_service)

        stream = asyncio.run(use_case.stream(_request()))

        assert stream.sources == []
        assert _drain(stream) == [NO_GROUNDED_ANSWER]

    def test_generation_error_propagates_mid_stream(
        self, retrieve_use_case, generation_service
    ):
        async def failing_stream(query, sources):
            yield "partial "
            raise GenerationServiceError("mid-stream boom")

        generation_service.stream_answer_question = failing_stream
        use_case = AnswerUseCase(retrieve_use_case, generation_service)

        stream = asyncio.run(use_case.stream(_request()))

        with pytest.raises(GenerationServiceError):
            _drain(stream)

    def test_retrieval_error_propagates_before_streaming(
        self, retrieve_use_case, generation_service
    ):
        retrieve_use_case.execute.side_effect = RetrievalExecutionError(
            trace_id="t", internal_message="corpus exploded", details={}
        )
        use_case = AnswerUseCase(retrieve_use_case, generation_service)

        with pytest.raises(RetrievalError):
            asyncio.run(use_case.stream(_request()))
