import pytest

from app.evals.answer_eval import (
    evaluate_answer,
    evaluate_answer_or_skip,
)
from app.services.generation_service import GenerationServiceError


class FakeGenerationService:
    def __init__(self, answer="answer", error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    def answer_question(self, question, sources):
        self.calls.append((question, sources))
        if self.error is not None:
            raise self.error
        return self.answer


def test_evaluate_answer_checks_phrase_groups_and_inline_citations():
    result = evaluate_answer(
        answer="Chunk overlap preserves context across a boundary [1]",
        expected_answer_facts=[("chunk overlap",), ("preserves", "context")],
        require_citation=True,
    )

    assert result.skipped is False
    assert result.has_inline_citation is True
    assert [fact.covered for fact in result.fact_results] == [True, True]


def test_evaluate_answer_or_skip_returns_skip_when_retrieval_fails():
    result = evaluate_answer_or_skip(
        retrieval_hit=False,
        generation_service=FakeGenerationService(),
        question="what is rag",
        contexts=[],
        expected_answer_facts=[],
        require_citation=False,
    )

    assert result.skipped is True
    assert result.skip_reason == "skipped_due_to_retrieval_failure"


def test_evaluate_answer_or_skip_raises_when_generation_is_requested_but_unavailable():
    with pytest.raises(GenerationServiceError, match="generation unavailable"):
        evaluate_answer_or_skip(
            retrieval_hit=True,
            generation_service=FakeGenerationService(
                error=GenerationServiceError("generation unavailable")
            ),
            question="what is rag",
            contexts=[{"text": "context", "original_filename": "doc.txt", "chunk_index": 0, "score": 0.9}],
            expected_answer_facts=[],
            require_citation=False,
        )
