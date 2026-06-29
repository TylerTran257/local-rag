"""Answer use case: shared, transport-agnostic grounded-answer orchestration.

See docs/adr/0002-answer-use-case.md.
"""
from app.answer.contracts import AnswerRequest, AnswerResult, AnswerStream
from app.answer.use_case import NO_GROUNDED_ANSWER, AnswerUseCase

__all__ = [
    "AnswerRequest",
    "AnswerResult",
    "AnswerStream",
    "AnswerUseCase",
    "NO_GROUNDED_ANSWER",
]
