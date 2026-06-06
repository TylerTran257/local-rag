import re
from dataclasses import dataclass
from typing import Sequence


INLINE_CITATION_PATTERN = re.compile(r"\[\d+\](?:\[\d+\])*")


@dataclass(frozen=True)
class FactCoverage:
    required_phrases: tuple[str, ...]
    covered: bool


@dataclass(frozen=True)
class AnswerEvaluation:
    answer: str
    fact_results: tuple[FactCoverage, ...]
    has_inline_citation: bool
    skipped: bool = False
    skip_reason: str | None = None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def has_inline_citation(answer: str) -> bool:
    return INLINE_CITATION_PATTERN.search(answer) is not None


def evaluate_answer(
    answer: str,
    expected_answer_facts: Sequence[Sequence[str]],
    require_citation: bool,
) -> AnswerEvaluation:
    normalized_answer = normalize_text(answer)
    fact_results = tuple(
        FactCoverage(
            required_phrases=tuple(fact_group),
            covered=all(
                normalize_text(required_phrase) in normalized_answer
                for required_phrase in fact_group
            ),
        )
        for fact_group in expected_answer_facts
    )

    return AnswerEvaluation(
        answer=answer,
        fact_results=fact_results,
        has_inline_citation=(not require_citation) or has_inline_citation(answer),
    )


def skipped_answer_evaluation(reason: str) -> AnswerEvaluation:
    return AnswerEvaluation(
        answer="",
        fact_results=tuple(),
        has_inline_citation=False,
        skipped=True,
        skip_reason=reason,
    )


def evaluate_answer_or_skip(
    retrieval_hit: bool,
    generation_service,
    question: str,
    contexts: Sequence[dict],
    expected_answer_facts: Sequence[Sequence[str]],
    require_citation: bool,
) -> AnswerEvaluation:
    if not retrieval_hit:
        return skipped_answer_evaluation("skipped_due_to_retrieval_failure")

    answer = generation_service.answer_question(question, list(contexts))
    return evaluate_answer(answer, expected_answer_facts, require_citation)
