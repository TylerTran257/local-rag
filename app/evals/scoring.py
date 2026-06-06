from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalScore:
    retrieved_documents: tuple[str, ...]
    relevant_rank: int | None
    hit: bool
    reciprocal_rank: float


def deduplicate_document_ranking(contexts: Sequence[dict]) -> list[str]:
    ranking: list[str] = []
    seen_documents: set[str] = set()

    for context in contexts:
        original_filename = context["original_filename"]
        if original_filename in seen_documents:
            continue

        seen_documents.add(original_filename)
        ranking.append(original_filename)

    return ranking


def score_retrieval(
    expected_documents: Sequence[str],
    contexts: Sequence[dict],
    top_k: int,
) -> RetrievalScore:
    retrieved_documents = tuple(deduplicate_document_ranking(contexts))
    relevant_rank = next(
        (
            rank
            for rank, document in enumerate(retrieved_documents, start=1)
            if document in expected_documents
        ),
        None,
    )
    hit = relevant_rank is not None and relevant_rank <= top_k
    reciprocal_rank = 0.0 if relevant_rank is None else 1 / relevant_rank

    return RetrievalScore(
        retrieved_documents=retrieved_documents,
        relevant_rank=relevant_rank,
        hit=hit,
        reciprocal_rank=reciprocal_rank,
    )
