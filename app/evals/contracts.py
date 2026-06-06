import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SeedExample:
    id: str
    query: str
    expected_documents: tuple[str, ...]
    expected_answer_facts: tuple[tuple[str, ...], ...]
    require_citation: bool = False


@dataclass(frozen=True)
class GoldenEvalSet:
    source_path: Path
    corpus_documents: tuple[str, ...]
    examples: tuple[SeedExample, ...]


def _load_json(eval_set_path: Path) -> dict:
    try:
        payload = json.loads(eval_set_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Golden eval set JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Golden eval set JSON must be an object")

    return payload


def _validate_non_empty_string(value: object, error_message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error_message)

    return value.strip()


def _validate_answer_fact_groups(raw_fact_groups: object) -> tuple[tuple[str, ...], ...]:
    if raw_fact_groups is None:
        return tuple()

    if not isinstance(raw_fact_groups, list):
        raise ValueError("expected_answer_facts must be a list of phrase groups")

    fact_groups: list[tuple[str, ...]] = []
    for raw_group in raw_fact_groups:
        if not isinstance(raw_group, list) or len(raw_group) == 0:
            raise ValueError("Each answer fact group must contain at least one phrase")

        phrases = tuple(
            _validate_non_empty_string(
                raw_phrase,
                "Each answer fact phrase must be a non-empty string",
            )
            for raw_phrase in raw_group
        )
        fact_groups.append(phrases)

    return tuple(fact_groups)


def load_golden_eval_set(eval_set_path: Path, repo_root: Path) -> GoldenEvalSet:
    payload = _load_json(eval_set_path)

    raw_corpus_documents = payload.get("corpus_documents")
    if not isinstance(raw_corpus_documents, list) or len(raw_corpus_documents) == 0:
        raise ValueError("corpus_documents must contain at least one document")

    corpus_documents = tuple(
        _validate_non_empty_string(
            raw_document,
            "Each corpus document must be a non-empty string",
        )
        for raw_document in raw_corpus_documents
    )

    for document_path in corpus_documents:
        if not (repo_root / document_path).exists():
            raise ValueError(f"Corpus document does not exist: {document_path}")

    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list) or len(raw_examples) == 0:
        raise ValueError("examples must contain at least one seed example")

    seen_ids: set[str] = set()
    examples: list[SeedExample] = []
    for raw_example in raw_examples:
        if not isinstance(raw_example, dict):
            raise ValueError("Each seed example must be an object")

        example_id = _validate_non_empty_string(
            raw_example.get("id"),
            "Each seed example must have a non-empty id",
        )
        if example_id in seen_ids:
            raise ValueError(f"Duplicate seed example id: {example_id}")
        seen_ids.add(example_id)

        query = _validate_non_empty_string(
            raw_example.get("query"),
            "Each seed example must have a non-empty query",
        )

        raw_expected_documents = raw_example.get("expected_documents")
        if (
            not isinstance(raw_expected_documents, list)
            or len(raw_expected_documents) == 0
        ):
            raise ValueError(
                "Each seed example must have at least one expected document"
            )

        expected_documents = tuple(
            _validate_non_empty_string(
                raw_document,
                "Each expected document must be a non-empty string",
            )
            for raw_document in raw_expected_documents
        )
        for expected_document in expected_documents:
            if expected_document not in corpus_documents:
                raise ValueError(
                    "Expected document must appear in corpus_documents: "
                    f"{expected_document}"
                )

        raw_require_citation = raw_example.get("require_citation", False)
        if not isinstance(raw_require_citation, bool):
            raise ValueError("require_citation must be a boolean when provided")

        examples.append(
            SeedExample(
                id=example_id,
                query=query,
                expected_documents=expected_documents,
                expected_answer_facts=_validate_answer_fact_groups(
                    raw_example.get("expected_answer_facts")
                ),
                require_citation=raw_require_citation,
            )
        )

    return GoldenEvalSet(
        source_path=eval_set_path,
        corpus_documents=corpus_documents,
        examples=tuple(examples),
    )


def select_examples(
    eval_set: GoldenEvalSet,
    example_ids: Sequence[str] | None,
) -> tuple[SeedExample, ...]:
    if not example_ids:
        return eval_set.examples

    requested_ids = {
        _validate_non_empty_string(example_id, "Seed example ids must be non-empty")
        for example_id in example_ids
    }
    known_ids = {example.id for example in eval_set.examples}
    missing_ids = sorted(requested_ids - known_ids)
    if missing_ids:
        raise ValueError(f"Unknown seed example id: {', '.join(missing_ids)}")

    return tuple(example for example in eval_set.examples if example.id in requested_ids)
