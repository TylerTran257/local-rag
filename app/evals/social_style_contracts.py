import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.social import StyleCategory


@dataclass(frozen=True)
class SocialStyleCorpusChunk:
    chunk_id: str
    tenant_id: str
    source_label: str
    text: str
    style_category: StyleCategory
    platform: str | None = None


@dataclass(frozen=True)
class SocialStyleEvalExample:
    id: str
    query: str
    tenant_id: str
    target_categories: tuple[StyleCategory, ...]
    expected_source_labels: tuple[str, ...] = tuple()
    expected_chunk_ids: tuple[str, ...] = tuple()
    expected_exclusions: tuple[StyleCategory, ...] = tuple()
    platform: str | None = None


@dataclass(frozen=True)
class SocialStyleEvalSet:
    source_path: Path
    corpus_chunks: tuple[SocialStyleCorpusChunk, ...]
    examples: tuple[SocialStyleEvalExample, ...]


def _load_json(eval_set_path: Path) -> dict:
    try:
        payload = json.loads(eval_set_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid social style eval set JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Social style eval set JSON must be an object")

    return payload


def _validate_non_empty_string(value: object, error_message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error_message)

    return value.strip()


def _parse_style_category(value: object, error_message: str) -> StyleCategory:
    normalized_value = _validate_non_empty_string(value, error_message)
    try:
        return StyleCategory(normalized_value)
    except ValueError as exc:
        raise ValueError(error_message) from exc


def _parse_category_list(value: object, field_name: str) -> tuple[StyleCategory, ...]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{field_name} must contain at least one category")

    return tuple(
        _parse_style_category(raw_category, f"{field_name} contains an invalid category")
        for raw_category in value
    )


def _parse_optional_string_list(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return tuple()

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")

    return tuple(
        _validate_non_empty_string(item, f"{field_name} must contain non-empty strings")
        for item in value
    )


def _parse_optional_category_list(value: object, field_name: str) -> tuple[StyleCategory, ...]:
    if value is None:
        return tuple()

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")

    return tuple(
        _parse_style_category(raw_category, f"{field_name} contains an invalid category")
        for raw_category in value
    )


def load_social_style_eval_set(eval_set_path: Path) -> SocialStyleEvalSet:
    payload = _load_json(eval_set_path)

    raw_corpus_chunks = payload.get("corpus_chunks")
    if not isinstance(raw_corpus_chunks, list) or len(raw_corpus_chunks) == 0:
        raise ValueError("corpus_chunks must contain at least one chunk")

    seen_chunk_ids: set[str] = set()
    seen_source_labels: set[str] = set()
    corpus_chunks: list[SocialStyleCorpusChunk] = []
    for raw_chunk in raw_corpus_chunks:
        if not isinstance(raw_chunk, dict):
            raise ValueError("Each corpus chunk must be an object")

        chunk_id = _validate_non_empty_string(
            raw_chunk.get("chunk_id"),
            "Each corpus chunk must define a non-empty chunk_id",
        )
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate corpus chunk id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)

        source_label = _validate_non_empty_string(
            raw_chunk.get("source_label"),
            "Each corpus chunk must define a non-empty source_label",
        )
        seen_source_labels.add(source_label)

        raw_platform = raw_chunk.get("platform")
        platform = None
        if raw_platform is not None:
            platform = _validate_non_empty_string(
                raw_platform,
                "Corpus chunk platform must be a non-empty string when provided",
            )

        corpus_chunks.append(
            SocialStyleCorpusChunk(
                chunk_id=chunk_id,
                tenant_id=_validate_non_empty_string(
                    raw_chunk.get("tenant_id"),
                    "Each corpus chunk must define a non-empty tenant_id",
                ),
                source_label=source_label,
                text=_validate_non_empty_string(
                    raw_chunk.get("text"),
                    "Each corpus chunk must define non-empty text",
                ),
                style_category=_parse_style_category(
                    raw_chunk.get("style_category"),
                    "Each corpus chunk must define a valid style_category",
                ),
                platform=platform,
            )
        )

    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list) or len(raw_examples) == 0:
        raise ValueError("examples must contain at least one example")

    seen_example_ids: set[str] = set()
    examples: list[SocialStyleEvalExample] = []
    for raw_example in raw_examples:
        if not isinstance(raw_example, dict):
            raise ValueError("Each social style eval example must be an object")

        example_id = _validate_non_empty_string(
            raw_example.get("id"),
            "Each social style eval example must define a non-empty id",
        )
        if example_id in seen_example_ids:
            raise ValueError(f"Duplicate social style eval example id: {example_id}")
        seen_example_ids.add(example_id)

        raw_platform = raw_example.get("platform")
        platform = None
        if raw_platform is not None:
            platform = _validate_non_empty_string(
                raw_platform,
                "Example platform must be a non-empty string when provided",
            )

        expected_source_labels = _parse_optional_string_list(
            raw_example.get("expected_source_labels"),
            "expected_source_labels",
        )
        expected_chunk_ids = _parse_optional_string_list(
            raw_example.get("expected_chunk_ids"),
            "expected_chunk_ids",
        )
        if len(expected_source_labels) == 0 and len(expected_chunk_ids) == 0:
            raise ValueError(
                "Each social style eval example must define expected_source_labels or expected_chunk_ids"
            )

        missing_source_labels = sorted(
            label for label in expected_source_labels if label not in seen_source_labels
        )
        if missing_source_labels:
            raise ValueError(
                "expected_source_labels must reference source labels present in corpus_chunks"
            )

        missing_chunk_ids = sorted(
            chunk_id for chunk_id in expected_chunk_ids if chunk_id not in seen_chunk_ids
        )
        if missing_chunk_ids:
            raise ValueError(
                "expected_chunk_ids must reference chunk ids present in corpus_chunks"
            )

        examples.append(
            SocialStyleEvalExample(
                id=example_id,
                query=_validate_non_empty_string(
                    raw_example.get("query"),
                    "Each social style eval example must define a non-empty query",
                ),
                tenant_id=_validate_non_empty_string(
                    raw_example.get("tenant_id"),
                    "Each social style eval example must define a non-empty tenant_id",
                ),
                target_categories=_parse_category_list(
                    raw_example.get("target_categories"),
                    "target_categories",
                ),
                expected_source_labels=expected_source_labels,
                expected_chunk_ids=expected_chunk_ids,
                expected_exclusions=_parse_optional_category_list(
                    raw_example.get("expected_exclusions"),
                    "expected_exclusions",
                ),
                platform=platform,
            )
        )

    return SocialStyleEvalSet(
        source_path=eval_set_path,
        corpus_chunks=tuple(corpus_chunks),
        examples=tuple(examples),
    )


def select_examples(
    eval_set: SocialStyleEvalSet,
    example_ids: Sequence[str] | None,
) -> tuple[SocialStyleEvalExample, ...]:
    if example_ids is None:
        return eval_set.examples

    requested_ids = tuple(
        _validate_non_empty_string(example_id, "example ids must be non-empty strings")
        for example_id in example_ids
    )

    example_by_id = {example.id: example for example in eval_set.examples}
    missing_ids = [example_id for example_id in requested_ids if example_id not in example_by_id]
    if missing_ids:
        raise ValueError(
            "Unknown social style eval example ids: " + ", ".join(sorted(missing_ids))
        )

    return tuple(example_by_id[example_id] for example_id in requested_ids)
