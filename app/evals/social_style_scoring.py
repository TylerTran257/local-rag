from dataclasses import dataclass

from app.evals.social_style_contracts import SocialStyleEvalExample
from app.social import StyleCategory, StyleContext


@dataclass(frozen=True)
class SocialStyleRetrievalScore:
    hit: bool
    matched_expected_count: int
    total_expected_count: int
    recall_at_k: float
    covered_category_count: int
    requested_category_count: int
    category_coverage: float
    missing_required_category_count: int
    unexpected_category_count: int
    matched_source_labels: tuple[str, ...]
    matched_chunk_ids: tuple[str, ...]


def _entries_by_category(context: StyleContext) -> dict[StyleCategory, list]:
    return {
        StyleCategory.VOICE_RULES: context.voice_rules,
        StyleCategory.HOOK_PATTERNS: context.hook_patterns,
        StyleCategory.CTA_PATTERNS: context.cta_patterns,
        StyleCategory.PAST_POST_PATTERNS: context.past_post_patterns,
        StyleCategory.AVOID_RULES: context.avoid_rules,
        StyleCategory.OFFER_POSITIONING: context.offer_positioning,
    }


def _build_expected_categories(
    example: SocialStyleEvalExample,
    source_label_categories: dict[str, StyleCategory],
    chunk_id_categories: dict[str, StyleCategory],
) -> dict[StyleCategory, dict[str, set[str]]]:
    expected_categories = {
        category: {"source_labels": set(), "chunk_ids": set()}
        for category in example.target_categories
    }

    for source_label in example.expected_source_labels:
        category = source_label_categories[source_label]
        if category in expected_categories:
            expected_categories[category]["source_labels"].add(source_label)

    for chunk_id in example.expected_chunk_ids:
        category = chunk_id_categories[chunk_id]
        if category in expected_categories:
            expected_categories[category]["chunk_ids"].add(chunk_id)

    return expected_categories


def score_style_retrieval(
    *,
    example: SocialStyleEvalExample,
    context: StyleContext,
    top_k: int,
    source_label_categories: dict[str, StyleCategory],
    chunk_id_categories: dict[str, StyleCategory],
) -> SocialStyleRetrievalScore:
    entries_by_category = _entries_by_category(context)
    expected_categories = _build_expected_categories(
        example,
        source_label_categories,
        chunk_id_categories,
    )

    matched_source_labels: set[str] = set()
    matched_chunk_ids: set[str] = set()
    matched_expected_count = 0
    total_expected_count = 0
    covered_category_count = 0
    category_hits: list[bool] = []

    for category in example.target_categories:
        entries = entries_by_category.get(category, [])[:top_k]
        if entries:
            covered_category_count += 1

        observed_source_labels = {entry.source_label for entry in entries}
        observed_chunk_ids = {
            str(entry.metadata.get("eval_chunk_id"))
            for entry in entries
            if entry.metadata.get("eval_chunk_id") is not None
        }
        expected_source_labels = expected_categories[category]["source_labels"]
        expected_chunk_ids = expected_categories[category]["chunk_ids"]
        total_expected_count += len(expected_source_labels) + len(expected_chunk_ids)

        matched_labels = observed_source_labels.intersection(expected_source_labels)
        matched_ids = observed_chunk_ids.intersection(expected_chunk_ids)

        matched_source_labels.update(matched_labels)
        matched_chunk_ids.update(matched_ids)
        matched_expected_count += len(matched_labels) + len(matched_ids)

        if len(expected_source_labels) == 0 and len(expected_chunk_ids) == 0:
            category_hits.append(len(entries) > 0)
        else:
            category_hits.append(bool(matched_labels or matched_ids))

    requested_category_count = len(example.target_categories)
    category_coverage = (
        0.0
        if requested_category_count == 0
        else covered_category_count / requested_category_count
    )
    missing_required_category_count = requested_category_count - covered_category_count

    requested_categories = set(example.target_categories)
    excluded_categories = set(example.expected_exclusions)
    unexpected_category_count = sum(
        1
        for category, entries in entries_by_category.items()
        if entries and (category not in requested_categories or category in excluded_categories)
    )

    recall_at_k = (
        0.0
        if total_expected_count == 0
        else matched_expected_count / total_expected_count
    )

    return SocialStyleRetrievalScore(
        hit=all(category_hits),
        matched_expected_count=matched_expected_count,
        total_expected_count=total_expected_count,
        recall_at_k=recall_at_k,
        covered_category_count=covered_category_count,
        requested_category_count=requested_category_count,
        category_coverage=category_coverage,
        missing_required_category_count=missing_required_category_count,
        unexpected_category_count=unexpected_category_count,
        matched_source_labels=tuple(sorted(matched_source_labels)),
        matched_chunk_ids=tuple(sorted(matched_chunk_ids)),
    )
