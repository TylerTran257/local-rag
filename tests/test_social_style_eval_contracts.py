import json

import pytest

from app.evals.social_style_contracts import load_social_style_eval_set, select_examples
from app.social import StyleCategory


def _write_eval_set(tmp_path, payload):
    eval_path = tmp_path / "social_style_eval_set.json"
    eval_path.write_text(json.dumps(payload), encoding="utf-8")
    return eval_path


def test_load_social_style_eval_set_parses_chunks_and_examples(tmp_path):
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_chunks": [
                {
                    "chunk_id": "voice-1",
                    "tenant_id": "tenant-a",
                    "platform": "linkedin",
                    "style_category": "voice_rules",
                    "source_label": "voice-guide",
                    "text": "Use a confident tone.",
                }
            ],
            "examples": [
                {
                    "id": "voice-example",
                    "query": "confident tone",
                    "tenant_id": "tenant-a",
                    "platform": "linkedin",
                    "target_categories": ["voice_rules"],
                    "expected_source_labels": ["voice-guide"],
                }
            ],
        },
    )

    eval_set = load_social_style_eval_set(eval_path)

    assert eval_set.source_path == eval_path
    assert eval_set.corpus_chunks[0].style_category == StyleCategory.VOICE_RULES
    assert eval_set.examples[0].target_categories == (StyleCategory.VOICE_RULES,)
    assert eval_set.examples[0].expected_source_labels == ("voice-guide",)


def test_load_social_style_eval_set_requires_example_expectations(tmp_path):
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_chunks": [
                {
                    "chunk_id": "voice-1",
                    "tenant_id": "tenant-a",
                    "style_category": "voice_rules",
                    "source_label": "voice-guide",
                    "text": "Use a confident tone.",
                }
            ],
            "examples": [
                {
                    "id": "voice-example",
                    "query": "confident tone",
                    "tenant_id": "tenant-a",
                    "target_categories": ["voice_rules"],
                }
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="must define expected_source_labels or expected_chunk_ids",
    ):
        load_social_style_eval_set(eval_path)


def test_select_examples_rejects_unknown_ids(tmp_path):
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_chunks": [
                {
                    "chunk_id": "voice-1",
                    "tenant_id": "tenant-a",
                    "style_category": "voice_rules",
                    "source_label": "voice-guide",
                    "text": "Use a confident tone.",
                }
            ],
            "examples": [
                {
                    "id": "voice-example",
                    "query": "confident tone",
                    "tenant_id": "tenant-a",
                    "target_categories": ["voice_rules"],
                    "expected_source_labels": ["voice-guide"],
                }
            ],
        },
    )

    eval_set = load_social_style_eval_set(eval_path)

    with pytest.raises(ValueError, match="Unknown social style eval example ids"):
        select_examples(eval_set, ["missing-example"])
