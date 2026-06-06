import json

import pytest

from app.evals.contracts import load_golden_eval_set, select_examples


def _write_eval_set(tmp_path, payload):
    eval_path = tmp_path / "golden_eval_set.json"
    eval_path.write_text(json.dumps(payload), encoding="utf-8")
    return eval_path


def test_load_golden_eval_set_rejects_duplicate_example_ids(tmp_path):
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_documents": ["doc.txt"],
            "examples": [
                {
                    "id": "duplicate-id",
                    "query": "first",
                    "expected_documents": ["doc.txt"],
                },
                {
                    "id": "duplicate-id",
                    "query": "second",
                    "expected_documents": ["doc.txt"],
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="Duplicate seed example id"):
        load_golden_eval_set(eval_path, tmp_path)


def test_load_golden_eval_set_rejects_non_object_top_level_payload(tmp_path):
    eval_path = tmp_path / "golden_eval_set.json"
    eval_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="Golden eval set JSON must be an object"):
        load_golden_eval_set(eval_path, tmp_path)


def test_load_golden_eval_set_rejects_expected_document_outside_corpus(tmp_path):
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "other.txt").write_text("world", encoding="utf-8")
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_documents": ["doc.txt"],
            "examples": [
                {
                    "id": "bad-reference",
                    "query": "first",
                    "expected_documents": ["other.txt"],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="Expected document must appear in corpus_documents"):
        load_golden_eval_set(eval_path, tmp_path)


def test_load_golden_eval_set_rejects_malformed_answer_fact_group(tmp_path):
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_documents": ["doc.txt"],
            "examples": [
                {
                    "id": "bad-facts",
                    "query": "first",
                    "expected_documents": ["doc.txt"],
                    "expected_answer_facts": [["chunk overlap"], []],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="Each answer fact group must contain at least one phrase"):
        load_golden_eval_set(eval_path, tmp_path)


def test_load_golden_eval_set_rejects_non_boolean_require_citation(tmp_path):
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_documents": ["doc.txt"],
            "examples": [
                {
                    "id": "bad-citation-flag",
                    "query": "first",
                    "expected_documents": ["doc.txt"],
                    "require_citation": "yes",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="require_citation must be a boolean"):
        load_golden_eval_set(eval_path, tmp_path)


def test_select_examples_filters_by_id_and_rejects_unknown_ids(tmp_path):
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_documents": ["doc.txt"],
            "examples": [
                {
                    "id": "first-example",
                    "query": "first",
                    "expected_documents": ["doc.txt"],
                },
                {
                    "id": "second-example",
                    "query": "second",
                    "expected_documents": ["doc.txt"],
                },
            ],
        },
    )

    eval_set = load_golden_eval_set(eval_path, tmp_path)

    selected = select_examples(eval_set, ["second-example"])
    assert [example.id for example in selected] == ["second-example"]

    with pytest.raises(ValueError, match="Unknown seed example id"):
        select_examples(eval_set, ["missing-example"])
