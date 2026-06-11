import pytest

from app.evals.social_style_eval import main


def test_help_flag_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0


def test_main_raises_for_missing_eval_set(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        main(["--eval-set", str(missing_path)])


def test_main_returns_error_for_unknown_example_id(tmp_path):
    import json

    eval_path = tmp_path / "eval_set.json"
    eval_path.write_text(
        json.dumps(
            {
                "corpus_chunks": [
                    {
                        "chunk_id": "c1",
                        "tenant_id": "t1",
                        "style_category": "voice_rules",
                        "source_label": "s1",
                        "text": "placeholder",
                    }
                ],
                "examples": [
                    {
                        "id": "e1",
                        "query": "test",
                        "tenant_id": "t1",
                        "target_categories": ["voice_rules"],
                        "expected_source_labels": ["s1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--eval-set", str(eval_path), "--example-id", "nonexistent"])

    assert exit_code == 2
