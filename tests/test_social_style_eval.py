import json
import shutil

from app.evals.social_style_eval import (
    EVAL_COLLECTION_NAME,
    EVAL_SERVICE_NAME,
    SocialStyleEvalRuntime,
    main,
    make_eval_tenant_id,
    run_social_style_eval,
)
from app.social import StyleCategory, StyleContext, StyleEntry


class FakeEngine:
    def dispose(self):
        return None


class FakeIngestUseCase:
    def __init__(self):
        self.calls = []

    def ingest_chunks(self, chunks):
        self.calls.append(chunks)


class FakeSocialService:
    def __init__(self, contexts_by_query):
        self.contexts_by_query = contexts_by_query
        self.calls = []

    def retrieve(self, request):
        self.calls.append(request)
        return self.contexts_by_query[request.query]


def _write_eval_set(tmp_path, payload):
    eval_path = tmp_path / "social_style_eval_set.json"
    eval_path.write_text(json.dumps(payload), encoding="utf-8")
    return eval_path


def _make_style_entry(source_label, style_category, eval_chunk_id, platform="linkedin"):
    return StyleEntry(
        content=f"content for {source_label}",
        source_label=source_label,
        score=0.95,
        metadata={
            "style_category": style_category,
            "platform": platform,
            "eval_chunk_id": eval_chunk_id,
        },
    )


def test_run_social_style_eval_aggregates_metrics_and_cleans_up_workspace(tmp_path):
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
                },
                {
                    "chunk_id": "hook-1",
                    "tenant_id": "tenant-b",
                    "platform": "twitter",
                    "style_category": "hook_patterns",
                    "source_label": "hook-guide",
                    "text": "Open with a question.",
                },
            ],
            "examples": [
                {
                    "id": "voice-example",
                    "query": "confident tone",
                    "tenant_id": "tenant-a",
                    "platform": "linkedin",
                    "target_categories": ["voice_rules"],
                    "expected_source_labels": ["voice-guide"],
                },
                {
                    "id": "hook-example",
                    "query": "question hook",
                    "tenant_id": "tenant-b",
                    "platform": "twitter",
                    "target_categories": ["hook_patterns"],
                    "expected_source_labels": ["hook-guide"],
                },
            ],
        },
    )

    fake_ingest = FakeIngestUseCase()
    fake_social = FakeSocialService(
        {
            "confident tone": StyleContext(
                voice_rules=[
                    _make_style_entry(
                        source_label="voice-guide",
                        style_category="voice_rules",
                        eval_chunk_id="voice-1",
                        platform="linkedin",
                    )
                ]
            ),
            "question hook": StyleContext(),
        }
    )

    report = run_social_style_eval(
        eval_set_path=eval_path,
        repo_root=tmp_path,
        runtime_factory=lambda workspace: SocialStyleEvalRuntime(
            ingest_use_case=fake_ingest,
            social_service=fake_social,
            engine=FakeEngine(),
        ),
    )

    assert report.example_count == 2
    assert report.hit_rate_at_k == 0.5
    assert report.recall_at_k == 0.5
    assert report.category_coverage == 0.5
    assert report.missing_required_category_count == 1
    assert report.unexpected_category_count == 0
    assert report.retrieval_passed is False
    assert report.workspace_path.exists() is False

    assert len(fake_ingest.calls) == 2
    first_ingested_chunk = fake_ingest.calls[0][0]
    assert first_ingested_chunk.service_name == EVAL_SERVICE_NAME
    assert first_ingested_chunk.collection == EVAL_COLLECTION_NAME
    assert first_ingested_chunk.tenant_id == make_eval_tenant_id("tenant-a")
    assert first_ingested_chunk.domain_metadata["style_category"] == "voice_rules"
    assert first_ingested_chunk.domain_metadata["eval_chunk_id"] == "voice-1"

    assert fake_social.calls[0].tenant_id == make_eval_tenant_id("tenant-a")
    assert fake_social.calls[1].tenant_id == make_eval_tenant_id("tenant-b")


def test_run_social_style_eval_keeps_artifacts_when_requested(tmp_path):
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

    report = run_social_style_eval(
        eval_set_path=eval_path,
        repo_root=tmp_path,
        keep_artifacts=True,
        runtime_factory=lambda workspace: SocialStyleEvalRuntime(
            ingest_use_case=FakeIngestUseCase(),
            social_service=FakeSocialService(
                {
                    "confident tone": StyleContext(
                        voice_rules=[
                            _make_style_entry(
                                source_label="voice-guide",
                                style_category="voice_rules",
                                eval_chunk_id="voice-1",
                            )
                        ]
                    )
                }
            ),
            engine=FakeEngine(),
        ),
    )

    assert report.workspace_path.exists() is True
    shutil.rmtree(report.workspace_path)


def test_social_style_eval_main_returns_failure_code(monkeypatch, tmp_path):
    failure_report = type("FailureReport", (), {
        "eval_set_path": tmp_path / "social_style_eval_set.json",
        "workspace_path": tmp_path,
        "example_results": tuple(),
        "example_count": 1,
        "hit_rate_at_k": 0.0,
        "recall_at_k": 0.0,
        "category_coverage": 0.0,
        "missing_required_category_count": 1,
        "unexpected_category_count": 0,
        "retrieval_passed": False,
        "kept_artifacts": False,
    })()

    monkeypatch.setattr(
        "app.evals.social_style_eval.run_social_style_eval",
        lambda **kwargs: failure_report,
    )

    exit_code = main([])

    assert exit_code == 1
