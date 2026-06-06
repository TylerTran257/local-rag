import json
import shutil
from pathlib import Path

from app.evals.answer_eval import AnswerEvaluation
from app.evals.golden_eval import (
    GoldenEvalRunReport,
    SeedExampleRunResult,
    main,
    run_golden_eval,
)
from app.evals.scoring import deduplicate_document_ranking, score_retrieval


class FakeEngine:
    def dispose(self):
        return None


class FakeDocumentService:
    def __init__(self, contexts_by_query):
        self.contexts_by_query = contexts_by_query
        self.ingested_documents = []

    def create_document_from_path(self, file_path, original_filename=None):
        document_id = original_filename or file_path.name
        self.ingested_documents.append(document_id)
        return {"document_id": document_id}

    def extract_text(self, document_id):
        return {"document_id": document_id, "status": "text_extracted"}

    def chunk_document(self, document_id):
        return {"document_id": document_id, "status": "chunked", "chunk_count": 1}

    def embed_document(self, document_id):
        return {"document_id": document_id, "status": "embedded", "embedding_count": 1}

    def retrieve_context(self, query, limit):
        return self.contexts_by_query[query][:limit]


def _write_eval_set(tmp_path, payload):
    eval_path = tmp_path / "golden_eval_set.json"
    eval_path.write_text(json.dumps(payload), encoding="utf-8")
    return eval_path


def _make_context(original_filename, chunk_index=0, score=0.9, text="chunk text"):
    return {
        "document_id": f"doc-{original_filename}-{chunk_index}",
        "original_filename": original_filename,
        "chunk_index": chunk_index,
        "score": score,
        "text": text,
    }


def test_deduplicate_document_ranking_preserves_first_appearance():
    contexts = [
        _make_context("doc-a.txt", chunk_index=0),
        _make_context("doc-a.txt", chunk_index=1),
        _make_context("doc-b.txt", chunk_index=0),
    ]

    ranking = deduplicate_document_ranking(contexts)

    assert ranking == ["doc-a.txt", "doc-b.txt"]


def test_score_retrieval_uses_document_level_hit_and_mrr():
    contexts = [
        _make_context("doc-b.txt", chunk_index=0),
        _make_context("doc-a.txt", chunk_index=0),
        _make_context("doc-a.txt", chunk_index=1),
    ]

    result = score_retrieval(
        expected_documents=["doc-a.txt"],
        contexts=contexts,
        top_k=3,
    )

    assert result.retrieved_documents == ("doc-b.txt", "doc-a.txt")
    assert result.relevant_rank == 2
    assert result.hit is True
    assert result.reciprocal_rank == 0.5


def test_run_golden_eval_filters_examples_and_cleans_up_workspace(tmp_path):
    (tmp_path / "doc-a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "doc-b.txt").write_text("world", encoding="utf-8")
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_documents": ["doc-a.txt", "doc-b.txt"],
            "examples": [
                {
                    "id": "first-example",
                    "query": "first",
                    "expected_documents": ["doc-a.txt"],
                },
                {
                    "id": "second-example",
                    "query": "second",
                    "expected_documents": ["doc-b.txt"],
                },
            ],
        },
    )

    fake_document_service = FakeDocumentService(
        {
            "first": [_make_context("doc-a.txt")],
            "second": [_make_context("doc-b.txt")],
        }
    )

    report = run_golden_eval(
        eval_set_path=eval_path,
        repo_root=tmp_path,
        example_ids=["second-example"],
        document_service_factory=lambda workspace: (fake_document_service, FakeEngine()),
    )

    assert [result.example_id for result in report.example_results] == ["second-example"]
    assert fake_document_service.ingested_documents == ["doc-a.txt", "doc-b.txt"]
    assert report.retrieval_passed is True
    assert report.workspace_path.exists() is False


def test_run_golden_eval_keeps_artifacts_when_requested(tmp_path):
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
                }
            ],
        },
    )

    report = run_golden_eval(
        eval_set_path=eval_path,
        repo_root=tmp_path,
        keep_artifacts=True,
        document_service_factory=lambda workspace: (
            FakeDocumentService({"first": [_make_context("doc.txt")]}),
            FakeEngine(),
        ),
    )

    assert report.workspace_path.exists() is True
    shutil.rmtree(report.workspace_path)


def test_run_golden_eval_reports_retrieval_failure(tmp_path):
    (tmp_path / "doc-a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "doc-b.txt").write_text("world", encoding="utf-8")
    eval_path = _write_eval_set(
        tmp_path,
        {
            "corpus_documents": ["doc-a.txt", "doc-b.txt"],
            "examples": [
                {
                    "id": "failing-example",
                    "query": "first",
                    "expected_documents": ["doc-a.txt"],
                }
            ],
        },
    )

    report = run_golden_eval(
        eval_set_path=eval_path,
        repo_root=tmp_path,
        document_service_factory=lambda workspace: (
            FakeDocumentService({"first": [_make_context("doc-b.txt")]}),
            FakeEngine(),
        ),
    )

    assert report.retrieval_passed is False
    assert report.hit_count == 0
    assert report.example_results[0].retrieval_result.relevant_rank is None


def test_run_golden_eval_runs_informational_answer_eval_with_fake_generation_service(
    tmp_path,
):
    class FakeGenerationService:
        def __init__(self):
            self.calls = []

        def answer_question(self, question, sources):
            self.calls.append((question, sources))
            return "Chunk overlap helps across a chunk boundary [1]"

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
                    "expected_answer_facts": [["chunk overlap"], ["chunk boundary"]],
                    "require_citation": True,
                }
            ],
        },
    )

    report = run_golden_eval(
        eval_set_path=eval_path,
        repo_root=tmp_path,
        with_answer_eval=True,
        document_service_factory=lambda workspace: (
            FakeDocumentService({"first": [_make_context("doc.txt")]}),
            FakeEngine(),
        ),
        generation_service_factory=FakeGenerationService,
    )

    answer_evaluation = report.example_results[0].answer_evaluation
    assert answer_evaluation is not None
    assert answer_evaluation.skipped is False
    assert answer_evaluation.has_inline_citation is True
    assert [fact.covered for fact in answer_evaluation.fact_results] == [True, True]


def test_main_returns_non_zero_and_prints_failure_details(capsys, monkeypatch):
    failure_report = GoldenEvalRunReport(
        eval_set_path=Path("/tmp/golden_eval_set.json"),
        workspace_path=Path("/tmp/golden-workspace"),
        example_results=(
            SeedExampleRunResult(
                example_id="failing-example",
                query="what is rag",
                expected_documents=("doc-a.txt",),
                retrieval_result=score_retrieval(
                    expected_documents=["doc-a.txt"],
                    contexts=[_make_context("doc-b.txt")],
                    top_k=3,
                ),
                answer_evaluation=AnswerEvaluation(
                    answer="",
                    fact_results=tuple(),
                    has_inline_citation=False,
                    skipped=True,
                    skip_reason="skipped_due_to_retrieval_failure",
                ),
            ),
        ),
        hit_count=0,
        example_count=1,
        hit_rate=0.0,
        mrr=0.0,
        retrieval_passed=False,
        with_answer_eval=False,
        kept_artifacts=False,
    )

    monkeypatch.setattr("app.evals.golden_eval.run_golden_eval", lambda **kwargs: failure_report)

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Retrieval gate: FAILED" in captured.out
    assert "expected: doc-a.txt" in captured.out
    assert "retrieved: doc-b.txt" in captured.out
