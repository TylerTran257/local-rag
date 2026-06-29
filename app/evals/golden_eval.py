import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from app.evals.answer_eval import AnswerEvaluation, evaluate_answer_or_skip
from app.evals.contracts import GoldenEvalSet, SeedExample, load_golden_eval_set, select_examples
from app.evals.scoring import RetrievalScore, score_retrieval
from app.evals.workspace import build_eval_rig, create_eval_workspace
from app.services.generation_service import GenerationService, GenerationServiceError

DEFAULT_RETRIEVAL_LIMIT = 3


@dataclass(frozen=True)
class SeedExampleRunResult:
    example_id: str
    query: str
    expected_documents: tuple[str, ...]
    retrieval_result: RetrievalScore
    answer_evaluation: AnswerEvaluation | None = None


@dataclass(frozen=True)
class GoldenEvalRunReport:
    eval_set_path: Path
    workspace_path: Path
    example_results: tuple[SeedExampleRunResult, ...]
    hit_count: int
    example_count: int
    hit_rate: float
    mrr: float
    retrieval_passed: bool
    with_answer_eval: bool
    kept_artifacts: bool


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_eval_set_path(repo_root: Path) -> Path:
    return repo_root / "app/evals/golden_eval_set.json"


def ingest_eval_corpus(
    rig,
    eval_set: GoldenEvalSet,
    repo_root: Path,
) -> None:
    for corpus_document in eval_set.corpus_documents:
        rig.ingest(
            repo_root / corpus_document,
            original_filename=corpus_document,
        )


def ensure_generation_available(generation_service) -> None:
    generation_service.answer_question(
        "Confirm generation endpoint availability.",
        [
            {
                "original_filename": "preflight.txt",
                "chunk_index": 0,
                "score": 1.0,
                "text": "This is a preflight source for the Golden eval runner.",
            }
        ],
    )


def _evaluate_seed_example(
    rig,
    example: SeedExample,
    with_answer_eval: bool,
    generation_service,
) -> SeedExampleRunResult:
    contexts = rig.retrieve(example.query, DEFAULT_RETRIEVAL_LIMIT)
    retrieval_result = score_retrieval(
        expected_documents=example.expected_documents,
        contexts=contexts,
        top_k=DEFAULT_RETRIEVAL_LIMIT,
    )

    answer_evaluation = None
    if with_answer_eval:
        answer_evaluation = evaluate_answer_or_skip(
            retrieval_hit=retrieval_result.hit,
            generation_service=generation_service,
            question=example.query,
            contexts=contexts,
            expected_answer_facts=example.expected_answer_facts,
            require_citation=example.require_citation,
        )

    return SeedExampleRunResult(
        example_id=example.id,
        query=example.query,
        expected_documents=example.expected_documents,
        retrieval_result=retrieval_result,
        answer_evaluation=answer_evaluation,
    )


def run_golden_eval(
    *,
    eval_set_path: Path | None = None,
    repo_root: Path | None = None,
    example_ids: Sequence[str] | None = None,
    with_answer_eval: bool = False,
    keep_artifacts: bool = False,
    rig_factory=build_eval_rig,
    generation_service_factory: Callable[[], object] = GenerationService,
) -> GoldenEvalRunReport:
    resolved_repo_root = repo_root or default_repo_root()
    resolved_eval_set_path = eval_set_path or default_eval_set_path(resolved_repo_root)
    eval_set = load_golden_eval_set(resolved_eval_set_path, resolved_repo_root)
    examples = select_examples(eval_set, example_ids)

    with create_eval_workspace(keep_artifacts=keep_artifacts) as workspace:
        rig, engine = rig_factory(workspace)

        try:
            ingest_eval_corpus(rig, eval_set, resolved_repo_root)

            generation_service = None
            if with_answer_eval:
                generation_service = generation_service_factory()
                ensure_generation_available(generation_service)

            example_results = tuple(
                _evaluate_seed_example(
                    rig=rig,
                    example=example,
                    with_answer_eval=with_answer_eval,
                    generation_service=generation_service,
                )
                for example in examples
            )
            hit_count = sum(
                1 for example_result in example_results if example_result.retrieval_result.hit
            )
            example_count = len(example_results)
            hit_rate = 0.0 if example_count == 0 else hit_count / example_count
            mrr = (
                0.0
                if example_count == 0
                else sum(
                    example_result.retrieval_result.reciprocal_rank
                    for example_result in example_results
                )
                / example_count
            )

            return GoldenEvalRunReport(
                eval_set_path=resolved_eval_set_path,
                workspace_path=workspace.root_dir,
                example_results=example_results,
                hit_count=hit_count,
                example_count=example_count,
                hit_rate=hit_rate,
                mrr=mrr,
                retrieval_passed=hit_count == example_count,
                with_answer_eval=with_answer_eval,
                kept_artifacts=keep_artifacts,
            )
        finally:
            engine.dispose()


def print_golden_eval_report(report: GoldenEvalRunReport) -> None:
    status = "PASSED" if report.retrieval_passed else "FAILED"
    print(f"Golden eval set: {report.eval_set_path}")
    print(f"Seed examples: {report.example_count}")
    print(f"Retrieval gate: {status}")
    print(
        f"Hit@{DEFAULT_RETRIEVAL_LIMIT}: {report.hit_count}/{report.example_count} "
        f"({report.hit_rate:.2%})"
    )
    print(f"MRR: {report.mrr:.3f}")

    if report.kept_artifacts:
        print(f"Workspace: {report.workspace_path}")

    retrieval_failures = [
        example_result
        for example_result in report.example_results
        if not example_result.retrieval_result.hit
    ]
    if retrieval_failures:
        print("Retrieval failures:")
        for example_result in retrieval_failures:
            print(f"- {example_result.example_id}")
            print(
                "  expected: "
                + ", ".join(example_result.expected_documents)
            )
            print(
                "  retrieved: "
                + ", ".join(example_result.retrieval_result.retrieved_documents)
            )

    if report.with_answer_eval:
        print("Informational answer eval:")
        for example_result in report.example_results:
            answer_evaluation = example_result.answer_evaluation
            if answer_evaluation is None:
                continue

            if answer_evaluation.skipped:
                print(
                    f"- {example_result.example_id}: {answer_evaluation.skip_reason}"
                )
                continue

            covered_facts = sum(
                1 for fact_result in answer_evaluation.fact_results if fact_result.covered
            )
            print(
                f"- {example_result.example_id}: facts {covered_facts}/"
                f"{len(answer_evaluation.fact_results)}, inline_citation="
                f"{answer_evaluation.has_inline_citation}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Golden eval set")
    parser.add_argument("--eval-set", type=Path, default=None)
    parser.add_argument(
        "--example-id",
        action="append",
        default=None,
        help="Filter to one or more seed example ids",
    )
    parser.add_argument(
        "--with-answer-eval",
        action="store_true",
        help="Enable informational answer evaluation",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the isolated eval workspace for debugging",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = run_golden_eval(
            eval_set_path=args.eval_set,
            example_ids=args.example_id,
            with_answer_eval=args.with_answer_eval,
            keep_artifacts=args.keep_artifacts,
        )
    except (ValueError, GenerationServiceError) as exc:
        print(f"Golden eval runner failed: {exc}")
        return 2

    print_golden_eval_report(report)
    return 0 if report.retrieval_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
