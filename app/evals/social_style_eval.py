import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.evals.social_style_contracts import (
    SocialStyleCorpusChunk,
    SocialStyleEvalExample,
    SocialStyleEvalSet,
    load_social_style_eval_set,
    select_examples,
)
from app.evals.social_style_scoring import SocialStyleRetrievalScore, score_style_retrieval
from app.evals.workspace import EvalWorkspace, create_eval_workspace
from app.ingest import IngestChunk
from app.ingest.use_case import IngestUseCase
from app.retrieval import NoOpTraceSink, PassthroughScopePolicy, SystemClock, UuidTraceIdGenerator
from app.retrieval.metadata_gateway import MetadataAwareRetrievalGateway
from app.retrieval.use_case import RetrieveUseCase
from app.services.embedding_service import EmbeddingService
from app.services.lexical_search_service import LexicalSearchService
from app.services.vector_store_service import VectorStoreService
from app.social import SocialStyleRetrievalService, StyleCategory, StyleContext, StyleRetrievalRequest

DEFAULT_TOP_K = 3
EVAL_SERVICE_NAME = "social-style-eval"
EVAL_COLLECTION_NAME = "style_memory_eval"
EVAL_TENANT_PREFIX = "eval-tenant"


class IngestUseCaseLike(Protocol):
    def ingest_chunks(self, chunks: list[IngestChunk]) -> object: ...


class SocialStyleServiceLike(Protocol):
    def retrieve(self, request: StyleRetrievalRequest) -> StyleContext: ...


@dataclass(frozen=True)
class SocialStyleEvalRuntime:
    ingest_use_case: IngestUseCaseLike
    social_service: SocialStyleServiceLike
    engine: object


@dataclass(frozen=True)
class SocialStyleExampleRunResult:
    example_id: str
    query: str
    tenant_id: str
    platform: str | None
    target_categories: tuple[StyleCategory, ...]
    retrieval_result: SocialStyleRetrievalScore


@dataclass(frozen=True)
class SocialStyleEvalRunReport:
    eval_set_path: Path
    workspace_path: Path
    example_results: tuple[SocialStyleExampleRunResult, ...]
    example_count: int
    hit_rate_at_k: float
    recall_at_k: float
    category_coverage: float
    missing_required_category_count: int
    unexpected_category_count: int
    retrieval_passed: bool
    kept_artifacts: bool


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_eval_set_path(repo_root: Path) -> Path:
    return repo_root / "app/evals/social_style_eval_set.json"


def make_eval_tenant_id(tenant_id: str) -> str:
    return f"{EVAL_TENANT_PREFIX}:{tenant_id}"


def build_social_style_eval_runtime(workspace: EvalWorkspace) -> SocialStyleEvalRuntime:
    engine = create_engine(
        f"sqlite:///{workspace.database_path}",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    embedding_service = EmbeddingService()
    vector_store_service = VectorStoreService(qdrant_path=workspace.qdrant_path)
    lexical_search_service = LexicalSearchService(session_factory=session_factory)
    ingest_use_case = IngestUseCase(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        lexical_search_service=lexical_search_service,
    )
    retrieve_use_case = RetrieveUseCase(
        gateway=MetadataAwareRetrievalGateway(
            vector_store_service=vector_store_service,
            lexical_search_service=lexical_search_service,
            embedding_service=embedding_service,
        ),
        scope_policy=PassthroughScopePolicy(),
        clock=SystemClock(),
        trace_id_generator=UuidTraceIdGenerator(),
        trace_sink=NoOpTraceSink(),
    )
    social_service = SocialStyleRetrievalService(
        retrieve_use_case=retrieve_use_case,
        service_name=EVAL_SERVICE_NAME,
    )
    return SocialStyleEvalRuntime(
        ingest_use_case=ingest_use_case,
        social_service=social_service,
        engine=engine,
    )


def _chunk_to_ingest_request(chunk: SocialStyleCorpusChunk) -> IngestChunk:
    domain_metadata = {
        "style_category": chunk.style_category.value,
        "eval_chunk_id": chunk.chunk_id,
    }
    if chunk.platform is not None:
        domain_metadata["platform"] = chunk.platform

    return IngestChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        service_name=EVAL_SERVICE_NAME,
        tenant_id=make_eval_tenant_id(chunk.tenant_id),
        collection=EVAL_COLLECTION_NAME,
        source_type="style_memory",
        source_label=chunk.source_label,
        domain_metadata=domain_metadata,
    )


def ingest_social_style_eval_corpus(
    ingest_use_case: IngestUseCaseLike,
    eval_set: SocialStyleEvalSet,
) -> None:
    for chunk in eval_set.corpus_chunks:
        ingest_use_case.ingest_chunks([_chunk_to_ingest_request(chunk)])


def _evaluate_social_style_example(
    social_service: SocialStyleServiceLike,
    example: SocialStyleEvalExample,
    top_k: int,
    source_label_categories: dict[str, StyleCategory],
    chunk_id_categories: dict[str, StyleCategory],
) -> SocialStyleExampleRunResult:
    context = social_service.retrieve(
        StyleRetrievalRequest(
            tenant_id=make_eval_tenant_id(example.tenant_id),
            query=example.query,
            style_categories=list(example.target_categories),
            platform=example.platform,
            per_category_limit=top_k,
            collection=EVAL_COLLECTION_NAME,
        )
    )
    retrieval_result = score_style_retrieval(
        example=example,
        context=context,
        top_k=top_k,
        source_label_categories=source_label_categories,
        chunk_id_categories=chunk_id_categories,
    )
    return SocialStyleExampleRunResult(
        example_id=example.id,
        query=example.query,
        tenant_id=example.tenant_id,
        platform=example.platform,
        target_categories=example.target_categories,
        retrieval_result=retrieval_result,
    )


def run_social_style_eval(
    *,
    eval_set_path: Path | None = None,
    repo_root: Path | None = None,
    example_ids: Sequence[str] | None = None,
    keep_artifacts: bool = False,
    top_k: int = DEFAULT_TOP_K,
    runtime_factory=build_social_style_eval_runtime,
) -> SocialStyleEvalRunReport:
    resolved_repo_root = repo_root or default_repo_root()
    resolved_eval_set_path = eval_set_path or default_eval_set_path(resolved_repo_root)
    eval_set = load_social_style_eval_set(resolved_eval_set_path)
    examples = select_examples(eval_set, example_ids)
    source_label_categories = {
        chunk.source_label: chunk.style_category for chunk in eval_set.corpus_chunks
    }
    chunk_id_categories = {
        chunk.chunk_id: chunk.style_category for chunk in eval_set.corpus_chunks
    }

    with create_eval_workspace(keep_artifacts=keep_artifacts) as workspace:
        runtime = runtime_factory(workspace)
        try:
            ingest_social_style_eval_corpus(runtime.ingest_use_case, eval_set)
            example_results = tuple(
                _evaluate_social_style_example(
                    social_service=runtime.social_service,
                    example=example,
                    top_k=top_k,
                    source_label_categories=source_label_categories,
                    chunk_id_categories=chunk_id_categories,
                )
                for example in examples
            )
            example_count = len(example_results)
            hit_count = sum(
                1 for example_result in example_results if example_result.retrieval_result.hit
            )
            total_expected_count = sum(
                example_result.retrieval_result.total_expected_count
                for example_result in example_results
            )
            matched_expected_count = sum(
                example_result.retrieval_result.matched_expected_count
                for example_result in example_results
            )
            total_requested_categories = sum(
                example_result.retrieval_result.requested_category_count
                for example_result in example_results
            )
            covered_categories = sum(
                example_result.retrieval_result.covered_category_count
                for example_result in example_results
            )
            missing_required_category_count = sum(
                example_result.retrieval_result.missing_required_category_count
                for example_result in example_results
            )
            unexpected_category_count = sum(
                example_result.retrieval_result.unexpected_category_count
                for example_result in example_results
            )

            hit_rate_at_k = 0.0 if example_count == 0 else hit_count / example_count
            recall_at_k = (
                0.0
                if total_expected_count == 0
                else matched_expected_count / total_expected_count
            )
            category_coverage = (
                0.0
                if total_requested_categories == 0
                else covered_categories / total_requested_categories
            )

            return SocialStyleEvalRunReport(
                eval_set_path=resolved_eval_set_path,
                workspace_path=workspace.root_dir,
                example_results=example_results,
                example_count=example_count,
                hit_rate_at_k=hit_rate_at_k,
                recall_at_k=recall_at_k,
                category_coverage=category_coverage,
                missing_required_category_count=missing_required_category_count,
                unexpected_category_count=unexpected_category_count,
                retrieval_passed=(
                    hit_count == example_count
                    and missing_required_category_count == 0
                    and unexpected_category_count == 0
                    and (total_expected_count == 0 or matched_expected_count == total_expected_count)
                ),
                kept_artifacts=keep_artifacts,
            )
        finally:
            runtime.engine.dispose()


def print_social_style_eval_report(report: SocialStyleEvalRunReport, top_k: int = DEFAULT_TOP_K) -> None:
    status = "PASSED" if report.retrieval_passed else "FAILED"
    print(f"Social style eval set: {report.eval_set_path}")
    print(f"Examples: {report.example_count}")
    print(f"Retrieval gate: {status}")
    print(f"Hit@{top_k}: {report.hit_rate_at_k:.2%}")
    print(f"Recall@{top_k}: {report.recall_at_k:.2%}")
    print(f"Category coverage: {report.category_coverage:.2%}")
    print(f"Missing required categories: {report.missing_required_category_count}")
    print(f"Unexpected categories: {report.unexpected_category_count}")

    if report.kept_artifacts:
        print(f"Workspace: {report.workspace_path}")

    failures = [
        example_result
        for example_result in report.example_results
        if not example_result.retrieval_result.hit
    ]
    if failures:
        print("Retrieval failures:")
        for example_result in failures:
            print(f"- {example_result.example_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the social style retrieval eval set")
    parser.add_argument("--eval-set", type=Path, default=None)
    parser.add_argument(
        "--example-id",
        action="append",
        default=None,
        help="Filter to one or more social style eval example ids",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the isolated eval workspace for debugging",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Per-category retrieval limit and scoring cutoff",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = run_social_style_eval(
            eval_set_path=args.eval_set,
            example_ids=args.example_id,
            keep_artifacts=args.keep_artifacts,
            top_k=args.top_k,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Social style eval runner failed: {exc}")
        return 2

    print_social_style_eval_report(report, top_k=args.top_k)
    return 0 if report.retrieval_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
