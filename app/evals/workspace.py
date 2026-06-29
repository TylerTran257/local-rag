import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.composition import build_metadata_aware_runtime
from app.db.database import Base
from app.ingest.contracts import IngestDocument
from app.ingest.use_case import IngestUseCase
from app.profiles.store import ProfileStore
from app.retrieval.types import RetrievalMode, RetrievalScope, RetrieveRequest
from app.retrieval.use_case import RetrieveUseCase
from app.services.embedding_service import EmbeddingService
from app.services.lexical_search_service import LexicalSearchService
from app.services.text_extractor import TextExtractor
from app.services.vector_store_service import VectorStoreService

# Fixed scope the golden eval ingests and retrieves under. The eval corpus is a
# single service/tenant/collection, so retrieval exercises the same scoped path
# production uses.
EVAL_SERVICE = "eval"
EVAL_TENANT = "eval"
EVAL_COLLECTION = "eval"
EVAL_SOURCE_TYPE = "document"


@dataclass(frozen=True)
class EvalWorkspace:
    root_dir: Path
    database_path: Path
    upload_dir: Path
    qdrant_path: Path


class EvalRig:
    """Ingest + retrieve over the production metadata-aware seam for evals.

    Wraps the real ``IngestUseCase`` and ``RetrieveUseCase`` under a fixed eval
    scope, so the golden eval scores the same retrieval path callers use. The
    returned contexts carry ``original_filename`` (the chunk's source label) so
    document-level scoring lines up with the corpus filenames.
    """

    def __init__(
        self,
        ingest_use_case: IngestUseCase,
        retrieve_use_case: RetrieveUseCase,
        text_extractor: TextExtractor,
    ) -> None:
        self._ingest = ingest_use_case
        self._retrieve = retrieve_use_case
        self._text_extractor = text_extractor

    def ingest(self, file_path: Path, original_filename: str) -> None:
        text = self._text_extractor.extract(file_path)
        self._ingest.ingest_document(
            IngestDocument(
                text=text,
                service_name=EVAL_SERVICE,
                tenant_id=EVAL_TENANT,
                collection=EVAL_COLLECTION,
                source_type=EVAL_SOURCE_TYPE,
                source_label=original_filename,
            )
        )

    def retrieve(self, query: str, limit: int) -> list[dict]:
        result = self._retrieve.execute(
            RetrieveRequest(
                query=query,
                retrieval_mode=RetrievalMode.HYBRID,
                limit=limit,
                scope=RetrievalScope(
                    service_name=EVAL_SERVICE,
                    tenant_id=EVAL_TENANT,
                    collections=[EVAL_COLLECTION],
                ),
            )
        )
        return [
            {
                "document_id": chunk.document_id,
                "original_filename": chunk.metadata.get("source_label", "unknown"),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "score": chunk.score,
                "text": chunk.content,
            }
            for chunk in result.chunks
        ]


def build_eval_rig(workspace: EvalWorkspace):
    engine = create_engine(
        f"sqlite:///{workspace.database_path}",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    runtime = build_metadata_aware_runtime(
        embedding_service=EmbeddingService(),
        vector_store_service=VectorStoreService(qdrant_path=workspace.qdrant_path),
        lexical_search_service=LexicalSearchService(session_factory=session_factory),
        profile_store=ProfileStore(session_factory=session_factory),
    )

    rig = EvalRig(
        ingest_use_case=runtime.ingest_use_case,
        retrieve_use_case=runtime.retrieve_use_case,
        text_extractor=TextExtractor(),
    )
    return rig, engine


@contextmanager
def create_eval_workspace(keep_artifacts: bool = False):
    root_dir = Path(tempfile.mkdtemp(prefix="golden-eval-"))
    workspace = EvalWorkspace(
        root_dir=root_dir,
        database_path=root_dir / "app.db",
        upload_dir=root_dir / "uploads",
        qdrant_path=root_dir / "qdrant_data",
    )
    workspace.upload_dir.mkdir(parents=True, exist_ok=True)
    workspace.qdrant_path.mkdir(parents=True, exist_ok=True)

    try:
        yield workspace
    finally:
        if not keep_artifacts:
            shutil.rmtree(root_dir, ignore_errors=True)
