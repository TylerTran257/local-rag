import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.lexical_search_service import LexicalSearchService
from app.services.text_extractor import TextExtractor
from app.services.vector_store_service import VectorStoreService


@dataclass(frozen=True)
class EvalWorkspace:
    root_dir: Path
    database_path: Path
    upload_dir: Path
    qdrant_path: Path


def build_eval_document_service(workspace: EvalWorkspace):
    engine = create_engine(
        f"sqlite:///{workspace.database_path}",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    lexical_search_service = LexicalSearchService(session_factory=session_factory)
    vector_store_service = VectorStoreService(qdrant_path=workspace.qdrant_path)
    document_service = DocumentService(
        embedding_service=EmbeddingService(),
        vector_store_service=vector_store_service,
        text_extractor=TextExtractor(),
        lexical_search_service=lexical_search_service,
        session_factory=session_factory,
        upload_dir=workspace.upload_dir,
    )
    return document_service, engine


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
