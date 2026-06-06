import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict

from app.db.database import SessionLocal
from app.db.models import Document, DocumentChunk, Job
from app.services.embedding_service import EmbeddingService
from app.services.lexical_search_service import LexicalSearchService
from app.services.text_extractor import TextExtractor
from app.services.vector_store_service import VectorStoreService
from app.settings import settings

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

ALLOWED_CONTENT_TYPES = {"text/plain", "application/pdf"}


class DocumentData(TypedDict, total=False):
    document_id: str
    original_filename: str
    stored_filename: str
    status: str
    size_bytes: int
    extracted_text: str | None
    chunks: list[str]
    chunk_embeddings: list[list[float]]


class JobData(TypedDict):
    job_id: str
    job_type: str
    document_id: str
    status: str
    error_message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class DocumentService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        text_extractor: TextExtractor,
        lexical_search_service: LexicalSearchService,
        session_factory=None,
        upload_dir: Path | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.text_extractor = text_extractor
        self.lexical_search_service = lexical_search_service
        self.session_factory = session_factory or SessionLocal
        self.upload_dir = upload_dir or settings.upload_dir

    def _validate_content_type(self, content_type: str) -> None:
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=400, detail=f"{content_type} is not allowed")

    def _content_type_for_path(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return "text/plain"

        if suffix == ".pdf":
            return "application/pdf"

        raise ValueError(f"Unsupported file type: {suffix}")

    def _persist_document_contents(
        self,
        original_filename: str,
        contents: bytes,
    ) -> DocumentData:
        if len(contents) > settings.max_file_size:
            raise HTTPException(
                status_code=400, detail="Document is too large. Max size is 1 MB"
            )

        self.upload_dir.mkdir(parents=True, exist_ok=True)

        document_id = str(uuid4())
        original_suffix = Path(original_filename).suffix.lower()
        saved_filename = f"{document_id}{original_suffix}"
        saved_path = self.upload_dir / saved_filename
        saved_path.write_bytes(contents)

        with self.session_factory() as session:
            document = Document(
                id=document_id,
                original_filename=original_filename,
                stored_filename=saved_filename,
                status="uploaded",
                size_bytes=len(contents),
                extracted_text=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return self.serialize_document(document)

    def _validate_query(self, query: str, limit: int) -> None:
        if not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query must not be empty",
            )

        if limit <= 0:
            raise HTTPException(
                status_code=400,
                detail="Limit must be greater than 0",
            )

    def _fuse_rankings_rrf(
        self, dense_results: list[dict], lexical_results: list[dict], limit: int
    ) -> list[dict]:
        if limit <= 0:
            return []

        fused_by_key: dict[tuple[str, int], dict] = {}
        rrf_k = settings.fusion_rrf_k

        def add_results(results: list[dict]) -> None:
            for rank, result in enumerate(results, start=1):
                key = (result["document_id"], result["chunk_index"])
                rrf_score = 1 / (rrf_k + rank)

                existing = fused_by_key.get(key)
                if existing is None:
                    fused_by_key[key] = {
                        "document_id": result["document_id"],
                        "original_filename": result["original_filename"],
                        "chunk_index": result["chunk_index"],
                        "text": result["text"],
                        "score": rrf_score,
                    }
                    continue

                existing["score"] += rrf_score

        add_results(dense_results)
        add_results(lexical_results)

        fused_results = list(fused_by_key.values())
        fused_results.sort(
            key=lambda item: (-item["score"], item["document_id"], item["chunk_index"])
        )

        return fused_results[:limit]

    async def create_document(self, file: UploadFile) -> DocumentData:
        started_at = perf_counter()
        if not file.filename:
            raise HTTPException(status_code=400, detail="Documentname is required")

        self._validate_content_type(file.content_type or "")

        contents = await file.read()

        result = self._persist_document_contents(file.filename or "unknown", contents)

        document_id = result.get("document_id")
        if document_id is None:
            raise RuntimeError("Document creation did not return a document_id")

        logger.info(
            "event=document_created document_id=%s filename=%s size_bytes=%s duration_ms=%s",
            document_id,
            file.filename or "unknown",
            len(contents),
            round((perf_counter() - started_at) * 1000, 2),
        )
        return result

    def create_document_from_path(
        self,
        file_path: Path,
        original_filename: str | None = None,
    ) -> DocumentData:
        started_at = perf_counter()
        self._content_type_for_path(file_path)
        contents = file_path.read_bytes()
        resolved_filename = original_filename or file_path.name
        result = self._persist_document_contents(resolved_filename, contents)

        document_id = result.get("document_id")
        if document_id is None:
            raise RuntimeError("Document creation did not return a document_id")

        logger.info(
            "event=document_created_from_path document_id=%s filename=%s size_bytes=%s duration_ms=%s",
            document_id,
            resolved_filename,
            len(contents),
            round((perf_counter() - started_at) * 1000, 2),
        )
        return result

    def get_document(self, document_id: str) -> DocumentData:
        with self.session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")
            return self.serialize_document(document)

    def search_document(
        self, document_id: str, query: str, limit: int
    ) -> dict[str, str | int | list[str]]:
        with self.session_factory() as session:
            document = session.get(Document, document_id)

            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

            if not query.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Query must not be empty",
                )

            if limit <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Limit must be greater than 0",
                )

            chunks = [chunk.text for chunk in document.chunks]
            normalized_query = query.lower()
            results = []

            for index, chunk in enumerate(chunks):
                score = chunk.lower().count(normalized_query)
                if score > 0:
                    results.append(
                        {
                            "chunk_index": index,
                            "score": score,
                            "text": chunk,
                        }
                    )

            results.sort(key=lambda item: item["score"], reverse=True)
            top_results = results[:limit]

            return {
                "document_id": document_id,
                "query": query,
                "match_count": len(top_results),
                "results": top_results,
            }

    def extract_text(self, document_id: str) -> DocumentData:
        started_at = perf_counter()
        with self.session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

            if document.status == "text_extracted":
                return self.serialize_document(document)

            if document.status != "uploaded":
                raise HTTPException(status_code=400, detail="Document state error")

            stored_filename = document.stored_filename
            saved_path = self.upload_dir / str(stored_filename)

            document.extracted_text = self.text_extractor.extract(saved_path)
            document.status = "text_extracted"
            document.updated_at = datetime.now()

            session.commit()
            session.refresh(document)
            result = self.serialize_document(document)

        logger.info(
            "event=text_extracted document_id=%s text_length=%s duration_ms=%s",
            document_id,
            len(result.get("extracted_text") or ""),
            round((perf_counter() - started_at) * 1000, 2),
        )
        return result

    def chunk_document(self, document_id: str) -> dict[str, str | int]:
        started_at = perf_counter()
        with self.session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

            if document.status == "chunked":
                chunks = document.chunks
                return {
                    "document_id": document_id,
                    "status": "chunked",
                    "chunk_count": len(chunks or []),
                }

            if document.status != "text_extracted":
                raise HTTPException(
                    status_code=409,
                    detail="Document must be text_extracted before chunking",
                )

            extracted_text = document.extracted_text
            if extracted_text is None:
                raise HTTPException(
                    status_code=409,
                    detail="Document must be text_extracted before chunking",
                )

            document.chunks.clear()

            chunks = text_splitter.split_text(extracted_text)
            for index, chunk_text in enumerate(chunks):
                document.chunks.append(
                    DocumentChunk(
                        chunk_index=index,
                        text=chunk_text,
                        created_at=datetime.now(),
                    )
                )

            document.status = "chunked"
            document.updated_at = datetime.now()
            session.commit()

            self.lexical_search_service.index_document_chunks(
                document_id, document.original_filename, document.chunks
            )

            result = {
                "document_id": document_id,
                "status": "chunked",
                "chunk_count": len(chunks),
            }

        logger.info(
            "event=document_chunked document_id=%s chunk_count=%s duration_ms=%s",
            document_id,
            result["chunk_count"],
            round((perf_counter() - started_at) * 1000, 2),
        )
        return result

    def embed_document(self, document_id: str) -> dict[str, str | int | list[str]]:
        started_at = perf_counter()
        with self.session_factory() as session:
            document = session.get(Document, document_id)

            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

            if document.status == "embedded":
                return {
                    "document_id": document.id or "",
                    "status": document.status or "",
                    "embedding_count": len(document.chunks),
                }

            if document.status != "chunked":
                raise HTTPException(
                    status_code=409,
                    detail="Document must be chunked before embedding",
                )

            chunks = [chunk.text for chunk in document.chunks]
            chunk_embeddings = self.embedding_service.embed_texts(chunks)

            self.vector_store_service.upsert_document_chunks(
                document_id,
                document.original_filename,
                chunks,
                chunk_embeddings,
            )

            document.status = "embedded"
            document.updated_at = datetime.now()
            session.commit()

            result = {
                "document_id": document.id,
                "status": "embedded",
                "embedding_count": len(chunk_embeddings),
            }

        logger.info(
            "event=document_embedded document_id=%s embedding_count=%s duration_ms=%s",
            document_id,
            result["embedding_count"],
            round((perf_counter() - started_at) * 1000, 2),
        )
        return result

    def retrieve_context(self, query: str, limit: int) -> list[dict]:
        return self.retrieve_context_dense(query, limit)

    def retrieve_context_dense(self, query: str, limit: int) -> list[dict]:
        started_at = perf_counter()
        self._validate_query(query, limit)

        if not self.vector_store_service.has_indexed_chunks():
            raise HTTPException(
                status_code=409,
                detail="At least one document must be embedded before semantic searching",
            )

        query_embedding = self.embedding_service.embed_text(query)

        results = self.vector_store_service.search(query_embedding, limit)
        logger.info(
            "event=retrieval_completed mode=dense query_length=%s requested_limit=%s result_count=%s duration_ms=%s",
            len(query),
            limit,
            len(results),
            round((perf_counter() - started_at) * 1000, 2),
        )
        return results

    def retrieve_context_lexical(self, query: str, limit: int) -> list[dict]:
        self._validate_query(query, limit)

        return self.lexical_search_service.search(query, limit)

    def retrieve_context_hybrid(self, query: str, limit: int) -> list[dict]:
        started_at = perf_counter()
        self._validate_query(query, limit)

        try:
            dense_results = self.retrieve_context_dense(
                query, settings.dense_retrieval_limit
            )
        except HTTPException as exc:
            if exc.status_code == 409:
                dense_results = []
            else:
                raise

        lexical_results = self.retrieve_context_lexical(
            query, settings.lexical_retrieval_limit
        )

        results = self._fuse_rankings_rrf(dense_results, lexical_results, limit)
        logger.info(
            "event=retrieval_completed mode=hybrid query_length=%s requested_limit=%s dense_candidate_count=%s lexical_candidate_count=%s result_count=%s duration_ms=%s",
            len(query),
            limit,
            len(dense_results),
            len(lexical_results),
            len(results),
            round((perf_counter() - started_at) * 1000, 2),
        )
        return results

    def semantic_search(
        self, query: str, limit: int
    ) -> dict[str, str | int | list[str] | list[dict[str, str | int]]]:
        results = self.retrieve_context(query, limit)

        return {"query": query, "match_count": len(results), "results": results}

    def hybrid_search(
        self, query: str, limit: int
    ) -> dict[str, str | int | list[str] | list[dict[str, str | int]]]:
        results = self.retrieve_context_hybrid(query, limit)

        return {"query": query, "match_count": len(results), "results": results}

    def serialize_document(self, document: Document) -> DocumentData:
        chunks = [chunk.text for chunk in document.chunks]

        return {
            "document_id": document.id,
            "original_filename": document.original_filename,
            "stored_filename": document.stored_filename,
            "status": document.status,
            "size_bytes": document.size_bytes,
            "extracted_text": document.extracted_text,
            "chunks": chunks,
        }

    def serialize_job(self, job: Job) -> JobData:
        return {
            "job_id": job.id,
            "job_type": job.job_type,
            "document_id": job.document_id,
            "status": job.status,
            "error_message": job.error_message,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        }

    def serialize_citations(self, contexts: list[dict]) -> list[dict]:
        citations = []
        for id, context in enumerate(contexts, start=1):
            citations.append(
                {
                    "id": id,
                    "document_id": context["document_id"],
                    "original_filename": context["original_filename"],
                    "chunk_index": context["chunk_index"],
                    "score": context["score"],
                    "text": context["text"],
                }
            )

        return citations

    def create_job(self, document_id: str) -> JobData:
        with self.session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

            job = Job(
                id=str(uuid4()),
                job_type="document_index",
                document_id=document_id,
                status="queued",
                error_message=None,
                created_at=datetime.now(),
                started_at=None,
                finished_at=None,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return self.serialize_job(job)

    def get_job(self, job_id: str) -> JobData:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")

            return self.serialize_job(job)

    def mark_job_running(self, job_id: str) -> JobData:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")

            job.status = "running"
            job.started_at = datetime.now()

            session.commit()
            session.refresh(job)
            return self.serialize_job(job)

    def mark_job_completed(self, job_id: str) -> JobData:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")

            job.status = "completed"
            job.error_message = None
            job.finished_at = datetime.now()

            session.commit()
            session.refresh(job)
            return self.serialize_job(job)

    def mark_job_failed(self, job_id: str, message: str) -> JobData:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")

            job.status = "failed"
            job.error_message = message
            job.finished_at = datetime.now() if not job.finished_at else job.finished_at

            session.commit()
            session.refresh(job)
            return self.serialize_job(job)

    def run_indexing_pipeline(self, document_id: str, job_id: str) -> None:
        started_at = perf_counter()
        try:
            logger.info(
                "event=index_job_started job_id=%s document_id=%s",
                job_id,
                document_id,
            )
            self.mark_job_running(job_id)
            self.extract_text(document_id)
            self.chunk_document(document_id)
            self.embed_document(document_id)
            self.mark_job_completed(job_id)
            logger.info(
                "event=index_job_completed job_id=%s document_id=%s duration_ms=%s",
                job_id,
                document_id,
                round((perf_counter() - started_at) * 1000, 2),
            )
        except Exception as exc:
            self.mark_job_failed(job_id, str(exc))
            logger.exception(
                "event=index_job_failed job_id=%s document_id=%s duration_ms=%s",
                job_id,
                document_id,
                round((perf_counter() - started_at) * 1000, 2),
            )
