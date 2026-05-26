import logging
from time import perf_counter
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.routing import APIRouter
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from app.api.schemas import AskRequest, SearchRequest
from app.core.logging import configure_logging
from app.db.database import Base, engine
from app.services.document_service import DocumentData, DocumentService, JobData
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService, GenerationServiceError
from app.services.lexical_search_service import LexicalSearchService
from app.services.text_extractor import TextExtractor
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")


def create_app(document_service=None, generation_service=None) -> FastAPI:
    configure_logging()
    logger.info("event=app_started")
    app = FastAPI()

    @app.middleware("http")
    async def log_request_timing(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        started_at = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%s",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "event=http_request_completed request_id=%s method=%s path=%s status_code=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    Base.metadata.create_all(bind=engine)

    resolved_document_service = document_service
    if document_service is None:
        embedding_service = EmbeddingService()
        vector_store_service = VectorStoreService()
        text_extractor = TextExtractor()
        lexical_search_service = LexicalSearchService()
        resolved_document_service = DocumentService(
            embedding_service,
            vector_store_service,
            text_extractor,
            lexical_search_service,
        )

    resolved_generation_service = generation_service
    if generation_service is None:
        resolved_generation_service = GenerationService()
    app.state.document_service = resolved_document_service
    app.state.generation_service = resolved_generation_service

    app.include_router(router)
    return app


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/upload_v1")
async def upload_document_v1(
    request: Request, file: UploadFile = File(...)
) -> DocumentData:
    return await request.app.state.document_service.create_document(file)


@router.get("/chat")
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})


@router.get("/{document_id}")
def get_document(request: Request, document_id: str) -> DocumentData:
    return request.app.state.document_service.get_document(document_id)


@router.post("/{document_id}/extract")
def extract_document(request: Request, document_id: str) -> DocumentData:
    return request.app.state.document_service.extract_text(document_id)


@router.post("/{document_id}/chunk")
def chunk_document(request: Request, document_id: str) -> dict[str, str | int]:
    return request.app.state.document_service.chunk_document(document_id)


@router.post("/{document_id}/search")
def search_document(
    request: Request, document_id: str, searchRequest: SearchRequest
) -> dict:
    return request.app.state.document_service.search_document(
        document_id, searchRequest.query, searchRequest.limit
    )


@router.post("/{document_id}/embed")
def embed_document(request: Request, document_id: str) -> dict:
    return request.app.state.document_service.embed_document(document_id)


@router.post("/upload_v2")
async def upload_document_v2(
    request: Request, file: UploadFile = File(...)
) -> DocumentData:
    uploaded_file = await request.app.state.document_service.create_document(file)
    document_id = uploaded_file.get("document_id")
    if document_id is None:
        raise HTTPException(status_code=500, detail="Document Id Not Found")

    request.app.state.document_service.extract_text(document_id)
    request.app.state.document_service.chunk_document(document_id)
    request.app.state.document_service.embed_document(document_id)

    return request.app.state.document_service.get_document(document_id)


@router.post("/semantic-search")
def semantic_search_document(request: Request, searchRequest: SearchRequest) -> dict:
    return request.app.state.document_service.semantic_search(
        searchRequest.query, searchRequest.limit
    )


@router.post("/hybrid-search")
def hybrid_search_document(request: Request, searchRequest: SearchRequest) -> dict:
    return request.app.state.document_service.hybrid_search(
        searchRequest.query, searchRequest.limit
    )


@router.post("/ask")
def ask(request: Request, askRequest: AskRequest) -> dict:
    contexts = request.app.state.document_service.retrieve_context(
        askRequest.query, askRequest.limit
    )
    if len(contexts) == 0:
        return {
            "query": askRequest.query,
            "answer": "",
            "match_count": 0,
            "sources": [],
            "citations": [],
        }
    try:
        answer = request.app.state.generation_service.answer_question(
            askRequest.query, contexts
        )
    except GenerationServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    citations = request.app.state.document_service.serialize_citations(contexts)

    return {
        "query": askRequest.query,
        "answer": answer if len(answer) != 0 else "",
        "match_count": len(contexts),
        "sources": contexts,
        "citations": citations,
    }


@router.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket):
    chat_id = str(uuid4())
    logger.info("event=websocket_connected chat_id=%s", chat_id)
    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()

            try:
                ask_request = AskRequest.model_validate(payload)
                turn_started_at = perf_counter()
                logger.info(
                    "event=chat_turn_started chat_id=%s query_length=%s",
                    chat_id,
                    len(ask_request.query),
                )
            except ValidationError:
                logger.error(
                    "event=chat_payload_invalid chat_id=%s",
                    chat_id,
                )
                await websocket.send_json(
                    {"type": "error", "message": "Invalid chat payload"}
                )
                continue

            await websocket.send_json(
                {"type": "status", "message": "retrieving context"}
            )

            try:
                contexts = websocket.app.state.document_service.retrieve_context(
                    ask_request.query,
                    ask_request.limit,
                )

            except HTTPException as exc:
                message = (
                    exc.detail
                    if isinstance(exc.detail, str)
                    else "Failed to retrieve document context"
                )
                await websocket.send_json({"type": "error", "message": message})
                logger.error(
                    "event=chat_retrieval_failed chat_id=%s error_message=%s",
                    chat_id,
                    message,
                )
                continue
            except Exception:
                await websocket.send_json(
                    {"type": "error", "message": "Failed to retrieve document context"}
                )
                logger.exception(
                    "event=chat_retrieval_failed chat_id=%s",
                    chat_id,
                )
                continue

            if not contexts:
                await websocket.send_json(
                    {"type": "done", "answer": "", "sources": [], "citations": []}
                )
                completed_duration_ms = round(
                    (perf_counter() - turn_started_at) * 1000, 2
                )
                logger.info(
                    "event=chat_turn_completed chat_id=%s query_length=%s source_count=0 duration_ms=%s",
                    chat_id,
                    len(ask_request.query),
                    completed_duration_ms,
                )
                continue
            await websocket.send_json(
                {
                    "type": "status",
                    "message": "generating answer",
                }
            )

            full_answer = ""
            try:
                async for (
                    token
                ) in websocket.app.state.generation_service.stream_answer_question(
                    ask_request.query,
                    contexts,
                ):
                    full_answer += token
                    await websocket.send_json({"type": "token", "value": token})
            except GenerationServiceError as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                logger.error(
                    "event=chat_generation_failed chat_id=%s error_message=%s",
                    chat_id,
                    str(exc),
                )
                continue

            citations = websocket.app.state.document_service.serialize_citations(
                contexts
            )
            await websocket.send_json(
                {
                    "type": "done",
                    "answer": full_answer,
                    "sources": contexts,
                    "citations": citations,
                }
            )
            completed_duration_ms = round((perf_counter() - turn_started_at) * 1000, 2)
            logger.info(
                "event=chat_turn_completed chat_id=%s query_length=%s source_count=%s duration_ms=%s",
                chat_id,
                len(ask_request.query),
                len(contexts),
                completed_duration_ms,
            )
    except WebSocketDisconnect:
        logger.info(
            "event=websocket_disconnected chat_id=%s",
            chat_id,
        )
        pass


@router.post("/upload_async", status_code=202)
async def upload_async(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> JobData:
    document = await request.app.state.document_service.create_document(file)
    job = request.app.state.document_service.create_job(document["document_id"])
    background_tasks.add_task(
        request.app.state.document_service.run_indexing_pipeline,
        document["document_id"],
        job["job_id"],
    )

    return job


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> JobData:
    job = request.app.state.document_service.get_job(job_id)

    return job
