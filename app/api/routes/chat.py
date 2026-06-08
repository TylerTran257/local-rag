import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.retrieval_helpers import build_default_scope
from app.api.schemas import AskRequest
from app.services.generation_service import GenerationServiceError
from app.retrieval import (
    RetrieveRequest,
    RetrievalMode,
    RetrievalError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


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
                retrieve_request = RetrieveRequest(
                    query=ask_request.query,
                    retrieval_mode=RetrievalMode.DENSE,
                    limit=ask_request.limit,
                    scope=build_default_scope(),
                )
                result = websocket.app.state.retrieve_use_case.execute(retrieve_request)

            except RetrievalError as exc:
                message = str(exc)
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

            contexts = [
                {
                    "document_id": chunk.metadata["document_id"],
                    "original_filename": chunk.metadata["source_label"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    "score": chunk.score,
                    "text": chunk.content,
                }
                for chunk in result.chunks
            ]

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
                async for token in websocket.app.state.generation_service.stream_answer_question(
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

            citations = [
                {
                    "id": i,
                    "document_id": ctx["document_id"],
                    "original_filename": ctx["original_filename"],
                    "chunk_index": ctx["chunk_index"],
                    "score": ctx["score"],
                    "text": ctx["text"],
                }
                for i, ctx in enumerate(contexts, start=1)
            ]
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
