import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.schemas import AskRequest
from app.services.generation_service import GenerationServiceError

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

            citations = websocket.app.state.document_service.serialize_citations(contexts)
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
