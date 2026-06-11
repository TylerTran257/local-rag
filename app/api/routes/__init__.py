from app.api.routes.answer import router as answer_router
from app.api.routes.ask import router as ask_router
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.documents_v2 import router as documents_v2_router
from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.pages import router as pages_router
from app.api.routes.retrieve import router as retrieve_router
from app.api.routes.search import router as search_router
from app.api.routes.social_style import router as social_style_router
from app.api.routes.uploads import router as uploads_router

__all__ = [
    "answer_router",
    "ask_router",
    "chat_router",
    "documents_router",
    "documents_v2_router",
    "health_router",
    "ingest_router",
    "jobs_router",
    "pages_router",
    "retrieve_router",
    "search_router",
    "social_style_router",
    "uploads_router",
]
