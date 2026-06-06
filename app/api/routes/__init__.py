from app.api.routes.ask import router as ask_router
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.pages import router as pages_router
from app.api.routes.search import router as search_router
from app.api.routes.uploads import router as uploads_router

__all__ = [
    "ask_router",
    "chat_router",
    "documents_router",
    "health_router",
    "jobs_router",
    "pages_router",
    "search_router",
    "uploads_router",
]
