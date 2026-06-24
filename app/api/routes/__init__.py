from app.api.routes.answer import router as answer_router
from app.api.routes.delete import router as delete_router
from app.api.routes.documents_v2 import router as documents_v2_router
from app.api.routes.health import router as health_router
from app.api.routes.profiles import router as profiles_router
from app.api.routes.retrieve import router as retrieve_router

__all__ = [
    "answer_router",
    "delete_router",
    "documents_v2_router",
    "health_router",
    "profiles_router",
    "retrieve_router",
]
