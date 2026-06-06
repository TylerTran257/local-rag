from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")


@router.get("/chat")
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})
