from fastapi import APIRouter, Depends

from app.dependencies import get_rag_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RagService

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    return await rag_service.answer(request)

