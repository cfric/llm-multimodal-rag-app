from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.config.settings import get_settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        vectorstore=settings.vectorstore.provider,
        llm_provider=settings.llm.provider,
    )
