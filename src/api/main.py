import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import documents, health, rag
from src.config.settings import get_settings
from src.monitoring.metrics import start_metrics_server
from src.utils.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    if settings.langsmith.enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key

    app = FastAPI(
        title="Multimodal RAG API",
        description="Production-ready multimodal RAG system",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(rag.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup() -> None:
        if settings.monitoring.enabled:
            start_metrics_server(settings.monitoring.prometheus_port)
        logger.info("Multimodal RAG API started")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("src.api.main:app", host=settings.api.host, port=settings.api.port, reload=True)
