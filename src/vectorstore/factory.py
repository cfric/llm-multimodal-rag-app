import logging

from src.config.settings import Settings
from src.core.exceptions import ConfigurationError
from src.core.interfaces import BaseEmbedder, BaseVectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    @staticmethod
    def create(settings: Settings, embedder: BaseEmbedder) -> BaseVectorStore:
        provider = settings.vectorstore.provider
        logger.info("Creating VectorStore: %s", provider)

        if provider == "chroma":
            from src.vectorstore.chroma_store import ChromaVectorStore

            return ChromaVectorStore(
                embedder=embedder,
                persist_directory=settings.vectorstore.persist_directory,
                collection_name=settings.vectorstore.collection_name,
                min_relevance_score=settings.vectorstore.min_relevance_score,
                image_min_relevance_score=settings.vectorstore.image_min_relevance_score,
                chroma_host=settings.vectorstore.chroma_host,
                chroma_port=settings.vectorstore.chroma_port,
            )

        raise ConfigurationError(f"Unknown VectorStore provider: {provider}")
