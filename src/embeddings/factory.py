import logging

from src.config.settings import Settings
from src.core.exceptions import ConfigurationError
from src.core.interfaces import BaseEmbedder

logger = logging.getLogger(__name__)


class EmbedderFactory:
    @staticmethod
    def create(settings: Settings) -> BaseEmbedder:
        provider = settings.embedding.provider
        logger.info("Creating embedder: %s", provider)

        if provider == "openai":
            from src.embeddings.openai_embeddings import OpenAIEmbedder

            return OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.embedding.model,
            )

        if provider == "huggingface":
            from src.embeddings.local_embeddings import LocalEmbedder

            return LocalEmbedder(model=settings.embedding.model)

        raise ConfigurationError(f"Unknown embedding provider: {provider}")
