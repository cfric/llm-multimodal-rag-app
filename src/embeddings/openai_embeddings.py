import logging

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.core.exceptions import ConfigurationError
from src.core.interfaces import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set")
        self._api_key = api_key
        self._model = model
        self._embeddings: Embeddings | None = None

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=self._api_key,
                model=self._model,
            )
            logger.info("OpenAI embeddings initialized: %s", self._model)
        return self._embeddings
