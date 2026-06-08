import logging

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.interfaces import BaseEmbedder

logger = logging.getLogger(__name__)


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model: str) -> None:
        self._model = model
        self._embeddings: Embeddings | None = None

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("HuggingFace embeddings initialized: %s", self._model)
        return self._embeddings
