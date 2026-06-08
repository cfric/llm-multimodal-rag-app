import logging

from langchain_core.documents import Document

from src.core.exceptions import RetrievalError
from src.core.interfaces import BaseRetriever, BaseVectorStore

logger = logging.getLogger(__name__)


class SemanticRetriever(BaseRetriever):
    def __init__(self, vector_store: BaseVectorStore) -> None:
        self._vector_store = vector_store

    def retrieve(self, query: str, k: int = 10) -> list[Document]:
        try:
            docs = self._vector_store.similarity_search(query, k=k)
            logger.debug("Retrieved %d documents for query", len(docs))
            return docs
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}") from e
