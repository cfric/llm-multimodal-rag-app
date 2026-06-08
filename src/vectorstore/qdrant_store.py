import logging

from langchain_core.documents import Document

from src.core.exceptions import VectorStoreError
from src.core.interfaces import BaseEmbedder, BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant VectorStore — Interface vorbereitet, Implementierung folgt bei Bedarf."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        url: str,
        collection_name: str,
        api_key: str = "",
    ) -> None:
        self._embedder = embedder
        self._url = url
        self._collection_name = collection_name
        self._api_key = api_key

    def _get_store(self):  # type: ignore[return]
        try:
            from langchain_qdrant import QdrantVectorStore as LCQdrant
            from qdrant_client import QdrantClient

            client = QdrantClient(url=self._url, api_key=self._api_key or None)
            return LCQdrant(
                client=client,
                collection_name=self._collection_name,
                embeddings=self._embedder.get_embeddings(),
            )
        except ImportError as e:
            raise VectorStoreError("langchain-qdrant not installed") from e

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self._get_store().add_documents(documents)

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        return self._get_store().similarity_search(query, k=k)

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError("Qdrant delete not yet implemented")

    def list_sources(self) -> list[dict]:
        raise NotImplementedError("Qdrant list_sources not yet implemented")

    def delete_by_source(self, source_file: str) -> None:
        raise NotImplementedError("Qdrant delete_by_source not yet implemented")
