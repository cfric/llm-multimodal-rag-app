import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.core.exceptions import VectorStoreError
from src.core.interfaces import BaseEmbedder, BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedder: BaseEmbedder,
        persist_directory: str,
        collection_name: str,
    ) -> None:
        self._embedder = embedder
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._store: Chroma | None = None

    def _get_store(self) -> Chroma:
        if self._store is None:
            Path(self._persist_directory).mkdir(parents=True, exist_ok=True)
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedder.get_embeddings(),
                persist_directory=self._persist_directory,
            )
            logger.info(
                "ChromaDB initialized: collection=%s, dir=%s",
                self._collection_name,
                self._persist_directory,
            )
        return self._store

    def add_documents(self, documents: list[Document]) -> list[str]:
        try:
            ids = self._get_store().add_documents(documents)
            logger.info("Added %d documents to ChromaDB", len(documents))
            return ids
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {e}") from e

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        try:
            return self._get_store().similarity_search(query, k=k)
        except Exception as e:
            raise VectorStoreError(f"Similarity search failed: {e}") from e

    def delete(self, ids: list[str]) -> None:
        try:
            self._get_store().delete(ids=ids)
            logger.info("Deleted %d documents from ChromaDB", len(ids))
        except Exception as e:
            raise VectorStoreError(f"Failed to delete documents: {e}") from e

    def list_sources(self) -> list[dict]:
        try:
            collection = self._get_store()._collection
            result = collection.get(include=["metadatas"])
            seen: dict[str, dict] = {}
            for meta in result["metadatas"]:
                source = meta.get("source_file", "unknown")
                if source not in seen:
                    seen[source] = {
                        "source_file": source,
                        "type": meta.get("type", "text"),
                    }
            return list(seen.values())
        except Exception as e:
            raise VectorStoreError(f"Failed to list sources: {e}") from e

    def delete_by_source(self, source_file: str) -> None:
        try:
            collection = self._get_store()._collection
            result = collection.get(include=["metadatas"])
            ids_to_delete = [
                doc_id
                for doc_id, meta in zip(result["ids"], result["metadatas"])
                if meta.get("source_file") == source_file
            ]
            if ids_to_delete:
                self.delete(ids_to_delete)
                logger.info(
                    "Deleted %d chunks for source: %s",
                    len(ids_to_delete),
                    source_file,
                )
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to delete by source: {e}") from e
