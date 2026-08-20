import logging
from pathlib import Path
from typing import Any, cast

import chromadb
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
        min_relevance_score: float = 0.35,
        image_min_relevance_score: float = 0.55,
        chroma_host: str = "",
        chroma_port: int = 8000,
    ) -> None:
        self._embedder = embedder
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._min_relevance_score = min_relevance_score
        self._image_min_relevance_score = image_min_relevance_score
        self._chroma_host = chroma_host
        self._chroma_port = chroma_port
        self._store: Chroma | None = None

    def _get_store(self) -> Chroma:
        if self._store is None:
            if self._chroma_host:
                client = chromadb.HttpClient(host=self._chroma_host, port=self._chroma_port)
                self._store = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embedder.get_embeddings(),
                    client=client,
                )
                logger.info(
                    "ChromaDB initialized: collection=%s, host=%s:%d",
                    self._collection_name,
                    self._chroma_host,
                    self._chroma_port,
                )
            else:
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
            results = self._get_store().similarity_search_with_relevance_scores(query, k=k)
            documents = []
            for document, score in results:
                min_score = (
                    self._image_min_relevance_score
                    if document.metadata.get("type") == "image"
                    else self._min_relevance_score
                )
                if score < min_score:
                    continue
                document.metadata["retrieval_score"] = score
                documents.append(document)
            logger.debug(
                "Retrieved %d/%d chunks above relevance threshold %.2f",
                len(documents),
                len(results),
                self._min_relevance_score,
            )
            return documents
        except Exception as e:
            raise VectorStoreError(f"Similarity search failed: {e}") from e

    def delete(self, ids: list[str]) -> None:
        try:
            self._get_store().delete(ids=ids)
            logger.info("Deleted %d documents from ChromaDB", len(ids))
        except Exception as e:
            raise VectorStoreError(f"Failed to delete documents: {e}") from e

    def list_sources(self) -> list[dict[str, str]]:
        try:
            collection = cast(Any, self._get_store()._collection)
            result = cast(dict[str, Any], collection.get(include=["metadatas"]))
            metadatas = cast(list[dict[str, Any]], result["metadatas"])
            seen: dict[str, dict[str, str]] = {}
            for meta in metadatas:
                source = str(meta.get("source_file", "unknown"))
                if source not in seen:
                    seen[source] = {
                        "source_file": source,
                        "type": str(meta.get("type", "text")),
                    }
            return list(seen.values())
        except Exception as e:
            raise VectorStoreError(f"Failed to list sources: {e}") from e

    def delete_by_source(self, source_file: str) -> None:
        try:
            collection = cast(Any, self._get_store()._collection)
            result = cast(dict[str, Any], collection.get(include=["metadatas"]))
            ids = cast(list[str], result["ids"])
            metadatas = cast(list[dict[str, Any]], result["metadatas"])
            ids_to_delete = [
                doc_id
                for doc_id, meta in zip(ids, metadatas)
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
