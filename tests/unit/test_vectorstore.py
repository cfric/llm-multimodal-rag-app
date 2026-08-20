from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.core.exceptions import VectorStoreError
from src.vectorstore.chroma_store import ChromaVectorStore


class TestChromaVectorStore:
    def _make_store(self, tmp_path):
        mock_embedder = MagicMock()
        mock_embedder.get_embeddings.return_value = MagicMock()
        return ChromaVectorStore(
            embedder=mock_embedder,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_collection",
        )

    def test_initialization(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store is not None

    def test_add_documents_returns_ids(self, tmp_path):
        store = self._make_store(tmp_path)
        mock_chroma = MagicMock()
        mock_chroma.add_documents.return_value = ["id1", "id2"]
        store._store = mock_chroma

        docs = [Document(page_content="Test", metadata={"source_file": "test.pdf"})]
        ids = store.add_documents(docs)
        assert len(ids) == 2

    def test_add_documents_raises_on_failure(self, tmp_path):
        store = self._make_store(tmp_path)
        mock_chroma = MagicMock()
        mock_chroma.add_documents.side_effect = Exception("DB error")
        store._store = mock_chroma

        with pytest.raises(VectorStoreError):
            store.add_documents([Document(page_content="Test", metadata={})])

    def test_similarity_search(self, tmp_path):
        store = ChromaVectorStore(
            embedder=MagicMock(),
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_collection",
            min_relevance_score=0.5,
            image_min_relevance_score=0.7,
        )
        relevant = Document(page_content="Ergebnis", metadata={})
        irrelevant = Document(page_content="Fremdes Thema", metadata={"type": "image"})
        mock_chroma = MagicMock()
        mock_chroma.similarity_search_with_relevance_scores.return_value = [
            (relevant, 0.8),
            (irrelevant, 0.6),
        ]
        store._store = mock_chroma

        result = store.similarity_search("Test-Query", k=5)
        assert result == [relevant]
        assert relevant.metadata["retrieval_score"] == 0.8
        mock_chroma.similarity_search_with_relevance_scores.assert_called_once_with(
            "Test-Query", k=5
        )
