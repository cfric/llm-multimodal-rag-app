import pytest
from unittest.mock import MagicMock

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
        store = self._make_store(tmp_path)
        expected = [Document(page_content="Ergebnis", metadata={})]
        mock_chroma = MagicMock()
        mock_chroma.similarity_search.return_value = expected
        store._store = mock_chroma

        result = store.similarity_search("Test-Query", k=5)
        assert result == expected
        mock_chroma.similarity_search.assert_called_once_with("Test-Query", k=5)
