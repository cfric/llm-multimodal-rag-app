import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.core.interfaces import ChatMessage
from src.rag.chain import MultimodalRAGChain
from src.retrieval.semantic_retriever import SemanticRetriever


class TestRAGPipeline:
    """Integration tests: Retriever + Reranker + Chain zusammen."""

    def _make_pipeline(self, mock_llm, docs_to_return):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = docs_to_return
        retriever = SemanticRetriever(mock_vs)
        return MultimodalRAGChain(retriever=retriever, llm=mock_llm, reranker=None)

    def test_full_rag_query_text_only(self, mock_llm):
        docs = [
            Document(
                page_content="Machine Learning ist ein Teilgebiet der KI.",
                metadata={"type": "text", "source_file": "ml_buch.pdf", "page": 5},
            )
        ]
        chain = self._make_pipeline(mock_llm, docs)
        response = chain.invoke("Was ist Machine Learning?", [])

        assert len(response.sources) == 1
        assert response.sources[0].source_file == "ml_buch.pdf"
        assert response.sources[0].type == "text"

    def test_full_rag_query_with_reranker(self, mock_llm):
        docs = [
            Document(page_content=f"Dokument {i}", metadata={"type": "text", "source_file": f"doc{i}.pdf", "page": i})
            for i in range(10)
        ]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = docs[:3]

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = docs
        retriever = SemanticRetriever(mock_vs)
        chain = MultimodalRAGChain(retriever=retriever, llm=mock_llm, reranker=mock_reranker)

        response = chain.invoke("Test?", [])
        assert len(response.sources) == 3
        mock_reranker.rerank.assert_called_once()

    def test_empty_vectorstore_returns_answer(self, mock_llm):
        chain = self._make_pipeline(mock_llm, [])
        response = chain.invoke("Frage ohne Kontext?", [])
        assert response.answer is not None
        assert response.sources == []
