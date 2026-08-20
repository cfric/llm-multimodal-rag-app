from langchain_core.documents import Document

from src.rag.chain import MultimodalRAGChain
from src.rag.memory import ConversationMemory


class TestMultimodalRAGChain:
    def _make_chain(self, mock_llm, mock_vector_store):
        from src.retrieval.semantic_retriever import SemanticRetriever

        retriever = SemanticRetriever(mock_vector_store)
        return MultimodalRAGChain(retriever=retriever, llm=mock_llm, reranker=None)

    def test_invoke_returns_rag_response(self, mock_llm, mock_vector_store):
        mock_vector_store.similarity_search.return_value = [
            Document(
                page_content="Kontext",
                metadata={"type": "text", "source_file": "test.pdf", "page": 1},
            )
        ]
        chain = self._make_chain(mock_llm, mock_vector_store)
        response = chain.invoke("Was ist ML?", [])

        assert response.answer == "Das ist eine Testantwort."
        assert response.model == "gpt-4o-mock"
        assert response.latency_ms >= 0

    def test_invoke_with_chat_history(self, mock_llm, mock_vector_store, sample_chat_history):
        chain = self._make_chain(mock_llm, mock_vector_store)
        response = chain.invoke("Erkläre mehr.", sample_chat_history)
        assert response.answer is not None

    def test_sources_contain_correct_types(
        self,
        mock_llm,
        mock_vector_store,
        sample_text_document,
        sample_image_document,
        sample_table_document,
    ):
        mock_vector_store.similarity_search.return_value = [
            sample_text_document,
            sample_image_document,
            sample_table_document,
        ]
        chain = self._make_chain(mock_llm, mock_vector_store)
        response = chain.invoke("Test?", [])

        types = {s.type for s in response.sources}
        assert "text" in types
        assert "image" in types
        assert "table" in types

    def test_table_source_has_full_content(
        self, mock_llm, mock_vector_store, sample_table_document
    ):
        mock_vector_store.similarity_search.return_value = [sample_table_document]
        chain = self._make_chain(mock_llm, mock_vector_store)
        response = chain.invoke("Zeig die Tabelle.", [])

        table_source = next(s for s in response.sources if s.type == "table")
        assert "Q1" in table_source.content

    def test_sources_are_deduplicated_by_document_page(self, mock_llm, mock_vector_store):
        mock_vector_store.similarity_search.return_value = [
            Document(
                page_content="Erster Chunk",
                metadata={"type": "text", "source_file": "test.pdf", "page": 1},
            ),
            Document(
                page_content="Zweiter Chunk",
                metadata={"type": "text", "source_file": "test.pdf", "page": 1},
            ),
        ]
        chain = self._make_chain(mock_llm, mock_vector_store)

        response = chain.invoke("Test?", [])

        assert len(response.sources) == 1


class TestConversationMemory:
    def test_add_and_retrieve_messages(self):
        memory = ConversationMemory()
        memory.add_user_message("Hallo")
        memory.add_assistant_message("Guten Tag")
        history = memory.get_history()
        assert len(history) == 2
        assert history[0].role == "human"
        assert history[1].role == "assistant"

    def test_max_messages_limit(self):
        memory = ConversationMemory(max_messages=4)
        for i in range(10):
            memory.add_user_message(f"Nachricht {i}")
        assert len(memory.get_history()) == 4

    def test_clear(self):
        memory = ConversationMemory()
        memory.add_user_message("Test")
        memory.clear()
        assert len(memory) == 0
