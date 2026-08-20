from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.core.interfaces import ChatMessage


@pytest.fixture
def sample_text_document():
    return Document(
        page_content="Das ist ein Testdokument über Machine Learning.",
        metadata={"type": "text", "source_file": "test.pdf", "page": 1},
    )


@pytest.fixture
def sample_image_document():
    return Document(
        page_content="Ein Diagramm zeigt den Trainingsverlust über 100 Epochen.",
        metadata={
            "type": "image",
            "source_file": "test.pdf",
            "page": 2,
            "image_path": "/tmp/test_image.png",
        },
    )


@pytest.fixture
def sample_table_document():
    return Document(
        page_content="Zusammenfassung: Tabelle mit Quartalsumsätzen 2024.",
        metadata={
            "type": "table",
            "source_file": "test.pdf",
            "page": 3,
            "table_content": "| Q1 | Q2 |\n|-----|-----|\n| 1M | 2M |",
            "caption": "Quartalsumsätze 2024",
        },
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model_name = "gpt-4o-mock"
    model = MagicMock()
    model.invoke.return_value = MagicMock(content="Das ist eine Testantwort.")

    async def _astream(*args, **kwargs):
        for text in ["Das ", "ist ", "eine ", "Antwort."]:
            yield MagicMock(content=text)

    model.astream = _astream
    llm.get_model.return_value = model
    return llm


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.similarity_search.return_value = []
    vs.add_documents.return_value = ["id1", "id2"]
    vs.list_sources.return_value = []
    return vs


@pytest.fixture
def sample_chat_history():
    return [
        ChatMessage(role="human", content="Was ist Machine Learning?"),
        ChatMessage(role="assistant", content="Machine Learning ist..."),
    ]
