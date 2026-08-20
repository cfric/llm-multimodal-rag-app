from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api.dependencies import _sessions, get_rag_chain
from src.api.main import create_app
from src.core.interfaces import RAGResponse


def test_query_uses_the_request_session_id() -> None:
    chain = MagicMock()
    chain.invoke.return_value = RAGResponse(
        answer="Antwort",
        sources=[],
        query="Frage",
        model="test-model",
        latency_ms=1.0,
    )
    app = create_app()
    app.dependency_overrides[get_rag_chain] = lambda: chain
    _sessions.clear()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rag/query", json={"question": "Frage", "session_id": "portfolio"}
        )

    assert response.status_code == 200
    assert [message.content for message in _sessions["portfolio"].get_history()] == [
        "Frage",
        "Antwort",
    ]
    assert "default" not in _sessions
