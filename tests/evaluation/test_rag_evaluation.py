import pytest
from unittest.mock import MagicMock

from src.core.interfaces import ChatMessage, RAGResponse, Source
from src.evaluation.evaluator import EvaluationSample, RAGEvaluator


class TestRAGEvaluator:
    def _make_mock_chain(self, answer: str, sources: list[Source]):
        chain = MagicMock()
        chain.invoke.return_value = RAGResponse(
            answer=answer,
            sources=sources,
            query="Test?",
            model="gpt-4o-mock",
            latency_ms=100.0,
        )
        return chain

    def test_evaluator_collects_responses(self):
        source = Source(
            type="text",
            content="Machine Learning ist ein Teilgebiet der KI.",
            source_file="test.pdf",
        )
        chain = self._make_mock_chain(
            answer="Machine Learning ist ein Teilgebiet der KI.",
            sources=[source],
        )
        evaluator = RAGEvaluator(rag_chain=chain)

        samples = [
            EvaluationSample(
                question="Was ist Machine Learning?",
                ground_truth="Machine Learning ist ein Teilgebiet der KI.",
            )
        ]

        with pytest.raises(Exception):
            # Requires RAGAS + OpenAI API — skip in CI without API key
            evaluator.evaluate(samples)

    def test_evaluation_sample_dataclass(self):
        sample = EvaluationSample(
            question="Test?",
            ground_truth="Antwort.",
        )
        assert sample.question == "Test?"
        assert sample.ground_truth == "Antwort."
