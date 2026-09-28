from unittest.mock import MagicMock

import pytest

from src.core.interfaces import RAGResponse, Source
from src.evaluation.dataset import load_evaluation_samples
from src.evaluation.evaluator import EvaluationSample, RAGEvaluator, _RagasComponents


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

    def test_evaluator_collects_responses(self, monkeypatch):
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
                id="sample_001",
                question="Was ist Machine Learning?",
                ground_truth="Machine Learning ist ein Teilgebiet der KI.",
            )
        ]

        pandas_result = MagicMock()
        pandas_result.mean.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_recall": 0.7,
            "context_precision": 0.6,
        }
        ragas_result = MagicMock()
        ragas_result.to_pandas.return_value = pandas_result
        ragas_evaluate = MagicMock(return_value=ragas_result)
        monkeypatch.setattr(
            "src.evaluation.evaluator._load_ragas",
            lambda: _RagasComponents(evaluate=ragas_evaluate, metrics=[]),
        )

        result = evaluator.evaluate(samples)

        assert result.to_dict() == {
            "samples_evaluated": 1,
            "ragas": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_recall": 0.7,
                "context_precision": 0.6,
            },
            "deterministic": {
                "source_evaluable_samples": 0,
                "source_hit_rate": 0.0,
                "source_mrr": 0.0,
                "modality_accuracy": 1.0,
                "non_empty_answer_rate": 1.0,
                "mean_latency_ms": 100.0,
            },
            "by_modality": {
                "text": {
                    "samples_evaluated": 1,
                    "source_evaluable_samples": 0,
                    "source_hit_rate": 0.0,
                    "source_mrr": 0.0,
                    "modality_accuracy": 1.0,
                    "non_empty_answer_rate": 1.0,
                    "mean_latency_ms": 100.0,
                }
            },
        }
        ragas_evaluate.assert_called_once()

    def test_evaluator_calculates_deterministic_metrics(self):
        chain = MagicMock()
        chain.invoke.side_effect = [
            RAGResponse(
                answer="Antwort 1",
                sources=[
                    Source(type="text", content="Falsch", source_file="other.md"),
                    Source(type="text", content="Richtig", source_file="demo.md"),
                ],
                query="Frage 1",
                model="mock",
                latency_ms=100.0,
            ),
            RAGResponse(
                answer="Antwort 2",
                sources=[
                    Source(type="image", content="Bild", source_file="diagram.png"),
                ],
                query="Frage 2",
                model="mock",
                latency_ms=200.0,
            ),
        ]
        samples = [
            EvaluationSample(
                id="text_001",
                modality="text",
                question="Frage 1",
                ground_truth="Antwort 1",
                expected_sources=("demo.md",),
            ),
            EvaluationSample(
                id="image_001",
                modality="image",
                question="Frage 2",
                ground_truth="Antwort 2",
                expected_sources=("diagram.png",),
            ),
        ]

        result = RAGEvaluator(chain).evaluate(samples, include_ragas=False)

        assert result.source_evaluable_samples == 2
        assert result.source_hit_rate == 1.0
        assert result.source_mrr == 0.75
        assert result.modality_accuracy == 1.0
        assert result.non_empty_answer_rate == 1.0
        assert result.mean_latency_ms == 150.0
        assert result.faithfulness is None
        assert result.by_modality["text"].source_mrr == 0.5
        assert result.by_modality["image"].source_mrr == 1.0

    def test_evaluator_marks_wrong_modality_and_empty_answer(self):
        chain = self._make_mock_chain(
            answer="",
            sources=[Source(type="text", content="Text", source_file="demo.md")],
        )
        sample = EvaluationSample(
            id="table_001",
            modality="table",
            question="Frage?",
            ground_truth="Antwort",
            expected_sources=("demo.md",),
        )

        result = RAGEvaluator(chain).evaluate([sample], include_ragas=False)

        assert result.source_hit_rate == 1.0
        assert result.modality_accuracy == 0.0
        assert result.non_empty_answer_rate == 0.0

    def test_evaluator_rejects_empty_dataset(self, mock_llm):
        with pytest.raises(Exception, match="At least one evaluation sample"):
            RAGEvaluator(mock_llm).evaluate([])

    def test_evaluation_sample_dataclass(self):
        sample = EvaluationSample(
            question="Test?",
            ground_truth="Antwort.",
        )
        assert sample.question == "Test?"
        assert sample.ground_truth == "Antwort."

    def test_loads_golden_dataset(self):
        samples = load_evaluation_samples("tests/evaluation/data/golden_dataset.json")

        assert len(samples) == 6
        assert {sample.modality for sample in samples} == {"text", "table", "image"}
        assert samples[0].expected_sources == ("demo_document.md",)
