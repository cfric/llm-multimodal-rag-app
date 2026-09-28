import logging
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast

from datasets import Dataset

from src.core.exceptions import EvaluationError
from src.core.interfaces import BaseRAGChain, RAGResponse

logger = logging.getLogger(__name__)

EvaluationModality = Literal["text", "image", "table", "mixed"]


class _RagasComponents(NamedTuple):
    evaluate: Any
    metrics: list[Any]


@dataclass
class EvaluationSample:
    question: str
    ground_truth: str
    id: str = "sample"
    modality: EvaluationModality = "text"
    expected_sources: tuple[str, ...] = ()


@dataclass
class EvaluationResult:
    faithfulness: float | None
    answer_relevancy: float | None
    context_recall: float | None
    context_precision: float | None
    samples_evaluated: int
    source_evaluable_samples: int
    source_hit_rate: float
    source_mrr: float
    modality_accuracy: float
    non_empty_answer_rate: float
    mean_latency_ms: float
    by_modality: dict[str, "ModalityMetrics"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_evaluated": self.samples_evaluated,
            "ragas": {
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "context_recall": self.context_recall,
                "context_precision": self.context_precision,
            },
            "deterministic": {
                "source_evaluable_samples": self.source_evaluable_samples,
                "source_hit_rate": self.source_hit_rate,
                "source_mrr": self.source_mrr,
                "modality_accuracy": self.modality_accuracy,
                "non_empty_answer_rate": self.non_empty_answer_rate,
                "mean_latency_ms": self.mean_latency_ms,
            },
            "by_modality": {
                modality: metrics.to_dict() for modality, metrics in self.by_modality.items()
            },
        }


@dataclass
class ModalityMetrics:
    samples_evaluated: int
    source_evaluable_samples: int
    source_hit_rate: float
    source_mrr: float
    modality_accuracy: float
    non_empty_answer_rate: float
    mean_latency_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "samples_evaluated": self.samples_evaluated,
            "source_evaluable_samples": self.source_evaluable_samples,
            "source_hit_rate": self.source_hit_rate,
            "source_mrr": self.source_mrr,
            "modality_accuracy": self.modality_accuracy,
            "non_empty_answer_rate": self.non_empty_answer_rate,
            "mean_latency_ms": self.mean_latency_ms,
        }


@dataclass
class _SampleMetrics:
    source_evaluable: bool
    source_hit: bool
    reciprocal_rank: float
    modality_hit: bool
    non_empty_answer: bool
    latency_ms: float


def _load_ragas() -> _RagasComponents:
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        raise EvaluationError("ragas is not installed") from e

    return _RagasComponents(
        evaluate=ragas_evaluate,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )


def _metric_value(scores: Any, metric_name: str) -> float:
    value = scores.get(metric_name, 0.0)
    if value != value:  # NaN is returned when a metric cannot be calculated.
        return 0.0
    return float(value)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _source_rank(response: RAGResponse, expected_sources: tuple[str, ...]) -> int | None:
    expected = set(expected_sources)
    if not expected:
        return None
    for rank, source in enumerate(response.sources, start=1):
        if source.source_file in expected:
            return rank
    return None


def _modality_matches(modality: EvaluationModality, response: RAGResponse) -> bool:
    source_types = {source.type for source in response.sources}
    if modality == "mixed":
        return len(source_types) >= 2
    return modality in source_types


def _calculate_sample_metrics(
    sample: EvaluationSample, response: RAGResponse
) -> _SampleMetrics:
    source_rank = _source_rank(response, sample.expected_sources)
    return _SampleMetrics(
        source_evaluable=bool(sample.expected_sources),
        source_hit=source_rank is not None,
        reciprocal_rank=1 / source_rank if source_rank is not None else 0.0,
        modality_hit=_modality_matches(sample.modality, response),
        non_empty_answer=bool(response.answer.strip()),
        latency_ms=response.latency_ms,
    )


def _aggregate_metrics(records: list[_SampleMetrics]) -> ModalityMetrics:
    source_records = [record for record in records if record.source_evaluable]
    return ModalityMetrics(
        samples_evaluated=len(records),
        source_evaluable_samples=len(source_records),
        source_hit_rate=_average([float(record.source_hit) for record in source_records]),
        source_mrr=_average([record.reciprocal_rank for record in source_records]),
        modality_accuracy=_average([float(record.modality_hit) for record in records]),
        non_empty_answer_rate=_average([float(record.non_empty_answer) for record in records]),
        mean_latency_ms=_average([record.latency_ms for record in records]),
    )


class RAGEvaluator:
    def __init__(self, rag_chain: BaseRAGChain) -> None:
        self._rag_chain = rag_chain

    def evaluate(
        self, samples: list[EvaluationSample], *, include_ragas: bool = True
    ) -> EvaluationResult:
        if not samples:
            raise EvaluationError("At least one evaluation sample is required")

        ragas = _load_ragas() if include_ragas else None

        logger.info("Starting RAG evaluation with %d samples", len(samples))

        questions, answers, contexts, ground_truths = [], [], [], []
        responses: list[RAGResponse] = []
        for sample in samples:
            response = self._rag_chain.invoke(sample.question, [])
            responses.append(response)
            questions.append(sample.question)
            answers.append(response.answer)
            contexts.append([s.content for s in response.sources])
            ground_truths.append(sample.ground_truth)

        records = [
            _calculate_sample_metrics(sample, response)
            for sample, response in zip(samples, responses)
        ]
        overall = _aggregate_metrics(records)
        by_modality: dict[str, ModalityMetrics] = {}
        for modality in sorted({sample.modality for sample in samples}):
            by_modality[modality] = _aggregate_metrics(
                [record for sample, record in zip(samples, records) if sample.modality == modality]
            )

        ragas_scores: dict[str, float | None] = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_recall": None,
            "context_precision": None,
        }
        if ragas is not None:
            dataset = Dataset.from_dict(
                {
                    "question": questions,
                    "answer": answers,
                    "contexts": contexts,
                    "ground_truth": ground_truths,
                }
            )
            try:
                result = ragas.evaluate(
                    dataset=dataset,
                    metrics=ragas.metrics,
                )
                scores = cast(Any, result).to_pandas().mean(numeric_only=True)
                ragas_scores = {
                    metric_name: _metric_value(scores, metric_name)
                    for metric_name in ragas_scores
                }
                logger.info("RAGAS evaluation complete: %s", ragas_scores)
            except Exception as e:
                raise EvaluationError(f"RAGAS evaluation failed: {e}") from e

        return EvaluationResult(
            faithfulness=ragas_scores["faithfulness"],
            answer_relevancy=ragas_scores["answer_relevancy"],
            context_recall=ragas_scores["context_recall"],
            context_precision=ragas_scores["context_precision"],
            samples_evaluated=overall.samples_evaluated,
            source_evaluable_samples=overall.source_evaluable_samples,
            source_hit_rate=overall.source_hit_rate,
            source_mrr=overall.source_mrr,
            modality_accuracy=overall.modality_accuracy,
            non_empty_answer_rate=overall.non_empty_answer_rate,
            mean_latency_ms=overall.mean_latency_ms,
            by_modality=by_modality,
        )
