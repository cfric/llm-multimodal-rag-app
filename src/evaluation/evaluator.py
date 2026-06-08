import logging
from dataclasses import dataclass

from datasets import Dataset

from src.core.exceptions import EvaluationError
from src.core.interfaces import BaseRAGChain, ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    question: str
    ground_truth: str


@dataclass
class EvaluationResult:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    samples_evaluated: int


class RAGEvaluator:
    def __init__(self, rag_chain: BaseRAGChain) -> None:
        self._rag_chain = rag_chain

    def evaluate(self, samples: list[EvaluationSample]) -> EvaluationResult:
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as e:
            raise EvaluationError("ragas not installed") from e

        logger.info("Starting RAG evaluation with %d samples", len(samples))

        questions, answers, contexts, ground_truths = [], [], [], []
        for sample in samples:
            response = self._rag_chain.invoke(sample.question, [])
            questions.append(sample.question)
            answers.append(response.answer)
            contexts.append([s.content for s in response.sources])
            ground_truths.append(sample.ground_truth)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        try:
            result = ragas_evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            )
            scores = result.to_pandas().mean()
            logger.info("Evaluation complete: %s", dict(scores))
            return EvaluationResult(
                faithfulness=float(scores.get("faithfulness", 0.0)),
                answer_relevancy=float(scores.get("answer_relevancy", 0.0)),
                context_recall=float(scores.get("context_recall", 0.0)),
                context_precision=float(scores.get("context_precision", 0.0)),
                samples_evaluated=len(samples),
            )
        except Exception as e:
            raise EvaluationError(f"RAGAS evaluation failed: {e}") from e
