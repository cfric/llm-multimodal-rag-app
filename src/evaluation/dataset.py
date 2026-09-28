from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from src.core.exceptions import EvaluationError
from src.evaluation.evaluator import EvaluationModality, EvaluationSample


def _required_string(item: dict[str, Any], field_name: str, index: int) -> str:
    value = item.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"Sample {index} requires a non-empty {field_name}")
    return value


def load_evaluation_samples(path: str | Path) -> list[EvaluationSample]:
    dataset_path = Path(path)
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise EvaluationError(f"Could not load evaluation dataset {dataset_path}: {e}") from e

    if not isinstance(payload, list) or not payload:
        raise EvaluationError("Evaluation dataset must be a non-empty JSON array")

    samples: list[EvaluationSample] = []
    for index, raw_item in enumerate(payload):
        if not isinstance(raw_item, dict):
            raise EvaluationError(f"Sample {index} must be a JSON object")

        sample_id = _required_string(raw_item, "id", index)
        question = _required_string(raw_item, "question", index)
        ground_truth = _required_string(raw_item, "ground_truth", index)
        modality = raw_item.get("modality", "text")
        expected_sources = raw_item.get("expected_sources", [])

        if modality not in {"text", "image", "table", "mixed"}:
            raise EvaluationError(f"Sample {sample_id} has an unsupported modality")
        if not isinstance(expected_sources, list) or not all(
            isinstance(source, str) and source.strip() for source in expected_sources
        ):
            raise EvaluationError(f"Sample {sample_id} has invalid expected_sources")

        samples.append(
            EvaluationSample(
                id=sample_id,
                modality=cast(EvaluationModality, modality),
                question=question,
                ground_truth=ground_truth,
                expected_sources=tuple(expected_sources),
            )
        )

    return samples
