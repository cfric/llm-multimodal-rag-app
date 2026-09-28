from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.api.dependencies import get_rag_chain
from src.config.settings import get_settings
from src.evaluation.dataset import load_evaluation_samples
from src.evaluation.evaluator import RAGEvaluator

DEFAULT_DATASET = Path("tests/evaluation/data/golden_dataset.json")
DEFAULT_OUTPUT = Path("reports/evaluation/latest.json")
RAGAS_EVALUATOR_MODEL = "gpt-4o-mini"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the multimodal RAG pipeline with RAGAS")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Only run deterministic source, modality, answer and latency metrics",
    )
    return parser.parse_args()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main() -> None:
    args = _parse_args()
    samples = load_evaluation_samples(args.dataset)
    settings = get_settings()
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    result = RAGEvaluator(get_rag_chain()).evaluate(
        samples,
        include_ragas=not args.skip_ragas,
    )
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset": {
            "path": str(args.dataset),
            "sha256": dataset_hash,
            "samples": len(samples),
            "modalities": sorted({sample.modality for sample in samples}),
        },
        "configuration": {
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
            "llm_temperature": settings.llm.temperature,
            "embedding_provider": settings.embedding.provider,
            "embedding_model": settings.embedding.model,
            "retrieval_k": settings.retrieval.k,
            "retrieval_search_type": settings.retrieval.search_type,
            "reranker_enabled": settings.reranker.enabled,
            "reranker_provider": settings.reranker.provider,
            "reranker_model": settings.reranker.model,
            "reranker_top_n": settings.reranker.top_n,
            "ragas_evaluator_model": RAGAS_EVALUATOR_MODEL,
        },
        "evaluation_mode": "deterministic" if args.skip_ragas else "deterministic+ragas",
        "evaluation": result.to_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), indent=2))
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
