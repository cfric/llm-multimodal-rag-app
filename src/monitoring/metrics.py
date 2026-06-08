import logging
import threading
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, start_http_server

logger = logging.getLogger(__name__)

RAG_QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total number of RAG queries",
    ["model", "status"],
)

RAG_QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "RAG query latency in seconds",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

RAG_TOKEN_USAGE = Counter(
    "rag_token_usage_total",
    "Total tokens used",
    ["model", "type"],
)

RAG_RETRIEVED_DOCS = Histogram(
    "rag_retrieved_docs_count",
    "Number of documents retrieved per query",
    buckets=[1, 2, 3, 5, 10, 20],
)

RAG_INGESTION_TOTAL = Counter(
    "rag_ingestion_total",
    "Total documents ingested",
    ["status"],
)

RAG_ERRORS_TOTAL = Counter(
    "rag_errors_total",
    "Total errors",
    ["component", "error_type"],
)

_server_started = False
_lock = threading.Lock()


def start_metrics_server(port: int = 8001) -> None:
    global _server_started
    with _lock:
        if not _server_started:
            start_http_server(port)
            _server_started = True
            logger.info("Prometheus metrics server started on port %d", port)


def record_query(model: str, latency_seconds: float, num_docs: int, success: bool = True) -> None:
    status = "success" if success else "error"
    RAG_QUERY_TOTAL.labels(model=model, status=status).inc()
    RAG_QUERY_LATENCY.labels(model=model).observe(latency_seconds)
    RAG_RETRIEVED_DOCS.observe(num_docs)


def record_ingestion(success: bool = True) -> None:
    status = "success" if success else "error"
    RAG_INGESTION_TOTAL.labels(status=status).inc()


def record_error(component: str, error_type: str) -> None:
    RAG_ERRORS_TOTAL.labels(component=component, error_type=error_type).inc()
