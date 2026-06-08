import logging
from typing import Literal

from langchain_core.documents import Document

from src.core.exceptions import RerankerError
from src.core.interfaces import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model: str) -> None:
        self._model_name = model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info("CrossEncoder reranker initialized: %s", self._model_name)
        return self._model

    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        if not documents:
            return []
        try:
            model = self._get_model()
            pairs = [(query, doc.page_content) for doc in documents]
            scores = model.predict(pairs)
            scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            results = []
            for score, doc in scored[:top_n]:
                doc.metadata["rerank_score"] = float(score)
                results.append(doc)
            logger.debug("Reranked %d → %d documents", len(documents), len(results))
            return results
        except Exception as e:
            raise RerankerError(f"CrossEncoder reranking failed: {e}") from e


class CohereReranker(BaseReranker):
    def __init__(self, api_key: str, model: str = "rerank-multilingual-v3.0") -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(self._api_key)
            logger.info("Cohere reranker initialized: %s", self._model)
        return self._client

    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        if not documents:
            return []
        try:
            client = self._get_client()
            texts = [doc.page_content for doc in documents]
            response = client.rerank(
                query=query,
                documents=texts,
                model=self._model,
                top_n=min(top_n, len(documents)),
            )
            results = []
            for result in response.results:
                doc = documents[result.index]
                doc.metadata["rerank_score"] = result.relevance_score
                results.append(doc)
            logger.debug("Cohere reranked %d → %d documents", len(documents), len(results))
            return results
        except Exception as e:
            raise RerankerError(f"Cohere reranking failed: {e}") from e


class RerankerFactory:
    @staticmethod
    def create(
        provider: Literal["crossencoder", "cohere"],
        model: str,
        cohere_api_key: str = "",
    ) -> BaseReranker:
        if provider == "crossencoder":
            return CrossEncoderReranker(model=model)
        if provider == "cohere":
            return CohereReranker(api_key=cohere_api_key, model=model)
        raise ValueError(f"Unknown reranker provider: {provider}")
