import base64
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.core.interfaces import (
    BaseLLM,
    BaseRAGChain,
    BaseReranker,
    BaseRetriever,
    ChatMessage,
    RAGResponse,
    Source,
)
from src.rag.prompt_templates import (
    IMAGE_CONTEXT_TEMPLATE,
    SYSTEM_PROMPT,
    TABLE_CONTEXT_TEMPLATE,
    TEXT_CONTEXT_TEMPLATE,
)

logger = logging.getLogger(__name__)


class MultimodalRAGChain(BaseRAGChain):
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        reranker: BaseReranker | None = None,
        retrieval_k: int = 20,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker
        self._retrieval_k = retrieval_k

    def invoke(self, query: str, chat_history: list[ChatMessage]) -> RAGResponse:
        start = time.monotonic()

        docs = self._retriever.retrieve(query, k=self._retrieval_k)
        if self._reranker:
            docs = self._reranker.rerank(query, docs)

        sources = self._build_sources(docs)
        messages = self._build_messages(query, docs, chat_history)

        response = self._llm.get_model().invoke(messages)
        latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "RAG query completed in %.0fms — %d sources returned",
            latency_ms,
            len(sources),
        )
        return RAGResponse(
            answer=str(response.content),
            sources=sources,
            query=query,
            model=self._llm.model_name,
            latency_ms=latency_ms,
        )

    async def astream(self, query: str, chat_history: list[ChatMessage]) -> AsyncIterator[str]:
        docs = self._retriever.retrieve(query, k=self._retrieval_k)
        if self._reranker:
            docs = self._reranker.rerank(query, docs)

        messages = self._build_messages(query, docs, chat_history)

        async for chunk in self._llm.get_model().astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield str(chunk.content)

    def _build_sources(self, docs: list[Document]) -> list[Source]:
        sources = []
        seen_sources: set[tuple[str, str, int | None, str | None]] = set()
        for doc in docs:
            meta = doc.metadata
            doc_type = meta.get("type", "text")
            source_file = meta.get("source_file", "unknown")
            page = meta.get("page")
            image_path = meta.get("image_path") if doc_type == "image" else None
            source_key = (doc_type, source_file, page, image_path)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            sources.append(
                Source(
                    type=doc_type,
                    content=meta.get("table_content", doc.page_content)
                    if doc_type == "table"
                    else doc.page_content,
                    source_file=source_file,
                    page=page,
                    image_path=image_path,
                    caption=meta.get("caption") if doc_type == "table" else None,
                    summary=doc.page_content if doc_type == "table" else None,
                    score=meta.get("rerank_score"),
                )
            )
        return sources

    def _build_messages(
        self,
        query: str,
        docs: list[Document],
        chat_history: list[ChatMessage],
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        for msg in chat_history[:-1] if chat_history else []:
            if msg.role == "human":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))

        content_parts: list[str | dict[Any, Any]] = []

        text_docs = [d for d in docs if d.metadata.get("type") == "text"]
        image_docs = [d for d in docs if d.metadata.get("type") == "image"]
        table_docs = [d for d in docs if d.metadata.get("type") == "table"]

        context_parts = []
        for doc in text_docs:
            context_parts.append(
                TEXT_CONTEXT_TEMPLATE.format(
                    source_file=doc.metadata.get("source_file", "?"),
                    page=doc.metadata.get("page", "?"),
                    content=doc.page_content,
                )
            )
        for doc in table_docs:
            table_content = doc.metadata.get("table_content", doc.page_content)
            context_parts.append(
                TABLE_CONTEXT_TEMPLATE.format(
                    source_file=doc.metadata.get("source_file", "?"),
                    page=doc.metadata.get("page", "?"),
                    table_content=table_content,
                )
            )

        if context_parts:
            content_parts.append({"type": "text", "text": "\n\n".join(context_parts)})

        for doc in image_docs:
            image_path = doc.metadata.get("image_path")
            if image_path and Path(image_path).exists():
                b64, mime = self._encode_image(image_path)
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                )
            content_parts.append(
                {
                    "type": "text",
                    "text": IMAGE_CONTEXT_TEMPLATE.format(
                        source_file=doc.metadata.get("source_file", "?"),
                        page=doc.metadata.get("page", "?"),
                        description=doc.page_content,
                    ),
                }
            )

        content_parts.append({"type": "text", "text": f"Frage: {query}"})
        messages.append(HumanMessage(content=content_parts))
        return messages

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        ext = Path(image_path).suffix.lstrip(".").lower()
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime = mime_map.get(ext, "image/png")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode(), mime
