import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.interfaces import BaseProcessor

logger = logging.getLogger(__name__)


class TextProcessor(BaseProcessor):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def process(self, documents: list[Document]) -> list[Document]:
        text_docs = [d for d in documents if d.metadata.get("type") == "text"]
        if not text_docs:
            return []
        chunks = self._splitter.split_documents(text_docs)
        logger.debug("Split %d text documents into %d chunks", len(text_docs), len(chunks))
        return chunks
