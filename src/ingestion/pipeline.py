import asyncio
import logging
from pathlib import Path

from langchain_core.documents import Document

from src.core.exceptions import IngestionError, UnsupportedFileTypeError
from src.core.interfaces import BaseLoader, BaseProcessor, BaseVectorStore
from src.ingestion.processors.image_processor import ImageProcessor
from src.ingestion.processors.table_processor import TableProcessor
from src.ingestion.processors.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        loaders: list[BaseLoader],
        text_processor: TextProcessor,
        image_processor: ImageProcessor,
        table_processor: TableProcessor,
        vector_store: BaseVectorStore,
    ) -> None:
        self._loaders = loaders
        self._text_processor = text_processor
        self._image_processor = image_processor
        self._table_processor = table_processor
        self._vector_store = vector_store

    def _get_loader(self, file_path: str) -> BaseLoader:
        for loader in self._loaders:
            if loader.supports(file_path):
                return loader
        raise UnsupportedFileTypeError(f"No loader found for: {file_path}")

    async def ingest(self, file_path: str) -> list[str]:
        path = Path(file_path)
        if not path.exists():
            raise IngestionError(f"File not found: {file_path}")

        logger.info("Starting ingestion: %s", path.name)

        try:
            loader = self._get_loader(file_path)
            raw_documents = loader.load(file_path)
        except Exception as e:
            raise IngestionError(f"Loading failed for {file_path}: {e}") from e

        text_chunks, image_chunks, table_chunks = await asyncio.gather(
            self._text_processor.process(raw_documents),
            self._image_processor.process(raw_documents),
            self._table_processor.process(raw_documents),
        )

        all_chunks = text_chunks + image_chunks + table_chunks
        if not all_chunks:
            logger.warning("No content extracted from: %s", file_path)
            return []

        try:
            ids = self._vector_store.add_documents(all_chunks)
        except Exception as e:
            raise IngestionError(f"VectorStore insertion failed: {e}") from e

        logger.info(
            "Ingestion complete: %s — %d text, %d image, %d table chunks",
            path.name,
            len(text_chunks),
            len(image_chunks),
            len(table_chunks),
        )
        return ids
