import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.ingestion.pipeline import IngestionPipeline
from src.core.exceptions import IngestionError


class TestIngestionPipeline:
    def _make_pipeline(self, tmp_path, mock_vector_store):
        from src.ingestion.loaders.text_loader import TextLoader
        from src.ingestion.processors.text_processor import TextProcessor
        from src.ingestion.processors.image_processor import ImageProcessor
        from src.ingestion.processors.table_processor import TableProcessor

        mock_vision_llm = MagicMock()
        mock_vision_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Bildbeschreibung"))

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Zusammenfassung"))

        return IngestionPipeline(
            loaders=[TextLoader()],
            text_processor=TextProcessor(),
            image_processor=ImageProcessor(mock_vision_llm, str(tmp_path / "images")),
            table_processor=TableProcessor(mock_llm),
            vector_store=mock_vector_store,
        )

    @pytest.mark.asyncio
    async def test_ingest_text_file(self, tmp_path, mock_vector_store):
        txt = tmp_path / "test.txt"
        txt.write_text("Das ist ein langer Testtext für die Integration. " * 20)
        mock_vector_store.add_documents.return_value = ["id1"]

        pipeline = self._make_pipeline(tmp_path, mock_vector_store)
        ids = await pipeline.ingest(str(txt))

        assert len(ids) > 0
        mock_vector_store.add_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_for_missing_file(self, tmp_path, mock_vector_store):
        pipeline = self._make_pipeline(tmp_path, mock_vector_store)
        with pytest.raises(IngestionError):
            await pipeline.ingest("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_raises_for_unsupported_type(self, tmp_path, mock_vector_store):
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("content")
        pipeline = self._make_pipeline(tmp_path, mock_vector_store)
        with pytest.raises(IngestionError):
            await pipeline.ingest(str(unsupported))
