import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from src.ingestion.processors.text_processor import TextProcessor
from src.ingestion.processors.image_processor import ImageProcessor
from src.ingestion.processors.table_processor import TableProcessor


class TestTextProcessor:
    def setup_method(self):
        self.processor = TextProcessor(chunk_size=100, chunk_overlap=20)

    @pytest.mark.asyncio
    async def test_splits_long_text(self):
        long_text = "Wort " * 200
        docs = [Document(page_content=long_text, metadata={"type": "text", "source_file": "test.pdf"})]
        result = await self.processor.process(docs)
        assert len(result) > 1

    @pytest.mark.asyncio
    async def test_skips_non_text_documents(self, sample_image_document):
        result = await self.processor.process([sample_image_document])
        assert result == []

    @pytest.mark.asyncio
    async def test_preserves_metadata(self):
        docs = [Document(
            page_content="Kurzer Text.",
            metadata={"type": "text", "source_file": "doc.pdf", "page": 1}
        )]
        result = await self.processor.process(docs)
        assert all(r.metadata["source_file"] == "doc.pdf" for r in result)


class TestTableProcessor:
    @pytest.mark.asyncio
    async def test_summarizes_table(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Tabelle zeigt Umsatzdaten."))
        processor = TableProcessor(llm=mock_llm)

        table_doc = Document(
            page_content="| A | B |\n|---|---|\n| 1 | 2 |",
            metadata={"type": "table", "source_file": "test.pdf", "raw_table": "| A | B |\n|---|---|\n| 1 | 2 |"},
        )
        result = await processor.process([table_doc])

        assert len(result) == 1
        assert result[0].page_content == "Tabelle zeigt Umsatzdaten."
        assert result[0].metadata["table_content"] == "| A | B |\n|---|---|\n| 1 | 2 |"

    @pytest.mark.asyncio
    async def test_skips_non_table_documents(self, sample_text_document):
        mock_llm = MagicMock()
        processor = TableProcessor(llm=mock_llm)
        result = await processor.process([sample_text_document])
        assert result == []
