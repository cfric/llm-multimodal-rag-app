import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.core.exceptions import TableProcessingError
from src.core.interfaces import BaseProcessor

logger = logging.getLogger(__name__)

SUMMARIZE_TABLE_PROMPT = (
    "Erstelle eine präzise semantische Zusammenfassung der folgenden Tabelle auf Deutsch. "
    "Beschreibe den Inhalt, die wichtigsten Datenpunkte, Trends und Erkenntnisse. "
    "Die Zusammenfassung wird als Embedding für eine semantische Suche verwendet. "
    "Tabelle:\n\n{table_content}"
)


class TableProcessor(BaseProcessor):
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def process(self, documents: list[Document]) -> list[Document]:
        table_docs = [d for d in documents if d.metadata.get("type") == "table"]
        if not table_docs:
            return []
        processed = []
        for doc in table_docs:
            try:
                result = await self._process_single(doc)
                processed.append(result)
            except Exception as e:
                logger.warning("Failed to process table, storing as-is: %s", e)
                processed.append(doc)
        logger.info("Processed %d table documents", len(processed))
        return processed

    async def _process_single(self, doc: Document) -> Document:
        table_content = doc.metadata.get("raw_table") or doc.page_content
        summary = await self._summarize(table_content)
        return Document(
            page_content=summary,
            metadata={
                **doc.metadata,
                "type": "table",
                "table_content": table_content,
                "caption": self._extract_caption(doc),
            },
        )

    async def _summarize(self, table_content: str) -> str:
        prompt = SUMMARIZE_TABLE_PROMPT.format(table_content=table_content)
        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            return str(response.content)
        except Exception as e:
            raise TableProcessingError(f"LLM summarization failed: {e}") from e

    def _extract_caption(self, doc: Document) -> str:
        return doc.metadata.get("caption") or doc.metadata.get("element_type", "Tabelle")
