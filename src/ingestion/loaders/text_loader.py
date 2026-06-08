import logging
from pathlib import Path

from langchain_core.documents import Document
from unstructured.partition.auto import partition

from src.core.exceptions import DocumentLoadError, UnsupportedFileTypeError
from src.core.interfaces import BaseLoader

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".html", ".htm", ".docx", ".doc", ".pptx"}


class TextLoader(BaseLoader):
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in TEXT_EXTENSIONS

    def load(self, file_path: str) -> list[Document]:
        if not self.supports(file_path):
            raise UnsupportedFileTypeError(f"TextLoader does not support: {file_path}")
        try:
            elements = partition(filename=file_path)
            documents = []
            for el in elements:
                el_type = type(el).__name__
                metadata = {
                    "type": "table" if el_type == "Table" else "text",
                    "source_file": Path(file_path).name,
                    "source_path": str(file_path),
                    "page": getattr(el.metadata, "page_number", None),
                    "element_type": el_type,
                }
                if el_type == "Table":
                    metadata["raw_table"] = str(el)
                documents.append(Document(page_content=str(el), metadata=metadata))
            logger.info("Loaded %d elements from: %s", len(documents), file_path)
            return documents
        except Exception as e:
            raise DocumentLoadError(f"Failed to load {file_path}: {e}") from e
