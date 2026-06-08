import logging
from pathlib import Path

from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf

from src.core.exceptions import DocumentLoadError, UnsupportedFileTypeError
from src.core.interfaces import BaseLoader

logger = logging.getLogger(__name__)

_IMAGES_OUTPUT_DIR = "data/images"


class PDFLoader(BaseLoader):
    def __init__(self, images_dir: str = _IMAGES_OUTPUT_DIR) -> None:
        self._images_dir = images_dir

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def load(self, file_path: str) -> list[Document]:
        if not self.supports(file_path):
            raise UnsupportedFileTypeError(f"PDFLoader does not support: {file_path}")
        try:
            Path(self._images_dir).mkdir(parents=True, exist_ok=True)
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                extract_images_in_pdf=True,
                image_output_dir_path=self._images_dir,
            )
            documents = []
            for el in elements:
                el_type = type(el).__name__
                metadata = {
                    "source_file": Path(file_path).name,
                    "source_path": str(file_path),
                    "page": getattr(el.metadata, "page_number", None),
                    "element_type": el_type,
                }
                if el_type == "Table":
                    metadata["type"] = "table"
                    metadata["raw_table"] = str(el)
                elif el_type == "Image":
                    metadata["type"] = "image"
                    image_path = getattr(el.metadata, "image_path", None)
                    if image_path and Path(image_path).exists():
                        metadata["raw_image_path"] = image_path
                    else:
                        metadata["type"] = "text"
                else:
                    metadata["type"] = "text"
                documents.append(Document(page_content=str(el), metadata=metadata))
            logger.info("Loaded %d elements from PDF: %s", len(documents), file_path)
            return documents
        except Exception as e:
            raise DocumentLoadError(f"Failed to load PDF {file_path}: {e}") from e
