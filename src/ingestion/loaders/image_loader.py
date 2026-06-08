import logging
from pathlib import Path

from langchain_core.documents import Document

from src.core.exceptions import DocumentLoadError, UnsupportedFileTypeError
from src.core.interfaces import BaseLoader

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


class ImageLoader(BaseLoader):
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS

    def load(self, file_path: str) -> list[Document]:
        if not self.supports(file_path):
            raise UnsupportedFileTypeError(f"ImageLoader does not support: {file_path}")
        try:
            path = Path(file_path)
            document = Document(
                page_content="",
                metadata={
                    "type": "image",
                    "source_file": path.name,
                    "source_path": str(file_path),
                    "raw_image_path": str(file_path),
                    "page": None,
                    "element_type": "Image",
                },
            )
            logger.info("Loaded image: %s", file_path)
            return [document]
        except Exception as e:
            raise DocumentLoadError(f"Failed to load image {file_path}: {e}") from e
