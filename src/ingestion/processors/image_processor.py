import base64
import logging
import shutil
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.core.exceptions import ImageProcessingError
from src.core.interfaces import BaseProcessor

logger = logging.getLogger(__name__)

DESCRIBE_IMAGE_PROMPT = (
    "Beschreibe dieses Bild detailliert auf Deutsch. Fokussiere auf alle sichtbaren "
    "Informationen, Texte, Diagramme, Personen, Objekte und deren Zusammenhänge. "
    "Die Beschreibung wird für ein RAG-System verwendet, also sei so präzise wie möglich."
)


class ImageProcessor(BaseProcessor):
    def __init__(self, vision_llm: BaseChatModel, images_dir: str) -> None:
        self._llm = vision_llm
        self._images_dir = Path(images_dir)
        self._images_dir.mkdir(parents=True, exist_ok=True)

    async def process(self, documents: list[Document]) -> list[Document]:
        image_docs = [d for d in documents if d.metadata.get("type") == "image"]
        if not image_docs:
            return []
        processed = []
        for doc in image_docs:
            try:
                result = await self._process_single(doc)
                if result:
                    processed.append(result)
            except Exception as e:
                logger.warning("Failed to process image document: %s", e)
        logger.info("Processed %d image documents", len(processed))
        return processed

    async def _process_single(self, doc: Document) -> Document | None:
        raw_path = doc.metadata.get("raw_image_path")
        if not raw_path or not Path(raw_path).exists():
            logger.warning("Image not found: %s", raw_path)
            return None

        stored_path = self._store_image(raw_path, doc.metadata.get("source_file", "unknown"))
        description = await self._describe_image(stored_path)

        return Document(
            page_content=description,
            metadata={
                **doc.metadata,
                "type": "image",
                "image_path": str(stored_path),
            },
        )

    def _store_image(self, source_path: str, source_file: str) -> Path:
        ext = Path(source_path).suffix or ".png"
        unique_name = f"{Path(source_file).stem}_{uuid.uuid4().hex[:8]}{ext}"
        dest = self._images_dir / unique_name
        shutil.copy2(source_path, dest)
        return dest

    async def _describe_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = image_path.suffix.lstrip(".").lower()
        mime = f"image/{ext}" if ext in {"jpg", "jpeg", "png", "gif", "webp"} else "image/png"
        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": DESCRIBE_IMAGE_PROMPT},
            ]
        )
        try:
            response = await self._llm.ainvoke([message])
            return str(response.content)
        except Exception as e:
            raise ImageProcessingError(f"Vision LLM failed for {image_path}: {e}") from e
