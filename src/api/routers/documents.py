import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status

from src.api.dependencies import get_ingestion_pipeline, get_vector_store
from src.api.schemas import DeleteResponse, DocumentInfo, DocumentListResponse, IngestionResponse
from src.config.settings import get_settings
from src.core.exceptions import IngestionError, UnsupportedFileTypeError
from src.core.interfaces import BaseVectorStore
from src.ingestion.pipeline import IngestionPipeline
from src.monitoring import metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".jpg", ".jpeg", ".png"}


@router.post("/upload", response_model=IngestionResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    max_upload_size_mb = get_settings().api.max_upload_size_mb
    max_upload_bytes = max_upload_size_mb * 1024 * 1024
    uploaded_bytes = 0
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        while chunk := await file.read(1024 * 1024):
            uploaded_bytes += len(chunk)
            if uploaded_bytes > max_upload_bytes:
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {max_upload_size_mb} MB limit",
                )
            tmp.write(chunk)
        tmp_path = tmp.name

    background_tasks.add_task(_run_ingestion, tmp_path, file.filename, pipeline)
    logger.info("Ingestion queued for: %s", file.filename)

    return IngestionResponse(
        filename=file.filename,
        chunks_created=0,
        status="processing",
    )


async def _run_ingestion(tmp_path: str, filename: str, pipeline: IngestionPipeline) -> None:
    try:
        ids = await pipeline.ingest(tmp_path, source_name=filename)
        metrics.record_ingestion(success=True)
        logger.info("Ingestion complete: %s — %d chunks", filename, len(ids))
    except (IngestionError, UnsupportedFileTypeError) as e:
        metrics.record_ingestion(success=False)
        metrics.record_error("ingestion", type(e).__name__)
        logger.error("Ingestion failed for %s: %s", filename, e)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> DocumentListResponse:
    sources = vector_store.list_sources()
    return DocumentListResponse(
        documents=[DocumentInfo(**source) for source in sources],
        total=len(sources),
    )


@router.delete("/{source_file}", response_model=DeleteResponse)
async def delete_document(
    source_file: str,
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> DeleteResponse:
    try:
        vector_store.delete_by_source(source_file)
        return DeleteResponse(source_file=source_file, status="deleted")
    except Exception as e:
        logger.error("Delete failed for %s: %s", source_file, e)
        raise HTTPException(status_code=500, detail="Document deletion failed") from e
