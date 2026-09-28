from __future__ import annotations

import asyncio
from pathlib import Path

from src.api.dependencies import get_ingestion_pipeline, get_vector_store

DEMO_DOCUMENT = Path("data/demo/demo_document.md")
DEMO_IMAGE = Path("data/demo/generated/demo_diagram.png")


async def index_demo_documents() -> None:
    if not DEMO_DOCUMENT.exists() or not DEMO_IMAGE.exists():
        raise FileNotFoundError("Run `make demo-assets` before indexing the demo documents")

    vector_store = get_vector_store()
    pipeline = get_ingestion_pipeline()
    for path in (DEMO_DOCUMENT, DEMO_IMAGE):
        vector_store.delete_by_source(path.name)
        await pipeline.ingest(str(path), source_name=path.name)


if __name__ == "__main__":
    asyncio.run(index_demo_documents())
