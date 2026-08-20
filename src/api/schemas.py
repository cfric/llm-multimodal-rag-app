from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=64)


class SourceResponse(BaseModel):
    type: str
    content: str
    source_file: str
    page: int | None = None
    image_path: str | None = None
    caption: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    query: str
    model: str
    latency_ms: float
    session_id: str


class IngestionResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str


class DocumentInfo(BaseModel):
    source_file: str
    type: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    source_file: str
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
    vectorstore: str
    llm_provider: str
