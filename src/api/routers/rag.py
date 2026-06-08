import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_rag_chain, get_session_memory
from src.api.schemas import QueryRequest, QueryResponse, SourceResponse
from src.core.exceptions import LLMError, RetrievalError
from src.monitoring import metrics
from src.rag.chain import MultimodalRAGChain
from src.rag.memory import ConversationMemory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    chain: MultimodalRAGChain = Depends(get_rag_chain),
    memory: ConversationMemory = Depends(get_session_memory),
) -> QueryResponse:
    try:
        memory.add_user_message(request.question)
        response = chain.invoke(request.question, memory.get_history())
        memory.add_assistant_message(response.answer)

        metrics.record_query(
            model=response.model,
            latency_seconds=response.latency_ms / 1000,
            num_docs=len(response.sources),
        )

        return QueryResponse(
            answer=response.answer,
            sources=[
                SourceResponse(
                    type=s.type,
                    content=s.content,
                    source_file=s.source_file,
                    page=s.page,
                    image_path=s.image_path,
                    caption=s.caption,
                    score=s.score,
                )
                for s in response.sources
            ],
            query=response.query,
            model=response.model,
            latency_ms=response.latency_ms,
        )
    except (LLMError, RetrievalError) as e:
        metrics.record_error("rag", type(e).__name__)
        logger.error("RAG query failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/stream")
async def stream_query(
    request: QueryRequest,
    chain: MultimodalRAGChain = Depends(get_rag_chain),
    memory: ConversationMemory = Depends(get_session_memory),
) -> StreamingResponse:
    memory.add_user_message(request.question)

    async def generate():
        full_response = []
        async for chunk in chain.astream(request.question, memory.get_history()):
            full_response.append(chunk)
            yield f"data: {chunk}\n\n"
        memory.add_assistant_message("".join(full_response))

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    memory: ConversationMemory = Depends(get_session_memory),
) -> dict:
    memory.clear()
    return {"session_id": session_id, "status": "cleared"}
