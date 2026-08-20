import logging
from functools import lru_cache

from src.config.settings import get_settings
from src.core.interfaces import BaseVectorStore
from src.embeddings.factory import EmbedderFactory
from src.ingestion.loaders.image_loader import ImageLoader
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.loaders.text_loader import TextLoader
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.processors.image_processor import ImageProcessor
from src.ingestion.processors.table_processor import TableProcessor
from src.ingestion.processors.text_processor import TextProcessor
from src.llm.factory import LLMFactory
from src.rag.chain import MultimodalRAGChain
from src.rag.memory import ConversationMemory
from src.retrieval.reranker import RerankerFactory
from src.retrieval.semantic_retriever import SemanticRetriever
from src.vectorstore.factory import VectorStoreFactory

logger = logging.getLogger(__name__)

_sessions: dict[str, ConversationMemory] = {}


@lru_cache(maxsize=1)
def _get_vector_store() -> BaseVectorStore:
    settings = get_settings()
    embedder = EmbedderFactory.create(settings)
    return VectorStoreFactory.create(settings, embedder)


@lru_cache(maxsize=1)
def _get_rag_chain() -> MultimodalRAGChain:
    settings = get_settings()
    vector_store = _get_vector_store()
    retriever = SemanticRetriever(vector_store)
    llm = LLMFactory.create(settings)

    reranker = None
    if settings.reranker.enabled:
        reranker = RerankerFactory.create(
            provider=settings.reranker.provider,
            model=settings.reranker.model,
            cohere_api_key=settings.reranker.cohere_api_key,
        )

    return MultimodalRAGChain(
        retriever=retriever,
        llm=llm,
        reranker=reranker,
        retrieval_k=settings.retrieval.k,
    )


@lru_cache(maxsize=1)
def _get_ingestion_pipeline() -> IngestionPipeline:
    settings = get_settings()
    embedder = EmbedderFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, embedder)
    vision_llm = LLMFactory.create(settings).get_model()

    return IngestionPipeline(
        loaders=[PDFLoader(images_dir=settings.images_dir), ImageLoader(), TextLoader()],
        text_processor=TextProcessor(),
        image_processor=ImageProcessor(vision_llm, settings.images_dir),
        table_processor=TableProcessor(vision_llm),
        vector_store=vector_store,
    )


def get_vector_store() -> BaseVectorStore:
    return _get_vector_store()


def get_rag_chain() -> MultimodalRAGChain:
    return _get_rag_chain()


def get_ingestion_pipeline() -> IngestionPipeline:
    return _get_ingestion_pipeline()


def get_session_memory(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]
