from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


@dataclass
class Source:
    type: Literal["text", "image", "table"]
    content: str
    source_file: str
    page: int | None = None
    image_path: str | None = None
    caption: str | None = None
    summary: str | None = None
    score: float | None = None


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source]
    query: str
    model: str
    latency_ms: float


@dataclass
class ChatMessage:
    role: Literal["human", "assistant"]
    content: str


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        pass


class BaseProcessor(ABC):
    @abstractmethod
    async def process(self, documents: list[Document]) -> list[Document]:
        pass


class BaseEmbedder(ABC):
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        pass


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass

    @abstractmethod
    def list_sources(self) -> list[dict]:
        pass

    @abstractmethod
    def delete_by_source(self, source_file: str) -> None:
        pass


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> list[Document]:
        pass


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        pass


class BaseLLM(ABC):
    @abstractmethod
    def get_model(self) -> BaseChatModel:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class BaseRAGChain(ABC):
    @abstractmethod
    def invoke(self, query: str, chat_history: list[ChatMessage]) -> RAGResponse:
        pass

    @abstractmethod
    async def astream(self, query: str, chat_history: list[ChatMessage]) -> AsyncIterator[str]:
        pass


class BaseUI(ABC):
    @abstractmethod
    def run(self) -> None:
        pass
