from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.interfaces import BaseEmbedder
from src.vectorstore.chroma_store import ChromaVectorStore


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(map(ord, text)) % 101)]


class TestEmbedder(BaseEmbedder):
    def get_embeddings(self) -> Embeddings:
        return DeterministicEmbeddings()


class TestChromaVectorStoreIntegration:
    def test_add_search_and_delete_documents(self, tmp_path) -> None:
        store = ChromaVectorStore(
            embedder=TestEmbedder(),
            persist_directory=str(tmp_path / "chroma"),
            collection_name="documents",
            min_relevance_score=-1.0,
        )
        store.add_documents(
            [Document(page_content="Transformer attention", metadata={"source_file": "paper.pdf"})]
        )

        results = store.similarity_search("Transformer attention", k=1)

        assert len(results) == 1
        assert results[0].metadata["source_file"] == "paper.pdf"
        assert store.list_sources() == [{"source_file": "paper.pdf", "type": "text"}]

        store.delete_by_source("paper.pdf")

        assert store.list_sources() == []
