from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    provider: Literal["openai", "azure", "ollama", "huggingface"] = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2048
    hf_mode: Literal["api", "local"] = "api"
    azure_deployment: str = ""
    azure_api_version: str = "2024-02-01"
    azure_endpoint: str = ""
    ollama_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="allow")


class EmbeddingSettings(BaseSettings):
    provider: Literal["openai", "huggingface"] = "openai"
    model: str = "text-embedding-3-large"

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="allow")


class VectorStoreSettings(BaseSettings):
    provider: Literal["chroma", "qdrant"] = "chroma"
    persist_directory: str = "./data/vectorstore"
    collection_name: str = "multimodal_rag"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    model_config = SettingsConfigDict(env_prefix="VECTORSTORE_", env_file=".env", extra="allow")


class RerankerSettings(BaseSettings):
    enabled: bool = True
    provider: Literal["crossencoder", "cohere"] = "crossencoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    cohere_api_key: str = ""

    model_config = SettingsConfigDict(env_prefix="RERANKER_", env_file=".env", extra="allow")


class RetrievalSettings(BaseSettings):
    k: int = 20
    search_type: Literal["similarity", "mmr"] = "similarity"

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", env_file=".env", extra="allow")


class MonitoringSettings(BaseSettings):
    enabled: bool = True
    prometheus_port: int = 8001

    model_config = SettingsConfigDict(env_prefix="MONITORING_", env_file=".env", extra="allow")


class LangSmithSettings(BaseSettings):
    enabled: bool = False
    project: str = "multimodal-rag"

    model_config = SettingsConfigDict(env_prefix="LANGSMITH_", env_file=".env", extra="allow")


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="allow")


class Settings(BaseSettings):
    openai_api_key: str = ""
    azure_openai_api_key: str = ""
    huggingface_api_key: str = ""
    langsmith_api_key: str = ""

    data_dir: str = "./data"
    documents_dir: str = "./data/documents"
    images_dir: str = "./data/images"
    logs_dir: str = "./logs"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    api: APISettings = Field(default_factory=APISettings)

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @classmethod
    def from_yaml(cls, yaml_path: str = "config/config.yaml") -> "Settings":
        path = Path(yaml_path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        llm_data = data.pop("llm", {})
        embedding_data = data.pop("embedding", {})
        vectorstore_data = data.pop("vectorstore", {})
        reranker_data = data.pop("reranker", {})
        retrieval_data = data.pop("retrieval", {})
        monitoring_data = data.pop("monitoring", {})
        langsmith_data = data.pop("langsmith", {})
        api_data = data.pop("api", {})
        return cls(
            **data,
            llm=LLMSettings(**llm_data),
            embedding=EmbeddingSettings(**embedding_data),
            vectorstore=VectorStoreSettings(**vectorstore_data),
            reranker=RerankerSettings(**reranker_data),
            retrieval=RetrievalSettings(**retrieval_data),
            monitoring=MonitoringSettings(**monitoring_data),
            langsmith=LangSmithSettings(**langsmith_data),
            api=APISettings(**api_data),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
    return _settings
