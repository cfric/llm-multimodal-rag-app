import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _merge_environment_overrides[SettingsType: BaseSettings](
    settings_type: type[SettingsType], yaml_values: dict[str, Any]
) -> dict[str, Any]:
    """Apply .env and process values over the YAML defaults for one settings section."""
    env_prefix = str(settings_type.model_config.get("env_prefix", ""))
    environment = {**dotenv_values(".env"), **os.environ}
    values = dict(yaml_values)
    for field_name in settings_type.model_fields:
        env_value = environment.get(f"{env_prefix}{field_name.upper()}")
        if env_value is not None:
            values[field_name] = env_value
    return values


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
    provider: Literal["chroma"] = "chroma"
    persist_directory: str = "./data/vectorstore"
    collection_name: str = "multimodal_rag"
    min_relevance_score: float = 0.35
    image_min_relevance_score: float = 0.55
    chroma_host: str = ""
    chroma_port: int = 8000
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
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    max_upload_size_mb: int = 25

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
            **_merge_environment_overrides(cls, data),
            llm=LLMSettings(**_merge_environment_overrides(LLMSettings, llm_data)),
            embedding=EmbeddingSettings(
                **_merge_environment_overrides(EmbeddingSettings, embedding_data)
            ),
            vectorstore=VectorStoreSettings(
                **_merge_environment_overrides(VectorStoreSettings, vectorstore_data)
            ),
            reranker=RerankerSettings(
                **_merge_environment_overrides(RerankerSettings, reranker_data)
            ),
            retrieval=RetrievalSettings(
                **_merge_environment_overrides(RetrievalSettings, retrieval_data)
            ),
            monitoring=MonitoringSettings(
                **_merge_environment_overrides(MonitoringSettings, monitoring_data)
            ),
            langsmith=LangSmithSettings(
                **_merge_environment_overrides(LangSmithSettings, langsmith_data)
            ),
            api=APISettings(**_merge_environment_overrides(APISettings, api_data)),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
    return _settings
