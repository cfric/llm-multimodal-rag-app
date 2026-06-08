import logging

from src.config.settings import Settings
from src.core.exceptions import ConfigurationError
from src.core.interfaces import BaseLLM

logger = logging.getLogger(__name__)


class LLMFactory:
    @staticmethod
    def create(settings: Settings) -> BaseLLM:
        provider = settings.llm.provider
        logger.info("Creating LLM provider: %s", provider)

        if provider == "openai":
            from src.llm.openai_llm import OpenAILLM
            return OpenAILLM(
                api_key=settings.openai_api_key,
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )

        if provider == "azure":
            from src.llm.azure_llm import AzureLLM
            return AzureLLM(
                api_key=settings.azure_openai_api_key,
                endpoint=settings.llm.azure_endpoint,
                deployment=settings.llm.azure_deployment,
                api_version=settings.llm.azure_api_version,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )

        if provider == "ollama":
            from src.llm.local_llm import LocalLLM
            return LocalLLM(
                base_url=settings.llm.ollama_base_url,
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )

        if provider == "huggingface":
            from src.llm.huggingface_llm import HuggingFaceLLM
            return HuggingFaceLLM(
                model=settings.llm.model,
                mode=settings.llm.hf_mode,
                api_key=settings.huggingface_api_key,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )

        raise ConfigurationError(f"Unknown LLM provider: {provider}")
