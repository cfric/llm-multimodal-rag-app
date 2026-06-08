import logging

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from src.core.exceptions import ConfigurationError
from src.llm.base import LLMBase

logger = logging.getLogger(__name__)


class AzureLLM(LLMBase):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        if not api_key:
            raise ConfigurationError("AZURE_OPENAI_API_KEY is not set")
        if not endpoint:
            raise ConfigurationError("LLM_AZURE_ENDPOINT is not set")
        if not deployment:
            raise ConfigurationError("LLM_AZURE_DEPLOYMENT is not set")
        self._api_key = api_key
        self._endpoint = endpoint
        self._deployment = deployment
        self._api_version = api_version
        super().__init__(deployment, temperature, max_tokens)

    def _build_model(self) -> BaseChatModel:
        return AzureChatOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._endpoint,
            azure_deployment=self._deployment,
            api_version=self._api_version,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
