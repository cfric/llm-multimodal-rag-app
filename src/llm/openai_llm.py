import logging

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.core.exceptions import ConfigurationError
from src.llm.base import LLMBase

logger = logging.getLogger(__name__)


class OpenAILLM(LLMBase):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set")
        self._api_key = api_key
        super().__init__(model, temperature, max_tokens)

    def _build_model(self) -> BaseChatModel:
        return ChatOpenAI(
            api_key=SecretStr(self._api_key),
            model=self._model_name,
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
        )
