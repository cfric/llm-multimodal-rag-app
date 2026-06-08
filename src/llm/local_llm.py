import logging

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.llm.base import LLMBase

logger = logging.getLogger(__name__)


class LocalLLM(LLMBase):
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int) -> None:
        self._base_url = base_url
        super().__init__(model, temperature, max_tokens)

    def _build_model(self) -> BaseChatModel:
        return ChatOllama(
            base_url=self._base_url,
            model=self._model_name,
            temperature=self._temperature,
            num_predict=self._max_tokens,
        )
