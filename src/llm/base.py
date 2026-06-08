import logging
from abc import abstractmethod

from langchain_core.language_models import BaseChatModel

from src.core.interfaces import BaseLLM

logger = logging.getLogger(__name__)


class LLMBase(BaseLLM):
    def __init__(self, model_name: str, temperature: float, max_tokens: int) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._model: BaseChatModel | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = self._build_model()
            logger.info("LLM initialized: %s", self._model_name)
        return self._model

    @abstractmethod
    def _build_model(self) -> BaseChatModel:
        pass
