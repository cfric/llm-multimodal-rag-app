import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

from src.core.exceptions import ConfigurationError
from src.llm.base import LLMBase

logger = logging.getLogger(__name__)


class HuggingFaceLLM(LLMBase):
    def __init__(
        self,
        model: str,
        mode: Literal["api", "local"],
        api_key: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        if mode == "api" and not api_key:
            raise ConfigurationError("HUGGINGFACE_API_KEY is not set")
        self._mode = mode
        self._api_key = api_key
        super().__init__(model, temperature, max_tokens)

    def _build_model(self) -> BaseChatModel:
        if self._mode == "api":
            endpoint = HuggingFaceEndpoint(
                repo_id=self._model_name,
                huggingfacehub_api_token=self._api_key,
                temperature=self._temperature,
                max_new_tokens=self._max_tokens,
            )
            return ChatHuggingFace(llm=endpoint)
        else:
            pipeline = HuggingFacePipeline.from_model_id(
                model_id=self._model_name,
                task="text-generation",
                pipeline_kwargs={
                    "temperature": self._temperature,
                    "max_new_tokens": self._max_tokens,
                    "do_sample": self._temperature > 0,
                },
            )
            return ChatHuggingFace(llm=pipeline)
