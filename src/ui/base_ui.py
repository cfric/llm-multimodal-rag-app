from abc import ABC, abstractmethod


class BaseUI(ABC):
    @abstractmethod
    def run(self) -> None:
        pass
