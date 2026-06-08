from collections import deque

from src.core.interfaces import ChatMessage


class ConversationMemory:
    def __init__(self, max_messages: int = 20) -> None:
        self._history: deque[ChatMessage] = deque(maxlen=max_messages)

    def add_user_message(self, content: str) -> None:
        self._history.append(ChatMessage(role="human", content=content))

    def add_assistant_message(self, content: str) -> None:
        self._history.append(ChatMessage(role="assistant", content=content))

    def get_history(self) -> list[ChatMessage]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)
