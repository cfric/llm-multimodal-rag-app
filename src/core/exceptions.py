class RAGException(Exception):
    pass


class ConfigurationError(RAGException):
    pass


class DocumentLoadError(RAGException):
    pass


class UnsupportedFileTypeError(RAGException):
    pass


class EmbeddingError(RAGException):
    pass


class VectorStoreError(RAGException):
    pass


class LLMError(RAGException):
    pass


class RetrievalError(RAGException):
    pass


class RerankerError(RAGException):
    pass


class IngestionError(RAGException):
    pass


class ImageProcessingError(RAGException):
    pass


class TableProcessingError(RAGException):
    pass


class EvaluationError(RAGException):
    pass
