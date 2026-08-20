class RAGError(Exception):
    pass


class ConfigurationError(RAGError):
    pass


class DocumentLoadError(RAGError):
    pass


class UnsupportedFileTypeError(RAGError):
    pass


class EmbeddingError(RAGError):
    pass


class VectorStoreError(RAGError):
    pass


class LLMError(RAGError):
    pass


class RetrievalError(RAGError):
    pass


class RerankerError(RAGError):
    pass


class IngestionError(RAGError):
    pass


class ImageProcessingError(RAGError):
    pass


class TableProcessingError(RAGError):
    pass


class EvaluationError(RAGError):
    pass
