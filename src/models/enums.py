from enum import StrEnum


class FileType(StrEnum):
    TXT = "txt"
    MD = "md"
    PDF = "pdf"


class ResponseStatus(StrEnum):
    SUCCESS = "success"
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    PROCESSING_FAILED = "processing_failed"


class RetrievalSource(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class GenerationBackend(StrEnum):
    GEMINI = "gemini"
    OLLAMA = "ollama"


class EmbeddingBackend(StrEnum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    GEMINI = "gemini"
    REMOTE = "remote"  # any HTTP embedding server (e.g. Colab + ngrok)


class VectorDBBackend(StrEnum):
    QDRANT = "qdrant"


class DistanceMethod(StrEnum):
    COSINE = "cosine"
    DOT = "dot"

class RetrieverName(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"

