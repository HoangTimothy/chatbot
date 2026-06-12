class RagError(Exception):
    """Base exception for RAG pipeline errors."""


class RetrievalNotConfiguredError(RagError):
    """Raised when retrieval infrastructure is required but unavailable."""

