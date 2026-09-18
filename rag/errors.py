# rag/errors.py
"""
All custom exceptions for the RAG pipeline.

Rule: every exception name answers "what went wrong?" specifically.
Callers catch the type they care about — never a bare Exception.

PEP 8: exception classes use PascalCase with an 'Error' suffix.
"""


class RagError(Exception):
    """Root exception. All pipeline errors inherit from this."""


class ConfigurationError(RagError):
    """A required environment variable is missing or empty."""


class IndexMissingError(RagError):
    """The vector index does not exist on disk and cannot be loaded."""


class NoSourceFilesError(RagError):
    """No valid source files were found in the data directory."""


class FileRejectedError(RagError):
    """A file failed MIME, size, or content validation and was rejected."""


class QuotaExceededError(RagError):
    """A client sent more queries than the per-minute limit allows."""
