# rag/indexing/index_store.py
"""
Creates and opens the searchable passage index (ChromaDB on disk).

What a searchable index is:
    A ChromaDB collection where every passage has been converted to a vector
    and stored alongside its original text and metadata.
    Given a question vector, ChromaDB finds the nearest passage vectors quickly.

Two public operations:
    create_searchable_index() — builds from scratch, wipes any existing index.
    open_existing_index()     — opens what is already on disk. No re-embedding.

Why we always wipe before building:
    ChromaDB appends by default. If you re-index without wiping, every passage
    gets duplicated. Duplicate passages skew retrieval scores silently.
    Wiping first ensures the index reflects exactly the current set of documents.

New joiner note:
    hnsw:space = "cosine" tells ChromaDB to use cosine similarity for search.
    If you leave this unset, it defaults to L2 (Euclidean distance), which gives
    incorrect results for normalised vectors. Always set this explicitly.
"""

import json
import logging
import shutil
from datetime import UTC, datetime

from langchain.schema import Document
from langchain_chroma import Chroma

from rag.errors import IndexMissingError
from rag.indexing.text_encoder import get_text_encoder
from rag.settings import (
    INDEX_COLLECTION_NAME,
    INDEX_INFO_FILE,
    INDEX_STORE_DIR,
)

logger = logging.getLogger(__name__)


def create_searchable_index(passages: list[Document]) -> Chroma:
    """
    Build a fresh ChromaDB index from a list of text passages.

    Wipes any existing index before building — prevents silent duplication.
    Writes an index_info.json file recording when the index was built and
    which source files it contains.

    Args:
        passages: Deduplicated, prefixed passages from segment_pages_into_passages().

    Returns:
        The newly built Chroma collection, ready for similarity search.
    """
    if INDEX_STORE_DIR.exists():
        shutil.rmtree(INDEX_STORE_DIR)
        logger.info("Removed previous index at '%s'.", INDEX_STORE_DIR)

    INDEX_STORE_DIR.mkdir(parents=True, exist_ok=True)

    index = Chroma.from_documents(
        documents=passages,
        embedding=get_text_encoder(),
        persist_directory=str(INDEX_STORE_DIR),
        collection_name=INDEX_COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    _save_index_info(
        passage_count=len(passages),
        source_files=sorted({p.metadata.get("source", "unknown") for p in passages}),
    )
    logger.info(
        "Index created: %d passages from %d files.",
        len(passages),
        len({p.metadata.get("source") for p in passages}),
    )
    return index


def open_existing_index() -> Chroma:
    """
    Open the index that was previously built and saved to disk.

    Does not re-embed anything — reads the existing vectors from disk.

    Returns:
        The Chroma collection ready for similarity search.

    Raises:
        IndexMissingError: The index directory does not exist or is empty.
    """
    if not is_index_ready():
        raise IndexMissingError(
            "No index found on disk. Upload source files and run indexing first."
        )
    return Chroma(
        persist_directory=str(INDEX_STORE_DIR),
        embedding_function=get_text_encoder(),
        collection_name=INDEX_COLLECTION_NAME,
    )


def is_index_ready() -> bool:
    """Return True if a non-empty index exists on disk."""
    return INDEX_STORE_DIR.exists() and any(INDEX_STORE_DIR.iterdir())


def get_index_info() -> dict | None:
    """
    Return metadata about the current index, or None if no index exists.

    The returned dict contains:
        built_at:      ISO 8601 timestamp of when the index was built.
        passage_count: Number of passages in the index.
        source_files:  List of filenames that were indexed.
    """
    if not INDEX_INFO_FILE.exists():
        return None
    return json.loads(INDEX_INFO_FILE.read_text())


def compute_index_staleness_hours() -> float | None:
    """
    Return how many hours have passed since the index was last built.

    Returns None if no index info file exists (index was never built).
    Used by the UI to warn when the index may not reflect the latest uploads.
    """
    info = get_index_info()
    if not info:
        return None
    built_at = datetime.fromisoformat(info["built_at"])
    return (datetime.now(UTC) - built_at).total_seconds() / 3600


def _save_index_info(passage_count: int, source_files: list[str]) -> None:
    """Write index metadata to disk for freshness tracking."""
    INDEX_INFO_FILE.write_text(
        json.dumps(
            {
                "built_at": datetime.now(UTC).isoformat(),
                "passage_count": passage_count,
                "source_files": source_files,
            },
            indent=2,
        )
    )
