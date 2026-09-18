# rag/indexing/index_store.py
"""
Creates and opens the searchable passage index on disk using ChromaDB.
"""
import json
import logging
import shutil
from datetime import UTC, datetime

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.errors import IndexMissingError
from rag.indexing.text_encoder import get_text_encoder
from rag.settings import (
    INDEX_INFO_FILE,
    INDEX_STORE_DIR,
    settings,
)

logger = logging.getLogger(__name__)


def create_searchable_index(passages: list[Document]) -> Chroma:
    """
    Build a fresh ChromaDB index from a list of text passages.

    Wipes any existing index before building - prevents silent duplication.
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
        collection_name=settings.index_collection_name,
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

    Does not re-embed anything - reads the existing vectors from disk.

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
        collection_name=settings.index_collection_name,
    )


def load_all_passages() -> list[Document]:
    """
    Reconstruct every passage stored in the ChromaDB index as LangChain Documents.

    This is the disk-fallback path used by app.py when the server has restarted
    and st.session_state["indexed_passages"] is gone but a valid index still
    exists on disk.  The BM25 retriever needs the raw passage corpus at
    construction time - this provides it without re-embedding anything.

    Implementation detail:
        We bypass the LangChain Chroma wrapper and talk directly to the
        chromadb PersistentClient.  The embedding function is deliberately
        NOT loaded here because fetching stored text + metadata requires
        no vector operations whatsoever.  Loading the encoder would add
        ~30 s of cold-start latency for zero benefit.

    Returns:
        List of Documents with page_content and metadata populated.
        Returns an empty list (not an exception) if the collection exists
        but holds no records - the caller decides what to do with that.

    Raises:
        IndexMissingError: The index directory does not exist or is empty,
                           meaning indexing has never been run.
    """
    if not is_index_ready():
        raise IndexMissingError(
            "No index found on disk. Upload source files and run indexing first."
        )

    client = chromadb.PersistentClient(path=str(INDEX_STORE_DIR))

    try:
        collection = client.get_collection(name=settings.index_collection_name)
    except Exception as exc:
        # Collection name mismatch or corrupted store - surface a clear message.
        raise IndexMissingError(
            f"Could not open ChromaDB collection '{settings.index_collection_name}': {exc}. "
            "The index may be corrupted - re-index your documents."
        ) from exc

    result = collection.get(include=["documents", "metadatas"])

    raw_texts: list[str] = result.get("documents") or []
    raw_metas: list[dict | None] = result.get("metadatas") or []

    if not raw_texts:
        logger.warning(
            "load_all_passages: collection '%s' exists but contains no records.",
            settings.index_collection_name,
        )
        return []

    # metadatas list is parallel to documents; individual entries can be None
    # when a document was stored without metadata - guard explicitly.
    passages = [
        Document(page_content=text, metadata=meta if meta is not None else {})
        for text, meta in zip(raw_texts, raw_metas)
    ]

    logger.info(
        "load_all_passages: restored %d passages from disk (server-restart path).",
        len(passages),
    )
    return passages


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
