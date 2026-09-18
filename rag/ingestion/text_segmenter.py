# rag/ingestion/text_segmenter.py
"""
Splits page-level Documents into smaller, searchable text passages using Chonkie.

"""
import hashlib
import logging
from langchain_core.documents import Document
from chonkie import SemanticChunker

from rag.settings import settings

logger = logging.getLogger(__name__)


_MINIMUM_PASSAGE_LENGTH = settings.minimum_segment_length


def segment_pages_into_passages(pages: list[Document]) -> list[Document]:
    """
    Split page-level Documents into deduplicated, prefixed text passages.

    Each returned passage has:
        page_content: "[Source: file.pdf | Page: N]\\n<passage text>"
        metadata:     source, page_number, total_pages, content_hash

    The content_hash in metadata enables incremental re-indexing - you can
    compare hashes to skip re-embedding passages that have not changed.

    Args:
        pages: Page-level Documents from read_source_files().

    Returns:
        Deduplicated list of passage Documents ready for embedding.
        Shorter than the input list - dedup removes identical boilerplate.

    Performance note:
        Chonkie SemanticChunker uses Model2Vec static embeddings for boundary
        detection by default. Model2Vec is ~500KB, CPU-only, and runs in
        milliseconds. BGE-M3 is NOT called during segmentation - only during
        the subsequent create_searchable_index() embedding step.
    """
    chunker = SemanticChunker(
        chunk_size=512,
        similarity_threshold=0.5,
    )

    all_passages: list[Document] = []

    for page in pages:
        source_name = page.metadata.get("source", "unknown")
        page_number  = page.metadata.get("page_number", "?")
        page_text    = page.page_content.strip()

        if not page_text:
            continue

        try:
            # Chonkie returns a list of Chunk objects with .text and .token_count
            chunks = chunker.chunk(page_text)
        except Exception as split_error:
            logger.warning(
                "Segmentation failed for '%s' page %s: %s",
                source_name,
                page_number,
                split_error,
            )
            continue

        for chunk in chunks:
            passage_text = chunk.text.strip()

            # Drop micro-passages: page numbers, isolated headers, single lines
            if len(passage_text) < _MINIMUM_PASSAGE_LENGTH:
                continue

            # Contextual prefix - bakes source identity into the embedding vector
            prefixed_content = (
                f"[Source: {source_name} | Page: {page_number}]\n{passage_text}"
            )

            all_passages.append(
                Document(
                    page_content=prefixed_content,
                    metadata={
                        **page.metadata,     
                        "token_count": chunk.token_count, 
                    },
                )
            )


    seen_hashes:     set[str]       = set()
    unique_passages: list[Document] = []

    for passage in all_passages:
        content_hash = hashlib.sha256(
            passage.page_content.encode("utf-8")
        ).hexdigest()

        if content_hash in seen_hashes:
            continue

        seen_hashes.add(content_hash)
        passage.metadata["content_hash"] = content_hash
        unique_passages.append(passage)

    duplicate_count = len(all_passages) - len(unique_passages)
    logger.info(
        "Segmentation complete: %d raw passages -> %d unique (%d duplicates removed).",
        len(all_passages),
        len(unique_passages),
        duplicate_count,
    )
    return unique_passages
