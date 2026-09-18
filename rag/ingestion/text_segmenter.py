# rag/ingestion/text_segmenter.py
"""
Splits page-level Documents into smaller, searchable text passages.

Why we split at all:
    A full PDF page is often too large to embed meaningfully.
    A vector of a 3000-word page captures the general topic, not specific facts.
    Smaller passages produce sharper vectors that match specific questions better.

Why semantic splitting (not fixed-size):
    Fixed-size splitting cuts at arbitrary character positions — mid-sentence,
    mid-table, mid-equation. The resulting passages lose coherence.
    SemanticChunker measures cosine similarity between adjacent sentences.
    When similarity drops sharply, it inserts a boundary. Passages stay coherent.

Why the source prefix on every passage:
    When a passage is embedded, its vector encodes the text content only.
    Prepending "[Source: report.pdf | Page: 3]" bakes provenance INTO the vector.
    The retriever then finds passages that are both topically relevant AND traceable.
    This is the Anthropic "contextual retrieval" technique from their 2024 paper.

Why deduplication:
    Legal disclaimers, footers, and cover page text appear identically across
    many documents. Without dedup, the index fills with redundant passages that
    waste space and dilute retrieval quality.

New joiner note:
    "Passage" is the standard NLP term for a retrieved text segment.
    It is more precise than "chunk" (which implies arbitrary size).
"""

import hashlib
import logging

from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker

from rag.indexing.text_encoder import get_text_encoder
from rag.settings import (
    MINIMUM_SEGMENT_LENGTH,
    SEGMENT_BOUNDARY_METHOD,
    SEGMENT_BOUNDARY_STRENGTH,
)

logger = logging.getLogger(__name__)


def segment_pages_into_passages(pages: list[Document]) -> list[Document]:
    """
    Split page-level Documents into deduplicated, prefixed text passages.

    Each returned passage has:
        page_content: "[Source: file.pdf | Page: N]\\n<passage text>"
        metadata:     source, page_number, total_pages, content_hash

    The content_hash in metadata supports incremental re-indexing — you can
    skip re-embedding passages whose hash already exists in the index.

    Args:
        pages: Page-level Documents from read_source_files().

    Returns:
        Deduplicated list of passage Documents ready for embedding.
        Shorter than the input list (dedup removes identical boilerplate).
    """
    encoder = get_text_encoder()

    splitter = SemanticChunker(
        embeddings=encoder,
        breakpoint_threshold_type=SEGMENT_BOUNDARY_METHOD,
        breakpoint_threshold_amount=SEGMENT_BOUNDARY_STRENGTH,
    )

    all_passages: list[Document] = []

    for page in pages:
        source_name = page.metadata.get("source", "unknown")
        page_number = page.metadata.get("page_number", "?")

        try:
            segments = splitter.create_documents(
                texts=[page.page_content],
                metadatas=[page.metadata],
            )
        except Exception as split_error:
            logger.warning(
                "Segmentation failed for '%s' page %s: %s",
                source_name,
                page_number,
                split_error,
            )
            continue

        for segment in segments:
            passage_text = segment.page_content.strip()

            # Drop micro-segments: page numbers, headers, single-line footers
            if len(passage_text) < MINIMUM_SEGMENT_LENGTH:
                continue

            # Prepend source context — bakes provenance into the embedding vector
            segment.page_content = f"[Source: {source_name} | Page: {page_number}]\n{passage_text}"
            all_passages.append(segment)

    # SHA-256 deduplication — identical text across documents → keep only first
    seen_hashes: set[str] = set()
    unique_passages: list[Document] = []

    for passage in all_passages:
        content_hash = hashlib.sha256(passage.page_content.encode("utf-8")).hexdigest()

        if content_hash in seen_hashes:
            continue

        seen_hashes.add(content_hash)
        passage.metadata["content_hash"] = content_hash
        unique_passages.append(passage)

    duplicate_count = len(all_passages) - len(unique_passages)
    logger.info(
        "Segmentation complete: %d raw passages → %d unique (%d duplicates removed).",
        len(all_passages),
        len(unique_passages),
        duplicate_count,
    )
    return unique_passages
