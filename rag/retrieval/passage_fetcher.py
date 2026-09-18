# rag/retrieval/passage_fetcher.py
"""
Builds the three-stage passage retrieval pipeline.

Stage 1 — Cast a wide net (40 candidates total):
    Semantic search (ChromaDB, top 20): finds passages similar in meaning.
    Keyword search (BM25, top 20):      finds passages sharing exact words.
    Together they cover what either misses alone.
    BM25 excels at exact identifiers: product codes, regulation numbers, names.
    Semantic search excels at paraphrase and synonyms.

Stage 2 — Merge the two lists:
    EnsembleRetriever uses Reciprocal Rank Fusion (RRF) to combine rankings.
    Weights: 70% semantic, 30% keyword. Adjust toward 60/40 for technical domains.

Stage 3 — Rerank to a precision set (top 5):
    CrossEncoderReranker scores each (question, passage) pair JOINTLY.
    Bi-encoders (semantic search) embed question and passage independently.
    Cross-encoders attend over both at once — far more precise.
    Typical improvement: +5 to +15 NDCG@10 over retrieval-only approaches.

Why LangChain's built-in CrossEncoderReranker (not a custom class):
    HuggingFaceCrossEncoder auto-selects CUDA > MPS > CPU.
    CrossEncoderReranker is tested, maintained, and LCEL-compatible.
    Our previous hand-rolled BaseRetriever subclass was 60 lines doing the same thing.

New joiner note:
    "Passage fetcher" means: give it a question, it returns the best passages.
    The returned object is a standard LangChain retriever — plug it into any chain.
"""

import logging

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

from rag.settings import (
    KEYWORD_RETRIEVAL_WEIGHT,
    KEYWORD_SEARCH_TOP_K,
    RERANKER_MODEL_NAME,
    SEMANTIC_RETRIEVAL_WEIGHT,
    SEMANTIC_SEARCH_TOP_K,
    TOP_PASSAGES_COUNT,
)

logger = logging.getLogger(__name__)


def create_passage_fetcher(
    index: Chroma,
    all_passages: list[Document],
) -> ContextualCompressionRetriever:
    """
    Assemble the three-stage retrieval + reranking pipeline.

    Args:
        index:        The ChromaDB collection built by create_searchable_index().
        all_passages: Every passage in the index — needed to build the BM25
                      in-memory keyword index. BM25 cannot be persisted to disk
                      in the same way as ChromaDB, so it is rebuilt each run.

    Returns:
        A ContextualCompressionRetriever compatible with any LCEL chain.
        Call .invoke(question_string) to get back a list of ranked Documents.
    """
    # Stage 1a: Semantic search — top 20 candidates by vector similarity
    semantic_search = index.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEMANTIC_SEARCH_TOP_K},
    )

    # Stage 1b: Keyword search — top 20 candidates by BM25 term frequency
    keyword_search = BM25Retriever.from_documents(
        all_passages,
        k=KEYWORD_SEARCH_TOP_K,
    )

    # Stage 2: Fuse both lists via Reciprocal Rank Fusion
    combined_search = EnsembleRetriever(
        retrievers=[semantic_search, keyword_search],
        weights=[SEMANTIC_RETRIEVAL_WEIGHT, KEYWORD_RETRIEVAL_WEIGHT],
    )

    # Stage 3: Cross-encoder reranker — precision over the fused 40 candidates
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
    passage_reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=TOP_PASSAGES_COUNT,
    )

    logger.debug(
        "Passage fetcher ready: semantic(k=%d) + keyword(k=%d) → reranker(top_%d).",
        SEMANTIC_SEARCH_TOP_K,
        KEYWORD_SEARCH_TOP_K,
        TOP_PASSAGES_COUNT,
    )

    return ContextualCompressionRetriever(
        base_compressor=passage_reranker,
        base_retriever=combined_search,
    )
