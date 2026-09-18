# rag/indexing/text_encoder.py
"""
Provides the BGE-M3 text encoder as a module-level singleton.

What a text encoder does:
    It converts a string of text into a list of floating-point numbers (a vector).
    Texts that mean similar things produce vectors that are close in space.
    This is what makes semantic search possible.

Why a module-level singleton:
    BGE-M3 weighs about 550 MB on disk.
    Loading it takes 20-30 seconds on first call.
    Keeping it in a module-level variable means it loads once per process.
    Streamlit's @st.cache_resource at the app layer ensures this for the UI too.

Why langchain-huggingface (not langchain-community):
    LangChain 1.x split HuggingFace integrations into their own package.
    The old langchain_community.embeddings classes emit DeprecationWarning.

Why normalize_embeddings=True:
    Cosine similarity is only meaningful when vectors are unit-normalised.
    ChromaDB uses cosine similarity for our collection. Normalise here,
    or every similarity score is wrong.

BGE-M3 does not need a query prefix instruction — it was trained without one.
"""

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from rag.settings import ENCODER_BATCH_SIZE, ENCODER_MODEL_NAME

logger = logging.getLogger(__name__)

# Single instance — lives for the lifetime of the Python process
_loaded_encoder: HuggingFaceEmbeddings | None = None


def get_text_encoder() -> HuggingFaceEmbeddings:
    """
    Return the BGE-M3 text encoder, loading it on first call.

    Subsequent calls return the same instance instantly — no re-loading.

    Returns:
        A HuggingFaceEmbeddings instance ready to encode text into vectors.
    """
    global _loaded_encoder

    if _loaded_encoder is None:
        logger.info(
            "Loading text encoder '%s' for the first time (~30s)...",
            ENCODER_MODEL_NAME,
        )
        _loaded_encoder = HuggingFaceEmbeddings(
            model_name=ENCODER_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": ENCODER_BATCH_SIZE,
            },
        )
        logger.info("Text encoder ready.")

    return _loaded_encoder
