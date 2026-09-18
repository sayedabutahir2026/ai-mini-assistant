# rag/indexing/text_encoder.py
"""
Provides the BGE-M3 text encoder as a module-level singleton.
"""
import logging
import os

from langchain_huggingface import HuggingFaceEmbeddings

from rag.settings import settings

logger = logging.getLogger(__name__)

_loaded_encoder: HuggingFaceEmbeddings | None = None


def _authenticate_huggingface() -> None:
    """
    Authenticate with HuggingFace Hub using HF_TOKEN from environment.

    Called once at module load time — before any model download.
    huggingface_hub.login() is safe to call multiple times; subsequent
    calls are no-ops when already authenticated.

    Why not rely on CLI login alone:
        huggingface-cli login writes to ~/.cache/huggingface/token.
        If HF_HOME is set, the path changes and the token is not found.
        Calling login() with the token directly bypasses path resolution.
    """
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            logger.info("HuggingFace Hub: authenticated via HF_TOKEN.")
        except Exception as auth_error:
            logger.warning(
                "HuggingFace Hub authentication failed: %s. "
                "Continuing with anonymous access.",
                auth_error,
            )
    else:
        logger.debug(
            "HF_TOKEN not set — using anonymous HuggingFace Hub access. "
            "Rate limits apply. Set HF_TOKEN in .env to authenticate."
        )


# Authenticate once at module import time — before any model request
_authenticate_huggingface()


def get_text_encoder() -> HuggingFaceEmbeddings:
    """
    Return the BGE-M3 text encoder, loading it on first call.
    Subsequent calls return the same instance instantly.
    """
    global _loaded_encoder
    if _loaded_encoder is None:
        logger.info(
            "Loading text encoder '%s' (first load — downloads once, then cached)...",
            settings.encoder_model_name,
        )
        _loaded_encoder = HuggingFaceEmbeddings(
            model_name=settings.encoder_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": settings.encoder_batch_size,
            },
        )
        logger.info("Text encoder ready.")
    return _loaded_encoder
