# rag/settings.py
"""
Single source of truth for every constant and environment variable.

Rules enforced here:
  - Constants:  UPPER_SNAKE_CASE  (PEP 8 §Constants)
  - Paths:      pathlib.Path only — never raw strings
  - API keys:   read once here, never via os.getenv() elsewhere
  - Validation: validate_environment() called once at startup

New joiner guide:
  Need to change a threshold or model name? Change it here.
  Nothing else needs to change — no grep required.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from rag.errors import ConfigurationError

load_dotenv()

# ── On-disk locations ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_FILES_DIR = BASE_DIR / "data"
INDEX_STORE_DIR = BASE_DIR / "index_store"
AUDIT_LOG_FILE = BASE_DIR / "file_audit.json"
INDEX_INFO_FILE = INDEX_STORE_DIR / "index_info.json"
THROTTLE_LOG_FILE = BASE_DIR / ".throttle_log.json"

# ── Model names ───────────────────────────────────────────────────────────────
# Text encoder: converts passages to vectors for semantic search
ENCODER_MODEL_NAME = "BAAI/bge-m3"

# Passage reranker: jointly scores (question, passage) pairs for precision
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Primary LLM: generates the final grounded answer shown to the user
PRIMARY_MODEL_NAME = "llama-3.3-70b-versatile"

# Fast classifier LLM: used for HyDE, quality gate, and groundedness check.
# Cheap, fast. llama-3.1-8b-instant was RETIRED on Groq free tier Aug 16 2026.
# qwen/qwen3-8b is the confirmed replacement as of Sep 2026.
FAST_MODEL_NAME = "qwen/qwen3-8b"

# ── Primary LLM generation settings ──────────────────────────────────────────
# temperature=0 → deterministic output — answers must be grounded, not creative
PRIMARY_MODEL_TEMPERATURE = 0
PRIMARY_MODEL_MAX_TOKENS = 1024

# Fast model settings — used only for short classification tasks
FAST_MODEL_TEMPERATURE = 0
FAST_MODEL_MAX_TOKENS = 200

# ── Retrieval tuning ──────────────────────────────────────────────────────────
# Cast a wide net first (20 + 20), then rerank down to the best 5.
# Rule of thumb: retrieve 20, rerank to 5, feed 5 to the LLM.
SEMANTIC_SEARCH_TOP_K = 20  # Dense vector search candidate count
KEYWORD_SEARCH_TOP_K = 20  # BM25 keyword search candidate count
TOP_PASSAGES_COUNT = 5  # Final passages delivered to the LLM
ENCODER_BATCH_SIZE = 32  # Passages encoded per GPU/CPU forward pass

# Retrieval fusion weights: semantic 70%, keyword 30%.
# Increase keyword weight for domains with many exact codes/IDs (e.g. legal, medical).
SEMANTIC_RETRIEVAL_WEIGHT = 0.7
KEYWORD_RETRIEVAL_WEIGHT = 0.3

# ChromaDB collection identifier
INDEX_COLLECTION_NAME = "source_passages"

# ── Text segmentation settings ────────────────────────────────────────────────
# Semantic chunker finds natural topic boundaries instead of splitting at fixed
# character counts. 95th percentile drops a new boundary at the largest
# similarity gaps — keeps semantically coherent segments together.
SEGMENT_BOUNDARY_METHOD = "percentile"
SEGMENT_BOUNDARY_STRENGTH = 95

# Segments shorter than this are noise (page numbers, headers). Drop them.
MINIMUM_SEGMENT_LENGTH = 50  # characters

# ── Security limits ───────────────────────────────────────────────────────────
ALLOWED_MIME_TYPES = {"application/pdf", "text/plain"}
MAX_UPLOAD_SIZE_MB = 50
MAX_QUESTIONS_PER_MIN = 10

# ── Index freshness ───────────────────────────────────────────────────────────
# Warn the user if the index was built more than this many hours ago.
STALE_INDEX_THRESHOLD_HOURS = 24

# ── API credentials ───────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def validate_environment() -> None:
    """
    Confirm all required environment variables are present.

    Called once at application startup before any component initialises.
    Raises ConfigurationError with the NAMES of missing keys — never their values.
    A named exception (not bare Exception) lets the caller handle config
    failures distinctly from runtime failures.

    Returns:
        None — raises on failure, returns silently on success.

    Raises:
        ConfigurationError: One or more required variables are not set.
    """
    required = {"GROQ_API_KEY": GROQ_API_KEY}
    missing = [name for name, value in required.items() if not value]

    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {missing}. "
            "Set them in .env for local development or in Streamlit secrets for deployment."
        )
