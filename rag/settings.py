# rag/settings.py

from pathlib import Path

import litellm
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.errors import ConfigurationError


class _PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    primary_model_name: str = Field(
        default="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
        description=(
            "LiteLLM model string for answer generation. "
            "NVIDIA NIM format: nvidia_nim/<model-id-from-build.nvidia.com>. "
            "Override via PRIMARY_MODEL_NAME env var."
        ),
    )
    fast_model_name: str = Field(
        default="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
        description=(
            "LiteLLM model string for HyDE and groundedness scoring. "
            "Override via FAST_MODEL_NAME env var."
        ),
    )

    nvidia_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("nvidia_api_key", "NVIDIA_API_KEY"),
        description=(
            "NVIDIA NIM API key from build.nvidia.com. "
        ),
    )
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google Gemini API key. Required for gemini/* models.",
    )
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Groq API key. Required for groq/* models.",
    )
    openrouter_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenRouter API key. Required for openrouter/* models.",
    )
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API key. Required for anthropic/* models.",
    )
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key. Required for openai/* models.",
    )

    primary_model_temperature: float = Field(
        default=0,
        description="Sampling temperature. 0 = deterministic.",
    )
    primary_model_max_tokens: int = Field(
        default=4096,
        description=(
            "Maximum tokens in the primary LLM response. "
            "High value needed for reasoning models - thinking tokens "
            "are consumed before the actual answer."
        ),
    )

    fast_model_temperature: float = Field(
        default=0,
        description="Sampling temperature for fast classification calls.",
    )
    fast_model_max_tokens: int = Field(
        default=512,
        description=(
            "Maximum tokens for fast classifier responses. "
            "Groundedness scoring needs ~100 tokens for a score + sentence. "
            "HyDE needs ~200-400 tokens for a synthetic passage. "
            "512 is sufficient and keeps latency low."
        ),
    )

    encoder_model_name: str = Field(
        default="BAAI/bge-m3",
        description=(
            "HuggingFace model ID for text-to-vector encoding. "
            "Runs locally on CPU. ~2.27 GB download once. "
            "Override via ENCODER_MODEL_NAME env var."
        ),
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description=(
            "HuggingFace cross-encoder model ID for passage reranking. "
            "MiniLM-L-6-v2: 22MB, ~2-3s on CPU for 20 candidates. "
            "Replaces bge-reranker-v2-m3 (568MB, 25s on CPU for 40 candidates). "
            "Quality difference for English RAG is negligible at this pipeline stage "
            "since BGE-M3 + BM25 + RRF already pre-filtered to highly relevant passages. "
            "To restore maximum quality: set RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3"
        ),
    )

    semantic_search_top_k: int = Field(
        default=10,
        description=(
            "Dense vector search candidates before reranking. "
            "Reduced from 20 to 10 — CrossEncoder time is O(n), "
            "10+10=20 candidates vs 20+20=40 halves reranker latency. "
            "Quality impact: negligible — BGE-M3 top-10 already contains the answer."
        ),
    )
    keyword_search_top_k: int = Field(
        default=10,
        description=(
            "BM25 keyword search candidates before reranking. "
            "Reduced from 20 to 10 — see semantic_search_top_k rationale."
        ),
    )

    primary_model_max_tokens: int = Field(
        default=1024,
        description=(
            "Maximum tokens in the primary LLM response. "
            "Reduced from 4096 to 1024. With 'detailed thinking off' in system prompt, "
            "Nemotron does not generate thinking tokens. "
            "RAG answers are 200-500 tokens. 1024 is sufficient and avoids "
            "the API holding the connection open for unused token budget."
        ),
    )

    fast_model_max_tokens: int = Field(
        default=256,
        description=(
            "Maximum tokens for HyDE generation and groundedness scoring. "
            "HyDE synthetic passage: ~150-200 tokens. "
            "Groundedness score: 1 integer token. "
            "256 is sufficient. Reduces API latency vs 512."
        ),
    )

    encoder_batch_size: int = Field(
        default=32,
        description=(
            "Passages encoded per forward pass through BGE-M3. "
            "Reduce if CPU RAM is limited. "
            "Override via ENCODER_BATCH_SIZE env var."
        ),
    )

    top_passages_count: int = Field(
        default=5,
        description="Final passage count delivered to the LLM after reranking.",
    )
    semantic_retrieval_weight: float = Field(
        default=0.7,
        description="RRF fusion weight for dense retrieval. BM25 gets 1 - this.",
    )
    keyword_retrieval_weight: float = Field(
        default=0.3,
        description="RRF fusion weight for BM25 retrieval.",
    )

    minimum_segment_length: int = Field(
        default=50,
        description="Minimum character length for a passage. Shorter ones are noise.",
    )

    index_collection_name: str = Field(
        default="source_passages",
        description="ChromaDB collection name.",
    )
    stale_index_threshold_hours: int = Field(
        default=24,
        description="Warn when index is older than this many hours.",
    )

    max_upload_size_mb: int = Field(default=50)
    max_file_size_mb: int = Field(default=50)
    max_questions_per_min: int = Field(default=10)
    enable_content_scanning: bool = Field(
        default=False,
        description=(
            "Enable NeMo Guardrails. "
            "False during local validation. "
            "True for production."
        ),
    )


settings = _PipelineSettings()

BASE_DIR          = Path(__file__).resolve().parent.parent
SOURCE_FILES_DIR  = BASE_DIR / "data"
INDEX_STORE_DIR   = BASE_DIR / "index_store"
AUDIT_LOG_FILE    = BASE_DIR / "file_audit.json"
INDEX_INFO_FILE   = INDEX_STORE_DIR / "index_info.json"
THROTTLE_LOG_FILE = BASE_DIR / ".throttle_log.json"

ALLOWED_MIME_TYPES = {"application/pdf", "text/plain"}

_LOCAL_PROVIDERS_WITHOUT_CREDENTIALS = {"ollama", "ollama_chat"}


def validate_environment() -> None:
    """
    Confirm the API key required by the configured provider is present.

    NVIDIA NIM: validates NVIDIA_API_KEY directly.
                Does NOT delegate to litellm.validate_environment()
                because LiteLLM looks for NVIDIA_NIM_API_KEY - a
                different name. We use ChatOpenAI for NVIDIA, not
                ChatLiteLLM, so only NVIDIA_API_KEY is needed.

    All others: delegates to litellm.validate_environment() which
                correctly identifies the required key per provider.

    Ollama: local, no credentials, skipped.
    """
    models_in_use = list({settings.primary_model_name, settings.fast_model_name})
    all_missing_keys: list[str] = []

    for model_name in models_in_use:
        provider = model_name.split("/")[0] if "/" in model_name else model_name

        if provider in _LOCAL_PROVIDERS_WITHOUT_CREDENTIALS:
            continue

        if provider == "nvidia_nim":
            if not settings.nvidia_api_key.get_secret_value():
                all_missing_keys.append("NVIDIA_API_KEY")
            continue

        result = litellm.validate_environment(model_name)
        if not result.get("keys_in_environment", True):
            missing = result.get("missing_keys", [])
            if missing:
                all_missing_keys.extend(missing)

    unique_missing = sorted(set(all_missing_keys))
    if unique_missing:
        raise ConfigurationError(
            f"Missing required API keys: {unique_missing}. "
            f"Required for PRIMARY_MODEL_NAME='{settings.primary_model_name}'. "
            "Add the missing keys to your .env file. "
            "For NVIDIA NIM: add NVIDIA_API_KEY=nvapi-... to .env."
        )


def needs_temperature_omitted(model_name: str) -> bool:
    """
    True only for Gemini 3.x API models - they reject the temperature parameter.
    All other providers including NVIDIA NIM accept temperature correctly.
    """
    return "gemini/gemini-3" in model_name
