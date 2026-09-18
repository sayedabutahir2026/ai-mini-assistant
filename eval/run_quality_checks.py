# eval/run_quality_checks.py
"""
Offline quality evaluation using RAGAS v0.2+ API.

Run this before every deployment - never after.

Exit codes (standard Unix convention):
    0 = all metrics above their minimum thresholds -> safe to deploy
    1 = one or more metrics below threshold -> block deployment

Design principle - why no local model in this script:
    The eval script is a CI tool. It runs quickly, needs only an API key,
    and produces a pass/fail signal. Loading BGE-M3 (550MB, requires PyTorch)
    to score 5-10 QA pairs is wrong - it adds 30s startup, a heavy dependency,
    and fails on Python version mismatches unrelated to eval logic.

    Instead: we use API-based embeddings matched to the configured LLM provider.
    Gemini's text-embedding-004 model is free, requires the same GEMINI_API_KEY
    already in .env, and runs in milliseconds with no local model download.

    Provider -> embedding model mapping (for new joiners):
        gemini/*      -> models/text-embedding-004  (via GoogleGenerativeAIEmbeddings)
        groq/*        -> models/text-embedding-004  (Groq has no embedding API.
                         We use Gemini embeddings regardless of LLM provider
                         because Gemini embeddings are free and keyless-compatible.)
        openrouter/*  -> models/text-embedding-004  (same reasoning)
        ollama/*      -> nomic-embed-text via Ollama (local, no key)

Known upstream bug - VertexAI shim:
    ragas/llms/base.py imports ChatVertexAI unconditionally at module load time.
    langchain-community 0.4.2+ removed that path.
    We register a stub before any ragas import.
    Track: github.com/vibrantlabsai/ragas/issues/2753

RAGAS v0.2+ field names (renamed from v0.1):
    user_input         ← was "question"
    response           ← was "answer"
    retrieved_contexts ← was "contexts"
    reference          ← was "ground_truth"
"""
import contextlib
import sys
import warnings
from pathlib import Path
from types import ModuleType

warnings.filterwarnings("ignore", category=DeprecationWarning)

if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── VertexAI shim - MUST run before ANY ragas import ─────────────────────────
# Register stubs for both the parent package AND the leaf module.
# Python resolves dotted imports by walking each segment in sequence.
# Both must be in sys.modules before ragas/__init__.py executes.
# Remove once ragas fixes upstream: github.com/vibrantlabsai/ragas/issues/2753
def _register_vertexai_stub() -> None:
    leaf_path   = "langchain_community.chat_models.vertexai"
    parent_path = "langchain_community.chat_models"

    if leaf_path not in sys.modules:
        if parent_path not in sys.modules:
            parent_stub = ModuleType(parent_path)
            sys.modules[parent_path] = parent_stub

        leaf_stub = ModuleType(leaf_path)
        leaf_stub.ChatVertexAI = None  # type: ignore[attr-defined]
        sys.modules[leaf_path] = leaf_stub


_register_vertexai_stub()
# ── end shim ──────────────────────────────────────────────────────────────────

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper           # noqa: E402
from ragas.llms import LangchainLLMWrapper                        # noqa: E402
from ragas.metrics import (                                        # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from langchain_litellm import ChatLiteLLM                          # noqa: E402
from rag.settings import settings, validate_environment            # noqa: E402
from rag.errors import ConfigurationError                          # noqa: E402


def _build_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    """
    Build an API-based embedding model for RAGAS evaluation.

    Design decision:
        We deliberately do NOT use get_text_encoder() (BGE-M3) here.
        BGE-M3 requires sentence-transformers -> PyTorch -> 550MB download.
        The eval script is a CI tool - it must run with only an API key.

        We use Google's text-embedding-004 model regardless of which LLM
        provider is configured for generation. Reasons:
          - It is free with the same GEMINI_API_KEY already in .env
          - It requires no local model download
          - It works on any Python version without PyTorch
          - Gemini embedding quality is sufficient for RAGAS metric scoring

        If the user has no GEMINI_API_KEY (e.g. pure Groq setup), the function
        falls back to a lightweight sentence-transformers model that is fast to
        download and does not require a GPU or full PyTorch install.

    Returns:
        LangchainEmbeddingsWrapper compatible with RAGAS metrics.
    """
    provider = settings.primary_model_name.split("/")[0]

    if provider == "ollama":
        # Ollama runs locally - use its built-in nomic-embed-text model
        # which downloads automatically on first use via the Ollama server.
        from langchain_community.embeddings import OllamaEmbeddings
        base_url = settings.ollama_api_base if hasattr(settings, "ollama_api_base") else "http://localhost:11434"
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=base_url,
        )
        return LangchainEmbeddingsWrapper(embeddings)

    # For all API providers (groq, gemini, openrouter, anthropic, openai):
    # Use Google text-embedding-004 - free, fast, no local model required.
    # Falls back to a lightweight sentence-transformers model if GEMINI_API_KEY
    # is not set (covers pure Groq setups where Gemini is not configured).
    gemini_key = settings.gemini_api_key.get_secret_value()

    if gemini_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_key,
        )
        return LangchainEmbeddingsWrapper(embeddings)

    # Final fallback: lightweight all-MiniLM-L6-v2 via sentence-transformers.
    # 80MB, CPU-only, downloads once. No GPU. No OPENAI_API_KEY.
    # Only reached when neither Gemini nor Ollama is configured.
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


# ── Evaluation dataset ────────────────────────────────────────────────────────
# Fill in real values before running this script.
#
# How to get real values in three steps:
#   1. uv run streamlit run app.py
#   2. Ask a question whose answer is clearly in your test PDF.
#   3. Copy: the question, the generated answer, the passage texts from
#      "Source Citations", and what you know the correct answer should be.
EVALUATION_SAMPLES: list[SingleTurnSample] = [
    SingleTurnSample(
        user_input="REPLACE: paste the question you asked here.",
        response="REPLACE: paste the answer the pipeline generated here.",
        retrieved_contexts=[
            "REPLACE: paste the full text of the first retrieved passage here.",
            "REPLACE: paste the full text of the second retrieved passage here.",
        ],
        reference="REPLACE: write the correct expected answer here.",
    ),
]

# ── Minimum acceptable scores - CI deployment gates ──────────────────────────
MINIMUM_ACCEPTABLE_SCORES: dict[str, float] = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.75,
}


def build_evaluation_dataset(
    samples: list[SingleTurnSample],
) -> EvaluationDataset:
    """
    Wrap SingleTurnSample objects in a RAGAS EvaluationDataset.

    Args:
        samples: One SingleTurnSample per question/answer pair.

    Returns:
        EvaluationDataset ready to pass to ragas.evaluate().
    """
    return EvaluationDataset(samples=samples)


def run_evaluation(evaluation_dataset: EvaluationDataset) -> dict[str, float]:
    """
    Run RAGAS evaluation on the given dataset.

    RunConfig explanation (for new joiners):
        RAGAS defaults to 16 concurrent async workers.
        Free-tier APIs (Gemini: 15 RPM, Groq: 30 RPM) cannot handle
        16 simultaneous requests - they rate-limit immediately, LiteLLM
        retries each one, the event loop fills with stalled coroutines,
        and the whole run cancels with CancelledError.

        max_workers=1 forces sequential execution - one metric scored,
        then the next. Slower (4 metrics X ~3s each = ~12s total for one
        sample) but guaranteed to complete on any free-tier provider.

        If you upgrade to a paid tier with higher RPM, raise max_workers
        to 4 or 8 and the eval will finish proportionally faster.
    """
    from ragas.run_config import RunConfig

    evaluator_llm = LangchainLLMWrapper(
        ChatLiteLLM(
            model=settings.primary_model_name,
            temperature=0,
            max_tokens=1024,
        )
    )

    evaluator_embeddings = _build_evaluator_embeddings()

    evaluation_metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    # max_workers=1 -> sequential, never hits rate limits on any free tier
    # timeout=120   -> 2 minutes per individual metric call before giving up
    run_config = RunConfig(
        max_workers=1,
        timeout=120,
    )

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=evaluation_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )
    return dict(result)

if __name__ == "__main__":
    try:
        validate_environment()
    except ConfigurationError as missing_config:
        print(f"\n❌  Configuration error: {missing_config}")
        print(
            "   Add the missing key(s) to your .env file and re-run.\n"
            f"   Currently configured models:\n"
            f"     Primary : {settings.primary_model_name}\n"
            f"     Fast    : {settings.fast_model_name}"
        )
        sys.exit(1)

    # Resolve which embedding backend will be used before starting
    provider       = settings.primary_model_name.split("/")[0]
    gemini_key_set = bool(settings.gemini_api_key.get_secret_value())

    if provider == "ollama":
        embedding_backend = "nomic-embed-text via Ollama (local)"
    elif gemini_key_set:
        embedding_backend = "models/text-embedding-004 via Gemini API (free)"
    else:
        embedding_backend = "all-MiniLM-L6-v2 via sentence-transformers (local fallback)"

    print(f"Evaluator LLM        : {settings.primary_model_name}")
    print(f"Evaluator embeddings : {embedding_backend}")
    print(f"Sample count         : {len(EVALUATION_SAMPLES)}")
    print("Running pipeline quality checks...")
    print("─" * 55)

    evaluation_dataset = build_evaluation_dataset(EVALUATION_SAMPLES)
    metric_scores      = run_evaluation(evaluation_dataset)

    failed_metrics: list[str] = []

    for metric_name, metric_score in metric_scores.items():
        minimum_threshold = MINIMUM_ACCEPTABLE_SCORES.get(metric_name)

        if minimum_threshold is None:
            print(f"  {metric_name:30s}: {metric_score:.4f}")
            continue

        passes = metric_score >= minimum_threshold
        status = "✅ PASS" if passes else "❌ FAIL"

        if not passes:
            failed_metrics.append(metric_name)

        print(
            f"  {metric_name:30s}: {metric_score:.4f}  "
            f"{status}  (minimum: {minimum_threshold})"
        )

    print("─" * 55)

    if failed_metrics:
        print(f"\n❌  Deployment blocked. Metrics below threshold: {failed_metrics}")
        print(
            "    Improve retrieval or generation quality, then re-run.\n"
            "    Run: uv run streamlit run app.py to test interactively."
        )
        sys.exit(1)

    print("\n✅  All checks passed. Safe to deploy.")
    sys.exit(0)
