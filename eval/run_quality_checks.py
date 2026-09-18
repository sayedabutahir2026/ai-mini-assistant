# eval/run_quality_checks.py
"""
Offline quality evaluation using RAGAS.

Run this before every deployment — never after.

Exit codes (standard Unix convention):
    0 = all metrics above their minimum thresholds → safe to deploy
    1 = one or more metrics below threshold → block deployment

The three metrics that matter most:
    groundedness_score  (RAGAS: faithfulness)   — are answers traceable to passages?
    answer_relevance    (RAGAS: answer_relevancy) — does the answer address the question?
    passage_precision   (RAGAS: context_precision) — are retrieved passages relevant?

How to use:
    1. Run your pipeline against a set of representative questions.
    2. Paste the questions, generated answers, retrieved passage texts,
       and expected answers into EVALUATION_DATASET below.
    3. Run: uv run python eval/run_quality_checks.py

New joiner note:
    RAGAS evaluates RAG pipelines without needing a large labelled dataset.
    It uses LLMs to score answers against retrieved passages automatically.
    The scores are proxies — they do not replace human review, but they catch
    obvious regressions before users do.
"""

import contextlib
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

# Compatibility shim: ragas 0.4.x unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai, which was removed in langchain-community 0.4.0.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_shim = ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_shim.ChatVertexAI = None  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_shim

from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Replace these with real question/answer pairs from your test documents
EVALUATION_DATASET = [
    {
        "question": "What is the refund policy described in the document?",
        "answer": "REPLACE: paste the answer your pipeline generated here.",
        "contexts": ["REPLACE: paste the retrieved passage text here."],
        "ground_truth": "REPLACE: write the correct expected answer here.",
    },
]

# Minimum acceptable score for each metric before deployment is allowed
MINIMUM_ACCEPTABLE_SCORES = {
    "faithfulness": 0.85,  # < 0.85 means too many ungrounded claims
    "answer_relevancy": 0.80,  # < 0.80 means answers are off-topic too often
    "context_precision": 0.75,  # < 0.75 means too many irrelevant passages retrieved
}


def evaluate_pipeline(dataset: list[dict]) -> dict:
    """
    Run RAGAS evaluation on a dataset of question/answer pairs.

    Args:
        dataset: List of dicts with keys: question, answer, contexts, ground_truth.

    Returns:
        Dict mapping metric names to float scores.
    """
    ragas_dataset = Dataset.from_list(dataset)
    return evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "⚠️  OPENAI_API_KEY is not set in your environment or .env file.\n"
            "   RAGAS uses OpenAI by default to evaluate quality metrics.\n"
            "   Please set OPENAI_API_KEY in .env before running this script."
        )
        sys.exit(1)

    print("Running pipeline quality checks...")
    print("─" * 55)

    results = evaluate_pipeline(EVALUATION_DATASET)
    failed_metrics: list[str] = []

    for metric_name, metric_score in results.items():
        minimum = MINIMUM_ACCEPTABLE_SCORES.get(metric_name)

        if minimum is None:
            print(f"  {metric_name:30s}: {metric_score:.4f}")
            continue

        passes = metric_score >= minimum
        status = "✅ PASS" if passes else "❌ FAIL"
        if not passes:
            failed_metrics.append(metric_name)

        print(f"  {metric_name:30s}: {metric_score:.4f}  {status}  (minimum: {minimum})")

    print("─" * 55)

    if failed_metrics:
        print(f"\n❌ Deployment blocked. Fix these metrics: {failed_metrics}")
        sys.exit(1)

    print("\n✅ All checks passed. Safe to deploy.")
    sys.exit(0)
