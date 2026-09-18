# rag/generation/quality_gate.py
"""
Decides whether retrieved passages contain enough information to answer
the user's question — BEFORE the expensive primary LLM call is made.

This is the CRAG (Corrective RAG) pattern:
    Without this gate, when retrieval fails (wrong passages, empty results),
    the primary LLM still generates an answer — just from irrelevant context.
    The answer sounds confident and fluent, and is wrong.
    The gate aborts the pipeline early when retrieval clearly fails.

Three possible decisions:
    SUFFICIENT:    Retrieved passages contain the answer. Proceed.
    INSUFFICIENT:  Retrieved passages clearly do not help. Abort.
    UNCERTAIN:     Cannot tell. Proceed with a lower confidence signal.

Uses the fast classifier LLM + Pydantic structured output:
    No string parsing. No .strip().upper(). The Enum type enforces valid values.

Failure policy:
    Any error → return UNCERTAIN (allow pipeline to continue).
    The gate must never block the pipeline due to its own failure.

New joiner note:
    "CRAG" stands for Corrective Retrieval-Augmented Generation.
    It refers to the practice of checking retrieval quality before generation.
    See arxiv.org/abs/2401.15884 for the original paper.
"""

import logging
from enum import StrEnum

from langchain.schema import Document
from langchain_groq import ChatGroq
from pydantic import BaseModel

from rag.settings import FAST_MODEL_NAME, GROQ_API_KEY

logger = logging.getLogger(__name__)


class PassageRelevance(StrEnum):
    """The three possible outcomes from the quality gate."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    UNCERTAIN = "UNCERTAIN"


class RelevanceDecision(BaseModel):
    """
    Structured output schema for the quality gate classifier.

    Using a Pydantic model with an Enum field means the LLM's response
    is validated against the Enum automatically. If the LLM returns anything
    other than SUFFICIENT, INSUFFICIENT, or UNCERTAIN, Pydantic raises a
    ValidationError which our except block catches.
    """

    decision: PassageRelevance


_RELEVANCE_DECISION_PROMPT = """\
Do the PASSAGES below contain enough information to answer the QUESTION?

QUESTION:
{question}

PASSAGES (first 200 characters of each):
{passage_sample}

Reply with one of: SUFFICIENT, INSUFFICIENT, or UNCERTAIN."""


def decide_if_passages_answer_question(
    user_question: str,
    retrieved_passages: list[Document],
) -> str:
    """
    Assess whether retrieved passages are relevant enough to answer the question.

    Args:
        user_question:      The original question from the user.
        retrieved_passages: Top passages returned by the passage fetcher.

    Returns:
        "SUFFICIENT", "INSUFFICIENT", or "UNCERTAIN".
        Returns "INSUFFICIENT" immediately if retrieved_passages is empty.
        Returns "UNCERTAIN" on any LLM error (fail open — never block the pipeline).
    """
    if not retrieved_passages:
        logger.warning("Quality gate: no passages retrieved — returning INSUFFICIENT.")
        return PassageRelevance.INSUFFICIENT.value

    # Use only the first 200 characters of the top 3 passages for speed
    passage_sample = "\n---\n".join(p.page_content[:200] for p in retrieved_passages[:3])

    try:
        fast_classifier = ChatGroq(
            api_key=GROQ_API_KEY,
            model=FAST_MODEL_NAME,
            temperature=0,
            max_tokens=10,
        )
        structured_output = fast_classifier.with_structured_output(RelevanceDecision)
        decision: RelevanceDecision = structured_output.invoke(
            _RELEVANCE_DECISION_PROMPT.format(
                question=user_question,
                passage_sample=passage_sample,
            )
        )
        logger.debug("Quality gate decision: %s.", decision.decision.value)
        return decision.decision.value

    except Exception as gate_failure:
        logger.warning("Quality gate failed — returning UNCERTAIN: %s", gate_failure)
        return PassageRelevance.UNCERTAIN.value
