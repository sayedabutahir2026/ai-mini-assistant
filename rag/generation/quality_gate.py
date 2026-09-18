# rag/generation/quality_gate.py
"""
CRAG quality gate - kept as a fallback.

"""
import logging
from enum import StrEnum

from langchain_core.documents import Document
from pydantic import BaseModel

from rag.llm_factory import get_fast_llm

logger = logging.getLogger(__name__)


class PassageRelevance(StrEnum):
    SUFFICIENT   = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    UNCERTAIN    = "UNCERTAIN"


class RelevanceDecision(BaseModel):
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
    if not retrieved_passages:
        logger.warning("Quality gate: no passages retrieved - INSUFFICIENT.")
        return PassageRelevance.INSUFFICIENT.value

    passage_sample = "\n---\n".join(
        p.page_content[:200] for p in retrieved_passages[:3]
    )

    try:
        structured_output = get_fast_llm().with_structured_output(RelevanceDecision)
        decision: RelevanceDecision = structured_output.invoke(
            _RELEVANCE_DECISION_PROMPT.format(
                question=user_question,
                passage_sample=passage_sample,
            )
        )
        logger.debug("Quality gate decision: %s.", decision.decision.value)
        return decision.decision.value

    except Exception as gate_failure:
        logger.warning("Quality gate failed - UNCERTAIN: %s", gate_failure)
        return PassageRelevance.UNCERTAIN.value
