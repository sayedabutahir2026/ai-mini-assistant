# rag/generation/groundedness_check.py
"""
Scores how well the generated answer is grounded in the retrieved passages.

"Grounded" means: every factual claim in the answer comes directly from
the retrieved passages. Nothing is invented from the LLM's training data.

This is the LLM-as-judge pattern:
    We use a second LLM call (cheap, fast model) to evaluate the first LLM's output.
    The judge sees: the passages, the question, and the generated answer.
    It scores groundedness on a 0-10 scale and writes one explanatory sentence.

Why Pydantic + with_structured_output():
    We need an integer score and a string verdict from the LLM.
    Parsing raw LLM text for a JSON object is fragile.
    with_structured_output() validates the response against a Pydantic schema.
    The score is guaranteed to be an integer between 0 and 10.
    No json.loads(). No markdown fence stripping. No .get() with fallbacks.

When to display a warning:
    score >= 6: answer is substantially grounded — display normally.
    score 3-5:  some claims may be extrapolated — soft warning.
    score < 3:  answer is likely hallucinated — strong warning.
    score == -1: the check itself failed — show no badge (do not alarm the user).

New joiner note:
    Groundedness ≠ correctness. An answer can be grounded in a wrong source.
    Groundedness means: is the answer traceable to the retrieved passages?
    Correctness means: is the answer objectively true?
    We measure groundedness. We cannot measure correctness without a ground truth.
"""

import logging

from langchain.schema import Document
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from rag.settings import FAST_MODEL_NAME, GROQ_API_KEY

logger = logging.getLogger(__name__)


class GroundednessRating(BaseModel):
    """
    Structured output schema for the groundedness judge.

    score:   Integer 0-10. Pydantic ge/le constraints enforce the range.
             If the LLM returns 11, Pydantic raises ValidationError → caught below.
    summary: One sentence explaining why this score was given.
    """

    score: int = Field(
        description="0 = fully hallucinated, 10 = every claim is in the passages.",
        ge=0,
        le=10,
    )
    summary: str = Field(description="One sentence explaining the score.")


_GROUNDEDNESS_SCORING_PROMPT = """\
Score how well the ANSWER is supported by the PASSAGES on a scale of 0 to 10.

0  = The answer contradicts or ignores the passages entirely.
5  = Some claims are in the passages; others are inferred or assumed.
10 = Every claim in the answer is stated directly in the passages.

PASSAGES:
{passages}

QUESTION: {question}

ANSWER: {answer}"""


def score_answer_groundedness(
    user_question: str,
    generated_answer: str,
    source_passages: list[Document],
) -> dict[str, int | str]:
    """
    Score how grounded the generated answer is in the retrieved passages.

    Args:
        user_question:    The original question asked by the user.
        generated_answer: The answer produced by the primary LLM.
        source_passages:  The passages that were given to the primary LLM.

    Returns:
        {"score": int, "summary": str}
        score is in [0, 10] on success.
        score is -1 if the check itself failed — caller should show no badge.
    """
    # Use first 300 characters of each of the top 3 passages for the evaluation
    passages_for_grading = "\n---\n".join(p.page_content[:300] for p in source_passages[:3])

    try:
        fast_classifier = ChatGroq(
            api_key=GROQ_API_KEY,
            model=FAST_MODEL_NAME,
            temperature=0,
            max_tokens=80,
        )
        structured_output = fast_classifier.with_structured_output(GroundednessRating)
        rating: GroundednessRating = structured_output.invoke(
            _GROUNDEDNESS_SCORING_PROMPT.format(
                passages=passages_for_grading,
                question=user_question,
                answer=generated_answer,
            )
        )
        return {"score": rating.score, "summary": rating.summary}

    except Exception as check_failure:
        logger.warning("Groundedness check failed: %s", check_failure)
        return {"score": -1, "summary": "Groundedness check unavailable."}
