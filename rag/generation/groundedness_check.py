# rag/generation/groundedness_check.py
"""
Scores how well the generated answer is grounded in the retrieved passages.

"""
import logging
import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag.llm_factory import get_fast_llm

logger = logging.getLogger(__name__)

_NO_THINK_MESSAGE = SystemMessage(content="/no_think")

_GROUNDEDNESS_PROMPT = """\
Score how well the ANSWER is grounded in the PASSAGES below.
Reply with ONLY a single integer from 0 to 10. Nothing else.

0  = answer contradicts or ignores the passages entirely
5  = some claims are in the passages, others are inferred
10 = every claim in the answer is directly stated in the passages

PASSAGES:
{passages}

QUESTION: {question}

ANSWER: {answer}

SCORE (integer 0-10):"""


def score_answer_groundedness(
    user_question: str,
    generated_answer: str,
    source_passages: list[Document],
) -> dict[str, int | str]:
    """
    Score how grounded the generated answer is in the retrieved passages.
    Returns {"score": int 0-10, "summary": str} or {"score": -1, ...}.
    """
    passages_text = "\n---\n".join(
        p.page_content[:300] for p in source_passages[:3]
    )

    prompt = _GROUNDEDNESS_PROMPT.format(
        passages=passages_text,
        question=user_question,
        answer=generated_answer,
    )

    try:
        response = get_fast_llm().invoke([
            _NO_THINK_MESSAGE,
            HumanMessage(content=prompt),
        ])

        raw_text = re.sub(
            r"<think>.*?</think>", "", response.content, flags=re.DOTALL
        ).strip()

        match = re.search(r"\b([0-9]|10)\b", raw_text)
        if not match:
            logger.warning(
                "Groundedness: no integer in response: %r", raw_text[:100]
            )
            return {"score": -1, "summary": "Groundedness check unavailable."}

        score = int(match.group(1))

        if score >= 8:
            summary = "Answer is well-grounded in the retrieved passages."
        elif score >= 5:
            summary = "Answer is partially grounded; some claims may be inferred."
        elif score >= 2:
            summary = "Answer has limited grounding in the retrieved passages."
        else:
            summary = "Answer appears to contradict or ignore the passages."

        return {"score": score, "summary": summary}

    except Exception as check_failure:
        logger.warning("Groundedness check failed: %s", check_failure)
        return {"score": -1, "summary": "Groundedness check unavailable."}
