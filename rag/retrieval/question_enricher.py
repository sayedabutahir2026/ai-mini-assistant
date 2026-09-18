# rag/retrieval/question_enricher.py
"""
Enriches a user's question to improve passage retrieval recall.

The technique (HyDE — Hypothetical Document Embeddings, Gao et al. 2022):
    User questions are short and sparse. "What is the refund period?" is
    semantically distant from "Customers may request a refund within 30 days
    of purchase by contacting support@example.com." — even though the second
    passage directly answers the first.

    Solution: use a fast LLM to draft a short synthetic passage that LOOKS like
    it could be the answer. Embed [original question + synthetic passage] together.
    The combined vector lands closer to real answer passages in the index.

    Consistent measured lift: 10-30% retrieval recall improvement.

Failure policy:
    If the fast LLM call fails for any reason, return the original question.
    The pipeline continues without enrichment — degraded recall, not a crash.

New joiner note:
    "Recall" means: of all the relevant passages that exist, what fraction did
    we actually retrieve? Higher recall = fewer relevant passages missed.
    HyDE improves recall because the enriched query matches more surface forms.
"""

import logging

from langchain_groq import ChatGroq

from rag.settings import (
    FAST_MODEL_MAX_TOKENS,
    FAST_MODEL_NAME,
    FAST_MODEL_TEMPERATURE,
    GROQ_API_KEY,
)

logger = logging.getLogger(__name__)

_SYNTHETIC_CONTEXT_PROMPT = """\
Write a 2-3 sentence passage that would directly answer the question below.
Use language and terminology likely to appear in a formal document or report.
Write as if you ARE the source document — do not say "The answer is".

Question: {question}

Passage:"""


def enrich_question_for_retrieval(user_question: str) -> str:
    """
    Append a synthetic context hint to a user question to improve retrieval recall.

    Args:
        user_question: The raw question from the user.

    Returns:
        user_question + newline + synthetic_context_hint.
        Falls back to user_question alone if the LLM call fails.
    """
    try:
        fast_classifier = ChatGroq(
            api_key=GROQ_API_KEY,
            model=FAST_MODEL_NAME,
            temperature=FAST_MODEL_TEMPERATURE,
            max_tokens=FAST_MODEL_MAX_TOKENS,
        )
        response = fast_classifier.invoke(_SYNTHETIC_CONTEXT_PROMPT.format(question=user_question))
        synthetic_context_hint = response.content.strip()
        enriched_question = f"{user_question}\n\n{synthetic_context_hint}"

        logger.debug(
            "Question enriched: %d chars → %d chars.",
            len(user_question),
            len(enriched_question),
        )
        return enriched_question

    except Exception as enrichment_failure:
        logger.warning(
            "Question enrichment failed — using original question: %s",
            enrichment_failure,
        )
        return user_question
