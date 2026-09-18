# rag/retrieval/question_enricher.py
import logging

from langchain_litellm import ChatLiteLLM

from rag.settings import settings

logger = logging.getLogger(__name__)

_SYNTHETIC_CONTEXT_PROMPT = """\
Write a 2-3 sentence passage that would directly answer the question below.
Use language and terminology likely to appear in a formal document or report.
Write as if you ARE the source document - do not say "The answer is".

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
        fast_classifier = ChatLiteLLM(
            model=settings.fast_model_name,
            temperature=settings.fast_model_temperature,
            max_tokens=settings.fast_model_max_tokens,
        )
        response = fast_classifier.invoke(_SYNTHETIC_CONTEXT_PROMPT.format(question=user_question))
        synthetic_context_hint = response.content.strip()
        enriched_question = f"{user_question}\n\n{synthetic_context_hint}"

        logger.debug(
            "Question enriched: %d chars -> %d chars.",
            len(user_question),
            len(enriched_question),
        )
        return enriched_question

    except Exception as enrichment_failure:
        logger.warning(
            "Question enrichment failed - using original question: %s",
            enrichment_failure,
        )
        return user_question
