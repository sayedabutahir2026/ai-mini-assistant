# rag/security/content_scanner.py
"""
Input and output content safety scanning via llm-guard.

Why this module exists:
    User-submitted text can contain prompt injection attacks.
    LLM-generated text can leak PII or be manipulated by poisoned documents.
    llm-guard catches both — locally, without sending data to any external service.

Two public functions:
    inspect_user_question()  — called before retrieval
    inspect_generated_answer() — called before displaying to the user

Both return (cleaned_text, is_safe).
Callers decide what to do when is_safe is False.
This module never raises — it logs and returns a safe default on any error.

New joiner note:
    "Fail open" means: if the scanner itself crashes, we let the request through
    rather than breaking the app. We log the scanner error loudly so it gets fixed.
    The alternative (fail closed) would take down the whole UI on a scanner bug.
"""

import logging

from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import InvisibleText, PromptInjection, Toxicity
from llm_guard.input_scanners.prompt_injection import MatchType
from llm_guard.output_scanners import NoRefusal, Sensitive

logger = logging.getLogger(__name__)

# Module-level singletons.
# Each scanner loads an ML model on first import (~2-5s).
# Keeping them at module level means they load once per process, not per request.
_input_scanners = [
    InvisibleText(),  # Zero-width Unicode injection
    PromptInjection(match_type=MatchType.FULL),  # ML-based, not regex
    Toxicity(threshold=0.9),
]
_output_scanners = [
    NoRefusal(),  # Detects jailbroken refusals dressed as compliance
    Sensitive(),  # PII in the generated answer (emails, phone numbers, etc.)
]


def inspect_user_question(user_question: str) -> tuple[str, bool]:
    """
    Check a user's question for prompt injection, hidden characters, and toxicity.

    Args:
        user_question: The raw string the user typed into the chat input.

    Returns:
        (cleaned_question, is_safe)
        is_safe=False means the pipeline should be aborted for this question.
        is_safe=True means proceed — the returned string is the cleaned version.

    Note:
        On scanner failure: returns (user_question, True) and logs the error.
        We fail open deliberately — a scanner crash must not break the UI.
    """
    try:
        cleaned_question, scan_results, is_safe = scan_prompt(_input_scanners, user_question)
        if not is_safe:
            logger.warning(
                "[SECURITY] Question blocked by input scanner. Results: %s",
                scan_results,
            )
        return cleaned_question, is_safe

    except Exception as scanner_failure:
        logger.error("[SECURITY] Input scanner error — failing open: %s", scanner_failure)
        return user_question, True


def inspect_generated_answer(
    original_question: str,
    generated_answer: str,
) -> tuple[str, bool]:
    """
    Check the LLM's generated answer for PII leakage and jailbreak indicators.

    Args:
        original_question: The question that produced this answer.
        generated_answer:  The raw text produced by the primary LLM.

    Returns:
        (cleaned_answer, is_safe)
        is_safe=False means the UI should warn the user, but still show the answer.
        We warn rather than suppress because suppressing can be more confusing.

    Note:
        On scanner failure: returns (generated_answer, True) and logs the error.
    """
    try:
        cleaned_answer, scan_results, is_safe = scan_output(
            _output_scanners, original_question, generated_answer
        )
        if not is_safe:
            logger.warning(
                "[SECURITY] Answer flagged by output scanner. Results: %s",
                scan_results,
            )
        return cleaned_answer, is_safe

    except Exception as scanner_failure:
        logger.error("[SECURITY] Output scanner error — failing open: %s", scanner_failure)
        return generated_answer, True
