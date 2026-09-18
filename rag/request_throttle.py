# rag/request_throttle.py
"""
Per-client sliding-window request throttle using the `limits` library.

"""
import hashlib
import logging

from limits import RateLimitItemPerMinute, storage, strategies

from rag.errors import QuotaExceededError
from rag.settings import settings

logger = logging.getLogger(__name__)


_memory_store = storage.MemoryStorage()

_rate_limiter = strategies.MovingWindowRateLimiter(_memory_store)

_rate_limit = RateLimitItemPerMinute(settings.max_questions_per_min)


def enforce_rate_limit(caller_id: str) -> None:
    """
    Check the caller's request rate and record this request.

    Uses a sliding window algorithm: requests are counted over a rolling
    60-second window, not a fixed clock-minute boundary.

    Args:
        caller_id: A stable string identifying the caller - from identify_caller().

    Raises:
        QuotaExceededError: This caller has exceeded MAX_QUESTIONS_PER_MIN
                            in the last 60 seconds.
    """
    request_allowed = _rate_limiter.hit(_rate_limit, caller_id)

    if not request_allowed:
        raise QuotaExceededError(
            f"You have reached the limit of {settings.max_questions_per_min} "
            "questions per minute. Please wait a moment before asking another question."
        )

    logger.debug("Rate limit: allowed request for caller '%s'.", caller_id)


def identify_caller() -> str:
    """
    Derive a stable, hashed identifier for the current Streamlit session.

    Returns:
        First 16 hex characters of SHA-256(session_id), or "anonymous".
        SHA-256 ensures the raw session ID is never stored or logged.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        context = get_script_run_ctx()
        if context:
            return hashlib.sha256(
                context.session_id.encode("utf-8")
            ).hexdigest()[:16]
    except Exception:
        pass
    return "anonymous"
