# rag/request_throttle.py
"""
Per-client sliding-window request throttle.

Why file-backed (not session-based):
    Streamlit creates a new session on every page refresh.
    A session-based rate limiter is bypassed by refreshing the page.
    Writing timestamps to a file on disk survives session resets.

How the sliding window works:
    Every time a client sends a question, we record a timestamp.
    Before processing the question, we count timestamps in the last 60 seconds.
    If the count equals or exceeds MAX_QUESTIONS_PER_MIN, we raise QuotaExceededError.
    Old timestamps (outside the 60-second window) are discarded.

Client identity:
    We identify clients by a SHA-256 hash of their Streamlit session ID.
    We hash it so the raw session ID is never written to disk.
    Falls back to "anonymous" if the session context is unavailable.

New joiner note:
    This is a best-effort throttle, not a hard security barrier.
    A determined user can bypass it by clearing cookies.
    Its purpose is to protect the Groq free-tier quota from accidents,
    not to prevent adversarial abuse (that is what content_scanner.py handles).
"""

import hashlib
import json
import logging
import time

from rag.errors import QuotaExceededError
from rag.settings import MAX_QUESTIONS_PER_MIN, THROTTLE_LOG_FILE

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60.0


def enforce_rate_limit(caller_id: str) -> None:
    """
    Check the caller's request rate and record this request.

    Args:
        caller_id: A stable string identifying the caller — from identify_caller().

    Raises:
        QuotaExceededError: This caller has exceeded MAX_QUESTIONS_PER_MIN
                            within the last 60 seconds.
    """
    now = time.time()
    window_start = now - _WINDOW_SECONDS

    log = _load_throttle_log()
    recent_requests = [ts for ts in log.get(caller_id, []) if ts > window_start]

    if len(recent_requests) >= MAX_QUESTIONS_PER_MIN:
        _save_throttle_log({**log, caller_id: recent_requests})
        raise QuotaExceededError(
            f"You have reached the limit of {MAX_QUESTIONS_PER_MIN} questions per minute. "
            "Please wait a moment before asking another question."
        )

    recent_requests.append(now)
    _save_throttle_log({**log, caller_id: recent_requests})


def identify_caller() -> str:
    """
    Derive a stable, hashed identifier for the current Streamlit session.

    Returns:
        First 16 hex characters of SHA-256(session_id), or "anonymous".
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        context = get_script_run_ctx()
        if context:
            return hashlib.sha256(context.session_id.encode("utf-8")).hexdigest()[:16]
    except Exception as exc:
        logger.debug("Could not determine Streamlit session ID: %s", exc)
    return "anonymous"


def _load_throttle_log() -> dict[str, list[float]]:
    if not THROTTLE_LOG_FILE.exists():
        return {}
    try:
        return json.loads(THROTTLE_LOG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_throttle_log(log: dict[str, list[float]]) -> None:
    THROTTLE_LOG_FILE.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
