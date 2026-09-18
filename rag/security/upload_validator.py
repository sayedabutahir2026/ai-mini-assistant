# rag/security/upload_validator.py
"""
Validates uploaded files before they enter the ingestion pipeline.

Two checks run in order:
    1. MIME type via magic bytes - reads the actual file header, not the name.
       A file named report.pdf but containing an executable is caught here.
    2. File size - rejects anything above MAX_UPLOAD_SIZE_MB.

Then: SHA-256 content hash is recorded in file_audit.json.
    - Same bytes, different filename -> skip re-ingestion (deduplication).
    - Gives an audit trail of every file that entered the system.

New joiner note:
    "Magic bytes" means reading the first few bytes of the file itself.
    Every file format (PDF, PNG, ZIP, EXE) starts with a known byte sequence.
    filetype.guess() reads those bytes and returns the real MIME type.
    This is why renaming malware.exe to report.pdf does not fool this check.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import filetype

from rag.errors import FileRejectedError
from rag.settings import ALLOWED_MIME_TYPES, AUDIT_LOG_FILE, settings

logger = logging.getLogger(__name__)

FileAuditRecord = dict[str, str | float]


def check_and_record_upload(file_path: Path) -> FileAuditRecord:
    """
    Validate a file's type and size, then record it in the audit log.

    If the same file content (same SHA-256) was already recorded, this
    function returns the existing record without re-writing the log.
    This prevents re-embedding the same document after a page reload.

    Args:
        file_path: Absolute path to the uploaded file on disk.

    Returns:
        The audit record for this file (new or previously existing).

    Raises:
        FileRejectedError: MIME type not in ALLOWED_MIME_TYPES, or file too large.
    """
    resolved = Path(file_path).resolve()

    detected = filetype.guess(str(resolved))

    if detected is not None:
        if detected.mime not in ALLOWED_MIME_TYPES:
            raise FileRejectedError(
                f"[SECURITY] '{resolved.name}' has MIME type '{detected.mime}'. "
                f"Allowed types: {ALLOWED_MIME_TYPES}. "
                "File rejected - the extension does not match the actual content."
            )
    else:
        if resolved.suffix.lower() not in {".txt", ".md"}:
            raise FileRejectedError(
                f"[SECURITY] Could not determine file type of '{resolved.name}'. "
                "Only PDF, TXT, and MD files are accepted."
            )

    size_in_mb = resolved.stat().st_size / (1024 * 1024)
    if size_in_mb > settings.max_upload_size_mb:
        raise FileRejectedError(
            f"[SECURITY] '{resolved.name}' is {size_in_mb:.1f} MB. "
            f"Maximum allowed size is {settings.max_upload_size_mb} MB."
        )

    content_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
    existing_records = _load_audit_log()
    records_by_hash = {r["content_hash"]: r for r in existing_records}

    if content_hash in records_by_hash:
        logger.info(
            "Skipping '%s' - identical content already recorded (hash: %s...).",
            resolved.name,
            content_hash[:12],
        )
        return records_by_hash[content_hash]

    new_record: FileAuditRecord = {
        "filename": resolved.name,
        "content_hash": content_hash,
        "size_mb": round(size_in_mb, 3),
        "mime_type": detected.mime if detected else "text/plain",
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    existing_records.append(new_record)
    _write_audit_log(existing_records)

    logger.info(
        "Recorded upload: '%s' | hash prefix: %s...",
        resolved.name,
        content_hash[:12],
    )
    return new_record


def get_all_audit_records() -> list[FileAuditRecord]:
    """Return every recorded file audit entry."""
    return _load_audit_log()


def _load_audit_log() -> list[FileAuditRecord]:
    """Read the audit log from disk. Returns an empty list if missing or corrupt."""
    if not AUDIT_LOG_FILE.exists():
        return []
    try:
        return json.loads(AUDIT_LOG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Audit log is corrupt - starting with an empty log.")
        return []


def _write_audit_log(records: list[FileAuditRecord]) -> None:
    """Persist all audit records to disk."""
    AUDIT_LOG_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
