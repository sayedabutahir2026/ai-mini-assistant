# rag/ingestion/file_reader.py
"""
Reads PDF source files from disk into LangChain Document objects.

One Document per page — this is intentional.
Page-level granularity means citations show the exact page number.
If we merged all pages into one Document, we could only cite the filename.

What this module does NOT do:
    - It does not split text into segments (that is text_segmenter.py).
    - It does not embed text (that is text_encoder.py).
    - It does not validate file security (that runs before this, in upload_validator.py).

Returns (pages, warnings) so the caller decides how to surface warnings.
This module never silently swallows problems — it always tells the caller.

New joiner note:
    fitz is the Python name for PyMuPDF. The library is imported as fitz
    for historical reasons. page.get_text() extracts the visible text layer.
    Scanned PDFs have no text layer — get_text() returns an empty string.
    We flag those pages explicitly rather than silently skipping them.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain.schema import Document

from rag.errors import FileRejectedError, NoSourceFilesError
from rag.security.upload_validator import check_and_record_upload
from rag.settings import SOURCE_FILES_DIR

logger = logging.getLogger(__name__)

# Type alias — communicates intent at call sites
PageText = str


def read_source_files(
    source_directory: Path = SOURCE_FILES_DIR,
) -> tuple[list[Document], list[str]]:
    """
    Read all PDF files in source_directory into page-level Document objects.

    Each Document contains:
        page_content: the visible text on that page
        metadata:     source filename, page number, total page count

    Args:
        source_directory: Directory to scan for PDF files. Defaults to SOURCE_FILES_DIR.

    Returns:
        A tuple of:
            pages:    One Document per successfully extracted page.
            warnings: Human-readable messages for pages with no extractable text,
                      or files that failed security validation.
                      The caller decides how to display these — this module does not.

    Raises:
        NoSourceFilesError: No PDFs found in source_directory,
                            or every file failed validation.
    """
    pdf_paths = sorted(source_directory.glob("*.pdf"))
    if not pdf_paths:
        raise NoSourceFilesError(
            f"No PDF files found in '{source_directory}'. "
            "Upload at least one document before indexing."
        )

    pages: list[Document] = []
    warnings: list[str] = []

    for pdf_path in pdf_paths:
        try:
            check_and_record_upload(pdf_path)
        except FileRejectedError as rejection:
            warnings.append(str(rejection))
            logger.warning("Skipping '%s' — rejected by upload validator.", pdf_path.name)
            continue

        with fitz.open(str(pdf_path)) as pdf:
            total_pages = len(pdf)
            pages_extracted = 0

            for page_index, page in enumerate(pdf, start=1):
                page_text: PageText = page.get_text().strip()

                if not page_text:
                    warnings.append(
                        f"'{pdf_path.name}' page {page_index}/{total_pages}: "
                        "No text found. This page is likely a scanned image. "
                        "Add tesseract + pymupdf-ocr to your dependencies for OCR support."
                    )
                    continue

                pages.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": pdf_path.name,
                            "page_number": page_index,
                            "total_pages": total_pages,
                        },
                    )
                )
                pages_extracted += 1

            logger.info(
                "Read '%s': %d of %d pages extracted.",
                pdf_path.name,
                pages_extracted,
                total_pages,
            )

    if not pages:
        raise NoSourceFilesError(
            "No text could be extracted from any uploaded file. "
            "Check the warnings list for details."
        )

    return pages, warnings
