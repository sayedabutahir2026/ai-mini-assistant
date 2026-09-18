# rag/ingestion/file_reader.py
"""
Reads PDF source files into LangChain Document objects using PyMuPDF4LLM.

"""
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

from rag.errors import FileRejectedError, NoSourceFilesError
from rag.settings import SOURCE_FILES_DIR
from rag.security.upload_validator import check_and_record_upload

logger = logging.getLogger(__name__)


def read_source_files(
    source_directory: Path = SOURCE_FILES_DIR,
) -> tuple[list[Document], list[str]]:
    """
    Read all PDF files in source_directory into page-level Document objects.

    Uses PyMuPDF4LLMLoader which produces Markdown-formatted page content.
    Markdown output preserves document structure (headings, tables, lists)
    which produces sharper, more coherent embedding vectors than raw text.

    Each Document contains:
        page_content: Markdown-formatted text of that page
        metadata:     source, page (0-indexed from loader), total_pages, file_path

    Args:
        source_directory: Directory to scan for PDF files. Defaults to SOURCE_FILES_DIR.

    Returns:
        A tuple of:
            pages:    One Document per successfully loaded page.
            warnings: Human-readable messages for files that failed validation.

    Raises:
        NoSourceFilesError: No PDFs found in source_directory,
                            or every file failed security validation.
    """
    pdf_paths = sorted(source_directory.glob("*.pdf"))
    if not pdf_paths:
        raise NoSourceFilesError(
            f"No PDF files found in '{source_directory}'. "
            "Upload at least one document before indexing."
        )

    pages: list[Document]  = []
    warnings: list[str]    = []

    for pdf_path in pdf_paths:
        # Security validation - runs before any file content is read
        try:
            check_and_record_upload(pdf_path)
        except FileRejectedError as rejection:
            warnings.append(str(rejection))
            logger.warning("Skipping '%s' - rejected by upload validator.", pdf_path.name)
            continue

        try:
            # mode="page" -> one Document per page, with page metadata attached
            loader = PyMuPDF4LLMLoader(
                file_path=str(pdf_path),
                mode="page",
            )
            file_pages = loader.load()

            # Normalise metadata key names to match what the rest of the
            # pipeline expects: page_number (1-indexed) and source (filename only)
            for doc in file_pages:
                # PyMuPDF4LLMLoader sets metadata["page"] as 0-indexed integer
                zero_indexed_page = doc.metadata.get("page", 0)
                doc.metadata["page_number"] = zero_indexed_page + 1
                doc.metadata["source"]      = pdf_path.name
                doc.metadata["total_pages"] = doc.metadata.get("total_pages", len(file_pages))

                # Drop pages with no extractable content (truly blank/corrupt pages)
                if not doc.page_content.strip():
                    logger.debug(
                        "Skipping blank page %d of '%s'.",
                        doc.metadata["page_number"],
                        pdf_path.name,
                    )
                    continue

                pages.append(doc)

            logger.info(
                "Loaded '%s': %d pages via PyMuPDF4LLMLoader.",
                pdf_path.name,
                len(file_pages),
            )

        except Exception as load_error:
            warning_message = (
                f"Failed to load '{pdf_path.name}': {load_error}. "
                "File may be corrupt, password-protected, or in an unsupported format."
            )
            warnings.append(warning_message)
            logger.warning(warning_message)
            continue

    if not pages:
        raise NoSourceFilesError(
            "No content could be extracted from any uploaded file. "
            "Check the warnings list for details."
        )

    return pages, warnings
