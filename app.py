# app.py
"""
Streamlit application — user interface only.

This file handles:
    - Rendering the layout (sidebar, chat history, input)
    - Responding to user events (button clicks, file uploads, questions)
    - Calling rag/* modules for all actual work

This file does NOT contain:
    - Business logic (segmentation, retrieval, generation)
    - Security decisions (those are in rag/security/)
    - Constants (those are in rag/settings.py)

New joiner note:
    @st.cache_resource caches a resource for the lifetime of the Streamlit server.
    It is the right choice for ML models and database connections — things that are
    expensive to create and safe to share across all user sessions.
    We do NOT cache retrievers or answer pipelines — those are per-request objects.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv

from rag.errors import (
    ConfigurationError,
    IndexMissingError,
    NoSourceFilesError,
    QuotaExceededError,
)
from rag.settings import (
    MAX_QUESTIONS_PER_MIN,
    SOURCE_FILES_DIR,
    STALE_INDEX_THRESHOLD_HOURS,
    validate_environment,
)

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Validate environment before rendering anything ────────────────────────────
try:
    validate_environment()
except ConfigurationError as config_error:
    st.error(str(config_error))
    st.stop()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resources — loaded once per server lifetime ───────────────────────
@st.cache_resource(show_spinner="Loading text encoder (BGE-M3, one-time ~30s)...")
def load_encoder_once() -> HuggingFaceEmbeddings:
    from rag.indexing.text_encoder import get_text_encoder

    return get_text_encoder()


@st.cache_resource(show_spinner="Loading security scanners...")
def load_scanners_once() -> tuple[
    Callable[[str], tuple[str, bool]],
    Callable[[str, str], tuple[str, bool]],
]:
    from rag.security.content_scanner import inspect_generated_answer, inspect_user_question

    return inspect_user_question, inspect_generated_answer


@st.cache_resource(show_spinner=False)
def open_index_once() -> Chroma:
    from rag.indexing.index_store import open_existing_index

    return open_existing_index()


# Pre-warm on startup so the first user question is not slow
load_encoder_once()
inspect_user_question, inspect_generated_answer = load_scanners_once()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Document Manager")

    from rag.indexing.index_store import (
        compute_index_staleness_hours,
        get_index_info,
        is_index_ready,
    )

    if is_index_ready():
        index_info = get_index_info()
        staleness_hours = compute_index_staleness_hours()
        if index_info:
            st.success(f"✅ Index ready — {index_info['passage_count']} passages")
            st.caption(f"Sources: {', '.join(index_info['source_files'])}")
            if staleness_hours and staleness_hours > STALE_INDEX_THRESHOLD_HOURS:
                st.warning(
                    f"⚠️ Index is {staleness_hours:.0f} hours old. "
                    "Re-index if you have uploaded new files."
                )
    else:
        st.warning("⚠️ No index found. Upload PDF files and click Index Files.")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum {MAX_QUESTIONS_PER_MIN} MB per file. Text-based PDFs only.",
    )
    if uploaded_files:
        SOURCE_FILES_DIR.mkdir(exist_ok=True)
        saved_names = []
        for uploaded_file in uploaded_files:
            save_path = SOURCE_FILES_DIR / uploaded_file.name
            save_path.write_bytes(uploaded_file.read())
            saved_names.append(uploaded_file.name)
        st.success(f"Saved: {', '.join(saved_names)}")

    if st.button("🔄 Index Files", type="primary", use_container_width=True):
        from rag.indexing.index_store import create_searchable_index
        from rag.ingestion.file_reader import read_source_files
        from rag.ingestion.text_segmenter import segment_pages_into_passages

        with st.status("Indexing pipeline running...", expanded=True) as status:
            try:
                st.write("📄 Reading and validating source files...")
                pages, load_warnings = read_source_files()

                for warning_message in load_warnings:
                    st.warning(warning_message)

                unique_sources = len({p.metadata["source"] for p in pages})
                st.write(f"✅ {len(pages)} pages read from {unique_sources} file(s).")

                st.write("✂️ Segmenting text into searchable passages...")
                passages = segment_pages_into_passages(pages)
                st.write(f"✅ {len(passages)} unique passages created.")

                st.write("🧠 Encoding passages and building the index...")
                create_searchable_index(passages)

                st.session_state["indexed_passages"] = passages
                open_index_once.clear()

                status.update(label="✅ Index built successfully!", state="complete")
                st.rerun()

            except NoSourceFilesError as missing_files_error:
                status.update(label="❌ No files found.", state="error")
                st.error(str(missing_files_error))
            except Exception as unexpected_error:
                status.update(label="❌ Indexing failed.", state="error")
                st.exception(unexpected_error)

    st.divider()

    from rag.security.upload_validator import get_all_audit_records

    with st.expander("🔒 File Audit Log"):
        audit_records = get_all_audit_records()
        if audit_records:
            for record in audit_records:
                st.caption(
                    f"**{record['filename']}** | "
                    f"{record['size_mb']} MB | "
                    f"Hash: `{record['content_hash'][:12]}...` | "
                    f"{record['recorded_at'][:10]}"
                )
        else:
            st.caption("No files have been indexed yet.")

    st.divider()
    st.caption(
        "**Stack:** Python 3.12 · uv · LangChain 1.x LCEL · "
        "BGE-M3 · ChromaDB · BM25 · RRF · BGE-Reranker-v2-m3 · "
        "HyDE · CRAG · llm-guard · Llama-3.3-70B · Groq"
    )

# ── Main chat interface ───────────────────────────────────────────────────────
st.title("📚 RAG Knowledge Assistant")
st.caption(
    "Answers come from your uploaded documents. "
    "Every answer includes source citations and a groundedness score."
)

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Render previous turns
for turn in st.session_state.conversation:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

        if turn.get("citations"):
            with st.expander("📎 Source Citations"):
                for citation in turn["citations"]:
                    st.markdown(
                        f"- **{citation.get('source', 'unknown')}** | "
                        f"Page {citation.get('page_number', '?')} | "
                        f"Reranker score: `{citation.get('relevance_score', 'N/A')}`"
                    )

        ground_result = turn.get("groundedness")
        if ground_result and ground_result.get("score", -1) != -1:
            ground_score = ground_result["score"]
            badge_colour = (
                "green" if ground_score >= 6 else "orange" if ground_score >= 3 else "red"
            )
            st.caption(
                f"🔍 Groundedness: :{badge_colour}[{ground_score}/10] "
                f"— {ground_result.get('summary', '')}"
            )

# Handle new question
if user_question := st.chat_input("Ask a question about your documents..."):
    # Enforce rate limit before doing any work
    from rag.request_throttle import enforce_rate_limit, identify_caller

    try:
        enforce_rate_limit(identify_caller())
    except QuotaExceededError as quota_error:
        st.error(str(quota_error))
        st.stop()

    if not is_index_ready():
        st.error("No index found. Upload PDF files and click 'Index Files' first.")
        st.stop()

    # Inspect user question for security issues
    safe_question, question_is_clean = inspect_user_question(user_question)
    if not question_is_clean:
        st.error(
            "⛔ Your question was blocked by the security scanner. Please rephrase and try again."
        )
        st.stop()

    # Record and show the user's message
    st.session_state.conversation.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate and display the answer
    with st.chat_message("assistant"):
        with st.spinner("Finding relevant passages and generating an answer..."):
            try:
                from rag.generation.answer_pipeline import build_answer_pipeline
                from rag.generation.groundedness_check import score_answer_groundedness
                from rag.generation.quality_gate import decide_if_passages_answer_question
                from rag.indexing.index_store import open_existing_index
                from rag.retrieval.passage_fetcher import create_passage_fetcher
                from rag.retrieval.question_enricher import enrich_question_for_retrieval

                # Step 1: Enrich the question to improve retrieval recall
                enriched_question = enrich_question_for_retrieval(safe_question)

                # Step 2: Open the index and build a passage fetcher
                index = open_existing_index()
                passages = st.session_state.get("indexed_passages")
                if not passages:
                    # Session was cleared — re-segment from disk for BM25
                    from rag.ingestion.file_reader import read_source_files
                    from rag.ingestion.text_segmenter import segment_pages_into_passages

                    recovered_pages, _ = read_source_files()
                    passages = segment_pages_into_passages(recovered_pages)
                    st.session_state["indexed_passages"] = passages

                passage_fetcher = create_passage_fetcher(index, passages)

                # Step 3: Fetch top passages using the enriched question
                top_passages = passage_fetcher.invoke(enriched_question)

                # Step 4: Quality gate — abort if passages clearly cannot answer
                retrieval_decision = decide_if_passages_answer_question(safe_question, top_passages)

                if retrieval_decision == "INSUFFICIENT":
                    refusal = (
                        "The uploaded documents do not appear to contain "
                        "information relevant to your question. "
                        "Try rephrasing, or upload documents that cover this topic."
                    )
                    st.warning(refusal)
                    st.session_state.conversation.append(
                        {
                            "role": "assistant",
                            "content": refusal,
                            "citations": [],
                            "groundedness": None,
                        }
                    )
                    st.stop()

                # Step 5: Generate a grounded answer
                answer_pipeline = build_answer_pipeline(passage_fetcher)
                generated_answer = answer_pipeline.invoke(safe_question)

                # Step 6: Inspect the generated answer for security issues
                safe_answer, answer_is_clean = inspect_generated_answer(
                    safe_question, generated_answer
                )
                if not answer_is_clean:
                    st.warning("⚠️ The answer was flagged by the output security scanner.")

                # Step 7: Score groundedness
                ground_result = score_answer_groundedness(safe_question, safe_answer, top_passages)

                # Step 8: Deduplicate citations by (source, page_number)
                seen_pages: set[tuple] = set()
                citations: list[dict] = []
                for passage in top_passages:
                    citation_key = (
                        passage.metadata.get("source"),
                        passage.metadata.get("page_number"),
                    )
                    if citation_key not in seen_pages:
                        seen_pages.add(citation_key)
                        citations.append(passage.metadata)

                # Step 9: Display everything
                ground_score = ground_result.get("score", -1)
                if ground_score != -1 and ground_score < 6:
                    st.warning(
                        f"⚠️ Low groundedness score ({ground_score}/10). "
                        "This answer may contain information not directly "
                        "stated in the source documents."
                    )

                st.markdown(safe_answer)

                with st.expander("📎 Source Citations"):
                    for citation in citations:
                        st.markdown(
                            f"- **{citation.get('source', 'unknown')}** | "
                            f"Page {citation.get('page_number', '?')} | "
                            f"Reranker score: `{citation.get('relevance_score', 'N/A')}`"
                        )

                if ground_score != -1:
                    badge_colour = (
                        "green" if ground_score >= 6 else "orange" if ground_score >= 3 else "red"
                    )
                    st.caption(
                        f"🔍 Groundedness: :{badge_colour}[{ground_score}/10] "
                        f"— {ground_result.get('summary', '')}"
                    )

                # Persist turn to conversation history
                st.session_state.conversation.append(
                    {
                        "role": "assistant",
                        "content": safe_answer,
                        "citations": citations,
                        "groundedness": ground_result,
                    }
                )

            except IndexMissingError as index_error:
                st.error(str(index_error))
            except Exception as pipeline_error:
                st.error(f"Something went wrong: {pipeline_error}")
                st.exception(pipeline_error)
                logging.exception("Unhandled pipeline error for question: %s", user_question)
