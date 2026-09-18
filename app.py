# app.py
"""
Streamlit application - user interface only.
Change from previous version:
    build_answer_pipeline(passage_fetcher) replaced by generate_answer(question, passages).
    Retrieval now runs exactly once per question.
    Citations and generated answer are guaranteed to use the same passages.
"""
import logging
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.settings import SOURCE_FILES_DIR, settings, validate_environment
from rag.errors import (
    ConfigurationError,
    IndexMissingError,
    NoSourceFilesError,
    QuotaExceededError,
)

try:
    validate_environment()
except ConfigurationError as config_error:
    st.error(str(config_error))
    st.stop()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading text encoder (BGE-M3, one-time ~30 s)...")
def load_encoder_once():
    from rag.indexing.text_encoder import get_text_encoder
    return get_text_encoder()


@st.cache_resource(show_spinner=False)
def open_index_once():
    from rag.indexing.index_store import open_existing_index
    return open_existing_index()


load_encoder_once()


def _resolve_passages():
    passages = st.session_state.get("indexed_passages")
    if passages:
        return passages
    from rag.indexing.index_store import load_all_passages
    passages = load_all_passages()
    if not passages:
        raise IndexMissingError(
            "Passage store is empty. Upload PDF files and click 'Index Files'."
        )
    st.session_state["indexed_passages"] = passages
    return passages


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
            st.success(f"✅ Index ready - {index_info['passage_count']} passages")
            st.caption(f"Sources: {', '.join(index_info['source_files'])}")
            if (
                staleness_hours
                and staleness_hours > settings.stale_index_threshold_hours
            ):
                st.warning(
                    f"⚠️ Index is {staleness_hours:.0f} h old. "
                    "Re-index if you have uploaded new files."
                )
    else:
        st.warning("⚠️ No index. Upload PDF files and click Index Files.")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum {settings.max_file_size_mb} MB per file. Text-based PDFs only.",
    )

    if uploaded_files:
        SOURCE_FILES_DIR.mkdir(exist_ok=True)
        saved_names = []
        for uf in uploaded_files:
            save_path = SOURCE_FILES_DIR / uf.name
            save_path.write_bytes(uf.read())
            saved_names.append(uf.name)
        st.success(f"Saved: {', '.join(saved_names)}")

    if st.button("🔄 Index Files", type="primary", use_container_width=True):
        from rag.ingestion.file_reader import read_source_files
        from rag.ingestion.text_segmenter import segment_pages_into_passages
        from rag.indexing.index_store import create_searchable_index

        with st.status("Indexing pipeline running...", expanded=True) as status:
            try:
                st.write("📄 Reading and validating source files...")
                pages, load_warnings = read_source_files()
                for w in load_warnings:
                    st.warning(w)
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

            except NoSourceFilesError as e:
                status.update(label="❌ No files found.", state="error")
                st.error(str(e))
            except Exception as e:
                status.update(label="❌ Indexing failed.", state="error")
                st.exception(e)

    st.divider()

    from rag.security.upload_validator import get_all_audit_records
    with st.expander("🔒 File Audit Log"):
        audit_records = get_all_audit_records()
        if audit_records:
            for rec in audit_records:
                st.caption(
                    f"**{rec['filename']}** | "
                    f"{rec['size_mb']} MB | "
                    f"Hash: `{rec['content_hash'][:12]}...` | "
                    f"{rec['recorded_at'][:10]}"
                )
        else:
            st.caption("No files have been indexed yet.")

    st.divider()

st.title("📚 RAG Knowledge Assistant")
st.caption(
    "Answers come from your uploaded documents. "
    "Every answer includes source citations and a groundedness score."
)

if "conversation" not in st.session_state:
    st.session_state.conversation = []

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
                "green"  if ground_score >= 6 else
                "orange" if ground_score >= 3 else
                "red"
            )
            st.caption(
                f"🔍 Groundedness: :{badge_colour}[{ground_score}/10] "
                f"- {ground_result.get('summary', '')}"
            )


if user_question := st.chat_input("Ask a question about your documents..."):
    from rag.request_throttle import enforce_rate_limit, identify_caller

    try:
        enforce_rate_limit(identify_caller())
    except QuotaExceededError as e:
        st.error(str(e))
        st.stop()

    # Input validation — reject single keywords, require a question
    if len(user_question.strip()) < 10:
        st.warning(
            "⚠️ Please ask a complete question. "
            "For example: 'What is Lunorsoft's recruitment process?'"
        )
        st.stop()

    if not is_index_ready():
        st.error("No index found. Upload PDF files and click 'Index Files' first.")
        st.stop()

    st.session_state.conversation.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            from rag.retrieval.passage_fetcher import create_passage_fetcher
            from rag.generation.answer_pipeline import stream_answer
            from rag.generation.groundedness_check import score_answer_groundedness

            index    = open_index_once()
            passages = _resolve_passages()

            with st.spinner("Searching documents..."):
                passage_fetcher = create_passage_fetcher(index, passages)
                top_passages    = passage_fetcher.invoke(user_question)

            if not top_passages:
                st.warning(
                    "The uploaded documents do not appear to contain "
                    "information relevant to your question. "
                    "Try rephrasing, or upload documents that cover this topic."
                )
                st.stop()

            generated_answer = st.write_stream(
                stream_answer(user_question, top_passages)
            )

            with st.spinner("Scoring groundedness..."):
                ground_result = score_answer_groundedness(
                    user_question, generated_answer, top_passages
                )

            seen_pages: set[tuple] = set()
            citations: list[dict] = []
            for passage in top_passages:
                key = (
                    passage.metadata.get("source"),
                    passage.metadata.get("page_number"),
                )
                if key not in seen_pages:
                    seen_pages.add(key)
                    citations.append(passage.metadata)

            with st.expander("📎 Source Citations"):
                for citation in citations:
                    st.markdown(
                        f"- **{citation.get('source', 'unknown')}** | "
                        f"Page {citation.get('page_number', '?')} | "
                        f"Reranker score: `{citation.get('relevance_score', 'N/A')}`"
                    )

            ground_score = ground_result.get("score", -1)
            if ground_score != -1:
                if ground_score < 6:
                    st.warning(
                        f"⚠️ Low groundedness score ({ground_score}/10). "
                        "Verify this answer against the source documents."
                    )
                badge_colour = (
                    "green"  if ground_score >= 6 else
                    "orange" if ground_score >= 3 else
                    "red"
                )
                st.caption(
                    f"🔍 Groundedness: :{badge_colour}[{ground_score}/10] "
                    f"— {ground_result.get('summary', '')}"
                )

            st.session_state.conversation.append({
                "role":         "assistant",
                "content":      generated_answer,
                "citations":    citations,
                "groundedness": ground_result,
            })

        except IndexMissingError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.exception(e)
            log.exception("Unhandled pipeline error for question: %s", user_question)
