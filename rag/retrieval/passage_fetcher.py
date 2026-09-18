# rag/retrieval/passage_fetcher.py
"""
Three-stage retrieval pipeline.

"""
import logging

# from langchain_community.embeddings import HypotheticalDocumentEmbedder

from langchain_classic.chains import HypotheticalDocumentEmbedder
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_core.retrievers import BaseRetriever

from rag.indexing.text_encoder import get_text_encoder
from rag.llm_factory import get_fast_llm
from rag.settings import settings

logger = logging.getLogger(__name__)

_loaded_reranker: HuggingFaceCrossEncoder | None = None

_THINKING_OFF = SystemMessage(content="detailed thinking off")


def _get_reranker() -> HuggingFaceCrossEncoder:
    global _loaded_reranker
    if _loaded_reranker is None:
        logger.info(
            "Loading reranker '%s' (first load — downloads once)...",
            settings.reranker_model_name,
        )
        _loaded_reranker = HuggingFaceCrossEncoder(
            model_name=settings.reranker_model_name,
        )
        logger.info("Reranker ready.")
    return _loaded_reranker


class _ThinkingOffWrapper(BaseChatModel):
    """
    Thin wrapper that prepends 'detailed thinking off' SystemMessage
    to every call before forwarding to the underlying LLM.
    """
    llm: BaseChatModel

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return f"thinking_off_wrapper({self.llm._llm_type})"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system:
            messages = [_THINKING_OFF] + list(messages)
        return self.llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system:
            messages = [_THINKING_OFF] + list(messages)
        return await self.llm._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


class _HydeDenseRetriever(BaseRetriever):
    """
    Wraps HypotheticalDocumentEmbedder + ChromaDB into a BaseRetriever.
    """
    vectorstore: Chroma
    hyde_embeddings: HypotheticalDocumentEmbedder
    k: int

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        query_vector = self.hyde_embeddings.embed_query(query)
        return self.vectorstore.similarity_search_by_vector(
            query_vector,
            k=self.k,
        )


def create_passage_fetcher(
    index: Chroma,
    all_passages: list[Document],
) -> ContextualCompressionRetriever:
    """
    Assemble HyDE-Dense(k=20) + BM25(k=20) -> RRF -> CrossEncoder(top_n=5).
    """
    fast_llm_wrapped = _ThinkingOffWrapper(llm=get_fast_llm())

    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm=fast_llm_wrapped,
        base_embeddings=get_text_encoder(),
        prompt_key="web_search",
    )

    hyde_retriever = _HydeDenseRetriever(
        vectorstore=index,
        hyde_embeddings=hyde_embeddings,
        k=settings.semantic_search_top_k,
    )

    keyword_retriever = BM25Retriever.from_documents(
        all_passages,
        k=settings.keyword_search_top_k,
    )

    fused_retriever = EnsembleRetriever(
        retrievers=[hyde_retriever, keyword_retriever],
        weights=[settings.semantic_retrieval_weight, settings.keyword_retrieval_weight],
    )

    reranker = CrossEncoderReranker(
        model=_get_reranker(),
        top_n=settings.top_passages_count,
    )

    logger.debug(
        "Passage fetcher: HyDE-Dense(k=%d) + BM25(k=%d) -> RRF -> CrossEncoder(top_%d).",
        settings.semantic_search_top_k,
        settings.keyword_search_top_k,
        settings.top_passages_count,
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=fused_retriever,
    )
