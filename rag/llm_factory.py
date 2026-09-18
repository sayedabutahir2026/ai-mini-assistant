# rag/llm_factory.py
"""
Single factory for all LLM instances in the pipeline.

"""
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from rag.settings import settings, needs_temperature_omitted

logger = logging.getLogger(__name__)

_NVIDIA_PREFIX   = "nvidia_nim/"
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

_NVIDIA_FAST_TIMEOUT = 600

_NVIDIA_HYDE_TIMEOUT = 600


def _is_nvidia_nim(model_name: str) -> bool:
    return model_name.startswith(_NVIDIA_PREFIX)


def _build_nvidia_llm(model: str, max_tokens: int, timeout: int) -> ChatOpenAI:
    """
    Build ChatOpenAI pointed at NVIDIA's OpenAI-compatible NIM endpoint.
    """
    bare_model = model.removeprefix(_NVIDIA_PREFIX)
    api_key    = settings.nvidia_api_key.get_secret_value()

    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. "
            "Add NVIDIA_API_KEY=nvapi-... to your .env file."
        )

    return ChatOpenAI(
        model=bare_model,
        api_key=api_key,
        base_url=_NVIDIA_BASE_URL,
        max_tokens=max_tokens,
        temperature=0,
        timeout=timeout,
        stream_usage=True,
    )


def _build_generic_llm(model: str, max_tokens: int, temperature: float) -> BaseChatModel:
    """Build ChatLiteLLM for non-NVIDIA providers."""
    from langchain_litellm import ChatLiteLLM

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
    }
    if not needs_temperature_omitted(model):
        kwargs["temperature"] = temperature

    return ChatLiteLLM(**kwargs)


def get_primary_llm() -> BaseChatModel:
    """Primary LLM for answer generation — used with streaming."""
    model = settings.primary_model_name
    if _is_nvidia_nim(model):
        # No aggressive timeout — streaming keeps the connection alive
        return _build_nvidia_llm(model, settings.primary_model_max_tokens, _NVIDIA_FAST_TIMEOUT)
    return _build_generic_llm(
        model, settings.primary_model_max_tokens, settings.primary_model_temperature
    )


def get_fast_llm() -> BaseChatModel:
    """Fast LLM for HyDE and groundedness scoring."""
    model = settings.fast_model_name
    if _is_nvidia_nim(model):
        return _build_nvidia_llm(model, settings.fast_model_max_tokens, _NVIDIA_HYDE_TIMEOUT)
    return _build_generic_llm(
        model, settings.fast_model_max_tokens, settings.fast_model_temperature
    )
