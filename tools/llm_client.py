import time
import logging
from typing import Type, TypeVar
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMUsageTracker:
    """Tracks cumulative latency and token usage across an agent session."""

    def __init__(self):
        self.total_calls = 0
        self.total_latency_ms = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def record(self, latency_ms: float, response_metadata: dict | None = None):
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        if response_metadata:
            usage = response_metadata.get("usage_metadata") or response_metadata.get("token_usage") or {}
            self.total_input_tokens += usage.get("input_tokens", usage.get("prompt_tokens", 0))
            self.total_output_tokens += usage.get("output_tokens", usage.get("completion_tokens", 0))

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


usage_tracker = LLMUsageTracker()


def get_gemini_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


def get_groq_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3), reraise=True)
def _call_text(llm, messages: list) -> str:
    start = time.perf_counter()
    resp = llm.invoke(messages)
    elapsed_ms = (time.perf_counter() - start) * 1000
    usage_tracker.record(elapsed_ms, getattr(resp, "response_metadata", None))
    return resp.content


def generate_text(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """Generates text respecting settings.llm_provider with automatic cross-provider fallback."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    mode = settings.llm_provider.lower().strip()

    if mode == "groq":
        try:
            return _call_text(get_groq_llm(temperature=temperature), messages)
        except Exception as groq_err:
            logger.warning(f"Groq generate_text failed ({groq_err}); falling back to Gemini.")
            return _call_text(get_gemini_llm(temperature=temperature), messages)

    try:
        return _call_text(get_gemini_llm(temperature=temperature), messages)
    except Exception as gemini_err:
        logger.warning(f"Gemini generate_text failed ({gemini_err}); falling back to Groq.")
        return _call_text(get_groq_llm(temperature=temperature), messages)


def generate_structured(
    schema: Type[T],
    system_prompt: str,
    user_prompt: str,
    prefer_fast: bool = False,
) -> T:
    """Generates a validated Pydantic object respecting settings.llm_provider with fallback."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    mode = settings.llm_provider.lower().strip()

    if mode == "groq":
        primary = get_groq_llm(temperature=0.0)
        backup = get_gemini_llm(temperature=0.1)
    elif mode == "gemini":
        primary = get_gemini_llm(temperature=0.1)
        backup = get_groq_llm(temperature=0.0)
    else:
        primary = get_groq_llm(temperature=0.0) if prefer_fast else get_gemini_llm(temperature=0.1)
        backup = get_gemini_llm(temperature=0.1) if prefer_fast else get_groq_llm(temperature=0.0)

    try:
        start = time.perf_counter()
        structured_llm = primary.with_structured_output(schema)
        result = structured_llm.invoke(messages)
        usage_tracker.record((time.perf_counter() - start) * 1000)
        return result
    except Exception as primary_err:
        logger.warning(f"Primary structured LLM failed ({primary_err}); trying backup LLM.")
        start = time.perf_counter()
        structured_backup = backup.with_structured_output(schema)
        result = structured_backup.invoke(messages)
        usage_tracker.record((time.perf_counter() - start) * 1000)
        return result
