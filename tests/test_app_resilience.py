"""
Offline Unit Tests for LLM Model Failover & 503 Resilience (`tests/test_app_resilience.py`).

Verifies:
1. Primary model HTTP 503 (`InternalServerError`) fails over immediately to the next fallback model
   without sleeping.
2. Non-retryable 4xx errors (`BadRequestError` / `AuthenticationError`) fail fast without cycling
   models or sleeping.
3. Total upstream 503 outage across all fallback models degrades gracefully in `guarded_business_chat`
   to `SAFE_WORK_FALLBACK` instead of crashing the Gradio worker thread.
4. `get_llm_client()` sets `max_retries=0` to prevent hidden SDK retry multiplication.
"""

import sys

sys.dont_write_bytecode = True

from types import SimpleNamespace
from typing import Any

import httpx
from openai import AuthenticationError, BadRequestError, InternalServerError
import pytest

import app
from context import SAFE_WORK_FALLBACK


def _make_status_error(cls: Any, status_code: int, message: str = "error") -> Any:
    """Constructs a synthetic OpenAI `APIStatusError` subclass for offline testing."""
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    response = httpx.Response(status_code=status_code, request=request, json={"error": {"message": message}})
    return cls(message=message, response=response, body={"error": {"message": message}})


def _make_chat_response(content: str, finish_reason: str = "stop") -> Any:
    """Builds a minimal mock OpenAI ChatCompletion response object."""
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def test_client_disables_hidden_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_llm_client()` must set `max_retries=0` to avoid 3x3 retry storms."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    built_client = app.get_llm_client()
    assert built_client.max_retries == 0


def test_503_on_primary_model_fails_over_immediately_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the first model returns HTTP 503, `call_llm_with_backoff` immediately tries the next model."""
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-2.5-flash")

    called_models: list[str] = []
    slept_durations: list[float] = []

    def fake_create(**kwargs: Any) -> Any:
        model = kwargs["model"]
        called_models.append(model)
        if model == "gemini-3.1-flash-lite":
            raise _make_status_error(InternalServerError, 503, "Model overloaded")
        return _make_chat_response("Fast failover reply from healthy model")

    monkeypatch.setattr(app.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(app.time, "sleep", lambda d: slept_durations.append(d))

    response = app.call_llm_with_backoff(
        model="gemini-3.1-flash-lite",
        messages=[{"role": "user", "content": "Hello"}],
        correlation_id="test503",
    )

    assert response.choices[0].message.content == "Fast failover reply from healthy model"
    assert called_models == ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
    assert slept_durations == []


def test_non_retryable_4xx_fails_fast_without_failover_or_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 400/401 errors must raise immediately on the first call without cycling fallback models."""
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-2.5-flash")

    called_models: list[str] = []
    slept_durations: list[float] = []

    def fake_create(**kwargs: Any) -> Any:
        called_models.append(kwargs["model"])
        raise _make_status_error(AuthenticationError, 401, "Invalid API key")

    monkeypatch.setattr(app.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(app.time, "sleep", lambda d: slept_durations.append(d))

    with pytest.raises(AuthenticationError):
        app.call_llm_with_backoff(
            model="gemini-3.5-flash-lite",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert called_models == ["gemini-3.5-flash-lite"]
    assert slept_durations == []


def test_total_503_outage_degrades_gracefully_in_guarded_business_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every model in the fallback chain returns 503 across all passes, return `SAFE_WORK_FALLBACK`."""
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-2.5-flash")

    called_models: list[str] = []
    slept_durations: list[float] = []

    def always_503(**kwargs: Any) -> Any:
        called_models.append(kwargs["model"])
        raise _make_status_error(InternalServerError, 503, "Service Unavailable")

    monkeypatch.setattr(app.client.chat.completions, "create", always_503)
    monkeypatch.setattr(app.time, "sleep", lambda d: slept_durations.append(d))

    reply = app.guarded_business_chat("Tell me about Ajhar's LangGraph automation services.")

    assert reply == SAFE_WORK_FALLBACK
    assert len(called_models) == 2 * app.MAX_API_RETRIES
    assert len(slept_durations) == app.MAX_API_RETRIES - 1
