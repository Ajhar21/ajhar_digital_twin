"""
Main Application Entry Point & Multi-LLM Evaluator-Optimizer Pipeline (`app.py`).

Deployable on Render (`python app.py`):
- Binds Gradio to `0.0.0.0` and reads `PORT` from environment variables (default `7860`).
- Orchestrates the primary Generator LLM with bounded tool calling (`MAX_TOOL_ROUNDS = 3`)
  and the secondary Evaluator-Optimizer loop (`MAX_EVAL_RETRIES = 2`).
"""

import sys

sys.dont_write_bytecode = True

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from context import EVALUATOR_SYSTEM_PROMPT, PRIMARY_SYSTEM_PROMPT, SAFE_WORK_FALLBACK
from styles import APP_DESCRIPTION, APP_TITLE, CUSTOM_CSS, EXAMPLE_PROMPTS
from tools import TOOLS_SCHEMA, WorkEvaluationResult, execute_tool_calls

# 1. Resolve environment variables (.env in current or parent directories)
load_dotenv(override=True)
if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    for parent in Path(__file__).resolve().parent.parents:
        candidate_env = parent / ".env"
        if candidate_env.exists():
            load_dotenv(candidate_env, override=True)
            break

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_PRIMARY_MODEL = "gemini-2.5-flash"
MODEL_NAME = os.getenv("GEMINI_MODEL", DEFAULT_PRIMARY_MODEL).strip() or DEFAULT_PRIMARY_MODEL
DEFAULT_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
API_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15.0"))
MAX_API_RETRIES = 2
MAX_TOOL_ROUNDS = 3
MAX_EVAL_RETRIES = 2

# HTTP status codes that merit trying the next model or retrying after backoff
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({404, 408, 409, 429, 500, 502, 503, 504})


def get_model_fallback_chain(primary_model: str | None = None) -> list[str]:
    """
    Builds a deduplicated, ordered model fallback list starting with `primary_model`
    (or `MODEL_NAME`), followed by `GEMINI_FALLBACK_MODELS` (comma-separated env var)
    or `DEFAULT_FALLBACK_MODELS`.
    """
    first_model = (primary_model or MODEL_NAME).strip()
    raw_fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "")
    configured_fallbacks = (
        [m.strip() for m in raw_fallbacks.split(",") if m.strip()]
        if raw_fallbacks.strip()
        else list(DEFAULT_FALLBACK_MODELS)
    )
    chain: list[str] = []
    for candidate in [first_model, *configured_fallbacks]:
        if candidate and candidate not in chain:
            chain.append(candidate)
    return chain


def get_llm_client() -> OpenAI:
    """
    Initializes and returns the OpenAI-compatible Gemini client.
    Sets `max_retries=0` so the SDK does not silently multiply retries on top of
    `call_llm_with_backoff`.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment variables or .env file!"
        )
    return OpenAI(
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
        timeout=API_TIMEOUT_SECONDS,
        max_retries=0,
    )


client = get_llm_client()


def _is_retryable_llm_error(exc: Exception) -> bool:
    """Returns True if the exception is a transient timeout/connection/rate-limit/5xx/404 model error."""
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, APIError):
        status_code = getattr(exc, "status_code", None)
        return status_code is None or status_code in RETRYABLE_STATUS_CODES
    return False


def call_llm_with_backoff(
    *,
    correlation_id: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Production wrapper around `client.chat.completions.create` with:
    1. Multi-model fallback routing (`get_model_fallback_chain`) when a model returns
       HTTP 503 (overloaded), 429 (rate-limited), 404 (deprecated), or times out.
    2. Bounded exponential backoff across passes so transient capacity spikes recover
       fast without stalling the Gradio worker thread.
    """
    requested_model = kwargs.pop("model", MODEL_NAME)
    model_chain = get_model_fallback_chain(requested_model)
    trace_label = f" | trace={correlation_id}" if correlation_id else ""
    delay = 0.8
    last_exc: Exception | None = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        for idx, candidate_model in enumerate(model_chain):
            try:
                return client.chat.completions.create(model=candidate_model, **kwargs)
            except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as exc:
                last_exc = exc
                if not _is_retryable_llm_error(exc):
                    raise
                status_code = getattr(exc, "status_code", "timeout")
                has_next_model = idx < len(model_chain) - 1
                if has_next_model:
                    next_model = model_chain[idx + 1]
                    print(
                        f"[RESILIENCE{trace_label}] Model '{candidate_model}' failed "
                        f"({type(exc).__name__} status={status_code}). "
                        f"Failing over immediately to '{next_model}'..."
                    )
                elif attempt < MAX_API_RETRIES:
                    print(
                        f"[RESILIENCE{trace_label}] Pass {attempt}/{MAX_API_RETRIES} exhausted "
                        f"on '{candidate_model}' ({type(exc).__name__} status={status_code}). "
                        f"Retrying model chain in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2.0

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM invocation failed with no recorded exception.")


def sanitize_gradio_history(history: list[Any]) -> list[dict[str, str]]:
    """Normalizes Gradio chat history into clean OpenAI-compatible role/content dicts."""
    cleaned: list[dict[str, str]] = []
    if not history:
        return cleaned
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            content_val = item["content"]
            if isinstance(content_val, list):
                content_val = " ".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in content_val
                )
            cleaned.append({"role": str(item["role"]), "content": str(content_val)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0] is not None:
                cleaned.append({"role": "user", "content": str(item[0])})
            if item[1] is not None:
                cleaned.append({"role": "assistant", "content": str(item[1])})
    return cleaned


def run_generator_agent(
    messages: list[Any],
    correlation_id: str | None = None,
) -> str:
    """Executes the primary Generator LLM with a bounded tool-calling loop (`MAX_TOOL_ROUNDS`)."""
    response = call_llm_with_backoff(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS_SCHEMA,
        temperature=0.3,
        correlation_id=correlation_id,
    )

    rounds = 0
    while response.choices[0].finish_reason == "tool_calls" and rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)

        tool_results = execute_tool_calls(
            assistant_msg.tool_calls or [],
            correlation_id=correlation_id,
        )
        messages.extend(tool_results)

        response = call_llm_with_backoff(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            temperature=0.3,
            correlation_id=correlation_id,
        )

    return (response.choices[0].message.content or "").strip()


def _extract_clean_json(raw_text: str) -> str:
    """Strips optional markdown code fences so Pydantic JSON parsing never trips on formatting."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def evaluate_candidate_reply(
    user_query: str,
    candidate_reply: str,
    correlation_id: str | None = None,
) -> WorkEvaluationResult:
    """
    Calls an independent Evaluator LLM to verify that the candidate reply
    is strictly related to work/business and grounded in Ajhar's profile.
    """
    eval_messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"### Visitor Message:\n{user_query}\n\n"
                f"### Proposed Assistant Reply:\n{candidate_reply}\n\n"
                "Evaluate whether the Proposed Assistant Reply is strictly related to work/business "
                "and grounded. Return JSON only."
            ),
        },
    ]

    eval_response = call_llm_with_backoff(
        model=MODEL_NAME,
        messages=eval_messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        correlation_id=correlation_id,
    )

    raw_content = _extract_clean_json(eval_response.choices[0].message.content or "{}")
    try:
        parsed_dict = json.loads(raw_content)
        result = WorkEvaluationResult(**parsed_dict)
        if not (result.is_strictly_work_related and result.is_grounded_and_accurate):
            result.approved = False
        return result
    except (json.JSONDecodeError, ValidationError) as exc:
        print(
            f"[EVALUATOR PARSE WARNING] Could not validate Evaluator JSON ({exc}). "
            "Failing closed for revision."
        )
        return WorkEvaluationResult(
            is_strictly_work_related=False,
            is_grounded_and_accurate=False,
            approved=False,
            reasoning="Evaluator schema validation failed; triggering safe work-focused regeneration.",
            critique=(
                "Rewrite the response so it is strictly professional and focuses only on "
                "Ajhar's AI automation and backend engineering services."
            ),
        )


def guarded_business_chat(message: str, history: list[Any] | None = None) -> str:
    """
    Full Multi-LLM Agent Pipeline:
    1. Primary LLM generates answer (and invokes deterministic tools if needed).
    2. Secondary Evaluator LLM audits that the reply is strictly related to work.
    3. If rejected, Optimizer loop feeds the critique back for self-correction (up to `MAX_EVAL_RETRIES`).
    4. If all upstream LLM endpoints are unreachable/overloaded, degrades gracefully to
       `SAFE_WORK_FALLBACK` instead of crashing the Gradio request.
    """
    correlation_id = str(uuid.uuid4())[:8]
    clean_history = sanitize_gradio_history(history or [])
    messages: list[Any] = [
        {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
        *clean_history,
        {"role": "user", "content": message},
    ]

    try:
        candidate_reply = run_generator_agent(messages, correlation_id=correlation_id)
    except Exception as exc:
        print(
            f"[PIPELINE ERROR | trace={correlation_id}] Generator LLM failed across all fallback "
            f"models ({type(exc).__name__}: {exc}). Returning graceful SAFE_WORK_FALLBACK."
        )
        return SAFE_WORK_FALLBACK

    for attempt in range(1, MAX_EVAL_RETRIES + 2):
        try:
            evaluation = evaluate_candidate_reply(
                user_query=message,
                candidate_reply=candidate_reply,
                correlation_id=correlation_id,
            )
        except Exception as exc:
            print(
                f"[PIPELINE ERROR | trace={correlation_id}] Evaluator LLM failed across all fallback "
                f"models ({type(exc).__name__}: {exc}). Failing closed to SAFE_WORK_FALLBACK."
            )
            return SAFE_WORK_FALLBACK

        print(
            f"[EVALUATOR AUDIT | trace={correlation_id} | attempt={attempt}] "
            f"Approved={evaluation.approved} "
            f"(WorkRelated={evaluation.is_strictly_work_related}, "
            f"Grounded={evaluation.is_grounded_and_accurate}) "
            f"| Reason: {evaluation.reasoning}"
        )

        if evaluation.approved:
            return candidate_reply

        if attempt <= MAX_EVAL_RETRIES:
            print(
                f"[OPTIMIZER SELF-CORRECTION | trace={correlation_id}] "
                f"Rejecting reply & regenerating with critique: {evaluation.critique}"
            )
            messages.append({"role": "assistant", "content": candidate_reply})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[COMPLIANCE EVALUATOR REJECTION]: Your previous response was blocked "
                        "because it violated our strict work-only policy. "
                        f"Evaluator Critique: {evaluation.critique or evaluation.reasoning}. "
                        "Rewrite your response immediately: remove ALL personal/off-topic details "
                        "and strictly discuss Ajhar's professional AI automation and backend engineering work."
                    ),
                }
            )
            try:
                candidate_reply = run_generator_agent(messages, correlation_id=correlation_id)
            except Exception as exc:
                print(
                    f"[PIPELINE ERROR | trace={correlation_id}] Optimizer regeneration failed "
                    f"({type(exc).__name__}: {exc}). Returning SAFE_WORK_FALLBACK."
                )
                return SAFE_WORK_FALLBACK

    print(
        f"[FALLBACK TRIGGERED | trace={correlation_id}] "
        "Max evaluation retries reached. Returning deterministic work-only fallback."
    )
    return SAFE_WORK_FALLBACK


def build_gradio_app() -> gr.ChatInterface:
    """Constructs the production Gradio ChatInterface."""
    return gr.ChatInterface(
        fn=guarded_business_chat,
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        examples=EXAMPLE_PROMPTS,
    )


if __name__ == "__main__":
    demo = build_gradio_app()
    is_cloud_env = bool(os.getenv("RENDER") or os.getenv("PORT"))
    default_host = "0.0.0.0" if is_cloud_env else "127.0.0.1"
    host = os.getenv("GRADIO_SERVER_NAME", default_host)
    port = int(os.getenv("PORT", "7860"))
    print(f"[SERVER] Launching Gradio server on http://{host}:{port} (model={MODEL_NAME})")
    demo.launch(
        server_name=host,
        server_port=port,
        css=CUSTOM_CSS,
    )
