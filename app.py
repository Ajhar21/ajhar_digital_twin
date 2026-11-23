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
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
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
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
API_TIMEOUT_SECONDS = 30.0
MAX_API_RETRIES = 3
MAX_TOOL_ROUNDS = 3
MAX_EVAL_RETRIES = 2


def get_llm_client() -> OpenAI:
    """Initializes and returns the OpenAI-compatible Gemini client."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment variables or .env file!"
        )
    return OpenAI(
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
        timeout=API_TIMEOUT_SECONDS,
    )


client = get_llm_client()


def call_llm_with_backoff(**kwargs: Any) -> Any:
    """
    Production wrapper around `client.chat.completions.create` with exponential backoff
    for transient RateLimitError (429) or API connection/timeout errors.
    """
    delay = 1.5
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APITimeoutError, APIError) as exc:
            if attempt == MAX_API_RETRIES:
                raise
            print(
                f"[RESILIENCE] Transient API error ({type(exc).__name__}). "
                f"Retrying attempt {attempt}/{MAX_API_RETRIES} in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= 2.0


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
        )

    return (response.choices[0].message.content or "").strip()


def _extract_clean_json(raw_text: str) -> str:
    """Strips optional markdown code fences so Pydantic JSON parsing never trips on formatting."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def evaluate_candidate_reply(user_query: str, candidate_reply: str) -> WorkEvaluationResult:
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
    """
    correlation_id = str(uuid.uuid4())[:8]
    clean_history = sanitize_gradio_history(history or [])
    messages: list[Any] = [
        {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
        *clean_history,
        {"role": "user", "content": message},
    ]

    candidate_reply = run_generator_agent(messages, correlation_id=correlation_id)

    for attempt in range(1, MAX_EVAL_RETRIES + 2):
        evaluation = evaluate_candidate_reply(
            user_query=message, candidate_reply=candidate_reply
        )
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
            candidate_reply = run_generator_agent(messages, correlation_id=correlation_id)

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
