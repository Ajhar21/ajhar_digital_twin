"""
Deterministic Business Tools & Strict Pydantic v2 Schemas (`tools.py`).

Decouples non-deterministic LLM tool invocations from deterministic validation and persistence:
- `LeadCaptureInput`: Regex email validation, normalization, and extra-field rejection.
- `UnansweredQuestionInput`: Structured knowledge-gap validation.
- `WorkEvaluationResult`: Strict schema for the independent Evaluator LLM audit verdict.
- `record_business_lead`: Persists and deduplicates leads across `leads.jsonl` and `leads.txt`,
  then fires a fail-open Pushover notification for new leads or changed inquiries.
- `record_unanswered_question`: Logs out-of-context business/technical questions to
  `unanswered_questions.jsonl`, then fires a fail-open Pushover notification.
"""

import sys

sys.dont_write_bytecode = True

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from notifier import (
    Notifier,
    build_lead_event,
    build_unanswered_event,
    get_notifier,
    mask_email,
)

BASE_DIR = Path(__file__).resolve().parent
LEADS_JSONL_PATH = BASE_DIR / "leads.jsonl"
LEADS_TXT_PATH = BASE_DIR / "leads.txt"
UNANSWERED_JSONL_PATH = BASE_DIR / "unanswered_questions.jsonl"

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]{2,}$")


class LeadCaptureInput(BaseModel):
    """Strict Pydantic v2 schema for capturing prospective client or recruiter leads."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(..., description="The visitor's valid email address (e.g. sarah@agency.io)")
    name: str = Field(default="Not Provided", description="The visitor's name or company name")
    inquiry: str = Field(
        default="General AI Automation / Engineering Inquiry",
        description="Concise summary of the business automation need, consulting project, or role",
    )

    @field_validator("email")
    @classmethod
    def validate_and_normalize_email(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not EMAIL_REGEX.match(cleaned):
            raise ValueError(
                f"Invalid email address format: '{value}'. Must be a valid user@domain.tld address."
            )
        return cleaned

    @field_validator("name", "inquiry")
    @classmethod
    def sanitize_text_fields(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        return cleaned if cleaned else "Not Provided"


class UnansweredQuestionInput(BaseModel):
    """Strict Pydantic v2 schema for logging unanswered work/business questions."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=3,
        description="The specific question that could not be answered from verified context",
    )
    category: str = Field(
        default="out_of_context",
        description="Category of gap: pricing, availability, unsupported_tech, or other",
    )


class WorkEvaluationResult(BaseModel):
    """Strict Pydantic v2 schema returned by the secondary Evaluator LLM."""

    model_config = ConfigDict(extra="forbid")

    is_strictly_work_related: bool = Field(
        ...,
        description=(
            "True ONLY if the response is strictly professional and focused on Ajhar's work, "
            "engineering skills, career, or business services. False if it discusses personal "
            "hobbies, food/cheese preferences, recipes, politics, or non-work trivia."
        ),
    )
    is_grounded_and_accurate: bool = Field(
        ...,
        description=(
            "True if the response is faithful to the provided profile/business context and "
            "does not hallucinate fake credentials."
        ),
    )
    approved: bool = Field(
        ...,
        description="True ONLY if both is_strictly_work_related and is_grounded_and_accurate are True.",
    )
    reasoning: str = Field(..., description="Concise justification for the pass/fail verdict.")
    critique: str = Field(
        default="",
        description=(
            "If approved is False, concrete instructions telling the primary assistant how to "
            "rewrite the reply so it is strictly work-related and professional."
        ),
    )


def _sync_leads_txt(records: list[dict[str, Any]]) -> None:
    """Writes a clean, human-readable summary of all deduplicated leads to `leads.txt`."""
    lines = [
        (
            f"[{r['timestamp']}] Name: {r['name']} | Email: {r['email']} | "
            f"Inquiry: {r['inquiry']} (Updates: {r.get('update_count', 1)})"
        )
        for r in records
    ]
    LEADS_TXT_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def record_business_lead(
    *,
    correlation_id: str | None = None,
    notifier: Notifier | None = None,
    **raw_kwargs: Any,
) -> dict[str, Any]:
    """
    Validates and persists a business lead with email deduplication, then dispatches a
    fail-open Pushover notification for new leads or changed inquiries.
    """
    try:
        validated = LeadCaptureInput(**raw_kwargs)
    except ValidationError as exc:
        error_msg = exc.errors()[0].get("msg", "Invalid lead parameters")
        print(f"[TOOL VALIDATION FAILED] record_business_lead: {error_msg}")
        return {
            "status": "validation_error",
            "message": (
                f"Could not record email due to validation error: {error_msg}. "
                "Please ask the user for a valid email address."
            ),
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    existing_records: list[dict[str, Any]] = []
    if LEADS_JSONL_PATH.exists():
        for line in LEADS_JSONL_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_records.append(json.loads(line))

    matched_record = next((r for r in existing_records if r["email"] == validated.email), None)
    inquiry_changed = False

    if matched_record:
        previous_inquiry = " ".join(str(matched_record.get("inquiry", "")).strip().split())
        matched_record["timestamp"] = timestamp
        if validated.name != "Not Provided":
            matched_record["name"] = validated.name
        if validated.inquiry != "General AI Automation / Engineering Inquiry":
            if validated.inquiry != previous_inquiry:
                inquiry_changed = True
            matched_record["inquiry"] = validated.inquiry
        matched_record["update_count"] = matched_record.get("update_count", 1) + 1
        action = "updated_existing_lead"
        target_record = matched_record
    else:
        new_record = {
            "lead_id": str(uuid.uuid4())[:8],
            "timestamp": timestamp,
            "name": validated.name,
            "email": validated.email,
            "inquiry": validated.inquiry,
            "update_count": 1,
        }
        existing_records.append(new_record)
        action = "created_new_lead"
        target_record = new_record

    LEADS_JSONL_PATH.write_text(
        "\n".join(json.dumps(r) for r in existing_records) + "\n",
        encoding="utf-8",
    )
    _sync_leads_txt(existing_records)

    print(
        f"[TOOL EXECUTION] record_business_lead ({action}) -> "
        f"Email: {mask_email(validated.email)} | Name: {validated.name} | Inquiry: {validated.inquiry}"
    )

    should_notify = action == "created_new_lead" or (
        action == "updated_existing_lead" and inquiry_changed
    )
    if should_notify:
        active_notifier = notifier if notifier is not None else get_notifier()
        event = build_lead_event(
            record=target_record,
            status=action,
            correlation_id=correlation_id,
        )
        active_notifier.notify(event)

    return {
        "status": "success",
        "action": action,
        "email": validated.email,
        "name": validated.name,
        "message": (
            f"Successfully recorded lead for {validated.email} ({action}). "
            "Ajhar will follow up shortly."
        ),
    }


def record_unanswered_question(
    *,
    correlation_id: str | None = None,
    notifier: Notifier | None = None,
    **raw_kwargs: Any,
) -> dict[str, Any]:
    """
    Validates and logs an unanswered question to `unanswered_questions.jsonl`, then dispatches
    a fail-open Pushover notification.
    """
    try:
        validated = UnansweredQuestionInput(**raw_kwargs)
    except ValidationError as exc:
        return {"status": "validation_error", "message": str(exc)}

    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "question": validated.question,
        "category": validated.category,
    }
    with open(UNANSWERED_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(
        f"[TOOL EXECUTION] record_unanswered_question -> "
        f"Category: {validated.category} | Question: '{validated.question}'"
    )

    active_notifier = notifier if notifier is not None else get_notifier()
    event = build_unanswered_event(
        question=validated.question,
        correlation_id=correlation_id,
    )
    active_notifier.notify(event)

    return {
        "status": "logged",
        "message": (
            "Question logged for Ajhar's direct review. Inform the visitor honestly that "
            "you do not have that specific detail and offer to connect them via email."
        ),
    }


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "record_business_lead": record_business_lead,
    "record_unanswered_question": record_unanswered_question,
}

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "record_business_lead",
            "description": (
                "Record a prospective client's or recruiter's email address, name, and business inquiry "
                "when they want to get in touch, request an AI automation consultation, or discuss a role. "
                "Only call this tool when the user has explicitly provided an email address."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The visitor's email address (e.g. client@company.com)",
                    },
                    "name": {
                        "type": "string",
                        "description": "The visitor's name or organization if provided",
                    },
                    "inquiry": {
                        "type": "string",
                        "description": "Summary of their business automation project, support need, or job opportunity",
                    },
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_unanswered_question",
            "description": (
                "Always call this tool when a visitor asks a professional or business question whose answer "
                "is not present in Ajhar's LinkedIn profile or business services knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The unanswered question asked by the visitor",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category of the gap (e.g. pricing, availability, unsupported_service, other)",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool_calls(
    tool_calls: list[Any],
    correlation_id: str | None = None,
) -> list[dict[str, str]]:
    """Dispatches LLM tool calls through `TOOL_REGISTRY` with deterministic error handling."""
    tool_messages: list[dict[str, str]] = []
    for tool_call in tool_calls:
        fn_name = tool_call.function.name
        try:
            parsed_args = json.loads(tool_call.function.arguments or "{}")
            raw_args = parsed_args if isinstance(parsed_args, dict) else {}
        except json.JSONDecodeError:
            raw_args = {}

        # Strip reserved internal arguments so LLM-supplied payloads cannot spoof them
        raw_args.pop("correlation_id", None)
        raw_args.pop("notifier", None)

        handler = TOOL_REGISTRY.get(fn_name)
        if handler:
            result_payload = handler(correlation_id=correlation_id, **raw_args)
        else:
            result_payload = {"status": "error", "message": f"Unknown tool '{fn_name}'"}

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result_payload),
            }
        )
    return tool_messages
