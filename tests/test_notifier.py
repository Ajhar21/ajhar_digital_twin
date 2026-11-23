"""
Offline Unit & Integration Tests for Pushover Notifications (`tests/test_notifier.py`).

Uses `httpx.MockTransport` and `tmp_path` so zero external network calls are made and
real workspace files (`leads.jsonl`, `leads.txt`, `unanswered_questions.jsonl`) are untouched.

Run with:
    .\\.venv\\Scripts\\python.exe -B -m pytest -p no:cacheprovider -q
"""

import sys

sys.dont_write_bytecode = True

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
from pydantic import SecretStr
import pytest

import notifier as notifier_module
from notifier import (
    MESSAGE_MAX_LEN,
    TITLE_MAX_LEN,
    NullNotifier,
    PushoverNotifier,
    PushoverSettings,
    build_lead_event,
    build_notifier,
)
import tools


VALID_TEST_USER = "u" + ("1" * 29)
VALID_TEST_TOKEN = "a" + ("2" * 29)


@pytest.fixture(autouse=True)
def isolate_tool_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirects all tool persistence paths to `tmp_path` for complete test isolation."""
    monkeypatch.setattr(tools, "LEADS_JSONL_PATH", tmp_path / "leads.jsonl")
    monkeypatch.setattr(tools, "LEADS_TXT_PATH", tmp_path / "leads.txt")
    monkeypatch.setattr(tools, "UNANSWERED_JSONL_PATH", tmp_path / "unanswered_questions.jsonl")
    return tmp_path


def _make_mock_notifier(
    handler: Any,
    *,
    max_retries: int = 3,
    async_dispatch: bool = False,
) -> tuple[PushoverNotifier, list[dict[str, str]]]:
    """Builds a `PushoverNotifier` wired to an `httpx.MockTransport` and captures parsed form payloads."""
    captured_requests: list[dict[str, str]] = []

    def _transport_handler(request: httpx.Request) -> httpx.Response:
        raw_body = request.content.decode("utf-8")
        parsed_form = {k: v[0] for k, v in parse_qs(raw_body, keep_blank_values=True).items()}
        captured_requests.append(parsed_form)
        return handler(request, parsed_form, len(captured_requests))

    settings = PushoverSettings(
        user_key=SecretStr(VALID_TEST_USER),
        api_token=SecretStr(VALID_TEST_TOKEN),
        enabled=True,
        timeout_seconds=2.0,
        max_retries=max_retries,
    )
    client = httpx.Client(transport=httpx.MockTransport(_transport_handler), timeout=2.0)
    push_notifier = PushoverNotifier(
        settings,
        http_client=client,
        async_dispatch=async_dispatch,
        retry_initial_wait=0.001,
        retry_max_wait=0.005,
    )
    return push_notifier, captured_requests


def test_1_new_lead_sends_single_priority_1_notification() -> None:
    """Case 1: A new lead -> exactly one `new_lead` event, priority 1, message <= 1024 chars."""
    push_notifier, captured = _make_mock_notifier(
        lambda req, form, count: httpx.Response(200, json={"status": 1, "request": "req-1"})
    )

    result = tools.record_business_lead(
        email="Sarah.Jenkins@ApexAgency.io",
        name="Sarah Jenkins",
        inquiry="Need LangGraph + n8n lead qualification pipeline",
        correlation_id="ab12cd34",
        notifier=push_notifier,
    )

    assert result["status"] == "success"
    assert result["action"] == "created_new_lead"
    assert len(captured) == 1
    sent = captured[0]
    assert sent["priority"] == "1"
    assert len(sent["title"]) <= TITLE_MAX_LEN
    assert len(sent["message"]) <= MESSAGE_MAX_LEN
    assert "sarah.jenkins@apexagency.io" in sent["message"]
    assert "Sarah Jenkins" in sent["message"]
    assert "trace=ab12cd34" in sent["message"]


def test_2_duplicate_lead_skips_unchanged_inquiry_and_notifies_on_changed_inquiry() -> None:
    """Case 2: Duplicate lead with same inquiry -> no push; changed inquiry -> one `lead_updated` push."""
    push_notifier, captured = _make_mock_notifier(
        lambda req, form, count: httpx.Response(200, json={"status": 1})
    )

    # Initial creation -> 1st push
    tools.record_business_lead(
        email="sarah@apexagency.io",
        name="Sarah Jenkins",
        inquiry="Build an n8n CRM workflow",
        correlation_id="trace001",
        notifier=push_notifier,
    )
    assert len(captured) == 1

    # Duplicate email with identical normalized inquiry (extra whitespace only) -> NO push
    res_same = tools.record_business_lead(
        email="SARAH@APEXAGENCY.IO",
        name="Sarah Jenkins",
        inquiry="  Build   an n8n CRM workflow  ",
        correlation_id="trace002",
        notifier=push_notifier,
    )
    assert res_same["action"] == "updated_existing_lead"
    assert len(captured) == 1

    # Duplicate email with a genuinely updated inquiry -> 1 `lead_updated` push (priority 0)
    res_changed = tools.record_business_lead(
        email="sarah@apexagency.io",
        name="Sarah Jenkins",
        inquiry="Build an n8n CRM workflow plus FastAPI PostgreSQL RAG",
        correlation_id="trace003",
        notifier=push_notifier,
    )
    assert res_changed["action"] == "updated_existing_lead"
    assert len(captured) == 2
    assert captured[1]["priority"] == "0"
    assert "FastAPI PostgreSQL RAG" in captured[1]["message"]
    assert "Updates: 3" in captured[1]["message"]
    assert "trace=trace003" in captured[1]["message"]


def test_3_unanswered_question_sends_notification() -> None:
    """Case 3: An unanswered question -> one `unanswered_question` push with priority 0."""
    push_notifier, captured = _make_mock_notifier(
        lambda req, form, count: httpx.Response(200, json={"status": 1})
    )

    res = tools.record_unanswered_question(
        question="Do you offer on-site mainframe soldering in Zurich at a fixed CHF rate?",
        category="unsupported_service",
        correlation_id="gap98765",
        notifier=push_notifier,
    )

    assert res["status"] == "logged"
    assert len(captured) == 1
    assert captured[0]["priority"] == "0"
    assert "Zurich" in captured[0]["message"]
    assert "trace=gap98765" in captured[0]["message"]


def test_4_pushover_500_retries_max_retries_and_fails_open(tmp_path: Path) -> None:
    """Case 4: Pushover 500 -> retried `max_retries` times, tool returns success and `leads.jsonl` is intact."""
    push_notifier, captured = _make_mock_notifier(
        lambda req, form, count: httpx.Response(500, json={"status": 0, "errors": ["server error"]}),
        max_retries=3,
    )

    res = tools.record_business_lead(
        email="resilient@client.io",
        name="Resilient Client",
        inquiry="Testing HTTP 500 fail-open retry",
        correlation_id="retry500",
        notifier=push_notifier,
    )

    assert len(captured) == 3
    assert res["status"] == "success"
    assert res["action"] == "created_new_lead"

    leads_jsonl = tmp_path / "leads.jsonl"
    assert leads_jsonl.exists()
    persisted = json.loads(leads_jsonl.read_text(encoding="utf-8").strip())
    assert persisted["email"] == "resilient@client.io"


def test_5_pushover_400_does_not_retry_and_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    """Case 5: Pushover 400 -> no retry (1 attempt), ERROR logged, tool unaffected."""
    push_notifier, captured = _make_mock_notifier(
        lambda req, form, count: httpx.Response(
            400, json={"status": 0, "errors": ["application token is invalid"]}
        ),
        max_retries=3,
    )

    with caplog.at_level(logging.ERROR, logger="notifier"):
        res = tools.record_business_lead(
            email="client400@agency.io",
            name="Client 400",
            inquiry="Testing 400 no-retry",
            correlation_id="bad400id",
            notifier=push_notifier,
        )

    assert len(captured) == 1
    assert res["status"] == "success"
    error_logs = [rec.getMessage() for rec in caplog.records if rec.levelname == "ERROR"]
    assert any("event=pushover_failed" in msg and "http_status=400" in msg and "attempts=1" in msg for msg in error_logs)


def test_6_missing_creds_uses_null_notifier_with_single_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Case 6: Missing creds -> `NullNotifier`, no exception, one startup warning."""
    settings = PushoverSettings(user_key=None, api_token=None, enabled=True)

    with caplog.at_level(logging.WARNING, logger="notifier"):
        built = build_notifier(settings)

    assert isinstance(built, NullNotifier)
    warning_logs = [rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"]
    assert len(warning_logs) == 1
    assert "event=pushover_disabled" in warning_logs[0]

    # Calling notify on NullNotifier must be a clean no-op
    res = tools.record_business_lead(
        email="null@agency.io",
        name="Null Test",
        inquiry="Should succeed silently",
        notifier=built,
    )
    assert res["status"] == "success"


def test_7_validation_error_lead_does_not_trigger_push() -> None:
    """Case 7: `validation_error` lead -> no push."""
    push_notifier, captured = _make_mock_notifier(
        lambda req, form, count: httpx.Response(200, json={"status": 1})
    )

    res = tools.record_business_lead(
        email="invalid-email-format",
        name="Bad Email User",
        inquiry="Should never trigger Pushover",
        correlation_id="valerr01",
        notifier=push_notifier,
    )

    assert res["status"] == "validation_error"
    assert len(captured) == 0


def test_8_formatter_truncates_and_strips_control_chars_on_malicious_inquiry() -> None:
    """Case 8: Formatter truncation + control-char stripping on a 5,000-char malicious inquiry."""
    malicious_inquiry = (
        "Need automation\x00\x07\x1b[31m\r\n\r\n\r\n   "
        + ("A" * 5000)
        + "\x08\x1f"
    )
    record = {
        "name": "Malicious\x00\n\nVisitor   " + ("N" * 400),
        "email": "attacker@example.com",
        "inquiry": malicious_inquiry,
        "update_count": 1,
        "timestamp": "2026-11-23 05:00:00 UTC",
    }

    event = build_lead_event(record, status="created_new_lead", correlation_id="sec12345")

    assert len(event.title) <= TITLE_MAX_LEN
    assert len(event.message) <= MESSAGE_MAX_LEN
    assert "…" in event.message
    assert "\x00" not in event.message
    assert "\x07" not in event.message
    assert "\x1b" not in event.message
    assert "\r" not in event.message
    assert "   " not in event.message
