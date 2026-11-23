"""
Production-Grade Pushover Notification Transport Layer (`notifier.py`).

Responsibilities:
- Strict Pydantic v2 configuration (`PushoverSettings`) and event (`NotificationEvent`) schemas.
- Pure, zero-I/O event formatters (`build_lead_event`, `build_unanswered_event`) that sanitize
  untrusted visitor/LLM strings, strip control characters, collapse whitespace floods, and enforce
  Pushover payload boundaries (title <= 250 chars, message <= 1024 chars with `…`).
- Non-blocking, fail-open delivery (`PushoverNotifier`) using a single reused `httpx.Client`,
  bounded `ThreadPoolExecutor(max_workers=2, thread_name_prefix="pushover")` with `atexit` drain,
  and `tenacity` exponential backoff + jitter retrying ONLY on transport/timeout errors, HTTP 429,
  and HTTP 5xx.
- Extensible `Notifier` Protocol with `NullNotifier` and `CompositeNotifier` (ready for a future
  `N8nWebhookNotifier` without changing `tools.py`).
"""

import sys

sys.dont_write_bytecode = True

import atexit
import concurrent.futures
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, Sequence

from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr
import tenacity
from tenacity import retry_if_exception_type, stop_after_attempt, wait_random_exponential

load_dotenv(override=False)

logger = logging.getLogger("notifier")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = True

TITLE_MAX_LEN = 250
MESSAGE_MAX_LEN = 1024
INQUIRY_SNIPPET_MAX_LEN = 650
QUESTION_SNIPPET_MAX_LEN = 750

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
_WHITESPACE_FLOOD_RE = re.compile(r"\s+")
_USER_KEY_SHAPE_RE = re.compile(r"^u[A-Za-z0-9]{29}$")
_API_TOKEN_SHAPE_RE = re.compile(r"^a[A-Za-z0-9]{29}$")
_EMAIL_IN_TEXT_RE = re.compile(r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]{2,})")


class PushoverSettings(BaseModel):
    """Strict Pydantic v2 configuration for Pushover transport."""

    model_config = ConfigDict(extra="forbid")

    user_key: SecretStr | None = None
    api_token: SecretStr | None = None
    enabled: bool = True
    timeout_seconds: float = Field(default=5.0, gt=0.0)
    max_retries: int = Field(default=3, ge=1)
    api_url: str = "https://api.pushover.net/1/messages.json"

    @classmethod
    def from_env(cls) -> "PushoverSettings":
        """Reads `PUSHOVER_USER`, `PUSHOVER_TOKEN`, and `PUSHOVER_ENABLED` from environment."""
        raw_user = (os.getenv("PUSHOVER_USER") or "").strip()
        raw_token = (os.getenv("PUSHOVER_TOKEN") or "").strip()
        raw_enabled = (os.getenv("PUSHOVER_ENABLED", "true") or "true").strip().lower()
        enabled = raw_enabled not in {"false", "0", "no", "off"}

        return cls(
            user_key=SecretStr(raw_user) if raw_user else None,
            api_token=SecretStr(raw_token) if raw_token else None,
            enabled=enabled,
        )


class NotificationEvent(BaseModel):
    """Strict Pydantic v2 schema representing a domain notification event."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["new_lead", "lead_updated", "unanswered_question"]
    title: str
    message: str
    priority: Literal[-1, 0, 1]
    correlation_id: str | None = None


class Notifier(Protocol):
    """Transport protocol for dispatching business notifications."""

    def notify(self, event: NotificationEvent) -> None:
        """Dispatches a notification event (must fail open and never raise)."""
        ...


def mask_email(email: str) -> str:
    """Masks an email address for safe INFO logging (e.g. `s***@apexagency.io`)."""
    cleaned = email.strip()
    if "@" not in cleaned:
        return "***"
    local, domain = cleaned.split("@", 1)
    prefix = local[0] if local else "*"
    return f"{prefix}***@{domain}"


def _mask_emails_in_text(text: str) -> str:
    """Masks any email addresses embedded inside a string before writing to INFO logs."""
    return _EMAIL_IN_TEXT_RE.sub(lambda m: f"{m.group(1)[0]}***@{m.group(2)}", text)


def sanitize_untrusted_text(raw: Any, max_len: int) -> str:
    """
    Strips control characters, collapses whitespace/newline floods into single spaces,
    and truncates to `max_len` characters using a single-character ellipsis (`…`).
    """
    if raw is None:
        text = ""
    else:
        text = str(raw)
    without_ctrl = _CONTROL_CHAR_RE.sub(" ", text)
    collapsed = _WHITESPACE_FLOOD_RE.sub(" ", without_ctrl).strip()
    if max_len <= 0:
        return ""
    if len(collapsed) > max_len:
        if max_len == 1:
            return "…"
        return collapsed[: max_len - 1].rstrip() + "…"
    return collapsed


def build_lead_event(
    record: dict[str, Any],
    status: str,
    correlation_id: str | None = None,
) -> NotificationEvent:
    """
    Pure formatter (no I/O) that builds a `NotificationEvent` for a created or updated business lead.
    """
    is_new = status == "created_new_lead"
    event_type: Literal["new_lead", "lead_updated"] = "new_lead" if is_new else "lead_updated"
    priority: Literal[-1, 0, 1] = 1 if is_new else 0

    name = sanitize_untrusted_text(record.get("name", "Not Provided"), max_len=80) or "Not Provided"
    email = sanitize_untrusted_text(record.get("email", "unknown@example.com"), max_len=120)
    inquiry = (
        sanitize_untrusted_text(record.get("inquiry", "General Inquiry"), max_len=INQUIRY_SNIPPET_MAX_LEN)
        or "General Inquiry"
    )
    update_count = int(record.get("update_count", 1))
    timestamp = sanitize_untrusted_text(
        record.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        max_len=40,
    )
    trace_val = sanitize_untrusted_text(correlation_id or "none", max_len=32) or "none"

    raw_title = (
        f"New Business Lead: {name} ({email})"
        if is_new
        else f"Lead Updated (#{update_count}): {name} ({email})"
    )
    title = sanitize_untrusted_text(raw_title, max_len=TITLE_MAX_LEN)

    body = (
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Inquiry: {inquiry}\n"
        f"Updates: {update_count} | Time: {timestamp} | trace={trace_val}"
    )
    if len(body) > MESSAGE_MAX_LEN:
        body = body[: MESSAGE_MAX_LEN - 1].rstrip() + "…"

    return NotificationEvent(
        event_type=event_type,
        title=title,
        message=body,
        priority=priority,
        correlation_id=correlation_id,
    )


def build_unanswered_event(
    question: str,
    correlation_id: str | None = None,
) -> NotificationEvent:
    """
    Pure formatter (no I/O) that builds a `NotificationEvent` for an unanswered visitor question.
    """
    clean_question = (
        sanitize_untrusted_text(question, max_len=QUESTION_SNIPPET_MAX_LEN) or "Unspecified question"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    trace_val = sanitize_untrusted_text(correlation_id or "none", max_len=32) or "none"

    title = sanitize_untrusted_text(
        f"Unanswered Visitor Question (trace={trace_val})",
        max_len=TITLE_MAX_LEN,
    )
    body = f"Question: {clean_question}\nTime: {timestamp} | trace={trace_val}"
    if len(body) > MESSAGE_MAX_LEN:
        body = body[: MESSAGE_MAX_LEN - 1].rstrip() + "…"

    return NotificationEvent(
        event_type="unanswered_question",
        title=title,
        message=body,
        priority=0,
        correlation_id=correlation_id,
    )


class _RetryablePushoverHTTPError(Exception):
    """Internal exception raised for HTTP 429 and 5xx responses to trigger tenacity retry."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class NullNotifier:
    """No-op notifier used when Pushover is disabled or credentials are not configured."""

    def notify(self, event: NotificationEvent) -> None:
        return None


class CompositeNotifier:
    """Fan-out notifier that forwards events to multiple child notifiers (e.g., Pushover + n8n)."""

    def __init__(self, notifiers: Sequence[Notifier]) -> None:
        self._notifiers = list(notifiers)

    def notify(self, event: NotificationEvent) -> None:
        for child in self._notifiers:
            try:
                child.notify(event)
            except Exception as exc:  # pragma: no cover - defensive isolation
                logger.error(
                    "event=composite_child_failed type=%s trace=%s error_type=%s",
                    event.event_type,
                    event.correlation_id or "none",
                    type(exc).__name__,
                )


class PushoverNotifier:
    """
    Production Pushover HTTP client with bounded background thread pool, exponential backoff + jitter,
    and strict fail-open error handling.
    """

    def __init__(
        self,
        settings: PushoverSettings,
        *,
        http_client: httpx.Client | None = None,
        async_dispatch: bool = True,
        retry_initial_wait: float = 0.3,
        retry_max_wait: float = 2.0,
    ) -> None:
        self._settings = settings
        self._client = http_client or httpx.Client(timeout=settings.timeout_seconds)
        self._owns_client = http_client is None
        self._async_dispatch = async_dispatch
        self._retry_initial_wait = retry_initial_wait
        self._retry_max_wait = retry_max_wait
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

        if self._async_dispatch:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="pushover",
            )
            atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Flushes pending tasks and closes the HTTP client cleanly on process exit."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=False)
            except Exception as exc:
                logger.warning("event=pushover_shutdown_warning error_type=%s", type(exc).__name__)
            finally:
                self._executor = None
        if self._owns_client:
            try:
                self._client.close()
            except Exception as exc:
                logger.warning("event=pushover_client_close_warning error_type=%s", type(exc).__name__)

    def notify(self, event: NotificationEvent) -> None:
        """Submits a notification for delivery; never blocks the caller or raises."""
        if not self._settings.enabled:
            return
        if self._async_dispatch and self._executor is not None:
            try:
                self._executor.submit(self._deliver_safe, event)
            except RuntimeError as exc:
                logger.error(
                    "event=pushover_failed type=%s trace=%s http_status=none attempts=0 error_type=%s",
                    event.event_type,
                    event.correlation_id or "none",
                    type(exc).__name__,
                )
        else:
            self._deliver_safe(event)

    def _deliver_safe(self, event: NotificationEvent) -> None:
        """Executes the HTTP request with tenacity retries and fail-open exception handling."""
        trace_id = event.correlation_id or "none"
        user_secret = self._settings.user_key.get_secret_value() if self._settings.user_key else ""
        token_secret = self._settings.api_token.get_secret_value() if self._settings.api_token else ""

        payload = {
            "token": token_secret,
            "user": user_secret,
            "title": sanitize_untrusted_text(event.title, max_len=TITLE_MAX_LEN),
            "message": event.message[:MESSAGE_MAX_LEN],
            "priority": str(event.priority),
        }

        attempt_counter = 0
        last_http_status: int | str = "none"
        start_ts = time.perf_counter()

        def _send_once() -> httpx.Response:
            nonlocal attempt_counter, last_http_status
            attempt_counter += 1
            response = self._client.post(self._settings.api_url, data=payload)
            last_http_status = response.status_code

            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise _RetryablePushoverHTTPError(
                    response.status_code,
                    f"Transient HTTP {response.status_code} from Pushover API",
                )
            return response

        retryer = tenacity.Retrying(
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_random_exponential(
                multiplier=self._retry_initial_wait,
                max=self._retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (httpx.TransportError, httpx.TimeoutException, _RetryablePushoverHTTPError)
            ),
            reraise=True,
        )

        try:
            response = retryer(_send_once)
            latency_ms = int((time.perf_counter() - start_ts) * 1000)

            if 400 <= response.status_code < 500:
                logger.error(
                    "event=pushover_failed type=%s trace=%s http_status=%s attempts=%d error_type=HTTPClientError",
                    event.event_type,
                    trace_id,
                    response.status_code,
                    attempt_counter,
                )
                return

            body_json = response.json()
            if response.status_code == 200 and isinstance(body_json, dict) and body_json.get("status") == 1:
                logger.info(
                    "event=pushover_sent type=%s trace=%s attempt=%d latency_ms=%d title=%s",
                    event.event_type,
                    trace_id,
                    attempt_counter,
                    latency_ms,
                    _mask_emails_in_text(event.title),
                )
            else:
                logger.error(
                    "event=pushover_failed type=%s trace=%s http_status=%s attempts=%d error_type=InvalidPushoverStatus",
                    event.event_type,
                    trace_id,
                    response.status_code,
                    attempt_counter,
                )
        except (_RetryablePushoverHTTPError, httpx.HTTPError, ValueError, tenacity.RetryError) as exc:
            logger.error(
                "event=pushover_failed type=%s trace=%s http_status=%s attempts=%d error_type=%s",
                event.event_type,
                trace_id,
                last_http_status,
                attempt_counter,
                type(exc).__name__,
            )


def build_notifier(
    settings: PushoverSettings | None = None,
    *,
    http_client: httpx.Client | None = None,
    async_dispatch: bool = True,
    retry_initial_wait: float = 0.3,
    retry_max_wait: float = 2.0,
) -> Notifier:
    """
    Validates `PushoverSettings` (or loads from environment) and returns either a configured
    `PushoverNotifier` or a fail-open `NullNotifier` after logging a single startup warning.
    """
    resolved = settings or PushoverSettings.from_env()
    if not resolved.enabled:
        logger.warning(
            "event=pushover_disabled reason=PUSHOVER_ENABLED_false action=using_NullNotifier"
        )
        return NullNotifier()

    user_val = resolved.user_key.get_secret_value().strip() if resolved.user_key else ""
    token_val = resolved.api_token.get_secret_value().strip() if resolved.api_token else ""

    if not user_val or not token_val:
        logger.warning(
            "event=pushover_disabled reason=missing_credentials action=using_NullNotifier"
        )
        return NullNotifier()

    if not _USER_KEY_SHAPE_RE.match(user_val):
        logger.warning(
            "event=pushover_credential_shape_warning field=PUSHOVER_USER expected='30 chars starting with u' actual_len=%d",
            len(user_val),
        )
    if not _API_TOKEN_SHAPE_RE.match(token_val):
        logger.warning(
            "event=pushover_credential_shape_warning field=PUSHOVER_TOKEN expected='30 chars starting with a' actual_len=%d",
            len(token_val),
        )

    return PushoverNotifier(
        resolved,
        http_client=http_client,
        async_dispatch=async_dispatch,
        retry_initial_wait=retry_initial_wait,
        retry_max_wait=retry_max_wait,
    )


_DEFAULT_NOTIFIER: Notifier = build_notifier()


def get_notifier() -> Notifier:
    """Returns the process-wide stateless `Notifier` instance created at import time."""
    return _DEFAULT_NOTIFIER
