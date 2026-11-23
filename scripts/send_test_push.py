"""
Opt-In Manual Smoke Test for Pushover Notifications (`scripts/send_test_push.py`).

Sends a single real test notification to your phone using `PUSHOVER_USER` and `PUSHOVER_TOKEN`
from `.env`. Does NOT run automatically during `pytest`.

Usage (PowerShell):
    uv run python -B scripts/send_test_push.py
"""

import sys

sys.dont_write_bytecode = True

from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env", override=True)

from notifier import PushoverNotifier, PushoverSettings, build_lead_event


def main() -> int:
    settings = PushoverSettings.from_env()
    if not settings.enabled:
        print("[ABORT] PUSHOVER_ENABLED is false in environment/.env.")
        return 1
    if not (settings.user_key and settings.api_token):
        print("[ABORT] Missing PUSHOVER_USER or PUSHOVER_TOKEN in environment/.env.")
        return 1

    notifier = PushoverNotifier(settings, async_dispatch=False)
    sample_record = {
        "name": "Smoke Test Visitor",
        "email": "smoke.test@ajharconsulting.io",
        "inquiry": "Manual opt-in Pushover smoke test from scripts/send_test_push.py",
        "update_count": 1,
    }
    event = build_lead_event(
        record=sample_record,
        status="created_new_lead",
        correlation_id="smoketst",
    )
    print(f"[SMOKE TEST] Sending Pushover notification (trace={event.correlation_id})...")
    notifier.notify(event)
    notifier.shutdown()
    print("[SMOKE TEST] Done. Check the log output above and your Pushover mobile app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
