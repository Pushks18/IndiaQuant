"""Append-only snapshot of every GlobalTabView served.

Every /global request and every /api/global/cards.json call writes one line
to `data/artifacts/global_tickets/YYYY-MM-DD.jsonl`. Each line is a
self-contained JSON record with the as_of timestamp, mode, capital,
artifact name, and the serialized cards. The file is the foundation for:

  - audit (was this card actually shown on day X at time Y?)
  - replay (run the dashboard's "what did we see at 09:30?" view)
  - backtesting (label every served ticket with its eventual outcome)

The writer is deliberately fire-and-forget: it never raises into the
caller, never blocks /global on disk I/O failures.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_DIR = Path("data/artifacts/global_tickets")


def _coerce(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    return obj


def write_view_snapshot(view, *, root: Path | str = DEFAULT_DIR) -> Path | None:
    """Append one line to today's snapshot file. Returns the path written
    (or None on failure — never raises)."""
    try:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        as_of = view.as_of if hasattr(view, "as_of") else datetime.now()
        day = as_of.date() if isinstance(as_of, datetime) else date.today()
        path = root_path / f"{day.isoformat()}.jsonl"

        record = {
            "as_of": as_of.isoformat() if isinstance(as_of, datetime) else str(as_of),
            "mode": getattr(view.mode, "value", str(view.mode)),
            "capital": float(view.capital),
            "artifact": (view.artifact_paths or {}).get("name", "stub"),
            "cards": [_coerce(dataclasses.asdict(c)) for c in view.cards],
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return path
    except Exception as exc:  # noqa: BLE001
        logger.debug("snapshot write failed (non-fatal): {}", exc)
        return None
