from __future__ import annotations

import random
from typing import Dict


TEMPLATES = [
    "{action} a meeting{attendees_clause} at {location} on {date} at {time}{duration_clause}{recurrence_clause}{notes_clause}.",
    "Please {action} an event on {date} at {time}{attendees_clause} in {location}{duration_clause}{recurrence_clause}{notes_clause}.",
    "Add a calendar entry: {action}, {date} {time}, {location}{attendees_clause}{duration_clause}{recurrence_clause}{notes_clause}.",
    "Could you {action} a session at {location} on {date} at {time}{attendees_clause}{duration_clause}{recurrence_clause}{notes_clause}?",
]


def _clause(prefix: str, value: str | None) -> str:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return ""
    return f" {prefix} {value}"


def render_from_json(j: Dict) -> str:
    atts = j.get("attendees") or []
    att_str = None
    if isinstance(atts, list) and len(atts) > 0:
        att_str = ", ".join([str(a) for a in atts if isinstance(a, (str, int, float))])
    attendees_clause = _clause("with", att_str) if att_str else ""
    duration_clause = _clause("for", j.get("duration"))
    recurrence_clause = _clause("repeating", j.get("recurrence"))
    notes_clause = _clause("(note:)", j.get("notes"))
    tpl = random.choice(TEMPLATES)
    return tpl.format(
        action=j.get("action", "create"),
        location=j.get("location", "the office"),
        date=j.get("date", "01/01/2025"),
        time=j.get("time", "10:00 AM"),
        attendees_clause=attendees_clause,
        duration_clause=duration_clause,
        recurrence_clause=recurrence_clause,
        notes_clause=notes_clause,
    )


