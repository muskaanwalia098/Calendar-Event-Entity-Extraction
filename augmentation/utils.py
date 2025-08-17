from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TARGET_KEYS = ["action","date","time","attendees","location","duration","recurrence","notes"]


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def canonicalize_output(obj: Dict) -> Dict:
    # Raw data uses "output"; ensure canonical structure {"event_text", "output"}
    event_text = obj.get("event_text", "").strip()
    payload = obj.get("output") or obj.get("json") or {}
    out_obj = {k: payload.get(k, None) for k in TARGET_KEYS}
    return {"event_text": event_text, "output": out_obj}


def dedupe_text_output(pairs: Iterable[Dict]) -> List[Dict]:
    seen: set[Tuple[str, str]] = set()
    unique: List[Dict] = []
    import orjson  # rely on env having this after reset
    for p in pairs:
        sig = (p["event_text"].strip(), orjson.dumps(p["output"], option=orjson.OPT_SORT_KEYS).decode())
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(p)
    return unique


