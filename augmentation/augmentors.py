from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple

from augmentation.utils import TARGET_KEYS

def replace_substrings(text: str, mapping: Dict[str, str]) -> str:
    out = text
    for src, dst in mapping.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    return out


def swap_entities_with_pools(event_text: str, j: Dict, att_pool: List[str], loc_pool: List[str]) -> Tuple[str, Dict] | Tuple[None, None]:
    """Return (new_text, new_output_json) if any swap applied, else (None, None)."""
    atts = j.get("attendees") or []
    loc = j.get("location")
    repl = {}
    new_j = {**j}
    if isinstance(atts, list) and atts:
        new_atts = []
        for a in atts:
            if isinstance(a, str) and a.strip():
                cand = random.choice(att_pool) if att_pool else a
                repl[a] = cand
                new_atts.append(cand)
            else:
                new_atts.append(a)
        new_j["attendees"] = new_atts
    if isinstance(loc, str) and loc.strip():
        cand_l = random.choice(loc_pool) if loc_pool else loc
        repl[loc] = cand_l
        new_j["location"] = cand_l
    if repl:
        new_text = replace_substrings(event_text, repl)
        return new_text, new_j
    return None, None



# ------------------- Sanity check helpers -------------------
def _event_text_signature(text: str) -> str:
    # Lowercase, strip, collapse inner whitespace
    t = (text or "").strip().lower()
    t = " ".join(t.split())
    return t


def ensure_output_schema_row(row: Dict) -> Dict:
    # Unify key name to "output" and ensure all TARGET_KEYS exist with None for missing
    event_text = (row.get("event_text") or "").strip()
    payload = row.get("output") or row.get("json") or {}
    normalized = {}
    for k in TARGET_KEYS:
        v = payload.get(k, None)
        if isinstance(v, str) and v.strip() == "":
            v = None
        normalized[k] = v
    return {"event_text": event_text, "output": normalized}


def drop_split_leakage(train_rows: List[Dict], eval_rows: List[Dict], test_rows: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    # Remove near-duplicate event_text across splits using normalized signature
    # Keep precedence: train > eval > test.
    # 1) Build sets of signatures
    train_sigs = {_event_text_signature(r.get("event_text", "")) for r in train_rows}
    # 2) Filter eval to exclude anything in train
    filtered_eval = [r for r in eval_rows if _event_text_signature(r.get("event_text", "")) not in train_sigs]
    eval_sigs = {_event_text_signature(r.get("event_text", "")) for r in filtered_eval}
    # 3) Filter test to exclude anything in train or eval
    filtered_test = [
        r for r in test_rows
        if _event_text_signature(r.get("event_text", "")) not in train_sigs
        and _event_text_signature(r.get("event_text", "")) not in eval_sigs
    ]
    return train_rows, filtered_eval, filtered_test

