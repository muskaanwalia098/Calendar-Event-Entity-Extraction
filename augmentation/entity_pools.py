from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple


def build_entity_pools(rows: List[Dict]) -> Tuple[List[str], List[str]]:
    att_pool: Counter = Counter()
    loc_pool: Counter = Counter()
    for r in rows:
        j = r.get("output", {}) or r.get("json", {}) or {}
        atts = j.get("attendees") or []
        if isinstance(atts, list):
            for a in atts:
                if isinstance(a, str) and a.strip():
                    att_pool[a.strip()] += 1
        loc = j.get("location")
        if isinstance(loc, str) and loc.strip():
            loc_pool[loc.strip()] += 1
    return [a for a, _ in att_pool.most_common(500)], [l for l, _ in loc_pool.most_common(500)]


