from __future__ import annotations

import json
from typing import Dict, List, Tuple

from augmentation.utils import TARGET_KEYS


def canonicalize(o: Dict) -> Dict:
    # Lowercase strings, keep None as is
    out = {}
    for k in TARGET_KEYS:
        v = o.get(k, None)
        if isinstance(v, str):
            v = v.strip().lower()
        out[k] = v
    return out


def json_valid(o: Dict) -> bool:
    # Must have all keys
    return all(k in o for k in TARGET_KEYS)


def per_field_f1(pred: Dict, gold: Dict) -> Tuple[float, Dict[str, float]]:
    p = canonicalize(pred)
    g = canonicalize(gold)
    field_scores = {}
    correct = 0
    for k in TARGET_KEYS:
        field_scores[k] = 1.0 if p.get(k) == g.get(k) else 0.0
        correct += field_scores[k]
    micro_f1 = correct / len(TARGET_KEYS)
    return micro_f1, field_scores


def exact_match(pred: Dict, gold: Dict) -> bool:
    p = canonicalize(pred)
    g = canonicalize(gold)
    return all(p.get(k) == g.get(k) for k in TARGET_KEYS)
