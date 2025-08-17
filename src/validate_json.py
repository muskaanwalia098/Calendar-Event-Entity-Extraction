from __future__ import annotations

from typing import Dict

from augmentation.utils import TARGET_KEYS


def ensure_schema(o: Dict) -> Dict:
    return {k: o.get(k, None) for k in TARGET_KEYS}
