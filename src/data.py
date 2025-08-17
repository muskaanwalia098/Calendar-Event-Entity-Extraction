from __future__ import annotations

import json
from typing import Dict, List, Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from augmentation.utils import TARGET_KEYS
from src.prompts import build_prompt


def ensure_schema_dict(o: Dict) -> Dict:
    return {k: (o.get(k, None) if o.get(k, None) not in ("",) else None) for k in TARGET_KEYS}


class CalendarJsonDataset(Dataset):
    def __init__(self, path: str):
        self.rows: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                event_text = (obj.get("event_text") or "").strip()
                payload = obj.get("output") or obj.get("json") or {}
                payload = ensure_schema_dict(payload)
                self.rows.append({"event_text": event_text, "output": payload})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


def build_feature(example: Dict, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dict:
    # If already tokenized (defensive), just pass through
    if "input_ids" in example and "labels" in example:
        return example

    event_text = (example.get("event_text") or "").strip()
    output = example.get("output") or example.get("json") or {}
    output = {k: (output.get(k, None) if output.get(k, None) not in ("",) else None) for k in TARGET_KEYS}

    prompt = build_prompt(event_text) + "\n"
    tgt = json.dumps({k: output.get(k, None) for k in TARGET_KEYS}, ensure_ascii=False)
    text = prompt + tgt
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    labels = tokenized["input_ids"].copy()
    # mask prompt tokens in labels
    prompt_tokenized = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_tokenized)
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


class FeatureMap:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict:
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for ex in batch:
            f = build_feature(ex, self.tokenizer, self.max_length)
            out["input_ids"].append(f["input_ids"]) 
            out["attention_mask"].append(f["attention_mask"]) 
            out["labels"].append(f["labels"]) 
        return out
