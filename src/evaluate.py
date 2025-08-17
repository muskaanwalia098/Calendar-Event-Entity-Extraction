from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import build_prompt
from src.metrics import json_valid, per_field_f1, exact_match
from augmentation.utils import TARGET_KEYS


def load_config(config_path: str) -> Dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_test_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def generate_json(model, tok, event_text: str, max_new_tokens: int = 160) -> Dict:
    prompt = build_prompt(event_text) + "\n"
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = tok.decode(gen[0], skip_special_tokens=True)
    gen_text = out[len(prompt):]
    try:
        return json.loads(gen_text)
    except Exception:
        return {}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--model_dir", type=str, default="")
    args = p.parse_args()

    cfg = load_config(args.config)
    test_path = cfg["paths"]["test"]
    model_dir = args.model_dir or cfg["paths"]["outputs"]

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")

    rows = load_test_rows(test_path)

    n = 0
    valid_json = 0
    exact = 0
    f1_sum = 0.0

    for r in rows:
        n += 1
        gold = r.get("output") or r.get("json") or {}
        pred = generate_json(model, tok, r["event_text"], max_new_tokens=cfg["model"].get("max_new_tokens", 160))
        if json_valid(pred):
            valid_json += 1
        f1, _ = per_field_f1(pred, gold)
        f1_sum += f1
        if exact_match(pred, gold):
            exact += 1

    print(json.dumps({
        "count": n,
        "json_valid": valid_json / max(n, 1),
        "per_field_f1": f1_sum / max(n, 1),
        "exact_match": exact / max(n, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
