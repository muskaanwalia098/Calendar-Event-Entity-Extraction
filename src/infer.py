from __future__ import annotations

import json
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import build_prompt


def load_model(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    return model, tok


def infer(event_text: str, model_dir: str) -> dict:
    model, tok = load_model(model_dir)
    prompt = build_prompt(event_text) + "\n"
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    out = tok.decode(gen[0], skip_special_tokens=True)
    # get substring after prompt
    gen_text = out[len(prompt):]
    try:
        return json.loads(gen_text)
    except Exception:
        return {}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--text", type=str, required=True)
    args = p.parse_args()
    print(json.dumps(infer(args.text, args.model), ensure_ascii=False))
