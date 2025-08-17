from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.evaluate_finetuned import compute_metrics as compute_metrics_ft


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


def evaluate_model_on_sample(model, tokenizer, event_text: str, max_new_tokens: int = 150) -> str:
    prompt = f"Extract calendar information from: {event_text}\nCalendar JSON:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_response[len(prompt):].strip()
    return generated_text


def main():
    import argparse
    p = argparse.ArgumentParser(description="Compare baseline vs fine-tuned on random test samples")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--base_model_id", type=str, default="HuggingFaceTB/SmolLM-360M")
    p.add_argument("--adapter_dir", type=str, default="simple_output/checkpoint-277")
    p.add_argument("--sample_size", type=int, default=50)
    p.add_argument("--output", type=str, default="results/comparison_evaluation.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    test_path = cfg["paths"]["test"]

    # Load test data (prompt/completion format)
    data = load_test_rows(test_path)
    if not data:
        print("No test data found.")
        return
    sample_size = min(args.sample_size, len(data))
    test_sample = random.sample(data, sample_size)

    # Load base tokenizer/model
    base_model_id = args.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)

    # Load fine-tuned (base + adapters)
    finetuned_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
    finetuned_model = PeftModel.from_pretrained(finetuned_model, args.adapter_dir)

    # Evaluate baseline and fine-tuned
    print(f"Evaluating {sample_size} random test examples...")
    baseline_predictions: List[str] = []
    finetuned_predictions: List[str] = []
    targets: List[Dict] = []

    for i, example in enumerate(test_sample):
        prompt_text = example["prompt"]
        # recover the original event text from the prompt
        event_text = prompt_text.replace("Extract calendar information from: ", "").replace("\\nCalendar JSON:", "")
        target = json.loads(example["completion"]) if isinstance(example.get("completion"), str) else example.get("completion", {})

        try:
            b_pred = evaluate_model_on_sample(base_model, tokenizer, event_text, max_new_tokens=cfg["model"].get("max_new_tokens", 160))
        except Exception:
            b_pred = ""
        try:
            f_pred = evaluate_model_on_sample(finetuned_model, tokenizer, event_text, max_new_tokens=cfg["model"].get("max_new_tokens", 160))
        except Exception:
            f_pred = ""

        baseline_predictions.append(b_pred)
        finetuned_predictions.append(f_pred)
        targets.append(target)

    # Compute metrics using the shared metric function
    baseline_metrics = compute_metrics_ft(baseline_predictions, targets)
    finetuned_metrics = compute_metrics_ft(finetuned_predictions, targets)

    # Save detailed results
    Path("results").mkdir(exist_ok=True)
    results = {
        "test_samples": sample_size,
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    def fmt(x: float) -> str:
        try:
            return f"{x:.3f}"
        except Exception:
            return str(x)

    print("\nEvaluation summary (random sample):")
    for metric in ["exact_match", "json_validity", "field_accuracy"]:
        b = baseline_metrics.get(metric, 0.0)
        ft = finetuned_metrics.get(metric, 0.0)
        print(f"{metric}: baseline={fmt(b)}  finetuned={fmt(ft)}  delta={fmt(ft - b)}")
    print(f"\nSaved detailed report to {args.output}")


if __name__ == "__main__":
    main()
