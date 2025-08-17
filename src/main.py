from __future__ import annotations

"""
Main entrypoint for fine-tuning the calendar-event extractor.

Data flow and preprocessing
  - Source data: reads JSONL files from `data/splits/{train,eval}.jsonl`. Each row has
    `event_text` and `output` (8 keys: action, date, time, attendees, location, duration,
    recurrence, notes). Missing values are JSON null.

  - Cleaning/schema: loader ensures all 8 keys are present, converts empty strings to null,
    and preserves the original `event_text` as-is (no lemmatization by default, to keep
    lexical variety intact).

  - Prompt/target: `src/prompts.py` builds a deterministic instruction prompt from
    `event_text`. Target is the canonical JSON string with stable key ordering. We feed the
    model a single sequence: `prompt + "\n" + target`.

  - Tokenization: Hugging Face tokenizer encodes the full sequence with truncation at
    `model.max_length` from `configs/default.yaml`. Choose a value large enough so the target
    JSON is not truncated (we default to 384).

  - Loss masking: labels are a copy of `input_ids`; all prompt tokens are masked to -100 so
    cross-entropy is computed only on the JSON target tokens.

  - Padding/collation: batches are padded to the longest sequence in the batch using the
    tokenizer `pad_token_id`. Label padding uses -100 so padded positions don’t contribute to
    the loss.

Training setup
  - Model: loads `HuggingFaceTB/SmolLM-360M` by default. LoRA adapters are attached to
    attention projections (q/k/v/o). Base weights are frozen; only LoRA parameters train.

  - qLoRA: auto-enabled only when CUDA is available; on CPU (Intel Mac) training falls back
    to standard LoRA without 4-bit quantization. `use_cache` is disabled when needed for
    checkpointing compatibility.

  - Trainer: builds TrainingArguments from YAML; logs to TensorBoard at `results/tb`, and
    optionally performs early stopping on `eval_loss` when supported by your Transformers
    version.

Configuration files
  - `configs/default.yaml`
    - `paths`: locations for train/eval/test, output checkpoints, and best checkpoint mirror.
    - `model`: `base_id`, `max_length`, `max_new_tokens`, precision flags (`bf16`/`fp16`).
    - `training`: epochs, per-device batch sizes, gradient accumulation, lr, weight decay,
      warmup ratio, logging/eval/save steps, early stopping patience, workers, gradient
      checkpointing flag, etc. Some fields (e.g., advanced scheduler/optimizer knobs) may be
      ignored by older Transformers versions; upgrade if needed.
      
  - `configs/lora.yaml`
    - `lora`: LoRA rank (`r`), `alpha`, `dropout`, and target modules (q_proj/k_proj/v_proj/o_proj).
    - `qlora`: flags for 4-bit NF4 quantization (effective only on CUDA devices).

Run:
  python -m src.main --config configs/default.yaml --lora configs/lora.yaml

View logs:
  tensorboard --logdir results/tb
"""

import argparse

from src.train import train_entry


def main():
    p = argparse.ArgumentParser(description="Train and evaluate calendar event extractor")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--lora", type=str, default="configs/lora.yaml")
    args = p.parse_args()
    train_entry(args.config, args.lora)


if __name__ == "__main__":
    main()
