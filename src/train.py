from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import EarlyStoppingCallback
import warnings
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from src.data import CalendarJsonDataset, FeatureMap
from src.metrics import json_valid, per_field_f1, exact_match


@dataclass
class RunConfig:
    base_id: str
    max_length: int
    epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    early_stopping_patience: int
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    qlora: bool


def load_config(default_yaml: str, lora_yaml: str) -> Dict:
    import yaml
    with open(default_yaml, "r") as f:
        base = yaml.safe_load(f)
    with open(lora_yaml, "r") as f:
        lp = yaml.safe_load(f)
    base["lora"] = lp.get("lora", {})
    base["qlora"] = lp.get("qlora", {})
    return base


def build_model_and_tokenizer(cfg: Dict):
    base_id = cfg["model"]["base_id"]
    qlora_cfg = cfg.get("qlora", {})
    use_qlora = qlora_cfg.get("enabled", True) and torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg.get("load_in_4bit", True),
            bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_id)

    # Required for gradient checkpointing compatibility per HF docs
    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_conf = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_conf.get("r", 16),
        lora_alpha=lora_conf.get("alpha", 32),
        lora_dropout=lora_conf.get("dropout", 0.05),
        target_modules=lora_conf.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # We will decode predictions using greedy argmax for metric computation
    # This is approximate but sufficient for sanity metrics.
    import numpy as np
    pred_ids = np.argmax(logits, axis=-1)
    return {"dummy": float((pred_ids == labels).mean())}


def train_entry(config_path: str = "configs/default.yaml", lora_path: str = "configs/lora.yaml"):
    cfg = load_config(config_path, lora_path)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    model, tokenizer = build_model_and_tokenizer(cfg)

    train_path = cfg["paths"]["train"]
    eval_path = cfg["paths"]["eval"]

    train_ds = CalendarJsonDataset(train_path)
    eval_ds = CalendarJsonDataset(eval_path)

    feature_map = FeatureMap(tokenizer, cfg["model"]["max_length"])

    pad_id = tokenizer.pad_token_id

    def collate(batch):
        feats = feature_map(batch)
        max_len = max(len(x) for x in feats["input_ids"])
        input_ids = []
        attention_mask = []
        labels = []
        for i in range(len(feats["input_ids"])):
            ids = feats["input_ids"][i]
            att = feats["attention_mask"][i]
            lab = feats["labels"][i]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(att + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)
        import torch
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # Disable mixed precision if no CUDA
    use_bf16 = cfg["model"].get("bf16", False) and torch.cuda.is_available()
    use_fp16 = cfg["model"].get("fp16", True) and torch.cuda.is_available()

    # Build TrainingArguments with backward-compatible fallback
    # Disable gradient checkpointing on CPU to avoid PyTorch checkpoint warnings/errors
    gc_enabled = cfg["training"].get("gradient_checkpointing", True) and torch.cuda.is_available()

    args_kwargs = dict(
        output_dir=cfg["paths"]["outputs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        num_train_epochs=cfg["training"]["epochs"],
        logging_steps=cfg["training"]["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        adam_beta1=cfg["training"]["adam_beta1"],
        adam_beta2=cfg["training"]["adam_beta2"],
        gradient_checkpointing=gc_enabled,
        report_to=["tensorboard"],
        logging_dir="results/tb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        no_cuda=True,
    )
    try:
        args = TrainingArguments(**args_kwargs)
        use_early_stopping = True
    except TypeError as e:
        warnings.warn(f"Falling back to older TrainingArguments signature due to: {e}")
        # Remove newer keys and try minimal config
        fallback_keys = {
            "evaluation_strategy",
            "eval_steps",
            "save_steps",
            "report_to",
            "logging_dir",
            "load_best_model_at_end",
            "metric_for_best_model",
            "greater_is_better",
        }
        minimal = {k: v for k, v in args_kwargs.items() if k not in fallback_keys}
        args = TrainingArguments(**minimal)
        use_early_stopping = False

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Ensure model parameters require grad (LoRA adapters) on CPU
    for n, p in model.named_parameters():
        if p.requires_grad:
            continue
        # base weights frozen; LoRA adapters should require grad automatically
        # No action needed unless all are frozen incorrectly
    # sanity assert at least some params trainable
    num_trainable = sum(p.requires_grad for p in model.parameters())
    if num_trainable == 0:
        raise RuntimeError("No trainable parameters found. Ensure LoRA adapters are attached correctly.")

    # Early stopping on eval_loss (only if supported)
    if use_early_stopping:
        patience = int(cfg["training"].get("early_stopping_patience", 2))
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer.train()
    # Save final checkpoint
    save_dir = cfg["paths"]["outputs"]
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    # If best model was loaded at end, also copy to best_outputs
    best_dir = cfg["paths"].get("best_outputs")
    if best_dir:
        import os, shutil
        os.makedirs(best_dir, exist_ok=True)
        for name in ["adapter_config.json", "adapter_model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "pytorch_model.bin"]:
            src = os.path.join(save_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(best_dir, name))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--lora", type=str, default="configs/lora.yaml")
    args = p.parse_args()
    train_entry(args.config, args.lora)
