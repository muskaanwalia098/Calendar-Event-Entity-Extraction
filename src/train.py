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
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

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


def detect_device():
    """Detect the best available device for Apple Silicon."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def build_model_and_tokenizer(cfg: Dict):
    base_id = cfg["model"]["base_id"]
    qlora_cfg = cfg.get("qlora", {})
    device = detect_device()
    
    # Only use QLoRA if explicitly enabled and on CUDA (not supported well on MPS)
    use_qlora = qlora_cfg.get("enabled", False) and device == "cuda" and HAS_BNB
    
    print(f"Using device: {device}")
    print(f"QLoRA enabled: {use_qlora}")

    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora and HAS_BNB:
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
        # Standard loading for MPS or CPU
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float32,  # Use float32 for MPS compatibility
        )
        
        # Move to appropriate device
        if device == "mps":
            model = model.to("mps")
        elif device == "cuda":
            model = model.to("cuda")
        # CPU stays on CPU by default

    # Required for gradient checkpointing compatibility per HF docs
    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_conf = cfg.get("lora", {})
    target_modules = lora_conf.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Filter target modules to only include those that exist in the model
    valid_modules = []
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name:
                valid_modules.append(target)
                break
    
    if not valid_modules:
        valid_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # fallback
    
    peft_config = LoraConfig(
        r=lora_conf.get("r", 16),
        lora_alpha=lora_conf.get("alpha", 32),
        lora_dropout=lora_conf.get("dropout", 0.05),
        target_modules=list(set(valid_modules)),  # remove duplicates
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
        """Simple collate function without complex device handling."""
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
        
        # Don't manually move to device - let Trainer handle it
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    device = detect_device()
    
    # Disable mixed precision for MPS (not well supported) and adjust for device
    use_bf16 = cfg["model"].get("bf16", False) and device == "cuda"
    use_fp16 = cfg["model"].get("fp16", False) and device == "cuda"
    
    # Set device-specific settings
    use_mps = device == "mps"
    no_cuda = device != "cuda"

    # Build TrainingArguments with backward-compatible fallback
    # Disable gradient checkpointing for MPS stability
    gc_enabled = cfg["training"].get("gradient_checkpointing", False) and device == "cuda"

    # Build training arguments with device-specific optimizations
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
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        adam_beta1=cfg["training"]["adam_beta1"],
        adam_beta2=cfg["training"]["adam_beta2"],
        gradient_checkpointing=gc_enabled,
        report_to=["tensorboard"],
        logging_dir="results/tb",
    )
    
    # Only add newer arguments if available in transformers version
    try:
        # Try with newer transformers arguments
        newer_args = {
            "evaluation_strategy": "steps",
            "eval_steps": cfg["training"]["eval_steps"],
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
        
        # Handle device selection more carefully
        if device == "cpu":
            newer_args["no_cuda"] = True
        elif device == "mps":
            # For MPS, we need to be more careful about setup
            newer_args["no_cuda"] = True  # Disable CUDA
            # Don't use use_mps_device as it may not be compatible
        
        test_args = {**args_kwargs, **newer_args}
        args = TrainingArguments(**test_args)
        use_early_stopping = True
    except (TypeError, ValueError) as e:
        print(f"Using basic TrainingArguments due to: {e}")
        # Fallback to basic arguments
        basic_args = {
            "output_dir": cfg["paths"]["outputs"],
            "per_device_train_batch_size": cfg["training"]["per_device_train_batch_size"],
            "per_device_eval_batch_size": cfg["training"]["per_device_eval_batch_size"],
            "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
            "learning_rate": cfg["training"]["lr"],
            "num_train_epochs": cfg["training"]["epochs"],
            "logging_steps": cfg["training"]["logging_steps"],
            "save_steps": cfg["training"]["save_steps"],
            "bf16": False,  # Disable for compatibility
            "fp16": False,  # Disable for compatibility
        }
        args = TrainingArguments(**basic_args)
        use_early_stopping = False
    # TrainingArguments already created above

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Ensure model parameters require grad (LoRA adapters)
    device = detect_device()
    
    # Move tokenizer pad token to same device as model if using MPS
    if device == "mps" and hasattr(tokenizer, 'pad_token_id'):
        # Ensure tokenizer works with MPS device
        pass  # tokenizer doesn't need to be moved
    
    # Verify trainable parameters
    num_trainable = sum(p.requires_grad for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
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
