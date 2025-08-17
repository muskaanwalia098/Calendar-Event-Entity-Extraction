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
                
                # Handle different data formats
                if "prompt" in obj and "completion" in obj:
                    # Assignment format: {"prompt": "...", "completion": "...", "full_text": "..."}
                    self.rows.append({
                        "prompt": obj["prompt"],
                        "completion": obj["completion"],
                        "full_text": obj.get("full_text", obj["prompt"] + " " + obj["completion"])
                    })
                elif "messages" in obj:
                    # Messages format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                    # Store messages for chat template processing
                    self.rows.append({"messages": obj["messages"]})
                elif "text" in obj:
                    # Legacy ChatML format: {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}
                    self.rows.append({"text": obj["text"]})
                else:
                    # Legacy format support
                    event_text = (obj.get("event_text") or "").strip()
                    payload = obj.get("output") or obj.get("json") or {}
                    payload = ensure_schema_dict(payload)
                    prompt = build_prompt(event_text)
                    response = json.dumps(payload, ensure_ascii=False)
                    self.rows.append({
                        "prompt": prompt,
                        "response": response
                    })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


def build_feature(example: Dict, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dict:
    # If already tokenized (defensive), just pass through
    if "input_ids" in example and "labels" in example:
        return example

    if "prompt" in example and "completion" in example:
        # Handle assignment format for base model
        prompt = example["prompt"]
        completion = example["completion"]
        
        # For base models, we train on the full text but mask the prompt
        full_text = prompt + " " + completion
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        labels = tokenized["input_ids"].copy()
        
        # Mask prompt tokens in labels (only train on completion)
        # Be more careful about where to start the completion
        prompt_only = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_only)
        
        # Mask the prompt part, leaving the completion (including the space and JSON)
        if prompt_len < len(labels):
            labels[:prompt_len] = [-100] * prompt_len
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    elif "messages" in example:
        # Handle messages format with chat template
        messages = example["messages"]
        
        # Apply chat template to get the full formatted text
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback to manual ChatML formatting
            formatted_messages = []
            for msg in messages:
                formatted_messages.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            full_text = "\n".join(formatted_messages)
        
        # Tokenize the full conversation
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        labels = tokenized["input_ids"].copy()
        
        # Find where assistant response starts for masking
        # We want to mask everything except the assistant's response
        user_part = tokenizer.apply_chat_template(
            messages[:-1],  # All messages except the last (assistant)
            tokenize=False,
            add_generation_prompt=True
        )
        
        user_tokenized = tokenizer(user_part, add_special_tokens=False)["input_ids"]
        user_len = len(user_tokenized)
        
        # Mask the user part (prompt), only train on assistant response
        if user_len < len(labels):
            labels[:user_len] = [-100] * user_len
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
    
    elif "text" in example:
        # Handle legacy ChatML format
        full_text = example["text"]
        
        # Find where assistant response starts to mask properly
        assistant_start = full_text.find("<|im_start|>assistant\n")
        if assistant_start == -1:
            # Fallback: just train on everything
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy(),
            }
        
        # Split to get prompt part
        prompt_part = full_text[:assistant_start + len("<|im_start|>assistant\n")]
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        labels = tokenized["input_ids"].copy()
        
        # Mask prompt tokens in labels (only train on assistant response)
        prompt_tokenized = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokenized)
        if prompt_len < len(labels):
            labels[:prompt_len] = [-100] * prompt_len
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
    
    else:
        # Handle legacy prompt/response format
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        
        # For instruction models, we want to train on the full conversation
        # but only compute loss on the assistant's response
        full_text = prompt + "\n" + response
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        labels = tokenized["input_ids"].copy()
        
        # Mask prompt tokens in labels (only train on response)
        prompt_tokenized = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
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
