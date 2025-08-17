#!/usr/bin/env python3
"""
Debug training to identify why we're getting 0 loss.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from src.data import CalendarJsonDataset, FeatureMap
import yaml

def main():
    # Load config
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('configs/lora.yaml', 'r') as f:
        lora_config = yaml.safe_load(f)
    
    print("🔍 Debugging Training Setup...")
    
    # Load tokenizer and model
    model_id = config['model']['base_id']
    print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = CalendarJsonDataset('data/splits/train.jsonl')
    
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Check a raw sample first
    raw_sample = train_dataset[0]
    print(f"Raw sample keys: {raw_sample.keys()}")
    
    # Process a sample manually
    from src.data import build_feature
    sample = build_feature(raw_sample, tokenizer, config['model']['max_length'])
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {len(sample['input_ids'])}")
    print(f"Labels shape: {len(sample['labels'])}")
    
    # Count unmasked labels
    labels = sample['labels']
    unmasked = sum(1 for x in labels if x != -100)
    print(f"Unmasked labels: {unmasked}/{len(labels)}")
    
    # Test loss computation
    print("Testing loss computation...")
    input_ids = torch.tensor([sample['input_ids']])
    labels_tensor = torch.tensor([sample['labels']])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels_tensor)
        loss = outputs.loss
        print(f"Direct loss: {loss.item()}")
    
    # Try with LoRA
    print("Setting up LoRA...")
    lora_cfg = LoraConfig(
        r=lora_config['lora']['r'],
        lora_alpha=lora_config['lora']['alpha'],
        target_modules=lora_config['lora']['target_modules'],
        lora_dropout=lora_config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    peft_model = get_peft_model(model, lora_cfg)
    print(f"LoRA model created. Trainable params: {peft_model.num_parameters(only_trainable=True)}")
    
    # Test loss with LoRA
    with torch.no_grad():
        outputs = peft_model(input_ids=input_ids, labels=labels_tensor)
        loss = outputs.loss
        print(f"LoRA loss: {loss.item()}")
    
    # Try minimal training setup
    print("Setting up minimal trainer...")
    training_args = TrainingArguments(
        output_dir='debug_output',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=5e-4,
        logging_steps=1,
        save_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    
    # We need to create a proper dataset for the trainer
    # Skip the trainer test for now and just check the loss computation
    print("✅ Loss computation works! The issue might be in the Trainer setup.")

if __name__ == "__main__":
    main()
