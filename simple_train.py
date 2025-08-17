#!/usr/bin/env python3
"""
Simplified training script to isolate the zero loss issue.
"""

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from src.data import CalendarJsonDataset, build_feature

def simple_collate_fn(batch):
    """Simple collate function without complex device handling."""
    input_ids = []
    attention_mask = []
    labels = []
    
    max_len = max(len(item['input_ids']) for item in batch)
    
    for item in batch:
        ids = item['input_ids']
        mask = item['attention_mask'] 
        labs = item['labels']
        
        # Pad to max length
        pad_len = max_len - len(ids)
        input_ids.append(ids + [0] * pad_len)  # pad with 0 
        attention_mask.append(mask + [0] * pad_len)
        labels.append(labs + [-100] * pad_len)
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

class SimpleDataset:
    """Simple dataset wrapper that applies feature processing."""
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        raw_item = self.dataset[idx]
        return build_feature(raw_item, self.tokenizer, self.max_length)

def main():
    print("🔄 Starting Simplified Training...")
    
    # Load configs
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('configs/lora.yaml', 'r') as f:
        lora_config = yaml.safe_load(f)
    
    # Load model and tokenizer
    model_id = config['model']['base_id']
    print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use FP32 for stability
    )
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_cfg = LoraConfig(
        r=lora_config['lora']['r'],
        lora_alpha=lora_config['lora']['alpha'],
        target_modules=lora_config['lora']['target_modules'],
        lora_dropout=lora_config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_cfg)
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    
    # Load data
    print("Loading datasets...")
    train_dataset = CalendarJsonDataset('data/splits/train.jsonl')
    eval_dataset = CalendarJsonDataset('data/splits/eval.jsonl')
    
    # Wrap with feature processing
    train_processed = SimpleDataset(train_dataset, tokenizer, config['model']['max_length'])
    eval_processed = SimpleDataset(eval_dataset, tokenizer, config['model']['max_length'])
    
    print(f"Train size: {len(train_processed)}")
    print(f"Eval size: {len(eval_processed)}")
    
    # Test a single sample to verify loss computation
    print("Testing single sample loss...")
    sample = train_processed[0]
    input_ids = torch.tensor([sample['input_ids']])
    labels = torch.tensor([sample['labels']])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        test_loss = outputs.loss
        print(f"Single sample loss: {test_loss.item():.4f}")
    
    if test_loss.item() == 0:
        print("❌ Issue: Single sample loss is 0!")
        return
    else:
        print("✅ Single sample loss looks good!")
    
    # Setup training with minimal config
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='simple_output',
        num_train_epochs=1,  # Just 1 epoch for testing
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-4,
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=eval_processed,
        tokenizer=tokenizer,
        data_collator=simple_collate_fn,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
