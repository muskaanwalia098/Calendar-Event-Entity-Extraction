#!/usr/bin/env python3
"""
Prepare the assignment dataset from event_text_mapping.jsonl
for fine-tuning SmolLM-360M base model.

This script creates train/eval/test splits and formats data
for the base model (not instruct model).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

def load_raw_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the provided event_text_mapping.jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def format_for_base_model(event_text: str, output: Dict[str, Any]) -> Dict[str, str]:
    """
    Format data for base SmolLM-360M model.
    Since it's not an instruct model, we use simple prompt-completion format.
    """
    # Create a simple, clear prompt for the base model
    prompt = f"Extract calendar information from: {event_text}\nCalendar JSON:"
    
    # Create the target JSON response
    response = json.dumps(output, ensure_ascii=False)
    
    return {
        "prompt": prompt,
        "completion": response,
        "full_text": prompt + " " + response  # For training
    }

def create_splits(data: List[Dict], train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15):
    """Create train/eval/test splits."""
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.shuffle(data)
    n = len(data)
    
    train_end = int(n * train_ratio)
    eval_end = train_end + int(n * eval_ratio)
    
    return {
        'train': data[:train_end],
        'eval': data[train_end:eval_end],
        'test': data[eval_end:]
    }

def save_split(data: List[Dict], output_path: str):
    """Save a data split to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def analyze_dataset(data: List[Dict]) -> Dict[str, Any]:
    """Analyze the dataset for insights."""
    analysis = {
        'total_examples': len(data),
        'actions': {},
        'has_attendees': 0,
        'has_location': 0,
        'has_duration': 0,
        'has_recurrence': 0,
        'has_notes': 0,
        'avg_text_length': 0
    }
    
    text_lengths = []
    for item in data:
        text_lengths.append(len(item['event_text']))
        output = item['output']
        
        # Count actions
        action = output.get('action', 'unknown')
        analysis['actions'][action] = analysis['actions'].get(action, 0) + 1
        
        # Count non-null fields
        if output.get('attendees') is not None:
            analysis['has_attendees'] += 1
        if output.get('location') is not None:
            analysis['has_location'] += 1
        if output.get('duration') is not None:
            analysis['has_duration'] += 1
        if output.get('recurrence') is not None:
            analysis['has_recurrence'] += 1
        if output.get('notes') is not None:
            analysis['has_notes'] += 1
    
    analysis['avg_text_length'] = sum(text_lengths) / len(text_lengths)
    analysis['max_text_length'] = max(text_lengths)
    analysis['min_text_length'] = min(text_lengths)
    
    return analysis

def main():
    """Main function to prepare assignment data."""
    print("🔄 Preparing Assignment Dataset...")
    
    # Load raw data
    raw_data = load_raw_data('data/raw/event_text_mapping.jsonl')
    print(f"📊 Loaded {len(raw_data)} examples from assignment dataset")
    
    # Analyze dataset
    analysis = analyze_dataset(raw_data)
    print(f"\n📈 Dataset Analysis:")
    print(f"  Total examples: {analysis['total_examples']}")
    print(f"  Avg text length: {analysis['avg_text_length']:.1f} chars")
    print(f"  Text length range: {analysis['min_text_length']}-{analysis['max_text_length']}")
    print(f"  Examples with attendees: {analysis['has_attendees']}")
    print(f"  Examples with location: {analysis['has_location']}")
    print(f"  Examples with duration: {analysis['has_duration']}")
    print(f"  Examples with recurrence: {analysis['has_recurrence']}")
    print(f"  Examples with notes: {analysis['has_notes']}")
    print(f"  Top 10 actions: {dict(list(sorted(analysis['actions'].items(), key=lambda x: x[1], reverse=True))[:10])}")
    
    # Format data for base model
    formatted_data = []
    for item in raw_data:
        formatted_item = format_for_base_model(item['event_text'], item['output'])
        formatted_data.append(formatted_item)
    
    # Create splits
    splits = create_splits(formatted_data, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15)
    
    print(f"\n📂 Creating Data Splits:")
    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Eval: {len(splits['eval'])} examples") 
    print(f"  Test: {len(splits['test'])} examples")
    
    # Save splits
    save_split(splits['train'], 'data/splits/train.jsonl')
    save_split(splits['eval'], 'data/splits/eval.jsonl')
    save_split(splits['test'], 'data/splits/test.jsonl')
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📝 Sample formatted examples:")
    
    for i, example in enumerate(splits['train'][:2]):
        print(f"\nExample {i+1}:")
        print(f"  Prompt: {repr(example['prompt'])}")
        print(f"  Completion: {repr(example['completion'])}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
