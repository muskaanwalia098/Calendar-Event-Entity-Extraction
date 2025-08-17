#!/usr/bin/env python3
"""
Evaluate baseline SmolLM-360M performance on calendar entity extraction task.
This establishes the baseline before fine-tuning.
"""

import json
import torch
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.loss import EntityMetrics
import random
from pathlib import Path

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def evaluate_model_on_sample(model, tokenizer, event_text: str, max_new_tokens: int = 128) -> str:
    """Evaluate model on a single sample."""
    prompt = f"Extract calendar information from: {event_text}\nCalendar JSON:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after the prompt)
    generated_text = full_response[len(prompt):].strip()
    
    return generated_text

def compute_metrics(predictions: List[str], targets: List[Dict]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {
        'exact_match': 0.0,
        'json_validity': 0.0,
        'field_accuracy': 0.0,
        'action_accuracy': 0.0,
        'date_accuracy': 0.0,
        'time_accuracy': 0.0,
        'attendees_accuracy': 0.0,
        'location_accuracy': 0.0,
        'duration_accuracy': 0.0,
        'recurrence_accuracy': 0.0,
        'notes_accuracy': 0.0,
    }
    
    valid_jsons = 0
    total_field_accuracy = 0.0
    field_counts = {field: 0 for field in ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']}
    
    for pred_text, target in zip(predictions, targets):
        # Try to extract JSON from prediction
        pred_json = EntityMetrics.extract_json_from_text(pred_text)
        
        if pred_json is not None:
            valid_jsons += 1
            
            # Compute field-level accuracy
            field_acc = EntityMetrics.compute_field_accuracy(pred_json, target)
            total_field_accuracy += field_acc
            
            # Check exact match
            if pred_json == target:
                metrics['exact_match'] += 1.0
            
            # Check individual field accuracies
            for field in field_counts.keys():
                if pred_json.get(field) == target.get(field):
                    field_counts[field] += 1
    
    total_samples = len(predictions)
    
    # Calculate final metrics
    metrics['json_validity'] = valid_jsons / total_samples
    metrics['field_accuracy'] = total_field_accuracy / total_samples
    
    for field, correct_count in field_counts.items():
        metrics[f'{field}_accuracy'] = correct_count / total_samples
    
    metrics['exact_match'] = metrics['exact_match'] / total_samples
    
    return metrics

def main():
    """Main evaluation function."""
    print("🔄 Evaluating Baseline SmolLM-360M Performance...")
    
    # Load model and tokenizer
    print("📥 Loading base model...")
    model_id = "HuggingFaceTB/SmolLM-360M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use CPU for baseline evaluation to avoid MPS issues
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    
    # Load test data
    print("📊 Loading test dataset...")
    test_data = load_test_data('data/splits/test.jsonl')
    print(f"Loaded {len(test_data)} test examples")
    
    # Sample subset for quick evaluation
    sample_size = min(50, len(test_data))  # Test on 50 examples for baseline
    test_sample = random.sample(test_data, sample_size)
    
    print(f"🧪 Evaluating on {sample_size} examples...")
    
    predictions = []
    targets = []
    
    for i, example in enumerate(test_sample):
        # Get the original event text from the prompt
        prompt_text = example['prompt']
        event_text = prompt_text.replace("Extract calendar information from: ", "").replace("\\nCalendar JSON:", "")
        
        # Get target from completion
        target = json.loads(example['completion'])
        
        # Generate prediction
        try:
            prediction = evaluate_model_on_sample(model, tokenizer, event_text)
            predictions.append(prediction)
            targets.append(target)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{sample_size} examples...")
                
        except Exception as e:
            print(f"  Error on example {i+1}: {e}")
            predictions.append("")  # Empty prediction for failed cases
            targets.append(target)
    
    # Compute metrics
    print("📈 Computing metrics...")
    metrics = compute_metrics(predictions, targets)
    
    # Display results
    print("\\n📊 Baseline Performance Results:")
    print("=" * 50)
    print(f"Exact Match Accuracy: {metrics['exact_match']:.3f}")
    print(f"JSON Validity Rate:   {metrics['json_validity']:.3f}")
    print(f"Field Accuracy:       {metrics['field_accuracy']:.3f}")
    print("\\nField-Level Accuracies:")
    print(f"  Action:     {metrics['action_accuracy']:.3f}")
    print(f"  Date:       {metrics['date_accuracy']:.3f}")
    print(f"  Time:       {metrics['time_accuracy']:.3f}")
    print(f"  Attendees:  {metrics['attendees_accuracy']:.3f}")
    print(f"  Location:   {metrics['location_accuracy']:.3f}")
    print(f"  Duration:   {metrics['duration_accuracy']:.3f}")
    print(f"  Recurrence: {metrics['recurrence_accuracy']:.3f}")
    print(f"  Notes:      {metrics['notes_accuracy']:.3f}")
    
    # Save results
    results = {
        'model': model_id,
        'test_samples': sample_size,
        'metrics': metrics,
        'examples': []
    }
    
    # Save a few examples
    for i in range(min(5, len(predictions))):
        results['examples'].append({
            'input': test_sample[i]['prompt'].replace("Extract calendar information from: ", "").replace("\\nCalendar JSON:", ""),
            'target': targets[i],
            'prediction': predictions[i],
            'predicted_json': EntityMetrics.extract_json_from_text(predictions[i])
        })
    
    # Save baseline results
    Path('results').mkdir(exist_ok=True)
    with open('results/baseline_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\\n✅ Baseline evaluation complete!")
    print("📄 Results saved to results/baseline_evaluation.json")
    
    # Show some examples
    print("\\n📝 Sample Predictions:")
    for i, example in enumerate(results['examples'][:3]):
        print(f"\\nExample {i+1}:")
        print(f"Input: {example['input']}")
        print(f"Target: {json.dumps(example['target'], ensure_ascii=False)}")
        print(f"Generated: {example['prediction']}")
        print(f"Parsed JSON: {example['predicted_json']}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    main()
