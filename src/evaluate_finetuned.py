#!/usr/bin/env python3
"""
Evaluate fine-tuned model performance and compare with baseline.
"""

import json
import torch
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.loss import EntityMetrics
import random

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def evaluate_model_on_sample(model, tokenizer, event_text: str, max_new_tokens: int = 100) -> str:
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
    print("🔄 Evaluating Fine-Tuned Model Performance...")
    
    # Load base model and tokenizer
    print("📥 Loading base model...")
    base_model_id = "HuggingFaceTB/SmolLM-360M"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
    )
    
    # Load fine-tuned model
    print("📥 Loading fine-tuned model...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_model, "simple_output/checkpoint-277")
    
    # Load test data
    print("📊 Loading test dataset...")
    test_data = load_test_data('data/splits/test.jsonl')
    print(f"Loaded {len(test_data)} test examples")
    
    # Sample for evaluation (using all test data since it's manageable)
    test_sample = test_data[:30]  # Use 30 examples for thorough evaluation
    
    print(f"🧪 Evaluating on {len(test_sample)} examples...")
    
    # Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_predictions = []
    targets = []
    
    for i, example in enumerate(test_sample):
        # Get the original event text from the prompt
        prompt_text = example['prompt']
        event_text = prompt_text.replace("Extract calendar information from: ", "").replace("\\nCalendar JSON:", "")
        
        # Get target from completion
        target = json.loads(example['completion'])
        
        try:
            prediction = evaluate_model_on_sample(base_model, tokenizer, event_text)
            baseline_predictions.append(prediction)
            targets.append(target)
            
            if (i + 1) % 10 == 0:
                print(f"  Baseline: {i + 1}/{len(test_sample)} completed")
                
        except Exception as e:
            print(f"  Baseline error on example {i+1}: {e}")
            baseline_predictions.append("")
            targets.append(target)
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    finetuned_predictions = []
    
    for i, example in enumerate(test_sample):
        prompt_text = example['prompt']
        event_text = prompt_text.replace("Extract calendar information from: ", "").replace("\\nCalendar JSON:", "")
        
        try:
            prediction = evaluate_model_on_sample(finetuned_model, tokenizer, event_text)
            finetuned_predictions.append(prediction)
            
            if (i + 1) % 10 == 0:
                print(f"  Fine-tuned: {i + 1}/{len(test_sample)} completed")
                
        except Exception as e:
            print(f"  Fine-tuned error on example {i+1}: {e}")
            finetuned_predictions.append("")
    
    # Compute metrics
    print("📈 Computing metrics...")
    baseline_metrics = compute_metrics(baseline_predictions, targets)
    finetuned_metrics = compute_metrics(finetuned_predictions, targets)
    
    # Display comparison results
    print("\\n📊 Performance Comparison Results:")
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for metric in ['exact_match', 'json_validity', 'field_accuracy']:
        baseline_val = baseline_metrics[metric]
        finetuned_val = finetuned_metrics[metric]
        improvement = finetuned_val - baseline_val
        print(f"{metric:<20} {baseline_val:<15.3f} {finetuned_val:<15.3f} {improvement:+.3f}")
    
    print("\\nField-Level Accuracy Comparison:")
    print("-" * 70)
    for field in ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']:
        baseline_val = baseline_metrics[f'{field}_accuracy']
        finetuned_val = finetuned_metrics[f'{field}_accuracy']
        improvement = finetuned_val - baseline_val
        print(f"{field:<20} {baseline_val:<15.3f} {finetuned_val:<15.3f} {improvement:+.3f}")
    
    # Save detailed results
    results = {
        'test_samples': len(test_sample),
        'baseline_metrics': baseline_metrics,
        'finetuned_metrics': finetuned_metrics,
        'examples': []
    }
    
    # Save some example comparisons
    for i in range(min(5, len(test_sample))):
        results['examples'].append({
            'input': test_sample[i]['prompt'].replace("Extract calendar information from: ", "").replace("\\nCalendar JSON:", ""),
            'target': targets[i],
            'baseline_prediction': baseline_predictions[i],
            'finetuned_prediction': finetuned_predictions[i],
            'baseline_json': EntityMetrics.extract_json_from_text(baseline_predictions[i]),
            'finetuned_json': EntityMetrics.extract_json_from_text(finetuned_predictions[i])
        })
    
    # Save results
    with open('results/comparison_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\\n✅ Evaluation complete!")
    print("📄 Detailed results saved to results/comparison_evaluation.json")
    
    # Show example comparisons
    print("\\n📝 Sample Predictions Comparison:")
    for i, example in enumerate(results['examples'][:3]):
        print(f"\\nExample {i+1}:")
        print(f"Input: {example['input']}")
        print(f"Target: {json.dumps(example['target'], ensure_ascii=False)}")
        print(f"Baseline: {example['baseline_prediction'][:100]}...")
        print(f"Fine-tuned: {example['finetuned_prediction'][:100]}...")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    main()
