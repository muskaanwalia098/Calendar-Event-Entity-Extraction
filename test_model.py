#!/usr/bin/env python3
"""
Simple script to test the fine-tuned calendar entity extraction model.
Run this to test your own examples!
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

def load_model():
    """Load the fine-tuned model and tokenizer."""
    print("🔄 Loading fine-tuned model...")
    
    # Load base model and tokenizer
    base_model_id = "HuggingFaceTB/SmolLM-360M"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
    )
    
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(model, "simple_output/checkpoint-277")
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def extract_json_from_text(text):
    """Extract the first JSON object from text."""
    try:
        # Find first { and matching }
        start = text.find('{')
        if start == -1:
            return None
        
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    json_str = text[start:i+1]
                    return json.loads(json_str)
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

def predict_calendar_event(model, tokenizer, event_text):
    """Extract calendar information from event text."""
    # Create prompt
    prompt = f"Extract calendar information from: {event_text}\nCalendar JSON:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_response[len(prompt):].strip()
    
    # Extract JSON
    extracted_json = extract_json_from_text(generated_text)
    
    return generated_text, extracted_json

def test_examples():
    """Test with some example inputs."""
    examples = [
        "Meeting with John tomorrow at 2pm for 1 hour",
        "Lunch with Sarah on Friday at 12:30pm at the cafe",
        "Team standup every Monday at 9am",
        "Doctor appointment on Dec 15th at 3:00pm",
        "Conference call with clients next Tuesday at 10am for 2 hours",
        "Birthday party for Mom on Saturday at 6pm at her house"
    ]
    
    print("\n🧪 Testing with example inputs...")
    print("=" * 60)
    
    model, tokenizer = load_model()
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example}")
        print("-" * 40)
        
        generated_text, extracted_json = predict_calendar_event(model, tokenizer, example)
        
        if extracted_json:
            print("✅ Extracted JSON:")
            print(json.dumps(extracted_json, indent=2, ensure_ascii=False))
        else:
            print("❌ Could not extract valid JSON")
            print(f"Raw output: {generated_text[:200]}...")

def interactive_mode():
    """Interactive mode for testing your own inputs."""
    print("\n🔧 Interactive Mode")
    print("=" * 60)
    print("Enter your calendar event descriptions to extract entities!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    model, tokenizer = load_model()
    
    while True:
        try:
            # Get user input
            user_input = input("📅 Enter calendar event: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                print("Please enter a calendar event description.")
                continue
            
            print("\n🔄 Processing...")
            
            # Get prediction
            generated_text, extracted_json = predict_calendar_event(model, tokenizer, user_input)
            
            print("\n📊 Results:")
            print("-" * 30)
            
            if extracted_json:
                print("✅ Extracted Calendar Information:")
                print(json.dumps(extracted_json, indent=2, ensure_ascii=False))
                
                # Show field summary
                print("\n📋 Field Summary:")
                for field in ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']:
                    value = extracted_json.get(field)
                    if value is not None:
                        print(f"  {field.capitalize()}: {value}")
                    else:
                        print(f"  {field.capitalize()}: (not specified)")
            else:
                print("❌ Could not extract valid JSON")
                print(f"Raw model output: {generated_text}")
            
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    print("🎯 Calendar Event Entity Extraction - Test Script")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Command line mode
        event_text = " ".join(sys.argv[1:])
        print(f"Testing: {event_text}")
        
        model, tokenizer = load_model()
        generated_text, extracted_json = predict_calendar_event(model, tokenizer, event_text)
        
        if extracted_json:
            print("\n✅ Extracted JSON:")
            print(json.dumps(extracted_json, indent=2, ensure_ascii=False))
        else:
            print("\n❌ Could not extract valid JSON")
            print(f"Raw output: {generated_text}")
    else:
        # Interactive mode
        print("Choose an option:")
        print("1. Test with example inputs")
        print("2. Interactive mode (enter your own text)")
        
        while True:
            choice = input("\nEnter choice (1/2): ").strip()
            
            if choice == "1":
                test_examples()
                break
            elif choice == "2":
                interactive_mode()
                break
            else:
                print("Please enter 1 or 2")

if __name__ == "__main__":
    main()
