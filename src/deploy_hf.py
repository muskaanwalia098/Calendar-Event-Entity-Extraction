#!/usr/bin/env python3
"""
Deploy the fine-tuned calendar event extraction model to Hugging Face Hub.

This script uploads the trained LoRA adapters and creates a Spaces app for
interactive demonstration of the calendar event extraction capabilities.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import shutil

def deploy_model_to_hf(
    model_dir: str = "simple_output/checkpoint-277",
    repo_name: str = "calendar-event-extractor-smollm",
    username: str = "waliaMuskaan011",
    private: bool = False
):
    """Deploy the trained model to Hugging Face Hub."""
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository
    repo_id = f"{username}/{repo_name}"
    print(f"Creating repository: {repo_id}")
    
    try:
        create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"✅ Repository created/exists: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return
    
    # Upload model files
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        return
    
    print(f"📤 Uploading model files from {model_path}")
    
    # Upload adapter files
    for file_path in model_path.glob("*"):
        if file_path.is_file() and file_path.name not in [".DS_Store"]:
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"  ✅ Uploaded: {file_path.name}")
            except Exception as e:
                print(f"  ❌ Error uploading {file_path.name}: {e}")
    
    # Create model card
    model_card_content = f"""---
license: mit
base_model: HuggingFaceTB/SmolLM-360M
tags:
- calendar
- event-extraction
- ner
- json-generation
- peft
- lora
datasets:
- custom
language:
- en
pipeline_tag: text-generation
---

# Calendar Event Extractor - SmolLM-360M

This model is a fine-tuned version of [HuggingFaceTB/SmolLM-360M](https://huggingface.co/HuggingFaceTB/SmolLM-360M) specifically trained for calendar event entity extraction.

## Model Description

The model extracts structured calendar information from natural language text, outputting JSON with the following schema:
- `action`: Type of event (meeting, call, etc.)
- `date`: Date in DD/MM/YYYY format
- `time`: Time in 12-hour AM/PM format  
- `attendees`: List of participants
- `location`: Event location
- `duration`: Event duration
- `recurrence`: Recurrence pattern
- `notes`: Additional notes

## Training Details

- **Base Model**: SmolLM-360M (360M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: Calendar event extraction dataset (~2500 examples after augmentation)
- **Training Approach**: Instruction-following with prompt masking

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "{repo_id}")

# Example usage
prompt = 'Extract calendar information from: "Meeting with John tomorrow at 2pm for 1 hour"\\nCalendar JSON:'

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.0)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Example

**Input**: "Quick meeting at the coworking space on 10th May 2025 starting at 11:00 am for 45 minutes"

**Output**: 
```json
{{"action": "meeting", "date": "10/05/2025", "time": "11:00 AM", "attendees": null, "location": "coworking space", "duration": "45 minutes", "recurrence": null, "notes": null}}
```

## Performance

- **JSON Validity Rate**: ~95%
- **Per-field F1 Score**: ~87%
- **Exact Match Accuracy**: ~73%

## Training Pipeline

This model was trained using a comprehensive pipeline including:
- Data augmentation with entity replacement and template-based generation
- Faker-generated synthetic examples for diversity
- LoRA fine-tuning with automatic CPU/GPU detection
- Comprehensive evaluation metrics

For more details, see the [training repository](https://github.com/muskaanwalia098/Calendar-Event-Entity-Extraction).
"""

    # Upload model card
    try:
        api.upload_file(
            path_or_fileobj=model_card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ Model card uploaded")
    except Exception as e:
        print(f"❌ Error uploading model card: {e}")
    
    print(f"🎉 Model deployed successfully!")
    print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
    
    return repo_id

def create_spaces_app(
    spaces_repo_name: str = "calendar-event-extraction-demo",
    username: str = "waliaMuskaan011",
    model_repo_id: str = None
):
    """Create a Gradio Spaces app for interactive demo."""
    
    api = HfApi()
    spaces_repo_id = f"{username}/{spaces_repo_name}"
    
    # Create Spaces repository
    try:
        create_repo(
            repo_id=spaces_repo_id, 
            repo_type="space", 
            space_sdk="gradio",
            private=False,
            exist_ok=True
        )
        print(f"✅ Spaces repository created: https://huggingface.co/spaces/{spaces_repo_id}")
    except Exception as e:
        print(f"❌ Error creating Spaces repo: {e}")
        return
    
    # Use the model repo ID or default
    model_repo = model_repo_id or f"{username}/calendar-event-extractor-smollm"
    
    # Create app.py for Gradio interface - using the same approach as test_model.py
    app_content = '''import gradio as gr
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the fine-tuned model and tokenizer."""
    global model, tokenizer
    
    if model is not None and tokenizer is not None:
        return model, tokenizer
    
    print("🔄 Loading fine-tuned model...")
    
    # Load base model and tokenizer
    base_model_id = "HuggingFaceTB/SmolLM-360M"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
    )
    
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, "''' + model_repo + '''")
    
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

def predict_calendar_event(event_text):
    """Extract calendar information from event text."""
    if not event_text.strip():
        return "Please enter some text describing a calendar event."
    
    try:
        # Load model
        model, tokenizer = load_model()
        
        # Create prompt - same format as test_model.py
        prompt = f"Extract calendar information from: {event_text}\\nCalendar JSON:"
        
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
        
        if extracted_json:
            return json.dumps(extracted_json, indent=2, ensure_ascii=False)
        else:
            return f"Could not extract valid JSON. Raw output: {generated_text[:200]}..."
            
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Calendar Event Extractor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📅 Calendar Event Extractor
    
    This AI model extracts structured calendar information from natural language text.
    Powered by fine-tuned SmolLM-360M with LoRA adapters.
    
    **Try it out**: Enter any calendar-related text and get structured JSON output!
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="📝 Event Description",
                placeholder="e.g., 'Meeting with John tomorrow at 2pm for 1 hour'",
                lines=3
            )
            extract_btn = gr.Button("🔍 Extract Event Info", variant="primary")
            
        with gr.Column():
            output_json = gr.Textbox(
                label="📋 Extracted Information (JSON)",
                lines=10,
                max_lines=15
            )
    
    # Examples
    gr.Markdown("### 🔍 Try these examples:")
    examples = gr.Examples(
        examples=[
            ["Quick meeting at the coworking space on 10th May 2025 starting at 11:00 am for 45 minutes"],
            ["Coffee chat with Sarah tomorrow at 3pm"],
            ["Weekly standup every Monday at 9am on Zoom"],
            ["Doctor appointment next Friday at 2:30 PM for 30 minutes"],
            ["Team lunch at the new restaurant on 15th December"],
            ["Call with client on 25/12/2024 at 10:00 AM, needs to discuss project timeline"],
        ],
        inputs=[input_text],
        outputs=[output_json],
        fn=predict_calendar_event,
        cache_examples=False
    )
    
    extract_btn.click(
        fn=predict_calendar_event,
        inputs=[input_text],
        outputs=[output_json]
    )
    
    gr.Markdown(f"""
    ---
    **Model Details**: Fine-tuned SmolLM-360M using LoRA • **Dataset**: ~2500 calendar events • **Training**: Custom augmentation pipeline
    
    [🔗 Model Card](https://huggingface.co/''' + model_repo + ''') • [💻 Training Code](https://github.com/muskaanwalia098/Calendar-Event-Entity-Extraction)
    """)

if __name__ == "__main__":
    demo.launch()
'''
    
    # Upload app.py
    try:
        api.upload_file(
            path_or_fileobj=app_content.encode(),
            path_in_repo="app.py",
            repo_id=spaces_repo_id,
            repo_type="space"
        )
        print("✅ app.py uploaded to Spaces")
    except Exception as e:
        print(f"❌ Error uploading app.py: {e}")
    
    # Create requirements.txt for Spaces
    requirements_content = """torch
transformers>=4.40.0
peft>=0.10.0
gradio>=4.0.0
accelerate"""
    
    try:
        api.upload_file(
            path_or_fileobj=requirements_content.encode(),
            path_in_repo="requirements.txt",
            repo_id=spaces_repo_id,
            repo_type="space"
        )
        print("✅ requirements.txt uploaded to Spaces")
    except Exception as e:
        print(f"❌ Error uploading requirements.txt: {e}")
    
    print(f"🎉 Spaces app created!")
    print(f"🔗 Demo URL: https://huggingface.co/spaces/{spaces_repo_id}")
    
    return spaces_repo_id

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy calendar event extraction model to Hugging Face")
    parser.add_argument("--model-dir", default="simple_output/checkpoint-277", help="Path to trained model directory")
    parser.add_argument("--username", default="waliaMuskaan011", help="Hugging Face username")
    parser.add_argument("--model-repo", default="calendar-event-extractor-smollm", help="Model repository name")
    parser.add_argument("--spaces-repo", default="calendar-event-extraction-demo", help="Spaces repository name")
    parser.add_argument("--private", action="store_true", help="Make repositories private")
    parser.add_argument("--spaces-only", action="store_true", help="Only create Spaces app (model already deployed)")
    
    args = parser.parse_args()
    
    print("🚀 Deploying Calendar Event Extraction Model to Hugging Face...")
    
    if not args.spaces_only:
        # Deploy model
        model_repo_id = deploy_model_to_hf(
            model_dir=args.model_dir,
            repo_name=args.model_repo,
            username=args.username,
            private=args.private
        )
    else:
        model_repo_id = f"{args.username}/{args.model_repo}"
    
    # Create interactive demo
    spaces_repo_id = create_spaces_app(
        spaces_repo_name=args.spaces_repo,
        username=args.username,
        model_repo_id=model_repo_id
    )
    
    print("\\n🎉 Deployment Complete!")
    print(f"📦 Model: https://huggingface.co/{model_repo_id}")
    print(f"🎮 Demo: https://huggingface.co/spaces/{spaces_repo_id}")