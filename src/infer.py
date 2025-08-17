from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# --------- Utilities ---------
def _load_base_id_from_adapter(model_dir: str) -> Optional[str]:
    cfg_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            data = json.load(f)
        return data.get("base_model_name_or_path")
    return None


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Finds the first valid top-level JSON object substring (handles prompt echo / trailing prose).
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def safe_json_load(text: str) -> Optional[dict]:
    blob = extract_first_json_object(text)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


# --------- Model loading ---------
def load_model_and_tokenizer(
    adapter_dir: str,
    base_id: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a base model and applies a PEFT adapter from adapter_dir.
    - If base_id is not provided, tries to read it from adapter_config.json.
    """
    if base_id is None:
        base_id = _load_base_id_from_adapter(adapter_dir)
        if base_id is None:
            raise ValueError(
                "Could not determine base_id. Pass base_id explicitly or ensure adapter_config.json has base_model_name_or_path."
            )

    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        # ensure pad exists and is aligned with model EOS
        tok.pad_token = tok.eos_token

    # dtype default: bfloat16 if available, else float16 on Apple MPS prefers float32
    if torch_dtype is None:
        if torch.backends.mps.is_available():
            torch_dtype = torch.float32  # MPS is safest in fp32
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)

    # Put in eval mode
    model.eval()
    return model, tok


# --------- Inference ---------
def infer(
    event_text: str,
    adapter_dir: str,
    base_id: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    add_trailing_newline: bool = True,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """
    Deterministic JSON-first decoding with robust parsing.
    Returns {} if no valid JSON could be parsed.
    """
    from src.prompts import build_prompt, build_chatml_prompt

    model, tok = load_model_and_tokenizer(
        adapter_dir=adapter_dir,
        base_id=base_id,
        trust_remote_code=trust_remote_code,
    )

    # Use simple prompt format for base SmolLM-360M model
    prompt = f"Extract calendar information from: {event_text}\nCalendar JSON:"
    if add_trailing_newline and not prompt.endswith("\n"):
        prompt += "\n"

    print(f"Prompt: {repr(prompt)}")

    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    
    print(f"Input token length: {enc['input_ids'].shape[1]}")

    gen_kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        repetition_penalty=1.1,  # Add repetition penalty
    )

    with torch.no_grad():
        gen = model.generate(**gen_kwargs)

    # Decode and strip prompt echo
    out = tok.decode(gen[0], skip_special_tokens=True)
    print(f"Full output: {repr(out)}")
    
    # If the tokenizer applies a chat template & echoes the prompt, fall back to substring after prompt
    gen_text = out[len(prompt) :] if out.startswith(prompt) else out
    print(f"Generated text after prompt: {repr(gen_text)}")

    # Try robust JSON parse first
    obj = safe_json_load(gen_text)
    if obj is not None:
        print(f"Successfully parsed JSON: {obj}")
        return obj

    print("Failed to parse JSON, trying to find any JSON in output...")
    
    # Try to find JSON anywhere in the output
    obj2 = safe_json_load(out)
    if obj2 is not None:
        print(f"Found JSON in full output: {obj2}")
        return obj2

    print("No valid JSON found, returning empty dict")
    return {}


def test_base_model(event_text: str, base_id: str = "HuggingFaceTB/SmolLM-360M", max_new_tokens: int = 256, prompt_style: str = "default"):
    """Test what the base model generates without any fine-tuning."""
    from src.prompts import build_prompt, build_simple_prompt, build_few_shot_prompt
    
    print(f"Testing base model: {base_id} with prompt style: {prompt_style}")
    
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float32 if torch.backends.mps.is_available() else torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Choose prompt style
    if prompt_style == "simple":
        prompt = build_simple_prompt(event_text) + " "
    elif prompt_style == "few_shot":
        prompt = build_few_shot_prompt(event_text) + " "
    else:
        prompt = build_prompt(event_text) + "\n"
    
    print(f"Base model prompt: {repr(prompt)}")
    
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=1.1,
        )
    
    out = tok.decode(gen[0], skip_special_tokens=True)
    gen_text = out[len(prompt):] if out.startswith(prompt) else out
    
    print(f"Base model output: {repr(gen_text)}")
    
    # Try to parse JSON
    obj = safe_json_load(gen_text)
    if obj:
        print(f"Successfully parsed: {obj}")
    else:
        print("Failed to parse JSON from base model")
    
    return gen_text


# --------- CLI ---------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=str, help="Path to PEFT adapter dir")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--base", type=str, default=None, help="(Optional) base model id/path")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--test-base-only", action="store_true", help="Test only base model without adapter")
    p.add_argument("--prompt-style", type=str, default="default", choices=["default", "simple", "few_shot"], help="Prompt style to use")
    args = p.parse_args()

    if args.test_base_only:
        # Test base model only
        base_id = args.base or "HuggingFaceTB/SmolLM-360M"
        result = test_base_model(args.text, base_id, args.max_new_tokens, args.prompt_style)
        print("Base model result:")
        print(result)
    else:
        # Test with adapter
        if not args.adapter:
            print("Error: --adapter required unless using --test-base-only")
            exit(1)
            
        result = infer(
            event_text=args.text,
            adapter_dir=args.adapter,
            base_id=args.base,
            max_new_tokens=args.max_new_tokens,
        )
        print("Final result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))