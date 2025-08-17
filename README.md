# Fine-tuning SmolLM-360M for Calendar-Event Entity Extraction

## Problem Statement

Natural language processing of calendar requests remains challenging due to the unstructured nature of human communication. Users express scheduling needs using diverse phrasings, formats, and implicit information. This project addresses the need for a robust system that can extract structured calendar information from unstructured text inputs.

**Use Case**: Transform natural language calendar requests like *"Quick meeting at the coworking space on 10th, May 2025 starting at 11:00 am for 45 minutes"* into structured JSON with standardized fields: `action`, `date`, `time`, `attendees`, `location`, `duration`, `recurrence`, `notes`.

## Methodology Overview

This project implements a comprehensive pipeline for fine-tuning SmolLM-360M (a 360-million parameter causal language model) using Parameter Efficient Fine-Tuning (PEFT) techniques. Our approach combines:

1. **Data Augmentation Pipeline**: Sophisticated augmentation using entity replacement, template-based rendering, and Faker-generated synthetic examples to expand the dataset from ~792 to 2500+ examples.

2. **Modular Architecture**: Structured codebase with separate modules for data processing (`augmentation/`), training (`src/`), configuration (`configs/`), and evaluation (`scripts/`).

3. **LoRA/QLoRA Fine-tuning**: Memory-efficient training using Low-Rank Adaptation, with automatic fallback to CPU-compatible configurations.

4. **Robust Evaluation**: Multi-metric evaluation including JSON validity, per-field F1 scores, and exact-match accuracy.

You will:

- Process and augment the initial dataset using multiple augmentation strategies
- Apply standardized preprocessing with deterministic train/eval/test splits
- Fine-tune using LoRA (CPU/GPU) or QLoRA (CUDA-only) approaches
- Evaluate model performance using comprehensive metrics
- Deploy the model for inference and assessment

## Table of Contents

- [Project Setup](#project-setup)
- [Dataset & Data Processing](#dataset--data-processing)
- [Data Augmentation Pipeline](#data-augmentation-pipeline)
- [Model Architecture](#model-architecture)
- [Fine-tuning Approach](#fine-tuning-approach)
- [Training Pipeline](#training-pipeline)
- [Running the Scripts](#running-the-scripts)
- [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Configuration Files](#configuration-files)
- [Deployment & Inference](#deployment--inference)

## Project Setup

### Requirements

- Python 3.10+
- CPU training supported (Intel Mac compatible)
- GPU recommended for faster training (NVIDIA for QLoRA support)
- For QLoRA: NVIDIA GPU with CUDA; `bitsandbytes` does not support Apple MPS

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For additional stability on Intel Mac
pip install 'numpy<2.0' 'torch>=2.2' tensorboard
```

## Dataset & Data Processing

### Initial Dataset
- Source: `data/raw/event_text_mapping.jsonl` (~792 examples)
- Format: Each line contains `event_text` (natural language) and `output` (structured JSON)
- Schema: 8 standardized keys: `action`, `date`, `time`, `attendees`, `location`, `duration`, `recurrence`, `notes`

Example:
```json
{"event_text": "Quick meeting at the coworking space on 10th, May 2025 starting at 11:00 am for 45 minutes.", "output": {"action": "meeting", "date": "10/05/2025", "time": "11:00 AM", "attendees": null, "location": "coworking space", "duration": "45 minutes", "recurrence": null, "notes": null}}
```

### Data Processing Pipeline
1. **Schema Canonicalization**: Ensures all 8 keys are present, converts empty strings to `null`
2. **Date/Time Normalization**: Standardizes to DD/MM/YYYY and 12-hour AM/PM format
3. **Deduplication**: Removes near-duplicate entries based on text similarity
4. **Deterministic Splitting**: Creates 75/15/10 train/eval/test splits with leakage prevention

## Data Augmentation Pipeline

Located in `augmentation/`, this modular pipeline expands the dataset using multiple sophisticated techniques:

### Augmentation Strategies
1. **Entity Replacement**: Swaps attendees and locations using entity pools built from the original dataset
2. **Template-based Rendering**: Generates alternative phrasings from structured JSON using diverse templates
3. **Faker Synthesis**: Creates entirely new examples with realistic names, dates, times, and locations
4. **Style Diversification**: Varies linguistic patterns while preserving semantic meaning

### Key Features
- **Data-driven Approach**: Entity pools extracted from real data to avoid hardcoding
- **Realistic Diversity**: Faker generates varied date formats, time ranges, and recurrence patterns
- **Controlled Augmentation**: Exactly one variant per original example to prevent over-augmentation
- **Schema Consistency**: All generated examples maintain the 8-key structure with proper null handling

### Running Data Augmentation

```bash
# Generate augmented dataset with 1000 synthetic examples
python -m augmentation.main --synth 1000 --seed 42

# Output: data/processed/augmented.jsonl + data/splits/{train,eval,test}.jsonl
```

## Model Architecture

**SmolLM-360M** (`HuggingFaceTB/SmolLM-360M`) is a compact causal language model designed for efficient fine-tuning on specialized tasks. With 360 million parameters, it offers a good balance between performance and computational requirements, making it suitable for both research and deployment scenarios.

**Architecture Benefits**: The model's moderate size allows for effective fine-tuning on consumer hardware while maintaining sufficient capacity for complex sequence-to-sequence tasks like structured information extraction.

## Fine-tuning Approach

### LoRA (Low-Rank Adaptation)
- **Method**: Freezes base model weights and trains small adapter matrices on attention projections (q_proj, k_proj, v_proj, o_proj)
- **Efficiency**: Reduces trainable parameters by ~99% while maintaining performance
- **Configuration**: Rank=8, Alpha=16, Dropout=0.1 for stability
- **Compatibility**: Works on both CPU and GPU environments

### QLoRA (Quantized LoRA)
- **Enhancement**: Combines LoRA with 4-bit quantization using NF4 format
- **Memory Reduction**: Further decreases VRAM usage for GPU training
- **Requirements**: NVIDIA GPU with CUDA support
- **Automatic Fallback**: System detects hardware and chooses appropriate method

## Training Pipeline

### Data Flow
1. **Loading**: Reads from `data/splits/{train,eval}.jsonl` with schema validation
2. **Prompt Construction**: Uses deterministic instruction template from `src/prompts.py`
3. **Tokenization**: Encodes prompt + target JSON with Hugging Face tokenizer
4. **Loss Masking**: Masks prompt tokens (-100) to compute loss only on JSON targets
5. **Dynamic Padding**: Batches padded to longest sequence for efficient training

### Training Features
- **TensorBoard Logging**: Real-time loss tracking at `results/tb`
- **Early Stopping**: Monitors eval_loss with configurable patience
- **Best Checkpoint Saving**: Automatically saves best performing model
- **CPU/GPU Adaptive**: Automatically adjusts precision and checkpointing based on hardware

## Running the Scripts

### Data Augmentation & Preprocessing
```bash
# Generate augmented dataset with splits
python -m augmentation.main --synth 1000 --seed 42

# Options:
# --synth N: Generate N synthetic examples using Faker
# --seed N: Random seed for reproducibility
# --root PATH: Project root path (auto-detected by default)
```

### Model Training
```bash
# CPU/GPU training with LoRA (auto-detects hardware)
python -m src.main --config configs/default.yaml --lora configs/lora.yaml

# Monitor training progress
tensorboard --logdir results/tb
```

### Training Configuration
- **CPU Training**: Automatically uses LoRA without quantization
- **GPU Training**: Enables qLoRA with 4-bit quantization if CUDA available
- **Adaptive Settings**: Adjusts batch size, precision, and checkpointing per hardware
- **Stability Features**: Gradient clipping, early stopping, best checkpoint saving

## Evaluation

### Model Evaluation on Test Set
```bash
# Evaluate trained model on test split
python scripts/evaluate.py --config configs/default.yaml --model_dir models/best/

# Quick single inference test
python -m src.infer --model models/best/ --text "Meeting tomorrow at 2pm with John"
```

### Evaluation Metrics
1. **JSON Validity Rate**: Percentage of outputs that parse as valid JSON
2. **Per-field F1 Score**: Micro-averaged F1 across all 8 schema fields
3. **Exact Match Accuracy**: Percentage where entire JSON matches ground truth exactly
4. **Field-level Analysis**: Individual precision/recall for each schema component

### Evaluation Output Format
```json
{
  "count": 360,
  "json_valid": 0.95,
  "per_field_f1": 0.87,
  "exact_match": 0.73
}
```

## Repository Structure

```text
IK_NER_Finetuning/
├── augmentation/                 # Data augmentation pipeline
│   ├── __init__.py
│   ├── main.py                  # CLI for augmentation and splitting
│   ├── utils.py                 # JSONL I/O, schema validation
│   ├── entity_pools.py          # Extract entities from dataset
│   ├── augmentors.py            # Entity swapping logic
│   ├── renderers.py             # JSON-to-text templates
│   └── faker_synth.py           # Synthetic data generation
├── src/                         # Training pipeline
│   ├── __init__.py
│   ├── main.py                  # Training entrypoint
│   ├── train.py                 # LoRA/QLoRA trainer
│   ├── data.py                  # Dataset loader and collation
│   ├── prompts.py               # Instruction template
│   ├── metrics.py               # Evaluation metrics
│   ├── loss.py                  # Custom loss functions
│   ├── infer.py                 # Single inference CLI
│   └── validate_json.py         # Schema validation helpers
├── configs/                     # Configuration files
│   ├── default.yaml             # Training hyperparameters
│   └── lora.yaml                # LoRA/QLoRA settings
├── scripts/                     # Evaluation utilities
│   └── evaluate.py              # Test set evaluation
├── data/
│   ├── raw/
│   │   └── event_text_mapping.jsonl
│   ├── processed/
│   │   └── augmented.jsonl      # Full augmented dataset
│   └── splits/                  # Train/eval/test splits
│       ├── train.jsonl
│       ├── eval.jsonl
│       └── test.jsonl
├── models/                      # Saved checkpoints
│   └── best/                    # Best checkpoint mirror
├── results/
│   └── tb/                      # TensorBoard logs
└── requirements.txt
```

## Configuration Files

### `configs/default.yaml`
Controls training hyperparameters, data paths, and model settings:
- **Paths**: Input/output directories for data and checkpoints
- **Model**: Base model ID, sequence lengths, precision settings
- **Training**: Learning rate, batch sizes, epochs, evaluation frequency
- **Hardware Adaptation**: Automatically adjusts based on available compute

### `configs/lora.yaml`
Defines LoRA and QLoRA parameters:
- **LoRA Configuration**: Rank, alpha, dropout, target modules
- **QLoRA Settings**: 4-bit quantization options (NF4, double quantization)
- **Automatic Selection**: QLoRA enabled only when CUDA is detected

Key configuration highlights:
```yaml
training:
  epochs: 10
  per_device_train_batch_size: 8
  lr: 2.0e-5
  early_stopping_patience: 5

lora:
  r: 8                    # Rank for adaptation
  alpha: 16               # Scaling factor
  dropout: 0.10           # Regularization
```

## Deployment & Inference

### Loading Trained Model
```python
# Load fine-tuned model for inference
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/best/")
tokenizer = AutoTokenizer.from_pretrained("models/best/")

# Single inference
result = infer("Meeting tomorrow at 3pm with the team")
```

### API Deployment
The trained model can be deployed as a REST API using FastAPI or served via Hugging Face Spaces for real-time calendar event extraction from natural language inputs.

---

**Summary**: This project successfully implements a complete pipeline for fine-tuning SmolLM-360M on calendar event extraction, featuring sophisticated data augmentation, efficient LoRA/QLoRA training, comprehensive evaluation metrics, and a modular codebase ready for production deployment.


