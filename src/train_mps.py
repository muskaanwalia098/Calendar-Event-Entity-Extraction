"""
Apple Silicon (MPS) optimized training script for Calendar Event Extraction.
This script provides better compatibility and performance on MacBook M4 Pro.
"""

from __future__ import annotations

import os
import torch
import warnings
from pathlib import Path

# Set environment variables for optimal MPS performance
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def setup_mps_environment():
    """Configure environment for optimal MPS performance."""
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon MPS backend available")
        print(f"✅ MPS device: {torch.backends.mps.is_built()}")
        
        # Disable some warnings that are common with MPS
        warnings.filterwarnings("ignore", category=UserWarning, message=".*MPS.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*fallback.*")
        
        return True
    else:
        print("❌ MPS not available, falling back to CPU")
        return False

def optimize_for_m4_pro():
    """Apply M4 Pro specific optimizations."""
    # Set optimal thread count for M4 Pro (12 cores: 8P + 4E)
    torch.set_num_threads(8)  # Use performance cores
    
    # Enable optimized attention for Apple Silicon
    if hasattr(torch.backends, 'opt_einsum'):
        torch.backends.opt_einsum.enabled = True
    
    # Memory optimization
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # Clear any existing MPS cache

def main():
    """Main training entry point with MPS optimizations."""
    print("🚀 Starting Calendar Event Extraction Training (Apple Silicon Optimized)")
    
    # Setup environment
    setup_mps_environment()
    optimize_for_m4_pro()
    
    # Import after environment setup to avoid issues
    from src.train import train_entry
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train calendar event extractor (MPS optimized)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--lora", type=str, default="configs/lora.yaml")
    args = parser.parse_args()
    
    try:
        train_entry(args.config, args.lora)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        raise

if __name__ == "__main__":
    main()
