"""
Debug script for position extension issues.

Run this BEFORE training to verify position extension works correctly.
"""

import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA for better error messages

from transformers import AutoTokenizer
from models import create_model

print("=" * 80)
print("Testing Position Extension with Pure Longformer")
print("=" * 80)

# Create model
print("\n1. Creating base Longformer model...")
model = create_model(
    model_type="longformer",
    model_name="allenai/longformer-base-4096",
    use_reconstruction=False,  # Disable for simpler testing
    pooling_strategy="mean",  # Start with simple mean pooling
)

print(f"\nInitial max_position_embeddings: {model.config.max_position_embeddings}")
print(f"Position embedding shape: {model.encoder.embeddings.position_embeddings.weight.shape}")

# Extend positions
print("\n2. Extending position embeddings to 16384...")
model.extend_position_embeddings(16384)

print(f"\nAfter extension max_position_embeddings: {model.config.max_position_embeddings}")
print(f"Position embedding shape: {model.encoder.embeddings.position_embeddings.weight.shape}")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n3. Moving model to device: {device}")
model = model.to(device)
model.eval()

# Create tokenizer
print("\n4. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# Test with different sequence lengths
test_lengths = [512, 2048, 4096, 8192, 12288, 16384]

for test_len in test_lengths:
    print(f"\n{'='*80}")
    print(f"Testing with sequence length: {test_len}")
    print('='*80)

    try:
        # Create dummy input
        dummy_text = " ".join(["word"] * test_len)
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            truncation=True,
            max_length=test_len,
            padding="max_length"
        )

        actual_length = inputs['input_ids'].size(1)
        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Actual tokens: {actual_length}")
        print(f"  Max position in input: {actual_length - 1}")
        print(f"  Model max_position_embeddings: {model.config.max_position_embeddings}")

        # Check if input length exceeds model capacity
        if actual_length > model.config.max_position_embeddings:
            print(f"  ⚠️  WARNING: Input length ({actual_length}) > max positions ({model.config.max_position_embeddings})")
            print(f"  This WILL cause CUDA error!")
            continue

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"  ✓ SUCCESS: Forward pass completed")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Logits: {outputs['logits']}")

    except RuntimeError as e:
        print(f"  ✗ FAILED with error:")
        print(f"  {type(e).__name__}: {e}")

        # Detailed diagnostics
        print(f"\n  Diagnostics:")
        print(f"  - Model max_position_embeddings: {model.config.max_position_embeddings}")
        print(f"  - Position embedding shape: {model.encoder.embeddings.position_embeddings.weight.shape}")
        print(f"  - Input length: {actual_length}")

        if hasattr(model.encoder.embeddings, 'position_ids'):
            print(f"  - position_ids shape: {model.encoder.embeddings.position_ids.shape}")
            print(f"  - position_ids max: {model.encoder.embeddings.position_ids.max().item()}")

        break

print("\n" + "=" * 80)
print("Testing Complete")
print("=" * 80)

# Test with attention pooling
print("\n\nTesting with ATTENTION pooling strategy:")
print("=" * 80)

try:
    model_attn = create_model(
        model_type="longformer",
        model_name="allenai/longformer-base-4096",
        use_reconstruction=False,
        pooling_strategy="attention",
        extend_position_embeddings=16384,
    )

    model_attn = model_attn.to(device)
    model_attn.eval()

    # Test with 8192 tokens
    dummy_text = " ".join(["word"] * 8192)
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_attn(**inputs)

    print(f"✓ Attention pooling test PASSED")
    print(f"  Logits shape: {outputs['logits'].shape}")

except Exception as e:
    print(f"✗ Attention pooling test FAILED:")
    print(f"  {type(e).__name__}: {e}")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print("""
If all tests pass:
  ✓ Position extension is working correctly
  ✓ You can proceed with training

If tests fail:
  ✗ Check the error messages above
  ✗ Most common issues:
     1. Input length > max_position_embeddings (should not happen after extension)
     2. position_ids not updated correctly
     3. Device mismatch (CPU vs CUDA)

Run this script with: CUDA_LAUNCH_BLOCKING=1 python debug_position_extension.py
""")
