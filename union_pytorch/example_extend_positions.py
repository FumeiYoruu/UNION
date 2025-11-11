"""
Example: Extending Longformer position embeddings to 16384 tokens

This demonstrates how to extend pure Longformer models from their native 4096
max positions to longer sequences (e.g., 16384) using position interpolation.
"""

import torch
from models import create_model

# Example 1: Automatic extension via create_model
print("=" * 80)
print("Example 1: Automatic position extension")
print("=" * 80)

model = create_model(
    model_type="longformer",
    model_name="allenai/longformer-base-4096",
    extend_position_embeddings=16384,  # Automatically extends from 4096 to 16384
    use_reconstruction=True,
)

print(f"\nModel created with max_position_embeddings: {model.config.max_position_embeddings}")


# Example 2: Manual extension after model creation
print("\n" + "=" * 80)
print("Example 2: Manual position extension")
print("=" * 80)

model2 = create_model(
    model_type="longformer",
    model_name="allenai/longformer-base-4096",
    use_reconstruction=True,
)

print(f"\nBefore extension: {model2.config.max_position_embeddings}")
model2.extend_position_embeddings(8192)  # Extend to 8192
print(f"After extension: {model2.config.max_position_embeddings}")


# Example 3: Test with actual input
print("\n" + "=" * 80)
print("Example 3: Test with long sequence")
print("=" * 80)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# Create a long dummy sequence (10000 tokens)
long_text = " ".join(["word"] * 10000)
tokens = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=10000)

model.eval()
with torch.no_grad():
    try:
        outputs = model(**tokens)
        print(f"✓ Successfully processed {tokens['input_ids'].size(1)} tokens")
        print(f"Logits shape: {outputs['logits'].shape}")
    except Exception as e:
        print(f"✗ Error: {e}")


print("\n" + "=" * 80)
print("Summary: Position Extension Methods")
print("=" * 80)
print("""
1. **Linear Interpolation** (implemented):
   - Smoothly extends position embeddings using torch.nn.functional.interpolate
   - Works well for moderate extensions (4096 → 8192 or 16384)
   - No additional training required

2. **Alternative methods** (not implemented):
   - Sinusoidal extension: Use sin/cos for new positions
   - Learned extension: Fine-tune on long sequences after extension
   - ALiBi: Replace position embeddings with attention biases

For best results with extended positions:
- Start with small extensions (4096 → 8192) before jumping to 16384
- Fine-tune on your long-document dataset after extension
- Monitor attention patterns to ensure model handles long contexts
""")
