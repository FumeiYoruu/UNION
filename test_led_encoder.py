"""Test LED encoder to diagnose classification loss issue."""

import torch
from transformers import LEDForConditionalGeneration, AutoTokenizer

# Load LED model
print("Loading LED model...")
model_name = "allenai/led-base-16384"
led_model = LEDForConditionalGeneration.from_pretrained(model_name)
encoder = led_model.get_encoder()
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"\nLED Encoder Config:")
print(f"  Model: {model_name}")
print(f"  Hidden size: {encoder.config.d_model}")
print(f"  Num layers: {encoder.config.encoder_layers}")
print(f"  Max positions: {encoder.config.max_encoder_position_embeddings}")
print(f"  Vocab size: {encoder.config.vocab_size}")

# Test with sample text
test_stories = [
    "Once upon a time, there was a little girl who loved to read books.",
    "The cat sat on the mat and watched the birds fly by.",
    "She went to the store and bought some groceries for dinner.",
]

print("\n" + "="*80)
print("Testing LED Encoder Representations")
print("="*80)

for i, story in enumerate(test_stories):
    print(f"\nStory {i+1}: {story[:50]}...")

    # Tokenize
    inputs = tokenizer(story, return_tensors="pt", padding=True, truncation=True)

    # Get encoder outputs
    with torch.no_grad():
        outputs = encoder(**inputs)

    # Get [CLS] token (first token)
    cls_token = outputs.last_hidden_state[:, 0, :]

    # Statistics
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  CLS token shape: {cls_token.shape}")
    print(f"  CLS token mean: {cls_token.mean().item():.6f}")
    print(f"  CLS token std: {cls_token.std().item():.6f}")
    print(f"  CLS token L2 norm: {torch.norm(cls_token).item():.6f}")

# Compare different positions
print("\n" + "="*80)
print("Comparing Different Token Positions")
print("="*80)

story = test_stories[0]
inputs = tokenizer(story, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = encoder(**inputs)
    hidden_states = outputs.last_hidden_state[0]  # [seq_len, hidden_size]

seq_len = inputs['input_ids'].shape[1]
print(f"\nSequence length: {seq_len}")

for pos in [0, seq_len//2, seq_len-1]:
    token_repr = hidden_states[pos]
    print(f"\nPosition {pos} (token_id={inputs['input_ids'][0, pos].item()}):")
    print(f"  Mean: {token_repr.mean().item():.6f}")
    print(f"  Std: {token_repr.std().item():.6f}")
    print(f"  L2 norm: {torch.norm(token_repr).item():.6f}")

# Check if LED encoder has pooler
print("\n" + "="*80)
print("Checking Encoder Attributes")
print("="*80)
print(f"\nHas pooler_output: {hasattr(outputs, 'pooler_output')}")
print(f"Has pooler module: {hasattr(encoder, 'pooler')}")

# List encoder modules
print("\nEncoder modules:")
for name, module in encoder.named_children():
    print(f"  - {name}: {type(module).__name__}")

print("\n" + "="*80)
print("Recommendation")
print("="*80)
print("""
The LED encoder is designed for seq2seq tasks, not classification.
Key observations:

1. LED encoder doesn't have a pooler layer (unlike BERT)
2. The first token [CLS] is not specially pre-trained for classification
3. LED was trained to feed into a decoder, not a classifier

SOLUTIONS:

Option 1: Use encoder-only Longformer instead
  --model_name allenai/longformer-base-4096

Option 2: Add mean pooling instead of [CLS] token
  Modify union_model.py to use mean pooling over all tokens

Option 3: Add a trainable pooler layer
  Add nn.Linear(hidden_size, hidden_size) + tanh as pooler

Option 4: Increase LoRA trainable parameters
  Use higher --lora_r (16 or 32) and add more modules_to_save
""")
