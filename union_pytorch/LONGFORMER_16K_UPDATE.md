# Longformer 16K Token Update

## Summary

Updated the default Longformer model from **4096 tokens** to **16384 tokens** (16k) to support much longer story contexts.

## Changes Made

### 1. Model Default (`models/union_model.py`)

**Changed:**
```python
# Before
"longformer": "allenai/longformer-base-4096"

# After
"longformer": "allenai/longformer-base-16384"
```

**Location:** models/union_model.py:241

### 2. Configuration (`config.py`)

**Updated comments and defaults:**
```python
# Before
model_name: str = "bert-base-uncased"  # or "allenai/longformer-base-4096"
max_seq_length: int = 512  # 512 for BERT, up to 4096 for Longformer

# After
model_name: str = "bert-base-uncased"  # or "allenai/longformer-base-16384"
max_seq_length: int = 512  # 512 for BERT, up to 16384 for Longformer
```

**Location:** config.py:14-15

### 3. Documentation (`README.md`)

**Updated multiple sections:**

- **Features**: Changed "up to 4096 tokens" → "up to 16384 tokens"
- **Training examples**: Updated model name and max_seq_length values
- **Configuration options**: Updated max length documentation
- **Comparison table**: Updated max length from 4096 to 16384
- **Performance tips**: Added guidance for 16k context
- **Hardware requirements**: Added requirements for different sequence lengths
- **New Example 4**: Added example for training with full 16k context

## Usage

### Default (automatically uses 16k model)

```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_wp \
    --dataset_mode wp \
    --max_seq_length 2048
```

This will now automatically use `allenai/longformer-base-16384`.

### Explicit Model Specification

```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --model_name allenai/longformer-base-16384 \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_wp \
    --dataset_mode wp \
    --max_seq_length 4096
```

### Using Full 16k Context

```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_16k \
    --dataset_mode wp \
    --max_seq_length 16384 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16
```

**⚠️ Warning**: Full 16k context requires 40GB+ GPU memory.

## Recommended Sequence Lengths

### By Story Length

| Story Type | Recommended `max_seq_length` | GPU Memory |
|------------|------------------------------|------------|
| ROCStories (5 sentences) | 200-512 | 8GB+ |
| Short WritingPrompts | 1024-2048 | 16GB+ |
| Medium WritingPrompts | 2048-4096 | 24GB+ |
| Long WritingPrompts | 4096-8192 | 32GB+ |
| Very Long Stories | 8192-16384 | 40GB+ |

### By GPU

| GPU | Max Recommended Length | Batch Size |
|-----|------------------------|------------|
| RTX 3080 (10GB) | 2048 | 4 |
| RTX 3090 (24GB) | 4096 | 4 |
| RTX 4090 (24GB) | 4096-8192 | 4 |
| A100 40GB | 8192 | 2-4 |
| A100 80GB | 16384 | 2-4 |

## Performance Considerations

### Memory Usage

Longformer memory usage grows roughly linearly with sequence length:

| Sequence Length | Relative Memory | Example GPU |
|-----------------|-----------------|-------------|
| 512 | 1x | 8GB |
| 1024 | 2x | 12GB |
| 2048 | 4x | 16GB |
| 4096 | 8x | 24GB |
| 8192 | 16x | 40GB |
| 16384 | 32x | 80GB |

### Speed Considerations

- **16k context** is ~4x slower than **4k context**
- Use smaller sequences (2048-4096) for most applications
- Reserve 16k for stories that genuinely need it

### Optimization Tips

1. **Start with smaller sequences**:
   ```bash
   --max_seq_length 2048  # Good starting point
   ```

2. **Use gradient accumulation for larger effective batch sizes**:
   ```bash
   --train_batch_size 2 \
   --gradient_accumulation_steps 8  # Effective batch size = 16
   ```

3. **Enable mixed precision** (saves ~30-40% memory):
   ```bash
   --fp16
   ```

4. **Reduce batch size for longer sequences**:
   - 2048 tokens: batch_size=4-8
   - 4096 tokens: batch_size=2-4
   - 8192 tokens: batch_size=1-2
   - 16384 tokens: batch_size=1

## Backward Compatibility

### Old Code Still Works

If you were explicitly specifying the 4k model:
```bash
--model_name allenai/longformer-base-4096
```

This will continue to work! The change only affects the **default** when you don't specify `--model_name`.

### Migration

To use the new 16k model, either:

**Option 1: Use default** (recommended)
```bash
python train.py \
    --model_type longformer \
    # ... other args, no --model_name needed
```

**Option 2: Explicit specification**
```bash
python train.py \
    --model_type longformer \
    --model_name allenai/longformer-base-16384 \
    # ... other args
```

## When to Use 16k Context

### Use 16k (or high max_seq_length) when:
✅ Stories are genuinely very long (>2000 words)
✅ You have sufficient GPU memory (40GB+)
✅ You need to process entire long documents
✅ Accuracy on long stories is critical

### Use smaller context (2048-4096) when:
✅ Most stories fit in shorter sequences
✅ GPU memory is limited (<24GB)
✅ Training speed is important
✅ Stories are <1000 words

### Performance/Accuracy Trade-off

For most WritingPrompts and ROCStories:
- **2048 tokens** captures most content
- **4096 tokens** captures nearly all content
- **16384 tokens** is overkill for average stories

Only use 16k if your dataset has many stories that genuinely exceed 4k tokens.

## Testing

To verify the 16k model is loaded:

```python
from transformers import AutoTokenizer
from models import create_model

# This should load the 16k model by default
model = create_model(model_type="longformer")

print(f"Model name: {model.model_name}")
print(f"Max position embeddings: {model.config.max_position_embeddings}")

# Should output:
# Model name: allenai/longformer-base-16384
# Max position embeddings: 16384
```

## Example Training Runs

### Recommended: 2048 tokens (balanced)
```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_2k \
    --dataset_mode wp \
    --max_seq_length 2048 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 4
```

### For longer stories: 4096 tokens
```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_4k \
    --dataset_mode wp \
    --max_seq_length 4096 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8
```

### For very long stories: 8192 tokens
```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_8k \
    --dataset_mode wp \
    --max_seq_length 8192 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --fp16
```

## Files Modified

1. **models/union_model.py** (line 241)
   - Changed default Longformer model
   - Updated docstring

2. **config.py** (lines 14-15)
   - Updated comments and max_seq_length guidance

3. **README.md**
   - Updated features section
   - Updated all training examples
   - Updated configuration options
   - Updated comparison table
   - Updated performance tips
   - Updated hardware requirements
   - Added new example for 16k context

## Questions?

See `README.md` for complete documentation and examples.
