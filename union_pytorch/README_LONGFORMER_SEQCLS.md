# Direct LongformerForSequenceClassification Fine-tuning

This script directly fine-tunes the official Hugging Face `LongformerForSequenceClassification` model with LoRA, rather than using the custom `UnionClassifier`.

## Why Use This Script?

**Advantages of `train_longformer_seqcls.py`:**
- ✅ Uses the **official Longformer sequence classification architecture**
- ✅ Built-in **CLS token pooling** as designed by the Longformer authors
- ✅ Simpler code with fewer custom components
- ✅ Standard Hugging Face model interface
- ✅ Directly compatible with Hugging Face ecosystem (Trainer, Inference API, etc.)

**Use the custom `UnionClassifier` scripts when:**
- ❌ You need reconstruction task (auxiliary masked LM objective)
- ❌ You need multi-layer pooling
- ❌ You need flexible pooling strategies (mean/attention pooling)
- ❌ You need combined dataset training

## Model Architecture

This script uses `LongformerForSequenceClassification` which consists of:

```
Input Tokens
    ↓
LongformerModel (encoder)
    ↓
LongformerClassificationHead:
  - Extract CLS token (position 0)
  - Dropout
  - Dense layer (768 → 768)
  - Tanh activation
  - Dropout
  - Output projection (768 → num_labels)
    ↓
Classification Logits
```

This is the **standard architecture** used in the Longformer paper and recommended by Allen AI.

## Installation

```bash
pip install peft transformers

# Optional: For performance
pip install xformers
```

## Quick Start

### Basic Training

```bash
python train_longformer_seqcls.py \
    --task_name train \
    --model_name allenai/longformer-base-4096 \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_seqcls_wp \
    --dataset_mode wp \
    --max_seq_length 4096 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --fp16 \
    --use_flash_attention
```

### With All Optimizations (Recommended)

```bash
python train_longformer_seqcls.py \
    --task_name train \
    --model_name allenai/longformer-base-4096 \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_seqcls_wp \
    --dataset_mode wp \
    --max_seq_length 4096 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --fp16 \
    --use_flash_attention \
    --compile_model \
    --save_steps 500 \
    --eval_steps 1000 \
    --seed 42
```

## Arguments

### Model Arguments

- `--model_name`: Pretrained Longformer model (default: `allenai/longformer-base-4096`)
  - `allenai/longformer-base-4096`: 4096 max tokens
  - `allenai/longformer-large-4096`: Larger model, better performance
- `--max_seq_length`: Maximum sequence length (default: 4096)
- `--num_labels`: Number of classification labels (default: 2 for binary)

### LoRA Arguments

- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--lora_dropout`: LoRA dropout (default: 0.1)
- `--lora_target_modules`: Target modules (default: `["query", "value"]`)

### Data Arguments

- `--data_dir`: Training data directory (required)
- `--output_dir`: Output directory (required)
- `--dataset_mode`: Dataset type (`roc`, `wp`, `award`)
- `--train_data_fraction`: Fraction of training data to use (default: 1.0)
- `--lazy_loading`: Tokenize on-the-fly
- `--padding_strategy`: Padding strategy (`dynamic`, `bucket`, `fixed`)

### Training Arguments

- `--train_batch_size`: Batch size (default: 2)
- `--eval_batch_size`: Eval batch size (default: 4)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--num_train_epochs`: Number of epochs (default: 3)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--fp16`: Use mixed precision
- `--use_flash_attention`: Use efficient attention
- `--compile_model`: Compile with torch.compile()

### Checkpointing

- `--save_steps`: Save every N steps (default: 500)
- `--eval_steps`: Evaluate every N steps (default: 1000)
- `--keep_last_n_checkpoints`: Keep last N checkpoints (default: 3)
- `--resume_from_checkpoint PATH`: Resume from checkpoint

## Examples

### Train on ROCStories

```bash
python train_longformer_seqcls.py \
    --task_name train \
    --model_name allenai/longformer-base-4096 \
    --data_dir ../Data/ROCStories \
    --output_dir ./output/longformer_seqcls_roc \
    --dataset_mode roc \
    --max_seq_length 512 \
    --train_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16
```

### Train on Award-winning Literature

```bash
python train_longformer_seqcls.py \
    --task_name train \
    --model_name allenai/longformer-base-4096 \
    --data_dir ../Data/Award-winning \
    --output_dir ./output/longformer_seqcls_award \
    --dataset_mode award \
    --max_seq_length 4096 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --fp16 \
    --use_flash_attention \
    --save_steps 500
```

### Resume Training

```bash
python train_longformer_seqcls.py \
    --resume_from_checkpoint ./output/longformer_seqcls_wp/checkpoint-1500 \
    --task_name train \
    --model_name allenai/longformer-base-4096 \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_seqcls_wp \
    --dataset_mode wp \
    --max_seq_length 4096 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --fp16 \
    --use_flash_attention
```

## Comparison: Which Script to Use?

| Feature | train_longformer_seqcls.py | train_lora_simple.py | train_lora.py |
|---------|---------------------------|---------------------|---------------|
| Base Model | `LongformerForSequenceClassification` | Custom `UnionClassifier` | Custom `UnionClassifier` |
| CLS Pooling | ✓ Built-in | ✓ Auto-enabled | ✓ Auto-enabled |
| Mean Pooling | ✗ | ✓ | ✓ |
| Attention Pooling | ✗ | ✓ | ✓ |
| Reconstruction Task | ✗ | ✗ | ✓ |
| Multi-layer Pooling | ✗ | ✗ | ✓ |
| Combined Datasets | ✗ | ✗ | ✓ |
| Standard HF Model | ✓✓✓ | ✗ | ✗ |
| Simplicity | ✓✓✓ | ✓✓ | ✓ |
| **Best For** | **Standard classification** | **Custom pooling** | **Research features** |

**Recommendation:**
- **Use `train_longformer_seqcls.py`** for standard sequence classification (most common use case)
- Use `train_lora_simple.py` if you need flexible pooling strategies
- Use `train_lora.py` if you need reconstruction or multi-layer pooling

## Performance

### Recommended Settings for 4096 Tokens

```bash
--train_batch_size 2 \
--gradient_accumulation_steps 8 \
--fp16 \
--use_flash_attention
```

**Effective batch size:** 2 × 8 = 16

### GPU Requirements

| Sequence Length | Batch Size | GPU Memory | Recommended GPU |
|----------------|------------|------------|-----------------|
| 512 tokens | 8 | 8GB | RTX 2060, GTX 1080 |
| 1024 tokens | 4 | 12GB | RTX 3060 |
| 2048 tokens | 2 | 16GB | V100, RTX 3080 |
| 4096 tokens | 2 | 24GB | A100, RTX 3090, RTX 4090 |
| 8192 tokens | 1 | 40GB | A100 40GB |

### Performance Optimizations

**Essential for long sequences:**

```bash
--fp16                    # 2x speedup + 50% memory reduction
--use_flash_attention     # 3-6x faster attention
--compile_model           # Additional 20-30% speedup
```

**Requirements:**
```bash
pip install xformers  # For flash attention
```

## Merge LoRA Weights

After training, you can merge LoRA adapters into the base model:

```bash
python train_longformer_seqcls.py \
    --task_name train \
    --merge_weights \
    ... (other args)
```

This creates `merged_model/` with a standard `LongformerForSequenceClassification` model that can be used without PEFT.

## Load and Use Trained Model

### With LoRA Adapters

```python
from transformers import AutoTokenizer, LongformerForSequenceClassification
from peft import PeftModel

# Load base model
base_model = LongformerForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "./output/longformer_seqcls_wp/best-5000"
)

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# Inference
text = "Once upon a time..."
inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)
outputs = model(**inputs)
logits = outputs.logits
```

### With Merged Model

```python
from transformers import AutoTokenizer, LongformerForSequenceClassification

# Load merged model (no PEFT needed)
model = LongformerForSequenceClassification.from_pretrained(
    "./output/longformer_seqcls_wp/merged_model"
)

tokenizer = AutoTokenizer.from_pretrained(
    "./output/longformer_seqcls_wp/merged_model"
)

# Inference
text = "Once upon a time..."
inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)
outputs = model(**inputs)
logits = outputs.logits
```

## Key Differences from Custom UnionClassifier

### Architecture

**LongformerForSequenceClassification:**
```python
LongformerModel (encoder)
    ↓
LongformerClassificationHead:
  - CLS token extraction
  - Dropout → Dense → Tanh → Dropout → Output
```

**Custom UnionClassifier:**
```python
LongformerModel (encoder)
    ↓
Custom pooling (mean/attention/cls)
    ↓
Optional multi-layer pooling
    ↓
Simple Linear classifier
    ↓
Optional reconstruction head
```

### Advantages of LongformerForSequenceClassification

1. **Official architecture** - Matches the Longformer paper
2. **Better CLS token handling** - Specialized head for CLS pooling
3. **Simpler** - No custom pooling logic needed
4. **Standard** - Works with all Hugging Face tools

### When to Use Custom UnionClassifier

Use the custom model when you need:
- Reconstruction task for better training
- Multi-layer pooling for richer representations
- Flexible pooling strategies (mean/attention)
- Research experiments with custom architectures

## Troubleshooting

### ModuleNotFoundError: No module named 'peft'

```bash
pip install peft
```

### CUDA Out of Memory

```bash
# Reduce batch size
--train_batch_size 1 --gradient_accumulation_steps 16

# Or reduce sequence length
--max_seq_length 2048
```

### Slow Training

```bash
# Install xFormers
pip install xformers

# Enable all optimizations
--fp16 --use_flash_attention --compile_model
```

## Citation

```bibtex
@inproceedings{union2020,
    title={UNION: An Unreferenced Metric for Evaluating Open-ended Story Generation},
    author={Jian Guan and Minlie Huang},
    booktitle={EMNLP},
    year={2020}
}

@article{longformer2020,
    title={Longformer: The Long-Document Transformer},
    author={Beltagy, Iz and Peters, Matthew E. and Cohan, Arman},
    journal={arXiv:2004.05150},
    year={2020}
}
```

## Combined Dataset Training

**NEW**: Train on multiple datasets simultaneously with per-dataset batch sizes!

### Basic Combined Training

```bash
python train_longformer_seqcls.py \
    --task_name train \
    --model_name allenai/longformer-base-4096 \
    --output_dir ./output/longformer_seqcls_combined \
    --dataset_mode combined \
    --award_data_dir ../Data/Award-winning \
    --wp_data_dir ../Data/WritingPrompts \
    --max_seq_length 4096 \
    --train_batch_size 2 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --fp16 \
    --use_flash_attention
```

### With Per-Dataset Batch Sizes (Recommended)

```bash
python train_longformer_seqcls.py \
    --task_name train \
    --model_name allenai/longformer-base-4096 \
    --output_dir ./output/longformer_seqcls_combined \
    --dataset_mode combined \
    --award_data_dir ../Data/Award-winning \
    --wp_data_dir ../Data/WritingPrompts \
    --max_seq_length 4096 \
    --award_batch_size 2 \      # Award-winning: longer stories, smaller batch
    --wp_batch_size 8 \          # WritingPrompts: shorter stories, larger batch
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --fp16 \
    --use_flash_attention \
    --save_steps 500 \
    --eval_steps 1000 \
    --seed 42
```

### Combined Dataset Arguments

- `--dataset_mode combined`: Enable combined dataset mode
- `--award_data_dir PATH`: Path to Award-winning dataset (optional)
- `--wp_data_dir PATH`: Path to WritingPrompts dataset (optional)
- `--award_batch_size N`: Batch size for Award-winning (default: uses `--train_batch_size`)
- `--wp_batch_size N`: Batch size for WritingPrompts (default: uses `--train_batch_size`)

**Requirements:**
- Must provide at least one of `--award_data_dir` or `--wp_data_dir`
- `--data_dir` is NOT used in combined mode

### Why Use Combined Training?

**Benefits:**
- ✓ **More diverse training data** - Learn from multiple sources
- ✓ **Better generalization** - Model learns story quality from different styles
- ✓ **Per-dataset batch sizes** - Optimize VRAM usage for different sequence lengths
- ✓ **Reproducible** - Same data order across epochs and checkpoints

**Per-Dataset Batch Sizes:**
Award-winning stories are typically longer (~8k-16k tokens), so we use smaller batch sizes (2-4). WritingPrompts stories are shorter (~2k-4k tokens), so we can use larger batch sizes (8-16). This maximizes VRAM utilization and training throughput.

### How It Works

The training alternates between datasets in a round-robin fashion:
1. Batch from Award-winning (size=2)
2. Batch from WritingPrompts (size=8)
3. Batch from Award-winning (size=2)
4. Batch from WritingPrompts (size=8)
5. ...

Checkpoints track the exact batch position for perfect resuming across datasets.

