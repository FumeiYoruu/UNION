# Longformer 16K Quick Reference Card

## ✅ What Changed

**Default Longformer model upgraded from 4k → 16k tokens**

```
Before: allenai/longformer-base-4096  (max 4,096 tokens)
After:  allenai/longformer-base-16384 (max 16,384 tokens)
```

## 🚀 Quick Start

### Basic Usage (2k tokens - Recommended)
```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_wp \
    --dataset_mode wp \
    --max_seq_length 2048 \
    --train_batch_size 4
```

### Medium Stories (4k tokens)
```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_4k \
    --dataset_mode wp \
    --max_seq_length 4096 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4
```

### Very Long Stories (8k tokens)
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

### Maximum Context (16k tokens) ⚠️
```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_16k \
    --dataset_mode wp \
    --max_seq_length 16384 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --fp16
```
**Requires: 40GB+ GPU (A100)**

## 📊 Sequence Length Guide

| Length | Words (~) | Use Case | GPU Memory | Batch Size |
|--------|-----------|----------|------------|------------|
| 512 | ~380 | Short stories | 8GB | 8 |
| 1024 | ~760 | ROCStories | 12GB | 4-8 |
| 2048 | ~1500 | Most WritingPrompts | 16GB | 4 |
| 4096 | ~3000 | Long WritingPrompts | 24GB | 2 |
| 8192 | ~6000 | Very long stories | 32GB | 1-2 |
| 16384 | ~12000 | Maximum length | 40GB+ | 1 |

## 💡 Best Practices

### Do's ✅
- **Start with 2048** tokens for most use cases
- **Use gradient accumulation** for larger effective batch sizes
- **Enable FP16** (`--fp16`) to save memory
- **Monitor truncation** - check if stories are being cut off

### Don'ts ❌
- **Don't use 16k** unless absolutely necessary
- **Don't use large batch sizes** with long sequences
- **Don't ignore GPU memory** - start small and increase
- **Don't forget** gradient accumulation for small batch sizes

## 🎯 Choose Your Length

```
Story Length?
│
├─ 0-500 words  → 512-1024 tokens  (BERT or Longformer)
├─ 500-1000     → 1024-2048 tokens (Longformer recommended)
├─ 1000-2000    → 2048-4096 tokens (Longformer)
├─ 2000-4000    → 4096-8192 tokens (Longformer + A100)
└─ 4000+        → 8192-16384 tokens (Longformer + A100 80GB)
```

## 🔧 Memory Optimization Tricks

### Out of Memory? Try these in order:

1. **Reduce batch size**
   ```bash
   --train_batch_size 2  # or even 1
   ```

2. **Add gradient accumulation**
   ```bash
   --train_batch_size 1 \
   --gradient_accumulation_steps 16  # effective batch = 16
   ```

3. **Enable mixed precision**
   ```bash
   --fp16
   ```

4. **Reduce sequence length**
   ```bash
   --max_seq_length 2048  # instead of 4096
   ```

5. **Disable multi-layer pooling** (if enabled)
   ```bash
   # Remove --use_all_layers flag
   ```

## 🖥️ GPU Quick Guide

| Your GPU | Max Comfortable Length | Recommended Config |
|----------|------------------------|-------------------|
| RTX 3060 (12GB) | 2048 | batch=2, grad_acc=8 |
| RTX 3080 (10GB) | 2048 | batch=2, grad_acc=8 |
| RTX 3090 (24GB) | 4096 | batch=4, grad_acc=4 |
| RTX 4090 (24GB) | 4096-8192 | batch=2, grad_acc=8 |
| A100 40GB | 8192 | batch=2, grad_acc=8 |
| A100 80GB | 16384 | batch=1, grad_acc=16 |

## 📝 Common Configurations

### Conservative (Works on most GPUs)
```bash
--max_seq_length 1024 \
--train_batch_size 4 \
--gradient_accumulation_steps 4
```

### Balanced (24GB GPU)
```bash
--max_seq_length 2048 \
--train_batch_size 4 \
--gradient_accumulation_steps 4 \
--fp16
```

### Aggressive (40GB+ GPU)
```bash
--max_seq_length 4096 \
--train_batch_size 2 \
--gradient_accumulation_steps 8 \
--fp16
```

### Maximum (80GB GPU only)
```bash
--max_seq_length 16384 \
--train_batch_size 1 \
--gradient_accumulation_steps 16 \
--fp16
```

## ⚡ Performance Tips

1. **Speed vs Context Trade-off**
   - 2048 tokens: 1x speed (baseline)
   - 4096 tokens: ~0.5x speed
   - 8192 tokens: ~0.25x speed
   - 16384 tokens: ~0.12x speed

2. **Memory-Efficient Training**
   ```bash
   # Maximize context within memory budget
   --max_seq_length 4096 \
   --train_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --fp16
   ```

3. **Speed-Optimized Training**
   ```bash
   # Faster training with good context
   --max_seq_length 2048 \
   --train_batch_size 8 \
   --fp16
   ```

## 🔍 Check Your Model

Verify you're using the 16k model:

```python
from models import create_model

model = create_model(model_type="longformer")
print(f"Model: {model.model_name}")
print(f"Max tokens: {model.config.max_position_embeddings}")

# Should print:
# Model: allenai/longformer-base-16384
# Max tokens: 16384
```

## 📚 More Information

- **Full docs**: See `LONGFORMER_16K_UPDATE.md`
- **General usage**: See `README.md`
- **Conditional reconstruction**: See `CONDITIONAL_RECONSTRUCTION.md`

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce `--train_batch_size` and/or `--max_seq_length` |
| Too slow | Reduce `--max_seq_length` to 2048 or 1024 |
| Stories truncated | Increase `--max_seq_length` (if GPU allows) |
| Low accuracy | Try `--use_all_layers` or increase sequence length |

---

**TL;DR**: Use `--max_seq_length 2048` for most cases. Only go higher if needed.
