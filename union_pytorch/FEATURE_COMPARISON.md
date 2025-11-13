# Feature Comparison: Training Scripts

This document compares the features available in each training script.

## Feature Matrix

| Feature | train_lora.py | train_lora_simple.py | train_longformer_seqcls.py |
|---------|---------------|---------------------|---------------------------|
| **Model Architecture** | | | |
| Custom UnionClassifier | ✓ | ✓ | ✗ |
| LongformerForSequenceClassification | ✗ | ✗ | ✓ |
| Reconstruction Task | ✓ | ✗ | ✗ |
| Multi-layer Pooling | ✓ | ✗ | ✗ |
| CLS Pooling | ✓ (auto) | ✓ (auto) | ✓ (built-in) |
| Mean Pooling | ✓ | ✓ | ✗ |
| Attention Pooling | ✓ | ✓ | ✗ |
| **Data Features** | | | |
| Single Dataset | ✓ | ✓ | ✓ |
| Combined Datasets | ✓ | ✗ | ✓ |
| Per-dataset Batch Sizes | ✓ | ✗ | ✓ |
| Data Fraction | ✓ | ✓ | ✓ |
| Lazy Loading | ✓ | ✓ | ✓ |
| Dynamic Padding | ✓ | ✓ | ✓ |
| Bucket Padding | ✓ | ✓ | ✓ |
| **Training Features** | | | |
| LoRA Fine-tuning | ✓ | ✓ | ✓ |
| FP16 Mixed Precision | ✓ | ✓ | ✓ |
| Flash Attention | ✓ | ✓ | ✓ |
| Model Compilation | ✓ | ✓ | ✓ |
| Gradient Accumulation | ✓ | ✓ | ✓ |
| Multi-GPU (DataParallel) | ✓ | ✓ | ✓ |
| **Checkpointing** | | | |
| Save Checkpoints | ✓ | ✓ | ✓ |
| Resume from Checkpoint | ✓ | ✓ | ✓ |
| Mid-epoch Resume | ✓ | ✓ | ✓ |
| Batch-level Resume | ✓ | ✓ | ✓ |
| Best Model Tracking | ✓ | ✓ | ✓ |
| Checkpoint Cleanup | ✓ | ✓ | ✓ |
| **Logging** | | | |
| TensorBoard | ✓ | ✓ | ✓ |
| Logging Steps | ✓ | ✓ | ✓ |
| Save Steps | ✓ | ✓ | ✓ |
| Eval Steps | ✓ | ✓ | ✓ |
| Performance Metrics | ✓ | ✓ | ✓ |
| **Reproducibility** | | | |
| Seeded Sampling | ✓ | ✓ | ✓ |
| Deterministic Training | ✓ | ✓ | ✓ |
| **Code Quality** | | | |
| Lines of Code | ~1200 | ~800 | ~700 |
| Complexity | High | Medium | Low |

## Detailed Feature Descriptions

### 1. Batch-level Resume

**All scripts support mid-epoch checkpoint resuming:**

```python
# Saves checkpoint with batch_step
save_checkpoint(
    model, optimizer, scheduler,
    epoch=2,
    global_step=1500,
    batch_step=350,  # Resume from batch 350 in epoch 2
)

# Resume skips already-processed batches
for step, batch in enumerate(dataloader):
    if start_step > 0 and step < start_step:
        continue  # Skip already processed batches
    # Process batch...
```

**Location in each script:**
- `train_lora.py`: Lines 375-379, 486
- `train_lora_simple.py`: Similar implementation
- `train_longformer_seqcls.py`: Lines 265-266, 336

### 2. Logging Steps

**All scripts support configurable logging intervals:**

```python
--logging_steps 100  # Log every 100 steps
```

Logs:
- Training loss
- Learning rate
- Performance metrics (data loading time, forward/backward time)

**Location in each script:**
- `train_lora.py`: Lines 450-464
- `train_lora_simple.py`: Similar
- `train_longformer_seqcls.py`: Lines 313-319

### 3. Save Steps

**All scripts support periodic checkpointing:**

```python
--save_steps 500              # Save every 500 steps
--keep_last_n_checkpoints 3   # Keep only last 3 checkpoints
```

Creates checkpoints:
- `checkpoint-{step}`: Regular checkpoints (auto-cleanup)
- `best-{step}`: Best F1 checkpoints (kept forever)
- `epoch-{step}`: End-of-epoch checkpoints (keeps last 2)

**Location in each script:**
- `train_lora.py`: Lines 475-487
- `train_lora_simple.py`: Similar
- `train_longformer_seqcls.py`: Lines 325-336

### 4. Eval Steps

**All scripts support periodic evaluation:**

```python
--eval_steps 1000  # Evaluate every 1000 steps
```

During training:
- Evaluates on 1% of validation data (fast)
- Saves best model when F1 improves
- Logs metrics to TensorBoard

End of epoch:
- Evaluates on 10% of validation data (more thorough)

**Location in each script:**
- `train_lora.py`: Lines 490-527
- `train_lora_simple.py`: Similar
- `train_longformer_seqcls.py`: Lines 340-374

### 5. Data Fraction

**All scripts support training on partial data:**

```python
--train_data_fraction 0.1  # Use only 10% of training data
```

Useful for:
- Quick experiments
- Debugging
- Limited compute resources

**Only applied to training set, not validation.**

**Location in each script:**
- `train_lora.py`: Lines 600, 633, 659
- `train_lora_simple.py`: Lines 460-467
- `train_longformer_seqcls.py`: Lines 431-432, 463

## Example Usage

### Resume from Mid-epoch Checkpoint

All three scripts support this:

```bash
# Training was interrupted at step 1500 (batch 350 in epoch 2)
python train_longformer_seqcls.py \
    --resume_from_checkpoint ./output/checkpoint-1500 \
    --task_name train \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output \
    ... (same args as before)

# Output:
# Resuming from MID of epoch 2
# Global step: 1500, will skip first 350 batches
# (Training continues from batch 351)
```

### Train with Logging and Checkpointing

All three scripts support this:

```bash
python train_longformer_seqcls.py \
    --task_name train \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output \
    --logging_steps 100 \      # Log every 100 steps
    --save_steps 500 \         # Save every 500 steps
    --eval_steps 1000 \        # Evaluate every 1000 steps
    --keep_last_n_checkpoints 3  # Keep last 3 checkpoints
```

### Train on Partial Data

All three scripts support this:

```bash
python train_longformer_seqcls.py \
    --task_name train \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output \
    --train_data_fraction 0.1  # Use only 10% of data
```

## Choosing the Right Script

### Use `train_longformer_seqcls.py` when:
- ✓ You want the **official LongformerForSequenceClassification** architecture
- ✓ You want the **simplest, cleanest code**
- ✓ You're doing **standard sequence classification**
- ✓ You don't need reconstruction or custom pooling

### Use `train_lora_simple.py` when:
- ✓ You need **flexible pooling strategies** (mean/attention/cls)
- ✓ You want the custom `UnionClassifier` architecture
- ✓ You want cleaner code than full `train_lora.py`
- ✓ You don't need reconstruction or multi-layer pooling

### Use `train_lora.py` when:
- ✓ You need **reconstruction task** (auxiliary masked LM)
- ✓ You need **multi-layer pooling**
- ✓ You need **combined dataset training** (Award-winning + WritingPrompts)
- ✓ You need **per-dataset batch sizes**
- ✓ You're doing research experiments

## Summary

**All three scripts have complete feature parity for:**
- ✓ Batch-level checkpoint resuming
- ✓ Logging steps
- ✓ Save steps
- ✓ Eval steps
- ✓ Data fraction
- ✓ Best model tracking
- ✓ TensorBoard integration
- ✓ Reproducible training

**The main differences are:**
- **Model architecture** (official vs custom)
- **Advanced features** (reconstruction, multi-layer pooling, combined datasets)
- **Code complexity**

Choose based on your needs: standard classification → `train_longformer_seqcls.py`, custom features → `train_lora.py`.
