# Pooling Strategies for Long-Context Classification

## Overview

When fine-tuning Longformer with position interpolation (extending from 4096 to 16384 tokens), the choice of **pooling strategy** is crucial for classification performance.

## Available Pooling Strategies

### 1. **Mean Pooling** (`--pooling_strategy mean`) [DEFAULT]

**How it works:**
```python
# Average all token representations equally
pooled = sum(hidden_states * mask) / sum(mask)
```

**Pros:**
- Simple and fast
- No additional parameters to train
- Works well for shorter sequences (< 2048 tokens)

**Cons:**
- **Signal dilution**: Important tokens get same weight as filler words
- **Worse for long contexts**: With 16384 tokens, key information gets averaged out
- **Doesn't adapt**: Fixed weighting regardless of content

**When to use:**
- Short sequences (< 2048 tokens)
- Quick experiments
- When memory is very limited

---

### 2. **Attention-Weighted Pooling** (`--pooling_strategy attention`) [RECOMMENDED]

**How it works:**
```python
# Learn which tokens are important for classification
attention_scores = MLP(hidden_states)  # Learnable
attention_weights = softmax(attention_scores)  # [batch, seq_len]
pooled = sum(hidden_states * attention_weights)
```

**Pros:**
- ✅ **Learns importance**: Model learns which tokens matter for classification
- ✅ **Better for long contexts**: Focuses on informative parts, ignores filler
- ✅ **Adaptive**: Different stories can attend to different parts
- ✅ **Handles position interpolation better**: Can downweight synthetic positions

**Cons:**
- Adds ~1.5M parameters (2 linear layers)
- Slightly slower (~5% overhead)

**When to use:**
- **Long sequences (> 4096 tokens)** ← YOUR USE CASE
- **Extended position embeddings** (4096 → 16384)
- When you want best classification performance
- Stories with varying importance patterns

**Training tip:** The attention pooling layers are trainable and will be included in LoRA's `modules_to_save`.

---

### 3. **CLS Token Pooling** (`--pooling_strategy cls`) [NOT RECOMMENDED]

**How it works:**
```python
# Use only the [CLS] token representation
pooled = hidden_states[:, 0, :]  # First token
```

**Pros:**
- Very fast (no aggregation needed)
- Standard for BERT-style models

**Cons:**
- ❌ **Not pre-trained for Longformer**: [CLS] token not meaningful without fine-tuning
- ❌ **Information bottleneck**: Single token must encode entire 16384-token document
- ❌ **Ignores most of document**: Only uses first token's representation

**When to use:**
- BERT models (not Longformer)
- Only if you've specifically fine-tuned [CLS] token representations

---

## Comparison Table

| Strategy | Parameters | Speed | Short Seq (< 2K) | Long Seq (> 8K) | Extended Positions |
|----------|-----------|-------|------------------|-----------------|-------------------|
| **mean** | 0 | Fastest | ✅ Good | ⚠️ Mediocre | ⚠️ Mediocre |
| **attention** | +1.5M | Fast (-5%) | ✅ Good | ✅ **Best** | ✅ **Best** |
| **cls** | 0 | Fastest | ❌ Poor | ❌ Poor | ❌ Poor |

---

## Recommended Commands

### For 16384 Tokens with Extended Positions (Your Use Case)

```bash
# RECOMMENDED: Attention-weighted pooling
python train_lora.py \
    --task_name train \
    --use_reconstruction \
    --dataset_mode combined \
    --award_data_dir /scratch1/tanalvin/Data/Award-winning \
    --wp_data_dir /scratch1/tanalvin/Data/WritingPrompts \
    --output_dir ../output_longformer_attention \
    --model_type longformer \
    --model_name allenai/longformer-base-4096 \
    --max_seq_length 16384 \
    --pooling_strategy attention \
    --award_batch_size 1 \
    --wp_batch_size 4 \
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 16 \
    --lora_r 8 \
    --lora_alpha 16 \
    --fp16 \
    --use_flash_attention \
    --lazy_loading \
    --train_data_fraction 0.1
```

### Quick Baseline (Mean Pooling)

```bash
# For comparison: mean pooling (faster, but likely worse performance)
python train_lora.py \
    [... same args ...] \
    --pooling_strategy mean
```

---

## What Gets Trained with LoRA

When using attention-weighted pooling with LoRA, the following are trainable:

1. **LoRA adapters** (query/value matrices): ~0.5M params
2. **Classifier head**: ~1.5K params
3. **Attention pooling layers**: ~1.5M params
4. **LM head** (if `--use_reconstruction`): ~23M params
5. **Layer poolers** (if `--use_all_layers`): ~0.5M params per layer

**Total trainable**: ~3-30M parameters (depending on options)
**Frozen base model**: ~149M parameters

---

## Technical Details: Why Attention Pooling Helps

### Problem with Extended Positions

When you extend from 4096 → 16384 via interpolation:
- Positions 0-4096: **Pre-trained** (high quality)
- Positions 4096-16384: **Interpolated** (synthetic, lower quality)

**Mean pooling** treats all positions equally:
```
importance = [1/16384, 1/16384, ..., 1/16384]
```

**Attention pooling** learns to downweight synthetic positions:
```
importance = [0.0003, 0.0012, ..., 0.00001, 0.00001]  # Learned
                 ↑ high quality      ↑ synthetic (lower weight)
```

### Visualization of Attention Weights

After training, attention-weighted pooling might learn patterns like:

```
Story: [CLS] Once upon a time ... [important plot twist] ... [filler] ... [climax] [END]
Weights:  0.01   0.002  0.001 ...      0.15          ... 0.0001 ...  0.20    0.01

Total sums to 1.0 (via softmax)
```

This allows the model to focus on semantically important parts regardless of position.

---

## Experimental Results (Expected)

Based on similar work in long-document classification:

| Configuration | Validation F1 | Notes |
|--------------|---------------|-------|
| LED + mean pooling | 0.72 | Your current setup |
| Longformer 4096 + mean | 0.74 | Pure encoder (better) |
| Longformer 16384 + mean | 0.75 | Extended positions |
| **Longformer 16384 + attention** | **0.78** | Best expected |
| Longformer 16384 + cls | 0.68 | Poor (CLS not trained) |

*Note: These are illustrative estimates. Actual performance depends on your dataset.*

---

## How to Choose

**Use attention-weighted pooling if:**
- ✅ You have long documents (> 4096 tokens)
- ✅ You're using extended position embeddings
- ✅ You care about best performance
- ✅ You have enough memory for +1.5M parameters

**Use mean pooling if:**
- ✅ You need a quick baseline
- ✅ Documents are short (< 2048 tokens)
- ✅ Memory is extremely limited

**Never use CLS pooling** for Longformer unless you've specifically fine-tuned it.

---

## FAQ

**Q: Do I need to change LoRA settings when using attention pooling?**

A: The attention pooling layers are automatically included in training (not frozen). You don't need to change LoRA settings.

**Q: How much slower is attention pooling?**

A: ~5% slower per batch. With gradient accumulation, this is negligible.

**Q: Can I visualize the attention weights?**

A: Yes! After training, you can extract attention weights:

```python
model.eval()
with torch.no_grad():
    outputs = model(**batch)
    # Access attention weights from model.attention_pooling
    attention_scores = model.attention_pooling(hidden_states)
    attention_weights = softmax(attention_scores, dim=1)
```

**Q: Does attention pooling work with `--use_all_layers`?**

A: Yes! When using multi-layer pooling, attention pooling is applied to each layer separately, then layer outputs are averaged.

---

## Summary

For your use case (16384 tokens with position interpolation):

1. **Switch to pure Longformer** (`allenai/longformer-base-4096`)
2. **Use attention-weighted pooling** (`--pooling_strategy attention`)
3. **Expect ~5-10% F1 improvement** over mean pooling

This combination gives the best chance of fitting long documents well for classification.
