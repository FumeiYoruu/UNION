# Conditional Reconstruction for Mixed Datasets

This guide explains how to use conditional reconstruction loss with datasets that have a mix of samples with and without original references.

## Overview

The UNION model supports **per-sample conditional reconstruction**, where:
- Some negative samples have corresponding original (human-written) stories → reconstruction loss is applied
- Some negative samples don't have original stories → reconstruction loss is skipped

This is useful when you have negative samples generated from different sources or with different corruption methods.

## How It Works

### 1. Model Behavior

The model automatically handles conditional reconstruction:
- If a sample has `ref_input_ids` (reference data), reconstruction loss is computed
- If `ref_input_ids` is `None`, reconstruction loss is skipped for that sample
- The classification loss is **always** computed for all samples

### 2. Data Format

You need three files for each split (train/dev/test):

```
Data/ROCStories/train_data/
├── train_negative.txt         # Your negative samples
├── train_human.txt             # Original human stories (only for samples that have refs)
└── train_negative_ref_map.txt  # NEW: Mapping file indicating which samples have refs
```

### Reference Mapping File Format

The `*_ref_map.txt` file contains one line per negative sample:
- `1` = this sample has a reference in `*_human.txt`
- `0` = this sample has no reference

**Example** (`train_negative_ref_map.txt`):
```
1
1
0
1
0
0
1
```

This means:
- Samples 0, 1, 3, 6 have references
- Samples 2, 4, 5 don't have references
- The `train_human.txt` file should contain exactly 4 stories (for samples 0, 1, 3, 6)

## Creating Reference Mapping Files

### Using the Helper Script

We provide `utils/create_ref_mapping.py` to generate mapping files:

#### Option 1: Specify indices with references
```bash
python utils/create_ref_mapping.py \
    --negative_file ./Data/ROCStories/train_data/train_negative.txt \
    --indices "0,1,3,6,10,15"
```

#### Option 2: All samples have references (backward compatible)
```bash
python utils/create_ref_mapping.py \
    --negative_file ./Data/ROCStories/train_data/train_negative.txt \
    --all
```

#### Option 3: No samples have references
```bash
python utils/create_ref_mapping.py \
    --negative_file ./Data/ROCStories/train_data/train_negative.txt \
    --none
```

#### Custom output location
```bash
python utils/create_ref_mapping.py \
    --negative_file ./Data/ROCStories/train_data/train_negative.txt \
    --output_file ./Data/ROCStories/train_data/custom_mapping.txt \
    --indices "0,5,10"
```

### Manual Creation

Create a text file with one line per negative sample:

```python
# Example: Create mapping for 100 samples where first 50 have refs
with open("train_negative_ref_map.txt", "w") as f:
    for i in range(100):
        if i < 50:
            f.write("1\n")  # Has reference
        else:
            f.write("0\n")  # No reference
```

## Training with Mixed Datasets

### 1. Prepare Your Data

```bash
# Structure your data
Data/ROCStories/train_data/
├── train_negative.txt         # 1000 negative samples
├── train_human.txt             # 300 original stories (for samples with refs)
└── train_negative_ref_map.txt  # 1000 lines (300 ones, 700 zeros)
```

### 2. Train with Reconstruction

```bash
python train.py \
    --data_dir ./Data/ROCStories \
    --output_dir ./output/union_mixed \
    --use_reconstruction \
    --reconstruction_weight 0.1 \
    --num_train_epochs 5 \
    --train_batch_size 16
```

The model will:
- Apply classification loss to all 1000 samples
- Apply reconstruction loss only to the 300 samples with references

### 3. Verify Training

Check the training logs. You should see:
```
Loaded 1000 train examples
Epoch 1/5
loss: 0.5234, cls_loss: 0.4821, rec_loss: 0.0413
```

If all samples had references, `rec_loss` would be higher. If none had references, `rec_loss` would be 0.

## Example Use Cases

### Case 1: Multiple Corruption Methods

You have negative samples from different sources:
- 500 samples from Story Cloze (have originals) → refs available
- 300 samples from GPT-2 generation (no originals) → no refs
- 200 samples from shuffling (have originals) → refs available

```bash
# Create mapping
python utils/create_ref_mapping.py \
    --negative_file ./Data/train_negative.txt \
    --indices "0,1,2,...,499,800,801,...,999"  # Samples 0-499 and 800-999
```

### Case 2: Progressive Dataset

Start with small dataset with references, gradually add samples without:

**Initial training** (all have refs):
```bash
python utils/create_ref_mapping.py --negative_file train_negative.txt --all
```

**Add more data** (update mapping):
```python
# Append 500 new samples without refs to train_negative.txt
# Update mapping file
with open("train_negative_ref_map.txt", "a") as f:
    for i in range(500):
        f.write("0\n")
```

### Case 3: Active Learning

Select high-quality negative samples for annotation:
```python
# Initially no refs
mapping = [0] * 1000

# After annotation, mark which samples now have refs
annotated_indices = [10, 25, 50, 100, 200]
for idx in annotated_indices:
    mapping[idx] = 1

with open("train_negative_ref_map.txt", "w") as f:
    for val in mapping:
        f.write(f"{val}\n")
```

## Backward Compatibility

**Without mapping file**: The system falls back to the original behavior:
- If `use_reconstruction=True` and `*_human.txt` exists, all samples get references
- If `use_reconstruction=False` or no `*_human.txt`, no samples get references

**With mapping file**: The mapping file takes precedence and enables per-sample control.

## Monitoring Reconstruction Usage

To see how many samples actually use reconstruction, you can add logging:

```python
from data import StoryDataset

dataset = StoryDataset(
    data_dir="./Data/ROCStories",
    tokenizer=tokenizer,
    mode="train",
    use_reconstruction=True,
)

# Count samples with references
num_with_refs = sum(
    1 for feat in dataset.features
    if feat.ref_input_ids is not None
)

print(f"Samples with reconstruction: {num_with_refs}/{len(dataset)}")
```

## Troubleshooting

### Warning: ref_map length doesn't match stories

**Problem**: Mapping file has different number of lines than stories in negative file.

**Solution**: Regenerate mapping file with correct number of entries:
```bash
python utils/create_ref_mapping.py --negative_file <file> --all
```

### Warning: Not enough references

**Problem**: Mapping file indicates more samples have refs than stories in human file.

**Example**:
- `train_negative_ref_map.txt` has 500 ones (500 samples should have refs)
- `train_human.txt` only has 300 stories

**Solution**:
1. Check your human file has correct number of stories
2. Update mapping file to match available references

### Reconstruction loss is 0

**Possible causes**:
1. No mapping file and no human file exists
2. All zeros in mapping file
3. `use_reconstruction=False` in training config

**Solution**: Verify files exist and are correctly formatted.

## Technical Details

### Data Flow

1. `StoryDataset._read_stories()` loads:
   - Negative samples from `*_negative.txt`
   - Reference mapping from `*_negative_ref_map.txt` (if exists)
   - Human stories from `*_human.txt` (if exists)

2. For each sample, if mapping indicates `has_ref=1`:
   - Loads corresponding human story from `*_human.txt`
   - Creates `ref_input_ids`, `ref_attention_mask`, `ref_labels`

3. `UnionClassifier.forward()` checks if `ref_input_ids is not None`:
   - If yes: computes reconstruction loss
   - If no: skips reconstruction for that sample

### Memory Considerations

Samples with references require ~3x more memory (main story + masked reference + labels).

**Example** with 1000 samples:
- 300 with refs: 300 × 3 = 900 tensors
- 700 without refs: 700 × 1 = 700 tensors
- Total: ~1600 tensors vs 3000 if all had refs

This allows training larger batches when mixing with/without references.
