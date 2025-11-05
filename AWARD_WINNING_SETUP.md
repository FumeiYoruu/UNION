# Award-winning Dataset Integration - Summary

This document summarizes the changes made to integrate the Award-winning dataset with the union_pytorch codebase.

## Overview

The Award-winning dataset has been successfully integrated into the UNION PyTorch training pipeline. All components support reconstruction task, with all negative samples labeled as having references (label=1 for reconstruction).

## Changes Made

### 1. Data Preparation Script

**File**: `Data/prepare_award_winning.py`

A new Python script that reformats the Award-winning dataset structure:

**Input Structure**:
- Positive samples: `Data/Award-winning/{story_name}/story_text.txt`
- Negative samples: `Data/Award-winning/train_data/negatives/{story_name}_negative.txt`

**Output Structure**:
- `train_human.txt`, `dev_human.txt`, `test_human.txt` (positive samples)
- `train_negative.txt`, `dev_negative.txt`, `test_negative.txt` (negative samples)
- `*_negative_ref_map.txt` (reference mapping files, all set to 1)

**Key Features**:
- Handles space-to-underscore conversion in negative filenames
- Removes story titles from text
- Converts multi-line stories to single-line format (required by dataloader)
- Configurable train/dev/test split (default: 70%/15%/15%)
- Random seed for reproducibility

**Usage**:
```bash
python Data/prepare_award_winning.py \
    --award_dir ./Data/Award-winning \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### 2. Configuration Updates

**File**: `union_pytorch/config.py`

Added "award" as a new dataset mode option:

```python
# Line 36
dataset_mode: str = "roc"  # "roc", "wp", or "award"

# Lines 89-91
parser.add_argument("--dataset_mode", type=str, default="roc",
                    choices=["roc", "wp", "award"],
                    help="Dataset mode: ROCStories, WritingPrompts, or Award-winning")
```

### 3. Dataset Loader Updates

**File**: `union_pytorch/data/dataset.py`

Updated to support "award" dataset mode:
- Documentation updated to mention "award" option (lines 63)
- "award" mode treated the same as "wp" mode (single-line format)
- Comments clarified that both WritingPrompts and Award-winning use single-line format (lines 96, 125)

The existing code structure already supported single-line formats, so no major changes were needed.

### 4. Example Scripts

**File**: `union_pytorch/run_example.sh`

Added Example 5 for training on Award-winning dataset:

```bash
python train.py \
    --task_name train \
    --model_type bert \
    --model_name bert-base-uncased \
    --data_dir ../Data/Award-winning \
    --output_dir ./output/bert_award \
    --dataset_mode award \
    --max_seq_length 512 \
    --use_reconstruction \
    --reconstruction_weight 0.1 \
    --train_batch_size 4 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --warmup_steps 100
```

### 5. Documentation

**File**: `Data/Award-winning/README.md`

Comprehensive documentation covering:
- Dataset structure (before and after preparation)
- Data preparation instructions
- Training examples (basic, with reconstruction, with Longformer)
- Key differences from ROCStories and WritingPrompts
- Reconstruction task details
- Dataset statistics

### 6. Test Script

**File**: `test_award_dataloader.py`

A test script to verify the dataloader works correctly with Award-winning dataset:
- Tests loading train/dev/test splits
- Verifies reconstruction task is enabled
- Checks data shapes and keys

## Dataset Statistics

After preparation:
- **Total stories**: 47 pairs (3 stories missing negative samples)
- **Train**: 32 stories → 64 samples (32 human + 32 negative)
- **Dev**: 7 stories → 14 samples (7 human + 7 negative)
- **Test**: 8 stories → 16 samples (8 human + 8 negative)

## Reconstruction Task

**All negative samples use reconstruction**:
- Reference mapping files (`*_negative_ref_map.txt`) contain all 1s
- Every negative sample is paired with its corresponding human story
- Reconstruction helps the model learn coherence by predicting masked portions of the reference story

## Quick Start

1. **Prepare the data**:
```bash
python Data/prepare_award_winning.py
```

2. **Train with BERT**:
```bash
cd union_pytorch
python train.py \
    --task_name train \
    --model_type bert \
    --model_name bert-base-uncased \
    --data_dir ../Data/Award-winning \
    --output_dir ./output/bert_award \
    --dataset_mode award \
    --max_seq_length 512 \
    --use_reconstruction \
    --reconstruction_weight 0.1 \
    --train_batch_size 4 \
    --num_train_epochs 5
```

3. **Train with Longformer (recommended for longer context)**:
```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --model_name allenai/longformer-base-16384 \
    --data_dir ../Data/Award-winning \
    --output_dir ./output/longformer_award \
    --dataset_mode award \
    --max_seq_length 2048 \
    --use_reconstruction \
    --reconstruction_weight 0.1 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4
```

## Key Design Decisions

1. **Single-line format**: Stories are converted to single-line format (internal newlines replaced with spaces) to match the dataloader's expectation of one story per line.

2. **Dataset mode**: Added explicit "award" mode for clarity, though it functions identically to "wp" mode.

3. **Reconstruction for all**: Since all negative samples have references, the `*_negative_ref_map.txt` files contain all 1s, ensuring reconstruction is used for all negative samples.

4. **Title removal**: Story titles are automatically detected and removed if they appear as a short first line without ending punctuation.

5. **Underscore handling**: The script handles the space-to-underscore conversion between directory names and negative filenames.

## File Changes Summary

```
New files:
  - Data/prepare_award_winning.py
  - Data/Award-winning/README.md
  - Data/Award-winning/train_data/train_human.txt
  - Data/Award-winning/train_data/train_negative.txt
  - Data/Award-winning/train_data/train_negative_ref_map.txt
  - Data/Award-winning/train_data/dev_human.txt
  - Data/Award-winning/train_data/dev_negative.txt
  - Data/Award-winning/train_data/dev_negative_ref_map.txt
  - Data/Award-winning/train_data/test_human.txt
  - Data/Award-winning/train_data/test_negative.txt
  - Data/Award-winning/train_data/test_negative_ref_map.txt
  - test_award_dataloader.py

Modified files:
  - union_pytorch/config.py (added "award" dataset mode)
  - union_pytorch/data/dataset.py (updated documentation)
  - union_pytorch/run_example.sh (added Example 5)
```

## Testing

Run the test script to verify everything works:

```bash
python test_award_dataloader.py
```

Expected output:
```
Testing Award-winning dataset loading...
Loaded 64 train examples
✓ Successfully loaded training data: 64 samples
✓ Sample keys: dict_keys([...])
✓ Input shape: torch.Size([512])
✓ Reference input shape: torch.Size([512])
✓ Reconstruction task enabled
Loaded 14 dev examples
✓ Successfully loaded dev data: 14 samples
Loaded 16 test examples
✓ Successfully loaded test data: 16 samples

==================================================
All tests passed! Award-winning dataset is ready for training.
==================================================
```

## Notes

- Award-winning stories are significantly longer (3000-5000 words) than ROCStories (50-100 words)
- Consider using Longformer for better handling of long contexts
- Use smaller batch sizes (2-4) due to story length
- All training should use `--use_reconstruction` flag
- The dataset is small (47 stories), so overfitting is a concern - monitor validation loss carefully

## Troubleshooting

1. **"Missing negative story" warnings**: Normal - 3 stories don't have negative samples
2. **Out of memory errors**: Reduce `train_batch_size` or `max_seq_length`
3. **Stories split incorrectly**: Ensure you ran the updated `prepare_award_winning.py` that converts stories to single-line format
