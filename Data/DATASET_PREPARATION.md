# Dataset Preparation Guide

This guide explains how to download and prepare the WritingPrompts dataset and combine it with the Award-winning dataset for UNION training.

## Quick Start

### Option 1: Automated Pipeline (Recommended)

Use the end-to-end preparation script:

```bash
# Prepare WritingPrompts only
python prepare_full_dataset.py --dataset wp

# Prepare Award-winning only
python prepare_full_dataset.py --dataset award

# Prepare combined dataset (WritingPrompts + Award-winning)
python prepare_full_dataset.py --dataset combined
```

### Option 2: Step-by-Step

#### Step 1: Download WritingPrompts

```bash
# Download from Hugging Face (recommended)
python download_writing_prompts.py --source huggingface

# Or get manual download instructions
python download_writing_prompts.py --source manual
```

#### Step 2: Generate Vocabulary and Negatives

```bash
# Generate vocabulary
python get_vocab.py wp

# Generate negative samples
python gen_train_data.py wp
```

#### Step 3: (Optional) Combine with Award-winning Dataset

```bash
# First prepare Award-winning dataset
python prepare_award_winning.py --award_dir ./Award-winning

# Then combine both datasets
python download_writing_prompts.py --combine \
    --wp_dir ./WritingPrompts \
    --award_dir ./Award-winning \
    --combined_output ./Combined
```

## Dataset Structure

### WritingPrompts Dataset

After preparation, the WritingPrompts directory will have:

```
WritingPrompts/
├── ini_data/
│   ├── train.wp_source          # Training prompts
│   ├── train.wp_target          # Training stories
│   ├── dev.wp_source            # Dev prompts
│   ├── dev.wp_target            # Dev stories
│   ├── test.wp_source           # Test prompts
│   ├── test.wp_target           # Test stories
│   └── entity_vocab.txt         # Generated vocabulary
└── train_data/
    ├── train_human.txt          # Human-written stories
    ├── train_negative.txt       # Negative samples
    ├── dev_human.txt
    ├── dev_negative.txt
    ├── test_human.txt
    └── test_negative.txt
```

### Award-winning Dataset

After preparation:

```
Award-winning/
└── train_data/
    ├── award_human.txt          # All award-winning stories
    ├── train_human.txt          # Training split
    ├── train_negative.txt
    ├── train_negative_ref_map.txt
    ├── dev_human.txt
    ├── dev_negative.txt
    ├── dev_negative_ref_map.txt
    ├── test_human.txt
    ├── test_negative.txt
    └── test_negative_ref_map.txt
```

### Combined Dataset

When combining both datasets:

```
Combined/
└── train_data/
    ├── train_human.txt          # Shuffled mix of WP + Award-winning
    ├── train_negative.txt
    ├── train_negative_ref_map.txt
    ├── dev_human.txt
    ├── dev_negative.txt
    ├── dev_negative_ref_map.txt
    ├── test_human.txt
    ├── test_negative.txt
    └── test_negative_ref_map.txt
```

## Data Format

All stories are in **single-line format**:
- Each line contains one complete story
- Paragraph breaks are replaced with spaces
- Stories are plain text without special formatting

Example:
```
This is the first paragraph of the story. This is the second paragraph. And this is the third paragraph.
```

## Training UNION

After preparing the dataset, train UNION with:

### WritingPrompts Only

```bash
python run_union.py \
    --data_dir ./Data/WritingPrompts \
    --output_dir ./model/union_wp \
    --task_name train \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/bert_model.ckpt
```

### Award-winning Only

```bash
python run_union.py \
    --data_dir ./Data/Award-winning \
    --output_dir ./model/union_award \
    --task_name train \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/bert_model.ckpt \
    --use_reconstruction
```

### Combined Dataset

```bash
python run_union.py \
    --data_dir ./Data/Combined \
    --output_dir ./model/union_combined \
    --task_name train \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/bert_model.ckpt \
    --use_reconstruction
```

**Note:** Make sure to download the BERT checkpoint first from [google-research/bert](https://github.com/google-research/bert). Use the uncased BERT-base model (110M parameters).

## Script Reference

### `download_writing_prompts.py`

Downloads and formats the WritingPrompts dataset.

**Usage:**
```bash
# Download from Hugging Face
python download_writing_prompts.py --source huggingface

# Show manual download instructions
python download_writing_prompts.py --source manual

# Verify dataset
python download_writing_prompts.py --verify

# Combine with Award-winning
python download_writing_prompts.py --combine
```

**Requirements:**
- For Hugging Face download: `pip install datasets`

### `prepare_award_winning.py`

Prepares the Award-winning dataset for training.

**Usage:**
```bash
python prepare_award_winning.py --award_dir ./Award-winning

# Customize train/dev/test split
python prepare_award_winning.py \
    --award_dir ./Award-winning \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15
```

### `prepare_full_dataset.py`

End-to-end pipeline that orchestrates everything.

**Usage:**
```bash
# Prepare WritingPrompts
python prepare_full_dataset.py --dataset wp

# Prepare Award-winning
python prepare_full_dataset.py --dataset award

# Prepare combined dataset
python prepare_full_dataset.py --dataset combined

# Skip download if data exists
python prepare_full_dataset.py --dataset wp --skip-download
```

## Troubleshooting

### Hugging Face Download Fails

If `pip install datasets` doesn't work or download fails:
1. Use manual download: `python download_writing_prompts.py --source manual`
2. Download from [THUcloud](https://cloud.tsinghua.edu.cn/d/b3bdeee2c9b647439746/) or [Google Drive](https://drive.google.com/drive/folders/1Cfc-YkQo-27ovVug2bfpqBclECimvgwu?usp=sharing)
3. Extract to `./WritingPrompts/`

### Missing Award-winning Dataset

The Award-winning dataset should be in `./Award-winning/` with story directories containing `story_text.txt` files. Make sure you have the negatives in `./Award-winning/train_data/negatives/`.

### Vocabulary Generation Fails

Make sure you have NLTK data downloaded:
```python
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

### ConceptNet Files Missing

Make sure these files are in `./Data/`:
- `conceptnet_entity.csv`
- `conceptnet_antonym.txt`
- `negation.txt`

## Dataset Statistics

### WritingPrompts

The full WritingPrompts dataset contains:
- **Train:** ~272,000 prompt-story pairs
- **Dev:** ~15,000 pairs
- **Test:** ~15,000 pairs

Stories are variable-length (typically 200-1000 tokens).

### Award-winning

The Award-winning dataset contains award-winning short stories from various sources. The exact count depends on your collection.

## Citation

If you use WritingPrompts dataset:
```
@inproceedings{fan2018hierarchical,
  title={Hierarchical Neural Story Generation},
  author={Fan, Angela and Lewis, Mike and Dauphin, Yann},
  booktitle={ACL},
  year={2018}
}
```

If you use UNION:
```
@inproceedings{guan2020union,
  title={UNION: An Unreferenced Metric for Evaluating Open-ended Story Generation},
  author={Guan, Jian and Huang, Minlie},
  booktitle={EMNLP},
  year={2020}
}
```
