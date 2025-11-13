# UNION Model Accuracy Testing

This guide explains how to test the accuracy of the fine-tuned UNION model on the WritingPrompts test dataset.

## Overview

The `test_union_accuracy.py` script evaluates UNION's binary classification performance on distinguishing human-written stories from negative samples (stories with defects like repetition, incoherence, contradictions, etc.).

## Prerequisites

1. **Python Environment**:
   - Python 3.7+ (3.7-3.10 recommended)
   - TensorFlow 1.14.0 OR TensorFlow 2.x (see installation below)
   - numpy
   - scikit-learn (for metrics)

2. **Required Files**:
   - Fine-tuned UNION checkpoint (download or train your own)
   - BERT vocabulary and config files
   - WritingPrompts test data

## Installation

### Option 1: TensorFlow 1.14.0 (Original Environment)

```bash
# Use the original UNION environment
pip install tensorflow-gpu==1.14.0  # or tensorflow==1.14.0 for CPU
pip install scikit-learn==0.22.1
pip install numpy==1.18.1
```

**Use this script**: `test_union_accuracy.py`

### Option 2: TensorFlow 2.x (Modern Environment)

```bash
# For newer systems with TensorFlow 2.x
pip install tensorflow>=2.0  # or tensorflow-gpu>=2.0
pip install scikit-learn
pip install numpy
```

**Use this script**: `test_union_accuracy_tf2.py`

**Note**: The TF2 version uses `tensorflow.compat.v1` to maintain compatibility with the original UNION model code.

## Data Preparation

### Option 1: Download Prepared Data

Download the full WritingPrompts dataset with test data from:
- [THUcloud](https://cloud.tsinghua.edu.cn/d/b3bdeee2c9b647439746/)
- [Google Drive](https://drive.google.com/drive/folders/1Cfc-YkQo-27ovVug2bfpqBclECimvgwu?usp=sharing)

Extract to `./Data/WP/` so you have:
```
Data/WP/
├── train_data/
│   ├── test_human.txt      # Human-written test stories
│   └── test_negative.txt   # Negative sample test stories
```

### Option 2: Generate Data Yourself

If you don't have the test data, you can generate it:

```bash
cd Data
python download_and_prepare_wp.py
```

This will download WritingPrompts from Hugging Face and generate the train/dev/test splits with negative samples.

## Download Fine-tuned UNION Model

### For WritingPrompts

Download the fine-tuned WritingPrompts checkpoint:
- [THUcloud](https://cloud.tsinghua.edu.cn/d/0154034b7a574d0498c9/)
- [GoogleDrive](https://drive.google.com/drive/folders/1Z6uYG4WQBR3jzZAykQGfAEHriWc8CA0l?usp=sharing)

Extract the checkpoint files to `./model/uncased_L-12_H-768_A-12/union_wp/`:
```
model/uncased_L-12_H-768_A-12/union_wp/
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
├── model.ckpt.meta
└── checkpoint
```

### BERT Base Model

You also need the BERT base model (uncased):

```bash
# Download from Google's BERT repository
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip -d ./model/

# Should create:
# model/uncased_L-12_H-768_A-12/
#   ├── bert_model.ckpt.*
#   ├── bert_config.json
#   └── vocab.txt
```

## Usage

### Basic Usage (WritingPrompts)

**For TensorFlow 1.14.0:**
```bash
python test_union_accuracy.py \
    --data_dir ./Data/WP \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --bert_config_file ./model/uncased_L-12_H-768_A-12/bert_config.json \
    --vocab_file ./model/uncased_L-12_H-768_A-12/vocab.txt \
    --output_dir ./test_results
```

**For TensorFlow 2.x:**
```bash
python test_union_accuracy_tf2.py \
    --data_dir ./Data/WP \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --bert_config_file ./model/uncased_L-12_H-768_A-12/bert_config.json \
    --vocab_file ./model/uncased_L-12_H-768_A-12/vocab.txt \
    --output_dir ./test_results
```

### For ROCStories Dataset

If you want to test on ROCStories instead:

```bash
python test_union_accuracy.py \
    --data_dir ./Data/ROCStories \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_roc/model.ckpt \
    --bert_config_file ./model/uncased_L-12_H-768_A-12/bert_config.json \
    --vocab_file ./model/uncased_L-12_H-768_A-12/vocab.txt \
    --output_dir ./test_results_roc
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./Data/WP` | Directory containing `train_data/test_human.txt` and `test_negative.txt` |
| `--init_checkpoint` | `./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt` | Path to fine-tuned UNION checkpoint |
| `--bert_config_file` | `./model/uncased_L-12_H-768_A-12/bert_config.json` | BERT config file |
| `--vocab_file` | `./model/uncased_L-12_H-768_A-12/vocab.txt` | BERT vocabulary file |
| `--output_dir` | `./test_results` | Where to save test results |
| `--max_seq_length` | `200` | Maximum sequence length (tokens) |
| `--predict_batch_size` | `32` | Batch size for prediction |
| `--use_reconstruction` | `False` | Set to `True` if model was trained with reconstruction task |

## Output

The script generates two files in `--output_dir`:

### 1. `test_metrics.txt`

Summary of classification metrics:

```
UNION Model Test Results
============================================================
Data directory: ./Data/WP
Model checkpoint: ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt
Total test examples: 2000
Human stories: 1000
Negative stories: 1000

Accuracy:  0.9450
Precision: 0.9523
Recall:    0.9370
F1 Score:  0.9446

Confusion Matrix:
[[950, 50], [63, 937]]

Per-Class Metrics:
Negative: P=0.9378, R=0.9500, F1=0.9439
Human: P=0.9523, R=0.9370, F1=0.9446
```

### 2. `predictions.txt`

Per-example predictions (TSV format):

```
ID      True_Label      Pred_Label      Probability
0       1               1               0.987654
1       1               1               0.923456
2       0               0               0.034567
...
```

## Console Output

While running, the script prints:

```
Loaded 1000 human stories and 1000 negative stories
Total test examples: 2000

Creating examples...
Converting to features...
Converting example 0/2000
Converting example 1000/2000

Building model...
Running predictions...

============================================================
CLASSIFICATION METRICS
============================================================
Accuracy:  0.9450
Precision: 0.9523
Recall:    0.9370
F1 Score:  0.9446

Confusion Matrix:
                 Predicted
                 Neg    Pos
Actual  Neg   [[  950     50]]
        Pos   [[   63    937]]

Detailed Classification Report:
              precision    recall  f1-score   support

    Negative     0.9378    0.9500    0.9439      1000
       Human     0.9523    0.9370    0.9446      1000

    accuracy                         0.9450      2000
   macro avg     0.9451    0.9435    0.9442      2000
weighted avg     0.9451    0.9450    0.9442      2000

============================================================

Per-Class Metrics:
Negative Stories (label=0):
  Precision: 0.9378
  Recall:    0.9500
  F1:        0.9439
  Support:   1000

Human Stories (label=1):
  Precision: 0.9523
  Recall:    0.9370
  F1:        0.9446
  Support:   1000

Results saved to:
  Metrics: ./test_results/test_metrics.txt
  Predictions: ./test_results/predictions.txt
```

## Metrics Explanation

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall proportion of correct predictions: (TP + TN) / Total |
| **Precision** | Of predicted human stories, how many are actually human: TP / (TP + FP) |
| **Recall** | Of actual human stories, how many are predicted as human: TP / (TP + FN) |
| **F1 Score** | Harmonic mean of precision and recall: 2 * (P * R) / (P + R) |

Where:
- **TP** (True Positives): Human stories correctly identified as human
- **TN** (True Negatives): Negative stories correctly identified as negative
- **FP** (False Positives): Negative stories incorrectly identified as human
- **FN** (False Negatives): Human stories incorrectly identified as negative

## Expected Performance

Based on the UNION paper (EMNLP 2020), expected test accuracy:

| Dataset | Accuracy | F1 Score |
|---------|----------|----------|
| ROCStories | ~94-96% | ~0.94-0.96 |
| WritingPrompts | ~92-94% | ~0.92-0.94 |

**Note**: Actual performance may vary depending on:
- Training data quality and size
- Negative sample construction strategy
- Model hyperparameters (learning rate, epochs, etc.)
- Random seed

## Troubleshooting

### Error: "AttributeError: module 'tensorflow' has no attribute 'flags'"

**Cause**: You're using TensorFlow 2.x but running the TF1 script.

**Solution**: Use the TF2-compatible script:
```bash
python test_union_accuracy_tf2.py \
    --data_dir ./Data/WP \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt
```

Or install TensorFlow 1.14.0:
```bash
pip install tensorflow==1.14.0
```

### Error: "Test data files not found"

**Solution**: Generate or download the test data first:
```bash
cd Data
python download_and_prepare_wp.py
```

### Error: "Checkpoint not found"

**Solution**: Ensure checkpoint files exist:
```bash
ls -la ./model/uncased_L-12_H-768_A-12/union_wp/
# Should show: model.ckpt.data-..., model.ckpt.index, model.ckpt.meta, checkpoint
```

### Low Accuracy (<50%)

**Possible causes**:
1. Model checkpoint doesn't match the dataset (e.g., using ROC model on WP data)
2. Data format mismatch (check that stories are one per line for WritingPrompts)
3. Wrong checkpoint path (ensure you're loading fine-tuned UNION, not base BERT)

**Solution**: Verify data format and checkpoint:
```bash
# Check first few lines of test data
head -3 ./Data/WP/train_data/test_human.txt

# Verify checkpoint files
ls ./model/uncased_L-12_H-768_A-12/union_wp/
```

### TensorFlow Warnings

If you see deprecation warnings, these are expected for TensorFlow 1.14.0 and can be ignored.

### GPU Out of Memory

**Solution**: Reduce batch size:
```bash
python test_union_accuracy.py \
    --data_dir ./Data/WP \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --predict_batch_size 8  # Reduced from 32
```

## Testing Your Own Stories

To test custom stories:

1. Create two files in the same format:
   - `custom_human.txt`: One story per line
   - `custom_negative.txt`: One story per line

2. Place them in a directory: `./Data/Custom/train_data/`

3. Run the test:
```bash
python test_union_accuracy.py \
    --data_dir ./Data/Custom \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --output_dir ./custom_test_results
```

## Comparing Multiple Checkpoints

To compare different model checkpoints:

```bash
# Test checkpoint at step 10000
python test_union_accuracy.py \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt-10000 \
    --output_dir ./results_step10k

# Test checkpoint at step 20000
python test_union_accuracy.py \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt-20000 \
    --output_dir ./results_step20k

# Compare results
diff ./results_step10k/test_metrics.txt ./results_step20k/test_metrics.txt
```

## Citation

If you use this testing script, please cite the original UNION paper:

```bibtex
@inproceedings{guan2020union,
    title={UNION: An Unreferenced Metric for Evaluating Open-ended Story Generation},
    author={Jian Guan and Minlie Huang},
    booktitle={EMNLP},
    year={2020},
    eprint={2009.07602},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Additional Resources

- **Original Paper**: https://arxiv.org/abs/2009.07602
- **Code Repository**: https://github.com/thu-coai/UNION
- **BERT**: https://github.com/google-research/bert
- **WritingPrompts Dataset**: Available via Hugging Face or official links

## Quick Start Summary

```bash
# 1. Install scikit-learn
pip install scikit-learn

# 2. Download/prepare data (if needed)
cd Data && python download_and_prepare_wp.py && cd ..

# 3. Download BERT base model
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip -d ./model/

# 4. Download fine-tuned UNION checkpoint (from THUcloud or Google Drive)
# Extract to ./model/uncased_L-12_H-768_A-12/union_wp/

# 5. Run test
python test_union_accuracy.py \
    --data_dir ./Data/WP \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt

# 6. Check results
cat ./test_results/test_metrics.txt

python test_union_accuracy.py \
      --data_dir /scratch1/tanalvin/Data/WritingPrompts \
      --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
      --bert_config_file ./model/uncased_L-12_H-768_A-12/bert_config.json \
      --vocab_file ./model/uncased_L-12_H-768_A-12/vocab.txt \
      --output_dir ./test_results
```
