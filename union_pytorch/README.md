# UNION PyTorch

A modern PyTorch reimplementation of **UNION** (An Unreferenced Metric for Evaluating Open-ended Story Generation) with support for long-context encoders like Longformer.

## Features

- **Modern PyTorch Implementation**: Clean, modular code using PyTorch 2.0+ and Hugging Face Transformers
- **Long Context Support**: Use Longformer (up to 16384 tokens) for longer stories
- **Compatible with Original Data**: Works with the same data format as the original TensorFlow implementation
- **Multi-layer Pooling**: Optional feature to use representations from all transformer layers
- **Reconstruction Task**: Optional auxiliary task for better training
- **Easy to Use**: Simple command-line interface with comprehensive configuration

## Quick Start

### Installation

```bash
cd union_pytorch
pip install -r requirements.txt
```

### Data Preparation

Use the same data preparation as the original UNION:

```bash
cd ../Data
python3 get_vocab.py roc        # For ROCStories
python3 gen_train_data.py roc   # Generate training data
```

This creates training data in `Data/ROCStories/train_data/`.

### Training with BERT

```bash
python train.py \
    --task_name train \
    --model_type bert \
    --model_name bert-base-uncased \
    --data_dir ../Data/ROCStories \
    --output_dir ./output/bert_roc \
    --dataset_mode roc \
    --max_seq_length 200 \
    --train_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_steps 500 \
    --save_steps 1000
```

### Training with Longformer (Longer Context - 16k tokens)

```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --model_name allenai/longformer-base-16384 \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_wp \
    --dataset_mode wp \
    --max_seq_length 2048 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --save_steps 1000
```

### Prediction and Evaluation

```bash
python predict.py \
    --task_name pred \
    --model_type longformer \
    --model_name allenai/longformer-base-16384 \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/predictions \
    --dataset_mode wp \
    --max_seq_length 2048 \
    --init_checkpoint ./output/longformer_wp/best-epoch3-step12000 \
    --eval_batch_size 16
```

This will output:
- UNION scores for each story
- Correlation metrics (Pearson, Spearman, Kendall) with human judgments

## Configuration Options

### Model Options

- `--model_type`: Choose encoder type
  - `bert`: BERT-base (512 tokens max)
  - `longformer`: Longformer (up to 16384 tokens)

- `--model_name`: Pretrained model name
  - `bert-base-uncased` (default for BERT)
  - `allenai/longformer-base-16384` (default for Longformer)
  - Any compatible model from Hugging Face

- `--max_seq_length`: Maximum sequence length
  - 200-512 for BERT
  - Up to 16384 for Longformer (2048-4096 recommended for most use cases)

### Dataset Options

- `--dataset_mode`: Dataset mode (default: `roc`)
  - `roc`: ROCStories (5-sentence stories)
  - `wp`: WritingPrompts (longer stories)
  - `award`: Award-winning literature
  - `combined`: **NEW** - Train on multiple datasets simultaneously

- `--data_dir`: Directory containing training data (required for single dataset modes; not needed for combined mode)

**Combined Dataset Mode Options:**

- `--award_data_dir PATH`: Path to Award-winning dataset directory (optional, for combined mode)
- `--wp_data_dir PATH`: Path to WritingPrompts dataset directory (optional, for combined mode)
- `--wp_has_reconstruction`: Enable reconstruction loss for WritingPrompts in combined mode (default: True)
- `--award_has_reconstruction`: Enable reconstruction loss for Award-winning in combined mode (default: True)
  - **Note**: Award-winning dataset uses `*_ref_map.txt` files to selectively apply reconstruction loss only to samples with references (1 = has reference, 0 = no reference)

### Advanced Features

- `--use_all_layers`: Enable multi-layer pooling (uses all transformer layers)
- `--use_reconstruction`: Enable reconstruction task for single dataset mode (auxiliary masked LM objective)
- `--reconstruction_weight`: Weight for reconstruction loss (default: 0.1)

### Training Options

- `--train_batch_size`: Batch size for training (default: 8)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--num_train_epochs`: Number of epochs (default: 3)
- `--warmup_steps`: Warmup steps for learning rate scheduler
- `--weight_decay`: Weight decay (default: 0.01)
- `--max_grad_norm`: Gradient clipping (default: 1.0)

### Device Options

- `--device`: Device to use
  - `cuda`: NVIDIA GPU (default)
  - `mps`: Apple Silicon GPU
  - `cpu`: CPU

- `--fp16`: Enable mixed precision training (faster on modern GPUs)

## Project Structure

```
union_pytorch/
├── README.md
├── requirements.txt
├── config.py              # Configuration classes
├── train.py               # Training script
├── predict.py             # Prediction script
├── models/
│   ├── __init__.py
│   └── union_model.py     # UNION model implementation
├── data/
│   ├── __init__.py
│   └── dataset.py         # Dataset classes
└── utils/
    ├── __init__.py
    └── training_utils.py  # Training utilities
```

## Model Architecture

### UNION Classifier

```
Story Input → Tokenizer
    ↓
Encoder (BERT/Longformer)
    ↓
[Optional] Multi-layer Pooling
    ↓
[CLS] Token → Classifier
    ↓
P(human-written)
```

### With Reconstruction Task

```
Story Input → Encoder → Classifier → Classification Loss
                ↓
Reference Story → Encoder → LM Head → Reconstruction Loss
                                ↓
                        Total Loss = cls_loss + 0.1 * rec_loss
```

## Comparison: PyTorch vs Original TensorFlow

| Feature | Original (TF 1.x) | PyTorch Version |
|---------|------------------|-----------------|
| Framework | TensorFlow 1.14 | PyTorch 2.0+ |
| Encoder | BERT only | BERT + Longformer |
| Max Length | 512 tokens | Up to 16384 tokens |
| Data Loading | TF Records | PyTorch Dataset |
| Training | TPUEstimator | PyTorch Trainer |
| Mixed Precision | N/A | FP16 support |
| Code Style | Legacy TF 1.x | Modern, modular |
| Compatibility | Original data ✓ | Original data ✓ |

## Examples

### Example 1: Train BERT on ROCStories

```bash
python train.py \
    --task_name train \
    --model_type bert \
    --data_dir ../Data/ROCStories \
    --output_dir ./output/bert_roc \
    --dataset_mode roc \
    --max_seq_length 200 \
    --train_batch_size 10 \
    --num_train_epochs 3
```

### Example 2: Train Longformer with Multi-layer Pooling (16k context)

```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --model_name allenai/longformer-base-16384 \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_multilayer \
    --dataset_mode wp \
    --max_seq_length 4096 \
    --use_all_layers \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8
```

### Example 3: Train with Reconstruction Task

```bash
python train.py \
    --task_name train \
    --model_type bert \
    --data_dir ../Data/ROCStories \
    --output_dir ./output/bert_reconstruction \
    --dataset_mode roc \
    --use_reconstruction \
    --reconstruction_weight 0.1 \
    --train_batch_size 8
```

### Example 4: Train with Full 16k Context (Very Long Stories)

```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --model_name allenai/longformer-base-16384 \
    --data_dir ../Data/WritingPrompts \
    --output_dir ./output/longformer_16k \
    --dataset_mode wp \
    --max_seq_length 16384 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 1e-5
```

**Note**: Full 16k context requires significant GPU memory (40GB+). Consider using smaller `max_seq_length` (2048-4096) for most applications.

### Example 5: Evaluate and Get Correlations

```bash
python predict.py \
    --task_name pred \
    --model_type bert \
    --data_dir ../Data/ROCStories \
    --output_dir ./output/results \
    --init_checkpoint ./output/bert_roc/best-epoch3-step10000 \
    --eval_batch_size 32
```

### Example 6: Train with Award Winning
```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --model_name allenai/longformer-base-16384 \
    --data_dir ../Data/Award-winning \
    --output_dir ./output/longformer_award \
    --dataset_mode award \
    --max_seq_length 4096 \
    --use_reconstruction \
    --reconstruction_weight 0.1 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --warmup_steps 500 \
    --fp16 \
    --save_steps 500
```

### Example 7: Train on Combined Datasets (Award-winning + WritingPrompts)

**NEW**: You can now train on multiple datasets simultaneously! This is particularly useful when combining Award-winning stories with WritingPrompts. Both datasets automatically use reconstruction loss where applicable:
- **WritingPrompts**: All entries use reconstruction (all have references)
- **Award-winning**: Uses `*_ref_map.txt` files to selectively apply reconstruction (only to samples with references)

```bash
python train.py \
    --task_name train \
    --model_type longformer \
    --model_name allenai/longformer-base-16384 \
    --output_dir ./output/longformer_combined \
    --dataset_mode combined \
    --award_data_dir ./Data/Award-winning \
    --wp_data_dir ./Data/WritingPrompts \
    --max_seq_length 4096 \
    --reconstruction_weight 0.1 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --fp16 \
    --save_steps 500
```

**Note**: Both `--wp_has_reconstruction` and `--award_has_reconstruction` default to `True`, so you don't need to specify them unless you want to disable reconstruction for a dataset.

**Combined Dataset Options:**

- `--dataset_mode combined`: Enable combined dataset mode
- `--award_data_dir PATH`: Path to Award-winning dataset (optional)
- `--wp_data_dir PATH`: Path to WritingPrompts dataset (optional)
- `--wp_has_reconstruction`: Enable reconstruction for WritingPrompts (default: True)
- `--award_has_reconstruction`: Enable reconstruction for Award-winning (default: True)
  - Uses `*_ref_map.txt` files to selectively apply reconstruction based on reference availability

**Benefits of Combined Training:**
- **More diverse training data**: Learn from both award-winning literature and creative writing prompts
- **Selective reconstruction**: Automatically applies reconstruction loss only to samples with references
  - WritingPrompts: All entries have references → all use reconstruction
  - Award-winning: Uses `*_ref_map.txt` to determine which samples have references
- **Better generalization**: Model learns story quality from multiple sources
- **Flexible configuration**: Choose which datasets to include and whether to use reconstruction for each

**Notes:**
- You must provide at least one of `--award_data_dir` or `--wp_data_dir` when using `--dataset_mode combined`
- The `--data_dir` parameter is **not required** in combined mode (only `--output_dir` is needed)
- **WritingPrompts**: All entries use reconstruction loss (all entries have references)
- **Award-winning**: Uses `*_ref_map.txt` files to selectively apply reconstruction loss
  - Samples with `1` in ref_map: Use reconstruction loss (have reference stories)
  - Samples with `0` in ref_map: Use classification loss only (no reference)
  - This allows mixed training where only samples with references get reconstruction loss

## Cloud Training & Checkpoints

### Checkpoint Management

The training script includes robust checkpoint management for cloud environments where jobs may be interrupted:

**Checkpoint Types:**

1. **Step checkpoints** (`checkpoint-epoch{N}-step{M}`):
   - Saved every `--save_steps` (default: 500 steps)
   - Automatically keeps only last `--keep_last_n_checkpoints` (default: 3)
   - Used for resuming interrupted training

2. **Best checkpoints** (`best-epoch{N}-step{M}`):
   - Saved when validation F1 improves
   - All best checkpoints are kept (no auto-cleanup)

3. **Epoch checkpoints** (`epoch-epoch{N}-step{M}`):
   - Saved at the end of each epoch
   - Keeps last 2 epochs

**Resume Training:**

```bash
# Continue from an interrupted checkpoint (loads model + optimizer + scheduler + training state)
python train.py \
    --resume_from_checkpoint ./output/longformer_combined/checkpoint-epoch2-step3500 \
    --output_dir ./output/longformer_combined \
    ... (other args same as original training)
```

**Start from Pretrained Weights:**

```bash
# Load only model weights, start fresh training (new optimizer/scheduler)
python train.py \
    --init_checkpoint ./output/previous_run/best-epoch5-step12000 \
    --output_dir ./output/new_training \
    ... (other args)
```

### Training Duration & Steps

**Typical Training Requirements:**

| Dataset | Dataset Size | Steps/Epoch | Total Steps (3 epochs) | Training Time* |
|---------|-------------|-------------|----------------------|---------------|
| ROCStories | ~50K pairs | ~6,250 | ~18,750 | 3-5 hours |
| WritingPrompts | ~200K pairs | ~25,000 | ~75,000 | 15-20 hours |
| Award-winning | ~10K pairs | ~1,250 | ~3,750 | 1-2 hours |
| **Combined (Award + WP)** | ~210K pairs | ~26,250 | **~78,750** | **16-22 hours** |

*Estimated with Longformer, batch_size=2, grad_accum=8, max_seq_length=4096, on A100 40GB

**Recommended Checkpoint Intervals for Cloud:**

| Training Duration | `--save_steps` | `--keep_last_n_checkpoints` | Disk Space |
|-------------------|---------------|---------------------------|------------|
| < 5 hours | 1000 | 3 | ~9 GB |
| 5-10 hours | 500 | 3 | ~9 GB |
| 10-20 hours | **250-500** | **3-5** | **9-15 GB** |
| > 20 hours | **250** | **5** | **15 GB** |

**For combined Award-winning + WritingPrompts (16-22 hours):**
```bash
--save_steps 500 \
--keep_last_n_checkpoints 5 \
--eval_steps 2000
```

This saves checkpoints every ~15-20 minutes, keeping the last 5 (allowing you to resume from at most 1.5 hours back).

### Cloud-Optimized Training Example

```bash
python union_pytorch/train.py \
    --task_name train \
    --dataset_mode combined \
    --award_data_dir ./Data/Award-winning \
    --wp_data_dir ./Data/WritingPrompts \
    --output_dir ./output/longformer_combined \
    --model_type longformer \
    --max_seq_length 4096 \
    --reconstruction_weight 0.1 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --save_steps 500 \
    --keep_last_n_checkpoints 5 \
    --eval_steps 2000 \
    --fp16
```

**If training gets interrupted:**
```bash
# Find the latest checkpoint
ls -lt ./output/longformer_combined/ | grep checkpoint

# Resume from it
python train.py \
    --resume_from_checkpoint ./output/longformer_combined/checkpoint-epoch3-step15500 \
    --task_name train \
    --dataset_mode combined \
    --award_data_dir ./Data/Award-winning \
    --wp_data_dir ./Data/WritingPrompts \
    --output_dir ./output/longformer_combined \
    ... (same args as before)
```

## Performance Tips

1. **For longer stories**: Use Longformer with `max_seq_length=2048` or higher (up to 16384)
2. **For limited GPU memory**:
   - Reduce `train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use `--fp16` for mixed precision
   - Use shorter `max_seq_length` (2048-4096 instead of 16384)
3. **For faster training**: Use BERT instead of Longformer for short stories
4. **For better accuracy**: Enable `--use_all_layers` for multi-layer pooling
5. **For cloud training**: Use `--save_steps 250-500` with `--keep_last_n_checkpoints 5` to protect against interruptions

## Hardware Requirements

### Minimum
- **BERT**: 8GB GPU RAM, 200 token sequences
- **Longformer (2048 tokens)**: 16GB GPU RAM
- **Longformer (4096 tokens)**: 24GB GPU RAM

### Recommended
- **BERT**: 16GB GPU RAM (e.g., V100, A10)
- **Longformer (2048-4096 tokens)**: 24GB+ GPU RAM (e.g., A100, RTX 3090, RTX 4090)
- **Longformer (16384 tokens)**: 40GB+ GPU RAM (e.g., A100 40GB/80GB)

## Troubleshooting

### Out of Memory Error
```bash
# Solution 1: Reduce batch size and use gradient accumulation
--train_batch_size 4 --gradient_accumulation_steps 4

# Solution 2: Reduce sequence length
--max_seq_length 512

# Solution 3: Use mixed precision
--fp16
```

### CUDA Out of Memory (Longformer)
```bash
# Use smaller batch size with Longformer
--train_batch_size 2 --gradient_accumulation_steps 8
```

### Data Not Found
```bash
# Make sure you've run data preparation first
cd ../Data
python3 get_vocab.py roc
python3 gen_train_data.py roc
```

## Citation

If you use this code, please cite the original UNION paper:

```bibtex
@inproceedings{union2020,
    title={UNION: An Unreferenced Metric for Evaluating Open-ended Story Generation},
    author={Jian Guan and Minlie Huang},
    booktitle={EMNLP},
    year={2020}
}
```

## License

This implementation follows the license of the original UNION repository.

## Acknowledgments

- Original UNION implementation: [thu-coai/UNION](https://github.com/thu-coai/UNION)
- Longformer: [allenai/longformer](https://github.com/allenai/longformer)
- Hugging Face Transformers: [huggingface/transformers](https://github.com/huggingface/transformers)
