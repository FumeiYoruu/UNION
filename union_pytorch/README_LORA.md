# LoRA Training for UNION PyTorch

This guide explains how to use `train_lora.py` for parameter-efficient fine-tuning of the UNION model using LoRA (Low-Rank Adaptation).

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- **Freezes** the pre-trained model weights
- **Trains** small low-rank matrices (adapters) that are added to specific layers
- **Reduces** memory usage by ~3x and training time significantly
- **Produces** small checkpoint files (~10-50MB instead of 400MB+)
- **Maintains** similar or better performance compared to full fine-tuning

## Installation

First, install the PEFT library:

```bash
pip install peft
```

Or if using the full requirements:
```bash
pip install -r requirements.txt  # Make sure it includes peft
```

## Basic Usage

### Training from Scratch with LoRA

```bash
python train_lora.py \
    --task_name train \
    --model_type bert \
    --model_name bert-base-uncased \
    --data_dir ./Data/ROCStories \
    --output_dir ./output/union_lora_roc \
    --dataset_mode roc \
    --train_batch_size 16 \
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1
```

### Training with Longformer/LED

```bash
python train_lora.py \
    --task_name train \
    --model_type longformer \
    --model_name allenai/led-base-16384 \
    --max_seq_length 4096 \
    --data_dir ./Data/WritingPrompts \
    --output_dir ./output/union_lora_wp \
    --dataset_mode wp \
    --train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --lora_r 8 \
    --lora_alpha 16
```

### Training with Combined Datasets

```bash
python train_lora.py \
    --task_name train \
    --model_type bert \
    --dataset_mode combined \
    --award_data_dir ./Data/Award-winning \
    --wp_data_dir ./Data/WritingPrompts \
    --award_has_reconstruction \
    --wp_has_reconstruction \
    --output_dir ./output/union_lora_combined \
    --train_batch_size 8 \
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32
```

## LoRA-Specific Parameters

### Core LoRA Parameters

- **`--lora_r`** (default: 8): LoRA rank (dimension of low-rank matrices)
  - Lower values (4, 8): Fewer parameters, faster training, less expressive
  - Higher values (16, 32, 64): More parameters, more expressive, better for complex tasks
  - Typical range: 4-64

- **`--lora_alpha`** (default: 16): LoRA scaling factor
  - Usually set to `2 * lora_r` or equal to `lora_r`
  - Controls the magnitude of LoRA updates
  - Typical range: 8-64

- **`--lora_dropout`** (default: 0.1): Dropout for LoRA layers
  - Standard dropout applied to LoRA matrices
  - Typical range: 0.0-0.3

### Target Modules

- **`--lora_target_modules`** (default: auto-detected): Which layers to apply LoRA to
  - For BERT: `--lora_target_modules query value` (default)
  - For Longformer: `--lora_target_modules q_proj v_proj` (default)
  - Can also target: `query value key dense` for more coverage

- **`--lora_modules_to_save`** (default: None): Additional modules to train
  - Use to train classifier head or other task-specific layers
  - Example: `--lora_modules_to_save classifier lm_head`

### Model Merging

- **`--merge_weights`**: Merge LoRA adapters with base model after training
  - Creates a single, standalone model without needing PEFT
  - Useful for deployment
  - Saved to `{output_dir}/merged_model/`

## Learning Rate Recommendations

LoRA typically uses **higher learning rates** than full fine-tuning:

- **Full fine-tuning**: 2e-5 (0.00002)
- **LoRA fine-tuning**: 3e-4 (0.0003) - recommended starting point
- Range for experimentation: 1e-4 to 5e-4

## Checkpoint Management

### Checkpoint Structure

LoRA checkpoints are much smaller than full model checkpoints:

```
output/
├── checkpoint-1000/
│   ├── adapter_config.json       # LoRA configuration
│   ├── adapter_model.bin         # LoRA weights (~10-50MB)
│   └── training_state.pt         # Optimizer/scheduler state
├── best-2500/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_state.pt
└── logs/                          # TensorBoard logs
```

### Resume Training

```bash
python train_lora.py \
    --task_name train \
    --resume_from_checkpoint ./output/union_lora_roc/checkpoint-1000 \
    ...other args...
```

### Load Pre-trained Base Model

```bash
python train_lora.py \
    --task_name train \
    --init_checkpoint ./pretrained_model/pytorch_model.bin \
    --model_type bert \
    ...other args...
```

## Multi-GPU Training

LoRA training supports DataParallel for training on multiple GPUs on a single machine.

### Enable Multi-GPU Training

```bash
python train_lora.py \
    --task_name train \
    --use_multi_gpu \  # This flag enables DataParallel
    --train_batch_size 32 \  # Total batch size (split across GPUs)
    --data_dir ./Data/ROCStories \
    --output_dir ./output/union_lora_multi_gpu \
    --dataset_mode roc \
    --learning_rate 3e-4 \
    --lora_r 8
```

### How It Works

- **Automatic GPU Detection**: Uses all available GPUs automatically
- **Batch Splitting**: Total batch size is split evenly across GPUs
- **Example**: With 4 GPUs and `--train_batch_size 32`:
  - Each GPU processes 8 samples per batch
  - Gradients are synchronized across GPUs after each batch
  - Effective batch size remains 32

### Multi-GPU Example (4 GPUs)

```bash
# Training with 4 GPUs
python train_lora.py \
    --task_name train \
    --model_type bert \
    --data_dir ./Data/WritingPrompts \
    --output_dir ./output/union_lora_wp_4gpu \
    --dataset_mode wp \
    --use_multi_gpu \
    --train_batch_size 64 \  # 16 per GPU
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --lora_r 16
```

### Expected Output

When multi-GPU is enabled, you'll see:
```
Using 4 GPUs with DataParallel!
GPU devices: ['NVIDIA A100-SXM4-40GB', 'NVIDIA A100-SXM4-40GB', ...]
Effective batch size: 64 (split across 4 GPUs)
...
Multi-GPU: Yes (DataParallel with 4 GPUs)
Per-GPU batch size: 16
```

### Batch Size Guidelines

| GPUs | Recommended Total Batch Size | Per-GPU Batch Size |
|------|----------------------------|-------------------|
| 2 | 32-64 | 16-32 |
| 4 | 64-128 | 16-32 |
| 8 | 128-256 | 16-32 |

**Note**: Keep per-GPU batch size at 16-32 for optimal memory usage and training speed.

### Multi-GPU + Memory Optimization

Combine multi-GPU with other optimizations for maximum efficiency:

```bash
python train_lora.py \
    --use_multi_gpu \
    --train_batch_size 64 \
    --gradient_accumulation_steps 2 \  # Effective batch = 128
    --gradient_checkpointing \
    --fp16 \
    --learning_rate 3e-4 \
    ...
```

### Important Notes

- **DataParallel limitations**: Single-node only (one machine with multiple GPUs)
- **Checkpoint saving**: Automatically handled - saves only LoRA adapters
- **Resume training**: Works seamlessly with `--resume_from_checkpoint`
- **Merge weights**: Works correctly with multi-GPU trained models

## Memory Optimization

LoRA is already memory-efficient, but you can optimize further:

### Gradient Checkpointing
```bash
--gradient_checkpointing  # Reduces memory by ~30-40%
```

### Gradient Accumulation
```bash
--train_batch_size 4 \
--gradient_accumulation_steps 8  # Effective batch size = 32
```

### Mixed Precision (FP16)
```bash
--fp16  # Reduces memory and speeds up training on compatible GPUs
```

### Combined Optimization Example
```bash
python train_lora.py \
    --task_name train \
    --model_type longformer \
    --data_dir ./Data/WritingPrompts \
    --output_dir ./output/union_lora_wp_optimized \
    --dataset_mode wp \
    --train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --fp16 \
    --learning_rate 3e-4 \
    --lora_r 8
```

## Lazy Loading for Large Datasets

For large datasets like WritingPrompts:

```bash
--lazy_loading  # Tokenize on-the-fly instead of pre-loading
--train_data_fraction 0.1  # Use 10% of training data
```

## Hyperparameter Tuning Recommendations

### For Small Datasets (ROCStories)
```bash
--lora_r 8
--lora_alpha 16
--learning_rate 3e-4
--train_batch_size 16
--num_train_epochs 5
```

### For Medium Datasets (Award-winning)
```bash
--lora_r 16
--lora_alpha 32
--learning_rate 2e-4
--train_batch_size 8
--num_train_epochs 3
```

### For Large Datasets (WritingPrompts)
```bash
--lora_r 16
--lora_alpha 32
--learning_rate 1e-4
--train_batch_size 4
--gradient_accumulation_steps 4
--num_train_epochs 2
--lazy_loading
```

## Comparing LoRA vs Full Fine-tuning

| Aspect | Full Fine-tuning | LoRA |
|--------|------------------|------|
| Trainable Parameters | 110M (BERT-base) | ~1-5M (1-5%) |
| Memory Usage | ~12GB | ~4GB |
| Checkpoint Size | ~400MB | ~10-50MB |
| Training Time | Baseline | 1.5-2x faster |
| Learning Rate | 2e-5 | 3e-4 |
| Recommended Epochs | 3-5 | 3-5 |

## Using LoRA Checkpoints

### Load for Inference

```python
from peft import PeftModel
from models import create_model

# Load base model
base_model = create_model(
    model_type="bert",
    model_name="bert-base-uncased"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./output/union_lora_roc/best-2500"
)
model.eval()

# Use for prediction
outputs = model(**batch)
```

### Merge and Export

```python
# Merge LoRA weights with base model
merged_model = model.merge_and_unload()

# Save as standalone model
merged_model.save_pretrained("./merged_model")
```

Or use the command-line flag:
```bash
--merge_weights  # Automatically merges after training
```

## Troubleshooting

### Error: "No module named 'peft'"
```bash
pip install peft
```

### Error: "Target modules not found"
Check your model architecture and specify correct target modules:
```bash
--lora_target_modules query value  # For BERT
--lora_target_modules q_proj v_proj  # For Longformer/LED
```

### Out of Memory (OOM)
1. Reduce batch size: `--train_batch_size 2`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Increase gradient accumulation: `--gradient_accumulation_steps 8`
4. Enable FP16: `--fp16`
5. Reduce LoRA rank: `--lora_r 4`

### Poor Performance
1. Increase LoRA rank: `--lora_r 16` or `--lora_r 32`
2. Adjust learning rate: try `1e-4` to `5e-4`
3. Train for more epochs: `--num_train_epochs 5`
4. Target more modules: `--lora_target_modules query value key dense`

## Complete Example

Here's a full command for training UNION with LoRA on ROCStories:

```bash
python train_lora.py \
    --task_name train \
    --use_multi_gpu \
    --model_type bert \
    --model_name bert-base-uncased \
    --data_dir ./Data/ROCStories \
    --output_dir ./output/union_lora_roc_v1 \
    --dataset_mode roc \
    --max_seq_length 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    --logging_steps 100 \
    --save_steps 500 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --device cuda \
    --seed 42
```

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685): "LoRA: Low-Rank Adaptation of Large Language Models"
- [PEFT Library](https://github.com/huggingface/peft): Hugging Face Parameter-Efficient Fine-Tuning
- [UNION Paper](https://arxiv.org/abs/2009.07602): Original UNION paper

## Next Steps

1. Start with the basic example above
2. Adjust hyperparameters based on your dataset size
3. Monitor training with TensorBoard: `tensorboard --logdir ./output/union_lora_roc/logs`
4. Compare results with full fine-tuning using `train.py`
5. Export merged model for deployment if needed
