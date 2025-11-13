"""Simplified LoRA training script for PyTorch UNION model.

This script is a streamlined version of train_lora.py that focuses on:
- Standard sequence classification (no reconstruction task)
- Single-layer pooling (no multi-layer pooling)
- Pure Longformer models with CLS token pooling
- Cleaner, more focused codebase

Use this for straightforward fine-tuning of Longformer/BERT for story quality classification.

Requires: pip install peft
"""

import os
import sys
import time
from tqdm import tqdm
import argparse
import warnings

# Suppress Longformer/LED attention window padding warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*automatically padded.*")
warnings.filterwarnings("ignore", message=".*attention.*window.*")

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model
from data import StoryDataset, DataCollatorWithDynamicPadding, DataCollatorWithFixedBuckets
from utils import (
    set_seed,
    get_device,
    create_optimizer_and_scheduler,
    AverageMeter,
    format_time,
    compute_metrics,
)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simplified PyTorch UNION Training with LoRA - No Reconstruction, Single-Layer Pooling"
    )

    # Model arguments
    parser.add_argument("--model_type", type=str, default="longformer",
                        choices=["bert", "longformer"],
                        help="Type of model to use")
    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096",
                        help="Pretrained model name or path")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--pooling_strategy", type=str, default="cls",
                        choices=["mean", "attention", "cls"],
                        help="Pooling strategy: 'cls' (RECOMMENDED for pure Longformer), "
                             "'attention' (learned weights), 'mean' (average)")

    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (dimension of low-rank matrices)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout probability")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=None,
                        help="Target modules for LoRA (default: ['query', 'value'])")
    parser.add_argument("--merge_weights", action="store_true",
                        help="Merge LoRA weights with base model after training")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model and outputs")
    parser.add_argument("--dataset_mode", type=str, default="wp",
                        choices=["roc", "wp", "award"],
                        help="Dataset mode: ROCStories, WritingPrompts, or Award-winning")
    parser.add_argument("--train_data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (e.g., 0.1 for 10%)")
    parser.add_argument("--lazy_loading", action="store_true",
                        help="Use lazy loading: tokenize data on-the-fly")
    parser.add_argument("--padding_strategy", type=str, default="dynamic",
                        choices=["dynamic", "bucket", "fixed"],
                        help="Padding strategy: 'dynamic' (longest in batch), 'bucket', 'fixed'")
    parser.add_argument("--padding_buckets", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384],
                        help="Bucket sizes for bucket padding strategy")

    # Training arguments
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["train", "pred"],
                        help="Task: train or predict")
    parser.add_argument("--train_batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (typically higher for LoRA, e.g., 3e-4)")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use efficient attention (xFormers or Flash Attention 2)")
    parser.add_argument("--compile_model", action="store_true",
                        help="Compile model with torch.compile() (PyTorch 2.0+)")

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every X steps")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=3,
                        help="Keep only the last N checkpoints")

    # Device and seed
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"],
                        help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Checkpoint
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Initial checkpoint to load (full model)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from LoRA checkpoint")

    return parser.parse_args()


def get_default_lora_target_modules(model_type):
    """Get default LoRA target modules based on model type."""
    if model_type == "bert":
        return ["query", "value"]
    elif model_type == "longformer":
        return ["query", "value"]
    else:
        return ["query", "value"]


def create_lora_model(base_model, args, device):
    """
    Wrap the base model with LoRA adapters.

    Args:
        base_model: The base UNION model
        args: Command line arguments
        device: Device to use

    Returns:
        Model wrapped with LoRA adapters
    """
    # Determine target modules
    if args.lora_target_modules is None:
        target_modules = get_default_lora_target_modules(args.model_type)
    else:
        target_modules = args.lora_target_modules

    # For simple mode: only save classifier (no reconstruction or multi-layer pooling)
    modules_to_save = ["classifier"]

    # Add attention_pooling if using attention pooling strategy
    if args.pooling_strategy == "attention":
        modules_to_save.append("attention_pooling")

    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {args.lora_r}")
    print(f"  Alpha: {args.lora_alpha}")
    print(f"  Dropout: {args.lora_dropout}")
    print(f"  Target modules: {target_modules}")
    print(f"  Additional trainable modules: {modules_to_save}")

    # Create LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=modules_to_save,
    )

    # Wrap model with LoRA
    model = get_peft_model(base_model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def save_lora_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    output_dir,
    prefix="checkpoint",
    keep_last_n=3,
    batch_step=0,
):
    """Save LoRA adapter checkpoint."""
    checkpoint_dir = os.path.join(output_dir, f"{prefix}-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Handle DataParallel wrapper
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save LoRA adapter weights
    model_to_save.save_pretrained(checkpoint_dir)

    # Save optimizer and scheduler states
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": global_step,
        "batch_step": batch_step,
    }, os.path.join(checkpoint_dir, "training_state.pt"))

    print(f"Saved LoRA checkpoint to {checkpoint_dir}")

    # Clean up old checkpoints
    if keep_last_n > 0:
        checkpoints = sorted([
            d for d in os.listdir(output_dir)
            if d.startswith(prefix) and os.path.isdir(os.path.join(output_dir, d))
        ], key=lambda x: int(x.split("-")[-1]))

        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint_path = os.path.join(output_dir, checkpoint)
                import shutil
                shutil.rmtree(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    global_step,
    args,
    writer,
    scaler=None,
    eval_dataloader=None,
    best_f1=0.0,
    start_step=0,
):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    # Track time breakdown
    data_time = 0
    forward_time = 0
    backward_time = 0
    step_start = time.time()

    for step, batch in enumerate(progress_bar):
        # Skip batches if resuming
        if start_step > 0 and step < start_step:
            continue

        data_time += time.time() - step_start

        # Move batch to device
        forward_start = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass with optional mixed precision
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    if loss.dim() > 0:
                        loss = loss.mean()
            else:
                outputs = model(**batch)
                loss = outputs["loss"]
                if loss.dim() > 0:
                    loss = loss.mean()

        forward_time += time.time() - forward_start

        # Backward pass
        backward_start = time.time()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        backward_time += time.time() - backward_start

        # Update meters
        loss_meter.update(loss.item() * args.gradient_accumulation_steps)

        # Update weights
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Calculate timing breakdown
            total_time = data_time + forward_time + backward_time

            # Logging
            if global_step % args.logging_steps == 0 and writer is not None:
                writer.add_scalar("train/loss", loss_meter.avg, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                # Log timing breakdown
                if total_time > 0:
                    writer.add_scalar("perf/data_time_pct", 100 * data_time / total_time, global_step)
                    writer.add_scalar("perf/forward_time_pct", 100 * forward_time / total_time, global_step)
                    writer.add_scalar("perf/backward_time_pct", 100 * backward_time / total_time, global_step)

                writer.flush()

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "data%": f"{100*data_time/total_time:.0f}" if total_time > 0 else "0",
            })

        # Save checkpoint
        if (step + 1) % args.gradient_accumulation_steps == 0 and global_step % args.save_steps == 0:
            print(f"\nSaving checkpoint at step {global_step}...")
            save_lora_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                args.output_dir,
                prefix="checkpoint",
                keep_last_n=args.keep_last_n_checkpoints,
                batch_step=step + 1,
            )

        # Evaluate during training
        if (step + 1) % args.gradient_accumulation_steps == 0 and args.eval_steps > 0 and global_step % args.eval_steps == 0 and eval_dataloader is not None:
            print(f"\n{'='*80}")
            print(f"Evaluating at step {global_step} (using 1% of eval data)...")
            print('='*80)

            eval_metrics = evaluate(model, eval_dataloader, device, eval_fraction=0.01)

            print(f"Validation metrics (step {global_step}):")
            print(f"  Loss: {eval_metrics['loss']:.4f}")
            print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Precision: {eval_metrics['precision']:.4f}")
            print(f"  Recall: {eval_metrics['recall']:.4f}")
            print(f"  F1: {eval_metrics['f1']:.4f}")

            # Log to tensorboard
            if writer is not None:
                for key, value in eval_metrics.items():
                    writer.add_scalar(f"eval/{key}", value, global_step)
                writer.flush()

            # Save best model
            if eval_metrics["f1"] > best_f1:
                best_f1 = eval_metrics["f1"]
                print(f"  New best F1: {best_f1:.4f} - Saving checkpoint...")
                save_lora_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    args.output_dir,
                    prefix="best",
                    keep_last_n=0,
                    batch_step=step + 1,
                )

            print('='*80 + '\n')
            model.train()

        # Reset timer
        step_start = time.time()

    return global_step, best_f1


def evaluate(model, dataloader, device, eval_fraction=1.0):
    """Evaluate model."""
    model.eval()

    all_preds = []
    all_labels = []
    loss_meter = AverageMeter()

    total_batches = len(dataloader)
    batches_to_eval = max(1, int(total_batches * eval_fraction))

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=batches_to_eval)):
            if batch_idx >= batches_to_eval:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                outputs = model(**batch)

            loss = outputs["loss"]
            if loss.dim() > 0:
                loss = loss.mean()

            logits = outputs["logits"]

            loss_meter.update(loss.item())

            all_preds.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = loss_meter.avg

    return metrics


def main():
    """Main training function."""
    args = get_args()

    # Validate arguments
    if not args.data_dir:
        raise ValueError("--data_dir is required")

    if not 0.0 < args.train_data_fraction <= 1.0:
        raise ValueError("train_data_fraction must be between 0.0 and 1.0")

    # Set seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Check for multiple GPUs
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if num_gpus > 1:
        print(f"Found {num_gpus} GPUs - will use DataParallel")
        print(f"GPU devices: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

    # Auto-set model_name for longformer
    if args.model_type == "longformer" and args.model_name == "bert-base-uncased":
        args.model_name = "allenai/longformer-base-4096"
        print(f"Auto-setting model_name to {args.model_name} for model_type={args.model_type}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load datasets
    print("Loading training data...")
    train_dataset = StoryDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        mode="train",
        dataset_type=args.dataset_mode,
        max_seq_length=args.max_seq_length,
        use_reconstruction=False,  # No reconstruction in simple mode
        data_fraction=args.train_data_fraction,
        lazy_loading=args.lazy_loading,
    )

    print("Loading validation data...")
    eval_dataset = StoryDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        mode="dev",
        dataset_type=args.dataset_mode,
        max_seq_length=args.max_seq_length,
        use_reconstruction=False,  # No reconstruction in simple mode
        lazy_loading=args.lazy_loading,
    )

    # Setup padding collator
    collate_fn = None
    if args.padding_strategy == "dynamic":
        collate_fn = DataCollatorWithDynamicPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8
        )
        print("Using dynamic padding (pads to longest in batch, multiple of 8)")
    elif args.padding_strategy == "bucket":
        collate_fn = DataCollatorWithFixedBuckets(
            tokenizer=tokenizer,
            buckets=args.padding_buckets
        )
        print(f"Using bucket padding with buckets: {args.padding_buckets}")
    else:
        print("Using fixed padding (max_seq_length)")

    # Create dataloaders
    num_workers = 32 if not args.lazy_loading else 32

    # Use seeded RandomSampler for reproducible shuffling
    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = RandomSampler(train_dataset, generator=generator)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    # Auto-adjust pooling strategy for pure Longformer
    pooling_strategy = args.pooling_strategy
    if args.model_type == "longformer" and "led" not in args.model_name.lower():
        if args.pooling_strategy == "mean":
            pooling_strategy = "cls"
            print(f"\nAuto-setting pooling_strategy='cls' for pure Longformer model")
            print(f"Note: Pure Longformer models have specialized CLS token for sequence classification\n")

    # Create base model (no reconstruction, no multi-layer pooling)
    print(f"Creating base model: {args.model_type} - {args.model_name}")
    base_model = create_model(
        model_type=args.model_type,
        model_name=args.model_name,
        use_all_layers=False,  # Simple mode: no multi-layer pooling
        use_reconstruction=False,  # Simple mode: no reconstruction
        gradient_checkpointing=False,
        pooling_strategy=pooling_strategy,
    )

    # Load initial weights if provided
    if args.init_checkpoint:
        print(f"\nLoading initial model weights from: {args.init_checkpoint}")
        from utils import load_model_weights
        load_model_weights(base_model, args.init_checkpoint, device)
        print("Loaded weights into base model\n")

    base_model.to(device)

    # Enable efficient attention if requested
    if args.use_flash_attention:
        attention_enabled = False

        try:
            import xformers
            if hasattr(base_model, 'enable_xformers_memory_efficient_attention'):
                base_model.enable_xformers_memory_efficient_attention()
                print(f"✓ xFormers memory-efficient attention enabled (version {xformers.__version__})")
                attention_enabled = True
            elif hasattr(base_model.encoder, 'enable_xformers_memory_efficient_attention'):
                base_model.encoder.enable_xformers_memory_efficient_attention()
                print(f"✓ xFormers memory-efficient attention enabled for encoder (version {xformers.__version__})")
                attention_enabled = True
        except ImportError:
            print("ℹ️  xFormers not found, trying Flash Attention 2...")
        except Exception as e:
            print(f"⚠️  Could not enable xFormers: {e}")

        if not attention_enabled:
            try:
                import flash_attn
                if hasattr(base_model.encoder, 'config'):
                    base_model.encoder.config._attn_implementation = "flash_attention_2"
                    print(f"✓ Flash Attention 2 enabled for encoder (version {flash_attn.__version__})")
                    attention_enabled = True
            except ImportError:
                print("⚠️  Flash Attention 2 not found")
            except Exception as e:
                print(f"⚠️  Could not enable Flash Attention 2: {e}")

        if not attention_enabled:
            print("\n❌ ERROR: --use_flash_attention specified but no library found!")
            print("Install: pip install xformers  OR  pip install flash-attn --no-build-isolation")

    # Wrap with LoRA
    print("\nWrapping model with LoRA adapters...")
    model = create_lora_model(base_model, args, device)

    # Compile model if requested
    if args.compile_model:
        try:
            torch._dynamo.config.suppress_errors = True
            print("\nCompiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compilation successful")
        except Exception as e:
            print(f"Warning: Could not compile model - {e}")

    # Multi-GPU training with DataParallel
    if num_gpus > 1:
        print(f"\nUsing DataParallel with {num_gpus} GPUs")
        print(f"Per-GPU batch size: {args.train_batch_size}")
        print(f"Effective global batch size: {args.train_batch_size * num_gpus}")
        model = DataParallel(model)

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * args.num_train_epochs

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        learning_rate=args.learning_rate,
        num_training_steps=num_training_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
    )

    # Load from checkpoint if resuming
    start_epoch = 0
    global_step = 0
    start_batch_step = 0

    if args.resume_from_checkpoint:
        print(f"\nResuming training from LoRA checkpoint: {args.resume_from_checkpoint}")

        # Load LoRA adapter weights
        model = PeftModel.from_pretrained(base_model, args.resume_from_checkpoint)
        model.to(device)

        # Compile if requested
        if args.compile_model:
            try:
                torch._dynamo.config.suppress_errors = True
                print("Compiling resumed model with torch.compile()...")
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compilation successful")
            except Exception as e:
                print(f"Warning: Could not compile resumed model - {e}")

        # Wrap with DataParallel if using multi-GPU
        if num_gpus > 1:
            print(f"Re-wrapping resumed model with DataParallel for {num_gpus} GPUs")
            model = DataParallel(model)

        # Load training state
        training_state_path = os.path.join(args.resume_from_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
            saved_epoch = training_state.get("epoch", 0)
            global_step = training_state.get("step", 0)
            start_batch_step = training_state.get("batch_step", 0)

            if start_batch_step == 0:
                start_epoch = saved_epoch
                print(f"Resuming from END of epoch {saved_epoch}, starting epoch {saved_epoch + 1}")
            else:
                start_epoch = saved_epoch - 1
                print(f"Resuming from MID of epoch {saved_epoch}")

            print(f"Global step: {global_step}, will skip first {start_batch_step} batches\n")
        else:
            print(f"Warning: training_state.pt not found\n")

    # Setup mixed precision training
    scaler = None
    if args.fp16:
        if device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision (FP16) training enabled")
        else:
            print(f"Warning: --fp16 specified but device is {device.type}")

    # Setup tensorboard
    tensorboard_log_dir = os.path.join(args.output_dir, "logs")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print(f"\nTensorBoard logging to: {tensorboard_log_dir}")
    print(f"Start TensorBoard with: tensorboard --logdir {tensorboard_log_dir}\n")

    # Print training info
    print("\n" + "=" * 80)
    print("Simplified LoRA Training Configuration:")
    print("=" * 80)
    print(f"Model: {args.model_type} - {args.model_name}")
    print(f"Pooling strategy: {pooling_strategy}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"Dataset mode: {args.dataset_mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Training data fraction: {args.train_data_fraction*100:.1f}%")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Total training steps: {num_training_steps}")
    print(f"Mixed precision (FP16): {args.fp16 and device.type == 'cuda'}")
    print(f"Device: {device}")
    if num_gpus > 1:
        print(f"Multi-GPU: Yes (DataParallel with {num_gpus} GPUs)")
    else:
        print(f"Multi-GPU: No")
    print(f"Model compilation: {args.compile_model}")
    print("=" * 80 + "\n")

    # Training loop
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(start_epoch, args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        print("-" * 80)

        current_start_batch = start_batch_step if epoch == start_epoch else 0

        # Train
        global_step, best_f1 = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            epoch + 1,
            global_step,
            args,
            writer,
            scaler=scaler,
            eval_dataloader=eval_dataloader,
            best_f1=best_f1,
            start_step=current_start_batch,
        )

        # Evaluate at end of epoch
        print(f"\n{'='*80}")
        print(f"End of Epoch {epoch + 1} - Evaluating (using 10% of eval data)...")
        print('='*80)
        eval_metrics = evaluate(model, eval_dataloader, device, eval_fraction=0.10)

        print(f"Validation metrics (end of epoch {epoch + 1}):")
        print(f"  Loss: {eval_metrics['loss']:.4f}")
        print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Precision: {eval_metrics['precision']:.4f}")
        print(f"  Recall: {eval_metrics['recall']:.4f}")
        print(f"  F1: {eval_metrics['f1']:.4f}")

        # Log to tensorboard
        if writer is not None:
            for key, value in eval_metrics.items():
                writer.add_scalar(f"eval/{key}", value, global_step)
            writer.flush()

        # Save best model
        if eval_metrics["f1"] > best_f1:
            best_f1 = eval_metrics["f1"]
            print(f"  New best F1: {best_f1:.4f} - Saving checkpoint...")
            save_lora_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                args.output_dir,
                prefix="best",
                keep_last_n=0,
                batch_step=0,
            )

        print('='*80)

        # Save epoch checkpoint
        save_lora_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            global_step,
            args.output_dir,
            prefix="epoch",
            keep_last_n=2,
            batch_step=0,
        )

    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"Total time: {format_time(elapsed_time)}")
    print(f"Best F1: {best_f1:.4f}")
    print("=" * 80)

    # Optionally merge and save full model
    if args.merge_weights:
        print("\nMerging LoRA weights with base model...")
        model_to_merge = model.module if hasattr(model, 'module') else model
        merged_model = model_to_merge.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "merged_model")
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged model to {merged_dir}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
