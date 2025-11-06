"""Training script for PyTorch UNION model with LoRA (Low-Rank Adaptation).

This script extends train.py to support parameter-efficient fine-tuning using LoRA.
LoRA freezes the base model weights and trains small low-rank matrices,
significantly reducing memory usage and training time.

Requires: pip install peft
"""

import os
import sys
import time
from tqdm import tqdm
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_args
from models import create_model
from data import StoryDataset, CombinedDataset
from utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    load_model_weights,
    create_optimizer_and_scheduler,
    AverageMeter,
    format_time,
    compute_metrics,
)


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lora_args():
    """Parse command line arguments with LoRA-specific options."""
    parser = argparse.ArgumentParser(description="PyTorch UNION Training with LoRA")

    # Model arguments
    parser.add_argument("--model_type", type=str, default="bert",
                        choices=["bert", "longformer"],
                        help="Type of model to use")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Pretrained model name or path")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--use_reconstruction", action="store_true",
                        help="Use reconstruction task")
    parser.add_argument("--reconstruction_weight", type=float, default=0.1,
                        help="Weight for reconstruction loss")
    parser.add_argument("--use_all_layers", action="store_true",
                        help="Use multi-layer pooling")

    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (dimension of low-rank matrices)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout probability")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=None,
                        help="Target modules for LoRA. If None, uses default for model type. "
                             "For BERT: ['query', 'value'], "
                             "For LED/Longformer: ['query', 'value']")
    parser.add_argument("--lora_modules_to_save", type=str, nargs="+",
                        default=None,
                        help="Additional modules to train besides LoRA (e.g., classifier head)")
    parser.add_argument("--merge_weights", action="store_true",
                        help="Merge LoRA weights with base model after training")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing training data (required for single dataset modes, optional for combined mode)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model and outputs")
    parser.add_argument("--dataset_mode", type=str, default="roc",
                        choices=["roc", "wp", "award", "combined"],
                        help="Dataset mode: ROCStories, WritingPrompts, Award-winning, or combined")
    parser.add_argument("--wp_data_dir", type=str, default=None,
                        help="Directory for WritingPrompts data (for combined mode)")
    parser.add_argument("--award_data_dir", type=str, default=None,
                        help="Directory for Award-winning data (for combined mode)")
    parser.add_argument("--wp_has_reconstruction", action="store_true", default=True,
                        help="Enable reconstruction for WritingPrompts in combined mode")
    parser.add_argument("--award_has_reconstruction", action="store_true", default=True,
                        help="Enable reconstruction for Award-winning in combined mode")
    parser.add_argument("--train_data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (e.g., 0.1 for 10%%, 1.0 for all data)")
    parser.add_argument("--lazy_loading", action="store_true",
                        help="Use lazy loading: tokenize data on-the-fly instead of pre-tokenizing")

    # Training arguments
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["train", "pred"],
                        help="Task: train or predict")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16,
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
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory usage")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use efficient attention (xFormers or Flash Attention 2). Automatically detects which is available.")
    parser.add_argument("--compile_model", action="store_true",
                        help="Compile model with torch.compile() for PyTorch 2.0+ (can provide 20-30%% speedup)")

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every X steps")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=3,
                        help="Keep only the last N checkpoints to save disk space (0 = keep all)")

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

    # Multi-GPU training (DDP is automatically enabled when launched with torchrun)
    # Use: torchrun --nproc_per_node=NUM_GPUS train_lora.py [args]
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (automatically set by torchrun)")

    return parser.parse_args()


def get_default_lora_target_modules(model_type):
    """Get default LoRA target modules based on model type."""
    if model_type == "bert":
        # BERT uses 'query' and 'value' in attention layers
        return ["query", "value"]
    elif model_type == "longformer":
        # LED encoder uses 'query' and 'value' (nested in longformer_self_attn)
        # Module paths: encoder.layers.X.self_attn.longformer_self_attn.query/value
        return ["query", "value"]
    else:
        # Default fallback
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

    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {args.lora_r}")
    print(f"  Alpha: {args.lora_alpha}")
    print(f"  Dropout: {args.lora_dropout}")
    print(f"  Target modules: {target_modules}")
    if args.lora_modules_to_save:
        print(f"  Additional trainable modules: {args.lora_modules_to_save}")

    # Create LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Sequence classification task
        modules_to_save=args.lora_modules_to_save,  # Additional modules to train
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
):
    """Save LoRA adapter checkpoint."""
    checkpoint_dir = os.path.join(output_dir, f"{prefix}-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Handle DDP/DataParallel wrapper - extract the underlying model
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save LoRA adapter weights (much smaller than full model)
    model_to_save.save_pretrained(checkpoint_dir)

    # Save optimizer and scheduler states
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": global_step,
    }, os.path.join(checkpoint_dir, "training_state.pt"))

    print(f"Saved LoRA checkpoint to {checkpoint_dir}")

    # Clean up old checkpoints if requested
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
    is_main_process=True,
):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    rec_loss_meter = AverageMeter()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    # Track time breakdown for performance debugging
    data_time = 0
    forward_time = 0
    backward_time = 0
    step_start = time.time()

    for step, batch in enumerate(progress_bar):
        # Track data loading time
        data_time += time.time() - step_start

        # Move batch to device
        forward_start = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass with optional mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs["loss"]
        else:
            outputs = model(**batch)
            loss = outputs["loss"]

        # Track forward pass time
        forward_time += time.time() - forward_start

        # Backward pass with gradient accumulation
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
        if "classification_loss" in outputs:
            cls_loss_meter.update(outputs["classification_loss"].item())
        if "reconstruction_loss" in outputs:
            rec_loss_meter.update(outputs["reconstruction_loss"].item())

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

            # Logging (only on main process)
            if global_step % args.logging_steps == 0 and writer is not None:
                writer.add_scalar("train/loss", loss_meter.avg, global_step)
                writer.add_scalar("train/cls_loss", cls_loss_meter.avg, global_step)
                if args.use_reconstruction:
                    writer.add_scalar("train/rec_loss", rec_loss_meter.avg, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                # Log timing breakdown
                total_time = data_time + forward_time + backward_time
                if total_time > 0:
                    writer.add_scalar("perf/data_time_pct", 100 * data_time / total_time, global_step)
                    writer.add_scalar("perf/forward_time_pct", 100 * forward_time / total_time, global_step)
                    writer.add_scalar("perf/backward_time_pct", 100 * backward_time / total_time, global_step)

                progress_bar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "cls_loss": f"{cls_loss_meter.avg:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "data%": f"{100*data_time/total_time:.0f}" if total_time > 0 else "0",
                })

            # Save checkpoint (only on main process)
            if is_main_process and global_step % args.save_steps == 0:
                save_lora_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    args.output_dir,
                    prefix="checkpoint",
                    keep_last_n=args.keep_last_n_checkpoints,
                )

            # Evaluate during training
            if args.eval_steps > 0 and global_step % args.eval_steps == 0 and eval_dataloader is not None:
                print(f"\n{'='*80}")
                print(f"Evaluating at step {global_step}...")
                print('='*80)

                eval_metrics = evaluate(model, eval_dataloader, device)

                print(f"Validation metrics (step {global_step}):")
                print(f"  Loss: {eval_metrics['loss']:.4f}")
                print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
                print(f"  Precision: {eval_metrics['precision']:.4f}")
                print(f"  Recall: {eval_metrics['recall']:.4f}")
                print(f"  F1: {eval_metrics['f1']:.4f}")

                # Log to tensorboard (only on main process)
                if writer is not None:
                    for key, value in eval_metrics.items():
                        writer.add_scalar(f"eval/{key}", value, global_step)

                # Save best model during training (only on main process)
                if eval_metrics["f1"] > best_f1:
                    best_f1 = eval_metrics["f1"]
                    if is_main_process:
                        print(f"  New best F1: {best_f1:.4f} - Saving checkpoint...")
                        save_lora_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            global_step,
                            args.output_dir,
                            prefix="best",
                        keep_last_n=0,  # Keep all best checkpoints
                    )

                print('='*80 + '\n')

                # Set model back to training mode
                model.train()

        # Reset timer for next iteration
        step_start = time.time()

    return global_step, best_f1


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()

    all_preds = []
    all_labels = []
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            loss = outputs["loss"]
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


def create_dataset(args, tokenizer, mode):
    """Create dataset based on dataset_mode configuration."""
    # Only apply data fraction to training set, not validation/test
    data_fraction = args.train_data_fraction if mode == "train" else 1.0

    if args.dataset_mode == "combined":
        # Create combined dataset from multiple sources
        datasets = []

        # Add Award-winning dataset if directory is provided
        if args.award_data_dir:
            print(f"  Loading Award-winning {mode} data from {args.award_data_dir}...")
            award_dataset = StoryDataset(
                data_dir=args.award_data_dir,
                tokenizer=tokenizer,
                mode=mode,
                dataset_type="award",
                max_seq_length=args.max_seq_length,
                use_reconstruction=args.award_has_reconstruction,
                lazy_loading=args.lazy_loading,
            )
            datasets.append(award_dataset)
            print(f"    Award-winning: {len(award_dataset)} examples (reconstruction: {args.award_has_reconstruction})")

        # Add WritingPrompts dataset if directory is provided
        if args.wp_data_dir:
            print(f"  Loading WritingPrompts {mode} data from {args.wp_data_dir}...")
            wp_dataset = StoryDataset(
                data_dir=args.wp_data_dir,
                tokenizer=tokenizer,
                mode=mode,
                dataset_type="wp",
                max_seq_length=args.max_seq_length,
                use_reconstruction=args.wp_has_reconstruction,
                data_fraction=data_fraction,
                lazy_loading=args.lazy_loading,
            )
            datasets.append(wp_dataset)
            print(f"    WritingPrompts: {len(wp_dataset)} examples (reconstruction: {args.wp_has_reconstruction})")

        if not datasets:
            raise ValueError(
                "Combined mode requires at least one dataset. "
                "Please provide --award_data_dir and/or --wp_data_dir"
            )

        # Combine datasets
        combined_dataset = CombinedDataset(datasets)
        print(f"  Combined total: {len(combined_dataset)} examples")
        return combined_dataset

    else:
        # Single dataset mode (roc, wp, or award)
        return StoryDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            mode=mode,
            dataset_type=args.dataset_mode,
            max_seq_length=args.max_seq_length,
            use_reconstruction=args.use_reconstruction,
            data_fraction=data_fraction,
            lazy_loading=args.lazy_loading,
        )


def main():
    """Main training function with LoRA."""
    args = get_lora_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    # Validate dataset directories based on mode
    if args.dataset_mode == "combined":
        if not args.award_data_dir and not args.wp_data_dir:
            raise ValueError(
                "Combined mode requires at least one dataset. "
                "Please provide --award_data_dir and/or --wp_data_dir"
            )
    else:
        if not args.data_dir:
            raise ValueError(
                f"Dataset mode '{args.dataset_mode}' requires --data_dir argument"
            )

    # Validate model type
    valid_model_types = ["bert", "longformer"]
    if args.model_type not in valid_model_types:
        raise ValueError(
            f"Invalid model_type '{args.model_type}'. "
            f"Must be one of: {', '.join(valid_model_types)}"
        )

    # Validate train_data_fraction
    if not 0.0 < args.train_data_fraction <= 1.0:
        raise ValueError(
            f"Invalid train_data_fraction '{args.train_data_fraction}'. "
            f"Must be between 0.0 (exclusive) and 1.0 (inclusive)"
        )

    # Set seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device (use local_rank for DDP)
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = get_device(args.device)

    if is_main_process:
        print(f"Using device: {device}")
        if world_size > 1:
            print(f"Distributed training with {world_size} GPUs (rank {rank}/{world_size-1})")

    # Auto-set model_name if using default bert-base-uncased with longformer type
    if args.model_type == "longformer" and args.model_name == "bert-base-uncased":
        args.model_name = "allenai/led-base-16384"
        print(f"Auto-setting model_name to {args.model_name} for model_type={args.model_type}")
        print(f"Note: Using LED (Long Encoder-Decoder) encoder with 16384 max positions")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load datasets
    print("Loading training data...")
    train_dataset = create_dataset(args, tokenizer, mode="train")

    print("Loading validation data...")
    eval_dataset = create_dataset(args, tokenizer, mode="dev")

    # Create dataloaders with DistributedSampler for multi-GPU
    # Use more workers for faster data loading (8-16 recommended for modern systems)
    # Use persistent_workers to avoid recreating workers each epoch
    num_workers = 32 if not args.lazy_loading else 32  # Fewer workers for lazy loading

    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        eval_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,  # Prefetch 2 batches per worker
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
    )

    # Create base model
    # Note: Gradient checkpointing should be disabled for LoRA since base model is frozen
    print(f"Creating base model: {args.model_type} - {args.model_name}")
    if args.gradient_checkpointing:
        print("Warning: Gradient checkpointing is not recommended with LoRA (base model is frozen)")
        print("Disabling gradient checkpointing...")
    base_model = create_model(
        model_type=args.model_type,
        model_name=args.model_name,
        use_all_layers=args.use_all_layers,
        use_reconstruction=args.use_reconstruction,
        reconstruction_weight=args.reconstruction_weight,
        gradient_checkpointing=False,  # Disable for LoRA
    )

    # Load initial weights if provided (before wrapping with LoRA)
    if args.init_checkpoint:
        print(f"\nLoading initial model weights from: {args.init_checkpoint}")
        load_model_weights(base_model, args.init_checkpoint, device)
        print("Loaded weights into base model\n")

    base_model.to(device)

    # Enable efficient attention if requested (xFormers or Flash Attention)
    if args.use_flash_attention:
        attention_enabled = False

        # Try xFormers first (easier installation, widely compatible)
        try:
            import xformers
            if hasattr(base_model, 'enable_xformers_memory_efficient_attention'):
                base_model.enable_xformers_memory_efficient_attention()
                print(f"✓ xFormers memory-efficient attention enabled (version {xformers.__version__})")
                attention_enabled = True
            elif hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'enable_xformers_memory_efficient_attention'):
                base_model.encoder.enable_xformers_memory_efficient_attention()
                print(f"✓ xFormers memory-efficient attention enabled for encoder (version {xformers.__version__})")
                attention_enabled = True
        except ImportError:
            print("ℹ️  xFormers not found, trying Flash Attention 2...")
        except Exception as e:
            print(f"⚠️  Could not enable xFormers: {e}")

        # Fallback to Flash Attention 2 if xFormers not available
        if not attention_enabled:
            try:
                import flash_attn
                if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'config'):
                    base_model.encoder.config._attn_implementation = "flash_attention_2"
                    print(f"✓ Flash Attention 2 enabled for encoder (version {flash_attn.__version__})")
                    attention_enabled = True
                else:
                    print("⚠️  Model structure not compatible with Flash Attention 2")
            except ImportError:
                print("⚠️  Flash Attention 2 not found")
            except Exception as e:
                print(f"⚠️  Could not enable Flash Attention 2: {e}")

        if not attention_enabled:
            print("\n❌ ERROR: --use_flash_attention specified but no efficient attention library found!")
            print("\nPlease install one of the following:")
            print("  RECOMMENDED: pip install xformers")
            print("  ALTERNATIVE: pip install flash-attn --no-build-isolation")
            print("\nContinuing without efficient attention (training will be slower)...")

    # Wrap with LoRA
    print("\nWrapping model with LoRA adapters...")
    model = create_lora_model(base_model, args, device)

    # Compile model for PyTorch 2.0+ (can provide significant speedup)
    # Note: Compilation happens BEFORE DDP wrapping
    if args.compile_model:
        try:
            torch._dynamo.config.suppress_errors = True
            if is_main_process:
                print("\nCompiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            if is_main_process:
                print("Model compilation successful (first few steps will be slower during compilation)")
        except Exception as e:
            if is_main_process:
                print(f"Warning: Could not compile model - {e}")
                print("torch.compile() requires PyTorch 2.0+")

    # Multi-GPU training with DistributedDataParallel
    if world_size > 1:
        if is_main_process:
            print(f"\nUsing {world_size} GPUs with DistributedDataParallel!")
            print(f"GPU devices: {[torch.cuda.get_device_name(i) for i in range(world_size)]}")
            print(f"Per-GPU batch size: {args.train_batch_size}")
            print(f"Effective global batch size: {args.train_batch_size * world_size}")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * args.num_train_epochs

    # Create optimizer and scheduler (only for trainable parameters)
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

    if args.resume_from_checkpoint:
        print(f"\nResuming training from LoRA checkpoint: {args.resume_from_checkpoint}")

        # Load LoRA adapter weights
        model = PeftModel.from_pretrained(base_model, args.resume_from_checkpoint)
        model.to(device)

        # Compile if requested (before DDP wrapping)
        if args.compile_model:
            try:
                torch._dynamo.config.suppress_errors = True
                if is_main_process:
                    print("Compiling resumed model with torch.compile()...")
                model = torch.compile(model, mode="reduce-overhead")
                if is_main_process:
                    print("Model compilation successful")
            except Exception as e:
                if is_main_process:
                    print(f"Warning: Could not compile resumed model - {e}")

        # Wrap with DDP if using multi-GPU
        if world_size > 1:
            if is_main_process:
                print(f"Re-wrapping resumed model with DDP for {world_size} GPUs")
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        # Load training state
        training_state_path = os.path.join(args.resume_from_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
            start_epoch = training_state.get("epoch", 0)
            global_step = training_state.get("step", 0)
            print(f"Continuing from epoch {start_epoch}, step {global_step}\n")
        else:
            print(f"Warning: training_state.pt not found, starting fresh optimizer/scheduler\n")

    # Setup mixed precision training
    scaler = None
    if args.fp16:
        if device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision (FP16) training enabled")
        else:
            print(f"Warning: --fp16 specified but device is {device.type}. FP16 training disabled.")

    # Setup tensorboard (only on main process)
    if is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    else:
        writer = None

    # Print training info
    print("\n" + "=" * 80)
    print("LoRA Training Configuration:")
    print("=" * 80)
    print(f"Model: {args.model_type} - {args.model_name}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"Dataset mode: {args.dataset_mode}")
    if args.dataset_mode == "combined":
        print(f"  - Award-winning data: {args.award_data_dir or 'None'} (reconstruction: {args.award_has_reconstruction})")
        print(f"  - WritingPrompts data: {args.wp_data_dir or 'None'} (reconstruction: {args.wp_has_reconstruction})")
    else:
        print(f"  - Data directory: {args.data_dir}")
        print(f"  - Use reconstruction: {args.use_reconstruction}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Training data fraction: {args.train_data_fraction*100:.1f}%")
    print(f"Lazy loading: {args.lazy_loading}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Total training steps: {num_training_steps}")
    print(f"Reconstruction weight: {args.reconstruction_weight}")
    print(f"Use all layers: {args.use_all_layers}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"Mixed precision (FP16): {args.fp16 and device.type == 'cuda'}")
    print(f"Device: {device}")
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Multi-GPU: Yes (DataParallel with {torch.cuda.device_count()} GPUs)")
        print(f"Per-GPU batch size: {args.train_batch_size // torch.cuda.device_count()}")
    else:
        print(f"Multi-GPU: No")
    print("=" * 80 + "\n")

    # Training loop
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(start_epoch, args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        print("-" * 80)

        # Set epoch for distributed sampler (for proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train (with optional in-training evaluation)
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
            is_main_process=is_main_process,
        )

        # Evaluate at end of epoch
        print(f"\n{'='*80}")
        print(f"End of Epoch {epoch + 1} - Evaluating...")
        print('='*80)
        eval_metrics = evaluate(model, eval_dataloader, device)

        print(f"Validation metrics (end of epoch {epoch + 1}):")
        print(f"  Loss: {eval_metrics['loss']:.4f}")
        print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Precision: {eval_metrics['precision']:.4f}")
        print(f"  Recall: {eval_metrics['recall']:.4f}")
        print(f"  F1: {eval_metrics['f1']:.4f}")

        # Log to tensorboard (only on main process)
        if writer is not None:
            for key, value in eval_metrics.items():
                writer.add_scalar(f"eval/{key}", value, global_step)

        # Save best model (only on main process)
        if eval_metrics["f1"] > best_f1:
            best_f1 = eval_metrics["f1"]
            if is_main_process:
                print(f"  New best F1: {best_f1:.4f} - Saving checkpoint...")
                save_lora_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    args.output_dir,
                    prefix="best",
                    keep_last_n=0,  # Keep all best checkpoints
                )

        print('='*80)

        # Save epoch checkpoint (only on main process)
        if is_main_process:
            save_lora_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                args.output_dir,
                prefix="epoch",
            keep_last_n=2,  # Keep last 2 epoch checkpoints
        )

    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"Total time: {format_time(elapsed_time)}")
    print(f"Best F1: {best_f1:.4f}")
    print("=" * 80)

    # Optionally merge and save full model (only on main process)
    if args.merge_weights and is_main_process:
        print("\nMerging LoRA weights with base model...")
        # Extract underlying model if wrapped with DDP/DataParallel
        model_to_merge = model.module if hasattr(model, 'module') else model
        merged_model = model_to_merge.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "merged_model")
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged model to {merged_dir}")

    if writer is not None:
        writer.close()

    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()
