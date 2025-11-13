"""Direct fine-tuning of LongformerForSequenceClassification with LoRA.

This script uses the standard Hugging Face LongformerForSequenceClassification model
directly, rather than the custom UnionClassifier. This is simpler and follows the
standard Hugging Face fine-tuning approach.

Advantages:
- Uses the official LongformerForSequenceClassification architecture
- Simpler code, fewer custom components
- Standard Hugging Face model interface
- Built-in CLS token pooling (as designed by the Longformer authors)

Requires: pip install peft transformers
"""

import os
import sys
import time
from tqdm import tqdm
import argparse
import warnings

# Suppress Longformer attention window padding warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*automatically padded.*")
warnings.filterwarnings("ignore", message=".*attention.*window.*")

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    LongformerConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import (
    StoryDataset,
    CombinedDataset,
    DataCollatorWithDynamicPadding,
    DataCollatorWithFixedBuckets,
    MultiDataLoaderIterator,
)
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
        description="Direct fine-tuning of LongformerForSequenceClassification with LoRA"
    )

    # Model arguments
    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096",
                        help="Pretrained Longformer model name")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of classification labels (default: 2 for binary)")

    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=None,
                        help="Target modules for LoRA (default: ['query', 'value'])")
    parser.add_argument("--merge_weights", action="store_true",
                        help="Merge LoRA weights after training")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing training data (required for single dataset modes)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model and outputs")
    parser.add_argument("--dataset_mode", type=str, default="wp",
                        choices=["roc", "wp", "award", "combined"],
                        help="Dataset mode: ROCStories, WritingPrompts, Award-winning, or combined")
    parser.add_argument("--wp_data_dir", type=str, default=None,
                        help="Directory for WritingPrompts data (for combined mode)")
    parser.add_argument("--award_data_dir", type=str, default=None,
                        help="Directory for Award-winning data (for combined mode)")
    parser.add_argument("--train_data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use")
    parser.add_argument("--lazy_loading", action="store_true",
                        help="Use lazy loading")
    parser.add_argument("--padding_strategy", type=str, default="dynamic",
                        choices=["dynamic", "bucket", "fixed"],
                        help="Padding strategy")
    parser.add_argument("--padding_buckets", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384],
                        help="Bucket sizes for bucket padding")

    # Training arguments
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["train", "pred"],
                        help="Task: train or predict")
    parser.add_argument("--train_batch_size", type=int, default=2,
                        help="Training batch size (default for single dataset or all datasets)")
    parser.add_argument("--wp_batch_size", type=int, default=None,
                        help="WritingPrompts batch size for combined mode (default: uses --train_batch_size)")
    parser.add_argument("--award_batch_size", type=int, default=None,
                        help="Award-winning batch size for combined mode (default: uses --train_batch_size)")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use efficient attention")
    parser.add_argument("--compile_model", action="store_true",
                        help="Compile model with torch.compile()")

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every X steps")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=3,
                        help="Keep only last N checkpoints")

    # Device and seed
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"],
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Checkpoint
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Initial checkpoint to load")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint")

    return parser.parse_args()


def create_lora_model(base_model, args):
    """Wrap model with LoRA adapters."""
    # Default LoRA target modules for Longformer
    if args.lora_target_modules is None:
        target_modules = ["query", "value"]
    else:
        target_modules = args.lora_target_modules

    # Save the classifier head (randomly initialized for our task)
    modules_to_save = ["classifier"]

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


def save_checkpoint(
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
    """Save checkpoint."""
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

    print(f"Saved checkpoint to {checkpoint_dir}")

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

    # Handle resuming for MultiDataLoaderIterator
    if isinstance(dataloader, MultiDataLoaderIterator) and start_step > 0:
        # For MultiDataLoaderIterator, use skip_batches method before creating iterator
        dataloader.skip_batches(start_step)
        start_step = 0  # Reset since we've already skipped

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        # Skip batches if resuming (for regular dataloaders)
        if start_step > 0 and step < start_step:
            continue

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(**batch)
                    loss = outputs.loss
                    if loss.dim() > 0:
                        loss = loss.mean()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()

        # Backward pass
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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

            # Logging
            if global_step % args.logging_steps == 0 and writer is not None:
                writer.add_scalar("train/loss", loss_meter.avg, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.flush()

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # Save checkpoint
        if (step + 1) % args.gradient_accumulation_steps == 0 and global_step % args.save_steps == 0:
            print(f"\nSaving checkpoint at step {global_step}...")
            save_checkpoint(
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
            print(f"Evaluating at step {global_step}...")
            print('='*80)

            eval_metrics = evaluate(model, eval_dataloader, device, eval_fraction=0.01)

            print(f"Validation metrics (step {global_step}):")
            print(f"  Loss: {eval_metrics['loss']:.4f}")
            print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Precision: {eval_metrics['precision']:.4f}")
            print(f"  Recall: {eval_metrics['recall']:.4f}")
            print(f"  F1: {eval_metrics['f1']:.4f}")

            if writer is not None:
                for key, value in eval_metrics.items():
                    writer.add_scalar(f"eval/{key}", value, global_step)
                writer.flush()

            # Save best model
            if eval_metrics["f1"] > best_f1:
                best_f1 = eval_metrics["f1"]
                print(f"  New best F1: {best_f1:.4f} - Saving checkpoint...")
                save_checkpoint(
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

            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()

            logits = outputs.logits

            loss_meter.update(loss.item())

            all_preds.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = loss_meter.avg

    return metrics


def create_dataset(args, tokenizer, mode):
    """Create dataset based on dataset_mode configuration.

    Returns:
        For combined mode: Tuple of (datasets, dataset_names)
        For single mode: Single dataset
    """
    # Only apply data fraction to training set, not validation/test
    data_fraction = args.train_data_fraction if mode == "train" else 1.0

    if args.dataset_mode == "combined":
        # Create separate datasets for combined mode
        datasets = []
        dataset_names = []

        # Add Award-winning dataset if directory is provided
        if args.award_data_dir:
            print(f"  Loading Award-winning {mode} data from {args.award_data_dir}...")
            award_dataset = StoryDataset(
                data_dir=args.award_data_dir,
                tokenizer=tokenizer,
                mode=mode,
                dataset_type="award",
                max_seq_length=args.max_seq_length,
                use_reconstruction=False,  # Not supported by LongformerForSequenceClassification
                lazy_loading=args.lazy_loading,
            )
            datasets.append(award_dataset)
            dataset_names.append("Award")
            print(f"    Award-winning: {len(award_dataset)} examples")

        # Add WritingPrompts dataset if directory is provided
        if args.wp_data_dir:
            print(f"  Loading WritingPrompts {mode} data from {args.wp_data_dir}...")
            wp_dataset = StoryDataset(
                data_dir=args.wp_data_dir,
                tokenizer=tokenizer,
                mode=mode,
                dataset_type="wp",
                max_seq_length=args.max_seq_length,
                use_reconstruction=False,  # Not supported by LongformerForSequenceClassification
                data_fraction=data_fraction,
                lazy_loading=args.lazy_loading,
            )
            datasets.append(wp_dataset)
            dataset_names.append("WP")
            print(f"    WritingPrompts: {len(wp_dataset)} examples")

        if not datasets:
            raise ValueError(
                "Combined mode requires at least one dataset. "
                "Please provide --award_data_dir and/or --wp_data_dir"
            )

        total_examples = sum(len(d) for d in datasets)
        print(f"  Combined total: {total_examples} examples across {len(datasets)} datasets")
        return datasets, dataset_names

    else:
        # Single dataset mode (roc, wp, or award)
        return StoryDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            mode=mode,
            dataset_type=args.dataset_mode,
            max_seq_length=args.max_seq_length,
            use_reconstruction=False,  # Not supported by LongformerForSequenceClassification
            data_fraction=data_fraction,
            lazy_loading=args.lazy_loading,
        )


def main():
    """Main training function."""
    args = get_args()

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

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load datasets
    print("Loading training data...")
    train_data = create_dataset(args, tokenizer, mode="train")

    print("Loading validation data...")
    eval_data = create_dataset(args, tokenizer, mode="dev")

    # Setup padding collator
    collate_fn = None
    if args.padding_strategy == "dynamic":
        collate_fn = DataCollatorWithDynamicPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8
        )
        print("Using dynamic padding")
    elif args.padding_strategy == "bucket":
        collate_fn = DataCollatorWithFixedBuckets(
            tokenizer=tokenizer,
            buckets=args.padding_buckets
        )
        print(f"Using bucket padding: {args.padding_buckets}")

    # Create dataloaders
    num_workers = 32 if not args.lazy_loading else 32

    # Check if combined mode (returns tuple) or single mode (returns dataset)
    is_combined_mode = isinstance(train_data, tuple)

    if is_combined_mode:
        # Combined mode: create separate dataloaders for each dataset
        train_datasets, train_dataset_names = train_data
        eval_datasets, eval_dataset_names = eval_data

        # Determine batch sizes for each dataset
        train_batch_sizes = []
        for name in train_dataset_names:
            if name == "WP" and args.wp_batch_size is not None:
                train_batch_sizes.append(args.wp_batch_size)
            elif name == "Award" and args.award_batch_size is not None:
                train_batch_sizes.append(args.award_batch_size)
            else:
                train_batch_sizes.append(args.train_batch_size)

        # Create separate dataloaders with reproducible shuffling
        train_dataloaders = []
        for dataset, batch_size, name in zip(train_datasets, train_batch_sizes, train_dataset_names):
            # Create reproducible sampler with fixed seed
            generator = torch.Generator().manual_seed(args.seed)
            sampler = RandomSampler(dataset, generator=generator)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True if device.type == "cuda" else False,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2,
                collate_fn=collate_fn,
            )
            train_dataloaders.append(dataloader)
            print(f"  {name} train dataloader: {len(dataloader)} batches (batch_size={batch_size})")

        # Wrap in MultiDataLoaderIterator
        train_dataloader = MultiDataLoaderIterator(train_dataloaders, train_dataset_names)
        print(f"\nCombined training: {len(train_dataloader)} total batches across {len(train_dataloaders)} datasets")
        print(f"Using reproducible shuffle (seed={args.seed}) - same order across epochs and checkpoints")

        # For evaluation, use combined dataset
        eval_dataset = CombinedDataset(eval_datasets)
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

    else:
        # Single dataset mode
        train_dataset = train_data
        eval_dataset = eval_data

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

    # Create LongformerForSequenceClassification model
    print(f"\nCreating LongformerForSequenceClassification: {args.model_name}")
    print(f"Number of labels: {args.num_labels}")

    config = LongformerConfig.from_pretrained(args.model_name)
    config.num_labels = args.num_labels

    base_model = LongformerForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
    )

    print(f"\nModel architecture:")
    print(f"  - Uses LongformerModel encoder")
    print(f"  - Uses LongformerClassificationHead (CLS token pooling)")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")
    print(f"  - Attention window: {config.attention_window}")

    base_model.to(device)

    # Enable efficient attention if requested
    if args.use_flash_attention:
        attention_enabled = False
        try:
            import xformers
            if hasattr(base_model.longformer, 'enable_xformers_memory_efficient_attention'):
                base_model.longformer.enable_xformers_memory_efficient_attention()
                print(f"✓ xFormers enabled (version {xformers.__version__})")
                attention_enabled = True
        except ImportError:
            print("ℹ️  xFormers not found")
        except Exception as e:
            print(f"⚠️  Could not enable xFormers: {e}")

        if not attention_enabled:
            print("Install: pip install xformers")

    # Wrap with LoRA
    print("\nWrapping model with LoRA adapters...")
    model = create_lora_model(base_model, args)

    # Compile model if requested
    if args.compile_model:
        try:
            torch._dynamo.config.suppress_errors = True
            print("\nCompiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compilation successful")
        except Exception as e:
            print(f"Warning: Could not compile model - {e}")

    # Multi-GPU training
    if num_gpus > 1:
        print(f"\nUsing DataParallel with {num_gpus} GPUs")
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
        print(f"\nResuming from checkpoint: {args.resume_from_checkpoint}")

        model = PeftModel.from_pretrained(base_model, args.resume_from_checkpoint)
        model.to(device)

        if args.compile_model:
            try:
                torch._dynamo.config.suppress_errors = True
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                print(f"Warning: Could not compile - {e}")

        if num_gpus > 1:
            model = DataParallel(model)

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
            else:
                start_epoch = saved_epoch - 1

            print(f"Resuming from epoch {saved_epoch}, step {global_step}")

    # Setup mixed precision
    scaler = None
    if args.fp16 and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed precision (FP16) enabled")

    # Setup tensorboard
    tensorboard_log_dir = os.path.join(args.output_dir, "logs")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print(f"\nTensorBoard: tensorboard --logdir {tensorboard_log_dir}\n")

    # Print training info
    print("\n" + "=" * 80)
    print("LongformerForSequenceClassification Training with LoRA:")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Architecture: LongformerForSequenceClassification (official Hugging Face)")
    print(f"  - Encoder: LongformerModel")
    print(f"  - Head: LongformerClassificationHead (CLS token pooling)")
    print(f"Max sequence length: {args.max_seq_length}")

    # Print dataset sizes
    if is_combined_mode:
        total_train = sum(len(d) for d in train_datasets)
        print(f"Training examples: {total_train} total")
        for name, dataset in zip(train_dataset_names, train_datasets):
            print(f"  - {name}: {len(dataset)} examples")
    else:
        print(f"Training examples: {len(train_dataset)}")

    print(f"Validation examples: {len(eval_dataset)}")
    print(f"Epochs: {args.num_train_epochs}")

    # Print batch sizes
    if is_combined_mode:
        print(f"Batch sizes:")
        for name, batch_size in zip(train_dataset_names, train_batch_sizes):
            print(f"  - {name}: {batch_size}")
    else:
        print(f"Batch size: {args.train_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Device: {device}")
    print("=" * 80 + "\n")

    # Training loop
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(start_epoch, args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        print("-" * 80)

        current_start_batch = start_batch_step if epoch == start_epoch else 0

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
        print(f"End of Epoch {epoch + 1} - Evaluating...")
        print('='*80)
        eval_metrics = evaluate(model, eval_dataloader, device, eval_fraction=0.10)

        print(f"Validation metrics:")
        print(f"  Loss: {eval_metrics['loss']:.4f}")
        print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Precision: {eval_metrics['precision']:.4f}")
        print(f"  Recall: {eval_metrics['recall']:.4f}")
        print(f"  F1: {eval_metrics['f1']:.4f}")

        if writer is not None:
            for key, value in eval_metrics.items():
                writer.add_scalar(f"eval/{key}", value, global_step)
            writer.flush()

        if eval_metrics["f1"] > best_f1:
            best_f1 = eval_metrics["f1"]
            print(f"  New best F1: {best_f1:.4f}")
            save_checkpoint(
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

        save_checkpoint(
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

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"Total time: {format_time(elapsed_time)}")
    print(f"Best F1: {best_f1:.4f}")
    print("=" * 80)

    # Merge weights if requested
    if args.merge_weights:
        print("\nMerging LoRA weights...")
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
