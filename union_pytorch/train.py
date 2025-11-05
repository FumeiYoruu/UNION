"""Training script for PyTorch UNION model."""

import os
import sys
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

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
):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    rec_loss_meter = AverageMeter()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)

        loss = outputs["loss"]

        # Backward pass with gradient accumulation
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        # Update meters
        loss_meter.update(loss.item() * args.gradient_accumulation_steps)
        if "classification_loss" in outputs:
            cls_loss_meter.update(outputs["classification_loss"].item())
        if "reconstruction_loss" in outputs:
            rec_loss_meter.update(outputs["reconstruction_loss"].item())

        # Update weights
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if global_step % args.logging_steps == 0:
                writer.add_scalar("train/loss", loss_meter.avg, global_step)
                writer.add_scalar("train/cls_loss", cls_loss_meter.avg, global_step)
                if args.use_reconstruction:
                    writer.add_scalar("train/rec_loss", rec_loss_meter.avg, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                progress_bar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "cls_loss": f"{cls_loss_meter.avg:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    args.output_dir,
                    prefix="checkpoint",
                    keep_last_n=args.keep_last_n_checkpoints,
                )

    return global_step


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
    """Create dataset based on dataset_mode configuration.

    Args:
        args: Command line arguments
        tokenizer: Tokenizer instance
        mode: "train", "dev", or "test"

    Returns:
        Dataset instance (StoryDataset or CombinedDataset)
    """
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
        )


def main():
    """Main training function."""
    args = get_args()

    # Validate dataset directories based on mode
    if args.dataset_mode == "combined":
        # Combined mode requires at least one of the dataset directories
        if not args.award_data_dir and not args.wp_data_dir:
            raise ValueError(
                "Combined mode requires at least one dataset. "
                "Please provide --award_data_dir and/or --wp_data_dir"
            )
    else:
        # Single dataset mode requires --data_dir
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

    # Set seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Auto-set model_name if using default bert-base-uncased with longformer type
    if args.model_type == "longformer" and args.model_name == "bert-base-uncased":
        args.model_name = "allenai/led-base-16384"
        print(f"Auto-setting model_name to {args.model_name} for model_type={args.model_type}")
        print(f"Note: Using LED (Long Encoder-Decoder) encoder with {16384} max positions")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load datasets
    print("Loading training data...")
    train_dataset = create_dataset(args, tokenizer, mode="train")

    print("Loading validation data...")
    eval_dataset = create_dataset(args, tokenizer, mode="dev")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    print(f"Creating model: {args.model_type} - {args.model_name}")
    model = create_model(
        model_type=args.model_type,
        model_name=args.model_name,
        use_all_layers=args.use_all_layers,
        use_reconstruction=args.use_reconstruction,
        reconstruction_weight=args.reconstruction_weight,
    )

    model.to(device)

    # Load initial weights if provided (before creating optimizer)
    if args.init_checkpoint:
        print(f"\nLoading initial model weights from: {args.init_checkpoint}")
        load_model_weights(model, args.init_checkpoint, device)
        print("Starting fresh training from loaded weights (no optimizer/scheduler state)\n")

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

    # Load from checkpoint if resuming (loads model, optimizer, scheduler, and training state)
    start_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint:
        print(f"\nResuming training from checkpoint: {args.resume_from_checkpoint}")
        training_state = load_checkpoint(
            model,
            args.resume_from_checkpoint,
            optimizer,
            scheduler,
            device,
        )
        start_epoch = training_state.get("epoch", 0)
        global_step = training_state.get("step", 0)
        print(f"Continuing from epoch {start_epoch}, step {global_step}\n")

    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Print training info
    print("\n" + "=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(f"Model: {args.model_type} - {args.model_name}")
    print(f"Dataset mode: {args.dataset_mode}")
    if args.dataset_mode == "combined":
        print(f"  - Award-winning data: {args.award_data_dir or 'None'} (reconstruction: {args.award_has_reconstruction})")
        print(f"  - WritingPrompts data: {args.wp_data_dir or 'None'} (reconstruction: {args.wp_has_reconstruction})")
    else:
        print(f"  - Data directory: {args.data_dir}")
        print(f"  - Use reconstruction: {args.use_reconstruction}")
    print(f"Max sequence length: {args.max_seq_length}")
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
    print(f"Device: {device}")
    print("=" * 80 + "\n")

    # Training loop
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(start_epoch, args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        print("-" * 80)

        # Train
        global_step = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            epoch + 1,
            global_step,
            args,
            writer,
        )

        # Evaluate
        print("\nEvaluating...")
        eval_metrics = evaluate(model, eval_dataloader, device)

        print(f"Validation metrics:")
        print(f"  Loss: {eval_metrics['loss']:.4f}")
        print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Precision: {eval_metrics['precision']:.4f}")
        print(f"  Recall: {eval_metrics['recall']:.4f}")
        print(f"  F1: {eval_metrics['f1']:.4f}")

        # Log to tensorboard
        for key, value in eval_metrics.items():
            writer.add_scalar(f"eval/{key}", value, global_step)

        # Save best model
        if eval_metrics["f1"] > best_f1:
            best_f1 = eval_metrics["f1"]
            print(f"New best F1: {best_f1:.4f}")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                args.output_dir,
                prefix="best",
                keep_last_n=0,  # Keep all best checkpoints
            )

        # Save epoch checkpoint
        save_checkpoint(
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

    writer.close()


if __name__ == "__main__":
    main()
