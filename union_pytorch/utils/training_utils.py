"""Training utilities."""

import os
import random
import numpy as np
import torch
from typing import Dict
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "cuda") -> torch.device:
    """Get torch device."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    output_dir: str,
    prefix: str = "checkpoint",
    keep_last_n: int = 3,
):
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        epoch: Current epoch
        step: Current step
        output_dir: Output directory
        prefix: Checkpoint prefix (e.g., "checkpoint", "best")
        keep_last_n: Keep only the last N checkpoints with this prefix (0 = keep all)
    """
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, f"{prefix}-epoch{epoch}-step{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)

    # Save optimizer and scheduler
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(scheduler.state_dict(), scheduler_path)

    # Save training state
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    torch.save({"epoch": epoch, "step": step}, state_path)

    print(f"✓ Checkpoint saved to {checkpoint_dir}")

    # Cleanup old checkpoints (keep only last N)
    if keep_last_n > 0 and prefix == "checkpoint":  # Only auto-cleanup regular checkpoints
        cleanup_old_checkpoints(output_dir, prefix, keep_last_n)

    return checkpoint_dir


def cleanup_old_checkpoints(output_dir: str, prefix: str, keep_last_n: int):
    """Remove old checkpoints, keeping only the last N."""
    import glob

    # Find all checkpoints with this prefix
    pattern = os.path.join(output_dir, f"{prefix}-epoch*-step*")
    checkpoints = glob.glob(pattern)

    if len(checkpoints) <= keep_last_n:
        return

    # Sort by step number (extract from dirname)
    def get_step(path):
        try:
            # Extract step from "checkpoint-epoch3-step1000" format
            step_str = path.split("-step")[-1]
            return int(step_str)
        except:
            return 0

    checkpoints.sort(key=get_step)

    # Remove oldest checkpoints
    for old_checkpoint in checkpoints[:-keep_last_n]:
        try:
            import shutil
            shutil.rmtree(old_checkpoint)
            print(f"  Removed old checkpoint: {os.path.basename(old_checkpoint)}")
        except Exception as e:
            print(f"  Warning: Could not remove {old_checkpoint}: {e}")


def load_model_weights(model, checkpoint_path: str, device=None):
    """Load only model weights (no optimizer/scheduler state).

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint directory or pytorch_model.bin file
        device: Device to load on

    Use this for --init_checkpoint to start training from a pretrained model.
    """
    # Handle both directory and file paths
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    else:
        model_path = checkpoint_path

    if not os.path.exists(model_path):
        raise ValueError(f"Model checkpoint not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Model weights loaded from {model_path}")


def load_checkpoint(
    model,
    checkpoint_dir: str,
    optimizer=None,
    scheduler=None,
    device=None,
):
    """Load full checkpoint including optimizer and scheduler states.

    Use this for --resume_from_checkpoint to continue interrupted training.
    """
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise ValueError(f"Model checkpoint not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Model loaded from {model_path}")

    # Load optimizer and scheduler if provided
    if optimizer is not None:
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            print(f"✓ Optimizer loaded from {optimizer_path}")

    if scheduler is not None:
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
            print(f"✓ Scheduler loaded from {scheduler_path}")

    # Load training state
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    training_state = {}
    if os.path.exists(state_path):
        training_state = torch.load(state_path, map_location=device)
        print(f"✓ Training state loaded: epoch={training_state.get('epoch')}, step={training_state.get('step')}")

    return training_state


def create_optimizer_and_scheduler(
    model,
    learning_rate: float,
    num_training_steps: int,
    warmup_steps: int,
    weight_decay: float = 0.01,
):
    """Create optimizer and learning rate scheduler."""
    # Prepare optimizer parameters with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_metrics(preds, labels) -> Dict[str, float]:
    """Compute evaluation metrics."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Convert to numpy if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get predicted labels
    pred_labels = np.argmax(preds, axis=1)

    # Compute metrics
    accuracy = accuracy_score(labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_labels, average="binary"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
