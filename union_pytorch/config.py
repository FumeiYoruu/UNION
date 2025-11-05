"""Configuration for PyTorch UNION model."""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""

    # Model architecture
    model_type: str = "bert"  # "bert" or "longformer" (uses LED encoder)
    model_name: str = "bert-base-uncased"  # or "allenai/led-base-16384"
    max_seq_length: int = 512  # 512 for BERT, up to 16384 for LED
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Task settings
    use_reconstruction: bool = False
    reconstruction_weight: float = 0.1
    num_labels: int = 2

    # Multi-layer pooling
    use_all_layers: bool = False
    layers_to_use: Optional[list] = None  # None = use all layers


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    data_dir: str = "./Data/ROCStories"
    output_dir: str = "./union_pytorch/output"
    dataset_mode: str = "roc"  # "roc", "wp", or "award"

    # Training hyperparameters
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Optimization
    gradient_accumulation_steps: int = 1
    fp16: bool = False

    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000

    # Device
    device: str = "cuda"  # "cuda", "cpu", or "mps"
    seed: int = 42

    # Checkpoint
    init_checkpoint: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PyTorch UNION Training")

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
                        help="Enable reconstruction for WritingPrompts in combined mode (default: True)")
    parser.add_argument("--award_has_reconstruction", action="store_true", default=True,
                        help="Enable reconstruction for Award-winning in combined mode; uses ref_map files to determine which samples have references (default: True)")

    # Training arguments
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["train", "pred"],
                        help="Task: train or predict")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
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

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps (default: 500, recommended: 250-500 for cloud training)")
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
                        help="Initial checkpoint to load")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint")

    return parser.parse_args()
