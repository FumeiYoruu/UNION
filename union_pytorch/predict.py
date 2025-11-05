"""Prediction script for PyTorch UNION model."""

import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_args
from models import create_model
from data import PredictionDataset
from utils import set_seed, get_device, load_checkpoint


def predict(model, dataloader, device):
    """Run prediction on dataset."""
    model.eval()

    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs["logits"]

            # Get probability of label=1 (human-written)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu())

    # Concatenate all predictions
    all_probs = torch.cat(all_probs, dim=0).numpy()

    return all_probs


def compute_correlations(predictions, human_scores):
    """Compute correlation metrics."""
    pearson_corr, pearson_p = pearsonr(predictions, human_scores)
    spearman_corr, spearman_p = spearmanr(predictions, human_scores)
    kendall_corr, kendall_p = kendalltau(predictions, human_scores)

    return {
        "pearson": pearson_corr,
        "pearson_p": pearson_p,
        "spearman": spearman_corr,
        "spearman_p": spearman_p,
        "kendall": kendall_corr,
        "kendall_p": kendall_p,
    }


def main():
    """Main prediction function."""
    args = get_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load prediction dataset
    print("Loading prediction data...")
    pred_dataset = PredictionDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    pred_dataloader = DataLoader(
        pred_dataset,
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
        use_reconstruction=False,  # No reconstruction for prediction
    )

    model.to(device)

    # Load checkpoint
    if args.init_checkpoint:
        print(f"Loading checkpoint: {args.init_checkpoint}")
        load_checkpoint(model, args.init_checkpoint, device=device)
    else:
        print("Warning: No checkpoint specified, using pretrained weights only")

    # Run prediction
    print("\nRunning predictions...")
    predictions = predict(model, pred_dataloader, device)

    # Get human scores
    human_scores = pred_dataset.get_human_scores()

    # Compute correlations
    print("\nComputing correlations with human judgments...")
    correlations = compute_correlations(predictions, human_scores)

    print("\n" + "=" * 80)
    print("Correlation Results:")
    print("=" * 80)
    print(f"Pearson correlation:  {correlations['pearson']:.4f} (p={correlations['pearson_p']:.4e})")
    print(f"Spearman correlation: {correlations['spearman']:.4f} (p={correlations['spearman_p']:.4e})")
    print(f"Kendall correlation:  {correlations['kendall']:.4f} (p={correlations['kendall_p']:.4e})")
    print("=" * 80)

    # Save predictions
    output_file = os.path.join(args.output_dir, "predictions.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("story_id\tunion_score\thuman_score\n")
        for i, (pred, human) in enumerate(zip(predictions, human_scores)):
            f.write(f"{i}\t{pred:.6f}\t{human:.6f}\n")

    print(f"\nPredictions saved to: {output_file}")

    # Save correlation results
    corr_file = os.path.join(args.output_dir, "correlations.txt")
    with open(corr_file, "w") as f:
        f.write("Correlation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Pearson:  {correlations['pearson']:.4f} (p={correlations['pearson_p']:.4e})\n")
        f.write(f"Spearman: {correlations['spearman']:.4f} (p={correlations['spearman_p']:.4e})\n")
        f.write(f"Kendall:  {correlations['kendall']:.4f} (p={correlations['kendall_p']:.4e})\n")

    print(f"Correlation results saved to: {corr_file}")

    # Print statistics
    print("\n" + "=" * 80)
    print("Prediction Statistics:")
    print("=" * 80)
    print(f"Number of stories: {len(predictions)}")
    print(f"UNION scores:")
    print(f"  Mean:   {np.mean(predictions):.4f}")
    print(f"  Std:    {np.std(predictions):.4f}")
    print(f"  Min:    {np.min(predictions):.4f}")
    print(f"  Max:    {np.max(predictions):.4f}")
    print(f"Human scores:")
    print(f"  Mean:   {np.mean(human_scores):.4f}")
    print(f"  Std:    {np.std(human_scores):.4f}")
    print(f"  Min:    {np.min(human_scores):.4f}")
    print(f"  Max:    {np.max(human_scores):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
