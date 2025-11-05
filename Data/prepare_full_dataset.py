"""
End-to-end script to prepare the full dataset for UNION training.

This script orchestrates the complete pipeline:
1. Download WritingPrompts dataset
2. Generate vocabulary
3. Generate negative samples
4. Optionally combine with Award-winning dataset
5. Prepare for training

Usage:
    # Download and prepare WritingPrompts only
    python prepare_full_dataset.py --dataset wp

    # Download and prepare WritingPrompts + Award-winning
    python prepare_full_dataset.py --dataset combined

    # Skip download if data already exists
    python prepare_full_dataset.py --dataset wp --skip-download
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """Run a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=cwd,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed!")
        print(f"Error: {e}")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"  ✓ {description}: {filepath}")
        return True
    else:
        print(f"  ✗ {description} NOT FOUND: {filepath}")
        return False


def prepare_writing_prompts(skip_download=False):
    """Prepare WritingPrompts dataset."""
    print("\n" + "="*70)
    print("PREPARING WRITINGPROMPTS DATASET")
    print("="*70 + "\n")

    data_dir = Path("./WritingPrompts")
    ini_data_dir = data_dir / "ini_data"

    # Step 1: Download dataset
    if not skip_download:
        success = run_command(
            "python download_writing_prompts.py --source huggingface --output_dir ./WritingPrompts",
            "Step 1: Downloading WritingPrompts dataset"
        )
        if not success:
            print("\nTrying alternative download method...")
            success = run_command(
                "python download_writing_prompts.py --source manual",
                "Alternative: Manual download instructions"
            )
            if not success:
                return False
    else:
        print("Skipping download (--skip-download flag set)")

    # Verify download
    print("\nVerifying downloaded files...")
    required_files = [
        ini_data_dir / "train.wp_source",
        ini_data_dir / "train.wp_target",
        ini_data_dir / "dev.wp_source",
        ini_data_dir / "dev.wp_target",
        ini_data_dir / "test.wp_source",
        ini_data_dir / "test.wp_target"
    ]

    all_exist = all(check_file_exists(f, f.name) for f in required_files)
    if not all_exist:
        print("\n✗ Required files missing! Please run download script manually.")
        return False

    # Step 2: Generate vocabulary
    success = run_command(
        "python get_vocab.py wp",
        "Step 2: Generating vocabulary"
    )
    if not success:
        return False

    # Verify vocabulary file
    vocab_file = ini_data_dir / "entity_vocab.txt"
    if not check_file_exists(vocab_file, "Vocabulary file"):
        return False

    # Step 3: Generate negative samples
    success = run_command(
        "python gen_train_data.py wp",
        "Step 3: Generating negative samples"
    )
    if not success:
        return False

    # Verify generated training data
    print("\nVerifying generated training data...")
    train_data_dir = data_dir / "train_data"
    train_files = [
        train_data_dir / "train_human.txt",
        train_data_dir / "train_negative.txt",
        train_data_dir / "dev_human.txt",
        train_data_dir / "dev_negative.txt",
        train_data_dir / "test_human.txt",
        train_data_dir / "test_negative.txt"
    ]

    all_exist = all(check_file_exists(f, f.name) for f in train_files)
    if not all_exist:
        print("\n✗ Training data generation incomplete!")
        return False

    print("\n" + "="*70)
    print("✓ WRITINGPROMPTS DATASET READY!")
    print("="*70)
    print(f"\nDataset location: {data_dir.absolute()}")

    return True


def prepare_award_winning():
    """Prepare Award-winning dataset."""
    print("\n" + "="*70)
    print("PREPARING AWARD-WINNING DATASET")
    print("="*70 + "\n")

    award_dir = Path("./Award-winning")

    # Check if Award-winning data exists
    if not award_dir.exists():
        print(f"✗ Award-winning directory not found: {award_dir}")
        print("Please ensure the Award-winning dataset is in ./Award-winning/")
        return False

    # Run preparation script
    success = run_command(
        "python prepare_award_winning.py --award_dir ./Award-winning",
        "Preparing Award-winning dataset"
    )
    if not success:
        return False

    # Verify generated files
    print("\nVerifying Award-winning training data...")
    train_data_dir = award_dir / "train_data"
    required_files = [
        train_data_dir / "train_human.txt",
        train_data_dir / "train_negative.txt",
        train_data_dir / "dev_human.txt",
        train_data_dir / "dev_negative.txt",
        train_data_dir / "test_human.txt",
        train_data_dir / "test_negative.txt"
    ]

    all_exist = all(check_file_exists(f, f.name) for f in required_files)
    if not all_exist:
        print("\n✗ Award-winning data preparation incomplete!")
        return False

    print("\n" + "="*70)
    print("✓ AWARD-WINNING DATASET READY!")
    print("="*70)
    print(f"\nDataset location: {award_dir.absolute()}")

    return True


def combine_datasets():
    """Combine WritingPrompts and Award-winning datasets."""
    print("\n" + "="*70)
    print("COMBINING DATASETS")
    print("="*70 + "\n")

    # Verify both datasets are ready
    wp_dir = Path("./WritingPrompts")
    award_dir = Path("./Award-winning")

    wp_train = wp_dir / "train_data" / "train_human.txt"
    award_train = award_dir / "train_data" / "train_human.txt"

    if not wp_train.exists():
        print(f"✗ WritingPrompts training data not found: {wp_train}")
        print("Please prepare WritingPrompts dataset first.")
        return False

    if not award_train.exists():
        print(f"✗ Award-winning training data not found: {award_train}")
        print("Please prepare Award-winning dataset first.")
        return False

    # Combine datasets
    success = run_command(
        "python download_writing_prompts.py --combine --wp_dir ./WritingPrompts "
        "--award_dir ./Award-winning --combined_output ./Combined",
        "Combining WritingPrompts and Award-winning datasets"
    )

    if not success:
        return False

    print("\n" + "="*70)
    print("✓ COMBINED DATASET READY!")
    print("="*70)
    print(f"\nDataset location: {Path('./Combined').absolute()}")

    return True


def print_training_instructions(dataset_type):
    """Print instructions for training UNION."""
    print("\n" + "="*70)
    print("NEXT STEPS: TRAINING UNION")
    print("="*70 + "\n")

    if dataset_type == "wp":
        data_dir = "./Data/WritingPrompts"
        print("To train UNION on WritingPrompts:\n")
        print(f"python run_union.py \\")
        print(f"    --data_dir {data_dir} \\")
        print(f"    --output_dir ./model/union_wp \\")
        print(f"    --task_name train \\")
        print(f"    --init_checkpoint ./model/uncased_L-12_H-768_A-12/bert_model.ckpt")

    elif dataset_type == "award":
        data_dir = "./Data/Award-winning"
        print("To train UNION on Award-winning:\n")
        print(f"python run_union.py \\")
        print(f"    --data_dir {data_dir} \\")
        print(f"    --output_dir ./model/union_award \\")
        print(f"    --task_name train \\")
        print(f"    --init_checkpoint ./model/uncased_L-12_H-768_A-12/bert_model.ckpt \\")
        print(f"    --use_reconstruction")

    elif dataset_type == "combined":
        data_dir = "./Data/Combined"
        print("To train UNION on Combined dataset (WritingPrompts + Award-winning):\n")
        print(f"python run_union.py \\")
        print(f"    --data_dir {data_dir} \\")
        print(f"    --output_dir ./model/union_combined \\")
        print(f"    --task_name train \\")
        print(f"    --init_checkpoint ./model/uncased_L-12_H-768_A-12/bert_model.ckpt \\")
        print(f"    --use_reconstruction")

    print("\nNote: Make sure you have downloaded the BERT checkpoint first:")
    print("https://github.com/google-research/bert")
    print("Use the uncased BERT-base model (110M parameters)")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end dataset preparation for UNION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare WritingPrompts only
  python prepare_full_dataset.py --dataset wp

  # Prepare Award-winning only
  python prepare_full_dataset.py --dataset award

  # Prepare combined dataset
  python prepare_full_dataset.py --dataset combined

  # Skip download if data already exists
  python prepare_full_dataset.py --dataset wp --skip-download
        """
    )

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["wp", "award", "combined"],
                        help="Dataset to prepare: 'wp' (WritingPrompts), 'award' (Award-winning), or 'combined'")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading (use existing data)")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("UNION DATASET PREPARATION PIPELINE")
    print("="*70)
    print(f"\nDataset type: {args.dataset}")
    print(f"Skip download: {args.skip_download}")
    print("="*70)

    success = True

    if args.dataset == "wp":
        # Prepare WritingPrompts only
        success = prepare_writing_prompts(args.skip_download)
        if success:
            print_training_instructions("wp")

    elif args.dataset == "award":
        # Prepare Award-winning only
        success = prepare_award_winning()
        if success:
            print_training_instructions("award")

    elif args.dataset == "combined":
        # Prepare both datasets and combine
        wp_success = prepare_writing_prompts(args.skip_download)
        if not wp_success:
            print("\n✗ WritingPrompts preparation failed!")
            success = False
        else:
            award_success = prepare_award_winning()
            if not award_success:
                print("\n✗ Award-winning preparation failed!")
                success = False
            else:
                success = combine_datasets()
                if success:
                    print_training_instructions("combined")

    if success:
        print("\n" + "="*70)
        print("✓ DATASET PREPARATION COMPLETE!")
        print("="*70 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("✗ DATASET PREPARATION FAILED")
        print("="*70)
        print("\nPlease check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
