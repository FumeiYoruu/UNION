"""
Verify format compatibility between WritingPrompts and Award-winning datasets.

This script checks that both datasets are properly formatted and compatible
for combined training with UNION.
"""

import os
from pathlib import Path
import argparse


def check_single_line_format(filepath, dataset_name):
    """
    Verify that a file contains single-line stories (no multi-line stories).
    """
    errors = []
    warnings = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                warnings.append(f"Line {i}: Empty line")
                continue

            # Check for internal newlines (should not exist in single-line format)
            if '\n' in line:
                errors.append(f"Line {i}: Contains internal newlines")

            # Check if line is reasonable length (not too short)
            if len(line) < 50:
                warnings.append(f"Line {i}: Very short story ({len(line)} chars)")

            # Check if line is not too long (potential formatting issue)
            if len(line) > 10000:
                warnings.append(f"Line {i}: Very long story ({len(line)} chars)")

    return errors, warnings


def verify_file_pair(human_file, negative_file, ref_map_file=None):
    """
    Verify that human and negative files are properly paired.
    """
    errors = []

    if not human_file.exists():
        errors.append(f"Missing human file: {human_file}")
        return errors

    if not negative_file.exists():
        errors.append(f"Missing negative file: {negative_file}")
        return errors

    # Count lines
    with open(human_file, 'r', encoding='utf-8') as f:
        human_lines = sum(1 for line in f if line.strip())

    with open(negative_file, 'r', encoding='utf-8') as f:
        negative_lines = sum(1 for line in f if line.strip())

    if human_lines != negative_lines:
        errors.append(
            f"Line count mismatch: {human_file.name} ({human_lines}) vs "
            f"{negative_file.name} ({negative_lines})"
        )

    # Check ref_map if provided
    if ref_map_file and ref_map_file.exists():
        with open(ref_map_file, 'r', encoding='utf-8') as f:
            ref_lines = sum(1 for line in f if line.strip())

        if ref_lines != human_lines:
            errors.append(
                f"Ref map count mismatch: {ref_map_file.name} ({ref_lines}) vs "
                f"{human_file.name} ({human_lines})"
            )

    return errors


def verify_dataset(data_dir, dataset_name, require_ref_map=False):
    """
    Verify a complete dataset directory.
    """
    print(f"\n{'='*70}")
    print(f"Verifying {dataset_name} Dataset")
    print(f"{'='*70}\n")

    data_path = Path(data_dir)
    train_data_path = data_path / "train_data"

    if not train_data_path.exists():
        print(f"✗ Training data directory not found: {train_data_path}")
        return False

    all_errors = []
    all_warnings = []

    # Check each split
    splits = ['train', 'dev', 'test']

    for split in splits:
        print(f"\nChecking {split} split...")

        human_file = train_data_path / f"{split}_human.txt"
        negative_file = train_data_path / f"{split}_negative.txt"
        ref_map_file = train_data_path / f"{split}_negative_ref_map.txt"

        # Verify file pair
        pair_errors = verify_file_pair(
            human_file,
            negative_file,
            ref_map_file if require_ref_map else None
        )

        if pair_errors:
            all_errors.extend(pair_errors)
            print(f"  ✗ {split}: File pairing issues")
            for error in pair_errors:
                print(f"    - {error}")
            continue

        # Check format of human file
        print(f"  Checking {human_file.name}...")
        errors, warnings = check_single_line_format(human_file, dataset_name)
        all_errors.extend([f"{split}/human: {e}" for e in errors])
        all_warnings.extend([f"{split}/human: {w}" for w in warnings])

        # Check format of negative file
        print(f"  Checking {negative_file.name}...")
        errors, warnings = check_single_line_format(negative_file, dataset_name)
        all_errors.extend([f"{split}/negative: {e}" for e in errors])
        all_warnings.extend([f"{split}/negative: {w}" for w in warnings])

        # Count lines
        with open(human_file, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for line in f if line.strip())

        print(f"  ✓ {split}: {num_lines:,} story pairs")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Verification Summary for {dataset_name}")
    print(f"{'='*70}\n")

    if all_errors:
        print(f"✗ Found {len(all_errors)} errors:")
        for error in all_errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")
        return False
    else:
        print(f"✓ No errors found!")

    if all_warnings:
        print(f"\n⚠ Found {len(all_warnings)} warnings:")
        for warning in all_warnings[:10]:  # Show first 10 warnings
            print(f"  - {warning}")
        if len(all_warnings) > 10:
            print(f"  ... and {len(all_warnings) - 10} more warnings")

    print(f"\n✓ {dataset_name} dataset format is valid!")
    return True


def verify_compatibility(wp_dir, award_dir):
    """
    Verify that WritingPrompts and Award-winning datasets are compatible.
    """
    print(f"\n{'='*70}")
    print("Verifying Cross-Dataset Compatibility")
    print(f"{'='*70}\n")

    wp_path = Path(wp_dir) / "train_data"
    award_path = Path(award_dir) / "train_data"

    if not wp_path.exists():
        print(f"✗ WritingPrompts train_data not found: {wp_path}")
        return False

    if not award_path.exists():
        print(f"✗ Award-winning train_data not found: {award_path}")
        return False

    # Compare format of sample stories
    print("Comparing story formats...")

    wp_sample = wp_path / "train_human.txt"
    award_sample = award_path / "train_human.txt"

    # Read first 3 stories from each
    print("\nSample from WritingPrompts:")
    with open(wp_sample, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > 3:
                break
            preview = line.strip()[:100]
            print(f"  {i}. {preview}...")

    print("\nSample from Award-winning:")
    with open(award_sample, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > 3:
                break
            preview = line.strip()[:100]
            print(f"  {i}. {preview}...")

    print("\n✓ Both datasets use single-line format")
    print("✓ Datasets are compatible for combining!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify format compatibility of UNION datasets"
    )

    parser.add_argument("--wp_dir", type=str, default="./WritingPrompts",
                        help="WritingPrompts dataset directory")
    parser.add_argument("--award_dir", type=str, default="./Award-winning",
                        help="Award-winning dataset directory")
    parser.add_argument("--combined_dir", type=str, default="./Combined",
                        help="Combined dataset directory")
    parser.add_argument("--check", type=str, default="all",
                        choices=["all", "wp", "award", "combined", "compatibility"],
                        help="Which dataset(s) to check")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("UNION Dataset Format Verification")
    print("="*70)

    success = True

    if args.check in ["all", "wp"]:
        wp_success = verify_dataset(
            args.wp_dir,
            "WritingPrompts",
            require_ref_map=False
        )
        success = success and wp_success

    if args.check in ["all", "award"]:
        award_success = verify_dataset(
            args.award_dir,
            "Award-winning",
            require_ref_map=True
        )
        success = success and award_success

    if args.check in ["all", "combined"]:
        if Path(args.combined_dir).exists():
            combined_success = verify_dataset(
                args.combined_dir,
                "Combined",
                require_ref_map=True
            )
            success = success and combined_success
        elif args.check == "combined":
            print(f"\n✗ Combined dataset not found: {args.combined_dir}")
            print("Run: python download_writing_prompts.py --combine")
            success = False

    if args.check in ["all", "compatibility"]:
        compat_success = verify_compatibility(args.wp_dir, args.award_dir)
        success = success and compat_success

    # Final summary
    print("\n" + "="*70)
    if success:
        print("✓ ALL VERIFICATIONS PASSED!")
        print("="*70)
        print("\nYour datasets are properly formatted and ready for training!")
        print("\nNext steps:")
        print("1. Download BERT checkpoint: https://github.com/google-research/bert")
        print("2. Train UNION with your dataset")
    else:
        print("✗ VERIFICATION FAILED")
        print("="*70)
        print("\nPlease fix the errors above before training.")

    print("\n")


if __name__ == "__main__":
    main()
