"""Helper script to create reference mapping files for mixed datasets."""

import os
import argparse


def create_ref_mapping(
    negative_file: str,
    output_file: str,
    has_ref_indices: list = None,
    all_have_refs: bool = False,
    none_have_refs: bool = False,
):
    """
    Create a reference mapping file for negative samples.

    Args:
        negative_file: Path to the negative samples file (e.g., train_negative.txt)
        output_file: Path to output mapping file (e.g., train_negative_ref_map.txt)
        has_ref_indices: List of indices (0-based) that have references
        all_have_refs: If True, all samples have references (output all 1s)
        none_have_refs: If True, no samples have references (output all 0s)
    """
    # Count number of stories in negative file
    with open(negative_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Determine format (ROC has blank lines every 6 lines, WP is one per line)
    is_roc = any(line.strip() == "" for line in lines)

    if is_roc:
        # Count stories (1 story = 5 sentences + 1 blank line)
        num_stories = sum(1 for line in lines if line.strip() == "")
    else:
        # WritingPrompts: one story per line
        num_stories = len([line for line in lines if line.strip()])

    print(f"Found {num_stories} stories in {negative_file}")

    # Create mapping
    if all_have_refs:
        mapping = [1] * num_stories
    elif none_have_refs:
        mapping = [0] * num_stories
    elif has_ref_indices is not None:
        mapping = [0] * num_stories
        for idx in has_ref_indices:
            if 0 <= idx < num_stories:
                mapping[idx] = 1
            else:
                print(f"Warning: Index {idx} out of range [0, {num_stories-1}]")
    else:
        raise ValueError("Must specify either has_ref_indices, all_have_refs, or none_have_refs")

    # Write mapping file
    with open(output_file, "w", encoding="utf-8") as f:
        for val in mapping:
            f.write(f"{val}\n")

    num_with_refs = sum(mapping)
    print(f"Created mapping file: {output_file}")
    print(f"  Total stories: {num_stories}")
    print(f"  With references: {num_with_refs}")
    print(f"  Without references: {num_stories - num_with_refs}")


def main():
    parser = argparse.ArgumentParser(
        description="Create reference mapping file for mixed datasets"
    )
    parser.add_argument(
        "--negative_file",
        type=str,
        required=True,
        help="Path to negative samples file (e.g., ./Data/ROCStories/train_data/train_negative.txt)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output mapping file (defaults to <negative_file>_ref_map.txt)",
    )
    parser.add_argument(
        "--indices",
        type=str,
        help="Comma-separated list of indices (0-based) that have references (e.g., '0,1,5,10')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="All samples have references",
    )
    parser.add_argument(
        "--none",
        action="store_true",
        help="No samples have references",
    )

    args = parser.parse_args()

    # Determine output file
    if args.output_file is None:
        base = args.negative_file.replace(".txt", "")
        args.output_file = f"{base}_ref_map.txt"

    # Parse indices
    has_ref_indices = None
    if args.indices:
        has_ref_indices = [int(idx.strip()) for idx in args.indices.split(",")]

    # Validate arguments
    if sum([args.all, args.none, has_ref_indices is not None]) != 1:
        parser.error("Must specify exactly one of: --indices, --all, or --none")

    create_ref_mapping(
        negative_file=args.negative_file,
        output_file=args.output_file,
        has_ref_indices=has_ref_indices,
        all_have_refs=args.all,
        none_have_refs=args.none,
    )


if __name__ == "__main__":
    main()
