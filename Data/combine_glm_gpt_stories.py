"""
Combine GLM/GPT stories with existing Award-winning dataset.

This script:
- Reads stories from bad_glm.txt, bad_gpt.txt, and good_gpt.txt (split by '-----')
- Labels good_gpt.txt as human samples (positive)
- Labels bad_glm.txt and bad_gpt.txt as negative samples
- Adds ref_map entries with 0 (no reconstruction task for these stories)
- Appends to existing train/dev/test files in Award-winning/train_data/
"""

import os
import random
from pathlib import Path


def read_stories_from_file(filepath):
    """
    Read stories from a text file where stories are separated by '-----'.

    Returns:
        List of story texts (cleaned and normalized)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by separator
    stories = content.split('-----')

    # Clean each story
    cleaned_stories = []
    for story in stories:
        story = story.strip()
        if story:  # Skip empty entries
            # Convert to single-line format (replace newlines with spaces)
            story = story.replace('\n', ' ')
            # Remove multiple spaces
            story = ' '.join(story.split())
            cleaned_stories.append(story)

    return cleaned_stories


def combine_glm_gpt_stories(
    data_dir="./",
    award_dir="./Award-winning",
    train_ratio=0.7,
    dev_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """
    Combine GLM/GPT stories with existing Award-winning dataset.

    Args:
        data_dir: Directory containing bad_glm.txt, bad_gpt.txt, good_gpt.txt
        award_dir: Path to Award-winning directory with train_data
        train_ratio: Proportion for training set
        dev_ratio: Proportion for dev set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)

    data_path = Path(data_dir)
    award_path = Path(award_dir)
    train_data_path = award_path / "train_data"

    # Check if train_data directory exists
    if not train_data_path.exists():
        print(f"Error: {train_data_path} does not exist!")
        print("Please run prepare_award_winning.py first or create the directory.")
        return

    # Read stories from files
    print("Reading stories from text files...")

    bad_glm_file = data_path / "bad_glm.txt"
    bad_gpt_file = data_path / "bad_gpt.txt"
    good_gpt_file = data_path / "good_gpt.txt"

    # Check if files exist
    for filepath in [bad_glm_file, bad_gpt_file, good_gpt_file]:
        if not filepath.exists():
            print(f"Error: {filepath} not found!")
            return

    bad_glm_stories = read_stories_from_file(bad_glm_file)
    bad_gpt_stories = read_stories_from_file(bad_gpt_file)
    good_gpt_stories = read_stories_from_file(good_gpt_file)

    print(f"Loaded {len(bad_glm_stories)} stories from bad_glm.txt")
    print(f"Loaded {len(bad_gpt_stories)} stories from bad_gpt.txt")
    print(f"Loaded {len(good_gpt_stories)} stories from good_gpt.txt")

    # Combine negative stories (bad_glm + bad_gpt)
    all_negative_stories = bad_glm_stories + bad_gpt_stories
    all_positive_stories = good_gpt_stories

    print(f"\nTotal positive (human) stories: {len(all_positive_stories)}")
    print(f"Total negative stories: {len(all_negative_stories)}")

    # Since we need pairs, use the minimum of positive and negative
    # Or we can append them independently based on the split
    # Let's shuffle and split each separately

    random.shuffle(all_positive_stories)
    random.shuffle(all_negative_stories)

    # Split positive stories
    n_pos = len(all_positive_stories)
    n_pos_train = int(n_pos * train_ratio)
    n_pos_dev = int(n_pos * dev_ratio)

    pos_train = all_positive_stories[:n_pos_train]
    pos_dev = all_positive_stories[n_pos_train:n_pos_train + n_pos_dev]
    pos_test = all_positive_stories[n_pos_train + n_pos_dev:]

    # Split negative stories
    n_neg = len(all_negative_stories)
    n_neg_train = int(n_neg * train_ratio)
    n_neg_dev = int(n_neg * dev_ratio)

    neg_train = all_negative_stories[:n_neg_train]
    neg_dev = all_negative_stories[n_neg_train:n_neg_train + n_neg_dev]
    neg_test = all_negative_stories[n_neg_train + n_neg_dev:]

    print(f"\nSplit for positive stories:")
    print(f"  Train: {len(pos_train)}, Dev: {len(pos_dev)}, Test: {len(pos_test)}")
    print(f"\nSplit for negative stories:")
    print(f"  Train: {len(neg_train)}, Dev: {len(neg_dev)}, Test: {len(neg_test)}")

    # Append to existing files
    splits = [
        ('train', pos_train, neg_train),
        ('dev', pos_dev, neg_dev),
        ('test', pos_test, neg_test)
    ]

    for split_name, pos_stories, neg_stories in splits:
        print(f"\nAppending to {split_name} split...")

        # Append to human stories file
        human_file = train_data_path / f"{split_name}_human.txt"
        with open(human_file, 'a', encoding='utf-8') as f:
            for story in pos_stories:
                f.write(story + '\n')
        print(f"  Added {len(pos_stories)} human stories to {human_file}")

        # Append to negative stories file
        negative_file = train_data_path / f"{split_name}_negative.txt"
        with open(negative_file, 'a', encoding='utf-8') as f:
            for story in neg_stories:
                f.write(story + '\n')
        print(f"  Added {len(neg_stories)} negative stories to {negative_file}")

        # Append to reference mapping file
        # All GLM/GPT stories have ref_map=0 (no reconstruction task)
        ref_map_file = train_data_path / f"{split_name}_negative_ref_map.txt"
        with open(ref_map_file, 'a', encoding='utf-8') as f:
            for _ in neg_stories:
                f.write('0\n')
        print(f"  Added {len(neg_stories)} ref_map entries (0) to {ref_map_file}")

    print("\n" + "="*60)
    print("Successfully combined GLM/GPT stories with Award-winning data!")
    print("="*60)
    print(f"\nFiles updated in: {train_data_path}")
    print("\nSummary:")
    print(f"  - Positive stories added: {len(all_positive_stories)}")
    print(f"  - Negative stories added: {len(all_negative_stories)}")
    print(f"  - All negative stories have ref_map=0 (no reconstruction)")
    print("\nNote: The existing Award-winning stories (with ref_map=1) are preserved.")
    print("      The dataset now contains a mix of stories with and without reconstruction.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine GLM/GPT stories with Award-winning dataset"
    )
    parser.add_argument("--data_dir", type=str, default="./",
                        help="Directory containing bad_glm.txt, bad_gpt.txt, good_gpt.txt")
    parser.add_argument("--award_dir", type=str, default="./Award-winning",
                        help="Path to Award-winning directory")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Proportion for training set")
    parser.add_argument("--dev_ratio", type=float, default=0.15,
                        help="Proportion for dev set")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="Proportion for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.dev_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0 (current sum: {total_ratio})")
        exit(1)

    combine_glm_gpt_stories(
        data_dir=args.data_dir,
        award_dir=args.award_dir,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
