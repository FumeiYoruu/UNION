"""
Prepare Award-winning dataset for UNION training.
This script reformats the Award-winning dataset structure to match the expected format
for union_pytorch training code.

Structure:
- Positive samples: Data/Award-winning/{story_name}/story_text.txt
- Negative samples: Data/Award-winning/train_data/negatives/{story_name}_negative.txt

Output:
- train_human.txt, dev_human.txt, test_human.txt (positive stories)
- train_negative.txt, dev_negative.txt, test_negative.txt (negative stories)
- *_negative_ref_map.txt (reference mapping: 1 for all since we use reconstruction)
"""

import os
import random
from pathlib import Path

def read_story_text(filepath):
    """Read and clean story text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        # Remove the title (first line) if it exists
        lines = content.split('\n', 1)
        if len(lines) > 1 and lines[0].strip():
            # If first line looks like a title (short, no punctuation at end)
            first_line = lines[0].strip()
            if len(first_line) < 100 and not first_line.endswith('.'):
                content = lines[1].strip()

        # Replace internal newlines with spaces to ensure single-line format
        # This is required for the dataloader which reads one story per line
        content = content.replace('\n', ' ')
        return content

def prepare_award_winning_data(
    award_dir="./Award-winning",
    train_ratio=0.7,
    dev_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """
    Prepare Award-winning dataset.

    Args:
        award_dir: Path to Award-winning directory
        train_ratio: Proportion for training set
        dev_ratio: Proportion for dev set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)

    award_path = Path(award_dir)
    train_data_path = award_path / "train_data"
    negatives_path = train_data_path / "negatives"

    # Get all story directories (exclude train_data)
    story_dirs = [d for d in award_path.iterdir()
                  if d.is_dir() and d.name != "train_data"]

    print(f"Found {len(story_dirs)} story directories")

    # Read all stories
    stories_data = []

    for story_dir in story_dirs:
        story_name = story_dir.name
        story_file = story_dir / "story_text.txt"
        # Negative files replace spaces with underscores
        negative_name = story_name.replace(' ', '_')
        negative_file = negatives_path / f"{negative_name}_negative.txt"

        # Check if both positive and negative exist
        if not story_file.exists():
            print(f"Warning: Missing positive story for {story_name}")
            continue

        if not negative_file.exists():
            print(f"Warning: Missing negative story for {story_name}")
            continue

        # Read texts
        positive_text = read_story_text(story_file)
        negative_text = read_story_text(negative_file)

        stories_data.append({
            'name': story_name,
            'positive': positive_text,
            'negative': negative_text
        })

    print(f"Successfully loaded {len(stories_data)} story pairs")

    # Shuffle and split
    random.shuffle(stories_data)

    n_total = len(stories_data)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)

    train_stories = stories_data[:n_train]
    dev_stories = stories_data[n_train:n_train + n_dev]
    test_stories = stories_data[n_train + n_dev:]

    print(f"Split: Train={len(train_stories)}, Dev={len(dev_stories)}, Test={len(test_stories)}")

    # Write files for each split
    splits = [
        ('train', train_stories),
        ('dev', dev_stories),
        ('test', test_stories)
    ]

    for split_name, split_stories in splits:
        if len(split_stories) == 0:
            print(f"Warning: No stories in {split_name} split, skipping")
            continue

        # Write human stories (positive)
        human_file = train_data_path / f"{split_name}_human.txt"
        with open(human_file, 'w', encoding='utf-8') as f:
            for story in split_stories:
                f.write(story['positive'] + '\n')
        print(f"Wrote {len(split_stories)} human stories to {human_file}")

        # Write negative stories
        negative_file = train_data_path / f"{split_name}_negative.txt"
        with open(negative_file, 'w', encoding='utf-8') as f:
            for story in split_stories:
                f.write(story['negative'] + '\n')
        print(f"Wrote {len(split_stories)} negative stories to {negative_file}")

        # Write reference mapping file for negatives
        # All negatives have a reference (human story) for reconstruction
        # So all entries are 1
        ref_map_file = train_data_path / f"{split_name}_negative_ref_map.txt"
        with open(ref_map_file, 'w', encoding='utf-8') as f:
            for _ in split_stories:
                f.write('1\n')
        print(f"Wrote reference mapping to {ref_map_file}")

    print("\nDataset preparation complete!")
    print(f"Files created in: {train_data_path}")
    print("\nYou can now train with:")
    print(f"  python union_pytorch/train.py --data_dir {award_dir} --dataset_type wp --use_reconstruction")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Award-winning dataset for UNION training")
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

    prepare_award_winning_data(
        award_dir=args.award_dir,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
