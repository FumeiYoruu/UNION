"""Example script demonstrating conditional reconstruction with mixed datasets."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from data import StoryDataset


def create_example_data(data_dir: str):
    """Create example mixed dataset with some samples having references."""
    os.makedirs(os.path.join(data_dir, "train_data"), exist_ok=True)

    # Create negative samples (ROCStories format: 5 sentences + blank line)
    negative_stories = [
        [
            "John woke up late for work.",
            "He rushed to get dressed.",
            "John skipped breakfast.",
            "He arrived at the office.",
            "John was only five minutes late.",
        ],
        [
            "Mary loves to paint.",
            "She bought new canvases.",
            "Mary painted all weekend.",
            "Her paintings were beautiful.",
            "Mary decided to sell her art.",
        ],
        [
            "The cat jumped on the table.",
            "It knocked over a vase.",
            "Water spilled everywhere.",
            "The cat ran away.",
            "The owner cleaned up the mess.",
        ],
        [
            "Tom wanted to learn guitar.",
            "He bought a guitar online.",
            "Tom practiced every day.",
            "He learned three chords.",
            "Tom played his first song.",
        ],
    ]

    # Create human reference stories (only for samples 0 and 2)
    human_stories = [
        [
            "John woke up early for work.",
            "He ate a healthy breakfast.",
            "John arrived at work on time.",
            "He had a productive day.",
            "John felt accomplished.",
        ],
        [
            "The cat sat on the windowsill.",
            "It watched birds outside.",
            "The cat was very peaceful.",
            "It took a long nap.",
            "The owner petted the cat.",
        ],
    ]

    # Write negative samples
    negative_file = os.path.join(data_dir, "train_data", "train_negative.txt")
    with open(negative_file, "w", encoding="utf-8") as f:
        for story in negative_stories:
            for sentence in story:
                f.write(sentence + "\n")
            f.write("\n")  # Blank line

    print(f"Created {negative_file} with {len(negative_stories)} stories")

    # Write human samples
    human_file = os.path.join(data_dir, "train_data", "train_human.txt")
    with open(human_file, "w", encoding="utf-8") as f:
        for story in human_stories:
            for sentence in story:
                f.write(sentence + "\n")
            f.write("\n")  # Blank line

    print(f"Created {human_file} with {len(human_stories)} stories")

    # Write reference mapping (samples 0 and 2 have references)
    mapping_file = os.path.join(data_dir, "train_data", "train_negative_ref_map.txt")
    mapping = [1, 0, 1, 0]  # Samples 0 and 2 have refs, 1 and 3 don't
    with open(mapping_file, "w", encoding="utf-8") as f:
        for val in mapping:
            f.write(f"{val}\n")

    print(f"Created {mapping_file}")
    print(f"  Samples with references: {sum(mapping)}/{len(mapping)}")
    print(f"  Samples without references: {len(mapping) - sum(mapping)}/{len(mapping)}")


def test_mixed_dataset(data_dir: str):
    """Test loading mixed dataset with conditional reconstruction."""
    print("\n" + "=" * 80)
    print("Testing Mixed Dataset Loading")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load dataset with reconstruction enabled
    print("\nLoading dataset with use_reconstruction=True...")
    dataset = StoryDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        mode="train",
        dataset_type="roc",
        max_seq_length=128,
        use_reconstruction=True,
    )

    print(f"Total examples: {len(dataset)}")

    # Check which samples have reconstruction data
    samples_with_refs = []
    samples_without_refs = []

    for i, feature in enumerate(dataset.features):
        if feature.ref_input_ids is not None:
            samples_with_refs.append(i)
        else:
            samples_without_refs.append(i)

    print(f"\nSamples with reconstruction data: {len(samples_with_refs)}")
    print(f"  Indices: {samples_with_refs}")
    print(f"\nSamples without reconstruction data: {len(samples_without_refs)}")
    print(f"  Indices: {samples_without_refs}")

    # Test a few samples
    print("\n" + "=" * 80)
    print("Sample Data Inspection")
    print("=" * 80)

    for idx in [0, 1]:
        print(f"\n--- Sample {idx} ---")
        example = dataset.examples[idx]
        feature = dataset.features[idx]

        print(f"Text: {' '.join(example.text[:2])}...")  # First 2 sentences
        print(f"Label: {example.label}")
        print(f"Has reference: {feature.ref_input_ids is not None}")

        if feature.ref_input_ids is not None:
            print(f"Reference tokens: {len([x for x in feature.ref_input_ids if x != 0])}")
            print(f"Reference text: {tokenizer.decode(feature.ref_input_ids[:30])}...")

    # Get a batch
    print("\n" + "=" * 80)
    print("Batch Example")
    print("=" * 80)

    import torch

    batch_indices = [0, 1]
    batch = {
        "input_ids": torch.stack([dataset[i]["input_ids"] for i in batch_indices]),
        "attention_mask": torch.stack([dataset[i]["attention_mask"] for i in batch_indices]),
        "token_type_ids": torch.stack([dataset[i]["token_type_ids"] for i in batch_indices]),
        "labels": torch.stack([dataset[i]["labels"] for i in batch_indices]),
    }

    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels: {batch['labels']}")

    # Check which samples in batch have reconstruction data
    batch_has_refs = [i for i in batch_indices if dataset.features[i].ref_input_ids is not None]
    print(f"\nBatch samples with reconstruction: {batch_has_refs}")

    print("\n" + "=" * 80)
    print("Success! Mixed dataset loaded correctly.")
    print("=" * 80)


def main():
    """Main function."""
    # Create example data directory
    data_dir = "./example_mixed_data"

    print("Creating example mixed dataset...")
    create_example_data(data_dir)

    print("\nTesting mixed dataset loading...")
    test_mixed_dataset(data_dir)

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    print("\nTo use this with your own data:")
    print("1. Prepare your negative samples in *_negative.txt")
    print("2. Prepare your human references in *_human.txt (only for samples with refs)")
    print("3. Create mapping file with utils/create_ref_mapping.py")
    print("4. Train with --use_reconstruction flag")
    print("\nSee CONDITIONAL_RECONSTRUCTION.md for detailed documentation.")


if __name__ == "__main__":
    main()
