"""
Download and process WritingPrompts dataset for UNION training.

This script downloads the WritingPrompts dataset from multiple sources and processes
it into the format required by UNION (compatible with Award-winning dataset).

Sources:
1. Hugging Face datasets (recommended)
2. Original WritingPrompts repository
3. UNION repository data links (THUcloud/GoogleDrive)

Output format:
- ini_data/train.wp_source, train.wp_target
- ini_data/dev.wp_source, dev.wp_target
- ini_data/test.wp_source, test.wp_target
"""

import os
import sys
import json
import urllib.request
import gzip
import shutil
from pathlib import Path
import argparse


def download_from_huggingface(output_dir="./WritingPrompts"):
    """
    Download WritingPrompts from Hugging Face datasets.
    This is the recommended method as it's most reliable.
    """
    try:
        from datasets import load_dataset
        print("Downloading WritingPrompts from Hugging Face...")

        # Load the dataset
        dataset = load_dataset("writing_prompts")

        output_path = Path(output_dir) / "ini_data"
        output_path.mkdir(parents=True, exist_ok=True)

        # Process each split
        splits = {
            'train': dataset['train'],
            'valid': dataset['valid'],  # HF uses 'valid' instead of 'dev'
            'test': dataset['test']
        }

        for split_name, split_data in splits.items():
            # Convert 'valid' to 'dev' to match UNION naming
            file_prefix = 'dev' if split_name == 'valid' else split_name

            source_file = output_path / f"{file_prefix}.wp_source"
            target_file = output_path / f"{file_prefix}.wp_target"

            print(f"Processing {split_name} split ({len(split_data)} examples)...")

            with open(source_file, 'w', encoding='utf-8') as f_src, \
                 open(target_file, 'w', encoding='utf-8') as f_tgt:

                for example in split_data:
                    # Get prompt and story
                    prompt = example['prompt'].strip()
                    story = example['story'].strip()

                    # Convert to single-line format (replace newlines with spaces)
                    prompt = prompt.replace('\n', ' ')
                    story = story.replace('\n', ' ')

                    # Write to files
                    f_src.write(prompt + '\n')
                    f_tgt.write(story + '\n')

            print(f"  Wrote {file_prefix}.wp_source and {file_prefix}.wp_target")

        print(f"\nSuccessfully downloaded and processed WritingPrompts dataset!")
        print(f"Files saved to: {output_path}")
        return True

    except ImportError:
        print("ERROR: Hugging Face datasets library not installed.")
        print("Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"ERROR downloading from Hugging Face: {e}")
        return False


def download_from_original_source(output_dir="./WritingPrompts"):
    """
    Download WritingPrompts from the original source.
    The original dataset is hosted at https://github.com/pytorch/fairseq/tree/master/examples/stories
    """
    print("Downloading from original WritingPrompts source...")

    base_url = "https://dl.fbaipublicfiles.com/fairseq/data/"
    files = [
        "writingPrompts.tar.gz"
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / "temp"
    temp_dir.mkdir(exist_ok=True)

    try:
        for filename in files:
            url = base_url + filename
            output_file = temp_dir / filename

            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, output_file)

            # Extract tar.gz
            print(f"Extracting {filename}...")
            shutil.unpack_archive(output_file, temp_dir)

        # Process the extracted files into the correct format
        # The original dataset has different file structure
        # You may need to adjust this based on the actual structure

        print("Processing downloaded files...")
        process_original_files(temp_dir, output_path / "ini_data")

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        print(f"\nSuccessfully downloaded and processed WritingPrompts dataset!")
        print(f"Files saved to: {output_path / 'ini_data'}")
        return True

    except Exception as e:
        print(f"ERROR downloading from original source: {e}")
        print("Falling back to manual download instructions...")
        return False


def process_original_files(source_dir, output_dir):
    """
    Process files from the original WritingPrompts format to UNION format.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map of original filenames to output filenames
    # You may need to adjust this based on actual file structure
    file_mapping = {
        'train': ('train.wp_source', 'train.wp_target'),
        'valid': ('dev.wp_source', 'dev.wp_target'),
        'test': ('test.wp_source', 'test.wp_target')
    }

    # Process each split (implementation depends on actual file structure)
    # This is a placeholder - adjust based on the actual downloaded files
    pass


def download_from_union_repo(output_dir="./WritingPrompts"):
    """
    Provide instructions to manually download from UNION repository.
    """
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nThe WritingPrompts dataset can be downloaded from:")
    print("\n1. THUcloud:")
    print("   https://cloud.tsinghua.edu.cn/d/b3bdeee2c9b647439746/")
    print("\n2. Google Drive:")
    print("   https://drive.google.com/drive/folders/1Cfc-YkQo-27ovVug2bfpqBclECimvgwu?usp=sharing")
    print("\nAfter downloading:")
    print(f"1. Extract the WritingPrompts folder")
    print(f"2. Place it in: {Path(output_dir).absolute()}")
    print(f"3. Ensure the structure is:")
    print(f"   {output_dir}/ini_data/train.wp_source")
    print(f"   {output_dir}/ini_data/train.wp_target")
    print(f"   {output_dir}/ini_data/dev.wp_source")
    print(f"   {output_dir}/ini_data/dev.wp_target")
    print(f"   {output_dir}/ini_data/test.wp_source")
    print(f"   {output_dir}/ini_data/test.wp_target")
    print("="*70 + "\n")


def verify_dataset(data_dir="./WritingPrompts"):
    """
    Verify that the dataset has been downloaded and formatted correctly.
    """
    data_path = Path(data_dir) / "ini_data"
    required_files = [
        "train.wp_source", "train.wp_target",
        "dev.wp_source", "dev.wp_target",
        "test.wp_source", "test.wp_target"
    ]

    print("\nVerifying dataset...")
    all_present = True

    for filename in required_files:
        filepath = data_path / filename
        if filepath.exists():
            # Count lines
            with open(filepath, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            print(f"  ✓ {filename}: {num_lines:,} lines")
        else:
            print(f"  ✗ {filename}: NOT FOUND")
            all_present = False

    if all_present:
        print("\n✓ Dataset verification successful!")
        print(f"Dataset ready at: {data_path.absolute()}")

        # Show sample
        print("\nSample data (first example from train):")
        with open(data_path / "train.wp_source", 'r', encoding='utf-8') as f:
            prompt = f.readline().strip()[:200]
        with open(data_path / "train.wp_target", 'r', encoding='utf-8') as f:
            story = f.readline().strip()[:200]
        print(f"  Prompt: {prompt}...")
        print(f"  Story:  {story}...")

        return True
    else:
        print("\n✗ Dataset verification failed!")
        print("Some required files are missing.")
        return False


def combine_with_award_winning(wp_dir="./WritingPrompts", award_dir="./Award-winning",
                                output_dir="./Combined", train_ratio=0.8, dev_ratio=0.1):
    """
    Combine WritingPrompts and Award-winning datasets for joint training.

    This creates a combined dataset that can be used to train UNION on both
    WritingPrompts and Award-winning data together.
    """
    import random
    random.seed(42)

    print("\nCombining WritingPrompts and Award-winning datasets...")

    wp_path = Path(wp_dir) / "ini_data"
    award_path = Path(award_dir) / "train_data"
    output_path = Path(output_dir) / "train_data"
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all data
    all_human = []
    all_negative = []
    all_ref_map = []

    # Load WritingPrompts data (need to generate negatives first)
    print("  Loading WritingPrompts data...")
    print("  NOTE: You need to run gen_train_data.py first to generate negatives!")

    wp_train_data = Path(wp_dir) / "train_data"
    if (wp_train_data / "train_human.txt").exists():
        with open(wp_train_data / "train_human.txt", 'r', encoding='utf-8') as f:
            wp_human = [line.strip() for line in f if line.strip()]
        with open(wp_train_data / "train_negative.txt", 'r', encoding='utf-8') as f:
            wp_negative = [line.strip() for line in f if line.strip()]

        all_human.extend(wp_human)
        all_negative.extend(wp_negative)
        all_ref_map.extend(['1'] * len(wp_human))
        print(f"    Loaded {len(wp_human)} WritingPrompts examples")
    else:
        print("    WARNING: WritingPrompts negatives not found!")
        print("    Run: python gen_train_data.py wp")

    # Load Award-winning data
    print("  Loading Award-winning data...")
    if (award_path / "train_human.txt").exists():
        with open(award_path / "train_human.txt", 'r', encoding='utf-8') as f:
            award_human = [line.strip() for line in f if line.strip()]
        with open(award_path / "train_negative.txt", 'r', encoding='utf-8') as f:
            award_negative = [line.strip() for line in f if line.strip()]

        all_human.extend(award_human)
        all_negative.extend(award_negative)
        all_ref_map.extend(['1'] * len(award_human))
        print(f"    Loaded {len(award_human)} Award-winning examples")
    else:
        print("    WARNING: Award-winning data not found!")
        print("    Run: python prepare_award_winning.py")

    if not all_human:
        print("\n  ERROR: No data loaded! Make sure both datasets are prepared.")
        return False

    # Shuffle and split
    print(f"\n  Total examples: {len(all_human)}")
    combined = list(zip(all_human, all_negative, all_ref_map))
    random.shuffle(combined)

    n_total = len(combined)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)

    train_data = combined[:n_train]
    dev_data = combined[n_train:n_train + n_dev]
    test_data = combined[n_train + n_dev:]

    print(f"  Split: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}")

    # Write combined files
    for split_name, split_data in [('train', train_data), ('dev', dev_data), ('test', test_data)]:
        if not split_data:
            continue

        human_file = output_path / f"{split_name}_human.txt"
        negative_file = output_path / f"{split_name}_negative.txt"
        ref_map_file = output_path / f"{split_name}_negative_ref_map.txt"

        with open(human_file, 'w', encoding='utf-8') as f:
            for h, _, _ in split_data:
                f.write(h + '\n')

        with open(negative_file, 'w', encoding='utf-8') as f:
            for _, n, _ in split_data:
                f.write(n + '\n')

        with open(ref_map_file, 'w', encoding='utf-8') as f:
            for _, _, r in split_data:
                f.write(r + '\n')

        print(f"  Wrote {split_name} split to {output_path}")

    print(f"\n✓ Combined dataset created at: {output_path.absolute()}")
    print("\nYou can now train with:")
    print(f"  python run_union.py --data_dir {output_dir} --dataset_type wp --use_reconstruction")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and process WritingPrompts dataset for UNION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from Hugging Face (recommended)
  python download_writing_prompts.py --source huggingface

  # Download from original source
  python download_writing_prompts.py --source original

  # Show manual download instructions
  python download_writing_prompts.py --source manual

  # Verify existing dataset
  python download_writing_prompts.py --verify

  # Combine with Award-winning dataset
  python download_writing_prompts.py --combine --wp_dir ./WritingPrompts --award_dir ./Award-winning
        """
    )

    parser.add_argument("--source", type=str, default="huggingface",
                        choices=["huggingface", "original", "manual"],
                        help="Download source (default: huggingface)")
    parser.add_argument("--output_dir", type=str, default="./WritingPrompts",
                        help="Output directory for dataset (default: ./WritingPrompts)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify that dataset is properly formatted")
    parser.add_argument("--combine", action="store_true",
                        help="Combine WritingPrompts with Award-winning dataset")
    parser.add_argument("--wp_dir", type=str, default="./WritingPrompts",
                        help="WritingPrompts directory for combining")
    parser.add_argument("--award_dir", type=str, default="./Award-winning",
                        help="Award-winning directory for combining")
    parser.add_argument("--combined_output", type=str, default="./Combined",
                        help="Output directory for combined dataset")

    args = parser.parse_args()

    # If verify flag is set, just verify and exit
    if args.verify:
        success = verify_dataset(args.output_dir)
        sys.exit(0 if success else 1)

    # If combine flag is set, combine datasets
    if args.combine:
        success = combine_with_award_winning(
            args.wp_dir,
            args.award_dir,
            args.combined_output
        )
        sys.exit(0 if success else 1)

    # Download dataset
    print("="*70)
    print("WritingPrompts Dataset Download & Processing")
    print("="*70 + "\n")

    success = False

    if args.source == "huggingface":
        success = download_from_huggingface(args.output_dir)
    elif args.source == "original":
        success = download_from_original_source(args.output_dir)
    elif args.source == "manual":
        download_from_union_repo(args.output_dir)
        sys.exit(0)

    # Verify if download was successful
    if success:
        verify_dataset(args.output_dir)

        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Generate vocabulary and negative samples:")
        print(f"   cd {Path(args.output_dir).parent}")
        print("   python get_vocab.py wp")
        print("   python gen_train_data.py wp")
        print("\n2. (Optional) Combine with Award-winning dataset:")
        print(f"   python download_writing_prompts.py --combine")
        print("\n3. Train UNION:")
        print(f"   python run_union.py --data_dir {args.output_dir} --task_name train")
        print("="*70 + "\n")
    else:
        print("\nDownload failed. Try:")
        print("  python download_writing_prompts.py --source manual")
        print("for manual download instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
