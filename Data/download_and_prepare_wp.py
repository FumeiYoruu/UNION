"""
Complete pipeline to download WritingPrompts and prepare it for UNION training.

This script:
1. Downloads WritingPrompts from Hugging Face to WP/ini_data
2. Calls get_vocab.py to generate vocabulary
3. Calls gen_train_data.py to generate negative samples
4. Outputs to WP/train_data/ directory

Usage:
    python download_and_prepare_wp.py
"""

import sys
import subprocess
from pathlib import Path

# Check if datasets library is available
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def download_writingprompts():
    """Download WritingPrompts dataset from Hugging Face."""
    if not HAS_DATASETS:
        print("ERROR: datasets library not installed")
        print("Install with: pip install datasets")
        sys.exit(1)

    print("\n" + "="*70)
    print("STEP 1: Downloading WritingPrompts Dataset")
    print("="*70 + "\n")

    output_dir = Path("./WP/ini_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from Hugging Face...")

    # Try different possible dataset names
    dataset_names = [
        "writingprompts",
        "writing_prompts",
        "euclaise/writingprompts",
        "facebook/writingprompts"
    ]

    dataset = None
    for name in dataset_names:
        try:
            print(f"  Trying '{name}'...")
            dataset = load_dataset(name)
            print(f"  ✓ Successfully loaded '{name}'")
            break
        except Exception:
            print(f"  ✗ Failed with '{name}'")
            continue

    if dataset is None:
        print("\n✗ Could not load dataset from Hugging Face")
        print("\nAlternative: Download manually from:")
        print("  https://cloud.tsinghua.edu.cn/d/b3bdeee2c9b647439746/")
        print("  https://drive.google.com/drive/folders/1Cfc-YkQo-27ovVug2bfpqBclECimvgwu")
        print("\nExtract to ./WP/ini_data/ and ensure files are named:")
        print("  train.wp_source, train.wp_target")
        print("  dev.wp_source, dev.wp_target")
        print("  test.wp_source, test.wp_target")
        print("\nThen run the rest of the pipeline:")
        print("  python get_vocab_wp.py")
        print("  python gen_train_data_wp.py")
        return False

    # Handle different possible split names
    print(f"\nAvailable splits: {list(dataset.keys())}")

    # Check the structure of the first example to see field names
    first_split = list(dataset.keys())[0]
    if len(dataset[first_split]) > 0:
        print(f"Dataset fields: {list(dataset[first_split].features.keys())}")

    splits = {}

    # Map train split
    if 'train' in dataset:
        splits['train'] = dataset['train']
    else:
        print("✗ No training split found!")
        return False

    # Map validation split (could be 'valid', 'validation', or 'dev')
    if 'valid' in dataset:
        splits['valid'] = dataset['valid']
    elif 'validation' in dataset:
        splits['valid'] = dataset['validation']
    elif 'dev' in dataset:
        splits['valid'] = dataset['dev']
    else:
        print("⚠ No validation split found, skipping dev set")
        splits['valid'] = None

    # Map test split
    if 'test' in dataset:
        splits['test'] = dataset['test']
    else:
        print("⚠ No test split found, skipping test set")
        splits['test'] = None

    # Determine field names from the dataset
    first_split = list(dataset.keys())[0]
    field_names = list(dataset[first_split].features.keys())

    # Try to find prompt field
    prompt_field = None
    for field in ['prompt', 'source', 'context', 'input']:
        if field in field_names:
            prompt_field = field
            break

    # Try to find story field
    story_field = None
    for field in ['story', 'target', 'output', 'completion', 'text']:
        if field in field_names:
            story_field = field
            break

    if not prompt_field or not story_field:
        print(f"\n✗ Could not identify prompt/story fields in dataset")
        print(f"Available fields: {field_names}")
        print("\nPlease manually download the dataset")
        return False

    print(f"\nUsing fields: prompt='{prompt_field}', story='{story_field}'")

    for split_name, split_data in splits.items():
        if split_data is None:
            continue

        file_prefix = 'dev' if split_name == 'valid' else split_name

        source_file = output_dir / f"{file_prefix}.wp_source"
        target_file = output_dir / f"{file_prefix}.wp_target"

        print(f"Processing {split_name} split ({len(split_data)} examples)...")

        with open(source_file, 'w', encoding='utf-8') as f_src, \
             open(target_file, 'w', encoding='utf-8') as f_tgt:

            for example in split_data:
                prompt = example[prompt_field].strip().replace('\n', ' ')
                story = example[story_field].strip().replace('\n', ' ')

                f_src.write(prompt + '\n')
                f_tgt.write(story + '\n')

        print(f"  ✓ Wrote {file_prefix}.wp_source and {file_prefix}.wp_target")

    print(f"\n✓ Dataset downloaded to: {output_dir.absolute()}")
    return True


def create_directories():
    """Ensure necessary directories exist."""
    Path("./WP/ini_data").mkdir(parents=True, exist_ok=True)
    Path("./WP/train_data").mkdir(parents=True, exist_ok=True)
    print("✓ Created WP directories")


def create_wp_scripts():
    """Create WP-specific versions of get_vocab.py and gen_train_data.py."""
    print("\n" + "="*70)
    print("Creating WP-specific processing scripts")
    print("="*70 + "\n")

    # Create get_vocab_wp.py
    get_vocab_content = '''from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer().lemmatize
import nltk
pos_tag = nltk.pos_tag
from nltk.corpus import stopwords

file_dir = "./WP/ini_data/"
file_name = "train.wp_target"

def get_avail_phrases():
    sw = set(stopwords.words('english'))
    avail_phrases = set()
    fin = open("./conceptnet_entity.csv", 'r', encoding='utf-8')
    for i, line in enumerate(fin):
        avail_phrases.add(' '.join(line.strip().split("|||")[:-1]))
    avail_phrases = avail_phrases - sw
    fin.close()

    fin = open("./negation.txt", 'r', encoding='utf-8')
    for i, line in enumerate(fin):
        avail_phrases.add(' '.join(line.strip().split()[1:]))
    fin.close()

    for w in [".", ",", "!", "?", "male", "female", "neutral"]:
        avail_phrases.add(w)

    return avail_phrases

avail_phrases = get_avail_phrases()

vocab = {}
with open("%s/%s"%(file_dir, file_name), "r", encoding='utf-8') as fin1:
    for kkk, line in enumerate(fin1):
        if kkk % 1000 == 0:
            print(kkk)
        tmp = line.strip().split()
        pos = pos_tag(tmp)
        for word_pos in pos:
            if lemma(word_pos[0], 'v' if word_pos[1][0] == 'V' else 'n') not in avail_phrases:
                continue
            if word_pos[0] in vocab:
                vocab[word_pos[0]]["number"] += 1
                if word_pos[1] in vocab[word_pos[0]]:
                    vocab[word_pos[0]][word_pos[1]] += 1
                else:
                    vocab[word_pos[0]][word_pos[1]] = 1
            else:
                vocab[word_pos[0]] = {word_pos[1]:1, "number":1}
vocab_list = sorted(vocab, key=lambda x: vocab[x]["number"], reverse=True)
with open("%s/entity_vocab.txt"%file_dir, "w") as fout:
    for v in vocab_list:
        pos_list = sorted(vocab[v], key=vocab[v].get, reverse=True)
        pos_list.remove("number")
        fout.write("%s %d|||"%(v, vocab[v]["number"]) + "|||".join(["%s %d"%(p, vocab[v][p]) for p in pos_list]) + "\\n")
'''

    with open("get_vocab_wp.py", "w", encoding='utf-8') as f:
        f.write(get_vocab_content)

    print("  ✓ Created get_vocab_wp.py")

    # Read original gen_train_data.py and modify it
    with open("gen_train_data.py", "r", encoding='utf-8') as f:
        gen_content = f.read()

    # Replace the data_dir and output_dir lines
    gen_content = gen_content.replace(
        'data_dir = "./%s/ini_data"%("WritingPrompts" if "w" in sys.argv[1] else "ROCStories")',
        'data_dir = "./WP/ini_data"'
    )
    gen_content = gen_content.replace(
        'output_dir = "%s/train_data"%("WritingPrompts" if "w" in sys.argv[1] else "ROCStories")',
        'output_dir = "./WP/train_data"'
    )

    with open("gen_train_data_wp.py", "w", encoding='utf-8') as f:
        f.write(gen_content)

    print("  ✓ Created gen_train_data_wp.py")


def generate_vocabulary():
    """Generate entity vocabulary."""
    print("\n" + "="*70)
    print("STEP 2: Generating Vocabulary")
    print("="*70 + "\n")

    try:
        subprocess.run(
            [sys.executable, "get_vocab_wp.py"],
            check=True,
            cwd=Path.cwd()
        )
        print("\n✓ Vocabulary generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Vocabulary generation failed: {e}")
        return False
    except FileNotFoundError:
        print("\n✗ get_vocab_wp.py not found")
        return False


def generate_negatives():
    """Generate negative samples."""
    print("\n" + "="*70)
    print("STEP 3: Generating Negative Samples")
    print("="*70 + "\n")

    # Create train_data directory
    Path("./WP/train_data").mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [sys.executable, "gen_train_data_wp.py"],
            check=True,
            cwd=Path.cwd()
        )
        print("\n✓ Negative samples generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Negative sample generation failed: {e}")
        return False
    except FileNotFoundError:
        print("\n✗ gen_train_data_wp.py not found")
        return False


def verify_output():
    """Verify that output is correctly formatted."""
    print("\n" + "="*70)
    print("STEP 4: Verifying Output")
    print("="*70 + "\n")

    train_data_dir = Path("./WP/train_data")

    for split in ['train', 'dev', 'test']:
        human_file = train_data_dir / f"{split}_human.txt"
        negative_file = train_data_dir / f"{split}_negative.txt"

        if human_file.exists() and negative_file.exists():
            with open(human_file, 'r', encoding='utf-8') as f:
                human_count = sum(1 for line in f if line.strip())

            with open(negative_file, 'r', encoding='utf-8') as f:
                negative_count = sum(1 for line in f if line.strip())

            if human_count == negative_count:
                print(f"  ✓ {split}: {human_count:,} story pairs")
            else:
                print(f"  ✗ {split}: Mismatch - {human_count} human vs {negative_count} negative")
        else:
            print(f"  ✗ {split}: Files not found")


def check_if_data_exists():
    """Check if data already exists in WP/ini_data/."""
    required_files = [
        Path("./WP/ini_data/train.wp_target"),
        Path("./WP/ini_data/dev.wp_target"),
        Path("./WP/ini_data/test.wp_target")
    ]
    return all(f.exists() for f in required_files)


def main():
    print("\n" + "="*70)
    print("WritingPrompts Dataset Preparation Pipeline")
    print("="*70)

    # Create necessary directories first
    create_directories()

    # Check if data already exists
    if check_if_data_exists():
        print("\n✓ Data already exists in WP/ini_data/")
        response = input("Skip download and proceed to vocabulary/negative generation? (y/n): ")
        if response.lower() != 'y':
            # Step 1: Download dataset
            if not download_writingprompts():
                print("\n✗ Download failed!")
                sys.exit(1)
        else:
            print("Skipping download step...")
    else:
        # Step 1: Download dataset
        if not download_writingprompts():
            print("\n✗ Download failed!")
            print("\nYou can manually download the dataset and place files in WP/ini_data/")
            print("Then run this script again.")
            sys.exit(1)

    # Step 1.5: Create WP-specific scripts
    create_wp_scripts()

    # Step 2: Generate vocabulary
    if not generate_vocabulary():
        print("\n✗ Vocabulary generation failed!")
        sys.exit(1)

    # Step 3: Generate negatives
    if not generate_negatives():
        print("\n✗ Negative generation failed!")
        sys.exit(1)

    # Step 4: Verify output
    verify_output()

    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nDataset ready at: {Path('./WP').absolute()}")
    print("\nDirectory structure:")
    print("  WP/")
    print("  ├── ini_data/")
    print("  │   ├── train.wp_source, train.wp_target")
    print("  │   ├── dev.wp_source, dev.wp_target")
    print("  │   ├── test.wp_source, test.wp_target")
    print("  │   └── entity_vocab.txt")
    print("  └── train_data/")
    print("      ├── train_human.txt, train_negative.txt")
    print("      ├── dev_human.txt, dev_negative.txt")
    print("      └── test_human.txt, test_negative.txt")
    print("\nYou can now train UNION with:")
    print("  python run_union.py --data_dir ./Data/WP --task_name train")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
