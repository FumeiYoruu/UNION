"""Dataset classes for UNION model compatible with original data format."""

import os
import numpy as np
from functools import reduce
import operator
from typing import List, Dict, Optional, Tuple
from itertools import cycle

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


class StoryExample:
    """A single story example."""

    def __init__(self, guid: str, text: List[str], label: int, ref: Optional[List[str]] = None):
        self.guid = guid
        self.text = text  # List of sentences for ROC, single string for WP
        self.label = label
        self.ref = ref  # Reference human story for reconstruction


class StoryFeatures:
    """Features for a single story."""

    def __init__(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        token_type_ids: List[int],
        label_id: int,
        ref_input_ids: Optional[List[int]] = None,
        ref_attention_mask: Optional[List[int]] = None,
        ref_labels: Optional[List[int]] = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.ref_input_ids = ref_input_ids
        self.ref_attention_mask = ref_attention_mask
        self.ref_labels = ref_labels


class StoryDataset(Dataset):
    """Dataset for story classification."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        mode: str = "train",
        dataset_type: str = "roc",
        max_seq_length: int = 512,
        use_reconstruction: bool = False,
        data_fraction: float = 1.0,
        lazy_loading: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing the data (e.g., ./Data/ROCStories)
            tokenizer: Tokenizer to use
            mode: "train", "dev", or "test"
            dataset_type: "roc" for ROCStories, "wp" for WritingPrompts, or "award" for Award-winning
            max_seq_length: Maximum sequence length
            use_reconstruction: Whether to load reference stories for reconstruction
            data_fraction: Fraction of data to use (0.0 to 1.0). Only applied if mode is "train".
            lazy_loading: If True, tokenize examples on-the-fly to save RAM. If False, pre-tokenize all examples (faster but uses more memory).
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.mode = mode
        self.dataset_type = dataset_type
        self.max_seq_length = max_seq_length
        self.use_reconstruction = use_reconstruction
        self.data_fraction = data_fraction
        self.lazy_loading = lazy_loading

        self.examples = self._load_examples()

        # Only pre-compute features if not using lazy loading
        if not self.lazy_loading:
            self.features = self._convert_examples_to_features()
            print(f"  Pre-tokenized all features (memory: ~{len(self.features) * self.max_seq_length * 4 / 1024 / 1024:.1f} MB)")
        else:
            self.features = None
            print(f"  Using lazy loading (will tokenize on-the-fly)")

    def _read_stories(self, input_file: str) -> List[Dict]:
        """Read stories from file (compatible with original format)."""
        stories = []

        def _read_roc_stories(fin):
            """Read ROCStories format (5 sentences, separated by blank line)."""
            story, tmp = [], []
            for k, line in enumerate(fin):
                i = k + 1
                if i % 6 == 0:  # Every 6th line is blank
                    story.append(tmp)
                    tmp = []
                else:
                    tmp.append(line.strip())
            return story

        with open(input_file + ".txt", "r", encoding="utf-8") as fin:
            if self.dataset_type == "roc":
                stories_text = _read_roc_stories(fin)
            else:  # WritingPrompts or Award-winning (both use single-line format)
                stories_text = [[s.strip()] for s in fin.readlines()]

        # Determine labels based on filename
        if "human" in input_file:
            labels = [1 for _ in range(len(stories_text))]
        else:
            labels = [0 for _ in range(len(stories_text))]

        # Load reference stories if using reconstruction
        refs = [[None] for _ in range(len(stories_text))]
        if self.use_reconstruction:
            ref_file = "_".join(input_file.split("_")[:-1] + ["human"])

            # Load reference mapping file if it exists (for mixed datasets)
            # This allows selective reconstruction: only samples marked with 1 get reconstruction loss
            # Example: Award-winning dataset has *_ref_map.txt files indicating which samples have references
            # Format: one line per sample, 1 = has reference (use reconstruction), 0 = no reference (classification only)
            ref_map_file = input_file + "_ref_map.txt"
            ref_map = None
            if os.path.exists(ref_map_file):
                print(f"    Found ref_map file: {ref_map_file}")
                with open(ref_map_file, "r", encoding="utf-8") as fin:
                    # Read mapping: 1 = has reference, 0 = no reference
                    ref_map = [int(line.strip()) for line in fin.readlines()]
                    if len(ref_map) != len(stories_text):
                        print(f"Warning: ref_map length ({len(ref_map)}) doesn't match stories ({len(stories_text)})")
                        ref_map = None
                    else:
                        num_with_ref = sum(ref_map)
                        print(f"    Ref_map: {num_with_ref}/{len(ref_map)} samples have references")

            if os.path.exists(ref_file + ".txt"):
                with open(ref_file + ".txt", "r", encoding="utf-8") as fin:
                    if self.dataset_type == "roc":
                        refs_text = _read_roc_stories(fin)
                    else:  # WritingPrompts or Award-winning
                        refs_text = [[s.strip()] for s in fin.readlines()]

                    if ref_map is not None:
                        # Use ref_map to assign references per sample
                        refs = []
                        ref_idx = 0
                        for i, has_ref in enumerate(ref_map):
                            if has_ref == 1:
                                if ref_idx < len(refs_text):
                                    refs.append(refs_text[ref_idx])
                                    ref_idx += 1
                                else:
                                    print(f"Warning: Not enough references in {ref_file}.txt")
                                    refs.append([None])
                            else:
                                refs.append([None])
                    else:
                        # Fallback: Replicate references to match story count (old behavior)
                        refs = []
                        for ref in refs_text:
                            for _ in range(len(stories_text) // len(refs_text)):
                                refs.append(ref)

        return [
            {"story": s, "label": l, "ref": r}
            for s, l, r in zip(stories_text, labels, refs)
        ]

    def _load_examples(self) -> List[StoryExample]:
        """Load examples from data directory."""
        examples = []
        name_list = ["human", "negative"]

        for name in name_list:
            file_path = os.path.join(
                self.data_dir, "train_data", f"{self.mode}_{name}"
            )
            if not os.path.exists(file_path + ".txt"):
                print(f"Warning: {file_path}.txt not found, skipping...")
                continue

            stories = self._read_stories(file_path)

            for i, story_dict in enumerate(stories):
                guid = f"{self.mode}-{name}-{i}"
                examples.append(
                    StoryExample(
                        guid=guid,
                        text=story_dict["story"],
                        label=story_dict["label"],
                        ref=story_dict["ref"] if self.use_reconstruction else None,
                    )
                )

        # Shuffle examples
        np.random.shuffle(examples)

        # Apply data fraction for training set
        original_size = len(examples)
        if self.mode == "train" and self.data_fraction < 1.0:
            subset_size = int(len(examples) * self.data_fraction)
            examples = examples[:subset_size]
            print(f"Using {self.data_fraction*100:.1f}% of training data: {len(examples)}/{original_size} examples")
        else:
            print(f"Loaded {len(examples)} {self.mode} examples")

        return examples

    def _tokenize_text(self, text) -> Tuple[List[str], List[int]]:
        """Tokenize text (handles both string and list of strings)."""
        if isinstance(text, str):
            tokens = self.tokenizer.tokenize(text)
            return tokens, [len(tokens)]
        elif isinstance(text, list):
            tokens_list = [self.tokenizer.tokenize(t) for t in text]
            lengths = [len(t) for t in tokens_list]
            # Flatten list
            tokens = reduce(operator.add, tokens_list) if tokens_list else []
            return tokens, lengths
        else:
            raise ValueError(f"Unexpected text type: {type(text)}")

    def _convert_example_to_feature(self, example: StoryExample) -> StoryFeatures:
        """Convert a single example to features (used for lazy loading)."""
        # Tokenize main story
        tokens, _ = self._tokenize_text(example.text)

        # Truncate if needed
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[: self.max_seq_length - 2]

        # Build input sequence: [CLS] tokens [SEP]
        input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # Pad to max length
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length

        # Process reference story for reconstruction
        ref_input_ids = None
        ref_attention_mask = None
        ref_labels = None

        if self.use_reconstruction and example.ref is not None and example.ref[0] is not None:
            ref_tokens, ref_lengths = self._tokenize_text(example.ref)

            if len(ref_tokens) > self.max_seq_length - 2:
                ref_tokens = ref_tokens[: self.max_seq_length - 2]

            ref_input_tokens = (
                [self.tokenizer.cls_token] + ref_tokens + [self.tokenizer.sep_token]
            )
            ref_input_ids = self.tokenizer.convert_tokens_to_ids(ref_input_tokens)

            # For reconstruction, we want to mask the first sentence and predict it
            # Create attention mask: 0 for first sentence, 1 for rest
            if ref_lengths and ref_lengths[0] < self.max_seq_length:
                # Mask first sentence (positions 1 to ref_lengths[0]+1, accounting for CLS)
                ref_attention_mask = (
                    [0] * (ref_lengths[0] + 1)
                    + [1] * (len(ref_input_ids) - ref_lengths[0] - 1)
                )
            else:
                ref_attention_mask = [0] * len(ref_input_ids)

            # Labels for masked language modeling
            ref_labels = ref_input_ids.copy()

            # Pad
            padding_length_ref = self.max_seq_length - len(ref_input_ids)
            ref_input_ids += [self.tokenizer.pad_token_id] * padding_length_ref
            ref_attention_mask += [0] * padding_length_ref
            ref_labels += [-100] * padding_length_ref  # -100 is ignored in loss

        return StoryFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label_id=example.label,
            ref_input_ids=ref_input_ids,
            ref_attention_mask=ref_attention_mask,
            ref_labels=ref_labels,
        )

    def _convert_examples_to_features(self) -> List[StoryFeatures]:
        """Convert all examples to features (eager loading)."""
        features = []
        for example in self.examples:
            features.append(self._convert_example_to_feature(example))
        return features

    def __len__(self):
        # Return number of examples (works for both lazy and eager loading)
        if self.lazy_loading:
            return len(self.examples)
        else:
            return len(self.features)

    def __getitem__(self, idx):
        # Get feature (either from pre-computed cache or compute on-the-fly)
        if self.lazy_loading:
            # Lazy loading: tokenize on-the-fly
            feature = self._convert_example_to_feature(self.examples[idx])
        else:
            # Eager loading: use pre-computed features
            feature = self.features[idx]

        item = {
            "input_ids": torch.tensor(feature.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(feature.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(feature.token_type_ids, dtype=torch.long),
            "labels": torch.tensor(feature.label_id, dtype=torch.long),
        }

        if feature.ref_input_ids is not None:
            item["ref_input_ids"] = torch.tensor(feature.ref_input_ids, dtype=torch.long)
            item["ref_attention_mask"] = torch.tensor(
                feature.ref_attention_mask, dtype=torch.long
            )
            item["ref_labels"] = torch.tensor(feature.ref_labels, dtype=torch.long)

        return item


class CombinedDataset(Dataset):
    """Combined dataset from multiple sources (e.g., Award-winning + WritingPrompts)."""

    def __init__(
        self,
        datasets: List[StoryDataset],
    ):
        """
        Args:
            datasets: List of StoryDataset instances to combine
        """
        self.datasets = datasets
        self.cumulative_sizes = self._get_cumulative_sizes()

    def _get_cumulative_sizes(self):
        """Calculate cumulative sizes for indexing."""
        cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            cumulative_sizes.append(total)
        return cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        """Get item from appropriate dataset."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                dataset_idx = i
                break

        # Calculate local index within that dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][local_idx]


class PredictionDataset(Dataset):
    """Dataset for prediction/evaluation on annotated stories."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.stories, self.scores = self._load_annotated_stories()
        self.features = self._convert_to_features()

    def _load_annotated_stories(self) -> Tuple[List[str], List[float]]:
        """Load annotated stories from ant_data.txt."""
        stories = []
        human_scores = []

        ant_file = os.path.join(self.data_dir, "ant_data", "ant_data.txt")
        with open(ant_file, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split("|||")
                if len(parts) >= 3:
                    story_text = parts[1].strip()
                    scores = list(map(float, parts[2].strip().split()))
                    stories.append(story_text)
                    human_scores.append(np.mean(scores))

        print(f"Loaded {len(stories)} annotated stories")
        return stories, human_scores

    def _convert_to_features(self) -> List[StoryFeatures]:
        """Convert stories to features."""
        features = []

        for story in self.stories:
            tokens = self.tokenizer.tokenize(story)

            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[: self.max_seq_length - 2]

            input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)

            # Pad
            padding_length = self.max_seq_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length

            features.append(
                StoryFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label_id=0,  # Dummy label
                )
            )

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        return {
            "input_ids": torch.tensor(feature.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(feature.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(feature.token_type_ids, dtype=torch.long),
        }

    def get_human_scores(self):
        """Return human judgment scores."""
        return self.scores


class MultiDataLoaderIterator:
    """
    Iterator that alternates between multiple dataloaders with different batch sizes.

    This allows training with different batch sizes for different datasets in combined mode,
    which is useful when datasets have different sequence lengths (e.g., Award-winning vs WritingPrompts).

    Example:
        dataloader1 = DataLoader(dataset1, batch_size=2)  # Award-winning (long sequences)
        dataloader2 = DataLoader(dataset2, batch_size=8)  # WritingPrompts (shorter sequences)

        multi_loader = MultiDataLoaderIterator([dataloader1, dataloader2])
        for batch in multi_loader:
            # Alternates between dataloader1 and dataloader2
            # Each batch has its configured batch size
            pass
    """

    def __init__(self, dataloaders: List[DataLoader], dataset_names: Optional[List[str]] = None):
        """
        Args:
            dataloaders: List of DataLoader instances to alternate between
            dataset_names: Optional names for each dataloader (for debugging)
        """
        self.dataloaders = dataloaders
        self.dataset_names = dataset_names or [f"Dataset{i}" for i in range(len(dataloaders))]

        # Calculate total length (sum of all dataloader lengths)
        self._length = sum(len(dl) for dl in dataloaders)

        # Track state for resuming
        self.current_loader_idx = 0
        self.current_loader_iter = None
        self.batches_consumed = 0  # Total batches consumed across all loaders

    def __len__(self):
        """Total number of batches across all dataloaders."""
        return self._length

    def __iter__(self):
        """Create iterators for all dataloaders and start alternating."""
        # Create iterators for each dataloader
        self.iterators = [iter(dl) for dl in self.dataloaders]

        # Track batches remaining in each dataloader
        self.batches_remaining = [len(dl) for dl in self.dataloaders]

        # Reset state
        self.current_loader_idx = 0
        self.batches_consumed = 0

        return self

    def __next__(self):
        """Get next batch, alternating between dataloaders."""
        # Check if all dataloaders are exhausted
        if all(remaining == 0 for remaining in self.batches_remaining):
            raise StopIteration

        # Find next non-empty dataloader (round-robin)
        attempts = 0
        while attempts < len(self.dataloaders):
            if self.batches_remaining[self.current_loader_idx] > 0:
                # Get batch from current dataloader
                try:
                    batch = next(self.iterators[self.current_loader_idx])
                    self.batches_remaining[self.current_loader_idx] -= 1
                    self.batches_consumed += 1

                    # Move to next dataloader for next iteration
                    self.current_loader_idx = (self.current_loader_idx + 1) % len(self.dataloaders)

                    return batch
                except StopIteration:
                    # Should not happen, but handle gracefully
                    self.batches_remaining[self.current_loader_idx] = 0

            # Try next dataloader
            self.current_loader_idx = (self.current_loader_idx + 1) % len(self.dataloaders)
            attempts += 1

        # All dataloaders exhausted
        raise StopIteration

    def skip_batches(self, num_batches: int):
        """
        Skip the first num_batches batches (for resuming from checkpoint).

        This must be called before iterating to resume from a specific batch.
        """
        if num_batches == 0:
            return

        print(f"Skipping first {num_batches} batches to resume from checkpoint...")

        # Create iterators
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.batches_remaining = [len(dl) for dl in self.dataloaders]
        self.current_loader_idx = 0
        self.batches_consumed = 0

        # Skip batches by calling __next__ without using the data
        skipped = 0
        while skipped < num_batches:
            try:
                _ = self.__next__()
                skipped += 1

                if skipped % 100 == 0:
                    print(f"  Skipped {skipped}/{num_batches} batches...")
            except StopIteration:
                print(f"Warning: Reached end of epoch while skipping (skipped {skipped}/{num_batches})")
                break

        print(f"Resuming from batch {num_batches}")

    def get_progress_info(self) -> str:
        """Get progress information string."""
        remaining_per_loader = [f"{name}:{rem}" for name, rem in zip(self.dataset_names, self.batches_remaining)]
        return f"Batches consumed: {self.batches_consumed}/{self._length}, Remaining: [{', '.join(remaining_per_loader)}]"
