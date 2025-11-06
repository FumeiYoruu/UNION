"""Custom data collators for efficient batching with dynamic padding."""

from typing import List, Dict, Any
import torch


class DataCollatorWithDynamicPadding:
    """
    Collate function that pads sequences to the longest in the batch.
    This significantly reduces computation when most sequences are shorter than max_seq_length.

    For example, if max_seq_length=16384 but most sequences are ~2000 tokens,
    this will save ~7x computation time.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: int = None):
        """
        Args:
            tokenizer: Tokenizer for getting pad_token_id
            pad_to_multiple_of: If set, pad to a multiple of this value (e.g., 8 for tensor cores)
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Pad features to the longest sequence in the batch.
        """
        batch = {}

        # Find the longest sequence in this batch
        max_length = max(f["attention_mask"].sum().item() for f in features)

        # Optionally pad to multiple (for tensor cores efficiency)
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1)
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)

        # Pad each feature to max_length
        for key in features[0].keys():
            if key in ["input_ids", "attention_mask", "token_type_ids"]:
                # Truncate and pad
                tensors = []
                for f in features:
                    tensor = f[key][:max_length]
                    if len(tensor) < max_length:
                        # Determine padding value
                        if key == "input_ids":
                            pad_value = self.tokenizer.pad_token_id
                        else:
                            pad_value = 0

                        # Pad
                        padding = torch.full(
                            (max_length - len(tensor),),
                            pad_value,
                            dtype=tensor.dtype
                        )
                        tensor = torch.cat([tensor, padding])
                    tensors.append(tensor)

                batch[key] = torch.stack(tensors)

            elif key.startswith("ref_"):
                # Handle reconstruction tensors similarly
                tensors = []
                for f in features:
                    tensor = f[key][:max_length]
                    if len(tensor) < max_length:
                        if key == "ref_input_ids":
                            pad_value = self.tokenizer.pad_token_id
                        elif key == "ref_labels":
                            pad_value = -100  # Ignore in loss
                        else:
                            pad_value = 0

                        padding = torch.full(
                            (max_length - len(tensor),),
                            pad_value,
                            dtype=tensor.dtype
                        )
                        tensor = torch.cat([tensor, padding])
                    tensors.append(tensor)

                batch[key] = torch.stack(tensors)

            elif key == "labels":
                # Classification labels (no padding needed)
                batch[key] = torch.stack([f[key] for f in features])

        return batch


class DataCollatorWithFixedBuckets:
    """
    Bucket sequences by length and pad to bucket size.
    More predictable than fully dynamic padding.

    Example buckets for max_seq_length=16384:
    - [0, 2048]: pad to 2048
    - [2049, 4096]: pad to 4096
    - [4097, 8192]: pad to 8192
    - [8193, 16384]: pad to 16384
    """

    def __init__(self, tokenizer, buckets: List[int] = None):
        """
        Args:
            tokenizer: Tokenizer for getting pad_token_id
            buckets: List of bucket sizes (e.g., [2048, 4096, 8192, 16384])
        """
        self.tokenizer = tokenizer
        self.buckets = sorted(buckets) if buckets else [2048, 4096, 8192, 16384]

    def _get_bucket_size(self, length: int) -> int:
        """Find the smallest bucket that fits this length."""
        for bucket in self.buckets:
            if length <= bucket:
                return bucket
        return self.buckets[-1]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad features to bucket size."""
        batch = {}

        # Find max length in batch
        max_length_in_batch = max(f["attention_mask"].sum().item() for f in features)

        # Find appropriate bucket
        pad_length = self._get_bucket_size(max_length_in_batch)

        # Pad each feature to bucket size
        for key in features[0].keys():
            if key in ["input_ids", "attention_mask", "token_type_ids"]:
                tensors = []
                for f in features:
                    tensor = f[key][:pad_length]
                    if len(tensor) < pad_length:
                        if key == "input_ids":
                            pad_value = self.tokenizer.pad_token_id
                        else:
                            pad_value = 0

                        padding = torch.full(
                            (pad_length - len(tensor),),
                            pad_value,
                            dtype=tensor.dtype
                        )
                        tensor = torch.cat([tensor, padding])
                    tensors.append(tensor)

                batch[key] = torch.stack(tensors)

            elif key.startswith("ref_"):
                tensors = []
                for f in features:
                    tensor = f[key][:pad_length]
                    if len(tensor) < pad_length:
                        if key == "ref_input_ids":
                            pad_value = self.tokenizer.pad_token_id
                        elif key == "ref_labels":
                            pad_value = -100
                        else:
                            pad_value = 0

                        padding = torch.full(
                            (pad_length - len(tensor),),
                            pad_value,
                            dtype=tensor.dtype
                        )
                        tensor = torch.cat([tensor, padding])
                    tensors.append(tensor)

                batch[key] = torch.stack(tensors)

            elif key == "labels":
                batch[key] = torch.stack([f[key] for f in features])

        return batch
