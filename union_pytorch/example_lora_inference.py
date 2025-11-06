"""Example script for loading and using a trained LoRA model for inference.

This script demonstrates how to:
1. Load a base UNION model
2. Load LoRA adapter weights
3. Run inference on stories
"""

import torch
from transformers import AutoTokenizer
from peft import PeftModel

from models import create_model


def load_lora_model(base_model_name, lora_checkpoint_path, model_type="bert", device="cuda"):
    """
    Load a UNION model with LoRA adapters.

    Args:
        base_model_name: Name of the base model (e.g., "bert-base-uncased")
        lora_checkpoint_path: Path to LoRA checkpoint directory
        model_type: "bert" or "longformer"
        device: Device to load model on

    Returns:
        model: Model with LoRA adapters loaded
        tokenizer: Tokenizer for the model
    """
    print(f"Loading base model: {base_model_name}")

    # Create base model
    base_model = create_model(
        model_type=model_type,
        model_name=base_model_name,
        use_all_layers=False,  # Set to match training config
        use_reconstruction=False,
        gradient_checkpointing=False,
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {lora_checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Model loaded successfully!")
    return model, tokenizer


def predict_story_quality(model, tokenizer, story_text, device="cuda", max_length=512):
    """
    Predict the quality score for a story.

    Args:
        model: UNION model with LoRA adapters
        tokenizer: Tokenizer
        story_text: Story text (can be multi-sentence)
        device: Device for inference
        max_length: Maximum sequence length

    Returns:
        score: Quality score (probability of being human-written)
        logits: Raw logits [negative_class, positive_class]
    """
    # Tokenize input
    encoding = tokenizer(
        story_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get probability of positive class (human-written)
    score = probs[0, 1].item()

    return score, logits[0].cpu().numpy()


def batch_predict(model, tokenizer, stories, device="cuda", max_length=512, batch_size=8):
    """
    Predict quality scores for multiple stories.

    Args:
        model: UNION model with LoRA adapters
        tokenizer: Tokenizer
        stories: List of story texts
        device: Device for inference
        max_length: Maximum sequence length
        batch_size: Batch size for inference

    Returns:
        scores: List of quality scores
    """
    scores = []

    for i in range(0, len(stories), batch_size):
        batch_stories = stories[i:i+batch_size]

        # Tokenize batch
        encoding = tokenizer(
            batch_stories,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Move to device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        batch_scores = probs[:, 1].cpu().numpy()

        scores.extend(batch_scores)

    return scores


def main():
    """Example usage."""
    # Configuration
    BASE_MODEL_NAME = "bert-base-uncased"
    LORA_CHECKPOINT = "./output/union_lora_roc/best-2500"
    MODEL_TYPE = "bert"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, tokenizer = load_lora_model(
        base_model_name=BASE_MODEL_NAME,
        lora_checkpoint_path=LORA_CHECKPOINT,
        model_type=MODEL_TYPE,
        device=DEVICE
    )

    # Example 1: Single story prediction
    print("\n" + "="*80)
    print("Example 1: Single Story Prediction")
    print("="*80)

    story1 = """Karen was assigned a roommate her first year of college.
    Her roommate asked her to go to a nearby city for a concert.
    Karen agreed happily. The show was absolutely exhilarating.
    Karen became good friends with her roommate."""

    score, logits = predict_story_quality(model, tokenizer, story1, device=DEVICE)
    print(f"\nStory: {story1}")
    print(f"Quality Score: {score:.4f}")
    print(f"Logits: {logits}")
    print(f"Interpretation: {'High quality' if score > 0.5 else 'Low quality'} story")

    # Example 2: Batch prediction
    print("\n" + "="*80)
    print("Example 2: Batch Prediction")
    print("="*80)

    stories = [
        "John went to the store. He bought some milk. Then he went home.",
        "The ancient castle stood majestically on the hilltop, its weathered stones telling tales of centuries past. As the sun set, casting golden rays across the valley, a lone traveler approached its gates.",
        "I love pizza. Pizza is great. Pizza pizza pizza. Pizza.",  # Low quality - repetitive
        "She had always dreamed of becoming an astronaut. Years of training finally paid off when she received the acceptance letter from NASA.",
    ]

    scores = batch_predict(model, tokenizer, stories, device=DEVICE, batch_size=2)

    print("\nBatch Predictions:")
    for i, (story, score) in enumerate(zip(stories, scores)):
        print(f"\nStory {i+1}: {story[:60]}...")
        print(f"Quality Score: {score:.4f}")

    # Example 3: Compare two stories
    print("\n" + "="*80)
    print("Example 3: Compare Two Stories")
    print("="*80)

    story_a = "The cat sat on the mat. The mat was red. The cat was happy."
    story_b = "Eleanor had never believed in fate until that rainy Tuesday afternoon when she stumbled into the old bookshop on Maple Street. Little did she know that the dusty journal she'd pick up would change her life forever."

    score_a, _ = predict_story_quality(model, tokenizer, story_a, device=DEVICE)
    score_b, _ = predict_story_quality(model, tokenizer, story_b, device=DEVICE)

    print(f"\nStory A: {story_a}")
    print(f"Score: {score_a:.4f}")

    print(f"\nStory B: {story_b}")
    print(f"Score: {score_b:.4f}")

    print(f"\n{'Story B' if score_b > score_a else 'Story A'} has higher quality")
    print(f"Difference: {abs(score_b - score_a):.4f}")


if __name__ == "__main__":
    main()
