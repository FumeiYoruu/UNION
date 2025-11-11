"""Test mean pooling vs CLS token for LED encoder classification."""

import torch
import torch.nn as nn
from transformers import LEDForConditionalGeneration, AutoTokenizer

def mean_pooling(hidden_state, attention_mask):
    """Mean pooling with attention mask (CORRECTED VERSION)."""
    # hidden_state: [batch_size, seq_len, hidden_size]
    # attention_mask: [batch_size, seq_len]
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    sum_embeddings = torch.sum(hidden_state * attention_mask_expanded, dim=1)  # [batch_size, hidden_size]

    # BUG FIX: Count valid tokens per sample (sum across sequence dimension only)
    sum_mask = attention_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]

    # BUG FIX: Clamp to avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    return sum_embeddings / sum_mask  # [batch_size, hidden_size]

# Load LED model
print("Loading LED model...")
model_name = "allenai/led-base-16384"
led_model = LEDForConditionalGeneration.from_pretrained(model_name)
encoder = led_model.get_encoder()
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"\nLED Encoder Config:")
print(f"  Model: {model_name}")
print(f"  Hidden size: {encoder.config.d_model}")
print(f"  Num layers: {encoder.config.encoder_layers}")
print(f"  Max positions: {encoder.config.max_encoder_position_embeddings}")

# Test with different story pairs
story_pairs = [
    ("A good story with coherent plot and characters.", "A bad story. Random words. No sense."),
    ("The hero saved the village from the dragon.", "The hero dragon village saved from the."),
    ("She carefully planned her escape from prison.", "She carefully planned her escape from prison."),  # Same story
]

print("\n" + "="*80)
print("Testing CLS Token vs Mean Pooling for Story Discrimination")
print("="*80)

for i, (story1, story2) in enumerate(story_pairs):
    print(f"\n{'='*80}")
    print(f"Pair {i+1}:")
    print(f"  Story A: {story1}")
    print(f"  Story B: {story2}")
    print(f"{'='*80}")

    # Tokenize both stories
    inputs1 = tokenizer(story1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(story2, return_tensors="pt", padding=True, truncation=True)

    # Get encoder outputs
    with torch.no_grad():
        outputs1 = encoder(**inputs1)
        outputs2 = encoder(**inputs2)

    # Method 1: CLS token (position 0)
    cls1 = outputs1.last_hidden_state[:, 0, :]  # [1, 768]
    cls2 = outputs2.last_hidden_state[:, 0, :]  # [1, 768]

    # Method 2: Mean pooling
    mean1 = mean_pooling(outputs1.last_hidden_state, inputs1['attention_mask'])
    mean2 = mean_pooling(outputs2.last_hidden_state, inputs2['attention_mask'])

    # Compute cosine similarity and L2 distance
    cos_sim_cls = torch.nn.functional.cosine_similarity(cls1, cls2)
    cos_sim_mean = torch.nn.functional.cosine_similarity(mean1, mean2)

    l2_dist_cls = torch.norm(cls1 - cls2)
    l2_dist_mean = torch.norm(mean1 - mean2)

    print(f"\n  [CLS] Token Representations:")
    print(f"    Story A - L2 norm: {torch.norm(cls1).item():.4f}, mean: {cls1.mean().item():.6f}, std: {cls1.std().item():.6f}")
    print(f"    Story B - L2 norm: {torch.norm(cls2).item():.4f}, mean: {cls2.mean().item():.6f}, std: {cls2.std().item():.6f}")
    print(f"    Cosine similarity: {cos_sim_cls.item():.4f}")
    print(f"    L2 distance: {l2_dist_cls.item():.4f}")

    print(f"\n  Mean Pooling Representations:")
    print(f"    Story A - L2 norm: {torch.norm(mean1).item():.4f}, mean: {mean1.mean().item():.6f}, std: {mean1.std().item():.6f}")
    print(f"    Story B - L2 norm: {torch.norm(mean2).item():.4f}, mean: {mean2.mean().item():.6f}, std: {mean2.std().item():.6f}")
    print(f"    Cosine similarity: {cos_sim_mean.item():.4f}")
    print(f"    L2 distance: {l2_dist_mean.item():.4f}")

    print(f"\n  Comparison:")
    if i < 2:  # Different stories
        print(f"    Expected: LOW similarity for different stories")
        print(f"    [CLS] similarity: {cos_sim_cls.item():.4f} {'✓ GOOD' if cos_sim_cls.item() < 0.95 else '✗ BAD (too similar!)'}")
        print(f"    Mean similarity: {cos_sim_mean.item():.4f} {'✓ GOOD' if cos_sim_mean.item() < 0.95 else '✗ BAD (too similar!)'}")
    else:  # Same story
        print(f"    Expected: HIGH similarity for identical stories")
        print(f"    [CLS] similarity: {cos_sim_cls.item():.4f} {'✓ GOOD' if cos_sim_cls.item() > 0.99 else '✗ BAD (should be ~1.0)'}")
        print(f"    Mean similarity: {cos_sim_mean.item():.4f} {'✓ GOOD' if cos_sim_mean.item() > 0.99 else '✗ BAD (should be ~1.0)'}")

# Test with a simple binary classifier
print("\n" + "="*80)
print("Simulating Binary Classification Task")
print("="*80)

# Create simple classifiers for both methods
cls_classifier = nn.Linear(768, 2)
mean_classifier = nn.Linear(768, 2)

# Label: 0 = bad story, 1 = good story
training_data = [
    ("The hero bravely fought the dragon and saved the kingdom.", 1),
    ("Random words jumbled together without meaning chaos.", 0),
    ("She wrote a beautiful novel that touched many hearts.", 1),
    ("Nonsense text. Bad coherence. No story here.", 0),
]

print(f"\nTraining data: {len(training_data)} examples")
print("  Good stories (label=1): Stories with coherent plots")
print("  Bad stories (label=0): Random/incoherent text")

# Get representations for training data
cls_features = []
mean_features = []
labels = []

with torch.no_grad():
    for story, label in training_data:
        inputs = tokenizer(story, return_tensors="pt", padding=True, truncation=True)
        outputs = encoder(**inputs)

        # CLS token
        cls_feat = outputs.last_hidden_state[:, 0, :]
        cls_features.append(cls_feat)

        # Mean pooling
        mean_feat = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        mean_features.append(mean_feat)

        labels.append(label)

cls_features = torch.cat(cls_features, dim=0)  # [4, 768]
mean_features = torch.cat(mean_features, dim=0)  # [4, 768]
labels = torch.tensor(labels)

# Simple gradient descent
lr = 0.01
epochs = 100

print(f"\nTraining simple classifiers (lr={lr}, epochs={epochs})...")

# Train CLS classifier
optimizer_cls = torch.optim.Adam(cls_classifier.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

cls_losses = []
for epoch in range(epochs):
    logits = cls_classifier(cls_features)
    loss = loss_fn(logits, labels)
    optimizer_cls.zero_grad()
    loss.backward()
    optimizer_cls.step()
    cls_losses.append(loss.item())

# Train Mean classifier
optimizer_mean = torch.optim.Adam(mean_classifier.parameters(), lr=lr)

mean_losses = []
for epoch in range(epochs):
    logits = mean_classifier(mean_features)
    loss = loss_fn(logits, labels)
    optimizer_mean.zero_grad()
    loss.backward()
    optimizer_mean.step()
    mean_losses.append(loss.item())

print(f"\n  [CLS] Token Classifier:")
print(f"    Initial loss: {cls_losses[0]:.4f}")
print(f"    Final loss: {cls_losses[-1]:.4f}")
print(f"    Loss reduction: {(1 - cls_losses[-1]/cls_losses[0])*100:.1f}%")

print(f"\n  Mean Pooling Classifier:")
print(f"    Initial loss: {mean_losses[0]:.4f}")
print(f"    Final loss: {mean_losses[-1]:.4f}")
print(f"    Loss reduction: {(1 - mean_losses[-1]/mean_losses[0])*100:.1f}%")

# Test accuracy
with torch.no_grad():
    cls_logits = cls_classifier(cls_features)
    cls_preds = cls_logits.argmax(dim=1)
    cls_acc = (cls_preds == labels).float().mean()

    mean_logits = mean_classifier(mean_features)
    mean_preds = mean_logits.argmax(dim=1)
    mean_acc = (mean_preds == labels).float().mean()

print(f"\n  Training Accuracy:")
print(f"    [CLS] Token: {cls_acc.item()*100:.1f}%")
print(f"    Mean Pooling: {mean_acc.item()*100:.1f}%")

print("\n" + "="*80)
print("Summary")
print("="*80)
print("""
Expected observations:

1. [CLS] Token Issues:
   - All CLS tokens have similar norms (~1.5)
   - High cosine similarity even for different stories (>0.95)
   - Poor discrimination between good/bad stories
   - Classification loss may not decrease well

2. Mean Pooling Benefits:
   - Larger representation norms (~4-6)
   - Better discrimination between different stories
   - Lower similarity for different content
   - Should converge better in classification task

Recommendation: Use MEAN POOLING for LED encoder classification tasks!
""")
