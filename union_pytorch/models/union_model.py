"""PyTorch UNION model with BERT and Longformer support."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from transformers import (
    AutoModel,
    AutoConfig,
    BertModel,
    LongformerModel,
    LEDForConditionalGeneration,
)


class UnionClassifier(nn.Module):
    """
    UNION classifier for story quality evaluation.

    Supports both BERT and LED (Long Encoder-Decoder) encoders with optional:
    - Multi-layer pooling
    - Reconstruction task (masked LM)

    Note: For long documents (model_type="longformer"), uses LED encoder
    which supports up to 16384 tokens.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        model_type: str = "bert",
        num_labels: int = 2,
        hidden_dropout_prob: float = 0.1,
        use_all_layers: bool = False,
        use_reconstruction: bool = False,
        reconstruction_weight: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            model_name: Pretrained model name (e.g., "bert-base-uncased", "allenai/led-base-16384")
            model_type: "bert" or "longformer" (longformer uses LED encoder)
            num_labels: Number of classification labels (default: 2 for binary classification)
            hidden_dropout_prob: Dropout probability
            use_all_layers: Whether to use multi-layer pooling
            use_reconstruction: Whether to use reconstruction task
            reconstruction_weight: Weight for reconstruction loss
            gradient_checkpointing: Enable gradient checkpointing to reduce memory usage
        """
        super().__init__()

        self.model_name = model_name
        self.model_type = model_type
        self.num_labels = num_labels
        self.use_all_layers = use_all_layers
        self.use_reconstruction = use_reconstruction
        self.reconstruction_weight = reconstruction_weight

        # Load encoder based on model type
        if model_type == "bert":
            # Standard BERT model
            self.config = AutoConfig.from_pretrained(model_name)
            self.config.hidden_dropout_prob = hidden_dropout_prob
            self.config.attention_probs_dropout_prob = hidden_dropout_prob
            self.config.output_hidden_states = use_all_layers
            self.encoder = BertModel.from_pretrained(model_name, config=self.config)
            self.hidden_size = self.config.hidden_size

        elif model_type == "longformer":
            # Use LED encoder for long documents (16384 positions)
            # LED is available while Longformer standalone is not
            print(f"Loading LED encoder from {model_name} for long document support...")
            led_model = LEDForConditionalGeneration.from_pretrained(model_name)
            self.encoder = led_model.get_encoder()

            # Update encoder config for our needs
            self.config = self.encoder.config
            self.config.output_hidden_states = use_all_layers
            self.encoder.config.output_hidden_states = use_all_layers
            self.hidden_size = self.config.d_model  # LED uses d_model instead of hidden_size

            # LED uses max_encoder_position_embeddings instead of max_position_embeddings
            max_pos = getattr(self.config, 'max_encoder_position_embeddings', 16384)
            print(f"LED encoder loaded with max_encoder_position_embeddings: {max_pos}")

        else:
            # Fallback to AutoModel
            self.config = AutoConfig.from_pretrained(model_name)
            self.config.hidden_dropout_prob = hidden_dropout_prob
            self.config.attention_probs_dropout_prob = hidden_dropout_prob
            self.config.output_hidden_states = use_all_layers
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
            self.hidden_size = self.config.hidden_size

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            if hasattr(self.encoder, 'gradient_checkpointing_enable'):
                self.encoder.gradient_checkpointing_enable()
                print(f"Gradient checkpointing enabled for {model_type} encoder")
            else:
                print(f"Warning: Gradient checkpointing not supported for this encoder")

        # Classification head
        self.dropout = nn.Dropout(hidden_dropout_prob)

        if use_all_layers:
            # Multi-layer pooling: create separate poolers for each layer
            self.num_hidden_layers = self.config.num_hidden_layers
            self.layer_poolers = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_hidden_layers)
            ])
            # Classifier takes concatenated or averaged multi-layer features
            # For simplicity, we'll average them
            classifier_input_size = self.hidden_size
        else:
            # Single layer pooling
            classifier_input_size = self.hidden_size

        self.classifier = nn.Linear(classifier_input_size, num_labels)

        # Reconstruction head (for masked LM task)
        if use_reconstruction:
            # Handle different vocab_size attribute names
            vocab_size = getattr(self.config, 'vocab_size', None)
            if vocab_size is None:
                # LED uses 'vocab_size' in config
                vocab_size = self.config.vocab_size
            self.lm_head = nn.Linear(self.hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ref_input_ids: Optional[torch.Tensor] = None,
        ref_attention_mask: Optional[torch.Tensor] = None,
        ref_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length]
            labels: Classification labels [batch_size]
            ref_input_ids: Reference input IDs for reconstruction [batch_size, seq_length]
            ref_attention_mask: Reference attention mask [batch_size, seq_length]
            ref_labels: Reference labels for reconstruction [batch_size, seq_length]

        Returns:
            Dictionary with loss, logits, and optional reconstruction loss
        """
        outputs = {}

        # Encode main story
        # Note: LED encoder doesn't support token_type_ids
        if self.model_type == "bert" and token_type_ids is not None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Get pooled output
        if self.use_all_layers:
            # Multi-layer pooling: pool [CLS] token from all layers
            all_hidden_states = encoder_outputs.hidden_states  # Tuple of (batch_size, seq_length, hidden_size)

            pooled_outputs = []
            for i, hidden_state in enumerate(all_hidden_states[1:]):  # Skip embedding layer
                # Extract [CLS] token (first token)
                cls_token = hidden_state[:, 0, :]  # [batch_size, hidden_size]
                # Apply layer-specific pooler with tanh activation
                pooled = torch.tanh(self.layer_poolers[i](cls_token))
                pooled_outputs.append(pooled)

            # Average pooled outputs from all layers
            pooled_output = torch.stack(pooled_outputs).mean(dim=0)  # [batch_size, hidden_size]
        else:
            # Use pooled output from encoder (or first token of last hidden state)
            if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
                pooled_output = encoder_outputs.pooler_output
            else:
                # Longformer doesn't have pooler_output, use [CLS] token
                pooled_output = encoder_outputs.last_hidden_state[:, 0, :]

        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs["logits"] = logits

        # Compute classification loss
        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            total_loss = classification_loss
            outputs["classification_loss"] = classification_loss

        # Reconstruction task
        if self.use_reconstruction and ref_input_ids is not None:
            # LED encoder doesn't support token_type_ids
            if self.model_type == "bert" and token_type_ids is not None:
                ref_outputs = self.encoder(
                    input_ids=ref_input_ids,
                    attention_mask=ref_attention_mask,
                    token_type_ids=token_type_ids,
                )
            else:
                ref_outputs = self.encoder(
                    input_ids=ref_input_ids,
                    attention_mask=ref_attention_mask,
                )

            # Get sequence output for masked LM
            sequence_output = ref_outputs.last_hidden_state

            # Predict tokens
            prediction_scores = self.lm_head(sequence_output)

            outputs["reconstruction_logits"] = prediction_scores

            # Compute reconstruction loss
            if ref_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                vocab_size = getattr(self.config, 'vocab_size', self.config.vocab_size)
                reconstruction_loss = loss_fct(
                    prediction_scores.view(-1, vocab_size),
                    ref_labels.view(-1)
                )
                outputs["reconstruction_loss"] = reconstruction_loss

                # Add to total loss
                if total_loss is not None:
                    total_loss = total_loss + self.reconstruction_weight * reconstruction_loss
                else:
                    total_loss = self.reconstruction_weight * reconstruction_loss

        outputs["loss"] = total_loss

        return outputs

    def get_encoder(self):
        """Get the encoder model."""
        return self.encoder

    def freeze_encoder(self, freeze: bool = True):
        """Freeze or unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def freeze_layers(self, num_layers: int):
        """Freeze first N layers of encoder."""
        if self.model_type == "bert":
            layers = self.encoder.encoder.layer
        elif self.model_type == "longformer":
            layers = self.encoder.encoder.layer
        else:
            return

        for i in range(min(num_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False


def create_model(
    model_type: str = "bert",
    model_name: Optional[str] = None,
    **kwargs
) -> UnionClassifier:
    """
    Create a UNION model.

    Args:
        model_type: "bert" or "longformer"
        model_name: Optional model name (uses default if None)
        **kwargs: Additional arguments for UnionClassifier

    Returns:
        UnionClassifier model
    """
    # Default model names
    default_models = {
        "bert": "bert-base-uncased",
        "longformer": "allenai/led-base-16384",  # Use LED encoder for long documents
    }

    if model_name is None:
        model_name = default_models.get(model_type, "bert-base-uncased")

    return UnionClassifier(
        model_name=model_name,
        model_type=model_type,
        **kwargs
    )
