# coding=utf-8
"""
Test script to evaluate UNION model accuracy on WritingPrompts test dataset.
TensorFlow 2.x compatible version using argparse instead of tf.flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

# IMPORTANT: Import TF2 compatibility setup BEFORE any other imports
import tf2_compat_setup

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import tensorflow as tf
import union_modeling as modeling
import tokenization


class InputExample(object):
    """A single test example for sequence classification."""

    def __init__(self, guid, text_a, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.label = label
        self.ref = None


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.ref_input_ids = [0] * len(input_ids)
        self.ref_input_mask = [0] * len(input_ids)
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def load_test_data(data_dir):
    """Load human-written and negative stories from test set."""

    def _read_line(fin):
        """Read stories from file (WritingPrompts format: one story per line)."""
        stories = [[line.strip()] for line in fin.readlines()]
        return stories

    # Load human-written stories (label=1)
    human_file = os.path.join(data_dir, "train_data/test_human.txt")
    with tf.gfile.Open(human_file, "r") as fin:
        human_stories = _read_line(fin)
    human_labels = [1] * len(human_stories)

    # Load negative stories (label=0)
    negative_file = os.path.join(data_dir, "train_data/test_negative.txt")
    with tf.gfile.Open(negative_file, "r") as fin:
        negative_stories = _read_line(fin)
    negative_labels = [0] * len(negative_stories)

    # Combine
    all_stories = human_stories + negative_stories
    all_labels = human_labels + negative_labels

    print(f"Loaded {len(human_stories)} human stories and {len(negative_stories)} negative stories")
    print(f"Total test examples: {len(all_stories)}")

    return all_stories, all_labels


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    """Converts a single InputExample into a single InputFeatures."""

    def token(text):
        if isinstance(text, str):
            token_text = tokenizer.tokenize(text)
            return token_text
        elif isinstance(text, list):
            # For WritingPrompts: list with single story
            token_text = [tokenizer.tokenize(t) for t in text]
            return [tok for sublist in token_text for tok in sublist]  # Flatten

    tokens_a = token(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=example.label,
        is_real_example=True
    )
    return feature


def create_examples(stories, labels):
    """Creates InputExample objects from stories and labels."""
    examples = []
    for i, (story, label) in enumerate(zip(stories, labels)):
        guid = f"test-{i}"
        text_a = [tokenization.convert_to_unicode(s) for s in story]
        examples.append(InputExample(guid=guid, text_a=text_a, label=label))
    return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Convert a set of InputExamples to a list of InputFeatures."""
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            print(f"Converting example {ex_index}/{len(examples)}")

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer)
        features.append(feature)
    return features


def input_fn_builder(features, seq_length, batch_size):
    """Creates an input_fn closure to be passed to TPUEstimator."""

    all_input_ids = [f.input_ids for f in features]
    all_input_mask = [f.input_mask for f in features]
    all_segment_ids = [f.segment_ids for f in features]
    all_label_ids = [f.label_id for f in features]

    def input_fn(params):
        """The actual input function."""
        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "input_mask": tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
            "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "label_ids": tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model (same as run_union.py)."""

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )

    with tf.variable_scope("loss"):
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)

        return logits, probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, use_tpu):
    """Returns model_fn closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        logits, probabilities = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, num_labels, use_one_hot_embeddings=use_tpu
        )

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        predictions = {
            "probabilities": probabilities,
            "predictions": tf.argmax(logits, axis=-1, output_type=tf.int32),
            "labels": label_ids
        }

        # Use regular EstimatorSpec for GPU/CPU instead of TPUEstimatorSpec
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

        return output_spec

    return model_fn


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate classification metrics."""

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1
    )

    conf_matrix = confusion_matrix(y_true, y_pred)

    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"Actual  Neg   [[{conf_matrix[0,0]:5d}  {conf_matrix[0,1]:5d}]]")
    print(f"        Pos   [[{conf_matrix[1,0]:5d}  {conf_matrix[1,1]:5d}]]")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Human'], digits=4))
    print("="*60)

    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])

    print("\nPer-Class Metrics:")
    print(f"Negative Stories (label=0):")
    print(f"  Precision: {precision_per_class[0]:.4f}")
    print(f"  Recall:    {recall_per_class[0]:.4f}")
    print(f"  F1:        {f1_per_class[0]:.4f}")
    print(f"  Support:   {support[0]}")

    print(f"\nHuman Stories (label=1):")
    print(f"  Precision: {precision_per_class[1]:.4f}")
    print(f"  Recall:    {recall_per_class[1]:.4f}")
    print(f"  F1:        {f1_per_class[1]:.4f}")
    print(f"  Support:   {support[1]}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Test UNION model accuracy')

    # Required parameters
    parser.add_argument('--data_dir', type=str, default='./Data/WP',
                        help='The input data dir containing train_data/ folder')
    parser.add_argument('--init_checkpoint', type=str,
                        default='./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt',
                        help='Initial checkpoint (fine-tuned UNION model)')
    parser.add_argument('--bert_config_file', type=str,
                        default='./model/uncased_L-12_H-768_A-12/bert_config.json',
                        help='The config json file corresponding to the pre-trained BERT model')
    parser.add_argument('--vocab_file', type=str,
                        default='./model/uncased_L-12_H-768_A-12/vocab.txt',
                        help='The vocabulary file that the BERT model was trained on')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='The output directory where results will be written')

    # Other parameters
    parser.add_argument('--do_lower_case', action='store_true', default=True,
                        help='Whether to lower case the input text')
    parser.add_argument('--max_seq_length', type=int, default=200,
                        help='Maximum total input sequence length after WordPiece tokenization')
    parser.add_argument('--predict_batch_size', type=int, default=32,
                        help='Total batch size for predictions')
    parser.add_argument('--use_reconstruction', action='store_true', default=False,
                        help='Whether the model was trained with reconstruction task')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Whether to use GPU if available (default: True)')
    parser.add_argument('--gpu_device', type=str, default='0',
                        help='GPU device to use (e.g., "0" or "0,1" for multiple GPUs)')

    args = parser.parse_args()

    # GPU configuration
    if not args.use_gpu:
        print("GPU disabled by user - using CPU only")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
        print(f"GPU device(s) set to: {args.gpu_device}")

    tf.logging.set_verbosity(tf.logging.INFO)

    # Check that data directory exists
    if not tf.gfile.Exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")

    test_human_file = os.path.join(args.data_dir, "train_data/test_human.txt")
    test_negative_file = os.path.join(args.data_dir, "train_data/test_negative.txt")

    if not tf.gfile.Exists(test_human_file) or not tf.gfile.Exists(test_negative_file):
        raise ValueError(
            f"Test data files not found in {args.data_dir}/train_data/\n"
            f"Please ensure test_human.txt and test_negative.txt exist.\n"
            f"You may need to run: cd Data && python download_and_prepare_wp.py"
        )

    # Load BERT config
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            f"Cannot use sequence length {args.max_seq_length} because BERT model "
            f"was only trained up to sequence length {bert_config.max_position_embeddings}"
        )

    # Create output directory
    tf.gfile.MakeDirs(args.output_dir)

    # Load tokenizer
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case
    )

    # Load test data
    print("\nLoading test data...")
    stories, labels = load_test_data(args.data_dir)

    # Create examples and features
    print("\nCreating examples...")
    examples = create_examples(stories, labels)

    print("\nConverting to features...")
    features = convert_examples_to_features(examples, args.max_seq_length, tokenizer)

    # Build model
    print("\nBuilding model...")
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint=args.init_checkpoint,
        use_tpu=False
    )

    # Create estimator - use regular Estimator for GPU/CPU instead of TPUEstimator
    run_config = tf.estimator.RunConfig(
        model_dir=args.output_dir,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=1
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    # Run predictions
    print("\nRunning predictions...")
    predict_input_fn = input_fn_builder(
        features=features,
        seq_length=args.max_seq_length,
        batch_size=args.predict_batch_size
    )

    result = list(estimator.predict(input_fn=predict_input_fn))

    # Extract predictions and probabilities
    y_pred = [r["predictions"] for r in result]
    y_probs = [r["probabilities"][1] for r in result]  # Probability of class 1 (human)
    y_true = labels

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs)

    # Save results
    output_file = os.path.join(args.output_dir, "test_metrics.txt")
    with tf.gfile.GFile(output_file, "w") as writer:
        writer.write("UNION Model Test Results\n")
        writer.write("="*60 + "\n")
        writer.write(f"Data directory: {args.data_dir}\n")
        writer.write(f"Model checkpoint: {args.init_checkpoint}\n")
        writer.write(f"Total test examples: {len(y_true)}\n")
        writer.write(f"Human stories: {sum(y_true)}\n")
        writer.write(f"Negative stories: {len(y_true) - sum(y_true)}\n\n")

        writer.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        writer.write(f"Precision: {metrics['precision']:.4f}\n")
        writer.write(f"Recall:    {metrics['recall']:.4f}\n")
        writer.write(f"F1 Score:  {metrics['f1']:.4f}\n\n")

        writer.write("Confusion Matrix:\n")
        writer.write(f"{metrics['confusion_matrix']}\n\n")

        writer.write("Per-Class Metrics:\n")
        writer.write(f"Negative: P={metrics['precision_per_class'][0]:.4f}, "
                    f"R={metrics['recall_per_class'][0]:.4f}, "
                    f"F1={metrics['f1_per_class'][0]:.4f}\n")
        writer.write(f"Human: P={metrics['precision_per_class'][1]:.4f}, "
                    f"R={metrics['recall_per_class'][1]:.4f}, "
                    f"F1={metrics['f1_per_class'][1]:.4f}\n")

    # Save predictions
    predictions_file = os.path.join(args.output_dir, "predictions.txt")
    with tf.gfile.GFile(predictions_file, "w") as writer:
        writer.write("ID\tTrue_Label\tPred_Label\tProbability\n")
        for i, (true_label, pred_label, prob) in enumerate(zip(y_true, y_pred, y_probs)):
            writer.write(f"{i}\t{true_label}\t{pred_label}\t{prob:.6f}\n")

    print(f"\nResults saved to:")
    print(f"  Metrics: {output_file}")
    print(f"  Predictions: {predictions_file}")


if __name__ == "__main__":
    main()
