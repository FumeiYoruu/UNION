# Quick Test Guide - UNION Model Accuracy

## TL;DR - Run This Now

Since you have **TensorFlow 2.x** installed, use these commands:

```bash
# Make sure you're in the UNION directory
cd /home1/tanalvin/Desktop/UNION

# Step 1: Test TensorFlow compatibility (optional but recommended)
python test_tf_compat.py

# Step 2: Run the accuracy test
python test_union_accuracy_tf2.py \
    --data_dir /scratch1/tanalvin/Data/WritingPrompts \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --bert_config_file ./model/uncased_L-12_H-768_A-12/bert_config.json \
    --vocab_file ./model/uncased_L-12_H-768_A-12/vocab.txt \
    --output_dir ./test_results
```

**Note**: The script uses `tf2_compat_setup.py` to patch TensorFlow 2.x for compatibility with the original UNION code.

## What You Need

✅ **You already have:**
- Fine-tuned UNION checkpoint at `./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt`
- BERT config and vocab files
- TensorFlow 2.x installed

❓ **You need to verify:**
- Test data exists at `/scratch1/tanalvin/Data/WritingPrompts/train_data/`
  - `test_human.txt` (human-written stories)
  - `test_negative.txt` (negative samples)

## Check if Data Exists

```bash
# Check if test data exists
ls -la /scratch1/tanalvin/Data/WritingPrompts/train_data/test_*.txt

# If files don't exist, you need to generate them first
# See Data Preparation section in TEST_ACCURACY_README.md
```

## Two Scripts Available

| Script | TensorFlow Version | When to Use |
|--------|-------------------|-------------|
| `test_union_accuracy.py` | 1.14.0 | Original UNION environment |
| `test_union_accuracy_tf2.py` | 2.x | Modern systems (your case) |

## Expected Output

```
Loaded 1000 human stories and 1000 negative stories
Total test examples: 2000

Creating examples...
Converting to features...
Building model...
Running predictions...

============================================================
CLASSIFICATION METRICS
============================================================
Accuracy:  0.92-0.94  (expected range for WritingPrompts)
Precision: 0.91-0.95
Recall:    0.90-0.94
F1 Score:  0.92-0.94
...

Results saved to:
  Metrics: ./test_results/test_metrics.txt
  Predictions: ./test_results/predictions.txt
```

## Common Issues

### Issue: "AttributeError: module 'tensorflow' has no attribute 'flags'" or "no attribute 'gfile'"
**Solution**: You're using TF2 but ran the TF1 script. Use `test_union_accuracy_tf2.py` instead.

The TF2 script automatically patches TensorFlow using `tf2_compat_setup.py` to add missing TF1 attributes (`tf.gfile`, `tf.flags`, etc.).

### Issue: "Test data files not found"
**Solution**: Generate test data:
```bash
cd /scratch1/tanalvin/Data
# Follow data preparation instructions
```

### Issue: "Checkpoint not found"
**Solution**: Your checkpoint is at:
```bash
ls -la ./model/uncased_L-12_H-768_A-12/union_wp/
# Should show: model.ckpt.data-*, model.ckpt.index, model.ckpt.meta, checkpoint
```

## GPU Usage

**Yes, the code uses GPU by default if available!**

When you run the compatibility test or main script, you'll see:
```
GPU(s) available: 1 device(s)
  GPU 0: /physical_device:GPU:0
GPU acceleration will be used by default
```

**To force CPU-only** (not recommended - 20-40x slower):
```bash
python test_union_accuracy_tf2.py --no-use_gpu ...
```

**To use specific GPU** (if you have multiple):
```bash
python test_union_accuracy_tf2.py --gpu_device 1 ...
```

See `GPU_SETUP.md` for detailed GPU configuration.

## Understanding Results

After running, check:

```bash
# Summary metrics
cat ./test_results/test_metrics.txt

# Per-example predictions (first 10)
head -10 ./test_results/predictions.txt
```

**Metrics Meaning:**
- **Accuracy**: Overall % correct (should be ~92-94% for WP)
- **Precision**: Of stories predicted as "human", % that are actually human
- **Recall**: Of actual human stories, % correctly identified
- **F1**: Balance between precision and recall

## Full Documentation

For complete details, see: `TEST_ACCURACY_README.md`

## What's Next

Once you have the accuracy results, you can:
1. Compare with paper benchmarks (EMNLP 2020)
2. Test on your own custom stories
3. Compare different checkpoints
4. Analyze error cases (false positives/negatives)
