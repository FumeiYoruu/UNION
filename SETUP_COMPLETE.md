# UNION Accuracy Testing - Setup Complete! 🎉

## Quick Start (TL;DR)

You're ready to test! Run these commands:

```bash
cd /home1/tanalvin/Desktop/UNION

# Step 1: Verify TensorFlow compatibility
python test_tf_compat.py

# Step 2: Run accuracy test (uses GPU by default)
python test_union_accuracy_tf2.py \
    --data_dir /scratch1/tanalvin/Data/WritingPrompts \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --bert_config_file ./model/uncased_L-12_H-768_A-12/bert_config.json \
    --vocab_file ./model/uncased_L-12_H-768_A-12/vocab.txt \
    --output_dir ./test_results
```

## What Was Fixed

### Problem 1: `AttributeError: module 'tensorflow' has no attribute 'flags'`
**Solution**: Created `tf2_compat_setup.py` that patches TensorFlow 2.x with TF1 APIs

### Problem 2: `AttributeError: module 'tensorflow' has no attribute 'gfile'`
**Solution**: Patches `tf.gfile` → `tf.io.gfile` in the compatibility layer

### Problem 3: `module 'tensorflow.compat.v1' has no attribute 'contrib'`
**Solution**: Created a custom `tf.contrib` stub that maps to `tf.compat.v1.estimator.tpu`

### Question: "Does this code use CUDA on default?"
**Answer**: **YES!** GPU is used automatically if available. See `GPU_SETUP.md` for details.

## Files Created

### Core Files (Use These!)
1. **`test_union_accuracy_tf2.py`** - Main test script for TF2
2. **`tf2_compat_setup.py`** - TensorFlow compatibility layer
3. **`test_tf_compat.py`** - Verify compatibility setup

### Documentation
4. **`QUICK_TEST_GUIDE.md`** - Quick reference (read this first!)
5. **`TEST_ACCURACY_README.md`** - Comprehensive guide
6. **`TF2_COMPATIBILITY_NOTES.md`** - Technical details on TF1/TF2 compatibility
7. **`GPU_SETUP.md`** - GPU configuration and troubleshooting
8. **`SETUP_COMPLETE.md`** - This file!

### Original Files (For TF1 users)
9. **`test_union_accuracy.py`** - TF1 version (requires tensorflow==1.14.0)

## How the Compatibility Works

```
┌─────────────────────────────────────────────────────┐
│ test_union_accuracy_tf2.py                          │
│   ↓                                                  │
│ import tf2_compat_setup  ← Patches TensorFlow       │
│   ↓                                                  │
│ import tensorflow as tf  ← Now has TF1 APIs         │
│   ↓                                                  │
│ import union_modeling    ← Uses tf.gfile (patched)  │
│ import tokenization      ← Uses tf.gfile (patched)  │
└─────────────────────────────────────────────────────┘

Key patches:
- tf.gfile → tf.io.gfile
- tf.flags → tf.compat.v1.flags
- tf.contrib → Custom stub with TPUEstimator
- tf.logging → tf.compat.v1.logging
```

## GPU Configuration

### Default Behavior
✅ **GPU is used automatically** if CUDA is installed and GPU is available

### How to Verify GPU Usage

Run the compatibility test:
```bash
python test_tf_compat.py
```

You should see:
```
GPU(s) available: 1 device(s)
  GPU 0: /physical_device:GPU:0
GPU acceleration will be used by default
```

### GPU Options

| Flag | Example | Description |
|------|---------|-------------|
| Default | (no flag) | Uses GPU 0 automatically |
| `--gpu_device N` | `--gpu_device 1` | Use specific GPU |
| `--gpu_device "0,1"` | `--gpu_device "0,1"` | Make GPUs 0,1 visible (doesn't parallelize) |
| `--no-use_gpu` | `--no-use_gpu` | Force CPU-only (slow!) |

### Monitor GPU Usage

In another terminal while test is running:
```bash
watch -n 1 nvidia-smi
```

## Expected Results

### Performance
- **With GPU**: 3-5 minutes for 2000 test examples
- **Without GPU**: 2-4 hours for 2000 test examples (40x slower!)

### Accuracy
Based on UNION paper (EMNLP 2020):
- **WritingPrompts**: ~92-94% accuracy
- **ROCStories**: ~94-96% accuracy

### Output Files

After running, check:
```bash
# Summary metrics
cat ./test_results/test_metrics.txt

# Per-example predictions
head -20 ./test_results/predictions.txt
```

Example output:
```
UNION Model Test Results
============================================================
Data directory: /scratch1/tanalvin/Data/WritingPrompts
Model checkpoint: ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt
Total test examples: 2000
Human stories: 1000
Negative stories: 1000

Accuracy:  0.9350
Precision: 0.9423
Recall:    0.9270
F1 Score:  0.9346

Confusion Matrix:
[[945, 55], [73, 927]]

Per-Class Metrics:
Negative: P=0.9283, R=0.9450, F1=0.9366
Human: P=0.9423, R=0.9270, F1=0.9346
```

## Troubleshooting

### Still Getting TensorFlow Errors?

Make sure you're using the **TF2 script**:
```bash
# Wrong (TF1 only):
python test_union_accuracy.py ...

# Correct (TF2 compatible):
python test_union_accuracy_tf2.py ...
```

### Test Data Not Found?

Your data should be at:
```
/scratch1/tanalvin/Data/WritingPrompts/
├── train_data/
│   ├── test_human.txt      ← Need this
│   └── test_negative.txt   ← Need this
```

If missing, see `TEST_ACCURACY_README.md` section "Data Preparation"

### GPU Not Detected?

Check CUDA installation:
```bash
nvidia-smi
nvcc --version
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

See `GPU_SETUP.md` for detailed troubleshooting.

## Next Steps

1. **Run the test** using the command at the top of this file
2. **Check results** in `./test_results/test_metrics.txt`
3. **Compare with paper** (should get ~92-94% accuracy on WP)
4. **Analyze errors** using `./test_results/predictions.txt`

## Files You Can Ignore

- `test_union_accuracy.py` - TF1 version (unless you install TF 1.14.0)
- `union_modeling.py`, `tokenization.py` - Original UNION code (unchanged)
- `run_union.py` - Training script (not needed for testing)

## Summary

✅ **TensorFlow 2.x compatibility** - Fixed with `tf2_compat_setup.py`
✅ **GPU acceleration** - Enabled by default
✅ **Comprehensive metrics** - Accuracy, precision, recall, F1, confusion matrix
✅ **Easy to use** - Single command to run
✅ **Well documented** - 5 guide files covering all aspects

You're all set to test the UNION model! 🚀

## Questions?

- **How it works**: Read `TF2_COMPATIBILITY_NOTES.md`
- **GPU setup**: Read `GPU_SETUP.md`
- **Quick reference**: Read `QUICK_TEST_GUIDE.md`
- **Full details**: Read `TEST_ACCURACY_README.md`
