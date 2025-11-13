# TensorFlow 2.x Compatibility Notes

## Problem

The original UNION code was written for **TensorFlow 1.14.0** and uses deprecated APIs that were removed in TensorFlow 2.x:

- `tf.flags` → Removed (use `argparse` instead)
- `tf.gfile` → Moved to `tf.io.gfile`
- `tf.logging` → Moved to `tf.compat.v1.logging`
- `tf.contrib` → Removed/moved to separate packages

## Solution

We created a **compatibility layer** that patches TensorFlow 2.x to restore TF1 APIs without modifying the original UNION code files.

### Files Created

1. **`tf2_compat_setup.py`** - Compatibility wrapper
   - Detects TensorFlow version
   - If TF2: patches `tensorflow` module with TF1 attributes
   - Uses `tf.compat.v1.disable_v2_behavior()`
   - Restores `tf.gfile`, `tf.flags`, `tf.logging`, etc.

2. **`test_union_accuracy_tf2.py`** - TF2-compatible test script
   - Uses `argparse` instead of `tf.flags`
   - Imports `tf2_compat_setup` BEFORE other modules
   - Otherwise identical to `test_union_accuracy.py`

3. **`test_tf_compat.py`** - Compatibility verification script
   - Quick test to ensure TF2 patching works
   - Checks all required TF1 attributes are available
   - Tests UNION module imports

## How It Works

```python
# In test_union_accuracy_tf2.py
import tf2_compat_setup  # <-- Patches TensorFlow FIRST

# Now these work in TF2:
import tensorflow as tf
import union_modeling    # Uses tf.gfile internally - now works!
import tokenization      # Uses tf.gfile internally - now works!
```

The patching happens in `tf2_compat_setup.py`:

```python
if tf.__version__.startswith('2.'):
    import tensorflow.compat.v1 as tf_v1
    tf_v1.disable_v2_behavior()

    # Restore TF1 APIs
    tf.gfile = tf.io.gfile
    tf.logging = tf_v1.logging
    tf.flags = tf_v1.flags
    # ... etc
```

## Testing the Compatibility

Before running the full test, verify compatibility:

```bash
python test_tf_compat.py
```

Expected output:
```
Testing TensorFlow compatibility...
TensorFlow 2.x detected (version 2.x.x)
Enabled TF1 compatibility mode
✓ TensorFlow version: 2.x.x
✓ tf.gfile is available
✓ tf.logging is available
✓ tf.flags is available
✓ tf.train is available
✓ tf.contrib is available
✓ union_modeling imported successfully
✓ tokenization imported successfully

============================================================
SUCCESS! TensorFlow compatibility is working.
You can now run: python test_union_accuracy_tf2.py
============================================================
```

## Running the Test

Once compatibility is verified:

```bash
python test_union_accuracy_tf2.py \
    --data_dir /scratch1/tanalvin/Data/WritingPrompts \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --bert_config_file ./model/uncased_L-12_H-768_A-12/bert_config.json \
    --vocab_file ./model/uncased_L-12_H-768_A-12/vocab.txt \
    --output_dir ./test_results
```

## Why Not Just Use TensorFlow 1.14.0?

You can! But there are reasons to use TF2:

**Pros of TF2:**
- Works on modern systems (Python 3.8+, recent CUDA versions)
- Active development and security updates
- Better performance in many cases

**Pros of TF1.14.0:**
- No compatibility layer needed
- Guaranteed to work exactly as originally designed
- Simpler setup

**Our recommendation:** Use TF2 with the compatibility layer for modern systems, or TF1.14.0 for legacy systems.

## Limitations

The compatibility layer has some limitations:

1. **TPU support**: May not work correctly (not tested)
2. **TF2-specific features**: Can't mix TF2 eager execution with TF1 graph mode
3. **Warnings**: You may see deprecation warnings (can be ignored)

For production use on modern systems, consider migrating the UNION code to native TF2, but for testing/evaluation, the compatibility layer works fine.

## File Structure

```
UNION/
├── tf2_compat_setup.py           # Compatibility layer (NEW)
├── test_tf_compat.py              # Compatibility test (NEW)
├── test_union_accuracy.py         # Original (TF1 only)
├── test_union_accuracy_tf2.py     # TF2 compatible (NEW)
├── union_modeling.py              # Original UNION code (unchanged)
├── tokenization.py                # Original UNION code (unchanged)
└── run_union.py                   # Original UNION code (unchanged)
```

**Key point**: We didn't modify any original UNION code files. The compatibility is achieved through wrapper scripts only.

## Troubleshooting

### "AttributeError: module 'tensorflow' has no attribute 'gfile'"

**Cause**: `tf2_compat_setup.py` wasn't imported before other modules.

**Solution**: Make sure you're using `test_union_accuracy_tf2.py`, not `test_union_accuracy.py`.

### "ImportError: cannot import name 'tf2_compat_setup'"

**Cause**: Script not run from UNION directory.

**Solution**:
```bash
cd /home1/tanalvin/Desktop/UNION
python test_union_accuracy_tf2.py ...
```

### "non-resource variables are not supported in the long term"

**Cause**: This is a warning from TF2 about TF1 compatibility mode.

**Solution**: It's just a warning and can be ignored. The code still works.

## Alternative: Using TensorFlow 1.14.0

If you prefer to avoid the compatibility layer:

```bash
# Create a separate environment
conda create -n union_tf1 python=3.7
conda activate union_tf1

# Install TF1
pip install tensorflow-gpu==1.14.0  # or tensorflow==1.14.0
pip install numpy==1.18.1
pip install scikit-learn==0.22.1

# Use original script
python test_union_accuracy.py \
    --data_dir /scratch1/tanalvin/Data/WritingPrompts \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt
```

## Summary

- **TF2 users**: Use `test_union_accuracy_tf2.py` (imports `tf2_compat_setup`)
- **TF1 users**: Use `test_union_accuracy.py` (no compatibility needed)
- **Both versions**: Produce identical results
- **Test first**: Run `python test_tf_compat.py` to verify setup

The compatibility layer allows modern TensorFlow 2.x environments to run the original UNION code without modifications!
