#!/usr/bin/env python
# coding=utf-8
"""
Quick test script to verify TensorFlow compatibility setup works.
Run this before the full test to ensure TF2 compatibility is working.
"""

import sys
print("Testing TensorFlow compatibility...")

# Import compatibility setup
import tf2_compat_setup

# Test TensorFlow imports
import tensorflow as tf
print(f"✓ TensorFlow version: {tf.__version__}")

# Test TF1 attributes are available
try:
    assert hasattr(tf, 'gfile'), "tf.gfile not found"
    print("✓ tf.gfile is available")

    assert hasattr(tf, 'logging'), "tf.logging not found"
    print("✓ tf.logging is available")

    assert hasattr(tf, 'flags'), "tf.flags not found"
    print("✓ tf.flags is available")

    assert hasattr(tf, 'train'), "tf.train not found"
    print("✓ tf.train is available")

    assert hasattr(tf, 'contrib'), "tf.contrib not found"
    print("✓ tf.contrib is available")

except AssertionError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test UNION module imports
try:
    import union_modeling as modeling
    print("✓ union_modeling imported successfully")

    import tokenization
    print("✓ tokenization imported successfully")

except ImportError as e:
    print(f"✗ Error importing UNION modules: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("SUCCESS! TensorFlow compatibility is working.")
print("You can now run: python test_union_accuracy_tf2.py")
print("="*60)
