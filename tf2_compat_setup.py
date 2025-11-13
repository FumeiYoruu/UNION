# coding=utf-8
"""
TensorFlow 2.x compatibility setup for UNION model.
This script must be imported BEFORE any UNION modules to patch TF compatibility.
"""

import sys
import tensorflow as tf

# Disable TF2 behavior and use TF1 compatibility mode
if tf.__version__.startswith('2.'):
    import tensorflow.compat.v1 as tf_v1
    tf_v1.disable_v2_behavior()

    # Patch the main tensorflow module to add TF1 attributes
    if not hasattr(tf, 'gfile'):
        tf.gfile = tf.io.gfile

    if not hasattr(tf, 'logging'):
        tf.logging = tf_v1.logging

    if not hasattr(tf, 'flags'):
        tf.flags = tf_v1.flags

    if not hasattr(tf, 'train'):
        tf.train = tf_v1.train

    if not hasattr(tf, 'contrib'):
        tf.contrib = tf_v1.contrib

    # Replace tensorflow in sys.modules so all imports use patched version
    sys.modules['tensorflow'] = tf_v1

    print(f"TensorFlow 2.x detected (version {tf.__version__})")
    print("Enabled TF1 compatibility mode")
else:
    print(f"TensorFlow 1.x detected (version {tf.__version__})")
