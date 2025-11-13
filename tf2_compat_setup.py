# coding=utf-8
"""
TensorFlow 2.x compatibility setup for UNION model.
This script must be imported BEFORE any UNION modules to patch TF compatibility.
"""

import sys
import os

# Set environment variables for GPU usage
# TensorFlow will use GPU by default if available
# To disable GPU, uncomment: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

    # tf.contrib was completely removed in TF2, need to create a stub
    # The UNION code uses tf.contrib.tpu which we need to provide
    if not hasattr(tf, 'contrib'):
        # Import TPU components first (outside class definition)
        try:
            # Try new location first (TF 2.x)
            from tensorflow.python.estimator import estimator_lib as estimator
            tpu_estimator = estimator.tpu
        except (ImportError, AttributeError):
            # Fallback to compat.v1 path
            try:
                from tensorflow.compat.v1.estimator import tpu as tpu_estimator
            except ImportError:
                # Final fallback - create minimal stubs
                print("WARNING: Could not import TPU estimator. Creating minimal stubs.")
                class MinimalTPU:
                    pass
                tpu_estimator = MinimalTPU()

        # Create a minimal stub for tf.contrib.tpu
        class ContribStub:
            pass

        class TPUStub:
            # Map to actual TPU components if available
            TPUEstimator = getattr(tpu_estimator, 'TPUEstimator', None)
            TPUEstimatorSpec = getattr(tpu_estimator, 'TPUEstimatorSpec', None)
            TPUConfig = getattr(tpu_estimator, 'TPUConfig', None)
            RunConfig = getattr(tpu_estimator, 'RunConfig', None)
            InputPipelineConfig = getattr(tpu_estimator, 'InputPipelineConfig', None)

        contrib_stub = ContribStub()
        contrib_stub.tpu = TPUStub()
        tf.contrib = contrib_stub

    # Replace tensorflow in sys.modules so all imports use patched version
    sys.modules['tensorflow'] = tf_v1

    print(f"TensorFlow 2.x detected (version {tf.__version__})")
    print("Enabled TF1 compatibility mode")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        print("GPU acceleration will be used by default")
    else:
        print("No GPU detected - using CPU")
        print("Note: CPU-only execution will be MUCH slower")
else:
    print(f"TensorFlow 1.x detected (version {tf.__version__})")

    # Check GPU for TF1
    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    gpus = [x for x in local_devices if x.device_type == 'GPU']
    if gpus:
        print(f"GPU(s) available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  {gpu.name}: {gpu.physical_device_desc}")
    else:
        print("No GPU detected - using CPU")
