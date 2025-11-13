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

    # Import TPU components first (outside class definition)
    try:
        # Try the estimator path that usually works
        from tensorflow.compat.v1.estimator import tpu as tpu_module
        # RunConfig is in the main estimator module, not tpu submodule
        from tensorflow.compat.v1 import estimator as estimator_module
        tpu_success = True
    except (ImportError, AttributeError) as e:
        try:
            # Alternative: Try importing from estimator directly
            import tensorflow.estimator as estimator_module
            tpu_module = estimator_module.tpu
            tpu_success = True
        except (ImportError, AttributeError):
            # Final fallback
            print("WARNING: Could not import TPU estimator from standard paths.")
            tpu_module = None
            estimator_module = None
            tpu_success = False

    # Create contrib.tpu stub
    class ContribTPU:
        """Stub for tf.contrib.tpu that maps to compat.v1.estimator.tpu"""
        def __init__(self):
            if tpu_success and tpu_module:
                # TPU-specific classes from tpu module
                self.TPUEstimator = getattr(tpu_module, 'TPUEstimator', None)
                self.TPUEstimatorSpec = getattr(tpu_module, 'TPUEstimatorSpec', None)
                self.TPUConfig = getattr(tpu_module, 'TPUConfig', None)
                self.InputPipelineConfig = getattr(tpu_module, 'InputPipelineConfig', None)

                # RunConfig is in the main estimator module
                if estimator_module:
                    self.RunConfig = getattr(estimator_module, 'RunConfig',
                                            getattr(tpu_module, 'RunConfig', None))
                else:
                    self.RunConfig = getattr(tpu_module, 'RunConfig', None)
            else:
                # If TPU module not found, these will be None
                self.TPUEstimator = None
                self.TPUEstimatorSpec = None
                self.TPUConfig = None
                self.RunConfig = None
                self.InputPipelineConfig = None

    class Contrib:
        """Stub for tf.contrib"""
        def __init__(self):
            self.tpu = ContribTPU()

    # Patch both tf and tf_v1 to have the contrib attribute
    contrib = Contrib()
    tf.contrib = contrib
    tf_v1.contrib = contrib

    # Patch tf.gfile - need to add TF1 method names (capitalized) to TF2's gfile
    class GFileWrapper:
        """Wrapper to add TF1-style capitalized method names to tf.io.gfile"""
        def __init__(self, gfile_module):
            self._gfile = gfile_module

        def __getattr__(self, name):
            # First, try to get the attribute directly (for things like GFile)
            if hasattr(self._gfile, name):
                return getattr(self._gfile, name)

            # Special mappings for TF1 -> TF2
            # In TF1: tf.gfile.Open() returns file object
            # In TF2: tf.io.gfile.GFile() returns file object
            if name == 'Open':
                return self._gfile.GFile

            # For TF1 compatibility, map capitalized methods to lowercase
            # TF1: Exists, MakeDirs, etc.
            # TF2: exists, makedirs, etc.
            lowercase_name = name.lower()

            # Try lowercase version
            if hasattr(self._gfile, lowercase_name):
                return getattr(self._gfile, lowercase_name)

            # If nothing works, raise AttributeError
            raise AttributeError(f"module 'tf.gfile' has no attribute '{name}'. "
                               f"Available: {dir(self._gfile)}")

    gfile_wrapper = GFileWrapper(tf.io.gfile)
    tf.gfile = gfile_wrapper
    tf_v1.gfile = gfile_wrapper

    if not hasattr(tf, 'logging'):
        tf.logging = tf_v1.logging

    if not hasattr(tf, 'flags'):
        tf.flags = tf_v1.flags

    if not hasattr(tf, 'train'):
        tf.train = tf_v1.train

    # Add estimator module - it's a separate import in TF2
    try:
        from tensorflow.compat.v1 import estimator as tf1_estimator
        tf.estimator = tf1_estimator
        tf_v1.estimator = tf1_estimator
    except ImportError:
        # Fallback - estimator might be in a different location
        try:
            import tensorflow_estimator as tf_est
            tf.estimator = tf_est
            tf_v1.estimator = tf_est
        except ImportError:
            print("WARNING: Could not import estimator module")
            pass

    # Replace tensorflow in sys.modules
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
