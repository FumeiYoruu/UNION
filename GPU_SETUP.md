# GPU Configuration Guide

## GPU Usage by Default

**Yes, the code uses CUDA/GPU by default if available.**

TensorFlow automatically detects and uses GPUs when:
1. CUDA and cuDNN are properly installed
2. GPU-compatible TensorFlow is installed (`tensorflow-gpu` for TF1 or `tensorflow>=2.0` for TF2)
3. GPUs are visible to TensorFlow

## Checking GPU Availability

Run the compatibility test to see GPU status:

```bash
python test_tf_compat.py
```

Expected output with GPU:
```
TensorFlow 2.x detected (version 2.x.x)
Enabled TF1 compatibility mode
GPU(s) available: 1 device(s)
  GPU 0: /physical_device:GPU:0
GPU acceleration will be used by default
✓ TensorFlow version: 2.x.x
...
```

Expected output without GPU (CPU-only):
```
TensorFlow 2.x detected (version 2.x.x)
Enabled TF1 compatibility mode
No GPU detected - using CPU
Note: CPU-only execution will be MUCH slower
...
```

## GPU Configuration Options

### Use Default GPU (GPU 0)

```bash
# Default - uses first available GPU
python test_union_accuracy_tf2.py \
    --data_dir /scratch1/tanalvin/Data/WritingPrompts \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt
```

### Use Specific GPU

```bash
# Use GPU 1
python test_union_accuracy_tf2.py \
    --data_dir /scratch1/tanalvin/Data/WritingPrompts \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --gpu_device 1
```

### Use Multiple GPUs

```bash
# Use GPUs 0 and 1
python test_union_accuracy_tf2.py \
    --data_dir /scratch1/tanalvin/Data/WritingPrompts \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --gpu_device "0,1"
```

**Note**: The UNION model doesn't support multi-GPU data parallelism out of the box, but you can run multiple tests in parallel on different GPUs.

### Force CPU-Only Execution

```bash
# Disable GPU - use CPU only
python test_union_accuracy_tf2.py \
    --data_dir /scratch1/tanalvin/Data/WritingPrompts \
    --init_checkpoint ./model/uncased_L-12_H-768_A-12/union_wp/model.ckpt \
    --no-use_gpu
```

Or set environment variable before running:
```bash
export CUDA_VISIBLE_DEVICES=-1
python test_union_accuracy_tf2.py ...
```

## Checking GPU Usage During Execution

While the script is running, you can monitor GPU usage in another terminal:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or one-time check
nvidia-smi
```

You should see:
- GPU memory being allocated (several GB)
- GPU utilization % increasing during model inference
- Process name: `python`

## Expected Performance

### Test Set Size
- WritingPrompts: ~2000 examples (1000 human + 1000 negative)
- ROCStories: ~2000 examples (1000 human + 1000 negative)

### Approximate Runtime

| Hardware | Batch Size | Time per Example | Total Time (2000 examples) |
|----------|------------|------------------|----------------------------|
| GPU (modern, e.g., V100) | 32 | ~0.05s | ~3-5 minutes |
| GPU (older, e.g., GTX 1080) | 32 | ~0.1s | ~6-10 minutes |
| CPU only | 32 | ~2-5s | **2-4 hours** |

**Recommendation**: Always use GPU if available. CPU-only execution is **20-40x slower**.

## Troubleshooting GPU Issues

### Issue: "No GPU detected" but GPU exists

**Check CUDA installation:**
```bash
nvidia-smi
nvcc --version
```

**Check TensorFlow GPU support:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Solution**: Install compatible CUDA/cuDNN versions for your TensorFlow version:
- TensorFlow 2.10+: CUDA 11.2+, cuDNN 8.1+
- TensorFlow 2.4-2.9: CUDA 11.0+, cuDNN 8.0+

### Issue: "Could not load dynamic library 'libcudart.so'"

**Cause**: CUDA libraries not in system path

**Solution**:
```bash
# Add CUDA to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or add to ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Issue: "Out of memory" error on GPU

**Cause**: Model + batch too large for GPU memory

**Solution 1**: Reduce batch size
```bash
python test_union_accuracy_tf2.py \
    --predict_batch_size 8  # Reduced from default 32
```

**Solution 2**: Use gradient checkpointing (if available in model)

**Solution 3**: Use CPU instead
```bash
python test_union_accuracy_tf2.py --no-use_gpu
```

### Issue: GPU utilization is low (~10-20%)

**Possible causes**:
1. Batch size too small - increase `--predict_batch_size`
2. Data loading bottleneck - GPU waiting for CPU to prepare data
3. Model is I/O bound rather than compute bound

**Solutions**:
- Increase batch size: `--predict_batch_size 64`
- This is normal for inference tasks (training typically has higher utilization)

## Multi-GPU Testing Strategy

If you have multiple GPUs, you can run tests in parallel:

```bash
# Terminal 1: Test on GPU 0
CUDA_VISIBLE_DEVICES=0 python test_union_accuracy_tf2.py \
    --data_dir ./Data/WritingPrompts \
    --output_dir ./results_wp

# Terminal 2: Test on GPU 1 (different dataset or checkpoint)
CUDA_VISIBLE_DEVICES=1 python test_union_accuracy_tf2.py \
    --data_dir ./Data/ROCStories \
    --output_dir ./results_roc
```

## Memory Requirements

### GPU Memory
- Model size: ~420MB (BERT-base parameters)
- Batch size 32: ~2-3GB total GPU memory
- Batch size 64: ~4-5GB total GPU memory
- Batch size 128: ~8-10GB total GPU memory

**Recommendation**: Use batch size 32 for GPUs with 6GB+ memory, batch size 8-16 for 4GB GPUs.

### CPU Memory (RAM)
- Model + data: ~2-4GB
- Recommended: 8GB+ RAM

## Verifying GPU is Being Used

Add this to check GPU usage in your script output:

When you run the test, you should see:
```
TensorFlow 2.x detected (version 2.x.x)
Enabled TF1 compatibility mode
GPU(s) available: 1 device(s)
  GPU 0: /physical_device:GPU:0
GPU acceleration will be used by default
GPU device(s) set to: 0
```

If you see "No GPU detected - using CPU", then GPU is NOT being used.

## Summary

✅ **Default behavior**: Uses GPU automatically if available
✅ **Change GPU**: Use `--gpu_device N` flag
✅ **Disable GPU**: Use `--no-use_gpu` flag
✅ **Check status**: Run `python test_tf_compat.py`
✅ **Monitor usage**: Use `nvidia-smi`

GPU acceleration provides **20-40x speedup** compared to CPU-only execution!
