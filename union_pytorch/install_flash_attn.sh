#!/bin/bash
# Helper script to install Flash Attention 2 with proper CUDA_HOME setup

set -e  # Exit on error

echo "============================================"
echo "Flash Attention 2 Installation Helper"
echo "============================================"
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc (CUDA compiler) not found in PATH"
    echo ""
    echo "Please install CUDA toolkit or add it to your PATH:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "✓ Found CUDA version: $CUDA_VERSION"

# Check CUDA version compatibility
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
if [ "$CUDA_MAJOR" -lt 11 ] || ([ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -lt 6 ]); then
    echo "⚠️  WARNING: Flash Attention requires CUDA 11.6 or higher"
    echo "   Your version: $CUDA_VERSION"
    echo "   Installation may fail!"
    echo ""
fi

# Detect or set CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
    echo ""
    echo "CUDA_HOME is not set. Attempting to detect..."

    # Try common locations
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo "✓ Found CUDA at: $CUDA_HOME"
    elif [ -d "/usr/local/cuda-$CUDA_MAJOR.$CUDA_MINOR" ]; then
        export CUDA_HOME=/usr/local/cuda-$CUDA_MAJOR.$CUDA_MINOR
        echo "✓ Found CUDA at: $CUDA_HOME"
    elif [ ! -z "$CONDA_PREFIX" ]; then
        export CUDA_HOME=$CONDA_PREFIX
        echo "✓ Using conda environment: $CUDA_HOME"
    else
        echo "❌ ERROR: Could not detect CUDA installation"
        echo ""
        echo "Please set CUDA_HOME manually:"
        echo "  export CUDA_HOME=/path/to/cuda"
        echo "  ./install_flash_attn.sh"
        exit 1
    fi
else
    echo "✓ CUDA_HOME already set: $CUDA_HOME"
fi

# Verify CUDA_HOME is valid
if [ ! -d "$CUDA_HOME" ]; then
    echo "❌ ERROR: CUDA_HOME points to non-existent directory: $CUDA_HOME"
    exit 1
fi

if [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "❌ ERROR: nvcc not found in $CUDA_HOME/bin/"
    exit 1
fi

# Set additional environment variables
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo ""
echo "Environment variables:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  PATH: $PATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check PyTorch installation
echo ""
echo "Checking PyTorch installation..."
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')" || {
    echo "❌ ERROR: PyTorch not installed"
    echo "Please install PyTorch first: pip install torch"
    exit 1
}

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'" && {
    python -c "import torch; print(f'✓ PyTorch CUDA version: {torch.version.cuda}')"
} || {
    echo "⚠️  WARNING: CUDA not available in PyTorch"
    echo "Flash Attention requires PyTorch with CUDA support"
}

# Check GPU compatibility
echo ""
echo "Checking GPU compatibility..."
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        compute_cap = torch.cuda.get_device_capability(i)
        print(f'✓ GPU {i}: {gpu_name} (compute capability {compute_cap[0]}.{compute_cap[1]})')
        if compute_cap[0] < 8:
            print(f'  ⚠️  WARNING: Flash Attention requires compute capability 8.0+ (Ampere or newer)')
            print(f'     Your GPU has {compute_cap[0]}.{compute_cap[1]}. Installation will likely fail.')
else:
    print('⚠️  WARNING: No CUDA GPUs detected')
" || echo "⚠️  Could not check GPU compatibility"

# Ask for confirmation
echo ""
echo "============================================"
echo "Ready to install Flash Attention 2"
echo "============================================"
echo ""
echo "This will run: pip install flash-attn --no-build-isolation"
echo "Compilation may take 10-30 minutes depending on your hardware."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

# Install Flash Attention
echo ""
echo "Installing Flash Attention 2..."
echo "This may take 10-30 minutes. Please be patient..."
echo ""

pip install flash-attn --no-build-isolation

# Verify installation
echo ""
echo "Verifying installation..."
python -c "from flash_attn import flash_attn_func; print('✓ Flash Attention installed successfully!')" || {
    echo "❌ ERROR: Flash Attention installation failed verification"
    exit 1
}

# Success message
echo ""
echo "============================================"
echo "✓ Installation Complete!"
echo "============================================"
echo ""
echo "Flash Attention 2 is now installed and ready to use."
echo ""
echo "To use it with train_lora.py:"
echo "  python train_lora.py --fp16 --use_flash_attention --compile_model [other args...]"
echo ""
echo "To make CUDA_HOME permanent, add to your ~/.bashrc or ~/.zshrc:"
echo "  echo 'export CUDA_HOME=$CUDA_HOME' >> ~/.bashrc"
echo "  echo 'export PATH=\$CUDA_HOME/bin:\$PATH' >> ~/.bashrc"
echo "  echo 'export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"
echo ""
