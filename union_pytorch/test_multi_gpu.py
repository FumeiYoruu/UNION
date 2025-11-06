"""Quick test script to verify multi-GPU setup is working correctly.

Run this before starting your full training to ensure multi-GPU works.
"""

import torch
import sys
import os

def test_gpu_availability():
    """Test GPU availability and configuration."""
    print("=" * 80)
    print("GPU Configuration Test")
    print("=" * 80)

    # Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("   ❌ CUDA not available. Check PyTorch installation and GPU drivers.")
        return False

    # Check number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"2. Number of GPUs: {num_gpus}")

    if num_gpus == 0:
        print("   ❌ No GPUs detected.")
        return False
    elif num_gpus == 1:
        print("   ⚠️  Only 1 GPU detected. Multi-GPU training will not be effective.")
        print("   💡 You can still train, but --use_multi_gpu flag is not needed.")
    else:
        print(f"   ✅ {num_gpus} GPUs detected. Multi-GPU training available!")

    # List GPU details
    print(f"\n3. GPU Details:")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"      - Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"      - Compute Capability: {props.major}.{props.minor}")
        print(f"      - Multi-Processor Count: {props.multi_processor_count}")

    # Check CUDA_VISIBLE_DEVICES
    print(f"\n4. CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all GPUs visible)')}")

    return num_gpus > 1


def test_dataparallel():
    """Test DataParallel functionality."""
    print("\n" + "=" * 80)
    print("DataParallel Functionality Test")
    print("=" * 80)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("\n⚠️  Skipping DataParallel test (need 2+ GPUs)")
        return True

    try:
        # Create a simple model
        print("\n1. Creating test model...")
        model = torch.nn.Linear(100, 10).cuda()
        print("   ✅ Model created on GPU")

        # Wrap with DataParallel
        print("\n2. Wrapping with DataParallel...")
        model = torch.nn.DataParallel(model)
        print(f"   ✅ DataParallel wrapper applied")
        print(f"   📊 Using {len(model.device_ids)} GPUs: {model.device_ids}")

        # Test forward pass
        print("\n3. Testing forward pass...")
        batch_size = 32
        x = torch.randn(batch_size, 100).cuda()
        output = model(x)
        print(f"   ✅ Forward pass successful")
        print(f"   📊 Input shape: {x.shape}, Output shape: {output.shape}")

        # Test backward pass
        print("\n4. Testing backward pass...")
        loss = output.sum()
        loss.backward()
        print("   ✅ Backward pass successful")

        print("\n✅ DataParallel test PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ DataParallel test FAILED: {str(e)}")
        return False


def test_memory_allocation():
    """Test memory allocation across GPUs."""
    print("\n" + "=" * 80)
    print("Memory Allocation Test")
    print("=" * 80)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("\n⚠️  Skipping memory test (need 2+ GPUs)")
        return True

    try:
        print("\n1. Initial memory state:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"   GPU {i}: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")

        # Create model with DataParallel
        print("\n2. Creating DataParallel model...")
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 100)
        ).cuda()
        model = torch.nn.DataParallel(model)

        # Allocate data
        batch_size = 64
        x = torch.randn(batch_size, 1000).cuda()

        # Forward pass
        output = model(x)
        loss = output.sum()
        loss.backward()

        print("\n3. Memory after model + forward/backward:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"   GPU {i}: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")

        print("\n4. Memory distribution check:")
        gpu0_mem = torch.cuda.memory_allocated(0)
        all_same = True
        for i in range(1, num_gpus):
            gpu_i_mem = torch.cuda.memory_allocated(i)
            ratio = gpu_i_mem / gpu0_mem if gpu0_mem > 0 else 0
            print(f"   GPU {i}/GPU 0 memory ratio: {ratio:.2f}")
            # Allow some variance (model replica might differ slightly)
            if ratio < 0.5 or ratio > 2.0:
                all_same = False

        if all_same:
            print("   ✅ Memory is reasonably balanced across GPUs")
        else:
            print("   ⚠️  Memory distribution is unbalanced (this can be normal)")

        # Cleanup
        del model, x, output, loss
        torch.cuda.empty_cache()

        print("\n✅ Memory allocation test PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ Memory allocation test FAILED: {str(e)}")
        return False


def print_recommendations():
    """Print recommendations based on GPU configuration."""
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)

    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("\n❌ No GPUs available")
        print("   - Use CPU training (very slow): --device cpu")
        print("   - Or install CUDA-compatible PyTorch")

    elif num_gpus == 1:
        print("\n✅ Single GPU Setup")
        print("   - Don't use --use_multi_gpu flag")
        print("   - Example command:")
        print("     python train_lora.py --train_batch_size 16 ...")

    elif num_gpus == 2:
        print("\n✅ Dual GPU Setup")
        print("   - Use --use_multi_gpu flag")
        print("   - Recommended batch size: 32-64")
        print("   - Example command:")
        print("     python train_lora.py --use_multi_gpu --train_batch_size 32 ...")

    elif num_gpus == 4:
        print("\n✅ Quad GPU Setup")
        print("   - Use --use_multi_gpu flag")
        print("   - Recommended batch size: 64-128")
        print("   - Example command:")
        print("     python train_lora.py --use_multi_gpu --train_batch_size 64 ...")

    else:  # 8 or more
        print(f"\n✅ {num_gpus} GPU Setup")
        print("   - Use --use_multi_gpu flag")
        print(f"   - Recommended batch size: {num_gpus * 16}-{num_gpus * 32}")
        print("   - Example command:")
        print(f"     python train_lora.py --use_multi_gpu --train_batch_size {num_gpus * 16} ...")


def main():
    """Run all tests."""
    print("\n🔍 UNION Multi-GPU Setup Test\n")

    # Test 1: GPU availability
    has_multi_gpu = test_gpu_availability()

    if not torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("❌ CANNOT PROCEED: No CUDA available")
        print("=" * 80)
        sys.exit(1)

    # Test 2: DataParallel
    dp_works = test_dataparallel()

    # Test 3: Memory
    mem_works = test_memory_allocation()

    # Print recommendations
    print_recommendations()

    # Final summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"✅ GPU Available: Yes")
    print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
    print(f"{'✅' if dp_works else '❌'} DataParallel: {'Working' if dp_works else 'Failed'}")
    print(f"{'✅' if mem_works else '❌'} Memory Allocation: {'Working' if mem_works else 'Failed'}")

    if torch.cuda.device_count() > 1 and dp_works and mem_works:
        print("\n🎉 Your system is ready for multi-GPU training!")
        print("   Use: python train_lora.py --use_multi_gpu ...")
    elif torch.cuda.device_count() == 1:
        print("\n✅ Your system is ready for single-GPU training!")
        print("   Use: python train_lora.py ...")
    else:
        print("\n⚠️  Some tests failed. Multi-GPU training may not work correctly.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
