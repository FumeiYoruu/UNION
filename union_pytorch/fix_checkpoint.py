#!/usr/bin/env python3
"""
Fix old checkpoints by adding batch_step field.
"""
import torch
import sys
import os

def fix_checkpoint(checkpoint_path, gradient_accumulation_steps=16):
    """Add batch_step to old checkpoint that doesn't have it."""
    training_state_path = os.path.join(checkpoint_path, "training_state.pt")

    if not os.path.exists(training_state_path):
        print(f"Error: {training_state_path} not found")
        return False

    # Load checkpoint
    state = torch.load(training_state_path, map_location='cpu')

    print(f"Current checkpoint contents:")
    print(f"  epoch: {state.get('epoch', 'NOT FOUND')}")
    print(f"  step: {state.get('step', 'NOT FOUND')}")
    print(f"  batch_step: {state.get('batch_step', 'NOT FOUND')}")

    # If batch_step already exists, no need to fix
    if 'batch_step' in state:
        print("\nCheckpoint already has batch_step field, no fix needed.")
        return True

    # Calculate batch_step from global step
    global_step = state.get('step', 0)
    epoch = state.get('epoch', 1)

    # Total batches processed up to this checkpoint
    total_batches_processed = global_step * gradient_accumulation_steps

    # For simplicity, assume we're still in the first epoch if this is an early checkpoint
    # The batch_step is the position within the current epoch
    # Since we don't have steps_per_epoch stored, we use total_batches as an approximation
    batch_step = total_batches_processed

    print(f"\nNote: If this checkpoint spans multiple epochs, you may need to manually")
    print(f"calculate batch_step as: (global_step * grad_accum_steps) % batches_per_epoch")

    print(f"\nCalculated batch_step: {batch_step}")
    print(f"  (global_step {global_step} × gradient_accumulation_steps {gradient_accumulation_steps})")

    # Add batch_step
    state['batch_step'] = batch_step

    # Backup original
    backup_path = training_state_path + ".backup"
    if not os.path.exists(backup_path):
        torch.save(state, backup_path)
        print(f"\nBackup saved to: {backup_path}")

    # Save fixed checkpoint
    torch.save(state, training_state_path)
    print(f"Fixed checkpoint saved to: {training_state_path}")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_checkpoint.py <checkpoint_dir> [gradient_accumulation_steps]")
        print("Example: python fix_checkpoint.py ../output/checkpoint-200 16")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    gradient_accumulation_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    print(f"Fixing checkpoint: {checkpoint_path}")
    print(f"Using gradient_accumulation_steps: {gradient_accumulation_steps}\n")

    success = fix_checkpoint(checkpoint_path, gradient_accumulation_steps)

    if success:
        print("\n✓ Checkpoint fixed successfully!")
    else:
        print("\n✗ Failed to fix checkpoint")
        sys.exit(1)
