"""
DETR Training Results - Visualization and Analysis

This script provides additional analysis and visualization utilities
for the trained DETR model. Run this after training is complete.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
CHECKPOINT_DIR = Path("./checkpoints")
TB_LOG_DIR = Path("./runs")

def plot_training_history(checkpoint_path):
    """
    Plot training history from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # This is a placeholder - in reality you'd need to log history separately
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_title('mAP@0.5 Evolution')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].text(0.5, 0.5, 
                   f"Best Checkpoint Info:\\n"
                   f"Epoch: {checkpoint['epoch']}\\n"
                   f"Val Loss: {checkpoint['val_loss']:.4f}\\n"
                   f"mAP@0.5: {checkpoint['val_map']:.4f}",
                   ha='center', va='center',
                   fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

def print_checkpoint_summary():
    """Print summary of available checkpoints."""
    print("="*60)
    print("DETR Training - Checkpoint Summary")
    print("="*60)
    
    checkpoints = {
        'best_loss_checkpoint.pth': 'Best Validation Loss',
        'best_map_checkpoint.pth': 'Best mAP@0.5',
        'last_checkpoint.pth': 'Last Epoch'
    }
    
    for ckpt_file, description in checkpoints.items():
        ckpt_path = CHECKPOINT_DIR / ckpt_file
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            print(f"\\n{description}:")
            print(f"  File: {ckpt_file}")
            print(f"  Epoch: {ckpt['epoch']}")
            print(f"  Train Loss: {ckpt['train_loss']:.4f}")
            print(f"  Val Loss: {ckpt['val_loss']:.4f}")
            print(f"  mAP@0.5: {ckpt['val_map']:.4f}")
        else:
            print(f"\\n{description}: Not found")
    
    print("\\n" + "="*60)

def compare_checkpoints():
    """Compare all available checkpoints."""
    checkpoints = {}
    
    for ckpt_file in CHECKPOINT_DIR.glob('*.pth'):
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            
            # Проверяем, что это действительно чекпоинт обучения (содержит нужные ключи)
            if 'epoch' in ckpt and 'val_loss' in ckpt and 'val_map' in ckpt:
                checkpoints[ckpt_file.stem] = {
                    'epoch': ckpt['epoch'],
                    'val_loss': ckpt['val_loss'],
                    'val_map': ckpt['val_map']
                }
            else:
                # Это не полный чекпоинт обучения (возможно, только веса модели)
                print(f"Skipping {ckpt_file.name} - not a training checkpoint")
        except Exception as e:
            print(f"Error loading {ckpt_file.name}: {e}")
            continue

    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(checkpoints.keys())
    val_losses = [checkpoints[n]['val_loss'] for n in names]
    val_maps = [checkpoints[n]['val_map'] for n in names]
    
    ax1.barh(names, val_losses, color='coral')
    ax1.set_xlabel('Validation Loss')
    ax1.set_title('Validation Loss by Checkpoint')
    ax1.grid(axis='x', alpha=0.3)
    
    ax2.barh(names, val_maps, color='skyblue')
    ax2.set_xlabel('mAP@0.5')
    ax2.set_title('mAP@0.5 by Checkpoint')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_model_size():
    """Analyze model checkpoint size and parameters."""
    best_ckpt = CHECKPOINT_DIR / 'best_map_checkpoint.pth'
    
    if not best_ckpt.exists():
        print("Best mAP checkpoint not found!")
        return
    
    try:
        ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' not in ckpt:
            print("\n" + "="*60)
            print("Model Size Analysis")
            print("="*60)
            print("Checkpoint doesn't contain 'model_state_dict' key")
            print("This might be a weights-only checkpoint.")
            print("="*60)
            return
            
        model_state = ckpt['model_state_dict']
        
        total_params = sum(p.numel() for p in model_state.values())
        file_size_mb = best_ckpt.stat().st_size / (1024 ** 2)
        
        print("\n" + "="*60)
        print("Model Size Analysis")
        print("="*60)
        print(f"Total Parameters: {total_params:,}")
        print(f"Checkpoint File Size: {file_size_mb:.2f} MB")
        print(f"Memory per Parameter: {file_size_mb / (total_params / 1e6):.2f} MB/M params")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print("Model Size Analysis")
        print("="*60)
        print(f"Error: {e}")
        print("="*60)

def generate_tensorboard_summary():
    """Generate summary of TensorBoard logs."""
    print("\n" + "="*60)
    print("TensorBoard Logs")
    print("="*60)
    
    if TB_LOG_DIR.exists():
        log_dirs = list(TB_LOG_DIR.glob("*"))
        print(f"\\nFound {len(log_dirs)} training run(s):")
        for log_dir in log_dirs:
            print(f"  - {log_dir.name}")
        
        print("\\nTo view in TensorBoard, run:")
        print(f"  tensorboard --logdir={TB_LOG_DIR}")
        print("  Then open: http://localhost:6006")
    else:
        print("No TensorBoard logs found!")
    
    print("="*60)

if __name__ == "__main__":
    print("\\n DETR Training - Results Analysis\\n")
    
    # Print checkpoint summary
    print_checkpoint_summary()
    
    # Analyze model size
    analyze_model_size()
    
    # TensorBoard info
    generate_tensorboard_summary()
    
    # Generate visualizations
    print("\\nGenerating visualizations...")
    
    try:
        fig = compare_checkpoints()
        plt.savefig('checkpoint_comparison.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: checkpoint_comparison.png")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\\nAnalysis complete!\\n")
