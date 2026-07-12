#!/usr/bin/env python3
"""
UNSW-NB15 Fine-Tuning Script

Fine-tunes the ThreatDiffusionModel on real UNSW-NB15 traffic data,
improving anomaly detection performance on real IoT traffic patterns.

Usage:
    python finetune_unsw_nb15.py [--epochs 10] [--batch-size 32] [--lr 1e-4] [--device cpu]

Output:
    /app/src/weights/threat_diffusion_unsw.pth  — Fine-tuned weights
"""

import sys
import os
import math
import time
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

PARQUET_PATH = '/app/UNSW-NB15/Network-Flows/UNSW_Flow.parquet'
RESULTS_DIR = Path(__file__).resolve().parent / 'results' / 'finetune'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_unsw_attributes(max_records: int = 5000) -> tuple:
    """
    Load UNSW-NB15 and extract attribute indices + labels.
    
    Returns:
        (attr_indices, policy_indices, labels) — each as numpy array
    """
    from evaluate_unsw_nb15 import load_unsw_nb15_from_parquet
    
    print(f"  Loading UNSW-NB15 from {PARQUET_PATH}...")
    samples = load_unsw_nb15_from_parquet(PARQUET_PATH, max_records)
    print(f"  Loaded {len(samples)} samples")
    
    # Extract attribute indices and labels
    attr_list = []
    label_list = []
    for s in samples:
        padded = list(s['attrs'])[:20]
        while len(padded) < 20:
            padded.append(0)
        attr_list.append(padded)
        label_list.append(s['label'])
    
    # Use attribute vector itself as policy condition (self-supervised)
    attr_array = np.array(attr_list, dtype=np.int64)
    label_array = np.array(label_list, dtype=np.int64)
    
    # Policy condition: use same attributes (model learns identity mapping
    # between request attributes and policy conditions)
    policy_array = attr_array.copy()
    
    return attr_array, policy_array, label_array


def create_dataloader(attr_array, policy_array, batch_size=32, shuffle=True):
    """Create DataLoader from attribute arrays."""
    dataset = TensorDataset(
        torch.tensor(attr_array, dtype=torch.long),
        torch.tensor(policy_array, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune on UNSW-NB15')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Lower learning rate for fine-tuning')
    parser.add_argument('--max-records', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model-weights', type=str, default=None,
                        help='Path to pretrained weights (default: threat_diffusion.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output weights path')
    args = parser.parse_args()
    
    device = args.device
    
    print("=" * 60)
    print("UNSW-NB15 Fine-Tuning")
    print("=" * 60)
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    
    # ---- Step 1: Load base model ----
    print("\n[Step 1] Loading base model...")
    from diffusion import ThreatDiffusionModel
    
    # Default pretrained path
    if args.model_weights is None:
        default_path = str(Path(_src_dir) / 'weights' / 'threat_diffusion.pth')
        if os.path.exists(default_path):
            args.model_weights = default_path
            print(f"  Found pretrained weights: {default_path}")
        else:
            print(f"  No pretrained weights found at {default_path}")
            print(f"  Will train from scratch (not recommended)")
    
    model = ThreatDiffusionModel(
        vocab_size=100,
        embed_dim=128,
        condition_dim=64,
        num_train_timesteps=100,
        device=device,
        pretrained_path=args.model_weights,
    )
    print(f"  Model trained: {model.is_trained}")
    
    # ---- Step 2: Load UNSW-NB15 data ----
    print(f"\n[Step 2] Loading UNSW-NB15 data...")
    attr_array, policy_array, label_array = load_unsw_attributes(args.max_records)
    print(f"  Attributes shape: {attr_array.shape}")
    print(f"  Normal: {np.sum(label_array == 0)}, Anomaly: {np.sum(label_array == 1)}")
    
    # ---- Step 3: Fine-tune ----
    print(f"\n[Step 3] Fine-tuning...")
    loader = create_dataloader(attr_array, policy_array, args.batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    losses = []
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_attrs, batch_policy in loader:
            batch_attrs = batch_attrs.to(device)
            batch_policy = batch_policy.to(device)
            
            loss = model.train_step(batch_attrs, batch_policy, optimizer)
            epoch_loss += loss
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}, Elapsed: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n  Fine-tuning complete! Total time: {total_time:.1f}s")
    print(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
    
    # ---- Step 4: Save fine-tuned weights ----
    print(f"\n[Step 4] Saving fine-tuned weights...")
    output_path = args.output or str(Path(_src_dir) / 'weights' / 'threat_diffusion_unsw.pth')
    model.save_weights(output_path)
    
    # Validate: verify the saved weights can be loaded back
    print("  Validating saved weights...")
    validation_model = ThreatDiffusionModel(
        vocab_size=100, embed_dim=128, condition_dim=64,
        num_train_timesteps=100, device=device,
        pretrained_path=output_path,
    )
    if validation_model.is_trained:
        print(f"  ✓ Weights validated successfully")
    else:
        print(f"  ⚠ Weight validation issue")
    
    # Save training curve
    training_curve = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'n_samples': len(attr_array),
        'loss_history': losses,
        'final_loss': losses[-1],
        'total_time_s': round(total_time, 2),
    }
    
    import json
    curve_path = RESULTS_DIR / 'finetune_curve.json'
    with open(curve_path, 'w') as f:
        json.dump(training_curve, f, indent=2)
    print(f"  Training curve saved to: {curve_path}")
    
    print(f"\n{'='*60}")
    print("Fine-tuning complete! Use --model-weights with fine-tuned path")
    print(f"  {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
