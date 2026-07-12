#!/usr/bin/env python3
"""
Grad-CAM Batch Analysis — Multi-Sample Statistics

Analyzes Grad-CAM attributions across a batch of synthetic auth requests
to produce statistical support for the claim:
"81% of True Positive detections are attributed to a compact subset of 7-12 dimensions"

Usage:
    python experiments/batch_gradcam_analysis.py
    python experiments/batch_gradcam_analysis.py --num-samples 200 --threshold 0.5
"""

import sys
import os
import json
import time
import argparse
import random
from pathlib import Path

import numpy as np

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from diffusion import ThreatDiffusionModel

RESULTS_DIR = Path(__file__).resolve().parent / 'results' / 'ablation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_batch_samples(num_samples: int, seed: int = 42):
    """Generate diverse auth request samples with known anomalies."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Normal attribute combinations (satisfy typical policies)
    normal_bases = [
        [10, 20, 30], [10, 21, 30], [11, 20, 31],
        [10, 20, 31], [11, 21, 30], [10, 21, 31],
    ]
    
    # Anomalous patterns
    anomaly_bases = [
        [12, 20, 30], [10, 22, 32], [11, 20, 32],
        [10, 99, 30], [12, 21, 31], [99, 20, 99],
    ]
    
    # Extended attribute pools for filling up to 20 attributes
    normal_pool = list(range(30, 50))  # Normal-like attributes
    anomaly_pool = [44, 45, 46, 47, 98, 99]  # Suspicious attributes
    
    samples = []
    for i in range(num_samples):
        is_anomaly = random.random() < 0.4  # 40% anomaly rate
        
        if is_anomaly:
            base = list(random.choice(anomaly_bases))
            fill_count = random.randint(10, 17)
            extra = random.choices(
                anomaly_pool + normal_pool,
                weights=[0.7, 0.7, 0.7, 0.7, 0.3, 0.3] + [0.2] * 20,
                k=fill_count
            )
        else:
            base = list(random.choice(normal_bases))
            fill_count = random.randint(10, 17)
            extra = random.choices(normal_pool, k=fill_count)
        
        attrs = base + extra
        random.shuffle(attrs)
        
        samples.append({
            'attrs': attrs[:20],  # cap at 20 attributes
            'is_anomaly': is_anomaly,
            'timestamp': f'2026-0{random.randint(1,6):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00',
            'device_id': f'device_{random.randint(1, 50):03d}',
        })
    
    return samples


def analyze_gradcam_batch(model, samples, threshold=0.5):
    """
    Run Grad-CAM on all samples and compute aggregate statistics.
    
    For each anomalous sample detected as True Positive:
    - Count how many embedding dimensions have high importance
    - Compute the distribution of "important dimension count"
    
    Returns statistics on attribute importance patterns.
    """
    results = []
    tp_dim_counts = []
    tp_top_attr_counts = []
    
    # Track importance distribution across all samples
    all_important_dims = []
    
    for i, sample in enumerate(samples):
        auth_request = {
            'attrs': sample['attrs'],
            'timestamp': sample.get('timestamp'),
            'device_id': sample.get('device_id'),
        }
        
        # Get Grad-CAM result
        gradcam_result = model.generate_gradcam(auth_request)
        anomaly_score = gradcam_result['anomaly_score']
        heatmap_embed = np.array(gradcam_result['heatmap_embed'])
        heatmap_attr = np.array(gradcam_result['heatmap_attr'])
        top_attrs = gradcam_result['top_attrs']
        
        # Prediction
        prediction = 1 if anomaly_score > threshold else 0
        true_label = 1 if sample['is_anomaly'] else 0
        
        # Count "important" embedding dimensions (importance > 0.1)
        important_dims = np.sum(heatmap_embed > 0.1)
        very_important_dims = np.sum(heatmap_embed > 0.3)
        
        # Count important attributes (attribute importance > 0.05)
        important_attrs = np.sum(heatmap_attr > 0.05)
        
        # Top attributes span
        if top_attrs:
            top_indices = [a['index'] for a in top_attrs]
            if len(top_indices) >= 2:
                attr_span = max(top_indices) - min(top_indices)
            else:
                attr_span = 0
        else:
            attr_span = 0
        
        entry = {
            'sample_id': i,
            'is_anomaly': sample['is_anomaly'],
            'prediction': bool(prediction),
            'is_true_positive': bool(prediction == 1 and true_label == 1),
            'anomaly_score': float(anomaly_score),
            'important_embed_dims': int(important_dims),
            'very_important_embed_dims': int(very_important_dims),
            'important_attrs': int(important_attrs),
            'top_attr_indices': top_indices,
            'attr_span': attr_span,
        }
        results.append(entry)
        
        all_important_dims.append(int(important_dims))
        
        if entry['is_true_positive']:
            tp_dim_counts.append(int(important_dims))
            tp_top_attr_counts.append(len(top_indices))
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(samples)}")
    
    # Compute aggregate statistics
    total_tp = sum(1 for r in results if r['is_true_positive'])
    total_anomalies = sum(1 for s in samples if s['is_anomaly'])
    total_detected = sum(1 for r in results if r['prediction'])
    
    stats = {
        'total_samples': len(samples),
        'total_anomalies': total_anomalies,
        'total_detected': total_detected,
        'total_true_positives': total_tp,
        'true_positive_rate': total_tp / max(total_anomalies, 1),
    }
    
    if tp_dim_counts:
        dim_counts = np.array(tp_dim_counts)
        stats['tp_embed_dim_stats'] = {
            'mean': float(np.mean(dim_counts)),
            'median': float(np.median(dim_counts)),
            'min': int(np.min(dim_counts)),
            'max': int(np.max(dim_counts)),
            'std': float(np.std(dim_counts)),
            # Percentage of TPs where important dims are in 7-12 range
            'pct_in_7_to_12': float(np.mean((dim_counts >= 7) & (dim_counts <= 12)) * 100),
            'pct_in_5_to_15': float(np.mean((dim_counts >= 5) & (dim_counts <= 15)) * 100),
        }
        stats['tp_value_counts'] = {
            str(k): int(v) for k, v in zip(*np.unique(dim_counts, return_counts=True))
        }
    
    if all_important_dims:
        all_dims = np.array(all_important_dims)
        stats['all_sample_dim_stats'] = {
            'mean': float(np.mean(all_dims)),
            'median': float(np.median(all_dims)),
            'min': int(np.min(all_dims)),
            'max': int(np.max(all_dims)),
        }
    
    return stats, results


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Batch Analysis')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='Number of samples to analyze')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Anomaly detection threshold')
    parser.add_argument('--model-weights', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Grad-CAM Batch Analysis")
    print(f"Samples: {args.num_samples}, Threshold: {args.threshold}")
    print("=" * 60)
    
    # Step 1: Load model
    print("\n[Step 1] Initializing model...")
    model = ThreatDiffusionModel(
        vocab_size=100,
        embed_dim=128,
        condition_dim=64,
        num_train_timesteps=100,
        device=args.device,
        pretrained_path=args.model_weights,
    )
    print(f"  Model trained: {model.is_trained}")
    
    # Train if needed
    if not model.is_trained:
        print("  Training model...")
        from train_diffusion import train_model as train_diffusion
        trained_model, losses = train_diffusion(
            num_epochs=20, batch_size=32, lr=1e-4, device=args.device
        )
        model = trained_model
        model.to(args.device)
        print("  Model trained.")
    
    # Step 2: Generate batch samples
    print(f"\n[Step 2] Generating {args.num_samples} samples...")
    samples = generate_batch_samples(args.num_samples)
    normal_count = sum(1 for s in samples if not s['is_anomaly'])
    anomaly_count = sum(1 for s in samples if s['is_anomaly'])
    print(f"  Normal: {normal_count}, Anomalous: {anomaly_count}")
    
    # Step 3: Batch Grad-CAM analysis
    print(f"\n[Step 3] Running Grad-CAM on all samples...")
    stats, per_sample = analyze_gradcam_batch(model, samples, args.threshold)
    
    # Step 4: Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples:     {stats['total_samples']}")
    print(f"  Total anomalies:   {stats['total_anomalies']}")
    print(f"  True Positives:    {stats['total_true_positives']}")
    print(f"  Detection Rate:    {stats['true_positive_rate']:.2%}")
    
    if 'tp_embed_dim_stats' in stats:
        ds = stats['tp_embed_dim_stats']
        print(f"\n  TP Embedding Dimension Analysis:")
        print(f"    Mean important dims:  {ds['mean']:.1f}")
        print(f"    Median important dims:{ds['median']:.1f}")
        print(f"    Range:                [{ds['min']}, {ds['max']}]")
        print(f"    % in [7, 12]:         {ds['pct_in_7_to_12']:.1f}%")
        print(f"    % in [5, 15]:         {ds['pct_in_5_to_15']:.1f}%")
        
        if 'tp_value_counts' in ds:
            print(f"\n  Distribution of important dim counts among TPs:")
            for k in sorted(ds['tp_value_counts'].keys(), key=int):
                v = ds['tp_value_counts'][k]
                bar = '█' * min(v, 30)
                print(f"    {int(k):3d} dims: {v:3d} TPs {bar}")
    
    # Step 5: Save results
    output = {
        'experiment_info': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'num_samples': args.num_samples,
            'threshold': args.threshold,
        },
        'aggregate_stats': stats,
        'per_sample': per_sample,
    }
    
    output_path = RESULTS_DIR / 'gradcam_batch_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to: {output_path}")


if __name__ == '__main__':
    main()
