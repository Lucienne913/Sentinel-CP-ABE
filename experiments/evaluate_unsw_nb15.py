#!/usr/bin/env python3
"""
UNSW-NB15 Diffusion Model Evaluation Script

Evaluates the ThreatDiffusionModel on UNSW-NB15 dataset.
Compares fixed threshold vs adaptive threshold (EWMA).

Usage:
    # Full pipeline (train + evaluate)
    python evaluate_unsw_nb15.py --mode full

    # Evaluate only (use existing preprocessed data and trained model)
    python evaluate_unsw_nb15.py --mode eval --model-weights ../src/weights/threat_diffusion.pth

    # Data preprocessing only (requires UNSW-NB15 CSV files)
    python evaluate_unsw_nb15.py --mode preprocess --data-dir ./data/unsw_nb15 --max-records 5000

Output:
    ./results/unsw_nb15/evaluation_results.json  — All metrics
    ./results/unsw_nb15/metrics_comparison.json  — Fixed vs Adaptive comparison
    ./results/unsw_nb15/threshold_history.json   — Adaptive threshold tracking
"""

import sys
import os
import json
import time
import math
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix, roc_curve
)

# Ensure src is in path
_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_experiments_dir = str(Path(__file__).resolve().parent)
if _experiments_dir not in sys.path:
    sys.path.insert(0, _experiments_dir)

from diffusion import ThreatDiffusionModel, generate_synthetic_auth_logs

RESULTS_DIR = Path(__file__).resolve().parent / 'results' / 'unsw_nb15'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default path for UNSW-NB15 parquet inside Docker container
DEFAULT_PARQUET_PATH = '/app/UNSW-NB15/Network-Flows/UNSW_Flow.parquet'


def load_unsw_nb15_from_parquet(parquet_path: str, max_records: int = 5000) -> list:
    """
    Load UNSW-NB15 dataset from parquet and map to ABE evaluation format.
    
    Args:
        parquet_path: Path to parquet file
        max_records: Maximum records to sample (5000 by default)
    
    Returns:
        List of dicts suitable for model evaluation
    """
    import pandas as pd
    
    print(f"  Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Total records: {len(df)}")
    
    # Filter IoT-related attack types (labels are lowercase in parquet)
    iot_attacks = ['fuzzers', 'analysis', 'backdoor', 'dos', 'exploits',
                   'generic', 'reconnaissance', 'shellcode', 'worms']
    if 'attack_label' in df.columns:
        df['attack_label'] = df['attack_label'].str.lower().str.strip()
        df_filtered = df[df['attack_label'].isin(iot_attacks + ['normal'])].copy()
    else:
        df_filtered = df.copy()
    print(f"  After IoT filtering: {len(df_filtered)}")
    
    # Stratified sampling to get balanced set
    if 'attack_label' in df_filtered.columns:
        samples_per_class = max_records // len(df_filtered['attack_label'].unique())
        sampled = df_filtered.groupby('attack_label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), samples_per_class), random_state=42)
        )
    else:
        sampled = df_filtered.sample(min(max_records, len(df_filtered)), random_state=42)
    
    sampled = sampled.reset_index(drop=True)
    print(f"  After stratified sampling: {len(sampled)}")
    
    # Map to ABE evaluation format
    eval_samples = []
    for _, row in sampled.iterrows():
        # Build ABE attributes from traffic features
        attrs = map_unsw_row_to_abe_attrs(row)
        
        # Determine label
        label = 1 if row.get('binary_label', 0) == 1 else 0
        
        # Determine context anomalies
        time_anomaly = _detect_time_anomaly(row)
        behavior_anomaly = _detect_behavior_anomaly(row)
        
        eval_samples.append({
            'attrs': attrs,
            'label': label,
            'time_anomaly': time_anomaly,
            'behavior_anomaly': behavior_anomaly,
            'source': 'unsw_nb15'
        })
    
    return eval_samples


def map_unsw_row_to_abe_attrs(row) -> list:
    """Map UNSW-NB15 row features to ABE attribute indices."""
    attrs = []
    
    # Protocol
    proto_map = {'tcp': 4, 'udp': 5, 'icmp': 6, 'http': 7}
    proto = str(row.get('protocol', 'tcp')).lower().strip()
    attrs.append(proto_map.get(proto, 0))
    
    # State
    state_map = {'CON': 26, 'INT': 27, 'FIN': 28, 'REQ': 29, 'RST': 30, 'no': 31}
    state = str(row.get('state', 'CON')).upper().strip()
    attrs.append(state_map.get(state, 26))
    
    # Traffic volume (based on total bytes)
    sbytes = float(row.get('sbytes', 0))
    dbytes = float(row.get('dbytes', 0))
    total_bytes = sbytes + dbytes
    if total_bytes < 1000:
        attrs.append(23)  # traffic:low
    elif total_bytes < 100000:
        attrs.append(24)  # traffic:medium
    else:
        attrs.append(25)  # traffic:high
    
    # Duration
    dur = float(row.get('dur', 0))
    if dur < 1:
        attrs.append(32)  # duration:short
    elif dur < 60:
        attrs.append(33)  # duration:medium
    else:
        attrs.append(34)  # duration:long
    
    # Service
    service_map = {'http': 7, 'ftp': 35, 'smtp': 36, 'ssh': 37, 'dns': 38, '-': 39}
    service = str(row.get('service', '-')).lower().strip()
    attrs.append(service_map.get(service, 39))
    
    # Connection state features
    sttl = float(row.get('sttl', 0))
    if sttl < 30:
        attrs.append(40)  # conn:short_ttl
    elif sttl < 100:
        attrs.append(41)  # conn:medium_ttl
    else:
        attrs.append(42)  # conn:long_ttl
    
    # Attack category (for ABE attribute space)
    attack_label = str(row.get('attack_label', 'normal')).strip().lower()
    threat_map = {
        'normal': 43, 'fuzzers': 44, 'analysis': 44, 'reconnaissance': 44,
        'dos': 45, 'exploits': 45, 'generic': 45,
        'backdoor': 46, 'shellcode': 46,
        'worms': 47,
    }
    attrs.append(threat_map.get(attack_label, 43))
    
    return attrs


def _detect_time_anomaly(row) -> bool:
    """Detect time-based anomalies (unusual connection duration or inter-packet timing)."""
    dur = float(row.get('dur', 0))
    sjit = float(row.get('sjit', 0))
    djit = float(row.get('djit', 0))
    
    # Extremely short duration with high jitter = possible scanning
    if dur < 0.001 and (sjit > 10 or djit > 10):
        return True
    # Extremely long duration = possible persistence attack
    if dur > 1000:
        return True
    return False


def _detect_behavior_anomaly(row) -> bool:
    """Detect behavior anomalies (unusual packet counts, byte ratios, etc.)."""
    sbytes = float(row.get('sbytes', 0))
    dbytes = float(row.get('dbytes', 0))
    spkts = float(row.get('spkts', 1))
    dpkts = float(row.get('dpkts', 1))
    
    # Unusual byte ratio (high outbound vs inbound)
    total = sbytes + dbytes
    if total > 0:
        byte_ratio = sbytes / total
        if byte_ratio > 0.95 or byte_ratio < 0.05:
            return True
    
    # Unusual packet size
    if spkts > 0:
        avg_pkt_size = sbytes / spkts
        if avg_pkt_size > 10000:
            return True
    
    return False


# ===== Synthetic Data Generator (for development/testing) =====

def generate_abe_evaluation_set(n_samples: int = 1000) -> list:
    """
    Generate synthetic evaluation set in the format expected by ThreatDiffusionModel.
    
    Generates realistic attribute combinations for:
    - Normal traffic (60%): Attributes that satisfy typical IoT policies
    - Anomalous traffic (40%): Attributes with suspicious patterns
    
    Returns:
        List of dicts: [{'attrs': [...], 'label': 0/1, 'time_anomaly': bool, 'behavior_anomaly': bool}]
    """
    normal_combos = [
        [10, 20, 30],   # role:engineer, dept:maintenance, location:factory
        [10, 21, 30],   # role:engineer, dept:operations, location:factory
        [11, 20, 31],   # role:admin, dept:maintenance, location:office
        [10, 20, 31],   # role:engineer, dept:maintenance, location:office
        [11, 21, 30],   # role:admin, dept:operations, location:factory
        [10, 21, 31],   # role:engineer, dept:operations, location:office
    ]
    attack_combos = [
        [12, 20, 30],   # role:intern (unauthorized), dept:maintenance, location:factory
        [10, 22, 32],   # role:engineer, dept:unknown, location:external
        [99, 99, 99],   # Completely random attributes
        [11, 20, 32],   # role:admin, dept:maintenance, location:external
        [10, 99, 30],   # role:engineer, dept:unknown, location:factory
        [99, 20, 99],   # Partial random
    ]
    
    samples = []
    for i in range(n_samples):
        if i < n_samples * 0.6:
            # Normal
            attrs = list(normal_combos[i % len(normal_combos)])
            label = 0
            time_anomaly = False
            behavior_anomaly = False
        else:
            # Anomalous
            attrs = list(attack_combos[i % len(attack_combos)])
            label = 1
            time_anomaly = random.random() > 0.5
            behavior_anomaly = random.random() > 0.3
        
        samples.append({
            'attrs': attrs,
            'label': label,
            'time_anomaly': time_anomaly,
            'behavior_anomaly': behavior_anomaly
        })
    
    return samples


# ===== Evaluation Helpers =====

def compute_metrics(labels: list, predictions: list, scores: list, prefix: str = "") -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomaly)
        predictions: Binary predictions from model
        scores: Raw anomaly scores (for AUC computation)
        prefix: Metric name prefix (e.g., "fixed_threshold" or "adaptive_threshold")
    
    Returns:
        Dictionary of all metrics
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    scores = np.array(scores)
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    
    metrics = {
        f'{prefix}accuracy': float(accuracy_score(labels, predictions)),
        f'{prefix}precision': float(precision_score(labels, predictions, zero_division=0)),
        f'{prefix}recall': float(recall_score(labels, predictions, zero_division=0)),
        f'{prefix}f1': float(f1_score(labels, predictions, zero_division=0)),
        f'{prefix}auc': float(roc_auc_score(labels, scores)),
        f'{prefix}fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        f'{prefix}tnr': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        f'{prefix}tp': int(tp),
        f'{prefix}fp': int(fp),
        f'{prefix}tn': int(tn),
        f'{prefix}fn': int(fn),
    }
    return metrics


def compute_latency_metrics(latencies: list, prefix: str = "") -> dict:
    """Compute latency statistics."""
    latencies = np.array(latencies)
    return {
        f'{prefix}avg_latency_ms': float(np.mean(latencies)),
        f'{prefix}p50_latency_ms': float(np.percentile(latencies, 50)),
        f'{prefix}p95_latency_ms': float(np.percentile(latencies, 95)),
        f'{prefix}p99_latency_ms': float(np.percentile(latencies, 99)),
        f'{prefix}max_latency_ms': float(np.max(latencies)),
        f'{prefix}min_latency_ms': float(np.min(latencies)),
    }


# ===== Main Evaluation Pipeline =====

def run_evaluation(model: ThreatDiffusionModel, eval_samples: list, device: str = 'cpu') -> dict:
    """
    Run full evaluation comparing fixed vs adaptive threshold.
    
    Args:
        model: Trained ThreatDiffusionModel
        eval_samples: List of evaluation samples
        device: Computing device
    
    Returns:
        Dictionary containing all results
    """
    print(f"\n{'='*60}")
    print(f"Running Evaluation on {len(eval_samples)} samples")
    print(f"{'='*60}")
    
    # ---- Fixed Threshold Evaluation ----
    print("\n[Fixed Threshold] Evaluating with fixed threshold (0.5)...")
    fixed_labels = []
    fixed_predictions = []
    fixed_scores = []
    fixed_latencies = []
    
    # Importance: For fixed threshold, use the original anomaly_score only (no adaptive update)
    # Re-initialize model threshold state to ensure clean comparison
    model.threshold_history = []
    model.threshold_mean = 0.5
    model.threshold_std = 0.1
    
    for i, sample in enumerate(eval_samples):
        auth_request = {'attrs': sample['attrs']}
        context = {
            'time_anomaly': sample.get('time_anomaly', False),
            'behavior_anomaly': sample.get('behavior_anomaly', False)
        }
        
        start = time.time()
        score = model.anomaly_score(auth_request, context)
        latency = (time.time() - start) * 1000  # ms
        
        fixed_scores.append(score)
        fixed_predictions.append(1 if score > 0.5 else 0)
        fixed_labels.append(sample['label'])
        fixed_latencies.append(latency)
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(eval_samples)}")
    
    fixed_metrics = compute_metrics(fixed_labels, fixed_predictions, fixed_scores, prefix="fixed_")
    fixed_latency_metrics = compute_latency_metrics(fixed_latencies, prefix="fixed_")
    
    print(f"\n  Fixed Threshold Results:")
    print(f"    F1:     {fixed_metrics['fixed_f1']:.4f}")
    print(f"    AUC:    {fixed_metrics['fixed_auc']:.4f}")
    print(f"    FPR:    {fixed_metrics['fixed_fpr']:.4f}")
    print(f"    Avg Lat: {fixed_latency_metrics['fixed_avg_latency_ms']:.2f}ms")
    
    # ---- Adaptive Threshold (EWMA) Evaluation ----
    print("\n[Adaptive Threshold] Evaluating with EWMA adaptive threshold...")
    adapt_labels = []
    adapt_predictions = []
    adapt_scores = []
    adapt_latencies = []
    adapt_thresholds = []
    
    # Reset threshold state for adaptive evaluation
    model.threshold_history = []
    model.threshold_mean = 0.5
    model.threshold_std = 0.1
    
    for i, sample in enumerate(eval_samples):
        auth_request = {'attrs': sample['attrs']}
        context = {
            'time_anomaly': sample.get('time_anomaly', False),
            'behavior_anomaly': sample.get('behavior_anomaly', False)
        }
        
        start = time.time()
        score, is_anomaly = model.predict(auth_request, context)
        latency = (time.time() - start) * 1000  # ms
        
        adapt_scores.append(score)
        adapt_predictions.append(1 if is_anomaly else 0)
        adapt_labels.append(sample['label'])
        adapt_latencies.append(latency)
        adapt_thresholds.append(model.get_adaptive_threshold())
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(eval_samples)}")
    
    adapt_metrics = compute_metrics(adapt_labels, adapt_predictions, adapt_scores, prefix="adaptive_")
    adapt_latency_metrics = compute_latency_metrics(adapt_latencies, prefix="adaptive_")
    
    print(f"\n  Adaptive Threshold Results:")
    print(f"    F1:     {adapt_metrics['adaptive_f1']:.4f}")
    print(f"    AUC:    {adapt_metrics['adaptive_auc']:.4f}")
    print(f"    FPR:    {adapt_metrics['adaptive_fpr']:.4f}")
    print(f"    Avg Lat: {adapt_latency_metrics['adaptive_avg_latency_ms']:.2f}ms")
    
    # ---- Compile Results ----
    results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'dataset': eval_samples[0].get('source', 'synthetic'),
            'num_samples': len(eval_samples),
            'device': device,
            'model_trained': model.is_trained,
        },
        'metrics': {**fixed_metrics, **fixed_latency_metrics, **adapt_metrics, **adapt_latency_metrics},
        'threshold_history': {
            'values': adapt_thresholds,
            'initial': 0.5,
            'final': adapt_thresholds[-1] if adapt_thresholds else 0.5,
            'mean': float(np.mean(adapt_thresholds)) if adapt_thresholds else 0.5,
            'alpha': model.threshold_alpha,
            'window_size': model.threshold_window_size,
        },
        'score_distribution': {
            'fixed': {
                'mean': float(np.mean(fixed_scores)),
                'std': float(np.std(fixed_scores)),
                'min': float(np.min(fixed_scores)),
                'max': float(np.max(fixed_scores)),
            },
            'adaptive': {
                'mean': float(np.mean(adapt_scores)),
                'std': float(np.std(adapt_scores)),
                'min': float(np.min(adapt_scores)),
                'max': float(np.max(adapt_scores)),
            }
        }
    }
    
    return results


def print_comparison_table(results: dict):
    """Print a comparison table of fixed vs adaptive threshold."""
    print(f"\n{'='*60}")
    print("Comparison: Fixed Threshold vs Adaptive Threshold (EWMA)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Fixed':<12} {'Adaptive':<12} {'Improvement':<12}")
    print(f"{'-'*56}")
    
    metrics_to_show = [
        ('f1', 'F1 Score'),
        ('auc', 'AUC'),
        ('fpr', 'FPR'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('avg_latency_ms', 'Avg Lat (ms)'),
    ]
    
    m = results['metrics']
    for key, name in metrics_to_show:
        fixed_val = m.get(f'fixed_{key}', 0)
        adapt_val = m.get(f'adaptive_{key}', 0)
        
        if key in ('fpr', 'avg_latency_ms'):
            # Lower is better
            improvement = ((fixed_val - adapt_val) / max(fixed_val, 1e-6)) * 100
            impr_str = f"{improvement:+.1f}%" if improvement != 0 else "—"
        else:
            # Higher is better
            improvement = ((adapt_val - fixed_val) / max(fixed_val, 1e-6)) * 100
            impr_str = f"{improvement:+.1f}%" if improvement != 0 else "—"
        
        print(f"{name:<20} {fixed_val:<12.4f} {adapt_val:<12.4f} {impr_str:<12}")


def save_results(results: dict, output_dir: Path = RESULTS_DIR):
    """Save evaluation results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_file}")
    
    # Summary for quick reference
    m = results['metrics']
    summary = {
        'timestamp': results['experiment_info']['timestamp'],
        'dataset': results['experiment_info']['dataset'],
        'num_samples': results['experiment_info']['num_samples'],
        'fixed_threshold': {
            'f1': m.get('fixed_f1'),
            'auc': m.get('fixed_auc'),
            'fpr': m.get('fixed_fpr'),
            'precision': m.get('fixed_precision'),
            'recall': m.get('fixed_recall'),
            'avg_latency_ms': m.get('fixed_avg_latency_ms'),
        },
        'adaptive_threshold': {
            'f1': m.get('adaptive_f1'),
            'auc': m.get('adaptive_auc'),
            'fpr': m.get('adaptive_fpr'),
            'precision': m.get('adaptive_precision'),
            'recall': m.get('adaptive_recall'),
            'avg_latency_ms': m.get('adaptive_avg_latency_ms'),
        },
        'threshold_alpha': results['threshold_history']['alpha'],
    }
    summary_file = output_dir / 'metrics_comparison.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='UNSW-NB15 Diffusion Model Evaluation')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'eval', 'preprocess', 'synthetic'],
                        help='Execution mode')
    parser.add_argument('--model-weights', type=str, default=None,
                        help='Path to pretrained model weights (.pth)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of evaluation samples')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Computing device (cpu/cuda)')
    parser.add_argument('--parquet-path', type=str, default=DEFAULT_PARQUET_PATH,
                        help='Path to UNSW-NB15 parquet file (inside Docker)')
    parser.add_argument('--max-records', type=int, default=5000,
                        help='Maximum UNSW-NB15 records to process')
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNSW-NB15 Diffusion Model Evaluation")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"{'='*60}")
    
    device = args.device
    
    # Step 1: Initialize model
    print("\n[Step 1] Initializing ThreatDiffusionModel...")
    model = ThreatDiffusionModel(
        vocab_size=100,
        embed_dim=128,
        condition_dim=64,
        num_train_timesteps=100,
        device=device,
        pretrained_path=args.model_weights,
    )
    print(f"  Model trained: {model.is_trained}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 2: Ensure model is trained
    if not model.is_trained and args.mode in ('full', 'eval', 'synthetic'):
        print("\n[Step 2] Model not trained. Training with synthetic data...")
        from train_diffusion import train_model as train_diffusion
        trained_model, losses = train_diffusion(
            num_epochs=20, batch_size=32, lr=1e-4, device=device
        )
        model = trained_model
        model.to(device)
        
        # Save weights
        weights_dir = Path(_src_dir) / 'weights'
        weights_dir.mkdir(exist_ok=True)
        weights_path = weights_dir / 'threat_diffusion.pth'
        model.save_weights(str(weights_path))
        print(f"  Model weights saved to: {weights_path}")
    
    # Step 3: Generate or load evaluation data
    print(f"\n[Step 3] Preparing evaluation data ({args.num_samples} samples)...")
    
    # Try to load real UNSW-NB15 data (inside Docker)
    use_unsw = False
    if args.mode in ('full', 'eval'):
        parquet_path = args.parquet_path
        if os.path.exists(parquet_path):
            print(f"  Loading real UNSW-NB15 data from parquet...")
            try:
                eval_samples = load_unsw_nb15_from_parquet(parquet_path, args.max_records)
                use_unsw = True
                print(f"  Successfully loaded {len(eval_samples)} UNSW-NB15 samples")
            except Exception as e:
                print(f"  Failed to load UNSW-NB15 data: {e}")
                print(f"  Falling back to synthetic data...")
    
    if not use_unsw:
        eval_samples = generate_abe_evaluation_set(n_samples=args.num_samples)
        print(f"  Using synthetic evaluation set")
    
    normal_count = sum(1 for s in eval_samples if s['label'] == 0)
    anomaly_count = sum(1 for s in eval_samples if s['label'] == 1)
    print(f"  Normal: {normal_count}, Anomaly: {anomaly_count}")
    
    # Step 4: Run evaluation
    results = run_evaluation(model, eval_samples, device)
    
    # Step 5: Print comparison
    print_comparison_table(results)
    
    # Step 6: Save results
    save_results(results)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
