#!/usr/bin/env python3
"""
EWMA η (threshold_sensitivity) Parameter Sweep

Tries different η values on the UNSW-NB15 evaluation data to find
the optimal sensitivity parameter that maximizes F1 score.

Usage:
    python tune_ewma_eta.py
    python tune_ewma_eta.py --model-weights ../src/weights/threat_diffusion.pth --num-samples 4671
"""

import sys
import os
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from diffusion import ThreatDiffusionModel
from evaluate_unsw_nb15 import (
    load_unsw_nb15_from_parquet,
    generate_abe_evaluation_set,
    compute_metrics,
    compute_latency_metrics,
)

RESULTS_DIR = Path(__file__).resolve().parent / 'results' / 'unsw_nb15'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PARQUET_PATH = '/app/UNSW-NB15/Network-Flows/UNSW_Flow.parquet'


def run_adaptive_evaluation(model, eval_samples, eta, device='cpu'):
    """
    Run adaptive (EWMA) evaluation with a specific η (threshold_sensitivity) value.
    Returns metrics dict.
    """
    # Reset model threshold state
    model.threshold_history = []
    model.threshold_mean = 0.5
    model.threshold_std = 0.1
    model.threshold_sensitivity = eta

    labels = []
    predictions = []
    scores = []
    latencies = []
    thresholds = []

    for i, sample in enumerate(eval_samples):
        auth_request = {'attrs': sample['attrs']}
        context = {
            'time_anomaly': sample.get('time_anomaly', False),
            'behavior_anomaly': sample.get('behavior_anomaly', False)
        }

        start = time.time()
        score, is_anomaly = model.predict(auth_request, context)
        latency = (time.time() - start) * 1000  # ms

        scores.append(score)
        predictions.append(1 if is_anomaly else 0)
        labels.append(sample['label'])
        latencies.append(latency)
        thresholds.append(model.get_adaptive_threshold())

    metrics = compute_metrics(labels, predictions, scores, prefix="adaptive_")
    lat_metrics = compute_latency_metrics(latencies, prefix="adaptive_")

    return {**metrics, **lat_metrics}, {
        'values': thresholds,
        'initial': 0.5,
        'final': float(np.mean(thresholds[-10:])) if len(thresholds) >= 10 else 0.5,
        'mean': float(np.mean(thresholds)),
    }


def main():
    parser = argparse.ArgumentParser(description='EWMA η Parameter Sweep')
    parser.add_argument('--model-weights', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=5000)
    parser.add_argument('--parquet-path', type=str, default=DEFAULT_PARQUET_PATH)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device

    print("=" * 60)
    print("EWMA η Parameter Sweep")
    print(f"Device: {device}")
    print("=" * 60)

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

    # Train if not trained
    if not model.is_trained:
        print("\n  Training model with synthetic data...")
        from train_diffusion import train_model as train_diffusion
        trained_model, losses = train_diffusion(
            num_epochs=20, batch_size=32, lr=1e-4, device=device
        )
        model = trained_model
        model.to(device)
        weights_dir = Path(_src_dir) / 'weights'
        weights_dir.mkdir(exist_ok=True)
        model.save_weights(str(weights_dir / 'threat_diffusion.pth'))
        print("  Model trained and saved.")

    # Step 2: Load evaluation data
    print(f"\n[Step 2] Loading evaluation data...")
    use_unsw = False
    if os.path.exists(args.parquet_path):
        print(f"  Loading real UNSW-NB15 data...")
        try:
            eval_samples = load_unsw_nb15_from_parquet(args.parquet_path, args.num_samples)
            use_unsw = True
            print(f"  Loaded {len(eval_samples)} UNSW-NB15 samples")
        except Exception as e:
            print(f"  Failed: {e}")

    if not use_unsw:
        print(f"  Using synthetic data...")
        eval_samples = generate_abe_evaluation_set(n_samples=args.num_samples)

    normal_count = sum(1 for s in eval_samples if s['label'] == 0)
    anomaly_count = sum(1 for s in eval_samples if s['label'] == 1)
    print(f"  Normal: {normal_count}, Anomaly: {anomaly_count}")

    # Step 3: Run fixed threshold evaluation (baseline, same for all η)
    print(f"\n[Step 3] Running fixed threshold baseline...")
    model.threshold_history = []
    model.threshold_mean = 0.5
    model.threshold_std = 0.1

    fixed_labels = []
    fixed_scores = []
    for sample in eval_samples:
        auth_request = {'attrs': sample['attrs']}
        context = {
            'time_anomaly': sample.get('time_anomaly', False),
            'behavior_anomaly': sample.get('behavior_anomaly', False)
        }
        score = model.anomaly_score(auth_request, context)
        fixed_scores.append(score)
        fixed_labels.append(sample['label'])

    fixed_preds = [1 if s > 0.5 else 0 for s in fixed_scores]
    fixed_metrics = compute_metrics(fixed_labels, fixed_preds, fixed_scores, prefix="fixed_")
    print(f"  Fixed (thresh=0.5): F1={fixed_metrics['fixed_f1']:.4f}, "
          f"AUC={fixed_metrics['fixed_auc']:.4f}, "
          f"FPR={fixed_metrics['fixed_fpr']:.4f}")

    # Step 4: η sweep
    eta_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    print(f"\n[Step 4] Sweeping η values: {eta_values}")
    sweep_results = {}
    best_result = {'eta': None, 'f1': -1.0}

    for eta in eta_values:
        print(f"\n  η = {eta} ...", end=" ", flush=True)
        
        (metrics, thresh_info), time_s = _time_it(
            run_adaptive_evaluation, model, eval_samples, eta, device
        )
        
        f1 = metrics['adaptive_f1']
        auc = metrics['adaptive_auc']
        precision = metrics['adaptive_precision']
        recall = metrics['adaptive_recall']
        fpr = metrics['adaptive_fpr']
        
        print(f"F1={f1:.4f}, AUC={auc:.4f}, Prec={precision:.4f}, "
              f"Rec={recall:.4f}, FPR={fpr:.4f} ({time_s:.1f}s)")

        sweep_results[str(eta)] = {
            'eta': eta,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tp': metrics['adaptive_tp'],
            'fp': metrics['adaptive_fp'],
            'tn': metrics['adaptive_tn'],
            'fn': metrics['adaptive_fn'],
            'threshold_final': thresh_info['final'],
            'threshold_mean': thresh_info['mean'],
            'time_s': time_s,
        }

        if f1 > best_result['f1']:
            best_result = {'eta': eta, 'f1': f1, 'auc': auc, 'precision': precision, 'recall': recall, 'fpr': fpr}

    # Step 5: Print results
    print(f"\n{'='*60}")
    print("SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'η':<8} {'F1':<8} {'AUC':<8} {'Prec':<8} {'Recall':<8} {'FPR':<8} {'TP':<6} {'FP':<6}")
    print(f"{'-'*60}")
    for eta_str, r in sorted(sweep_results.items(), key=lambda x: float(x[0])):
        print(f"{r['eta']:<8.2f} {r['f1']:<8.4f} {r['auc']:<8.4f} {r['precision']:<8.4f} "
              f"{r['recall']:<8.4f} {r['fpr']:<8.4f} {r['tp']:<6} {r['fp']:<6}")

    print(f"\n  BEST: η={best_result['eta']} with F1={best_result['f1']:.4f}, "
          f"AUC={best_result['auc']:.4f}, Prec={best_result['precision']:.4f}")

    # Step 6: Save results
    output = {
        'experiment_info': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'num_samples': len(eval_samples),
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
        },
        'fixed_baseline': {
            'f1': fixed_metrics['fixed_f1'],
            'auc': fixed_metrics['fixed_auc'],
            'fpr': fixed_metrics['fixed_fpr'],
            'precision': fixed_metrics['fixed_precision'],
            'recall': fixed_metrics['fixed_recall'],
        },
        'sweep': sweep_results,
        'best': best_result,
    }

    output_path = RESULTS_DIR / 'ewma_eta_sweep.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Sweep results saved to: {output_path}")


def _time_it(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


if __name__ == '__main__':
    main()
