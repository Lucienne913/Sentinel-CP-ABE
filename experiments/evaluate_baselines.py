#!/usr/bin/env python3
"""
Baseline Comparison Script

Compares our scheme (diffusion model + adaptive threshold) against state-of-the-art
anomaly detection baselines on the UNSW-NB15 dataset.

Baselines:
  1. IsolationForest (时序森林) — sklearn
  2. One-Class SVM              — sklearn
  3. Simple Autoencoder (时序AE) — PyTorch lightweight

Metrics: FPR, F1, AUC, Precision, Recall, Latency

Usage:
    python evaluate_baselines.py [--num-samples 5000] [--device cpu]

Output:
    ./results/comparison/sota_comparison.json
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix
)

warnings.filterwarnings('ignore')

# Ensure src is in path
_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

RESULTS_DIR = Path(__file__).resolve().parent / 'results' / 'comparison'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PARQUET_PATH = '/app/UNSW-NB15/Network-Flows/UNSW_Flow.parquet'


# =====================================================================
#  Baseline: Simple Autoencoder (轻量时序AE)
# =====================================================================

class SimpleAE(nn.Module):
    """Simple Autoencoder for anomaly detection on attribute embeddings."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def anomaly_score(self, x):
        """Reconstruction error as anomaly score."""
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction='none').mean(dim=1)


# =====================================================================
#  Data Loading (reuses UNSW-NB15 loader from evaluate_unsw_nb15)
# =====================================================================

def load_unsw_data(max_records: int = 5000):
    """Load UNSW-NB15 data in ABE evaluation format."""
    from evaluate_unsw_nb15 import load_unsw_nb15_from_parquet
    parquet_path = os.environ.get('UNSW_PARQUET_PATH', DEFAULT_PARQUET_PATH)
    
    if not os.path.exists(parquet_path):
        print(f"  ⚠ Parquet not found at {parquet_path}")
        print(f"  Check UNSW_PARQUET_PATH env var or use --parquet-path")
        return None
    
    samples = load_unsw_nb15_from_parquet(parquet_path, max_records)
    return samples


def generate_synthetic_data(n_samples: int = 1000):
    """Generate synthetic evaluation set for dev/testing."""
    from evaluate_unsw_nb15 import generate_abe_evaluation_set
    return generate_abe_evaluation_set(n_samples)


def samples_to_embeddings(samples, model):
    """Convert attribute indices to embeddings using diffusion model."""
    model.eval()
    attrs_list = []
    for s in samples:
        padded = list(s['attrs'])[:model.max_attrs]
        while len(padded) < model.max_attrs:
            padded.append(0)
        attrs_list.append(padded)
    
    attr_tensor = torch.tensor(attrs_list, dtype=torch.long).to(model.device)
    with torch.no_grad():
        embeddings = model.attr_embedding(attr_tensor)
    return embeddings.cpu().numpy()


def get_labels(samples):
    return np.array([s['label'] for s in samples])


# =====================================================================
#  Baseline Training & Evaluation
# =====================================================================

def train_isolation_forest(X_train):
    """Train IsolationForest."""
    print("  Training IsolationForest...")
    start = time.time()
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.2f}s")
    return model, elapsed


def train_oneclass_svm(X_train):
    """Train One-Class SVM."""
    print("  Training One-Class SVM...")
    start = time.time()
    model = OneClassSVM(
        nu=0.1,
        kernel='rbf',
        gamma='scale',
    )
    model.fit(X_train)
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.2f}s")
    return model, elapsed


def train_autoencoder(X_train, X_val, input_dim=128, hidden_dim=64,
                      epochs=50, lr=1e-3, device='cpu'):
    """Train Simple Autoencoder."""
    print(f"  Training Autoencoder ({epochs} epochs)...")
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(X_train)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = SimpleAE(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            loss = F.mse_loss(recon, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.6f}")
    
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.2f}s")
    return model, elapsed


def evaluate_baseline(name, scores, labels, latency_ms=0):
    """Compute metrics for a single baseline."""
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Find best threshold by maximizing F1 on test set
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_f1 = 0
    best_pred = np.zeros_like(labels)
    
    for thresh in thresholds:
        pred = (scores > thresh).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_pred = pred
    
    tn, fp, fn, tp = confusion_matrix(labels, best_pred, labels=[0, 1]).ravel()
    
    metrics = {
        'model': name,
        'accuracy': float(accuracy_score(labels, best_pred)),
        'precision': float(precision_score(labels, best_pred, zero_division=0)),
        'recall': float(recall_score(labels, best_pred, zero_division=0)),
        'f1': float(best_f1),
        'auc': float(roc_auc_score(labels, scores)),
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'tnr': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'avg_latency_ms': round(latency_ms, 4),
        'n_test': len(labels),
    }
    return metrics


def balance_data(samples, anomaly_ratio=0.3, seed=42):
    """
    Balance dataset by downsampling majority class.
    
    UNSW-NB15 has ~90% anomaly rate which is unrealistic for IoT.
    This creates a more realistic ratio for meaningful metric comparison.
    
    Args:
        samples: List of sample dicts with 'label' field
        anomaly_ratio: Target anomaly ratio (default 0.3, realistic IoT)
        seed: Random seed
        
    Returns:
        Balanced list of samples
    """
    rng = np.random.RandomState(seed)
    normal = [s for s in samples if s['label'] == 0]
    anomaly = [s for s in samples if s['label'] == 1]
    
    n_normal = len(normal)
    n_anomaly_target = int(n_normal * anomaly_ratio / (1 - anomaly_ratio))
    n_anomaly_target = min(n_anomaly_target, len(anomaly))
    
    # Downsample anomalies to target ratio
    idx = rng.choice(len(anomaly), n_anomaly_target, replace=False)
    anomaly_sampled = [anomaly[i] for i in idx]
    
    balanced = normal + anomaly_sampled
    rng.shuffle(balanced)
    
    return balanced


# =====================================================================
#  Main Pipeline
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Baseline Comparison')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Max UNSW-NB15 records to use')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data instead of UNSW-NB15')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-weights', type=str, default=None,
                        help='Path to fine-tuned weights')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    
    print("=" * 60)
    print("Baseline Comparison: Anomaly Detection on UNSW-NB15")
    print("=" * 60)
    print(f"Samples: {args.num_samples}, Device: {device}")
    
    # ---- Step 1: Initialize diffusion model ----
    print("\n[Step 1] Initializing diffusion model...")
    from diffusion import ThreatDiffusionModel
    
    # Default: try fine-tuned UNSW weights, then fall back
    model_weights = args.model_weights
    if model_weights is None:
        unsw_path = str(Path(_src_dir) / 'weights' / 'threat_diffusion_unsw.pth')
        if os.path.exists(unsw_path):
            model_weights = unsw_path
            print(f"  Using fine-tuned UNSW weights: {unsw_path}")
        else:
            default_path = str(Path(_src_dir) / 'weights' / 'threat_diffusion.pth')
            if os.path.exists(default_path):
                model_weights = default_path
                print(f"  Using pretrained weights: {default_path}")
    
    model = ThreatDiffusionModel(
        vocab_size=100, embed_dim=128, condition_dim=64,
        num_train_timesteps=100, device=device,
        pretrained_path=model_weights,
    )
    print(f"  Model trained: {model.is_trained}")
    
    # Train only if no weights available at all
    if not model.is_trained:
        print("\n[Step 2] Training diffusion model from scratch...")
        from train_diffusion import train_model as train_diffusion
        model, losses = train_diffusion(
            num_epochs=20, batch_size=32, lr=1e-4, device=device
        )
        model.to(device)
        
        weights_dir = Path(_src_dir) / 'weights'
        weights_dir.mkdir(exist_ok=True)
        weights_path = weights_dir / 'threat_diffusion.pth'
        model.save_weights(str(weights_path))
        print(f"  Weights saved to: {weights_path}")
    
    # ---- Step 3: Load and balance data ----
    print(f"\n[Step 3] Loading data...")
    if args.synthetic:
        samples = generate_synthetic_data(args.num_samples)
    else:
        samples = load_unsw_data(args.num_samples)
        if samples is None:
            print("  ⚠ UNSW-NB15 not available, falling back to synthetic data")
            samples = generate_synthetic_data(args.num_samples)
    
    print(f"  Raw total: {len(samples)}")
    raw_normal = sum(1 for s in samples if s['label'] == 0)
    raw_anomaly = sum(1 for s in samples if s['label'] == 1)
    print(f"  Raw Normal: {raw_normal}, Anomaly: {raw_anomaly} ({raw_anomaly/max(len(samples),1)*100:.1f}%)")
    
    # Balance dataset for meaningful comparison
    samples = balance_data(samples, anomaly_ratio=0.3, seed=args.seed)
    print(f"\n  After balance: {len(samples)} samples")
    normal = sum(1 for s in samples if s['label'] == 0)
    anomaly = sum(1 for s in samples if s['label'] == 1)
    print(f"  Normal: {normal}, Anomaly: {anomaly} ({anomaly/max(len(samples),1)*100:.1f}%)")
    
    # Train/val/test split (60/20/20)
    n = len(samples)
    idx = np.random.permutation(n)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    print(f"\n  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # ---- Step 4: Extract embeddings ----
    print(f"\n[Step 4] Extracting embeddings...")
    X_train = samples_to_embeddings(train_samples, model)
    X_val = samples_to_embeddings(val_samples, model)
    X_test = samples_to_embeddings(test_samples, model)
    y_train = get_labels(train_samples)
    y_val = get_labels(val_samples)
    y_test = get_labels(test_samples)
    print(f"  Embedding dimension: {X_train.shape[1]}")
    
    # Standardize for baselines that need it (SVM, AE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ---- Step 5: Train and evaluate baselines ----
    all_results = []
    
    # --- 5a. IsolationForest ---
    print(f"\n[Step 5a] IsolationForest...")
    if_model, if_train_time = train_isolation_forest(X_train)
    
    # IF produces -1 for anomaly, 1 for normal
    start = time.time()
    if_scores = if_model.decision_function(X_test)  # Higher = more normal
    if_scores = -if_scores  # Convert: higher = more anomalous
    if_latency = (time.time() - start) / len(X_test) * 1000
    
    if_results = evaluate_baseline('IsolationForest', if_scores, y_test, if_latency)
    if_results['train_time_s'] = round(if_train_time, 2)
    all_results.append(if_results)
    print(f"    F1: {if_results['f1']:.4f}, AUC: {if_results['auc']:.4f}, FPR: {if_results['fpr']:.4f}")
    
    # --- 5b. One-Class SVM ---
    print(f"\n[Step 5b] One-Class SVM...")
    svm_model, svm_train_time = train_oneclass_svm(X_train_scaled)
    
    start = time.time()
    svm_scores = svm_model.decision_function(X_test_scaled)  # Positive = anomaly
    svm_latency = (time.time() - start) / len(X_test) * 1000
    
    svm_results = evaluate_baseline('OneClassSVM', svm_scores, y_test, svm_latency)
    svm_results['train_time_s'] = round(svm_train_time, 2)
    all_results.append(svm_results)
    print(f"    F1: {svm_results['f1']:.4f}, AUC: {svm_results['auc']:.4f}, FPR: {svm_results['fpr']:.4f}")
    
    # --- 5c. Autoencoder ---
    print(f"\n[Step 5c] Autoencoder...")
    ae_model, ae_train_time = train_autoencoder(
        X_train, X_val, input_dim=128, hidden_dim=64,
        epochs=30, lr=1e-3, device=device,
    )
    
    ae_model.eval()
    start = time.time()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        ae_scores = ae_model.anomaly_score(X_test_tensor).cpu().numpy()
    ae_latency = (time.time() - start) / len(X_test) * 1000
    
    ae_results = evaluate_baseline('Autoencoder', ae_scores, y_test, ae_latency)
    ae_results['train_time_s'] = round(ae_train_time, 2)
    all_results.append(ae_results)
    print(f"    F1: {ae_results['f1']:.4f}, AUC: {ae_results['auc']:.4f}, FPR: {ae_results['fpr']:.4f}")
    
    # --- 5d. Our Diffusion Model (on test set) ---
    print(f"\n[Step 5d] Diffusion Model (Ours)...")
    model.eval()
    diff_scores = []
    diff_latencies = []
    
    for sample in test_samples:
        auth_req = {'attrs': sample['attrs']}
        ctx = {
            'time_anomaly': sample.get('time_anomaly', False),
            'behavior_anomaly': sample.get('behavior_anomaly', False),
        }
        start = time.time()
        score = model.anomaly_score(auth_req, ctx)
        diff_latencies.append((time.time() - start) * 1000)
        diff_scores.append(score)
    
    diff_latency = float(np.mean(diff_latencies))
    diff_results = evaluate_baseline('DiffusionModel(Ours)', diff_scores, y_test, diff_latency)
    all_results.append(diff_results)
    print(f"    F1: {diff_results['f1']:.4f}, AUC: {diff_results['auc']:.4f}, FPR: {diff_results['fpr']:.4f}")
    
    # ---- Step 6: Compile final comparison table ----
    print(f"\n{'='*60}")
    print("SOTA Comparison Results")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'F1':<8} {'AUC':<8} {'FPR':<8} {'Prec':<8} {'Rec':<8} {'Lat(ms)':<10}")
    print(f"{'-'*67}")
    
    for r in all_results:
        print(f"{r['model']:<25} {r['f1']:<8.4f} {r['auc']:<8.4f} "
              f"{r['fpr']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f} {r['avg_latency_ms']:<10.4f}")
    
    # ---- Save results ----
    timestamp = datetime.now().isoformat()
    output = {
        'experiment_info': {
            'timestamp': timestamp,
            'dataset': 'unsw_nb15' if not args.synthetic else 'synthetic',
            'n_train': len(train_samples),
            'n_val': len(val_samples),
            'n_test': len(test_samples),
            'n_features': X_train.shape[1],
            'normal_ratio': f"{normal}/{len(samples)}",
            'anomaly_ratio': f"{anomaly}/{len(samples)}",
        },
        'models': all_results,
    }
    
    output_path = RESULTS_DIR / 'sota_comparison.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")
    
    print(f"\n{'='*60}")
    print("Baseline Comparison Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
