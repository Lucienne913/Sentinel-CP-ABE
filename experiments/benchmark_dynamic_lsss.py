#!/usr/bin/env python3
"""
Benchmark: AI-driven Dynamic LSSS Update — Multi-iteration Latency & Throughput

Measures the end-to-end closed-loop latency of:
  1. Anomaly detection (EWMA + diffusion score)
  2. Grad-CAM attribution (top-k anomalous attributes)
  3. Policy computation (LSSS update via adaptive_policy_update)
  4. Attribute revocation (T_CP_ABE.revoke_attribute)
  5. Ciphertext re-encryption (T_CP_ABE.encrypt with new policy)

Usage:
    python experiments/benchmark_dynamic_lsss.py
    python experiments/benchmark_dynamic_lsss.py --iterations 200
    python experiments/benchmark_dynamic_lsss.py --save

Output:
    - Console: per-step mean latency ± std (ms)
    - JSON:    experiments/results/ai_closed_loop.json (with --save)
"""

import sys
import os
import time
import json
import argparse
import statistics
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, GT
from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, PolicyParser
from diffusion import ThreatDiffusionModel


def run_benchmark(iterations=100, save=False):
    """Run the full closed-loop benchmark."""
    print("=" * 65)
    print("  AI-Driven Dynamic LSSS Update — Benchmark")
    print(f"  Iterations: {iterations}  |  Charm-Crypto: SS512")
    print("=" * 65)
    
    # ── Setup ────────────────────────────────────────────────
    print("\n[Setup] Initializing T-CP-ABE + Diffusion model...")
    setup = T_CP_ABE_Setup(group_name='SS512', security_level=80)
    PP, MK = setup.setup(max_attrs=50)
    tcabe = T_CP_ABE(PP)
    parser = PolicyParser()
    group = PP['group']
    
    model = ThreatDiffusionModel(
        vocab_size=100, embed_dim=128, condition_dim=64,
        device='cpu', pretrained_path=None
    )
    
    # Create initial ciphertext
    base_policy_str = "role:engineer AND dept:maintenance"
    base_policy_tree = parser.parse(base_policy_str)
    message = group.random(GT)
    ct = tcabe.encrypt(message, base_policy_tree)
    ct['policy_str'] = base_policy_str
    
    # Warm up EWMA (40 samples)
    warmup_ctx = {
        'attrs': [10, 20, 30],
        'time_anomaly': False, 'behavior_anomaly': False, 'suspicious_attrs': False
    }
    for _ in range(40):
        model.predict({'attrs': [10, 20, 30]}, warmup_ctx)
    
    # ── Pre-compute messages for re-encryption ───────────────
    messages = [group.random(GT) for _ in range(iterations)]
    
    # ── Benchmark loop ───────────────────────────────────────
    timings = {
        'anomaly_detection': [],
        'gradcam': [],
        'policy_compute': [],
        'revocation': [],
        're_encryption': [],
        'total': [],
        'trigger_rate': 0,
    }
    
    import gc
    for i in range(iterations):
        # Vary context per iteration to get realistic timing mix
        if i % 3 == 0:
            ctx = {
                'attrs': [85, 33, 52],
                'time_anomaly': True, 'behavior_anomaly': True, 'suspicious_attrs': True
            }
        else:
            ctx = {
                'attrs': [10 + (i % 20), 20 + (i % 15), 30 + (i % 10)],
                'time_anomaly': False, 'behavior_anomaly': False, 'suspicious_attrs': False
            }
        
        # Fresh ciphertext per iteration to avoid revocation conflict
        ct_i = tcabe.encrypt(messages[i], base_policy_tree)
        ct_i['policy_str'] = base_policy_str
        
        t_start = time.time()
        
        # 1. Anomaly detection
        t0 = time.time()
        auth_req = {'attrs': ctx.get('attrs', [])}
        score, is_anomaly = model.predict(auth_req, ctx)
        timings['anomaly_detection'].append((time.time() - t0) * 1000)
        
        # 2. Grad-CAM attribution (only if anomalous for realistic data)
        t1 = time.time()
        gradcam_result = None
        if is_anomaly:
            gradcam_result = model.generate_gradcam(auth_req)
        else:
            gradcam_result = {'top_attrs': [{'index': 0, 'importance': 0.0}]}
        timings['gradcam'].append((time.time() - t1) * 1000)
        
        # 3. Policy computation
        t2 = time.time()
        new_policy, revoke_list, tl = model.adaptive_policy_update(
            score, base_policy_str, gradcam_result if is_anomaly else None
        )
        timings['policy_compute'].append((time.time() - t2) * 1000)
        
        # 4. Attribute revocation (only if revoke_list non-empty)
        t3 = time.time()
        for attr in revoke_list:
            tcabe.revoke_attribute(attr)
        timings['revocation'].append((time.time() - t3) * 1000)
        
        # 5. Re-encryption with new policy
        t4 = time.time()
        if is_anomaly and new_policy != base_policy_str:
            new_tree = parser.parse(new_policy)
            new_ct = tcabe.encrypt(messages[i], new_tree)
            new_ct['policy_str'] = new_policy
        timings['re_encryption'].append((time.time() - t4) * 1000)
        
        timings['total'].append((time.time() - t_start) * 1000)
        timings['trigger_rate'] += (1 if is_anomaly else 0)
        
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{iterations}")
            gc.collect()
    
    timings['trigger_rate'] = timings['trigger_rate'] / iterations * 100
    
    # ── Statistics ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Latency Results (ms)")
    print("=" * 65)
    print(f"  {'Step':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 58)
    
    results = {}
    for step in ['anomaly_detection', 'gradcam', 'policy_compute',
                 'revocation', 're_encryption', 'total']:
        data = timings[step]
        if data:
            mean = statistics.mean(data)
            std = statistics.stdev(data) if len(data) > 1 else 0.0
            mn = min(data)
            mx = max(data)
            print(f"  {step:<22} {mean:>8.2f} {std:>8.2f} {mn:>8.2f} {mx:>8.2f}")
            results[step] = {
                'mean_ms': round(mean, 2),
                'std_ms': round(std, 2),
                'min_ms': round(mn, 2),
                'max_ms': round(mx, 2),
                'samples': len(data)
            }
        else:
            print(f"  {step:<22} {'N/A':>8}")
    
    print(f"\n  Trigger rate: {timings['trigger_rate']:.1f}% ({int(timings['trigger_rate']/100*iterations)}/{iterations})")
    results['trigger_rate_pct'] = round(timings['trigger_rate'], 1)
    results['iterations'] = iterations
    results['curve'] = 'SS512'
    
    # ── Save ─────────────────────────────────────────────────
    if save:
        output_dir = Path(__file__).resolve().parent / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'ai_closed_loop.json'
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Save] Results written to {output_path}")
    
    print("=" * 65)
    return results


def main():
    parser = argparse.ArgumentParser(description='Dynamic LSSS Update Benchmark')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--save', action='store_true',
                       help='Save results to experiments/results/ai_closed_loop.json')
    
    args = parser.parse_args()
    run_benchmark(iterations=args.iterations, save=args.save)


if __name__ == '__main__':
    main()
