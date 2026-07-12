#!/usr/bin/env python3
"""
Full Closed-Loop End-to-End Benchmark — Detection → Grad-CAM → Policy → 
Re-encryption → Re-keying, matching the "910 ms" claim at paper.tex L879.

Extends benchmark_dynamic_lsss.py by including the RE-KEYING phase:
re-generating secret keys for all users whose attributes were revoked.

Usage:
    python experiments/benchmark_full_closed_loop.py
    python experiments/benchmark_full_closed_loop.py --save
"""

import sys
import os
import time
import json
import statistics
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from charm.toolbox.pairinggroup import PairingGroup, ZR, GT
from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, PolicyParser
from diffusion import ThreatDiffusionModel
from distributed_ta import DistributedTA

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_benchmark(iterations=50, affected_users=10, save=False):
    """Run full closed-loop benchmark with re-keying."""
    print("=" * 65)
    print("  Full Closed-Loop Benchmark — with Re-keying")
    print(f"  Iterations: {iterations}  |  Affected users: {affected_users}")
    print("=" * 65)

    # ── Setup ────────────────────────────────────────────────
    print("\n[Setup] Initializing T-CP-ABE + Diffusion + DTA...")
    group_obj = PairingGroup('SS512')
    setup_obj = T_CP_ABE_Setup(group_name='SS512', security_level=80)
    PP, MK = setup_obj.setup(max_attrs=50)
    tcabe = T_CP_ABE(PP)
    parser = PolicyParser()

    # Initialize DTA (t=3, n=5)
    dta = DistributedTA(group_obj, n_tas=5, threshold=3)
    PP_dta = dta.distributed_setup(setup_obj)

    # Initialize diffusion model
    model = ThreatDiffusionModel(
        vocab_size=100, embed_dim=128, condition_dim=64,
        device='cpu', pretrained_path=None
    )

    # Create initial ciphertext
    base_policy_str = "role:engineer AND dept:maintenance"
    base_policy_tree = parser.parse(base_policy_str)
    message = group_obj.random(GT)
    ct = tcabe.encrypt(message, base_policy_tree)
    ct['policy_str'] = base_policy_str

    # Warm up EWMA (40 samples)
    warmup_ctx = {
        'attrs': [10, 20, 30],
        'time_anomaly': False, 'behavior_anomaly': False, 'suspicious_attrs': False
    }
    for _ in range(40):
        model.predict({'attrs': [10, 20, 30]}, warmup_ctx)

    # Pre-generate affected users' attribute sets
    user_attr_sets = []
    for u in range(affected_users):
        user_attr_sets.append([
            'role:engineer', 'dept:maintenance',
            f'uid:{u}', f'group:{(u % 3) + 1}'
        ])

    # Generate initial SKs for all users (for re-keying later)
    user_sks = []
    for attrs in user_attr_sets:
        sk = dta.threshold_keygen(tcabe, attrs, available_nodes=[0, 1, 2])
        user_sks.append(sk)

    # Pre-compute messages for re-encryption
    messages = [group_obj.random(GT) for _ in range(iterations)]

    # ── Benchmark loop ───────────────────────────────────────
    timings = {
        'anomaly_detection': [],
        'gradcam': [],
        'policy_compute': [],
        'revocation': [],
        're_encryption': [],
        're_keying': [],
        'total': [],
        'total_without_rekey': [],
        'trigger_rate': 0,
    }

    import gc
    for i in range(iterations):
        # Vary context per iteration
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

        # Fresh ciphertext per iteration
        ct_i = tcabe.encrypt(messages[i], base_policy_tree)
        ct_i['policy_str'] = base_policy_str

        t_start = time.time()

        # ── 1. Anomaly detection ─────────────────────────────
        t0 = time.time()
        auth_req = {'attrs': ctx.get('attrs', [])}
        score, is_anomaly = model.predict(auth_req, ctx)
        timings['anomaly_detection'].append((time.time() - t0) * 1000)

        # ── 2. Grad-CAM attribution ──────────────────────────
        t1 = time.time()
        if is_anomaly:
            gradcam_result = model.generate_gradcam(auth_req)
        else:
            gradcam_result = {'top_attrs': [{'index': 0, 'importance': 0.0}]}
        timings['gradcam'].append((time.time() - t1) * 1000)

        # ── 3. Policy computation ────────────────────────────
        t2 = time.time()
        new_policy, revoke_list, tl = model.adaptive_policy_update(
            score, base_policy_str, gradcam_result if is_anomaly else None
        )
        timings['policy_compute'].append((time.time() - t2) * 1000)

        # ── 4. Attribute revocation ──────────────────────────
        t3 = time.time()
        for attr in revoke_list:
            tcabe.revoke_attribute(attr)
        timings['revocation'].append((time.time() - t3) * 1000)

        # ── 5. Re-encryption ─────────────────────────────────
        t4 = time.time()
        if is_anomaly and new_policy != base_policy_str:
            new_tree = parser.parse(new_policy)
            new_ct = tcabe.encrypt(messages[i], new_tree)
        timings['re_encryption'].append((time.time() - t4) * 1000)

        without_rekey = (time.time() - t_start) * 1000

        # ── 6. RE-KEYING: regenerate SKs for affected users ──
        t5 = time.time()
        for u_idx in range(affected_users):
            # Use 2nd, 5th, 8th... node groups for variety
            nodes = [(u_idx * 2) % 5, (u_idx * 2 + 1) % 5, (u_idx * 2 + 2) % 5]
            dta.threshold_keygen(tcabe, user_attr_sets[u_idx], available_nodes=nodes)
        timings['re_keying'].append((time.time() - t5) * 1000)

        timings['total_without_rekey'].append(without_rekey)
        timings['total'].append((time.time() - t_start) * 1000)
        timings['trigger_rate'] += (1 if is_anomaly else 0)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{iterations}")
            gc.collect()

    timings['trigger_rate'] = timings['trigger_rate'] / iterations * 100

    # ── Statistics ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Latency Results (ms)")
    print("=" * 65)
    print(f"  {'Step':<24} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 58)

    results = {}
    for step in ['anomaly_detection', 'gradcam', 'policy_compute',
                 'revocation', 're_encryption', 're_keying',
                 'total_without_rekey', 'total']:
        data = timings[step]
        if data:
            mean = statistics.mean(data)
            std = statistics.stdev(data) if len(data) > 1 else 0.0
            mn = min(data)
            mx = max(data)
            print(f"  {step:<24} {mean:>8.2f} {std:>8.2f} {mn:>8.2f} {mx:>8.2f}")
            results[step] = {
                'mean_ms': round(mean, 2),
                'std_ms': round(std, 2),
                'min_ms': round(mn, 2),
                'max_ms': round(mx, 2),
                'samples': len(data)
            }
        else:
            print(f"  {step:<24} {'N/A':>8}")

    print(f"\n  Trigger rate: {timings['trigger_rate']:.1f}%")
    print(f"  Affected users re-keyed per loop: {affected_users}")
    results['trigger_rate_pct'] = round(timings['trigger_rate'], 1)
    results['iterations'] = iterations
    results['affected_users'] = affected_users
    results['curve'] = 'SS512'

    if save:
        output_path = RESULTS_DIR / 'full_closed_loop_benchmark.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Save] Results saved to: {output_path}")

    print("=" * 65)
    return results


def main():
    parser = argparse.ArgumentParser(description='Full Closed-Loop Benchmark')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations (default: 50)')
    parser.add_argument('--affected-users', type=int, default=10,
                       help='Users to re-key per loop (default: 10)')
    parser.add_argument('--save', action='store_true',
                       help='Save to results/full_closed_loop_benchmark.json')

    args = parser.parse_args()
    run_benchmark(iterations=args.iterations,
                  affected_users=args.affected_users,
                  save=args.save)


if __name__ == '__main__':
    main()
