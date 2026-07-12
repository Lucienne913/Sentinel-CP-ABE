#!/usr/bin/env python3
"""
DTA Performance Benchmark — Independent Split/Reconstruct Latency

Measures Shamir SSS split and reconstruct latencies separately for
each (t,n) configuration, producing data for paper Table V.

Usage:
    python experiments/benchmark_dta_performance.py
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from charm.toolbox.pairinggroup import PairingGroup, ZR
from distributed_ta import DistributedSecretSharing

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_config(sss, group, n, t, iterations=100):
    """Benchmark split and reconstruct separately."""
    secret = group.random(ZR)
    split_times = []
    recon_times = []

    for _ in range(iterations):
        # Split only
        t0 = time.perf_counter()
        shares = sss.split_secret(secret, n, t)
        split_times.append((time.perf_counter() - t0) * 1000)

        # Reconstruct only (from first t shares)
        t0 = time.perf_counter()
        recovered = sss.reconstruct(shares[:t], t)
        recon_times.append((time.perf_counter() - t0) * 1000)

    return {
        'n': n,
        't': t,
        'split': {
            'avg_ms': round(sum(split_times) / len(split_times), 4),
            'min_ms': round(min(split_times), 4),
            'max_ms': round(max(split_times), 4),
            'p95_ms': round(sorted(split_times)[int(0.95 * len(split_times))], 4),
            'std_ms': round(__import__('statistics').stdev(split_times), 4),
        },
        'reconstruct': {
            'avg_ms': round(sum(recon_times) / len(recon_times), 4),
            'min_ms': round(min(recon_times), 4),
            'max_ms': round(max(recon_times), 4),
            'p95_ms': round(sorted(recon_times)[int(0.95 * len(recon_times))], 4),
            'std_ms': round(__import__('statistics').stdev(recon_times), 4),
        },
        'total_avg_ms': round((sum(split_times) + sum(recon_times)) / len(split_times), 4),
    }


def main():
    print("=" * 60)
    print("DTA Performance Benchmark")
    print("=" * 60)

    group = PairingGroup('SS512')
    sss = DistributedSecretSharing(group)
    
    configs = [(2, 3), (3, 5), (4, 7), (5, 10), (7, 10)]
    results = []
    
    print(f"\n{'Config':<10} {'Split(ms)':<12} {'Recon(ms)':<12} {'Total(ms)':<12}")
    print(f"{'-'*46}")
    
    for t, n in configs:
        data = benchmark_config(sss, group, n, t, iterations=100)
        results.append(data)
        print(f"{f'{t}-of-{n}':<10} {data['split']['avg_ms']:<12.4f} "
              f"{data['reconstruct']['avg_ms']:<12.4f} {data['total_avg_ms']:<12.4f}")
    
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'curve': 'SS512',
        'iterations_per_config': 100,
        'configs': results,
    }
    
    output_path = RESULTS_DIR / 'dta_performance.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
