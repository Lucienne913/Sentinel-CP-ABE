#!/usr/bin/env python3
"""
DTA Full Distributed Setup Benchmark — Includes MK generation + SSS split + 
Feldman commitment + node distribution, matching the claim at paper.tex L706.

Measures:
  - distributed_setup() total latency (what paper calls "5.7 ms")
  - Per-step breakdown for transparency

Usage:
    python experiments/benchmark_dta_full_setup.py
    python experiments/benchmark_dta_full_setup.py --save
"""

import sys
import os
import json
import time
import statistics
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2
from setup import T_CP_ABE_Setup
from distributed_ta import DistributedTA, DistributedSecretSharing

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_dta_setup(iterations=100, t=3, n=5):
    """Benchmark DistributedTA.distributed_setup() full latency."""
    print(f"\n  Config: t={t}, n={n}  |  Iterations: {iterations}")

    setup_latencies = []
    setup_obj = T_CP_ABE_Setup(group_name='SS512', security_level=80)

    for i in range(iterations):
        # Create fresh DTA each iteration (new MK each time)
        dta = DistributedTA(setup_obj.group, n_tas=n, threshold=t)

        t0 = time.perf_counter()
        PP = dta.distributed_setup(setup_obj)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        setup_latencies.append(elapsed_ms)

    mean_ms = statistics.mean(setup_latencies)
    std_ms = statistics.stdev(setup_latencies) if len(setup_latencies) > 1 else 0.0
    min_ms = min(setup_latencies)
    max_ms = max(setup_latencies)
    p95_ms = sorted(setup_latencies)[int(0.95 * len(setup_latencies))]

    print(f"  Mean: {mean_ms:.4f} ms  |  Std: {std_ms:.4f} ms")
    print(f"  Min: {min_ms:.4f} ms  |  Max: {max_ms:.4f} ms  |  P95: {p95_ms:.4f} ms")

    return {
        'config': {'t': t, 'n': n},
        'iterations': iterations,
        'curve': 'SS512',
        'mean_ms': round(mean_ms, 4),
        'std_ms': round(std_ms, 4),
        'min_ms': round(min_ms, 4),
        'max_ms': round(max_ms, 4),
        'p95_ms': round(p95_ms, 4),
        'all_latencies_ms': [round(x, 4) for x in setup_latencies],
    }


def benchmark_dta_setup_configs(configs=None, iterations=100):
    """Benchmark all (t,n) configurations, plus per-step breakdown for (3,5)."""
    if configs is None:
        configs = [(2, 3), (3, 5), (4, 7), (5, 10), (7, 10)]

    all_results = []
    print("=" * 60)
    print("  DTA Full Setup Benchmark — All Configurations")
    print("=" * 60)
    print(f"  {'Config':<12} {'Mean(ms)':<12} {'Std(ms)':<12} {'Min(ms)':<12} {'P95(ms)':<12}")
    print(f"  {'-' * 58}")

    for t, n in configs:
        data = benchmark_dta_setup(iterations=iterations, t=t, n=n)
        all_results.append(data)
        print(f"  {f'{t}-of-{n}':<12} {data['mean_ms']:<12.4f} {data['std_ms']:<12.4f} "
              f"{data['min_ms']:<12.4f} {data['p95_ms']:<12.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='DTA Full Setup Benchmark')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Iterations per config (default: 100)')
    parser.add_argument('--save', action='store_true',
                       help='Save to experiments/results/dta_setup_benchmark.json')

    args = parser.parse_args()

    results = benchmark_dta_setup_configs(iterations=args.iterations)

    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'benchmark': 'DTA Full Distributed Setup',
        'includes': ['MK generation', 'Shamir SSS split', 'Share distribution to n nodes',
                     'Feldman commitment', 'Broadcast'],
        'iterations_per_config': args.iterations,
        'configs': results,
    }

    if args.save:
        output_path = RESULTS_DIR / 'dta_setup_benchmark.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"\n[Save] Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
