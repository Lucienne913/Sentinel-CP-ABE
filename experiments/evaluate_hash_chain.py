#!/usr/bin/env python3
"""
Hash Chain Performance Benchmark: Original vs Optimized (Checkpoint + Sliding Window)

Compares TimeTokenAuthority verification performance across chain lengths
and token positions. Demonstrates O(N)→O(1) improvement.

Usage:
    python evaluate_hash_chain.py [--chain-lengths 1000 8760 100000]

Output:
    ./results/ablation/hash_chain_performance.json  — All benchmark data
"""

import sys
import os
import json
import time
import math
import argparse
from pathlib import Path
from datetime import datetime

# Ensure src is in path
_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from t_cp_abe import TimeTokenAuthority

RESULTS_DIR = Path(__file__).resolve().parent / 'results' / 'ablation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_verify(tta, positions, n_repeats=30, label=''):
    """
    Benchmark verify_token at specified positions.
    
    Args:
        tta: TimeTokenAuthority instance
        positions: List of (position_index, n_hashes) tuples
        n_repeats: Number of repeats per position
        label: Label for output
    
    Returns:
        dict: Latency metrics per position
    """
    results = {}
    
    for pos, n_hashes_expected in positions:
        token = tta._chain[pos]
        latencies = []
        hash_counts = []
        
        for _ in range(n_repeats):
            start = time.perf_counter()
            result = tta.verify_token(pos, token)
            elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
            
            if not result:
                print(f"  ⚠ Verification FAILED at position {pos}!")
            latencies.append(elapsed)
        
        avg_lat = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
        
        results[f'pos_{pos}'] = {
            'position': pos,
            'distance_from_tip': tta.chain_length - 1 - pos,
            'expected_hashes': n_hashes_expected,
            'avg_latency_us': round(avg_lat, 3),
            'p95_latency_us': round(p95, 3),
            'p99_latency_us': round(p99, 3),
            'n_repeats': n_repeats,
        }
    
    return results


def run_chain_benchmark(chain_length, checkpoint_interval=100, window_size=10):
    """
    Run benchmark for a single chain length.
    
    Creates two instances:
    - Original: checkpoint_interval=chain_length (effectively disabled)
    - Optimized: checkpoint_interval=100
    
    Args:
        chain_length: Length of hash chain
        checkpoint_interval: Checkpoint interval for optimized version
        window_size: Sliding window size for optimized version
    
    Returns:
        dict: Benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Chain Length: {chain_length}")
    print(f"{'='*60}")
    
    # Create original instance (checkpoints disabled)
    print("\n[Original] Creating baseline (checkpoints disabled)...")
    tta_orig = TimeTokenAuthority(
        chain_length=chain_length,
        checkpoint_interval=chain_length,  # Effectively disable checkpoints
        window_size=0,                      # No sliding window
    )
    
    # Create optimized instance
    print(f"[Optimized] Creating with checkpoints (interval={checkpoint_interval}, window={window_size})...")
    tta_opt = TimeTokenAuthority(
        chain_length=chain_length,
        checkpoint_interval=checkpoint_interval,
        window_size=window_size,
    )
    opt_checkpoints = len(tta_opt._checkpoints)
    print(f"  Checkpoints created: {opt_checkpoints}")
    print(f"  Checkpoint storage: {opt_checkpoints * 32 / 1024:.2f} KB")
    
    # Define test positions
    positions = []
    for pct in [0, 10, 25, 50, 75, 90, 99]:
        pos = max(0, int(chain_length * pct / 100) - 1)
        n_hashes_expected = chain_length - 1 - pos
        positions.append((pos, n_hashes_expected))
        print(f"  Position {pos} ({pct}% from start, {n_hashes_expected} hashes to tip)")
    
    # Benchmark original
    print("\n[Original] Benchmarking...")
    orig_results = benchmark_verify(tta_orig, positions, n_repeats=30, label='original')
    
    # Reset optimized instance (clear sliding window)
    tta_opt._sliding_window = []
    
    # Benchmark optimized
    print("[Optimized] Benchmarking...")
    opt_results = benchmark_verify(tta_opt, positions, n_repeats=30, label='optimized')
    
    # Benchmark sliding window hit rate
    print("\n[Sliding Window] Benchmarking hot token hit rate...")
    hot_positions = []
    hot_token = tta_opt._chain[chain_length - 2]
    for _ in range(20):
        tta_opt.verify_token(chain_length - 2, hot_token)
    
    window_hit_before = len(tta_opt._sliding_window)
    hot_latencies = []
    for _ in range(100):
        start = time.perf_counter()
        tta_opt.verify_token(chain_length - 2, hot_token)
        hot_latencies.append((time.perf_counter() - start) * 1_000_000)
    avg_hot_lat = sum(hot_latencies) / len(hot_latencies)
    
    # Compile comparison
    comparison = {}
    speedups = []
    for key in orig_results:
        pos = orig_results[key]['position']
        orig_lat = orig_results[key]['avg_latency_us']
        opt_lat = opt_results[key]['avg_latency_us']
        speedup = orig_lat / max(opt_lat, 0.001)
        speedups.append(speedup)
        
        comparison[f'pos_{pos}'] = {
            'position': pos,
            'distance_from_tip': orig_results[key]['distance_from_tip'],
            'expected_hashes': orig_results[key]['distance_from_tip'],
            'original_latency_us': orig_lat,
            'optimized_latency_us': opt_lat,
            'speedup_x': round(speedup, 2),
        }
        
        print(f"  pos {pos:6d}: orig={orig_lat:8.2f}us → opt={opt_lat:8.2f}us  ({speedup:.1f}x)")
    
    avg_speedup = sum(speedups) / len(speedups)
    max_speedup = max(speedups)
    
    return {
        'chain_length': chain_length,
        'checkpoint_interval': checkpoint_interval,
        'window_size': window_size,
        'num_checkpoints': opt_checkpoints,
        'checkpoint_storage_kb': round(opt_checkpoints * 32 / 1024, 2),
        'orig': orig_results,
        'opt': opt_results,
        'comparison': comparison,
        'avg_speedup_x': round(avg_speedup, 2),
        'max_speedup_x': round(max_speedup, 2),
        'hot_token_latency_us': round(avg_hot_lat, 3),
    }


def main():
    parser = argparse.ArgumentParser(description='Hash Chain Performance Benchmark')
    parser.add_argument('--chain-lengths', type=int, nargs='+',
                        default=[1000, 8760, 100000],
                        help='Chain lengths to benchmark')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Checkpoint interval for optimized version')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Sliding window size')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hash Chain Performance Benchmark")
    print("=" * 60)
    print(f"Chain lengths: {args.chain_lengths}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Window size: {args.window_size}")
    
    all_results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'chain_lengths': args.chain_lengths,
            'checkpoint_interval': args.checkpoint_interval,
            'window_size': args.window_size,
        },
        'chains': [],
    }
    
    for length in args.chain_lengths:
        result = run_chain_benchmark(
            length,
            checkpoint_interval=args.checkpoint_interval,
            window_size=args.window_size,
        )
        all_results['chains'].append(result)
        
        print(f"\n  Summary for chain length {length}:")
        print(f"    Avg speedup: {result['avg_speedup_x']}x")
        print(f"    Max speedup: {result['max_speedup_x']}x")
        print(f"    Checkpoints: {result['num_checkpoints']} ({result['checkpoint_storage_kb']} KB)")
        print(f"    Hot token (window hit): {result['hot_token_latency_us']:.3f}us")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("Overall Summary")
    print(f"{'='*60}")
    print(f"{'Chain Len':<12} {'Avg Speedup':<14} {'Max Speedup':<14} {'Checkpoints':<14} {'Hot Token(us)':<14}")
    print(f"{'-'*68}")
    for c in all_results['chains']:
        print(f"{c['chain_length']:<12} {c['avg_speedup_x']:<14.1f}x {c['max_speedup_x']:<14.1f}x "
              f"{c['num_checkpoints']:<14} {c['hot_token_latency_us']:<14.3f}")
    
    # Save results
    output_path = args.output or (RESULTS_DIR / 'hash_chain_performance.json')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
