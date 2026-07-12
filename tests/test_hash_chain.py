#!/usr/bin/env python3
"""Experiment 2: Long hash chain performance test."""
import sys, os, time, json, statistics, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def sha256_hash(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()

def build_hash_chain(seed, N):
    """Build hash chain of length N: T_0=H(seed), T_i=H(T_{i-1})."""
    chain = [None] * (N + 1)
    chain[0] = sha256_hash(seed)
    for i in range(1, N + 1):
        chain[i] = sha256_hash(chain[i - 1])
    return chain

def verify_token(chain_tip, token, N, i):
    """Verify T_i by computing H^{N-i}(T_i) and comparing with T_N."""
    current = token
    for _ in range(N - i):
        current = sha256_hash(current)
    return current == chain_tip

def verify_with_checkpoint(checkpoints, checkpoint_interval, token, N, i, chain_tip):
    """Verify using nearest checkpoint."""
    # Find nearest checkpoint <= i
    nearest_ckpt_idx = (i // checkpoint_interval) * checkpoint_interval
    if nearest_ckpt_idx == 0:
        # No useful checkpoint before, verify from token to tip
        return verify_token(chain_tip, token, N, i)

    ckpt = checkpoints[nearest_ckpt_idx]
    # Hash FROM checkpoint TO token: H^{i - ckpt_idx}(ckpt) should == token
    # (because T_i = H^{i-ckpt}(T_{ckpt}))
    current = ckpt
    steps_to_token = i - nearest_ckpt_idx
    for _ in range(steps_to_token):
        current = sha256_hash(current)
    if current != token:
        return False
    # Hash from checkpoint to tip
    current = ckpt
    steps_to_tip = N - nearest_ckpt_idx
    for _ in range(steps_to_tip):
        current = sha256_hash(current)
    return current == chain_tip

if __name__ == '__main__':
    print("=" * 60)
    print("Experiment 2: Long Hash Chain Performance")
    print("=" * 60)

    chain_lengths = [100, 500, 1000, 5000, 10000]
    checkpoint_interval = 1000
    n_trials = 5
    results = []

    for N in chain_lengths:
        print(f"\n--- Chain Length N={N} ---")

        # 1. Precomputation time
        precomp_times = []
        for _ in range(n_trials):
            seed = os.urandom(32).hex()
            start = time.perf_counter()
            chain = build_hash_chain(seed, N)
            precomp_times.append((time.perf_counter() - start) * 1000)

        # 2. Token generation (just index lookup, but measure hash derivation)
        # Token i is chain[i], which is already computed
        # In practice, TTA stores chain and releases T_i
        token_gen_times = []
        for _ in range(n_trials):
            i = N // 2  # middle of chain
            start = time.perf_counter()
            _ = chain[i]
            token_gen_times.append((time.perf_counter() - start) * 1000)

        # 3. Verification WITHOUT checkpoint: H^{N-i}(T_i) == T_N
        verify_times = []
        i = N // 2
        for _ in range(n_trials):
            start = time.perf_counter()
            result = verify_token(chain[N], chain[i], N, i)
            verify_times.append((time.perf_counter() - start) * 1000)
            assert result

        # 4. Verification WITH checkpoint
        checkpoints = {}
        for idx in range(0, N + 1, checkpoint_interval):
            checkpoints[idx] = chain[idx]

        verify_ckpt_times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            result = verify_with_checkpoint(checkpoints, checkpoint_interval, chain[i], N, i, chain[N])
            verify_ckpt_times.append((time.perf_counter() - start) * 1000)
            assert result

        # 5. Token generation from seed (full chain build)
        token_from_seed_times = []
        for _ in range(n_trials):
            seed = os.urandom(32).hex()
            start = time.perf_counter()
            c = build_hash_chain(seed, N)
            _ = c[i]
            token_from_seed_times.append((time.perf_counter() - start) * 1000)

        result = {
            'N': N,
            'precompute_ms': round(statistics.mean(precomp_times), 1),
            'precompute_std': round(statistics.stdev(precomp_times) if len(precomp_times) > 1 else 0, 1),
            'token_gen_ms': round(statistics.mean(token_gen_times), 3),
            'verify_no_ckpt_ms': round(statistics.mean(verify_times), 1),
            'verify_no_ckpt_std': round(statistics.stdev(verify_times) if len(verify_times) > 1 else 0, 1),
            'verify_with_ckpt_ms': round(statistics.mean(verify_ckpt_times), 3),
            'token_from_seed_ms': round(statistics.mean(token_from_seed_times), 1),
        }
        results.append(result)
        print(f"  Precompute: {result['precompute_ms']} ms")
        print(f"  Token gen (lookup): {result['token_gen_ms']} ms")
        print(f"  Verify (no checkpoint, i=N/2): {result['verify_no_ckpt_ms']} ms")
        print(f"  Verify (with checkpoint): {result['verify_with_ckpt_ms']} ms")
        print(f"  Full chain build from seed: {result['token_from_seed_ms']} ms")

    output_path = '/app/experiments/results/hash_chain_performance.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'N':>6} | {'Precompute (ms)':>16} | {'Verify no-ckpt (ms)':>20} | {'Verify ckpt (ms)':>18}")
    print("-" * 70)
    for r in results:
        print(f"{r['N']:>6} | {r['precompute_ms']:>16.1f} | {r['verify_no_ckpt_ms']:>20.1f} | {r['verify_with_ckpt_ms']:>18.3f}")
