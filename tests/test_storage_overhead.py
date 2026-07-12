#!/usr/bin/env python3
"""Experiment 4: Measure actual key size and ciphertext size in bytes."""
import sys, os, time, json, pickle, statistics
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, PolicyParser
from charm.toolbox.pairinggroup import GT, ZR, G1, G2

n_trials = 5

def measure_element_size(elem, group):
    """Get serialized size of a Charm group element."""
    return len(group.serialize(elem))

def measure_sk_size(SK, group):
    """Measure secret key size by serializing each group element."""
    total = 0
    total += measure_element_size(SK['K0'], group)
    if 'K_time' in SK:
        total += measure_element_size(SK['K_time'], group)
    for attr, components in SK['K'].items():
        total += measure_element_size(components['K_attr'], group)
        total += measure_element_size(components['K_prime_attr'], group)
    return total

def measure_ct_size(CT, group):
    """Measure ciphertext size by serializing each group element."""
    total = 0
    for key in ['C0', 'C1', 'C2', 'C_tilde', 'policy']:
        if key in CT:
            total += measure_element_size(CT[key], group)
    if 'leaf_components' in CT:
        for leaf_id, comp in CT['leaf_components'].items():
            for v in comp.values():
                total += measure_element_size(v, group)
    # Fallback: measure all values that are group elements
    for key, val in CT.items():
        if key in ('C0', 'C1', 'C2', 'C_tilde', 'policy', 'leaf_components'):
            continue
        try:
            total += measure_element_size(val, group)
        except Exception:
            pass
    return total

def make_and_policy(parser, n):
    attrs = [f'attr{i}' for i in range(n)]
    policy_str = "AND(" + ", ".join(attrs) + ")"
    return parser.parse(policy_str), attrs

if __name__ == '__main__':
    print("=" * 60)
    print("Experiment 4: Storage Overhead Measurement")
    print("=" * 60)

    setup_obj = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
    PP, MSK = setup_obj.setup(max_attrs=200)
    tcabe = T_CP_ABE(PP)
    parser = PolicyParser()
    group = PP['group']
    msg = group.random(GT)
    work_time = datetime(2026, 6, 3, 14, 30)

    attr_counts = [10, 20, 50, 100]
    results = []

    for n in attr_counts:
        print(f"\n--- Attributes: {n} ---")
        policy_tree, attrs = make_and_policy(parser, n)
        user_attrs = attrs[:n]

        sk_sizes = []
        ct_sizes = []
        for trial in range(n_trials):
            SK = tcabe.keygen(MSK, user_attrs)
            CT = tcabe.encrypt(msg, policy_tree)

            sk_size = measure_sk_size(SK, group)
            ct_size = measure_ct_size(CT, group)

            sk_sizes.append(sk_size)
            ct_sizes.append(ct_size)

        result = {
            'n_attrs': n,
            'sk_size_bytes': round(statistics.mean(sk_sizes)),
            'sk_size_kb': round(statistics.mean(sk_sizes) / 1024, 2),
            'ct_size_bytes': round(statistics.mean(ct_sizes)),
            'ct_size_kb': round(statistics.mean(ct_sizes) / 1024, 2),
            'sk_per_attr_bytes': round(statistics.mean(sk_sizes) / n),
        }
        results.append(result)
        print(f"  SK: {result['sk_size_bytes']} bytes ({result['sk_size_kb']} KB), "
              f"per attr: {result['sk_per_attr_bytes']} bytes")
        print(f"  CT: {result['ct_size_bytes']} bytes ({result['ct_size_kb']} KB)")

    # Save results
    output_path = '/app/experiments/results/storage_overhead.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Attrs':>6} | {'SK (KB)':>10} | {'CT (KB)':>10} | {'SK/Attr (B)':>12}")
    print("-" * 50)
    for r in results:
        print(f"{r['n_attrs']:>6} | {r['sk_size_kb']:>10.2f} | {r['ct_size_kb']:>10.2f} | {r['sk_per_attr_bytes']:>12}")
