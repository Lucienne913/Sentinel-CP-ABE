"""
Policy Scalability Tests: Fixed user attribute count, varying policy attribute count, measure encrypt/decrypt time

Purpose: Prove that decrypt time varies with policy size (not constant), address scalability concerns
"""

import pytest
import time
import gc
import json
import sys
from pathlib import Path
from charm.toolbox.pairinggroup import GT

sys.path.insert(0, str(Path(__file__).parent.parent))

POLICY_SIZES = [10, 50, 100, 500, 1000]
NUM_REPEATS = 20
FIXED_USER_ATTRS = 1010  # User attribute count >= max policy attribute count (1000), ensure decryption possible


def make_attrs(n):
    return [f'attr_{i}' for i in range(n)]


def make_policy(attrs):
    """Generate AND policy, max 10 attributes per nested group"""
    if len(attrs) <= 10:
        return '(' + ' AND '.join(attrs) + ')'
    # Nested grouping: each 10 attributes as a group, joined with AND
    groups = []
    for i in range(0, len(attrs), 10):
        chunk = attrs[i:i+10]
        groups.append('(' + ' AND '.join(chunk) + ')')
    return ' AND '.join(groups)


def compute_timing_stats(times_ms):
    sorted_times = sorted(times_ms)
    p95_idx = int(len(sorted_times) * 0.95)
    return {
        'mean_ms': round(sum(times_ms) / len(times_ms), 2),
        'std_ms': round((sum((x - sum(times_ms)/len(times_ms))**2 for x in times_ms) / (len(times_ms)-1))**0.5, 2) if len(times_ms) > 1 else 0,
        'min_ms': round(min(times_ms), 2),
        'max_ms': round(max(times_ms), 2),
        'p95_ms': round(sorted_times[p95_idx], 2),
        'num_repeats': len(times_ms),
    }


@pytest.mark.slow
class TestPolicyScalability:

    def test_policy_scalability(self, system_setup, heartbeat, oom_protection):
        """Policy scalability: fixed user attributes, varying policy attribute count"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']

        user_attrs = make_attrs(FIXED_USER_ATTRS)
        SK = tcabe.keygen(MSK, user_attrs)

        results = {}
        total_steps = len(POLICY_SIZES)
        step = 0

        for policy_size in POLICY_SIZES:
            gc.collect()
            policy_attrs = make_attrs(policy_size)
            policy = make_policy(policy_attrs)
            policy_tree = parser.parse(policy)

            encrypt_times = []
            decrypt_times = []

            for rep in range(NUM_REPEATS):
                M = group.random(GT)

                start = time.time()
                CT = tcabe.encrypt(M, policy_tree)
                encrypt_times.append((time.time() - start) * 1000)

                start = time.time()
                decrypted = tcabe.decrypt(SK, CT)
                decrypt_times.append((time.time() - start) * 1000)

                assert decrypted == M

            results[policy_size] = {
                'encrypt': compute_timing_stats(encrypt_times),
                'decrypt': compute_timing_stats(decrypt_times),
            }

            step += 1
            enc_mean = results[policy_size]['encrypt']['mean_ms']
            dec_mean = results[policy_size]['decrypt']['mean_ms']
            heartbeat(step, total_steps,
                      f'Policy {policy_size} attrs ({NUM_REPEATS} reps): enc={enc_mean:.1f}ms dec={dec_mean:.1f}ms')
            oom_protection.check_and_protect()

        # Save results
        output_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'scalability'
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'policy_scalability_table.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print(f"Policy Scalability (fixed {FIXED_USER_ATTRS} user attrs, {NUM_REPEATS} reps)")
        print("=" * 80)
        print(f"{'Policy Attrs':12} {'Enc mean':12} {'std':8} {'Dec mean':12} {'std':8}")
        print("-" * 80)
        for ps, r in results.items():
            print(f"{ps:12} {r['encrypt']['mean_ms']:10.1f}ms {r['encrypt']['std_ms']:6.1f}ms "
                  f"{r['decrypt']['mean_ms']:10.1f}ms {r['decrypt']['std_ms']:6.1f}ms")
        print("=" * 80)