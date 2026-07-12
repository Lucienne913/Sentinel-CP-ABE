#!/usr/bin/env python3
"""
SOTA Comparison Tests

Real baseline implementations:
1. BSW07 CP-ABE - Charm-Crypto built-in (Bethencourt-Sahai-Waters 2007)
2. Lightweight ABE - BSW07 engine with AND-gate only policies

Other recent schemes: cite published data (see experiments/SOTA_Comparison_Guide.md)
"""

import pytest
import gc
import time
import json
import sys
import statistics
from pathlib import Path
from charm.toolbox.pairinggroup import GT

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.bsw07_abe import BSW07Baseline
from src.baselines.lightweight_abe import BSW07_AND


ATTR_COUNTS = [10, 50, 100, 200, 500, 1000, 5000, 10000]
NUM_REPEATS = 100


def get_repeats(num_attrs):
    """Adaptive repeat count: 100 repeats for small scale, reduced for large scale to ensure feasibility"""
    if num_attrs <= 500:
        return 100
    elif num_attrs <= 1000:
        return 20
    elif num_attrs <= 5000:
        return 10
    else:
        return 5


def make_attrs(n):
    attrs = []
    for i in range(n):
        if i < 26:
            attrs.append(chr(65 + i))
        else:
            attrs.append(chr(65 + (i % 26)) + str(i // 26))
    return attrs


def make_policy(attrs, max_depth=10, operator='AND'):
    return '(' + f' {operator} '.join(attrs[:min(max_depth, len(attrs))]) + ')'


def compute_timing_stats(times_ms):
    sorted_times = sorted(times_ms)
    p95_idx = int(len(sorted_times) * 0.95)
    return {
        'mean_ms': round(statistics.mean(times_ms), 2),
        'std_ms': round(statistics.stdev(times_ms), 2) if len(times_ms) > 1 else 0,
        'min_ms': round(min(times_ms), 2),
        'max_ms': round(max(times_ms), 2),
        'p95_ms': round(sorted_times[p95_idx], 2),
        'num_repeats': len(times_ms),
    }


@pytest.mark.slow
@pytest.mark.comparison
class TestSOTAComparison:

    @staticmethod
    def _save_comparison_data(data, filename):
        output_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def test_bsw07_baseline(self, system_setup, heartbeat, oom_protection):
        PP = system_setup['PP']
        group = PP['group']

        bsw07 = BSW07Baseline('SS1024')
        bsw07.setup()

        results = {}
        total_steps = len(ATTR_COUNTS)
        step = 0

        for num_attrs in ATTR_COUNTS:
            gc.collect()
            attrs = make_attrs(num_attrs)
            policy = make_policy(attrs, operator='and')
            num_reps = get_repeats(num_attrs)

            keygen_times = []
            encrypt_times = []
            decrypt_times = []

            for rep in range(num_reps):
                start = time.time()
                sk = bsw07.keygen(attrs)
                keygen_times.append((time.time() - start) * 1000)

                M = group.random(GT)

                start = time.time()
                ct = bsw07.encrypt(M, policy)
                encrypt_times.append((time.time() - start) * 1000)

                start = time.time()
                decrypted = bsw07.decrypt(ct, sk)
                decrypt_times.append((time.time() - start) * 1000)

                assert decrypted == M

            results[num_attrs] = {
                'keygen': compute_timing_stats(keygen_times),
                'encrypt': compute_timing_stats(encrypt_times),
                'decrypt': compute_timing_stats(decrypt_times),
            }

            step += 1
            kg_mean = results[num_attrs]['keygen']['mean_ms']
            heartbeat(step, total_steps,
                      f'BSW07 {num_attrs} attrs ({num_reps} reps): keygen={kg_mean:.1f}ms')
            oom_protection.check_and_protect()

        self._save_comparison_data(results, 'bsw07_baseline.json')

        print("\n" + "=" * 80)
        print(f"BSW07 CP-ABE Baseline (adaptive repetitions)")
        print("=" * 80)
        print(f"{'Attrs':8} {'KeyGen mean':12} {'std':8} {'Enc mean':12} {'std':8} {'Dec mean':12} {'std':8}")
        print("-" * 80)
        for num_attrs, r in results.items():
            print(f"{num_attrs:8} {r['keygen']['mean_ms']:10.1f}ms {r['keygen']['std_ms']:6.1f}ms "
                  f"{r['encrypt']['mean_ms']:10.1f}ms {r['encrypt']['std_ms']:6.1f}ms "
                  f"{r['decrypt']['mean_ms']:10.1f}ms {r['decrypt']['std_ms']:6.1f}ms")
        print("=" * 80)

    def test_lightweight_abe_baseline(self, system_setup, heartbeat, oom_protection):
        PP = system_setup['PP']
        group = PP['group']

        lw_abe = BSW07_AND('SS1024')
        lw_abe.setup()

        results = {}
        total_steps = len(ATTR_COUNTS)
        step = 0

        for num_attrs in ATTR_COUNTS:
            gc.collect()
            attrs = make_attrs(num_attrs)
            num_reps = get_repeats(num_attrs)

            keygen_times = []
            encrypt_times = []
            decrypt_times = []

            for rep in range(num_reps):
                start = time.time()
                sk = lw_abe.keygen(attrs)
                keygen_times.append((time.time() - start) * 1000)

                M = group.random(GT)

                start = time.time()
                ct = lw_abe.encrypt(M, attrs[:10])
                encrypt_times.append((time.time() - start) * 1000)

                start = time.time()
                decrypted = lw_abe.decrypt(ct, sk)
                decrypt_times.append((time.time() - start) * 1000)

                assert decrypted == M

            results[num_attrs] = {
                'keygen': compute_timing_stats(keygen_times),
                'encrypt': compute_timing_stats(encrypt_times),
                'decrypt': compute_timing_stats(decrypt_times),
            }

            step += 1
            kg_mean = results[num_attrs]['keygen']['mean_ms']
            heartbeat(step, total_steps,
                      f'BSW07-AND {num_attrs} attrs ({num_reps} reps): keygen={kg_mean:.1f}ms')
            oom_protection.check_and_protect()

        self._save_comparison_data(results, 'bsw07_and_baseline.json')

        print("\n" + "=" * 80)
        print(f"Lightweight ABE Baseline (adaptive repetitions)")
        print("=" * 80)
        print(f"{'Attrs':8} {'KeyGen mean':12} {'std':8} {'Enc mean':12} {'std':8} {'Dec mean':12} {'std':8}")
        print("-" * 80)
        for num_attrs, r in results.items():
            print(f"{num_attrs:8} {r['keygen']['mean_ms']:10.1f}ms {r['keygen']['std_ms']:6.1f}ms "
                  f"{r['encrypt']['mean_ms']:10.1f}ms {r['encrypt']['std_ms']:6.1f}ms "
                  f"{r['decrypt']['mean_ms']:10.1f}ms {r['decrypt']['std_ms']:6.1f}ms")
        print("=" * 80)

    def test_our_scheme_performance(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']

        results = {}
        total_steps = len(ATTR_COUNTS)
        step = 0

        for num_attrs in ATTR_COUNTS:
            gc.collect()
            attrs = make_attrs(num_attrs)
            policy = make_policy(attrs)
            policy_tree = parser.parse(policy)
            num_reps = get_repeats(num_attrs)

            keygen_times = []
            encrypt_times = []
            decrypt_times = []

            for rep in range(num_reps):
                start = time.time()
                SK = tcabe.keygen(MSK, attrs)
                keygen_times.append((time.time() - start) * 1000)

                M = group.random(GT)

                start = time.time()
                CT = tcabe.encrypt(M, policy_tree)
                encrypt_times.append((time.time() - start) * 1000)

                start = time.time()
                decrypted = tcabe.decrypt(SK, CT)
                decrypt_times.append((time.time() - start) * 1000)

                assert decrypted == M

            results[num_attrs] = {
                'keygen': compute_timing_stats(keygen_times),
                'encrypt': compute_timing_stats(encrypt_times),
                'decrypt': compute_timing_stats(decrypt_times),
            }

            step += 1
            kg_mean = results[num_attrs]['keygen']['mean_ms']
            heartbeat(step, total_steps,
                      f'OurScheme {num_attrs} attrs ({num_reps} reps): keygen={kg_mean:.1f}ms')
            oom_protection.check_and_protect()

        self._save_comparison_data(results, 'our_scheme_performance.json')

        print("\n" + "=" * 80)
        print(f"OurScheme T-CP-ABE Performance (adaptive repetitions)")
        print("=" * 80)
        print(f"{'Attrs':8} {'KeyGen mean':12} {'std':8} {'Enc mean':12} {'std':8} {'Dec mean':12} {'std':8}")
        print("-" * 80)
        for num_attrs, r in results.items():
            print(f"{num_attrs:8} {r['keygen']['mean_ms']:10.1f}ms {r['keygen']['std_ms']:6.1f}ms "
                  f"{r['encrypt']['mean_ms']:10.1f}ms {r['encrypt']['std_ms']:6.1f}ms "
                  f"{r['decrypt']['mean_ms']:10.1f}ms {r['decrypt']['std_ms']:6.1f}ms")
        print("=" * 80)

    def test_generate_comparison_table(self, system_setup, heartbeat, oom_protection):
        PP = system_setup['PP']
        group = PP['group']
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        parser = system_setup['parser']

        bsw07 = BSW07Baseline('SS1024')
        bsw07.setup()
        lw_abe = BSW07_AND('SS1024')
        lw_abe.setup()

        all_results = {}
        total_steps = len(ATTR_COUNTS) * 3
        step = 0

        for num_attrs in ATTR_COUNTS:
            gc.collect()
            attrs = make_attrs(num_attrs)
            bsw07_policy = make_policy(attrs, operator='and')
            our_policy = make_policy(attrs)

            start = time.time()
            sk = bsw07.keygen(attrs)
            keygen_time = time.time() - start
            M = group.random(GT)
            start = time.time()
            ct = bsw07.encrypt(M, bsw07_policy)
            encrypt_time = time.time() - start
            start = time.time()
            decrypted = bsw07.decrypt(ct, sk)
            decrypt_time = time.time() - start
            assert decrypted == M
            all_results[f'BSW07_{num_attrs}'] = {
                'scheme': 'BSW07', 'num_attrs': num_attrs,
                'keygen_ms': round(keygen_time * 1000, 2),
                'encrypt_ms': round(encrypt_time * 1000, 2),
                'decrypt_ms': round(decrypt_time * 1000, 2),
            }
            step += 1
            heartbeat(step, total_steps, f'BSW07 {num_attrs} attrs')
            oom_protection.check_and_protect()

            start = time.time()
            sk = lw_abe.keygen(attrs)
            keygen_time = time.time() - start
            M = group.random(GT)
            start = time.time()
            ct = lw_abe.encrypt(M, attrs[:10])
            encrypt_time = time.time() - start
            start = time.time()
            decrypted = lw_abe.decrypt(ct, sk)
            decrypt_time = time.time() - start
            assert decrypted == M
            all_results[f'BSW07-AND_{num_attrs}'] = {
                'scheme': 'BSW07-AND', 'num_attrs': num_attrs,
                'keygen_ms': round(keygen_time * 1000, 2),
                'encrypt_ms': round(encrypt_time * 1000, 2),
                'decrypt_ms': round(decrypt_time * 1000, 2),
            }
            step += 1
            heartbeat(step, total_steps, f'BSW07-AND {num_attrs} attrs')
            oom_protection.check_and_protect()

            policy_tree = parser.parse(our_policy)
            M = group.random(GT)
            start = time.time()
            SK = tcabe.keygen(MSK, attrs)
            keygen_time = time.time() - start
            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            encrypt_time = time.time() - start
            start = time.time()
            decrypted = tcabe.decrypt(SK, CT)
            decrypt_time = time.time() - start
            assert decrypted == M
            all_results[f'OurScheme_{num_attrs}'] = {
                'scheme': 'OurScheme (T-CP-ABE)', 'num_attrs': num_attrs,
                'keygen_ms': round(keygen_time * 1000, 2),
                'encrypt_ms': round(encrypt_time * 1000, 2),
                'decrypt_ms': round(decrypt_time * 1000, 2),
            }
            step += 1
            heartbeat(step, total_steps, f'OurScheme {num_attrs} attrs')
            oom_protection.check_and_protect()

        self._save_comparison_data(all_results, 'sota_comparison_table.json')

        print("\n" + "=" * 100)
        print("SOTA Comparison Table (All Real Measurements)")
        print("Note: Other recent schemes - cite published data (see experiments/SOTA_Comparison_Guide.md)")
        print("=" * 100)
        print(f"{'Scheme':25} {'Attrs':10} {'KeyGen(ms)':12} {'Encrypt(ms)':12} {'Decrypt(ms)':12}")
        print("-" * 100)
        for key, r in all_results.items():
            print(f"{r['scheme']:25} {r['num_attrs']:10} {r['keygen_ms']:11.1f} {r['encrypt_ms']:11.1f} {r['decrypt_ms']:11.1f}")
        print("=" * 100)
