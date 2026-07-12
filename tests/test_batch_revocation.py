#!/usr/bin/env python3
"""Experiment 3: Batch attribute revocation performance."""
import sys, os, time, json, statistics
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, PolicyParser
from charm.toolbox.pairinggroup import GT

n_trials = 5

def make_threshold_policy(parser, attrs, k):
    policy_str = f"THRESHOLD({k}, " + ", ".join(attrs) + ")"
    return parser.parse(policy_str)

if __name__ == '__main__':
    print("=" * 60)
    print("Experiment 3: Batch Attribute Revocation")
    print("=" * 60)

    setup_obj = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
    PP, MSK = setup_obj.setup(max_attrs=100)
    parser = PolicyParser()
    msg = PP['group'].random(GT)
    work_time = datetime(2026, 6, 3, 14, 30)

    total_attrs = 20
    all_attrs = [f'attr{i}' for i in range(total_attrs)]
    revoke_counts = [1, 5, 10, 20]
    results = []

    for n_revoke in revoke_counts:
        print(f"\n--- Revoking {n_revoke}/{total_attrs} attributes ---")

        # Fresh setup for each test
        tcabe = T_CP_ABE(PP)

        # Generate keys for all attributes
        SK = tcabe.keygen(MSK, all_attrs)

        # Use threshold policy requiring half the attributes
        k = total_attrs // 2
        policy_tree = make_threshold_policy(parser, all_attrs, k)
        CT = tcabe.encrypt(msg, policy_tree)

        # Verify decryption works before revocation
        result_before = tcabe.decrypt(SK, CT, work_time)
        assert result_before == msg, "Decryption should work before revocation"

        # Batch revoke
        revoke_attrs = all_attrs[:n_revoke]
        revoke_times = []
        for _ in range(n_trials):
            tcabe2 = T_CP_ABE(PP)
            tcabe2.keygen(MSK, all_attrs)
            start = time.perf_counter()
            for attr in revoke_attrs:
                tcabe2.revoke_attribute(attr)
            revoke_times.append((time.perf_counter() - start) * 1000)

        # Verify decryption fails after revocation
        tcabe_failed = T_CP_ABE(PP)
        tcabe_failed.keygen(MSK, all_attrs)
        for attr in revoke_attrs:
            tcabe_failed.revoke_attribute(attr)

        decrypt_fail_count = 0
        decrypt_times_after = []
        for _ in range(n_trials):
            start = time.perf_counter()
            try:
                result_after = tcabe_failed.decrypt(SK, CT, work_time)
                if result_after != msg:
                    decrypt_fail_count += 1
            except:
                decrypt_fail_count += 1
            decrypt_times_after.append((time.perf_counter() - start) * 1000)

        # Encrypt with new version (should work with fresh key)
        tcabe_new = T_CP_ABE(PP)
        for attr in revoke_attrs:
            tcabe_new.revoke_attribute(attr)
        new_SK = tcabe_new.keygen(MSK, all_attrs)
        new_CT = tcabe_new.encrypt(msg, policy_tree)

        encrypt_times_new = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = tcabe_new.encrypt(msg, policy_tree)
            encrypt_times_new.append((time.perf_counter() - start) * 1000)

        result = {
            'n_revoke': n_revoke,
            'n_total': total_attrs,
            'revoke_pct': round(n_revoke / total_attrs * 100),
            'revoke_time_ms': round(statistics.mean(revoke_times), 1),
            'revoke_time_std': round(statistics.stdev(revoke_times) if len(revoke_times) > 1 else 0, 1),
            'old_key_decrypt_fail_rate': round(decrypt_fail_count / n_trials * 100),
            'new_encrypt_ms': round(statistics.mean(encrypt_times_new), 1),
            'new_encrypt_std': round(statistics.stdev(encrypt_times_new) if len(encrypt_times_new) > 1 else 0, 1),
        }
        results.append(result)
        print(f"  Revoke time: {result['revoke_time_ms']} ms")
        print(f"  Old key decrypt fail rate: {result['old_key_decrypt_fail_rate']}%")
        print(f"  New encrypt time: {result['new_encrypt_ms']} ms")

    output_path = '/app/experiments/results/batch_revocation.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Revoke':>8} | {'Revoke%':>8} | {'Revoke Time (ms)':>17} | {'Fail Rate':>10} | {'New Encrypt (ms)':>17}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_revoke']:>8} | {r['revoke_pct']:>7}% | {r['revoke_time_ms']:>17.1f} | {r['old_key_decrypt_fail_rate']:>9}% | {r['new_encrypt_ms']:>17.1f}")
