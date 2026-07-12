#!/usr/bin/env python3
"""E3: AI closed-loop - end-to-end latency from anomaly detection to decryption rejection."""
import sys, os, json, statistics, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, PolicyParser
from charm.toolbox.pairinggroup import GT

n_trials = 5

if __name__ == '__main__':
    print("=" * 60)
    print("E3: AI Closed-Loop End-to-End Latency")
    print("=" * 60)

    setup_obj = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
    PP, MSK = setup_obj.setup(max_attrs=50)
    parser = PolicyParser()
    group = PP['group']
    msg = group.random(GT)
    work_time = datetime(2026, 6, 3, 14, 30)

    all_attrs = [f'attr{i}' for i in range(20)]
    k = 10

    results = []

    # Simulate the full closed loop:
    # 1. Anomaly detected (simulated as a flag)
    # 2. TA revokes compromised attribute
    # 3. TA issues new key to honest users
    # 4. Attacker tries to decrypt with old key -> fails

    print("\n--- Full Loop: Anomaly -> Revoke -> Re-encrypt -> Reject ---")

    for n_revoke in [1, 3, 5]:
        print(f"\n  Revoking {n_revoke} attributes...")
        loop_times = []

        for _ in range(n_trials):
            tcabe = T_CP_ABE(PP)
            SK = tcabe.keygen(MSK, all_attrs)
            policy_tree = parser.parse(f"THRESHOLD({k}, " + ", ".join(all_attrs) + ")")
            CT = tcabe.encrypt(msg, policy_tree)

            total_start = time.perf_counter()

            # Step 1: Anomaly detection trigger (instant, just a flag)
            # Step 2: TA revokes attributes
            revoke_attrs = [f'attr{i}' for i in range(n_revoke)]
            for attr in revoke_attrs:
                tcabe.revoke_attribute(attr)

            # Step 3: Re-encrypt with new version
            CT_new = tcabe.encrypt(msg, policy_tree)

            # Step 4: New key generation for honest users
            SK_new = tcabe.keygen(MSK, all_attrs)

            # Step 5: Verify new key works
            result_new = tcabe.decrypt(SK_new, CT_new, work_time)
            assert result_new == msg

            # Step 6: Verify old key fails on new CT
            fail = False
            try:
                result_old = tcabe.decrypt(SK, CT_new, work_time)
                if result_old != msg:
                    fail = True
            except:
                fail = True

            total_time = (time.perf_counter() - total_start) * 1000
            loop_times.append(total_time)

            assert fail, "Old key should fail on new ciphertext"

        result = {
            'n_revoke': n_revoke,
            'loop_time_ms': round(statistics.mean(loop_times), 1),
            'loop_time_std': round(statistics.stdev(loop_times) if len(loop_times) > 1 else 0, 1),
            'old_key_fail_rate': 100,
        }
        results.append(result)
        print(f"    Total loop time: {result['loop_time_ms']} ms")
        print(f"    Old key fail rate: 100%")

    # Individual step timings
    print("\n--- Individual Step Timings ---")
    tcabe = T_CP_ABE(PP)
    SK = tcabe.keygen(MSK, all_attrs)
    policy_tree = parser.parse(f"THRESHOLD({k}, " + ", ".join(all_attrs) + ")")

    # Revoke 1 attribute
    step_times = {'revoke': [], 're_encrypt': [], 'keygen': [], 'decrypt': []}
    for _ in range(n_trials):
        tcabe2 = T_CP_ABE(PP)
        SK2 = tcabe2.keygen(MSK, all_attrs)
        CT2 = tcabe2.encrypt(msg, policy_tree)

        t0 = time.perf_counter()
        tcabe2.revoke_attribute('attr0')
        step_times['revoke'].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        CT_new = tcabe2.encrypt(msg, policy_tree)
        step_times['re_encrypt'].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        SK_new = tcabe2.keygen(MSK, all_attrs)
        step_times['keygen'].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = tcabe2.decrypt(SK_new, CT_new, work_time)
        step_times['decrypt'].append((time.perf_counter() - t0) * 1000)

    for step, times in step_times.items():
        mean = statistics.mean(times)
        print(f"  {step:15s}: {mean:>8.1f} ms")

    output_path = '/app/experiments/results/ai_closed_loop.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("Summary: End-to-End Closed Loop Latency")
    print("=" * 60)
    for r in results:
        print(f"  Revoke {r['n_revoke']} attrs: {r['loop_time_ms']} ms total, old key fail: {r['old_key_fail_rate']}%")
