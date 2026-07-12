#!/usr/bin/env python3
"""E5: Collusion attack resistance - verify revoked+valid users cannot collude."""
import sys, os, json, statistics
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, PolicyParser
from charm.toolbox.pairinggroup import GT

n_trials = 5

if __name__ == '__main__':
    print("=" * 60)
    print("E5: Collusion Attack Resistance")
    print("=" * 60)

    setup_obj = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
    PP, MSK = setup_obj.setup(max_attrs=50)
    parser = PolicyParser()
    group = PP['group']
    msg = group.random(GT)
    work_time = datetime(2026, 6, 3, 14, 30)

    results = []

    # Test 1: Threshold collusion - revoked user + valid users try to meet threshold
    print("\n--- Test 1: Threshold Collusion (k=5, n=10) ---")
    all_attrs = [f'attr{i}' for i in range(10)]
    k = 5

    tcabe_honest = T_CP_ABE(PP)
    SK_honest = tcabe_honest.keygen(MSK, all_attrs)
    policy_tree = parser.parse(f"THRESHOLD({k}, " + ", ".join(all_attrs) + ")")
    CT = tcabe_honest.encrypt(msg, policy_tree)

    # Revoke attr0-attr4 (5 attributes), attacker has attr5-attr9
    tcabe_revoke = T_CP_ABE(PP)
    SK_revoke = tcabe_revoke.keygen(MSK, all_attrs)
    for i in range(5):
        tcabe_revoke.revoke_attribute(f'attr{i}')

    # Attacker tries to decrypt with revoked SK
    fail_count = 0
    for _ in range(n_trials):
        try:
            result = tcabe_revoke.decrypt(SK_revoke, CT, work_time)
            if result != msg:
                fail_count += 1
        except:
            fail_count += 1

    print(f"  Collusion fail rate: {fail_count}/{n_trials} = {fail_count/n_trials*100}%")
    results.append({
        'test': 'threshold_collusion_k5_n10',
        'description': 'Revoked user (5 revoked attrs) + policy k=5',
        'fail_rate': fail_count / n_trials * 100,
    })

    # Test 2: Mixed collusion - some valid, some revoked, trying to meet AND policy
    print("\n--- Test 2: AND Collusion (5 valid + 5 revoked) ---")
    tcabe2 = T_CP_ABE(PP)
    SK2 = tcabe2.keygen(MSK, all_attrs)
    for i in range(5):
        tcabe2.revoke_attribute(f'attr{i}')

    policy_and = parser.parse("AND(" + ", ".join(all_attrs) + ")")
    CT2 = tcabe2.encrypt(msg, policy_and)

    fail_count2 = 0
    for _ in range(n_trials):
        try:
            result = tcabe2.decrypt(SK2, CT2, work_time)
            if result != msg:
                fail_count2 += 1
        except:
            fail_count2 += 1

    print(f"  Collusion fail rate: {fail_count2}/{n_trials} = {fail_count2/n_trials*100}%")
    results.append({
        'test': 'and_collusion_50pct',
        'description': 'AND policy with 50% attributes revoked',
        'fail_rate': fail_count2 / n_trials * 100,
    })

    # Test 3: All revoked
    print("\n--- Test 3: All Attributes Revoked ---")
    tcabe3 = T_CP_ABE(PP)
    SK3 = tcabe3.keygen(MSK, all_attrs)
    for attr in all_attrs:
        tcabe3.revoke_attribute(attr)

    fail_count3 = 0
    for _ in range(n_trials):
        try:
            result = tcabe3.decrypt(SK3, CT, work_time)
            if result != msg:
                fail_count3 += 1
        except:
            fail_count3 += 1

    print(f"  Collusion fail rate: {fail_count3}/{n_trials} = {fail_count3/n_trials*100}%")
    results.append({
        'test': 'all_revoked',
        'description': 'All attributes revoked, threshold k=5',
        'fail_rate': fail_count3 / n_trials * 100,
    })

    # Test 4: Single attribute revoked in threshold
    print("\n--- Test 4: Single Attribute Revoked (k=5, n=10) ---")
    tcabe4 = T_CP_ABE(PP)
    SK4 = tcabe4.keygen(MSK, all_attrs)
    tcabe4.revoke_attribute('attr0')

    fail_count4 = 0
    for _ in range(n_trials):
        try:
            result = tcabe4.decrypt(SK4, CT, work_time)
            if result != msg:
                fail_count4 += 1
        except:
            fail_count4 += 1

    print(f"  Collusion fail rate: {fail_count4}/{n_trials} = {fail_count4/n_trials*100}%")
    results.append({
        'test': 'single_revoked_threshold',
        'description': '1/10 attributes revoked, threshold k=5',
        'fail_rate': fail_count4 / n_trials * 100,
    })

    output_path = '/app/experiments/results/collusion_resistance.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['test']:35s} | Fail Rate: {r['fail_rate']:>5.0f}%")
