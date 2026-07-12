#!/usr/bin/env python3
"""E4: Storage overhead comparison - T-CP-ABE vs BSW07."""
import sys, os, json, statistics, pickle
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'baselines'))
from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, PolicyParser
from charm.toolbox.pairinggroup import GT, ZR, G1, G2, PairingGroup
from charm.schemes.abenc.abenc_bsw07 import CPabe_BSW07

n_trials = 3

def measure_element_size(elem, group):
    return len(group.serialize(elem))

def measure_bsw07_sk_size(sk, group):
    total = 0
    # D: single element
    total += measure_element_size(sk['D'], group)
    # Dj, Djp: dict of elements
    for attr in sk['Dj']:
        total += measure_element_size(sk['Dj'][attr], group)
    for attr in sk['Djp']:
        total += measure_element_size(sk['Djp'][attr], group)
    return total

def measure_bsw07_ct_size(ct, group):
    total = 0
    total += measure_element_size(ct['C_tilde'], group)
    total += measure_element_size(ct['C'], group)
    for attr in ct['Cy']:
        total += measure_element_size(ct['Cy'][attr], group)
    for attr in ct['Cyp']:
        total += measure_element_size(ct['Cyp'][attr], group)
    return total

def make_and_policy(n):
    attrs = [f'ATTR{i}' for i in range(n)]
    policy = "(" + " and ".join(attrs) + ")"
    return policy, attrs

if __name__ == '__main__':
    print("=" * 60)
    print("E4: Storage Overhead - T-CP-ABE vs BSW07")
    print("=" * 60)

    attr_counts = [10, 20, 50, 100]
    results = []

    for n in attr_counts:
        print(f"\n--- Attributes: {n} ---")
        policy_str, attrs = make_and_policy(n)

        # --- BSW07 ---
        group = PairingGroup('SS512')
        cpabe = CPabe_BSW07(group)
        (pk, msk) = cpabe.setup()
        sk_bsw = cpabe.keygen(pk, msk, attrs)
        msg = group.random(GT)
        ct_bsw = cpabe.encrypt(pk, msg, policy_str)

        sk_bsw_sizes = []
        ct_bsw_sizes = []
        for _ in range(n_trials):
            sk_bsw_sizes.append(measure_bsw07_sk_size(sk_bsw, group))
            ct_bsw_sizes.append(measure_bsw07_ct_size(ct_bsw, group))

        # --- T-CP-ABE ---
        setup_obj = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
        PP, MSK = setup_obj.setup(max_attrs=n + 10)
        tcabe = T_CP_ABE(PP)
        parser = PolicyParser()
        msg2 = PP['group'].random(GT)
        policy_tree = parser.parse(f"AND(" + ", ".join(attrs) + ")")
        SK = tcabe.keygen(MSK, attrs[:n])
        CT = tcabe.encrypt(msg2, policy_tree)

        sk_tcp_sizes = []
        ct_tcp_sizes = []
        for _ in range(n_trials):
            sk_tcp_sizes.append(measure_element_size(SK['K0'], PP['group']) +
                               sum(measure_element_size(v['K_attr'], PP['group']) +
                                   measure_element_size(v['K_prime_attr'], PP['group'])
                                   for v in SK['K'].values()))
            ct_tcp_sizes.append(measure_element_size(CT['C0'], PP['group']) +
                               measure_element_size(CT['C1'], PP['group']))

        result = {
            'n_attrs': n,
            'bsw_sk_kb': round(statistics.mean(sk_bsw_sizes) / 1024, 2),
            'bsw_ct_kb': round(statistics.mean(ct_bsw_sizes) / 1024, 2),
            'tcp_sk_kb': round(statistics.mean(sk_tcp_sizes) / 1024, 2),
            'tcp_ct_kb': round(statistics.mean(ct_tcp_sizes) / 1024, 2),
            'sk_ratio': round(statistics.mean(sk_tcp_sizes) / statistics.mean(sk_bsw_sizes), 2),
            'ct_ratio': round(statistics.mean(ct_tcp_sizes) / statistics.mean(ct_bsw_sizes), 2),
        }
        results.append(result)
        print(f"  BSW07: SK={result['bsw_sk_kb']}KB, CT={result['bsw_ct_kb']}KB")
        print(f"  T-CP:  SK={result['tcp_sk_kb']}KB, CT={result['tcp_ct_kb']}KB")
        print(f"  Ratio: SK={result['sk_ratio']}x, CT={result['ct_ratio']}x")

    output_path = '/app/experiments/results/storage_comparison.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Attrs':>6} | {'BSW SK':>8} | {'TCP SK':>8} | {'SK Ratio':>9} | {'BSW CT':>8} | {'TCP CT':>8} | {'CT Ratio':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_attrs']:>6} | {r['bsw_sk_kb']:>7.2f}K | {r['tcp_sk_kb']:>7.2f}K | {r['sk_ratio']:>8.2f}x | {r['bsw_ct_kb']:>7.2f}K | {r['tcp_ct_kb']:>7.2f}K | {r['ct_ratio']:>8.2f}x")
