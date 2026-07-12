#!/usr/bin/env python3
"""
Distributed TA (DTA) Performance & Correctness Test Suite

Evaluates the (t,n)-threshold Shamir secret sharing implementation against
the three criteria required by reviewer comments Section 3.2:
  1. Correctness: distributed keygen → encrypt → decrypt matches
  2. Security: <t compromised nodes cannot reconstruct MK
  3. Performance: split/reconstruct latency across configurations

Usage:
    python experiments/test_distributed_ta.py          # run all tests
    python experiments/test_distributed_ta.py --quick   # skip slow benchmarks

Output:
    - Console: per-test pass/fail and timing
    - JSON:    tests/results/dta_benchmark.json (if --save)
"""

import sys
import os
import time
import json
import argparse

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, GT
from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, AccessPolicyTree as APT
from distributed_ta import DistributedSecretSharing, DistributedTA


# ============================================================== #
#  Test Helpers                                                    #
# ============================================================== #

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def heading(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def subheading(title):
    print(f"\n  --- {title} ---")


def check(condition, message):
    status = PASS if condition else FAIL
    print(f"  {status}: {message}")
    return condition


# ============================================================== #
#  Test Cases                                                     #
# ============================================================== #

def test_sss_correctness(group):
    """Test 1: Basic SSS (t,n)-split/reconstruct correctness."""
    subheading("SSS Split/Reconstruct")
    sss = DistributedSecretSharing(group)
    all_ok = True

    for n, t in [(3, 2), (5, 3), (7, 4), (10, 5), (10, 10)]:
        secret = group.random(ZR)
        shares = sss.split_secret(secret, n, t)
        recovered = sss.reconstruct(shares[:t], t)
        ok = check(secret == recovered, f"{t}-of-{n}: exact recovery")
        all_ok &= ok

        # <t shares must fail
        if t > 2:
            bad = sss.reconstruct(shares[:t - 1], t)
            ok2 = check(secret != bad, f"{t}-of-{n}: {t-1} shares fail (info-theoretic)")
            all_ok &= ok2

    return all_ok


def test_sss_verification(group):
    """Test 2: Feldman commitment verification."""
    subheading("Feldman Share Verification")
    sss = DistributedSecretSharing(group)
    all_ok = True

    n, t = 5, 3
    test_secret = group.random(ZR)
    coeffs = [test_secret, group.random(ZR), group.random(ZR)]
    g = group.random(G1)
    commitments = sss.compute_commitment(g, coeffs)

    # Generate shares from the same coeffs
    for i in range(1, n + 1):
        xi = group.init(ZR, i)
        val = coeffs[-1]
        for j in range(len(coeffs) - 2, -1, -1):
            val = val * xi + coeffs[j]
        valid = sss.verify_share(g, i, val, commitments)
        ok = check(valid, f"Share {i} verified against commitments")
        all_ok &= ok

    # Tampered share must fail
    tampered_val = group.random(ZR)
    valid_tampered = sss.verify_share(g, 1, tampered_val, commitments)
    ok2 = check(not valid_tampered, "Tampered share rejected")
    all_ok &= ok2

    return all_ok


def test_distributed_setup(group):
    """Test 3: DistributedTA setup and node initialization."""
    subheading("Distributed Setup")
    all_ok = True

    base = T_CP_ABE_Setup(group_name='SS512', security_level=80)
    dta = DistributedTA(group, n_tas=5, threshold=3)
    PP = dta.distributed_setup(base)

    ok = check(len(dta.node_shares) == 5, "5 TA nodes initialized")
    all_ok &= ok
    ok = check(dta.commitments is not None, "Feldman commitments computed")
    all_ok &= ok

    for node_id in range(5):
        ok = check(dta.verify_share_integrity(node_id), f"Node {node_id} share integrity")
        all_ok &= ok

    ok = check('g' in PP and 'h' in PP and 'e_gg_alpha' in PP, "PP fields present")
    all_ok &= ok

    return all_ok, PP, dta


def test_threshold_keygen(group, PP, dta):
    """Test 4: Threshold key generation and encrypt/decrypt."""
    subheading("Threshold KeyGen & End-to-End Encrypt/Decrypt")
    all_ok = True

    abe = T_CP_ABE(PP)
    user_attrs = ['role:engineer', 'dept:maintenance', 'location:factory']
    
    # 2-of-3 threshold policy
    policy_tree = APT('THRESHOLD', threshold=2, children=[
        APT('LEAF', value='role:engineer'),
        APT('LEAF', value='dept:maintenance'),
        APT('LEAF', value='location:factory')
    ])

    # Generate key using threshold (3 of 5 nodes)
    SK = dta.threshold_keygen(abe, user_attrs)
    ok = check('K0' in SK, "SK contains K0")
    all_ok &= ok
    ok = check(len(SK['K']) == 3, f"SK has {len(SK['K'])} attribute keys")
    all_ok &= ok

    # Encrypt
    M = group.random(GT)
    CT = abe.encrypt(M, policy_tree)
    ok = check('C0' in CT and 'C1' in CT, "CT contains both ciphertext components")
    all_ok &= ok

    # Decrypt
    M_dec = abe.decrypt(SK, CT)
    ok = check(M == M_dec, "Decrypted message matches original")
    all_ok &= ok

    # Test different node subsets
    for nodes in [[0, 1, 2], [2, 3, 4], [0, 3, 4]]:
        SK2 = dta.threshold_keygen(abe, user_attrs, available_nodes=nodes)
        M_dec2 = abe.decrypt(SK2, CT)
        ok = check(M == M_dec2, f"Decrypt with nodes {nodes}: MATCH")
        all_ok &= ok

    return all_ok


def test_node_compromise(group, dta):
    """Test 5: <t nodes cannot reconstruct MK."""
    subheading("Compromise Resistance")
    all_ok = True

    for k in range(1, dta.threshold):
        secure = dta.simulate_node_compromise(list(range(k)))
        ok = check(secure, f"{k} compromised node(s) ({dta.threshold} required)")
        all_ok &= ok

    return all_ok


def test_different_thresholds(group):
    """Test 6: Different (t,n) configurations."""
    subheading("Multiple (t,n) Configurations")
    all_ok = True

    configs = [(3, 5), (4, 7), (5, 10), (7, 10)]
    for t, n in configs:
        base = T_CP_ABE_Setup(group_name='SS512', security_level=80)
        dta = DistributedTA(group, n_tas=n, threshold=t)
        PP = dta.distributed_setup(base)
        abe = T_CP_ABE(PP)
        user_attrs = [f'attr:{i}' for i in range(5)]
        policy_tree = APT('THRESHOLD', threshold=2, children=[
            APT('LEAF', value='attr:0'),
            APT('LEAF', value='attr:1')
        ])

        SK = dta.threshold_keygen(abe, user_attrs)
        M = group.random(GT)
        CT = abe.encrypt(M, policy_tree)
        M_dec = abe.decrypt(SK, CT)
        ok = check(M == M_dec, f"{t}-of-{n}: encrypt/decrypt OK")
        all_ok &= ok

    return all_ok


def test_share_renewal(group):
    """Test 7: Proactive share renewal (zero-polynomial)."""
    subheading("Share Renewal")
    sss = DistributedSecretSharing(group)
    all_ok = True

    n, t = 5, 3
    secret = group.random(ZR)
    shares = sss.split_secret(secret, n, t)
    deltas = sss.renew_shares(n, t)

    # Add delta to each share
    renewed_shares = []
    for i in range(n):
        idx = shares[i][0]
        new_val = shares[i][1] + deltas[i][1]
        renewed_shares.append((idx, new_val))

    # Reconstruct from renewed shares (should get original secret)
    recovered = sss.reconstruct(renewed_shares[:t], t)
    ok = check(secret == recovered, "Renewed shares reconstruct to same secret")
    all_ok &= ok

    # Old (unrenewed) shares should still work too (renewal = proactive, not revocation)
    # But verify they're different from renewed shares
    ok = check(shares[0][1] != renewed_shares[0][1], "Individual share value changed after renewal")
    all_ok &= ok

    return all_ok


def test_performance_benchmark(group, dta):
    """Test 8: Performance benchmarks."""
    subheading("Performance Benchmarks")
    sss = DistributedSecretSharing(group)

    print(f"  Setup latency: {dta.get_setup_latency():.2f} ms")

    for n, t in [(3, 2), (5, 3), (7, 4), (10, 5), (20, 10)]:
        test_secret = group.random(ZR)
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            shr = sss.split_secret(test_secret, n, t)
            _ = sss.reconstruct(shr[:t], t)
            latencies.append((time.perf_counter() - t0) * 1000)
        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        print(f"  {t}-of-{n}: avg={avg:.3f} ms, p95={p95:.3f} ms")

    return True


# ============================================================== #
#  Main                                                           #
# ============================================================== #

def main():
    parser = argparse.ArgumentParser(description='Distributed TA Test Suite')
    parser.add_argument('--quick', action='store_true', help='Skip performance benchmarks')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    args = parser.parse_args()

    print("=" * 60)
    print("  Distributed TA (DTA) — Phase 2 Test Suite")
    print(f"  Curve: SS512 | {6 * '='}>  Reviewer Section 3.2")
    print("=" * 60)

    group = PairingGroup('SS512')
    results = {}
    all_tests_pass = True

    # Test 1: SSS Correctness
    heading("Test 1: SSS Correctness")
    t1_ok = test_sss_correctness(group)
    results['sss_correctness'] = t1_ok
    all_tests_pass &= t1_ok

    # Test 2: Verification
    heading("Test 2: Share Verification (Feldman)")
    t2_ok = test_sss_verification(group)
    results['share_verification'] = t2_ok
    all_tests_pass &= t2_ok

    # Test 3: Distributed Setup
    heading("Test 3: Distributed Setup")
    t3_ok, PP, dta = test_distributed_setup(group)
    results['distributed_setup'] = t3_ok
    all_tests_pass &= t3_ok

    # Test 4: End-to-End
    heading("Test 4: Threshold KeyGen & Encrypt/Decrypt")
    t4_ok = test_threshold_keygen(group, PP, dta)
    results['end_to_end'] = t4_ok
    all_tests_pass &= t4_ok

    # Test 5: Compromise Resistance
    heading("Test 5: Compromise Resistance")
    t5_ok = test_node_compromise(group, dta)
    results['compromise_resistance'] = t5_ok
    all_tests_pass &= t5_ok

    # Test 6: Different thresholds
    heading("Test 6: Configurations")
    t6_ok = test_different_thresholds(group)
    results['configurations'] = t6_ok
    all_tests_pass &= t6_ok

    # Test 7: Share Renewal
    heading("Test 7: Proactive Share Renewal")
    t7_ok = test_share_renewal(group)
    results['share_renewal'] = t7_ok
    all_tests_pass &= t7_ok

    # Test 8: Performance
    if not args.quick:
        heading("Test 8: Performance")
        test_performance_benchmark(group, dta)
        results['benchmark'] = True

    # Summary
    heading("Summary")
    passed = sum(1 for v in results.values() if v is True)
    total = sum(1 for v in results.values() if isinstance(v, bool))
    print(f"  Passed: {passed}/{total}")
    verdict = "ALL TESTS PASSED" if all_tests_pass else "SOME TESTS FAILED"
    print(f"  Verdict: {verdict}")
    print(f"  {'=' * 60}")

    # Save results
    if args.save:
        os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
        path = os.path.join(os.path.dirname(__file__), 'results', 'dta_benchmark.json')
        with open(path, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'curve': 'SS512',
                'results': results,
                'all_pass': all_tests_pass,
            }, f, indent=2)
        print(f"  Results saved to: {path}")

    return 0 if all_tests_pass else 1


if __name__ == "__main__":
    sys.exit(main())
