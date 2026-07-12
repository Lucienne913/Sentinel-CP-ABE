#!/usr/bin/env python3
"""
P3: Attribute Revocation Experiment

Test version-based attribute revocation mechanism:
1. Normal decryption before revocation
2. Old key cannot decrypt after revocation
3. Normal decryption after regenerating key
4. Revocation performance measurement
5. Multi-user revocation scenario
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, PolicyParser
from charm.toolbox.pairinggroup import GT
from datetime import datetime


def test_revocation():
    # 1. Initialize system
    setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
    PP, MSK = setup.setup(max_attrs=50)
    tcabe = T_CP_ABE(PP)
    parser = PolicyParser()
    work_time = datetime(2026, 6, 3, 14, 30)

    print("=" * 60)
    print("P3: Attribute Revocation Experiment")
    print("=" * 60)

    # ========== Test 1: Basic revocation process ==========
    print("\n[Test1] Basic revocation process")

    # User A has role:engineer and dept:maintenance
    user_a_attrs = ['role:engineer', 'dept:maintenance']
    SK_a = tcabe.keygen_with_version(MSK, user_a_attrs, user_id='user_a')

    # Encrypt a message
    policy_str = "role:engineer AND dept:maintenance"
    policy_tree = parser.parse(policy_str)
    M = PP['group'].random(GT)
    CT = tcabe.encrypt(M, policy_tree)

    # Before revocation: User A can decrypt
    try:
        decrypted = tcabe.decrypt(SK_a, CT, work_time)
        assert decrypted == M
        print("  Decryption before revocation: SUCCESS")
    except ValueError as e:
        print("  Decryption before revocation: FAILED -", str(e))
        return False

    # Revoke role:engineer
    new_version = tcabe.revoke_attribute('role:engineer')
    print("  Revoked role:engineer, new version:", new_version)

    # After revocation: User A's old key cannot decrypt
    try:
        decrypted = tcabe.decrypt(SK_a, CT, work_time)
        print("  Decryption after revocation: SHOULD NOT SUCCEED")
        return False
    except ValueError as e:
        print("  Decryption after revocation: Correctly rejected -", str(e))

    # User A regenerates key
    SK_a_new = tcabe.keygen_with_version(MSK, user_a_attrs, user_id='user_a')
    try:
        decrypted = tcabe.decrypt(SK_a_new, CT, work_time)
        assert decrypted == M
        print("  Decryption after regenerating key: SUCCESS")
    except ValueError as e:
        print("  Decryption after regenerating key: FAILED -", str(e))
        return False

    # ========== Test 2: Multiple attribute revocation ==========
    print("\n[Test2] Multiple attribute revocation")

    tcabe2 = T_CP_ABE(PP)
    user_b_attrs = ['role:admin', 'dept:security', 'level:high']
    SK_b = tcabe2.keygen_with_version(MSK, user_b_attrs, user_id='user_b')

    policy_str2 = "role:admin AND dept:security"
    policy_tree2 = parser.parse(policy_str2)
    CT2 = tcabe2.encrypt(M, policy_tree2)

    # Before revocation
    try:
        decrypted = tcabe2.decrypt(SK_b, CT2, work_time)
        assert decrypted == M
        print("  Decryption before revocation: SUCCESS")
    except ValueError as e:
        print("  Decryption before revocation: FAILED -", str(e))
        return False

    # Revoke role:admin
    tcabe2.revoke_attribute('role:admin')
    try:
        decrypted = tcabe2.decrypt(SK_b, CT2, work_time)
        print("  Decryption after revoking role:admin: SHOULD NOT SUCCEED")
        return False
    except ValueError as e:
        print("  Decryption after revoking role:admin: Correctly rejected")

    # Regenerate key
    SK_b_new = tcabe2.keygen_with_version(MSK, user_b_attrs, user_id='user_b')
    try:
        decrypted = tcabe2.decrypt(SK_b_new, CT2, work_time)
        assert decrypted == M
        print("  Decryption after regenerating key: SUCCESS")
    except ValueError as e:
        print("  Decryption after regenerating key: FAILED -", str(e))
        return False

    # ========== Test 3: Revocation does not affect other users ==========
    print("\n[Test3] Revocation does not affect other users")

    tcabe3 = T_CP_ABE(PP)
    user_c_attrs = ['role:engineer', 'dept:research']
    user_d_attrs = ['role:engineer', 'dept:maintenance']

    SK_c = tcabe3.keygen_with_version(MSK, user_c_attrs, user_id='user_c')
    SK_d = tcabe3.keygen_with_version(MSK, user_d_attrs, user_id='user_d')

    policy_str3 = "role:engineer AND dept:research"
    policy_tree3 = parser.parse(policy_str3)
    CT3 = tcabe3.encrypt(M, policy_tree3)

    # Revoke dept:maintenance (affects user_d, not user_c)
    tcabe3.revoke_attribute('dept:maintenance')

    # user_c can still decrypt (not revoked)
    try:
        decrypted = tcabe3.decrypt(SK_c, CT3, work_time)
        assert decrypted == M
        print("  user_c (not revoked) decryption: SUCCESS")
    except ValueError as e:
        print("  user_c decryption: FAILED -", str(e))
        return False

    # ========== Test 4: Revocation performance measurement ==========
    print("\n[Test4] Revocation performance measurement")

    tcabe4 = T_CP_ABE(PP)
    revoke_attrs = ['attr_%d' % i for i in range(50)]

    # Generate multiple user keys
    num_users = 100
    users = []
    for i in range(num_users):
        user_attrs = ['attr_%d' % (i % 50), 'attr_%d' % ((i + 1) % 50)]
        SK = tcabe4.keygen_with_version(MSK, user_attrs, user_id='user_%d' % i)
        users.append(SK)

    # Measure revocation time
    revoke_times = []
    for attr in revoke_attrs[:10]:  # Revoke 10 attributes
        start = time.time()
        tcabe4.revoke_attribute(attr)
        revoke_times.append(time.time() - start)

    avg_revoke_time = sum(revoke_times) / len(revoke_times)
    print("  Revoked 10 attributes, average time: %.4f ms" % (avg_revoke_time * 1000))
    print("  Total users:", num_users)

    # Measure version check time
    check_times = []
    for SK in users[:20]:
        start = time.time()
        tcabe4._check_attr_versions(SK)
        check_times.append(time.time() - start)

    avg_check_time = sum(check_times) / len(check_times)
    print("  Version check average time: %.4f ms" % (avg_check_time * 1000))

    # ========== Test 5: Continuous revocation ==========
    print("\n[Test5] Continuous revocation")

    tcabe5 = T_CP_ABE(PP)
    user_e_attrs = ['role:engineer']
    SK_e = tcabe5.keygen_with_version(MSK, user_e_attrs, user_id='user_e')

    # Revoke 3 times continuously
    for i in range(3):
        tcabe5.revoke_attribute('role:engineer')
        SK_e_new = tcabe5.keygen_with_version(MSK, user_e_attrs, user_id='user_e')

        # New key should be able to decrypt
        policy_str5 = "role:engineer"
        policy_tree5 = parser.parse(policy_str5)
        CT5 = tcabe5.encrypt(M, policy_tree5)

        try:
            decrypted = tcabe5.decrypt(SK_e_new, CT5, work_time)
            assert decrypted == M
            print("  %d-th revocation, regenerate key: SUCCESS" % (i + 1))
        except ValueError as e:
            print("  %d-th revocation decryption: FAILED - %s" % (i + 1, str(e)))
            return False

    # ========== Test 6: Revocation combined with THRESHOLD ==========
    print("\n[Test6] Revocation combined with THRESHOLD")

    tcabe6 = T_CP_ABE(PP)
    user_f_attrs = ['role:engineer', 'role:admin', 'role:manager']
    SK_f = tcabe6.keygen_with_version(MSK, user_f_attrs, user_id='user_f')

    policy_str6 = "THRESHOLD(2, role:engineer, role:admin, role:manager)"
    policy_tree6 = parser.parse(policy_str6)
    CT6 = tcabe6.encrypt(M, policy_tree6)

    # Before revocation
    try:
        decrypted = tcabe6.decrypt(SK_f, CT6, work_time)
        assert decrypted == M
        print("  THRESHOLD decryption before revocation: SUCCESS")
    except ValueError as e:
        print("  THRESHOLD decryption before revocation: FAILED -", str(e))
        return False

    # Revoke role:engineer
    tcabe6.revoke_attribute('role:engineer')

    # Old key should fail
    try:
        decrypted = tcabe6.decrypt(SK_f, CT6, work_time)
        print("  Old key decryption after revocation: SHOULD NOT SUCCEED")
        return False
    except ValueError as e:
        print("  Old key decryption after revocation: Correctly rejected")

    # New key should succeed
    SK_f_new = tcabe6.keygen_with_version(MSK, user_f_attrs, user_id='user_f')
    try:
        decrypted = tcabe6.decrypt(SK_f_new, CT6, work_time)
        assert decrypted == M
        print("  New key THRESHOLD decryption: SUCCESS")
    except ValueError as e:
        print("  New key THRESHOLD decryption: FAILED -", str(e))
        return False

    print("\n" + "=" * 60)
    print("P3: Attribute Revocation Experiment PASSED")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_revocation()
    sys.exit(0 if success else 1)
