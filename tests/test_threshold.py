#!/usr/bin/env python3
"""P2: THRESHOLD Gate Support Tests"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, PolicyParser
from charm.toolbox.pairinggroup import GT
from datetime import datetime

def test_threshold():
    # 1. Initialize system
    setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
    PP, MSK = setup.setup(max_attrs=50)
    tcabe = T_CP_ABE(PP)
    parser = PolicyParser()

    print("=" * 60)
    print("P2: THRESHOLD Gate Support Tests")
    print("=" * 60)

    # 2. Test THRESHOLD(2, A, B, C) - 2-of-3
    print("\n[Test1] THRESHOLD(2, role:engineer, role:admin, role:manager)")
    policy_str = "THRESHOLD(2, role:engineer, role:admin, role:manager)"
    policy_tree = parser.parse(policy_str)
    print("  Policy Type:", policy_tree.node_type)
    print("  Threshold Value:", policy_tree.threshold)
    print("  Child Node Count:", len(policy_tree.children))
    assert policy_tree.node_type == 'THRESHOLD'
    assert policy_tree.threshold == 2
    assert len(policy_tree.children) == 3

    # 3. Test Serialize/Deserialize
    print("\n[Test2] Serialize/Deserialize")
    policy_str_back = tcabe._policy_to_str(policy_tree)
    print("  Serialize Result:", policy_str_back)
    policy_tree2 = parser.parse(policy_str_back)
    assert policy_tree2.node_type == 'THRESHOLD'
    assert policy_tree2.threshold == 2
    assert len(policy_tree2.children) == 3
    print("  Serialize/Deserialize: OK")

    # 4. Test policy satisfaction (2-of-3, providing 2 attributes)
    print("\n[Test3] Policy Satisfaction Test (2 attributes, satisfies THRESHOLD(2,...))")
    M = PP['group'].random(GT)
    CT = tcabe.encrypt(M, policy_tree)
    attrs_2 = ['role:engineer', 'role:admin']
    SK_2 = tcabe.keygen(MSK, attrs_2)
    work_time = datetime(2026, 6, 3, 14, 30)
    try:
        decrypted = tcabe.decrypt(SK_2, CT, work_time)
        if decrypted == M:
            print("  2-of-3 (2 attrs): SUCCESS")
        else:
            print("  2-of-3 (2 attrs): FAILED (message mismatch)")
            return False
    except ValueError as e:
        print("  2-of-3 (2 attrs): FAILED (%s)" % str(e))
        return False

    # 5. Test policy satisfaction (2-of-3, providing 3 attributes)
    print("\n[Test4] Policy Satisfaction Test (3 attributes, satisfies THRESHOLD(2,...))")
    attrs_3 = ['role:engineer', 'role:admin', 'role:manager']
    SK_3 = tcabe.keygen(MSK, attrs_3)
    try:
        decrypted = tcabe.decrypt(SK_3, CT, work_time)
        if decrypted == M:
            print("  2-of-3 (3 attrs): SUCCESS")
        else:
            print("  2-of-3 (3 attrs): FAILED (message mismatch)")
            return False
    except ValueError as e:
        print("  2-of-3 (3 attrs): FAILED (%s)" % str(e))
        return False

    # 6. Test policy not satisfied (2-of-3, providing only 1 attribute)
    print("\n[Test5] Policy Not Satisfied Test (1 attribute, does not satisfy THRESHOLD(2,...))")
    attrs_1 = ['role:engineer']
    SK_1 = tcabe.keygen(MSK, attrs_1)
    try:
        decrypted = tcabe.decrypt(SK_1, CT, work_time)
        print("  2-of-3 (1 attr): SHOULD NOT SUCCEED")
        return False
    except ValueError as e:
        print("  2-of-3 (1 attr): Correctly rejected -", str(e))

    # 7. Test THRESHOLD(1, A, B, C) - equivalent to OR
    print("\n[Test6] THRESHOLD(1, ...) equivalent to OR")
    policy_or = "THRESHOLD(1, role:engineer, role:admin, role:manager)"
    policy_tree_or = parser.parse(policy_or)
    CT_or = tcabe.encrypt(M, policy_tree_or)
    try:
        decrypted = tcabe.decrypt(SK_1, CT_or, work_time)
        if decrypted == M:
            print("  THRESHOLD(1,...) with 1 attr: SUCCESS")
        else:
            print("  THRESHOLD(1,...) with 1 attr: FAILED")
            return False
    except ValueError as e:
        print("  THRESHOLD(1,...) with 1 attr: FAILED (%s)" % str(e))
        return False

    # 8. Test THRESHOLD(3, A, B, C) - equivalent to AND
    print("\n[Test7] THRESHOLD(3, ...) equivalent to AND")
    policy_and = "THRESHOLD(3, role:engineer, role:admin, role:manager)"
    policy_tree_and = parser.parse(policy_and)
    CT_and = tcabe.encrypt(M, policy_tree_and)
    try:
        decrypted = tcabe.decrypt(SK_3, CT_and, work_time)
        if decrypted == M:
            print("  THRESHOLD(3,...) with 3 attrs: SUCCESS")
        else:
            print("  THRESHOLD(3,...) with 3 attrs: FAILED")
            return False
    except ValueError as e:
        print("  THRESHOLD(3,...) with 3 attrs: FAILED (%s)" % str(e))
        return False

    # 9. Test THRESHOLD combined with AND/OR
    print("\n[Test8] THRESHOLD combined with AND")
    policy_combo = "THRESHOLD(2, role:engineer, role:admin, role:manager) AND dept:maintenance"
    policy_tree_combo = parser.parse(policy_combo)
    print("  Policy Type:", policy_tree_combo.node_type)
    CT_combo = tcabe.encrypt(M, policy_tree_combo)
    attrs_combo = ['role:engineer', 'role:admin', 'dept:maintenance']
    SK_combo = tcabe.keygen(MSK, attrs_combo)
    try:
        decrypted = tcabe.decrypt(SK_combo, CT_combo, work_time)
        if decrypted == M:
            print("  THRESHOLD AND combo: SUCCESS")
        else:
            print("  THRESHOLD AND combo: FAILED")
            return False
    except ValueError as e:
        print("  THRESHOLD AND combo: FAILED (%s)" % str(e))
        return False

    # 10. Test Time Predicate combined with THRESHOLD
    print("\n[Test9] THRESHOLD combined with Time Predicate")
    time_predicates = {'work': {'hour': (8, 18), 'weekday': [1,2,3,4,5]}}
    parser_with_time = PolicyParser(time_predicates=time_predicates)
    policy_time = "THRESHOLD(2, role:engineer, role:admin, role:manager) AND time:work"
    policy_tree_time = parser_with_time.parse(policy_time)
    CT_time = tcabe.encrypt(M, policy_tree_time)
    try:
        decrypted = tcabe.decrypt(SK_3, CT_time, work_time)
        if decrypted == M:
            print("  THRESHOLD + time:work: SUCCESS")
        else:
            print("  THRESHOLD + time:work: FAILED")
            return False
    except ValueError as e:
        print("  THRESHOLD + time:work: FAILED (%s)" % str(e))
        return False

    # Night time test (should fail)
    night_time = datetime(2026, 6, 3, 22, 0)
    try:
        decrypted = tcabe.decrypt(SK_3, CT_time, night_time)
        print("  THRESHOLD + night: SHOULD NOT SUCCEED")
        return False
    except ValueError as e:
        print("  THRESHOLD + night: Correctly rejected")

    print("\n" + "=" * 60)
    print("P2: THRESHOLD Gate Support Tests PASSED")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = test_threshold()
    sys.exit(0 if success else 1)
