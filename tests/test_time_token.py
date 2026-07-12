#!/usr/bin/env python3
"""P1: Time Predicate Hash-Chain Cryptographic Binding Tests"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from setup import T_CP_ABE_Setup
from t_cp_abe import T_CP_ABE, TimeTokenAuthority, PolicyParser
from charm.toolbox.pairinggroup import GT
from datetime import datetime

def test_time_token():
    # 1. Initialize system
    setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
    PP, MSK = setup.setup(max_attrs=50)

    # 2. Create TTA
    ttt = TimeTokenAuthority(chain_length=100)
    print("TTA chain_tip:", ttt.chain_tip[:16].hex(), "...")
    print("TTA chain_length:", ttt.chain_length)

    # 3. Create T-CP-ABE instance (with TTA)
    tcabe = T_CP_ABE(PP, time_token_authority=ttt)

    # 4. Key Generation (with Time Token)
    attrs = ['role:engineer', 'dept:maintenance']
    SK = tcabe.keygen_with_time_token(MSK, attrs)
    print("Time token index:", SK["time_token_index"])
    print("Time token:", SK["time_token"][:16].hex(), "...")

    # 5. Verify token validity
    token = SK['time_token']
    index = SK['time_token_index']
    is_valid = ttt.verify_token(index, token)
    print("Token valid:", is_valid)
    assert is_valid, "Token should be valid"

    # 6. Encryption (with Time Token binding)
    time_predicates = {'work': {'hour': (8, 18), 'weekday': [1,2,3,4,5]}}
    parser = PolicyParser(time_predicates=time_predicates)
    policy_tree = parser.parse('role:engineer AND dept:maintenance AND time:work')

    M = PP['group'].random(GT)
    CT = tcabe.encrypt_with_time_token(M, policy_tree)
    print("Time token binding:", CT["time_token_binding"][:32], "...")
    print("Time token index in CT:", CT["time_token_index"])

    # 7. Decryption (with Time Token verification)
    work_time = datetime(2026, 6, 3, 14, 30)  # Wednesday 14:30
    try:
        decrypted = tcabe.decrypt(SK, CT, work_time)
        if decrypted == M:
            print("Decryption with time token: SUCCESS")
        else:
            print("Decryption with time token: FAILED (message mismatch)")
            return False
    except ValueError as e:
        print("Decryption with time token: FAILED (%s)" % str(e))
        return False

    # 8. Test invalid token (simulate adversary forging token)
    fake_token = b'fake_token_123456789012345678901234'
    fake_SK = dict(SK)
    fake_SK['time_token'] = fake_token
    try:
        decrypted = tcabe.decrypt(fake_SK, CT, work_time)
        print("Fake token decryption: SHOULD NOT SUCCEED")
        return False
    except ValueError as e:
        print("Fake token correctly rejected:", str(e))

    # 9. Test TTA token advancement
    new_index, new_token = ttt.advance()
    print("Advanced to index:", new_index)
    is_new_valid = ttt.verify_token(new_index, new_token)
    print("New token valid:", is_new_valid)
    assert is_new_valid, "New token should be valid"

    # 10. Test old token still valid (forward security verification)
    is_old_valid = ttt.verify_token(index, token)
    print("Old token still valid:", is_old_valid)
    assert is_old_valid, "Old token should still be valid (forward verification)"

    print()
    print("P1 Time Token Test PASSED")
    return True

if __name__ == '__main__':
    success = test_time_token()
    sys.exit(0 if success else 1)
