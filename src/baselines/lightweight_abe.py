"""BSW07-AND: AND-gate only policy variant of BSW07 for IoT baseline comparison.

This implements a policy-restricted variant of BSW07 that uses only AND-gate
expressions (no OR or THRESHOLD gates), approximating the performance
characteristics of lightweight CP-ABE schemes while maintaining the same
cryptographic foundation.

This is NOT a real implementation of lightweight ABE schemes like:
- Yao et al. "A lightweight attribute-based encryption scheme for the IoT" (2019)
- Odelu et al. "CP-ABE with constant-size keys for lightweight devices" (2016)

The actual lightweight schemes have different cryptographic constructions
(e.g., constant-size ciphertext, different key structures) that cannot be
fully replicated using BSW07. This implementation only captures the policy
expressiveness limitation (AND-gate only vs full LSSS).

In the paper, this is labeled as:
"BSW07-AND (AND-gate only policy variant)"

References for real lightweight ABE schemes:
- Yao et al. "A lightweight attribute-based encryption scheme for the IoT" (2019)
- Odelu et al. "CP-ABE with constant-size keys for lightweight devices" (2016)
"""

from charm.toolbox.pairinggroup import PairingGroup
from charm.schemes.abenc.abenc_bsw07 import CPabe_BSW07


class BSW07_AND:
    """BSW07-AND: AND-gate only policy variant of BSW07.

    This class implements a policy-restricted variant of BSW07 that uses only
    AND-gate expressions (no OR or THRESHOLD gates), approximating the performance
    characteristics of lightweight CP-ABE schemes.

    Limitations compared to real lightweight schemes:
    1. Does not implement constant-size ciphertext (lightweight schemes' key feature)
    2. Does not use the same key generation structure as real lightweight schemes
    3. Only approximates the policy expressiveness limitation

    For accurate comparison with real lightweight schemes, researchers should:
    1. Implement the actual scheme from the referenced paper
    2. Or clearly label this as "AND-gate only BSW07 variant" in publications
    """

    def __init__(self, group_name='SS1024'):
        self.group = PairingGroup(group_name)
        self.cpabe = CPabe_BSW07(self.group)
        self.pk = None
        self.msk = None

    def setup(self):
        """Setup phase - uses BSW07's setup"""
        (self.pk, self.msk) = self.cpabe.setup()
        return self.pk, self.msk

    def keygen(self, attributes):
        """Key generation - uses BSW07's keygen with attribute list"""
        if self.msk is None:
            self.setup()
        return self.cpabe.keygen(self.pk, self.msk, attributes)

    def encrypt(self, message, attributes):
        """Encryption with AND-gate only policy

        This constructs an AND-gate policy from the attribute list,
        which is the key restriction that differentiates BSW07-AND
        from the full BSW07 scheme.
        """
        if self.pk is None:
            self.setup()
        # Construct AND-gate only policy (this is the key restriction)
        policy_str = '(' + ' and '.join(attributes) + ')'
        return self.cpabe.encrypt(self.pk, message, policy_str)

    def decrypt(self, ct, sk):
        """Decryption - uses BSW07's decrypt"""
        return self.cpabe.decrypt(self.pk, sk, ct)

    def get_scheme_info(self):
        """Return information about this scheme for documentation purposes"""
        return {
            'scheme_name': 'BSW07-AND',
            'base_scheme': 'BSW07',
            'policy_restriction': 'AND-gate only',
            'limitations': [
                'No constant-size ciphertext',
                'Different key structure from real lightweight schemes',
                'Only approximates the policy expressiveness limitation'
            ]
        }
