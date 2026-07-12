"""BSW07 CP-ABE baseline wrapper for fair comparison."""

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
from charm.schemes.abenc.abenc_bsw07 import CPabe_BSW07


class BSW07Baseline:
    """BSW07 CP-ABE (Bethencourt-Sahai-Waters 2007) baseline implementation.

    Uses Charm-Crypto's built-in implementation for fair comparison.
    """

    def __init__(self, group_name='SS1024'):
        self.group = PairingGroup(group_name)
        self.cpabe = CPabe_BSW07(self.group)
        self.msk = None
        self.pk = None

    def setup(self):
        (self.pk, self.msk) = self.cpabe.setup()
        return self.pk, self.msk

    def keygen(self, attributes):
        if self.msk is None:
            self.setup()
        return self.cpabe.keygen(self.pk, self.msk, attributes)

    def encrypt(self, message, policy_str):
        if self.pk is None:
            self.setup()
        return self.cpabe.encrypt(self.pk, message, policy_str)

    def decrypt(self, ct, sk):
        return self.cpabe.decrypt(self.pk, sk, ct)
