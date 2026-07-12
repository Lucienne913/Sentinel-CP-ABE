"""Zhang2024 Proxy Re-encryption CP-ABE baseline for comparison.

Simulated baseline based on published results from:
  Zhang et al., "Fine-grained data sharing based on proxy re-encryption
  in IIoT," IEEE TDSC, vol. 21, no. 1, 2024.

Since Zhang2024 uses proxy re-encryption (PRE) on top of CP-ABE, it has
additional computational overhead for the re-encryption step but enables
data sharing without exposing the original decryption key. This module
provides performance estimates based on the published complexity analysis.

Published performance characteristics (from Zhang2024, Table V-VI):
- KeyGen: O(n) pairings, ~1.1x of BSW07 KeyGen time
- Encrypt: O(n) pairings, ~1.05x of BSW07 Encrypt time
- Decrypt: O(k) pairings (k = satisfied attributes), ~1.15x of BSW07
- Re-encrypt: O(k) pairings (additional step, not in BSW07)

IMPORTANT: The values returned by this module are ESTIMATES derived from
the published paper's complexity analysis and reported experimental data.
They are NOT measured from an actual implementation. In the paper, these
are clearly labeled as "Zhang24* (estimated from [17])".
"""

import numpy as np


class Zhang2024Baseline:
    """Zhang2024 Proxy Re-encryption CP-ABE baseline (simulated).

    This class provides performance estimates for the Zhang2024 scheme based on
    the published complexity analysis. It is used for fair comparison in the
    SOTA comparison table.

    The estimates are calibrated against BSW07 baseline measurements using
    the complexity ratios reported in the original paper.
    """

    def __init__(self):
        self.scheme_name = "Zhang2024"
        self.source = "Zhang et al., IEEE TDSC, 2024"
        self.note = "Estimated from published complexity analysis"

    def estimate_keygen_time(self, num_attrs, bsw07_keygen_ms):
        """Estimate KeyGen time based on BSW07 and published ratios.

        Zhang2024 KeyGen includes proxy re-encryption key generation.
        Published ratio: ~1.10x of BSW07 KeyGen time (from Table VI).
        """
        ratio = 1.10
        return bsw07_keygen_ms * ratio

    def estimate_encrypt_time(self, num_attrs, bsw07_encrypt_ms):
        """Estimate Encrypt time based on BSW07 and published ratios.

        Zhang2024 Encrypt includes PRE encoding overhead.
        Published ratio: ~1.05x of BSW07 Encrypt time (from Table VI).
        """
        ratio = 1.05
        return bsw07_encrypt_ms * ratio

    def estimate_decrypt_time(self, num_attrs, bsw07_decrypt_ms):
        """Estimate Decrypt time based on BSW07 and published ratios.

        Zhang2024 Decrypt requires additional PRE decoding.
        Published ratio: ~1.15x of BSW07 Decrypt time (from Table VI).
        """
        ratio = 1.15
        return bsw07_decrypt_ms * ratio

    def get_scheme_info(self):
        """Return scheme metadata for documentation."""
        return {
            'scheme_name': 'Zhang2024',
            'full_name': 'Fine-grained data sharing based on proxy '
                         're-encryption in IIoT',
            'source': 'Zhang et al., IEEE TDSC, vol. 21, no. 1, 2024',
            'crypto_basis': 'CP-ABE with Proxy Re-encryption (pairing-based)',
            'security_model': 'CPA under DBDH + q-BDHE assumptions',
            'features': [
                'Proxy re-encryption for data sharing',
                'Fine-grained access control',
                'IIoT-optimized ciphertext structure',
            ],
            'limitations_vs_ours': [
                'No temporal access control',
                'No THRESHOLD gate support',
                'No integrated revocation mechanism',
                'Requires trusted proxy for re-encryption',
                'No digital twin integration',
            ],
            'data_source': 'Estimated from published results (not directly measured)',
        }
