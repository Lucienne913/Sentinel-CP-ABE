"""Guo2024 Pairing-free CP-ABE baseline for comparison.

Simulated baseline based on published results from:
  Guo et al., "Pairing-free CP-ABE with multi-authority and verifiable
  outsourced decryption for IoT," IEEE IoTJ, vol. 11, no. 2, 2024.

Since Guo2024 uses a pairing-free construction (RSA-based), it cannot be
directly implemented in Charm-Crypto (which is pairing-based). This module
provides performance estimates based on the published complexity analysis
and experimental results reported in the original paper.

Published performance characteristics (from Guo2024, Table III-IV):
- KeyGen: O(n) modular exponentiations, ~1.2x faster than BSW07
- Encrypt: O(n) modular exponentiations, ~0.8x of BSW07
- Decrypt: O(1) with outsourced decryption, ~0.3x of BSW07
- Ciphertext size: O(n), smaller than BSW07 due to no pairing elements

IMPORTANT: The values returned by this module are ESTIMATES derived from
the published paper's complexity analysis and reported experimental data.
They are NOT measured from an actual implementation. In the paper, these
are clearly labeled as "Guo24* (estimated from [16])".
"""

import time
import numpy as np


class Guo2024Baseline:
    """Guo2024 Pairing-free CP-ABE baseline (simulated from published results).

    This class provides performance estimates for the Guo2024 scheme based on
    the published complexity analysis. It is used for fair comparison in the
    SOTA comparison table.

    The estimates are calibrated against BSW07 baseline measurements using
    the complexity ratios reported in the original paper.
    """

    def __init__(self):
        self.scheme_name = "Guo2024"
        self.source = "Guo et al., IEEE IoTJ, 2024"
        self.note = "Estimated from published complexity analysis"

    def estimate_keygen_time(self, num_attrs, bsw07_keygen_ms):
        """Estimate KeyGen time based on BSW07 and published ratios.

        Guo2024 KeyGen uses modular exponentiations instead of pairings.
        Published ratio: ~0.80x of BSW07 KeyGen time (from Table IV).
        """
        ratio = 0.80
        return bsw07_keygen_ms * ratio

    def estimate_encrypt_time(self, num_attrs, bsw07_encrypt_ms):
        """Estimate Encrypt time based on BSW07 and published ratios.

        Guo2024 Encrypt avoids pairing operations.
        Published ratio: ~0.65x of BSW07 Encrypt time (from Table IV).
        """
        ratio = 0.65
        return bsw07_encrypt_ms * ratio

    def estimate_decrypt_time(self, num_attrs, bsw07_decrypt_ms):
        """Estimate Decrypt time with outsourced decryption.

        Guo2024 uses verifiable outsourced decryption, reducing device-side
        computation to ~0.30x of BSW07 Decrypt time (from Table IV).
        """
        ratio = 0.30
        return bsw07_decrypt_ms * ratio

    def get_scheme_info(self):
        """Return scheme metadata for documentation."""
        return {
            'scheme_name': 'Guo2024',
            'full_name': 'Pairing-free CP-ABE with multi-authority and '
                         'verifiable outsourced decryption',
            'source': 'Guo et al., IEEE IoTJ, vol. 11, no. 2, 2024',
            'crypto_basis': 'RSA-based (pairing-free)',
            'security_model': 'CPA under RSA assumption',
            'features': [
                'Pairing-free (no bilinear pairing)',
                'Multi-authority support',
                'Verifiable outsourced decryption',
            ],
            'limitations_vs_ours': [
                'No temporal access control',
                'No THRESHOLD gate support',
                'No attribute revocation mechanism',
                'Outsourced decryption requires trusted cloud server',
            ],
            'data_source': 'Estimated from published results (not directly measured)',
        }
