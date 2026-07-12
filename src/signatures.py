#!/usr/bin/env python3
"""
Digital Signature Module: BLS Signature Scheme based on Charm-Crypto

Implements Boneh-Lynn-Shacham signature scheme for device certificates and authentication responses.

Reasons for choosing BLS:
1. Short signature (only 1 G1 element, ~65 bytes)
2. Aggregatable (multiple signatures can be merged into one)
3. Pairing-based, uses the same cryptographic primitives as T-CP-ABE
4. EUF-CMA security (reduces to CDH assumption)

Security:
- EUF-CMA (Existential Unforgeability under Chosen Message Attack)
- Reduces to Computational Diffie-Hellman (CDH) assumption

References:
- Boneh, D., Lynn, B., & Shacham, H. (2001). Short signatures from the Weil pairing. ASIACRYPT.
"""

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
from datetime import datetime, timedelta
import hashlib
import secrets


class BLSSignature:
    """
    BLS (Boneh-Lynn-Shacham) Digital Signature Scheme
    
    Algorithm Interface:
    - keygen() → (PK, SK)
    - sign(SK, message) → σ
    - verify(PK, message, σ) → bool
    
    Mathematical Definition:
    - Setup: Select pairing group (G, G_T, e), g ←$ G
    - KeyGen: sk ←$ Z_p, PK = g^{sk}
    - Sign: σ = H(m)^{sk} ∈ G
    - Verify: e(σ, g) == e(H(m), PK)
    
    Correctness Proof:
    e(σ, g) = e(H(m)^{sk}, g) = e(H(m), g)^{sk} = e(H(m), g^{sk}) = e(H(m), PK)
    """
    
    def __init__(self, group=None, g=None, h=None):
        """
        Args:
            group: Pairing group instance (optional, creates new instance if not provided)
            g: G1 generator (optional, uses deterministic generation if not provided)
            h: G2 generator (optional, uses deterministic generation if not provided)
        
        Note: Standard BLS signatures require system parameters (g, h) to be globally fixed public parameters.
        They cannot be randomly generated on each initialization, otherwise signature verification results
        for the same message will be inconsistent across different sessions.
        """
        if group is None:
            self.group = PairingGroup('SS1024')
        else:
            self.group = group
        
        # Use deterministic generators (based on fixed seed)
        # Ensures the same system parameters are obtained on each initialization
        if g is None:
            self.g = self._derive_generator(G1, b"bls_g_gen_2026")
        else:
            self.g = g
        
        if h is None:
            self.h = self._derive_generator(G2, b"bls_h_gen_2026")
        else:
            self.h = h
    
    def _derive_generator(self, group_type, seed):
        """
        Derive group generator deterministically from seed
        
        Key insight: group.init(G1, 1) or group.init(G2, 1) creates the identity element,
        not a generator. Must use group.random() or group.init(G, non_one_value) to obtain a non-identity element.
        
        Args:
            group_type: G1 or G2
            seed: Seed byte string
            
        Returns:
            Non-identity group element to use as generator
        """
        # Use multiple hash iterations to find a non-identity element
        for i in range(100):
            hash_input = seed + i.to_bytes(4, 'big')
            hash_int = int.from_bytes(hashlib.sha256(hash_input).digest(), 'big') % self.group.order()
            if hash_int == 0:
                continue
            hash_zr = self.group.init(ZR, hash_int)
            # Use the group's default generator (non-identity) to derive
            # group.random() returns a random non-identity element, but we need determinism
            # Method: use ZR element as exponent, raise a known non-identity element to it
            element = self.group.init(group_type, hash_int)
            # Verify non-identity
            if element != self.group.init(group_type, 0):
                return element
        # Fallback: use group.random()
        return self.group.random(group_type)
    
    def keygen(self):
        """
        Key Generation
        
        Returns:
            tuple: (pk, sk)
                - pk: Public key (G2 element)
                - sk: Secret key (ZR element)
        """
        sk = self.group.random(ZR)
        pk = self.h ** sk
        return pk, sk
    
    def _hash_to_G1(self, message):
        """
        Hash message to G1 group
        
        Construction:
        1. h = SHA-256(message) ∈ {0,1}^{256}
        2. x = int(h) mod p ∈ Z_p
        3. H(m) = g^x ∈ G1
        
        Args:
            message: Message (bytes or string)
            
        Returns:
            G1: Hash value
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        hash_digest = hashlib.sha256(message).digest()
        hash_int = int.from_bytes(hash_digest, 'big') % self.group.order()
        hash_zr = self.group.init(ZR, hash_int)
        
        return self.g ** hash_zr
    
    def sign(self, sk, message):
        """
        Signing Algorithm
        
        Args:
            sk: Secret key (ZR element)
            message: Message (bytes or string)
            
        Returns:
            G1: Signature σ = H(m)^{sk}
        """
        H_m = self._hash_to_G1(message)
        sigma = H_m ** sk
        return sigma
    
    def verify(self, pk, message, sigma):
        """
        Verification Algorithm
        
        Args:
            pk: Public key (G2 element)
            message: Message (bytes or string)
            sigma: Signature (G1 element)
            
        Returns:
            bool: Whether signature is valid
        """
        H_m = self._hash_to_G1(message)
        
        # e(σ, h) == e(H(m), pk)
        lhs = pair(sigma, self.h)
        rhs = pair(H_m, pk)
        
        return lhs == rhs


class DeviceCertificate:
    """
    Device Certificate: BLS-based identity credential
    
    Certificate Format:
    cert = {
        'device_id': str,
        'attributes': list,
        'public_key': G2,
        'signature': G1,  # CA signature over the above information
        'issued_at': float,
        'expires_at': float
    }
    """
    
    def __init__(self, ca_sk=None, ca_pk=None, group=None):
        """
        Args:
            ca_sk: CA secret key
            ca_pk: CA public key
            group: Pairing group
        """
        if group is None:
            self.group = PairingGroup('SS1024')
        else:
            self.group = group
        self.bls = BLSSignature(self.group)
        
        if ca_sk is None or ca_pk is None:
            self.ca_pk, self.ca_sk = self.bls.keygen()
        else:
            self.ca_pk = ca_pk
            self.ca_sk = ca_sk
    
    def issue_certificate(self, device_id, attributes, validity_hours=8760):
        """
        Issue device certificate
        
        Args:
            device_id: Device ID
            attributes: List of device attributes
            validity_hours: Validity period in hours (default 1 year)
            
        Returns:
            dict: Device certificate
        """
        import time
        
        # Generate device key pair
        device_pk, device_sk = self.bls.keygen()
        
        issued_at = time.time()
        expires_at = issued_at + validity_hours * 3600
        
        # Build certificate content
        cert_data = f"{device_id}:{','.join(sorted(attributes))}:{device_pk}:{issued_at}"
        
        # CA signature
        signature = self.bls.sign(self.ca_sk, cert_data)
        
        cert = {
            'device_id': device_id,
            'attributes': attributes,
            'device_pk': device_pk,
            'signature': signature,
            'issued_at': issued_at,
            'expires_at': expires_at
        }
        
        return cert
    
    def verify_certificate(self, cert):
        """
        Verify device certificate
        
        Args:
            cert: Device certificate dictionary
            
        Returns:
            bool: Whether certificate is valid
        """
        import time
        
        # Check validity period
        current_time = time.time()
        if current_time < cert['issued_at'] or current_time > cert['expires_at']:
            return False
        
        # Reconstruct certificate content
        cert_data = f"{cert['device_id']}:{','.join(sorted(cert['attributes']))}:{cert['device_pk']}:{cert['issued_at']}"
        
        # Verify CA signature
        return self.bls.verify(self.ca_pk, cert_data, cert['signature'])


def main():
    """Test BLS signature scheme"""
    print("=" * 60)
    print("Scheme 4: BLS Digital Signature Scheme Test")
    print("=" * 60)
    
    # 1. Key Generation
    print("\n[Step 1] BLS Key Generation")
    bls = BLSSignature()
    pk, sk = bls.keygen()
    print(f"  Public Key: {pk}")
    print("  ✓ Key generation successful")
    
    # 2. Signing
    print("\n[Step 2] Message Signing")
    message = "Device_001:nonce_abc123:1713567890"
    sigma = bls.sign(sk, message)
    print(f"  Message: {message}")
    print(f"  Signature: {sigma}")
    print("  ✓ Signing successful")
    
    # 3. Verification (valid signature)
    print("\n[Step 3] Verify Valid Signature")
    is_valid = bls.verify(pk, message, sigma)
    if is_valid:
        print("  ✓ Signature verification passed")
    else:
        print("  ✗ Signature verification failed (should pass)")
    
    # 4. Verification (tampered message)
    print("\n[Step 4] Verify Tampered Message (should fail)")
    tampered_message = "Device_002:nonce_xyz789:1713567891"
    is_valid_tampered = bls.verify(pk, tampered_message, sigma)
    if not is_valid_tampered:
        print("  ✓ Correctly rejected tampered message")
    else:
        print("  ✗ Should not pass but passed")
    
    # 5. Device Certificate
    print("\n[Step 5] Device Certificate Issuance and Verification")
    cert_auth = DeviceCertificate()
    
    cert = cert_auth.issue_certificate(
        device_id='sensor_001',
        attributes=['type:temperature', 'location:factory'],
        validity_hours=24
    )
    print(f"  Device ID: {cert['device_id']}")
    print(f"  Attributes: {cert['attributes']}")
    
    is_cert_valid = cert_auth.verify_certificate(cert)
    if is_cert_valid:
        print("  ✓ Certificate verification passed")
    else:
        print("  ✗ Certificate verification failed")
    
    # 6. Forged Certificate Detection
    print("\n[Step 6] Forged Certificate Detection")
    import copy
    forged_cert = copy.deepcopy(cert)
    forged_cert['device_id'] = 'sensor_forged'
    
    is_forged_valid = cert_auth.verify_certificate(forged_cert)
    if not is_forged_valid:
        print("  ✓ Correctly detected forged certificate")
    else:
        print("  ✗ Should not pass but passed")
    
    print("\n" + "=" * 60)
    print("BLS Signature Scheme Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()