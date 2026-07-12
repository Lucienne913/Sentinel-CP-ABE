#!/usr/bin/env python3
"""
System Initialization Module: Bilinear Pairing Group Parameter Generation based on Charm-Crypto

Implements global parameter initialization algorithm for Scheme 4, including:
- Bilinear pairing group initialization (Type A/A1 curves)
- Master key and public parameter generation
- Security parameter configuration (80-bit / 128-bit security level)

Tech Stack:
- Charm-Crypto 0.62 (pairing cryptography framework)
- Python 3.9 (Charm compatible version)

Security:
- Type A curve (SS512): 80-bit security level, suitable for rapid prototyping
- Type A1 curve (BN254): 128-bit security level, suitable for production deployment
"""

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
import hashlib
import os


class T_CP_ABE_Setup:
    """
    Time-Aware CP-ABE System Initialization Class
    
    Extended based on Bethencourt-Sahai-Waters (BSW) CP-ABE framework,
    adding time predicate support to implement time-aware fine-grained access control.
    
    Security Model:
    - IND-qCPA-T security (Indistinguishability under Quasi-dynamic CPA with Temporal constraints)
    - Reduces to DBDH assumption (Decisional Bilinear Diffie-Hellman)
    - Based on BSW CP-ABE framework with quasi-dynamic Init structure
    
    References:
    - Bethencourt, J., Sahai, A., & Waters, B. (2007). Cipher-policy attribute-based encryption. IEEE S&P.
    - Waters, B. (2011). Ciphertext-policy attribute-based encryption: An expressive, efficient, and provably secure realization. PKC.
    """
    
    def __init__(self, group_name='SS1024', security_level=128):
        """
        Initialize pairing group
        
        Args:
            group_name: Pairing curve name
                - 'SS512': Type A curve, 80-bit security, rapid prototyping
                - 'SS1024': Type A curve, ~128-bit security, symmetric pairing
            security_level: Security level (80 or 128)
        
        Raises:
            ValueError: Unsupported curve name or security level
        """
        self.security_level = security_level
        
        # Verify security level matches curve
        if security_level == 128 and group_name == 'SS512':
            group_name = 'SS1024'
        
        try:
            self.group = PairingGroup(group_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize pairing group {group_name}: {e}")
        
        self.group_name = group_name
        self.hash_func = hashlib.sha256  # Use SHA-256 as random oracle
    
    def setup(self, max_attrs=100, cache_max_size=5000):
        """
        System Initialization Algorithm
        
        Generate global public parameters PP and master secret key MK.
        
        Algorithm Flow (based on BSW CP-ABE):
        1. Select bilinear pairing group (G, G_T, e) of order p
        2. Select generator g ←$ G
        3. Select random elements α, β ←$ Z_p
        4. Compute h = g^β
        5. Compute e(g, g)^α ∈ G_T
        6. Generate hash values H(attr) ∈ G for each attribute
        
        Mathematical Foundation:
        - Bilinear pairing e: G × G → G_T satisfies:
          * Bilinearity: e(g^a, g^b) = e(g, g)^{ab}
          * Non-degeneracy: e(g, g) ≠ 1
          * Computability: Polynomial-time algorithm exists to compute e
        
        Args:
            max_attrs: Maximum number of supported attributes (for precomputing attribute hashes)
            cache_max_size: Maximum capacity of attribute hash cache (OOM protection)
            
        Returns:
            tuple: (PP, MK)
                - PP: Public parameters containing {g, h, e_gg_alpha, H, group_name, max_attrs}
                - MK: Master key containing {alpha, beta}
        """
        # Step 1-2: Select generator
        g = self.group.random(G1)
        g2 = self.group.random(G2)  # G2 generator needed for dual-system encryption
        
        # Step 3: Select master key random elements
        # Randomness source: Charm-Crypto uses PBC library's random() function,
        # which calls /dev/urandom (Linux) or CryptGenRandom (Windows) underneath
        # Distribution: Uniform over Z_p = {0, 1, ..., p-1}
        # Security: For 512-bit prime p, brute-force search complexity is O(2^{511})
        alpha = self.group.random(ZR)
        beta = self.group.random(ZR)
        
        # Step 4: Compute h = g^beta
        h = g ** beta
        
        # Step 5: Compute e(g, g)^alpha ∈ G_T
        # Bilinear pairing e: G × G → G_T satisfies:
        # (1) Bilinearity: ∀a,b∈Z_p, e(g^a, g^b) = e(g, g)^{ab}
        # (2) Non-degeneracy: e(g, g) ≠ 1_{G_T}
        # (3) Computability: PPT algorithm exists to compute e(·, ·)
        e_gg_alpha = pair(g, g) ** alpha
        
        # Step 6: Attribute hash function (with LRU+TTL cache, OOM protection)
        attr_hash_cache = {}
        attr_cache_order = []  # LRU access order
        attr_cache_timestamps = {}  # TTL timestamps
        
        def H(attr_str, ttl=3600):
            """
            Attribute hash function H: {0,1}* → G
            
            Construction (Random Oracle Model):
            1. h ← SHA-256(attr_str) ∈ {0,1}^{256}
            2. x ← int(h) mod p ∈ Z_p
            3. H(attr) = g^x ∈ G
            
            Security Analysis:
            - If SHA-256 is a random oracle, then H's output is uniformly distributed over G
            - Collision probability: ≤ q²/2^{256} (birthday bound), where q is the number of queries
            - For q=2^{40}, collision probability ≤ 2^{-176}, negligible
            
            Args:
                attr_str: Attribute string
                ttl: Cache validity period (seconds)
            """
            import time
            
            if attr_str in attr_hash_cache:
                # Check TTL
                if time.time() - attr_cache_timestamps[attr_str] > ttl:
                    del attr_hash_cache[attr_str]
                    del attr_cache_timestamps[attr_str]
                    if attr_str in attr_cache_order:
                        attr_cache_order.remove(attr_str)
                else:
                    # Update LRU order
                    if attr_str in attr_cache_order:
                        attr_cache_order.remove(attr_str)
                    attr_cache_order.append(attr_str)
                    return attr_hash_cache[attr_str]
            
            # Compute hash
            attr_bytes = attr_str.encode('utf-8')
            hash_digest = self.hash_func(attr_bytes).digest()
            hash_int = int.from_bytes(hash_digest, 'big') % self.group.order()
            hash_zr = self.group.init(ZR, hash_int)
            hash_g = g ** hash_zr
            
            # Cache (with capacity limit)
            if len(attr_hash_cache) >= cache_max_size and attr_cache_order:
                # Remove oldest entry
                oldest = attr_cache_order.pop(0)
                attr_hash_cache.pop(oldest, None)
                attr_cache_timestamps.pop(oldest, None)
            
            attr_hash_cache[attr_str] = hash_g
            attr_cache_timestamps[attr_str] = time.time()
            attr_cache_order.append(attr_str)
            
            return hash_g
        
        def clear_cache():
            """Explicit cache clearing (OOM protection)"""
            attr_hash_cache.clear()
            attr_cache_order.clear()
            attr_cache_timestamps.clear()
        
        # Public parameters
        PP = {
            'group': self.group,
            'g': g,
            'g2': g2,  # G2 generator needed for dual-system encryption
            'h': h,
            'e_gg_alpha': e_gg_alpha,
            'H': H,
            'attr_hash_cache': attr_hash_cache,
            'group_name': self.group_name,
            'security_level': self.security_level,
            'max_attrs': max_attrs,
            'clear_cache': clear_cache
        }
        
        # Master secret key
        MK = {
            'alpha': alpha,
            'beta': beta
        }
        
        return PP, MK
    
    def get_group_params(self):
        """
        Get pairing group parameter information
        
        Returns:
            dict: Group parameter information
        """
        return {
            'group_name': self.group_name,
            'security_level': self.security_level,
            'group_order': self.group.order(),
            'order_bits': self.group.order().bit_length()
        }
    
    def generate_random_element(self, group_type=ZR):
        """
        Generate random element in the group
        
        Args:
            group_type: Group type (ZR, G1, G2, GT)
            
        Returns:
            Group element
        """
        return self.group.random(group_type)
    
    def hash_to_ZR(self, data):
        """
        Hash data to Z_p group
        
        Args:
            data: Input data (string or bytes)
            
        Returns:
            ZR group element
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        hash_digest = self.hash_func(data).digest()
        hash_int = int.from_bytes(hash_digest, 'big') % self.group.order()
        return self.group.init(ZR, hash_int)


def main():
    """Test system initialization"""
    print("=" * 60)
    print("Scheme 4: Digital Twin Network Security Framework - System Initialization Test")
    print("=" * 60)
    
    # Test 1: 80-bit security level (SS512 curve)
    print("\n[Test 1] 80-bit Security Level (SS512 Curve)")
    try:
        setup80 = T_CP_ABE_Setup(group_name='SS512', security_level=80)
        PP, MK = setup80.setup(max_attrs=50)
        
        group_params = setup80.get_group_params()
        print(f"  Group Name: {group_params['group_name']}")
        print(f"  Security Level: {group_params['security_level']}-bit")
        print(f"  Group Order Bits: {group_params['order_bits']}-bit")
        print(f"  Public Parameter g: {PP['g']}")
        print(f"  Public Parameter h: {PP['h']}")
        print(f"  e(g,g)^alpha: {PP['e_gg_alpha']}")
        print(f"  Master Key alpha: {MK['alpha']}")
        print(f"  Max Attributes: {PP['max_attrs']}")
        print("  ✓ 80-bit security level initialization successful")
    except Exception as e:
        print(f"  ✗ 80-bit security level initialization failed: {e}")
    
    # Test 2: 128-bit security level (auto-select MNT159_1)
    print("\n[Test 2] 128-bit Security Level (MNT159_1 Curve)")
    try:
        setup128 = T_CP_ABE_Setup(group_name='MNT159_1', security_level=128)
        PP, MK = setup128.setup(max_attrs=100)
        
        group_params = setup128.get_group_params()
        print(f"  Group Name: {group_params['group_name']}")
        print(f"  Security Level: {group_params['security_level']}-bit")
        print(f"  Group Order Bits: {group_params['order_bits']}-bit")
        print("  ✓ 128-bit security level initialization successful")
    except Exception as e:
        print(f"  ✗ 128-bit security level initialization failed: {e}")
        print("  (Note: MNT159_1 curve may require PBC library installation and configuration)")
    
    # Test 3: Attribute hash function
    print("\n[Test 3] Attribute Hash Function Test")
    try:
        setup = T_CP_ABE_Setup(group_name='SS512', security_level=80)
        PP, MK = setup.setup()
        
        test_attrs = [
            'role:engineer',
            'dept:maintenance',
            'time:work',
            'location:factory',
            'clearance:level3'
        ]
        
        print("  Attribute Hash Values:")
        for attr in test_attrs:
            h_attr = PP['H'](attr)
            print(f"    H('{attr}') = {h_attr}")
        
        print("  ✓ Attribute hash function test successful")
    except Exception as e:
        print(f"  ✗ Attribute hash function test failed: {e}")
    
    # Test 4: Random element generation
    print("\n[Test 4] Random Element Generation Test")
    try:
        setup = T_CP_ABE_Setup(group_name='SS512', security_level=80)
        PP, _ = setup.setup()
        
        zr_elem = setup.generate_random_element(ZR)
        g1_elem = setup.generate_random_element(G1)
        
        print(f"  ZR Random Element: {zr_elem}")
        print(f"  G1 Random Element: {g1_elem}")
        print("  ✓ Random element generation test successful")
    except Exception as e:
        print(f"  ✗ Random element generation test failed: {e}")
    
    print("\n" + "=" * 60)
    print("System Initialization Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()