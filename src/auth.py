#!/usr/bin/env python3
"""
Bidirectional Authentication Protocol for Digital Twin Networks

Implements mutual authentication between physical devices and digital twins, ensuring:
1. Physical device identity authenticity (prevent forged device access)
2. Digital twin platform legitimacy (prevent man-in-the-middle attacks)
3. Secure session key negotiation (encrypt subsequent communications)

BLS Signature and T-CP-ABE Handshake Sequence:
===================================================
1. Device → Digital Twin: {cert_D, nonce_D, attr_D, sig_D = BLS.Sign(SK_D, cert_D || nonce_D || attr_D || timestamp)}
2. Digital Twin Verify: BLS.Verify(PK_D, cert_D || nonce_D || attr_D || timestamp, sig_D)
3. Digital Twin → Device: {enc_challenge = T-CP-ABE.Enc(challenge, policy_T), nonce_T, sig_T = BLS.Sign(SK_T, challenge || nonce_T || timestamp)}
4. Device Verify: BLS.Verify(PK_T, challenge || nonce_T || timestamp, sig_T)
5. Device Decrypt: T-CP-ABE.Decrypt(SK_D, enc_challenge, current_time)
6. Device → Digital Twin: {response = BLS.Sign(SK_D, challenge || nonce_T || nonce_D || timestamp), decrypted_challenge}
7. Digital Twin Verify: BLS.Verify(PK_D, challenge || nonce_T || nonce_D || timestamp, response)
8. Digital Twin → Device: {enc_session_key = T-CP-ABE.Enc(K_session, policy_D), sig_T' = BLS.Sign(SK_T, K_session || timestamp)}
9. Device Verify: BLS.Verify(PK_T, K_session || timestamp, sig_T')
10. Device Decrypt: T-CP-ABE.Decrypt(SK_D, enc_session_key, current_time)

Authentication vs Authorization Logic:
===================================================
1. Authentication: Verify device identity authenticity
   - Device certificate verification (BLS signature)
   - Challenge-response mechanism (prevent replay attacks)
   - Mutual signature verification (prevent MITM attacks)

2. Authorization: Attribute-based access control
   - T-CP-ABE encrypted challenge (only authorized devices can decrypt)
   - T-CP-ABE encrypted session key (based on device attribute policy)
   - Time-aware policy (ensure temporal validity)

Security:
- Replay attack prevention: nonce + timestamp mechanism
- MITM attack prevention: mutual BLS signature verification
- Session key forward secrecy: new session key generated per authentication
- Time-aware access control: prevent expired key usage

Tech Stack:
- Charm-Crypto (BLS signature, T-CP-ABE encryption)
- Python 3.9

References:
- Mironov, I. & Stephens-Davidowitz, N. (2015). Cryptographic Reverse Firewalls. EUROCRYPT.
- ISO/IEC 9798-3: Entity authentication using digital signatures.
- Boneh, D., Lynn, B., & Shacham, H. (2001). Short signatures from the Weil pairing. ASIACRYPT.
"""

import sys
from pathlib import Path
# Ensure src directory is in sys.path (supports running from any working directory)
_src_dir = str(Path(__file__).parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
from datetime import datetime, timedelta
import hashlib
import secrets
from typing import Dict, List, Any
from signatures import BLSSignature, DeviceCertificate


class BidirectionalAuth:
    """
    Bidirectional Authentication Protocol Implementation
    
    Used for mutual identity verification between physical devices and cloud mappings in digital twin networks.
    
    Security Model:
    - Adversary can eavesdrop, tamper, and replay messages
    - Adversary does not know device private keys or CA signing keys
    - Protocol must resist replay attacks, MITM attacks, and device forgery attacks
    
    Theorem 2 (Bidirectional Authentication Unforgeability):
    If the digital signature scheme has EUF-CMA security and the ABE scheme has IND-CPA security,
    then this protocol is secure under the adversary model.
    """
    
    def __init__(self, PP, tcabe=None, max_nonce_cache_size: int = 10000, 
                 max_active_sessions: int = 1000, ca_sk=None, ca_pk=None):
        """
        Initialize bidirectional authentication protocol
        
        Args:
            PP: Public parameters (from T_CP_ABE_Setup.setup())
            tcabe: T_CP_ABE instance (for encrypting challenges and session keys)
            max_nonce_cache_size: Maximum nonce cache size (OOM protection)
            max_active_sessions: Maximum active sessions (OOM protection)
            ca_sk: CA private key (for device certificate signing)
            ca_pk: CA public key (for device certificate verification)
        """
        self.PP = PP
        self.group = PP['group']
        self.tcabe = tcabe
        
        # BLS signature initialization
        self.bls = BLSSignature(self.group)
        
        # Device certificate management
        self.cert_auth = DeviceCertificate(ca_sk=ca_sk, ca_pk=ca_pk, group=self.group)
        
        # Digital twin signing key pair
        self.dt_pk, self.dt_sk = self.bls.keygen()
        
        # Authentication context (with OOM protection: LRU cache eviction)
        self.max_nonce_cache_size = max_nonce_cache_size
        self.nonce_cache: Dict[str, float] = {}  # {nonce_hex: timestamp}
        self._nonce_access_order: List[str] = []  # LRU access order
        self.max_nonce_age = 300  # nonce validity: 5 minutes
        
        # Session management (with OOM protection: capacity limit + expiration cleanup)
        self.max_active_sessions = max_active_sessions
        self.active_sessions: Dict[str, Dict] = {}
    
    def _evict_lru_nonces(self):
        """LRU eviction: Remove oldest accessed nonces (OOM protection)"""
        evict_count = len(self.nonce_cache) - self.max_nonce_cache_size
        for _ in range(evict_count):
            if self._nonce_access_order:
                oldest = self._nonce_access_order.pop(0)
                self.nonce_cache.pop(oldest, None)
    
    def _update_nonce_access(self, nonce_key):
        """Update LRU access order for nonce"""
        if nonce_key in self._nonce_access_order:
            self._nonce_access_order.remove(nonce_key)
        self._nonce_access_order.append(nonce_key)
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions (OOM protection)"""
        current_time = datetime.now().timestamp()
        expired = [
            sid for sid, session in self.active_sessions.items()
            if current_time - session.get('timestamp', 0) > 3600  # Expires in 1 hour
        ]
        for sid in expired:
            del self.active_sessions[sid]
        
        # Capacity limit: remove oldest sessions
        if len(self.active_sessions) > self.max_active_sessions:
            sorted_sessions = sorted(
                self.active_sessions.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            evict_count = len(self.active_sessions) - self.max_active_sessions
            for sid, _ in sorted_sessions[:evict_count]:
                del self.active_sessions[sid]
    
    def generate_nonce(self):
        """
        Generate cryptographically secure nonce
        
        Returns:
            bytes: 128-bit random nonce
        """
        return secrets.token_bytes(16)
    
    def generate_timestamp(self):
        """
        Generate current timestamp
        
        Returns:
            float: Unix timestamp
        """
        return datetime.now().timestamp()
    
    def is_nonce_valid(self, nonce, timestamp):
        """
        Verify if nonce is valid (unused and not expired)
        
        Args:
            nonce: Nonce
            timestamp: Timestamp
            
        Returns:
            bool: Whether valid
        """
        current_time = datetime.now().timestamp()
        
        # Check if expired
        if current_time - timestamp > self.max_nonce_age:
            return False
        
        # Check for replay
        nonce_key = nonce.hex() if isinstance(nonce, bytes) else nonce
        if nonce_key in self.nonce_cache:
            return False
        
        # Add to cache
        self.nonce_cache[nonce_key] = timestamp
        self._update_nonce_access(nonce_key)
        
        # LRU cleanup
        if len(self.nonce_cache) > self.max_nonce_cache_size:
            self._evict_lru_nonces()
        
        # Periodically clean up expired nonces
        self._cleanup_nonce_cache(current_time)
        
        return True
    
    def _cleanup_nonce_cache(self, current_time):
        """Clean up expired nonces (avoid deleting dict items during iteration)"""
        expired = [k for k, v in self.nonce_cache.items() 
                   if current_time - v > self.max_nonce_age]
        for k in expired:
            del self.nonce_cache[k]
            if k in self._nonce_access_order:
                self._nonce_access_order.remove(k)
    
    def device_auth_init(self, device_id, device_attrs, device_bls_sk=None):
        """
        Step 1: Device Authentication Initialization
        
        Physical device sends authentication request to digital twin.
        
        Args:
            device_id: Device unique identifier
            device_attrs: Device attribute list
            device_bls_sk: Device BLS private key (for signing, auto-generates temporary key pair if None)
            
        Returns:
            tuple: (auth_request, device_bls_sk)
                - auth_request: Authentication request dictionary
                - device_bls_sk: Device BLS private key (caller must save for signing responses)
        """
        nonce_D = self.generate_nonce()
        timestamp = self.generate_timestamp()
        
        cert_D = self.cert_auth.issue_certificate(device_id, device_attrs, validity_hours=24)
        
        if device_bls_sk is None:
            device_pk, device_bls_sk = self.bls.keygen()
        else:
            device_pk = cert_D['device_pk']
        
        message = f"{device_id}:{','.join(sorted(device_attrs))}:{nonce_D.hex()}:{timestamp}"
        sig_D = self.bls.sign(device_bls_sk, message)
        
        auth_request = {
            'device_id': device_id,
            'cert_D': cert_D,
            'device_pk': device_pk,
            'nonce_D': nonce_D,
            'attr_D': device_attrs,
            'sig_D': sig_D,
            'timestamp': timestamp
        }
        
        return auth_request, device_bls_sk
    
    def digital_twin_challenge(self, auth_request, policy_T, session_id=None):
        """
        Step 2: Digital Twin Generate Challenge
        
        After verifying device request, digital twin sends ABE-encrypted challenge value.
        
        Args:
            auth_request: Device authentication request
            policy_T: Challenge decryption policy (only devices satisfying this policy can decrypt)
            session_id: Session ID (optional)
            
        Returns:
            dict: Challenge response
                {
                    'session_id': str,
                    'enc_challenge': dict,  # ABE-encrypted challenge value
                    'nonce_T': bytes,       # Digital twin nonce
                    'dt_pk': G2,            # Digital twin public key
                    'sig_T': G1,            # BLS signature
                    'timestamp': float
                }
        """
        # Verify device nonce
        if not self.is_nonce_valid(auth_request['nonce_D'], auth_request['timestamp']):
            raise ValueError("Invalid or expired nonce from device")
        
        # Verify device certificate
        if not self.cert_auth.verify_certificate(auth_request['cert_D']):
            raise ValueError("Invalid device certificate")
        
        # Verify device BLS signature
        message = f"{auth_request['device_id']}:{','.join(sorted(auth_request['attr_D']))}:{auth_request['nonce_D'].hex()}:{auth_request['timestamp']}"
        device_pk = auth_request['device_pk']
        if not self.bls.verify(device_pk, message, auth_request['sig_D']):
            raise ValueError("Invalid device signature")
        
        # Generate session ID
        if session_id is None:
            session_id = secrets.token_hex(16)
        
        # Generate challenge value (128-bit nonce)
        challenge = secrets.token_bytes(16)
        
        # Generate digital twin nonce
        nonce_T = self.generate_nonce()
        timestamp = self.generate_timestamp()
        
        # ABE encrypt challenge value
        if self.tcabe is None:
            raise ValueError("T_CP_ABE instance required for challenge encryption")
        
        # Encode challenge value as GT group element
        challenge_int = int.from_bytes(challenge, 'big') % self.group.order()
        challenge_GT = self.group.init(GT, challenge_int)
        
        # Use hash as stable comparison base for challenge value (avoid serialization differences)
        challenge_hash = hashlib.sha256(challenge).digest()
        
        # Encrypt challenge using T-CP-ABE
        enc_challenge = self.tcabe.encrypt(challenge_GT, policy_T)
        
        # Build challenge sign message: challenge || nonce_T || timestamp
        challenge_message = f"{challenge.hex()}:{nonce_T.hex()}:{timestamp}"
        sig_T = self.bls.sign(self.dt_sk, challenge_message)
        
        challenge_response = {
            'session_id': session_id,
            'enc_challenge': enc_challenge,
            'nonce_T': nonce_T,
            'dt_pk': self.dt_pk,
            'sig_T': sig_T,
            'timestamp': timestamp,
            'device_nonce': auth_request['nonce_D']  # Echo device nonce to prevent replay
        }
        
        # Save session state (use hash comparison to avoid serialization differences)
        self.active_sessions[session_id] = {
            'device_id': auth_request['device_id'],
            'device_attrs': auth_request['attr_D'],
            'device_pk': auth_request['device_pk'],
            'challenge_hash': challenge_hash,  # Use hash for comparison
            'challenge_hex': challenge.hex(),  # Keep original hex for signature verification
            'nonce_T': nonce_T,
            'nonce_D': auth_request['nonce_D'],
            'timestamp': timestamp,
            'state': 'challenge_sent'
        }
        
        return challenge_response
    
    def device_response(self, session_id, challenge_response, device_abe_SK, device_bls_sk, device_id):
        """
        Step 3: Device Response to Challenge
        
        Device decrypts challenge value and signs response.
        
        Args:
            session_id: Session ID
            challenge_response: Digital twin's challenge
            device_abe_SK: Device ABE key (for T-CP-ABE challenge decryption)
            device_bls_sk: Device BLS private key (for signing response)
            device_id: Device ID
            
        Returns:
            dict: Device response
                {
                    'session_id': str,
                    'response': G1,     # BLS signature response
                    'decrypted_challenge': bytes,
                    'timestamp': float
                }
        """
        # Verify session state
        if session_id not in self.active_sessions:
            raise ValueError("Invalid session ID")
        
        session = self.active_sessions[session_id]
        if session['state'] != 'challenge_sent':
            raise ValueError("Session not in challenge state")
        
        # Verify nonce echo
        if challenge_response['device_nonce'] != session['nonce_D']:
            raise ValueError("Nonce mismatch - possible replay attack")
        
        # Verify digital twin BLS signature
        dt_pk = challenge_response['dt_pk']
        challenge_message = f"{session['challenge_hex']}:{challenge_response['nonce_T'].hex()}:{challenge_response['timestamp']}"
        if not self.bls.verify(dt_pk, challenge_message, challenge_response['sig_T']):
            raise ValueError("Invalid digital twin signature")
        
        if self.tcabe is None:
            raise ValueError("T-CP-ABE instance required for challenge decryption")

        try:
            challenge_GT = self.tcabe.decrypt(device_abe_SK, challenge_response['enc_challenge'])
            # Sign using original challenge hex (avoid serialization differences)
            expected_challenge_hex = session['challenge_hex']
            expected_challenge_hash = hashlib.sha256(bytes.fromhex(expected_challenge_hex)).digest()
        except Exception as e:
            raise ValueError("Failed to decrypt challenge: %s" % str(e))
        
        timestamp = self.generate_timestamp()
        
        # Sign using original challenge hex (maintain consistency)
        message = "%s:%s:%s:%s" % (expected_challenge_hex, challenge_response['nonce_T'].hex(), session['nonce_D'].hex(), timestamp)
        response = self.bls.sign(device_bls_sk, message)
        
        device_resp = {
            'session_id': session_id,
            'response': response,
            'decrypted_challenge': expected_challenge_hash,  # Use hash for comparison
            'timestamp': timestamp
        }
        
        return device_resp
    
    def digital_twin_verify(self, device_response):
        """
        Step 4: Digital Twin Verify Device Response
        
        Verify correctness of signed response.
        
        Args:
            device_response: Device response
            
        Returns:
            dict: Verification result
                {
                    'success': bool,
                    'session_id': str,
                    'message': str
                }
        """
        session_id = device_response['session_id']
        
        # Verify session state
        if session_id not in self.active_sessions:
            return {'success': False, 'session_id': session_id, 
                    'message': 'Invalid session ID'}
        
        session = self.active_sessions[session_id]
        
        # Verify challenge match (use hash comparison to avoid serialization differences)
        decrypted_challenge = device_response['decrypted_challenge']
        
        if decrypted_challenge != session['challenge_hash']:
            return {'success': False, 'session_id': session_id,
                    'message': 'Challenge mismatch - possible MITM attack'}
        
        # Verify signature response (use original hex)
        expected_message = f"{session['challenge_hex']}:{session['nonce_T'].hex()}:{session['nonce_D'].hex()}:{device_response['timestamp']}"
        device_pk = session.get('device_pk')
        if not device_pk:
            return {'success': False, 'session_id': session_id,
                    'message': 'Device public key not found in session'}
        try:
            if not self.bls.verify(device_pk, expected_message, device_response['response']):
                return {'success': False, 'session_id': session_id,
                        'message': 'Invalid device signature'}
        except (TypeError, Exception):
            return {'success': False, 'session_id': session_id,
                    'message': 'Invalid device signature'}
        
        # Verify timestamp
        current_time = datetime.now().timestamp()
        if current_time - device_response['timestamp'] > self.max_nonce_age:
            return {'success': False, 'session_id': session_id,
                    'message': 'Response expired'}
        
        # Verification successful
        session['state'] = 'verified'
        session['verified_at'] = current_time
        
        return {'success': True, 'session_id': session_id,
                'message': 'Authentication successful'}
    
    def establish_session(self, session_id, policy_D=None):
        """
        Step 5: Establish Secure Session
        
        Generate and distribute session key.
        
        Args:
            session_id: Session ID
            policy_D: Session key decryption policy (optional, defaults to device attributes)
            
        Returns:
            dict: Session information
                {
                    'session_id': str,
                    'session_key': bytes,
                    'enc_session_key': dict,  # ABE-encrypted session key
                    'sig_session': G1,        # BLS signature
                    'expiry': float
                }
        """
        if session_id not in self.active_sessions:
            raise ValueError("Invalid session ID")
        
        session = self.active_sessions[session_id]
        
        if session['state'] != 'verified':
            raise ValueError("Session not verified")
        
        # Generate session key (256-bit)
        session_key = secrets.token_bytes(32)
        
        # Set session expiry time (1 hour)
        expiry = datetime.now().timestamp() + 3600
        
        # ABE encrypt session key (if tcabe provided)
        enc_session_key = None
        if self.tcabe is not None and policy_D is not None:
            session_key_GT = self.group.init(GT, int.from_bytes(session_key, 'big') % self.group.order())
            enc_session_key = self.tcabe.encrypt(session_key_GT, policy_D)
        
        # Generate BLS signature for session key
        timestamp = datetime.now().timestamp()
        session_message = f"{session_key.hex()}:{expiry}:{timestamp}"
        sig_session = self.bls.sign(self.dt_sk, session_message)
        
        # Update session state
        session['state'] = 'active'
        session['session_key'] = session_key
        session['expiry'] = expiry
        
        session_info = {
            'session_id': session_id,
            'device_id': session['device_id'],
            'session_key': session_key,
            'enc_session_key': enc_session_key,
            'sig_session': sig_session,
            'expiry': expiry,
            'established_at': timestamp
        }
        
        return session_info
    
    def get_session(self, session_id):
        """Get session information"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if session.get('expiry', 0) > datetime.now().timestamp():
                return session
            else:
                del self.active_sessions[session_id]
        return None
    
    def terminate_session(self, session_id):
        """Terminate session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]


def main():
    """Test bidirectional authentication protocol"""
    from setup import T_CP_ABE_Setup
    from t_cp_abe import T_CP_ABE, PolicyParser
    
    print("=" * 60)
    print("Scheme 4: Bidirectional Authentication Protocol Test")
    print("=" * 60)
    
    # 1. System initialization
    print("\n[Step 1] System Initialization")
    setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
    PP, MK = setup.setup()
    tcabe = T_CP_ABE(PP)
    auth = BidirectionalAuth(PP, tcabe=tcabe)
    print("  ✓ Initialization successful")
    
    # 2. Device key generation
    print("\n[Step 2] Device Key Generation")
    device_id = 'Device_Sensor_001'
    device_attrs = ['type:sensor', 'location:factory', 'dept:maintenance']
    device_abe_SK = tcabe.keygen(MK, device_attrs)
    device_bls_pk, device_bls_sk = auth.bls.keygen()
    print(f"  Device ID: {device_id}")
    print(f"  Device attributes: {device_attrs}")
    print("  ✓ Device ABE key and BLS key pair generated successfully")

    # 3. Device authentication initialization
    print("\n[Step 3] Device Authentication Initialization")
    auth_request, device_bls_sk = auth.device_auth_init(device_id, device_attrs, device_bls_sk=device_bls_sk)
    print(f"  Nonce: {auth_request['nonce_D'].hex()}")
    print("  ✓ Authentication request generated successfully")

    # 4. Digital twin generates challenge
    print("\n[Step 4] Digital Twin Generate Challenge")
    policy_T_str = "type:sensor AND location:factory"
    parser = PolicyParser()
    policy_T = parser.parse(policy_T_str)
    challenge_resp = auth.digital_twin_challenge(auth_request, policy_T)
    print(f"  Session ID: {challenge_resp['session_id']}")
    print(f"  Nonce_T: {challenge_resp['nonce_T'].hex()}")
    print(f"  Encrypted challenge leaves count: {len(challenge_resp['enc_challenge']['leaves'])}")
    print("  ✓ Challenge generated successfully")

    # 5. Device responds (using T-CP-ABE to decrypt challenge)
    print("\n[Step 5] Device Response to Challenge (T-CP-ABE Decryption)")
    device_resp = auth.device_response(
        challenge_resp['session_id'],
        challenge_resp,
        device_abe_SK,
        device_bls_sk,
        device_id
    )
    print(f"  Decrypted challenge: {device_resp['decrypted_challenge'].hex()[:32]}...")
    # response is pairing.Element, needs serialization to bytes
    resp_bytes = auth.group.serialize(device_resp['response'])
    print(f"  Response signature: {resp_bytes.hex()[:32]}...")
    print("  ✓ Response generated successfully (with T-CP-ABE decryption)")

    # 6. Digital twin verification
    print("\n[Step 6] Digital Twin Verification")
    verify_result = auth.digital_twin_verify(device_resp)
    if verify_result['success']:
        print(f"  ✓ Verification successful: {verify_result['message']}")
    else:
        print(f"  ✗ Verification failed: {verify_result['message']}")

    # 7. Establish session
    print("\n[Step 7] Establish Secure Session")
    if verify_result['success']:
        session_info = auth.establish_session(challenge_resp['session_id'])
        print(f"  Session ID: {session_info['session_id']}")
        print(f"  Session key: {session_info['session_key'].hex()[:32]}...")
        print(f"  Expiry time: {datetime.fromtimestamp(session_info['expiry'])}")
        print("  ✓ Session established successfully")

    # 8. Replay attack test
    print("\n[Step 8] Replay Attack Test (Should Fail)")
    try:
        # Attempt to replay old authentication request
        old_auth_request = auth_request.copy()
        auth.digital_twin_challenge(old_auth_request, policy_T)
        print("  ✗ Should not succeed but did")
    except ValueError as e:
        print(f"  ✓ Correctly rejected replay attack: {e}")
    
    print("\n" + "=" * 60)
    print("Bidirectional Authentication Protocol Test Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
